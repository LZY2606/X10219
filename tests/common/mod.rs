//! Shared harness for hermetic model/contract tests: a confined filesystem
//! sandbox, fake HTTP/env capabilities, and an ordered event log.
#![allow(dead_code)]

use std::collections::BTreeMap;
use std::path::{Component, Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use pklr::capabilities::{BoxFuture, EvalCapabilities};

/// FNV-1a 64-bit; mirrors `pklr::package_cache`'s key/checksum function.
pub fn fnv1a(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf29ce484222325_u64;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

pub fn cache_key(url: &str) -> String {
    format!("{:016x}", fnv1a(url.as_bytes()))
}

/// Longest-prefix URL rewrite, mirroring `Evaluator::rewrite_url`.
pub fn apply_rewrites(rules: &[(String, String)], url: &str) -> String {
    match rules
        .iter()
        .filter(|(src, _)| url.starts_with(src.as_str()))
        .max_by_key(|(src, _)| src.len())
    {
        Some((src, tgt)) => format!("{}{}", tgt, &url[src.len()..]),
        None => url.to_string(),
    }
}

static ROOT_COUNTER: AtomicU64 = AtomicU64::new(0);

/// A unique temporary directory that is reclaimed on `close`/drop.
pub struct TestRoot {
    canonical: PathBuf,
    closed: bool,
}

impl TestRoot {
    pub fn new(tag: &str) -> Self {
        let tag: String = tag
            .chars()
            .map(|c| if c.is_ascii_alphanumeric() { c } else { '-' })
            .collect();
        let dir = std::env::temp_dir().join(format!(
            "pklr-contract-{tag}-{}-{}",
            std::process::id(),
            ROOT_COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let canonical = dir.canonicalize().unwrap();
        Self {
            canonical,
            closed: false,
        }
    }

    pub fn path(&self) -> &Path {
        &self.canonical
    }

    /// Reclaim the directory and assert it is gone.
    pub fn close(mut self) {
        self.closed = true;
        std::fs::remove_dir_all(&self.canonical).unwrap();
        assert!(
            !self.canonical.exists(),
            "temp dir not reclaimed: {}",
            self.canonical.display()
        );
    }
}

impl Drop for TestRoot {
    fn drop(&mut self) {
        if !self.closed {
            let _ = std::fs::remove_dir_all(&self.canonical);
        }
    }
}

/// Capability event kinds recorded by the sandbox.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Ev {
    ReadFile,
    ReadBytes,
    PathExists,
    Canonicalize,
    CreateDirAll,
    WriteBytes,
    FlushFile,
    Rename,
    SyncDir,
    RemoveFile,
    ReadEnv,
    FetchText,
    FetchBytes,
    TempDir,
    Glob,
}

pub type Event = (Ev, String);

/// Lexically resolve `.`/`..` without touching the filesystem.
pub fn lexical_normalize(path: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for component in path.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                out.pop();
            }
            other => out.push(other.as_os_str()),
        }
    }
    out
}

/// Replace the nondeterministic temp-name suffix (`.tmp-<pid>-<counter>`)
/// with a stable marker so event logs compare across drivers.
pub fn normalize_temp_names(target: &str) -> String {
    match target.find(".tmp-") {
        Some(index) => format!("{}*", &target[..index + ".tmp-".len()]),
        None => target.to_string(),
    }
}

/// Hermetic capabilities: real filesystem confined to a [`TestRoot`], fake
/// HTTP and environment, and an ordered event log. No network, no user cache,
/// no home-directory access.
#[derive(Clone)]
pub struct SandboxCaps {
    root: PathBuf,
    pub http: Arc<BTreeMap<String, Vec<u8>>>,
    pub env: Arc<BTreeMap<String, String>>,
    pub events: Arc<Mutex<Vec<Event>>>,
}

impl SandboxCaps {
    pub fn new(
        root: &TestRoot,
        http: BTreeMap<String, Vec<u8>>,
        env: BTreeMap<String, String>,
    ) -> Self {
        Self {
            root: root.path().to_path_buf(),
            http: Arc::new(http),
            env: Arc::new(env),
            events: Arc::new(Mutex::new(Vec::new())),
        }
    }

    pub fn events(&self) -> Vec<Event> {
        self.events.lock().unwrap().clone()
    }

    fn log(&self, kind: Ev, target: String) {
        self.events
            .lock()
            .unwrap()
            .push((kind, normalize_temp_names(&target)));
    }

    /// Normalize a path to a root-relative string; panics on sandbox escape.
    pub fn rel(&self, path: &Path) -> String {
        let normalized = lexical_normalize(path);
        let rel = normalized.strip_prefix(&self.root).unwrap_or_else(|_| {
            panic!(
                "sandbox escape: {} is outside {}",
                path.display(),
                self.root.display()
            )
        });
        rel.to_string_lossy().replace('\\', "/")
    }

    fn confine(&self, path: &Path) -> Result<PathBuf, pklr::Error> {
        let normalized = lexical_normalize(path);
        if normalized.strip_prefix(&self.root).is_err() {
            return Err(pklr::Error::Eval(format!(
                "sandbox escape: {} is outside {}",
                path.display(),
                self.root.display()
            )));
        }
        Ok(normalized)
    }

    fn io_err(path: &Path, error: std::io::Error) -> pklr::Error {
        pklr::Error::Io(path.to_path_buf(), error)
    }
}

impl EvalCapabilities for SandboxCaps {
    fn read_to_string<'a>(&'a mut self, path: &'a Path) -> BoxFuture<'a, pklr::Result<String>> {
        self.log(Ev::ReadFile, self.rel(path));
        let result = self.confine(path);
        Box::pin(async move {
            let path = result?;
            std::fs::read_to_string(&path).map_err(|e| Self::io_err(&path, e))
        })
    }

    fn path_exists<'a>(&'a mut self, path: &'a Path) -> BoxFuture<'a, pklr::Result<bool>> {
        self.log(Ev::PathExists, self.rel(path));
        let result = self.confine(path);
        Box::pin(async move { Ok(result.is_ok_and(|p| p.exists())) })
    }

    fn canonicalize<'a>(&'a mut self, path: &'a Path) -> BoxFuture<'a, pklr::Result<PathBuf>> {
        self.log(Ev::Canonicalize, self.rel(path));
        let result = self.confine(path);
        Box::pin(async move {
            let path = result?;
            match path.canonicalize() {
                Ok(canonical) => Ok(canonical),
                // Not-yet-existing paths (e.g. cache files) stay lexical.
                Err(_) => Ok(path),
            }
        })
    }

    fn read_bytes<'a>(&'a mut self, path: &'a Path) -> BoxFuture<'a, pklr::Result<Vec<u8>>> {
        self.log(Ev::ReadBytes, self.rel(path));
        let result = self.confine(path);
        Box::pin(async move {
            let path = result?;
            std::fs::read(&path).map_err(|e| Self::io_err(&path, e))
        })
    }

    fn create_dir_all<'a>(&'a mut self, path: &'a Path) -> BoxFuture<'a, pklr::Result<()>> {
        self.log(Ev::CreateDirAll, self.rel(path));
        let result = self.confine(path);
        Box::pin(async move {
            let path = result?;
            std::fs::create_dir_all(&path).map_err(|e| Self::io_err(&path, e))
        })
    }

    fn write_bytes<'a>(
        &'a mut self,
        path: &'a Path,
        bytes: &'a [u8],
    ) -> BoxFuture<'a, pklr::Result<()>> {
        self.log(Ev::WriteBytes, self.rel(path));
        let result = self.confine(path);
        let bytes = bytes.to_vec();
        Box::pin(async move {
            let path = result?;
            use std::io::Write;
            std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&path)
                .and_then(|mut file| file.write_all(&bytes))
                .map_err(|e| Self::io_err(&path, e))
        })
    }

    fn flush_file<'a>(&'a mut self, path: &'a Path) -> BoxFuture<'a, pklr::Result<()>> {
        self.log(Ev::FlushFile, self.rel(path));
        let result = self.confine(path);
        Box::pin(async move {
            let path = result?;
            std::fs::OpenOptions::new()
                .write(true)
                .open(&path)
                .and_then(|file| file.sync_all())
                .map_err(|e| Self::io_err(&path, e))
        })
    }

    fn rename<'a>(&'a mut self, from: &'a Path, to: &'a Path) -> BoxFuture<'a, pklr::Result<()>> {
        self.log(Ev::Rename, format!("{} -> {}", self.rel(from), self.rel(to)));
        let from = self.confine(from);
        let to = self.confine(to);
        Box::pin(async move {
            let from = from?;
            let to = to?;
            std::fs::rename(&from, &to).map_err(|e| Self::io_err(&to, e))
        })
    }

    fn sync_dir<'a>(&'a mut self, path: &'a Path) -> BoxFuture<'a, pklr::Result<()>> {
        self.log(Ev::SyncDir, self.rel(path));
        let result = self.confine(path);
        Box::pin(async move {
            let path = result?;
            #[cfg(unix)]
            {
                std::fs::File::open(&path)
                    .and_then(|dir| dir.sync_all())
                    .map_err(|e| Self::io_err(&path, e))
            }
            #[cfg(not(unix))]
            {
                let _ = path;
                Ok(())
            }
        })
    }

    fn remove_file<'a>(&'a mut self, path: &'a Path) -> BoxFuture<'a, pklr::Result<()>> {
        self.log(Ev::RemoveFile, self.rel(path));
        let result = self.confine(path);
        Box::pin(async move {
            let path = result?;
            std::fs::remove_file(&path).map_err(|e| Self::io_err(&path, e))
        })
    }

    fn read_env<'a>(&'a mut self, name: &'a str) -> BoxFuture<'a, pklr::Result<Option<String>>> {
        self.log(Ev::ReadEnv, name.to_string());
        let value = self.env.get(name).cloned();
        Box::pin(async move { Ok(value) })
    }

    fn fetch_text<'a>(&'a mut self, url: &'a str) -> BoxFuture<'a, pklr::Result<String>> {
        self.log(Ev::FetchText, url.to_string());
        let body = self.http.get(url).cloned();
        let url = url.to_string();
        Box::pin(async move {
            let bytes = body.ok_or_else(|| pklr::Error::ImportNotFound(url.clone()))?;
            String::from_utf8(bytes).map_err(|e| pklr::Error::Eval(e.to_string()))
        })
    }

    fn fetch_bytes<'a>(&'a mut self, url: &'a str) -> BoxFuture<'a, pklr::Result<Vec<u8>>> {
        self.log(Ev::FetchBytes, url.to_string());
        let body = self.http.get(url).cloned();
        let url = url.to_string();
        Box::pin(async move { body.ok_or_else(|| pklr::Error::ImportNotFound(url)) })
    }

    fn temp_dir<'a>(&'a mut self, prefix: &'a str) -> BoxFuture<'a, pklr::Result<PathBuf>> {
        self.log(Ev::TempDir, prefix.to_string());
        let root = self.root.clone();
        let prefix = prefix.to_string();
        Box::pin(async move {
            let parent = root.join("tmp");
            std::fs::create_dir_all(&parent).map_err(|e| Self::io_err(&parent, e))?;
            for attempt in 0..100u32 {
                let dir = parent.join(format!("{prefix}-{attempt}"));
                match std::fs::create_dir(&dir) {
                    Ok(()) => return Ok(dir),
                    Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
                    Err(e) => return Err(Self::io_err(&dir, e)),
                }
            }
            Err(pklr::Error::Eval("temp dir exhausted".to_string()))
        })
    }

    fn glob<'a>(
        &'a mut self,
        base: &'a Path,
        pattern: &'a str,
    ) -> BoxFuture<'a, pklr::Result<Vec<PathBuf>>> {
        self.log(Ev::Glob, format!("{} :: {}", self.rel(base), pattern));
        let result = self.confine(base);
        let pattern = pattern.to_string();
        Box::pin(async move {
            let base = result?;
            pklr::eval::expand_glob(&base, &pattern)
        })
    }
}

/// Count events by (kind, target).
pub fn count_events(events: &[Event]) -> BTreeMap<Event, usize> {
    let mut counts = BTreeMap::new();
    for event in events {
        *counts.entry(event.clone()).or_insert(0) += 1;
    }
    counts
}

/// First-occurrence index of an event in the log.
pub fn first_index(events: &[Event], kind: Ev, target: &str) -> Option<usize> {
    events.iter().position(|(k, t)| *k == kind && t == target)
}
