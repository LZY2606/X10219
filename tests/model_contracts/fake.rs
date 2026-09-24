//! Fake host capabilities for model-based contract tests.
//!
//! The fake provides a fully in-memory file system, environment, and HTTP
//! surface. Every capability call appends to an ordered event log so a test
//! can compare the implementation's observable behavior against the
//! simplified model. No real network, real cache, or user directory is ever
//! touched; the only real-FS usage in this suite is explicit unique temp
//! dirs created by `TempHome` in `main.rs`.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use pklr::capabilities::BoxFuture;
use pklr::{EvalCapabilities, Error};

/// Virtual locations inside the fake world.
pub const WORK_ROOT: &str = "/work";
pub const CACHE_DIR: &str = "/cache";
pub const TEMP_ROOT: &str = "/tmp";

/// One observable capability call. The log is ordered; fields carry exactly
/// the arguments the evaluator passed (paths are not normalized here), so
/// the log is independent of scheduling and platform details.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Ev {
    ReadFile(String),
    ReadBytes(String),
    Exists(String),
    Canon(String),
    Env(String),
    FetchText(String),
    FetchBytes(String),
    MkdirAll(String),
    WriteAtomic(String),
    RemoveFile(String),
    TempDir(String),
    ExtractZip(String),
    Glob(String, String),
}

impl Ev {
    /// Scheduling-only fields would be dropped here before comparison. The
    /// fake world is fully deterministic, so normalization is the identity;
    /// it exists to document the comparison contract.
    pub fn normalized(&self) -> &Ev {
        self
    }
}

/// Atomic-write stages mirrored from `pklr::eval::AtomicWriteStage` so the
/// fake can inject the same durability-boundary faults in-memory.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FakeStage {
    CreateTemp,
    Write,
    Flush,
    Rename,
    SyncDir,
}

#[derive(Default)]
pub struct FakeState {
    /// Normalized path ("/a/b.pkl") -> contents.
    pub files: BTreeMap<String, Vec<u8>>,
    /// Explicitly created directories (normalized).
    pub dirs: BTreeSet<String>,
    pub env: BTreeMap<String, String>,
    /// URL -> response bytes.
    pub http: BTreeMap<String, Vec<u8>>,
    pub events: Vec<Ev>,
    pub temp_counter: u64,
    pub temp_file_counter: u64,
    /// One-shot fault injected into the next `write_atomic` call.
    pub write_fault: Option<FakeStage>,
}

/// Lexically normalize an absolute path: resolve `.` and `..` components.
/// The fake world has no symlinks, so this is the canonical form.
pub fn normalize_path(path: &str) -> String {
    let mut parts: Vec<&str> = Vec::new();
    for part in path.split('/') {
        match part {
            "" | "." => {}
            ".." => {
                parts.pop();
            }
            other => parts.push(other),
        }
    }
    format!("/{}", parts.join("/"))
}

fn path_str(path: &Path) -> String {
    path.to_string_lossy().replace('\\', "/")
}

fn io_not_found(path: &Path) -> Error {
    Error::Io(
        path.to_path_buf(),
        std::io::Error::from(std::io::ErrorKind::NotFound),
    )
}

fn injected_fault(stage: FakeStage) -> Error {
    Error::Eval(format!("injected write fault at {stage:?}"))
}

/// Cloneable handle to the fake world. The evaluator owns one clone; the
/// test keeps another to inspect state and the event log afterwards.
#[derive(Clone, Default)]
pub struct Fake {
    pub st: Arc<Mutex<FakeState>>,
}

impl Fake {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn events(&self) -> Vec<Ev> {
        self.st.lock().unwrap().events.clone()
    }

    pub fn env_reads_log(&self) -> Vec<String> {
        self.st
            .lock()
            .unwrap()
            .events
            .iter()
            .filter_map(|ev| match ev {
                Ev::Env(name) => Some(name.clone()),
                _ => None,
            })
            .collect()
    }

    pub fn file_string(&self, path: &str) -> Option<String> {
        let st = self.st.lock().unwrap();
        st.files
            .get(&normalize_path(path))
            .map(|bytes| String::from_utf8_lossy(bytes).into_owned())
    }

    pub fn has_file(&self, path: &str) -> bool {
        self.st
            .lock()
            .unwrap()
            .files
            .contains_key(&normalize_path(path))
    }

    pub fn put_file(&self, path: &str, bytes: Vec<u8>) {
        self.st
            .lock()
            .unwrap()
            .files
            .insert(normalize_path(path), bytes);
    }

    pub fn arm_write_fault(&self, stage: FakeStage) {
        self.st.lock().unwrap().write_fault = Some(stage);
    }

    fn is_dir(state: &FakeState, normalized: &str) -> bool {
        state.dirs.contains(normalized)
            || state
                .files
                .keys()
                .any(|file| file.starts_with(&format!("{normalized}/")))
    }
}

impl EvalCapabilities for Fake {
    fn read_to_string<'a>(&'a mut self, path: &'a Path) -> BoxFuture<'a, pklr::Result<String>> {
        let mut st = self.st.lock().unwrap();
        st.events.push(Ev::ReadFile(path_str(path)));
        let bytes = st.files.get(&normalize_path(&path_str(path))).cloned();
        drop(st);
        Box::pin(async move {
            let bytes = bytes.ok_or_else(|| io_not_found(path))?;
            String::from_utf8(bytes)
                .map_err(|error| Error::Eval(format!("fake file is not UTF-8: {error}")))
        })
    }

    fn path_exists<'a>(&'a mut self, path: &'a Path) -> BoxFuture<'a, pklr::Result<bool>> {
        let mut st = self.st.lock().unwrap();
        st.events.push(Ev::Exists(path_str(path)));
        let normalized = normalize_path(&path_str(path));
        let exists = st.files.contains_key(&normalized) || Self::is_dir(&st, &normalized);
        drop(st);
        Box::pin(async move { Ok(exists) })
    }

    fn canonicalize<'a>(&'a mut self, path: &'a Path) -> BoxFuture<'a, pklr::Result<PathBuf>> {
        let mut st = self.st.lock().unwrap();
        st.events.push(Ev::Canon(path_str(path)));
        let normalized = normalize_path(&path_str(path));
        let exists = st.files.contains_key(&normalized) || Self::is_dir(&st, &normalized);
        drop(st);
        Box::pin(async move {
            if exists {
                Ok(PathBuf::from(normalized))
            } else {
                Err(io_not_found(path))
            }
        })
    }

    fn read_bytes<'a>(&'a mut self, path: &'a Path) -> BoxFuture<'a, pklr::Result<Vec<u8>>> {
        let mut st = self.st.lock().unwrap();
        st.events.push(Ev::ReadBytes(path_str(path)));
        let bytes = st.files.get(&normalize_path(&path_str(path))).cloned();
        drop(st);
        Box::pin(async move { bytes.ok_or_else(|| io_not_found(path)) })
    }

    fn create_dir_all<'a>(&'a mut self, path: &'a Path) -> BoxFuture<'a, pklr::Result<()>> {
        let mut st = self.st.lock().unwrap();
        st.events.push(Ev::MkdirAll(path_str(path)));
        st.dirs.insert(normalize_path(&path_str(path)));
        drop(st);
        Box::pin(async { Ok(()) })
    }

    fn write_atomic<'a>(
        &'a mut self,
        path: &'a Path,
        bytes: &'a [u8],
    ) -> BoxFuture<'a, pklr::Result<()>> {
        let bytes = bytes.to_vec();
        let mut st = self.st.lock().unwrap();
        st.events.push(Ev::WriteAtomic(path_str(path)));
        let fault = st.write_fault.take();
        let normalized = normalize_path(&path_str(path));
        st.temp_file_counter += 1;
        let temp = pklr::eval::cache_temp_file_name(
            normalized.rsplit('/').next().unwrap_or("cache"),
            4242,
            st.temp_file_counter,
        );
        let temp_path = format!(
            "{}/{}",
            normalized.rsplit_once('/').map(|(dir, _)| dir).unwrap_or(""),
            temp
        );
        match fault {
            Some(FakeStage::CreateTemp) => {
                drop(st);
                return Box::pin(async { Err(injected_fault(FakeStage::CreateTemp)) });
            }
            _ => {}
        }
        st.files.insert(temp_path.clone(), Vec::new());
        if fault == Some(FakeStage::Write) {
            st.files.remove(&temp_path);
            drop(st);
            return Box::pin(async { Err(injected_fault(FakeStage::Write)) });
        }
        st.files.insert(temp_path.clone(), bytes.clone());
        if fault == Some(FakeStage::Flush) {
            st.files.remove(&temp_path);
            drop(st);
            return Box::pin(async { Err(injected_fault(FakeStage::Flush)) });
        }
        if fault == Some(FakeStage::Rename) {
            st.files.remove(&temp_path);
            drop(st);
            return Box::pin(async { Err(injected_fault(FakeStage::Rename)) });
        }
        st.files.remove(&temp_path);
        st.files.insert(normalized, bytes);
        if fault == Some(FakeStage::SyncDir) {
            drop(st);
            return Box::pin(async { Err(injected_fault(FakeStage::SyncDir)) });
        }
        drop(st);
        Box::pin(async { Ok(()) })
    }

    fn remove_file<'a>(&'a mut self, path: &'a Path) -> BoxFuture<'a, pklr::Result<()>> {
        let mut st = self.st.lock().unwrap();
        st.events.push(Ev::RemoveFile(path_str(path)));
        let removed = st.files.remove(&normalize_path(&path_str(path)));
        drop(st);
        Box::pin(async move {
            removed.map(|_| ()).ok_or_else(|| io_not_found(path))
        })
    }

    fn extract_zip<'a>(
        &'a mut self,
        bytes: Vec<u8>,
        destination: &'a Path,
    ) -> BoxFuture<'a, pklr::Result<()>> {
        let dest = normalize_path(&path_str(destination));
        let mut st = self.st.lock().unwrap();
        st.events.push(Ev::ExtractZip(dest.clone()));
        drop(st);
        let st = self.st.clone();
        Box::pin(async move {
            let mut archive = zip::ZipArchive::new(std::io::Cursor::new(bytes))
                .map_err(|error| Error::Eval(format!("fake zip error: {error}")))?;
            let mut st = st.lock().unwrap();
            for index in 0..archive.len() {
                let mut entry = archive
                    .by_index(index)
                    .map_err(|error| Error::Eval(format!("fake zip entry error: {error}")))?;
                // Same zip-slip protection as the real extractor: entries
                // escaping the destination are skipped.
                let Some(name) = entry.enclosed_name().map(|p| p.to_path_buf()) else {
                    continue;
                };
                let mut bytes = Vec::new();
                std::io::Read::read_to_end(&mut entry, &mut bytes)
                    .map_err(|error| Error::Eval(format!("fake zip read error: {error}")))?;
                let target = normalize_path(&format!("{dest}/{}", name.to_string_lossy()));
                st.files.insert(target, bytes);
            }
            Ok(())
        })
    }

    fn read_env<'a>(&'a mut self, name: &'a str) -> BoxFuture<'a, pklr::Result<Option<String>>> {
        let mut st = self.st.lock().unwrap();
        st.events.push(Ev::Env(name.to_string()));
        let value = st.env.get(name).cloned();
        drop(st);
        Box::pin(async move { Ok(value) })
    }

    fn fetch_text<'a>(&'a mut self, url: &'a str) -> BoxFuture<'a, pklr::Result<String>> {
        let mut st = self.st.lock().unwrap();
        st.events.push(Ev::FetchText(url.to_string()));
        let body = st.http.get(url).cloned();
        drop(st);
        Box::pin(async move {
            let bytes = body.ok_or_else(|| Error::ImportNotFound(url.to_string()))?;
            String::from_utf8(bytes)
                .map_err(|error| Error::Eval(format!("fake response is not UTF-8: {error}")))
        })
    }

    fn fetch_bytes<'a>(&'a mut self, url: &'a str) -> BoxFuture<'a, pklr::Result<Vec<u8>>> {
        let mut st = self.st.lock().unwrap();
        st.events.push(Ev::FetchBytes(url.to_string()));
        let body = st.http.get(url).cloned();
        drop(st);
        Box::pin(async move { body.ok_or_else(|| Error::ImportNotFound(url.to_string())) })
    }

    fn temp_dir<'a>(&'a mut self, prefix: &'a str) -> BoxFuture<'a, pklr::Result<PathBuf>> {
        let mut st = self.st.lock().unwrap();
        st.events.push(Ev::TempDir(prefix.to_string()));
        st.temp_counter += 1;
        let dir = format!("{TEMP_ROOT}/{}-{}", prefix, st.temp_counter);
        st.dirs.insert(dir.clone());
        drop(st);
        Box::pin(async move { Ok(PathBuf::from(dir)) })
    }

    fn glob<'a>(
        &'a mut self,
        base: &'a Path,
        pattern: &'a str,
    ) -> BoxFuture<'a, pklr::Result<Vec<PathBuf>>> {
        let mut st = self.st.lock().unwrap();
        st.events
            .push(Ev::Glob(path_str(base), pattern.to_string()));
        drop(st);
        Box::pin(async { Ok(Vec::new()) })
    }
}
