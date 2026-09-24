//! Shared test support for the model-based contract tests.
//!
//! `ScriptedHost` is a fake `EvalCapabilities` implementation: the filesystem
//! is a real, per-test temporary directory (so the atomic-write pipeline and
//! cache layout are genuinely exercised), while HTTP and environment access
//! are served from in-memory maps. Every capability call is appended to an
//! ordered event log; paths are recorded relative to the test root and temp
//! file names are normalized so that logs from independent runs of the same
//! case are comparable.

use std::collections::HashMap;
use std::path::{Component, Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use pklr::capabilities::{BoxFuture, EvalCapabilities};

static ROOT_COUNTER: AtomicU64 = AtomicU64::new(0);

/// A unique temporary directory that is removed on drop.
pub struct TestRoot(PathBuf);

impl TestRoot {
    pub fn new(tag: &str) -> Self {
        let id = ROOT_COUNTER.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "pklr-contract-{tag}-{}-{id}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&path);
        std::fs::create_dir_all(&path).unwrap();
        Self(path)
    }

    pub fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TestRoot {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// Lexical path normalization: resolves `.` and `..` without touching the
/// filesystem, so it is deterministic and platform-independent.
pub fn lex_normalize(path: &Path) -> PathBuf {
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

/// One recorded capability call. Paths are root-relative with `/` separators.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Event {
    Read(String),
    Exists(String),
    Env(String),
    FetchText(String),
    FetchBytes(String),
    TempDir(String),
    TempCreate(String),
    WriteTemp(String),
    FlushTemp(String),
    Rename(String, String),
    SyncDir(String),
    Remove(String),
    CreateDir(String),
    ListDir(String),
    Glob(String),
}

/// Cache-write fault boundaries that `ScriptedHost` can inject.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Fault {
    /// Creating the temp file for any atomic write fails.
    CreateTemp,
    /// Writing the content blob (non-metadata target) fails.
    WriteBlob,
    /// Writing the metadata file fails.
    WriteMeta,
    /// Flushing a written temp file fails.
    FlushTemp,
    /// Renaming a staged temp file into place fails.
    Rename,
    /// Syncing the directory after a rename fails.
    SyncDir,
}

impl Fault {
    fn message(self) -> pklr::Error {
        pklr::Error::Eval(format!("injected cache fault: {self:?}"))
    }
}

fn is_meta_temp(name: &str) -> bool {
    name.contains(".meta.tmp-")
}

/// Fake host capabilities backed by a real temp directory and virtual
/// HTTP/environment maps.
pub struct ScriptedHost {
    pub root: PathBuf,
    pub http_text: HashMap<String, String>,
    pub http_bytes: HashMap<String, Vec<u8>>,
    pub env: HashMap<String, String>,
    pub log: Arc<Mutex<Vec<Event>>>,
    pub fault: Option<Fault>,
}

impl ScriptedHost {
    pub fn new(root: &Path) -> Self {
        Self {
            root: root.to_path_buf(),
            http_text: HashMap::new(),
            http_bytes: HashMap::new(),
            env: HashMap::new(),
            log: Arc::new(Mutex::new(Vec::new())),
            fault: None,
        }
    }

    fn rel(&self, path: &Path) -> String {
        let normalized = lex_normalize(path);
        let stripped = normalized.strip_prefix(&self.root).unwrap_or(&normalized);
        let mut text = stripped.to_string_lossy().replace('\\', "/");
        if let Some(index) = text.find(".tmp-") {
            text = format!("{}.tmp-N", &text[..index]);
        }
        text
    }

    fn record(&self, event: Event) {
        self.log.lock().unwrap().push(event);
    }

    fn io_error(path: &Path, error: std::io::Error) -> pklr::Error {
        pklr::Error::Io(path.to_path_buf(), error)
    }

    pub fn take_log(&self) -> Vec<Event> {
        std::mem::take(&mut *self.log.lock().unwrap())
    }
}

impl EvalCapabilities for ScriptedHost {
    fn read_to_string<'a>(&'a mut self, path: &'a Path) -> BoxFuture<'a, pklr::Result<String>> {
        self.record(Event::Read(self.rel(path)));
        let path = path.to_path_buf();
        Box::pin(async move {
            std::fs::read_to_string(&path).map_err(|error| Self::io_error(&path, error))
        })
    }

    fn read_bytes<'a>(&'a mut self, path: &'a Path) -> BoxFuture<'a, pklr::Result<Vec<u8>>> {
        self.record(Event::Read(self.rel(path)));
        let path = path.to_path_buf();
        Box::pin(async move { std::fs::read(&path).map_err(|error| Self::io_error(&path, error)) })
    }

    fn path_exists<'a>(&'a mut self, path: &'a Path) -> BoxFuture<'a, pklr::Result<bool>> {
        self.record(Event::Exists(self.rel(path)));
        let path = path.to_path_buf();
        Box::pin(async move { Ok(path.exists()) })
    }

    fn canonicalize<'a>(&'a mut self, path: &'a Path) -> BoxFuture<'a, pklr::Result<PathBuf>> {
        let normalized = lex_normalize(path);
        Box::pin(async move { Ok(normalized) })
    }

    fn create_dir_all<'a>(&'a mut self, path: &'a Path) -> BoxFuture<'a, pklr::Result<()>> {
        self.record(Event::CreateDir(self.rel(path)));
        let path = path.to_path_buf();
        Box::pin(async move {
            std::fs::create_dir_all(&path).map_err(|error| Self::io_error(&path, error))
        })
    }

    fn create_temp_file<'a>(&'a mut self, target: &'a Path) -> BoxFuture<'a, pklr::Result<PathBuf>> {
        self.record(Event::TempCreate(self.rel(target)));
        let fault = self.fault;
        let target = target.to_path_buf();
        Box::pin(async move {
            if fault == Some(Fault::CreateTemp) {
                return Err(fault.unwrap().message());
            }
            pklr::eval::native_create_temp_file(&target)
                .map_err(|error| Self::io_error(&target, error))
        })
    }

    fn write_temp_file<'a>(
        &'a mut self,
        temp: &'a Path,
        bytes: &'a [u8],
    ) -> BoxFuture<'a, pklr::Result<()>> {
        self.record(Event::WriteTemp(self.rel(temp)));
        let fault = self.fault;
        let temp = temp.to_path_buf();
        let bytes = bytes.to_vec();
        Box::pin(async move {
            if let Some(stage) = fault {
                let meta = is_meta_temp(&temp.to_string_lossy());
                let hit = match stage {
                    Fault::WriteBlob => !meta,
                    Fault::WriteMeta => meta,
                    _ => false,
                };
                if hit {
                    return Err(stage.message());
                }
            }
            pklr::eval::native_write_temp_file(&temp, &bytes)
                .map_err(|error| Self::io_error(&temp, error))
        })
    }

    fn flush_temp_file<'a>(&'a mut self, temp: &'a Path) -> BoxFuture<'a, pklr::Result<()>> {
        self.record(Event::FlushTemp(self.rel(temp)));
        let fault = self.fault;
        let temp = temp.to_path_buf();
        Box::pin(async move {
            if fault == Some(Fault::FlushTemp) {
                return Err(fault.unwrap().message());
            }
            pklr::eval::native_flush_temp_file(&temp)
                .map_err(|error| Self::io_error(&temp, error))
        })
    }

    fn rename_file<'a>(&'a mut self, from: &'a Path, to: &'a Path) -> BoxFuture<'a, pklr::Result<()>> {
        self.record(Event::Rename(self.rel(from), self.rel(to)));
        let fault = self.fault;
        let from = from.to_path_buf();
        let to = to.to_path_buf();
        Box::pin(async move {
            if fault == Some(Fault::Rename) {
                return Err(fault.unwrap().message());
            }
            pklr::eval::native_rename_file(&from, &to).map_err(|error| Self::io_error(&to, error))
        })
    }

    fn sync_dir<'a>(&'a mut self, dir: &'a Path) -> BoxFuture<'a, pklr::Result<()>> {
        self.record(Event::SyncDir(self.rel(dir)));
        let fault = self.fault;
        let dir = dir.to_path_buf();
        Box::pin(async move {
            if fault == Some(Fault::SyncDir) {
                return Err(fault.unwrap().message());
            }
            pklr::eval::native_sync_dir(&dir).map_err(|error| Self::io_error(&dir, error))
        })
    }

    fn remove_file<'a>(&'a mut self, path: &'a Path) -> BoxFuture<'a, pklr::Result<()>> {
        self.record(Event::Remove(self.rel(path)));
        let path = path.to_path_buf();
        Box::pin(async move {
            std::fs::remove_file(&path).map_err(|error| Self::io_error(&path, error))
        })
    }

    fn list_dir<'a>(&'a mut self, path: &'a Path) -> BoxFuture<'a, pklr::Result<Vec<PathBuf>>> {
        self.record(Event::ListDir(self.rel(path)));
        let path = path.to_path_buf();
        Box::pin(async move {
            match std::fs::read_dir(&path) {
                Ok(entries) => {
                    let mut paths = Vec::new();
                    for entry in entries {
                        let entry = entry.map_err(|error| Self::io_error(&path, error))?;
                        paths.push(entry.path());
                    }
                    // Directory iteration order is OS-dependent; sort so the
                    // event log is deterministic across runs.
                    paths.sort();
                    Ok(paths)
                }
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(Vec::new()),
                Err(error) => Err(Self::io_error(&path, error)),
            }
        })
    }

    fn read_env<'a>(&'a mut self, name: &'a str) -> BoxFuture<'a, pklr::Result<Option<String>>> {
        self.record(Event::Env(name.to_string()));
        let value = self.env.get(name).cloned();
        Box::pin(async move { Ok(value) })
    }

    fn fetch_text<'a>(&'a mut self, url: &'a str) -> BoxFuture<'a, pklr::Result<String>> {
        self.record(Event::FetchText(url.to_string()));
        let body = self.http_text.get(url).cloned();
        let url = url.to_string();
        Box::pin(async move { body.ok_or(pklr::Error::ImportNotFound(url)) })
    }

    fn fetch_bytes<'a>(&'a mut self, url: &'a str) -> BoxFuture<'a, pklr::Result<Vec<u8>>> {
        self.record(Event::FetchBytes(url.to_string()));
        let body = self.http_bytes.get(url).cloned();
        let url = url.to_string();
        Box::pin(async move { body.ok_or(pklr::Error::ImportNotFound(url)) })
    }

    fn temp_dir<'a>(&'a mut self, prefix: &'a str) -> BoxFuture<'a, pklr::Result<PathBuf>> {
        self.record(Event::TempDir(prefix.to_string()));
        let dir = self.root.join("tmp").join(prefix);
        Box::pin(async move {
            std::fs::create_dir_all(&dir).map_err(|error| Self::io_error(&dir, error))?;
            Ok(dir)
        })
    }

    fn glob<'a>(
        &'a mut self,
        base: &'a Path,
        pattern: &'a str,
    ) -> BoxFuture<'a, pklr::Result<Vec<PathBuf>>> {
        self.record(Event::Glob(pattern.to_string()));
        let base = base.to_path_buf();
        let pattern = pattern.to_string();
        Box::pin(async move { pklr::eval::expand_glob(&base, &pattern) })
    }
}

/// Collect the relative paths of every file under `dir` (empty when missing).
pub fn list_files(dir: &Path) -> Vec<String> {
    let mut out = Vec::new();
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            out.push(entry.file_name().to_string_lossy().to_string());
        }
    }
    out.sort();
    out
}

/// Build a zip archive in memory with a single text entry.
pub fn zip_bytes(entry: &str, source: &str) -> Vec<u8> {
    use std::io::Write;

    let mut bytes = Vec::new();
    {
        let mut zip = zip::ZipWriter::new(std::io::Cursor::new(&mut bytes));
        zip.start_file(
            entry,
            zip::write::SimpleFileOptions::default()
                .compression_method(zip::CompressionMethod::Stored),
        )
        .unwrap();
        zip.write_all(source.as_bytes()).unwrap();
        zip.finish().unwrap();
    }
    bytes
}
