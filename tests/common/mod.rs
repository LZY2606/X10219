#![allow(dead_code)]

//! Shared model-contract test host: a deterministic in-memory filesystem,
//! controllable network and environment, and an ordered event log.
//!
//! Nothing here touches the real network, the user cache, or
//! platform-specific temp directories.

use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use pklr::capabilities::{BoxFuture, EvalCapabilities};
use pklr::package_cache::FileKind;
use pklr::{Error, Result};

/// Scheduler flavor used while an event was recorded. Only the async host
/// yields, and yields are scheduling-only and ignored during comparison.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    Sync,
    Async,
}

/// A single observable interaction between evaluator and host.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Event {
    Canonicalize(String),
    Exists(String),
    ReadFile(String),
    ReadEnv(String),
    FetchText(String),
    FetchBytes(String),
    CacheSweep,
    CacheList(String),
    CacheRemove(String),
    CacheMkdir(String),
    CacheCreate {
        kind: &'static str,
        path: String,
        bytes_len: usize,
    },
    CacheFlush {
        kind: &'static str,
        path: String,
    },
    CacheRename {
        kind: &'static str,
        from: String,
        to: String,
    },
    CacheSync {
        kind: &'static str,
        path: String,
    },
    TempDir(String),
    Glob(String),
    ExtractZip(String),
    /// Scheduling-only: emitted by the async host between capability calls.
    Yield,
}

fn key_of(path: &Path) -> String {
    path.to_string_lossy().replace('\\', "/")
}

/// Which cache boundary fails, and how many matching calls succeed first.
#[derive(Debug, Clone)]
pub struct CacheFault {
    pub boundary: &'static str,
    pub kind: Option<FileKind>,
    pub skip: usize,
}

/// In-memory filesystem plus network/environment answers.
#[derive(Clone)]
pub struct FakeHost {
    pub state: Arc<Mutex<HostState>>,
    mode: Mode,
}

pub struct HostState {
    pub files: HashMap<PathBuf, Vec<u8>>,
    pub env: BTreeMap<String, String>,
    pub http_text: HashMap<String, String>,
    pub http_bytes: HashMap<String, Vec<u8>>,
    pub events: Vec<Event>,
    pub fault: Option<CacheFault>,
    pub fault_hits: usize,
    pub next_temp: usize,
}

impl FakeHost {
    pub fn new(mode: Mode) -> Self {
        Self {
            state: Arc::new(Mutex::new(HostState {
                files: HashMap::new(),
                env: BTreeMap::new(),
                http_text: HashMap::new(),
                http_bytes: HashMap::new(),
                events: Vec::new(),
                fault: None,
                fault_hits: 0,
                next_temp: 0,
            })),
            mode,
        }
    }

    pub fn install_fault(&self, fault: CacheFault) {
        let mut state = self.state.lock().unwrap();
        state.fault = Some(fault);
        state.fault_hits = 0;
    }

    pub fn snapshot_events(&self) -> Vec<Event> {
        self.state.lock().unwrap().events.clone()
    }

    /// Canonical event sequence: drop scheduler-only yields.
    pub fn canonical_events(&self) -> Vec<Event> {
        self.snapshot_events()
            .into_iter()
            .filter(|event| *event != Event::Yield)
            .collect()
    }

    pub fn put_file(&self, path: impl Into<PathBuf>, bytes: impl Into<Vec<u8>>) {
        self.state
            .lock()
            .unwrap()
            .files
            .insert(path.into(), bytes.into());
    }

    pub fn put_text(&self, path: impl Into<PathBuf>, text: impl Into<String>) {
        self.put_file(path, text.into().into_bytes());
    }

    pub fn read_file(&self, path: &Path) -> Option<Vec<u8>> {
        self.state.lock().unwrap().files.get(path).cloned()
    }

    pub fn file_names(&self) -> Vec<String> {
        let mut names = self
            .state
            .lock()
            .unwrap()
            .files
            .keys()
            .map(|path| key_of(path))
            .collect::<Vec<_>>();
        names.sort();
        names
    }

    pub fn put_env(&self, name: impl Into<String>, value: impl Into<String>) {
        self.state
            .lock()
            .unwrap()
            .env
            .insert(name.into(), value.into());
    }

    pub fn put_http_text(&self, url: impl Into<String>, text: impl Into<String>) {
        self.state
            .lock()
            .unwrap()
            .http_text
            .insert(url.into(), text.into());
    }

    pub fn put_http_bytes(&self, url: impl Into<String>, bytes: impl Into<Vec<u8>>) {
        self.state
            .lock()
            .unwrap()
            .http_bytes
            .insert(url.into(), bytes.into());
    }

    fn maybe_yield(&self) {
        if self.mode == Mode::Async {
            self.state.lock().unwrap().events.push(Event::Yield);
        }
    }
}

fn io_not_found(path: &Path) -> Error {
    Error::Io(
        path.to_path_buf(),
        std::io::Error::new(std::io::ErrorKind::NotFound, "fake fs: not found"),
    )
}

impl EvalCapabilities for FakeHost {
    fn read_to_string<'a>(&'a mut self, path: &'a Path) -> BoxFuture<'a, Result<String>> {
        self.maybe_yield();
        let state = self.state.clone();
        let owned = path.to_path_buf();
        Box::pin(async move {
            let mut state = state.lock().unwrap();
            state.events.push(Event::ReadFile(key_of(&owned)));
            match state.files.get(&owned).cloned() {
                Some(bytes) => String::from_utf8(bytes)
                    .map_err(|error| Error::Eval(format!("fake fs: invalid UTF-8: {error}"))),
                None => Err(io_not_found(&owned)),
            }
        })
    }

    fn path_exists<'a>(&'a mut self, path: &'a Path) -> BoxFuture<'a, Result<bool>> {
        self.maybe_yield();
        let state = self.state.clone();
        let owned = path.to_path_buf();
        Box::pin(async move {
            let exists = {
                let mut state = state.lock().unwrap();
                state.events.push(Event::Exists(key_of(&owned)));
                state.files.contains_key(&owned)
            };
            Ok(exists)
        })
    }

    fn canonicalize<'a>(&'a mut self, path: &'a Path) -> BoxFuture<'a, Result<PathBuf>> {
        self.maybe_yield();
        let state = self.state.clone();
        let owned = path.to_path_buf();
        Box::pin(async move {
            state
                .lock()
                .unwrap()
                .events
                .push(Event::Canonicalize(key_of(&owned)));
            Ok(owned)
        })
    }

    fn read_bytes<'a>(&'a mut self, path: &'a Path) -> BoxFuture<'a, Result<Vec<u8>>> {
        self.maybe_yield();
        let state = self.state.clone();
        let owned = path.to_path_buf();
        Box::pin(async move {
            let mut state = state.lock().unwrap();
            state.events.push(Event::ReadFile(key_of(&owned)));
            state.files.get(&owned).cloned().ok_or_else(|| io_not_found(&owned))
        })
    }

    fn create_dir_all<'a>(&'a mut self, _path: &'a Path) -> BoxFuture<'a, Result<()>> {
        Box::pin(async { Ok(()) })
    }

    fn write_atomic<'a>(
        &'a mut self,
        path: &'a Path,
        bytes: &'a [u8],
    ) -> BoxFuture<'a, Result<()>> {
        let state = self.state.clone();
        let owned = path.to_path_buf();
        let bytes = bytes.to_vec();
        Box::pin(async move {
            state.lock().unwrap().files.insert(owned, bytes);
            Ok(())
        })
    }

    fn remove_file<'a>(&'a mut self, path: &'a Path) -> BoxFuture<'a, Result<()>> {
        self.maybe_yield();
        let state = self.state.clone();
        let owned = path.to_path_buf();
        Box::pin(async move {
            let mut guard = state.lock().unwrap();
            guard.events.push(Event::CacheRemove(key_of(&owned)));
            guard.files.remove(&owned);
            Ok(())
        })
    }

    fn read_env<'a>(&'a mut self, name: &'a str) -> BoxFuture<'a, Result<Option<String>>> {
        self.maybe_yield();
        let state = self.state.clone();
        let name = name.to_string();
        Box::pin(async move {
            let mut guard = state.lock().unwrap();
            guard.events.push(Event::ReadEnv(name.clone()));
            Ok(guard.env.get(&name).cloned())
        })
    }

    fn fetch_text<'a>(&'a mut self, url: &'a str) -> BoxFuture<'a, Result<String>> {
        self.maybe_yield();
        let state = self.state.clone();
        let url = url.to_string();
        Box::pin(async move {
            let mut guard = state.lock().unwrap();
            guard.events.push(Event::FetchText(url.clone()));
            guard
                .http_text
                .get(&url)
                .cloned()
                .ok_or_else(|| Error::ImportNotFound(url.clone()))
        })
    }

    fn fetch_bytes<'a>(&'a mut self, url: &'a str) -> BoxFuture<'a, Result<Vec<u8>>> {
        self.maybe_yield();
        let state = self.state.clone();
        let url = url.to_string();
        Box::pin(async move {
            let mut guard = state.lock().unwrap();
            guard.events.push(Event::FetchBytes(url.clone()));
            guard
                .http_bytes
                .get(&url)
                .cloned()
                .ok_or_else(|| Error::ImportNotFound(url.clone()))
        })
    }

    fn temp_dir<'a>(&'a mut self, prefix: &'a str) -> BoxFuture<'a, Result<PathBuf>> {
        self.maybe_yield();
        let state = self.state.clone();
        let prefix = prefix.to_string();
        Box::pin(async move {
            let mut guard = state.lock().unwrap();
            guard.events.push(Event::TempDir(prefix.clone()));
            guard.next_temp += 1;
            Ok(PathBuf::from(format!("/tmp/{prefix}-{}", guard.next_temp)))
        })
    }

    fn glob<'a>(
        &'a mut self,
        base: &'a Path,
        pattern: &'a str,
    ) -> BoxFuture<'a, Result<Vec<PathBuf>>> {
        self.maybe_yield();
        let state = self.state.clone();
        let spec = format!("{}/{}", key_of(base), pattern);
        Box::pin(async move {
            state
                .lock()
                .unwrap()
                .events
                .push(Event::Glob(spec));
            Ok(Vec::new())
        })
    }

    fn cache_mkdir<'a>(&'a mut self, path: &'a Path) -> BoxFuture<'a, Result<()>> {
        self.maybe_yield();
        let state = self.state.clone();
        let owned = path.to_path_buf();
        Box::pin(async move {
            let mut guard = state.lock().unwrap();
            guard.events.push(Event::CacheMkdir(key_of(&owned)));
            if let Some(error) = Self::consume_fault(&mut guard, "mkdir", FileKind::Blob) {
                return Err(error);
            }
            Ok(())
        })
    }

    fn cache_create_file<'a>(
        &'a mut self,
        kind: FileKind,
        path: &'a Path,
        bytes: &'a [u8],
    ) -> BoxFuture<'a, Result<()>> {
        self.maybe_yield();
        let state = self.state.clone();
        let owned = path.to_path_buf();
        let bytes = bytes.to_vec();
        Box::pin(async move {
            let mut guard = state.lock().unwrap();
            guard.events.push(Event::CacheCreate {
                kind: kind.as_str(),
                path: key_of(&owned),
                bytes_len: bytes.len(),
            });
            if let Some(error) = Self::consume_fault(&mut guard, "create", kind) {
                return Err(error);
            }
            if guard.files.contains_key(&owned) {
                return Err(Error::Io(
                    owned,
                    std::io::Error::new(
                        std::io::ErrorKind::AlreadyExists,
                        "fake fs: staged file already exists",
                    ),
                ));
            }
            guard.files.insert(owned, bytes);
            Ok(())
        })
    }

    fn cache_flush_file<'a>(
        &'a mut self,
        kind: FileKind,
        path: &'a Path,
    ) -> BoxFuture<'a, Result<()>> {
        self.maybe_yield();
        let state = self.state.clone();
        let owned = path.to_path_buf();
        Box::pin(async move {
            let mut guard = state.lock().unwrap();
            guard.events.push(Event::CacheFlush {
                kind: kind.as_str(),
                path: key_of(&owned),
            });
            if let Some(error) = Self::consume_fault(&mut guard, "flush", kind) {
                return Err(error);
            }
            if !guard.files.contains_key(&owned) {
                return Err(io_not_found(&owned));
            }
            Ok(())
        })
    }

    fn cache_rename<'a>(
        &'a mut self,
        kind: FileKind,
        from: &'a Path,
        to: &'a Path,
    ) -> BoxFuture<'a, Result<()>> {
        self.maybe_yield();
        let state = self.state.clone();
        let from_owned = from.to_path_buf();
        let to_owned = to.to_path_buf();
        Box::pin(async move {
            let mut guard = state.lock().unwrap();
            guard.events.push(Event::CacheRename {
                kind: kind.as_str(),
                from: key_of(&from_owned),
                to: key_of(&to_owned),
            });
            if let Some(error) = Self::consume_fault(&mut guard, "rename", kind) {
                return Err(error);
            }
            let bytes = guard
                .files
                .remove(&from_owned)
                .ok_or_else(|| io_not_found(&from_owned))?;
            guard.files.insert(to_owned, bytes);
            Ok(())
        })
    }

    fn cache_sync_dir<'a>(
        &'a mut self,
        kind: FileKind,
        path: &'a Path,
    ) -> BoxFuture<'a, Result<()>> {
        self.maybe_yield();
        let state = self.state.clone();
        let owned = path.to_path_buf();
        Box::pin(async move {
            let mut guard = state.lock().unwrap();
            guard.events.push(Event::CacheSync {
                kind: kind.as_str(),
                path: key_of(&owned),
            });
            if let Some(error) = Self::consume_fault(&mut guard, "sync", kind) {
                return Err(error);
            }
            Ok(())
        })
    }

    fn list_dir<'a>(&'a mut self, path: &'a Path) -> BoxFuture<'a, Result<Vec<PathBuf>>> {
        self.maybe_yield();
        let state = self.state.clone();
        let owned = path.to_path_buf();
        Box::pin(async move {
            let mut guard = state.lock().unwrap();
            guard.events.push(Event::CacheList(key_of(&owned)));
            let mut paths = guard
                .files
                .keys()
                .filter(|candidate| candidate.parent() == Some(owned.as_path()))
                .cloned()
                .collect::<Vec<_>>();
            paths.sort();
            Ok(paths)
        })
    }

}


impl FakeHost {
    fn fault_error(message: &str) -> Error {
        Error::Io(
            PathBuf::from("<cache>"),
            std::io::Error::new(std::io::ErrorKind::Other, message.to_string()),
        )
    }

    fn consume_fault(state: &mut HostState, boundary: &'static str, kind: FileKind) -> Option<Error> {
        let fault = state.fault.clone()?;
        if fault.boundary != boundary {
            return None;
        }
        if let Some(expected_kind) = fault.kind
            && expected_kind != kind
        {
            return None;
        }
        let fail = state.fault_hits >= fault.skip;
        state.fault_hits += 1;
        if fail {
            // A consumed one-shot fault is cleared so retry behavior stays
            // deterministic and cannot leak into later operations.
            state.fault = None;
            Some(Self::fault_error(&format!(
                "injected cache failure at {boundary} ({})",
                kind.as_str()
            )))
        } else {
            None
        }
    }
}



// --- Execution modes ---------------------------------------------------------

/// Run an evaluation to completion synchronously (blocking executor).
pub fn run_sync<F>(host: FakeHost, configure: F, path: &Path) -> Result<pklr::Value>
where
    F: FnOnce(&mut pklr::Evaluator),
{
    let mut evaluator = pklr::Evaluator::with_capabilities(host);
    configure(&mut evaluator);
    evaluator.set_base_path(path.parent().unwrap_or(Path::new(".")));
    pollster::block_on(evaluator.eval_file_pub(path))
}

/// Same evaluation driven through a Tokio runtime (async executor).
///
/// Both modes use the same deterministic host; only the scheduling changes.
pub fn run_async<F>(host: FakeHost, configure: F, path: &Path) -> Result<pklr::Value>
where
    F: FnOnce(&mut pklr::Evaluator) + Send,
{
    let path = path.to_path_buf();
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .expect("tokio runtime builds");
    runtime.block_on(async move {
        let mut evaluator = pklr::Evaluator::with_capabilities(host);
        configure(&mut evaluator);
        evaluator.set_base_path(path.parent().unwrap_or(Path::new(".")));
        evaluator.eval_file_pub(&path).await
    })
}

// --- Cache layout helpers ----------------------------------------------------

pub fn digest(bytes: &[u8]) -> String {
    pklr::package_cache::sha256_hex(bytes)
}

pub fn meta_file(cache: &Path, url: &str) -> PathBuf {
    pklr::package_cache::meta_path(cache, url)
}

pub fn blob_file(cache: &Path, _url: &str, extension: &str, bytes: &[u8]) -> PathBuf {
    pklr::package_cache::blob_path(cache, &digest(bytes), extension)
}

pub fn meta_json(url: &str, extension: &str, bytes: &[u8]) -> String {
    let digest = digest(bytes);
    format!(
        "{{\"version\":1,\"url\":\"{url}\",\"extension\":\"{extension}\",\"sha256\":\"{digest}\",\"blob\":\"blobs/{digest}.{extension}\"}}"
    )
}

pub fn legacy_blob(cache: &Path, url: &str, extension: &str) -> PathBuf {
    pklr::package_cache::legacy_paths(cache, url, extension).0
}

pub fn legacy_url_sidecar(cache: &Path, url: &str, extension: &str) -> PathBuf {
    pklr::package_cache::legacy_paths(cache, url, extension).1
}
