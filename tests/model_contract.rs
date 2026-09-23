//! Model-based contract tests for pklr's evaluation pipeline.
//!
//! A recording fake-capabilities host turns one evaluation into an ordered
//! event log. A small, independent reference model computes the expected log
//! and outcome for bounded, deterministically generated cases covering URI
//! rewrites (longest prefix), relative file imports, package metadata/blob
//! caches, preloads, offline hit/miss, amend edges, and cycle detection.
//!
//! The generator uses fixed seeds with explicit state-space bounds; on
//! failure the test prints the seed (full deterministic replay via
//! `PKLR_MODEL_REPLAY=<seed>`) plus a greedily minimized case.
//!
//! Every generated case runs through both the synchronous (pollster) and
//! asynchronous (tokio) drivers; event logs must agree after normalizing
//! scheduling-only fields (temp directory names).

use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use pklr::capabilities::BoxFuture;
use pklr::{EvalCapabilities, Evaluator};

// ============================================================================
// Ordered event log + recording fake capabilities
// ============================================================================

#[derive(Debug, Clone, PartialEq, Eq)]
enum Event {
    ReadFile(String),
    ReadBytes(String),
    MkdirAll(String),
    WriteFile(String, usize),
    RemoveFile(String),
    EnvRead(String),
    FetchText(String),
    FetchBytes(String),
    TempDir(String),
    Unzip(usize, String),
}

impl Event {
    fn strings(&self) -> Vec<&str> {
        match self {
            Event::ReadFile(s)
            | Event::ReadBytes(s)
            | Event::MkdirAll(s)
            | Event::RemoveFile(s)
            | Event::EnvRead(s)
            | Event::FetchText(s)
            | Event::FetchBytes(s)
            | Event::TempDir(s) => vec![s],
            Event::WriteFile(s, _) | Event::Unzip(_, s) => vec![s],
        }
    }

    fn map_strings(&self, f: &dyn Fn(&str) -> String) -> Event {
        let m = |s: &str| f(s);
        match self {
            Event::ReadFile(s) => Event::ReadFile(m(s)),
            Event::ReadBytes(s) => Event::ReadBytes(m(s)),
            Event::MkdirAll(s) => Event::MkdirAll(m(s)),
            Event::WriteFile(s, n) => Event::WriteFile(m(s), *n),
            Event::RemoveFile(s) => Event::RemoveFile(m(s)),
            Event::EnvRead(s) => Event::EnvRead(m(s)),
            Event::FetchText(s) => Event::FetchText(m(s)),
            Event::FetchBytes(s) => Event::FetchBytes(m(s)),
            Event::TempDir(s) => Event::TempDir(m(s)),
            Event::Unzip(n, s) => Event::Unzip(*n, m(s)),
        }
    }
}

/// Normalize scheduling-only fields (temp directory names) so logs from
/// different execution drivers can be compared for equivalence.
fn normalize_log(log: &[Event]) -> Vec<Event> {
    let mut temps: Vec<String> = Vec::new();
    for event in log {
        for s in event.strings() {
            if let Some(rest) = s.strip_prefix("/virtual-tmp/")
                && let Some(dir) = rest.split('/').next()
            {
                let dir = format!("/virtual-tmp/{dir}");
                if !temps.contains(&dir) {
                    temps.push(dir);
                }
            }
        }
    }
    log.iter()
        .map(|event| {
            event.map_strings(&|s| {
                let mut out = s.to_string();
                for (index, dir) in temps.iter().enumerate() {
                    out = out.replace(dir.as_str(), &format!("/virtual-tmp/TMP{index}"));
                }
                out
            })
        })
        .collect()
}

#[derive(Default)]
struct FakeHost {
    files: BTreeMap<String, Vec<u8>>,
    env: BTreeMap<String, String>,
    http_text: BTreeMap<String, String>,
    http_bytes: BTreeMap<String, Vec<u8>>,
    temp_counter: usize,
    log: Vec<Event>,
}

#[derive(Clone, Default)]
struct RecordingCaps {
    host: Arc<Mutex<FakeHost>>,
}

impl RecordingCaps {
    fn log(&self) -> Vec<Event> {
        self.host.lock().unwrap().log.clone()
    }
}

fn not_found(path: &Path) -> pklr::Error {
    pklr::Error::Io(
        path.to_path_buf(),
        std::io::Error::from(std::io::ErrorKind::NotFound),
    )
}

impl EvalCapabilities for RecordingCaps {
    fn read_to_string<'a>(&'a mut self, path: &'a Path) -> BoxFuture<'a, pklr::Result<String>> {
        let key = path.display().to_string();
        let mut host = self.host.lock().unwrap();
        host.log.push(Event::ReadFile(key.clone()));
        let bytes = host.files.get(&key).cloned();
        let path = path.to_path_buf();
        Box::pin(async move {
            let bytes = bytes.ok_or_else(|| not_found(&path))?;
            String::from_utf8(bytes).map_err(|error| pklr::Error::Eval(error.to_string()))
        })
    }

    fn path_exists<'a>(&'a mut self, path: &'a Path) -> BoxFuture<'a, pklr::Result<bool>> {
        let key = path.display().to_string();
        let exists = self.host.lock().unwrap().files.contains_key(&key);
        Box::pin(async move { Ok(exists) })
    }

    fn canonicalize<'a>(&'a mut self, path: &'a Path) -> BoxFuture<'a, pklr::Result<PathBuf>> {
        let path = path.to_path_buf();
        Box::pin(async move { Ok(path) })
    }

    fn read_bytes<'a>(&'a mut self, path: &'a Path) -> BoxFuture<'a, pklr::Result<Vec<u8>>> {
        let key = path.display().to_string();
        let mut host = self.host.lock().unwrap();
        host.log.push(Event::ReadBytes(key.clone()));
        let bytes = host.files.get(&key).cloned();
        let path = path.to_path_buf();
        Box::pin(async move { bytes.ok_or_else(|| not_found(&path)) })
    }

    fn create_dir_all<'a>(&'a mut self, path: &'a Path) -> BoxFuture<'a, pklr::Result<()>> {
        let key = path.display().to_string();
        self.host.lock().unwrap().log.push(Event::MkdirAll(key));
        Box::pin(async { Ok(()) })
    }

    fn write_atomic<'a>(
        &'a mut self,
        path: &'a Path,
        bytes: &'a [u8],
    ) -> BoxFuture<'a, pklr::Result<()>> {
        let key = path.display().to_string();
        let mut host = self.host.lock().unwrap();
        host.log.push(Event::WriteFile(key.clone(), bytes.len()));
        host.files.insert(key, bytes.to_vec());
        Box::pin(async { Ok(()) })
    }

    fn remove_file<'a>(&'a mut self, path: &'a Path) -> BoxFuture<'a, pklr::Result<()>> {
        let key = path.display().to_string();
        let mut host = self.host.lock().unwrap();
        host.log.push(Event::RemoveFile(key.clone()));
        host.files.remove(&key);
        Box::pin(async { Ok(()) })
    }

    fn extract_zip<'a>(
        &'a mut self,
        bytes: Vec<u8>,
        destination: &'a Path,
    ) -> BoxFuture<'a, pklr::Result<()>> {
        let dest = destination.display().to_string();
        let mut host = self.host.lock().unwrap();
        host.log.push(Event::Unzip(bytes.len(), dest.clone()));
        let mut archive = zip::ZipArchive::new(std::io::Cursor::new(bytes))
            .expect("generated zips must parse");
        for index in 0..archive.len() {
            let mut entry = archive.by_index(index).expect("zip entry");
            let mut content = Vec::new();
            std::io::Read::read_to_end(&mut entry, &mut content).expect("zip entry body");
            let name = entry.name().to_string();
            host.files.insert(format!("{dest}/{name}"), content);
        }
        Box::pin(async { Ok(()) })
    }

    fn read_env<'a>(&'a mut self, name: &'a str) -> BoxFuture<'a, pklr::Result<Option<String>>> {
        let mut host = self.host.lock().unwrap();
        host.log.push(Event::EnvRead(name.to_string()));
        let value = host.env.get(name).cloned();
        Box::pin(async move { Ok(value) })
    }

    fn fetch_text<'a>(&'a mut self, url: &'a str) -> BoxFuture<'a, pklr::Result<String>> {
        let mut host = self.host.lock().unwrap();
        host.log.push(Event::FetchText(url.to_string()));
        let source = host.http_text.get(url).cloned();
        let url = url.to_string();
        Box::pin(async move { source.ok_or(pklr::Error::ImportNotFound(url)) })
    }

    fn fetch_bytes<'a>(&'a mut self, url: &'a str) -> BoxFuture<'a, pklr::Result<Vec<u8>>> {
        let mut host = self.host.lock().unwrap();
        host.log.push(Event::FetchBytes(url.to_string()));
        let bytes = host.http_bytes.get(url).cloned();
        let url = url.to_string();
        Box::pin(async move { bytes.ok_or(pklr::Error::ImportNotFound(url)) })
    }

    fn temp_dir<'a>(&'a mut self, prefix: &'a str) -> BoxFuture<'a, pklr::Result<PathBuf>> {
        let mut host = self.host.lock().unwrap();
        host.log.push(Event::TempDir(prefix.to_string()));
        let counter = host.temp_counter;
        host.temp_counter += 1;
        let dir = PathBuf::from(format!("/virtual-tmp/{prefix}-{counter}"));
        Box::pin(async move { Ok(dir) })
    }

    fn glob<'a>(
        &'a mut self,
        _base: &'a Path,
        _pattern: &'a str,
    ) -> BoxFuture<'a, pklr::Result<Vec<PathBuf>>> {
        Box::pin(async { Ok(Vec::new()) })
    }
}
