//! Versioned metadata + content-addressed blob storage for downloaded packages.
//!
//! Every cache entry is a JSON metadata document (`packages/<fnv>.json`) that
//! points at a blob in `blobs/`. Writes are staged through uniquely named temp
//! files and committed with rename, so a crash or an injected failure at any
//! single boundary leaves the previous entry intact. Stale temp files are
//! reclaimed the next time the cache is opened.

use std::collections::HashMap;
use std::io::Write;
use std::path::{Component, Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use crate::error::{Error, Result};

const METADATA_VERSION: u32 = 1;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CommitStage {
    WriteMetadata,
    WriteBlob,
    Flush,
    Rename,
    SyncDir,
}

impl CommitStage {
    fn as_str(self) -> &'static str {
        match self {
            CommitStage::WriteMetadata => "write-metadata",
            CommitStage::WriteBlob => "write-blob",
            CommitStage::Flush => "flush",
            CommitStage::Rename => "rename",
            CommitStage::SyncDir => "sync-dir",
        }
    }
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf29ce484222325_u64;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

/// Stable, filesystem-safe identity for a cached package URL + extension.
pub fn entry_key(url: &str, extension: &str) -> String {
    format!("{:016x}", fnv1a64(format!("{extension}\u{0}{url}").as_bytes()))
}

fn blob_key(bytes: &[u8]) -> String {
    format!("{:016x}", fnv1a64(bytes))
}

fn safe_hex_name(name: &str) -> bool {
    name.len() == 16 && name.bytes().all(|byte| byte.is_ascii_hexdigit())
}

/// Resolve `path` strictly inside `root`. No symlinks are followed: containment
/// is decided from lexical components alone so the rule works on in-memory and
/// real filesystems alike.
fn contained_path(root: &Path, path: &Path) -> Result<PathBuf> {
    let mut joined = PathBuf::new();
    for component in path.components() {
        match component {
            Component::Normal(part) => joined.push(part),
            Component::CurDir => {}
            Component::ParentDir | Component::RootDir | Component::Prefix(_) => {
                return Err(Error::Eval(format!(
                    "cache path escapes the cache directory: {}",
                    path.display()
                )));
            }
        }
    }
    Ok(root.join(joined))
}

/// A handle to an open temp file. Flushing writes buffered data, syncing fsyncs
/// it, and renaming publishes it at its final location.
pub trait CacheTempSink: Write {
    fn flush_data(&mut self) -> std::io::Result<()>;
    fn sync_data(&mut self) -> std::io::Result<()>;
    fn rename_to(self: Box<Self>, destination: &Path) -> std::io::Result<()>;
}

/// Restricted filesystem seam used by the package cache. The evaluator ships a
/// real implementation; tests substitute controllable or fault-injecting ones
/// without touching the public evaluation API.
pub trait CacheFs: Send + Sync {
    fn read(&self, path: &Path) -> std::io::Result<Vec<u8>>;
    fn exists(&self, path: &Path) -> bool;
    fn create_dir_all(&self, path: &Path) -> std::io::Result<()>;
    fn list_dir(&self, path: &Path) -> std::io::Result<Vec<String>>;
    fn remove_file(&self, path: &Path) -> std::io::Result<()>;
    fn sync_dir(&self, path: &Path) -> std::io::Result<()>;
    fn create_temp(&self, path: &Path) -> std::io::Result<Box<dyn CacheTempSink>>;

    /// Stage at which the next mutating operation fails. Tests override this;
    /// the real filesystem never fails on demand.
    fn injected_failure(&self, _stage: CommitStage) -> Option<std::io::ErrorKind> {
        None
    }
}

/// Filesystem-backed [`CacheFs`] used by the real evaluator.
pub struct RealCacheFs;

struct RealTempSink {
    file: std::fs::File,
    temp_path: PathBuf,
}

impl Write for RealTempSink {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.file.write(buf)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.file.flush()
    }
}

impl CacheTempSink for RealTempSink {
    fn flush_data(&mut self) -> std::io::Result<()> {
        self.file.flush()
    }

    fn sync_data(&mut self) -> std::io::Result<()> {
        self.file.sync_all()
    }

    fn rename_to(self: Box<Self>, destination: &Path) -> std::io::Result<()> {
        let RealTempSink { file, temp_path } = *self;
        drop(file);
        match std::fs::rename(&temp_path, destination) {
            Ok(()) => Ok(()),
            // Windows rename cannot replace; remove and retry once.
            Err(_) if destination.exists() => {
                std::fs::remove_file(destination)?;
                std::fs::rename(&temp_path, destination)
            }
            Err(error) => Err(error),
        }
    }
}

impl CacheFs for RealCacheFs {
    fn read(&self, path: &Path) -> std::io::Result<Vec<u8>> {
        std::fs::read(path)
    }

    fn exists(&self, path: &Path) -> bool {
        path.exists()
    }

    fn create_dir_all(&self, path: &Path) -> std::io::Result<()> {
        std::fs::create_dir_all(path)
    }

    fn list_dir(&self, path: &Path) -> std::io::Result<Vec<String>> {
        let mut names = Vec::new();
        for entry in std::fs::read_dir(path)? {
            names.push(entry?.file_name().to_string_lossy().into_owned());
        }
        Ok(names)
    }

    fn remove_file(&self, path: &Path) -> std::io::Result<()> {
        match std::fs::remove_file(path) {
            Ok(()) => Ok(()),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(error) => Err(error),
        }
    }

    fn sync_dir(&self, path: &Path) -> std::io::Result<()> {
        #[cfg(windows)]
        {
            let _ = path;
            Ok(())
        }
        #[cfg(not(windows))]
        {
            use std::os::unix::fs::OpenOptionsExt;
            let dir = std::fs::OpenOptions::new()
                .read(true)
                .custom_flags(O_DIRECTORY_FLAG)
                .open(path)?;
            dir.sync_all()
        }
    }

    fn create_temp(&self, path: &Path) -> std::io::Result<Box<dyn CacheTempSink>> {
        use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let file_name = path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("cache");
        let temp_path = path.with_file_name(format!(
            ".{file_name}.tmp-{}-{}",
            std::process::id(),
            COUNTER.fetch_add(1, AtomicOrdering::Relaxed)
        ));
        let file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temp_path)?;
        Ok(Box::new(RealTempSink { file, temp_path }))
    }
}

#[cfg(not(windows))]
const O_DIRECTORY_FLAG: i32 = 0o0200000;

struct EntryMetadata {
    version: u32,
    url: String,
    extension: String,
    digest: String,
}

impl EntryMetadata {
    fn to_json_bytes(&self) -> Vec<u8> {
        serde_json::json!({
            "version": self.version,
            "url": self.url,
            "extension": self.extension,
            "digest": self.digest,
        })
        .to_string()
        .into_bytes()
    }

    fn from_json_slice(raw: &[u8]) -> std::result::Result<Self, String> {
        let value: serde_json::Value =
            serde_json::from_slice(raw).map_err(|error| error.to_string())?;
        let object = value
            .as_object()
            .ok_or_else(|| "metadata is not a JSON object".to_string())?;
        let version = object
            .get("version")
            .and_then(|value| value.as_u64())
            .ok_or_else(|| "missing numeric version".to_string())?;
        let url = object
            .get("url")
            .and_then(|value| value.as_str())
            .ok_or_else(|| "missing string url".to_string())?;
        let extension = object
            .get("extension")
            .and_then(|value| value.as_str())
            .ok_or_else(|| "missing string extension".to_string())?;
        let digest = object
            .get("digest")
            .and_then(|value| value.as_str())
            .ok_or_else(|| "missing string digest".to_string())?;
        Ok(Self {
            version: version as u32,
            url: url.to_string(),
            extension: extension.to_string(),
            digest: digest.to_string(),
        })
    }
}

/// Why a cache lookup refused to serve an entry.
#[derive(Debug, PartialEq, Eq)]
pub enum LookupRejection {
    /// No metadata for the key.
    Missing,
    /// Metadata exists but cannot be parsed.
    CorruptMetadata(String),
    /// Metadata advertises a format this build cannot read.
    UnsupportedVersion(u32),
    /// Metadata does not describe the URL/extension that was requested.
    IdentityMismatch,
    /// Referenced blob is absent or stored outside the cache tree.
    UnsafeBlobPath,
    /// Blob digest differs from metadata (including truncated blobs).
    DigestMismatch,
}

impl LookupRejection {
    fn to_error(&self) -> Error {
        match self {
            LookupRejection::Missing => Error::Eval("cache entry missing".to_string()),
            LookupRejection::CorruptMetadata(reason) => {
                Error::Eval(format!("corrupt cache metadata: {reason}"))
            }
            LookupRejection::UnsupportedVersion(version) => {
                Error::Eval(format!("unsupported cache metadata version: {version}"))
            }
            LookupRejection::IdentityMismatch => {
                Error::Eval("cache metadata identity mismatch".to_string())
            }
            LookupRejection::UnsafeBlobPath => {
                Error::Eval("cache blob path escapes the cache directory".to_string())
            }
            LookupRejection::DigestMismatch => {
                Error::Eval("cache blob digest mismatch".to_string())
            }
        }
    }
}

/// Versioned package cache rooted at a directory and mediated by a [`CacheFs`].
pub struct PackageCacheStore {
    root: PathBuf,
    fs: Arc<dyn CacheFs>,
    temp_counter: AtomicU64,
}

fn io_error(path: &Path, error: std::io::Error) -> Error {
    Error::Io(path.to_path_buf(), error)
}

impl PackageCacheStore {
    /// Open a cache rooted at `root`, creating its layout and reclaiming any
    /// temp files left behind by interrupted writes.
    pub fn open(root: impl Into<PathBuf>, fs: Arc<dyn CacheFs>) -> Result<Arc<Self>> {
        let root = root.into();
        let packages = root.join("packages");
        let blobs = root.join("blobs");
        fs.create_dir_all(&packages).map_err(|error| io_error(&packages, error))?;
        fs.create_dir_all(&blobs).map_err(|error| io_error(&blobs, error))?;
        let store = Arc::new(Self {
            root,
            fs,
            temp_counter: AtomicU64::new(0),
        });
        store.sweep_temp_files()?;
        Ok(store)
    }

    fn packages_dir(&self) -> PathBuf {
        self.root.join("packages")
    }

    fn blobs_dir(&self) -> PathBuf {
        self.root.join("blobs")
    }

    fn metadata_path(&self, key: &str) -> Result<PathBuf> {
        contained_path(&self.packages_dir(), Path::new(&format!("{key}.json")))
    }

    fn blob_path(&self, name: &str) -> Result<PathBuf> {
        if !safe_hex_name(name) {
            return Err(Error::Eval(
                "cache blob path escapes the cache directory".to_string(),
            ));
        }
        contained_path(&self.blobs_dir(), Path::new(name))
    }

    /// Remove temp files left behind by crashed or failed commits.
    fn sweep_temp_files(&self) -> Result<()> {
        for directory in [self.packages_dir(), self.blobs_dir()] {
            let names = match self.fs.list_dir(&directory) {
                Ok(names) => names,
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => continue,
                Err(error) => return Err(io_error(&directory, error)),
            };
            for name in names {
                if name.starts_with('.') && name.contains(".tmp-") {
                    let _ = self.fs.remove_file(&directory.join(&name));
                }
            }
        }
        Ok(())
    }

    /// Serve an entry when it is complete and valid.
    pub fn lookup(&self, url: &str, extension: &str) -> Result<Option<Vec<u8>>> {
        let key = entry_key(url, extension);
        let metadata_path = self.metadata_path(&key)?;
        let raw = match self.fs.read(&metadata_path) {
            Ok(raw) => raw,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(error) => return Err(io_error(&metadata_path, error)),
        };
        let rejection = match self.validate_metadata(&raw, url, extension) {
            Ok(metadata) => match self.serve_blob(&metadata) {
                Ok(bytes) => return Ok(Some(bytes)),
                Err(rejection) => rejection,
            },
            Err(rejection) => rejection,
        };
        // A half-written or corrupt entry is never a hit; drop it so the next
        // online fetch can repopulate cleanly.
        if !matches!(rejection, LookupRejection::Missing) {
            let _ = self.fs.remove_file(&metadata_path);
        }
        Err(rejection.to_error())
    }

    fn validate_metadata(
        &self,
        raw: &[u8],
        url: &str,
        extension: &str,
    ) -> std::result::Result<EntryMetadata, LookupRejection> {
        let metadata = EntryMetadata::from_json_slice(raw)
            .map_err(LookupRejection::CorruptMetadata)?;
        if metadata.version != METADATA_VERSION {
            return Err(LookupRejection::UnsupportedVersion(metadata.version));
        }
        if metadata.url != url || metadata.extension != extension {
            return Err(LookupRejection::IdentityMismatch);
        }
        Ok(metadata)
    }

    fn serve_blob(
        &self,
        metadata: &EntryMetadata,
    ) -> std::result::Result<Vec<u8>, LookupRejection> {
        let blob_path = self
            .blob_path(&metadata.digest)
            .map_err(|_| LookupRejection::UnsafeBlobPath)?;
        let bytes = match self.fs.read(&blob_path) {
            Ok(bytes) => bytes,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                return Err(LookupRejection::Missing);
            }
            Err(_) => return Err(LookupRejection::UnsafeBlobPath),
        };
        if blob_key(&bytes) != metadata.digest {
            return Err(LookupRejection::DigestMismatch);
        }
        Ok(bytes)
    }

    /// Publish `bytes` for `url`. Each disk boundary is a named stage so fault
    /// injection can target it precisely.
    pub fn commit(&self, url: &str, extension: &str, bytes: &[u8]) -> Result<()> {
        let digest = blob_key(bytes);
        let blob_path = self.blob_path(&digest)?;
        self.stage_commit(CommitStage::WriteBlob, &blob_path, bytes)?;

        let metadata = EntryMetadata {
            version: METADATA_VERSION,
            url: url.to_string(),
            extension: extension.to_string(),
            digest,
        };
        let metadata_bytes = metadata.to_json_bytes();
        let key = entry_key(url, extension);
        let metadata_path = self.metadata_path(&key)?;
        self.stage_commit(CommitStage::WriteMetadata, &metadata_path, &metadata_bytes)
    }

    fn stage_commit(&self, stage: CommitStage, final_path: &Path, bytes: &[u8]) -> Result<()> {
        let parent = final_path
            .parent()
            .ok_or_else(|| Error::Eval("cache path has no parent directory".to_string()))?;
        self.fs
            .create_dir_all(parent)
            .map_err(|error| io_error(parent, error))?;

        let temp_path = self.unique_temp_path(final_path);
        let mut sink = self
            .fs
            .create_temp(&temp_path)
            .map_err(|error| io_error(&temp_path, error))?;

        if let Some(kind) = self.fs.injected_failure(stage) {
            drop(sink);
            let _ = self.fs.remove_file(&temp_path);
            return Err(stage_error(stage, &temp_path, kind));
        }
        if let Err(error) = sink.write_all(bytes) {
            return Err(self.abandon_temp(&temp_path, sink, stage, error));
        }

        if let Some(kind) = self.fs.injected_failure(CommitStage::Flush) {
            drop(sink);
            let _ = self.fs.remove_file(&temp_path);
            return Err(stage_error(CommitStage::Flush, &temp_path, kind));
        }
        if let Err(error) = sink.flush_data() {
            return Err(self.abandon_temp(&temp_path, sink, CommitStage::Flush, error));
        }
        if let Err(error) = sink.sync_data() {
            return Err(self.abandon_temp(&temp_path, sink, CommitStage::Flush, error));
        }

        if let Some(kind) = self.fs.injected_failure(CommitStage::Rename) {
            drop(sink);
            let _ = self.fs.remove_file(&temp_path);
            return Err(stage_error(CommitStage::Rename, &temp_path, kind));
        }
        if let Err(error) = sink.rename_to(final_path) {
            // Sink ownership moved into rename; if it failed before publishing,
            // any leftover temp file is reclaimed on the next cache open.
            return Err(io_error(final_path, error));
        }

        if let Some(kind) = self.fs.injected_failure(CommitStage::SyncDir) {
            return Err(stage_error(CommitStage::SyncDir, parent, kind));
        }
        self.fs
            .sync_dir(parent)
            .map_err(|error| io_error(parent, error))?;
        Ok(())
    }

    fn abandon_temp(
        &self,
        temp_path: &Path,
        sink: Box<dyn CacheTempSink>,
        stage: CommitStage,
        error: std::io::Error,
    ) -> Error {
        drop(sink);
        let _ = self.fs.remove_file(temp_path);
        let _ = stage;
        io_error(temp_path, error)
    }

    fn unique_temp_path(&self, final_path: &Path) -> PathBuf {
        let file_name = final_path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("cache");
        let counter = self.temp_counter.fetch_add(1, Ordering::Relaxed);
        final_path.with_file_name(format!(
            ".{file_name}.tmp-{}-{counter}",
            std::process::id()
        ))
    }

    /// Best-effort deletion of a metadata entry (blobs are content-addressed
    /// and shared, so they are left for later reclamation).
    pub(crate) fn evict_metadata(&self, url: &str, extension: &str) {
        if let Ok(path) = self.metadata_path(&entry_key(url, extension)) {
            let _ = self.fs.remove_file(&path);
        }
    }
}

fn stage_error(stage: CommitStage, path: &Path, kind: std::io::ErrorKind) -> Error {
    Error::Io(path.to_path_buf(), std::io::Error::new(
        kind,
        format!("injected cache failure at {}", stage.as_str()),
    ))
}

#[cfg(any(test, feature = "native-io"))]
pub mod testing {
    //! Controllable cache filesystems for tests. Not a stability boundary.

    use super::*;

    type EntryMap = HashMap<PathBuf, Vec<u8>>;

    /// In-memory [`CacheFs`]: deterministically records every operation in
    /// `events` so tests can compare evaluation trajectories.
    #[derive(Clone, Default)]
    pub struct InMemoryCacheFs {
        entries: Arc<Mutex<EntryMap>>,
        events: Arc<Mutex<Vec<String>>>,
    }

    impl InMemoryCacheFs {
        pub fn new() -> Self {
            Self::default()
        }

        pub fn snapshot(&self) -> EntryMap {
            self.entries.lock().unwrap().clone()
        }

        pub fn put(&self, path: PathBuf, bytes: Vec<u8>) {
            self.entries.lock().unwrap().insert(path, bytes);
        }

        pub fn contains(&self, path: &Path) -> bool {
            self.entries.lock().unwrap().contains_key(path)
        }

        pub fn events(&self) -> Vec<String> {
            self.events.lock().unwrap().clone()
        }

        pub fn clear_events(&self) {
            self.events.lock().unwrap().clear();
        }

        pub fn arc(self) -> Arc<Self> {
            Arc::new(self)
        }

        fn record(&self, event: String) {
            self.events.lock().unwrap().push(event);
        }
    }

    struct MemoryTempSink {
        path: PathBuf,
        buffer: Vec<u8>,
        entries: Arc<Mutex<EntryMap>>,
        events: Arc<Mutex<Vec<String>>>,
        finished: bool,
    }

    impl Write for MemoryTempSink {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            self.buffer.extend_from_slice(buf);
            Ok(buf.len())
        }

        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    impl Drop for MemoryTempSink {
        fn drop(&mut self) {
            if !self.finished {
                self.entries.lock().unwrap().remove(&self.path);
            }
        }
    }

    impl CacheTempSink for MemoryTempSink {
        fn flush_data(&mut self) -> std::io::Result<()> {
            self.events
                .lock()
                .unwrap()
                .push(format!("flush:{}", self.path.display()));
            Ok(())
        }

        fn sync_data(&mut self) -> std::io::Result<()> {
            Ok(())
        }

        fn rename_to(mut self: Box<Self>, destination: &Path) -> std::io::Result<()> {
            self.finished = true;
            let mut entries = self.entries.lock().unwrap();
            entries.remove(&self.path);
            entries.insert(destination.to_path_buf(), self.buffer.clone());
            Ok(())
        }
    }

    impl CacheFs for InMemoryCacheFs {
        fn read(&self, path: &Path) -> std::io::Result<Vec<u8>> {
            self.record(format!("read:{}", path.display()));
            self.entries
                .lock()
                .unwrap()
                .get(path)
                .cloned()
                .ok_or_else(|| std::io::Error::from(std::io::ErrorKind::NotFound))
        }

        fn exists(&self, path: &Path) -> bool {
            self.entries.lock().unwrap().contains_key(path)
        }

        fn create_dir_all(&self, _path: &Path) -> std::io::Result<()> {
            Ok(())
        }

        fn list_dir(&self, path: &Path) -> std::io::Result<Vec<String>> {
            let entries = self.entries.lock().unwrap();
            let prefix = format!("{}/", path.display());
            let mut names = Vec::new();
            for key in entries.keys() {
                let rendered = key.display().to_string();
                if let Some(rest) = rendered.strip_prefix(&prefix)
                    && !rest.contains('/')
                {
                    names.push(rest.to_string());
                }
            }
            Ok(names)
        }

        fn remove_file(&self, path: &Path) -> std::io::Result<()> {
            self.record(format!("remove:{}", path.display()));
            self.entries.lock().unwrap().remove(path);
            Ok(())
        }

        fn sync_dir(&self, path: &Path) -> std::io::Result<()> {
            self.record(format!("sync-dir:{}", path.display()));
            Ok(())
        }

        fn create_temp(&self, path: &Path) -> std::io::Result<Box<dyn CacheTempSink>> {
            self.record(format!("create-temp:{}", path.display()));
            self.entries.lock().unwrap().insert(path.to_path_buf(), Vec::new());
            Ok(Box::new(MemoryTempSink {
                path: path.to_path_buf(),
                buffer: Vec::new(),
                entries: Arc::clone(&self.entries),
                events: Arc::clone(&self.events),
                finished: false,
            }))
        }
    }

    /// Wraps another filesystem and fails one named commit boundary a fixed
    /// number of times. Shared clones observe the same failure arm and counter.
    #[derive(Clone)]
    pub struct FaultInjectingFs {
        inner: Arc<dyn CacheFs>,
        arm: Arc<Mutex<Option<FaultArm>>>,
        fired: Arc<Mutex<Vec<String>>>,
    }

    #[derive(Clone, Copy)]
    struct FaultArm {
        stage: CommitStage,
        remaining: u64,
    }

    impl FaultInjectingFs {
        pub fn new(inner: Arc<dyn CacheFs>) -> Arc<Self> {
            Arc::new(Self {
                inner,
                arm: Arc::new(Mutex::new(None)),
                fired: Arc::new(Mutex::new(Vec::new())),
            })
        }

        /// Fail on the `occurrence`-th crossing of `stage` (1-based), once.
        pub fn fail_once(&self, stage: CommitStage, occurrence: u64) {
            *self.arm.lock().unwrap() = Some(FaultArm {
                stage,
                remaining: occurrence,
            });
        }

        /// Fail every crossing of `stage` until [`disarm`](Self::disarm).
        pub fn fail_always(&self, stage: CommitStage) {
            *self.arm.lock().unwrap() = Some(FaultArm {
                stage,
                remaining: u64::MAX,
            });
        }

        pub fn disarm(&self) {
            *self.arm.lock().unwrap() = None;
        }

        pub fn fired(&self) -> Vec<String> {
            self.fired.lock().unwrap().clone()
        }
    }

    impl CacheFs for FaultInjectingFs {
        fn read(&self, path: &Path) -> std::io::Result<Vec<u8>> {
            self.inner.read(path)
        }

        fn exists(&self, path: &Path) -> bool {
            self.inner.exists(path)
        }

        fn create_dir_all(&self, path: &Path) -> std::io::Result<()> {
            self.inner.create_dir_all(path)
        }

        fn list_dir(&self, path: &Path) -> std::io::Result<Vec<String>> {
            self.inner.list_dir(path)
        }

        fn remove_file(&self, path: &Path) -> std::io::Result<()> {
            self.inner.remove_file(path)
        }

        fn sync_dir(&self, path: &Path) -> std::io::Result<()> {
            self.inner.sync_dir(path)
        }

        fn create_temp(&self, path: &Path) -> std::io::Result<Box<dyn CacheTempSink>> {
            self.inner.create_temp(path)
        }

        fn injected_failure(&self, stage: CommitStage) -> Option<std::io::ErrorKind> {
            let mut arm_guard = self.arm.lock().unwrap();
            if let Some(arm) = *arm_guard
                && arm.stage == stage
            {
                if arm.remaining == u64::MAX {
                    self.fired
                        .lock()
                        .unwrap()
                        .push(stage.as_str().to_string());
                    return Some(std::io::ErrorKind::PermissionDenied);
                }
                if arm.remaining == 1 {
                    self.fired
                        .lock()
                        .unwrap()
                        .push(stage.as_str().to_string());
                    *arm_guard = None;
                    return Some(std::io::ErrorKind::PermissionDenied);
                }
                if let Some(arm) = arm_guard.as_mut() {
                    arm.remaining -= 1;
                }
            }
            None
        }
    }
}

