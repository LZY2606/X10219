use std::future::Future;
use std::path::{Path, PathBuf};
use std::pin::Pin;
#[cfg(feature = "native-io")]
use std::sync::atomic::{AtomicU64, Ordering};
#[cfg(any(feature = "http", feature = "blocking"))]
use std::time::Duration;

use crate::Result;

pub type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + 'a>>;

/// Boundary at which a durable cache write is made to fail.
///
/// This is the narrow dependency-injection seam used by the cache contract
/// tests: embedders never need it, and evaluation semantics do not depend on
/// it. `Blob*` boundaries are only crossed while committing a package blob and
/// `Metadata*` boundaries only while committing its metadata sidecar.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheWriteFault {
    /// `create` of the staging file failed.
    Create,
    /// `write`/`flush` of the staging bytes failed.
    Flush,
    /// `fsync` of the staging file failed.
    FileSync,
    /// `rename` of the staging file to its destination failed.
    Rename,
    /// `fsync` of the destination directory failed (metadata commit only).
    DirSync,
}

/// Host-provided IO for evaluating Pkl modules.
///
/// The default evaluator still uses the native implementation, but embedders
/// can provide their own file, environment, HTTP, temp-dir, and glob behavior.
pub trait EvalCapabilities: Send + Sync {
    fn read_to_string<'a>(&'a mut self, path: &'a Path) -> BoxFuture<'a, Result<String>>;

    fn path_exists<'a>(&'a mut self, path: &'a Path) -> BoxFuture<'a, Result<bool>>;

    fn canonicalize<'a>(&'a mut self, path: &'a Path) -> BoxFuture<'a, Result<PathBuf>>;

    fn read_bytes<'a>(&'a mut self, path: &'a Path) -> BoxFuture<'a, Result<Vec<u8>>> {
        Box::pin(async move {
            std::fs::read(path).map_err(|error| crate::Error::Io(path.to_path_buf(), error))
        })
    }

    fn create_dir_all<'a>(&'a mut self, path: &'a Path) -> BoxFuture<'a, Result<()>> {
        Box::pin(async move {
            std::fs::create_dir_all(path)
                .map_err(|error| crate::Error::Io(path.to_path_buf(), error))
        })
    }

    fn write_atomic<'a>(
        &'a mut self,
        path: &'a Path,
        bytes: &'a [u8],
    ) -> BoxFuture<'a, Result<()>> {
        Box::pin(async move {
            crate::eval::write_atomic(path, bytes)
                .map_err(|error| crate::Error::Io(path.to_path_buf(), error))
        })
    }

    fn remove_file<'a>(&'a mut self, path: &'a Path) -> BoxFuture<'a, Result<()>> {
        Box::pin(async move {
            std::fs::remove_file(path).map_err(|error| crate::Error::Io(path.to_path_buf(), error))
        })
    }

    /// List directory entries. Defaults to the native filesystem.
    fn list_dir<'a>(&'a mut self, path: &'a Path) -> BoxFuture<'a, Result<Vec<PathBuf>>> {
        Box::pin(async move {
            match std::fs::read_dir(path) {
                Ok(entries) => {
                    let mut paths = Vec::new();
                    for entry in entries {
                        let entry = entry.map_err(|error| {
                            crate::Error::Io(path.to_path_buf(), error)
                        })?;
                        paths.push(entry.path());
                    }
                    Ok(paths)
                }
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(Vec::new()),
                Err(error) => Err(crate::Error::Io(path.to_path_buf(), error)),
            }
        })
    }

    /// Inject a failure at a durable-write boundary. The default
    /// implementation never fails; test capabilities override it.
    fn cache_write_fault(&mut self, _fault: CacheWriteFault) -> Result<()> {
        Ok(())
    }

    /// Commit `bytes` to `destination` via a staging file in `staging_dir`.
    ///
    /// The sequence is: ensure the destination parent exists, create a
    /// staging file, write + flush + fsync it, rename it into place, and
    /// fsync the destination directory when `sync_dir` is set. Each boundary
    /// consults [`EvalCapabilities::cache_write_fault`] so cache write
    /// failures can be injected without changing public evaluation semantics.
    fn durable_write<'a>(
        &'a mut self,
        destination: &'a Path,
        staging_dir: &'a Path,
        bytes: &'a [u8],
        sync_dir: bool,
    ) -> BoxFuture<'a, Result<()>> {
        Box::pin(async move {
            durable_write_default(self, destination, staging_dir, bytes, sync_dir).await
        })
    }

    #[cfg(feature = "package-zip-core")]
    fn extract_zip<'a>(
        &'a mut self,
        bytes: Vec<u8>,
        destination: &'a Path,
    ) -> BoxFuture<'a, Result<()>> {
        Box::pin(async move { extract_zip_sync(bytes, destination) })
    }

    fn read_env<'a>(&'a mut self, name: &'a str) -> BoxFuture<'a, Result<Option<String>>>;

    fn fetch_text<'a>(&'a mut self, url: &'a str) -> BoxFuture<'a, Result<String>>;

    fn fetch_bytes<'a>(&'a mut self, url: &'a str) -> BoxFuture<'a, Result<Vec<u8>>>;

    #[cfg(feature = "http")]
    fn set_http_client(&mut self, client: reqwest::Client) -> Result<()> {
        drop(client);
        Err(crate::Error::Unsupported(
            "this evaluator's capabilities do not accept a reqwest HTTP client".to_string(),
        ))
    }

    fn temp_dir<'a>(&'a mut self, prefix: &'a str) -> BoxFuture<'a, Result<PathBuf>>;

    fn glob<'a>(
        &'a mut self,
        base: &'a Path,
        pattern: &'a str,
    ) -> BoxFuture<'a, Result<Vec<PathBuf>>>;
}

#[cfg(feature = "native-io")]
#[derive(Debug, Clone)]
pub struct NativeCapabilities {
    #[cfg(feature = "http")]
    http_client: reqwest::Client,
}

/// Synchronous host IO used by the blocking evaluator.
#[cfg(feature = "blocking")]
#[derive(Debug, Clone)]
pub struct BlockingCapabilities {
    http_agent: ureq::Agent,
}

#[cfg(feature = "blocking")]
impl BlockingCapabilities {
    pub fn new() -> Self {
        Self {
            http_agent: default_blocking_http_agent(),
        }
    }

    pub fn with_http_agent(http_agent: ureq::Agent) -> Self {
        ensure_blocking_crypto_provider();
        Self { http_agent }
    }
}

#[cfg(feature = "blocking")]
fn ensure_blocking_crypto_provider() {
    if rustls::crypto::CryptoProvider::get_default().is_none() {
        let _ = rustls::crypto::aws_lc_rs::default_provider().install_default();
    }
}

#[cfg(feature = "blocking")]
fn default_blocking_http_agent() -> ureq::Agent {
    ensure_blocking_crypto_provider();
    ureq::Agent::config_builder()
        .timeout_connect(Some(Duration::from_secs(10)))
        .timeout_global(Some(Duration::from_secs(30)))
        .build()
        .into()
}

#[cfg(feature = "blocking")]
const BLOCKING_HTTP_BODY_LIMIT: u64 = 64 * 1024 * 1024;

#[cfg(feature = "blocking")]
impl Default for BlockingCapabilities {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(feature = "native-io")]
impl NativeCapabilities {
    pub fn new() -> Self {
        Self {
            #[cfg(feature = "http")]
            http_client: default_http_client(),
        }
    }

    #[cfg(feature = "http")]
    pub fn with_http_client(http_client: reqwest::Client) -> Self {
        Self { http_client }
    }
}

#[cfg(feature = "native-io")]
impl Default for NativeCapabilities {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(all(feature = "native-io", feature = "http"))]
fn default_http_client() -> reqwest::Client {
    reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(10))
        .timeout(Duration::from_secs(30))
        .build()
        .expect("default reqwest client should build")
}

#[cfg(feature = "native-io")]
impl EvalCapabilities for NativeCapabilities {
    fn read_to_string<'a>(&'a mut self, path: &'a Path) -> BoxFuture<'a, Result<String>> {
        Box::pin(async move {
            #[cfg(feature = "async")]
            let result = if tokio::runtime::Handle::try_current().is_ok() {
                tokio::fs::read_to_string(path).await
            } else {
                std::fs::read_to_string(path)
            };
            #[cfg(not(feature = "async"))]
            let result = std::fs::read_to_string(path);
            result.map_err(|error| crate::Error::Io(path.to_path_buf(), error))
        })
    }

    fn path_exists<'a>(&'a mut self, path: &'a Path) -> BoxFuture<'a, Result<bool>> {
        Box::pin(async move {
            #[cfg(feature = "async")]
            let exists = if tokio::runtime::Handle::try_current().is_ok() {
                tokio::fs::try_exists(path).await.unwrap_or(false)
            } else {
                path.exists()
            };
            #[cfg(not(feature = "async"))]
            let exists = path.exists();
            Ok(exists)
        })
    }

    fn canonicalize<'a>(&'a mut self, path: &'a Path) -> BoxFuture<'a, Result<PathBuf>> {
        Box::pin(async move {
            #[cfg(feature = "async")]
            let result = if tokio::runtime::Handle::try_current().is_ok() {
                tokio::fs::canonicalize(path).await
            } else {
                path.canonicalize()
            };
            #[cfg(not(feature = "async"))]
            let result = path.canonicalize();
            result.map_err(|error| crate::Error::Io(path.to_path_buf(), error))
        })
    }

    fn read_bytes<'a>(&'a mut self, path: &'a Path) -> BoxFuture<'a, Result<Vec<u8>>> {
        Box::pin(async move {
            #[cfg(feature = "async")]
            let result = if tokio::runtime::Handle::try_current().is_ok() {
                tokio::fs::read(path).await
            } else {
                std::fs::read(path)
            };
            #[cfg(not(feature = "async"))]
            let result = std::fs::read(path);
            result.map_err(|error| crate::Error::Io(path.to_path_buf(), error))
        })
    }

    fn create_dir_all<'a>(&'a mut self, path: &'a Path) -> BoxFuture<'a, Result<()>> {
        Box::pin(async move {
            #[cfg(feature = "async")]
            let result = if tokio::runtime::Handle::try_current().is_ok() {
                tokio::fs::create_dir_all(path).await
            } else {
                std::fs::create_dir_all(path)
            };
            #[cfg(not(feature = "async"))]
            let result = std::fs::create_dir_all(path);
            result.map_err(|error| crate::Error::Io(path.to_path_buf(), error))
        })
    }

    fn write_atomic<'a>(
        &'a mut self,
        path: &'a Path,
        bytes: &'a [u8],
    ) -> BoxFuture<'a, Result<()>> {
        Box::pin(async move {
            #[cfg(feature = "async")]
            {
                if tokio::runtime::Handle::try_current().is_err() {
                    return crate::eval::write_atomic(path, bytes)
                        .map_err(|error| crate::Error::Io(path.to_path_buf(), error));
                }
                let path = path.to_path_buf();
                let task_path = path.clone();
                let bytes = bytes.to_vec();
                tokio::task::spawn_blocking(move || crate::eval::write_atomic(&task_path, &bytes))
                    .await
                    .map_err(|error| {
                        crate::Error::Eval(format!("atomic write task failed: {error}"))
                    })?
                    .map_err(|error| crate::Error::Io(path, error))
            }
            #[cfg(not(feature = "async"))]
            crate::eval::write_atomic(path, bytes)
                .map_err(|error| crate::Error::Io(path.to_path_buf(), error))
        })
    }

    fn remove_file<'a>(&'a mut self, path: &'a Path) -> BoxFuture<'a, Result<()>> {
        Box::pin(async move {
            #[cfg(feature = "async")]
            let result = if tokio::runtime::Handle::try_current().is_ok() {
                tokio::fs::remove_file(path).await
            } else {
                std::fs::remove_file(path)
            };
            #[cfg(not(feature = "async"))]
            let result = std::fs::remove_file(path);
            result.map_err(|error| crate::Error::Io(path.to_path_buf(), error))
        })
    }

    #[cfg(feature = "package-zip-core")]
    fn extract_zip<'a>(
        &'a mut self,
        bytes: Vec<u8>,
        destination: &'a Path,
    ) -> BoxFuture<'a, Result<()>> {
        Box::pin(async move {
            #[cfg(feature = "async")]
            {
                if tokio::runtime::Handle::try_current().is_err() {
                    return extract_zip_sync(bytes, destination);
                }
                let destination = destination.to_path_buf();
                tokio::task::spawn_blocking(move || extract_zip_sync(bytes, &destination))
                    .await
                    .map_err(|error| {
                        crate::Error::Eval(format!("package extraction task failed: {error}"))
                    })?
            }
            #[cfg(not(feature = "async"))]
            extract_zip_sync(bytes, destination)
        })
    }

    fn list_dir<'a>(&'a mut self, path: &'a Path) -> BoxFuture<'a, Result<Vec<PathBuf>>> {
        Box::pin(async move {
            #[cfg(feature = "async")]
            {
                if tokio::runtime::Handle::try_current().is_err() {
                    return list_dir_sync(path);
                }
                let path = path.to_path_buf();
                tokio::task::spawn_blocking(move || list_dir_sync(&path))
                    .await
                    .map_err(|error| {
                        crate::Error::Eval(format!("list directory task failed: {error}"))
                    })?
            }
            #[cfg(not(feature = "async"))]
            list_dir_sync(path)
        })
    }

    fn read_env<'a>(&'a mut self, name: &'a str) -> BoxFuture<'a, Result<Option<String>>> {
        Box::pin(async move { Ok(std::env::var(name).ok()) })
    }

    fn fetch_text<'a>(&'a mut self, url: &'a str) -> BoxFuture<'a, Result<String>> {
        #[cfg(feature = "http")]
        let client = self.http_client.clone();
        Box::pin(async move {
            #[cfg(feature = "http")]
            {
                client
                    .get(url)
                    .send()
                    .await
                    .map_err(|error| {
                        crate::Error::Eval(format!("HTTP fetch failed for {url}: {error}"))
                    })?
                    .error_for_status()
                    .map_err(|error| {
                        if error.status() == Some(reqwest::StatusCode::NOT_FOUND) {
                            crate::Error::ImportNotFound(url.to_string())
                        } else {
                            crate::Error::Eval(format!("HTTP error for {url}: {error}"))
                        }
                    })?
                    .text()
                    .await
                    .map_err(|error| {
                        crate::Error::Eval(format!("HTTP read failed for {url}: {error}"))
                    })
            }
            #[cfg(not(feature = "http"))]
            {
                Err(crate::Error::Unsupported(format!(
                    "HTTP fetch requires pklr's 'http' feature: {url}"
                )))
            }
        })
    }

    fn fetch_bytes<'a>(&'a mut self, url: &'a str) -> BoxFuture<'a, Result<Vec<u8>>> {
        #[cfg(feature = "http")]
        let client = self.http_client.clone();
        Box::pin(async move {
            #[cfg(feature = "http")]
            {
                let bytes = client
                    .get(url)
                    .send()
                    .await
                    .map_err(|error| {
                        crate::Error::Eval(format!("HTTP fetch failed for {url}: {error}"))
                    })?
                    .error_for_status()
                    .map_err(|error| {
                        if error.status() == Some(reqwest::StatusCode::NOT_FOUND) {
                            crate::Error::ImportNotFound(url.to_string())
                        } else {
                            crate::Error::Eval(format!("HTTP error for {url}: {error}"))
                        }
                    })?
                    .bytes()
                    .await
                    .map_err(|error| {
                        crate::Error::Eval(format!("HTTP read failed for {url}: {error}"))
                    })?;
                Ok(bytes.to_vec())
            }
            #[cfg(not(feature = "http"))]
            {
                Err(crate::Error::Unsupported(format!(
                    "HTTP byte fetch requires pklr's 'http' feature: {url}"
                )))
            }
        })
    }

    #[cfg(feature = "http")]
    fn set_http_client(&mut self, client: reqwest::Client) -> Result<()> {
        self.http_client = client;
        Ok(())
    }

    fn temp_dir<'a>(&'a mut self, prefix: &'a str) -> BoxFuture<'a, Result<PathBuf>> {
        Box::pin(async move {
            #[cfg(feature = "async")]
            {
                if tokio::runtime::Handle::try_current().is_err() {
                    return unique_temp_dir(prefix);
                }
                let prefix = prefix.to_string();
                tokio::task::spawn_blocking(move || unique_temp_dir(&prefix))
                    .await
                    .map_err(|error| {
                        crate::Error::Eval(format!("temp directory task failed: {error}"))
                    })?
            }
            #[cfg(not(feature = "async"))]
            unique_temp_dir(prefix)
        })
    }

    fn glob<'a>(
        &'a mut self,
        base: &'a Path,
        pattern: &'a str,
    ) -> BoxFuture<'a, Result<Vec<PathBuf>>> {
        Box::pin(async move {
            #[cfg(feature = "async")]
            {
                if tokio::runtime::Handle::try_current().is_err() {
                    return crate::eval::expand_glob(base, pattern);
                }
                let base = base.to_path_buf();
                let pattern = pattern.to_string();
                tokio::task::spawn_blocking(move || crate::eval::expand_glob(&base, &pattern))
                    .await
                    .map_err(|error| crate::Error::Eval(format!("glob task failed: {error}")))?
            }
            #[cfg(not(feature = "async"))]
            crate::eval::expand_glob(base, pattern)
        })
    }
}

#[cfg(feature = "blocking")]
impl EvalCapabilities for BlockingCapabilities {
    fn read_to_string<'a>(&'a mut self, path: &'a Path) -> BoxFuture<'a, Result<String>> {
        Box::pin(async move {
            std::fs::read_to_string(path)
                .map_err(|error| crate::Error::Io(path.to_path_buf(), error))
        })
    }

    fn path_exists<'a>(&'a mut self, path: &'a Path) -> BoxFuture<'a, Result<bool>> {
        Box::pin(async move { Ok(path.exists()) })
    }

    fn canonicalize<'a>(&'a mut self, path: &'a Path) -> BoxFuture<'a, Result<PathBuf>> {
        Box::pin(async move {
            path.canonicalize()
                .map_err(|error| crate::Error::Io(path.to_path_buf(), error))
        })
    }

    fn read_env<'a>(&'a mut self, name: &'a str) -> BoxFuture<'a, Result<Option<String>>> {
        Box::pin(async move { Ok(std::env::var(name).ok()) })
    }

    fn fetch_text<'a>(&'a mut self, url: &'a str) -> BoxFuture<'a, Result<String>> {
        let agent = self.http_agent.clone();
        Box::pin(async move {
            let mut response = agent
                .get(url)
                .call()
                .map_err(|error| blocking_http_error(url, error))?;
            response
                .body_mut()
                .with_config()
                .limit(BLOCKING_HTTP_BODY_LIMIT)
                .lossy_utf8(false)
                .read_to_string()
                .map_err(|error| crate::Error::Eval(format!("HTTP read failed for {url}: {error}")))
        })
    }

    fn fetch_bytes<'a>(&'a mut self, url: &'a str) -> BoxFuture<'a, Result<Vec<u8>>> {
        let agent = self.http_agent.clone();
        Box::pin(async move {
            let mut response = agent
                .get(url)
                .call()
                .map_err(|error| blocking_http_error(url, error))?;
            response
                .body_mut()
                .with_config()
                .limit(BLOCKING_HTTP_BODY_LIMIT)
                .read_to_vec()
                .map_err(|error| crate::Error::Eval(format!("HTTP read failed for {url}: {error}")))
        })
    }

    fn temp_dir<'a>(&'a mut self, prefix: &'a str) -> BoxFuture<'a, Result<PathBuf>> {
        Box::pin(async move { unique_temp_dir(prefix) })
    }

    fn glob<'a>(
        &'a mut self,
        base: &'a Path,
        pattern: &'a str,
    ) -> BoxFuture<'a, Result<Vec<PathBuf>>> {
        Box::pin(async move { crate::eval::expand_glob(base, pattern) })
    }
}

#[cfg(feature = "blocking")]
fn blocking_http_error(url: &str, error: ureq::Error) -> crate::Error {
    if matches!(error, ureq::Error::StatusCode(404)) {
        crate::Error::ImportNotFound(url.to_string())
    } else {
        crate::Error::Eval(format!("HTTP fetch failed for {url}: {error}"))
    }
}

#[cfg(feature = "native-io")]
fn list_dir_sync(path: &Path) -> Result<Vec<PathBuf>> {
    match std::fs::read_dir(path) {
        Ok(entries) => {
            let mut paths = Vec::new();
            for entry in entries {
                let entry = entry.map_err(|error| crate::Error::Io(path.to_path_buf(), error))?;
                paths.push(entry.path());
            }
            Ok(paths)
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(Vec::new()),
        Err(error) => Err(crate::Error::Io(path.to_path_buf(), error)),
    }
}

/// Generic durable-write protocol shared by the trait default and the
/// off-runtime branch of the native capabilities. It never recurses through
/// `EvalCapabilities::durable_write`; only the narrower primitives
/// (`create_dir_all`, `cache_write_fault`) are dispatched virtually.
async fn durable_write_default<T: EvalCapabilities + ?Sized>(
    capabilities: &mut T,
    destination: &Path,
    staging_dir: &Path,
    bytes: &[u8],
    sync_dir: bool,
) -> Result<()> {
    use std::io::Write;

    if let Some(parent) = destination.parent() {
        capabilities.create_dir_all(parent).await?;
    }
    capabilities.create_dir_all(staging_dir).await?;
    let staging_name =
        crate::package_cache::staging_name(destination).unwrap_or_else(|| "stage".to_string());
    let staging = staging_dir.join(staging_name);

    capabilities.cache_write_fault(CacheWriteFault::Create)?;
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&staging)
        .map_err(|error| crate::Error::Io(staging.clone(), error))?;

    capabilities.cache_write_fault(CacheWriteFault::Flush)?;
    file.write_all(bytes)
        .and_then(|()| file.flush())
        .map_err(|error| crate::Error::Io(staging.clone(), error))?;

    capabilities.cache_write_fault(CacheWriteFault::FileSync)?;
    file.sync_all()
        .map_err(|error| crate::Error::Io(staging.clone(), error))?;
    drop(file);

    capabilities.cache_write_fault(CacheWriteFault::Rename)?;
    // Windows rename cannot replace; fall back to remove-then-rename there,
    // matching the atomic writer used elsewhere in pklr.
    match std::fs::rename(&staging, destination) {
        Ok(()) => {}
        Err(_) if destination.exists() => {
            std::fs::remove_file(destination)
                .and_then(|()| std::fs::rename(&staging, destination))
                .map_err(|error| crate::Error::Io(destination.to_path_buf(), error))?;
        }
        Err(error) => return Err(crate::Error::Io(destination.to_path_buf(), error)),
    }

    if sync_dir {
        capabilities.cache_write_fault(CacheWriteFault::DirSync)?;
        sync_parent_dir(destination)?;
    }
    Ok(())
}

#[cfg(feature = "native-io")]
static TEMP_DIR_COUNTER: AtomicU64 = AtomicU64::new(0);

#[cfg(feature = "native-io")]
fn unique_temp_dir(prefix: &str) -> Result<PathBuf> {
    let base = std::env::temp_dir();
    for _ in 0..100 {
        let counter = TEMP_DIR_COUNTER.fetch_add(1, Ordering::Relaxed);
        let dir = base.join(format!("{prefix}-{}-{counter}", std::process::id()));
        match std::fs::create_dir(&dir) {
            Ok(()) => return Ok(dir),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(error) => {
                return Err(crate::Error::Eval(format!(
                    "mkdir failed for {}: {error}",
                    dir.display()
                )));
            }
        }
    }
    Err(crate::Error::Eval(format!(
        "mkdir failed for {}: unable to create a unique directory",
        base.join(prefix).display()
    )))
}


/// Fsync the parent directory so a metadata rename is durable. Best-effort on
/// platforms where directory handles cannot be synced.
fn sync_parent_dir(destination: &Path) -> Result<()> {
    let Some(parent) = destination.parent() else {
        return Ok(());
    };
    match std::fs::File::open(parent) {
        Ok(dir) => dir
            .sync_all()
            .map_err(|error| crate::Error::Io(parent.to_path_buf(), error)),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(crate::Error::Io(parent.to_path_buf(), error)),
    }
}

#[cfg(feature = "package-zip-core")]
fn extract_zip_sync(bytes: Vec<u8>, destination: &Path) -> Result<()> {
    let mut archive = zip::ZipArchive::new(std::io::Cursor::new(bytes))
        .map_err(|error| crate::Error::Eval(format!("zip error: {error}")))?;
    archive
        .extract(destination)
        .map_err(|error| crate::Error::Eval(format!("zip extract error: {error}")))?;
    Ok(())
}

#[cfg(all(test, feature = "native-io"))]
mod tests {
    use super::{EvalCapabilities, NativeCapabilities};

    #[cfg(feature = "blocking")]
    #[test]
    fn blocking_capabilities_install_a_crypto_provider() {
        let _ = super::BlockingCapabilities::new();

        assert!(rustls::crypto::CryptoProvider::get_default().is_some());
    }

    #[tokio::test]
    async fn native_temp_dirs_are_unique_and_empty() {
        let mut capabilities = NativeCapabilities::new();
        let first = capabilities
            .temp_dir("pklr-capabilities-test")
            .await
            .unwrap();
        let second = capabilities
            .temp_dir("pklr-capabilities-test")
            .await
            .unwrap();

        assert_ne!(first, second);
        assert!(std::fs::read_dir(&first).unwrap().next().is_none());
        assert!(std::fs::read_dir(&second).unwrap().next().is_none());

        std::fs::remove_dir(&first).unwrap();
        std::fs::remove_dir(&second).unwrap();
    }

    #[tokio::test]
    async fn native_path_exists_treats_metadata_errors_as_missing() {
        let mut capabilities = NativeCapabilities::new();

        assert!(
            !capabilities
                .path_exists(std::path::Path::new("invalid\0path"))
                .await
                .unwrap()
        );
    }
}
