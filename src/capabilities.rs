use std::future::Future;
use std::path::{Path, PathBuf};
use std::pin::Pin;
#[cfg(feature = "native-io")]
use std::sync::atomic::{AtomicU64, Ordering};
#[cfg(any(feature = "http", feature = "blocking"))]
use std::time::Duration;

use crate::Result;

pub type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + 'a>>;

#[cfg(unix)]
extern "C" {
    fn fsync(fd: std::os::unix::io::RawFd) -> std::ffi::c_int;
}

#[cfg(unix)]
unsafe fn libc_fsync(fd: std::os::unix::io::RawFd) -> std::ffi::c_int {
    // SAFETY: the caller passes an owned, open descriptor; fsync only blocks
    // until the kernel has flushed it.
    unsafe { fsync(fd) }
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
            // The default implementation crosses four individually overridable
            // boundaries so hosts (and tests) can observe or fail each stage:
            // temp-file creation, content flush, atomic rename, and directory
            // sync. None of the stages ever exposes a half-written destination.
            let target_name = path
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or("cache");
            let Some(parent) = path.parent() else {
                return Err(crate::Error::Eval(format!(
                    "atomic write target has no parent directory: {}",
                    path.display()
                )));
            };
            let temp_path = path.with_file_name(crate::cache::temp_file_name(target_name));
            let write_result = self.create_temp_file(&temp_path, bytes).await;
            if let Err(error) = write_result {
                let _ = self.remove_file(&temp_path).await;
                return Err(error);
            }
            if let Err(error) = self.rename(&temp_path, path).await {
                let _ = self.remove_file(&temp_path).await;
                return Err(error);
            }
            if let Err(error) = self.sync_directory(parent).await {
                return Err(crate::Error::Io(path.to_path_buf(), error));
            }
            Ok(())
        })
    }

    /// Create `path` exclusively and flush `bytes` (`write_all` + `fsync`).
    fn create_temp_file<'a>(
        &'a mut self,
        path: &'a Path,
        bytes: &'a [u8],
    ) -> BoxFuture<'a, Result<()>> {
        Box::pin(async move {
            use std::io::Write;
            let mut file = std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(path)
                .map_err(|error| crate::Error::Io(path.to_path_buf(), error))?;
            file.write_all(bytes)
                .and_then(|()| file.sync_all())
                .map_err(|error| crate::Error::Io(path.to_path_buf(), error))
        })
    }

    /// Atomically move `from` to `to`, replacing an existing destination.
    fn rename<'a>(
        &'a mut self,
        from: &'a Path,
        to: &'a Path,
    ) -> BoxFuture<'a, Result<()>> {
        Box::pin(async move {
            match std::fs::rename(from, to) {
                Ok(()) => Ok(()),
                // Windows rename does not replace an existing file.
                Err(_) if to.exists() => std::fs::remove_file(to)
                    .and_then(|()| std::fs::rename(from, to))
                    .map_err(|error| crate::Error::Io(to.to_path_buf(), error)),
                Err(error) => Err(crate::Error::Io(to.to_path_buf(), error)),
            }
        })
    }

    /// fsync a directory so a recent rename survives a crash.
    fn sync_directory<'a>(&'a mut self, path: &'a Path) -> BoxFuture<'a, Result<()>> {
        Box::pin(async move {
            #[cfg(unix)]
            {
                use std::os::unix::io::AsRawFd;
                let dir = std::fs::File::open(path)
                    .map_err(|error| crate::Error::Io(path.to_path_buf(), error))?;
                // SAFETY: `dir` owns a valid, open file descriptor.
                let result = unsafe { libc_fsync(dir.as_raw_fd()) };
                if result == 0 {
                    Ok(())
                } else {
                    Err(crate::Error::Io(
                        path.to_path_buf(),
                        std::io::Error::last_os_error(),
                    ))
                }
            }
            #[cfg(not(unix))]
            {
                let _ = path;
                Ok(())
            }
        })
    }

    /// List immediate children of a directory as `(file_name, is_directory)`.
    /// Missing directories report an empty listing.
    fn list_directory<'a>(
        &'a mut self,
        path: &'a Path,
    ) -> BoxFuture<'a, Result<Vec<(std::ffi::OsString, bool)>>> {
        Box::pin(async move {
            let entries = match std::fs::read_dir(path) {
                Ok(entries) => entries,
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                    return Ok(Vec::new())
                }
                Err(error) => {
                    return Err(crate::Error::Io(path.to_path_buf(), error));
                }
            };
            let mut children = Vec::new();
            for entry in entries {
                let entry = entry.map_err(|error| crate::Error::Io(path.to_path_buf(), error))?;
                let is_dir = entry
                    .file_type()
                    .map(|file_type| file_type.is_dir())
                    .unwrap_or(false);
                children.push((entry.file_name(), is_dir));
            }
            Ok(children)
        })
    }

    fn remove_file<'a>(&'a mut self, path: &'a Path) -> BoxFuture<'a, Result<()>> {
        Box::pin(async move {
            std::fs::remove_file(path).map_err(|error| crate::Error::Io(path.to_path_buf(), error))
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
                    return crate::cache::write_atomic_native(path, bytes)
                        .map_err(|error| crate::Error::Io(path.to_path_buf(), error));
                }
                let path = path.to_path_buf();
                let task_path = path.clone();
                let bytes = bytes.to_vec();
                tokio::task::spawn_blocking(move || crate::cache::write_atomic_native(&task_path, &bytes))
                    .await
                    .map_err(|error| {
                        crate::Error::Eval(format!("atomic write task failed: {error}"))
                    })?
                    .map_err(|error| crate::Error::Io(path, error))
            }
            #[cfg(not(feature = "async"))]
            crate::cache::write_atomic_native(path, bytes)
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
