//! Persistent `package://` content cache: a metadata file plus a blob file
//! per package URL, written atomically and validated on read.
//!
//! Layout under `<cache_dir>/packages/`:
//!
//! - `<fnv1a(url)>.blob` — the package bytes.
//! - `<fnv1a(url)>.meta.json` — JSON metadata: format version, source URL,
//!   extension, blob file name, and a checksum of the blob.
//!
//! Writes go through temp files (recognizable by the
//! [`TEMP_NAME_MARKER`] substring) that are flushed and renamed into place;
//! the metadata rename — the commit point — is written last, so an interrupted
//! write never produces a half-entry that reads as a hit. Stale temp files are
//! removed on startup.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use crate::capabilities::EvalCapabilities;
use crate::error::{Error, Result};

/// Metadata format version understood by this implementation.
pub const CACHE_FORMAT_VERSION: u32 = 1;

/// Substring marking in-progress cache writes; used to recognize and clean
/// up stale temp files on startup.
pub const TEMP_NAME_MARKER: &str = ".tmp-";

static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Why a cache entry was rejected instead of served.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CacheRejection {
    /// The metadata file is not valid JSON or is missing required fields.
    CorruptMetadata,
    /// The metadata format version is not supported by this implementation.
    UnsupportedVersion(u32),
    /// The blob bytes do not match the checksum recorded in the metadata.
    ChecksumMismatch,
    /// The metadata points at a blob path outside the cache directory.
    PathEscape,
}

/// Outcome of reading a cache entry.
#[derive(Debug)]
pub enum CacheRead {
    /// Valid cached bytes.
    Hit(Vec<u8>),
    /// No usable entry (absent, incomplete, or owned by another URL).
    Miss,
    /// An entry exists but failed validation; it must not be served.
    Rejected(CacheRejection),
}

/// Metadata + blob cache for downloaded `package://` content.
#[derive(Debug, Clone)]
pub struct PackageCache {
    packages_dir: PathBuf,
}

/// FNV-1a 64-bit, used for cache keys and blob checksums: stable across
/// platforms and processes without depending on filesystem path rules.
pub(crate) fn fnv1a(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf29ce484222325_u64;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

fn cache_key(url: &str) -> String {
    format!("{:016x}", fnv1a(url.as_bytes()))
}

/// Whether `name` looks like an in-progress cache write.
pub fn is_temp_name(name: &str) -> bool {
    name.contains(TEMP_NAME_MARKER)
}

impl PackageCache {
    /// Open the cache rooted at `cache_dir` (entries live under `packages/`).
    pub fn new(cache_dir: &Path) -> Self {
        Self {
            packages_dir: cache_dir.join("packages"),
        }
    }

    /// Directory holding the cache entries.
    pub fn packages_dir(&self) -> &Path {
        &self.packages_dir
    }

    /// The (blob, metadata) paths for `url`.
    pub fn paths(&self, url: &str) -> (PathBuf, PathBuf) {
        let key = cache_key(url);
        (
            self.packages_dir.join(format!("{key}.blob")),
            self.packages_dir.join(format!("{key}.meta.json")),
        )
    }

    fn temp_path(&self, final_path: &Path) -> PathBuf {
        let file_name = final_path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("cache");
        final_path.with_file_name(format!(
            ".{file_name}.tmp-{}-{}",
            std::process::id(),
            TEMP_COUNTER.fetch_add(1, Ordering::Relaxed)
        ))
    }

    fn metadata(url: &str, extension: &str, bytes: &[u8]) -> String {
        serde_json::json!({
            "version": CACHE_FORMAT_VERSION,
            "url": url,
            "extension": extension,
            "blob": format!("{}.blob", cache_key(url)),
            "checksum": format!("{:016x}", fnv1a(bytes)),
        })
        .to_string()
    }

    /// Read the cached bytes for `url`, validating metadata and checksum.
    pub async fn read(
        &self,
        capabilities: &mut dyn EvalCapabilities,
        url: &str,
        extension: &str,
    ) -> Result<CacheRead> {
        let (blob_path, meta_path) = self.paths(url);
        let meta_text = match capabilities.read_to_string(&meta_path).await {
            Ok(text) => text,
            Err(Error::Io(_, error)) if error.kind() == std::io::ErrorKind::NotFound => {
                return Ok(CacheRead::Miss);
            }
            Err(error) => return Err(error),
        };
        let meta: serde_json::Value = match serde_json::from_str(&meta_text) {
            Ok(meta) => meta,
            Err(_) => return Ok(CacheRead::Rejected(CacheRejection::CorruptMetadata)),
        };
        let (Some(version), Some(meta_url), Some(meta_extension), Some(blob_name), Some(checksum)) = (
            meta.get("version").and_then(|v| v.as_u64()),
            meta.get("url").and_then(|v| v.as_str()),
            meta.get("extension").and_then(|v| v.as_str()),
            meta.get("blob").and_then(|v| v.as_str()),
            meta.get("checksum").and_then(|v| v.as_str()),
        ) else {
            return Ok(CacheRead::Rejected(CacheRejection::CorruptMetadata));
        };
        if version != u64::from(CACHE_FORMAT_VERSION) {
            return Ok(CacheRead::Rejected(CacheRejection::UnsupportedVersion(
                version as u32,
            )));
        }
        // A hash collision or a different URL's entry: simply not our cache line.
        if meta_url != url || meta_extension != extension {
            return Ok(CacheRead::Miss);
        }
        // The blob name must be the expected file inside the packages dir.
        let expected_blob_name = format!("{}.blob", cache_key(url));
        let blob_is_safe = blob_name == expected_blob_name
            && !blob_name.contains('/')
            && !blob_name.contains('\\')
            && Path::new(blob_name).file_name().and_then(|n| n.to_str()) == Some(blob_name);
        if !blob_is_safe {
            return Ok(CacheRead::Rejected(CacheRejection::PathEscape));
        }
        let bytes = match capabilities.read_bytes(&blob_path).await {
            Ok(bytes) => bytes,
            Err(Error::Io(_, error)) if error.kind() == std::io::ErrorKind::NotFound => {
                // Metadata without its blob is a half-written entry, not a hit.
                return Ok(CacheRead::Miss);
            }
            Err(error) => return Err(error),
        };
        if format!("{:016x}", fnv1a(&bytes)) != checksum {
            return Ok(CacheRead::Rejected(CacheRejection::ChecksumMismatch));
        }
        Ok(CacheRead::Hit(bytes))
    }

    /// Atomically store `bytes` for `url`. The metadata file is the commit
    /// point and is written last; on failure the previous entry (if any) is
    /// never served as a hit for different bytes.
    pub async fn write(
        &self,
        capabilities: &mut dyn EvalCapabilities,
        url: &str,
        extension: &str,
        bytes: &[u8],
    ) -> Result<()> {
        capabilities.create_dir_all(&self.packages_dir).await?;
        let (blob_path, meta_path) = self.paths(url);
        let meta_text = Self::metadata(url, extension, bytes);
        for (final_path, contents) in [
            (&blob_path, bytes.to_vec()),
            (&meta_path, meta_text.into_bytes()),
        ] {
            let temp_path = self.temp_path(final_path);
            let result: Result<()> = async {
                capabilities.write_bytes(&temp_path, &contents).await?;
                capabilities.flush_file(&temp_path).await?;
                capabilities.rename(&temp_path, final_path).await
            }
            .await;
            if let Err(error) = result {
                let _ = capabilities.remove_file(&temp_path).await;
                return Err(error);
            }
        }
        capabilities.sync_dir(&self.packages_dir).await
    }

    /// Remove any entry for `url`. Missing files are ignored.
    pub async fn remove(&self, capabilities: &mut dyn EvalCapabilities, url: &str) {
        let (blob_path, meta_path) = self.paths(url);
        let _ = capabilities.remove_file(&blob_path).await;
        let _ = capabilities.remove_file(&meta_path).await;
    }

    /// Remove stale temp files left by interrupted writes. Returns how many
    /// were removed.
    pub async fn cleanup_stale_temps(
        &self,
        capabilities: &mut dyn EvalCapabilities,
    ) -> Result<usize> {
        if !capabilities.path_exists(&self.packages_dir).await? {
            return Ok(0);
        }
        let temps = capabilities
            .glob(&self.packages_dir, &format!("*{TEMP_NAME_MARKER}*"))
            .await?;
        let mut removed = 0;
        for temp in temps {
            let is_temp = temp
                .file_name()
                .and_then(|name| name.to_str())
                .is_some_and(is_temp_name);
            if is_temp && capabilities.remove_file(&temp).await.is_ok() {
                removed += 1;
            }
        }
        Ok(removed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn temp_names_are_recognizable() {
        let cache = PackageCache::new(Path::new("cache"));
        let (_, meta_path) = cache.paths("https://example.com/pkg.pkl");
        let temp = cache.temp_path(&meta_path);
        let name = temp.file_name().unwrap().to_str().unwrap();
        assert!(is_temp_name(name));
        assert!(!is_temp_name("0123abcd.meta.json"));
    }

    #[test]
    fn cache_keys_are_stable_and_url_sensitive() {
        assert_eq!(cache_key("https://a.test/x"), cache_key("https://a.test/x"));
        assert_ne!(cache_key("https://a.test/x"), cache_key("https://a.test/y"));
    }
}
