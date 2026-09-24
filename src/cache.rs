//! Persistent package cache: on-disk layout, metadata, and the staged write
//! pipeline used to update it atomically.
//!
//! The cache lives under a caller-provided directory:
//!
//! ```text
//! <cache>/packages/<fnv1a64(url)>.meta.json        metadata (version 1)
//! <cache>/packages/blobs/<fnv1a64(bytes)>.<ext>    content-addressed blob
//! <cache>/packages/<fnv1a64(url)>.<ext>            legacy (v0) blob
//! <cache>/packages/<fnv1a64(url)>.url              legacy (v0) URL sidecar
//! ```
//!
//! Writes stage bytes in a `.<name>.tmp-<pid>-<seq>` sibling file, flush them,
//! rename onto the final path, and sync the containing directory. The blob is
//! content-addressed and written before the metadata, so an interrupted write
//! can leave an orphan blob but never metadata pointing at a missing or
//! mismatched blob. Leftover staging files are removed by [`sweep_staged`] on
//! the first cache access of a new evaluator.
use std::path::{Component, Path, PathBuf};

use crate::error::{Error, Result};

/// Cache metadata version written by this implementation.
pub const METADATA_VERSION: u32 = 1;

/// Rejection reasons surfaced in [`Error::Cache`] messages. They are part of
/// the cache contract so tests can match on them without parsing prose.
pub const ERR_METADATA_CORRUPT: &str = "package cache metadata is corrupt";
pub const ERR_METADATA_VERSION: &str = "unsupported package cache metadata version";
pub const ERR_DIGEST_MISMATCH: &str = "package cache digest mismatch";
pub const ERR_BLOB_PATH: &str = "package cache blob path escapes cache directory";

/// On-disk layout of one cache entry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CacheEntryPaths {
    /// Versioned metadata file (`<hash>.meta.json`).
    pub metadata: PathBuf,
    /// Legacy (v0) blob location next to the metadata (`<hash>.<ext>`).
    pub blob: PathBuf,
    /// Legacy (v0) URL sidecar (`<hash>.url`).
    pub legacy_url: PathBuf,
}

/// FNV-1a 64-bit digest. Stable across platforms and process runs, which is
/// what the persistent cache contract requires.
pub fn fnv1a_64(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf29ce484222325_u64;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

/// Digest string stored in metadata: `fnv1a-64:<hex>`.
pub fn digest_string(bytes: &[u8]) -> String {
    format!("fnv1a-64:{:016x}", fnv1a_64(bytes))
}

fn url_key(url: &str) -> String {
    format!("{:016x}", fnv1a_64(url.as_bytes()))
}

/// Resolve the on-disk layout for `url` inside `cache_dir`.
pub fn entry_paths(cache_dir: &Path, url: &str, extension: &str) -> CacheEntryPaths {
    let base = cache_dir.join("packages").join(url_key(url));
    CacheEntryPaths {
        metadata: base.with_extension("meta.json"),
        blob: base.with_extension(extension),
        legacy_url: base.with_extension("url"),
    }
}

/// Staging path for an in-progress write of `final_path`.
pub fn staging_path(final_path: &Path, process_id: u32, sequence: u64) -> PathBuf {
    let file_name = final_path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("cache");
    final_path.with_file_name(format!(".{file_name}.tmp-{process_id}-{sequence}"))
}

/// Whether `file_name` looks like a leftover staging file.
pub fn is_staging_file_name(file_name: &str) -> bool {
    file_name.starts_with('.') && file_name.contains(".tmp-")
}

/// Parsed cache metadata (version 1).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CacheMetadata {
    pub version: u32,
    pub url: String,
    pub extension: String,
    pub digest: String,
    /// Blob location relative to the cache root. Always confined to it.
    pub blob: String,
}

impl CacheMetadata {
    /// Build metadata for `bytes` downloaded from `url`.
    pub fn new(url: &str, extension: &str, bytes: &[u8]) -> Self {
        Self {
            version: METADATA_VERSION,
            url: url.to_string(),
            extension: extension.to_string(),
            digest: digest_string(bytes),
            blob: format!("packages/blobs/{:016x}.{extension}", fnv1a_64(bytes)),
        }
    }

    pub fn to_json(&self) -> String {
        serde_json::json!({
            "version": self.version,
            "url": self.url,
            "extension": self.extension,
            "digest": self.digest,
            "blob": self.blob,
        })
        .to_string()
    }

    /// Parse and validate metadata. Rejects corrupt JSON, unsupported
    /// versions, and blob paths that would escape the cache directory.
    pub fn from_json(text: &str) -> Result<Self> {
        let value: serde_json::Value = serde_json::from_str(text)
            .map_err(|error| Error::Cache(format!("{ERR_METADATA_CORRUPT}: {error}")))?;
        let corrupt = |field: &str| {
            Error::Cache(format!("{ERR_METADATA_CORRUPT}: missing or invalid '{field}'"))
        };
        let version = value
            .get("version")
            .and_then(serde_json::Value::as_u64)
            .ok_or_else(|| corrupt("version"))?;
        if version != u64::from(METADATA_VERSION) {
            return Err(Error::Cache(format!("{ERR_METADATA_VERSION}: {version}")));
        }
        let string_field = |field: &str| {
            value
                .get(field)
                .and_then(serde_json::Value::as_str)
                .map(str::to_string)
                .ok_or_else(|| corrupt(field))
        };
        let metadata = Self {
            version: version as u32,
            url: string_field("url")?,
            extension: string_field("extension")?,
            digest: string_field("digest")?,
            blob: string_field("blob")?,
        };
        metadata.validate_blob_path()?;
        Ok(metadata)
    }

    /// Reject blob paths that are absolute or escape the cache root.
    pub fn validate_blob_path(&self) -> Result<()> {
        let path = Path::new(&self.blob);
        let confined = !path.is_absolute()
            && !self.blob.is_empty()
            && path
                .components()
                .all(|component| matches!(component, Component::Normal(_)));
        if !confined {
            return Err(Error::Cache(format!(
                "{ERR_BLOB_PATH}: {}",
                self.blob
            )));
        }
        Ok(())
    }

    /// Absolute blob path inside `cache_dir`.
    pub fn blob_path(&self, cache_dir: &Path) -> PathBuf {
        cache_dir.join(&self.blob)
    }

    /// Verify that `bytes` match the recorded digest.
    pub fn verify_digest(&self, bytes: &[u8]) -> Result<()> {
        let actual = digest_string(bytes);
        if actual != self.digest {
            return Err(Error::Cache(format!(
                "{ERR_DIGEST_MISMATCH}: {}: expected {}, got {actual}",
                self.url, self.digest
            )));
        }
        Ok(())
    }
}

/// Stage `bytes` at `staging`, refusing to overwrite an existing file.
pub fn stage_write(staging: &Path, bytes: &[u8]) -> std::io::Result<()> {
    use std::io::Write;
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(staging)?;
    file.write_all(bytes)
}

/// Flush a staged file's contents to stable storage.
pub fn flush_file(path: &Path) -> std::io::Result<()> {
    let file = std::fs::OpenOptions::new().write(true).open(path)?;
    file.sync_all()
}

/// Atomically replace `final_path` with the staged file.
pub fn promote(staging: &Path, final_path: &Path) -> std::io::Result<()> {
    match std::fs::rename(staging, final_path) {
        Ok(()) => Ok(()),
        // Windows rename does not replace an existing file. There is a brief
        // non-atomic gap on that platform, but the destination must not retain
        // stale bytes while its metadata is updated.
        Err(_) if final_path.exists() => {
            std::fs::remove_file(final_path).and_then(|()| std::fs::rename(staging, final_path))
        }
        Err(error) => Err(error),
    }
}

/// Durably sync a directory that received a promoted file. Unsupported on
/// non-Unix platforms, where it is a no-op.
pub fn sync_dir(dir: &Path) -> std::io::Result<()> {
    #[cfg(unix)]
    {
        std::fs::File::open(dir)?.sync_all()
    }
    #[cfg(not(unix))]
    {
        let _ = dir;
        Ok(())
    }
}

/// Remove leftover staging files directly under `dir`, returning the removed
/// paths in directory order. A missing directory sweeps nothing.
pub fn sweep_staged(dir: &Path) -> std::io::Result<Vec<PathBuf>> {
    let mut removed = Vec::new();
    let entries = match std::fs::read_dir(dir) {
        Ok(entries) => entries,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(removed),
        Err(error) => return Err(error),
    };
    for entry in entries {
        let entry = entry?;
        let file_name = entry.file_name();
        let Some(file_name) = file_name.to_str() else {
            continue;
        };
        if is_staging_file_name(file_name) {
            std::fs::remove_file(entry.path())?;
            removed.push(entry.path());
        }
    }
    Ok(removed)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "pklr-cache-{name}-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn metadata_roundtrips() {
        let metadata = CacheMetadata::new("https://example.com/pkg@1.0.0.zip", "zip", b"bytes");
        let parsed = CacheMetadata::from_json(&metadata.to_json()).unwrap();
        assert_eq!(parsed, metadata);
    }

    #[test]
    fn metadata_rejects_corrupt_json() {
        let error = CacheMetadata::from_json("{not json").unwrap_err();
        assert!(error.to_string().contains(ERR_METADATA_CORRUPT), "{error}");
    }

    #[test]
    fn metadata_rejects_unsupported_version() {
        let text = serde_json::json!({
            "version": 99,
            "url": "https://example.com/pkg.zip",
            "extension": "zip",
            "digest": digest_string(b"x"),
            "blob": "packages/blobs/x.zip",
        })
        .to_string();
        let error = CacheMetadata::from_json(&text).unwrap_err();
        assert!(error.to_string().contains(ERR_METADATA_VERSION), "{error}");
    }

    #[test]
    fn metadata_rejects_escaping_blob_paths() {
        for blob in ["../outside.zip", "packages/../../evil.zip", "/abs/path.zip"] {
            let text = serde_json::json!({
                "version": 1,
                "url": "https://example.com/pkg.zip",
                "extension": "zip",
                "digest": digest_string(b"x"),
                "blob": blob,
            })
            .to_string();
            let error = CacheMetadata::from_json(&text).unwrap_err();
            assert!(
                error.to_string().contains(ERR_BLOB_PATH),
                "{blob}: {error}"
            );
        }
    }

    #[test]
    fn digest_verification_detects_mismatch() {
        let metadata = CacheMetadata::new("https://example.com/pkg.zip", "zip", b"original");
        metadata.verify_digest(b"original").unwrap();
        let error = metadata.verify_digest(b"tampered").unwrap_err();
        assert!(error.to_string().contains(ERR_DIGEST_MISMATCH), "{error}");
    }

    #[test]
    fn staged_pipeline_writes_and_sweep_cleans_leftovers() {
        let dir = test_dir("pipeline");
        let final_path = dir.join("data.bin");
        let staging = staging_path(&final_path, std::process::id(), 0);

        stage_write(&staging, b"payload").unwrap();
        flush_file(&staging).unwrap();
        promote(&staging, &final_path).unwrap();
        sync_dir(&dir).unwrap();
        assert_eq!(std::fs::read(&final_path).unwrap(), b"payload");

        // Leftover staging files are swept; other files are kept.
        let leftover = staging_path(&final_path, std::process::id(), 1);
        std::fs::write(&leftover, b"orphan").unwrap();
        let removed = sweep_staged(&dir).unwrap();
        assert_eq!(removed, vec![leftover.clone()]);
        assert!(!leftover.exists());
        assert!(final_path.exists());

        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn staging_names_are_recognized() {
        let staging = staging_path(Path::new("cache/packages/x.meta.json"), 42, 7);
        let name = staging.file_name().unwrap().to_str().unwrap();
        assert_eq!(name, ".x.meta.json.tmp-42-7");
        assert!(is_staging_file_name(name));
        assert!(!is_staging_file_name("x.meta.json"));
        assert!(!is_staging_file_name("x.tmp-1-2"));
    }
}
