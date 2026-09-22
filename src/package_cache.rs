//! On-disk package cache with versioned metadata and content-addressed blobs.
//!
//! Layout under `<cache_dir>/packages/`:
//!
//! - `<urlhash>.meta` — JSON metadata (`version`, `url`, `extension`,
//!   `sha256`, `blob`). The metadata file is the commit record: a blob
//!   without metadata is never a cache hit.
//! - `<sha256>.<extension>` — immutable, content-addressed payload.
//! - `.<name>.tmp-*` — staging files cleaned up on startup.
//!
//! All IO goes through [`EvalCapabilities`], so the cache is exercised
//! hermetically in tests with an in-memory filesystem.

use std::path::{Path, PathBuf};

use serde_json::json;

use crate::capabilities::{AtomicWriteFault, AtomicWriteFaults, EvalCapabilities, WriteStage};
use crate::sha256::sha256_hex;
use crate::Result;

/// Metadata format version understood by this build. Older metadata is
/// rejected with a distinct error rather than misinterpreted.
pub const METADATA_VERSION: u32 = 1;
pub const PACKAGES_DIR: &str = "packages";

/// Corruption reasons reported by [`read_entry`]. Each maps to a stable
/// `cache:`-prefixed error string so callers and tests can distinguish them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheCorruption {
    /// Metadata is missing or is not valid JSON.
    Metadata,
    /// Metadata declares a version this build does not understand.
    Version,
    /// Blob digest does not match metadata.
    Digest,
    /// Metadata references a path outside the packages directory.
    Escape,
}

impl CacheCorruption {
    pub fn tag(self) -> &'static str {
        match self {
            Self::Metadata => "cache: corrupt package metadata",
            Self::Version => "cache: unsupported package metadata version",
            Self::Digest => "cache: package digest mismatch",
            Self::Escape => "cache: package metadata escapes cache directory",
        }
    }

    fn error(self) -> crate::Error {
        crate::Error::Eval(self.tag().to_string())
    }
}

/// FNV-1a hash of a package URL: stable, filesystem-safe metadata file name.
fn url_hash(url: &str) -> u64 {
    let mut hash = 0xcbf29ce484222325_u64;
    for byte in url.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

fn metadata_path(cache_dir: &Path, url: &str) -> PathBuf {
    cache_dir
        .join(PACKAGES_DIR)
        .join(format!("{:016x}.meta", url_hash(url)))
}

fn packages_dir(cache_dir: &Path) -> PathBuf {
    cache_dir.join(PACKAGES_DIR)
}

fn is_not_found(error: &crate::Error) -> bool {
    matches!(error, crate::Error::Io(_, io) if io.kind() == std::io::ErrorKind::NotFound)
}

/// Validate that a metadata-declared blob name is a plain file in the same
/// directory. Absolute paths, separators, and `..` components are rejected,
/// so a forged metadata record cannot make the evaluator read or delete files
/// outside the packages directory.
fn safe_blob_name(name: &str) -> Option<&str> {
    if name.is_empty() || name.contains('/') || name.contains('\\') || name == "." || name == ".."
    {
        return None;
    }
    if Path::new(name)
        .components()
        .any(|component| !matches!(component, std::path::Component::Normal(_)))
    {
        return None;
    }
    Some(name)
}

/// Read and fully validate a cached package.
///
/// Returns `Ok(None)` when there is no metadata record (cold cache). Any
/// structural problem with a present record is a distinct [`CacheCorruption`]
/// error: a half-written blob or forged metadata must never look like a hit.
pub async fn read_entry(
    capabilities: &mut dyn EvalCapabilities,
    cache_dir: &Path,
    url: &str,
    extension: &str,
) -> Result<Option<Vec<u8>>> {
    let meta_path = metadata_path(cache_dir, url);
    let metadata = match capabilities.read_to_string(&meta_path).await {
        Ok(text) => text,
        Err(error) if is_not_found(&error) => return Ok(None),
        Err(error) => return Err(error),
    };
    let parsed: serde_json::Value = serde_json::from_str(&metadata)
        .map_err(|_| CacheCorruption::Metadata.error())?;

    if parsed.get("version").and_then(|value| value.as_u64()) != Some(u64::from(METADATA_VERSION))
    {
        return Err(CacheCorruption::Version.error());
    }
    if parsed.get("url").and_then(|value| value.as_str()) != Some(url) {
        return Err(CacheCorruption::Metadata.error());
    }
    if parsed.get("extension").and_then(|value| value.as_str()) != Some(extension) {
        return Err(CacheCorruption::Metadata.error());
    }
    let blob_name = parsed
        .get("blob")
        .and_then(|value| value.as_str())
        .and_then(safe_blob_name)
        .ok_or_else(|| CacheCorruption::Escape.error())?;
    let expected_digest = parsed
        .get("sha256")
        .and_then(|value| value.as_str())
        .ok_or_else(|| CacheCorruption::Metadata.error())?;

    let blob_path = packages_dir(cache_dir).join(blob_name);
    let bytes = match capabilities.read_bytes(&blob_path).await {
        Ok(bytes) => bytes,
        Err(error) if is_not_found(&error) => {
            return Err(CacheCorruption::Metadata.error());
        }
        Err(error) => return Err(error),
    };
    if sha256_hex(&bytes) != expected_digest {
        return Err(CacheCorruption::Digest.error());
    }
    Ok(Some(bytes))
}

/// Write a validated package: content-addressed blob first, metadata last.
/// The metadata file is the atomic commit record, so a failure at any earlier
/// boundary cannot poison future lookups.
///
/// `faults` optionally injects a failure at one of the write boundaries
/// (metadata write, blob write, flush, rename, directory synchronization) for
/// contract tests.
pub async fn write_entry(
    capabilities: &mut dyn EvalCapabilities,
    cache_dir: &Path,
    url: &str,
    extension: &str,
    bytes: &[u8],
    faults: Option<&dyn AtomicWriteFaults>,
) -> Result<()> {
    let dir = packages_dir(cache_dir);
    capabilities.create_dir_all(&dir).await?;

    let digest = sha256_hex(bytes);
    let blob_name = format!("{digest}.{extension}");
    let blob_path = dir.join(&blob_name);
    let meta_path = metadata_path(cache_dir, url);

    // Content-addressed blobs are immutable. Skip the write when the bytes are
    // already present; only the metadata commit is still required.
    if capabilities
        .read_bytes(&blob_path)
        .await
        .map(|existing| existing == bytes)
        .unwrap_or(false)
    {
        // Nothing to write.
    } else {
        capabilities
            .write_atomic_faultable(&blob_path, bytes, faults, WriteStage::Blob)
            .await?;
    }

    let metadata = json!({
        "version": METADATA_VERSION,
        "url": url,
        "extension": extension,
        "sha256": digest,
        "blob": blob_name,
    })
    .to_string();
    capabilities
        .write_atomic_faultable(meta_path.as_path(), metadata.as_bytes(), faults, WriteStage::Metadata)
        .await?;
    capabilities.sync_directory(&dir, faults).await
}

/// Remove a cache entry. Only the plain blob name recorded in metadata is
/// deleted, so a forged record cannot trigger a delete outside the cache.
pub async fn remove_entry(
    capabilities: &mut dyn EvalCapabilities,
    cache_dir: &Path,
    url: &str,
    extension: &str,
) {
    let meta_path = metadata_path(cache_dir, url);
    if let Ok(metadata) = capabilities.read_to_string(&meta_path).await
        && let Ok(parsed) = serde_json::from_str::<serde_json::Value>(&metadata)
        && let Some(blob_name) = parsed
            .get("blob")
            .and_then(|value| value.as_str())
            .and_then(safe_blob_name)
    {
        let _ = capabilities
            .remove_file(&packages_dir(cache_dir).join(blob_name))
            .await;
    }
    let _ = capabilities.remove_file(&meta_path).await;
}

/// File name prefix used by [`crate::eval::write_atomic`] for staging files.
pub const STAGING_PREFIX: char = '.';

/// Remove staging files left behind by a crashed or failed write.
///
/// Only files that look like our staging names (`.name.tmp-...`) are touched;
/// every deleted path is verified to stay inside the packages directory.
/// Errors are best effort and never block evaluation.
pub async fn sweep_staging(capabilities: &mut dyn EvalCapabilities, cache_dir: &Path) {
    let dir = packages_dir(cache_dir);
    let entries = capabilities.list_dir(&dir).await;
    let Ok(entries) = entries else {
        return;
    };
    for entry in entries {
        let Some(name) = entry.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        let looks_like_staging = name.starts_with(STAGING_PREFIX) && name.contains(".tmp-");
        if !looks_like_staging {
            continue;
        }
        let candidate = dir.join(name);
        if !candidate.starts_with(&dir) {
            continue;
        }
        let _ = capabilities.remove_file(&candidate).await;
    }
}

/// Select the fault to inject at a write boundary. This is the single place
/// that maps an embedder's [`AtomicWriteFaults`] policy onto the free atomic
/// write function, so native and custom capabilities behave identically.
pub(crate) fn select_fault(
    faults: Option<&dyn AtomicWriteFaults>,
    path: &Path,
    stage: WriteStage,
) -> Option<AtomicWriteFault> {
    faults.and_then(|policy| policy.fault_for(path, stage))
}
