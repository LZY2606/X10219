//! Persistent content-addressed package cache.
//!
//! Every downloaded package is stored inside `<cache_dir>/packages/` in a
//! directory derived from the package URL. A directory holds:
//!
//! * `meta.json` — versioned metadata (origin URL, extension, digest),
//! * a content-addressed blob named after its SHA-256 digest.
//!
//! Writes are staged with unique temp files and committed in two phases:
//! the blob lands before the metadata, so a crash can never expose a
//! half-written entry as a cache hit. All paths are confined to the cache
//! root; temp files use the `.staging-` prefix and are swept on startup.

use std::path::{Component, Path, PathBuf};

use serde_json::json;
use sha2::{Digest, Sha256};

use crate::capabilities::EvalCapabilities;
use crate::Result;

/// On-disk metadata format version understood by this build.
pub const META_VERSION: u32 = 1;
/// Metadata sidecar file name.
pub const META_FILE_NAME: &str = "meta.json";
/// Prefix used for staged (not yet committed) cache files.
pub const STAGING_PREFIX: &str = ".staging-";

/// A single physical durability boundary exercised while committing an entry.
///
/// Each variant is one place a crash or injected failure can interrupt a
/// cache write.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WriteBoundary {
    /// Create and fill the staged metadata file.
    MetadataWrite,
    /// Create and fill the staged blob file.
    BlobWrite,
    /// Flush a staged file to stable storage.
    Flush,
    /// Rename a staged file over its final name (the commit point).
    Rename,
    /// Flush the containing directory (before the committing rename).
    DirSync,
}

impl WriteBoundary {
    pub fn as_str(self) -> &'static str {
        match self {
            WriteBoundary::MetadataWrite => "metadata_write",
            WriteBoundary::BlobWrite => "blob_write",
            WriteBoundary::Flush => "flush",
            WriteBoundary::Rename => "rename",
            WriteBoundary::DirSync => "dir_sync",
        }
    }
}

/// Why a persisted cache entry could not be used.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CacheReadError {
    /// `meta.json` exists but cannot be parsed as metadata.
    CorruptMetadata(String),
    /// The blob digest does not match the digest recorded in metadata.
    DigestMismatch { expected: String, actual: String },
    /// Metadata carries a `version` this build does not understand.
    UnsupportedVersion(u32),
    /// Metadata names a blob outside the cache directory.
    UnsafePath(String),
}

impl CacheReadError {
    pub fn message(&self) -> String {
        match self {
            CacheReadError::CorruptMetadata(detail) => {
                format!("corrupt package cache metadata: {detail}")
            }
            CacheReadError::DigestMismatch { expected, actual } => format!(
                "package cache digest mismatch: expected {expected}, found {actual}"
            ),
            CacheReadError::UnsupportedVersion(version) => {
                format!("unsupported package cache metadata version: {version}")
            }
            CacheReadError::UnsafePath(path) => {
                format!("package cache metadata points outside the cache directory: {path}")
            }
        }
    }
}

impl std::fmt::Display for CacheReadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message())
    }
}

impl std::error::Error for CacheReadError {}

/// Outcome of looking up a package in the persistent cache.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CacheLookup {
    /// A complete, validated entry was present.
    Hit(Vec<u8>),
    /// Nothing committed exists for the package.
    Miss,
    /// Persisted state is present but invalid.
    Invalid(CacheReadError),
}

/// Compute the lowercase hex SHA-256 digest of `bytes`.
pub fn digest_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    let digest = hasher.finalize();
    let mut out = String::with_capacity(64);
    for byte in digest {
        out.push_str(&format!("{byte:02x}"));
    }
    out
}

fn fnv1a64(input: &str) -> u64 {
    let mut hash = 0xcbf29ce4_84222325_u64;
    for byte in input.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x1000_00001b3);
    }
    hash
}

/// Directory that stores the entry for `url` (with a given file extension).
pub fn entry_dir(cache_dir: &Path, url: &str, extension: &str) -> PathBuf {
    cache_dir
        .join("packages")
        .join(format!("{:016x}", fnv1a64(&format!("{extension}\u{1}{url}"))))
}

fn staged_name(final_name: &str, serial: u64) -> String {
    format!(
        "{STAGING_PREFIX}{final_name}-{pid}-{serial}",
        pid = std::process::id()
    )
}

/// Whether `name` is a leftover staged file that may be reclaimed.
pub fn is_staging_file(name: &str) -> bool {
    name.starts_with(STAGING_PREFIX)
}

fn normalize(path: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for component in path.components() {
        match component {
            Component::Prefix(prefix) => out.push(prefix.as_os_str()),
            Component::RootDir => out.push(Component::RootDir.as_os_str()),
            Component::CurDir => {}
            Component::ParentDir => {
                out.pop();
            }
            Component::Normal(part) => out.push(part),
        }
    }
    out
}

fn is_within(root: &Path, target: &Path) -> bool {
    let joined = if target.is_absolute() {
        target.to_path_buf()
    } else {
        root.join(target)
    };
    normalize(&joined).starts_with(normalize(root))
}

fn parse_metadata(source: &str) -> std::result::Result<serde_json::Value, String> {
    let value: serde_json::Value =
        serde_json::from_str(source).map_err(|error| error.to_string())?;
    if !value.is_object() {
        return Err("metadata is not a JSON object".to_string());
    }
    Ok(value)
}

use std::sync::atomic::{AtomicU64, Ordering};

static STAGING_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Remove staged files left behind by interrupted writes.
///
/// Walks `cache_dir` (depth-first, sorted) and deletes every file or empty
/// directory whose name starts with [`STAGING_PREFIX`]. Returns the paths
/// that were reclaimed. Missing cache roots are treated as an empty sweep.
pub async fn sweep_staging<C: EvalCapabilities + ?Sized>(
    caps: &mut C,
    cache_dir: &Path,
) -> Result<Vec<PathBuf>> {
    let mut removed = Vec::new();
    sweep_dir(caps, cache_dir, &mut removed).await?;
    Ok(removed)
}

#[async_recursion::async_recursion(?Send)]
async fn sweep_dir<C: EvalCapabilities + ?Sized>(
    caps: &mut C,
    dir: &Path,
    removed: &mut Vec<PathBuf>,
) -> Result<()> {
    let mut entries = match caps.cache_list_dir(dir).await {
        Ok(entries) => entries,
        Err(crate::Error::Io(_, error)) if error.kind() == std::io::ErrorKind::NotFound => {
            return Ok(());
        }
        Err(error) => return Err(error),
    };
    entries.sort();
    for entry in entries {
        let name = entry
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or_default();
        match caps.cache_list_dir(&entry).await {
            Ok(_) => {
                sweep_dir(caps, &entry, removed).await?;
                let empty = caps
                    .cache_list_dir(&entry)
                    .await
                    .map(|items| items.is_empty())
                    .unwrap_or(false);
                if is_staging_file(name) && empty {
                    let _ = caps.cache_remove_dir_all(&entry).await;
                    removed.push(entry);
                }
            }
            Err(_) if is_staging_file(name) => {
                let _ = caps.remove_file(&entry).await;
                removed.push(entry);
            }
            Err(_) => {}
        }
    }
    Ok(())
}

/// Load and validate the persisted entry for `url`.
pub async fn load_entry<C: EvalCapabilities + ?Sized>(
    caps: &mut C,
    cache_dir: &Path,
    url: &str,
    extension: &str,
) -> Result<CacheLookup> {
    let directory = entry_dir(cache_dir, url, extension);
    let metadata_path = directory.join(META_FILE_NAME);
    let source = match caps.read_to_string(&metadata_path).await {
        Ok(source) => source,
        Err(crate::Error::Io(_, error)) if error.kind() == std::io::ErrorKind::NotFound => {
            return Ok(CacheLookup::Miss);
        }
        Err(error) => return Err(error),
    };
    let metadata = match parse_metadata(&source) {
        Ok(metadata) => metadata,
        Err(detail) => return Ok(CacheLookup::Invalid(CacheReadError::CorruptMetadata(detail))),
    };
    let version = metadata
        .get("version")
        .and_then(|value| value.as_u64())
        .ok_or_else(|| {
            crate::Error::Eval(CacheReadError::CorruptMetadata(
                "missing numeric version".to_string(),
            )
            .message())
        })? as u32;
    if version != META_VERSION {
        return Ok(CacheLookup::Invalid(CacheReadError::UnsupportedVersion(
            version,
        )));
    }
    let cached_url = metadata.get("url").and_then(|value| value.as_str());
    if cached_url != Some(url) {
        // Hash collision or a stale entry from a different URL: never a hit.
        return Ok(CacheLookup::Miss);
    }
    let cached_extension = metadata
        .get("extension")
        .and_then(|value| value.as_str());
    if cached_extension != Some(extension) {
        return Ok(CacheLookup::Miss);
    }
    let recorded_digest = metadata
        .get("digest")
        .and_then(|value| value.as_str())
        .ok_or_else(|| {
            crate::Error::Eval(CacheReadError::CorruptMetadata(
                "missing digest".to_string(),
            )
            .message())
        })?;
    let blob_name = metadata
        .get("blob")
        .and_then(|value| value.as_str())
        .ok_or_else(|| {
            crate::Error::Eval(CacheReadError::CorruptMetadata("missing blob".to_string())
                .message())
        })?;
    if blob_name.is_empty()
        || Path::new(blob_name).is_absolute()
        || !is_within(&directory, Path::new(blob_name))
    {
        return Ok(CacheLookup::Invalid(CacheReadError::UnsafePath(
            blob_name.to_string(),
        )));
    }
    let blob_path = directory.join(blob_name);
    let bytes = match caps.read_bytes(&blob_path).await {
        Ok(bytes) => bytes,
        Err(crate::Error::Io(_, error)) if error.kind() == std::io::ErrorKind::NotFound => {
            return Ok(CacheLookup::Miss);
        }
        Err(error) => return Err(error),
    };
    let actual_digest = digest_hex(&bytes);
    if actual_digest != recorded_digest {
        return Ok(CacheLookup::Invalid(CacheReadError::DigestMismatch {
            expected: recorded_digest.to_string(),
            actual: actual_digest,
        }));
    }
    Ok(CacheLookup::Hit(bytes))
}

/// Commit `bytes` for `url`, staging blob-first and metadata-last.
///
/// Fails atomically: an interruption at any [`WriteBoundary`] leaves any
/// previously committed entry intact and never produces a new hit.
pub async fn put_entry<C: EvalCapabilities + ?Sized>(
    caps: &mut C,
    cache_dir: &Path,
    url: &str,
    extension: &str,
    bytes: &[u8],
) -> Result<()> {
    let directory = entry_dir(cache_dir, url, extension);
    caps.create_dir_all(&directory).await?;
    let digest = digest_hex(bytes);
    let blob_name = digest.clone();
    stage_file(caps, &directory, &blob_name, bytes).await?;
    commit_metadata(caps, &directory, url, extension, &digest, &blob_name).await
}

async fn commit_metadata<C: EvalCapabilities + ?Sized>(
    caps: &mut C,
    directory: &Path,
    url: &str,
    extension: &str,
    digest: &str,
    blob_name: &str,
) -> Result<()> {
    let metadata = json!({
        "version": META_VERSION,
        "url": url,
        "extension": extension,
        "digest": digest,
        "blob": blob_name,
    });
    let bytes = serde_json::to_vec(&metadata).expect("metadata always serializes");
    stage_file(caps, directory, META_FILE_NAME, &bytes).await
}

async fn stage_file<C: EvalCapabilities + ?Sized>(
    caps: &mut C,
    directory: &Path,
    final_name: &str,
    bytes: &[u8],
) -> Result<()> {
    let serial = STAGING_COUNTER.fetch_add(1, Ordering::Relaxed);
    let staged = directory.join(staged_name(final_name, serial));
    if let Err(error) = caps.cache_write_new(&staged, bytes).await {
        return Err(error);
    }
    if let Err(error) = caps.cache_sync_file(&staged).await {
        let _ = caps.remove_file(&staged).await;
        return Err(error);
    }
    // Sync the directory while the file is still staged. This is the last
    // fault boundary before the commit: after this point the rename is the
    // only remaining step, so a failure at any of the injected boundaries can
    // never leave a partially committed entry behind.
    if let Err(error) = caps.cache_sync_dir(directory).await {
        let _ = caps.remove_file(&staged).await;
        return Err(error);
    }
    let final_path = directory.join(final_name);
    if let Err(error) = caps.cache_rename(&staged, &final_path).await {
        let _ = caps.remove_file(&staged).await;
        return Err(error);
    }
    Ok(())
}

/// Delete the persisted entry for `url` (best-effort; a missing entry is fine).
pub async fn evict_entry<C: EvalCapabilities + ?Sized>(
    caps: &mut C,
    cache_dir: &Path,
    url: &str,
    extension: &str,
) -> Result<()> {
    let directory = entry_dir(cache_dir, url, extension);
    match caps.cache_remove_dir_all(&directory).await {
        Ok(()) => Ok(()),
        Err(crate::Error::Io(_, error)) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error),
    }
}
