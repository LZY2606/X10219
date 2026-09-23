//! Persistent, content-addressed package cache.
//!
//! Each cached package is stored as a blob plus a metadata sidecar:
//!
//! ```text
//! <cache-dir>/packages/blobs/<sha256>.<ext>        content-addressed bytes
//! <cache-dir>/packages/meta/<fnv1a64(url)>.<ext>.json   metadata sidecar
//! <cache-dir>/tmp/.stage-...                        write staging files
//! ```
//!
//! The metadata sidecar is the *commit* marker: it is written last and
//! atomically renamed into place, so a crash at any earlier boundary leaves
//! the previous committed entry fully readable. Staging files live under a
//! dedicated `tmp` directory and are reclaimed on the next startup.
//!
//! Metadata integrity is checked on every read: unsupported schema versions,
//! malformed JSON, digest mismatches, and metadata pointing outside of the
//! blob directory are all rejected instead of being treated as hits.

use std::path::{Component, Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use crate::capabilities::EvalCapabilities;
use crate::sha256::digest_hex;
use crate::{Error, Result};

pub(crate) const METADATA_VERSION: u32 = 1;
const STAGING_PREFIX: &str = ".stage-";

#[derive(Debug, PartialEq, Eq)]
pub(crate) struct PackageMetadata {
    pub(crate) version: u32,
    pub(crate) url: String,
    pub(crate) extension: String,
    pub(crate) sha256: String,
    pub(crate) blob: String,
}

/// Storage locations derived from a package URL.
#[derive(Debug)]
pub(crate) struct CachePaths {
    pub(crate) root: PathBuf,
    pub(crate) metadata: PathBuf,
}

impl CachePaths {
    pub(crate) fn for_url(cache_dir: &Path, url: &str, extension: &str) -> Self {
        let mut hash = 0xcbf29ce484222325_u64;
        for byte in url.as_bytes() {
            hash ^= u64::from(*byte);
            hash = hash.wrapping_mul(0x100000001b3);
        }
        let metadata = cache_dir
            .join("packages")
            .join("meta")
            .join(format!("{hash:016x}.{extension}.json"));
        Self {
            root: cache_dir.to_path_buf(),
            metadata,
        }
    }

    fn blobs_dir(&self) -> PathBuf {
        self.root.join("packages").join("blobs")
    }

    fn tmp_dir(&self) -> PathBuf {
        self.root.join("tmp")
    }

    fn blob_path(&self, file_name: &str) -> PathBuf {
        self.blobs_dir().join(file_name)
    }
}

/// Why a cache lookup could not return a hit.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum CacheRejection {
    Unreadable,
    UnsupportedVersion { found: u32 },
    MalformedMetadata,
    DigestMismatch,
    UnsafeBlobPath,
    BlobMissing,
}

impl CacheRejection {
    pub(crate) fn as_error(&self, url: &str) -> Error {
        match self {
            CacheRejection::UnsupportedVersion { found } => Error::Eval(format!(
                "package cache metadata uses unsupported version {found} for {url} (supported: {METADATA_VERSION})"
            )),
            CacheRejection::MalformedMetadata => {
                Error::Eval(format!("package cache metadata is corrupt for {url}"))
            }
            CacheRejection::DigestMismatch => {
                Error::Eval(format!("package cache digest mismatch for {url}"))
            }
            CacheRejection::UnsafeBlobPath => Error::Eval(format!(
                "package cache metadata points outside the blob directory for {url}"
            )),
            CacheRejection::Unreadable => {
                Error::Eval(format!("package cache metadata is unreadable for {url}"))
            }
            CacheRejection::BlobMissing => {
                Error::Eval(format!("package cache blob is missing for {url}"))
            }
        }
    }
}

fn metadata_string_field<'a>(
    value: &'a serde_json::Map<String, serde_json::Value>,
    name: &str,
) -> std::result::Result<&'a str, CacheRejection> {
    value
        .get(name)
        .and_then(|field| field.as_str())
        .ok_or(CacheRejection::MalformedMetadata)
}

fn parse_metadata(bytes: &[u8]) -> std::result::Result<PackageMetadata, CacheRejection> {
    let value: serde_json::Value =
        serde_json::from_slice(bytes).map_err(|_| CacheRejection::MalformedMetadata)?;
    let object = value
        .as_object()
        .ok_or(CacheRejection::MalformedMetadata)?;
    let version = object
        .get("version")
        .and_then(|field| field.as_u64())
        .ok_or(CacheRejection::MalformedMetadata)? as u32;
    if version != METADATA_VERSION {
        return Err(CacheRejection::UnsupportedVersion { found: version });
    }
    Ok(PackageMetadata {
        version,
        url: metadata_string_field(object, "url")?.to_string(),
        extension: metadata_string_field(object, "extension")?.to_string(),
        sha256: metadata_string_field(object, "sha256")?.to_string(),
        blob: metadata_string_field(object, "blob")?.to_string(),
    })
}

/// Reject any metadata `blob` name that is not a plain file located directly
/// inside the cache blob directory.
fn safe_blob_path(paths: &CachePaths, blob: &str) -> Option<PathBuf> {
    if blob.is_empty() || blob.contains('/') || blob.contains('\\') || blob == "." || blob == ".."
    {
        return None;
    }
    let candidate = paths.blob_path(blob);
    if candidate.components().any(|component| {
        matches!(
            component,
            Component::ParentDir | Component::RootDir | Component::Prefix(_)
        )
    }) {
        return None;
    }
    let normalized = normalize_path(&candidate);
    if !normalized.starts_with(&normalize_path(&paths.blobs_dir())) {
        return None;
    }
    Some(candidate)
}

fn normalize_path(path: &Path) -> PathBuf {
    let mut normalized = Vec::new();
    for component in path.components() {
        match component {
            Component::ParentDir => {
                normalized.pop();
            }
            Component::CurDir => {}
            other => normalized.push(other.as_os_str().to_os_string()),
        }
    }
    normalized.into_iter().collect()
}

/// Load and fully validate a cache entry.
pub(crate) async fn read_entry(
    capabilities: &mut dyn EvalCapabilities,
    cache_dir: &Path,
    url: &str,
    extension: &str,
) -> std::result::Result<Option<Vec<u8>>, CacheRejection> {
    let paths = CachePaths::for_url(cache_dir, url, extension);
    let raw = match capabilities.read_to_string(&paths.metadata).await {
        Ok(raw) => raw,
        Err(Error::Io(_, error)) if error.kind() == std::io::ErrorKind::NotFound => {
            return Ok(None)
        }
        Err(_) => return Err(CacheRejection::Unreadable),
    };
    let metadata = parse_metadata(raw.as_bytes())?;
    if metadata.url != url || metadata.extension != extension {
        return Err(CacheRejection::MalformedMetadata);
    }
    if metadata.sha256.len() != 64
        || !metadata.sha256.chars().all(|c| c.is_ascii_hexdigit())
    {
        return Err(CacheRejection::MalformedMetadata);
    }
    let blob_path = safe_blob_path(&paths, &metadata.blob).ok_or(CacheRejection::UnsafeBlobPath)?;
    let bytes = match capabilities.read_bytes(&blob_path).await {
        Ok(bytes) => bytes,
        Err(Error::Io(_, error)) if error.kind() == std::io::ErrorKind::NotFound => {
            return Err(CacheRejection::BlobMissing)
        }
        Err(_) => return Err(CacheRejection::BlobMissing),
    };
    if digest_hex(&bytes) != metadata.sha256 {
        return Err(CacheRejection::DigestMismatch);
    }
    Ok(Some(bytes))
}

fn metadata_json(url: &str, extension: &str, digest: &str, blob: &str) -> Vec<u8> {
    serde_json::json!({
        "version": METADATA_VERSION,
        "url": url,
        "extension": extension,
        "sha256": digest,
        "blob": blob,
    })
    .to_string()
    .into_bytes()
}

/// Stage and commit `bytes` for `url`.
///
/// Failures injected at [`CacheWriteFault`] boundaries surface to the caller.
/// The commit order (blob first, metadata sidecar last) guarantees a
/// pre-existing committed entry is never turned into a half-file hit.
pub(crate) async fn write_entry(
    capabilities: &mut dyn EvalCapabilities,
    cache_dir: &Path,
    url: &str,
    extension: &str,
    bytes: &[u8],
) -> Result<()> {
    let paths = CachePaths::for_url(cache_dir, url, extension);
    let digest = digest_hex(bytes);
    let blob_name = format!("{digest}.{extension}");
    let blob_path = paths.blob_path(&blob_name);
    let tmp_dir = paths.tmp_dir();

    capabilities
        .durable_write(&blob_path, &tmp_dir, bytes, false)
        .await?;
    capabilities
        .durable_write(
            &paths.metadata,
            &tmp_dir,
            &metadata_json(url, extension, &digest, &blob_name),
            true,
        )
        .await?;
    Ok(())
}

/// Remove a rejected metadata sidecar. Content-addressed blobs are left alone:
/// they are immutable and shared between sidecars.
pub(crate) async fn evict_metadata(
    capabilities: &mut dyn EvalCapabilities,
    cache_dir: &Path,
    url: &str,
    extension: &str,
) {
    let paths = CachePaths::for_url(cache_dir, url, extension);
    let _ = capabilities.remove_file(&paths.metadata).await;
}

/// Delete leftover staging files from interrupted writes. This is safe to run
/// on every startup: only files matching the staging prefix are removed and
/// a failure to list or remove one entry does not stop the sweep.
pub(crate) async fn sweep_staging(
    capabilities: &mut dyn EvalCapabilities,
    cache_dir: &Path,
) -> Result<()> {
    let tmp_dir = cache_dir.join("tmp");
    let entries = capabilities.list_dir(&tmp_dir).await?;
    for entry in entries {
        let is_staging = entry
            .file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| name.starts_with(STAGING_PREFIX));
        if is_staging {
            let _ = capabilities.remove_file(&entry).await;
        }
    }
    Ok(())
}

static STAGING_COUNTER: AtomicU64 = AtomicU64::new(0);

/// The staging file name used by the durable-write protocol, exposed so tests
/// can stage interrupted writes deterministically.
pub(crate) fn staging_name(destination: &Path) -> Option<String> {
    let file_name = destination
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("cache")
        .replace('/', "_");
    Some(format!(
        "{STAGING_PREFIX}{file_name}-{}-{}",
        std::process::id(),
        STAGING_COUNTER.fetch_add(1, Ordering::Relaxed)
    ))
}
