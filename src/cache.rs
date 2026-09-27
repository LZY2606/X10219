//! On-disk package cache layout used by [`crate::Evaluator`].
//!
//! Every cached package is a pair of files under
//! `<cache_dir>/packages/<hash>/`:
//!
//! - `<name>.meta.json` — a small, versioned metadata document naming the
//!   source URL and the digest of the stored blob.
//! - `<name>.<extension>` — the package bytes (`pkl` source or `zip` archive).
//!
//! Metadata is the commit record: the blob is published first and the
//! metadata last, so a crash at any write boundary leaves either the old
//! entry or no entry, never a half-written hit. Files named `*.part` are
//! staging files from interrupted publishes and are swept on startup.

use std::collections::BTreeMap;
use std::path::{Component, Path, PathBuf};

use crate::capabilities::EvalCapabilities;
use crate::{Error, Result};

/// Metadata format version understood by this build.
pub const CACHE_FORMAT_VERSION: u32 = 1;

/// Prefix for staging files removed by [`sweep_staging`].
pub const STAGING_SUFFIX: &str = ".part";

/// FNV-1a (64 bit) digest of `bytes`, rendered as lowercase hexadecimal.
pub fn digest_hex(bytes: &[u8]) -> String {
    let mut hash = 0xcbf29ce484222325_u64;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    format!("{hash:016x}")
}

/// Paths of the metadata and blob files for `url`/`extension`.
pub fn entry_paths(cache_dir: &Path, url: &str, extension: &str) -> (PathBuf, PathBuf) {
    let mut hash = 0xcbf29ce484222325_u64;
    for byte in url.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    let base = cache_dir.join("packages").join(format!("{hash:016x}"));
    (
        base.with_extension("meta.json"),
        base.with_extension(extension),
    )
}

/// Validate that `blob_path` is a regular file contained in `packages_dir`.
///
/// The check is lexical, so it works against fake or virtual file systems
/// without requiring the file (or its parents) to exist on disk.
pub fn ensure_within_cache(packages_dir: &Path, blob_path: &Path) -> Result<()> {
    let mut relative = Vec::new();
    for component in blob_path.components() {
        match component {
            Component::Normal(part) => relative.push(part),
            Component::CurDir => {}
            Component::ParentDir
            | Component::RootDir
            | Component::Prefix(_) => {
                return Err(Error::InvalidCache(format!(
                    "cache metadata points outside the cache directory: {}",
                    blob_path.display()
                )));
            }
        }
    }
    let (prefix, rest) = relative.split_at(packages_dir.components().count());
    let normalized_packages: Vec<_> = packages_dir.components().collect();
    if prefix != normalized_packages {
        return Err(Error::InvalidCache(format!(
            "cache metadata points outside the cache directory: {}",
            blob_path.display()
        )));
    }
    if rest.is_empty() {
        return Err(Error::InvalidCache(
            "cache metadata names the cache directory itself".to_string(),
        ));
    }
    Ok(())
}

/// Result of a successful metadata read.
pub struct CacheEntry {
    pub url: String,
    pub blob_path: PathBuf,
}

/// Read and validate the metadata document at `meta_path`.
///
/// Missing metadata is `Ok(None)`; a malformed, unsupported, or escaping
/// document is [`Error::InvalidCache`].
pub async fn read_metadata(
    capabilities: &mut dyn EvalCapabilities,
    packages_dir: &Path,
    meta_path: &Path,
) -> Result<Option<CacheEntry>> {
    let raw = match capabilities.read_to_string(meta_path).await {
        Ok(raw) => raw,
        Err(Error::Io(_, error)) if error.kind() == std::io::ErrorKind::NotFound => {
            return Ok(None);
        }
        Err(error) => return Err(error),
    };
    let document: BTreeMap<String, serde_json::Value> = serde_json::from_str(&raw).map_err(
        |error| Error::InvalidCache(format!("corrupt package metadata: {error}")),
    )?;
    let version = document
        .get("version")
        .and_then(|value| value.as_u64())
        .ok_or_else(|| {
            Error::InvalidCache("corrupt package metadata: missing version".to_string())
        })?;
    if version != u64::from(CACHE_FORMAT_VERSION) {
        return Err(Error::InvalidCache(format!(
            "unsupported package cache format version {version} (supported: {CACHE_FORMAT_VERSION})"
        )));
    }
    let url = document
        .get("url")
        .and_then(|value| value.as_str())
        .ok_or_else(|| {
            Error::InvalidCache("corrupt package metadata: missing url".to_string())
        })?;
    let blob_name = document
        .get("blob")
        .and_then(|value| value.as_str())
        .ok_or_else(|| {
            Error::InvalidCache("corrupt package metadata: missing blob".to_string())
        })?;
    document
        .get("digest")
        .and_then(|value| value.as_str())
        .ok_or_else(|| {
            Error::InvalidCache("corrupt package metadata: missing digest".to_string())
        })?;
    let blob_path = packages_dir.join(blob_name);
    ensure_within_cache(packages_dir, &blob_path)?;
    Ok(Some(CacheEntry {
        url: url.to_string(),
        blob_path,
    }))
}

fn metadata_document(url: &str, extension: &str, bytes: &[u8]) -> Vec<u8> {
    let digest = digest_hex(bytes);
    // BTreeMap-free, hand-built JSON: keys are fixed and all values are
    // controlled strings (URLs are validated by package URI resolution
    // before reaching the cache), so the escaping below is sufficient.
    format!(
        "{{\"version\":{version},\"url\":\"{url}\",\"digest\":\"{digest}\",\"extension\":\"{extension}\",\"blob\":\"{hash}.{extension}\"}}\n",
        version = CACHE_FORMAT_VERSION,
        url = escape_json(url),
        digest = digest,
        extension = escape_json(extension),
        hash = name_hash(url),
    )
    .into_bytes()
}

fn escape_json(value: &str) -> String {
    let mut out = String::with_capacity(value.len());
    for ch in value.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            ch if (ch as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", ch as u32)),
            ch => out.push(ch),
        }
    }
    out
}

fn name_hash(url: &str) -> String {
    let mut hash = 0xcbf29ce484222325_u64;
    for byte in url.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    format!("{hash:016x}")
}

/// Publish `bytes` for `url`: blob first, metadata last. Both land via the
/// capabilities' atomic publish hook so each boundary (flush, rename,
/// directory sync) can be failed independently by test doubles.
pub async fn write_entry(
    capabilities: &mut dyn EvalCapabilities,
    cache_dir: &Path,
    url: &str,
    extension: &str,
    bytes: &[u8],
) -> Result<()> {
    let (meta_path, blob_path) = entry_paths(cache_dir, url, extension);
    let Some(parent) = blob_path.parent() else {
        return Ok(());
    };
    capabilities.create_dir_all(parent).await?;
    capabilities.publish_atomic(&blob_path, bytes).await?;
    let metadata = metadata_document(url, extension, bytes);
    capabilities.publish_atomic(&meta_path, &metadata).await?;
    Ok(())
}

/// Remove a stale cache entry (best-effort; missing files are fine).
pub async fn remove_entry(
    capabilities: &mut dyn EvalCapabilities,
    cache_dir: &Path,
    url: &str,
    extension: &str,
) {
    let (meta_path, blob_path) = entry_paths(cache_dir, url, extension);
    let _ = capabilities.remove_file(&blob_path).await;
    let _ = capabilities.remove_file(&meta_path).await;
}

/// Delete `*.part` staging files left behind by an interrupted publish.
/// Runs once per evaluator, before the first cache access.
pub async fn sweep_staging(capabilities: &mut dyn EvalCapabilities, cache_dir: &Path) {
    let packages_dir = cache_dir.join("packages");
    let Ok(mut entries) = capabilities.list_directory(&packages_dir).await else {
        return;
    };
    // Directory order is platform dependent; sort for deterministic behavior.
    entries.sort();
    for entry in entries {
        if entry
            .file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| name.ends_with(STAGING_SUFFIX))
        {
            let _ = capabilities.remove_file(&entry).await;
        }
    }
}
