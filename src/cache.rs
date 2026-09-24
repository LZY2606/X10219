//! Persistent `package://` cache layout, integrity validation, and cleanup.
//!
//! This module is the dependency-injection boundary used by the evaluator's
//! package cache. It is public so embedders and contract tests can reason
//! about the on-disk layout, but it is not covered by semantic versioning.
//!
//! Layout under `<cache_dir>/packages/`:
//!
//! - `<urlhash>.meta` — JSON metadata: `version`, `url`, `extension`,
//!   `digest`, `length`. The presence of a valid meta file is what makes a
//!   blob a cache hit; a blob without valid metadata is never a hit.
//! - `<urlhash>.<digest>.<ext>` — content blob, named by its own digest so a
//!   newly written blob can never corrupt the entry an older meta points to.
//! - `.<name>.tmp-<pid>-<counter>` — in-flight atomic-write temp files. They
//!   are recognizable by name and removed by [`cleanup_stale`] on startup.

use std::path::{Path, PathBuf};

use crate::capabilities::EvalCapabilities;
use crate::error::{Error, Result};

/// Cache metadata format version. Entries with any other version are rejected.
pub const CACHE_FORMAT_VERSION: u64 = 1;

/// Stable, filesystem-safe hash of a package URL (FNV-1a, 64-bit, hex).
pub fn url_hash(url: &str) -> String {
    let mut hash = 0xcbf29ce484222325_u64;
    for byte in url.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    format!("{hash:016x}")
}

/// Content digest used for cache integrity (FNV-1a, 128-bit, hex).
///
/// This guards against torn writes and corruption, not adversarial input;
/// it keeps the cache free of platform-specific crypto dependencies.
pub fn content_digest(bytes: &[u8]) -> String {
    let mut hash = 0x6c62272e07bb014262b821756295c58d_u128;
    for byte in bytes {
        hash ^= u128::from(*byte);
        hash = hash.wrapping_mul(0x0000_0000_0100_0000_0000_0000_0000_013b);
    }
    format!("{hash:032x}")
}

/// Directory that holds all cache entries for `cache_dir`.
pub fn packages_dir(cache_dir: &Path) -> PathBuf {
    cache_dir.join("packages")
}

/// Path of the metadata file for `url`.
pub fn meta_path(cache_dir: &Path, url: &str) -> PathBuf {
    packages_dir(cache_dir).join(format!("{}.meta", url_hash(url)))
}

/// Path of the content blob for `url` with content digest `digest`.
///
/// The digest becomes part of a file name, so anything that is not a plain
/// 32-character hex digest is rejected: a digest containing separators or
/// `..` would point outside the cache directory.
pub fn blob_path(cache_dir: &Path, url: &str, digest: &str, extension: &str) -> Result<PathBuf> {
    if digest.contains('/')
        || digest.contains('\\')
        || digest.contains("..")
        || digest.contains('\0')
    {
        return Err(Error::Eval(format!(
            "package cache metadata points outside the cache directory: {url}"
        )));
    }
    if digest.len() != 32 || !digest.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err(Error::Eval(format!(
            "corrupt package cache metadata for {url}: malformed digest"
        )));
    }
    Ok(packages_dir(cache_dir).join(format!(
        "{}.{digest}.{extension}",
        url_hash(url)
    )))
}

/// Whether `name` looks like an in-flight atomic-write temp file.
pub fn is_temp_file_name(name: &str) -> bool {
    let Some(rest) = name.strip_prefix('.') else {
        return false;
    };
    let Some((_, suffix)) = rest.rsplit_once(".tmp-") else {
        return false;
    };
    // Temp names end in `.tmp-<pid>-<counter>`.
    let mut parts = suffix.split('-');
    let pid = parts.next().unwrap_or("");
    let counter = parts.next().unwrap_or("");
    parts.next().is_none()
        && !pid.is_empty()
        && pid.bytes().all(|b| b.is_ascii_digit())
        && !counter.is_empty()
        && counter.bytes().all(|b| b.is_ascii_digit())
}

/// Parse a blob file name into `(urlhash, digest)` when it matches the layout.
fn parse_blob_file_name(name: &str) -> Option<(&str, &str)> {
    let mut parts = name.split('.');
    let urlhash = parts.next()?;
    let digest = parts.next()?;
    let extension = parts.next()?;
    if parts.next().is_some() {
        return None;
    }
    if urlhash.len() != 16
        || !urlhash.bytes().all(|b| b.is_ascii_hexdigit())
        || digest.len() != 32
        || !digest.bytes().all(|b| b.is_ascii_hexdigit())
        || extension.is_empty()
        || extension == "meta"
        || !extension.bytes().all(|b| b.is_ascii_alphanumeric())
    {
        return None;
    }
    Some((urlhash, digest))
}

struct Metadata {
    url: String,
    extension: String,
    digest: String,
    length: u64,
}

fn encode_metadata(url: &str, extension: &str, digest: &str, length: u64) -> String {
    // serde_json::json! performs the string escaping for us.
    serde_json::json!({
        "version": CACHE_FORMAT_VERSION,
        "url": url,
        "extension": extension,
        "digest": digest,
        "length": length,
    })
    .to_string()
}

fn decode_metadata(url: &str, text: &str) -> Result<Metadata> {
    let corrupt = |reason: &str| {
        Error::Eval(format!(
            "corrupt package cache metadata for {url}: {reason}"
        ))
    };
    let value: serde_json::Value =
        serde_json::from_str(text).map_err(|error| corrupt(&error.to_string()))?;
    let version = value
        .get("version")
        .and_then(serde_json::Value::as_u64)
        .ok_or_else(|| corrupt("missing version"))?;
    if version != CACHE_FORMAT_VERSION {
        return Err(Error::Eval(format!(
            "unsupported package cache metadata version {version} for {url} (expected {CACHE_FORMAT_VERSION})"
        )));
    }
    let get_str = |key: &str| {
        value
            .get(key)
            .and_then(serde_json::Value::as_str)
            .map(str::to_string)
            .ok_or_else(|| corrupt(&format!("missing {key}")))
    };
    let length = value
        .get("length")
        .and_then(serde_json::Value::as_u64)
        .ok_or_else(|| corrupt("missing length"))?;
    Ok(Metadata {
        url: get_str("url")?,
        extension: get_str("extension")?,
        digest: get_str("digest")?,
        length,
    })
}

/// Read a cache entry, validating metadata, version, digest, and length.
///
/// Returns `Ok(None)` on a plain miss (no metadata, or a hash collision where
/// the stored URL or extension differs). Corrupt metadata, unsupported
/// versions, digest mismatches, and metadata pointing outside the cache
/// directory are rejected with distinct errors.
pub async fn read_entry(
    capabilities: &mut dyn EvalCapabilities,
    cache_dir: &Path,
    url: &str,
    extension: &str,
) -> Result<Option<Vec<u8>>> {
    let meta_file = meta_path(cache_dir, url);
    let meta_text = match capabilities.read_to_string(&meta_file).await {
        Ok(text) => text,
        Err(Error::Io(_, error)) if error.kind() == std::io::ErrorKind::NotFound => {
            return Ok(None);
        }
        Err(error) => return Err(error),
    };
    let meta = decode_metadata(url, &meta_text)?;
    if meta.url != url || meta.extension != extension {
        // Hash collision or a different representation of the same URL: a
        // plain miss, matching the behavior of the legacy URL sidecar.
        return Ok(None);
    }
    let blob_file = blob_path(cache_dir, url, &meta.digest, extension)?;
    let bytes = match capabilities.read_bytes(&blob_file).await {
        Ok(bytes) => bytes,
        Err(Error::Io(_, error)) if error.kind() == std::io::ErrorKind::NotFound => {
            return Err(Error::Eval(format!(
                "corrupt package cache for {url}: metadata exists but blob is missing"
            )));
        }
        Err(error) => return Err(error),
    };
    if bytes.len() as u64 != meta.length || content_digest(&bytes) != meta.digest {
        return Err(Error::Eval(format!(
            "package cache digest mismatch for {url}"
        )));
    }
    Ok(Some(bytes))
}

/// Write a cache entry. The blob is written under its digest name first and
/// the metadata is committed last, so a failure at any point leaves a
/// previously cached entry fully usable and never exposes a half-written
/// file as a hit.
pub async fn write_entry(
    capabilities: &mut dyn EvalCapabilities,
    cache_dir: &Path,
    url: &str,
    extension: &str,
    bytes: &[u8],
) -> Result<()> {
    let digest = content_digest(bytes);
    let blob_file = blob_path(cache_dir, url, &digest, extension)?;
    let meta_file = meta_path(cache_dir, url);
    let Some(parent) = blob_file.parent() else {
        return Ok(());
    };
    capabilities.create_dir_all(parent).await?;
    capabilities.write_atomic(&blob_file, bytes).await?;
    let meta = encode_metadata(url, extension, &digest, bytes.len() as u64);
    capabilities.write_atomic(&meta_file, meta.as_bytes()).await
}

/// Remove the cache entry for `url`, including its blob when known.
pub async fn remove_entry(
    capabilities: &mut dyn EvalCapabilities,
    cache_dir: &Path,
    url: &str,
    extension: &str,
) -> Result<()> {
    let meta_file = meta_path(cache_dir, url);
    if let Ok(text) = capabilities.read_to_string(&meta_file).await
        && let Ok(meta) = decode_metadata(url, &text)
        && let Ok(blob_file) = blob_path(cache_dir, url, &meta.digest, extension)
    {
        let _ = capabilities.remove_file(&blob_file).await;
    }
    match capabilities.remove_file(&meta_file).await {
        Ok(()) => Ok(()),
        Err(Error::Io(_, error)) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error),
    }
}

/// Remove recognizable stale files from a previous run: in-flight temp files
/// and blobs no valid metadata references. Unknown files are left untouched.
pub async fn cleanup_stale(
    capabilities: &mut dyn EvalCapabilities,
    cache_dir: &Path,
) -> Result<()> {
    let packages = packages_dir(cache_dir);
    let entries = capabilities.list_dir(&packages).await?;
    let mut referenced_blobs: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut temp_files = Vec::new();
    let mut blobs = Vec::new();
    for entry in entries {
        let Some(name) = entry
            .file_name()
            .and_then(|name| name.to_str().map(str::to_string))
        else {
            continue;
        };
        if is_temp_file_name(&name) {
            temp_files.push(entry);
        } else if let Some((urlhash, digest)) = parse_blob_file_name(&name) {
            blobs.push((entry, format!("{urlhash}.{digest}")));
        } else if let Some(urlhash) = name.strip_suffix(".meta") {
            let urlhash = urlhash.to_string();
            if let Ok(text) = capabilities.read_to_string(&entry).await
                && let Ok(value) = serde_json::from_str::<serde_json::Value>(&text)
                && value.get("version").and_then(serde_json::Value::as_u64)
                    == Some(CACHE_FORMAT_VERSION)
                && let Some(digest) = value.get("digest").and_then(serde_json::Value::as_str)
            {
                referenced_blobs.insert(format!("{urlhash}.{digest}"));
            }
        }
    }
    for temp in temp_files {
        let _ = capabilities.remove_file(&temp).await;
    }
    for (blob, key) in blobs {
        if !referenced_blobs.contains(&key) {
            let _ = capabilities.remove_file(&blob).await;
        }
    }
    Ok(())
}
