//! On-disk layout and validation primitives for the persistent package cache.
//!
//! Cache entries live under `<cache_dir>/packages/` and use two files:
//!
//! - `<fnv64(url)>-<digest>.blob` — the package bytes, content-addressed so a
//!   failed or interrupted write can never overwrite a usable entry.
//! - `<fnv64(url)>.meta.pklr` — small text metadata naming the URL, format
//!   version, and digest of the blob. This file is replaced last, so its
//!   presence is the commit point for an entry.
//!
//! Both files are written through temporary files named
//! `.<destination>.tmp-<pid>-<n>`. Stale temporary files left by a crashed
//! process are recognized by that suffix and swept out of the cache directory
//! on the next evaluation.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use crate::error::{Error, Result};

/// Metadata format version understood by this build.
pub const CACHE_METADATA_VERSION: u32 = 1;

const PACKAGES_DIR: &str = "packages";
const META_SUFFIX: &str = "meta.pklr";
const BLOB_SUFFIX: &str = "blob";

static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

/// 64-bit FNV-1a of `input`, used for filesystem-safe URL names.
pub fn fnv1a_64(input: &str) -> u64 {
    let mut hash = 0xcbf29ce484222325_u64;
    for byte in input.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

/// 128-bit FNV-1a digest of `bytes`, rendered as 32 lowercase hex characters.
///
/// This is a cheap integrity check for cache corruption, not a cryptographic
/// guarantee; downloads are already served over TLS.
pub fn digest_hex(bytes: &[u8]) -> String {
    let mut hash = 0x6c62272e07bb014262b821756295c58d_u128;
    for byte in bytes {
        hash ^= u128::from(*byte);
        hash = hash.wrapping_mul(0x0000000001000000000000000000013b);
    }
    format!("{hash:032x}")
}

/// Validate a package extension used to derive cache paths.
///
/// Extensions are interpolated into file names internally, so they must be a
/// single path component: no separators, parent traversal, or dots that could
/// escape the cache directory.
pub fn validate_extension(extension: &str) -> Result<()> {
    let valid = !extension.is_empty()
        && extension
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || ch == '-');
    if valid {
        Ok(())
    } else {
        Err(Error::Cache(format!(
            "invalid package cache extension: {extension:?}"
        )))
    }
}

fn packages_dir(cache_dir: &Path) -> Result<PathBuf> {
    let dir = cache_dir.join(PACKAGES_DIR);
    contained_path(cache_dir, &dir)?;
    Ok(dir)
}

fn contained_path(cache_dir: &Path, path: &Path) -> Result<()> {
    let packages = cache_dir.join(PACKAGES_DIR);
    if path.strip_prefix(&packages).is_ok() {
        Ok(())
    } else {
        Err(Error::Cache(format!(
            "package cache path {} escapes the cache directory {}",
            path.display(),
            packages.display()
        )))
    }
}

/// Metadata sidecar path for `url`.
pub fn meta_path(cache_dir: &Path, url: &str) -> Result<PathBuf> {
    let path = packages_dir(cache_dir)?
        .join(format!("{:016x}.{META_SUFFIX}", fnv1a_64(url)));
    contained_path(cache_dir, &path)?;
    Ok(path)
}

/// Content-addressed blob path for `url` carrying bytes with `digest`.
pub fn blob_path_for_digest(cache_dir: &Path, url: &str, digest: &str) -> Result<PathBuf> {
    let path = packages_dir(cache_dir)?
        .join(format!("{:016x}-{digest}.{BLOB_SUFFIX}", fnv1a_64(url)));
    contained_path(cache_dir, &path)?;
    Ok(path)
}

/// Content-addressed blob path for `url` carrying `bytes`.
pub fn blob_path(cache_dir: &Path, url: &str, bytes: &[u8]) -> Result<PathBuf> {
    blob_path_for_digest(cache_dir, url, &digest_hex(bytes))
}

/// Serialize cache metadata.
pub fn encode_metadata(url: &str, digest: &str) -> Vec<u8> {
    format!(
        "pklr-cache-version: {CACHE_METADATA_VERSION}\npklr-url: {url}\npklr-digest: {digest}\n"
    )
    .into_bytes()
}

/// Parsed cache metadata: the originating URL and blob digest.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Metadata {
    pub version: u32,
    pub url: String,
    pub digest: String,
}

/// Parse and validate cache metadata text.
pub fn decode_metadata(text: &str) -> Result<Metadata> {
    let mut version = None;
    let mut url = None;
    let mut digest = None;
    for line in text.split('\n') {
        if line.is_empty() {
            continue;
        }
        let (key, value) = line
            .split_once(':')
            .ok_or_else(|| Error::Cache(format!("malformed cache metadata line: {line:?}")))?;
        let value = value.strip_prefix(' ').unwrap_or(value);
        match key {
            "pklr-cache-version" => {
                let parsed = value.parse::<u32>().map_err(|_| {
                    Error::Cache(format!("malformed cache metadata version: {value:?}"))
                })?;
                version = Some(parsed);
            }
            "pklr-url" => url = Some(value.to_string()),
            "pklr-digest" => digest = Some(value.to_string()),
            other => {
                return Err(Error::Cache(format!(
                    "unknown cache metadata field: {other:?}"
                )));
            }
        }
    }
    let version = version
        .ok_or_else(|| Error::Cache("cache metadata is missing pklr-cache-version".to_string()))?;
    if version != CACHE_METADATA_VERSION {
        return Err(Error::Cache(format!(
            "unsupported cache metadata version: {version}"
        )));
    }
    let url = url
        .ok_or_else(|| Error::Cache("cache metadata is missing pklr-url".to_string()))?;
    let digest = digest
        .ok_or_else(|| Error::Cache("cache metadata is missing pklr-digest".to_string()))?;
    if digest.len() != 32 || !digest.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(Error::Cache(format!(
            "cache metadata has an invalid digest: {digest:?}"
        )));
    }
    Ok(Metadata {
        version,
        url,
        digest: digest.to_ascii_lowercase(),
    })
}

/// Verify that `bytes` match metadata and the package extension's basic
/// content guarantees (`.pkl` blobs must be UTF-8).
pub fn validate_stored_bytes(
    expected_url: &str,
    extension: &str,
    metadata: &Metadata,
    bytes: &[u8],
) -> Result<()> {
    if metadata.url != expected_url {
        return Err(Error::Cache(format!(
            "cache metadata URL mismatch: expected {expected_url}, found {}",
            metadata.url
        )));
    }
    let actual = digest_hex(bytes);
    if actual != metadata.digest {
        return Err(Error::Cache(format!(
            "cache digest mismatch for {expected_url}: metadata {}, bytes {actual}",
            metadata.digest
        )));
    }
    if extension == "pkl"
        && std::str::from_utf8(bytes).is_err()
    {
        return Err(Error::Cache(format!(
            "cached package source is not UTF-8: {expected_url}"
        )));
    }
    Ok(())
}

/// Temporary file name used while writing `destination`, recognizable later
/// by [`is_temp_file_name`].
pub fn temp_file_name(destination_name: &str) -> String {
    let counter = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
    format!(".{destination_name}.tmp-{}-{counter}", std::process::id())
}

/// Whether `name` is an unfinished staged cache write.
pub fn is_temp_file_name(name: &str) -> bool {
    name.starts_with('.') && name.contains(".tmp-")
}

fn sibling_temp_path(destination: &Path) -> Result<PathBuf> {
    let file_name = destination
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| Error::Cache(format!("invalid cache path: {}", destination.display())))?;
    Ok(destination.with_file_name(temp_file_name(file_name)))
}

/// Create a fresh, empty temporary file next to `destination`.
pub fn create_temp_file(destination: &Path) -> std::io::Result<PathBuf> {
    let temp_path = sibling_temp_path(destination)
        .map_err(|error| std::io::Error::new(std::io::ErrorKind::InvalidInput, error.to_string()))?;
    std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temp_path)?;
    Ok(temp_path)
}

/// Write `bytes` to a staged temporary file. A partial write may remain on
/// disk; callers decide whether to discard the staging file or leave it for
/// startup sweeping.
pub fn write_temp(temp_path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .truncate(true)
        .open(temp_path)?;
    file.write_all(bytes)
}

/// Flush staged bytes to disk and fsync the file.
pub fn flush_temp_file(temp_path: &Path) -> std::io::Result<()> {
    let mut file = std::fs::OpenOptions::new().write(true).open(temp_path)?;
    file.flush()?;
    file.sync_all()
}

/// Publish a staged temporary file at `destination`.
pub fn rename_temp_file(temp_path: &Path, destination: &Path) -> std::io::Result<()> {
    match std::fs::rename(temp_path, destination) {
        Ok(()) => Ok(()),
        // Windows rename does not replace an existing file.
        Err(_) if destination.exists() => {
            std::fs::remove_file(destination)?;
            std::fs::rename(temp_path, destination)
        }
        Err(error) => Err(error),
    }
}

/// Flush a directory entry (needed after rename on POSIX for durability).
pub fn sync_directory(dir: &Path) -> std::io::Result<()> {
    let file = std::fs::File::open(dir)?;
    file.sync_all()
}

/// Remove stale staged writes from each cache directory.
pub fn sweep_temp_files(dirs: &[PathBuf]) -> std::io::Result<Vec<PathBuf>> {
    let mut removed = Vec::new();
    for dir in dirs {
        let entries = match std::fs::read_dir(dir) {
            Ok(entries) => entries,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => continue,
            Err(error) => return Err(error),
        };
        for entry in entries {
            let entry = entry?;
            if entry.file_type()?.is_file()
                && entry
                    .file_name()
                    .to_str()
                    .is_some_and(is_temp_file_name)
            {
                std::fs::remove_file(entry.path())?;
                removed.push(entry.path());
            }
        }
    }
    Ok(removed)
}
