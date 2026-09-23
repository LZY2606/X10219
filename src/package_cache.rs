//! On-disk package cache with digest-checked metadata.
//!
//! Layout under a configured cache directory:
//!
//! ```text
//! <cache>/packages/<fnv1a>.meta.json  metadata (v1), committed last
//! <cache>/blobs/<sha256>.<ext>       content-addressed package bytes
//! <cache>/packages/<fnv1a>.<ext>     legacy blob layout (pre-v1 caches)
//! <cache>/packages/<fnv1a>.url       legacy URL sidecar (pre-v1 caches)
//! ```
//!
//! Writes are staged as `*.pklrtmp` temporary files. Every commit boundary
//! (create/write, flush, rename, directory sync) is a distinct capabilities
//! call so hosts can inject failures, and stale temporary files are swept the
//! first time a cache directory is used by a new evaluator.

use std::path::{Component, Path, PathBuf};

use serde_json::json;

use crate::capabilities::EvalCapabilities;
use crate::{Error, Result};

/// Metadata format version this build reads and writes.
pub const METADATA_VERSION: u32 = 1;
/// Filename suffix used for every staged cache write.
pub const TEMP_SUFFIX: &str = "pklrtmp";

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum FileKind {
    Blob,
    Metadata,
}

impl FileKind {
    pub fn as_str(self) -> &'static str {
        match self {
            FileKind::Blob => "blob",
            FileKind::Metadata => "metadata",
        }
    }
}

#[derive(Debug, Clone)]
pub struct CacheEntry {
    pub url: String,
    pub extension: String,
    pub bytes: Vec<u8>,
}

/// FNV-1a hash of a package URL, matching the historical cache file naming.
pub fn url_hash(url: &str) -> u64 {
    let mut hash = 0xcbf29ce484222325_u64;
    for byte in url.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

pub fn meta_path(cache_dir: &Path, url: &str) -> PathBuf {
    cache_dir
        .join("packages")
        .join(format!("{:016x}.meta.json", url_hash(url)))
}

pub fn legacy_paths(cache_dir: &Path, url: &str, extension: &str) -> (PathBuf, PathBuf) {
    let base = cache_dir
        .join("packages")
        .join(format!("{:016x}", url_hash(url)));
    (base.with_extension(extension), base.with_extension("url"))
}

pub fn blob_path(cache_dir: &Path, digest: &str, extension: &str) -> PathBuf {
    cache_dir
        .join("blobs")
        .join(format!("{digest}.{extension}"))
}

/// Resolve `relative` against `root` and reject anything that escapes it.
///
/// The check is lexical (`..`, absolute paths, prefixes, Windows prefixes), so
/// it works identically on an in-memory fake filesystem and on a real disk.
pub fn resolve_within_root(root: &Path, relative: &str) -> Result<PathBuf> {
    let candidate = Path::new(relative);
    if candidate.is_absolute()
        || candidate
            .components()
            .any(|component| !matches!(component, Component::Normal(_)))
    {
        return Err(Error::Eval(format!(
            "cache path escapes the cache directory: {relative}"
        )));
    }
    Ok(root.join(candidate))
}

fn hex_lower(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        out.push(std::char::from_digit(u32::from(*byte) >> 4, 16).unwrap());
        out.push(std::char::from_digit(u32::from(*byte) & 0xf, 16).unwrap());
    }
    out
}

/// SHA-256 digest of `bytes` as lowercase hex, matching `sha256sum` output.
pub fn sha256_hex(bytes: &[u8]) -> String {
    hex_lower(&sha256::digest(bytes))
}

/// Minimal self-contained SHA-256 implementation (FIPS 180-4).
mod sha256 {
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ];

    pub fn digest(data: &[u8]) -> [u8; 32] {
        let mut hash = [
            0x6a09e667_u32,
            0xbb67ae85,
            0x3c6ef372,
            0xa54ff53a,
            0x510e527f,
            0x9b05688c,
            0x1f83d9ab,
            0x5be0cd19,
        ];

        let bit_len = u64::try_from(data.len()).unwrap() * 8;
        let mut padded = data.to_vec();
        padded.push(0x80);
        while padded.len() % 64 != 56 {
            padded.push(0);
        }
        padded.extend_from_slice(&bit_len.to_be_bytes());

        for chunk in padded.chunks_exact(64) {
            let mut w = [0_u32; 64];
            for (index, word) in chunk.chunks_exact(4).enumerate() {
                w[index] = u32::from_be_bytes([word[0], word[1], word[2], word[3]]);
            }
            for index in 16..64 {
                let s0 = w[index - 15].rotate_right(7) ^ w[index - 15].rotate_right(18)
                    ^ (w[index - 15] >> 3);
                let s1 = w[index - 2].rotate_right(17) ^ w[index - 2].rotate_right(19)
                    ^ (w[index - 2] >> 10);
                w[index] = w[index - 16]
                    .wrapping_add(s0)
                    .wrapping_add(w[index - 7])
                    .wrapping_add(s1);
            }

            let mut working = hash;
            for index in 0..64 {
                let s1 = working[4].rotate_right(6) ^ working[4].rotate_right(11)
                    ^ working[4].rotate_right(25);
                let choice = (working[4] & working[5]) ^ ((!working[4]) & working[6]);
                let temp1 = working[7]
                    .wrapping_add(s1)
                    .wrapping_add(choice)
                    .wrapping_add(K[index])
                    .wrapping_add(w[index]);
                let s0 = working[0].rotate_right(2) ^ working[0].rotate_right(13)
                    ^ working[0].rotate_right(22);
                let majority =
                    (working[0] & working[1]) ^ (working[0] & working[2]) ^ (working[1] & working[2]);
                let temp2 = s0.wrapping_add(majority);
                working[7] = working[6];
                working[6] = working[5];
                working[5] = working[4];
                working[4] = working[3].wrapping_add(temp1);
                working[3] = working[2];
                working[2] = working[1];
                working[1] = working[0];
                working[0] = temp1.wrapping_add(temp2);
            }
            for slot in 0..8 {
                hash[slot] = hash[slot].wrapping_add(working[slot]);
            }
        }

        let mut out = [0_u8; 32];
        for (word, slot) in hash.iter().zip(out.chunks_exact_mut(4)) {
            slot.copy_from_slice(&word.to_be_bytes());
        }
        out
    }
}

#[derive(Debug)]
struct Metadata {
    version: u32,
    url: String,
    extension: String,
    sha256: String,
    blob: String,
}

async fn read_metadata(caps: &mut dyn EvalCapabilities, path: &Path) -> Result<Option<Metadata>> {
    let raw = match caps.read_to_string(path).await {
        Ok(raw) => raw,
        Err(Error::Io(_, error)) if error.kind() == std::io::ErrorKind::NotFound => {
            return Ok(None)
        }
        Err(error) => return Err(error),
    };
    let value: serde_json::Value = serde_json::from_str(&raw).map_err(|error| {
        Error::Eval(format!(
            "corrupt package cache metadata at {}: {error}",
            path.display()
        ))
    })?;
    let object = value.as_object().ok_or_else(|| {
        Error::Eval(format!(
            "corrupt package cache metadata at {}: not a JSON object",
            path.display()
        ))
    })?;
    let field_str = |name: &str| -> Result<String> {
        object
            .get(name)
            .and_then(|value| value.as_str())
            .map(str::to_string)
            .ok_or_else(|| {
                Error::Eval(format!(
                    "corrupt package cache metadata at {}: missing string field '{name}'",
                    path.display()
                ))
            })
    };
    let version = object
        .get("version")
        .and_then(|value| value.as_u64())
        .ok_or_else(|| {
            Error::Eval(format!(
                "corrupt package cache metadata at {}: missing numeric field 'version'",
                path.display()
            ))
        })?;
    let metadata = Metadata {
        version: u32::try_from(version).map_err(|_| {
            Error::Eval(format!(
                "corrupt package cache metadata at {}: invalid version value",
                path.display()
            ))
        })?,
        url: field_str("url")?,
        extension: field_str("extension")?,
        sha256: field_str("sha256")?,
        blob: field_str("blob")?,
    };
    if metadata.version != METADATA_VERSION {
        return Err(Error::Eval(format!(
            "unsupported package cache metadata version {} at {} (supported: {})",
            metadata.version,
            path.display(),
            METADATA_VERSION
        )));
    }
    if metadata.url.is_empty() || metadata.extension.is_empty() || metadata.sha256.is_empty() {
        return Err(Error::Eval(format!(
            "corrupt package cache metadata at {}: a required field is empty",
            path.display()
        )));
    }
    Ok(Some(metadata))
}

async fn read_legacy(
    caps: &mut dyn EvalCapabilities,
    cache_dir: &Path,
    url: &str,
    extension: &str,
) -> Result<Option<Vec<u8>>> {
    let (data_path, url_path) = legacy_paths(cache_dir, url, extension);
    let cached_url = match caps.read_to_string(&url_path).await {
        Ok(cached_url) => cached_url,
        Err(Error::Io(_, error)) if error.kind() == std::io::ErrorKind::NotFound => {
            return Ok(None)
        }
        Err(error) => return Err(error),
    };
    if cached_url != url {
        return Ok(None);
    }
    match caps.read_bytes(&data_path).await {
        Ok(bytes) => Ok(Some(bytes)),
        Err(Error::Io(_, error)) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(error),
    }
}

/// Load and fully validate a cached package.
///
/// v1 metadata wins; otherwise the pre-v1 sidecar layout is accepted. Every
/// error message distinguishes corruption kinds so callers (and tests) can
/// assert the rejection reason.
pub async fn read_entry(
    caps: &mut dyn EvalCapabilities,
    cache_dir: &Path,
    url: &str,
    extension: &str,
) -> Result<Option<Vec<u8>>> {
    let meta_file = meta_path(cache_dir, url);
    if let Some(metadata) = read_metadata(caps, &meta_file).await? {
        if metadata.url != url {
            return Err(Error::Eval(format!(
                "corrupt package cache metadata at {}: URL mismatch",
                meta_file.display()
            )));
        }
        if metadata.extension != extension {
            return Err(Error::Eval(format!(
                "corrupt package cache metadata at {}: extension mismatch",
                meta_file.display()
            )));
        }
        let blob_path = resolve_within_root(cache_dir, &metadata.blob)?;
        let bytes = match caps.read_bytes(&blob_path).await {
            Ok(bytes) => bytes,
            Err(Error::Io(_, error)) if error.kind() == std::io::ErrorKind::NotFound => {
                return Ok(None)
            }
            Err(error) => return Err(error),
        };
        let actual = sha256_hex(&bytes);
        if actual != metadata.sha256 {
            return Err(Error::Eval(format!(
                "package cache digest mismatch for {url}: expected {}, got {actual}",
                metadata.sha256
            )));
        }
        return Ok(Some(bytes));
    }
    read_legacy(caps, cache_dir, url, extension).await
}

/// Remove an entry's metadata and legacy sidecars. Content-addressed blobs are
/// shared and left in place.
pub async fn remove_entry(
    caps: &mut dyn EvalCapabilities,
    cache_dir: &Path,
    url: &str,
    extension: &str,
) {
    let _ = caps.remove_file(&meta_path(cache_dir, url)).await;
    let (data_path, url_path) = legacy_paths(cache_dir, url, extension);
    let _ = caps.remove_file(&data_path).await;
    let _ = caps.remove_file(&url_path).await;
}

/// Stage and commit one cache file. Every boundary is a separate capability
/// call, so a host can fail the write, the flush, the rename, or the sync.
async fn commit_file(
    caps: &mut dyn EvalCapabilities,
    kind: FileKind,
    target: &Path,
    bytes: &[u8],
) -> Result<()> {
    let Some(parent) = target.parent() else {
        return Err(Error::Eval(format!(
            "cache target has no parent directory: {}",
            target.display()
        )));
    };
    caps.cache_mkdir(parent).await?;
    let temp_path = target.with_file_name(format!(
        ".{}.tmp-{}",
        target
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("cache"),
        TEMP_SUFFIX
    ));
    caps.cache_create_file(kind, &temp_path, bytes).await?;
    caps.cache_flush_file(kind, &temp_path).await?;
    caps.cache_rename(kind, &temp_path, target).await?;
    caps.cache_sync_dir(kind, parent).await
}

/// Persist package bytes as a digest-addressed blob plus v1 metadata.
pub async fn write_entry(
    caps: &mut dyn EvalCapabilities,
    cache_dir: &Path,
    url: &str,
    extension: &str,
    bytes: &[u8],
) -> Result<()> {
    let digest = sha256_hex(bytes);
    let blob_rel = format!("blobs/{digest}.{extension}");
    let blob_path = resolve_within_root(cache_dir, &blob_rel)?;
    if !caps.path_exists(&blob_path).await? {
        Box::pin(commit_file(caps, FileKind::Blob, &blob_path, bytes)).await?;
    }
    let metadata = Metadata {
        version: METADATA_VERSION,
        url: url.to_string(),
        extension: extension.to_string(),
        sha256: digest,
        blob: blob_rel,
    };
    let meta_json = json!({
        "version": metadata.version,
        "url": metadata.url,
        "extension": metadata.extension,
        "sha256": metadata.sha256,
        "blob": metadata.blob,
    });
    let meta_bytes = serde_json::to_vec(&meta_json)
        .map_err(|error| Error::Eval(format!("failed to encode cache metadata: {error}")))?;
    commit_file(caps, FileKind::Metadata, &meta_path(cache_dir, url), &meta_bytes).await
}

/// Delete leftover `*.pklrtmp` staging files from the cache directories.
pub async fn sweep_temp_files(caps: &mut dyn EvalCapabilities, cache_dir: &Path) {
    for dir in [cache_dir.join("packages"), cache_dir.join("blobs")] {
        let paths = match caps.list_dir(&dir).await {
            Ok(paths) => paths,
            Err(_) => continue,
        };
        for path in paths {
            let is_temp = path
                .file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.starts_with('.') && name.ends_with(TEMP_SUFFIX));
            if is_temp {
                let _ = caps.remove_file(&path).await;
            }
        }
    }
}
