//! Persistent package cache: content-addressed blobs plus sidecar metadata.
//!
//! Layout:
//!
//! ```text
//! <cache_dir>/packages/<fnv1a64(url)>/entry.json
//! <cache_dir>/packages/<fnv1a64(url)>/blobs/<sha256>.bin
//! ```
//!
//! Blobs are written into a same-directory staging file (`*.pklrtmp`) and
//! atomically renamed into place; metadata is published afterwards with the
//! same two-phase commit. Every write boundary goes through
//! [`EvalCapabilities`] so hosts (and tests) can inject failures at each
//! boundary.

use std::path::{Component, Path, PathBuf};

use serde_json::json;

use crate::capabilities::EvalCapabilities;
use crate::{Error, Result};

/// Metadata format version accepted by this build.
pub const CACHE_METADATA_VERSION: u32 = 1;
/// Suffix shared by every staging file the cache creates.
pub const STAGING_SUFFIX: &str = "pklrtmp";

/// Deterministic cache key for `url`, matching the historical FNV-1a hash.
pub fn key_for_url(url: &str) -> String {
    let mut hash = 0xcbf29ce484222325_u64;
    for byte in url.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    format!("{hash:016x}")
}

fn entry_dir(cache_dir: &Path, url: &str) -> PathBuf {
    cache_dir.join("packages").join(key_for_url(url))
}

fn metadata_path(cache_dir: &Path, url: &str) -> PathBuf {
    entry_dir(cache_dir, url).join("entry.json")
}

fn blob_path(cache_dir: &Path, url: &str, digest: &str) -> Option<PathBuf> {
    let name = safe_blob_name(digest)?;
    Some(entry_dir(cache_dir, url).join("blobs").join(name))
}

/// Only plain `<64 hex>.bin` names are ever read back, so a tampered digest
/// cannot address a file outside the entry directory.
fn safe_blob_name(digest: &str) -> Option<PathBuf> {
    if digest.len() != 64 || !digest.bytes().all(|b| b.is_ascii_hexdigit()) {
        return None;
    }
    let path = Path::new(digest);
    if path.components().count() != 1 {
        return None;
    }
    if matches!(path.components().next(), Some(Component::Normal(_))) {
        Some(PathBuf::from(format!("{digest}.bin")))
    } else {
        None
    }
}

/// SHA-256 without a third-party dependency (FIPS 180-4 reference algorithm).
pub fn sha256_hex(input: &[u8]) -> String {
    const K: [u32; 64] = [
        0x428a_2f98, 0x7137_4491, 0xb5c0_fbcf, 0xe9b5_dba5, 0x3956_c25b, 0x59f1_11f1, 0x923f_82a4,
        0xab1c_5ed5, 0xd807_aa98, 0x1283_5b01, 0x2431_85be, 0x550c_7dc3, 0x72be_5d74, 0x80de_b1fe,
        0x9bdc_06a7, 0xc19b_f174, 0xe49b_69c1, 0xefbe_4786, 0x0fc1_9dc6, 0x240c_a1cc, 0x2de9_2c6f,
        0x4a74_84aa, 0x5cb0_a9dc, 0x76f9_88da, 0x983e_5152, 0xa831_c66d, 0xb003_27c8, 0xbf59_7fc7,
        0xc6e0_0bf3, 0xd5a7_9147, 0x06ca_6351, 0x1429_2967, 0x27b7_0a85, 0x2e1b_2138, 0x4d2c_6dfc,
        0x5338_0d13, 0x650a_7354, 0x766a_0abb, 0x81c2_c92e, 0x9272_2c85, 0xa2bf_e8a1, 0xa81a_664b,
        0xc24b_8b70, 0xc76c_51a3, 0xd192_e819, 0xd699_0624, 0xf40e_3585, 0x106a_a070, 0x19a4_c116,
        0x1e37_6c08, 0x2748_774c, 0x34b0_bcb5, 0x391c_0cb3, 0x4ed8_aa4a, 0x5b9c_ca4f, 0x682e_6ff3,
        0x748f_82ee, 0x78a5_636f, 0x84c8_7814, 0x8cc7_0208, 0x90be_fffa, 0xa450_6ceb, 0xbef9_a3f7,
        0xc671_78f2,
    ];

    let mut hash = [
        0x6a09_e667_u32,
        0xbb67_ae85,
        0x3c6e_f372,
        0xa54f_f53a,
        0x510e_527f,
        0x9b05_688c,
        0x1f83_d9ab,
        0x5be0_cd19,
    ];

    let bit_len = u64::try_from(input.len()).unwrap_or(0) * 8;
    let mut padded = input.to_vec();
    padded.push(0x80);
    while padded.len() % 64 != 56 {
        padded.push(0);
    }
    padded.extend_from_slice(&bit_len.to_be_bytes());

    for chunk in padded.chunks_exact(64) {
        let mut words = [0_u32; 64];
        for (i, word) in chunk.chunks_exact(4).enumerate() {
            words[i] = u32::from_be_bytes([word[0], word[1], word[2], word[3]]);
        }
        for i in 16..64 {
            let s0 = words[i - 15].rotate_right(7)
                ^ words[i - 15].rotate_right(18)
                ^ (words[i - 15] >> 3);
            let s1 = words[i - 2].rotate_right(17)
                ^ words[i - 2].rotate_right(19)
                ^ (words[i - 2] >> 10);
            words[i] = words[i - 16]
                .wrapping_add(s0)
                .wrapping_add(words[i - 7])
                .wrapping_add(s1);
        }

        let mut work = hash;
        for (i, &word) in words.iter().enumerate() {
            let big_s1 =
                work[4].rotate_right(6) ^ work[4].rotate_right(11) ^ work[4].rotate_right(25);
            let ch = (work[4] & work[5]) ^ ((!work[4]) & work[6]);
            let temp1 = work[7]
                .wrapping_add(big_s1)
                .wrapping_add(ch)
                .wrapping_add(K[i])
                .wrapping_add(word);
            let big_s0 =
                work[0].rotate_right(2) ^ work[0].rotate_right(13) ^ work[0].rotate_right(22);
            let maj = (work[0] & work[1]) ^ (work[0] & work[2]) ^ (work[1] & work[2]);
            let temp2 = big_s0.wrapping_add(maj);
            work[7] = work[6];
            work[6] = work[5];
            work[5] = work[4];
            work[4] = work[3].wrapping_add(temp1);
            work[3] = work[2];
            work[2] = work[1];
            work[1] = work[0];
            work[0] = temp1.wrapping_add(temp2);
        }

        for (slot, value) in hash.iter_mut().zip(work) {
            *slot = slot.wrapping_add(value);
        }
    }

    let mut out = String::with_capacity(64);
    for word in hash {
        out.push_str(&format!("{word:08x}"));
    }
    out
}

fn corrupted(message: impl Into<String>) -> Error {
    Error::Eval(format!("corrupted package cache: {}", message.into()))
}

/// Load and validate the cached bytes for `url`.
///
/// * A missing entry returns `Ok(None)`.
/// * Malformed metadata, an unsupported version, a digest mismatch, and a
///   digest pointing outside the entry directory are distinct errors.
pub async fn read_entry(
    caps: &mut dyn EvalCapabilities,
    cache_dir: &Path,
    url: &str,
    extension: &str,
) -> Result<Option<Vec<u8>>> {
    let meta_path = metadata_path(cache_dir, url);
    let raw_meta = match caps.read_to_string(&meta_path).await {
        Ok(raw) => raw,
        Err(Error::Io(_, error)) if error.kind() == std::io::ErrorKind::NotFound => {
            return Ok(None);
        }
        Err(error) => return Err(error),
    };
    let meta: serde_json::Value = serde_json::from_str(&raw_meta)
        .map_err(|error| corrupted(format!("metadata for {url} is not valid JSON: {error}")))?;

    let version = meta
        .get("version")
        .and_then(serde_json::Value::as_u64)
        .ok_or_else(|| corrupted(format!("metadata for {url} is missing a numeric version")))?;
    if version != u64::from(CACHE_METADATA_VERSION) {
        return Err(corrupted(format!(
            "metadata for {url} uses unsupported cache version {version}"
        )));
    }
    if meta.get("url").and_then(serde_json::Value::as_str) != Some(url) {
        return Err(corrupted(format!(
            "metadata for {url} has a mismatched URL"
        )));
    }
    if meta.get("extension").and_then(serde_json::Value::as_str) != Some(extension) {
        return Err(corrupted(format!(
            "metadata for {url} has a mismatched extension"
        )));
    }
    let digest = meta
        .get("sha256")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| corrupted(format!("metadata for {url} is missing a sha256 digest")))?;
    let blob_path = blob_path(cache_dir, url, digest)
        .ok_or_else(|| corrupted(format!("metadata for {url} names a blob outside the cache")))?;
    let bytes = match caps.read_bytes(&blob_path).await {
        Ok(bytes) => bytes,
        Err(Error::Io(_, error)) if error.kind() == std::io::ErrorKind::NotFound => {
            return Err(corrupted(format!(
                "metadata for {url} points at a missing blob"
            )));
        }
        Err(error) => return Err(error),
    };
    if sha256_hex(&bytes) != digest {
        return Err(corrupted(format!("blob checksum mismatch for {url}")));
    }
    Ok(Some(bytes))
}

/// Persist `bytes` for `url`.
///
/// Blobs are content-addressed, so a failed or superseded publish never
/// destroys an older valid entry.
pub async fn write_entry(
    caps: &mut dyn EvalCapabilities,
    cache_dir: &Path,
    url: &str,
    extension: &str,
    bytes: &[u8],
) -> Result<()> {
    let dir = entry_dir(cache_dir, url);
    let blob_dir = dir.join("blobs");
    caps.create_dir_all(&blob_dir).await?;

    let digest = sha256_hex(bytes);
    let blob_path = blob_path(cache_dir, url, &digest)
        .ok_or_else(|| corrupted("computed blob digest is not a safe file name"))?;

    if !caps.path_exists(&blob_path).await? {
        let staging = blob_dir.join(format!(".{digest}.bin.{STAGING_SUFFIX}"));
        caps.cache_write_temp(&staging, bytes).await?;
        caps.cache_flush_temp(&staging).await?;
        caps.cache_rename(&staging, &blob_path).await?;
        caps.cache_dir_sync(&blob_dir).await?;
    }

    let metadata = json!({
        "version": CACHE_METADATA_VERSION,
        "url": url,
        "extension": extension,
        "sha256": digest,
    })
    .to_string();
    let meta_path = metadata_path(cache_dir, url);
    let meta_staging = dir.join(format!(".entry.json.{STAGING_SUFFIX}"));
    caps.cache_write_temp(&meta_staging, metadata.as_bytes())
        .await?;
    caps.cache_flush_temp(&meta_staging).await?;
    caps.cache_rename(&meta_staging, &meta_path).await?;
    caps.cache_dir_sync(&dir).await?;
    Ok(())
}

/// Remove a cache entry (best effort).
pub async fn remove_entry(caps: &mut dyn EvalCapabilities, cache_dir: &Path, url: &str) {
    let dir = entry_dir(cache_dir, url);
    let _ = caps.remove_file(&dir.join("entry.json")).await;
}

/// Delete staging files left behind by a crashed previous process.
///
/// Files created with [`EvalCapabilities::cache_write_temp`] always end in
/// [`STAGING_SUFFIX`]; committed files never do, so sweeping can only remove
/// half-written data.
pub async fn sweep_staging(caps: &mut dyn EvalCapabilities, cache_dir: &Path) {
    let packages = cache_dir.join("packages");
    let mut pending = vec![packages];
    while let Some(dir) = pending.pop() {
        let entries = match caps.cache_list_dir(&dir).await {
            Ok(entries) => entries,
            Err(_) => continue,
        };
        for entry in entries {
            let name = entry
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or_default()
                .to_string();
            if name.ends_with(STAGING_SUFFIX) {
                let _ = caps.remove_file(&entry).await;
            } else if !entry.extension().is_some_and(|ext| ext == "bin" || ext == "json") {
                // Directories (entry dirs and their `blobs` subdirs) are
                // recursed into; committed files are left alone.
                pending.push(entry);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{key_for_url, sha256_hex};

    #[test]
    fn sha256_matches_known_vectors() {
        assert_eq!(
            sha256_hex(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        let long = vec![b'a'; 1_000_000];
        assert_eq!(
            sha256_hex(&long),
            "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0"
        );
    }

    #[test]
    fn url_keys_are_stable() {
        assert_eq!(key_for_url("https://example.com/a.pkl"), key_for_url("https://example.com/a.pkl"));
        assert_ne!(key_for_url("https://example.com/a.pkl"), key_for_url("https://example.com/b.pkl"));
    }
}
