//! Persistent package cache layout and filesystem staging primitives.
//!
//! Every cached package is represented by two files under
//! `<cache_dir>/packages/<fnv1a64(url)>/`:
//!
//! - `metadata.json`: versioned metadata (format version, source URL, and the
//!   hex digest of the blob). Metadata is written only after the blob commit
//!   succeeds, so a missing or unreadable metadata file can never turn a
//!   half-written blob into a cache hit.
//! - `<sha256>.{pkl,zip}`: the content-addressed package bytes. Content
//!   addressing means a later, different version never overwrites a blob that
//!   existing metadata still points at, so a write fault at any boundary
//!   leaves the previously usable cache entry intact.

use std::path::{Component, Path, PathBuf};

use crate::Result;

/// Metadata format version understood by this build. Older or newer versions
/// are rejected instead of being interpreted with the wrong layout.
pub const METADATA_VERSION: u32 = 1;

pub const METADATA_FILE_NAME: &str = "metadata.json";

/// Prefix of staging files left behind when a process dies mid-commit.
pub const STAGING_PREFIX: &str = ".staging-";

/// Prefix used for the brief backup file created when replacing an existing
/// destination on platforms where rename does not atomically replace.
pub const BACKUP_PREFIX: &str = ".backup-";

#[derive(serde::Serialize, serde::Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct PackageMetadata {
    pub version: u32,
    pub url: String,
    pub extension: String,
    /// Hex-encoded SHA-256 digest of the blob bytes.
    pub digest: String,
}

/// FNV-1a 64-bit hash of a package URL. Gives package URLs stable,
/// filesystem-safe directory names without depending on platform path rules.
pub fn fnv1a64(input: &str) -> u64 {
    let mut hash = 0xcbf29ce484222325_u64;
    for byte in input.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

/// Return the per-package directory for `url`, and the metadata path within it.
pub fn package_dir(cache_dir: &Path, url: &str) -> PathBuf {
    cache_dir
        .join("packages")
        .join(format!("{:016x}", fnv1a64(url)))
}

pub fn metadata_path(cache_dir: &Path, url: &str) -> PathBuf {
    package_dir(cache_dir, url).join(METADATA_FILE_NAME)
}

/// Content-addressed blob path for a verified digest.
pub fn blob_path(cache_dir: &Path, url: &str, extension: &str, digest_hex: &str) -> PathBuf {
    package_dir(cache_dir, url).join(format!("{digest_hex}.{extension}"))
}

pub fn encode_metadata(metadata: &PackageMetadata) -> Result<Vec<u8>> {
    serde_json::to_vec_pretty(metadata)
        .map_err(|error| crate::Error::Eval(format!("failed to encode cache metadata: {error}")))
}

pub fn decode_metadata(bytes: &[u8]) -> Result<PackageMetadata> {
    let metadata: PackageMetadata = serde_json::from_slice(bytes).map_err(|error| {
        crate::Error::Eval(format!("corrupt package cache metadata: {error}"))
    })?;
    if metadata.version != METADATA_VERSION {
        return Err(crate::Error::Eval(format!(
            "unsupported package cache metadata version: {} (supported: {METADATA_VERSION})",
            metadata.version
        )));
    }
    if !is_safe_digest_hex(&metadata.digest) || !is_safe_extension(&metadata.extension) {
        return Err(crate::Error::Eval(
            "corrupt package cache metadata: invalid digest or extension".to_string(),
        ));
    }
    Ok(metadata)
}

fn is_safe_digest_hex(digest: &str) -> bool {
    digest.len() == 64 && digest.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn is_safe_extension(extension: &str) -> bool {
    !extension.is_empty()
        && extension
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || byte == b'-')
}

/// Reject a path that escapes `root` after lexical normalization. `root` and
/// `path` must both be absolute or both relative in comparable ways; callers
/// normalize with [`normalize_within`] first.
pub fn ensure_within(root: &Path, path: &Path) -> Result<()> {
    let root = normalize_components(root);
    let path = normalize_components(path);
    if path.strip_prefix(&root).is_ok() {
        Ok(())
    } else {
        Err(crate::Error::Eval(format!(
            "refusing to access a cache path outside {}: {}",
            root.display(),
            path.display()
        )))
    }
}

fn normalize_components(path: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for component in path.components() {
        match component {
            Component::ParentDir => {
                out.pop();
            }
            Component::CurDir => {}
            other => out.push(other.as_os_str()),
        }
    }
    out
}

/// SHA-256 implementation (FIPS 180-4). Kept local so the cache format does
/// not pull in a hashing dependency for embedders.
pub fn sha256_hex(input: &[u8]) -> String {
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

    let mut hash: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ];

    let bit_len = u64::try_from(input.len()).unwrap() * 8;
    let mut padded = input.to_vec();
    padded.push(0x80);
    while padded.len() % 64 != 56 {
        padded.push(0);
    }
    padded.extend_from_slice(&bit_len.to_be_bytes());

    for chunk in padded.chunks_exact(64) {
        let mut words = [0u32; 64];
        for (index, bytes) in chunk.chunks_exact(4).enumerate() {
            words[index] = u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        }
        for index in 16..64 {
            let s0 = words[index - 15].rotate_right(7) ^ words[index - 15].rotate_right(18)
                ^ (words[index - 15] >> 3);
            let s1 = words[index - 2].rotate_right(17) ^ words[index - 2].rotate_right(19)
                ^ (words[index - 2] >> 10);
            words[index] = words[index - 16]
                .wrapping_add(s0)
                .wrapping_add(words[index - 7])
                .wrapping_add(s1);
        }

        let mut a = hash[0];
        let mut b = hash[1];
        let mut c = hash[2];
        let mut d = hash[3];
        let mut e = hash[4];
        let mut f = hash[5];
        let mut g = hash[6];
        let mut h = hash[7];

        for index in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let temp1 = h
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[index])
                .wrapping_add(words[index]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let temp2 = s0.wrapping_add(maj);
            h = g;
            g = f;
            f = e;
            e = d.wrapping_add(temp1);
            d = c;
            c = b;
            b = a;
            a = temp1.wrapping_add(temp2);
        }

        hash[0] = hash[0].wrapping_add(a);
        hash[1] = hash[1].wrapping_add(b);
        hash[2] = hash[2].wrapping_add(c);
        hash[3] = hash[3].wrapping_add(d);
        hash[4] = hash[4].wrapping_add(e);
        hash[5] = hash[5].wrapping_add(f);
        hash[6] = hash[6].wrapping_add(g);
        hash[7] = hash[7].wrapping_add(h);
    }

    let mut hex = String::with_capacity(64);
    for word in hash {
        use std::fmt::Write;
        let _ = write!(hex, "{word:08x}");
    }
    hex
}

/// A directory entry surfaced by [`crate::EvalCapabilities::list_dir`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DirEntry {
    pub file_name: String,
    pub is_dir: bool,
}

use std::sync::atomic::{AtomicU64, Ordering};

static STAGING_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Build the unique staging path for `destination`.
pub fn staging_path_for(destination: &Path) -> PathBuf {
    let file_name = destination
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("cache");
    let counter = STAGING_COUNTER.fetch_add(1, Ordering::Relaxed);
    destination.with_file_name(format!(
        "{STAGING_PREFIX}{file_name}-{}-{counter}",
        std::process::id()
    ))
}

/// Whether `file_name` is a staging or backup file created by this module.
pub fn is_recoverable_file_name(file_name: &str) -> bool {
    file_name.starts_with(STAGING_PREFIX) || file_name.starts_with(BACKUP_PREFIX)
}

/// Filesystem boundary for the "write bytes to a unique staging file" phase.
/// Returns the staging path on success so callers can commit it.
pub fn fs_write_staging(destination: &Path, bytes: &[u8]) -> std::io::Result<PathBuf> {
    use std::io::Write;

    let staging = staging_path_for(destination);
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&staging)?;
    let result = file.write_all(bytes);
    drop(file);
    if let Err(error) = result {
        let _ = std::fs::remove_file(&staging);
        return Err(error);
    }
    Ok(staging)
}

/// Filesystem boundary for flushing staged bytes to stable storage.
pub fn fs_flush(path: &Path) -> std::io::Result<()> {
    let file = std::fs::OpenOptions::new().write(true).open(path)?;
    file.sync_all()
}

/// fsync the containing directory after a rename so the commit is durable.
/// Best effort on platforms without directory fsync support.
pub fn fs_sync_parent_dir(path: &Path) -> std::io::Result<()> {
    let Some(parent) = path.parent() else {
        return Ok(());
    };
    match std::fs::File::open(parent) {
        Ok(directory) => match directory.sync_all() {
            Ok(()) => Ok(()),
            // Directory fsync is unsupported on some platforms; the write
            // itself already succeeded.
            Err(error) if error.kind() == std::io::ErrorKind::InvalidInput => Ok(()),
            Err(error) => Err(error),
        },
        // Nothing to sync when the parent directory does not exist yet.
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error),
    }
}

/// Commit a staging file into place. Replaces an existing destination, moving
/// it to a backup first so a failed replacement leaves the old file in place
/// whenever the platform allows recovery.
pub fn fs_commit(staging: &Path, destination: &Path) -> std::io::Result<()> {
    match std::fs::rename(staging, destination) {
        Ok(()) => Ok(()),
        Err(_) if destination.exists() => {
            let backup = destination.with_file_name(format!(
                "{BACKUP_PREFIX}{}",
                destination
                    .file_name()
                    .and_then(|name| name.to_str())
                    .unwrap_or("cache")
            ));
            let _ = std::fs::remove_file(&backup);
            std::fs::rename(destination, &backup)?;
            match std::fs::rename(staging, destination) {
                Ok(()) => {
                    let _ = std::fs::remove_file(&backup);
                    Ok(())
                }
                Err(error) => {
                    let _ = std::fs::rename(&backup, destination);
                    Err(error)
                }
            }
        }
        Err(error) => Err(error),
    }
}

/// One-shot atomic write expressed through the staging/flush/rename/sync
/// boundaries. Used by the default filesystem capabilities.
pub fn fs_write_atomic(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    let staging = fs_write_staging(path, bytes)?;
    fs_flush(&staging)?;
    fs_commit(&staging, path)?;
    fs_sync_parent_dir(path)
}


/// Validate that a zip entry name stays inside the extraction directory.
/// Mirrors the sanitization applied to package URI fragments and blocks the
/// classic "zip slip" traversal as well as absolute and Windows-style paths.
pub fn ensure_zip_entry_safe(name: &str) -> Result<()> {
    if name.is_empty()
        || name.starts_with('/')
        || name.contains('\\')
        || name.split('/').any(|part| part == "..")
    {
        return Err(crate::Error::Eval(format!(
            "refusing to extract package entry outside the cache directory: {name}"
        )));
    }
    let path = Path::new(name);
    if path.components().any(|component| {
        matches!(
            component,
            Component::ParentDir | Component::RootDir | Component::Prefix(_)
        )
    }) {
        return Err(crate::Error::Eval(format!(
            "refusing to extract package entry outside the cache directory: {name}"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

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
    }

    #[test]
    fn metadata_round_trips_and_rejects_unknown_versions() {
        let metadata = PackageMetadata {
            version: METADATA_VERSION,
            url: "https://example.com/p@1.pkl".to_string(),
            extension: "pkl".to_string(),
            digest: sha256_hex(b"answer = 1\n"),
        };
        let bytes = encode_metadata(&metadata).unwrap();
        assert_eq!(decode_metadata(&bytes).unwrap(), metadata);

        let mut other = metadata.clone();
        other.version = METADATA_VERSION + 1;
        let bytes = encode_metadata(&other).unwrap();
        assert!(decode_metadata(&bytes)
            .unwrap_err()
            .to_string()
            .contains("unsupported package cache metadata version"));

        let mut other = metadata.clone();
        other.digest = "not-a-digest".to_string();
        let bytes = encode_metadata(&other).unwrap();
        assert!(decode_metadata(&bytes)
            .unwrap_err()
            .to_string()
            .contains("corrupt package cache metadata"));
    }

    #[test]
    fn corrupt_metadata_json_is_rejected() {
        let error = decode_metadata(b"{ not json").unwrap_err().to_string();
        assert!(error.contains("corrupt package cache metadata"), "{error}");
    }

    #[test]
    fn traversal_paths_are_rejected() {
        let root = Path::new("/tmp/cache-root/packages/abc");
        ensure_within(root, &root.join("metadata.json")).unwrap();
        let error = ensure_within(root, &root.join("..").join("secret")).unwrap_err();
        assert!(error.to_string().contains("outside"));
        ensure_zip_entry_safe("Config.pkl").unwrap();
        ensure_zip_entry_safe("nested/Config.pkl").unwrap();
        assert!(ensure_zip_entry_safe("../escape.pkl").is_err());
        assert!(ensure_zip_entry_safe("/abs.pkl").is_err());
        assert!(ensure_zip_entry_safe("nested/../escape.pkl").is_err());
    }
}
