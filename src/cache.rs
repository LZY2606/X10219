//! Persistent package-cache primitives shared by every IO backend.
//!
//! A cache entry is two adjacent files inside `<cache_dir>/packages`:
//! `<hash>.<ext>` holds the blob and `<hash>.url` holds a JSON envelope with
//! the originating URL, format version and blob digest.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use serde_json::json;

/// Current cache envelope format. Older or newer formats are rejected.
pub const CACHE_FORMAT_VERSION: u32 = 1;

/// The envelope stored next to every cached blob.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CacheMetadata {
    pub url: String,
    pub extension: String,
    pub version: u32,
    pub sha256: String,
}

impl CacheMetadata {
    pub fn new(url: impl Into<String>, extension: impl Into<String>, bytes: &[u8]) -> Self {
        Self {
            url: url.into(),
            extension: extension.into(),
            version: CACHE_FORMAT_VERSION,
            sha256: sha256_hex(bytes),
        }
    }

    pub fn to_json(&self) -> String {
        json!({
            "version": self.version,
            "url": self.url,
            "extension": self.extension,
            "sha256": self.sha256,
        })
        .to_string()
    }

    /// Parse and structurally validate an envelope.
    ///
    /// Distinct error kinds keep "corrupt metadata", "unsupported version",
    /// "digest mismatch", and "escaped path" failures independently testable.
    pub fn parse(raw: &str) -> Result<Self, CacheError> {
        let value: serde_json::Value = serde_json::from_str(raw)
            .map_err(|error| CacheError::CorruptMetadata(error.to_string()))?;
        let object = value.as_object().ok_or_else(|| {
            CacheError::CorruptMetadata("envelope is not a JSON object".to_string())
        })?;
        let version = object
            .get("version")
            .and_then(|value| value.as_u64())
            .ok_or_else(|| CacheError::CorruptMetadata("missing numeric version".to_string()))?
            as u32;
        if version != CACHE_FORMAT_VERSION {
            return Err(CacheError::UnsupportedVersion(version));
        }
        let url = object
            .get("url")
            .and_then(|value| value.as_str())
            .ok_or_else(|| CacheError::CorruptMetadata("missing url".to_string()))?
            .to_string();
        let extension = object
            .get("extension")
            .and_then(|value| value.as_str())
            .ok_or_else(|| CacheError::CorruptMetadata("missing extension".to_string()))?
            .to_string();
        let sha256 = object
            .get("sha256")
            .and_then(|value| value.as_str())
            .ok_or_else(|| CacheError::CorruptMetadata("missing sha256".to_string()))?
            .to_string();
        if sha256.len() != 64 || !sha256.bytes().all(|byte| byte.is_ascii_hexdigit()) {
            return Err(CacheError::CorruptMetadata(
                "sha256 is not 64 hex digits".to_string(),
            ));
        }
        Ok(Self {
            url,
            extension,
            version,
            sha256,
        })
    }

    /// Ensure the envelope matches the lookup and that `bytes` has the digest
    /// recorded when the entry was written.
    pub fn verify(&self, url: &str, extension: &str, bytes: &[u8]) -> Result<(), CacheError> {
        if self.url != url {
            return Err(CacheError::UrlMismatch {
                expected: url.to_string(),
                recorded: self.url.clone(),
            });
        }
        if self.extension != extension {
            return Err(CacheError::ExtensionMismatch {
                expected: extension.to_string(),
                recorded: self.extension.clone(),
            });
        }
        if self.sha256 != sha256_hex(bytes) {
            return Err(CacheError::DigestMismatch {
                url: url.to_string(),
            });
        }
        Ok(())
    }
}

/// Errors produced while validating persistent cache entries.
#[derive(Debug, thiserror::Error)]
pub enum CacheError {
    #[error("corrupt package cache metadata: {0}")]
    CorruptMetadata(String),
    #[error("unsupported package cache format version: {0}")]
    UnsupportedVersion(u32),
    #[error("cached blob digest mismatch for {url}")]
    DigestMismatch { url: String },
    #[error("cached entry was written for {recorded}, not {expected}")]
    UrlMismatch {
        expected: String,
        recorded: String,
    },
    #[error("cached entry extension was {recorded}, not {expected}")]
    ExtensionMismatch {
        expected: String,
        recorded: String,
    },
    #[error("cache path escapes the cache directory: {}", .0.display())]
    PathEscape(PathBuf),
}

/// FNV-1a hash used to turn package URLs into filesystem-safe names.
pub fn url_hash(url: &str) -> u64 {
    let mut hash = 0xcbf29ce484222325_u64;
    for byte in url.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

/// Return `(blob_path, metadata_path)` for an entry and verify both stay
/// inside `cache_dir`. Hash-derived names cannot escape, but the guard is
/// explicit so host-supplied cache roots are never trusted blindly.
pub fn entry_paths(
    cache_dir: &Path,
    url: &str,
    extension: &str,
) -> Result<(PathBuf, PathBuf), CacheError> {
    let root = cache_dir.join("packages");
    let base = root.join(format!("{:016x}", url_hash(url)));
    let blob = base.with_extension(extension);
    let metadata = base.with_extension("url");
    ensure_within(&root, &blob)?;
    ensure_within(&root, &metadata)?;
    Ok((blob, metadata))
}

/// Reject any path that is not strictly contained in `root`.
pub fn ensure_within(root: &Path, candidate: &Path) -> Result<(), CacheError> {
    if candidate.components().count() <= root.components().count()
        || !candidate.starts_with(root)
        || candidate
            .components()
            .any(|component| matches!(component, std::path::Component::ParentDir))
    {
        return Err(CacheError::PathEscape(candidate.to_path_buf()));
    }
    Ok(())
}

/// Whether `name` is an abandoned atomic-write temp file. Matches the naming
/// used by [`temp_file_name`] so crash leftovers can be swept on startup.
pub fn is_temp_file_name(name: &str) -> bool {
    // Format: `.<target>.tmp-<pid>-<counter>`, where pid and counter are
    // decimal. Leftovers from previous processes must also be recognized.
    let Some(rest) = name.strip_prefix('.') else {
        return false;
    };
    let Some((_, suffix)) = rest.split_once(".tmp-") else {
        return false;
    };
    let mut parts = suffix.split('-');
    matches!(parts.next(), Some(pid) if !pid.is_empty() && pid.bytes().all(|b| b.is_ascii_digit()))
        && matches!(parts.next(), Some(counter) if !counter.is_empty()
            && counter.bytes().all(|b| b.is_ascii_digit()))
        && parts.next().is_none()
}

static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Temp-file naming shared by native IO and sweeps:
/// `.<target>.tmp-<pid>-<counter>`.
pub fn temp_file_name(target_name: &str) -> String {
    format!(
        ".{target_name}.tmp-{}-{}",
        std::process::id(),
        TEMP_COUNTER.fetch_add(1, Ordering::Relaxed)
    )
}

/// Native-filesystem atomic write using the shared temp naming convention.
///
/// Failures at create, flush/sync, and rename boundaries all leave only temp
/// files behind, never a half-written destination.
pub fn write_atomic_native(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    use std::io::Write;

    let Some(parent) = path.parent() else {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "atomic write target has no parent directory",
        ));
    };
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("cache");
    let temp_path = path.with_file_name(temp_file_name(file_name));
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temp_path)?;
    let write_result = file.write_all(bytes).and_then(|()| file.sync_all());
    drop(file);
    if let Err(error) = write_result {
        let _ = std::fs::remove_file(&temp_path);
        return Err(error);
    }
    let rename_result = match std::fs::rename(&temp_path, path) {
        Ok(()) => return sync_dir_native(parent),
        // Windows rename does not replace an existing file.
        Err(_) if path.exists() => {
            std::fs::remove_file(path).and_then(|()| std::fs::rename(&temp_path, path))
        }
        Err(error) => Err(error),
    };
    match rename_result {
        Ok(()) => sync_dir_native(parent),
        Err(error) => {
            let _ = std::fs::remove_file(&temp_path);
            Err(error)
        }
    }
}

#[cfg(unix)]
fn sync_dir_native(dir: &Path) -> std::io::Result<()> {
    use std::os::unix::io::AsRawFd;
    let file = std::fs::File::open(dir)?;
    // SAFETY: the descriptor is owned by `file` and stays open for the call.
    let code = unsafe { fsync(file.as_raw_fd()) };
    if code == 0 {
        Ok(())
    } else {
        Err(std::io::Error::last_os_error())
    }
}

#[cfg(not(unix))]
fn sync_dir_native(_dir: &Path) -> std::io::Result<()> {
    Ok(())
}

#[cfg(unix)]
extern "C" {
    fn fsync(fd: std::os::unix::io::RawFd) -> std::ffi::c_int;
}

/// Compute the lowercase hex SHA-256 digest of `input`.
///
/// A dependency-free implementation keeps the cache verifiable without
/// pulling a crypto crate into the default feature set.
pub fn sha256_hex(input: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(input);
    hasher.finish_hex()
}

struct Sha256 {
    state: [u32; 8],
    buffer: [u8; 64],
    buffer_len: usize,
    total_len: u64,
}

impl Sha256 {
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

    fn new() -> Self {
        Self {
            state: [
                0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c,
                0x1f83d9ab, 0x5be0cd19,
            ],
            buffer: [0; 64],
            buffer_len: 0,
            total_len: 0,
        }
    }

    fn update(&mut self, mut input: &[u8]) {
        self.total_len = self.total_len.wrapping_add(input.len() as u64);
        if self.buffer_len > 0 {
            let needed = 64 - self.buffer_len;
            let taken = needed.min(input.len());
            self.buffer[self.buffer_len..self.buffer_len + taken]
                .copy_from_slice(&input[..taken]);
            self.buffer_len += taken;
            input = &input[taken..];
            if self.buffer_len == 64 {
                let block = self.buffer;
                self.compress(&block);
                self.buffer_len = 0;
            }
        }
        while input.len() >= 64 {
            let mut block = [0u8; 64];
            block.copy_from_slice(&input[..64]);
            self.compress(&block);
            input = &input[64..];
        }
        if !input.is_empty() {
            self.buffer[..input.len()].copy_from_slice(input);
            self.buffer_len = input.len();
        }
    }

    fn finish(mut self) -> [u8; 32] {
        let bit_len = self.total_len.wrapping_mul(8);
        if self.buffer_len >= 56 {
            self.buffer[self.buffer_len] = 0x80;
            for byte in &mut self.buffer[self.buffer_len + 1..] {
                *byte = 0;
            }
            let block = self.buffer;
            self.compress(&block);
            self.buffer = [0; 64];
        } else {
            self.buffer[self.buffer_len] = 0x80;
            for byte in &mut self.buffer[self.buffer_len + 1..] {
                *byte = 0;
            }
        }
        self.buffer[56..].copy_from_slice(&bit_len.to_be_bytes());
        let block = self.buffer;
        self.compress(&block);
        let mut out = [0u8; 32];
        for (word, chunk) in self.state.iter().zip(out.chunks_exact_mut(4)) {
            chunk.copy_from_slice(&word.to_be_bytes());
        }
        out
    }

    fn finish_hex(self) -> String {
        let mut out = String::with_capacity(64);
        for byte in self.finish() {
            out.push_str(&format!("{byte:02x}"));
        }
        out
    }

    fn compress(&mut self, block: &[u8; 64]) {
        let mut words = [0u32; 64];
        for (index, chunk) in block.chunks_exact(4).enumerate() {
            words[index] = u32::from_be_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        }
        for index in 16..64 {
            let s0 = words[index - 15].rotate_right(7)
                ^ words[index - 15].rotate_right(18)
                ^ (words[index - 15] >> 3);
            let s1 = words[index - 2].rotate_right(17)
                ^ words[index - 2].rotate_right(19)
                ^ (words[index - 2] >> 10);
            words[index] = words[index - 16]
                .wrapping_add(s0)
                .wrapping_add(words[index - 7])
                .wrapping_add(s1);
        }
        let mut working = self.state;
        for index in 0..64 {
            let s1 = working[4].rotate_right(6)
                ^ working[4].rotate_right(11)
                ^ working[4].rotate_right(25);
            let choice = (working[4] & working[5]) ^ ((!working[4]) & working[6]);
            let temp1 = working[7]
                .wrapping_add(s1)
                .wrapping_add(choice)
                .wrapping_add(Self::K[index])
                .wrapping_add(words[index]);
            let s0 = working[0].rotate_right(2)
                ^ working[0].rotate_right(13)
                ^ working[0].rotate_right(22);
            let majority = (working[0] & working[1]) ^ (working[0] & working[2]) ^ (working[1] & working[2]);
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
        for (state, word) in self.state.iter_mut().zip(working) {
            *state = state.wrapping_add(word);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sha256_matches_known_vectors() {
        // NIST FIPS 180-2 examples.
        assert_eq!(
            sha256_hex(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        assert_eq!(
            sha256_hex(b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"),
            "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
        );
    }

    #[test]
    fn metadata_roundtrips_and_detects_tampering() {
        let metadata = CacheMetadata::new("https://example.com/a.pkl", "pkl", b"value = 1\n");
        let parsed = CacheMetadata::parse(&metadata.to_json()).unwrap();
        assert_eq!(parsed, metadata);
        parsed
            .verify("https://example.com/a.pkl", "pkl", b"value = 1\n")
            .unwrap();
        assert!(matches!(
            parsed.verify("https://example.com/a.pkl", "pkl", b"tampered"),
            Err(CacheError::DigestMismatch { .. })
        ));
    }

    #[test]
    fn rejects_corrupt_and_versioned_envelopes() {
        assert!(matches!(
            CacheMetadata::parse("not json"),
            Err(CacheError::CorruptMetadata(_))
        ));
        assert!(matches!(
            CacheMetadata::parse(
                &json!({"version": 99, "url": "u", "extension": "pkl", "sha256": "x"})
                    .to_string()
            ),
            Err(CacheError::UnsupportedVersion(99))
        ));
    }

    #[test]
    fn temp_names_are_recognized_across_processes() {
        let name = temp_file_name("deadbeef.pkl");
        assert!(is_temp_file_name(&name));
        assert!(is_temp_file_name(".deadbeef.pkl.tmp-42-7"));
        assert!(!is_temp_file_name("deadbeef.pkl"));
        assert!(!is_temp_file_name(".deadbeef.pkl.tmp-x-1"));
    }

    #[test]
    fn entry_paths_stay_inside_the_packages_dir() {
        let (blob, metadata) =
            entry_paths(Path::new("/cache"), "https://example.com/a.zip", "zip").unwrap();
        assert!(blob.starts_with("/cache/packages"));
        assert!(metadata.starts_with("/cache/packages"));
        assert!(ensure_within(
            Path::new("/cache/packages"),
            Path::new("/cache/packages/a.zip")
        )
        .is_ok());
        assert!(matches!(
            ensure_within(
                Path::new("/cache/packages"),
                Path::new("/cache/escape.zip")
            ),
            Err(CacheError::PathEscape(_))
        ));
        assert!(matches!(
            ensure_within(
                Path::new("/cache/packages"),
                Path::new("/cache/packages/../escape.zip")
            ),
            Err(CacheError::PathEscape(_))
        ));
    }
}
