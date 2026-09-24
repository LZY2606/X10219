//! Versioned on-disk layout for the persistent package cache.
//!
//! Every cached package occupies a single directory containing:
//!
//! * `meta.json` - the commit marker, written last;
//! * the blob (`data.<extension>`) - validated package bytes.
//!
//! An entry without a parseable, version-supported `meta.json` whose digest
//! matches the blob is never treated as a hit. Writes land beside the entry
//! as dot-prefixed staging files so that interrupted writes are recognizable
//! and removable on the next startup without touching valid entries.

use std::path::{Component, Path, PathBuf};

use crate::error::{Error, Result};

/// Metadata format version understood by this build.
pub const CACHE_METADATA_VERSION: u32 = 1;

const METADATA_FILE_NAME: &str = "meta.json";
const MAX_METADATA_BYTES: usize = 64 * 1024;
const MAX_SWEEP_DEPTH: usize = 4;

/// Filesystem-safe identity for one cached package URL.
pub fn entry_id(url: &str) -> String {
    // FNV-1a keeps the layout stable and independent of the URL's characters.
    // The URL stored in `meta.json` detects hash collisions.
    let mut hash = 0xcbf29ce484222325_u64;
    for byte in url.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    format!("{hash:016x}")
}

/// Directory holding `meta.json` and the blob for one package.
pub fn entry_dir(cache_dir: &Path, url: &str) -> PathBuf {
    cache_dir.join("packages").join(entry_id(url))
}

/// `(metadata_path, blob_path)` for one package cache entry.
pub fn entry_paths(cache_dir: &Path, url: &str, extension: &str) -> (PathBuf, PathBuf) {
    let dir = entry_dir(cache_dir, url);
    (
        dir.join(METADATA_FILE_NAME),
        dir.join(format!("data.{}", safe_extension(extension))),
    )
}

fn safe_extension(extension: &str) -> String {
    let cleaned: String = extension
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() {
                ch
            } else {
                '-'
            }
        })
        .collect();
    if cleaned.is_empty() {
        "bin".to_string()
    } else {
        cleaned
    }
}

/// Serialized metadata for a cache entry. The JSON document is the commit
/// marker: until it exists and validates, the entry is a cache miss.
#[derive(Debug, PartialEq, Eq)]
pub struct CacheMetadata {
    pub version: u32,
    pub url: String,
    pub extension: String,
    pub sha256: String,
    pub blob: String,
}

impl CacheMetadata {
    pub fn new(url: &str, extension: &str, sha256: &str) -> Self {
        Self {
            version: CACHE_METADATA_VERSION,
            url: url.to_string(),
            extension: safe_extension(extension),
            sha256: sha256.to_string(),
            blob: format!("data.{}", safe_extension(extension)),
        }
    }

    pub fn to_json(&self) -> String {
        format!(
            "{{\"version\":{},\"url\":{},\"extension\":{},\"sha256\":{},\"blob\":{}}}\n",
            self.version,
            json_string(&self.url),
            json_string(&self.extension),
            json_string(&self.sha256),
            json_string(&self.blob),
        )
    }

    /// Parse metadata, rejecting unknown versions, malformed JSON, and
    /// unsafe blob references.
    pub fn parse(bytes: &[u8]) -> Result<Self> {
        if bytes.len() > MAX_METADATA_BYTES {
            return Err(Error::Eval("package cache metadata is too large".into()));
        }
        let value: serde_json::Value = serde_json::from_slice(bytes)
            .map_err(|error| Error::Eval(format!("corrupt package cache metadata: {error}")))?;
        let object = value.as_object().ok_or_else(|| {
            Error::Eval("corrupt package cache metadata: expected a JSON object".into())
        })?;
        let version = object
            .get("version")
            .and_then(|value| value.as_u64())
            .ok_or_else(|| Error::Eval("corrupt package cache metadata: missing version".into()))?;
        if version != u64::from(CACHE_METADATA_VERSION) {
            return Err(Error::Eval(format!(
                "unsupported package cache metadata version: {version}"
            )));
        }
        let url = object
            .get("url")
            .and_then(|value| value.as_str())
            .ok_or_else(|| Error::Eval("corrupt package cache metadata: missing url".into()))?;
        let extension = object
            .get("extension")
            .and_then(|value| value.as_str())
            .ok_or_else(|| {
                Error::Eval("corrupt package cache metadata: missing extension".into())
            })?;
        let sha256 = object
            .get("sha256")
            .and_then(|value| value.as_str())
            .ok_or_else(|| {
                Error::Eval("corrupt package cache metadata: missing sha256".into())
            })?;
        let blob = object
            .get("blob")
            .and_then(|value| value.as_str())
            .ok_or_else(|| Error::Eval("corrupt package cache metadata: missing blob".into()))?;
        if !is_safe_blob_name(blob) {
            return Err(Error::Eval(format!(
                "package cache metadata points outside the cache entry: {blob}"
            )));
        }
        Ok(Self {
            version: version as u32,
            url: url.to_string(),
            extension: extension.to_string(),
            sha256: sha256.to_string(),
            blob: blob.to_string(),
        })
    }
}

/// Blob names stay inside the entry directory: no absolutes, roots, or `..`.
pub fn is_safe_blob_name(name: &str) -> bool {
    if name.is_empty() || name == METADATA_FILE_NAME {
        return false;
    }
    let path = Path::new(name);
    if path.is_absolute() {
        return false;
    }
    path.components().all(|component| {
        matches!(component, Component::Normal(part) if is_safe_file_name(&part.to_string_lossy()))
    })
}

fn is_safe_file_name(name: &str) -> bool {
    !name.is_empty()
        && name != "."
        && name != ".."
        && !name.starts_with('.')
        && name != METADATA_FILE_NAME
}

fn json_string(value: &str) -> String {
    serde_json::Value::String(value.to_string()).to_string()
}

/// Lowercase hex SHA-256 digest of `bytes`.
pub fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut out = String::with_capacity(64);
    for byte in digest {
        out.push_str(&format!("{byte:02x}"));
    }
    out
}

/// A staging file name left by an interrupted atomic write.
pub fn is_staging_file_name(name: &str) -> bool {
    // Keep this in sync with `crate::eval::write_atomic`'s temp file scheme.
    name.starts_with('.') && name.contains(".tmp-")
}

/// List `(file_name, full_path)` for regular files under `root`, bounded by
/// [`MAX_SWEEP_DEPTH`]. Unreadable directories are skipped.
pub fn list_files_bounded(root: &Path) -> Vec<(String, PathBuf)> {
    let mut found = Vec::new();
    collect_files(root, root, 0, &mut found);
    found
}

fn collect_files(root: &Path, dir: &Path, depth: usize, found: &mut Vec<(String, PathBuf)>) {
    if depth > MAX_SWEEP_DEPTH {
        return;
    }
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let file_type = match entry.file_type() {
            Ok(file_type) => file_type,
            Err(_) => continue,
        };
        if file_type.is_dir() {
            collect_files(root, &path, depth + 1, found);
        } else if file_type.is_file() {
            let relative = path
                .strip_prefix(root)
                .map(|path| path.to_string_lossy().replace('\\', "/"))
                .unwrap_or_else(|_| path.to_string_lossy().to_string());
            found.push((relative, path));
        }
    }
}

/// Minimal, dependency-free SHA-256 (FIPS 180-4).
#[derive(Clone)]
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

    fn digest(mut data: &[u8]) -> [u8; 32] {
        let mut hasher = Self::new();
        while data.len() >= 64 {
            hasher.compress(data);
            hasher.total_len = hasher.total_len.wrapping_add(64);
            data = &data[64..];
        }
        hasher.total_len = hasher.total_len.wrapping_add(data.len() as u64);
        hasher.buffer[..data.len()].copy_from_slice(data);
        hasher.buffer_len = data.len();
        hasher.finalize()
    }

    fn compress(&mut self, block: &[u8]) {
        let mut words = [0u32; 64];
        for (index, word) in block.chunks_exact(4).enumerate() {
            words[index] = u32::from_be_bytes([word[0], word[1], word[2], word[3]]);
        }
        for index in 16..64 {
            let small = words[index - 15];
            let large = words[index - 2];
            let s0 = small.rotate_right(7) ^ small.rotate_right(18) ^ (small >> 3);
            let s1 = large.rotate_right(17) ^ large.rotate_right(19) ^ (large >> 10);
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
            let choose = (working[4] & working[5]) ^ ((!working[4]) & working[6]);
            let temp1 = working[7]
                .wrapping_add(s1)
                .wrapping_add(choose)
                .wrapping_add(Self::K[index])
                .wrapping_add(words[index]);
            let s0 = working[0].rotate_right(2)
                ^ working[0].rotate_right(13)
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
        for (state, value) in self.state.iter_mut().zip(working) {
            *state = state.wrapping_add(value);
        }
    }

    fn finalize(mut self) -> [u8; 32] {
        let bit_len = self.total_len.wrapping_mul(8);
        self.buffer[self.buffer_len] = 0x80;
        for byte in &mut self.buffer[self.buffer_len + 1..] {
            *byte = 0;
        }
        let zero_pad_from = if self.buffer_len < 56 {
            self.buffer_len + 1
        } else {
            self.compress(&self.buffer.clone());
            for byte in &mut self.buffer[..] {
                *byte = 0;
            }
            0
        };
        self.buffer[zero_pad_from..56].fill(0);
        self.buffer[56..].copy_from_slice(&bit_len.to_be_bytes());
        self.compress(&self.buffer.clone());
        let mut digest = [0u8; 32];
        for (word, chunk) in self.state.iter().zip(digest.chunks_mut(4)) {
            chunk.copy_from_slice(&word.to_be_bytes());
        }
        digest
    }
}

#[cfg(test)]
mod tests {
    use super::{CacheMetadata, entry_paths, is_safe_blob_name, sha256_hex};

    #[test]
    fn sha256_matches_known_vectors() {
        // FIPS 180-4 examples, without pulling in a crypto dependency.
        assert_eq!(
            sha256_hex(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        assert_eq!(
            sha256_hex(b"a".repeat(1_000_000).as_bytes()),
            "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0"
        );
    }

    #[test]
    fn metadata_round_trips_and_rejects_unsafe_blobs() {
        let metadata = CacheMetadata::new("https://example.com/p.zip", "zip", "digest");
        let parsed =
            CacheMetadata::parse(metadata.to_json().as_bytes()).expect("metadata should parse");
        assert_eq!(parsed, metadata);
        assert!(is_safe_blob_name("data.zip"));
        assert!(!is_safe_blob_name("../escape"));
        assert!(!is_safe_blob_name("/abs/path"));
        assert!(!is_safe_blob_name("dir/../../x"));
    }

    #[test]
    fn metadata_rejects_unknown_versions() {
        let error = CacheMetadata::parse(
            br#"{"version":99,"url":"u","extension":"zip","sha256":"x","blob":"data.zip"}"#,
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("unsupported package cache metadata version"));
    }

    #[test]
    fn entry_paths_are_stable() {
        let (meta, blob) = entry_paths(std::path::Path::new("cache"), "https://x/y.zip", "zip");
        assert!(meta.ends_with("meta.json"));
        assert!(blob.ends_with("data.zip"));
        assert_eq!(meta.parent(), blob.parent());
    }
}
