//! Persistent on-disk cache for downloaded package content.
//!
//! Every cached package consists of two sibling files:
//!
//! ```text
//! <cache-dir>/packages/<fnv1a64(url)>.<ext>.blob
//! <cache-dir>/packages/<fnv1a64(url)>.<ext>.meta.json
//! ```
//!
//! The metadata file is versioned (`v1`) and records the canonical URL of the
//! content, its extension and a SHA-256 digest. A cache hit is only served when
//! the metadata parses, its version is supported, the URL matches, the blob
//! path stays inside the cache directory and the digest verifies. Anything else
//! is rejected as corruption instead of being treated as a hit.
//!
//! Writes are staged through uniquely named temp files and published with a
//! rename so a crash never leaves a half-written entry that reads as a hit.
//! Leftover temp files from a previous process are reclaimed on startup.

use std::path::{Path, PathBuf};

use crate::error::Result;

/// Metadata format version understood by this build.
pub const CACHE_METADATA_VERSION: u32 = 1;
/// Metadata revision that will never be produced by this build.
#[allow(dead_code)]
pub const FUTURE_CACHE_METADATA_VERSION: u32 = 999;
const PACKAGES_DIR: &str = "packages";
const META_SUFFIX: &str = "meta.json";

#[cfg(feature = "eval-core")]
pub(crate) mod digest {
    /// Minimal, dependency-free SHA-256 (FIPS 180-4).
    ///
    /// Used for package blob integrity checks in the persistent cache.
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

        let mut h: [u32; 8] = [
            0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
            0x5be0cd19,
        ];

        let bit_len = (input.len() as u64).wrapping_mul(8);
        let mut padded = input.to_vec();
        padded.push(0x80);
        while padded.len() % 64 != 56 {
            padded.push(0);
        }
        padded.extend_from_slice(&bit_len.to_be_bytes());

        for chunk in padded.chunks_exact(64) {
            let mut w = [0u32; 64];
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

            let mut a = h[0];
            let mut b = h[1];
            let mut c = h[2];
            let mut d = h[3];
            let mut e = h[4];
            let mut f = h[5];
            let mut g = h[6];
            let mut hh = h[7];

            for index in 0..64 {
                let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
                let ch = (e & f) ^ ((!e) & g);
                let temp1 = hh
                    .wrapping_add(s1)
                    .wrapping_add(ch)
                    .wrapping_add(K[index])
                    .wrapping_add(w[index]);
                let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
                let maj = (a & b) ^ (a & c) ^ (b & c);
                let temp2 = s0.wrapping_add(maj);
                hh = g;
                g = f;
                f = e;
                e = d.wrapping_add(temp1);
                d = c;
                c = b;
                b = a;
                a = temp1.wrapping_add(temp2);
            }

            h[0] = h[0].wrapping_add(a);
            h[1] = h[1].wrapping_add(b);
            h[2] = h[2].wrapping_add(c);
            h[3] = h[3].wrapping_add(d);
            h[4] = h[4].wrapping_add(e);
            h[5] = h[5].wrapping_add(f);
            h[6] = h[6].wrapping_add(g);
            h[7] = h[7].wrapping_add(hh);
        }

        let mut out = String::with_capacity(64);
        for word in h {
            out.push_str(&format!("{word:08x}"));
        }
        out
    }

    #[cfg(test)]
    mod tests {
        use super::sha256_hex;

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
            let repeated = vec![b'a'; 1000];
            assert_eq!(
                sha256_hex(&repeated),
                "41edece42d63e8d9bf515a9ba6932e1c20cbc9f5a5d134645adb5db1b9737ea3"
            );
        }
    }
}

#[cfg(feature = "eval-core")]
pub(crate) fn fnv1a64(input: &str) -> u64 {
    let mut hash = 0xcbf29ce484222325_u64;
    for byte in input.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

/// Sibling paths `(blob, metadata)` for a package URL.
#[cfg(feature = "eval-core")]
pub(crate) fn package_cache_paths(
    cache_dir: &Path,
    url: &str,
    extension: &str,
) -> (PathBuf, PathBuf) {
    let base = cache_dir
        .join(PACKAGES_DIR)
        .join(format!("{:016x}", fnv1a64(url)));
    (
        base.with_extension(format!("{extension}.blob")),
        base.with_extension(format!("{extension}.{META_SUFFIX}")),
    )
}

#[cfg(feature = "eval-core")]
fn metadata_json(url: &str, extension: &str, digest: &str) -> String {
    format!(
        "{{\"version\":{version},\"url\":{url},\"extension\":{extension},\"digest\":{digest}}}",
        version = CACHE_METADATA_VERSION,
        url = serde_json::Value::from(url),
        extension = serde_json::Value::from(extension),
        digest = serde_json::Value::from(digest),
    )
}

/// Parse and fully validate a cache metadata document.
///
/// `expected_url`/`expected_extension` identify the entry being looked up.
/// Returns the validated blob path together with the digest it must match.
#[cfg(feature = "eval-core")]
pub(crate) fn parse_metadata(
    metadata_text: &str,
    expected_url: &str,
    expected_extension: &str,
    cache_dir: &Path,
) -> Result<(PathBuf, String)> {
    let document: serde_json::Value = serde_json::from_str(metadata_text).map_err(|error| {
        crate::Error::Eval(format!("corrupt package cache metadata: {error}"))
    })?;
    let object = document.as_object().ok_or_else(|| {
        crate::Error::Eval("corrupt package cache metadata: expected a JSON object".to_string())
    })?;

    let version = object
        .get("version")
        .and_then(|value| value.as_u64())
        .ok_or_else(|| {
            crate::Error::Eval("corrupt package cache metadata: missing version".to_string())
        })?;
    if version != u64::from(CACHE_METADATA_VERSION) {
        return Err(crate::Error::Eval(format!(
            "unsupported package cache metadata version: {version}"
        )));
    }

    let stored_url = object
        .get("url")
        .and_then(|value| value.as_str())
        .ok_or_else(|| {
            crate::Error::Eval("corrupt package cache metadata: missing url".to_string())
        })?;
    if stored_url != expected_url {
        return Err(crate::Error::Eval(format!(
            "corrupt package cache metadata: url mismatch (stored {stored_url})"
        )));
    }

    let stored_extension = object
        .get("extension")
        .and_then(|value| value.as_str())
        .ok_or_else(|| {
            crate::Error::Eval("corrupt package cache metadata: missing extension".to_string())
        })?;
    if stored_extension != expected_extension {
        return Err(crate::Error::Eval(format!(
            "corrupt package cache metadata: extension mismatch (stored {stored_extension})"
        )));
    }

    let stored_digest = object
        .get("digest")
        .and_then(|value| value.as_str())
        .ok_or_else(|| {
            crate::Error::Eval("corrupt package cache metadata: missing digest".to_string())
        })?;

    let (blob_path, _) = package_cache_paths(cache_dir, expected_url, expected_extension);
    if let Some(value) = object.get("blob") {
        let name = value.as_str().ok_or_else(|| {
            crate::Error::Eval(
                "corrupt package cache metadata: blob must be a string".to_string(),
            )
        })?;
        let referenced = Path::new(name);
        if referenced.is_absolute()
            || referenced
                .components()
                .any(|component| !matches!(component, std::path::Component::Normal(_)))
        {
            return Err(crate::Error::Eval(format!(
                "corrupt package cache metadata: blob path escapes the cache directory: {name}"
            )));
        }
        let resolved = cache_dir.join(PACKAGES_DIR).join(referenced);
        if resolved != blob_path || !stays_within(&resolved, cache_dir) {
            return Err(crate::Error::Eval(format!(
                "corrupt package cache metadata: blob path escapes the cache directory: {name}"
            )));
        }
    }

    Ok((blob_path, stored_digest.to_string()))
}

#[cfg(feature = "eval-core")]
fn stays_within(path: &Path, root: &Path) -> bool {
    use std::path::Component;
    let mut anchor: Vec<&std::ffi::OsStr> = Vec::new();
    for component in root.components() {
        if let Component::Normal(part) = component {
            anchor.push(part);
        }
    }
    let mut current: Vec<&std::ffi::OsStr> = Vec::new();
    for component in path.components() {
        match component {
            Component::Normal(part) => current.push(part),
            Component::ParentDir => {
                if current.pop().is_none() {
                    return false;
                }
            }
            Component::CurDir => {}
            Component::RootDir | Component::Prefix(_) => {}
        }
    }
    current.starts_with(&anchor)
}

#[cfg(feature = "eval-core")]
pub(crate) fn encode_metadata(url: &str, extension: &str, bytes: &[u8]) -> Vec<u8> {
    metadata_json(
        url,
        extension,
        &digest::sha256_hex(bytes),
    )
    .into_bytes()
}

#[cfg(all(test, feature = "eval-core"))]
mod metadata_tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn valid_metadata_round_trips() {
        let root = Path::new("/cache");
        let bytes = b"answer = 42\n";
        let encoded = encode_metadata("https://x/y.zip", "zip", bytes);
        let (blob, digest) =
            parse_metadata(&String::from_utf8(encoded).unwrap(), "https://x/y.zip", "zip", root)
                .unwrap();
        assert!(blob.starts_with(root));
        assert_eq!(digest, digest::sha256_hex(bytes));
    }

    #[test]
    fn unsupported_versions_are_rejected() {
        let document = format!(
            "{{\"version\":{v},\"url\":\"u\",\"extension\":\"zip\",\"digest\":\"d\"}}",
            v = FUTURE_CACHE_METADATA_VERSION
        );
        let error = parse_metadata(&document, "u", "zip", Path::new("/cache")).unwrap_err();
        assert!(error.to_string().contains("unsupported package cache metadata version"));
    }

    #[test]
    fn escaping_blob_paths_are_rejected() {
        let document = r#"{"version":1,"url":"u","extension":"zip","digest":"d","blob":"../../../etc/passwd"}"#;
        let error = parse_metadata(document, "u", "zip", Path::new("/cache")).unwrap_err();
        assert!(error
            .to_string()
            .contains("escapes the cache directory"));

        let absolute = r#"{"version":1,"url":"u","extension":"zip","digest":"d","blob":"/etc/passwd"}"#;
        let error = parse_metadata(absolute, "u", "zip", Path::new("/cache")).unwrap_err();
        assert!(error
            .to_string()
            .contains("escapes the cache directory"));
    }

    #[test]
    fn mismatched_url_is_rejected() {
        let document = r#"{"version":1,"url":"other","extension":"zip","digest":"d"}"#;
        let error = parse_metadata(document, "u", "zip", Path::new("/cache")).unwrap_err();
        assert!(error.to_string().contains("url mismatch"), "{error}");
    }
}

/// Which half of a cache entry a storage boundary is operating on.
#[cfg(feature = "eval-core")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheFile {
    Blob,
    Metadata,
}

impl CacheFile {
    pub fn label(self) -> &'static str {
        match self {
            Self::Blob => "blob",
            Self::Metadata => "metadata",
        }
    }
}

/// Unique temp path for staging `destination`.
#[cfg(feature = "eval-core")]
pub(crate) fn temp_sibling_for(destination: &Path) -> PathBuf {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let file_name = destination
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("cache");
    destination.with_file_name(format!(
        ".{file_name}.tmp-{}-{}",
        std::process::id(),
        COUNTER.fetch_add(1, Ordering::Relaxed)
    ))
}

/// Filename component shared by temp files left behind by interrupted writes.
#[cfg(feature = "eval-core")]
pub(crate) const TEMP_FILE_MARKER: &str = ".tmp-";

/// Boundaries at which cache storage can fail. Each maps to one distinct
/// observable operation so fault injection tests can target it precisely.
#[cfg(feature = "eval-core")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheStage {
    /// Creating and writing the temp file.
    Write,
    /// Flushing file contents (`fsync` on the file).
    Flush,
    /// Publishing the temp file with a rename.
    Rename,
    /// Flushing the containing directory after the metadata rename.
    DirectorySync,
}

impl CacheStage {
    pub fn label(self) -> &'static str {
        match self {
            Self::Write => "write",
            Self::Flush => "flush",
            Self::Rename => "rename",
            Self::DirectorySync => "directory-sync",
        }
    }
}
