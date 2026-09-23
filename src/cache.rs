//! Persistent package cache layout and validation.
//!
//! Layout under the configured cache directory:
//!
//! ```text
//! packages/<fnv1a64(url)>.<extension>   blob (package bytes)
//! packages/<fnv1a64(url)>.meta          JSON metadata sidecar (commit point)
//! ```
//!
//! The metadata file is written last and is the commit point: a blob without a
//! valid metadata sidecar is never treated as a cache hit. Metadata records the
//! format version, the source URL, and the SHA-256 of the blob. Temporary files
//! left behind by an interrupted atomic write match [`is_stale_temp_file`] and
//! are swept on cache startup.

use std::path::{Path, PathBuf};

use crate::sha256::sha256_hex;

/// Cache metadata format version written by this build.
pub const CACHE_FORMAT_VERSION: u32 = 1;

/// Metadata for one cached package blob.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CacheMeta {
    pub version: u32,
    pub url: String,
    pub sha256: String,
}

impl CacheMeta {
    pub fn for_blob(url: &str, blob: &[u8]) -> Self {
        Self {
            version: CACHE_FORMAT_VERSION,
            url: url.to_string(),
            sha256: sha256_hex(blob),
        }
    }

    pub fn encode(&self) -> Vec<u8> {
        serde_json::json!({
            "version": self.version,
            "url": self.url,
            "sha256": self.sha256,
        })
        .to_string()
        .into_bytes()
    }

    pub fn decode(bytes: &[u8]) -> Option<Self> {
        let value: serde_json::Value = serde_json::from_slice(bytes).ok()?;
        Some(Self {
            version: value.get("version")?.as_u64()? as u32,
            url: value.get("url")?.as_str()?.to_string(),
            sha256: value.get("sha256")?.as_str()?.to_string(),
        })
    }
}

/// Why a cache entry was rejected. Each variant maps to a distinct corruption
/// or incompatibility that must never be treated as a cache hit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CacheRejection {
    /// No metadata sidecar: the write never committed.
    MissingMetadata,
    /// Metadata is not well-formed JSON with the required fields.
    CorruptMetadata,
    /// Metadata version is newer than this build understands.
    UnsupportedVersion(u32),
    /// The entry belongs to a different URL (hash collision or relocation).
    UrlMismatch,
    /// Blob bytes do not match the recorded digest.
    ChecksumMismatch,
}

impl std::fmt::Display for CacheRejection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingMetadata => write!(f, "cache entry has no metadata"),
            Self::CorruptMetadata => write!(f, "cache metadata is corrupt"),
            Self::UnsupportedVersion(version) => {
                write!(f, "unsupported cache metadata version {version}")
            }
            Self::UrlMismatch => write!(f, "cache entry URL does not match"),
            Self::ChecksumMismatch => write!(f, "cache blob digest mismatch"),
        }
    }
}

/// Validate `meta_bytes` and `blob` for `url`. Returns the rejection reason
/// when the entry must not be used.
pub fn validate_entry(meta_bytes: &[u8], url: &str, blob: &[u8]) -> Result<(), CacheRejection> {
    let meta = CacheMeta::decode(meta_bytes).ok_or(CacheRejection::CorruptMetadata)?;
    if meta.version != CACHE_FORMAT_VERSION {
        return Err(CacheRejection::UnsupportedVersion(meta.version));
    }
    if meta.url != url {
        return Err(CacheRejection::UrlMismatch);
    }
    if meta.sha256 != sha256_hex(blob) {
        return Err(CacheRejection::ChecksumMismatch);
    }
    Ok(())
}

/// FNV-1a 64-bit of `url`, hex-encoded: a stable, filesystem-safe entry name
/// that cannot contain separators or `..`, so cache entries can never escape
/// the cache directory.
pub fn entry_file_name(url: &str) -> String {
    let mut hash = 0xcbf29ce484222325_u64;
    for byte in url.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    format!("{hash:016x}")
}

/// `(blob_path, meta_path)` for `url` under `cache_dir`.
pub fn entry_paths(cache_dir: &Path, url: &str, extension: &str) -> (PathBuf, PathBuf) {
    let base = cache_dir.join("packages").join(entry_file_name(url));
    (base.with_extension(extension), base.with_extension("meta"))
}

/// Legacy URL sidecar written by pklr versions before the metadata format.
pub fn legacy_url_path(cache_dir: &Path, url: &str) -> PathBuf {
    cache_dir
        .join("packages")
        .join(entry_file_name(url))
        .with_extension("url")
}

/// True when `path` stays within `dir` after lexical normalization (no
/// component escapes via `..`). Used to prove cache and extraction paths
/// cannot point outside their configured root.
pub fn is_contained_in(dir: &Path, path: &Path) -> bool {
    fn normalize(path: &Path) -> Vec<std::ffi::OsString> {
        let mut out: Vec<std::ffi::OsString> = Vec::new();
        for component in path.components() {
            match component {
                std::path::Component::ParentDir => {
                    out.pop();
                }
                std::path::Component::CurDir => {}
                other => out.push(other.as_os_str().to_os_string()),
            }
        }
        out
    }
    let dir_components = normalize(dir);
    let path_components = normalize(path);
    path_components.len() >= dir_components.len()
        && path_components
            .iter()
            .zip(dir_components.iter())
            .all(|(a, b)| a == b)
}

/// True for temp files left by an interrupted atomic cache write. The pattern
/// matches `write_atomic`'s `.{name}.tmp-{pid}-{counter}` naming.
pub fn is_stale_temp_file(file_name: &str) -> bool {
    let Some(rest) = file_name.strip_prefix('.') else {
        return false;
    };
    let Some((name, suffix)) = rest.split_once(".tmp-") else {
        return false;
    };
    if name.is_empty() {
        return false;
    }
    // suffix is "{pid}-{counter}" with numeric parts
    let mut parts = suffix.split('-');
    let pid = parts.next().unwrap_or("");
    let counter = parts.next().unwrap_or("");
    parts.next().is_none()
        && !pid.is_empty()
        && pid.bytes().all(|b| b.is_ascii_digit())
        && !counter.is_empty()
        && counter.bytes().all(|b| b.is_ascii_digit())
}

/// Reject zip archives whose entries would extract outside `destination`
/// (zip-slip). Returns the offending entry name on failure.
#[cfg(feature = "package-zip-core")]
pub fn validate_zip_entries(bytes: &[u8]) -> crate::Result<()> {
    let mut archive = zip::ZipArchive::new(std::io::Cursor::new(bytes))
        .map_err(|error| crate::Error::Eval(format!("zip error: {error}")))?;
    for index in 0..archive.len() {
        let entry = archive
            .by_index(index)
            .map_err(|error| crate::Error::Eval(format!("zip error: {error}")))?;
        if entry.enclosed_name().is_none() {
            return Err(crate::Error::Eval(format!(
                "package archive entry escapes its destination: {}",
                entry.name()
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn meta_round_trips() {
        let meta = CacheMeta::for_blob("https://example.com/pkg.zip", b"payload");
        let decoded = CacheMeta::decode(&meta.encode()).unwrap();
        assert_eq!(decoded, meta);
    }

    #[test]
    fn validate_entry_accepts_fresh_write() {
        let blob = b"answer = 42\n";
        let meta = CacheMeta::for_blob("https://example.com/a.pkl", blob);
        assert_eq!(validate_entry(&meta.encode(), "https://example.com/a.pkl", blob), Ok(()));
    }

    #[test]
    fn validate_entry_rejects_each_corruption() {
        let blob = b"data";
        let url = "https://example.com/a.pkl";
        let meta = CacheMeta::for_blob(url, blob);

        assert_eq!(
            validate_entry(b"{not json", url, blob),
            Err(CacheRejection::CorruptMetadata)
        );
        assert_eq!(
            validate_entry(b"{}", url, blob),
            Err(CacheRejection::CorruptMetadata)
        );

        let mut future = meta.clone();
        future.version = CACHE_FORMAT_VERSION + 1;
        assert_eq!(
            validate_entry(&future.encode(), url, blob),
            Err(CacheRejection::UnsupportedVersion(CACHE_FORMAT_VERSION + 1))
        );

        let other = CacheMeta::for_blob("https://example.com/b.pkl", blob);
        assert_eq!(
            validate_entry(&other.encode(), url, blob),
            Err(CacheRejection::UrlMismatch)
        );

        assert_eq!(
            validate_entry(&meta.encode(), url, b"tampered"),
            Err(CacheRejection::ChecksumMismatch)
        );
    }

    #[test]
    fn entry_names_stay_inside_the_cache_dir() {
        let cache_dir = Path::new("/cache");
        for url in [
            "https://example.com/../../etc/passwd",
            "https://example.com/%2e%2e/secret",
            "package://host/../../../escape#/x.pkl",
            "https://example.com/\0nul",
        ] {
            let (blob, meta) = entry_paths(cache_dir, url, "zip");
            assert!(is_contained_in(cache_dir, &blob), "{blob:?}");
            assert!(is_contained_in(cache_dir, &meta), "{meta:?}");
        }
    }

    #[test]
    fn containment_rejects_escapes() {
        let dir = Path::new("/cache");
        assert!(is_contained_in(dir, &dir.join("packages").join("abc.zip")));
        assert!(!is_contained_in(dir, &dir.join("..").join("etc").join("passwd")));
        assert!(!is_contained_in(dir, Path::new("/etc/passwd")));
        assert!(is_contained_in(dir, &dir.join("a").join("..").join("b")));
    }

    #[test]
    fn stale_temp_file_pattern_matches_write_atomic_names() {
        assert!(is_stale_temp_file(".abc123.zip.tmp-4242-7"));
        assert!(is_stale_temp_file(".abc123.meta.tmp-1-0"));
        assert!(!is_stale_temp_file("abc123.zip"));
        assert!(!is_stale_temp_file(".abc123.zip"));
        assert!(!is_stale_temp_file(".tmp-1-2"));
        assert!(!is_stale_temp_file(".abc.tmp-1"));
        assert!(!is_stale_temp_file(".abc.tmp-1-2-3"));
        assert!(!is_stale_temp_file(".abc.tmp-pid-1"));
    }

    #[cfg(feature = "package-zip-core")]
    #[test]
    fn zip_slip_entries_are_rejected() {
        use std::io::Write;

        let mut bytes = Vec::new();
        {
            let cursor = std::io::Cursor::new(&mut bytes);
            let mut archive = zip::ZipWriter::new(cursor);
            archive
                .start_file("../evil.pkl", zip::write::SimpleFileOptions::default())
                .unwrap();
            archive.write_all(b"nope").unwrap();
            archive.finish().unwrap();
        }
        let error = validate_zip_entries(&bytes).unwrap_err().to_string();
        assert!(error.contains("escapes its destination"), "{error}");
    }
}
