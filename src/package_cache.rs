//! On-disk package cache layout and metadata handling.
//!
//! Every cached package is stored under `<cache>/packages/` as two siblings:
//! a content-addressed binary blob (`<hash>.<ext>`) and a JSON metadata file
//! (`<hash>.meta.json`). Metadata records the format version, source URL, and
//! a SHA-256 digest of the blob so corruption and a mismatched blob are both
//! rejected instead of served as a cache hit.

use std::path::{Component, Path, PathBuf};

use sha2::{Digest, Sha256};

/// Metadata format version understood by this build.
pub const CACHE_VERSION: u32 = 1;
/// Only this metadata format is accepted.
pub const SUPPORTED_VERSION: u32 = 1;

/// Metadata persisted next to a cached package blob.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct PackageMetadata {
    pub version: u32,
    pub url: String,
    pub extension: String,
    pub sha256: String,
    /// Blob location relative to the package directory. `None` means the
    /// conventional `<hash>.<ext>` sibling.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub blob: Option<String>,
}

impl PackageMetadata {
    pub fn new(url: &str, extension: &str, bytes: &[u8]) -> Self {
        Self {
            version: CACHE_VERSION,
            url: url.to_string(),
            extension: extension.to_string(),
            sha256: digest_hex(bytes),
            blob: None,
        }
    }

    pub fn to_json(&self) -> Vec<u8> {
        serde_json::to_vec(self).expect("package metadata always serializes")
    }
}

/// Compute the lowercase hex SHA-256 digest of `bytes`.
pub fn digest_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    let digest = hasher.finalize();
    let mut out = String::with_capacity(digest.len() * 2);
    for byte in digest {
        use std::fmt::Write;
        let _ = write!(out, "{byte:02x}");
    }
    out
}

/// The on-disk paths of a cache entry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CachePaths {
    pub package_dir: PathBuf,
    pub blob: PathBuf,
    pub metadata: PathBuf,
}

/// Filesystem-safe entry key derived from the source URL (FNV-1a 64-bit).
pub fn cache_key(url: &str) -> u64 {
    let mut hash = 0xcbf29ce484222325_u64;
    for byte in url.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

pub fn cache_paths(cache_dir: &Path, url: &str, extension: &str) -> CachePaths {
    let package_dir = cache_dir.join("packages");
    let key = format!("{:016x}", cache_key(url));
    CachePaths {
        package_dir: package_dir.clone(),
        blob: package_dir.join(format!("{key}.{extension}")),
        metadata: package_dir.join(format!("{key}.meta.json")),
    }
}

/// Why a cache entry could not be served.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MetadataError {
    /// Metadata file does not exist.
    Missing,
    /// Metadata is not valid JSON or does not match [`PackageMetadata`].
    Corrupt(String),
    /// Metadata advertises a format this build cannot read.
    UnsupportedVersion(u32),
    /// Metadata URL does not match the URL being looked up.
    UrlMismatch,
    /// Metadata extension does not match the lookup extension.
    ExtensionMismatch,
    /// Declared blob path escapes the package directory.
    BlobPathEscapes,
    /// Blob file does not exist.
    BlobMissing,
    /// Blob digest differs from the metadata claim.
    DigestMismatch,
}

impl MetadataError {
    pub fn message(&self) -> String {
        match self {
            MetadataError::Missing => "cache metadata is missing".to_string(),
            MetadataError::Corrupt(reason) => {
                format!("cache metadata is corrupt: {reason}")
            }
            MetadataError::UnsupportedVersion(version) => {
                format!("unsupported cache metadata version: {version}")
            }
            MetadataError::UrlMismatch => "cache metadata URL mismatch".to_string(),
            MetadataError::ExtensionMismatch => "cache metadata extension mismatch".to_string(),
            MetadataError::BlobPathEscapes => {
                "cache metadata points outside the cache directory".to_string()
            }
            MetadataError::BlobMissing => "cached package blob is missing".to_string(),
            MetadataError::DigestMismatch => "cached package digest mismatch".to_string(),
        }
    }
}

/// Parse and fully validate metadata bytes for a lookup of `url`/`extension`.
pub fn parse_metadata(
    raw: &[u8],
    url: &str,
    extension: &str,
) -> Result<PackageMetadata, MetadataError> {
    let metadata: PackageMetadata = serde_json::from_slice(raw)
        .map_err(|error| MetadataError::Corrupt(error.to_string()))?;
    if metadata.version != SUPPORTED_VERSION {
        return Err(MetadataError::UnsupportedVersion(metadata.version));
    }
    if metadata.url != url {
        return Err(MetadataError::UrlMismatch);
    }
    if metadata.extension != extension {
        return Err(MetadataError::ExtensionMismatch);
    }
    if let Some(blob) = &metadata.blob
        && resolve_blob_path_safe(&PathBuf::from("packages"), Path::new(blob)).is_none()
    {
        return Err(MetadataError::BlobPathEscapes);
    }
    Ok(metadata)
}

/// Resolve a metadata-declared blob relative to the cache root, rejecting any
/// path (absolute, prefix-relative, or containing `..`) that escapes it.
pub fn resolve_blob_path_safe(package_dir: &Path, relative: &Path) -> Option<PathBuf> {
    let mut resolved = PathBuf::new();
    for component in relative.components() {
        match component {
            Component::Normal(part) => resolved.push(part),
            Component::CurDir => {}
            Component::RootDir | Component::Prefix(_) | Component::ParentDir => return None,
        }
    }
    if resolved.as_os_str().is_empty() {
        return None;
    }
    Some(package_dir.join(resolved))
}

/// Validate a blob against its parsed metadata.
pub fn validate_blob(metadata: &PackageMetadata, bytes: &[u8]) -> Result<(), MetadataError> {
    if digest_hex(bytes) != metadata.sha256 {
        return Err(MetadataError::DigestMismatch);
    }
    Ok(())
}
