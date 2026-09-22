// The Lex/Parse variant fields are read by miette's Diagnostic derive macro,
// but rustc can't see through the proc-macro expansion.
#![allow(unused_assignments)]

use std::path::PathBuf;

#[cfg(feature = "miette-diagnostics")]
use miette::NamedSource;

pub type Result<T> = std::result::Result<T, Error>;

#[cfg_attr(feature = "miette-diagnostics", derive(miette::Diagnostic))]
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("IO error reading {0}: {1}")]
    Io(PathBuf, #[source] std::io::Error),

    #[error("{message}")]
    #[cfg_attr(feature = "miette-diagnostics", diagnostic())]
    Lex {
        #[cfg(feature = "miette-diagnostics")]
        #[source_code]
        src: NamedSource<String>,
        #[cfg(feature = "miette-diagnostics")]
        #[label("{message}")]
        span: miette::SourceOffset,
        #[cfg(not(feature = "miette-diagnostics"))]
        source_name: String,
        #[cfg(not(feature = "miette-diagnostics"))]
        offset: usize,
        message: String,
    },

    #[error("{message}")]
    #[cfg_attr(feature = "miette-diagnostics", diagnostic())]
    Parse {
        #[cfg(feature = "miette-diagnostics")]
        #[source_code]
        src: NamedSource<String>,
        #[cfg(feature = "miette-diagnostics")]
        #[label("{message}")]
        span: miette::SourceOffset,
        #[cfg(not(feature = "miette-diagnostics"))]
        source_name: String,
        #[cfg(not(feature = "miette-diagnostics"))]
        offset: usize,
        message: String,
    },

    #[error("Eval error: {0}")]
    Eval(String),

    #[error("Import not found: {0}")]
    ImportNotFound(String),

    /// Persistent package cache metadata could not be trusted.
    #[error("invalid package cache for {url}: {reason}")]
    CacheInvalid { url: String, reason: CacheInvalidReason },

    /// Writing to the persistent package cache failed at a durability boundary.
    #[error("package cache write failed for {url}: {reason}")]
    CacheWrite { url: String, reason: String },

    #[error("Unsupported feature: {0}")]
    Unsupported(String),
}

/// Why a persisted package cache entry was rejected.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CacheInvalidReason {
    /// Metadata was absent, truncated, or not valid JSON.
    CorruptMetadata,
    /// Metadata advertised a format this build cannot read.
    UnsupportedVersion(u32),
    /// The blob digest did not match the metadata claim.
    DigestMismatch,
    /// The blob path pointed outside of the cache directory.
    BlobPathEscape,
    /// The metadata belonged to a different URL.
    UrlMismatch,
    /// The blob referenced by metadata was missing.
    MissingBlob,
}

impl std::fmt::Display for CacheInvalidReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::CorruptMetadata => write!(f, "metadata is corrupt"),
            Self::UnsupportedVersion(version) => {
                write!(f, "unsupported cache metadata version {version}")
            }
            Self::DigestMismatch => write!(f, "blob digest mismatch"),
            Self::BlobPathEscape => write!(f, "blob path escapes the cache directory"),
            Self::UrlMismatch => write!(f, "metadata does not belong to the requested url"),
            Self::MissingBlob => write!(f, "cached blob is missing"),
        }
    }
}

impl Error {
    pub fn lex(source_name: &str, source: &str, offset: usize, message: String) -> Self {
        #[cfg(feature = "miette-diagnostics")]
        {
            Self::Lex {
                src: NamedSource::new(source_name, source.to_string()),
                span: miette::SourceOffset::from(offset),
                message,
            }
        }
        #[cfg(not(feature = "miette-diagnostics"))]
        {
            let _ = source;
            Self::Lex {
                source_name: source_name.to_string(),
                offset,
                message,
            }
        }
    }

    pub fn parse(source_name: &str, source: &str, offset: usize, message: String) -> Self {
        #[cfg(feature = "miette-diagnostics")]
        {
            Self::Parse {
                src: NamedSource::new(source_name, source.to_string()),
                span: miette::SourceOffset::from(offset),
                message,
            }
        }
        #[cfg(not(feature = "miette-diagnostics"))]
        {
            let _ = source;
            Self::Parse {
                source_name: source_name.to_string(),
                offset,
                message,
            }
        }
    }

    pub fn source_offset(&self) -> Option<usize> {
        match self {
            #[cfg(feature = "miette-diagnostics")]
            Self::Lex { span, .. } | Self::Parse { span, .. } => Some(span.offset()),
            #[cfg(not(feature = "miette-diagnostics"))]
            Self::Lex { offset, .. } | Self::Parse { offset, .. } => Some(*offset),
            _ => None,
        }
    }
}
