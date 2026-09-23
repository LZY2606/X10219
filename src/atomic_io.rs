//! Atomic file writes for the package cache, decomposed into individually
//! fault-injectable steps.
//!
//! A cache write must never leave a half-written file at its final path:
//! bytes go to a recognizable temporary file in the same directory, are
//! flushed to disk, and are then renamed into place. The parent directory is
//! synced afterwards so the rename itself is durable. Each boundary
//! (create, write, flush, rename, directory sync) goes through
//! [`AtomicWriteBackend`] so tests can inject failures without changing the
//! production code path.

use std::io;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

/// A file being written as part of an atomic write.
#[doc(hidden)]
pub trait AtomicWriteFile {
    fn write_all(&mut self, bytes: &[u8]) -> io::Result<()>;
    fn sync_all(&mut self) -> io::Result<()>;
}

/// Filesystem primitives behind [`write_atomic_with`].
///
/// This is a restricted dependency-injection seam extracted for fault
/// injection tests; it is not a public stability guarantee.
#[doc(hidden)]
pub trait AtomicWriteBackend {
    fn create_temp_file(&mut self, path: &Path) -> io::Result<Box<dyn AtomicWriteFile>>;
    fn rename(&mut self, from: &Path, to: &Path) -> io::Result<()>;
    fn remove_file(&mut self, path: &Path) -> io::Result<()>;
    fn sync_dir(&mut self, path: &Path) -> io::Result<()>;
    fn exists(&mut self, path: &Path) -> bool;
}

/// Production backend talking to the real filesystem.
#[doc(hidden)]
#[derive(Debug, Default)]
pub struct FsAtomicWriteBackend;

struct FsAtomicWriteFile(std::fs::File);

impl AtomicWriteFile for FsAtomicWriteFile {
    fn write_all(&mut self, bytes: &[u8]) -> io::Result<()> {
        use std::io::Write;
        self.0.write_all(bytes)
    }

    fn sync_all(&mut self) -> io::Result<()> {
        self.0.sync_all()
    }
}

impl AtomicWriteBackend for FsAtomicWriteBackend {
    fn create_temp_file(&mut self, path: &Path) -> io::Result<Box<dyn AtomicWriteFile>> {
        let file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)?;
        Ok(Box::new(FsAtomicWriteFile(file)))
    }

    fn rename(&mut self, from: &Path, to: &Path) -> io::Result<()> {
        match std::fs::rename(from, to) {
            Ok(()) => Ok(()),
            // Windows rename does not replace an existing file. There is a
            // brief non-atomic gap on that platform, but the destination must
            // not retain stale bytes while its metadata sidecar is updated.
            Err(_) if to.exists() => {
                std::fs::remove_file(to).and_then(|()| std::fs::rename(from, to))
            }
            Err(error) => Err(error),
        }
    }

    fn remove_file(&mut self, path: &Path) -> io::Result<()> {
        std::fs::remove_file(path)
    }

    fn sync_dir(&mut self, path: &Path) -> io::Result<()> {
        sync_dir(path)
    }

    fn exists(&mut self, path: &Path) -> bool {
        path.exists()
    }
}

#[cfg(unix)]
fn sync_dir(path: &Path) -> io::Result<()> {
    std::fs::File::open(path)?.sync_all()
}

#[cfg(not(unix))]
fn sync_dir(path: &Path) -> io::Result<()> {
    // Opening a directory requires platform-specific flags outside unix; a
    // best-effort attempt that ignores "unsupported" errors keeps the write
    // path portable without weakening durability where it is supported.
    match std::fs::File::open(path).and_then(|dir| dir.sync_all()) {
        Ok(()) => Ok(()),
        Err(error)
            if matches!(
                error.kind(),
                io::ErrorKind::Unsupported | io::ErrorKind::InvalidInput
            ) =>
        {
            Ok(())
        }
        Err(error) => Err(error),
    }
}

static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

/// The marker that makes an interrupted atomic write recognizable later.
const TEMP_MARKER: &str = ".tmp-";

/// Does `file_name` look like a temporary file left behind by
/// [`write_atomic_with`] (`.<name>.tmp-<pid>-<counter>`)?
#[doc(hidden)]
pub fn is_atomic_temp_file(file_name: &str) -> bool {
    let Some(rest) = file_name.strip_prefix('.') else {
        return false;
    };
    let Some((_, suffix)) = rest.split_once(TEMP_MARKER) else {
        return false;
    };
    // The suffix is `<pid>-<counter>`; both parts are decimal.
    let mut parts = suffix.split('-');
    let pid = parts.next().unwrap_or_default();
    let counter = parts.next().unwrap_or_default();
    parts.next().is_none()
        && !pid.is_empty()
        && !counter.is_empty()
        && pid.bytes().all(|b| b.is_ascii_digit())
        && counter.bytes().all(|b| b.is_ascii_digit())
}

/// Remove recognizable stale atomic-write temp files directly inside `dir`.
///
/// Called when a cache directory is opened so temp files orphaned by an
/// interrupted process are reclaimed on the next startup. Only files
/// matching the [`is_atomic_temp_file`] pattern are touched. Returns the
/// number of files removed.
#[doc(hidden)]
pub fn cleanup_stale_temp_files(dir: &Path) -> io::Result<usize> {
    let mut removed = 0;
    let entries = match std::fs::read_dir(dir) {
        Ok(entries) => entries,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(0),
        Err(error) => return Err(error),
    };
    for entry in entries {
        let entry = entry?;
        let is_stale_temp = entry
            .file_name()
            .to_str()
            .is_some_and(is_atomic_temp_file);
        if is_stale_temp && entry.file_type().is_ok_and(|ty| ty.is_file()) {
            std::fs::remove_file(entry.path())?;
            removed += 1;
        }
    }
    Ok(removed)
}

/// Write `bytes` to `path` atomically using `backend` for every IO step.
#[doc(hidden)]
pub fn write_atomic_with(
    backend: &mut dyn AtomicWriteBackend,
    path: &Path,
    bytes: &[u8],
) -> io::Result<()> {
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("cache");
    let temp_path = path.with_file_name(format!(
        ".{file_name}{TEMP_MARKER}{}-{}",
        std::process::id(),
        TEMP_COUNTER.fetch_add(1, Ordering::Relaxed)
    ));
    let mut file = backend.create_temp_file(&temp_path)?;
    let write_result = file
        .write_all(bytes)
        .and_then(|()| file.sync_all());
    drop(file);
    if let Err(error) = write_result {
        let _ = backend.remove_file(&temp_path);
        return Err(error);
    }
    if let Err(error) = backend.rename(&temp_path, path) {
        let _ = backend.remove_file(&temp_path);
        return Err(error);
    }
    // The rename is durable only once the containing directory is synced.
    // The new bytes are already visible at `path`, so a directory-sync
    // failure is reported but does not roll the file back.
    if let Some(parent) = path.parent() {
        backend.sync_dir(parent)?;
    }
    Ok(())
}

/// Write `bytes` to `path` atomically on the real filesystem.
pub(crate) fn write_atomic(path: &Path, bytes: &[u8]) -> io::Result<()> {
    write_atomic_with(&mut FsAtomicWriteBackend, path, bytes)
}

#[cfg(test)]
mod tests {
    use super::{cleanup_stale_temp_files, is_atomic_temp_file, write_atomic};

    #[test]
    fn temp_file_pattern_is_recognized() {
        assert!(is_atomic_temp_file(".abc.pkl.tmp-123-0"));
        assert!(is_atomic_temp_file(".cache.tmp-1-42"));
        assert!(!is_atomic_temp_file("abc.pkl"));
        assert!(!is_atomic_temp_file(".abc.pkl"));
        assert!(!is_atomic_temp_file("abc.tmp-1-2"));
        assert!(!is_atomic_temp_file(".abc.tmp-1"));
        assert!(!is_atomic_temp_file(".abc.tmp-1-2-3"));
        assert!(!is_atomic_temp_file(".abc.tmp-pid-0"));
    }

    #[test]
    fn cleanup_removes_only_stale_temp_files() {
        let dir = std::env::temp_dir().join(format!(
            "pklr-atomic-io-test-{}-{}",
            std::process::id(),
            1
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join(".data.pkl.tmp-99999-0"), b"partial").unwrap();
        std::fs::write(dir.join("data.pkl"), b"complete").unwrap();
        std::fs::write(dir.join(".data.pkl"), b"hidden-but-not-temp").unwrap();

        let removed = cleanup_stale_temp_files(&dir).unwrap();

        assert_eq!(removed, 1);
        assert!(!dir.join(".data.pkl.tmp-99999-0").exists());
        assert_eq!(std::fs::read(dir.join("data.pkl")).unwrap(), b"complete");
        assert!(dir.join(".data.pkl").exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn write_atomic_round_trips_and_leaves_no_temp_files() {
        let dir = std::env::temp_dir().join(format!(
            "pklr-atomic-io-test-{}-{}",
            std::process::id(),
            2
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("entry.bin");

        write_atomic(&path, b"one").unwrap();
        write_atomic(&path, b"two").unwrap();

        assert_eq!(std::fs::read(&path).unwrap(), b"two");
        let leftovers: Vec<_> = std::fs::read_dir(&dir)
            .unwrap()
            .filter_map(|entry| entry.ok())
            .filter(|entry| {
                entry
                    .file_name()
                    .to_str()
                    .is_some_and(is_atomic_temp_file)
            })
            .collect();
        assert!(leftovers.is_empty(), "leftover temp files: {leftovers:?}");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
