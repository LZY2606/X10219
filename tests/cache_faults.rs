//! Cache write-fault injection and on-disk corruption contract tests.
//!
//! These tests never touch the real network or the developer's home
//! directory: persistence goes through an in-memory filesystem and packages
//! come from scripted byte responses. A small companion suite that needs real
//! `fsync`/rename semantics uses per-test temp directories that are deleted
//! when the test finishes.

mod common;

use std::path::{Path, PathBuf};

use pklr::package_cache::{
    self, CacheLookup, CacheReadError, META_FILE_NAME, STAGING_PREFIX, WriteBoundary,
};
use pklr::Evaluator;

use common::{semantic, FakeCapabilities, FakeWorld, Scope, SemanticEvent};

const URL: &str = "https://example.test/pkg@1.0.0.zip";
const EXT: &str = "zip";
const CACHE_ROOT: &str = "/cache";
const FETCH_URL: &str = "https://cdn.example.test/pkg@1.0.0.zip";

fn config_pkl() -> String {
    "amends \"package://example.test/pkg@1.0.0#/Config.pkl\"\n".to_string()
}

fn zip_with_answer(answer: u32) -> Vec<u8> {
    use std::io::Write;
    let mut bytes = Vec::new();
    {
        let mut archive = zip::ZipWriter::new(std::io::Cursor::new(&mut bytes));
        archive
            .start_file(
                "Config.pkl",
                zip::write::SimpleFileOptions::default()
                    .compression_method(zip::CompressionMethod::Stored),
            )
            .unwrap();
        archive
            .write_all(format!("answer = {answer}\n").as_bytes())
            .unwrap();
        archive.finish().unwrap();
    }
    bytes
}

fn entry_dir() -> PathBuf {
    package_cache::entry_dir(Path::new(CACHE_ROOT), URL, EXT)
}

fn seeded_world(served_answer: Option<u32>) -> FakeWorld {
    let world = FakeWorld::new();
    world.write_text("/work/main.pkl", config_pkl());
    if let Some(answer) = served_answer {
        world.serve_bytes(FETCH_URL, zip_with_answer(answer));
    }
    world
}

async fn put_seed(world: &FakeWorld, answer: u32) {
    let mut caps = FakeCapabilities::new(world.clone());
    package_cache::put_entry(
        &mut caps,
        Path::new(CACHE_ROOT),
        URL,
        EXT,
        &zip_with_answer(answer),
    )
    .await
    .unwrap();
}

async fn load(world: &FakeWorld) -> CacheLookup {
    let mut caps = FakeCapabilities::new(world.clone());
    package_cache::load_entry(&mut caps, Path::new(CACHE_ROOT), URL, EXT)
        .await
        .unwrap()
}

fn assert_no_staging(world: &FakeWorld) {
    let left = world.staged_paths();
    assert!(
        left.is_empty(),
        "staged files must be renamed or removed: {left:?}"
    );
}

fn evaluator(world: &FakeWorld, offline: bool) -> Evaluator {
    let mut evaluator = Evaluator::with_capabilities(
        FakeCapabilities::new(world.clone()).with_cache_root(CACHE_ROOT),
    );
    evaluator.set_base_path(Path::new("/work"));
    evaluator.set_package_cache_dir(CACHE_ROOT);
    evaluator.set_offline(offline);
    evaluator.set_http_rewrites(&[format!("https://example.test/={}/", "https://cdn.example.test")]);
    evaluator
}

#[test]
fn each_write_boundary_failure_keeps_the_prior_entry_readable() {
    const CASES: &[(WriteBoundary, Scope, bool)] = &[
        (WriteBoundary::MetadataWrite, Scope::Metadata, false),
        (WriteBoundary::BlobWrite, Scope::Blob, false),
        (WriteBoundary::Flush, Scope::Metadata, false),
        (WriteBoundary::Flush, Scope::Blob, false),
        (WriteBoundary::Rename, Scope::Metadata, false),
        (WriteBoundary::Rename, Scope::Blob, false),
        // Directory sync runs after every rename, but the metadata commit is
        // last: a crash at either dirsync leaves the previous metadata in
        // place, so the old entry remains the hit.
        (WriteBoundary::DirSync, Scope::Blob, false),
        (WriteBoundary::DirSync, Scope::Metadata, false),
    ];
    for (boundary, scope, expect_new) in CASES {
        let world = seeded_world(Some(42));
        pollster::block_on(put_seed(&world, 7));
        world.inject_fault(*boundary, *scope, 1);

        let result = pollster::block_on(async {
            let mut caps = FakeCapabilities::new(world.clone());
            package_cache::put_entry(
                &mut caps,
                Path::new(CACHE_ROOT),
                URL,
                EXT,
                &zip_with_answer(42),
            )
            .await
        });
        if result.is_ok() {
            eprintln!("NO-FAULT case={boundary:?}/{scope:?}");
            panic!("{boundary:?}/{scope:?} must surface the fault");
        }
        assert_no_staging(&world);

        let lookup = pollster::block_on(load(&world));
        if !matches!(&lookup, CacheLookup::Hit(b) if *b == zip_with_answer(if *expect_new {42} else {7})) {
            eprintln!("DBG case={boundary:?}/{scope:?} lookup={lookup:?}");
        }
        match lookup {
            CacheLookup::Hit(bytes) => {
                let expected = if *expect_new { 42 } else { 7 };
                // The blob is a valid zip; compare via the contained answer by
                // validating raw bytes against the seed content.
                assert_eq!(
                    bytes,
                    zip_with_answer(expected),
                    "{boundary:?}/{scope:?}: old entry must still be a valid hit"
                );
            }
            other => panic!("{boundary:?}/{scope:?}: half-written entry read as {other:?}"),
        }
    }
}

#[test]
fn failed_first_write_is_never_a_hit_and_online_fetch_still_succeeds() {
    let world = seeded_world(Some(42));
    // Fail while staging the very first blob; no committed entry exists.
    world.inject_fault(WriteBoundary::BlobWrite, Scope::Blob, 1);

    let value = pollster::block_on(async {
        let mut evaluator = evaluator(&world, false);
        evaluator.eval_file_pub(Path::new("/work/main.pkl")).await
    })
    .unwrap();
    assert_eq!(value.to_json()["answer"], 42);
    assert_no_staging(&world);

    // Offline with the failed cache must be an actionable miss, not a hit
    // containing a half file.
    let error = pollster::block_on(async {
        let mut evaluator = evaluator(&world, true);
        evaluator.eval_file_pub(Path::new("/work/main.pkl")).await
    })
    .unwrap_err()
    .to_string();
    assert!(
        error.contains("offline mode"),
        "half-written blob must not be a cache hit: {error}"
    );
}

#[test]
fn stale_staging_files_are_reclaimed_on_next_startup() {
    let world = seeded_world(Some(42));
    pollster::block_on(put_seed(&world, 9));
    let directory = entry_dir();
    // Simulate crashes that left staged files at several levels.
    world.write_file(
        directory.join(format!("{STAGING_PREFIX}{META_FILE_NAME}-1-1")),
        b"{".to_vec(),
    );
    world.write_file(
        directory.join(format!("{STAGING_PREFIX}deadbeef-1-2")),
        b"partial".to_vec(),
    );
    let orphan_dir = Path::new(CACHE_ROOT).join("packages").join(".staging-orphan");
    world.write_file(orphan_dir.join(format!("{STAGING_PREFIX}x-1-3")), b"x");

    // A fresh evaluator is the "next startup".
    let value = pollster::block_on(async {
        let mut evaluator = evaluator(&world, true);
        evaluator.eval_file_pub(Path::new("/work/main.pkl")).await
    })
    .unwrap();
    assert_eq!(value.to_json()["answer"], 9);

    let swept: Vec<PathBuf> = semantic(&world.events())
        .into_iter()
        .filter_map(|event| match event {
            SemanticEvent::RemoveFile(path) | SemanticEvent::RemoveDirAll(path) => Some(path),
            _ => None,
        })
        .collect();
    assert!(
        swept
            .iter()
            .any(|path| path
                .file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.starts_with(STAGING_PREFIX))),
        "startup sweep must reclaim staged files: {swept:?}"
    );
    assert!(!world.exists(&orphan_dir), "staged directory is reclaimed");
    assert!(
        !world.exists(
            &directory.join(format!("{STAGING_PREFIX}{META_FILE_NAME}-1-1"))
        ),
        "staged metadata is reclaimed"
    );
}

#[test]
fn sweep_can_be_driven_directly_and_touches_nothing_committed() {
    let world = seeded_world(None);
    pollster::block_on(put_seed(&world, 11));
    let directory = entry_dir();
    world.write_file(directory.join(format!("{STAGING_PREFIX}blob-2-9")), b"z");

    let removed = pollster::block_on(async {
        let mut caps = FakeCapabilities::new(world.clone());
        package_cache::sweep_staging(&mut caps, Path::new(CACHE_ROOT))
            .await
            .unwrap()
    });
    assert_eq!(removed.len(), 1);
    assert_eq!(
        removed[0].file_name().and_then(|name| name.to_str()),
        Some(
            directory
                .join(format!("{STAGING_PREFIX}blob-2-9"))
                .file_name()
                .unwrap()
                .to_str()
                .unwrap()
        )
    );
    match pollster::block_on(load(&world)) {
        CacheLookup::Hit(bytes) => assert_eq!(bytes, zip_with_answer(11)),
        other => panic!("committed entry must survive the sweep: {other:?}"),
    }
}

fn write_meta(world: &FakeWorld, json: &str) {
    world.write_text(entry_dir().join(META_FILE_NAME), json);
}

#[test]
fn corrupt_metadata_is_rejected() {
    let world = FakeWorld::new();
    write_meta(&world, "{ not json");
    let error = pollster::block_on(load(&world));
    assert!(
        matches!(error, CacheLookup::Invalid(CacheReadError::CorruptMetadata(_))),
        "got {error:?}"
    );

    // Offline evaluation surfaces the explicit reason.
    world.write_text("/work/main.pkl", config_pkl());
    let err = pollster::block_on(async {
        let mut evaluator = evaluator(&world, true);
        evaluator.eval_file_pub(Path::new("/work/main.pkl")).await
    })
    .unwrap_err()
    .to_string();
    assert!(err.contains("corrupt package cache metadata"), "{err}");
}

#[test]
fn digest_mismatch_is_rejected() {
    let world = FakeWorld::new();
    let blob_name = "a".repeat(64);
    write_meta(
        &world,
        &format!(
            r#"{{"version":1,"url":"{URL}","extension":"{EXT}","digest":"{bad}","blob":"{blob_name}"}}"#,
            bad = "f".repeat(64)
        ),
    );
    world.write_file(entry_dir().join(&blob_name), b"tampered");
    let error = pollster::block_on(load(&world));
    match error {
        CacheLookup::Invalid(CacheReadError::DigestMismatch { expected, actual }) => {
            assert_eq!(expected, "f".repeat(64));
            assert_eq!(actual.len(), 64);
            assert_ne!(actual, expected);
        }
        other => panic!("expected digest mismatch, got {other:?}"),
    }
}

#[test]
fn unsupported_metadata_version_is_rejected() {
    let world = FakeWorld::new();
    let blob_name = "b".repeat(64);
    write_meta(
        &world,
        &format!(
            r#"{{"version":99,"url":"{URL}","extension":"{EXT}","digest":"{d}","blob":"{blob_name}"}}"#,
            d = "1".repeat(64)
        ),
    );
    world.write_file(entry_dir().join(&blob_name), b"x");
    assert!(matches!(
        pollster::block_on(load(&world)),
        CacheLookup::Invalid(CacheReadError::UnsupportedVersion(99))
    ));
}

#[test]
fn metadata_paths_outside_the_cache_are_rejected() {
    for escaped in ["../escape.pkl", "../../etc/passwd", "/etc/passwd", "dir/../../../out"] {
        let world = FakeWorld::new();
        write_meta(
            &world,
            &format!(
                r#"{{"version":1,"url":"{URL}","extension":"{EXT}","digest":"{d}","blob":"{escaped}"}}"#,
                d = "2".repeat(64)
            ),
        );
        let error = pollster::block_on(load(&world));
        assert!(
            matches!(error, CacheLookup::Invalid(CacheReadError::UnsafePath(_))),
            "{escaped} must be rejected, got {error:?}"
        );
    }
}

#[test]
fn online_evaluation_evicts_invalid_entries_then_fetches() {
    let world = seeded_world(Some(42));
    write_meta(&world, "{ broken");
    let value = pollster::block_on(async {
        let mut evaluator = evaluator(&world, false);
        evaluator.eval_file_pub(Path::new("/work/main.pkl")).await
    })
    .unwrap();
    assert_eq!(value.to_json()["answer"], 42);

    // The invalid entry was evicted and replaced by the fetched one.
    match pollster::block_on(load(&world)) {
        CacheLookup::Hit(bytes) => assert_eq!(bytes, zip_with_answer(42)),
        other => panic!("expected repaired cache entry, got {other:?}"),
    }
}

#[test]
fn stored_metadata_round_trip_itself() {
    let world = FakeWorld::new();
    pollster::block_on(put_seed(&world, 5));
    let meta_path = entry_dir().join(META_FILE_NAME);
    let raw = world.read_file(&meta_path).unwrap();
    let json: serde_json::Value = serde_json::from_slice(&raw).unwrap();
    assert_eq!(json["version"], 1);
    assert_eq!(json["url"], URL);
    assert_eq!(json["extension"], EXT);
    assert_eq!(json["blob"], json["digest"]);
}
