//! Cache write fault injection and corruption contract tests.
//!
//! Every test uses the in-memory `FakeHost`, so no real temp directory, home
//! directory, network, or user cache is ever touched.

mod common;

use std::path::{Path, PathBuf};

use common::{
    CacheFault, FakeHost, Mode, blob_file, legacy_blob, legacy_url_sidecar, meta_file, meta_json,
};
use pklr::package_cache::FileKind;

const CACHE: &str = "/cache";
const ENTRY: &str = "/work/main.pkl";
const OLD_URL: &str = "https://example.com/packages/old@1.pkl";
const NEW_URL: &str = "https://example.com/packages/new@1.pkl";
const OLD_SOURCE: &str = "value = \"old\"\n";
const NEW_SOURCE: &str = "value = \"new\"\n";

fn entry_path() -> PathBuf {
    PathBuf::from(ENTRY)
}

/// Entry module imports two direct packages: an already-cached one and a fresh
/// one whose cache write is about to fail.
fn entry_source() -> String {
    r#"import "package://example.com/packages/old@1#/Old.pkl" as Old
import "package://example.com/packages/new@1#/New.pkl" as New
old = Old.value
new = New.value
"#
    .to_string()
}

fn seed_old_cache(host: &FakeHost) -> Vec<u8> {
    let cache = Path::new(CACHE);
    let old_bytes = OLD_SOURCE.to_string().into_bytes();
    host.put_file(
        blob_file(cache, OLD_URL, "pkl", &old_bytes),
        old_bytes.clone(),
    );
    host.put_file(meta_file(cache, OLD_URL), meta_json(OLD_URL, "pkl", &old_bytes));
    old_bytes
}

/// Clone the virtual filesystem into a brand new host (simulating restart into
/// the same cache directory), dropping all in-process caches and events.
fn restart_into(host: &FakeHost, mode: Mode) -> FakeHost {
    let next = FakeHost::new(mode);
    for (path, bytes) in host.state.lock().unwrap().files.clone() {
        next.put_file(path, bytes);
    }
    next
}

fn assert_no_staging_files(host: &FakeHost, case: &str) {
    let leftover = host
        .file_names()
        .into_iter()
        .filter(|name| {
            let file = Path::new(name)
                .file_name()
                .and_then(|value| value.to_str())
                .unwrap_or("");
            file.starts_with('.') && file.ends_with(pklr::package_cache::TEMP_SUFFIX)
        })
        .collect::<Vec<_>>();
    assert!(
        leftover.is_empty(),
        "{case}: staging files survived a restart sweep: {leftover:?}"
    );
}

#[test]
fn failed_commit_boundaries_keep_old_cache_usable() {
    let cases = [
        ("create", Some(FileKind::Blob)),
        ("create", Some(FileKind::Metadata)),
        ("flush", Some(FileKind::Blob)),
        ("flush", Some(FileKind::Metadata)),
        ("rename", Some(FileKind::Blob)),
        ("rename", Some(FileKind::Metadata)),
        ("sync", Some(FileKind::Blob)),
        ("sync", Some(FileKind::Metadata)),
        ("mkdir", None),
    ];

    for (boundary, kind) in cases {
        let case = format!("boundary {boundary}");
        let host = FakeHost::new(Mode::Sync);
        host.put_text(entry_path(), entry_source());
        seed_old_cache(&host);
        host.put_http_text(NEW_URL, NEW_SOURCE);
        host.install_fault(CacheFault {
            boundary,
            kind,
            skip: 0,
        });

        // Online evaluation succeeds from the fetched bytes despite the
        // best-effort cache write failing.
        let value = common::run_sync(
            host.clone(),
            |evaluator| {
                evaluator.set_package_cache_dir(CACHE);
            },
            &entry_path(),
        )
        .unwrap_or_else(|error| panic!("{case}: online eval failed: {error}"));
        assert_eq!(value.to_json()["old"], "old", "{case}");
        assert_eq!(value.to_json()["new"], "new", "{case}");

        // Simulated restart, offline: the old package must still be a hit.
        let offline_host = restart_into(&host, Mode::Sync);
        let result = common::run_sync(
            offline_host,
            |evaluator| {
                evaluator.set_package_cache_dir(CACHE);
                evaluator.set_offline(true);
            },
            &entry_path(),
        );
        assert!(
            result.is_err(),
            "{case}: half-written new package was treated as an offline hit"
        );

        // Old package alone must still read after the failed commit.
        let old_only = PathBuf::from("/work/old_only.pkl");
        let host2 = restart_into(&host, Mode::Sync);
        host2.put_text(
            &old_only,
            "import \"package://example.com/packages/old@1#/Old.pkl\" as Old\nvalue = Old.value\n",
        );
        let old_value = common::run_sync(
            host2.clone(),
            |evaluator| {
                evaluator.set_package_cache_dir(CACHE);
                evaluator.set_offline(true);
            },
            &old_only,
        )
        .unwrap_or_else(|error| panic!("{case}: old cache became unusable: {error}"));
        assert_eq!(old_value.to_json()["value"], "old", "{case}");
        assert_no_staging_files(&host2, &case);
    }
}

#[test]
fn half_written_blob_without_metadata_is_never_a_hit() {
    let host = FakeHost::new(Mode::Sync);
    seed_old_cache(&host);
    // Simulate a crash after the new blob was renamed but before metadata was
    // committed (plus the metadata staging temp left behind).
    let partial = NEW_SOURCE.to_string().into_bytes();
    host.put_file(
        blob_file(Path::new(CACHE), NEW_URL, "pkl", &partial),
        partial,
    );
    host.put_text(
        Path::new(&format!(
            "{CACHE}/packages/.{:016x}.meta.json.tmp-pklrtmp",
            pklr::package_cache::url_hash(NEW_URL)
        )),
        "partial",
    );
    let path = PathBuf::from("/work/new_only.pkl");
    host.put_text(
        &path,
        "import \"package://example.com/packages/new@1#/New.pkl\" as New\nvalue = New.value\n",
    );

    // Offline: metadata is absent, so this must be a miss even though a blob
    // with the right content exists.
    let error = common::run_sync(
        host.clone(),
        |evaluator| {
            evaluator.set_package_cache_dir(CACHE);
            evaluator.set_offline(true);
        },
        &path,
    )
    .expect_err("orphan blob must not be treated as a hit");
    assert!(error.to_string().contains("offline"), "{error}");

    // The startup sweep removed the leftover staging temp.
    assert_no_staging_files(&host, "orphan blob");
}

#[test]
fn stale_temp_files_are_swept_on_next_startup() {
    let stale_blob = PathBuf::from(format!("{CACHE}/blobs/.deadbeef.pkl.tmp-pklrtmp"));
    let stale_meta =
        PathBuf::from(format!("{CACHE}/packages/.abc.meta.json.tmp-pklrtmp"));
    let unrelated = PathBuf::from(format!("{CACHE}/packages/keep.txt"));
    let host = FakeHost::new(Mode::Sync);
    host.put_file(&stale_blob, b"stale blob");
    host.put_file(&stale_meta, b"stale meta");
    host.put_file(&unrelated, b"keep me");
    let path = PathBuf::from("/work/old_only.pkl");
    host.put_text(
        &path,
        "import \"package://example.com/packages/old@1#/Old.pkl\" as Old\nvalue = Old.value\n",
    );
    seed_old_cache(&host);

    let value = common::run_sync(
        host.clone(),
        |evaluator| {
            evaluator.set_package_cache_dir(CACHE);
            evaluator.set_offline(true);
        },
        &path,
    )
    .expect("sweep must not disturb valid entries");
    assert_eq!(value.to_json()["value"], "old");
    assert!(!host.read_file(&stale_blob).is_some(), "stale blob remains");
    assert!(!host.read_file(&stale_meta).is_some(), "stale meta remains");
    assert!(host.read_file(&unrelated).is_some(), "unrelated file removed");
}

#[test]
fn failed_preload_does_not_publish_half_files() {
    for (boundary, kind) in [
        ("create", Some(FileKind::Blob)),
        ("flush", Some(FileKind::Blob)),
        ("rename", Some(FileKind::Blob)),
        ("sync", Some(FileKind::Blob)),
        ("create", Some(FileKind::Metadata)),
        ("flush", Some(FileKind::Metadata)),
        ("rename", Some(FileKind::Metadata)),
        ("sync", Some(FileKind::Metadata)),
        ("mkdir", None),
    ] {
        let case = format!("preload {boundary}");
        let host = FakeHost::new(Mode::Sync);
        host.install_fault(CacheFault {
            boundary,
            kind,
            skip: 0,
        });
        let bytes = NEW_SOURCE.to_string().into_bytes();

        let mut evaluator = pklr::Evaluator::with_capabilities(host.clone());
        evaluator.set_package_cache_dir(CACHE);
        let error = evaluator
            .preload_package(NEW_URL, "pkl", &bytes)
            .expect_err("injected preload failure must surface");
        assert!(
            !error.to_string().is_empty(),
            "{case}: empty preload error"
        );

        // No final blob/metadata names exist for the failed preload.
        let final_names = host
            .file_names()
            .into_iter()
            .filter(|name| {
                let file = Path::new(name)
                    .file_name()
                    .and_then(|value| value.to_str())
                    .unwrap_or("");
                !file.starts_with('.')
            })
            .collect::<Vec<_>>();
        assert!(
            final_names
                .iter()
                .all(|name| name.ends_with(".pklrtmp") == false),
            "{case}: staged file leaked under a final name: {final_names:?}"
        );

        // A new evaluator reusing the filesystem sweeps and succeeds on retry.
        let retry_host = restart_into(&host, Mode::Sync);
        let mut retry = pklr::Evaluator::with_capabilities(retry_host.clone());
        retry.set_package_cache_dir(CACHE);
        retry
            .preload_package(NEW_URL, "pkl", &bytes)
            .unwrap_or_else(|error| panic!("{case}: retry preload failed: {error}"));
        assert_no_staging_files(&retry_host, &case);
    }
}

fn single_package_eval(host: FakeHost) -> pklr::Result<pklr::Value> {
    let path = PathBuf::from("/work/new_only.pkl");
    host.put_text(
        &path,
        "import \"package://example.com/packages/new@1#/New.pkl\" as New\nvalue = New.value\n",
    );
    common::run_sync(
        host,
        |evaluator| {
            evaluator.set_package_cache_dir(CACHE);
            evaluator.set_offline(true);
        },
        &path,
    )
}

#[test]
fn corrupt_metadata_is_rejected() {
    let host = FakeHost::new(Mode::Sync);
    host.put_file(meta_file(Path::new(CACHE), NEW_URL), b"{not json".to_vec());
    let error = single_package_eval(host).expect_err("corrupt metadata rejected");
    assert!(
        error.to_string().contains("corrupt package cache metadata"),
        "{error}"
    );
}

#[test]
fn digest_mismatch_is_rejected() {
    let host = FakeHost::new(Mode::Sync);
    let actual_bytes = NEW_SOURCE.to_string().into_bytes();
    let wrong_bytes = b"value = \"tampered\"\n".to_vec();
    host.put_file(
        blob_file(Path::new(CACHE), NEW_URL, "pkl", &actual_bytes),
        wrong_bytes,
    );
    host.put_file(
        meta_file(Path::new(CACHE), NEW_URL),
        meta_json(NEW_URL, "pkl", &actual_bytes),
    );
    let error = single_package_eval(host).expect_err("digest mismatch rejected");
    assert!(
        error.to_string().contains("package cache digest mismatch"),
        "{error}"
    );
}

#[test]
fn unsupported_metadata_version_is_rejected() {
    let host = FakeHost::new(Mode::Sync);
    let bytes = NEW_SOURCE.to_string().into_bytes();
    host.put_file(
        blob_file(Path::new(CACHE), NEW_URL, "pkl", &bytes),
        bytes.clone(),
    );
    let mut metadata: serde_json::Value =
        serde_json::from_str(&meta_json(NEW_URL, "pkl", &bytes)).unwrap();
    metadata["version"] = serde_json::json!(999);
    host.put_file(
        meta_file(Path::new(CACHE), NEW_URL),
        serde_json::to_vec(&metadata).unwrap(),
    );
    let error = single_package_eval(host).expect_err("future version rejected");
    assert!(
        error
            .to_string()
            .contains("unsupported package cache metadata version"),
        "{error}"
    );
}

#[test]
fn metadata_blob_path_outside_cache_is_rejected() {
    let escapes = [
        "../../../etc/passwd",
        "/etc/passwd",
        "../escape.pkl",
        "blobs/../../out.pkl",
    ];
    for escape in escapes {
        let host = FakeHost::new(Mode::Sync);
        let bytes = NEW_SOURCE.to_string().into_bytes();
        let digest = common::digest(&bytes);
        let metadata = format!(
            "{{\"version\":1,\"url\":\"{NEW_URL}\",\"extension\":\"pkl\",\
             \"sha256\":\"{digest}\",\"blob\":\"{escape}\"}}"
        );
        host.put_file(
            meta_file(Path::new(CACHE), NEW_URL),
            metadata.into_bytes(),
        );
        let error = single_package_eval(host).expect_err(&format!(
            "escape path {escape:?} must be rejected"
        ));
        assert!(
            error.to_string().contains("escapes the cache directory"),
            "escape {escape:?}: {error}"
        );
    }
}

#[test]
fn pre_v1_legacy_cache_layout_is_still_readable() {
    let host = FakeHost::new(Mode::Sync);
    let bytes = NEW_SOURCE.to_string().into_bytes();
    host.put_file(
        legacy_blob(Path::new(CACHE), NEW_URL, "pkl"),
        bytes.clone(),
    );
    host.put_file(
        legacy_url_sidecar(Path::new(CACHE), NEW_URL, "pkl"),
        NEW_URL.to_string().into_bytes(),
    );
    let value = single_package_eval(host).expect("legacy cache still loads");
    assert_eq!(value.to_json()["value"], "new");
}

#[test]
fn legacy_sidecar_url_mismatch_is_not_a_hit() {
    let host = FakeHost::new(Mode::Sync);
    let bytes = NEW_SOURCE.to_string().into_bytes();
    host.put_file(
        legacy_blob(Path::new(CACHE), NEW_URL, "pkl"),
        bytes.clone(),
    );
    host.put_file(
        legacy_url_sidecar(Path::new(CACHE), NEW_URL, "pkl"),
        b"https://example.com/different@1.pkl".to_vec(),
    );
    let error = single_package_eval(host).expect_err("sidecar mismatch is a miss");
    assert!(error.to_string().contains("offline"), "{error}");
}
