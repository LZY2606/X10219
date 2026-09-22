#[path = "contract_support/mod.rs"]
mod support;

use std::collections::BTreeMap;

use support::model::{predict, Case, GateOp, Location, Node};
use support::{
    edge, http_node, local_node, materialize, pkg_direct_node, prop_gate, prop_int, prop_ref,
    run_async, run_sync, FaultPolicy,
};

/// Hand-built, high-signal cases covering every modelled feature. The random
/// generator below exercises combinations; these pin the feature semantics.
fn handbuilt_cases() -> Vec<Case> {
    let mut out = Vec::new();

    // Local import chain: main -> dep -> leaf.
    {
        let mut main = local_node("/virt/proj/main.pkl");
        main.imports.push(edge(1, "dep.pkl", "Dep"));
        main.props.push(prop_ref("answer", "Dep", "answer"));
        let mut dep = local_node("/virt/proj/dep.pkl");
        dep.imports.push(edge(2, "leaf.pkl", "Leaf"));
        dep.props.push(prop_ref("answer", "Leaf", "value"));
        let mut leaf = local_node("/virt/proj/leaf.pkl");
        leaf.props.push(prop_int("value", 7));
        out.push(case("local_chain", vec![main, dep, leaf], 0));
    }

    // Amend merge with override and inherited key.
    {
        let mut base = local_node("/virt/proj/base.pkl");
        base.props.push(prop_int("a", 1));
        base.props.push(prop_int("b", 2));
        let mut main = local_node("/virt/proj/main.pkl");
        main.amends = Some(edge(1, "base.pkl", "Base"));
        main.props.push(prop_int("b", 20));
        main.props.push(prop_int("c", 3));
        out.push(case("amend_merge", vec![main, base], 0));
    }

    // Two-level amend chain.
    {
        let mut root = local_node("/virt/proj/root.pkl");
        root.props.push(prop_int("deep", 1));
        let mut base = local_node("/virt/proj/base.pkl");
        base.amends = Some(edge(2, "root.pkl", "Root"));
        base.props.push(prop_int("mid", 2));
        let mut main = local_node("/virt/proj/main.pkl");
        main.amends = Some(edge(1, "base.pkl", "Base"));
        main.props.push(prop_int("top", 3));
        out.push(case("amend_chain", vec![main, base, root], 0));
    }

    // HTTP import served through a longest-prefix rewrite.
    {
        let mut remote = http_node("http://fake.test/Remote.pkl");
        remote.props.push(prop_int("value", 99));
        let mut main = local_node("/virt/proj/main.pkl");
        main.imports.push(edge(1, "https://example.com/Remote.pkl", "Remote"));
        main.props.push(prop_ref("answer", "Remote", "value"));
        let mut c = case("http_rewrite", vec![main, remote], 0);
        c.rewrites = vec![
            ("https://example.com/short/".to_string(), "http://other/".to_string()),
            ("https://example.com/".to_string(), "http://fake.test/".to_string()),
        ];
        out.push(c);
    }

    // Relative HTTP import resolved against the importing module's directory.
    {
        let mut sibling = http_node("http://fake.test/Sibling.pkl");
        sibling.props.push(prop_int("value", 5));
        let mut remote = http_node("http://fake.test/cfg/Remote.pkl");
        remote.imports.push(edge(2, "../Sibling.pkl", "Sibling"));
        remote.props.push(prop_ref("value", "Sibling", "value"));
        let mut main = local_node("/virt/proj/main.pkl");
        main.imports.push(edge(1, "http://fake.test/cfg/Remote.pkl", "Remote"));
        main.props.push(prop_ref("answer", "Remote", "value"));
        out.push(case("http_relative", vec![main, remote, sibling], 0));
    }

    // Direct package, offline, served from a preloaded cache entry.
    {
        let pkg_uri =
            "package://pkg.pkl-lang.org/github.com/acme/pkg@1.0.0#/P.pkl";
        let download_url = "https://github.com/acme/pkg/releases/download/1.0.0/P.pkl";
        let mut pkg = pkg_direct_node(pkg_uri, download_url);
        pkg.props.push(prop_int("value", 42));
        let mut main = local_node("/virt/proj/main.pkl");
        main.imports.push(edge(1, pkg_uri, "P"));
        main.props.push(prop_ref("answer", "P", "value"));
        let mut c = case("package_preload_offline", vec![main, pkg], 0);
        c.offline = true;
        if let Location::PkgDirect { download_url, .. } = c.nodes[1].loc.as_ref().unwrap() {
            let source = support::render_node(&c.nodes, 1);
            let download_url = download_url.clone();
            c.preloads.push(support::model::Preload {
                url: download_url.clone(),
                extension: "pkl".to_string(),
                bytes: source.into_bytes(),
            });
        }
        out.push(c);
    }

    // Direct package, online, fetched and written to cache (no preload).
    {
        let pkg_uri =
            "package://pkg.pkl-lang.org/github.com/acme/other@2.0.0#/Q.pkl";
        let download_url = "https://github.com/acme/other/releases/download/2.0.0/Q.pkl";
        let mut pkg = pkg_direct_node(pkg_uri, download_url);
        pkg.props.push(prop_int("value", 8));
        let mut main = local_node("/virt/proj/main.pkl");
        main.imports.push(edge(1, pkg_uri, "Q"));
        main.props.push(prop_ref("answer", "Q", "value"));
        out.push(case("package_online_fetch", vec![main, pkg], 0));
    }

    // Offline package miss (no preload, offline).
    {
        let pkg_uri =
            "package://pkg.pkl-lang.org/github.com/acme/missing@9.0.0#/M.pkl";
        let download_url = "https://github.com/acme/missing/releases/download/9.0.0/M.pkl";
        let mut pkg = pkg_direct_node(pkg_uri, download_url);
        pkg.props.push(prop_int("value", 1));
        let mut main = local_node("/virt/proj/main.pkl");
        main.imports.push(edge(1, pkg_uri, "M"));
        main.props.push(prop_ref("answer", "M", "value"));
        let mut c = case("package_offline_miss", vec![main, pkg], 0);
        c.offline = true;
        out.push(c);
    }

    // Short-circuit AND with a false left: right key never read.
    {
        let mut main = local_node("/virt/proj/main.pkl");
        main.props.push(prop_gate("flag", GateOp::And, "LEFT", "RIGHT"));
        let mut c = case("short_circuit_and", vec![main], 0);
        c.env.insert("LEFT".to_string(), String::new());
        out.push(c);
    }

    // Short-circuit OR with a true left.
    {
        let mut main = local_node("/virt/proj/main.pkl");
        main.props.push(prop_gate("flag", GateOp::Or, "LEFT", "RIGHT"));
        let mut c = case("short_circuit_or", vec![main], 0);
        c.env.insert("LEFT".to_string(), "yes".to_string());
        out.push(c);
    }

    // Both branches evaluated when the left is non-deciding.
    {
        let mut main = local_node("/virt/proj/main.pkl");
        main.props.push(prop_gate("flag", GateOp::And, "LEFT", "RIGHT"));
        let mut c = case("both_branches_and", vec![main], 0);
        c.env.insert("LEFT".to_string(), "yes".to_string());
        out.push(c);
    }

    out
}

fn case(name: &str, nodes: Vec<Node>, entry: usize) -> Case {
    Case {
        name: name.to_string(),
        nodes,
        entry,
        rewrites: vec![],
        offline: false,
        env: BTreeMap::new(),
        preloads: vec![],
        prestate: vec![],
    }
}

fn assert_case_matches_model(case: &Case) {
    let model_mat = materialize(case);
    let files: BTreeMap<String, Vec<u8>> = model_mat
        .fs
        .files
        .lock()
        .unwrap()
        .iter()
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();
    let prediction = predict(
        case,
        files,
        model_mat.http_sources.clone(),
        model_mat.http_bytes.clone(),
    );

    // Each API gets a freshly materialized world: preloads and fetches mutate
    // the virtual filesystem, so runs must not share cache state.
    let sync_mat = materialize(case);
    let async_mat = materialize(case);
    let (sync_result, _) = run_sync(case, &sync_mat, FaultPolicy::default());
    let (async_result, _) =
        pollster::FutureExt::block_on(run_async(case, &async_mat, FaultPolicy::default()));

    // Sync and async logs differ only by scheduling markers.
    let sync_raw: Vec<String> = sync_result
        .raw
        .iter()
        .map(|(event, seq, mode)| format!("{}|{seq}|{mode}", event.label()))
        .collect();
    let async_raw: Vec<String> = async_result
        .raw
        .iter()
        .map(|(event, seq, mode)| format!("{}|{seq}|{mode}", event.label()))
        .collect();
    assert_ne!(
        sync_raw.iter().filter(|line| line.ends_with("|sync")).count(),
        0
    );
    assert!(
        async_raw.iter().any(|line| line.starts_with("sched|")),
        "async run must contain scheduling markers: {async_raw:?}"
    );
    assert!(
        sync_raw.iter().all(|line| !line.starts_with("sched|")),
        "sync run must not contain scheduling markers: {sync_raw:?}"
    );

    eprintln!(
        "CASE {} outcome model={:?} sync={:?}",
        case.name, prediction.outcome, sync_result.error_category
    );
    assert_eq!(
        sync_result.normalized, prediction.trace,
        "case {}: sync/model trace mismatch",
        case.name
    );
    assert_eq!(
        async_result.normalized, prediction.trace,
        "case {}: async/model trace mismatch",
        case.name
    );
    assert_eq!(
        sync_result.normalized, async_result.normalized,
        "case {}: sync vs async normalized traces differ",
        case.name
    );

    match &prediction.outcome {
        support::model::Outcome::Ok(expected_json) => {
            assert_eq!(
                sync_result.json.as_deref(),
                Some(expected_json.as_str()),
                "case {}: sync json mismatch",
                case.name
            );
            assert_eq!(
                async_result.json.as_deref(),
                Some(expected_json.as_str()),
                "case {}: async json mismatch",
                case.name
            );
            assert_eq!(
                sync_result.env_reads, prediction.env_reads,
                "case {}: sync env reads mismatch",
                case.name
            );
            assert_eq!(
                async_result.env_reads, prediction.env_reads,
                "case {}: async env reads mismatch",
                case.name
            );
        }
        support::model::Outcome::Err(expected_category) => {
            assert_eq!(
                sync_result.error_category.as_deref(),
                Some(expected_category.as_str()),
                "case {}: sync error category mismatch",
                case.name
            );
            assert_eq!(
                async_result.error_category.as_deref(),
                Some(expected_category.as_str()),
                "case {}: async error category mismatch",
                case.name
            );
        }
    }
}

#[test]
fn handbuilt_contract_cases_match_model() {
    for case in handbuilt_cases() {
        assert_case_matches_model(&case);
    }
}

// ---------------------------------------------------------------------------
// Cycle handling
// ---------------------------------------------------------------------------

#[test]
fn import_cycle_is_handled_with_sync_async_parity() {
    // a imports b; b imports a. The second back-reference resolves to the
    // in-flight placeholder, so referencing a field errors.
    let mut a = local_node("/virt/proj/a.pkl");
    a.imports.push(edge(1, "b.pkl", "B"));
    a.props.push(prop_ref("v", "B", "v"));
    let mut b = local_node("/virt/proj/b.pkl");
    b.imports.push(edge(0, "a.pkl", "A"));
    b.props.push(prop_int("v", 1));
    let case = case("import_cycle", vec![a, b], 0);

    let sync_mat = materialize(&case);
    let async_mat = materialize(&case);
    let (sync_result, _) = run_sync(&case, &sync_mat, FaultPolicy::default());
    let (async_result, _) =
        pollster::FutureExt::block_on(run_async(&case, &async_mat, FaultPolicy::default()));

    assert_eq!(sync_result.error_category.as_deref(), Some("field-not-found"));
    assert_eq!(
        async_result.error_category.as_deref(),
        Some("field-not-found")
    );
    assert_eq!(sync_result.normalized, async_result.normalized);
}

#[test]
fn mutual_amend_cycle_is_bounded_with_parity() {
    // Mutual amends walk until the in-flight guard turns a frame into an empty
    // module; evaluation still terminates with identical results both modes.
    let mut a = local_node("/virt/proj/a.pkl");
    a.amends = Some(edge(1, "b.pkl", "B"));
    a.props.push(prop_int("x", 1));
    let mut b = local_node("/virt/proj/b.pkl");
    b.amends = Some(edge(0, "a.pkl", "A"));
    b.props.push(prop_int("y", 2));
    let case = case("mutual_amend_cycle", vec![a, b], 0);

    let sync_mat = materialize(&case);
    let async_mat = materialize(&case);
    let (sync_result, _) = run_sync(&case, &sync_mat, FaultPolicy::default());
    let (async_result, _) =
        pollster::FutureExt::block_on(run_async(&case, &async_mat, FaultPolicy::default()));

    // Both modes terminate and agree on JSON and the full normalized trace.
    assert!(sync_result.json.is_some(), "sync: {:?}", sync_result.error_category);
    assert_eq!(sync_result.json, async_result.json);
    assert_eq!(sync_result.normalized, async_result.normalized);
    // The trace must be bounded (well below the depth limit blowup of 32).
    assert!(
        sync_result.normalized.len() < 80,
        "amend cycle trace unexpectedly long: {}",
        sync_result.normalized.len()
    );
}
