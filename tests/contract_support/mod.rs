//! Shared scaffolding for pklr's model-based contract tests.
//!
//! Everything here is hermetic: an in-memory filesystem stands in for the
//! real OS, fetching never touches the network, and every host interaction is
//! recorded in an ordered event log. The same generated case is driven through
//! both the synchronous and asynchronous public APIs and compared against a
//! simplified reference interpreter.

pub mod model;

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use pklr::capabilities::BoxFuture;
use pklr::EvalCapabilities;

/// Virtual filesystem root. Paths are normalized to this POSIX-style tree on
/// every platform so cases never depend on a real Unix directory or a home dir.
pub const VFS_ROOT: &str = "/virt";
/// Virtual package cache directory.
pub const CACHE_DIR: &str = "/virt/cache";
/// Virtual scratch area used for extracted package archives.
pub const TMP_ROOT: &str = "/virt/tmp";

/// Normalize a logical POSIX path (`.`, `..`, duplicate slashes) lexically.
pub fn normalize(path: &str) -> String {
    let mut stack: Vec<&str> = Vec::new();
    for component in path.split('/') {
        match component {
            "" | "." => {}
            ".." => {
                stack.pop();
            }
            other => stack.push(other),
        }
    }
    let mut out = String::from("/");
    out.push_str(&stack.join("/"));
    out
}

fn to_virtual(path: &Path) -> String {
    let raw = path.to_string_lossy().replace('\\', "/");
    if raw.starts_with('/') {
        normalize(&raw)
    } else {
        normalize(&format!("{VFS_ROOT}/{raw}"))
    }
}

/// A single observed host interaction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Event {
    /// A module source file was read (path or http URL).
    ReadSource(String),
    /// Existence of a local path was checked.
    PathExists(String),
    /// Canonicalization of a local path was requested.
    Canonicalize(String),
    /// A remote text module was fetched.
    FetchText(String),
    /// Remote package bytes were fetched.
    FetchBytes(String),
    /// An environment variable was read.
    EnvRead(String),
    /// Package archive extracted into a directory.
    ExtractZip(String),
    /// Cache sub-step: blob write boundary (`flush`/`rename`/`sync`).
    BlobStage(String),
    /// Cache sub-step: metadata write boundary.
    MetadataStage(String),
    /// Cache directory was synced during a staged write.
    SyncDir(String),
    /// Staging file from a crashed previous run was deleted.
    SweepStaging(String),
    /// Cooperative scheduling point (async only; stripped during comparison).
    SchedYield,
}

impl Event {
    pub fn label(&self) -> &'static str {
        match self {
            Event::ReadSource(_) => "read",
            Event::PathExists(_) => "exists",
            Event::Canonicalize(_) => "canon",
            Event::FetchText(_) => "fetch_text",
            Event::FetchBytes(_) => "fetch_bytes",
            Event::EnvRead(_) => "env",
            Event::ExtractZip(_) => "extract_zip",
            Event::BlobStage(_) => "blob_stage",
            Event::MetadataStage(_) => "meta_stage",
            Event::SyncDir(_) => "sync_dir",
            Event::SweepStaging(_) => "sweep",
            Event::SchedYield => "sched",
        }
    }

    pub fn arg(&self) -> Option<&str> {
        match self {
            Event::ReadSource(a)
            | Event::PathExists(a)
            | Event::Canonicalize(a)
            | Event::FetchText(a)
            | Event::FetchBytes(a)
            | Event::EnvRead(a)
            | Event::ExtractZip(a)
            | Event::BlobStage(a)
            | Event::MetadataStage(a)
            | Event::SyncDir(a)
            | Event::SweepStaging(a) => Some(a),
            Event::SchedYield => None,
        }
    }
}

/// Events plus the monotonic sequence counter and runtime mode are scheduling
/// metadata and are dropped when comparing sync and async runs.
pub fn normalize_events(events: &[(Event, u64, &str)]) -> Vec<(String, String)> {
    events
        .iter()
        .filter(|(event, _, _)| !matches!(event, Event::SchedYield))
        .map(|(event, _, _)| {
            (
                event.label().to_string(),
                event.arg().unwrap_or("").to_string(),
            )
        })
        .collect()
}

/// Which staged-write boundary should fail, and on which file class.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct FaultPolicy {
    /// `"flush"`, `"rename"`, or `"sync"`.
    pub fail_phase: Option<String>,
    /// Apply only to `"blob"` or `"metadata"` files; None applies to both.
    pub file_class: Option<String>,
    /// Fail at most this many times (None = every matching write).
    pub remaining: Option<u32>,
}

impl FaultPolicy {
    pub fn failing(file_class: &str, phase: &str) -> Self {
        Self {
            fail_phase: Some(phase.to_string()),
            file_class: Some(file_class.to_string()),
            remaining: Some(1),
        }
    }
}

/// In-memory filesystem shared between clones of the same evaluator run.
#[derive(Clone, Default)]
pub struct VirtualFs {
    pub files: Arc<Mutex<BTreeMap<String, Vec<u8>>>>,
}

impl VirtualFs {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn write(&self, path: &str, bytes: impl Into<Vec<u8>>) {
        self.files
            .lock()
            .unwrap()
            .insert(normalize(path), bytes.into());
    }

    pub fn read(&self, path: &str) -> Option<Vec<u8>> {
        self.files.lock().unwrap().get(&normalize(path)).cloned()
    }

    pub fn exists(&self, path: &str) -> bool {
        self.files.lock().unwrap().contains_key(&normalize(path))
    }

    pub fn remove(&self, path: &str) -> bool {
        self.files.lock().unwrap().remove(&normalize(path)).is_some()
    }

    pub fn list_dir(&self, dir: &str) -> Vec<String> {
        let dir = normalize(dir);
        self.files
            .lock()
            .unwrap()
            .keys()
            .filter_map(|path| {
                Path::new(path)
                    .parent()
                    .map(|parent| parent.to_string_lossy().into_owned())
                    .filter(|parent| parent == &dir)
                    .and_then(|_| {
                        Path::new(path)
                            .file_name()
                            .map(|name| name.to_string_lossy().into_owned())
                    })
            })
            .collect()
    }

    pub fn staging_leftovers(&self) -> Vec<String> {
        self.files
            .lock()
            .unwrap()
            .keys()
            .filter(|path| {
                Path::new(path)
                    .file_name()
                    .and_then(|name| name.to_str())
                    .is_some_and(|name| name.contains(".staging-"))
            })
            .cloned()
            .collect()
    }
}

#[derive(Clone)]
pub struct FakeCapabilities {
    pub fs: VirtualFs,
    pub env: Arc<BTreeMap<String, String>>,
    /// Remote text modules: rewritten/absolute URL -> source.
    pub http_sources: Arc<BTreeMap<String, String>>,
    /// Remote package bytes: URL -> bytes.
    pub http_bytes: Arc<BTreeMap<String, Vec<u8>>>,
    pub log: Arc<Mutex<Vec<(Event, u64, String)>>>,
    pub counter: Arc<Mutex<u64>>,
    /// When true, insert cooperative scheduling yields (async path).
    pub schedule_markers: bool,
    /// Number of package extractions already performed, for temp-dir naming.
    pub extraction_count: Arc<Mutex<u32>>,
    /// Faults injected into staged writes.
    pub fault: Arc<Mutex<FaultPolicy>>,
}

impl FakeCapabilities {
    pub fn new(
        fs: VirtualFs,
        env: BTreeMap<String, String>,
        http_sources: BTreeMap<String, String>,
        http_bytes: BTreeMap<String, Vec<u8>>,
        schedule_markers: bool,
    ) -> (Self, Arc<Mutex<Vec<(Event, u64, String)>>>) {
        let log = Arc::new(Mutex::new(Vec::new()));
        let caps = Self {
            fs,
            env: Arc::new(env),
            http_sources: Arc::new(http_sources),
            http_bytes: Arc::new(http_bytes),
            log: log.clone(),
            counter: Arc::new(Mutex::new(0)),
            schedule_markers,
            extraction_count: Arc::new(Mutex::new(0)),
            fault: Arc::new(Mutex::new(FaultPolicy::default())),
        };
        (caps, log)
    }

    pub fn set_fault(&self, fault: FaultPolicy) {
        *self.fault.lock().unwrap() = fault;
    }

    fn record(&self, event: Event) {
        let mut counter = self.counter.lock().unwrap();
        *counter += 1;
        let mode = if self.schedule_markers { "async" } else { "sync" }.to_string();
        self.log.lock().unwrap().push((event, *counter, mode));
    }

    fn maybe_yield(&self) {
        if self.schedule_markers {
            self.record(Event::SchedYield);
        }
    }

    fn io_err(message: &str) -> pklr::Error {
        pklr::Error::Io(
            PathBuf::from(message),
            std::io::Error::new(std::io::ErrorKind::Other, message),
        )
    }
}

impl EvalCapabilities for FakeCapabilities {
    fn read_to_string<'a>(&'a mut self, path: &'a Path) -> BoxFuture<'a, pklr::Result<String>> {
        self.maybe_yield();
        let key = path.to_string_lossy().replace('\\', "/");
        self.record(Event::ReadSource(key.clone()));
        let bytes = self.fs.read(&to_virtual(path));
        Box::pin(async move {
            let bytes = bytes.ok_or_else(|| {
                pklr::Error::Io(
                    PathBuf::from(key.clone()),
                    std::io::Error::from(std::io::ErrorKind::NotFound),
                )
            })?;
            String::from_utf8(bytes).map_err(|error| pklr::Error::Eval(error.to_string()))
        })
    }

    fn path_exists<'a>(&'a mut self, path: &'a Path) -> BoxFuture<'a, pklr::Result<bool>> {
        self.maybe_yield();
        let key = to_virtual(path);
        self.record(Event::PathExists(key.clone()));
        let exists = self.fs.exists(&key);
        Box::pin(async move { Ok(exists) })
    }

    fn canonicalize<'a>(&'a mut self, path: &'a Path) -> BoxFuture<'a, pklr::Result<PathBuf>> {
        self.maybe_yield();
        let key = to_virtual(path);
        self.record(Event::Canonicalize(key.clone()));
        Box::pin(async move { Ok(PathBuf::from(key)) })
    }

    fn read_bytes<'a>(&'a mut self, path: &'a Path) -> BoxFuture<'a, pklr::Result<Vec<u8>>> {
        self.maybe_yield();
        let key = to_virtual(path);
        self.record(Event::ReadSource(key.clone()));
        let bytes = self.fs.read(&key);
        Box::pin(async move {
            bytes.ok_or_else(|| {
                pklr::Error::Io(
                    PathBuf::from(key),
                    std::io::Error::from(std::io::ErrorKind::NotFound),
                )
            })
        })
    }

    fn create_dir_all<'a>(&'a mut self, _path: &'a Path) -> BoxFuture<'a, pklr::Result<()>> {
        Box::pin(async { Ok(()) })
    }

    fn stage_write<'a>(
        &'a mut self,
        path: &'a Path,
        bytes: &'a [u8],
        on_phase: &'a mut dyn FnMut(&str),
    ) -> BoxFuture<'a, pklr::Result<()>> {
        self.maybe_yield();
        let target = to_virtual(path);
        let is_metadata = target.contains("/packages/");
        let file_class = if is_metadata { "metadata" } else { "blob" };
        let bytes = bytes.to_vec();
        let fs = self.fs.clone();
        let fault = self.fault.clone();
        let log = self.log.clone();
        let counter = self.counter.clone();
        let schedule_markers = self.schedule_markers;
        Box::pin(async move {
            let record = |event: Event| {
                let mut counter = counter.lock().unwrap();
                *counter += 1;
                let mode = if schedule_markers { "async" } else { "sync" }.to_string();
                log.lock().unwrap().push((event, *counter, mode));
            };
            let staging = format!("{target}.staging-{}-{}", std::process::id(), {
                use std::sync::atomic::{AtomicU64, Ordering};
                static N: AtomicU64 = AtomicU64::new(0);
                N.fetch_add(1, Ordering::Relaxed)
            });
            let consume_fault = |phase: &str| -> bool {
                let mut policy = fault.lock().unwrap();
                let matches = policy.fail_phase.as_deref() == Some(phase)
                    && policy
                        .file_class
                        .as_deref()
                        .map_or(true, |class| class == file_class);
                if matches
                    && let Some(remaining) = policy.remaining.as_mut()
                {
                    *remaining = remaining.saturating_sub(1);
                    if *remaining == 0 {
                        *policy = FaultPolicy::default();
                    }
                }
                matches
            };

            // Flush boundary: the staging file becomes visible with full bytes.
            fs.write(&staging, bytes.clone());
            on_phase("flush");
            if is_metadata {
                record(Event::MetadataStage("flush".to_string()));
            } else {
                record(Event::BlobStage("flush".to_string()));
            }
            if consume_fault("flush") {
                fs.remove(&staging);
                return Err(FakeCapabilities::io_err("injected flush failure"));
            }

            // Rename boundary: on failure the staging file remains unpublished.
            if consume_fault("rename") {
                return Err(FakeCapabilities::io_err("injected rename failure"));
            }
            fs.remove(&staging);
            fs.write(&target, bytes);
            on_phase("rename");
            if is_metadata {
                record(Event::MetadataStage("rename".to_string()));
            } else {
                record(Event::BlobStage("rename".to_string()));
            }

            // Directory sync boundary: content is durable afterward.
            if consume_fault("sync") {
                return Err(FakeCapabilities::io_err("injected sync failure"));
            }
            on_phase("sync");
            if is_metadata {
                record(Event::MetadataStage("sync".to_string()));
            } else {
                record(Event::BlobStage("sync".to_string()));
            }
            Ok(())
        })
    }

    fn sync_dir<'a>(&'a mut self, _path: &'a Path) -> BoxFuture<'a, pklr::Result<()>> {
        Box::pin(async { Ok(()) })
    }

    fn list_dir<'a>(&'a mut self, path: &'a Path) -> BoxFuture<'a, pklr::Result<Vec<String>>> {
        let key = to_virtual(path);
        let names = self.fs.list_dir(&key);
        Box::pin(async move { Ok(names) })
    }

    fn remove_file<'a>(&'a mut self, path: &'a Path) -> BoxFuture<'a, pklr::Result<()>> {
        let key = to_virtual(path);
        if key.contains(".staging-") {
            self.record(Event::SweepStaging(key.clone()));
        }
        let removed = self.fs.remove(&key);
        Box::pin(async move {
            if removed {
                Ok(())
            } else {
                Err(pklr::Error::Io(
                    PathBuf::from(key),
                    std::io::Error::from(std::io::ErrorKind::NotFound),
                ))
            }
        })
    }

    #[cfg(feature = "package-zip-core")]
    fn extract_zip<'a>(
        &'a mut self,
        bytes: Vec<u8>,
        destination: &'a Path,
    ) -> BoxFuture<'a, pklr::Result<()>> {
        self.maybe_yield();
        let dest = to_virtual(destination);
        self.record(Event::ExtractZip(dest.clone()));
        let fs = self.fs.clone();
        Box::pin(async move {
            let mut archive = zip::ZipArchive::new(std::io::Cursor::new(bytes))
                .map_err(|error| pklr::Error::Eval(format!("zip error: {error}")))?;
            for index in 0..archive.len() {
                let mut entry = archive
                    .by_index(index)
                    .map_err(|error| pklr::Error::Eval(format!("zip error: {error}")))?;
                if entry.is_dir() {
                    continue;
                }
                let mut content = Vec::new();
                std::io::copy(&mut entry, &mut content)
                    .map_err(|error| pklr::Error::Eval(format!("zip error: {error}")))?;
                fs.write(&format!("{dest}/{}", entry.name()), content);
            }
            Ok(())
        })
    }

    fn read_env<'a>(&'a mut self, name: &'a str) -> BoxFuture<'a, pklr::Result<Option<String>>> {
        self.maybe_yield();
        self.record(Event::EnvRead(name.to_string()));
        let value = self.env.get(name).cloned();
        Box::pin(async move { Ok(value) })
    }

    fn fetch_text<'a>(&'a mut self, url: &'a str) -> BoxFuture<'a, pklr::Result<String>> {
        self.maybe_yield();
        self.record(Event::FetchText(url.to_string()));
        let source = self.http_sources.get(url).cloned();
        Box::pin(async move {
            source.ok_or_else(|| pklr::Error::ImportNotFound(url.to_string()))
        })
    }

    fn fetch_bytes<'a>(&'a mut self, url: &'a str) -> BoxFuture<'a, pklr::Result<Vec<u8>>> {
        self.maybe_yield();
        self.record(Event::FetchBytes(url.to_string()));
        let bytes = self.http_bytes.get(url).cloned();
        Box::pin(async move {
            bytes.ok_or_else(|| pklr::Error::ImportNotFound(url.to_string()))
        })
    }

    fn temp_dir<'a>(&'a mut self, prefix: &'a str) -> BoxFuture<'a, pklr::Result<PathBuf>> {
        let mut count = self.extraction_count.lock().unwrap();
        *count += 1;
        let dir = PathBuf::from(format!("{TMP_ROOT}/{prefix}-{count}"));
        Box::pin(async move { Ok(dir) })
    }

    fn glob<'a>(
        &'a mut self,
        _base: &'a Path,
        _pattern: &'a str,
    ) -> BoxFuture<'a, pklr::Result<Vec<PathBuf>>> {
        Box::pin(async { Ok(Vec::new()) })
    }
}

// ---------------------------------------------------------------------------
// Pkl rendering
// ---------------------------------------------------------------------------

use model::{Case, Edge, GateOp, Location, Node, Prop, PropKind};

pub fn render_prop(prop: &Prop) -> String {
    match &prop.kind {
        PropKind::Int(value) => format!("{} = {value}\n", prop.name),
        PropKind::Gate { op, left, right } => {
            let symbol = match op {
                GateOp::And => "&&",
                GateOp::Or => "||",
            };
            format!(
                "{} = read?(\"env:{left}\") != null {symbol} read?(\"env:{right}\") != null\n",
                prop.name
            )
        }
        PropKind::Ref { alias, field } => {
            format!("{} = {alias}.{field}\n", prop.name)
        }
    }
}

pub fn render_node(nodes: &[Node], index: usize) -> String {
    let node = &nodes[index];
    let mut out = String::new();
    for edge in &node.imports {
        out.push_str(&format!("import \"{}\" as {}\n", edge.uri, edge.alias));
    }
    if let Some(edge) = &node.amends {
        out.push_str(&format!("amends \"{}\"\n", edge.uri));
    }
    for prop in &node.props {
        out.push_str(&render_prop(prop));
    }
    let _ = nodes;
    out
}

// ---------------------------------------------------------------------------
// World materialization + dual execution
// ---------------------------------------------------------------------------

pub struct Materialized {
    pub fs: VirtualFs,
    pub http_sources: BTreeMap<String, String>,
    pub http_bytes: BTreeMap<String, Vec<u8>>,
    pub entry_path: PathBuf,
}

pub fn materialize(case: &Case) -> Materialized {
    let fs = VirtualFs::new();
    let mut http_sources: BTreeMap<String, String> = BTreeMap::new();
    let mut http_bytes: BTreeMap<String, Vec<u8>> = BTreeMap::new();

    for file in &case.prestate {
        fs.write(&file.path, file.bytes.clone());
    }

    for (index, node) in case.nodes.iter().enumerate() {
        let source = render_node(&case.nodes, index);
        match node.loc.as_ref().unwrap() {
            Location::Local(path) => fs.write(path, source.clone()),
            Location::Http(url) => {
                http_sources.insert(url.clone(), source.clone());
            }
            Location::PkgDirect {
                download_url, ..
            } => {
                http_bytes.insert(download_url.clone(), source.into_bytes());
            }
        }
    }

    let entry_path = match case.nodes[case.entry].loc.as_ref().unwrap() {
        Location::Local(path) => PathBuf::from(path),
        Location::Http(url) => PathBuf::from(url),
        Location::PkgDirect { download_url, .. } => PathBuf::from(download_url),
    };

    Materialized {
        fs,
        http_sources,
        http_bytes,
        entry_path,
    }
}

#[derive(Clone)]
pub struct RunResult {
    pub json: Option<String>,
    pub error_category: Option<String>,
    pub env_reads: BTreeMap<String, bool>,
    pub normalized: Vec<(String, String)>,
    pub raw: Vec<(Event, u64, String)>,
}

pub async fn run_async(
    case: &Case,
    mat: &Materialized,
    fault: FaultPolicy,
) -> (RunResult, FakeCapabilities) {
    let (caps, log) = FakeCapabilities::new(
        mat.fs.clone(),
        case.env.clone(),
        mat.http_sources.clone(),
        mat.http_bytes.clone(),
        true,
    );
    caps.set_fault(fault);
    let mut builder = pklr::AsyncEvaluatorBuilder::new()
        .offline(case.offline)
        .package_cache_dir(CACHE_DIR)
        .capabilities(caps.clone());
    for (src, tgt) in &case.rewrites {
        builder = builder.http_rewrites(vec![format!("{src}={tgt}")]);
    }
    for preload in &case.preloads {
        builder = builder.preload_package(
            preload.url.clone(),
            preload.extension.clone(),
            preload.bytes.clone(),
        );
    }
    let outcome = builder.eval(&mat.entry_path).await;
    finish_run_simple(outcome, log, caps)
}

pub fn run_sync(
    case: &Case,
    mat: &Materialized,
    fault: FaultPolicy,
) -> (RunResult, FakeCapabilities) {
    let (caps, log) = FakeCapabilities::new(
        mat.fs.clone(),
        case.env.clone(),
        mat.http_sources.clone(),
        mat.http_bytes.clone(),
        false,
    );
    caps.set_fault(fault);
    let mut builder = pklr::EvaluatorBuilder::new()
        .offline(case.offline)
        .package_cache_dir(CACHE_DIR)
        .capabilities(caps.clone());
    for (src, tgt) in &case.rewrites {
        builder = builder.http_rewrites(vec![format!("{src}={tgt}")]);
    }
    for preload in &case.preloads {
        builder = builder.preload_package(
            preload.url.clone(),
            preload.extension.clone(),
            std::borrow::Cow::Owned(preload.bytes.clone()),
        );
    }
    let outcome = builder.eval(&mat.entry_path);
    finish_run_simple(outcome, log, caps)
}

fn finish_run_simple(
    outcome: pklr::Result<pklr::EvalOutcome>,
    log: Arc<Mutex<Vec<(Event, u64, String)>>>,
    caps: FakeCapabilities,
) -> (RunResult, FakeCapabilities) {
    let raw = log.lock().unwrap().clone();
    let normalized = normalize_events(
        &raw.iter()
            .map(|(event, seq, mode)| (event.clone(), *seq, mode.as_str()))
            .collect::<Vec<_>>(),
    );
    let (json, error_category, env_reads) = match outcome {
        Ok(outcome) => (
            Some(outcome.json.to_string()),
            None,
            outcome
                .env_reads
                .into_iter()
                .map(|(key, value)| (key, value.is_some()))
                .collect(),
        ),
        Err(error) => (None, Some(category_of(&error.to_string())), BTreeMap::new()),
    };
    (
        RunResult {
            json,
            error_category,
            env_reads,
            normalized,
            raw,
        },
        caps,
    )
}

/// Coarse, stable error categories shared with the reference model.
pub fn category_of(message: &str) -> String {
    if message.contains("offline") {
        "offline-miss".to_string()
    } else if message.contains("Import not found") {
        "import-not-found".to_string()
    } else if message.contains("field not found") {
        "field-not-found".to_string()
    } else if message.contains("corrupt package cache") {
        if message.contains("digest mismatch") {
            "digest-mismatch".to_string()
        } else if message.contains("unsupported cache version") {
            "unsupported-version".to_string()
        } else if message.contains("escapes") {
            "containment".to_string()
        } else {
            "corrupt-metadata".to_string()
        }
    } else if message.contains("environment variable not found") {
        "env-missing".to_string()
    } else {
        "other".to_string()
    }
}

pub fn edge(to: usize, uri: &str, alias: &str) -> Edge {
    Edge {
        to,
        uri: uri.to_string(),
        alias: alias.to_string(),
        used: true,
    }
}

pub fn unused_edge(to: usize, uri: &str, alias: &str) -> Edge {
    let mut edge = edge(to, uri, alias);
    edge.used = false;
    edge
}

pub fn prop_int(name: &str, value: i64) -> Prop {
    Prop {
        name: name.to_string(),
        kind: PropKind::Int(value),
    }
}

pub fn prop_gate(name: &str, op: GateOp, left: &str, right: &str) -> Prop {
    Prop {
        name: name.to_string(),
        kind: PropKind::Gate {
            op,
            left: left.to_string(),
            right: right.to_string(),
        },
    }
}

pub fn prop_ref(name: &str, alias: &str, field: &str) -> Prop {
    Prop {
        name: name.to_string(),
        kind: PropKind::Ref {
            alias: alias.to_string(),
            field: field.to_string(),
        },
    }
}

pub fn local_node(path: &str) -> Node {
    Node {
        loc: Some(Location::Local(path.to_string())),
        ..Default::default()
    }
}

pub fn http_node(url: &str) -> Node {
    Node {
        loc: Some(Location::Http(url.to_string())),
        ..Default::default()
    }
}

pub fn pkg_direct_node(pkg_uri: &str, download_url: &str) -> Node {
    Node {
        loc: Some(Location::PkgDirect {
            pkg_uri: pkg_uri.to_string(),
            download_url: download_url.to_string(),
        }),
        ..Default::default()
    }
}
