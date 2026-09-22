//! Model-based contract tests for pklr's evaluation pipeline.
//!
//! A simplified model of URI normalization, HTTP rewrite (longest prefix),
//! relative file imports, the package metadata/blob cache, preloads, offline
//! mode, amend edges, and cycle detection is evaluated alongside the real
//! evaluator over generated import/amend graphs. Both sides consume the same
//! seeded case; the implementation runs against fake capabilities that record
//! an ordered event log, and the resulting trace (fetches, cache writes, env
//! reads, final JSON or error) must match the model exactly.
//!
//! Every generated case is also evaluated through the synchronous
//! (pollster-driven) and asynchronous (tokio-driven) paths; the two event
//! logs must be equivalent after normalizing scheduling-only fields (temp
//! directory names).
//!
//! The generator uses a fixed seed and explicit state-space bounds. On
//! failure the case is greedily shrunk and printed as a JSON blob that can be
//! replayed with `PKLR_MODEL_REPLAY='<json>' cargo test --test model_contract`.
//! `PKLR_MODEL_SEED` / `PKLR_MODEL_CASES` override the seed and case count.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use pklr::capabilities::{BoxFuture, EvalCapabilities};
use pklr::{Error, Evaluator};
use serde_json::{Value as J, json};

// --- Explicit state-space bounds -------------------------------------------
const MAX_FILE_MODULES: usize = 4;
const MAX_HTTP_MODULES: usize = 2;
const MAX_PACKAGES: usize = 2;
const MAX_IMPORTS_PER_MODULE: usize = 2;
const MAX_EXTRA_PROPS: usize = 2;
const DEFAULT_CASES: usize = 64;
const DEFAULT_SEED: u64 = 0x9e37_79b9_7f4a_7c15;
const ENV_KEYS: [&str; 6] = [
    "PKLR_MODEL_ENV_0",
    "PKLR_MODEL_ENV_1",
    "PKLR_MODEL_ENV_2",
    "PKLR_MODEL_ENV_3",
    "PKLR_MODEL_ENV_4",
    "PKLR_MODEL_ENV_5",
];

// --- Deterministic RNG (splitmix64) -----------------------------------------
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }

    fn chance(&mut self, pct: u64) -> bool {
        self.next() % 100 < pct
    }
}

// --- Ordered event log ------------------------------------------------------
#[derive(Clone, Debug, PartialEq, Eq)]
enum Event {
    ReadFile(String),
    ReadBytes(String),
    PathExists(String),
    Canonicalize(String),
    ListDir(String),
    CreateDir(String),
    WriteAtomic(String),
    RemoveFile(String),
    ReadEnv(String),
    ReadProp(String),
    FetchText(String),
    FetchBytes(String),
    TempDir(String),
    ExtractZip(String),
}

/// Normalize scheduling-only fields out of an event log: temp directory
/// names depend on extraction order and host temp roots, so they are
/// replaced with a stable placeholder before comparing two runs.
fn normalize_log(events: &[Event]) -> Vec<Event> {
    events
        .iter()
        .map(|event| {
            let normalized = match event {
                Event::TempDir(_) => Event::TempDir("<temp>".to_string()),
                Event::ExtractZip(_) => Event::ExtractZip("<temp>".to_string()),
                other => other.clone(),
            };
            normalized
        })
        .collect()
}

// --- Fake capabilities over an in-memory filesystem -------------------------
#[derive(Default)]
struct FakeState {
    files: BTreeMap<String, Vec<u8>>,
    dirs: BTreeSet<String>,
    env: BTreeMap<String, String>,
    props: BTreeMap<String, String>,
    http_text: HashMap<String, String>,
    http_bytes: HashMap<String, Vec<u8>>,
}

#[derive(Clone, Default)]
struct Fake {
    state: Arc<Mutex<FakeState>>,
    log: Arc<Mutex<Vec<Event>>>,
}

impl Fake {
    fn with_state(state: FakeState) -> Self {
        Self {
            state: Arc::new(Mutex::new(state)),
            log: Arc::new(Mutex::new(Vec::new())),
        }
    }

    fn record(&self, event: Event) {
        self.log.lock().unwrap().push(event);
    }

    fn events(&self) -> Vec<Event> {
        self.log.lock().unwrap().clone()
    }
}

/// Lexical path normalization shared by the fake filesystem and the model:
/// forward slashes, no `.`/`..` segments, always rooted at `/`.
fn norm(path: &Path) -> String {
    norm_str(&path.to_string_lossy())
}

fn norm_str(raw: &str) -> String {
    let raw = raw.replace('\\', "/");
    let mut parts: Vec<&str> = Vec::new();
    for seg in raw.split('/') {
        match seg {
            "" | "." => {}
            ".." => {
                parts.pop();
            }
            seg => parts.push(seg),
        }
    }
    format!("/{}", parts.join("/"))
}

fn join_str(base: &str, rel: &str) -> String {
    norm_str(&format!("{base}/{rel}"))
}

fn parent_str(path: &str) -> String {
    match path.rfind('/') {
        Some(0) | None => "/".to_string(),
        Some(idx) => path[..idx].to_string(),
    }
}

fn not_found(path: &Path) -> Error {
    Error::Io(
        path.to_path_buf(),
        std::io::Error::from(std::io::ErrorKind::NotFound),
    )
}

impl EvalCapabilities for Fake {
    fn read_to_string<'a>(&'a mut self, path: &'a Path) -> BoxFuture<'a, pklr::Result<String>> {
        let key = norm(path);
        self.record(Event::ReadFile(key.clone()));
        let bytes = self.state.lock().unwrap().files.get(&key).cloned();
        Box::pin(async move {
            let bytes = bytes.ok_or_else(|| not_found(Path::new(&key)))?;
            String::from_utf8(bytes).map_err(|error| Error::Eval(error.to_string()))
        })
    }

    fn read_bytes<'a>(&'a mut self, path: &'a Path) -> BoxFuture<'a, pklr::Result<Vec<u8>>> {
        let key = norm(path);
        self.record(Event::ReadBytes(key.clone()));
        let bytes = self.state.lock().unwrap().files.get(&key).cloned();
        Box::pin(async move { bytes.ok_or_else(|| not_found(Path::new(&key))) })
    }

    fn path_exists<'a>(&'a mut self, path: &'a Path) -> BoxFuture<'a, pklr::Result<bool>> {
        let key = norm(path);
        self.record(Event::PathExists(key.clone()));
        let exists = {
            let state = self.state.lock().unwrap();
            state.files.contains_key(&key) || state.dirs.contains(&key)
        };
        Box::pin(async move { Ok(exists) })
    }

    fn canonicalize<'a>(&'a mut self, path: &'a Path) -> BoxFuture<'a, pklr::Result<PathBuf>> {
        let key = norm(path);
        self.record(Event::Canonicalize(key.clone()));
        Box::pin(async move { Ok(PathBuf::from(key)) })
    }

    fn create_dir_all<'a>(&'a mut self, path: &'a Path) -> BoxFuture<'a, pklr::Result<()>> {
        let key = norm(path);
        self.record(Event::CreateDir(key.clone()));
        let mut state = self.state.lock().unwrap();
        let mut current = String::new();
        for seg in key.split('/').filter(|seg| !seg.is_empty()) {
            current.push('/');
            current.push_str(seg);
            state.dirs.insert(current.clone());
        }
        Box::pin(async { Ok(()) })
    }

    fn write_atomic<'a>(
        &'a mut self,
        path: &'a Path,
        bytes: &'a [u8],
    ) -> BoxFuture<'a, pklr::Result<()>> {
        let key = norm(path);
        self.record(Event::WriteAtomic(key.clone()));
        let bytes = bytes.to_vec();
        let mut state = self.state.lock().unwrap();
        state.dirs.insert(parent_str(&key));
        state.files.insert(key, bytes);
        Box::pin(async { Ok(()) })
    }

    fn remove_file<'a>(&'a mut self, path: &'a Path) -> BoxFuture<'a, pklr::Result<()>> {
        let key = norm(path);
        self.record(Event::RemoveFile(key.clone()));
        let removed = self.state.lock().unwrap().files.remove(&key);
        Box::pin(async move {
            removed
                .map(|_| ())
                .ok_or_else(|| not_found(Path::new(&key)))
        })
    }

    fn list_dir<'a>(&'a mut self, path: &'a Path) -> BoxFuture<'a, pklr::Result<Vec<PathBuf>>> {
        let key = norm(path);
        self.record(Event::ListDir(key.clone()));
        let state = self.state.lock().unwrap();
        if !state.dirs.contains(&key) {
            let err = not_found(Path::new(&key));
            return Box::pin(async move { Err(err) });
        }
        let prefix = format!("{key}/");
        let mut children = BTreeSet::new();
        for entry in state.files.keys().chain(state.dirs.iter()) {
            if let Some(rest) = entry.strip_prefix(&prefix)
                && !rest.contains('/')
            {
                children.insert(entry.clone());
            }
        }
        let paths: Vec<PathBuf> = children.into_iter().map(PathBuf::from).collect();
        Box::pin(async move { Ok(paths) })
    }

    fn extract_zip<'a>(
        &'a mut self,
        bytes: Vec<u8>,
        destination: &'a Path,
    ) -> BoxFuture<'a, pklr::Result<()>> {
        let dest = norm(destination);
        self.record(Event::ExtractZip(dest.clone()));
        let mut state = self.state.lock().unwrap();
        let result = (|| -> pklr::Result<()> {
            let mut archive = zip::ZipArchive::new(std::io::Cursor::new(bytes))
                .map_err(|error| Error::Eval(format!("zip error: {error}")))?;
            for index in 0..archive.len() {
                let mut entry = archive
                    .by_index(index)
                    .map_err(|error| Error::Eval(format!("zip error: {error}")))?;
                if entry.is_dir() {
                    continue;
                }
                let mut data = Vec::new();
                std::io::Read::read_to_end(&mut entry, &mut data)
                    .map_err(|error| Error::Eval(format!("zip read error: {error}")))?;
                let path = join_str(&dest, entry.name());
                state.dirs.insert(parent_str(&path));
                state.files.insert(path, data);
            }
            Ok(())
        })();
        Box::pin(async move { result })
    }

    fn read_env<'a>(&'a mut self, name: &'a str) -> BoxFuture<'a, pklr::Result<Option<String>>> {
        self.record(Event::ReadEnv(name.to_string()));
        let value = self.state.lock().unwrap().env.get(name).cloned();
        Box::pin(async move { Ok(value) })
    }

    fn read_prop<'a>(&'a mut self, name: &'a str) -> BoxFuture<'a, pklr::Result<Option<String>>> {
        self.record(Event::ReadProp(name.to_string()));
        let value = self.state.lock().unwrap().props.get(name).cloned();
        Box::pin(async move { Ok(value) })
    }

    fn fetch_text<'a>(&'a mut self, url: &'a str) -> BoxFuture<'a, pklr::Result<String>> {
        self.record(Event::FetchText(url.to_string()));
        let source = self.state.lock().unwrap().http_text.get(url).cloned();
        let url = url.to_string();
        Box::pin(async move { source.ok_or(Error::ImportNotFound(url)) })
    }

    fn fetch_bytes<'a>(&'a mut self, url: &'a str) -> BoxFuture<'a, pklr::Result<Vec<u8>>> {
        self.record(Event::FetchBytes(url.to_string()));
        let bytes = self.state.lock().unwrap().http_bytes.get(url).cloned();
        let url = url.to_string();
        Box::pin(async move { bytes.ok_or(Error::ImportNotFound(url)) })
    }

    fn temp_dir<'a>(&'a mut self, prefix: &'a str) -> BoxFuture<'a, pklr::Result<PathBuf>> {
        self.record(Event::TempDir(prefix.to_string()));
        let path = format!("/tmp/{prefix}");
        let mut state = self.state.lock().unwrap();
        if state.dirs.contains(&path) {
            let err = Error::Eval(format!("temp dir already exists: {path}"));
            return Box::pin(async move { Err(err) });
        }
        state.dirs.insert(path.clone());
        Box::pin(async move { Ok(PathBuf::from(path)) })
    }

    fn glob<'a>(
        &'a mut self,
        _base: &'a Path,
        _pattern: &'a str,
    ) -> BoxFuture<'a, pklr::Result<Vec<PathBuf>>> {
        Box::pin(async { Ok(Vec::new()) })
    }
}

// --- Generated case model ---------------------------------------------------

#[derive(Clone, Debug, PartialEq)]
enum Target {
    File(usize),
    Http(usize),
    Package(usize, usize),
    Missing,
}

#[derive(Clone, Debug, PartialEq)]
struct Import {
    alias: String,
    target: Target,
}

#[derive(Clone, Debug, PartialEq)]
enum Prop {
    Const(String, i64),
    Env(String, String),
    CondEnv(String, bool, String, String),
    Ref(String, String),
}

impl Prop {
    fn name(&self) -> &str {
        match self {
            Prop::Const(name, _)
            | Prop::Env(name, _)
            | Prop::CondEnv(name, _, _, _)
            | Prop::Ref(name, _) => name,
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
struct ModSpec {
    amends: Option<Target>,
    imports: Vec<Import>,
    props: Vec<Prop>,
}

#[derive(Clone, Debug, Default, PartialEq)]
struct HttpSpec {
    env_read: Option<String>,
    rel_import: Option<usize>,
}

#[derive(Clone, Debug, Default, PartialEq)]
struct PkgSpec {
    /// Optional env key read by each entry module.
    entries: Vec<Option<String>>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Seed {
    Absent,
    Valid,
    CorruptMeta,
    BadVersion,
    UrlMismatch,
    BlobEscape,
    ChecksumMismatch,
    MissingBlob,
}

#[derive(Clone, Debug, Default, PartialEq)]
struct Case {
    files: Vec<ModSpec>,
    https: Vec<HttpSpec>,
    pkgs: Vec<PkgSpec>,
    rewrites: Vec<(String, String)>,
    cache_seed: Vec<Seed>,
    preloads: Vec<usize>,
    offline: bool,
    env: BTreeMap<String, String>,
}

fn file_path(idx: usize) -> String {
    format!("/src/mod{idx}.pkl")
}

fn file_uri(idx: usize) -> String {
    // Exercise lexical normalization: odd indices are reached through a
    // `sub/..` detour that must canonicalize back to the same module.
    if idx % 2 == 1 {
        format!("sub/../mod{idx}.pkl")
    } else {
        format!("mod{idx}.pkl")
    }
}

fn http_url(idx: usize) -> String {
    format!("https://h0.test/dir/mod{idx}.pkl")
}

fn pkg_zip_url(idx: usize) -> String {
    format!("https://pkg{idx}.test/packages/p{idx}@1.0.0.zip")
}

fn pkg_uri(pkg: usize, entry: usize) -> String {
    format!("package://pkg{pkg}.test/packages/p{pkg}@1.0.0#/entry{entry}.pkl")
}

fn target_uri(target: &Target) -> String {
    match target {
        Target::File(idx) => file_uri(*idx),
        Target::Http(idx) => http_url(*idx),
        Target::Package(pkg, entry) => pkg_uri(*pkg, *entry),
        Target::Missing => "missing.pkl".to_string(),
    }
}

fn apply_rewrites(rules: &[(String, String)], url: &str) -> String {
    let best = rules
        .iter()
        .filter(|(src, _)| url.starts_with(src.as_str()))
        .max_by_key(|(src, _)| src.len());
    match best {
        Some((src, tgt)) => format!("{}{}", tgt, &url[src.len()..]),
        None => url.to_string(),
    }
}

// --- Generator (fixed seed, bounded state space) ----------------------------

fn generate(rng: &mut Rng) -> Case {
    let mut case = Case::default();
    let n_files = 1 + rng.below(MAX_FILE_MODULES);
    let n_http = rng.below(MAX_HTTP_MODULES + 1);
    let n_pkgs = rng.below(MAX_PACKAGES + 1);

    for idx in 0..n_http {
        let mut spec = HttpSpec::default();
        if rng.chance(40) {
            spec.env_read = Some(ENV_KEYS[rng.below(ENV_KEYS.len())].to_string());
        }
        if idx > 0 && rng.chance(40) {
            spec.rel_import = Some(rng.below(idx));
        }
        case.https.push(spec);
    }
    for _ in 0..n_pkgs {
        let n_entries = 1 + rng.below(2);
        let entries = (0..n_entries)
            .map(|_| {
                if rng.chance(40) {
                    Some(ENV_KEYS[rng.below(ENV_KEYS.len())].to_string())
                } else {
                    None
                }
            })
            .collect();
        case.pkgs.push(PkgSpec { entries });
    }

    for idx in 0..n_files {
        let mut module = ModSpec::default();
        module.props.push(Prop::Const("id".to_string(), idx as i64));
        if rng.chance(40) {
            module.amends = Some(gen_target(rng, idx, n_files, n_http, n_pkgs, true));
        }
        let n_imports = rng.below(MAX_IMPORTS_PER_MODULE + 1);
        for k in 0..n_imports {
            let mut target = gen_target(rng, idx, n_files, n_http, n_pkgs, false);
            // Avoid the deferred-inherited-import path: never import the exact
            // file this module also amends.
            if let (Some(Target::File(amended)), Target::File(imported)) =
                (&module.amends, &target)
                && amended == imported
            {
                target = Target::Missing;
            }
            module.imports.push(Import {
                alias: format!("i{idx}_{k}"),
                target,
            });
        }
        // Most imports are referenced so they are actually evaluated (pklr
        // imports are lazy); leave some unreferenced to cover laziness.
        for k in 0..module.imports.len() {
            if rng.chance(70) {
                let alias = module.imports[k].alias.clone();
                module.props.push(Prop::Ref(format!("r{idx}_{k}"), alias));
            }
        }
        let n_props = rng.below(MAX_EXTRA_PROPS + 1);
        for k in 0..n_props {
            let name = format!("p{idx}_{k}");
            let prop = match rng.below(10) {
                0..=2 => Prop::Const(name, rng.below(100) as i64),
                3..=5 => Prop::Env(name, ENV_KEYS[rng.below(ENV_KEYS.len())].to_string()),
                6..=7 => Prop::CondEnv(
                    name,
                    rng.chance(50),
                    ENV_KEYS[rng.below(ENV_KEYS.len())].to_string(),
                    ENV_KEYS[rng.below(ENV_KEYS.len())].to_string(),
                ),
                _ => {
                    if module.imports.is_empty() {
                        Prop::Const(name, rng.below(100) as i64)
                    } else {
                        let alias = module.imports[rng.below(module.imports.len())].alias.clone();
                        Prop::Ref(name, alias)
                    }
                }
            };
            module.props.push(prop);
        }
        case.files.push(module);
    }

    // Rewrite rules with overlapping prefixes to exercise longest-prefix wins.
    let mut pool: Vec<(String, String)> = Vec::new();
    if n_http > 0 {
        pool.push(("https://h0.test/".into(), "https://mirror0.test/x/".into()));
        pool.push((
            "https://h0.test/dir/".into(),
            "https://deep0.test/y/".into(),
        ));
    }
    for p in 0..n_pkgs {
        pool.push((
            format!("https://pkg{p}.test/"),
            format!("https://pkmirror{p}.test/z/"),
        ));
    }
    while !pool.is_empty() && case.rewrites.len() < 3 && rng.chance(45) {
        let idx = rng.below(pool.len());
        let rule = pool.swap_remove(idx);
        if !case.rewrites.contains(&rule) {
            case.rewrites.push(rule);
        }
    }

    for _ in 0..n_pkgs {
        let seed = match rng.below(100) {
            0..=29 => Seed::Valid,
            30..=64 => Seed::Absent,
            65..=70 => Seed::CorruptMeta,
            71..=76 => Seed::BadVersion,
            77..=82 => Seed::UrlMismatch,
            83..=88 => Seed::BlobEscape,
            89..=94 => Seed::ChecksumMismatch,
            _ => Seed::MissingBlob,
        };
        case.cache_seed.push(seed);
    }
    let n_preloads = rng.below(n_pkgs + 1);
    for _ in 0..n_preloads {
        let pkg = rng.below(n_pkgs);
        if !case.preloads.contains(&pkg) {
            case.preloads.push(pkg);
        }
    }
    case.offline = rng.chance(35);
    for key in ENV_KEYS {
        if rng.chance(60) {
            case.env.insert(key.to_string(), format!("val-{key}"));
        }
    }
    case
}

fn gen_target(
    rng: &mut Rng,
    self_idx: usize,
    n_files: usize,
    n_http: usize,
    n_pkgs: usize,
    for_amends: bool,
) -> Target {
    loop {
        let roll = rng.below(100);
        let target = match roll {
            0..=54 => Target::File(rng.below(n_files)),
            55..=69 => {
                if n_http > 0 {
                    Target::Http(rng.below(n_http))
                } else {
                    continue;
                }
            }
            70..=89 => {
                if n_pkgs > 0 {
                    let pkg = rng.below(n_pkgs);
                    let entry = 0; // entry 0 always exists
                    Target::Package(pkg, entry)
                } else {
                    continue;
                }
            }
            _ => Target::Missing,
        };
        // Amending an http module's own index is fine; nothing to filter here.
        let _ = self_idx;
        let _ = for_amends;
        return target;
    }
}

// --- Source materialization ---------------------------------------------------

fn prop_source(prop: &Prop) -> String {
    match prop {
        Prop::Const(name, value) => format!("{name} = {value}\n"),
        Prop::Env(name, key) => format!("{name} = read(\"env:{key}\")\n"),
        Prop::CondEnv(name, cond, k1, k2) => {
            format!("{name} = if ({cond}) read(\"env:{k1}\") else read(\"env:{k2}\")\n")
        }
        Prop::Ref(name, alias) => format!("{name} = {alias}.id\n"),
    }
}

fn file_module_source(module: &ModSpec) -> String {
    let mut source = String::new();
    if let Some(target) = &module.amends {
        source.push_str(&format!("amends \"{}\"\n", target_uri(target)));
    }
    for import in &module.imports {
        source.push_str(&format!(
            "import \"{}\" as {}\n",
            target_uri(&import.target),
            import.alias
        ));
    }
    for prop in &module.props {
        source.push_str(&prop_source(prop));
    }
    source
}

fn http_module_source(idx: usize, spec: &HttpSpec) -> String {
    let mut source = String::new();
    if let Some(rel) = spec.rel_import {
        source.push_str(&format!("import \"./mod{rel}.pkl\"\n"));
    }
    source.push_str(&format!("id = {}\n", 100 + idx as i64));
    if let Some(rel) = spec.rel_import {
        source.push_str(&format!("ph{idx} = mod{rel}.id\n"));
    }
    if let Some(key) = &spec.env_read {
        source.push_str(&format!("he{idx} = read(\"env:{key}\")\n"));
    }
    source
}

fn pkg_entry_source(pkg: usize, entry: usize, env_read: &Option<String>) -> String {
    let mut source = format!("id = {}\n", 200 + 10 * pkg as i64 + entry as i64);
    if let Some(key) = env_read {
        source.push_str(&format!("pe{pkg}_{entry} = read(\"env:{key}\")\n"));
    }
    source
}

fn build_zip(pkg: &PkgSpec, pkg_idx: usize) -> Vec<u8> {
    use std::io::Write;
    let mut bytes = Vec::new();
    {
        let cursor = std::io::Cursor::new(&mut bytes);
        let mut writer = zip::ZipWriter::new(cursor);
        let options = zip::write::SimpleFileOptions::default();
        for (idx, env_read) in pkg.entries.iter().enumerate() {
            writer
                .start_file(format!("entry{idx}.pkl"), options)
                .unwrap();
            writer
                .write_all(pkg_entry_source(pkg_idx, idx, env_read).as_bytes())
                .unwrap();
        }
        writer.finish().unwrap();
    }
    bytes
}

/// FNV-1a, mirroring the evaluator's cache layout so the test can seed and
/// corrupt cache entries directly.
fn url_hash(url: &str) -> u64 {
    let mut hash = 0xcbf29ce484222325_u64;
    for byte in url.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

fn cache_meta_path(url: &str) -> String {
    format!("/cache/packages/{:016x}.json", url_hash(url))
}

fn materialize(case: &Case) -> FakeState {
    let mut state = FakeState::default();
    for dir in ["/", "/src", "/tmp"] {
        state.dirs.insert(dir.to_string());
    }
    for (idx, module) in case.files.iter().enumerate() {
        state.files.insert(
            file_path(idx),
            file_module_source(module).into_bytes(),
        );
    }
    for (idx, spec) in case.https.iter().enumerate() {
        let url = apply_rewrites(&case.rewrites, &http_url(idx));
        state.http_text.insert(url, http_module_source(idx, spec));
    }
    for (idx, pkg) in case.pkgs.iter().enumerate() {
        let url = apply_rewrites(&case.rewrites, &pkg_zip_url(idx));
        state.http_bytes.insert(url, build_zip(pkg, idx));
    }
    state.env = case.env.clone();
    state
}

/// Seed the persistent cache. Valid entries are created through the public
/// preload API (so the test does not duplicate the checksum format); corrupt
/// states mutate the seeded files afterwards.
fn seed_cache(case: &Case, state: &mut FakeState) {
    let needs_valid = |seed: Seed| !matches!(seed, Seed::Absent);
    for (idx, seed) in case.cache_seed.iter().enumerate() {
        if !needs_valid(*seed) {
            continue;
        }
        let url = pkg_zip_url(idx);
        let bytes = build_zip(&case.pkgs[idx], idx);
        let fake = Fake {
            state: Arc::new(Mutex::new(std::mem::take(state))),
            log: Arc::new(Mutex::new(Vec::new())),
        };
        let mut evaluator = Evaluator::with_capabilities(fake.clone());
        evaluator.set_package_cache_dir("/cache");
        pollster::block_on(evaluator.preload_package_async(&url, "zip", &bytes))
            .expect("seeding a valid cache entry must succeed");
        *state = fake.state.lock().unwrap().clone_for_seed();
        let meta_path = cache_meta_path(&url);
        let meta: Vec<u8> = state.files.get(&meta_path).cloned().unwrap_or_default();
        match seed {
            Seed::Valid | Seed::Absent => {}
            Seed::CorruptMeta => {
                state.files.insert(meta_path, b"{not-json".to_vec());
            }
            Seed::BadVersion => {
                let mut json: J = serde_json::from_slice(&meta).unwrap();
                json["version"] = J::from(99);
                state
                    .files
                    .insert(meta_path, serde_json::to_vec(&json).unwrap());
            }
            Seed::UrlMismatch => {
                let mut json: J = serde_json::from_slice(&meta).unwrap();
                json["url"] = J::from("https://other.test/foreign.zip");
                state
                    .files
                    .insert(meta_path, serde_json::to_vec(&json).unwrap());
            }
            Seed::BlobEscape => {
                let mut json: J = serde_json::from_slice(&meta).unwrap();
                json["blob"] = J::from("../evil.zip");
                state
                    .files
                    .insert(meta_path, serde_json::to_vec(&json).unwrap());
                // Trap: a valid zip outside the cache directory. The evaluator
                // must reject the escaping path, never read this file.
                state
                    .files
                    .insert("/cache/evil.zip".to_string(), bytes.clone());
            }
            Seed::ChecksumMismatch => {
                let json: J = serde_json::from_slice(&meta).unwrap();
                let blob = json["blob"].as_str().unwrap().to_string();
                let blob_path = format!("/cache/packages/{blob}");
                let mut data = state.files.get(&blob_path).cloned().unwrap();
                data[0] ^= 0xff;
                state.files.insert(blob_path, data);
            }
            Seed::MissingBlob => {
                let json: J = serde_json::from_slice(&meta).unwrap();
                let blob = json["blob"].as_str().unwrap().to_string();
                state.files.remove(&format!("/cache/packages/{blob}"));
            }
        }
    }
}

impl FakeState {
    fn clone_for_seed(&self) -> Self {
        FakeState {
            files: self.files.clone(),
            dirs: self.dirs.clone(),
            env: self.env.clone(),
            props: self.props.clone(),
            http_text: self.http_text.clone(),
            http_bytes: self.http_bytes.clone(),
        }
    }
}

// --- Simplified model evaluator ---------------------------------------------
//
// Mirrors pklr's observable behavior for the generated language subset:
// import caching with cycle placeholders, scoped amends with in-flight cycle
// detection, requested-field filtering, lazy imports, HTTP fetch caching,
// longest-prefix rewrites, the persistent package cache, preloads, and
// offline mode. Produces the expected JSON (or error) plus the expected
// projection of capability events.

#[derive(Clone, Debug, PartialEq, Eq)]
enum MEvent {
    FetchText(String),
    FetchBytes(String),
    WriteBlob(String),
    WriteMeta(String),
}

#[derive(Clone, Debug)]
struct MErr {
    category: &'static str,
    substr: String,
}

fn eval_err(substr: impl Into<String>) -> MErr {
    MErr {
        category: "eval",
        substr: substr.into(),
    }
}

struct Model<'a> {
    case: &'a Case,
    import_cache: HashMap<String, J>,
    in_flight: HashSet<String>,
    http_cache: HashSet<String>,
    pkg_dirs: HashMap<usize, String>,
    cache: Vec<Seed>,
    events: Vec<MEvent>,
    env_reads: BTreeMap<String, Option<String>>,
}

impl<'a> Model<'a> {
    fn new(case: &'a Case) -> Self {
        Self {
            case,
            import_cache: HashMap::new(),
            in_flight: HashSet::new(),
            http_cache: HashSet::new(),
            pkg_dirs: HashMap::new(),
            cache: case.cache_seed.clone(),
            events: Vec::new(),
            env_reads: BTreeMap::new(),
        }
    }

    fn run(&mut self) -> Result<J, MErr> {
        // Builder preloads happen before evaluation.
        for &pkg in &self.case.preloads {
            if self.cache[pkg] != Seed::Valid {
                let url = pkg_zip_url(pkg);
                self.events.push(MEvent::WriteBlob(url.clone()));
                self.events.push(MEvent::WriteMeta(url.clone()));
                self.cache[pkg] = Seed::Valid;
            }
        }
        let root = file_path(0);
        self.import_cache.insert(root.clone(), J::Object(Default::default()));
        let spec = self.case.files[0].clone();
        let value = self.eval_module(&spec, &root, None)?;
        self.import_cache.insert(root, value.clone());
        Ok(value)
    }

    fn http_modspec(&self, idx: usize) -> ModSpec {
        let spec = &self.case.https[idx];
        let mut module = ModSpec::default();
        module
            .props
            .push(Prop::Const("id".to_string(), 100 + idx as i64));
        if let Some(rel) = spec.rel_import {
            module.imports.push(Import {
                alias: format!("mod{rel}"),
                target: Target::Http(rel),
            });
            module
                .props
                .push(Prop::Ref(format!("ph{idx}"), format!("mod{rel}")));
        }
        if let Some(key) = &spec.env_read {
            module
                .props
                .push(Prop::Env(format!("he{idx}"), key.clone()));
        }
        module
    }

    fn entry_modspec(&self, pkg: usize, entry: usize) -> ModSpec {
        let mut module = ModSpec::default();
        module.props.push(Prop::Const(
            "id".to_string(),
            200 + 10 * pkg as i64 + entry as i64,
        ));
        if let Some(key) = &self.case.pkgs[pkg].entries[entry] {
            module
                .props
                .push(Prop::Env(format!("pe{pkg}_{entry}"), key.clone()));
        }
        module
    }

    fn eval_module(
        &mut self,
        spec: &ModSpec,
        path: &str,
        requested: Option<&BTreeSet<String>>,
    ) -> Result<J, MErr> {
        // pklr walks the amends chain for reference analysis before touching
        // imports; that pass performs the chain's fetches (and package
        // extractions) up front, so the model must replay it first.
        self.analysis_fetch(spec, path, 0)?;
        let active: Vec<&Prop> = spec
            .props
            .iter()
            .filter(|prop| requested.is_none_or(|req| req.contains(prop.name())))
            .collect();
        let referenced: HashSet<&str> = active
            .iter()
            .filter_map(|prop| match prop {
                Prop::Ref(_, alias) => Some(alias.as_str()),
                _ => None,
            })
            .collect();

        // Imports (lazy: only referenced aliases are evaluated). An alias is
        // referenced when an active property uses it, or when any module in
        // the amends chain references it (pklr's inherited_reference_roots
        // walks the unfiltered chain sources). Aliases referenced only
        // through the chain are evaluated in full (requested = None).
        let inherited = self.inherited_refs(spec, path);
        let mut scope: HashMap<String, J> = HashMap::new();
        for import in &spec.imports {
            let actively_used = referenced.contains(import.alias.as_str());
            if !actively_used && !inherited.contains(&import.alias) {
                continue;
            }
            let requested_import = actively_used.then(|| BTreeSet::from(["id".to_string()]));
            let value = self.eval_import(&import.target, path, requested_import.as_ref())?;
            scope.insert(import.alias.clone(), value);
        }

        // Amends.
        let mut out = match &spec.amends {
            Some(target) => self.eval_amends(target, path)?,
            None => J::Object(Default::default()),
        };

        // Own entries.
        let out_map = out.as_object_mut().expect("module value is an object");
        for prop in active {
            let value = self.eval_prop(prop, &scope)?;
            out_map.insert(prop.name().to_string(), value);
        }
        if let Some(req) = requested {
            out_map.retain(|name, _| req.contains(name));
        }
        Ok(out)
    }

    fn eval_import(
        &mut self,
        target: &Target,
        current_path: &str,
        requested: Option<&BTreeSet<String>>,
    ) -> Result<J, MErr> {
        match target {
            Target::File(idx) => {
                let path = join_str(&parent_str(current_path), &file_uri(*idx));
                let spec = self.case.files[*idx].clone();
                self.eval_file_cached(&spec, &path, requested)
            }
            Target::Http(idx) => {
                self.fetch_text(&http_url(*idx))?;
                let spec = self.http_modspec(*idx);
                let url = http_url(*idx);
                self.eval_module(&spec, &url, requested)
            }
            Target::Package(pkg, entry) => {
                self.ensure_package(*pkg)?;
                let dir = self.pkg_dirs[pkg].clone();
                let local = format!("{dir}/entry{entry}.pkl");
                let spec = self.entry_modspec(*pkg, *entry);
                self.eval_file_cached(&spec, &local, requested)
            }
            Target::Missing => Err(MErr {
                category: "import-not-found",
                substr: String::new(),
            }),
        }
    }

    /// Import-cache semantics: a placeholder breaks cycles. Fully requested
    /// imports stay in the cache; requested-field imports are removed after
    /// evaluation, and failures always evict the placeholder.
    fn eval_file_cached(
        &mut self,
        spec: &ModSpec,
        path: &str,
        requested: Option<&BTreeSet<String>>,
    ) -> Result<J, MErr> {
        if let Some(cached) = self.import_cache.get(path) {
            return Ok(cached.clone());
        }
        self.import_cache
            .insert(path.to_string(), J::Object(Default::default()));
        let result = self.eval_module(spec, path, requested);
        match (requested, result) {
            (None, Ok(value)) => {
                self.import_cache.insert(path.to_string(), value.clone());
                Ok(value)
            }
            (None, Err(error)) => {
                self.import_cache.remove(path);
                Err(error)
            }
            (Some(_), result) => {
                self.import_cache.remove(path);
                result
            }
        }
    }

    /// Aliases referenced anywhere in the amends chain's unfiltered module
    /// bodies, mirroring pklr's inherited_reference_roots walk.
    fn inherited_refs(&self, spec: &ModSpec, path: &str) -> HashSet<String> {
        let mut out = HashSet::new();
        self.collect_inherited_refs(spec, path, 0, &mut out);
        out
    }

    fn collect_inherited_refs(
        &self,
        spec: &ModSpec,
        path: &str,
        depth: usize,
        out: &mut HashSet<String>,
    ) {
        if depth > 32 {
            return;
        }
        let Some(target) = &spec.amends else {
            return;
        };
        let (chain_spec, next_path) = match target {
            Target::File(idx) => (
                self.case.files[*idx].clone(),
                join_str(&parent_str(path), &file_uri(*idx)),
            ),
            Target::Http(idx) => (self.http_modspec(*idx), http_url(*idx)),
            Target::Package(pkg, entry) => {
                let dir = format!("/tmp/pklr-pkg-{}", self.pkg_dirs.len());
                (self.entry_modspec(*pkg, *entry), format!("{dir}/entry{entry}.pkl"))
            }
            Target::Missing => return,
        };
        for prop in &chain_spec.props {
            if let Prop::Ref(_, alias) = prop {
                out.insert(alias.clone());
            }
        }
        self.collect_inherited_refs(&chain_spec, &next_path, depth + 1, out);
    }

    fn eval_amends(&mut self, target: &Target, current_path: &str) -> Result<J, MErr> {
        match target {
            Target::File(idx) => {
                let path = join_str(&parent_str(current_path), &file_uri(*idx));
                if !self.in_flight.insert(path.clone()) {
                    return Ok(J::Object(Default::default()));
                }
                let spec = self.case.files[*idx].clone();
                let result = self.eval_module(&spec, &path, None);
                self.in_flight.remove(&path);
                result
            }
            Target::Http(idx) => {
                self.fetch_text(&http_url(*idx))?;
                let spec = self.http_modspec(*idx);
                let url = http_url(*idx);
                self.eval_module(&spec, &url, None)
            }
            Target::Package(pkg, entry) => {
                self.ensure_package(*pkg)?;
                let dir = self.pkg_dirs[pkg].clone();
                let local = format!("{dir}/entry{entry}.pkl");
                let spec = self.entry_modspec(*pkg, *entry);
                self.eval_module(&spec, &local, None)
            }
            Target::Missing => Ok(J::Object(Default::default())),
        }
    }

    fn eval_prop(&mut self, prop: &Prop, scope: &HashMap<String, J>) -> Result<J, MErr> {
        match prop {
            Prop::Const(_, value) => Ok(J::from(*value)),
            Prop::Env(_, key) => self.read_env(key),
            Prop::CondEnv(_, cond, k1, k2) => {
                let key = if *cond { k1 } else { k2 };
                self.read_env(key)
            }
            Prop::Ref(_, alias) => {
                let module = scope.get(alias).ok_or_else(|| {
                    eval_err(format!("undefined variable: {alias}"))
                })?;
                module
                    .get("id")
                    .cloned()
                    .ok_or_else(|| eval_err("field not found: id"))
            }
        }
    }

    fn read_env(&mut self, key: &str) -> Result<J, MErr> {
        let value = self.case.env.get(key).cloned();
        self.env_reads.insert(key.to_string(), value.clone());
        value
            .map(J::from)
            .ok_or_else(|| eval_err(format!("environment variable not found: {key}")))
    }

    fn fetch_text(&mut self, url: &str) -> Result<(), MErr> {
        let rewritten = apply_rewrites(&self.case.rewrites, url);
        if self.http_cache.contains(&rewritten) {
            return Ok(());
        }
        if self.case.offline {
            return Err(eval_err(format!(
                "offline mode prevented HTTP fetch for {url}"
            )));
        }
        self.events.push(MEvent::FetchText(rewritten.clone()));
        self.http_cache.insert(rewritten);
        Ok(())
    }

    /// Replay the fetch side effects of pklr's amends-chain reference
    /// analysis (`inherited_reference_roots`): before a module's imports are
    /// processed, the whole amends chain is walked and its http/package
    /// sources are fetched (and zips extracted, allocating temp dirs).
    fn analysis_fetch(&mut self, spec: &ModSpec, path: &str, depth: usize) -> Result<(), MErr> {
        if depth > 32 {
            return Ok(());
        }
        let Some(target) = &spec.amends else {
            return Ok(());
        };
        match target {
            Target::Http(idx) => {
                self.fetch_text(&http_url(*idx))?;
                let spec = self.http_modspec(*idx);
                self.analysis_fetch(&spec, &http_url(*idx), depth + 1)
            }
            Target::Package(pkg, entry) => {
                self.ensure_package(*pkg)?;
                let dir = self.pkg_dirs[pkg].clone();
                let local = format!("{dir}/entry{entry}.pkl");
                let spec = self.entry_modspec(*pkg, *entry);
                self.analysis_fetch(&spec, &local, depth + 1)
            }
            Target::File(idx) => {
                let next = join_str(&parent_str(path), &file_uri(*idx));
                let spec = self.case.files[*idx].clone();
                self.analysis_fetch(&spec, &next, depth + 1)
            }
            Target::Missing => Ok(()),
        }
    }

    fn ensure_package(&mut self, pkg: usize) -> Result<(), MErr> {
        if self.pkg_dirs.contains_key(&pkg) {
            return Ok(());
        }
        let url = pkg_zip_url(pkg);
        match self.cache[pkg] {
            Seed::Valid => {}
            Seed::Absent | Seed::UrlMismatch | Seed::MissingBlob => {
                if self.case.offline {
                    return Err(eval_err(
                        "package is not cached and offline mode is enabled",
                    ));
                }
                self.fetch_package(pkg);
            }
            Seed::CorruptMeta => {
                if self.case.offline {
                    return Err(eval_err("corrupt package cache metadata"));
                }
                self.fetch_package(pkg);
            }
            Seed::BadVersion => {
                if self.case.offline {
                    return Err(eval_err("unsupported package cache metadata version"));
                }
                self.fetch_package(pkg);
            }
            Seed::BlobEscape => {
                if self.case.offline {
                    return Err(eval_err("escapes the cache directory"));
                }
                self.fetch_package(pkg);
            }
            Seed::ChecksumMismatch => {
                if self.case.offline {
                    return Err(eval_err("package cache checksum mismatch"));
                }
                self.fetch_package(pkg);
            }
        }
        let dir = format!("/tmp/pklr-pkg-{}", self.pkg_dirs.len());
        self.pkg_dirs.insert(pkg, dir);
        Ok(())
    }

    fn fetch_package(&mut self, pkg: usize) {
        let url = pkg_zip_url(pkg);
        let rewritten = apply_rewrites(&self.case.rewrites, &url);
        self.events.push(MEvent::FetchBytes(rewritten));
        self.events.push(MEvent::WriteBlob(url.clone()));
        self.events.push(MEvent::WriteMeta(url));
        self.cache[pkg] = Seed::Valid;
    }
}

// --- Implementation runner ----------------------------------------------------

struct Run {
    json: Result<J, (String, String)>,
    events: Vec<Event>,
    env_reads: BTreeMap<String, Option<String>>,
    prop_reads: BTreeMap<String, Option<String>>,
}

fn error_category(error: &Error) -> &'static str {
    match error {
        Error::Io(..) => "io",
        Error::Lex { .. } => "lex",
        Error::Parse { .. } => "parse",
        Error::Eval(_) => "eval",
        Error::ImportNotFound(_) => "import-not-found",
        Error::Unsupported(_) => "unsupported",
    }
}

fn build_evaluator(fake: Fake, case: &Case) -> Evaluator {
    let mut evaluator = Evaluator::with_capabilities(fake);
    evaluator.set_base_path(Path::new("/src"));
    let rules: Vec<String> = case
        .rewrites
        .iter()
        .map(|(src, tgt)| format!("{src}={tgt}"))
        .collect();
    evaluator.set_http_rewrites(&rules);
    evaluator.set_package_cache_dir("/cache");
    evaluator.set_offline(case.offline);
    evaluator
}

async fn eval_async(mut evaluator: Evaluator, case: &Case) -> (Result<J, Error>, Evaluator) {
    for &pkg in &case.preloads {
        let bytes = build_zip(&case.pkgs[pkg], pkg);
        evaluator
            .preload_package_async(&pkg_zip_url(pkg), "zip", &bytes)
            .await
            .expect("generated preload bytes are valid");
    }
    let result = evaluator
        .eval_file_pub(Path::new("/src/mod0.pkl"))
        .await
        .map(|value| value.to_json());
    (result, evaluator)
}

fn run_impl(case: &Case, mode: Mode) -> Run {
    let mut state = materialize(case);
    seed_cache(case, &mut state);
    let fake = Fake {
        state: Arc::new(Mutex::new(state)),
        log: Arc::new(Mutex::new(Vec::new())),
    };
    let events = fake.log.clone();
    let evaluator = build_evaluator(fake, case);
    let (result, mut evaluator) = match mode {
        Mode::Sync => pollster::block_on(eval_async(evaluator, case)),
        Mode::Async => tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
            .block_on(eval_async(evaluator, case)),
    };
    Run {
        json: result.map_err(|error| (error_category(&error).to_string(), error.to_string())),
        events: events.lock().unwrap().clone(),
        env_reads: evaluator.env_reads().clone(),
        prop_reads: evaluator.prop_reads().clone(),
    }
}

#[derive(Clone, Copy)]
enum Mode {
    Sync,
    Async,
}

/// Project the raw capability log onto the events the model tracks.
fn projection(events: &[Event], case: &Case) -> Vec<MEvent> {
    let hash_to_url: HashMap<String, String> = (0..case.pkgs.len())
        .map(|idx| (format!("{:016x}", url_hash(&pkg_zip_url(idx))), pkg_zip_url(idx)))
        .collect();
    let mut out = Vec::new();
    for event in events {
        match event {
            Event::FetchText(url) => out.push(MEvent::FetchText(url.clone())),
            Event::FetchBytes(url) => out.push(MEvent::FetchBytes(url.clone())),
            Event::WriteAtomic(path) => {
                let name = path.rsplit('/').next().unwrap_or(path);
                if let Some(hash) = name.strip_suffix(".json")
                    && let Some(url) = hash_to_url.get(hash)
                {
                    out.push(MEvent::WriteMeta(url.clone()));
                    continue;
                }
                let hash = name.split('-').next().unwrap_or("");
                if let Some(url) = hash_to_url.get(hash) {
                    out.push(MEvent::WriteBlob(url.clone()));
                }
            }
            _ => {}
        }
    }
    out
}

fn check_case(case: &Case) -> Result<(), String> {
    let mut model = Model::new(case);
    let expected = model.run();
    let run = run_impl(case, Mode::Sync);

    match (&expected, &run.json) {
        (Ok(expected_json), Ok(actual_json)) => {
            if expected_json != actual_json {
                return Err(format!(
                    "json mismatch\nmodel: {}\nimpl:  {}",
                    serde_json::to_string_pretty(expected_json).unwrap(),
                    serde_json::to_string_pretty(actual_json).unwrap()
                ));
            }
        }
        (Err(expected_err), Err((category, message))) => {
            if &expected_err.category != category {
                return Err(format!(
                    "error category mismatch: model={} impl={} ({message})",
                    expected_err.category, category
                ));
            }
            if !expected_err.substr.is_empty() && !message.contains(&expected_err.substr) {
                return Err(format!(
                    "error message mismatch: expected substring {:?} in {message:?}",
                    expected_err.substr
                ));
            }
        }
        (expected, actual) => {
            return Err(format!(
                "outcome mismatch: model={} impl={}",
                match expected {
                    Ok(_) => "ok".to_string(),
                    Err(error) => format!("err({}: {})", error.category, error.substr),
                },
                match actual {
                    Ok(json) => format!("ok({json})"),
                    Err((category, message)) => format!("err({category}: {message})"),
                }
            ));
        }
    }

    if model.env_reads != run.env_reads {
        return Err(format!(
            "env read mismatch\nmodel: {:?}\nimpl:  {:?}",
            model.env_reads, run.env_reads
        ));
    }

    let impl_events = projection(&run.events, case);
    if std::env::var("PKLR_MODEL_DEBUG").is_ok() {
        eprintln!(
            "case: pkgs={} seed={:?} preloads={:?} offline={} amends={:?} imports={:?} model_events={:?} impl_events={:?}",
            case.pkgs.len(), case.cache_seed, case.preloads, case.offline,
            case.files.iter().map(|m| m.amends.clone()).collect::<Vec<_>>(),
            case.files.iter().map(|m| m.imports.iter().map(|i| i.target.clone()).collect::<Vec<_>>()).collect::<Vec<_>>(),
            model.events, impl_events
        );
    }
    if impl_events != model.events {
        return Err(format!(
            "event projection mismatch\nmodel: {:?}\nimpl:  {:?}",
            model.events, impl_events
        ));
    }

    // The same case through the async API must produce an equivalent event
    // log once scheduling-only fields are ignored.
    let async_run = run_impl(case, Mode::Async);
    if normalize_log(&run.events) != normalize_log(&async_run.events) {
        return Err(format!(
            "sync/async event log mismatch\nsync:  {:?}\nasync: {:?}",
            normalize_log(&run.events),
            normalize_log(&async_run.events)
        ));
    }
    if run.json != async_run.json
        || run.env_reads != async_run.env_reads
        || run.prop_reads != async_run.prop_reads
    {
        return Err("sync/async outcome mismatch".to_string());
    }
    Ok(())
}

// --- Replay serialization -----------------------------------------------------

fn target_to_json(target: &Target) -> J {
    match target {
        Target::File(idx) => json!({"file": idx}),
        Target::Http(idx) => json!({"http": idx}),
        Target::Package(pkg, entry) => json!({"package": [pkg, entry]}),
        Target::Missing => json!("missing"),
    }
}

fn target_from_json(value: &J) -> Target {
    if value == "missing" {
        return Target::Missing;
    }
    if let Some(idx) = value.get("file") {
        return Target::File(idx.as_u64().unwrap() as usize);
    }
    if let Some(idx) = value.get("http") {
        return Target::Http(idx.as_u64().unwrap() as usize);
    }
    let pair = value["package"].as_array().unwrap();
    Target::Package(pair[0].as_u64().unwrap() as usize, pair[1].as_u64().unwrap() as usize)
}

fn prop_to_json(prop: &Prop) -> J {
    match prop {
        Prop::Const(name, value) => json!({"const": [name, value]}),
        Prop::Env(name, key) => json!({"env": [name, key]}),
        Prop::CondEnv(name, cond, k1, k2) => json!({"cond": [name, cond, k1, k2]}),
        Prop::Ref(name, alias) => json!({"ref": [name, alias]}),
    }
}

fn prop_from_json(value: &J) -> Prop {
    if let Some(pair) = value.get("const") {
        return Prop::Const(
            pair[0].as_str().unwrap().to_string(),
            pair[1].as_i64().unwrap(),
        );
    }
    if let Some(pair) = value.get("env") {
        return Prop::Env(
            pair[0].as_str().unwrap().to_string(),
            pair[1].as_str().unwrap().to_string(),
        );
    }
    if let Some(pair) = value.get("cond") {
        return Prop::CondEnv(
            pair[0].as_str().unwrap().to_string(),
            pair[1].as_bool().unwrap(),
            pair[2].as_str().unwrap().to_string(),
            pair[3].as_str().unwrap().to_string(),
        );
    }
    let pair = value["ref"].as_array().unwrap();
    Prop::Ref(
        pair[0].as_str().unwrap().to_string(),
        pair[1].as_str().unwrap().to_string(),
    )
}

fn modspec_to_json(module: &ModSpec) -> J {
    json!({
        "amends": module.amends.as_ref().map(target_to_json),
        "imports": module.imports.iter().map(|import| json!({
            "alias": import.alias,
            "target": target_to_json(&import.target),
        })).collect::<Vec<_>>(),
        "props": module.props.iter().map(prop_to_json).collect::<Vec<_>>(),
    })
}

fn modspec_from_json(value: &J) -> ModSpec {
    ModSpec {
        amends: value.get("amends").and_then(|v| {
            if v.is_null() {
                None
            } else {
                Some(target_from_json(v))
            }
        }),
        imports: value["imports"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| Import {
                alias: v["alias"].as_str().unwrap().to_string(),
                target: target_from_json(&v["target"]),
            })
            .collect(),
        props: value["props"]
            .as_array()
            .unwrap()
            .iter()
            .map(prop_from_json)
            .collect(),
    }
}

fn seed_to_json(seed: Seed) -> J {
    J::from(match seed {
        Seed::Absent => "absent",
        Seed::Valid => "valid",
        Seed::CorruptMeta => "corrupt-meta",
        Seed::BadVersion => "bad-version",
        Seed::UrlMismatch => "url-mismatch",
        Seed::BlobEscape => "blob-escape",
        Seed::ChecksumMismatch => "checksum-mismatch",
        Seed::MissingBlob => "missing-blob",
    })
}

fn seed_from_json(value: &J) -> Seed {
    match value.as_str().unwrap() {
        "absent" => Seed::Absent,
        "valid" => Seed::Valid,
        "corrupt-meta" => Seed::CorruptMeta,
        "bad-version" => Seed::BadVersion,
        "url-mismatch" => Seed::UrlMismatch,
        "blob-escape" => Seed::BlobEscape,
        "checksum-mismatch" => Seed::ChecksumMismatch,
        "missing-blob" => Seed::MissingBlob,
        other => panic!("unknown seed state {other}"),
    }
}

fn case_to_json(case: &Case) -> J {
    json!({
        "files": case.files.iter().map(modspec_to_json).collect::<Vec<_>>(),
        "https": case.https.iter().map(|spec| json!({
            "env_read": spec.env_read,
            "rel_import": spec.rel_import,
        })).collect::<Vec<_>>(),
        "pkgs": case.pkgs.iter().map(|pkg| json!({
            "entries": pkg.entries,
        })).collect::<Vec<_>>(),
        "rewrites": case.rewrites,
        "cache_seed": case.cache_seed.iter().map(|seed| seed_to_json(*seed)).collect::<Vec<_>>(),
        "preloads": case.preloads,
        "offline": case.offline,
        "env": case.env,
    })
}

fn case_from_json(value: &J) -> Case {
    Case {
        files: value["files"]
            .as_array()
            .unwrap()
            .iter()
            .map(modspec_from_json)
            .collect(),
        https: value["https"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| HttpSpec {
                env_read: v["env_read"].as_str().map(str::to_string),
                rel_import: v["rel_import"].as_u64().map(|idx| idx as usize),
            })
            .collect(),
        pkgs: value["pkgs"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| PkgSpec {
                entries: v["entries"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|e| e.as_str().map(str::to_string))
                    .collect(),
            })
            .collect(),
        rewrites: value["rewrites"]
            .as_array()
            .unwrap()
            .iter()
            .map(|pair| {
                (
                    pair[0].as_str().unwrap().to_string(),
                    pair[1].as_str().unwrap().to_string(),
                )
            })
            .collect(),
        cache_seed: value["cache_seed"]
            .as_array()
            .unwrap()
            .iter()
            .map(seed_from_json)
            .collect(),
        preloads: value["preloads"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect(),
        offline: value["offline"].as_bool().unwrap(),
        env: value["env"]
            .as_object()
            .unwrap()
            .iter()
            .map(|(k, v)| (k.clone(), v.as_str().unwrap().to_string()))
            .collect(),
    }
}

// --- Shrinking: reduce a failing case to a minimal replayable one -------------

fn reductions(case: &Case) -> Vec<Case> {
    let mut out = Vec::new();
    // Remove individual properties (keep at least one per module).
    for (m, module) in case.files.iter().enumerate() {
        for p in 0..module.props.len() {
            if module.props.len() <= 1 {
                break;
            }
            let mut candidate = case.clone();
            candidate.files[m].props.remove(p);
            out.push(candidate);
        }
        for i in 0..module.imports.len() {
            let mut candidate = case.clone();
            candidate.files[m].imports.remove(i);
            out.push(candidate);
        }
        if module.amends.is_some() {
            let mut candidate = case.clone();
            candidate.files[m].amends = None;
            out.push(candidate);
        }
    }
    for i in 0..case.rewrites.len() {
        let mut candidate = case.clone();
        candidate.rewrites.remove(i);
        out.push(candidate);
    }
    for i in 0..case.preloads.len() {
        let mut candidate = case.clone();
        candidate.preloads.remove(i);
        out.push(candidate);
    }
    for i in 0..case.cache_seed.len() {
        if case.cache_seed[i] != Seed::Absent {
            let mut candidate = case.clone();
            candidate.cache_seed[i] = Seed::Absent;
            out.push(candidate);
        }
    }
    for key in case.env.keys() {
        let mut candidate = case.clone();
        candidate.env.remove(key);
        out.push(candidate);
    }
    out
}

fn shrink(case: Case) -> Case {
    let mut best = case;
    loop {
        let mut improved = false;
        for candidate in reductions(&best) {
            if check_case(&candidate).is_err() {
                best = candidate;
                improved = true;
                break;
            }
        }
        if !improved {
            return best;
        }
    }
}

// --- Tests ---------------------------------------------------------------------

fn env_or<T: std::str::FromStr>(key: &str, default: T) -> T {
    std::env::var(key)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

#[test]
fn model_contract_generated_cases() {
    if let Ok(replay) = std::env::var("PKLR_MODEL_REPLAY") {
        let case = case_from_json(&serde_json::from_str(&replay).expect("valid replay json"));
        check_case(&case).expect("replayed case should pass");
        return;
    }
    let seed = env_or("PKLR_MODEL_SEED", DEFAULT_SEED);
    let cases = env_or("PKLR_MODEL_CASES", DEFAULT_CASES);
    let mut rng = Rng(seed);
    let mut stats = CaseStats::default();
    for idx in 0..cases {
        let case = generate(&mut rng);
        stats.observe(&case);
        if let Err(report) = check_case(&case) {
            let original = serde_json::to_string(&case_to_json(&case)).unwrap();
            let shrunk = shrink(case);
            let replay = serde_json::to_string(&case_to_json(&shrunk)).unwrap();
            panic!(
                "model contract violation in case {idx} (seed {seed:#x}):\n{report}\n\
                 original case:\n{original}\n\
                 minimal replayable case (PKLR_MODEL_REPLAY='{}'):\n{}",
                replay, replay
            );
        }
    }
    eprintln!("model contract stats: {stats}");
}

#[derive(Default)]
struct CaseStats {
    total: usize,
    with_packages: usize,
    with_http: usize,
    offline: usize,
    with_rewrites: usize,
    with_preloads: usize,
    with_cache_states: usize,
    with_amends: usize,
    with_cycles: usize,
    errors: usize,
}

impl CaseStats {
    fn observe(&mut self, case: &Case) {
        self.total += 1;
        self.with_packages += usize::from(!case.pkgs.is_empty());
        self.with_http += usize::from(!case.https.is_empty());
        self.offline += usize::from(case.offline);
        self.with_rewrites += usize::from(!case.rewrites.is_empty());
        self.with_preloads += usize::from(!case.preloads.is_empty());
        self.with_cache_states +=
            usize::from(case.cache_seed.iter().any(|seed| *seed != Seed::Absent));
        self.with_amends += usize::from(case.files.iter().any(|m| m.amends.is_some()));
        self.with_cycles += usize::from(has_amend_or_import_cycle(case));
        let mut model = Model::new(case);
        self.errors += usize::from(model.run().is_err());
    }
}

impl std::fmt::Display for CaseStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} cases (packages={}, http={}, offline={}, rewrites={}, preloads={}, cache_states={}, amends={}, cycles={}, errors={})",
            self.total,
            self.with_packages,
            self.with_http,
            self.offline,
            self.with_rewrites,
            self.with_preloads,
            self.with_cache_states,
            self.with_amends,
            self.with_cycles,
            self.errors
        )
    }
}

/// Detect import/amend cycles among file modules (for coverage stats).
fn has_amend_or_import_cycle(case: &Case) -> bool {
    fn visit(case: &Case, node: usize, stack: &mut Vec<usize>) -> bool {
        if stack.contains(&node) {
            return true;
        }
        stack.push(node);
        let module = &case.files[node];
        let mut targets = module
            .imports
            .iter()
            .filter_map(|import| match import.target {
                Target::File(idx) => Some(idx),
                _ => None,
            })
            .collect::<Vec<_>>();
        if let Some(Target::File(idx)) = module.amends {
            targets.push(idx);
        }
        let found = targets.into_iter().any(|next| visit(case, next, stack));
        stack.pop();
        found
    }
    (0..case.files.len()).any(|idx| visit(case, idx, &mut Vec::new()))
}

#[test]
fn sync_and_async_apis_agree_on_a_real_filesystem_case() {
    // The public blocking and async entry points must agree on identical
    // input. Uses a file-only case so no network or system temp dirs are
    // involved; everything lives under one reclaimed temp dir.
    let dir = std::env::temp_dir().join(format!(
        "pklr-model-sync-async-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("base.pkl"), "id = 1\nname = \"base\"\n").unwrap();
    std::fs::write(
        dir.join("main.pkl"),
        "amends \"base.pkl\"\nimport \"base.pkl\" as b\nid = 2\nfrom_import = b.id\n",
    )
    .unwrap();
    let main = dir.join("main.pkl");

    let sync_json = pklr::eval_to_json(&main).unwrap();
    let async_json = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
        .block_on(pklr::eval_to_json_async(&main))
        .unwrap();
    assert_eq!(sync_json, async_json);
    assert_eq!(sync_json["id"], 2);
    assert_eq!(sync_json["from_import"], 1);
    assert_eq!(sync_json["name"], "base");

    std::fs::remove_dir_all(&dir).unwrap();
    assert!(!dir.exists(), "temp dir must be reclaimed");
}

// --- Environment and property read contracts ----------------------------------

fn eval_source_with_fake(
    files: &[(&str, &str)],
    env: &[(&str, &str)],
    props: &[(&str, &str)],
    root: &str,
) -> Run {
    let mut state = FakeState::default();
    for dir in ["/", "/src"] {
        state.dirs.insert(dir.to_string());
    }
    for (path, source) in files {
        state.files.insert(path.to_string(), source.as_bytes().to_vec());
    }
    for (key, value) in env {
        state.env.insert(key.to_string(), value.to_string());
    }
    for (key, value) in props {
        state.props.insert(key.to_string(), value.to_string());
    }
    let fake = Fake {
        state: Arc::new(Mutex::new(state)),
        log: Arc::new(Mutex::new(Vec::new())),
    };
    let events = fake.log.clone();
    let mut evaluator = Evaluator::with_capabilities(fake);
    evaluator.set_base_path(Path::new("/src"));
    let root = root.to_string();
    let (result, evaluator) = pollster::block_on(async move {
        let result = evaluator
            .eval_file_pub(Path::new(&root))
            .await
            .map(|value| value.to_json());
        (result, evaluator)
    });
    let mut evaluator = evaluator;
    Run {
        json: result.map_err(|error| (error_category(&error).to_string(), error.to_string())),
        events: events.lock().unwrap().clone(),
        env_reads: evaluator.env_reads().clone(),
        prop_reads: evaluator.prop_reads().clone(),
    }
}

#[test]
fn env_reads_only_record_the_evaluated_branch() {
    let run = eval_source_with_fake(
        &[(
            "/src/main.pkl",
            "a = if (false) read(\"env:PKLR_MC_A\") else \"x\"\n\
             b = if (true) \"y\" else read(\"env:PKLR_MC_B\")\n\
             c = false && read(\"env:PKLR_MC_C\") == \"1\"\n\
             d = true || read(\"env:PKLR_MC_D\") == \"1\"\n\
             e = read(\"env:PKLR_MC_E\")\n",
        )],
        &[("PKLR_MC_E", "present")],
        &[],
        "/src/main.pkl",
    );
    let json = run.json.expect("evaluation succeeds");
    assert_eq!(json["e"], "present");
    assert_eq!(
        run.env_reads,
        BTreeMap::from([("PKLR_MC_E".to_string(), Some("present".to_string()))]),
        "short-circuited branches must not register env dependencies"
    );
}

#[test]
fn missing_env_var_is_recorded_with_none() {
    let run = eval_source_with_fake(
        &[("/src/main.pkl", "a = read?(\"env:PKLR_MC_MISSING\")\n")],
        &[],
        &[],
        "/src/main.pkl",
    );
    assert!(run.json.is_ok());
    assert_eq!(
        run.env_reads,
        BTreeMap::from([("PKLR_MC_MISSING".to_string(), None)])
    );
}

#[test]
fn unreferenced_import_is_never_read() {
    let run = eval_source_with_fake(
        &[
            ("/src/main.pkl", "import \"dep.pkl\" as dep\nx = 1\n"),
            ("/src/dep.pkl", "secret = read(\"env:PKLR_MC_DEP\")\n"),
        ],
        &[],
        &[],
        "/src/main.pkl",
    );
    assert!(run.json.is_ok());
    assert!(run.env_reads.is_empty());
    assert!(
        !run.events
            .iter()
            .any(|event| matches!(event, Event::ReadFile(path) if path == "/src/dep.pkl")),
        "unreferenced import must not even be read: {:?}",
        run.events
    );
}

#[test]
fn unrequested_import_fields_do_not_register_env_reads() {
    let run = eval_source_with_fake(
        &[
            (
                "/src/main.pkl",
                "import \"dep.pkl\" as dep\nx = dep.id\n",
            ),
            (
                "/src/dep.pkl",
                "id = 7\nsecret = read(\"env:PKLR_MC_SECRET\")\n",
            ),
        ],
        &[],
        &[],
        "/src/main.pkl",
    );
    let json = run.json.expect("evaluation succeeds");
    assert_eq!(json["x"], 7);
    assert!(
        run.env_reads.is_empty(),
        "fields outside the requested set must not be evaluated: {:?}",
        run.env_reads
    );
}

#[test]
fn import_cache_hit_reads_env_only_once() {
    let run = eval_source_with_fake(
        &[
            (
                "/src/main.pkl",
                "import \"a.pkl\" as a\nimport \"b.pkl\" as b\nx = a\ny = b\n",
            ),
            ("/src/a.pkl", "import \"c.pkl\" as c\nav = c\n"),
            ("/src/b.pkl", "import \"c.pkl\" as c\nbv = c\n"),
            ("/src/c.pkl", "cv = read(\"env:PKLR_MC_ONCE\")\n"),
        ],
        &[("PKLR_MC_ONCE", "1")],
        &[],
        "/src/main.pkl",
    );
    assert!(run.json.is_ok());
    let env_read_count = run
        .events
        .iter()
        .filter(|event| matches!(event, Event::ReadEnv(key) if key == "PKLR_MC_ONCE"))
        .count();
    assert_eq!(
        env_read_count, 1,
        "a cached import must evaluate (and read env) only once: {:?}",
        run.events
    );
}

#[test]
fn retry_after_failure_does_not_leak_previous_reads() {
    let files: &[(&str, &str)] = &[
        (
            "/src/first.pkl",
            "a = read(\"env:PKLR_MC_FIRST\")\nb = read(\"env:PKLR_MC_GONE\")\n",
        ),
        ("/src/second.pkl", "c = read(\"env:PKLR_MC_SECOND\")\n"),
    ];
    let mut state = FakeState::default();
    state.dirs.insert("/".into());
    state.dirs.insert("/src".into());
    for (path, source) in files {
        state.files.insert(path.to_string(), source.as_bytes().to_vec());
    }
    state
        .env
        .insert("PKLR_MC_FIRST".to_string(), "1".to_string());
    state
        .env
        .insert("PKLR_MC_SECOND".to_string(), "2".to_string());
    let fake = Fake {
        state: Arc::new(Mutex::new(state)),
        log: Arc::new(Mutex::new(Vec::new())),
    };
    let mut evaluator = Evaluator::with_capabilities(fake);
    evaluator.set_base_path(Path::new("/src"));

    pollster::block_on(async {
        // First evaluation fails on the missing variable.
        let first = evaluator.eval_file_pub(Path::new("/src/first.pkl")).await;
        assert!(first.is_err());
        assert!(evaluator.env_reads().contains_key("PKLR_MC_FIRST"));

        // The retry (a different case) must not leak the previous read set.
        evaluator
            .eval_file_pub(Path::new("/src/second.pkl"))
            .await
            .unwrap();
        assert_eq!(
            evaluator.env_reads().clone(),
            BTreeMap::from([("PKLR_MC_SECOND".to_string(), Some("2".to_string()))]),
            "reads from the failed evaluation must not leak into the retry"
        );
    });
}

#[test]
fn prop_reads_are_tracked_like_env_reads() {
    let run = eval_source_with_fake(
        &[(
            "/src/main.pkl",
            "a = read(\"prop:pklr.mc.answer\")\n\
             b = if (false) read(\"prop:pklr.mc.skipped\") else \"z\"\n",
        )],
        &[],
        &[("pklr.mc.answer", "42")],
        "/src/main.pkl",
    );
    let json = run.json.expect("evaluation succeeds");
    assert_eq!(json["a"], "42");
    assert_eq!(
        run.prop_reads,
        BTreeMap::from([("pklr.mc.answer".to_string(), Some("42".to_string()))]),
        "only properties on the evaluated path may be recorded"
    );
    assert!(run.env_reads.is_empty());
}

#[test]
fn missing_prop_is_recorded_with_none() {
    let run = eval_source_with_fake(
        &[("/src/main.pkl", "a = read?(\"prop:pklr.mc.missing\")\n")],
        &[],
        &[],
        "/src/main.pkl",
    );
    assert!(run.json.is_ok());
    assert_eq!(
        run.prop_reads,
        BTreeMap::from([("pklr.mc.missing".to_string(), None)])
    );
}

#[test]
fn default_capabilities_still_reject_prop_reads() {
    // Public semantics are unchanged: without an injected property source,
    // `prop:` reads keep failing.
    let dir = std::env::temp_dir().join(format!("pklr-model-prop-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let main = dir.join("main.pkl");
    std::fs::write(&main, "a = read(\"prop:pklr.mc.unsupported\")\n").unwrap();
    let error = pklr::eval_to_json(&main).unwrap_err().to_string();
    assert!(
        error.contains("system property not available"),
        "unexpected error: {error}"
    );
    std::fs::remove_dir_all(&dir).unwrap();
    assert!(!dir.exists(), "temp dir must be reclaimed");
}

#[test]
fn rewrite_longest_prefix_wins() {
    let mut evaluator = Evaluator::with_capabilities(Fake::default());
    evaluator.set_http_rewrites(&[
        "https://a.test/=https://short.test/".to_string(),
        "https://a.test/deep/=https://long.test/".to_string(),
        "https://a.test/deep/deeper/=https://longest.test/".to_string(),
    ]);
    assert_eq!(
        evaluator.rewrite_url("https://a.test/x.pkl"),
        "https://short.test/x.pkl"
    );
    assert_eq!(
        evaluator.rewrite_url("https://a.test/deep/x.pkl"),
        "https://long.test/x.pkl"
    );
    assert_eq!(
        evaluator.rewrite_url("https://a.test/deep/deeper/x.pkl"),
        "https://longest.test/x.pkl"
    );
    assert_eq!(
        evaluator.rewrite_url("https://other.test/x.pkl"),
        "https://other.test/x.pkl"
    );
}

#[test]
fn debug_repro_case_187() {
    if std::env::var("PKLR_DEBUG_REPRO").is_err() {
        return;
    }
    let case = case_from_json(&serde_json::from_str(r#"{"cache_seed":[],"env":{"PKLR_MODEL_ENV_0":"v0","PKLR_MODEL_ENV_2":"v2"},"files":[{"amends":{"file":2},"imports":[{"alias":"i0_0","target":{"file":1}},{"alias":"i0_1","target":{"file":0}}],"props":[{"const":["id",0]},{"ref":["r0_0","i0_0"]}]},{"amends":null,"imports":[],"props":[{"const":["id",1]}]},{"amends":{"file":0},"imports":[{"alias":"i2_0","target":{"file":2}},{"alias":"i2_1","target":"missing"}],"props":[{"const":["id",2]},{"ref":["r2_0","i2_0"]},{"cond":["p2_0",true,"PKLR_MODEL_ENV_0","PKLR_MODEL_ENV_2"]},{"ref":["p2_1","i2_0"]}]}],"https":[],"offline":true,"pkgs":[],"preloads":[],"rewrites":[]}"#).unwrap());
    let run = run_impl(&case, Mode::Sync);
    eprintln!("result: {:?}", run.json);
    for event in &run.events {
        eprintln!("  {:?}", event);
    }
    let mut model = Model::new(&case);
    eprintln!("model: {:?}", model.run());
    eprintln!("model events: {:?}", model.events);
}
