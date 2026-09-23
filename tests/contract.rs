//! Model-based contract tests for pklr's evaluation IO behavior.
//!
//! A single evaluation is expressed as an ordered event log recorded by fake
//! capabilities over a fully in-memory file system. No real network, user
//! cache, home directory, or platform-specific path is ever touched: every
//! path lives under the virtual root `/virtual`.
//!
//! For each generated case the same seed drives three artifacts:
//!
//! 1. the synchronous public API ([`pklr::EvaluatorBuilder`]),
//! 2. the asynchronous public API ([`pklr::AsyncEvaluatorBuilder`]),
//! 3. a simplified reference model ([`model`]) that re-implements URI
//!    normalization, longest-prefix rewrites, relative file imports, package
//!    metadata/blob caching, preload hits, offline hit/miss, amend edges, and
//!    cycle detection.
//!
//! The sync and async event logs must be identical after normalizing
//! scheduling-only details (temp-dir numbering), and the implementation's
//! semantic trace must equal the model's trace. Generation uses fixed seeds
//! with explicit state-space caps; on failure the case is minimized and the
//! minimal replayable case is printed.

use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::rc::Rc;

use pklr::capabilities::{BoxFuture, EvalCapabilities};

const WORK: &str = "/virtual/work";
const CACHE: &str = "/virtual/cache";
const TMP: &str = "/virtual/tmp";

/// Cache write boundaries where failures are injected.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WriteBoundary {
    MetadataWrite,
    MetadataFlush,
    BlobWrite,
    BlobFlush,
    Rename,
    DirSync,
}

impl WriteBoundary {
    const ALL: [WriteBoundary; 6] = [
        WriteBoundary::MetadataWrite,
        WriteBoundary::MetadataFlush,
        WriteBoundary::BlobWrite,
        WriteBoundary::BlobFlush,
        WriteBoundary::Rename,
        WriteBoundary::DirSync,
    ];

    fn name(self) -> &'static str {
        match self {
            WriteBoundary::MetadataWrite => "metadata-write",
            WriteBoundary::MetadataFlush => "metadata-flush",
            WriteBoundary::BlobWrite => "blob-write",
            WriteBoundary::BlobFlush => "blob-flush",
            WriteBoundary::Rename => "rename",
            WriteBoundary::DirSync => "dir-sync",
        }
    }
}

/// One recorded capability call.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Event {
    ReadFile(String),
    ReadBytes(String),
    Exists(String),
    Canon(String),
    MkDir(String),
    WriteAtomic(String),
    RemoveFile(String),
    CacheWriteTemp(String),
    CacheFlush(String),
    CacheRename(String, String),
    CacheDirSync(String),
    CacheListDir(String),
    ReadEnv(String),
    FetchText(String),
    FetchBytes(String),
    TempDir(String),
    ExtractZip(String),
    Glob(String, String),
}

impl Event {
    /// Map every path through `normalize` and render a stable string.
    fn render(&self, normalize: &dyn Fn(&str) -> String) -> String {
        let n = |p: &str| normalize(p);
        match self {
            Event::ReadFile(p) => format!("read {}", n(p)),
            Event::ReadBytes(p) => format!("read-bytes {}", n(p)),
            Event::Exists(p) => format!("exists {}", n(p)),
            Event::Canon(p) => format!("canon {}", n(p)),
            Event::MkDir(p) => format!("mkdir {}", n(p)),
            Event::WriteAtomic(p) => format!("write-atomic {}", n(p)),
            Event::RemoveFile(p) => format!("remove {}", n(p)),
            Event::CacheWriteTemp(p) => format!("cache-write {}", n(p)),
            Event::CacheFlush(p) => format!("cache-flush {}", n(p)),
            Event::CacheRename(a, b) => format!("cache-rename {} -> {}", n(a), n(b)),
            Event::CacheDirSync(p) => format!("cache-dir-sync {}", n(p)),
            Event::CacheListDir(p) => format!("cache-list {}", n(p)),
            Event::ReadEnv(k) => format!("read-env {k}"),
            Event::FetchText(u) => format!("fetch-text {u}"),
            Event::FetchBytes(u) => format!("fetch-bytes {u}"),
            Event::TempDir(p) => format!("temp-dir {}", n(p)),
            Event::ExtractZip(p) => format!("extract-zip {}", n(p)),
            Event::Glob(b, p) => format!("glob {} {}", n(b), p),
        }
    }
}

fn io_not_found() -> pklr::Error {
    pklr::Error::Io(
        PathBuf::from("/virtual"),
        std::io::Error::from(std::io::ErrorKind::NotFound),
    )
}

fn io_not_found_at(path: &Path) -> pklr::Error {
    pklr::Error::Io(
        path.to_path_buf(),
        std::io::Error::from(std::io::ErrorKind::NotFound),
    )
}

fn injected_fault(path: &Path) -> pklr::Error {
    pklr::Error::Io(
        path.to_path_buf(),
        std::io::Error::new(std::io::ErrorKind::PermissionDenied, "injected fault"),
    )
}

fn key(path: &Path) -> String {
    path.to_string_lossy().replace('\\', "/")
}

/// Fully in-memory file system plus scripted HTTP/env behavior.
#[derive(Default)]
struct FakeFs {
    files: BTreeMap<String, Vec<u8>>,
    dirs: BTreeSet<String>,
    events: Vec<Event>,
    env: BTreeMap<String, String>,
    http_text: HashMap<String, String>,
    http_bytes: HashMap<String, Vec<u8>>,
    temp_counter: usize,
    fail_at: Option<WriteBoundary>,
}

impl FakeFs {
    fn new() -> Self {
        let mut fs = FakeFs::default();
        fs.dirs.insert("/virtual".to_string());
        fs.dirs.insert(WORK.to_string());
        fs.dirs.insert(TMP.to_string());
        fs
    }

    fn write_file(&mut self, path: &str, bytes: &[u8]) {
        let mut dir = String::new();
        for part in path.split('/').skip(1) {
            if part.contains('.') {
                break;
            }
            dir.push('/');
            dir.push_str(part);
            self.dirs.insert(dir.clone());
        }
        self.files.insert(path.to_string(), bytes.to_vec());
    }

    fn children(&self, dir: &str) -> Vec<String> {
        let prefix = format!("{dir}/");
        let mut out = BTreeSet::new();
        for path in self.files.keys().chain(self.dirs.iter()) {
            if let Some(rest) = path.strip_prefix(&prefix)
                && !rest.contains('/')
            {
                out.insert(path.clone());
            }
        }
        out.into_iter().collect()
    }

    fn boundary_for_temp(path: &str, flush: bool) -> WriteBoundary {
        let blob = path.contains(".bin.");
        match (blob, flush) {
            (true, true) => WriteBoundary::BlobFlush,
            (true, false) => WriteBoundary::BlobWrite,
            (false, true) => WriteBoundary::MetadataFlush,
            (false, false) => WriteBoundary::MetadataWrite,
        }
    }

    fn take_fault(&mut self, boundary: WriteBoundary) -> bool {
        if self.fail_at == Some(boundary) {
            self.fail_at = None;
            true
        } else {
            false
        }
    }
}

type Shared = Rc<RefCell<FakeFs>>;

struct FakeCaps {
    fs: Shared,
}

impl FakeCaps {
    fn record(&self, event: Event) {
        self.fs.borrow_mut().events.push(event);
    }
}

impl EvalCapabilities for FakeCaps {
    fn read_to_string<'a>(&'a mut self, path: &'a Path) -> BoxFuture<'a, pklr::Result<String>> {
        let key = key(path);
        self.record(Event::ReadFile(key.clone()));
        let bytes = self.fs.borrow().files.get(&key).cloned();
        Box::pin(async move {
            let bytes = bytes.ok_or_else(|| io_not_found_at(Path::new(&key)))?;
            String::from_utf8(bytes).map_err(|error| pklr::Error::Eval(error.to_string()))
        })
    }

    fn path_exists<'a>(&'a mut self, path: &'a Path) -> BoxFuture<'a, pklr::Result<bool>> {
        let key = key(path);
        self.record(Event::Exists(key.clone()));
        let exists = {
            let fs = self.fs.borrow();
            fs.files.contains_key(&key) || fs.dirs.contains(&key)
        };
        Box::pin(async move { Ok(exists) })
    }

    fn canonicalize<'a>(&'a mut self, path: &'a Path) -> BoxFuture<'a, pklr::Result<PathBuf>> {
        let key = key(path);
        self.record(Event::Canon(key.clone()));
        Box::pin(async move { Ok(PathBuf::from(key)) })
    }

    fn read_bytes<'a>(&'a mut self, path: &'a Path) -> BoxFuture<'a, pklr::Result<Vec<u8>>> {
        let key = key(path);
        self.record(Event::ReadBytes(key.clone()));
        let bytes = self.fs.borrow().files.get(&key).cloned();
        Box::pin(async move { bytes.ok_or_else(|| io_not_found_at(Path::new(&key))) })
    }

    fn create_dir_all<'a>(&'a mut self, path: &'a Path) -> BoxFuture<'a, pklr::Result<()>> {
        let key = key(path);
        self.record(Event::MkDir(key.clone()));
        self.fs.borrow_mut().dirs.insert(key);
        Box::pin(async { Ok(()) })
    }

    fn write_atomic<'a>(&'a mut self, path: &'a Path, bytes: &'a [u8]) -> BoxFuture<'a, pklr::Result<()>> {
        let key = key(path);
        self.record(Event::WriteAtomic(key.clone()));
        self.fs.borrow_mut().files.insert(key, bytes.to_vec());
        Box::pin(async { Ok(()) })
    }

    fn remove_file<'a>(&'a mut self, path: &'a Path) -> BoxFuture<'a, pklr::Result<()>> {
        let key = key(path);
        self.record(Event::RemoveFile(key.clone()));
        let existed = self.fs.borrow_mut().files.remove(&key).is_some();
        Box::pin(async move {
            if existed {
                Ok(())
            } else {
                Err(io_not_found_at(Path::new(&key)))
            }
        })
    }

    fn cache_write_temp<'a>(&'a mut self, path: &'a Path, bytes: &'a [u8]) -> BoxFuture<'a, pklr::Result<()>> {
        let key = key(path);
        self.record(Event::CacheWriteTemp(key.clone()));
        let boundary = FakeFs::boundary_for_temp(&key, false);
        if self.fs.borrow_mut().take_fault(boundary) {
            return Box::pin(async move { Err(injected_fault(Path::new(&key))) });
        }
        self.fs.borrow_mut().files.insert(key, bytes.to_vec());
        Box::pin(async { Ok(()) })
    }

    fn cache_flush_temp<'a>(&'a mut self, path: &'a Path) -> BoxFuture<'a, pklr::Result<()>> {
        let key = key(path);
        self.record(Event::CacheFlush(key.clone()));
        let boundary = FakeFs::boundary_for_temp(&key, true);
        if self.fs.borrow_mut().take_fault(boundary) {
            // The staging file stays behind, exactly like an fsync failure;
            // the next startup's sweep must reclaim it.
            return Box::pin(async move { Err(injected_fault(Path::new(&key))) });
        }
        Box::pin(async { Ok(()) })
    }

    fn cache_rename<'a>(&'a mut self, from: &'a Path, to: &'a Path) -> BoxFuture<'a, pklr::Result<()>> {
        let (from, to) = (key(from), key(to));
        self.record(Event::CacheRename(from.clone(), to.clone()));
        if self.fs.borrow_mut().take_fault(WriteBoundary::Rename) {
            return Box::pin(async move { Err(injected_fault(Path::new(&to))) });
        }
        let bytes = self.fs.borrow_mut().files.remove(&from);
        match bytes {
            Some(bytes) => {
                self.fs.borrow_mut().files.insert(to, bytes);
                Box::pin(async { Ok(()) })
            }
            None => Box::pin(async move { Err(io_not_found_at(Path::new(&from))) }),
        }
    }

    fn cache_dir_sync<'a>(&'a mut self, path: &'a Path) -> BoxFuture<'a, pklr::Result<()>> {
        let key = key(path);
        self.record(Event::CacheDirSync(key.clone()));
        if self.fs.borrow_mut().take_fault(WriteBoundary::DirSync) {
            return Box::pin(async move { Err(injected_fault(Path::new(&key))) });
        }
        Box::pin(async { Ok(()) })
    }

    fn cache_list_dir<'a>(&'a mut self, path: &'a Path) -> BoxFuture<'a, pklr::Result<Vec<PathBuf>>> {
        let key = key(path);
        self.record(Event::CacheListDir(key.clone()));
        let children = self.fs.borrow().children(&key);
        let exists = self.fs.borrow().dirs.contains(&key);
        Box::pin(async move {
            if exists {
                Ok(children.into_iter().map(PathBuf::from).collect())
            } else {
                Err(io_not_found())
            }
        })
    }

    fn extract_zip<'a>(&'a mut self, bytes: Vec<u8>, destination: &'a Path) -> BoxFuture<'a, pklr::Result<()>> {
        let dest = key(destination);
        self.record(Event::ExtractZip(dest.clone()));
        let fs = self.fs.clone();
        Box::pin(async move {
            let mut archive = zip::ZipArchive::new(std::io::Cursor::new(bytes))
                .map_err(|error| pklr::Error::Eval(format!("zip error: {error}")))?;
            for index in 0..archive.len() {
                let mut entry = archive
                    .by_index(index)
                    .map_err(|error| pklr::Error::Eval(format!("zip error: {error}")))?;
                let name = entry
                    .enclosed_name()
                    .ok_or_else(|| pklr::Error::Eval("zip entry escapes".to_string()))?;
                let mut data = Vec::new();
                std::io::Read::read_to_end(&mut entry, &mut data)
                    .map_err(|error| pklr::Error::Eval(format!("zip read: {error}")))?;
                let path = format!("{dest}/{}", name.to_string_lossy());
                fs.borrow_mut().write_file(&path, &data);
            }
            Ok(())
        })
    }

    fn read_env<'a>(&'a mut self, name: &'a str) -> BoxFuture<'a, pklr::Result<Option<String>>> {
        self.record(Event::ReadEnv(name.to_string()));
        let value = self.fs.borrow().env.get(name).cloned();
        Box::pin(async move { Ok(value) })
    }

    fn fetch_text<'a>(&'a mut self, url: &'a str) -> BoxFuture<'a, pklr::Result<String>> {
        self.record(Event::FetchText(url.to_string()));
        let body = self.fs.borrow().http_text.get(url).cloned();
        let url = url.to_string();
        Box::pin(async move { body.ok_or(pklr::Error::ImportNotFound(url)) })
    }

    fn fetch_bytes<'a>(&'a mut self, url: &'a str) -> BoxFuture<'a, pklr::Result<Vec<u8>>> {
        self.record(Event::FetchBytes(url.to_string()));
        let body = self.fs.borrow().http_bytes.get(url).cloned();
        let url = url.to_string();
        Box::pin(async move { body.ok_or(pklr::Error::ImportNotFound(url)) })
    }

    fn temp_dir<'a>(&'a mut self, prefix: &'a str) -> BoxFuture<'a, pklr::Result<PathBuf>> {
        let mut fs = self.fs.borrow_mut();
        let path = format!("{TMP}/{prefix}-{}", fs.temp_counter);
        fs.temp_counter += 1;
        fs.dirs.insert(path.clone());
        drop(fs);
        self.record(Event::TempDir(path.clone()));
        Box::pin(async move { Ok(PathBuf::from(path)) })
    }

    fn glob<'a>(&'a mut self, base: &'a Path, pattern: &'a str) -> BoxFuture<'a, pklr::Result<Vec<PathBuf>>> {
        self.record(Event::Glob(key(base), pattern.to_string()));
        Box::pin(async { Ok(Vec::new()) })
    }
}

// --- Case model ------------------------------------------------------------

/// Explicit caps on the generated state space. The generator never exceeds
/// these, so the contract suite is a bounded deterministic enumeration, not
/// unbounded fuzzing.
const MAX_NODES: usize = 5;
const MAX_PACKAGES: usize = 2;
const MAX_IMPORTS_PER_NODE: usize = 2;
const GENERATED_SEEDS: u64 = 24;

#[derive(Debug, Clone, PartialEq, Eq)]
enum Loc {
    Local,
    Http,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Target {
    Node(usize),
    Pkg(usize),
}

#[derive(Debug, Clone)]
struct Node {
    loc: Loc,
    amends: Option<Target>,
    imports: Vec<Target>,
    env_read: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum CacheState {
    Miss,
    Valid,
}

#[derive(Debug, Clone)]
struct Package {
    /// "pkl" (direct download) or "zip".
    ext: &'static str,
    cache: CacheState,
}

#[derive(Debug, Clone)]
struct Case {
    seed: u64,
    nodes: Vec<Node>,
    packages: Vec<Package>,
    rewrites: Vec<(String, String)>,
    offline: bool,
}

impl Case {
    fn node_path(&self, idx: usize) -> String {
        format!("{WORK}/m{idx}.pkl")
    }

    fn node_url(&self, idx: usize) -> String {
        // Alternate hosts so rewrite rules have both matching and
        // non-matching inputs.
        if idx % 2 == 0 {
            format!("https://cdn.example.com/deep/m{idx}.pkl")
        } else {
            format!("https://cdn.example.com/m{idx}.pkl")
        }
    }

    fn pkg_url(&self, idx: usize) -> String {
        match self.packages[idx].ext {
            "pkl" => format!("https://github.com/acme/pkg{idx}/releases/download/1.0.{idx}/main.pkl"),
            _ => format!("https://pkg.example.com/pkg{idx}@1.0.0.zip"),
        }
    }

    fn pkg_entry(&self, idx: usize) -> String {
        format!("Pkg{idx}.pkl")
    }

    /// Longest-prefix rewrite, mirroring `Evaluator::rewrite_url`.
    fn rewrite(&self, url: &str) -> String {
        let best = self
            .rewrites
            .iter()
            .filter(|(src, _)| url.starts_with(src.as_str()))
            .max_by_key(|(src, _)| src.len());
        match best {
            Some((src, tgt)) => format!("{}{}", tgt, &url[src.len()..]),
            None => url.to_string(),
        }
    }

    fn target_alias(target: &Target) -> String {
        match target {
            Target::Node(i) => format!("A{i}"),
            Target::Pkg(k) => format!("P{k}"),
        }
    }

    fn target_uri(&self, target: &Target) -> String {
        match target {
            Target::Node(i) => match self.nodes[*i].loc {
                Loc::Local => format!("m{i}.pkl"),
                Loc::Http => self.node_url(*i),
            },
            Target::Pkg(k) => match self.packages[*k].ext {
                "pkl" => format!("package://pkg.pkl-lang.org/github.com/acme/pkg{k}@1.0.{k}#/main.pkl"),
                _ => format!("package://pkg.example.com/pkg{k}@1.0.0#/{}", self.pkg_entry(*k)),
            },
        }
    }

    fn node_source(&self, idx: usize) -> String {
        let node = &self.nodes[idx];
        let mut out = String::new();
        if let Some(amends) = &node.amends {
            out.push_str(&format!("amends \"{}\"\n", self.target_uri(amends)));
        }
        for (n, target) in node.imports.iter().enumerate() {
            out.push_str(&format!(
                "import \"{}\" as {}\n",
                self.target_uri(target),
                Self::target_alias(target)
            ));
            out.push_str(&format!("used_{n} = {}\n", Self::target_alias(target)));
        }
        if let Some(var) = &node.env_read {
            out.push_str(&format!("envv = read(\"env:{var}\")\n"));
        }
        out.push_str(&format!("val = {idx}\n"));
        out
    }

    fn pkg_source(&self, idx: usize) -> String {
        format!("pv = {}\n", 100 + idx)
    }
}

/// Deterministic RNG (SplitMix64) so every seed replays exactly.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }

    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }

    fn chance(&mut self, numerator: u64, denominator: u64) -> bool {
        self.below(denominator) < numerator
    }
}

fn generate(seed: u64) -> Case {
    let mut rng = Rng(seed.wrapping_mul(0x9e3779b97f4a7c15).wrapping_add(1));
    let node_count = 2 + rng.below((MAX_NODES - 1) as u64) as usize;
    let mut nodes = Vec::new();
    for i in 0..node_count {
        let loc = if i > 0 && rng.chance(1, 4) { Loc::Http } else { Loc::Local };
        nodes.push(Node {
            loc,
            amends: None,
            imports: Vec::new(),
            env_read: None,
        });
    }
    let pkg_count = rng.below((MAX_PACKAGES + 1) as u64) as usize;
    let mut packages = Vec::new();
    for _ in 0..pkg_count {
        packages.push(Package {
            ext: if rng.chance(1, 2) { "pkl" } else { "zip" },
            cache: if rng.chance(1, 2) {
                CacheState::Valid
            } else {
                CacheState::Miss
            },
        });
    }
    // Edges only point to strictly smaller indices: the import/amend graph is
    // a bounded DAG by construction (cycles get dedicated tests).
    for i in 1..node_count {
        let mut targets: Vec<Target> = (0..i).map(Target::Node).collect();
        targets.extend((0..pkg_count).map(Target::Pkg));
        if rng.chance(1, 3) && !targets.is_empty() {
            let pick = targets[rng.below(targets.len() as u64) as usize].clone();
            nodes[i].amends = Some(pick);
        }
        let import_count = rng.below((MAX_IMPORTS_PER_NODE + 1) as u64) as usize;
        let mut available = targets.clone();
        for _ in 0..import_count {
            if available.is_empty() {
                break;
            }
            let pick = available.remove(rng.below(available.len() as u64) as usize);
            if nodes[i].amends.as_ref() == Some(&pick) {
                continue;
            }
            nodes[i].imports.push(pick);
        }
        if rng.chance(1, 4) {
            nodes[i].env_read = Some(format!("PKLR_CONTRACT_VAR_{i}"));
        }
    }
    // The entry node may import anything (never itself).
    let entry_imports = rng.below(2) as usize;
    let mut available: Vec<Target> = (1..node_count).map(Target::Node).collect();
    available.extend((0..pkg_count).map(Target::Pkg));
    for _ in 0..entry_imports {
        if available.is_empty() {
            break;
        }
        nodes[0]
            .imports
            .push(available.remove(rng.below(available.len() as u64) as usize));
    }

    let mut rewrites = Vec::new();
    if nodes.iter().any(|n| n.loc == Loc::Http) || pkg_count > 0 {
        // A short and a long prefix for the same host: the long one must win.
        rewrites.push((
            "https://cdn.example.com/".to_string(),
            "https://mirror.example.com/cdn/".to_string(),
        ));
        rewrites.push((
            "https://cdn.example.com/deep/".to_string(),
            "https://mirror2.example.com/deep/".to_string(),
        ));
        if rng.chance(1, 2) {
            rewrites.push((
                "https://pkg.example.com/".to_string(),
                "https://pkg-mirror.example.com/".to_string(),
            ));
        }
    }
    let offline = rng.chance(1, 4);
    if offline {
        // Offline generated cases stay successful: everything is preloaded
        // and no plain HTTP imports remain.
        for package in &mut packages {
            package.cache = CacheState::Valid;
        }
        for node in &mut nodes {
            if node.loc == Loc::Http {
                node.loc = Loc::Local;
            }
        }
    }
    Case {
        seed,
        nodes,
        packages,
        rewrites,
        offline,
    }
}
