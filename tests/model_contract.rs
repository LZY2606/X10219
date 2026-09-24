//! Model-based contract tests for pklr's evaluation pipeline.
//!
//! A `Case` describes a bounded import/amend graph, URI rewrite rules, an
//! initial package-cache state, preloads, an offline flag, and environment
//! variables. Each case is evaluated twice — once driven synchronously
//! (pollster) and once on a tokio runtime — through `ScriptedHost`, a fake
//! capabilities implementation backed by a real temporary directory and
//! virtual HTTP/env maps. No real network or user cache is touched.
//!
//! A simplified reference model (`Model`) independently computes the expected
//! abstract event trace and outcome: URI normalization, longest-prefix
//! rewrites, relative file imports, package metadata/blob cache hits and
//! misses, preload hits, offline hit/miss, amend edges, and cycle detection.
//! The implementation trace must match the model, and the synchronous and
//! asynchronous runs must produce identical event logs (temp-file counters
//! are the only scheduling-dependent fields, and they are normalized away).
//!
//! Generated cases use fixed seeds with explicit state-space bounds; golden
//! handwritten cases pin down each contract individually, so the generator
//! is never the only gate. On failure the runner prints a replayable seed
//! and a greedily shrunk minimal case spec (replay with
//! `PKLR_MODEL_REPLAY=<seed>` or `PKLR_MODEL_REPLAY_SPEC='<spec>'`).

mod contract_support;

use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::{Path, PathBuf};

use contract_support::{Event, ScriptedHost, TestRoot, lex_normalize, zip_bytes};
use pklr::Evaluator;

// --- Explicit state-space bounds -------------------------------------------

const MAX_LOCAL_UNITS: usize = 5;
const MAX_REMOTE_UNITS: usize = 2;
const MAX_PACKAGES: usize = 2;
const MAX_IMPORTS_PER_UNIT: usize = 2;
const MAX_ENV_READS_PER_UNIT: usize = 2;
const MAX_REWRITE_RULES: usize = 2;
const GENERATED_CASES: u64 = 48;
const MAX_SHRINK_ROUNDS: usize = 24;

// --- Case specification -----------------------------------------------------

#[derive(Clone, Debug, PartialEq)]
struct Case {
    units: Vec<Unit>,
    packages: Vec<Package>,
    entry: usize,
    rewrites: Vec<String>,
    offline: bool,
    preloads: Vec<usize>,
    cache_seeds: Vec<(usize, CacheSeed)>,
    env: BTreeMap<String, String>,
}

#[derive(Clone, Debug, PartialEq)]
struct Unit {
    name: String,
    remote: bool,
    dir: String,
    id: i64,
    imports: Vec<Import>,
    amends: Option<usize>,
    env_reads: Vec<String>,
    opt_reads: Vec<String>,
}

#[derive(Clone, Debug, PartialEq)]
struct Import {
    alias: String,
    target: Target,
    uri: String,
    referenced: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Target {
    Local(usize),
    Remote(usize),
    Package(usize),
    Missing,
}

#[derive(Clone, Debug, PartialEq)]
struct Package {
    name: String,
    direct: bool,
    answer: i64,
    stale_answer: i64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CacheSeed {
    Absent,
    Valid,
    ValidStale,
    CorruptMeta,
    BadDigest,
    UnsupportedVersion,
    StaleTemp,
}

impl CacheSeed {
    fn spec_name(self) -> &'static str {
        match self {
            Self::Absent => "absent",
            Self::Valid => "valid",
            Self::ValidStale => "stale",
            Self::CorruptMeta => "corruptmeta",
            Self::BadDigest => "baddigest",
            Self::UnsupportedVersion => "badversion",
            Self::StaleTemp => "staletemp",
        }
    }

    fn from_spec(name: &str) -> Option<Self> {
        Some(match name {
            "absent" => Self::Absent,
            "valid" => Self::Valid,
            "stale" => Self::ValidStale,
            "corruptmeta" => Self::CorruptMeta,
            "baddigest" => Self::BadDigest,
            "badversion" => Self::UnsupportedVersion,
            "staletemp" => Self::StaleTemp,
            _ => return None,
        })
    }
}

impl Unit {
    fn path(&self) -> String {
        format!("{}/{}.pkl", self.dir, self.name)
    }

    fn url(&self) -> String {
        format!("https://remote.test/pkg/{}.pkl", self.name)
    }

    fn source(&self, case: &Case, root: &Path) -> String {
        let mut out = String::new();
        if let Some(amends) = self.amends {
            let uri = local_import_uri(&self.dir, &case.units[amends].path());
            out.push_str(&format!("amends \"{uri}\"\n"));
        }
        for import in &self.imports {
            let uri = import.uri.replace("{ROOT}", &root.display().to_string());
            out.push_str(&format!("import \"{uri}\" as {}\n", import.alias));
        }
        out.push_str(&format!("id = {}\n", self.id));
        for import in &self.imports {
            if import.referenced {
                out.push_str(&format!("copy_{} = {}\n", import.alias, import.alias));
            }
        }
        for (index, var) in self.env_reads.iter().enumerate() {
            out.push_str(&format!("env_{}_{index} = read(\"env:{var}\")\n", self.name));
        }
        for (index, var) in self.opt_reads.iter().enumerate() {
            out.push_str(&format!("opt_{}_{index} = read?(\"env:{var}\")\n", self.name));
        }
        let never = |kind: &str| format!("PKLR_MODEL_NEVER_{kind}_{}", self.name.to_uppercase());
        out.push_str(&format!(
            "short_circuit_and = false && read(\"env:{}\")\n",
            never("AND")
        ));
        out.push_str(&format!(
            "short_circuit_or = true || read(\"env:{}\")\n",
            never("OR")
        ));
        out.push_str(&format!(
            "cond_branch = if (false) read(\"env:{}\") else \"z\"\n",
            never("IF")
        ));
        out
    }
}

impl Package {
    fn entry(&self) -> String {
        format!("{}Config.pkl", self.name)
    }

    fn ext(&self) -> &'static str {
        if self.direct { "pkl" } else { "zip" }
    }

    fn uri(&self) -> String {
        if self.direct {
            format!(
                "package://pkg.pkl-lang.org/github.com/acme/{}@2.0.0#/{}",
                self.name,
                self.entry()
            )
        } else {
            format!("package://example.com/{}@1.0.0#/{}", self.name, self.entry())
        }
    }

    /// The https URL the package URI resolves to (see `resolve_package_uri`).
    fn fetch_url(&self) -> String {
        if self.direct {
            format!(
                "https://github.com/acme/{}/releases/download/2.0.0/{}",
                self.name,
                self.entry()
            )
        } else {
            format!("https://example.com/{}@1.0.0.zip", self.name)
        }
    }

    fn source(answer: i64) -> String {
        format!("answer = {answer}\n")
    }

    fn bytes(&self) -> Vec<u8> {
        if self.direct {
            Self::source(self.answer).into_bytes()
        } else {
            zip_bytes(&self.entry(), &Self::source(self.answer))
        }
    }

    fn stale_bytes(&self) -> Vec<u8> {
        if self.direct {
            Self::source(self.stale_answer).into_bytes()
        } else {
            zip_bytes(&self.entry(), &Self::source(self.stale_answer))
        }
    }
}

impl Case {
    fn local_units(&self) -> impl Iterator<Item = (usize, &Unit)> {
        self.units
            .iter()
            .enumerate()
            .filter(|(_, unit)| !unit.remote)
    }

    fn local_paths(&self) -> HashSet<String> {
        self.local_units().map(|(_, unit)| unit.path()).collect()
    }

    fn entry_path(&self) -> String {
        self.units[self.entry].path()
    }
}

/// Relative import path from an importer directory to a target module path.
fn local_import_uri(importer_dir: &str, target_path: &str) -> String {
    let importer: Vec<&str> = importer_dir.split('/').collect();
    let mut target: Vec<&str> = target_path.split('/').collect();
    let mut common = 0;
    while common < importer.len()
        && common < target.len()
        && importer[common] == target[common]
    {
        common += 1;
    }
    let mut parts = vec![".."; importer.len() - common];
    parts.append(&mut target.split_off(common));
    parts.join("/")
}

/// Longest-prefix rewrite, matching `Evaluator::rewrite_url`: among matching
/// rules the longest source prefix wins; ties keep the later rule.
fn rewrite(rules: &[(String, String)], url: &str) -> String {
    let mut best: Option<&(String, String)> = None;
    for rule in rules {
        if url.starts_with(&rule.0) && best.is_none_or(|current| rule.0.len() >= current.0.len()) {
            best = Some(rule);
        }
    }
    match best {
        Some((src, tgt)) => format!("{}{}", tgt, &url[src.len()..]),
        None => url.to_string(),
    }
}

fn parse_rewrites(rules: &[String]) -> Vec<(String, String)> {
    rules
        .iter()
        .filter_map(|rule| rule.split_once('='))
        .map(|(src, tgt)| (src.to_string(), tgt.to_string()))
        .collect()
}

// --- Abstract event trace ---------------------------------------------------

/// Events the model predicts, projected from the raw capability log.
#[derive(Clone, Debug, PartialEq, Eq)]
enum AEvent {
    /// First read of a module source (root-relative path).
    Read(String),
    /// First read of an environment variable.
    Env(String),
    /// HTTP text fetch (post-rewrite URL).
    FetchText(String),
    /// HTTP bytes fetch for a package (post-rewrite URL).
    FetchBytes(String),
    /// Package zip extraction temp directory.
    TempDir(String),
    /// Atomic cache write staged for this final path.
    CacheWrite(String),
}

fn project(events: &[Event]) -> Vec<AEvent> {
    let mut seen_reads = HashSet::new();
    let mut seen_env = HashSet::new();
    let mut out = Vec::new();
    for event in events {
        match event {
            Event::Read(path) if !path.starts_with("cache/") => {
                if seen_reads.insert(path.clone()) {
                    out.push(AEvent::Read(path.clone()));
                }
            }
            Event::Env(name) => {
                if seen_env.insert(name.clone()) {
                    out.push(AEvent::Env(name.clone()));
                }
            }
            Event::FetchText(url) => out.push(AEvent::FetchText(url.clone())),
            Event::FetchBytes(url) => out.push(AEvent::FetchBytes(url.clone())),
            Event::TempDir(prefix) => out.push(AEvent::TempDir(prefix.clone())),
            Event::TempCreate(path) if path.starts_with("cache/") => {
                out.push(AEvent::CacheWrite(path.clone()));
            }
            _ => {}
        }
    }
    out
}

// --- Reference model --------------------------------------------------------

#[derive(Clone, Copy, PartialEq)]
enum CacheContent {
    Real,
    Stale,
}

struct Model<'c> {
    case: &'c Case,
    events: Vec<AEvent>,
    seen_reads: HashSet<String>,
    seen_env: HashSet<String>,
    memo: HashMap<String, serde_json::Value>,
    scoped: HashSet<String>,
    http_text_fetched: HashSet<String>,
    pkg_source_cached: HashSet<String>,
    extracted: HashMap<usize, String>,
    cache: HashMap<usize, CacheContent>,
    corrupt: HashMap<usize, &'static str>,
    rewrites: Vec<(String, String)>,
}

impl<'c> Model<'c> {
    fn new(case: &'c Case) -> Self {
        Self {
            case,
            events: Vec::new(),
            seen_reads: HashSet::new(),
            seen_env: HashSet::new(),
            memo: HashMap::new(),
            scoped: HashSet::new(),
            http_text_fetched: HashSet::new(),
            pkg_source_cached: HashSet::new(),
            extracted: HashMap::new(),
            cache: HashMap::new(),
            corrupt: HashMap::new(),
            rewrites: parse_rewrites(&case.rewrites),
        }
    }

    fn run(mut self) -> (Vec<AEvent>, Result<serde_json::Value, String>) {
        self.apply_cache_seeds();
        self.apply_preloads();
        // eval_source_inner seeds the entry into the import cache as an empty
        // placeholder before evaluating, then stores the final value.
        let entry_path = self.case.entry_path();
        self.memo.insert(entry_path.clone(), serde_json::json!({}));
        let result = self.eval_module(self.case.entry, 0);
        (self.events, result)
    }

    fn apply_cache_seeds(&mut self) {
        for (pkg, seed) in &self.case.cache_seeds {
            match seed {
                CacheSeed::Valid => {
                    self.cache.insert(*pkg, CacheContent::Real);
                }
                CacheSeed::ValidStale => {
                    self.cache.insert(*pkg, CacheContent::Stale);
                }
                CacheSeed::CorruptMeta => {
                    self.corrupt.insert(*pkg, "corrupt package cache metadata");
                }
                CacheSeed::BadDigest => {
                    self.corrupt.insert(*pkg, "package cache digest mismatch");
                }
                CacheSeed::UnsupportedVersion => {
                    self.corrupt
                        .insert(*pkg, "unsupported package cache metadata version");
                }
                CacheSeed::Absent | CacheSeed::StaleTemp => {}
            }
        }
    }

    fn apply_preloads(&mut self) {
        for pkg in self.case.preloads.clone() {
            // A valid cached entry wins; preloading never overrides it.
            if self.cache.contains_key(&pkg) {
                continue;
            }
            self.corrupt.remove(&pkg);
            let bytes = self.case.packages[pkg].bytes();
            let url = self.case.packages[pkg].fetch_url();
            let ext = self.case.packages[pkg].ext();
            let digest = pklr::cache::content_digest(&bytes);
            let blob = pklr::cache::blob_path(Path::new("cache"), &url, &digest, ext).unwrap();
            let meta = pklr::cache::meta_path(Path::new("cache"), &url);
            self.events.push(AEvent::CacheWrite(rel_string(&blob)));
            self.events.push(AEvent::CacheWrite(rel_string(&meta)));
            self.cache.insert(pkg, CacheContent::Real);
        }
    }

    fn note_read(&mut self, path: String) {
        if self.seen_reads.insert(path.clone()) {
            self.events.push(AEvent::Read(path));
        }
    }

    fn note_env(&mut self, name: &str) {
        if self.seen_env.insert(name.to_string()) {
            self.events.push(AEvent::Env(name.to_string()));
        }
    }

    /// Aliases usable from `unit`'s scope, including names referenced by
    /// amended base modules (the evaluator computes `referenced_imports`
    /// from the module body plus inherited reference roots).
    fn inherited_refs(&self, unit: usize) -> HashSet<String> {
        let mut refs = HashSet::new();
        let mut frontier = vec![unit];
        let mut visited = HashSet::new();
        while let Some(current) = frontier.pop() {
            if !visited.insert(current) {
                continue;
            }
            if current != unit {
                for import in &self.case.units[current].imports {
                    if import.referenced {
                        refs.insert(import.alias.clone());
                    }
                }
            }
            if let Some(base) = self.case.units[current].amends {
                frontier.push(base);
            }
        }
        refs
    }

    /// `inherited_reference_roots`: reads the amends-chain sources.
    fn scan_inherited(&mut self, unit: usize, visited: &mut HashSet<usize>) {
        if !visited.insert(unit) {
            return;
        }
        if let Some(base) = self.case.units[unit].amends {
            self.note_read(self.case.units[base].path());
            self.scan_inherited(base, visited);
        }
    }

    fn eval_module(&mut self, index: usize, depth: usize) -> Result<serde_json::Value, String> {
        if depth > 32 {
            return Err("max import depth".to_string());
        }
        let unit = &self.case.units[index];
        if !unit.remote {
            self.note_read(unit.path());
        }
        let mut visited = HashSet::new();
        self.scan_inherited(index, &mut visited);
        let inherited = self.inherited_refs(index);
        let mut bindings = Vec::new();
        for import in &unit.imports {
            if !import.referenced && !inherited.contains(&import.alias) {
                continue;
            }
            let value = self.eval_import(index, import, depth + 1)?;
            bindings.push((import.alias.clone(), value));
        }
        let mut out = serde_json::Map::new();
        if let Some(base) = unit.amends {
            let base_value = self.eval_scoped(base, depth + 1)?;
            if let serde_json::Value::Object(map) = base_value {
                out = map;
            }
        }
        out.insert("id".to_string(), serde_json::json!(unit.id));
        for (alias, value) in bindings {
            out.insert(format!("copy_{alias}"), value);
        }
        for (var_index, var) in unit.env_reads.iter().enumerate() {
            self.note_env(var);
            let value = self
                .case
                .env
                .get(var)
                .cloned()
                .ok_or_else(|| format!("environment variable not found: {var}"))?;
            out.insert(
                format!("env_{}_{var_index}", unit.name),
                serde_json::Value::String(value),
            );
        }
        for (var_index, var) in unit.opt_reads.iter().enumerate() {
            self.note_env(var);
            out.insert(format!("opt_{}_{var_index}", unit.name), serde_json::Value::Null);
        }
        out.insert("short_circuit_and".to_string(), serde_json::json!(false));
        out.insert("short_circuit_or".to_string(), serde_json::json!(true));
        out.insert("cond_branch".to_string(), serde_json::json!("z"));
        Ok(serde_json::Value::Object(out))
    }

    /// Amends/extends evaluation: scoped, cycle-broken to an empty object,
    /// never memoized.
    fn eval_scoped(&mut self, index: usize, depth: usize) -> Result<serde_json::Value, String> {
        let key = self.case.units[index].path();
        if !self.scoped.insert(key.clone()) {
            return Ok(serde_json::json!({}));
        }
        let result = self.eval_module(index, depth);
        self.scoped.remove(&key);
        result
    }

    /// Local import evaluation: memoized with an in-flight placeholder.
    fn eval_file_local(
        &mut self,
        index: usize,
        depth: usize,
    ) -> Result<serde_json::Value, String> {
        let key = self.case.units[index].path();
        if let Some(value) = self.memo.get(&key) {
            return Ok(value.clone());
        }
        self.memo.insert(key.clone(), serde_json::json!({}));
        let value = self.eval_module(index, depth)?;
        self.memo.insert(key, value.clone());
        Ok(value)
    }

    fn eval_import(
        &mut self,
        importer: usize,
        import: &Import,
        depth: usize,
    ) -> Result<serde_json::Value, String> {
        match import.target {
            Target::Local(target) => {
                let resolved = self.resolve_local(importer, &import.uri);
                if !self.case.local_paths().contains(&resolved) {
                    return Err("Import not found".to_string());
                }
                self.eval_file_local(target, depth)
            }
            Target::Missing => Err("Import not found".to_string()),
            Target::Remote(target) => {
                let url = self.resolve_remote(importer, import);
                self.fetch_text(&url)?;
                self.eval_module(target, depth)
            }
            Target::Package(pkg) => self.eval_package(pkg, depth),
        }
    }

    fn resolve_local(&self, importer: usize, uri: &str) -> String {
        if let Some(rest) = uri.strip_prefix("file://") {
            return rest
                .strip_prefix("{ROOT}/")
                .unwrap_or(rest)
                .to_string();
        }
        let importer_dir = &self.case.units[importer].dir;
        lex_normalize(Path::new(importer_dir).join(uri).as_path())
            .to_string_lossy()
            .replace('\\', "/")
    }

    fn resolve_remote(&self, importer: usize, import: &Import) -> String {
        if import.uri.contains("://") {
            return import.uri.clone();
        }
        let base = self.case.units[importer].url();
        url::Url::parse(&base)
            .and_then(|base| base.join(&import.uri))
            .map(|url| url.to_string())
            .unwrap_or(import.uri.clone())
    }

    fn fetch_text(&mut self, url: &str) -> Result<(), String> {
        let rewritten = rewrite(&self.rewrites, url);
        if self.http_text_fetched.contains(&rewritten) {
            return Ok(());
        }
        if self.case.offline {
            return Err("offline mode prevented HTTP fetch".to_string());
        }
        self.events.push(AEvent::FetchText(rewritten.clone()));
        self.http_text_fetched.insert(rewritten);
        Ok(())
    }

    fn fetch_package_bytes(&mut self, pkg: usize) -> Result<(), String> {
        if self.cache.contains_key(&pkg) {
            return Ok(());
        }
        if let Some(pattern) = self.corrupt.get(&pkg).copied() {
            if self.case.offline {
                return Err(pattern.to_string());
            }
            // Corrupt entries are dropped and refetched while online.
            self.corrupt.remove(&pkg);
        }
        if self.case.offline {
            return Err("package is not cached and offline mode is enabled".to_string());
        }
        let package = &self.case.packages[pkg];
        let url = package.fetch_url();
        let rewritten = rewrite(&self.rewrites, &url);
        self.events.push(AEvent::FetchBytes(rewritten));
        let bytes = package.bytes();
        let digest = pklr::cache::content_digest(&bytes);
        let blob = pklr::cache::blob_path(Path::new("cache"), &url, &digest, package.ext()).unwrap();
        let meta = pklr::cache::meta_path(Path::new("cache"), &url);
        self.events.push(AEvent::CacheWrite(rel_string(&blob)));
        self.events.push(AEvent::CacheWrite(rel_string(&meta)));
        self.cache.insert(pkg, CacheContent::Real);
        Ok(())
    }

    fn effective_answer(&self, pkg: usize) -> i64 {
        let package = &self.case.packages[pkg];
        match self.cache.get(&pkg) {
            Some(CacheContent::Stale) => package.stale_answer,
            _ => package.answer,
        }
    }

    fn eval_package(&mut self, pkg: usize, depth: usize) -> Result<serde_json::Value, String> {
        let package = &self.case.packages[pkg];
        let url = package.fetch_url();
        if package.direct {
            if !self.pkg_source_cached.contains(&url) {
                self.fetch_package_bytes(pkg)?;
                self.pkg_source_cached.insert(url);
            }
            return Ok(serde_json::json!({ "answer": self.effective_answer(pkg) }));
        }
        if !self.extracted.contains_key(&pkg) {
            self.fetch_package_bytes(pkg)?;
            let prefix = format!("pklr-pkg-{}", self.extracted.len());
            self.events.push(AEvent::TempDir(prefix.clone()));
            self.extracted.insert(pkg, format!("tmp/{prefix}"));
        }
        let entry_path = format!("{}/{}", self.extracted[&pkg], package.entry());
        if let Some(value) = self.memo.get(&entry_path) {
            return Ok(value.clone());
        }
        self.memo.insert(entry_path.clone(), serde_json::json!({}));
        self.note_read(entry_path.clone());
        let value = serde_json::json!({ "answer": self.effective_answer(pkg) });
        self.memo.insert(entry_path, value.clone());
        let _ = depth;
        Ok(value)
    }
}

fn rel_string(path: &Path) -> String {
    path.to_string_lossy().replace('\\', "/")
}

// --- Fixed-seed generator ---------------------------------------------------

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }

    fn chance(&mut self, percent: u64) -> bool {
        self.next() % 100 < percent
    }
}

const REWRITE_POOL: &[(&str, &str)] = &[
    ("https://remote.test/", "https://mirror-a.test/"),
    ("https://remote.test/pkg/", "https://mirror-b.test/deep/"),
    ("https://example.com/", "https://cdn-example.test/"),
    ("https://github.com/acme/", "https://gh-mirror.test/acme/"),
];

fn import_uri(
    units: &[Unit],
    packages: &[Package],
    importer: usize,
    target: &Target,
    missing_key: &str,
) -> String {
    match target {
        Target::Local(j) => local_import_uri(&units[importer].dir, &units[*j].path()),
        Target::Remote(j) => {
            if units[importer].remote {
                format!("{}.pkl", units[*j].name)
            } else {
                units[*j].url()
            }
        }
        Target::Package(p) => packages[*p].uri(),
        Target::Missing => format!("missing-{missing_key}.pkl"),
    }
}

fn generate(seed: u64) -> Case {
    let mut rng = Rng(seed.wrapping_add(0x5EED));
    let n_local = 2 + rng.below(MAX_LOCAL_UNITS - 1);
    let n_remote = rng.below(MAX_REMOTE_UNITS + 1);
    let n_pkg = rng.below(MAX_PACKAGES + 1);

    let mut units = Vec::new();
    for i in 0..n_local {
        units.push(Unit {
            name: format!("m{i}"),
            remote: false,
            dir: if rng.chance(30) { "mod/sub".into() } else { "mod".into() },
            id: 10 + i as i64,
            imports: Vec::new(),
            amends: None,
            env_reads: Vec::new(),
            opt_reads: Vec::new(),
        });
    }
    for i in 0..n_remote {
        units.push(Unit {
            name: format!("r{i}"),
            remote: true,
            dir: "remote".into(),
            id: 100 + i as i64,
            imports: Vec::new(),
            amends: None,
            env_reads: Vec::new(),
            opt_reads: Vec::new(),
        });
    }
    let packages: Vec<Package> = (0..n_pkg)
        .map(|i| Package {
            name: format!("p{i}"),
            direct: rng.chance(40),
            answer: 1000 + i as i64,
            stale_answer: -(1000 + i as i64),
        })
        .collect();

    for index in 0..units.len() {
        let n_imports = rng.below(MAX_IMPORTS_PER_UNIT + 1);
        let mut used_aliases = HashSet::new();
        for k in 0..n_imports {
            let choice = rng.below(100);
            let remote_unit = units[index].remote;
            let (target, alias) = if remote_unit {
                // Remote modules may import later remote modules (a DAG, so
                // remote evaluation cannot diverge) or packages.
                if choice < 50 {
                    let candidates: Vec<usize> =
                        (n_local..units.len()).filter(|&j| j > index).collect();
                    if candidates.is_empty() {
                        continue;
                    }
                    let target = candidates[rng.below(candidates.len())];
                    (Target::Remote(target), units[target].name.clone())
                } else if !packages.is_empty() {
                    let pkg = rng.below(packages.len());
                    (Target::Package(pkg), packages[pkg].name.clone())
                } else {
                    continue;
                }
            } else {
                match choice {
                    0..=39 => {
                        let j = rng.below(n_local);
                        (Target::Local(j), units[j].name.clone())
                    }
                    40..=59 if n_remote > 0 => {
                        let j = n_local + rng.below(n_remote);
                        (Target::Remote(j), units[j].name.clone())
                    }
                    60..=79 if !packages.is_empty() => {
                        let p = rng.below(packages.len());
                        (Target::Package(p), packages[p].name.clone())
                    }
                    80..=89 => (Target::Missing, format!("x{index}_{k}")),
                    _ => continue,
                }
            };
            if !used_aliases.insert(alias.clone()) {
                continue;
            }
            let missing_key = format!("{index}_{k}");
            let uri = import_uri(&units, &packages, index, &target, &missing_key);
            let referenced = target == Target::Missing || rng.chance(80);
            units[index].imports.push(Import {
                alias,
                target,
                uri,
                referenced,
            });
        }
    }

    for index in 0..n_local {
        if rng.chance(30) {
            let target = rng.below(n_local);
            if target != index {
                units[index].amends = Some(target);
                // Importing the amended module hits the deferred-inherited
                // path, which is outside the model's state space.
                let alias = units[target].name.clone();
                units[index].imports.retain(|imp| imp.alias != alias);
            }
        }
    }

    let mut env = BTreeMap::new();
    for unit in &mut units {
        let n_env = rng.below(MAX_ENV_READS_PER_UNIT + 1);
        for k in 0..n_env {
            let var = format!("PKLR_MODEL_{}_{k}", unit.name.to_uppercase());
            unit.env_reads.push(var.clone());
            env.insert(var, format!("val-{}-{k}", unit.name));
        }
        if rng.chance(40) {
            unit.opt_reads
                .push(format!("PKLR_MODEL_MISSING_{}", unit.name.to_uppercase()));
        }
    }

    let mut rewrites = Vec::new();
    let n_rules = rng.below(MAX_REWRITE_RULES + 1);
    let mut pool: Vec<usize> = (0..REWRITE_POOL.len()).collect();
    for _ in 0..n_rules {
        let pick = rng.below(pool.len());
        let index = pool.remove(pick);
        let (src, tgt) = REWRITE_POOL[index];
        rewrites.push(format!("{src}={tgt}"));
    }

    let offline = rng.chance(25);
    let preloads: Vec<usize> = (0..n_pkg).filter(|_| rng.chance(40)).collect();
    let mut cache_seeds = Vec::new();
    for pkg in 0..n_pkg {
        let seed = match rng.below(100) {
            0..=34 => CacheSeed::Absent,
            35..=54 => CacheSeed::Valid,
            55..=64 => CacheSeed::ValidStale,
            65..=72 => CacheSeed::CorruptMeta,
            73..=80 => CacheSeed::BadDigest,
            81..=87 => CacheSeed::UnsupportedVersion,
            _ => CacheSeed::StaleTemp,
        };
        if seed != CacheSeed::Absent {
            cache_seeds.push((pkg, seed));
        }
    }

    Case {
        units,
        packages,
        entry: 0,
        rewrites,
        offline,
        preloads,
        cache_seeds,
        env,
    }
}

// --- Replayable case specs --------------------------------------------------

impl Case {
    fn to_spec(&self) -> String {
        let mut out = String::from("case v1\n");
        out.push_str(&format!("entry {}\n", self.units[self.entry].name));
        out.push_str(&format!("offline {}\n", u8::from(self.offline)));
        for rule in &self.rewrites {
            out.push_str(&format!("rewrite {rule}\n"));
        }
        for (name, value) in &self.env {
            out.push_str(&format!("env {name} {value}\n"));
        }
        for unit in &self.units {
            if unit.remote {
                out.push_str(&format!("unit {} remote - {}\n", unit.name, unit.id));
            } else {
                out.push_str(&format!("unit {} local {} {}\n", unit.name, unit.dir, unit.id));
            }
        }
        for unit in &self.units {
            if let Some(amends) = unit.amends {
                out.push_str(&format!("amends {} {}\n", unit.name, self.units[amends].name));
            }
            for var in &unit.env_reads {
                out.push_str(&format!("envread {} {var}\n", unit.name));
            }
            for var in &unit.opt_reads {
                out.push_str(&format!("optread {} {var}\n", unit.name));
            }
            for import in &unit.imports {
                let (kind, target) = match import.target {
                    Target::Local(j) => ("local", self.units[j].name.clone()),
                    Target::Remote(j) => ("remote", self.units[j].name.clone()),
                    Target::Package(p) => ("package", self.packages[p].name.clone()),
                    Target::Missing => ("missing", "-".to_string()),
                };
                let referenced = if import.referenced { "ref" } else { "unref" };
                out.push_str(&format!(
                    "import {} {} {kind} {target} {referenced} {}\n",
                    unit.name, import.alias, import.uri
                ));
            }
        }
        for package in &self.packages {
            let kind = if package.direct { "direct" } else { "zip" };
            out.push_str(&format!(
                "package {} {kind} {} {}\n",
                package.name, package.answer, package.stale_answer
            ));
        }
        for pkg in &self.preloads {
            out.push_str(&format!("preload {}\n", self.packages[*pkg].name));
        }
        for (pkg, seed) in &self.cache_seeds {
            out.push_str(&format!(
                "cache {} {}\n",
                self.packages[*pkg].name,
                seed.spec_name()
            ));
        }
        out
    }

    fn from_spec(spec: &str) -> Result<Case, String> {
        let mut lines = spec.lines();
        if lines.next() != Some("case v1") {
            return Err("spec must start with `case v1`".to_string());
        }
        let mut entry_name = None;
        let mut offline = false;
        let mut rewrites = Vec::new();
        let mut env = BTreeMap::new();
        let mut units: Vec<Unit> = Vec::new();
        let mut packages: Vec<Package> = Vec::new();
        let mut preloads = Vec::new();
        let mut cache_seeds = Vec::new();
        let mut amends: Vec<(String, String)> = Vec::new();
        let mut imports: Vec<(String, Import, String)> = Vec::new();
        let mut env_reads: Vec<(String, String)> = Vec::new();
        let mut opt_reads: Vec<(String, String)> = Vec::new();

        let unit_index = |units: &[Unit], name: &str| -> Result<usize, String> {
            units
                .iter()
                .position(|unit| unit.name == name)
                .ok_or_else(|| format!("unknown unit {name}"))
        };
        let pkg_index = |packages: &[Package], name: &str| -> Result<usize, String> {
            packages
                .iter()
                .position(|pkg| pkg.name == name)
                .ok_or_else(|| format!("unknown package {name}"))
        };

        for line in lines {
            let parts: Vec<&str> = line.split(' ').collect();
            match parts.as_slice() {
                ["entry", name] => entry_name = Some(name.to_string()),
                ["offline", value] => offline = *value == "1",
                ["rewrite", rule] => rewrites.push(rule.to_string()),
                ["env", name, value] => {
                    env.insert(name.to_string(), value.to_string());
                }
                ["unit", name, kind, dir, id] => units.push(Unit {
                    name: name.to_string(),
                    remote: *kind == "remote",
                    dir: dir.to_string(),
                    id: id.parse().map_err(|_| "bad unit id".to_string())?,
                    imports: Vec::new(),
                    amends: None,
                    env_reads: Vec::new(),
                    opt_reads: Vec::new(),
                }),
                ["amends", unit, target] => {
                    amends.push((unit.to_string(), target.to_string()));
                }
                ["envread", unit, var] => {
                    env_reads.push((unit.to_string(), var.to_string()));
                }
                ["optread", unit, var] => {
                    opt_reads.push((unit.to_string(), var.to_string()));
                }
                ["import", unit, alias, kind, target, referenced, uri] => {
                    let target = match *kind {
                        "local" => Target::Local(usize::MAX),
                        "remote" => Target::Remote(usize::MAX),
                        "package" => Target::Package(usize::MAX),
                        "missing" => Target::Missing,
                        other => return Err(format!("bad import kind {other}")),
                    };
                    imports.push((
                        unit.to_string(),
                        Import {
                            alias: alias.to_string(),
                            target,
                            uri: uri.to_string(),
                            referenced: *referenced == "ref",
                        },
                        target_name_or_dash(kind, target),
                    ));
                }
                ["package", name, kind, answer, stale] => packages.push(Package {
                    name: name.to_string(),
                    direct: *kind == "direct",
                    answer: answer.parse().map_err(|_| "bad answer".to_string())?,
                    stale_answer: stale.parse().map_err(|_| "bad stale answer".to_string())?,
                }),
                ["preload", name] => preloads.push(name.to_string()),
                ["cache", name, seed] => cache_seeds.push((
                    name.to_string(),
                    CacheSeed::from_spec(seed).ok_or("bad cache seed")?,
                )),
                other => return Err(format!("unrecognized spec line: {other:?}")),
            }
        }

        fn target_name_or_dash(kind: &str, target: &str) -> String {
            if kind == "missing" {
                "-".to_string()
            } else {
                target.to_string()
            }
        }

        let mut case = Case {
            entry: 0,
            offline,
            rewrites,
            env,
            preloads: Vec::new(),
            cache_seeds: Vec::new(),
            units,
            packages,
        };
        case.entry = unit_index(
            &case.units,
            entry_name.as_deref().ok_or("missing entry line")?,
        )?;
        for (unit, target) in amends {
            let index = unit_index(&case.units, &unit)?;
            case.units[index].amends = Some(unit_index(&case.units, &target)?);
        }
        for (unit, var) in env_reads {
            let index = unit_index(&case.units, &unit)?;
            case.units[index].env_reads.push(var);
        }
        for (unit, var) in opt_reads {
            let index = unit_index(&case.units, &unit)?;
            case.units[index].opt_reads.push(var);
        }
        for (unit, mut import, target_name) in imports {
            let index = unit_index(&case.units, &unit)?;
            import.target = match import.target {
                Target::Local(_) => Target::Local(unit_index(&case.units, &target_name)?),
                Target::Remote(_) => Target::Remote(unit_index(&case.units, &target_name)?),
                Target::Package(_) => Target::Package(pkg_index(&case.packages, &target_name)?),
                Target::Missing => Target::Missing,
            };
            case.units[index].imports.push(import);
        }
        for name in preloads {
            case.preloads.push(pkg_index(&case.packages, &name)?);
        }
        for (name, seed) in cache_seeds {
            case.cache_seeds.push((pkg_index(&case.packages, &name)?, seed));
        }
        Ok(case)
    }
}

// --- Driver -----------------------------------------------------------------

fn meta_json(version: u64, url: &str, ext: &str, digest: &str, length: usize) -> String {
    serde_json::json!({
        "version": version,
        "url": url,
        "extension": ext,
        "digest": digest,
        "length": length,
    })
    .to_string()
}

fn write_cache_entry(cache: &Path, url: &str, ext: &str, bytes: &[u8]) {
    let digest = pklr::cache::content_digest(bytes);
    write_cache_entry_with(cache, url, ext, bytes, &digest, 1);
}

fn write_cache_entry_with(
    cache: &Path,
    url: &str,
    ext: &str,
    bytes: &[u8],
    digest: &str,
    version: u64,
) {
    let blob = pklr::cache::blob_path(cache, url, digest, ext).unwrap();
    let meta = pklr::cache::meta_path(cache, url);
    std::fs::create_dir_all(blob.parent().unwrap()).unwrap();
    std::fs::write(&blob, bytes).unwrap();
    std::fs::write(meta, meta_json(version, url, ext, digest, bytes.len())).unwrap();
}

fn seed_cache(case: &Case, root: &Path) {
    let cache = root.join("cache");
    for (pkg_index, seed) in &case.cache_seeds {
        let package = &case.packages[*pkg_index];
        let url = package.fetch_url();
        match seed {
            CacheSeed::Absent => {}
            CacheSeed::Valid => write_cache_entry(&cache, &url, package.ext(), &package.bytes()),
            CacheSeed::ValidStale => {
                write_cache_entry(&cache, &url, package.ext(), &package.stale_bytes());
            }
            CacheSeed::CorruptMeta => {
                let meta = pklr::cache::meta_path(&cache, &url);
                std::fs::create_dir_all(meta.parent().unwrap()).unwrap();
                std::fs::write(meta, b"{ not valid json").unwrap();
            }
            CacheSeed::BadDigest => {
                let wrong = pklr::cache::content_digest(b"not the real content");
                write_cache_entry_with(&cache, &url, package.ext(), &package.bytes(), &wrong, 1);
            }
            CacheSeed::UnsupportedVersion => {
                let digest = pklr::cache::content_digest(&package.bytes());
                write_cache_entry_with(&cache, &url, package.ext(), &package.bytes(), &digest, 99);
            }
            CacheSeed::StaleTemp => {
                let packages = pklr::cache::packages_dir(&cache);
                std::fs::create_dir_all(&packages).unwrap();
                std::fs::write(packages.join(".half-written.tmp-123-4"), b"partial").unwrap();
                let orphan_digest = pklr::cache::content_digest(b"orphan");
                let orphan = pklr::cache::blob_path(
                    &cache,
                    "https://orphan.test/x.zip",
                    &orphan_digest,
                    "zip",
                )
                .unwrap();
                std::fs::write(orphan, b"orphan").unwrap();
            }
        }
    }
}

fn build_host(case: &Case, root: &Path) -> ScriptedHost {
    let mut host = ScriptedHost::new(root);
    for (_, unit) in case.local_units() {
        let path = root.join(unit.path());
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, unit.source(case, root)).unwrap();
    }
    let rules = parse_rewrites(&case.rewrites);
    for unit in case.units.iter().filter(|unit| unit.remote) {
        let url = rewrite(&rules, &unit.url());
        host.http_text.insert(url, unit.source(case, root));
    }
    for package in &case.packages {
        let url = rewrite(&rules, &package.fetch_url());
        host.http_bytes.insert(url, package.bytes());
    }
    host.env = case.env.clone().into_iter().collect();
    seed_cache(case, root);
    host
}

async fn evaluate(case: &Case, root: &Path) -> (Result<serde_json::Value, String>, Vec<Event>) {
    let host = build_host(case, root);
    let log = host.log.clone();
    let mut evaluator = Evaluator::with_capabilities(host);
    evaluator.set_base_path(root);
    evaluator.set_http_rewrites(&case.rewrites);
    evaluator.set_package_cache_dir(root.join("cache"));
    evaluator.set_offline(case.offline);
    for pkg in &case.preloads {
        let package = &case.packages[*pkg];
        evaluator
            .preload_package_async(&package.fetch_url(), package.ext(), &package.bytes())
            .await
            .unwrap();
    }
    let entry = root.join(case.entry_path());
    let root_display = root.display().to_string();
    let result = evaluator
        .eval_file_pub(&entry)
        .await
        .map(|value| value.to_json())
        .map_err(|error| error.to_string().replace(&root_display, "<root>"));
    let events = log.lock().unwrap().clone();
    (result, events)
}

/// Post-run filesystem invariants: startup cleanup must remove recognizable
/// temp files, and no blob may exist without metadata referencing it.
fn check_run_artifacts(root: &Path, log: &[Event]) -> Result<(), String> {
    let cache_touched = log.iter().any(|event| match event {
        Event::ListDir(path) | Event::TempCreate(path) | Event::Read(path) => {
            path.starts_with("cache/")
        }
        _ => false,
    });
    if !cache_touched {
        return Ok(());
    }
    let packages = root.join("cache").join("packages");
    let files = contract_support::list_files(&packages);
    for name in &files {
        if pklr::cache::is_temp_file_name(name) {
            return Err(format!("stale temp file survived startup cleanup: {name}"));
        }
    }
    let mut referenced = HashSet::new();
    for name in &files {
        let Some(urlhash) = name.strip_suffix(".meta") else {
            continue;
        };
        let text = std::fs::read_to_string(packages.join(name)).unwrap();
        let Ok(value) = serde_json::from_str::<serde_json::Value>(&text) else {
            continue;
        };
        if value.get("version").and_then(serde_json::Value::as_u64)
            == Some(pklr::cache::CACHE_FORMAT_VERSION)
            && let Some(digest) = value.get("digest").and_then(serde_json::Value::as_str)
        {
            referenced.insert(format!("{urlhash}.{digest}"));
        }
    }
    for name in &files {
        let parts: Vec<&str> = name.split('.').collect();
        if parts.len() == 3 && parts[2] != "meta" {
            let key = format!("{}.{}", parts[0], parts[1]);
            if !referenced.contains(&key) {
                return Err(format!("orphan blob survived startup cleanup: {name}"));
            }
        }
    }
    Ok(())
}

fn first_divergence<T: std::fmt::Debug + PartialEq>(left: &[T], right: &[T]) -> String {
    let mut out = String::new();
    let index = left
        .iter()
        .zip(right.iter())
        .position(|(a, b)| a != b)
        .unwrap_or(left.len().min(right.len()));
    out.push_str(&format!("first divergence at index {index}:\n"));
    out.push_str(&format!("  left:  {:?}\n", left.get(index)));
    out.push_str(&format!("  right: {:?}\n", right.get(index)));
    out.push_str(&format!("left ({} events):\n", left.len()));
    for (i, event) in left.iter().enumerate() {
        out.push_str(&format!("  {i}: {event:?}\n"));
    }
    out.push_str(&format!("right ({} events):\n", right.len()));
    for (i, event) in right.iter().enumerate() {
        out.push_str(&format!("  {i}: {event:?}\n"));
    }
    out
}

fn check_case(case: &Case) -> Result<(), String> {
    let (model_events, model_outcome) = Model::new(case).run();

    let sync_root = TestRoot::new("sync");
    let (sync_outcome, sync_log) = pollster::block_on(evaluate(case, sync_root.path()));
    check_run_artifacts(sync_root.path(), &sync_log)?;

    let async_root = TestRoot::new("async");
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let (async_outcome, async_log) = runtime.block_on(evaluate(case, async_root.path()));
    check_run_artifacts(async_root.path(), &async_log)?;

    if sync_log != async_log {
        return Err(format!(
            "synchronous and asynchronous event logs diverge:\n{}",
            first_divergence(&sync_log, &async_log)
        ));
    }
    if sync_outcome != async_outcome {
        return Err(format!(
            "synchronous and asynchronous outcomes diverge:\nsync:  {sync_outcome:?}\nasync: {async_outcome:?}"
        ));
    }
    let projected = project(&sync_log);
    if projected != model_events {
        return Err(format!(
            "implementation trace diverges from model:\n{}",
            first_divergence(&projected, &model_events)
        ));
    }
    match (&model_outcome, &sync_outcome) {
        (Ok(expected), Ok(actual)) if expected == actual => Ok(()),
        (Err(pattern), Err(actual)) if actual.contains(pattern.as_str()) => Ok(()),
        (model, actual) => Err(format!(
            "outcome mismatch:\nmodel: {model:?}\nimpl:  {actual:?}"
        )),
    }
}

// --- Greedy shrinker ----------------------------------------------------------

fn shrink_candidates(case: &Case) -> Vec<Case> {
    let mut out = Vec::new();
    for (u, unit) in case.units.iter().enumerate() {
        for i in 0..unit.imports.len() {
            let mut candidate = case.clone();
            candidate.units[u].imports.remove(i);
            out.push(candidate);
        }
        if unit.amends.is_some() {
            let mut candidate = case.clone();
            candidate.units[u].amends = None;
            out.push(candidate);
        }
        for i in 0..unit.env_reads.len() {
            let mut candidate = case.clone();
            let var = candidate.units[u].env_reads.remove(i);
            candidate.env.remove(&var);
            out.push(candidate);
        }
        for i in 0..unit.opt_reads.len() {
            let mut candidate = case.clone();
            candidate.units[u].opt_reads.remove(i);
            out.push(candidate);
        }
    }
    for i in 0..case.preloads.len() {
        let mut candidate = case.clone();
        candidate.preloads.remove(i);
        out.push(candidate);
    }
    for i in 0..case.cache_seeds.len() {
        let mut candidate = case.clone();
        candidate.cache_seeds.remove(i);
        out.push(candidate);
    }
    for i in 0..case.rewrites.len() {
        let mut candidate = case.clone();
        candidate.rewrites.remove(i);
        out.push(candidate);
    }
    for p in 0..case.packages.len() {
        let referenced = case
            .units
            .iter()
            .any(|unit| unit.imports.iter().any(|imp| imp.target == Target::Package(p)))
            || case.preloads.contains(&p)
            || case.cache_seeds.iter().any(|(pkg, _)| *pkg == p);
        if referenced {
            continue;
        }
        let mut candidate = case.clone();
        candidate.packages.remove(p);
        for unit in &mut candidate.units {
            for imp in &mut unit.imports {
                if let Target::Package(x) = &mut imp.target
                    && *x > p
                {
                    *x -= 1;
                }
            }
        }
        candidate.preloads = candidate
            .preloads
            .iter()
            .map(|&x| if x > p { x - 1 } else { x })
            .collect();
        candidate.cache_seeds = candidate
            .cache_seeds
            .iter()
            .filter(|(pkg, _)| *pkg != p)
            .map(|(pkg, seed)| (if *pkg > p { pkg - 1 } else { *pkg }, *seed))
            .collect();
        out.push(candidate);
    }
    for u in 0..case.units.len() {
        if u == case.entry {
            continue;
        }
        let referenced = case.units.iter().any(|other| {
            other.amends == Some(u)
                || other
                    .imports
                    .iter()
                    .any(|imp| imp.target == Target::Local(u) || imp.target == Target::Remote(u))
        });
        if referenced {
            continue;
        }
        let mut candidate = case.clone();
        candidate.units.remove(u);
        if candidate.entry > u {
            candidate.entry -= 1;
        }
        for unit in &mut candidate.units {
            if let Some(amends) = &mut unit.amends
                && *amends > u
            {
                *amends -= 1;
            }
            for imp in &mut unit.imports {
                match &mut imp.target {
                    Target::Local(x) | Target::Remote(x) if *x > u => *x -= 1,
                    _ => {}
                }
            }
        }
        out.push(candidate);
    }
    out
}

fn shrink(case: &Case) -> Case {
    let mut current = case.clone();
    'outer: for _ in 0..MAX_SHRINK_ROUNDS {
        for candidate in shrink_candidates(&current) {
            if check_case(&candidate).is_err() {
                current = candidate;
                continue 'outer;
            }
        }
        break;
    }
    current
}
