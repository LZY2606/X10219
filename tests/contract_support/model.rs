//! Simplified reference model for a restricted Pkl subset.
//!
//! The model mirrors the observable protocol of the real evaluator on this
//! subset: URI normalization, longest-prefix rewrites, relative file imports,
//! package metadata/blob caching, preloads, offline hit/miss, amend edges, and
//! cycle handling. It produces (a) a semantic ordered event trace and
//! (b) the expected JSON result / error category. Contract tests compare the
//! implementation against both.

use std::collections::{BTreeMap, BTreeSet};

use super::{CACHE_DIR, normalize, Event};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GateOp {
    And,
    Or,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PropKind {
    Int(i64),
    /// `read?("env:left") != null <op> read?("env:right") != null`
    Gate {
        op: GateOp,
        left: String,
        right: String,
    },
    /// `name = alias.field`
    Ref { alias: String, field: String },
}

#[derive(Debug, Clone)]
pub struct Prop {
    pub name: String,
    pub kind: PropKind,
}

#[derive(Debug, Clone)]
pub enum Location {
    /// Virtual absolute path, e.g. `/virt/proj/main.pkl`.
    Local(String),
    /// Absolute http(s) URL.
    Http(String),
    /// Direct-download package: package URI plus resolved download URL.
    PkgDirect { pkg_uri: String, download_url: String },
}

impl Location {
    pub fn key(&self) -> String {
        match self {
            Location::Local(path) => path.clone(),
            Location::Http(url) => url.clone(),
            Location::PkgDirect { download_url, .. } => download_url.clone(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Edge {
    /// Index of the target node.
    pub to: usize,
    /// URI as written in the importing source.
    pub uri: String,
    pub alias: String,
    pub used: bool,
}

#[derive(Debug, Clone, Default)]
pub struct Node {
    pub loc: Option<Location>,
    pub props: Vec<Prop>,
    pub imports: Vec<Edge>,
    pub amends: Option<Edge>,
}

#[derive(Debug, Clone)]
pub struct Preload {
    pub url: String,
    pub extension: String,
    pub bytes: Vec<u8>,
}

#[derive(Debug, Clone)]
pub struct PrestateFile {
    pub path: String,
    pub bytes: Vec<u8>,
}

#[derive(Debug, Clone, Default)]
pub struct Case {
    pub name: String,
    pub nodes: Vec<Node>,
    pub entry: usize,
    pub rewrites: Vec<(String, String)>,
    pub offline: bool,
    pub env: BTreeMap<String, String>,
    pub preloads: Vec<Preload>,
    pub prestate: Vec<PrestateFile>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Outcome {
    Ok(String),
    Err(String),
}

/// Model-predicted observation of one evaluation.
pub struct Prediction {
    pub trace: Vec<(String, String)>,
    pub outcome: Outcome,
    pub env_reads: BTreeMap<String, bool>,
}

// ---------------------------------------------------------------------------
// URI helpers (subset, mirrors the implementation's rules)
// ---------------------------------------------------------------------------

pub fn rewrite(rewrites: &[(String, String)], url: &str) -> String {
    let best = rewrites
        .iter()
        .filter(|(src, _)| url.starts_with(src.as_str()))
        .max_by_key(|(src, _)| src.len());
    match best {
        Some((src, tgt)) => format!("{tgt}{}", &url[src.len()..]),
        None => url.to_string(),
    }
}

/// Resolve a relative reference against an http(s) base URL (RFC 3986 join).
pub fn http_resolve(base: &str, uri: &str) -> Option<String> {
    if uri.contains("://") || uri.starts_with("pkl:") {
        return None;
    }
    let base_url = url::Url::parse(base).ok()?;
    Some(base_url.join(uri).ok()?.to_string())
}

fn parent(path: &str) -> String {
    let normalized = normalize(path);
    match normalized.rfind('/') {
        Some(index) => normalized[..=index].to_string(),
        None => "/".to_string(),
    }
}

fn join_local(base_path: &str, uri: &str) -> String {
    normalize(&format!("{}{}", parent(base_path), uri))
}

// ---------------------------------------------------------------------------
// Cache layout (must match src/eval.rs)
// ---------------------------------------------------------------------------

fn fnv1a64(input: &str) -> u64 {
    let mut hash = 0xcbf29ce484222325_u64;
    for byte in input.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

fn metadata_path(url: &str) -> String {
    format!("{CACHE_DIR}/packages/{:016x}.json", fnv1a64(url))
}

fn sha256(input: &[u8]) -> [u8; 32] {
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ];
    let mut h: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ];
    let bit_len = (input.len() as u64).wrapping_mul(8);
    let mut padded = input.to_vec();
    padded.push(0x80);
    while padded.len() % 64 != 56 {
        padded.push(0);
    }
    padded.extend_from_slice(&bit_len.to_be_bytes());
    for chunk in padded.chunks_exact(64) {
        let mut w = [0u32; 64];
        for (i, word) in chunk.chunks_exact(4).enumerate() {
            w[i] = u32::from_be_bytes([word[0], word[1], word[2], word[3]]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }
        let (mut a, mut b, mut c, mut d) = (h[0], h[1], h[2], h[3]);
        let (mut e, mut f, mut g, mut hh) = (h[4], h[5], h[6], h[7]);
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let t1 = hh
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[i])
                .wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let t2 = s0.wrapping_add(maj);
            hh = g;
            g = f;
            f = e;
            e = d.wrapping_add(t1);
            d = c;
            c = b;
            b = a;
            a = t1.wrapping_add(t2);
        }
        h[0] = h[0].wrapping_add(a);
        h[1] = h[1].wrapping_add(b);
        h[2] = h[2].wrapping_add(c);
        h[3] = h[3].wrapping_add(d);
        h[4] = h[4].wrapping_add(e);
        h[5] = h[5].wrapping_add(f);
        h[6] = h[6].wrapping_add(g);
        h[7] = h[7].wrapping_add(hh);
    }
    let mut digest = [0u8; 32];
    for (i, v) in h.iter().enumerate() {
        digest[i * 4..i * 4 + 4].copy_from_slice(&v.to_be_bytes());
    }
    digest
}

fn blob_name(bytes: &[u8], extension: &str) -> String {
    let hex: String = sha256(bytes).iter().map(|b| format!("{b:02x}")).collect();
    format!("{hex}.{extension}")
}

fn blob_path(blob: &str) -> String {
    format!("{CACHE_DIR}/blobs/{blob}")
}

// ---------------------------------------------------------------------------
// Interpreter
// ---------------------------------------------------------------------------

struct World<'a> {
    case: &'a Case,
    files: BTreeMap<String, Vec<u8>>,
    http_sources: BTreeMap<String, String>,
    http_bytes: BTreeMap<String, Vec<u8>>,
    trace: Vec<(String, String)>,
    env_reads: BTreeMap<String, bool>,
    /// Evaluated module values (location key -> ordered JSON-ish map).
    module_values: BTreeMap<String, BTreeMap<String, String>>,
    /// Modules with an in-flight import evaluation (placeholder).
    import_in_flight: BTreeSet<String>,
    /// Local modules with an in-flight scoped amend evaluation.
    amend_in_flight: BTreeSet<String>,
    http_cache: BTreeSet<String>,
    swept: bool,
}

impl<'a> World<'a> {
    fn ev(&mut self, label: &str, arg: impl Into<String>) {
        self.trace.push((label.to_string(), arg.into()));
    }

    fn error(&self, category: &str) -> Outcome {
        Outcome::Err(category.to_string())
    }

    fn read_file(&mut self, path: &str) -> Option<Vec<u8>> {
        let path = normalize(path);
        self.ev("read", path.clone());
        self.files.get(&path).cloned()
    }

    fn exists(&mut self, path: &str) -> bool {
        let path = normalize(path);
        self.ev("exists", path.clone());
        self.files.contains_key(&path)
    }

    fn canon(&mut self, path: &str) -> String {
        let path = normalize(path);
        self.ev("canon", path.clone());
        path
    }

    fn sweep(&mut self) {
        if self.swept {
            return;
        }
        self.swept = true;
        let mut leftovers = Vec::new();
        for dir in [format!("{CACHE_DIR}/packages"), format!("{CACHE_DIR}/blobs")] {
            for name in self
                .files
                .keys()
                .filter_map(|path| {
                    let prefix = format!("{dir}/");
                    path.strip_prefix(&prefix)
                        .filter(|rest| !rest.contains('/') && rest.contains(".staging-"))
                        .map(|rest| (dir.clone(), rest.to_string()))
                })
            {
                leftovers.push(name);
            }
        }
        leftovers.sort();
        for (dir, name) in leftovers {
            let path = format!("{dir}/{name}");
            self.files.remove(&path);
            self.ev("sweep", path);
        }
    }

    /// Fetch a direct-package .pkl module through the persistent cache.
    fn package_pkl(&mut self, download_url: &str) -> Result<String, Outcome> {
        self.sweep();
        let meta_path = metadata_path(download_url);
        self.ev("read", meta_path.clone());
        let cached = match self.lookup_cache(download_url, "pkl") {
            Ok(bytes) => bytes,
            Err(outcome) => return Err(outcome),
        };
        if let Some(bytes) = cached {
            return String::from_utf8(bytes).map_err(|_| self.error("invalid-utf8"));
        }
        if self.case.offline {
            return Err(self.error("offline-miss"));
        }
        let fetch_url = rewrite(&self.case.rewrites, download_url);
        self.ev("fetch_bytes", fetch_url.clone());
        let bytes = self
            .http_bytes
            .get(&fetch_url)
            .cloned()
            .ok_or_else(|| self.error("fetch-failed"))?;
        self.publish_cache(download_url, "pkl", &bytes);
        String::from_utf8(bytes).map_err(|_| self.error("invalid-utf8"))
    }

    fn lookup_cache(
        &mut self,
        url: &str,
        extension: &str,
    ) -> Result<Option<Vec<u8>>, Outcome> {
        let meta_path = metadata_path(url);
        let raw_meta = match self.files.get(&meta_path) {
            Some(raw) => raw.clone(),
            None => return Ok(None),
        };
        let value: serde_json::Value = serde_json::from_slice(&raw_meta)
            .map_err(|_| self.error("corrupt-metadata"))?;
        let version = value
            .get("version")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| self.error("corrupt-metadata"))?;
        if version != 1 {
            return Err(self.error("unsupported-version"));
        }
        if value.get("url").and_then(|v| v.as_str()) != Some(url) {
            return Err(self.error("corrupt-metadata"));
        }
        if value.get("extension").and_then(|v| v.as_str()) != Some(extension) {
            return Err(self.error("corrupt-metadata"));
        }
        let blob = value
            .get("blob")
            .and_then(|v| v.as_str())
            .ok_or_else(|| self.error("corrupt-metadata"))?;
        let digest = value
            .get("sha256")
            .and_then(|v| v.as_str())
            .ok_or_else(|| self.error("corrupt-metadata"))?;
        if blob.contains('/')
            || blob.contains('\\')
            || blob.contains("..")
            || blob.split('.').count() != 2
            || digest.len() != 64
            || !digest.bytes().all(|b| b.is_ascii_hexdigit())
        {
            return Err(self.error("containment"));
        }
        let blob_p = blob_path(blob);
        self.ev("read", blob_p.clone());
        let bytes = match self.files.get(&blob_p) {
            Some(bytes) => bytes.clone(),
            None => return Err(self.error("corrupt-metadata")),
        };
        let actual: String = sha256(&bytes).iter().map(|b| format!("{b:02x}")).collect();
        if actual != digest {
            return Err(self.error("digest-mismatch"));
        }
        Ok(Some(bytes))
    }

    fn publish_cache(&mut self, url: &str, extension: &str, bytes: &[u8]) {
        let blob = blob_name(bytes, extension);
        let blob_p = blob_path(&blob);
        // The evaluator checks blob immutability through path_exists.
        let already_present = {
            let path = blob_p.clone();
            self.ev("exists", path);
            self.files.contains_key(&blob_p)
        };
        if !already_present {
            self.files.insert(blob_p.clone(), bytes.to_vec());
            for phase in ["flush", "rename", "sync"] {
                self.ev("blob_stage", phase);
            }
        }
        let meta = serde_json::json!({
            "version": 1,
            "url": url,
            "extension": extension,
            "sha256": sha256(bytes).iter().map(|b| format!("{b:02x}")).collect::<String>(),
            "blob": blob,
        });
        let meta_bytes = serde_json::to_vec(&meta).unwrap();
        let meta_path = metadata_path(url);
        self.files.insert(meta_path, meta_bytes);
        for phase in ["flush", "rename", "sync"] {
            self.ev("meta_stage", phase);
        }
    }

    fn eval_module_entry(&mut self, node_idx: usize) -> Result<(), Outcome> {
        self.eval_module_inner(node_idx, false, true)
    }

    /// Evaluate imports/amends of a module, returning its merged property map.
    fn eval_module(&mut self, node_idx: usize, via_amend: bool) -> Result<(), Outcome> {
        self.eval_module_inner(node_idx, via_amend, false)
    }

    fn eval_module_inner(
        &mut self,
        node_idx: usize,
        via_amend: bool,
        is_entry: bool,
    ) -> Result<(), Outcome> {
        let key = self.case.nodes[node_idx].loc.as_ref().unwrap().key();
        if self.module_values.contains_key(&key) {
            return Ok(());
        }
        if via_amend {
            if !self.amend_in_flight.insert(key.clone()) {
                // Mutual amend cycle: this frame contributes an empty module.
                self.module_values
                    .insert(key.clone(), BTreeMap::new());
                return Ok(());
            }
        } else if !self.import_in_flight.insert(key.clone()) {
            self.module_values
                .insert(key.clone(), BTreeMap::new());
            return Ok(());
        }

        // Type-namespace canonicalization leads off every locally imported
        // module body. Scoped amend bodies are entered from eval_file_with_scope,
        // which emits this canonicalization itself.
        if !via_amend
            && !is_entry
            && let Some(Location::Local(path)) = &self.case.nodes[node_idx].loc
        {
            self.canon(path);
        }

        let mut inherited_roots: Vec<usize> = Vec::new();
        if let Some(amend) = &self.case.nodes[node_idx].amends {
            let target = amend.to;
            match &self.case.nodes[target].loc {
                // Local amend: inherited_reference_roots calls
                // load_module_source, which reads the base directly (no
                // existence check) and then recurses through its amends.
                Some(Location::Local(path)) => {
                    if self.exists(path) {
                        self.read_file(path).ok_or_else(|| self.error("io"))?;
                        self.collect_inherited_roots(target, &mut BTreeSet::new())?;
                    }
                }
                Some(_) => {
                    self.collect_inherited_roots(target, &mut BTreeSet::new())?;
                }
                None => {}
            }
        }

        // Imports are processed in declaration order.
        let edges: Vec<Edge> = self.case.nodes[node_idx].imports.clone();
        for edge in edges {
            if !edge.used {
                continue;
            }
            self.eval_import_edge(node_idx, &edge)?;
        }

        // Amend base merges into this module's properties.
        let mut base_values = BTreeMap::new();
        if let Some(amend) = &self.case.nodes[node_idx].amends {
            let target = amend.to;
            match self.case.nodes[target].loc.clone() {
                Some(Location::Local(path)) => {
                    // eval_file_with_scope: existence guard, then canonicalize
                    // (in-flight set) and a second source read.
                    if self.exists(&path) {
                        // eval_file_with_scope: canonicalize (in-flight set),
                        // parse (read), then the type-namespace canonicalize.
                        self.canon(&path);
                        self.read_file(&path).ok_or_else(|| self.error("io"))?;
                        self.canon(&path);
                        if !self.amend_in_flight.contains(&path) {
                            self.eval_module(target, true)?;
                        }
                        // The amends section re-checks existence and reloads
                        // the base source after the base body completes.
                        if self.exists(&path) {
                            self.read_file(&path).ok_or_else(|| self.error("io"))?;
                        }
                        // Scoped evaluation closes with a final canonicalize.
                        self.canon(&path);
                        if let Some(values) = self.module_values.get(&path).cloned() {
                            base_values = values;
                        }
                    }
                }
                Some(loc) => {
                    match &loc {
                        Location::Http(url) => {
                            self.fetch_http(url)?;
                        }
                        Location::PkgDirect { download_url, .. } => {
                            self.package_pkl(download_url)?;
                        }
                        Location::Local(_) => unreachable!("local amends handled above"),
                    }
                    if !self.module_values.contains_key(&loc.key()) {
                        self.eval_module(target, true)?;
                    }
                    if let Some(values) = self.module_values.get(&loc.key()).cloned() {
                        base_values = values;
                    }
                }
                None => {}
            }
        }

        // Evaluate this module's own properties against the merged scope.
        let mut values = base_values;
        let aliases = self.alias_values(node_idx);
        let props = self.case.nodes[node_idx].props.clone();
        for prop in props {
            let value = self.eval_prop(&prop, &aliases, &values)?;
            values.insert(prop.name, value);
        }
        self.module_values.insert(key, values);
        Ok(())
    }

    fn collect_inherited_roots(
        &mut self,
        node_idx: usize,
        seen: &mut BTreeSet<usize>,
    ) -> Result<(), Outcome> {
        if !seen.insert(node_idx) {
            return Ok(());
        }
        // Mirrors inherited_reference_roots: load base source (no eval) and
        // recurse through its amends chain.
        if let Some(amend) = self.case.nodes[node_idx].amends.clone() {
            let target = amend.to;
            // load_module_source for a local edge checks existence first.
            match &self.case.nodes[target].loc {
                Some(Location::Local(path)) => {
                    if self.exists(path) {
                        self.read_file(path).ok_or_else(|| self.error("io"))?;
                    }
                }
                Some(Location::Http(url)) => {
                    self.fetch_http(url)?;
                }
                Some(Location::PkgDirect { download_url, .. }) => {
                    self.package_pkl(download_url)?;
                }
                None => {}
            }
            self.collect_inherited_roots(target, seen)?;
        }
        Ok(())
    }

    fn load_source(&mut self, from_idx: usize, uri: &str) -> Result<(), Outcome> {
        let from = self.case.nodes[from_idx].loc.as_ref().unwrap();
        match from {
            Location::Local(path) => {
                let target_path = join_local(path, uri);
                self.read_file(&target_path).ok_or_else(|| self.error("io"))?;
            }
            Location::Http(base) => {
                let url = http_resolve(base, uri).unwrap_or_else(|| uri.to_string());
                self.fetch_http(&url)?;
            }
            Location::PkgDirect { download_url, .. } => {
                let url = http_resolve(download_url, uri)
                    .unwrap_or_else(|| uri.to_string());
                self.fetch_http(&url)?;
            }
        }
        Ok(())
    }

    fn fetch_http(&mut self, url: &str) -> Result<String, Outcome> {
        if self.http_cache.contains(url) {
            return Ok(self.http_sources.get(url).cloned().unwrap_or_default());
        }
        if self.case.offline {
            return Err(self.error("offline-miss"));
        }
        let final_url = rewrite(&self.case.rewrites, url);
        self.ev("fetch_text", final_url.clone());
        let source = self
            .http_sources
            .get(&final_url)
            .cloned()
            .ok_or_else(|| self.error("fetch-failed"))?;
        self.http_cache.insert(url.to_string());
        Ok(source)
    }
}

impl<'a> World<'a> {
    fn eval_import_edge(&mut self, owner_idx: usize, edge: &Edge) -> Result<(), Outcome> {
        let owner = self.case.nodes[owner_idx].loc.as_ref().unwrap().clone();
        let target = edge.to;
        match self.case.nodes[target].loc.clone().unwrap() {
            Location::Local(path) => {
                // URI resolution: bare relative reference against the owner dir.
                let resolved = match &owner {
                    Location::Local(owner_path) => join_local(owner_path, &edge.uri),
                    Location::Http(base) | Location::PkgDirect { download_url: base, .. } => {
                        http_resolve(base, &edge.uri).unwrap_or_else(|| edge.uri.clone())
                    }
                };
                let _ = resolved;
                if !self.exists(&path) {
                    return Err(self.error("import-not-found"));
                }
                self.canon(&path);
                self.read_file(&path).ok_or_else(|| self.error("io"))?;
                if !self.module_values.contains_key(&path) {
                    self.eval_module(target, false)?;
                }
                self.canon(&path);
            }
            Location::Http(url) => {
                let resolved = match &owner {
                    Location::Http(base) | Location::PkgDirect { download_url: base, .. } => {
                        http_resolve(base, &edge.uri).unwrap_or(url.clone())
                    }
                    _ => url.clone(),
                };
                self.fetch_http(&resolved)?;
                if !self.module_values.contains_key(&self.case.nodes[target].loc.as_ref().unwrap().key()) {
                    self.eval_module(target, false)?;
                }
            }
            Location::PkgDirect {
                pkg_uri: _,
                download_url,
            } => {
                self.package_pkl(&download_url)?;
                if !self.module_values.contains_key(&download_url) {
                    self.eval_module(target, false)?;
                }
            }
        }
        Ok(())
    }

    fn alias_values(&self, owner_idx: usize) -> BTreeMap<String, String> {
        // Map alias -> target module key for Ref resolution.
        let mut map = BTreeMap::new();
        for edge in &self.case.nodes[owner_idx].imports {
            if !edge.used {
                continue;
            }
            if let Some(loc) = &self.case.nodes[edge.to].loc {
                map.insert(edge.alias.clone(), loc.key());
            }
        }
        map
    }

    fn eval_prop(
        &mut self,
        prop: &Prop,
        aliases: &BTreeMap<String, String>,
        own: &BTreeMap<String, String>,
    ) -> Result<String, Outcome> {
        match &prop.kind {
            PropKind::Int(value) => Ok(value.to_string()),
            PropKind::Gate { op, left, right } => {
                let l = self.read_env_gate(left);
                let left_true = l;
                let take_right = match op {
                    GateOp::And => left_true,
                    GateOp::Or => !left_true,
                };
                let r = if take_right {
                    self.read_env_gate(right)
                } else {
                    false
                };
                let result = match op {
                    GateOp::And => left_true && r,
                    GateOp::Or => left_true || r,
                };
                Ok(result.to_string())
            }
            PropKind::Ref { alias, field } => {
                let key = aliases
                    .get(alias)
                    .ok_or_else(|| self.error("field-not-found"))?;
                let value = self
                    .module_values
                    .get(key)
                    .and_then(|values| values.get(field))
                    .or_else(|| own.get(field))
                    .cloned();
                value.clone().or_else(|| {{
                    eprintln!("DBG miss alias={alias} field={field} aliases={:?} keys={:?}", aliases, self.module_values.keys().collect::<Vec<_>>());
                    None
                }})
                .ok_or_else(|| self.error("field-not-found"))
            }
        }
    }

    /// `read?("env:NAME") != null`: records the read only when the branch is
    /// actually evaluated; missing variables read as `null` (false).
    fn read_env_gate(&mut self, name: &str) -> bool {
        self.ev("env", name);
        let present = self.case.env.contains_key(name);
        self.env_reads.insert(name.to_string(), present);
        present
    }

    fn run(mut self) -> Prediction {
        // Preloads publish to the cache before evaluation.
        let preloads = self.case.preloads.clone();
        for preload in preloads {
            self.sweep();
            let meta_path = metadata_path(&preload.url);
            self.ev("read", meta_path.clone());
            // A valid pre-existing entry wins; model cases never pre-seed a
            // valid one alongside a preload, so always publish.
            if !self.files.contains_key(&meta_path) {
                self.publish_cache(&preload.url, &preload.extension, &preload.bytes);
            }
        }

        let entry = self.case.entry;
        // Entry read + two canonicalizations (placeholder seed + post-eval).
        if let Some(Location::Local(path)) = self.case.nodes[entry].loc.clone() {
            self.ev("read", path.clone());
            self.canon(&path);
            self.canon(&path);
            let outcome = match self.eval_module_entry(entry) {
                Ok(()) => {
                    self.canon(&path);
                    match self.module_values.get(&path) {
                        Some(values) => Outcome::Ok(render_json(values)),
                        None => Outcome::Err("no-entry".to_string()),
                    }
                }
                Err(outcome) => outcome,
            };
            Prediction {
                trace: self.trace,
                outcome,
                env_reads: self.env_reads,
            }
        } else {
            let outcome = self
                .eval_module(entry, false)
                .map(|_| Outcome::Ok("{}".to_string()))
                .unwrap_or_else(|outcome| outcome);
            Prediction {
                trace: self.trace,
                outcome,
                env_reads: self.env_reads,
            }
        }
    }
}

fn render_json(values: &BTreeMap<String, String>) -> String {
    let mut map = serde_json::Map::new();
    for (key, raw) in values {
        let value = if raw == "true" {
            serde_json::Value::Bool(true)
        } else if raw == "false" {
            serde_json::Value::Bool(false)
        } else if let Ok(integer) = raw.parse::<i64>() {
            serde_json::Value::from(integer)
        } else {
            serde_json::Value::String(raw.clone())
        };
        map.insert(key.clone(), value);
    }
    serde_json::Value::Object(map).to_string()
}

/// Predict a case from an explicit virtual world state.
pub fn predict(
    case: &Case,
    files: BTreeMap<String, Vec<u8>>,
    http_sources: BTreeMap<String, String>,
    http_bytes: BTreeMap<String, Vec<u8>>,
) -> Prediction {
    World {
        case,
        files,
        http_sources,
        http_bytes,
        trace: Vec::new(),
        env_reads: BTreeMap::new(),
        module_values: BTreeMap::new(),
        import_in_flight: BTreeSet::new(),
        amend_in_flight: BTreeSet::new(),
        http_cache: BTreeSet::new(),
        swept: false,
    }
    .run()
}
