#![allow(dead_code)]

//! Modeled, hermetic test harness: an in-memory filesystem, an ordered event
//! log, scripted HTTP/env answers, and cache write-fault injection. Nothing in
//! here touches the real network or the developer's home directory.

use std::collections::{BTreeMap, HashMap};
use std::path::{Component, Path, PathBuf};
use std::sync::{Arc, Mutex};

use pklr::capabilities::BoxFuture;
use pklr::{Error, EvalCapabilities};

/// Scheduling flavor of a capability call. The model strips this field before
/// comparing traces, because it is the only sync/async-specific observation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dispatch {
    Immediate,
    Spawned,
}

impl Dispatch {
    fn current() -> Dispatch {
        if tokio::runtime::Handle::try_current().is_ok() {
            Dispatch::Spawned
        } else {
            Dispatch::Immediate
        }
    }
}

/// One observable capability interaction, in call order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Event {
    ReadText(Dispatch, PathBuf),
    PathExists(Dispatch, PathBuf),
    Canonicalize(Dispatch, PathBuf),
    ReadBytes(Dispatch, PathBuf),
    CreateDirAll(Dispatch, PathBuf),
    WriteNew(Dispatch, PathBuf),
    SyncFile(Dispatch, PathBuf),
    Rename(Dispatch, PathBuf, PathBuf),
    SyncDir(Dispatch, PathBuf),
    ListDir(Dispatch, PathBuf),
    RemoveDirAll(Dispatch, PathBuf),
    RemoveFile(Dispatch, PathBuf),
    ReadEnv(Dispatch, String),
    FetchText(Dispatch, String),
    FetchBytes(Dispatch, String),
    TempDir(Dispatch, String),
    ExtractZip(Dispatch, PathBuf),
    Glob(Dispatch, PathBuf, String),
}

impl Event {
    /// Drop scheduling-only data so sync and async traces can be compared.
    pub fn semantic(self) -> SemanticEvent {
        match self {
            Event::ReadText(_, p) => SemanticEvent::ReadText(p),
            Event::PathExists(_, p) => SemanticEvent::PathExists(p),
            Event::Canonicalize(_, p) => SemanticEvent::Canonicalize(p),
            Event::ReadBytes(_, p) => SemanticEvent::ReadBytes(p),
            Event::CreateDirAll(_, p) => SemanticEvent::CreateDirAll(p),
            Event::WriteNew(_, p) => SemanticEvent::WriteNew(p),
            Event::SyncFile(_, p) => SemanticEvent::SyncFile(p),
            Event::Rename(_, a, b) => SemanticEvent::Rename(a, b),
            Event::SyncDir(_, p) => SemanticEvent::SyncDir(p),
            Event::ListDir(_, p) => SemanticEvent::ListDir(p),
            Event::RemoveDirAll(_, p) => SemanticEvent::RemoveDirAll(p),
            Event::RemoveFile(_, p) => SemanticEvent::RemoveFile(p),
            Event::ReadEnv(_, k) => SemanticEvent::ReadEnv(k),
            Event::FetchText(_, u) => SemanticEvent::FetchText(u),
            Event::FetchBytes(_, u) => SemanticEvent::FetchBytes(u),
            Event::TempDir(_, p) => SemanticEvent::TempDir(p),
            Event::ExtractZip(_, p) => SemanticEvent::ExtractZip(p),
            Event::Glob(_, b, p) => SemanticEvent::Glob(b, p),
        }
    }
}

/// An [`Event`] after stripping the sync/async scheduling tag.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SemanticEvent {
    ReadText(PathBuf),
    PathExists(PathBuf),
    Canonicalize(PathBuf),
    ReadBytes(PathBuf),
    CreateDirAll(PathBuf),
    WriteNew(PathBuf),
    SyncFile(PathBuf),
    Rename(PathBuf, PathBuf),
    SyncDir(PathBuf),
    ListDir(PathBuf),
    RemoveDirAll(PathBuf),
    RemoveFile(PathBuf),
    ReadEnv(String),
    FetchText(String),
    FetchBytes(String),
    TempDir(String),
    ExtractZip(PathBuf),
    Glob(PathBuf, String),
}

pub fn semantic(events: &[Event]) -> Vec<SemanticEvent> {
    events.iter().cloned().map(Event::semantic).collect()
}
impl SemanticEvent {
    pub fn from_event(event: &Event) -> SemanticEvent {
        event.clone().semantic()
    }
}

#[derive(Debug, Clone)]
enum Node {
    File(Vec<u8>),
    Dir,
}

/// Which durable-write step a fault rule targets.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Scope {
    /// Both metadata and blob stages.
    Any,
    Metadata,
    Blob,
}

#[derive(Debug, Clone)]
struct FaultRule {
    boundary: pklr::package_cache::WriteBoundary,
    scope: Scope,
    remaining: u32,
}

/// Controlled filesystem + environment + network used by [`FakeCapabilities`].
#[derive(Clone)]
pub struct FakeWorld {
    inner: Arc<Mutex<WorldState>>,
}

struct WorldState {
    nodes: HashMap<PathBuf, Node>,
    env: BTreeMap<String, Option<String>>,
    http_text: HashMap<String, String>,
    http_bytes: HashMap<String, Vec<u8>>,
    events: Vec<Event>,
    faults: Vec<FaultRule>,
    temp_counter: u64,
}

impl Default for FakeWorld {
    fn default() -> Self {
        let world = Self {
            inner: Arc::new(Mutex::new(WorldState {
                nodes: HashMap::new(),
                env: BTreeMap::new(),
                http_text: HashMap::new(),
                http_bytes: HashMap::new(),
                events: Vec::new(),
                faults: Vec::new(),
                temp_counter: 0,
            })),
        };
        world.with(|state| {
            state.nodes.insert(PathBuf::from("/"), Node::Dir);
        });
        world
    }
}

impl FakeWorld {
    pub fn new() -> Self {
        Self::default()
    }

    fn with<R>(&self, f: impl FnOnce(&mut WorldState) -> R) -> R {
        f(&mut self.inner.lock().unwrap())
    }

    pub fn write_file(&self, path: impl AsRef<Path>, bytes: impl Into<Vec<u8>>) {
        let path = normalize(path.as_ref());
        self.with(|state| {
            if let Some(parent) = path.parent() {
                ensure_dir(state, parent);
            }
            state.nodes.insert(path, Node::File(bytes.into()));
        });
    }

    pub fn write_text(&self, path: impl AsRef<Path>, text: impl Into<String>) {
        self.write_file(path, text.into().into_bytes());
    }

    pub fn read_file(&self, path: impl AsRef<Path>) -> Option<Vec<u8>> {
        let path = normalize(path.as_ref());
        self.with(|state| match state.nodes.get(&path) {
            Some(Node::File(bytes)) => Some(bytes.clone()),
            _ => None,
        })
    }

    pub fn exists(&self, path: impl AsRef<Path>) -> bool {
        let path = normalize(path.as_ref());
        self.with(|state| state.nodes.contains_key(&path))
    }

    pub fn set_env(&self, name: impl Into<String>, value: Option<String>) {
        self.with(|state| state.env.insert(name.into(), value));
    }

    pub fn serve_text(&self, url: impl Into<String>, body: impl Into<String>) {
        self.with(|state| state.http_text.insert(url.into(), body.into()));
    }

    pub fn serve_bytes(&self, url: impl Into<String>, bytes: impl Into<Vec<u8>>) {
        self.with(|state| state.http_bytes.insert(url.into(), bytes.into()));
    }

    /// Fail the first `times` matching durability operations.
    pub fn inject_fault(
        &self,
        boundary: pklr::package_cache::WriteBoundary,
        scope: Scope,
        times: u32,
    ) {
        self.with(|state| {
            state.faults.push(FaultRule {
                boundary,
                scope,
                remaining: times,
            })
        });
    }

    pub fn events(&self) -> Vec<Event> {
        self.with(|state| state.events.clone())
    }

    /// All staged (dot-prefixed) files currently present in the filesystem.
    pub fn staged_paths(&self) -> Vec<PathBuf> {
        self.with(|state| {
            let mut out: Vec<PathBuf> = state
                .nodes
                .iter()
                .filter_map(|(path, node)| match node {
                    Node::File(..) if path
                        .file_name()
                        .and_then(|name| name.to_str())
                        .is_some_and(|name| name.starts_with(pklr::package_cache::STAGING_PREFIX)) =>
                    {
                        Some(path.clone())
                    }
                    _ => None,
                })
                .collect();
            out.sort();
            out
        })
    }

    pub fn event_count(&self, predicate: impl Fn(&SemanticEvent) -> bool) -> usize {
        self.with(|state| {
            state
                .events
                .iter()
                .filter(|event| predicate(&SemanticEvent::from_event(event)))
                .count()
        })
    }
}

fn ensure_dir(state: &mut WorldState, path: &Path) {
    let mut current = PathBuf::new();
    for component in path.components() {
        if matches!(component, Component::RootDir) {
            current.push("/");
            state.nodes.entry(current.clone()).or_insert(Node::Dir);
        } else if let Component::Normal(part) = component {
            current.push(part);
            state.nodes.entry(current.clone()).or_insert(Node::Dir);
        }
    }
}

fn normalize(path: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for component in path.components() {
        match component {
            Component::Prefix(prefix) => out.push(prefix.as_os_str()),
            Component::RootDir => out.push("/"),
            Component::CurDir => {}
            Component::ParentDir => {
                out.pop();
            }
            Component::Normal(part) => out.push(part),
        }
    }
    out
}

fn not_found(path: &Path) -> Error {
    Error::Io(
        path.to_path_buf(),
        std::io::Error::from(std::io::ErrorKind::NotFound),
    )
}

/// Whether a staged cache file name stages metadata or a blob.
fn staged_scope(path: &Path) -> Scope {
    let name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or_default();
    if let Some(rest) = name.strip_prefix(pklr::package_cache::STAGING_PREFIX) {
        // Layout: <staging><final-name>-<pid>-<serial>. Blob names are
        // 64-char hex digests, so strip the last two dash-separated segments.
        let without_serial = rest.rsplit_once('-').map(|(head, _)| head).unwrap_or(rest);
        let final_name = without_serial
            .rsplit_once('-')
            .map(|(head, _)| head)
            .unwrap_or(without_serial);
        if final_name == pklr::package_cache::META_FILE_NAME {
            return Scope::Metadata;
        }
    }
    Scope::Blob
}

/// Capabilities backed entirely by [`FakeWorld`].
#[derive(Clone)]
pub struct FakeCapabilities {
    world: FakeWorld,
    /// Cache root; faults inside this tree are scoped by file role.
    cache_root: PathBuf,
}

impl FakeCapabilities {
    pub fn new(world: FakeWorld) -> Self {
        Self {
            world,
            cache_root: PathBuf::from("/cache"),
        }
    }

    pub fn with_cache_root(mut self, root: impl Into<PathBuf>) -> Self {
        self.cache_root = normalize(&root.into());
        self
    }

    fn log(&self, event: Event) {
        self.world.with(|state| state.events.push(event));
    }

    fn maybe_fault(
        &self,
        boundary: pklr::package_cache::WriteBoundary,
        path: &Path,
        scope_override: Option<Scope>,
    ) -> pklr::Result<()> {
        let target_scope = scope_override.unwrap_or_else(|| staged_scope(path));
        self.world.with(|state| {
            for rule in &mut state.faults {
                if rule.boundary != boundary || rule.remaining == 0 {
                    continue;
                }
                let write_boundary = matches!(
                    boundary,
                    pklr::package_cache::WriteBoundary::BlobWrite
                        | pklr::package_cache::WriteBoundary::MetadataWrite
                );
                // Write boundary rules already carry their role; other
                // boundaries are shared by the blob and metadata phases.
                let scope_matches = match (write_boundary, rule.scope, target_scope) {
                    (true, Scope::Any, _) => true,
                    (true, scope, target) => scope == target,
                    (false, Scope::Any, _) => true,
                    (false, scope, target) => scope == target,
                };
                if scope_matches {
                    rule.remaining -= 1;
                    return Err(Error::Io(
                        path.to_path_buf(),
                        std::io::Error::new(
                            std::io::ErrorKind::Other,
                            format!("injected fault at {}", boundary.as_str()),
                        ),
                    ));
                }
            }
            Ok(())
        })
    }
}

impl FakeCapabilities {
    fn list_in(state: &WorldState, dir: &Path) -> std::io::Result<Vec<PathBuf>> {
        let dir = normalize(dir);
        if !matches!(state.nodes.get(&dir), Some(Node::Dir)) {
            return Err(std::io::Error::from(std::io::ErrorKind::NotFound));
        }
        let mut out = Vec::new();
        for candidate in state.nodes.keys() {
            if let Some(parent) = candidate.parent()
                && parent == dir
            {
                out.push(candidate.clone());
            }
        }
        out.sort();
        Ok(out)
    }
}

impl EvalCapabilities for FakeCapabilities {
    fn read_to_string<'a>(&'a mut self, path: &'a Path) -> BoxFuture<'a, pklr::Result<String>> {
        self.log(Event::ReadText(Dispatch::current(), path.to_path_buf()));
        Box::pin(async move {
            self.world.with(|state| match state.nodes.get(path) {
                Some(Node::File(bytes)) => String::from_utf8(bytes.clone())
                    .map_err(|error| Error::Eval(error.to_string())),
                Some(Node::Dir) => Err(Error::Io(
                    path.to_path_buf(),
                    std::io::Error::from(std::io::ErrorKind::IsADirectory),
                )),
                None => Err(not_found(path)),
            })
        })
    }

    fn path_exists<'a>(&'a mut self, path: &'a Path) -> BoxFuture<'a, pklr::Result<bool>> {
        self.log(Event::PathExists(Dispatch::current(), path.to_path_buf()));
        Box::pin(async move { Ok(self.world.with(|state| state.nodes.contains_key(path))) })
    }

    fn canonicalize<'a>(&'a mut self, path: &'a Path) -> BoxFuture<'a, pklr::Result<PathBuf>> {
        self.log(Event::Canonicalize(Dispatch::current(), path.to_path_buf()));
        let path = path.to_path_buf();
        Box::pin(async move { Ok(path) })
    }

    fn read_bytes<'a>(&'a mut self, path: &'a Path) -> BoxFuture<'a, pklr::Result<Vec<u8>>> {
        self.log(Event::ReadBytes(Dispatch::current(), path.to_path_buf()));
        Box::pin(async move {
            self.world.with(|state| match state.nodes.get(path) {
                Some(Node::File(bytes)) => Ok(bytes.clone()),
                _ => Err(not_found(path)),
            })
        })
    }

    fn create_dir_all<'a>(&'a mut self, path: &'a Path) -> BoxFuture<'a, pklr::Result<()>> {
        self.log(Event::CreateDirAll(Dispatch::current(), path.to_path_buf()));
        let path = normalize(path);
        Box::pin(async move {
            self.world.with(|state| ensure_dir(state, &path));
            Ok(())
        })
    }

    fn cache_write_new<'a>(
        &'a mut self,
        path: &'a Path,
        bytes: &'a [u8],
    ) -> BoxFuture<'a, pklr::Result<()>> {
        self.log(Event::WriteNew(Dispatch::current(), path.to_path_buf()));
        let boundary = match staged_scope(path) {
            Scope::Metadata => pklr::package_cache::WriteBoundary::MetadataWrite,
            Scope::Blob => pklr::package_cache::WriteBoundary::BlobWrite,
            Scope::Any => pklr::package_cache::WriteBoundary::BlobWrite,
        };
        if let Err(error) = self.maybe_fault(boundary, path, None) {
            return Box::pin(async move { Err(error) });
        }
        let path = normalize(path);
        let bytes = bytes.to_vec();
        Box::pin(async move {
            self.world.with(|state| {
                if state.nodes.contains_key(&path) {
                    return Err(Error::Io(
                        path,
                        std::io::Error::from(std::io::ErrorKind::AlreadyExists),
                    ));
                }
                ensure_dir(state, path.parent().unwrap_or(Path::new("/")));
                state.nodes.insert(path, Node::File(bytes));
                Ok(())
            })
        })
    }

    fn cache_sync_file<'a>(&'a mut self, path: &'a Path) -> BoxFuture<'a, pklr::Result<()>> {
        self.log(Event::SyncFile(Dispatch::current(), path.to_path_buf()));
        let result = self.maybe_fault(pklr::package_cache::WriteBoundary::Flush, path);
        if result.is_err() {
            return Box::pin(async move { result });
        }
        Box::pin(async { Ok(()) })
    }

    fn cache_rename<'a>(
        &'a mut self,
        from: &'a Path,
        to: &'a Path,
    ) -> BoxFuture<'a, pklr::Result<()>> {
        self.log(Event::Rename(
            Dispatch::current(),
            from.to_path_buf(),
            to.to_path_buf(),
        ));
        if let Err(error) = self.maybe_fault(pklr::package_cache::WriteBoundary::Rename, from, None) {
            return Box::pin(async move { Err(error) });
        }
        let from = normalize(from);
        let to = normalize(to);
        Box::pin(async move {
            self.world.with(|state| {
                match state.nodes.remove(&from) {
                    Some(node) => {
                        if let Some(parent) = to.parent() {
                            ensure_dir(state, parent);
                        }
                        // POSIX rename(2) atomically replaces an existing file.
                        state.nodes.insert(to, node);
                    }
                    None => return Err(not_found(&from)),
                }
                Ok(())
            })
        })
    }

    fn cache_sync_dir<'a>(&'a mut self, path: &'a Path) -> BoxFuture<'a, pklr::Result<()>> {
        self.log(Event::SyncDir(Dispatch::current(), path.to_path_buf()));
        // Directory syncs follow the staged file's role: its role cannot be
        // inferred from the directory name, so derive it from the staged
        // children present in that directory.
        let scope = {
            let directory = path.to_path_buf();
            self.world.with(|state| {
                let mut roles = Vec::new();
                if let Ok(entries) = FakeCapabilities::list_in(&state, &directory) {
                    for entry in entries {
                        if let Some(name) = entry.file_name().and_then(|name| name.to_str())
                            && name.starts_with(pklr::package_cache::STAGING_PREFIX)
                        {
                            roles.push(staged_scope(&entry));
                        }
                    }
                }
                if roles.contains(&Scope::Metadata) {
                    Scope::Metadata
                } else {
                    Scope::Blob
                }
            })
        };
        let result =
            self.maybe_fault(pklr::package_cache::WriteBoundary::DirSync, path, Some(scope));
        Box::pin(async move { result })
    }

    fn cache_list_dir<'a>(
        &'a mut self,
        path: &'a Path,
    ) -> BoxFuture<'a, pklr::Result<Vec<PathBuf>>> {
        self.log(Event::ListDir(Dispatch::current(), path.to_path_buf()));
        let dir = normalize(path);
        Box::pin(async move {
            self.world.with(|state| {
                if !matches!(state.nodes.get(&dir), Some(Node::Dir)) {
                    return Err(not_found(&dir));
                }
                let mut out = Vec::new();
                for candidate in state.nodes.keys() {
                    if let Some(parent) = candidate.parent()
                        && parent == dir
                    {
                        out.push(candidate.clone());
                    }
                }
                out.sort();
                Ok(out)
            })
        })
    }

    fn cache_remove_dir_all<'a>(
        &'a mut self,
        path: &'a Path,
    ) -> BoxFuture<'a, pklr::Result<()>> {
        self.log(Event::RemoveDirAll(Dispatch::current(), path.to_path_buf()));
        let root = normalize(path);
        Box::pin(async move {
            self.world.with(|state| {
                state.nodes.retain(|candidate, _| !candidate.starts_with(&root));
            });
            Ok(())
        })
    }

    fn remove_file<'a>(&'a mut self, path: &'a Path) -> BoxFuture<'a, pklr::Result<()>> {
        self.log(Event::RemoveFile(Dispatch::current(), path.to_path_buf()));
        let path = normalize(path);
        Box::pin(async move {
            self.world.with(|state| {
                state.nodes.remove(&path);
            });
            Ok(())
        })
    }

    fn read_env<'a>(&'a mut self, name: &'a str) -> BoxFuture<'a, pklr::Result<Option<String>>> {
        self.log(Event::ReadEnv(Dispatch::current(), name.to_string()));
        let name = name.to_string();
        Box::pin(async move {
            Ok(self
                .world
                .with(|state| state.env.get(&name).cloned())
                .unwrap_or(None))
        })
    }

    fn fetch_text<'a>(&'a mut self, url: &'a str) -> BoxFuture<'a, pklr::Result<String>> {
        self.log(Event::FetchText(Dispatch::current(), url.to_string()));
        let url = url.to_string();
        Box::pin(async move {
            self.world.with(|state| {
                state
                    .http_text
                    .get(&url)
                    .cloned()
                    .ok_or_else(|| Error::Eval(format!("scripted network is offline: {url}")))
            })
        })
    }

    fn fetch_bytes<'a>(&'a mut self, url: &'a str) -> BoxFuture<'a, pklr::Result<Vec<u8>>> {
        self.log(Event::FetchBytes(Dispatch::current(), url.to_string()));
        let url = url.to_string();
        Box::pin(async move {
            self.world.with(|state| {
                state
                    .http_bytes
                    .get(&url)
                    .cloned()
                    .ok_or_else(|| Error::Eval(format!("scripted network is offline: {url}")))
            })
        })
    }

    fn temp_dir<'a>(&'a mut self, prefix: &'a str) -> BoxFuture<'a, pklr::Result<PathBuf>> {
        self.log(Event::TempDir(Dispatch::current(), prefix.to_string()));
        let path = self.world.with(|state| {
            state.temp_counter += 1;
            PathBuf::from(format!("/tmp/{prefix}-{}", state.temp_counter))
        });
        let path_clone = path.clone();
        Box::pin(async move {
            self.world
                .with(|state| ensure_dir(state, &path_clone));
            Ok(path)
        })
    }

    fn extract_zip<'a>(
        &'a mut self,
        bytes: Vec<u8>,
        destination: &'a Path,
    ) -> BoxFuture<'a, pklr::Result<()>> {
        self.log(Event::ExtractZip(Dispatch::current(), destination.to_path_buf()));
        let destination = normalize(destination);
        Box::pin(async move {
            let mut archive = zip::ZipArchive::new(std::io::Cursor::new(bytes))
                .map_err(|error| Error::Eval(format!("zip error: {error}")))?;
            for index in 0..archive.len() {
                use std::io::Read;
                let mut entry = archive
                    .by_index(index)
                    .map_err(|error| Error::Eval(format!("zip entry error: {error}")))?;
                if entry.is_dir() {
                    continue;
                }
                let name = entry.name().to_string();
                let mut bytes = Vec::new();
                entry
                    .read_to_end(&mut bytes)
                    .map_err(|error| Error::Eval(format!("zip read error: {error}")))?;
                let target = destination.join(&name);
                self.world.write_file(&target, bytes);
            }
            Ok(())
        })
    }

    fn glob<'a>(
        &'a mut self,
        base: &'a Path,
        pattern: &'a str,
    ) -> BoxFuture<'a, pklr::Result<Vec<PathBuf>>> {
        self.log(Event::Glob(
            Dispatch::current(),
            base.to_path_buf(),
            pattern.to_string(),
        ));
        Box::pin(async { Ok(Vec::new()) })
    }
}
