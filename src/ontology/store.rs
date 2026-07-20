//! Per-domain ontology persistence.
//!
//! [`OntologyStore`] is the schema-persistence contract the ontology
//! subsystem is built on. Unlike the older whole-catalog
//! [`crate::graph::OntologyCatalogStorage`] (load/save the entire map), it
//! is **per-domain** — the granularity a management API actually needs —
//! and namespace-scoped, so one backend serves every tenant.
//!
//! It is deliberately **schema-only**: it never touches embeddings. The
//! embedding refresh + garbage-collection that must accompany a schema
//! write lives one layer up, in the ontology service, which composes a
//! store with an embedder (see the module docs). Keeping the store pure
//! is what lets the file / in-memory backends exist at all — they have no
//! embedder — and keeps the Qdrant backend's schema path independent of
//! the embedding path even though both address the same collection.
//!
//! Backends in this module:
//! * [`InMemoryOntologyStore`] — a process map; tests and programmatic use.
//! * [`JsonFileOntologyStore`] — one flat-catalog JSON file per namespace;
//!   the on-disk shape is identical to [`crate::graph::OntologyCatalog`]'s,
//!   so an existing catalog file is readable as-is.
//!
//! The Qdrant single-store backend lives in [`super::qdrant`] behind the
//! `qdrant` feature.

use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use tokio::fs;

use super::namespace::Namespace;
use crate::graph::{DomainOntology, OntologyCatalog, OntologyError};

/// Write token / version for a domain: epoch-milliseconds of the last
/// write. Doubles as the value the UI shows as "updated at" and as the
/// optimistic-concurrency guard a `patch` compares against (see the
/// ontology service). Monotonic in practice at human edit cadence.
pub type Version = u64;

/// Lightweight per-domain projection for listings: enough for a UI table
/// (name, description, counts, freshness) without shipping every spec.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
pub struct DomainSummary {
    pub domain: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub entity_count: usize,
    pub relation_count: usize,
    /// Epoch-ms of the last write; see [`Version`].
    pub version: Version,
}

impl DomainSummary {
    pub(crate) fn of(domain: &str, ontology: &DomainOntology, version: Version) -> Self {
        Self {
            domain: domain.to_string(),
            description: ontology.description.clone(),
            entity_count: ontology.entity_types.len(),
            relation_count: ontology.relation_types.len(),
            version,
        }
    }
}

/// Outcome of [`OntologyStore::rename`]. A bare `Option` is not enough:
/// the two failure modes map to different HTTP statuses upstream (404 vs
/// 409).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RenameOutcome {
    Renamed(DomainSummary),
    /// No domain named `from` in this namespace.
    NotFound,
    /// A domain named `to` already exists in this namespace.
    Conflict,
}

/// Namespace-scoped, per-domain ontology persistence. Cheap to clone
/// (typically an `Arc` wrapper) and safe to share across tasks.
///
/// `put` is an idempotent create-or-replace. The create-vs-replace
/// distinction (409 when creating over an existing domain) and PATCH
/// semantics are policy the service layer applies on top of `get` + `put`
/// — the store stays a small, mechanical CRUD surface.
#[async_trait]
pub trait OntologyStore: Send + Sync + std::fmt::Debug {
    /// Summaries of every domain in `ns`, ordered by domain name.
    async fn list(&self, ns: &Namespace) -> Result<Vec<DomainSummary>, OntologyError>;

    /// Full ontology for one domain, or `None` when absent.
    async fn get(
        &self,
        ns: &Namespace,
        domain: &str,
    ) -> Result<Option<DomainOntology>, OntologyError>;

    /// Create-or-replace one domain. Returns the fresh summary (with the
    /// new [`Version`]).
    async fn put(
        &self,
        ns: &Namespace,
        domain: &str,
        ontology: DomainOntology,
    ) -> Result<DomainSummary, OntologyError>;

    /// Remove one domain. `false` when it did not exist.
    async fn delete(&self, ns: &Namespace, domain: &str) -> Result<bool, OntologyError>;

    /// Rename a domain's key. Content is carried over unchanged.
    async fn rename(
        &self,
        ns: &Namespace,
        from: &str,
        to: &str,
    ) -> Result<RenameOutcome, OntologyError>;

    /// Fetch a domain's ontology together with its current [`Version`] —
    /// the read a PATCH needs for its optimistic-concurrency check. The
    /// default composes `get` + `list`; a backend that carries the version
    /// alongside the ontology can override to do it in one round trip.
    async fn get_with_version(
        &self,
        ns: &Namespace,
        domain: &str,
    ) -> Result<Option<(DomainOntology, Version)>, OntologyError> {
        let Some(ontology) = self.get(ns, domain).await? else {
            return Ok(None);
        };
        let version = self
            .list(ns)
            .await?
            .into_iter()
            .find(|s| s.domain == domain)
            .map(|s| s.version)
            .unwrap_or(0);
        Ok(Some((ontology, version)))
    }

    /// Load every domain in `ns` as an [`OntologyCatalog`] — the shape the
    /// [`crate::core::Pipeline`] consumes for query lowering and schema
    /// enrichment.
    async fn load_catalog(&self, ns: &Namespace) -> Result<OntologyCatalog, OntologyError>;
}

pub(crate) fn now_ms() -> Version {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

// ---------------------------------------------------------------------
// In-memory backend
// ---------------------------------------------------------------------

/// Process-local, namespace-scoped store. Read-write, thread-safe, non
/// durable. For tests and callers that build ontologies programmatically.
#[derive(Debug, Default)]
pub struct InMemoryOntologyStore {
    // ns token → (domain → (ontology, version))
    inner: Mutex<HashMap<String, BTreeMap<String, (DomainOntology, Version)>>>,
}

impl InMemoryOntologyStore {
    pub fn new() -> Self {
        Self::default()
    }
}

#[async_trait]
impl OntologyStore for InMemoryOntologyStore {
    async fn list(&self, ns: &Namespace) -> Result<Vec<DomainSummary>, OntologyError> {
        let guard = self.inner.lock().unwrap();
        Ok(guard
            .get(ns.token())
            .map(|domains| {
                domains
                    .iter()
                    .map(|(name, (onto, v))| DomainSummary::of(name, onto, *v))
                    .collect()
            })
            .unwrap_or_default())
    }

    async fn get(
        &self,
        ns: &Namespace,
        domain: &str,
    ) -> Result<Option<DomainOntology>, OntologyError> {
        let guard = self.inner.lock().unwrap();
        Ok(guard
            .get(ns.token())
            .and_then(|d| d.get(domain))
            .map(|(onto, _)| {
                let mut onto = onto.clone();
                onto.name = Some(domain.to_string());
                onto
            }))
    }

    async fn put(
        &self,
        ns: &Namespace,
        domain: &str,
        mut ontology: DomainOntology,
    ) -> Result<DomainSummary, OntologyError> {
        ontology.name = Some(domain.to_string());
        let version = now_ms();
        let summary = DomainSummary::of(domain, &ontology, version);
        let mut guard = self.inner.lock().unwrap();
        guard
            .entry(ns.token().to_string())
            .or_default()
            .insert(domain.to_string(), (ontology, version));
        Ok(summary)
    }

    async fn delete(&self, ns: &Namespace, domain: &str) -> Result<bool, OntologyError> {
        let mut guard = self.inner.lock().unwrap();
        Ok(guard
            .get_mut(ns.token())
            .map(|d| d.remove(domain).is_some())
            .unwrap_or(false))
    }

    async fn rename(
        &self,
        ns: &Namespace,
        from: &str,
        to: &str,
    ) -> Result<RenameOutcome, OntologyError> {
        let mut guard = self.inner.lock().unwrap();
        let domains = match guard.get_mut(ns.token()) {
            Some(d) => d,
            None => return Ok(RenameOutcome::NotFound),
        };
        if !domains.contains_key(from) {
            return Ok(RenameOutcome::NotFound);
        }
        if domains.contains_key(to) {
            return Ok(RenameOutcome::Conflict);
        }
        let (mut onto, _) = domains.remove(from).expect("checked above");
        onto.name = Some(to.to_string());
        let version = now_ms();
        let summary = DomainSummary::of(to, &onto, version);
        domains.insert(to.to_string(), (onto, version));
        Ok(RenameOutcome::Renamed(summary))
    }

    async fn load_catalog(&self, ns: &Namespace) -> Result<OntologyCatalog, OntologyError> {
        let guard = self.inner.lock().unwrap();
        let mut catalog = OntologyCatalog::default();
        if let Some(domains) = guard.get(ns.token()) {
            for (name, (onto, _)) in domains {
                catalog.insert(name.clone(), onto.clone());
            }
        }
        Ok(catalog)
    }
}

// ---------------------------------------------------------------------
// JSON-file backend
// ---------------------------------------------------------------------

/// One flat-catalog JSON file per namespace under `base_dir`
/// (`{base_dir}/{token}.json`). The file content is exactly an
/// [`OntologyCatalog`] (`{ "<domain>": { … } }`), so a catalog written by
/// any other tool — or the legacy on-disk cache — round-trips unchanged.
///
/// [`Version`] is derived from the file's mtime, which is coarse; this
/// backend is meant for single-tenant CLI / test use, not the concurrent
/// multi-tenant server (that is the Qdrant backend).
#[derive(Debug, Clone)]
pub struct JsonFileOntologyStore {
    base_dir: PathBuf,
}

impl JsonFileOntologyStore {
    pub fn new(base_dir: impl Into<PathBuf>) -> Self {
        Self {
            base_dir: base_dir.into(),
        }
    }

    fn path(&self, ns: &Namespace) -> PathBuf {
        self.base_dir.join(format!("{}.json", ns.token()))
    }

    async fn read_catalog(&self, ns: &Namespace) -> Result<OntologyCatalog, OntologyError> {
        let path = self.path(ns);
        match fs::read(&path).await {
            Ok(bytes) if bytes.is_empty() => Ok(OntologyCatalog::default()),
            Ok(bytes) => {
                let mut catalog: OntologyCatalog = serde_json::from_slice(&bytes)?;
                catalog.hydrate_domain_names();
                Ok(catalog)
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(OntologyCatalog::default()),
            Err(e) => Err(OntologyError::Io(e)),
        }
    }

    async fn write_catalog(
        &self,
        ns: &Namespace,
        catalog: &OntologyCatalog,
    ) -> Result<(), OntologyError> {
        if !self.base_dir.as_os_str().is_empty() {
            fs::create_dir_all(&self.base_dir).await?;
        }
        let path = self.path(ns);
        let body = serde_json::to_vec_pretty(catalog)?;
        let tmp = path.with_extension("json.tmp");
        fs::write(&tmp, &body).await?;
        fs::rename(&tmp, &path).await?;
        Ok(())
    }

    async fn version(&self, ns: &Namespace) -> Version {
        file_version(&self.path(ns)).await
    }
}

async fn file_version(path: &Path) -> Version {
    match fs::metadata(path).await {
        Ok(m) => m
            .modified()
            .ok()
            .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
            .map(|d| d.as_millis() as u64)
            .unwrap_or_else(now_ms),
        Err(_) => now_ms(),
    }
}

#[async_trait]
impl OntologyStore for JsonFileOntologyStore {
    async fn list(&self, ns: &Namespace) -> Result<Vec<DomainSummary>, OntologyError> {
        let catalog = self.read_catalog(ns).await?;
        let version = self.version(ns).await;
        Ok(catalog
            .domains_view()
            .iter()
            .map(|(name, onto)| DomainSummary::of(name, onto, version))
            .collect())
    }

    async fn get(
        &self,
        ns: &Namespace,
        domain: &str,
    ) -> Result<Option<DomainOntology>, OntologyError> {
        Ok(self.read_catalog(ns).await?.get(domain).cloned())
    }

    async fn put(
        &self,
        ns: &Namespace,
        domain: &str,
        ontology: DomainOntology,
    ) -> Result<DomainSummary, OntologyError> {
        let mut catalog = self.read_catalog(ns).await?;
        catalog.insert(domain.to_string(), ontology);
        self.write_catalog(ns, &catalog).await?;
        let onto = catalog.get(domain).expect("just inserted");
        Ok(DomainSummary::of(domain, onto, self.version(ns).await))
    }

    async fn delete(&self, ns: &Namespace, domain: &str) -> Result<bool, OntologyError> {
        let mut catalog = self.read_catalog(ns).await?;
        if catalog.domains.remove(domain).is_none() {
            return Ok(false);
        }
        self.write_catalog(ns, &catalog).await?;
        Ok(true)
    }

    async fn rename(
        &self,
        ns: &Namespace,
        from: &str,
        to: &str,
    ) -> Result<RenameOutcome, OntologyError> {
        let mut catalog = self.read_catalog(ns).await?;
        if catalog.get(from).is_none() {
            return Ok(RenameOutcome::NotFound);
        }
        if catalog.get(to).is_some() {
            return Ok(RenameOutcome::Conflict);
        }
        let onto = catalog.domains.remove(from).expect("checked above");
        catalog.insert(to.to_string(), onto);
        self.write_catalog(ns, &catalog).await?;
        let onto = catalog.get(to).expect("just inserted");
        Ok(RenameOutcome::Renamed(DomainSummary::of(
            to,
            onto,
            self.version(ns).await,
        )))
    }

    async fn load_catalog(&self, ns: &Namespace) -> Result<OntologyCatalog, OntologyError> {
        self.read_catalog(ns).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::EntityTypeSpec;

    fn sample() -> DomainOntology {
        DomainOntology {
            name: None,
            description: Some("demo domain".into()),
            entity_types: vec![
                EntityTypeSpec::with_description("Client", "a customer"),
                EntityTypeSpec::new("Order"),
            ],
            relation_types: vec![],
            example: Some("few-shot".into()),
        }
    }

    async fn exercise(store: &dyn OntologyStore) {
        let ns = Namespace::new("ws_test");

        // empty namespace
        assert!(store.list(&ns).await.unwrap().is_empty());
        assert!(store.get(&ns, "demo").await.unwrap().is_none());

        // put + get
        let s = store.put(&ns, "demo", sample()).await.unwrap();
        assert_eq!(s.domain, "demo");
        assert_eq!(s.entity_count, 2);
        assert_eq!(s.description.as_deref(), Some("demo domain"));

        let got = store.get(&ns, "demo").await.unwrap().unwrap();
        assert_eq!(got.name.as_deref(), Some("demo"));
        assert_eq!(got.example.as_deref(), Some("few-shot"));
        assert_eq!(got.entity_types.len(), 2);

        // list
        let list = store.list(&ns).await.unwrap();
        assert_eq!(list.len(), 1);
        assert_eq!(list[0].domain, "demo");

        // load_catalog hydrates names
        let cat = store.load_catalog(&ns).await.unwrap();
        assert_eq!(cat.get("demo").unwrap().name.as_deref(), Some("demo"));

        // rename: not found / conflict / ok
        assert_eq!(
            store.rename(&ns, "missing", "x").await.unwrap(),
            RenameOutcome::NotFound
        );
        store.put(&ns, "other", sample()).await.unwrap();
        assert_eq!(
            store.rename(&ns, "demo", "other").await.unwrap(),
            RenameOutcome::Conflict
        );
        match store.rename(&ns, "demo", "renamed").await.unwrap() {
            RenameOutcome::Renamed(s) => assert_eq!(s.domain, "renamed"),
            other => panic!("expected Renamed, got {other:?}"),
        }
        assert!(store.get(&ns, "demo").await.unwrap().is_none());
        assert!(store.get(&ns, "renamed").await.unwrap().is_some());

        // delete
        assert!(store.delete(&ns, "renamed").await.unwrap());
        assert!(!store.delete(&ns, "renamed").await.unwrap());
    }

    #[tokio::test]
    async fn in_memory_roundtrips() {
        exercise(&InMemoryOntologyStore::new()).await;
    }

    #[tokio::test]
    async fn json_file_roundtrips() {
        let dir = std::env::temp_dir().join(format!(
            "lg-onto-store-{}",
            uuid::Uuid::new_v4().simple()
        ));
        exercise(&JsonFileOntologyStore::new(&dir)).await;
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn namespaces_are_isolated() {
        let store = InMemoryOntologyStore::new();
        let a = Namespace::new("ws_1");
        let b = Namespace::new("ws_2");
        store.put(&a, "demo", sample()).await.unwrap();
        assert_eq!(store.list(&a).await.unwrap().len(), 1);
        assert!(store.list(&b).await.unwrap().is_empty());
    }
}
