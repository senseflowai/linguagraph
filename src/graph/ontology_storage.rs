//! Pluggable storage for [`OntologyCatalog`].
//!
//! The default backend, [`JsonFileOntologyCatalogStorage`], reads and
//! writes a single JSON file. Custom backends (Postgres, S3, an HTTP
//! service) implement the [`OntologyCatalogStorage`] trait and can be
//! plugged into [`crate::prompt::PromptGenerator`] via
//! [`crate::prompt::PromptGenerator::from_storage`].

use std::path::{Path, PathBuf};
use std::sync::Arc;

use async_trait::async_trait;
use tokio::fs;

use super::ontology::{OntologyCatalog, OntologyError};

/// Async storage backend for ontology catalogs.
///
/// Implementors are expected to be cheap to clone (typically wrapped in
/// `Arc`) and safe to share across tasks.
#[async_trait]
pub trait OntologyCatalogStorage: Send + Sync + std::fmt::Debug {
    /// Load the full catalog.
    async fn load(&self) -> Result<OntologyCatalog, OntologyError>;

    /// Replace the stored catalog with `catalog`.
    ///
    /// The default implementation returns
    /// [`OntologyError::Unsupported`] for read-only backends.
    async fn save(&self, catalog: &OntologyCatalog) -> Result<(), OntologyError> {
        let _ = catalog;
        Err(OntologyError::Unsupported("save".into()))
    }
}

/// A shared, dynamically-typed storage handle. Convenient for wiring a
/// backend into [`crate::prompt::PromptGenerator`] without leaking the concrete
/// type through call sites.
pub type SharedOntologyCatalogStorage = Arc<dyn OntologyCatalogStorage>;

/// Default on-disk location for the JSON catalog cache.
pub const DEFAULT_ONTOLOGY_CATALOG_CACHE_PATH: &str = ".linguagraph/ontology_catalog.json";

/// Filesystem-backed storage: reads and atomically rewrites a single
/// JSON file holding the full catalog.
#[derive(Debug, Clone)]
pub struct JsonFileOntologyCatalogStorage {
    path: PathBuf,
}

impl JsonFileOntologyCatalogStorage {
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self { path: path.into() }
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn default_path() -> PathBuf {
        PathBuf::from(DEFAULT_ONTOLOGY_CATALOG_CACHE_PATH)
    }
}

impl Default for JsonFileOntologyCatalogStorage {
    fn default() -> Self {
        Self::new(Self::default_path())
    }
}

#[async_trait]
impl OntologyCatalogStorage for JsonFileOntologyCatalogStorage {
    async fn load(&self) -> Result<OntologyCatalog, OntologyError> {
        match fs::read(&self.path).await {
            Ok(bytes) => {
                if bytes.is_empty() {
                    return Ok(OntologyCatalog::default());
                }
                Ok(serde_json::from_slice(&bytes)?)
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                Err(OntologyError::NotFound(self.path.clone()))
            }
            Err(e) => Err(OntologyError::Io(e)),
        }
    }

    async fn save(&self, catalog: &OntologyCatalog) -> Result<(), OntologyError> {
        if let Some(parent) = self.path.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent).await?;
            }
        }
        let body = serde_json::to_vec_pretty(catalog)?;
        let tmp = self.path.with_extension("json.tmp");
        fs::write(&tmp, &body).await?;
        fs::rename(&tmp, &self.path).await?;
        Ok(())
    }
}

/// In-memory storage. Useful for tests and for callers that build the
/// catalog programmatically rather than from durable storage.
#[derive(Debug, Clone, Default)]
pub struct InMemoryOntologyCatalogStorage {
    catalog: OntologyCatalog,
}

impl InMemoryOntologyCatalogStorage {
    pub fn new(catalog: OntologyCatalog) -> Self {
        Self { catalog }
    }
}

#[async_trait]
impl OntologyCatalogStorage for InMemoryOntologyCatalogStorage {
    async fn load(&self) -> Result<OntologyCatalog, OntologyError> {
        Ok(self.catalog.clone())
    }
    // `save` falls back to the trait default (Unsupported); the in-memory
    // backend is read-only by design — callers that need mutation should
    // build a fresh storage from the updated catalog.
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::super::{DomainOntology, EntityTypeSpec, RelationTypeSpec};
    use super::*;

    static COUNTER: AtomicU64 = AtomicU64::new(0);

    fn tmp_path(name: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!("linguagraph-onto-{nanos}-{n}-{name}.json"))
    }

    fn sample_catalog() -> OntologyCatalog {
        let mut cat = OntologyCatalog::default();
        cat.insert(
            "demo",
            DomainOntology {
                entity_types: vec![EntityTypeSpec::with_description("Foo", "A foo.")],
                relation_types: vec![RelationTypeSpec::new("KNOWS")],
            },
        );
        cat
    }

    #[tokio::test]
    async fn missing_file_errors_with_not_found() {
        let storage = JsonFileOntologyCatalogStorage::new(tmp_path("missing"));
        let err = storage.load().await.unwrap_err();
        assert!(matches!(err, OntologyError::NotFound(_)));
    }

    #[tokio::test]
    async fn save_then_load_round_trips() {
        let path = tmp_path("round-trip");
        let storage = JsonFileOntologyCatalogStorage::new(&path);
        let cat = sample_catalog();
        storage.save(&cat).await.unwrap();
        let back = storage.load().await.unwrap();
        assert_eq!(back.get("demo").unwrap().entity_types[0].name, "Foo");
        let _ = std::fs::remove_file(&path);
    }

    #[tokio::test]
    async fn in_memory_storage_returns_a_clone() {
        let storage = InMemoryOntologyCatalogStorage::new(sample_catalog());
        let back = storage.load().await.unwrap();
        assert!(back.get("demo").is_some());
    }

    #[tokio::test]
    async fn in_memory_save_is_unsupported() {
        let storage = InMemoryOntologyCatalogStorage::new(sample_catalog());
        let err = storage.save(&OntologyCatalog::default()).await.unwrap_err();
        assert!(matches!(err, OntologyError::Unsupported(_)));
    }
}
