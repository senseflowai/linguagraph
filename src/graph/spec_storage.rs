//! JSON-cache-file storage for [`GraphSpecification`].
//!
//! The file backend treats a missing cache as an empty specification and
//! rewrites saves atomically through a sibling temporary file.

use std::path::{Path, PathBuf};

use async_trait::async_trait;
use thiserror::Error;
use tokio::fs;

use super::GraphSpecification;

pub const DEFAULT_GRAPH_SPECIFICATION_CACHE_PATH: &str = ".linguagraph/graph_specification.json";

#[derive(Debug, Error)]
pub enum GraphSpecificationStorageError {
    #[error("I/O error accessing graph specification cache: {0}")]
    Io(#[from] std::io::Error),

    #[error("invalid graph specification cache contents: {0}")]
    Decode(#[from] serde_json::Error),
}

#[async_trait]
pub trait GraphSpecificationStorage: Send + Sync + std::fmt::Debug {
    async fn load(&self) -> Result<GraphSpecification, GraphSpecificationStorageError>;

    async fn save(
        &self,
        specification: &GraphSpecification,
    ) -> Result<(), GraphSpecificationStorageError>;

    async fn update(
        &self,
        incoming: &GraphSpecification,
    ) -> Result<GraphSpecification, GraphSpecificationStorageError> {
        let mut current = self.load().await?;
        current.merge(incoming);
        self.save(&current).await?;
        Ok(current)
    }
}

#[derive(Debug, Clone)]
pub struct FileGraphSpecificationStorage {
    path: PathBuf,
}

impl FileGraphSpecificationStorage {
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self { path: path.into() }
    }

    pub fn default_path() -> PathBuf {
        PathBuf::from(DEFAULT_GRAPH_SPECIFICATION_CACHE_PATH)
    }

    pub fn path(&self) -> &Path {
        &self.path
    }
}

impl Default for FileGraphSpecificationStorage {
    fn default() -> Self {
        Self::new(Self::default_path())
    }
}

#[async_trait]
impl GraphSpecificationStorage for FileGraphSpecificationStorage {
    async fn load(&self) -> Result<GraphSpecification, GraphSpecificationStorageError> {
        match fs::read(&self.path).await {
            Ok(bytes) => {
                if bytes.is_empty() {
                    return Ok(GraphSpecification::new());
                }
                Ok(serde_json::from_slice(&bytes)?)
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(GraphSpecification::new()),
            Err(e) => Err(GraphSpecificationStorageError::Io(e)),
        }
    }

    async fn save(
        &self,
        specification: &GraphSpecification,
    ) -> Result<(), GraphSpecificationStorageError> {
        if let Some(parent) = self.path.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent).await?;
            }
        }

        let body = serde_json::to_vec_pretty(specification)?;
        let tmp = self.path.with_extension("json.tmp");
        fs::write(&tmp, &body).await?;
        fs::rename(&tmp, &self.path).await?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};

    use crate::graph::PropertyType;

    use super::*;

    static COUNTER: AtomicU64 = AtomicU64::new(0);

    fn tmp_path(name: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!("linguagraph-spec-{nanos}-{n}-{name}.json"))
    }

    #[tokio::test]
    async fn missing_file_loads_as_empty_specification() {
        let storage = FileGraphSpecificationStorage::new(tmp_path("missing"));
        let spec = storage.load().await.unwrap();
        assert!(spec.is_empty());
    }

    #[test]
    fn default_uses_linguagraph_cache_dir() {
        let storage = FileGraphSpecificationStorage::default();
        assert_eq!(
            storage.path(),
            Path::new(DEFAULT_GRAPH_SPECIFICATION_CACHE_PATH)
        );
        assert_eq!(
            FileGraphSpecificationStorage::default_path(),
            PathBuf::from(DEFAULT_GRAPH_SPECIFICATION_CACHE_PATH)
        );
    }

    #[tokio::test]
    async fn empty_file_loads_as_empty_specification() {
        let path = tmp_path("empty");
        fs::write(&path, "").await.unwrap();

        let storage = FileGraphSpecificationStorage::new(&path);
        let spec = storage.load().await.unwrap();

        assert!(spec.is_empty());
        let _ = std::fs::remove_file(&path);
    }

    #[tokio::test]
    async fn save_then_load_round_trips() {
        let path = tmp_path("round-trip");
        let storage = FileGraphSpecificationStorage::new(&path);
        let spec = GraphSpecification::new()
            .with_entity("Company", "A legal organization.")
            .with_property(
                "Company",
                "name",
                PropertyType::Text,
                "Human-readable company name.",
            );

        storage.save(&spec).await.unwrap();

        let reloaded = storage.load().await.unwrap();
        assert_eq!(reloaded, spec);
        let _ = std::fs::remove_file(&path);
    }
}
