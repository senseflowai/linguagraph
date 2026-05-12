//! JSON-cache-file [`MetadataStore`] implementation.
//!
//! The file is rewritten atomically (write to a sibling tempfile, then
//! `rename`) so concurrent ingests can't observe a half-written cache.

use std::path::{Path, PathBuf};

use async_trait::async_trait;
use tokio::fs;

use super::{MetadataError, MetadataStore, PropertyMetadata};

/// On-disk metadata cache. The path is created on first save; the parent
/// directory is created if needed.
#[derive(Debug, Clone)]
pub struct FileMetadataStore {
    path: PathBuf,
}

impl FileMetadataStore {
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self { path: path.into() }
    }

    pub fn path(&self) -> &Path {
        &self.path
    }
}

#[async_trait]
impl MetadataStore for FileMetadataStore {
    async fn load(&self) -> Result<PropertyMetadata, MetadataError> {
        match fs::read(&self.path).await {
            Ok(bytes) => {
                if bytes.is_empty() {
                    return Ok(PropertyMetadata::new());
                }
                Ok(serde_json::from_slice(&bytes)?)
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(PropertyMetadata::new()),
            Err(e) => Err(MetadataError::Io(e)),
        }
    }

    async fn save(&self, meta: &PropertyMetadata) -> Result<(), MetadataError> {
        if let Some(parent) = self.path.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent).await?;
            }
        }
        let body = serde_json::to_vec_pretty(meta)?;
        let tmp = self.path.with_extension("json.tmp");
        fs::write(&tmp, &body).await?;
        fs::rename(&tmp, &self.path).await?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};

    static COUNTER: AtomicU64 = AtomicU64::new(0);

    fn tmp_path(name: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!("linguagraph-meta-{nanos}-{n}-{name}.json"))
    }

    #[tokio::test]
    async fn missing_file_loads_as_empty() {
        let store = FileMetadataStore::new(tmp_path("missing"));
        let meta = store.load().await.unwrap();
        assert!(meta.is_empty());
    }

    #[tokio::test]
    async fn save_then_load_round_trips() {
        let path = tmp_path("rt");
        let store = FileMetadataStore::new(&path);
        let mut meta = PropertyMetadata::new();
        meta.insert("Camera.state", "active or inactive");
        store.save(&meta).await.unwrap();

        let again = store.load().await.unwrap();
        assert_eq!(again, meta);
        let _ = std::fs::remove_file(&path);
    }

    #[tokio::test]
    async fn update_merges_with_existing() {
        let path = tmp_path("merge");
        let store = FileMetadataStore::new(&path);

        let mut first = PropertyMetadata::new();
        first.insert("Camera.state", "old");
        first.insert("Camera.id", "primary key");
        store.save(&first).await.unwrap();

        let mut incoming = PropertyMetadata::new();
        incoming.insert("Camera.state", "new");
        incoming.insert("Place.address", "installation address");
        let merged = store.update(&incoming).await.unwrap();

        assert_eq!(merged.get("Camera.state"), Some("new"));
        assert_eq!(merged.get("Camera.id"), Some("primary key"));
        assert_eq!(merged.get("Place.address"), Some("installation address"));

        // Persisted result matches the returned snapshot.
        let reloaded = store.load().await.unwrap();
        assert_eq!(reloaded, merged);
        let _ = std::fs::remove_file(&path);
    }
}
