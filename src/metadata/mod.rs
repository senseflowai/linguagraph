//! Property metadata: human-readable descriptions for graph node properties.
//!
//! Mapping authors annotate node properties with `description` strings. The
//! ingester collects these into a flat key-value map keyed by the **property
//! path in the graph node** (e.g. `Camera.state`) and persists them through a
//! pluggable [`MetadataStore`].
//!
//! The default backend is a JSON cache file ([`FileMetadataStore`]). The
//! trait abstraction is the seam for a future SQL-backed store: callers
//! depend on `Arc<dyn MetadataStore>`, never on the concrete file impl.

mod extract;
mod file;

use std::collections::BTreeMap;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use thiserror::Error;

pub use extract::collect_from_mapping;
pub use file::FileMetadataStore;

/// Flat map from `<NodeLabel>.<property>` (or `<NodeLabel>` for entity-level
/// descriptions) to the human-readable description.
///
/// `BTreeMap` keeps iteration order stable, which matters for snapshot tests
/// and for keeping prompt output deterministic.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct PropertyMetadata {
    pub entries: BTreeMap<String, String>,
}

impl PropertyMetadata {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn get(&self, key: &str) -> Option<&str> {
        self.entries.get(key).map(String::as_str)
    }

    pub fn insert(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.entries.insert(key.into(), value.into());
    }

    /// Merge `other` into `self`. Entries in `other` win on conflict — the
    /// freshest mapping is the source of truth.
    pub fn merge(&mut self, other: &PropertyMetadata) {
        for (k, v) in &other.entries {
            self.entries.insert(k.clone(), v.clone());
        }
    }
}

#[derive(Debug, Error)]
pub enum MetadataError {
    #[error("I/O error accessing metadata cache: {0}")]
    Io(#[from] std::io::Error),

    #[error("invalid metadata cache contents: {0}")]
    Decode(#[from] serde_json::Error),
}

/// Abstract metadata persistence. Implementations may store on disk, in a
/// SQL database, or in memory; the pipeline doesn't care.
#[async_trait]
pub trait MetadataStore: Send + Sync + std::fmt::Debug {
    /// Read the full metadata snapshot. Missing storage returns an empty
    /// snapshot rather than erroring — first-run ingests must succeed.
    async fn load(&self) -> Result<PropertyMetadata, MetadataError>;

    /// Replace the stored metadata with `meta`.
    async fn save(&self, meta: &PropertyMetadata) -> Result<(), MetadataError>;

    /// Merge `incoming` into the existing snapshot (incoming wins on
    /// conflict) and persist the result. Returns the merged snapshot.
    async fn update(
        &self,
        incoming: &PropertyMetadata,
    ) -> Result<PropertyMetadata, MetadataError> {
        let mut current = self.load().await?;
        current.merge(incoming);
        self.save(&current).await?;
        Ok(current)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn merge_overwrites_existing_keys() {
        let mut a = PropertyMetadata::new();
        a.insert("Camera.state", "old");
        a.insert("Camera.id", "primary");

        let mut b = PropertyMetadata::new();
        b.insert("Camera.state", "new");
        b.insert("Place.address", "where");

        a.merge(&b);
        assert_eq!(a.get("Camera.state"), Some("new"));
        assert_eq!(a.get("Camera.id"), Some("primary"));
        assert_eq!(a.get("Place.address"), Some("where"));
    }
}
