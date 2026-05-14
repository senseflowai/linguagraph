//! Property metadata: per-property descriptions **and** field-type tags
//! pulled from the mapping.
//!
//! Mapping authors annotate node properties with `description` strings and,
//! for typed fields, a `type` tag (e.g. `SemanticText`). The ingester
//! collects both into a flat key-value structure keyed by the **property
//! path in the graph node** (e.g. `Camera.state`) and persists it through
//! a pluggable [`MetadataStore`].
//!
//! The default backend is a JSON cache file ([`FileMetadataStore`]). The
//! trait abstraction is the seam for a future SQL-backed store: callers
//! depend on `Arc<dyn MetadataStore>`, never on the concrete file impl.
//!
//! ## Why store types
//!
//! At ingestion time the planner already consults `PropertyMapping.field_type`
//! directly. At **query** time the DSL parser does not have access to the
//! mapping, so the only way for a filter like
//! `{"field": "c.name", "op": "search", "value": "apple"}` to resolve a
//! handler without an explicit `"type": "SemanticText"` tag is for the
//! lowering step to look the type up in `PropertyMetadata`. That keeps
//! the DSL terse and prevents type drift between mapping and queries.

mod extract;
mod file;

use std::collections::BTreeMap;

use async_trait::async_trait;
use serde::{Deserialize, Deserializer, Serialize};
use thiserror::Error;

pub use extract::collect_from_mapping;
pub use file::FileMetadataStore;

/// Per-property annotation captured from the mapping.
///
/// Both fields are independently optional: a description without a type
/// is the common case for plain string properties; a type without a
/// description is legitimate for fields where the type alone is the
/// documentation.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct PropertyInfo {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none", rename = "type")]
    pub field_type: Option<String>,
}

impl PropertyInfo {
    pub fn description(d: impl Into<String>) -> Self {
        Self {
            description: Some(d.into()),
            field_type: None,
        }
    }

    pub fn typed(field_type: impl Into<String>) -> Self {
        Self {
            description: None,
            field_type: Some(field_type.into()),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.description.is_none() && self.field_type.is_none()
    }
}

/// Flat map from `<NodeLabel>.<property>` (or `<NodeLabel>` for entity-level
/// descriptions) to the captured [`PropertyInfo`].
///
/// `BTreeMap` keeps iteration order stable, which matters for snapshot tests
/// and for keeping prompt output deterministic.
///
/// ## On-disk format
///
/// New caches serialise as `{"Camera.state": {"description": "..."}}`.
/// **Legacy** caches written by earlier versions used a flat
/// `{"Camera.state": "..."}` map. The custom [`Deserialize`] impl
/// transparently accepts both forms; legacy strings are loaded as the
/// `description` field, with `field_type` left empty.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
#[serde(transparent)]
pub struct PropertyMetadata {
    pub entries: BTreeMap<String, PropertyInfo>,
}

impl<'de> Deserialize<'de> for PropertyMetadata {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        // Accept either {key: string} (legacy) or {key: {description, type}} (current).
        #[derive(Deserialize)]
        #[serde(untagged)]
        enum Either {
            Legacy(String),
            Modern(PropertyInfo),
        }
        let raw: BTreeMap<String, Either> = BTreeMap::deserialize(d)?;
        let mut entries = BTreeMap::new();
        for (k, v) in raw {
            let info = match v {
                Either::Legacy(s) => PropertyInfo::description(s),
                Either::Modern(info) => info,
            };
            entries.insert(k, info);
        }
        Ok(PropertyMetadata { entries })
    }
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

    /// Description for `key`, if any. Kept as `&str` for the prompt
    /// generator's existing call sites.
    pub fn get(&self, key: &str) -> Option<&str> {
        self.entries.get(key).and_then(|i| i.description.as_deref())
    }

    /// Field-type tag for `key`, if any. The DSL lowerer uses this to
    /// auto-select a [`crate::types::TypeHandler`] when a filter omits
    /// `"type"`.
    pub fn get_type(&self, key: &str) -> Option<&str> {
        self.entries.get(key).and_then(|i| i.field_type.as_deref())
    }

    /// Full [`PropertyInfo`] for `key`, if any.
    pub fn info(&self, key: &str) -> Option<&PropertyInfo> {
        self.entries.get(key)
    }

    /// Insert a description (legacy ergonomic helper).
    pub fn insert(&mut self, key: impl Into<String>, value: impl Into<String>) {
        let entry = self.entries.entry(key.into()).or_default();
        entry.description = Some(value.into());
    }

    /// Set the field-type tag for `key`, creating the entry if needed.
    pub fn insert_type(&mut self, key: impl Into<String>, field_type: impl Into<String>) {
        let entry = self.entries.entry(key.into()).or_default();
        entry.field_type = Some(field_type.into());
    }

    /// Insert a fully-formed entry, replacing any prior info under `key`.
    pub fn insert_info(&mut self, key: impl Into<String>, info: PropertyInfo) {
        self.entries.insert(key.into(), info);
    }

    /// Merge `other` into `self`. Per-field semantics:
    ///
    /// * `description` and `field_type` are updated independently.
    /// * Where `other` has `Some(_)`, it wins; where `other` has `None`,
    ///   the existing value is preserved.
    ///
    /// This means re-running ingest with a freshly-stripped mapping
    /// won't blow away annotations from earlier runs.
    pub fn merge(&mut self, other: &PropertyMetadata) {
        for (k, v) in &other.entries {
            let entry = self.entries.entry(k.clone()).or_default();
            if v.description.is_some() {
                entry.description = v.description.clone();
            }
            if v.field_type.is_some() {
                entry.field_type = v.field_type.clone();
            }
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
    /// conflict, per-field) and persist the result. Returns the merged
    /// snapshot.
    async fn update(&self, incoming: &PropertyMetadata) -> Result<PropertyMetadata, MetadataError> {
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
    fn merge_overwrites_description_per_field() {
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

    #[test]
    fn merge_preserves_existing_type_when_other_has_none() {
        let mut a = PropertyMetadata::new();
        a.insert_type("Company.name", "SemanticText");
        a.insert("Company.name", "the company name");

        let mut b = PropertyMetadata::new();
        b.insert("Company.name", "official company name"); // description-only update

        a.merge(&b);
        assert_eq!(a.get("Company.name"), Some("official company name"));
        assert_eq!(a.get_type("Company.name"), Some("SemanticText"));
    }

    #[test]
    fn merge_updates_type_when_other_has_some() {
        let mut a = PropertyMetadata::new();
        a.insert_type("Company.name", "Keyword");

        let mut b = PropertyMetadata::new();
        b.insert_type("Company.name", "SemanticText");

        a.merge(&b);
        assert_eq!(a.get_type("Company.name"), Some("SemanticText"));
    }

    #[test]
    fn legacy_flat_map_form_deserialises() {
        // Cache written by an earlier linguagraph version.
        let raw = r#"{
            "Camera": "An IP camera",
            "Camera.state": "active or inactive"
        }"#;
        let meta: PropertyMetadata = serde_json::from_str(raw).unwrap();
        assert_eq!(meta.get("Camera"), Some("An IP camera"));
        assert_eq!(meta.get("Camera.state"), Some("active or inactive"));
        assert_eq!(meta.get_type("Camera"), None);
    }

    #[test]
    fn modern_form_round_trips() {
        let mut meta = PropertyMetadata::new();
        meta.insert("Company.name", "the name");
        meta.insert_type("Company.name", "SemanticText");
        let json = serde_json::to_string(&meta).unwrap();
        let back: PropertyMetadata = serde_json::from_str(&json).unwrap();
        assert_eq!(back, meta);
        // Sanity: the modern form serialises as a struct, not a string.
        assert!(json.contains("\"description\""));
        assert!(json.contains("\"type\":\"SemanticText\""));
    }

    #[test]
    fn mixed_legacy_and_modern_entries_deserialise() {
        let raw = r#"{
            "Camera": "An IP camera",
            "Company.name": {"description": "the name", "type": "SemanticText"}
        }"#;
        let meta: PropertyMetadata = serde_json::from_str(raw).unwrap();
        assert_eq!(meta.get("Camera"), Some("An IP camera"));
        assert_eq!(meta.get("Company.name"), Some("the name"));
        assert_eq!(meta.get_type("Company.name"), Some("SemanticText"));
    }
}
