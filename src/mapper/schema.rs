//! Typed mirror of the mapping JSON.
//!
//! Mapping authors describe how to lift raw JSON into graph nodes and edges.
//! This module is purely structural; semantic checks (paths exist, primary
//! keys resolve, relationship endpoints are defined) live in
//! [`super::extractor`] and [`crate::ingest::planner`].

use std::path::Path;

use serde::{Deserialize, Serialize};
use tokio::fs;

use super::MapperError;

/// Top-level mapping document.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Mapping {
    pub entities: Vec<EntityMapping>,
    #[serde(default)]
    pub relationships: Vec<RelationshipMapping>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityMapping {
    /// Cypher node label, e.g. "Camera".
    #[serde(rename = "type")]
    pub kind: String,

    /// JSONPath that selects one row per match.
    pub source_path: String,

    /// JSONPath that resolves to the row's stable identifier.
    pub primary_key: String,

    #[serde(default)]
    pub properties: Vec<PropertyMapping>,

    /// Optional human-readable name (description-only metadata; the mapper
    /// does not insert this as a property unless it's a plain JSONPath).
    #[serde(default)]
    pub name: Option<String>,

    #[serde(default)]
    pub description: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PropertyMapping {
    pub name: String,
    pub source_path: String,
    #[serde(default)]
    pub description: Option<String>,
    /// Optional type tag matching a registered [`crate::types::TypeHandler`]
    /// (e.g. `"SemanticText"`). When present, ingestion delegates the
    /// property to the handler — which may rewrite the value, drop it,
    /// or queue side effects (embeddings, geo indexing, …).
    #[serde(default, rename = "type", skip_serializing_if = "Option::is_none")]
    pub field_type: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelationshipMapping {
    #[serde(rename = "type")]
    pub kind: String,
    pub from: String,
    pub to: String,
}

impl Mapping {
    pub async fn load(p: &Path) -> Result<Self, MapperError> {
        let raw = fs::read_to_string(p).await?;
        Self::from_str(&raw)
    }

    pub fn from_str(raw: &str) -> Result<Self, MapperError> {
        let m: Mapping = serde_json::from_str(raw)?;
        Ok(m)
    }
}
