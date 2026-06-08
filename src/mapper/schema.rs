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
    #[serde(default)]
    pub source: Option<String>,
    /// Ontology domain stamped on every entity produced from this
    /// mapping. Defaults to [`crate::mapper::DEFAULT_MAPPING_DOMAIN`]
    /// when not supplied. The domain is also the key under which the
    /// derived [`crate::graph::OntologyCatalog`] entry is filed.
    #[serde(default)]
    pub domain: Option<String>,
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
    /// Type tag matching a registered [`crate::types::TypeHandler`]
    /// (e.g. `"Text"`, `"Number"`, `"SemanticText"`). **Required** —
    /// mappings without an explicit type are rejected at load time so
    /// authors don't accidentally rely on the lossy default JSON →
    /// [`crate::ast::query::Literal`] conversion. Use [`Mapping::validate`]
    /// to surface a precise [`MapperError::MissingPropertyType`] before
    /// any extraction work is done.
    ///
    /// Held as `Option<String>` at the serde layer so we can produce a
    /// better error than the default `serde` "missing field" message.
    /// After validation it is guaranteed to be `Some`.
    #[serde(default, rename = "type", skip_serializing_if = "Option::is_none")]
    pub field_type: Option<String>,
}

impl PropertyMapping {
    /// Returns the declared type. Panics in debug builds if the
    /// mapping has not been validated (production callers should
    /// always validate). Use [`Self::field_type`] for the raw `Option`.
    pub fn type_name(&self) -> &str {
        debug_assert!(
            self.field_type.is_some(),
            "PropertyMapping::type_name called on an unvalidated property"
        );
        self.field_type.as_deref().unwrap_or("")
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelationshipMapping {
    #[serde(rename = "type")]
    pub kind: String,
    pub from: String,
    pub to: String,
    /// Optional foreign-key JSONPath on the `from` entity (e.g.
    /// `"$.cameras[*].place_id"`). When set, the relationship is resolved
    /// by **value join** — a `from` row links to a `to` row whose
    /// [`Self::to_key`] value equals this key's value — instead of the
    /// default array-context alignment. This is what makes relationships
    /// between sibling top-level arrays (no nesting) expressible.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub from_key: Option<String>,
    /// Optional JSONPath of the matching key on the `to` entity (e.g.
    /// `"$.places[*].id"`). Only meaningful together with
    /// [`Self::from_key`]. When omitted it defaults to the `to` entity's
    /// `primary_key`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub to_key: Option<String>,
}

impl Mapping {
    pub async fn load(p: &Path) -> Result<Self, MapperError> {
        let raw = fs::read_to_string(p).await?;
        Self::from_str(&raw)
    }

    /// Parse and validate a mapping JSON. Validation runs eagerly so a
    /// bad mapping never reaches the extractor.
    pub fn from_str(raw: &str) -> Result<Self, MapperError> {
        let m: Mapping = serde_json::from_str(raw)?;
        m.validate()?;
        Ok(m)
    }

    /// Run all schema-level checks. Currently:
    ///
    /// * every property declares a non-empty `type`.
    ///
    /// Other consistency checks (path validity, primary key shape, …)
    /// live in the extractor and the ingest planner — they need access
    /// to the registered type handlers and the input data, which the
    /// schema layer doesn't see.
    pub fn validate(&self) -> Result<(), MapperError> {
        for ent in &self.entities {
            for prop in &ent.properties {
                match prop.field_type.as_deref() {
                    Some(t) if !t.trim().is_empty() => {}
                    _ => {
                        return Err(MapperError::MissingPropertyType {
                            entity: ent.kind.clone(),
                            property: prop.name.clone(),
                        });
                    }
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn validate_rejects_property_without_type() {
        let mapping: Mapping = serde_json::from_value(json!({
            "entities": [{
                "type": "Camera",
                "source_path": "$.cameras[*]",
                "primary_key": "$.cameras[*].id",
                "properties": [
                    {"name": "name", "source_path": "$.cameras[*].name"}
                ]
            }]
        }))
        .unwrap();
        let err = mapping.validate().unwrap_err();
        match err {
            MapperError::MissingPropertyType { entity, property } => {
                assert_eq!(entity, "Camera");
                assert_eq!(property, "name");
            }
            other => panic!("expected MissingPropertyType, got {other:?}"),
        }
    }

    #[test]
    fn validate_rejects_empty_type() {
        let mapping: Mapping = serde_json::from_value(json!({
            "entities": [{
                "type": "Camera",
                "source_path": "$.cameras[*]",
                "primary_key": "$.cameras[*].id",
                "properties": [
                    {"name": "n", "source_path": "$.cameras[*].n", "type": "  "}
                ]
            }]
        }))
        .unwrap();
        assert!(matches!(
            mapping.validate(),
            Err(MapperError::MissingPropertyType { .. })
        ));
    }

    #[test]
    fn validate_passes_when_every_property_has_a_type() {
        let mapping: Mapping = serde_json::from_value(json!({
            "entities": [{
                "type": "Camera",
                "source_path": "$.cameras[*]",
                "primary_key": "$.cameras[*].id",
                "properties": [
                    {"name": "id", "source_path": "$.cameras[*].id", "type": "Text"},
                    {"name": "n",  "source_path": "$.cameras[*].n",  "type": "Number"}
                ]
            }]
        }))
        .unwrap();
        mapping.validate().unwrap();
    }

    #[test]
    fn from_str_runs_validation() {
        let raw = r#"{
            "entities": [{
                "type": "Camera",
                "source_path": "$.cameras[*]",
                "primary_key": "$.cameras[*].id",
                "properties": [
                    {"name": "name", "source_path": "$.cameras[*].name"}
                ]
            }]
        }"#;
        assert!(matches!(
            Mapping::from_str(raw),
            Err(MapperError::MissingPropertyType { .. })
        ));
    }
}
