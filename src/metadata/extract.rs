//! Pull description annotations and field-type tags out of a [`Mapping`]
//! document.
//!
//! Keys are the property path in the *graph node*, not the JSONPath in the
//! source document — `Camera.state`, not `$.cameras[*].state`. That's the
//! shape the prompt generator and downstream consumers need.
//!
//! Field types travel alongside descriptions so that, at query time, the
//! DSL lowerer can auto-resolve a [`crate::types::TypeHandler`] for a
//! property without requiring the LLM-emitted DSL to repeat the type tag.

use crate::mapper::Mapping;

use super::{PropertyInfo, PropertyMetadata};

/// Build a metadata snapshot from a mapping. Entries with neither a
/// description nor a type are skipped — the cache stores only annotated
/// paths.
pub fn collect_from_mapping(mapping: &Mapping) -> PropertyMetadata {
    let mut meta = PropertyMetadata::new();
    for ent in &mapping.entities {
        if let Some(desc) = ent.description.as_deref() {
            if !desc.is_empty() {
                meta.insert(ent.kind.clone(), desc);
            }
        }
        for prop in &ent.properties {
            let key = format!("{}.{}", ent.kind, prop.name);
            let mut info = PropertyInfo::default();
            if let Some(desc) = prop.description.as_deref() {
                if !desc.is_empty() {
                    info.description = Some(desc.to_string());
                }
            }
            if let Some(ty) = prop.field_type.as_deref() {
                if !ty.is_empty() {
                    info.field_type = Some(ty.to_string());
                }
            }
            if !info.is_empty() {
                meta.insert_info(key, info);
            }
        }
    }
    meta
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn collects_entity_and_property_descriptions() {
        let mapping: Mapping = serde_json::from_value(json!({
            "entities": [{
                "type": "Camera",
                "source_path": "$.cameras[*]",
                "primary_key": "$.cameras[*].id",
                "description": "An IP camera",
                "properties": [
                    {"name": "id", "source_path": "$.cameras[*].id"},
                    {
                        "name": "state",
                        "source_path": "$.cameras[*].state",
                        "description": "active or inactive"
                    }
                ]
            }]
        }))
        .unwrap();

        let meta = collect_from_mapping(&mapping);
        assert_eq!(meta.get("Camera"), Some("An IP camera"));
        assert_eq!(meta.get("Camera.state"), Some("active or inactive"));
        assert_eq!(meta.get("Camera.id"), None);
        assert_eq!(meta.len(), 2);
    }

    #[test]
    fn collects_field_types() {
        let mapping: Mapping = serde_json::from_value(json!({
            "entities": [{
                "type": "Company",
                "source_path": "$.companies[*]",
                "primary_key": "$.companies[*].id",
                "properties": [
                    {"name": "id", "source_path": "$.companies[*].id"},
                    {
                        "name": "name",
                        "source_path": "$.companies[*].name",
                        "type": "SemanticText",
                        "description": "the company name"
                    },
                    {
                        "name": "industry",
                        "source_path": "$.companies[*].industry"
                    }
                ]
            }]
        }))
        .unwrap();

        let meta = collect_from_mapping(&mapping);
        // Typed property: both description and type captured.
        assert_eq!(meta.get_type("Company.name"), Some("SemanticText"));
        assert_eq!(meta.get("Company.name"), Some("the company name"));
        // Untyped, undocumented properties are still skipped entirely.
        assert!(meta.info("Company.industry").is_none());
    }

    #[test]
    fn type_only_property_still_recorded() {
        let mapping: Mapping = serde_json::from_value(json!({
            "entities": [{
                "type": "Company",
                "source_path": "$.companies[*]",
                "primary_key": "$.companies[*].id",
                "properties": [
                    {"name": "id", "source_path": "$.companies[*].id"},
                    {
                        "name": "name",
                        "source_path": "$.companies[*].name",
                        "type": "SemanticText"
                    }
                ]
            }]
        }))
        .unwrap();
        let meta = collect_from_mapping(&mapping);
        assert_eq!(meta.get_type("Company.name"), Some("SemanticText"));
        assert_eq!(meta.get("Company.name"), None);
    }

    #[test]
    fn skips_empty_descriptions_and_types() {
        let mapping: Mapping = serde_json::from_value(json!({
            "entities": [{
                "type": "X",
                "source_path": "$.x[*]",
                "primary_key": "$.x[*].id",
                "description": "",
                "properties": [
                    {"name": "y", "source_path": "$.x[*].y", "description": ""}
                ]
            }]
        }))
        .unwrap();
        assert!(collect_from_mapping(&mapping).is_empty());
    }
}
