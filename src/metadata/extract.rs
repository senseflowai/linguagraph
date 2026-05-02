//! Pull description annotations out of a [`Mapping`] document.
//!
//! Keys are the property path in the *graph node*, not the JSONPath in the
//! source document — `Camera.state`, not `$.cameras[*].state`. That's the
//! shape the prompt generator and downstream consumers need.

use crate::mapper::Mapping;

use super::PropertyMetadata;

/// Build a metadata snapshot from a mapping. Entities and properties without
/// descriptions are skipped — the cache stores only annotated paths.
pub fn collect_from_mapping(mapping: &Mapping) -> PropertyMetadata {
    let mut meta = PropertyMetadata::new();
    for ent in &mapping.entities {
        if let Some(desc) = ent.description.as_deref() {
            if !desc.is_empty() {
                meta.insert(ent.kind.clone(), desc);
            }
        }
        for prop in &ent.properties {
            if let Some(desc) = prop.description.as_deref() {
                if !desc.is_empty() {
                    meta.insert(format!("{}.{}", ent.kind, prop.name), desc);
                }
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
    fn skips_empty_descriptions() {
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
