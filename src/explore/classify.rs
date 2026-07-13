//! Property classification and label reduction for explorer views.
//!
//! The ontology catalog knows each property's [`OntologyPropertyType`];
//! this module folds raw `properties(n)` maps into the display-oriented
//! [`PropertyGroups`] buckets and reduces raw Cypher label sets to the
//! one business label a UI should show.

use std::collections::BTreeMap;

use serde_json::Value as JsonValue;

use crate::graph::{OntologyCatalog, OntologyPropertyType, Scope};

use super::dto::PropertyGroups;

/// Property names never shown to users (`_canonical` and friends).
pub(crate) fn is_system_property(name: &str) -> bool {
    name.starts_with('_')
}

/// Split a raw property map into display buckets.
///
/// Classification prefers the ontology (`catalog.get_property`); unknown
/// properties fall back to value-shape inference (numbers/bools →
/// `facts`, everything else → `other`).
pub(crate) fn classify_properties(
    entity_type: &str,
    props: BTreeMap<String, JsonValue>,
    catalog: Option<&OntologyCatalog>,
) -> PropertyGroups {
    let mut groups = PropertyGroups::default();
    for (name, value) in props {
        if is_system_property(&name) {
            continue;
        }
        let spec_type =
            catalog.and_then(|c| c.get_property(entity_type, &name).map(|s| s.property_type));
        let bucket = match spec_type {
            Some(OntologyPropertyType::Keyword) | Some(OntologyPropertyType::List) => {
                &mut groups.identifiers
            }
            Some(OntologyPropertyType::Text) => &mut groups.descriptions,
            Some(OntologyPropertyType::Number) | Some(OntologyPropertyType::Bool) => {
                &mut groups.facts
            }
            Some(OntologyPropertyType::Datetime) => &mut groups.dates,
            None => match &value {
                JsonValue::Number(_) | JsonValue::Bool(_) => &mut groups.facts,
                _ => &mut groups.other,
            },
        };
        bucket.insert(name, value);
    }
    groups
}

/// Pick the business label out of a raw Cypher label set: skip the tenant
/// prefix, scope labels (`scope_*`) and ontology domain names; prefer a
/// label the catalog declares as an entity type.
pub(crate) fn primary_label(
    labels: &[String],
    prefix_label: Option<&str>,
    catalog: Option<&OntologyCatalog>,
) -> String {
    let candidates: Vec<&String> = labels
        .iter()
        .filter(|l| Some(l.as_str()) != prefix_label)
        .filter(|l| Scope::from_cypher_label(l).is_none())
        .filter(|l| !catalog.is_some_and(|c| c.get(l).is_some()))
        .collect();

    if let Some(catalog) = catalog {
        if let Some(declared) = candidates
            .iter()
            .find(|l| catalog.get_entity(l.as_str()).is_some())
        {
            return (*declared).clone();
        }
    }
    candidates
        .first()
        .map(|l| (*l).to_string())
        .unwrap_or_else(|| "Unknown".to_string())
}

/// Display name for a node: `name` → `title` → the public id.
pub(crate) fn display_name(props: &BTreeMap<String, JsonValue>, id: &str) -> String {
    for key in ["name", "title"] {
        if let Some(JsonValue::String(s)) = props.get(key) {
            if !s.trim().is_empty() {
                return s.clone();
            }
        }
    }
    id.to_string()
}

/// The `confidence` convention: surface the property when present, never
/// compute it.
pub(crate) fn confidence(props: &BTreeMap<String, JsonValue>) -> Option<f64> {
    props.get("confidence").and_then(JsonValue::as_f64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{DomainOntology, EntityTypeSpec, PropertySpec};
    use serde_json::json;

    fn property(name: &str, property_type: OntologyPropertyType) -> PropertySpec {
        PropertySpec {
            name: name.to_string(),
            description: None,
            property_type,
            required: false,
            allowed_values: Vec::new(),
        }
    }

    fn catalog_with_movie() -> OntologyCatalog {
        let mut catalog = OntologyCatalog::default();
        catalog.insert(
            "movies",
            DomainOntology {
                name: None,
                description: None,
                entity_types: vec![EntityTypeSpec {
                    name: "Movie".to_string(),
                    description: None,
                    properties: vec![
                        property("id", OntologyPropertyType::Keyword),
                        property("tagline", OntologyPropertyType::Text),
                        property("released", OntologyPropertyType::Datetime),
                        property("votes", OntologyPropertyType::Number),
                    ],
                }],
                relation_types: Vec::new(),
            },
        );
        catalog
    }

    #[test]
    fn classify_buckets_by_catalog_and_falls_back_by_shape() {
        let props = BTreeMap::from([
            ("id".to_string(), json!("m1")),
            ("tagline".to_string(), json!("Welcome to the Real World")),
            ("released".to_string(), json!("1999-03-31")),
            ("votes".to_string(), json!(4500)),
            ("box_office".to_string(), json!(463.5)), // not in catalog → shape
            ("note".to_string(), json!("uncatalogued")), // not in catalog → other
            ("_canonical".to_string(), json!("hidden")), // system → dropped
        ]);
        let groups = classify_properties("Movie", props, Some(&catalog_with_movie()));
        assert_eq!(groups.identifiers.get("id"), Some(&json!("m1")));
        assert!(groups.descriptions.contains_key("tagline"));
        assert!(groups.dates.contains_key("released"));
        assert_eq!(groups.facts.get("votes"), Some(&json!(4500)));
        assert_eq!(groups.facts.get("box_office"), Some(&json!(463.5)));
        assert!(groups.other.contains_key("note"));
        assert!(groups.iter().all(|(k, _)| k != "_canonical"));
    }

    #[test]
    fn classify_without_catalog_uses_shape_only() {
        let props = BTreeMap::from([
            ("age".to_string(), json!(42)),
            ("active".to_string(), json!(true)),
            ("name".to_string(), json!("Alice")),
        ]);
        let groups = classify_properties("Person", props, None);
        assert!(groups.facts.contains_key("age"));
        assert!(groups.facts.contains_key("active"));
        assert!(groups.other.contains_key("name"));
    }

    #[test]
    fn primary_label_skips_prefix_scope_and_domain_labels() {
        let labels = vec![
            "E2E_TCM".to_string(),
            "scope_structured".to_string(),
            "movies".to_string(),
            "Movie".to_string(),
        ];
        let catalog = catalog_with_movie();
        assert_eq!(
            primary_label(&labels, Some("E2E_TCM"), Some(&catalog)),
            "Movie"
        );
        // Without the catalog, domain label can't be recognized — first
        // non-prefix, non-scope label wins.
        assert_eq!(primary_label(&labels, Some("E2E_TCM"), None), "movies");
    }

    #[test]
    fn display_name_prefers_name_then_title_then_id() {
        let with_name = BTreeMap::from([("name".to_string(), json!("Alice"))]);
        assert_eq!(display_name(&with_name, "x"), "Alice");
        let with_title = BTreeMap::from([("title".to_string(), json!("The Matrix"))]);
        assert_eq!(display_name(&with_title, "x"), "The Matrix");
        assert_eq!(display_name(&BTreeMap::new(), "m1"), "m1");
    }

    #[test]
    fn confidence_is_read_never_invented() {
        assert_eq!(
            confidence(&BTreeMap::from([("confidence".to_string(), json!(0.9))])),
            Some(0.9)
        );
        assert_eq!(confidence(&BTreeMap::new()), None);
    }
}
