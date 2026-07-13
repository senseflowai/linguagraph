//! Subgraph → `GraphBuilder::from_json`-compatible export.
//!
//! The produced document (`{entities, relations}`) round-trips through
//! [`crate::graph::GraphBuilder::from_json`], so an exported fragment
//! can be re-ingested, diffed, or shipped to another linguagraph
//! deployment. Property types are recovered from the display
//! classification ([`PropertyGroups`]) so re-ingesting keeps semantic
//! `Text` fields embedded and dates typed.

use serde_json::{json, Map, Value as JsonValue};

use super::dto::{ExportDoc, PropertyGroups, Subgraph};

/// Render a subgraph as GraphBuilder JSON. Edges whose endpoints are not
/// both present in `subgraph.nodes` are dropped (the importer could not
/// resolve them).
pub(crate) fn export_subgraph(subgraph: &Subgraph) -> ExportDoc {
    let entities: Vec<JsonValue> = subgraph
        .nodes
        .iter()
        .map(|node| {
            json!({
                "id": node.id,
                "type": node.entity_type,
                "properties": typed_properties(&node.properties, node.confidence),
            })
        })
        .collect();

    let node_ids: Vec<&str> = subgraph.nodes.iter().map(|n| n.id.as_str()).collect();
    let relations: Vec<JsonValue> = subgraph
        .edges
        .iter()
        .filter(|e| node_ids.contains(&e.from.as_str()) && node_ids.contains(&e.to.as_str()))
        .map(|edge| {
            let mut props = Map::new();
            for (name, value) in &edge.properties {
                props.insert(name.clone(), value.clone());
            }
            json!({
                "from": edge.from,
                "to": edge.to,
                "type": edge.edge_type,
                "properties": props,
            })
        })
        .collect();

    ExportDoc(json!({
        "entities": entities,
        "relations": relations,
    }))
}

/// Rebuild `{name: {type, value}}` typed property maps from the display
/// buckets.
fn typed_properties(groups: &PropertyGroups, confidence: Option<f64>) -> Map<String, JsonValue> {
    fn put(out: &mut Map<String, JsonValue>, name: &str, ty: &str, value: &JsonValue) {
        out.insert(name.to_string(), json!({ "type": ty, "value": value }));
    }
    let mut out = Map::new();
    for (name, value) in &groups.identifiers {
        let ty = if value.is_array() { "List" } else { "Keyword" };
        put(&mut out, name, ty, value);
    }
    for (name, value) in &groups.descriptions {
        put(&mut out, name, "Text", value);
    }
    for (name, value) in &groups.facts {
        let ty = if value.is_boolean() { "Bool" } else { "Number" };
        put(&mut out, name, ty, value);
    }
    for (name, value) in &groups.dates {
        put(&mut out, name, "Datetime", value);
    }
    for (name, value) in &groups.other {
        // Untyped — let the importer's shape inference decide.
        out.insert(name.clone(), value.clone());
    }
    if let Some(confidence) = confidence {
        if !out.contains_key("confidence") {
            put(&mut out, "confidence", "Number", &json!(confidence));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::explore::dto::{EdgeView, NodeView};
    use crate::graph::GraphBuilder;
    use serde_json::json;
    use std::collections::BTreeMap;

    fn subgraph() -> Subgraph {
        let person = NodeView {
            id: "p1".into(),
            name: "Keanu Reeves".into(),
            entity_type: "Person".into(),
            labels: vec!["E2E".into(), "Person".into()],
            properties: PropertyGroups {
                identifiers: BTreeMap::from([
                    ("id".to_string(), json!("p1")),
                    ("name".to_string(), json!("Keanu Reeves")),
                ]),
                ..Default::default()
            },
            confidence: Some(0.9),
            ephemeral_handle: false,
        };
        let movie = NodeView {
            id: "m1".into(),
            name: "The Matrix".into(),
            entity_type: "Movie".into(),
            labels: vec!["E2E".into(), "Movie".into()],
            properties: PropertyGroups {
                identifiers: BTreeMap::from([("title".to_string(), json!("The Matrix"))]),
                descriptions: BTreeMap::from([(
                    "tagline".to_string(),
                    json!("Welcome to the Real World"),
                )]),
                dates: BTreeMap::from([("released".to_string(), json!("1999-03-31"))]),
                facts: BTreeMap::from([("votes".to_string(), json!(4500))]),
                ..Default::default()
            },
            confidence: None,
            ephemeral_handle: false,
        };
        let edge = EdgeView {
            id: "p1:ACTED_IN:m1".into(),
            edge_type: "ACTED_IN".into(),
            from: "p1".into(),
            to: "m1".into(),
            properties: BTreeMap::from([("roles".to_string(), json!(["Neo"]))]),
            confidence: None,
        };
        let dangling = EdgeView {
            id: "p1:KNOWS:ghost".into(),
            edge_type: "KNOWS".into(),
            from: "p1".into(),
            to: "ghost".into(), // not in the node set → dropped
            properties: BTreeMap::new(),
            confidence: None,
        };
        Subgraph {
            nodes: vec![person, movie],
            edges: vec![edge, dangling],
            truncated: false,
        }
    }

    #[test]
    fn export_round_trips_through_graph_builder() {
        let doc = export_subgraph(&subgraph());
        let raw = doc.0.to_string();
        let graph = GraphBuilder::from_json(&raw).expect("export re-imports cleanly");

        // 2 exported entities (the importer may add built-ins, never fewer).
        let user_entities: Vec<_> = graph
            .entities()
            .iter()
            .filter(|e| e.r#type == "Person" || e.r#type == "Movie")
            .collect();
        assert_eq!(user_entities.len(), 2);

        let acted_in: Vec<_> = graph
            .relations()
            .iter()
            .filter(|r| r.r#type == "ACTED_IN")
            .collect();
        assert_eq!(acted_in.len(), 1, "dangling edge dropped");

        // Typed properties survive the round trip.
        let movie = graph
            .entities()
            .iter()
            .find(|e| e.r#type == "Movie")
            .unwrap();
        use crate::graph::PropertyType;
        assert_eq!(
            movie.properties["tagline"].property_type,
            PropertyType::Text
        );
        assert_eq!(
            movie.properties["released"].property_type,
            PropertyType::Datetime
        );
        assert_eq!(movie.properties["votes"].property_type, PropertyType::Number);
    }

    #[test]
    fn confidence_exports_as_numeric_property() {
        let doc = export_subgraph(&subgraph());
        let person = doc.0["entities"]
            .as_array()
            .unwrap()
            .iter()
            .find(|e| e["id"] == "p1")
            .unwrap();
        assert_eq!(person["properties"]["confidence"]["value"], json!(0.9));
    }
}
