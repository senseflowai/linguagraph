//! Pure conversion from raw [`QueryResult`]s into the graph DTOs.
//!
//! Kept free of I/O so it can be unit-tested against hand-built results
//! with no database. The input column shapes are the ones emitted by
//! [`crate::builder::build_read_graph`] (the `nodes` / `edges` lists) and
//! by [`Pipeline::entity_detail`](crate::core::Pipeline::entity_detail) /
//! [`relation_detail`](crate::core::Pipeline::relation_detail).

use std::collections::BTreeMap;

use serde_json::{Map, Value};

use crate::builder::{GRAPH_EDGES_COLUMN, GRAPH_NODES_COLUMN};
use crate::db::{NodeType, QueryResult, Row, Value as DbValue};

use super::dto::{
    Endpoint, EntityDetail, GraphEdge, GraphNode, RelationDetail, RelationSummary,
};

/// Fold a graph-shaped [`QueryResult`] (columns `nodes` / `edges`, each a
/// list of maps) into deduplicated node and edge DTOs. Nodes and edges
/// repeated across rows collapse by id; entries without an id are skipped
/// (e.g. an unmatched OPTIONAL endpoint).
pub fn query_result_to_graph(result: &QueryResult) -> (Vec<GraphNode>, Vec<GraphEdge>) {
    let mut nodes: BTreeMap<String, GraphNode> = BTreeMap::new();
    let mut edges: BTreeMap<String, GraphEdge> = BTreeMap::new();

    for row in &result.rows {
        if let Some(arr) = field_array(row, GRAPH_NODES_COLUMN) {
            for item in arr {
                if let Some(node) = parse_node(item) {
                    nodes.entry(node.id.clone()).or_insert(node);
                }
            }
        }
        if let Some(arr) = field_array(row, GRAPH_EDGES_COLUMN) {
            for item in arr {
                if let Some(edge) = parse_edge(item) {
                    edges.entry(edge.id.clone()).or_insert(edge);
                }
            }
        }
    }

    (nodes.into_values().collect(), edges.into_values().collect())
}

/// Shape the first row of an `entity_detail` result into an
/// [`EntityDetail`]. Returns `None` when the id was unknown (no rows).
pub fn to_entity_detail(result: &QueryResult) -> Option<EntityDetail> {
    let row = result.rows.first()?;
    let id = field(row, "id").and_then(json_id_string)?;
    let labels = field(row, "labels").map(labels_vec).unwrap_or_default();
    let properties = field(row, "props").map(object_map).unwrap_or_default();
    let sources = field(row, "sources").and_then(value_array).unwrap_or_default();
    let relations = field(row, "relations")
        .and_then(Value::as_array)
        .map(|arr| arr.iter().filter_map(parse_relation_summary).collect())
        .unwrap_or_default();

    Some(EntityDetail {
        id,
        kind: node_kind(&labels),
        label: primary_label(&labels),
        name: display_name(&properties),
        properties,
        sources,
        relations,
    })
}

/// Shape the first row of a `relation_detail` result into a
/// [`RelationDetail`]. Returns `None` when the id was unknown.
pub fn to_relation_detail(result: &QueryResult) -> Option<RelationDetail> {
    let row = result.rows.first()?;
    let id = field(row, "id").and_then(json_id_string)?;
    let rel = field(row, "type").and_then(Value::as_str).unwrap_or_default().to_string();
    let from = endpoint(row, "from", "from_labels", "from_props")?;
    let to = endpoint(row, "to", "to_labels", "to_props")?;
    let properties = field(row, "props").map(object_map).unwrap_or_default();

    Some(RelationDetail { id, rel, from, to, properties })
}

fn endpoint(row: &Row, id_col: &str, labels_col: &str, props_col: &str) -> Option<Endpoint> {
    let id = field(row, id_col).and_then(json_id_string)?;
    let labels = field(row, labels_col).map(labels_vec).unwrap_or_default();
    let properties = field(row, props_col).map(object_map).unwrap_or_default();
    Some(Endpoint {
        id,
        kind: node_kind(&labels),
        label: primary_label(&labels),
        name: display_name(&properties),
        properties,
    })
}

fn parse_node(item: &Value) -> Option<GraphNode> {
    let obj = item.as_object()?;
    let id = obj.get("id").and_then(json_id_string)?;
    let labels = obj.get("labels").map(labels_vec).unwrap_or_default();
    let properties = obj.get("props").map(object_map).unwrap_or_default();
    let sources = obj.get("sources").and_then(value_array).unwrap_or_default();
    Some(GraphNode {
        id,
        kind: node_kind(&labels),
        label: primary_label(&labels),
        name: display_name(&properties),
        properties,
        sources,
    })
}

fn parse_edge(item: &Value) -> Option<GraphEdge> {
    let obj = item.as_object()?;
    let id = obj.get("id").and_then(json_id_string)?;
    let from = obj.get("from").and_then(json_id_string)?;
    let to = obj.get("to").and_then(json_id_string)?;
    let rel = obj.get("rel").and_then(Value::as_str).unwrap_or_default().to_string();
    let properties = obj.get("props").map(object_map).unwrap_or_default();
    Some(GraphEdge { id, from, to, rel, properties })
}

fn parse_relation_summary(item: &Value) -> Option<RelationSummary> {
    let obj = item.as_object()?;
    let id = obj.get("id").and_then(json_id_string)?;
    let rel = obj.get("type").and_then(Value::as_str).unwrap_or_default().to_string();
    let from = obj.get("from").and_then(json_id_string)?;
    let to = obj.get("to").and_then(json_id_string)?;
    let other_id = obj.get("other_id").and_then(json_id_string)?;
    let other_labels = obj.get("other_labels").map(labels_vec).unwrap_or_default();
    let other_props = obj.get("other_props").map(object_map).unwrap_or_default();
    let properties = obj.get("props").map(object_map).unwrap_or_default();
    let other_label = {
        let l = primary_label(&other_labels);
        (!l.is_empty()).then_some(l)
    };
    Some(RelationSummary {
        id,
        rel,
        from,
        to,
        other_id,
        other_label,
        other_name: display_name(&other_props),
        properties,
    })
}

// ── small value helpers ────────────────────────────────────────────────

/// Borrow a row column's inner JSON value. Every column these queries
/// project is decoded as [`DbValue::Json`]; anything else is treated as
/// absent.
fn field<'a>(row: &'a Row, name: &str) -> Option<&'a Value> {
    match row.fields.get(name) {
        Some(DbValue::Json(v)) => Some(v),
        _ => None,
    }
}

fn field_array<'a>(row: &'a Row, name: &str) -> Option<&'a Vec<Value>> {
    field(row, name).and_then(Value::as_array)
}

/// Stringify a JSON integer id. Returns `None` for null / non-integers so
/// unmatched OPTIONAL endpoints drop out cleanly.
fn json_id_string(v: &Value) -> Option<String> {
    v.as_i64().map(|i| i.to_string())
}

fn labels_vec(v: &Value) -> Vec<String> {
    v.as_array()
        .map(|arr| {
            arr.iter()
                .filter_map(|l| l.as_str().map(str::to_string))
                .collect()
        })
        .unwrap_or_default()
}

fn object_map(v: &Value) -> Map<String, Value> {
    v.as_object().cloned().unwrap_or_default()
}

fn value_array(v: &Value) -> Option<Vec<Value>> {
    v.as_array().cloned()
}

fn primary_label(labels: &[String]) -> String {
    labels.first().cloned().unwrap_or_default()
}

fn node_kind(labels: &[String]) -> NodeType {
    let mut kind = NodeType::Entity;
    for label in labels {
        match NodeType::from_label(label) {
            NodeType::Source => return NodeType::Source,
            NodeType::Chunk => kind = NodeType::Chunk,
            NodeType::Entity => {}
        }
    }
    kind
}

fn display_name(props: &Map<String, Value>) -> Option<String> {
    for key in ["name", "title"] {
        if let Some(Value::String(s)) = props.get(key) {
            return Some(s.clone());
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::{Column, QueryResult, Row};
    use serde_json::json;

    fn json_row(pairs: Vec<(&str, Value)>) -> Row {
        let mut row = Row::default();
        for (k, v) in pairs {
            row.fields.insert(k.to_string(), DbValue::Json(v));
        }
        row
    }

    #[test]
    fn builds_graph_from_node_and_edge_lists() {
        let row = json_row(vec![
            (
                "nodes",
                json!([
                    {"alias":"p","id":1,"labels":["Person"],"props":{"name":"Elena"},"sources":[{"name":"Doc1"}]},
                    {"alias":"c","id":2,"labels":["Company"],"props":{"name":"Acme"},"sources":[]}
                ]),
            ),
            (
                "edges",
                json!([{"id":10,"rel":"OWNS","from":1,"to":2,"props":{"share":80}}]),
            ),
        ]);
        let result = QueryResult {
            columns: vec![Column::new("nodes"), Column::new("edges")],
            rows: vec![row],
        };

        let (nodes, edges) = query_result_to_graph(&result);
        assert_eq!(nodes.len(), 2);
        let elena = nodes.iter().find(|n| n.id == "1").unwrap();
        assert_eq!(elena.name.as_deref(), Some("Elena"));
        assert_eq!(elena.kind, NodeType::Entity);
        assert_eq!(elena.label, "Person");
        assert_eq!(elena.sources.len(), 1);

        assert_eq!(edges.len(), 1);
        assert_eq!(edges[0].from, "1");
        assert_eq!(edges[0].to, "2");
        assert_eq!(edges[0].rel, "OWNS");
    }

    #[test]
    fn dedups_nodes_and_edges_across_rows() {
        let mk = || {
            json_row(vec![
                (
                    "nodes",
                    json!([{"alias":"p","id":1,"labels":["Person"],"props":{},"sources":[]}]),
                ),
                (
                    "edges",
                    json!([{"id":10,"rel":"OWNS","from":1,"to":2,"props":{}}]),
                ),
            ])
        };
        let result = QueryResult {
            columns: vec![Column::new("nodes"), Column::new("edges")],
            rows: vec![mk(), mk()],
        };
        let (nodes, edges) = query_result_to_graph(&result);
        assert_eq!(nodes.len(), 1);
        assert_eq!(edges.len(), 1);
    }

    #[test]
    fn skips_null_ids_and_empty_result() {
        let (nodes, edges) = query_result_to_graph(&QueryResult::default());
        assert!(nodes.is_empty() && edges.is_empty());

        let row = json_row(vec![(
            "nodes",
            json!([{"alias":"c","id":null,"labels":["Company"],"props":{},"sources":[]}]),
        )]);
        let result = QueryResult {
            columns: vec![Column::new("nodes")],
            rows: vec![row],
        };
        let (nodes, _) = query_result_to_graph(&result);
        assert!(nodes.is_empty(), "null-id node should be skipped");
    }

    #[test]
    fn source_label_maps_to_source_kind() {
        let row = json_row(vec![(
            "nodes",
            json!([{"alias":"s","id":7,"labels":["Source"],"props":{"name":"Doc"},"sources":[]}]),
        )]);
        let result = QueryResult {
            columns: vec![Column::new("nodes")],
            rows: vec![row],
        };
        let (nodes, _) = query_result_to_graph(&result);
        assert_eq!(nodes[0].kind, NodeType::Source);
    }

    #[test]
    fn shapes_entity_detail() {
        let row = json_row(vec![
            ("id", json!(1)),
            ("labels", json!(["Person"])),
            ("props", json!({"name":"Elena","role":"CEO"})),
            ("sources", json!([{"name":"Doc1"}])),
            (
                "relations",
                json!([{
                    "id":10,"type":"OWNS","from":1,"to":2,
                    "other_id":2,"other_labels":["Company"],
                    "other_props":{"name":"Acme"},"props":{"share":80}
                }]),
            ),
        ]);
        let result = QueryResult {
            columns: vec![],
            rows: vec![row],
        };
        let detail = to_entity_detail(&result).expect("entity");
        assert_eq!(detail.id, "1");
        assert_eq!(detail.name.as_deref(), Some("Elena"));
        assert_eq!(detail.relations.len(), 1);
        let r = &detail.relations[0];
        assert_eq!(r.rel, "OWNS");
        assert_eq!(r.other_id, "2");
        assert_eq!(r.other_name.as_deref(), Some("Acme"));

        assert!(to_entity_detail(&QueryResult::default()).is_none());
    }

    #[test]
    fn shapes_relation_detail() {
        let row = json_row(vec![
            ("id", json!(10)),
            ("type", json!("OWNS")),
            ("from", json!(1)),
            ("to", json!(2)),
            ("from_labels", json!(["Person"])),
            ("from_props", json!({"name":"Elena"})),
            ("to_labels", json!(["Company"])),
            ("to_props", json!({"name":"Acme"})),
            ("props", json!({"share":80})),
        ]);
        let result = QueryResult {
            columns: vec![],
            rows: vec![row],
        };
        let detail = to_relation_detail(&result).expect("relation");
        assert_eq!(detail.rel, "OWNS");
        assert_eq!(detail.from.name.as_deref(), Some("Elena"));
        assert_eq!(detail.to.label, "Company");
        assert_eq!(detail.from.id, "1");
    }
}
