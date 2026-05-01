//! Live schema introspection.
//!
//! The function in this module asks a [`GraphClient`] enough questions to
//! reconstruct a [`GraphSchema`] suitable for prompt generation. It is
//! deliberately driver-agnostic — every query goes through the trait — so
//! it can be exercised against the [`MockClient`](super::MockClient) or
//! any future backend without depending on `neo4rs`.
//!
//! The introspection uses portable Cypher (no Memgraph- or Neo4j-specific
//! procedures) so it works on Memgraph community as well as enterprise.
//! Property types are *inferred* from a small per-label sample, not
//! discovered from a registry.

use std::collections::BTreeMap;

use serde_json::Value as Json;

use crate::ast::query::Literal;
use crate::builder::CypherQuery;
use crate::prompt::{GraphSchema, NodeKind, Property, PropertyType, RelKind};

use super::result::Value;
use super::{DbError, GraphClient};

/// Knobs for [`introspect_schema`].
#[derive(Debug, Clone, Copy)]
pub struct IntrospectOptions {
    /// Maximum number of nodes / relationships sampled per type when
    /// inferring property types. The cost of introspection scales with
    /// this; 100 is enough to identify primitive types reliably.
    pub sample_size: u64,
}

impl Default for IntrospectOptions {
    fn default() -> Self {
        Self { sample_size: 100 }
    }
}

/// Build a [`GraphSchema`] by querying `client`.
///
/// Issues a small fixed sequence of parameterised Cypher queries:
///
/// 1. distinct node labels,
/// 2. for each label, sampled keys + values for type inference,
/// 3. distinct relationship types,
/// 4. for each rel type, distinct `(from_label, to_label)` endpoints,
/// 5. for each rel type, sampled keys + values.
///
/// The result is deterministic up to the order in which the database
/// returns rows; downstream consumers (the prompt generator) sort what
/// they need to.
pub async fn introspect_schema(
    client: &dyn GraphClient,
    opts: IntrospectOptions,
) -> Result<GraphSchema, DbError> {
    let labels = fetch_node_labels(client).await?;
    let mut nodes = Vec::with_capacity(labels.len());
    for label in &labels {
        let properties = fetch_props(
            client,
            "MATCH (n) WHERE $label IN labels(n) \
             WITH n LIMIT $sample_size \
             UNWIND keys(n) AS k \
             WITH k, n[k] AS v \
             RETURN k AS key, collect(v) AS samples",
            Some(("label", Literal::String(label.clone()))),
            opts.sample_size,
        )
        .await?;
        nodes.push(NodeKind { label: label.clone(), properties });
    }

    let rel_types = fetch_rel_types(client).await?;
    let mut relationships = Vec::new();
    for rt in &rel_types {
        let endpoints = fetch_rel_endpoints(client, rt).await?;
        let properties = fetch_props(
            client,
            "MATCH ()-[r]->() WHERE type(r) = $type \
             WITH r LIMIT $sample_size \
             UNWIND keys(r) AS k \
             WITH k, r[k] AS v \
             RETURN k AS key, collect(v) AS samples",
            Some(("type", Literal::String(rt.clone()))),
            opts.sample_size,
        )
        .await?;
        if endpoints.is_empty() {
            // Rel type exists but no endpoints could be resolved — emit
            // a single edge with `from`/`to` left unset rather than
            // dropping the type.
            relationships.push(RelKind {
                label: rt.clone(),
                from: None,
                to: None,
                properties,
            });
        } else {
            for (from, to) in endpoints {
                relationships.push(RelKind {
                    label: rt.clone(),
                    from: Some(from),
                    to: Some(to),
                    properties: properties.clone(),
                });
            }
        }
    }

    Ok(GraphSchema { nodes, relationships })
}

// ─── Query helpers ──────────────────────────────────────────────────────────

async fn fetch_node_labels(client: &dyn GraphClient) -> Result<Vec<String>, DbError> {
    let q = cypher(
        "MATCH (n) UNWIND labels(n) AS label RETURN DISTINCT label",
        BTreeMap::new(),
    );
    let res = client.execute(&q).await?;
    let mut out: Vec<String> = res
        .rows
        .iter()
        .filter_map(|r| r.fields.get("label").and_then(value_as_string))
        .collect();
    out.sort();
    out.dedup();
    Ok(out)
}

async fn fetch_rel_types(client: &dyn GraphClient) -> Result<Vec<String>, DbError> {
    let q = cypher(
        "MATCH ()-[r]->() RETURN DISTINCT type(r) AS type",
        BTreeMap::new(),
    );
    let res = client.execute(&q).await?;
    let mut out: Vec<String> = res
        .rows
        .iter()
        .filter_map(|r| r.fields.get("type").and_then(value_as_string))
        .collect();
    out.sort();
    out.dedup();
    Ok(out)
}

async fn fetch_rel_endpoints(
    client: &dyn GraphClient,
    rel_type: &str,
) -> Result<Vec<(String, String)>, DbError> {
    let mut params = BTreeMap::new();
    params.insert("type".to_string(), Literal::String(rel_type.to_string()));
    let q = cypher(
        "MATCH (a)-[r]->(b) WHERE type(r) = $type \
         RETURN DISTINCT labels(a)[0] AS from, labels(b)[0] AS to",
        params,
    );
    let res = client.execute(&q).await?;
    let mut out: Vec<(String, String)> = res
        .rows
        .iter()
        .filter_map(|r| {
            let from = r.fields.get("from").and_then(value_as_string)?;
            let to = r.fields.get("to").and_then(value_as_string)?;
            Some((from, to))
        })
        .collect();
    out.sort();
    out.dedup();
    Ok(out)
}

async fn fetch_props(
    client: &dyn GraphClient,
    query: &str,
    extra: Option<(&str, Literal)>,
    sample_size: u64,
) -> Result<Vec<Property>, DbError> {
    let mut params = BTreeMap::new();
    params.insert(
        "sample_size".to_string(),
        Literal::Int(sample_size.try_into().unwrap_or(i64::MAX)),
    );
    if let Some((k, v)) = extra {
        params.insert(k.to_string(), v);
    }
    let q = cypher(query, params);
    let res = client.execute(&q).await?;

    let mut props: Vec<Property> = res
        .rows
        .iter()
        .filter_map(|r| {
            let key = r.fields.get("key").and_then(value_as_string)?;
            let samples = r
                .fields
                .get("samples")
                .map(value_as_json)
                .unwrap_or(Json::Null);
            let sample_vec: Vec<Json> = match samples {
                Json::Array(a) => a,
                other => vec![other],
            };
            Some(Property { name: key, ty: infer_type(&sample_vec) })
        })
        .collect();
    props.sort_by(|a, b| a.name.cmp(&b.name));
    props.dedup_by(|a, b| a.name == b.name);
    Ok(props)
}

fn cypher(text: &str, params: BTreeMap<String, Literal>) -> CypherQuery {
    CypherQuery::new(text.to_string(), params)
}

fn value_as_string(v: &Value) -> Option<String> {
    match v {
        Value::String(s) => Some(s.clone()),
        Value::Json(Json::String(s)) => Some(s.clone()),
        _ => None,
    }
}

fn value_as_json(v: &Value) -> Json {
    match v {
        Value::Null => Json::Null,
        Value::Bool(b) => Json::Bool(*b),
        Value::Int(i) => Json::from(*i),
        Value::Float(f) => serde_json::Number::from_f64(*f)
            .map(Json::Number)
            .unwrap_or(Json::Null),
        Value::String(s) => Json::String(s.clone()),
        Value::Json(j) => j.clone(),
    }
}

/// Pick the most specific [`PropertyType`] consistent with every non-null
/// sample. The order of fall-throughs matters:
///
/// * a sample of pure ints → `Int`
/// * any float forces `Float` (Memgraph stores numerics as `Float|Int`)
/// * strings that all look like ISO datetimes/dates promote to
///   `Datetime`/`Date` — purely a hint for the LLM, since Memgraph
///   itself stores them as strings
/// * heterogeneous samples fall back to `String`.
pub fn infer_type(samples: &[Json]) -> PropertyType {
    let mut seen_string = false;
    let mut seen_int = false;
    let mut seen_float = false;
    let mut seen_bool = false;
    let mut seen_list = false;
    let mut seen_other = false;
    let mut all_strings_look_like_datetime = true;
    let mut all_strings_look_like_date = true;
    let mut any_non_null = false;

    for v in samples {
        match v {
            Json::Null => {}
            Json::Bool(_) => {
                any_non_null = true;
                seen_bool = true;
            }
            Json::Number(n) => {
                any_non_null = true;
                if n.is_i64() || n.is_u64() {
                    seen_int = true;
                } else {
                    seen_float = true;
                }
            }
            Json::String(s) => {
                any_non_null = true;
                seen_string = true;
                if !looks_like_datetime(s) {
                    all_strings_look_like_datetime = false;
                }
                if !looks_like_date(s) {
                    all_strings_look_like_date = false;
                }
            }
            Json::Array(_) => {
                any_non_null = true;
                seen_list = true;
            }
            Json::Object(_) => {
                any_non_null = true;
                seen_other = true;
            }
        }
    }

    if !any_non_null {
        return PropertyType::String;
    }

    let only = (seen_string as u8)
        + (seen_int as u8)
        + (seen_float as u8)
        + (seen_bool as u8)
        + (seen_list as u8)
        + (seen_other as u8);
    match only {
        1 => {
            if seen_int {
                PropertyType::Int
            } else if seen_float {
                PropertyType::Float
            } else if seen_bool {
                PropertyType::Bool
            } else if seen_list {
                PropertyType::List
            } else if seen_string {
                if all_strings_look_like_datetime {
                    PropertyType::Datetime
                } else if all_strings_look_like_date {
                    PropertyType::Date
                } else {
                    PropertyType::String
                }
            } else {
                PropertyType::String
            }
        }
        _ => {
            // Numeric mix → Float. Anything else mixed → String.
            if seen_int && seen_float && !seen_string && !seen_bool && !seen_list && !seen_other {
                PropertyType::Float
            } else {
                PropertyType::String
            }
        }
    }
}

fn looks_like_date(s: &str) -> bool {
    // Cheap shape check: YYYY-MM-DD, exactly 10 chars.
    let b = s.as_bytes();
    b.len() == 10
        && b[0..4].iter().all(|c| c.is_ascii_digit())
        && b[4] == b'-'
        && b[5..7].iter().all(|c| c.is_ascii_digit())
        && b[7] == b'-'
        && b[8..10].iter().all(|c| c.is_ascii_digit())
}

fn looks_like_datetime(s: &str) -> bool {
    // YYYY-MM-DDTHH:MM:SS… — must have a T at offset 10 and a digit pair
    // after it. Anything beyond second-level precision (millis, tz) is
    // tolerated.
    let b = s.as_bytes();
    b.len() >= 19
        && b[..10].iter().all(|c| c.is_ascii()) // optional safety
        && looks_like_date(&s[..10])
        && b[10] == b'T'
        && b[11..13].iter().all(|c| c.is_ascii_digit())
        && b[13] == b':'
        && b[14..16].iter().all(|c| c.is_ascii_digit())
        && b[16] == b':'
        && b[17..19].iter().all(|c| c.is_ascii_digit())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::result::{Row, Value};
    use crate::db::MockClient;
    use serde_json::json;

    fn row(pairs: &[(&str, Value)]) -> Row {
        let mut r = Row::default();
        for (k, v) in pairs {
            r.fields.insert((*k).to_string(), v.clone());
        }
        r
    }

    #[tokio::test]
    async fn introspect_against_mock_client() {
        let mock = MockClient::new();

        // Order matters: MockClient pops results LIFO, so enqueue in
        // *reverse* of the call order.
        // Call order:
        //   1. fetch_node_labels
        //   2. fetch_props for "Person"
        //   3. fetch_rel_types
        //   4. fetch_rel_endpoints for "KNOWS"
        //   5. fetch_props for "KNOWS"

        // 5. KNOWS rel props (empty)
        mock.enqueue(crate::db::QueryResult::default());
        // 4. KNOWS endpoints
        mock.enqueue(crate::db::QueryResult {
            columns: vec!["from".into(), "to".into()],
            rows: vec![row(&[
                ("from", Value::String("Person".into())),
                ("to", Value::String("Person".into())),
            ])],
        });
        // 3. rel types
        mock.enqueue(crate::db::QueryResult {
            columns: vec!["type".into()],
            rows: vec![row(&[("type", Value::String("KNOWS".into()))])],
        });
        // 2. Person props
        mock.enqueue(crate::db::QueryResult {
            columns: vec!["key".into(), "samples".into()],
            rows: vec![
                row(&[
                    ("key", Value::String("name".into())),
                    ("samples", Value::Json(json!(["Ada", "Bob"]))),
                ]),
                row(&[
                    ("key", Value::String("age".into())),
                    ("samples", Value::Json(json!([30, 40, 50]))),
                ]),
            ],
        });
        // 1. node labels
        mock.enqueue(crate::db::QueryResult {
            columns: vec!["label".into()],
            rows: vec![row(&[("label", Value::String("Person".into()))])],
        });

        let schema = introspect_schema(&mock, IntrospectOptions::default())
            .await
            .unwrap();

        assert_eq!(schema.nodes.len(), 1);
        assert_eq!(schema.nodes[0].label, "Person");
        assert_eq!(schema.nodes[0].properties.len(), 2);
        // Sorted alphabetically: age, name.
        assert_eq!(schema.nodes[0].properties[0].name, "age");
        assert_eq!(schema.nodes[0].properties[0].ty, PropertyType::Int);
        assert_eq!(schema.nodes[0].properties[1].name, "name");
        assert_eq!(schema.nodes[0].properties[1].ty, PropertyType::String);

        assert_eq!(schema.relationships.len(), 1);
        assert_eq!(schema.relationships[0].label, "KNOWS");
        assert_eq!(schema.relationships[0].from.as_deref(), Some("Person"));
        assert_eq!(schema.relationships[0].to.as_deref(), Some("Person"));
    }

    #[test]
    fn type_inference_picks_int_for_integer_samples() {
        assert_eq!(infer_type(&[json!(1), json!(2)]), PropertyType::Int);
    }

    #[test]
    fn type_inference_promotes_mixed_numerics_to_float() {
        assert_eq!(infer_type(&[json!(1), json!(2.5)]), PropertyType::Float);
    }

    #[test]
    fn type_inference_detects_datetime_strings() {
        let s = json!("2026-04-01T10:00:00Z");
        assert_eq!(infer_type(&[s]), PropertyType::Datetime);
    }

    #[test]
    fn type_inference_detects_date_strings() {
        let s = json!("2026-04-01");
        assert_eq!(infer_type(&[s]), PropertyType::Date);
    }

    #[test]
    fn type_inference_falls_back_to_string_on_mixed_kinds() {
        assert_eq!(
            infer_type(&[json!(1), json!("oops")]),
            PropertyType::String
        );
    }

    #[test]
    fn type_inference_handles_only_nulls() {
        assert_eq!(infer_type(&[json!(null)]), PropertyType::String);
    }

    #[test]
    fn type_inference_picks_list() {
        assert_eq!(
            infer_type(&[json!([1, 2]), json!([])]),
            PropertyType::List
        );
    }
}
