//! Cypher round-trip for soft-merge candidate retrieval.
//!
//! One query per `(label, field)` group: for every embedded candidate
//! row, retrieve up to `top_k` hits at or above `similarity_threshold`,
//! sorted by score descending, with each hit carrying its canonical
//! field value AND the full property map of the matched node so the
//! downstream decision pipeline can compute lexical similarity and
//! detect hard conflicts.

use std::collections::BTreeMap;

use serde_json::{Map as JsonMap, Value as JsonValue};

use crate::ast::query::Literal;
use crate::builder::CypherQuery;
use crate::config::SoftMergeConfig;
use crate::db::Value as DbValue;
use crate::ingest::IngestError;

use super::candidates::EmbeddedCandidate;

/// One candidate hit returned by `libqlink.search_labeled`, joined back
/// to its Memgraph node properties. The decision pipeline consumes a
/// `Vec<Hit>` per incoming entity.
#[derive(Debug, Clone)]
pub(super) struct Hit {
    pub id: i64,
    pub score: f64,
    pub canonical: String,
    pub props: JsonMap<String, JsonValue>,
}

/// Build the per-group Cypher. The shape of the returned rows is:
///   `idx: i64` — the index into `Graph::entities`
///   `hits: List<Map>` — each map has `id, score, canonical, props`,
///                       flattened to `DbValue::Json(Array(...))` by
///                       the driver.
pub(super) fn build_search_query(
    collection: &str,
    label: &str,
    field: &str,
    cfg: &SoftMergeConfig,
    group: &[EmbeddedCandidate],
) -> CypherQuery {
    let label = sanitize_ident(label);
    let field = sanitize_ident(field);

    let rows: Vec<Literal> = group
        .iter()
        .map(|c| {
            let mut row: BTreeMap<String, Literal> = BTreeMap::new();
            row.insert("idx".into(), Literal::Int(c.entity_index as i64));
            row.insert(
                "vec".into(),
                Literal::List(c.vec.iter().map(|f| Literal::Float(*f as f64)).collect()),
            );
            Literal::Object(row)
        })
        .collect();

    let mut params: BTreeMap<String, Literal> = BTreeMap::new();
    params.insert("coll".into(), Literal::String(collection.to_string()));
    params.insert("label".into(), Literal::String(label.clone()));
    params.insert("rows".into(), Literal::List(rows));
    params.insert("top_k".into(), Literal::Int(cfg.top_k as i64));
    params.insert("threshold".into(), Literal::Float(cfg.similarity_threshold));

    // The query keeps every hit above `$threshold` (the consideration
    // floor) and folds them into one `hits` list per candidate sorted
    // by score descending. Each hit carries the matched node's
    // canonical field value and full property map so the resolver can
    // compute lexical similarity and detect disambiguating-property
    // conflicts without a follow-up round-trip.
    let text = format!(
        "UNWIND $rows AS row\n\
         CALL libqlink.search_labeled([$coll], row.vec, $top_k, $label) \
             YIELD id AS hit_id, score AS hit_score\n\
         WITH row, hit_id, hit_score\n\
         WHERE hit_score >= $threshold\n\
         MATCH (n:{label}) WHERE id(n) = hit_id\n\
         WITH row, hit_score, hit_id, n.{field} AS canonical, properties(n) AS props\n\
         ORDER BY hit_score DESC\n\
         WITH row.idx AS idx, \
              collect({{id: hit_id, score: hit_score, canonical: canonical, props: props}}) AS hits\n\
         RETURN idx, hits",
    );

    CypherQuery::new(text, params)
}

/// Parse a row-cell that holds a list of hit maps. The driver
/// flattens nested Memgraph values to `DbValue::Json`, so a `hits`
/// cell arrives as `DbValue::Json(Array(...))` in both production and
/// `MockClient` tests that mimic the shape. Null/missing cells decode
/// to an empty list.
pub(super) fn parse_hits(field: &DbValue) -> Result<Vec<Hit>, IngestError> {
    match field {
        DbValue::Null => Ok(Vec::new()),
        DbValue::Json(JsonValue::Null) => Ok(Vec::new()),
        DbValue::Json(JsonValue::Array(items)) => items.iter().map(parse_hit_json).collect(),
        other => Err(IngestError::SoftMerge(format!(
            "soft-merge query returned non-list hits: {other:?}"
        ))),
    }
}

fn parse_hit_json(value: &JsonValue) -> Result<Hit, IngestError> {
    let JsonValue::Object(map) = value else {
        return Err(IngestError::SoftMerge(format!(
            "soft-merge hit must be an object, got: {value:?}"
        )));
    };
    let id = map
        .get("id")
        .and_then(JsonValue::as_i64)
        .ok_or_else(|| IngestError::SoftMerge(format!("hit missing/non-int `id`: {map:?}")))?;
    let score = map
        .get("score")
        .and_then(JsonValue::as_f64)
        .ok_or_else(|| IngestError::SoftMerge(format!("hit missing/non-float `score`: {map:?}")))?;
    let canonical = match map.get("canonical") {
        Some(JsonValue::String(s)) => s.clone(),
        Some(JsonValue::Null) | None => String::new(),
        Some(other) => other.to_string(),
    };
    let props = match map.get("props") {
        Some(JsonValue::Object(o)) => o.clone(),
        Some(JsonValue::Null) | None => JsonMap::new(),
        Some(other) => {
            return Err(IngestError::SoftMerge(format!(
                "hit `props` must be a map, got: {other:?}"
            )))
        }
    };
    Ok(Hit {
        id,
        score,
        canonical,
        props,
    })
}

/// Read a result-row cell as an `i64`, tolerant of both the native
/// `DbValue::Int` form (used by `MockClient` and tests) and the
/// `DbValue::Json(Number(...))` form Memgraph's neo4rs driver
/// produces for every scalar. Returns `None` for nulls or any
/// non-numeric shape — callers decide whether that's a soft skip or
/// a hard error.
pub(super) fn field_as_i64(v: &DbValue) -> Option<i64> {
    match v {
        DbValue::Int(i) => Some(*i),
        DbValue::Float(f) if f.fract() == 0.0 => Some(*f as i64),
        DbValue::Json(JsonValue::Number(n)) => n.as_i64(),
        _ => None,
    }
}

/// Strict allow-list for Cypher identifiers that we splice into the
/// query string instead of binding as a parameter (labels and property
/// names can't be parameterized in Cypher). Anything malformed
/// falls back to a filtered version — the worst case is a Cypher
/// parse error from Memgraph, never injection, because the planner
/// upstream already validates labels.
fn sanitize_ident(s: &str) -> String {
    let mut chars = s.chars();
    let first = chars.next();
    let valid = matches!(first, Some(c) if c.is_ascii_alphabetic() || c == '_')
        && chars.all(|c| c.is_ascii_alphanumeric() || c == '_');
    if valid {
        s.to_string()
    } else {
        s.chars()
            .filter(|c| c.is_ascii_alphanumeric() || *c == '_')
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::SoftMergeConfig;

    fn cfg() -> SoftMergeConfig {
        SoftMergeConfig {
            top_k: 10,
            similarity_threshold: 0.85,
            ..SoftMergeConfig::default()
        }
    }

    #[test]
    fn build_search_query_returns_top_k_with_properties() {
        let group = vec![EmbeddedCandidate {
            entity_index: 7,
            vec: vec![1.0, 0.0],
        }];
        let q = build_search_query("c1__name", "LegalConcept", "name", &cfg(), &group);
        assert!(
            q.text.contains("properties(n) AS props"),
            "Cypher must pull full props: {}",
            q.text
        );
        assert!(
            q.text.contains("collect({id: hit_id"),
            "Cypher must collect hits as a list, not pick [0]: {}",
            q.text
        );
        assert!(
            !q.text.contains(")[0] AS best"),
            "Cypher must not single-pick best hit: {}",
            q.text
        );
        assert!(
            q.text.contains("WHERE hit_score >= $threshold"),
            "Cypher must gate hits at the threshold: {}",
            q.text
        );
    }

    #[test]
    fn build_search_query_binds_top_k_and_threshold() {
        let group = vec![EmbeddedCandidate {
            entity_index: 0,
            vec: vec![1.0],
        }];
        let q = build_search_query("c1", "Label", "name", &cfg(), &group);
        assert_eq!(q.params.get("top_k"), Some(&Literal::Int(10)));
        assert_eq!(q.params.get("threshold"), Some(&Literal::Float(0.85)));
    }

    #[test]
    fn parse_hits_accepts_json_array() {
        let raw = DbValue::Json(serde_json::json!([
            {"id": 7, "score": 0.95, "canonical": "Microsoft",
             "props": {"name": "Microsoft", "country": "US"}},
            {"id": 9, "score": 0.85, "canonical": "Microsoft Inc",
             "props": {"name": "Microsoft Inc"}}
        ]));
        let hits = parse_hits(&raw).unwrap();
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].id, 7);
        assert!((hits[0].score - 0.95).abs() < 1e-9);
        assert_eq!(hits[0].canonical, "Microsoft");
        assert_eq!(hits[0].props.get("country"), Some(&serde_json::json!("US")));
        assert_eq!(hits[1].id, 9);
    }

    #[test]
    fn parse_hits_returns_empty_for_null() {
        let hits = parse_hits(&DbValue::Null).unwrap();
        assert!(hits.is_empty());
        let hits = parse_hits(&DbValue::Json(JsonValue::Null)).unwrap();
        assert!(hits.is_empty());
    }

    #[test]
    fn parse_hits_rejects_non_list() {
        let err = parse_hits(&DbValue::String("oops".into())).unwrap_err();
        assert!(matches!(err, IngestError::SoftMerge(_)));
    }

    #[test]
    fn parse_hits_rejects_malformed_hit_missing_canonical_is_ok() {
        // Missing canonical → empty string, missing props → empty map.
        // The decision pipeline can still reason about score; absent
        // canonical just yields a 0.0 lexical score.
        let raw = DbValue::Json(serde_json::json!([
            {"id": 1, "score": 0.9}
        ]));
        let hits = parse_hits(&raw).unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].canonical, "");
        assert!(hits[0].props.is_empty());
    }

    #[test]
    fn parse_hits_rejects_missing_id() {
        let raw = DbValue::Json(serde_json::json!([{"score": 0.9}]));
        let err = parse_hits(&raw).unwrap_err();
        assert!(matches!(err, IngestError::SoftMerge(_)));
    }
}
