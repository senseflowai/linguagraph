//! Soft-merge resolver: dedupe `PrimaryKey::Soft` entities by vector
//! similarity *before* the standard MERGE planner runs.
//!
//! The knowledge-extraction pipeline emits entities without any stable
//! identifier — only a `type` and a free-text `name`. The graph
//! schema marks such entities with `PrimaryKey::Soft("name")`,
//! meaning "find a semantically equivalent existing node of the same
//! label, and merge with it if you find one; otherwise create a new
//! node keyed by `name`".
//!
//! This module is the synchronous "find" half of that contract. It
//! runs once per ingest, embeds every soft entity's primary-key
//! property in one batch, then issues one Cypher round-trip per
//! label that consults `libqlink.search_labeled` against the same
//! Qdrant collection the `SemanticText` handler writes into. Hits at
//! or above the configured similarity threshold cause the soft
//! entity's property to be rewritten to the canonical value held by
//! the existing node — when the planner then emits its `MERGE`, the
//! two collapse into one.
//!
//! The "store" half is unchanged: the `SemanticText` side effect
//! runs after the MERGE and upserts the (now canonical) embedding.
//!
//! Failure is loud by design. Soft-merge without an embedder, or
//! without a working `GraphClient`, is treated as a configuration
//! error rather than silently regressing to exact-string MERGE —
//! callers who don't want similarity merging should not set
//! `PrimaryKey::Soft` in the first place.

use std::collections::BTreeMap;

use serde_json::Value;

use crate::ast::query::Literal;
use crate::builder::CypherQuery;
use crate::config::SoftMergeConfig;
use crate::db::{GraphClient, Value as DbValue};
use crate::embeddings::Embedder;
use crate::graph::{Graph, PrimaryKey};
use crate::ingest::IngestError;
use crate::types::handlers::semantic_text;

/// Per-call telemetry. Cheap to construct; surfaced to callers (and to
/// tests) that want to assert "this many entities got rewritten".
#[derive(Debug, Clone, Default)]
pub struct SoftMergeReport {
    /// Number of soft-merge candidates considered (entities with
    /// `PrimaryKey::Soft` and a non-empty key property).
    pub candidates: usize,
    /// Number of candidates that were rewritten to a canonical value.
    pub rewrites: usize,
}

/// Resolve `PrimaryKey::Soft` entities in `graph` against the existing
/// graph. Returns a report; mutates `graph` in place so the standard
/// ingest path sees the rewritten primary-key values.
pub async fn resolve_soft_keys(
    graph: &mut Graph,
    embedder: &dyn Embedder,
    client: &dyn GraphClient,
    cfg: &SoftMergeConfig,
    semantic_collection_base: &str,
    prefix_index: Option<&str>,
) -> Result<SoftMergeReport, IngestError> {
    let candidates = collect_candidates(graph)?;
    if candidates.is_empty() {
        return Ok(SoftMergeReport::default());
    }

    let texts: Vec<&str> = candidates.iter().map(|c| c.text.as_str()).collect();
    let vectors = embedder
        .embed_batch(&texts)
        .map_err(|e| IngestError::SoftMerge(format!("embed_batch: {e}")))?;
    if vectors.len() != candidates.len() {
        return Err(IngestError::SoftMerge(format!(
            "embedder returned {} vectors for {} inputs",
            vectors.len(),
            candidates.len()
        )));
    }

    // Group by (label, field) — every row in a group hits the same
    // Qdrant collection and can be UNWIND-ed into a single Cypher call.
    type GroupKey = (String, String);
    let mut groups: BTreeMap<GroupKey, Vec<EmbeddedCandidate>> = BTreeMap::new();
    for (cand, vec) in candidates.into_iter().zip(vectors.into_iter()) {
        groups
            .entry((cand.label.clone(), cand.field.clone()))
            .or_default()
            .push(EmbeddedCandidate {
                entity_index: cand.entity_index,
                vec,
            });
    }

    let mut report = SoftMergeReport::default();
    for ((label, field), group) in groups {
        report.candidates += group.len();
        let collection =
            semantic_text::with_prefix_index(prefix_index, &format!("{semantic_collection_base}__{field}"));

        let query = build_search_query(&collection, &label, &field, cfg, &group);
        let result = client
            .execute(&query)
            .await
            .map_err(|e| IngestError::SoftMerge(format!("client.execute: {e}")))?;

        for row in result.rows {
            let idx = match row.fields.get("idx") {
                Some(DbValue::Int(i)) => *i as usize,
                Some(other) => {
                    return Err(IngestError::SoftMerge(format!(
                        "soft-merge query returned non-integer idx: {other:?}"
                    )));
                }
                None => continue,
            };
            let canonical = match row.fields.get("canonical") {
                Some(DbValue::String(s)) => s.clone(),
                Some(DbValue::Null) | None => continue,
                Some(other) => {
                    return Err(IngestError::SoftMerge(format!(
                        "soft-merge query returned non-string canonical: {other:?}"
                    )));
                }
            };

            let entity = graph
                .entities_mut()
                .get_mut(idx)
                .ok_or_else(|| IngestError::SoftMerge(format!("entity idx {idx} out of bounds")))?;
            if let Some(prop) = entity.properties.get_mut(&field) {
                prop.value = Value::String(canonical);
                report.rewrites += 1;
            }
        }
    }

    Ok(report)
}

/// Did `graph` declare any soft-merge candidates? Cheap pre-check so
/// `Pipeline::ingest` can skip allocating the resolver path entirely
/// when there's nothing to do.
pub fn has_soft_merge_candidates(graph: &Graph) -> bool {
    graph
        .entities()
        .iter()
        .any(|e| matches!(e.primary_key, Some(PrimaryKey::Soft(_))))
}

#[derive(Debug)]
struct Candidate {
    entity_index: usize,
    label: String,
    field: String,
    text: String,
}

#[derive(Debug)]
struct EmbeddedCandidate {
    entity_index: usize,
    vec: Vec<f32>,
}

fn collect_candidates(graph: &Graph) -> Result<Vec<Candidate>, IngestError> {
    let mut out = Vec::new();
    for (idx, entity) in graph.entities().iter().enumerate() {
        let field = match &entity.primary_key {
            Some(PrimaryKey::Soft(f)) => f.clone(),
            _ => continue,
        };
        let property = entity.properties.get(&field).ok_or_else(|| {
            IngestError::MissingGraphPrimaryKeyValue {
                label: entity.r#type.clone(),
                field: field.clone(),
            }
        })?;
        let text = json_to_text(&property.value);
        if text.is_empty() {
            return Err(IngestError::MissingGraphPrimaryKeyValue {
                label: entity.r#type.clone(),
                field,
            });
        }
        out.push(Candidate {
            entity_index: idx,
            label: entity.r#type.clone(),
            field,
            text,
        });
    }
    Ok(out)
}

fn json_to_text(value: &Value) -> String {
    match value {
        Value::String(s) => s.clone(),
        Value::Null => String::new(),
        other => other.to_string(),
    }
}

/// Build a single Cypher round-trip that searches Qdrant for every
/// candidate in `group` and returns, for each hit at or above the
/// configured threshold, the candidate's index in `Graph::entities`
/// along with the canonical value of the soft-merge field on the
/// matched node. We use `id(n)` to join back to the Memgraph node
/// because `libqlink.search_labeled` yields Memgraph internal ids,
/// not domain keys.
fn build_search_query(
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
    params.insert(
        "threshold".into(),
        Literal::Float(cfg.similarity_threshold),
    );

    // `search_labeled` is the pure-KNN qlink procedure: it returns the
    // top-k neighbors of `row.vec` whose payload label matches
    // `$label`, ranked by cosine. We keep only the best hit per row
    // and gate it against the configured threshold; the property we
    // pull off the matched node is the canonical merge value.
    let text = format!(
        "UNWIND $rows AS row\n\
         CALL libqlink.search_labeled([$coll], row.vec, $top_k, $label) \
             YIELD id AS hit_id, score AS hit_score\n\
         WITH row, hit_id, hit_score\n\
         ORDER BY hit_score DESC\n\
         WITH row, collect({{id: hit_id, score: hit_score}})[0] AS best\n\
         WHERE best IS NOT NULL AND best.score >= $threshold\n\
         MATCH (n:{label}) WHERE id(n) = best.id\n\
         RETURN row.idx AS idx, n.{field} AS canonical",
    );

    CypherQuery::new(text, params)
}

/// Strict allow-list for Cypher identifiers that we splice into the
/// query string instead of binding as a parameter (labels and property
/// names can't be parameterized in Cypher). Anything malformed
/// falls back to the raw input — the worst case is a Cypher parse
/// error from Memgraph, never injection, because the planner upstream
/// already validates labels.
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
    use std::sync::Arc;

    use serde_json::json;

    use super::*;
    use crate::db::{result::Row, MockClient, QueryResult, Value as DbValue};
    use crate::embeddings::MockEmbedder;
    use crate::graph::{EntityGraph, GraphBuilder, PropertyType};

    fn cfg() -> SoftMergeConfig {
        SoftMergeConfig {
            similarity_threshold: 0.8,
            top_k: 1,
        }
    }

    fn entity_named(name: &str) -> EntityGraph {
        EntityGraph::new("LegalConcept")
            .soft_primary_key("name")
            .property("name", PropertyType::Text, name)
    }

    #[test]
    fn collect_candidates_picks_soft_entities_only() {
        let mut b = GraphBuilder::new();
        b.add_entity(entity_named("общественное согласие"));
        b.add_entity(
            EntityGraph::new("Person")
                .strict_primary_key("id")
                .property("id", PropertyType::String, "p1"),
        );
        let graph = b.build();

        let got = collect_candidates(&graph).unwrap();
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].entity_index, 0);
        assert_eq!(got[0].label, "LegalConcept");
        assert_eq!(got[0].field, "name");
    }

    #[test]
    fn collect_candidates_errors_when_soft_field_missing() {
        let mut b = GraphBuilder::new();
        // PrimaryKey::Soft("name") but no `name` property at all.
        b.add_entity(EntityGraph::new("LegalConcept").soft_primary_key("name"));
        let err = collect_candidates(&b.build()).unwrap_err();
        assert!(matches!(
            err,
            IngestError::MissingGraphPrimaryKeyValue { ref label, ref field }
                if label == "LegalConcept" && field == "name"
        ));
    }

    #[tokio::test]
    async fn resolver_rewrites_property_to_canonical_when_hit_above_threshold() {
        // The mock client returns one canonical row for entity idx=0.
        let client = Arc::new(MockClient::new());
        let mut canonical_row = Row::default();
        canonical_row
            .fields
            .insert("idx".into(), DbValue::Int(0));
        canonical_row.fields.insert(
            "canonical".into(),
            DbValue::String("общественное согласие".into()),
        );
        client.enqueue(QueryResult {
            columns: vec!["idx".into(), "canonical".into()],
            rows: vec![canonical_row],
        });

        let mut b = GraphBuilder::new();
        b.add_entity(entity_named("общественное соглас."));
        let mut graph = b.build();

        let embedder = MockEmbedder::new(8);
        let report = resolve_soft_keys(
            &mut graph,
            &embedder,
            client.as_ref(),
            &cfg(),
            "semantic_text",
            None,
        )
        .await
        .unwrap();

        assert_eq!(report.candidates, 1);
        assert_eq!(report.rewrites, 1);
        assert_eq!(
            graph.entities()[0].properties["name"].value,
            json!("общественное согласие")
        );
    }

    #[tokio::test]
    async fn resolver_leaves_property_when_no_hit_returned() {
        // Empty result set — no canonical row, no rewrite.
        let client = Arc::new(MockClient::new());
        client.enqueue(QueryResult::default());

        let mut b = GraphBuilder::new();
        b.add_entity(entity_named("уникальная сущность"));
        let mut graph = b.build();

        let embedder = MockEmbedder::new(8);
        let report = resolve_soft_keys(
            &mut graph,
            &embedder,
            client.as_ref(),
            &cfg(),
            "semantic_text",
            None,
        )
        .await
        .unwrap();

        assert_eq!(report.candidates, 1);
        assert_eq!(report.rewrites, 0);
        assert_eq!(
            graph.entities()[0].properties["name"].value,
            json!("уникальная сущность")
        );
    }

    #[tokio::test]
    async fn resolver_no_candidates_does_not_touch_client_or_embedder() {
        let client = Arc::new(MockClient::new());
        let mut b = GraphBuilder::new();
        b.add_entity(
            EntityGraph::new("Person")
                .strict_primary_key("id")
                .property("id", PropertyType::String, "p1"),
        );
        let mut graph = b.build();

        let embedder = MockEmbedder::new(8);
        let report = resolve_soft_keys(
            &mut graph,
            &embedder,
            client.as_ref(),
            &cfg(),
            "semantic_text",
            None,
        )
        .await
        .unwrap();

        assert_eq!(report.candidates, 0);
        assert_eq!(report.rewrites, 0);
        assert!(
            client.captured.lock().unwrap().is_empty(),
            "no candidates → no DB round-trip"
        );
    }

    #[tokio::test]
    async fn resolver_threads_prefix_index_through_collection_name() {
        // Capture the executed Cypher so we can assert on the
        // collection parameter actually sent to libqlink.
        let client = Arc::new(MockClient::new());
        client.enqueue(QueryResult::default());

        let mut b = GraphBuilder::new();
        b.add_entity(entity_named("общественное согласие"));
        let mut graph = b.build();

        let embedder = MockEmbedder::new(8);
        resolve_soft_keys(
            &mut graph,
            &embedder,
            client.as_ref(),
            &cfg(),
            "semantic_text",
            Some("Tenant1"),
        )
        .await
        .unwrap();

        let captured = client.captured.lock().unwrap();
        assert_eq!(captured.len(), 1);
        let coll = captured[0]
            .params
            .get("coll")
            .expect("coll param must be bound");
        assert_eq!(
            coll,
            &Literal::String("Tenant1__semantic_text__name".into()),
            "soft-merge collection must fold in the prefix_index"
        );
    }
}
