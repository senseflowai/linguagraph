//! Soft-merge resolver: dedupe `PrimaryKey::Soft` entities by a staged
//! decision pipeline that runs *before* the standard MERGE planner.
//!
//! For every soft candidate we:
//!   1. embed the primary-key text (one batch round-trip),
//!   2. in-batch dedup near-identical embeddings within the same ingest,
//!   3. retrieve top-K hits from Qdrant via `libqlink.search_labeled`,
//!   4. classify the candidate against its hits on multiple signals
//!      (top1 score, top1-top2 margin, lexical similarity vs name,
//!      close-candidate ambiguity, hard-conflict on disambiguating
//!      properties, type-only canonical guard),
//!   5. route to AutoMerge / NeedsReview / NoMerge.
//!
//! AutoMerge rewrites the entity's primary-key property to the existing
//! node's canonical value; the standard Cypher MERGE later folds the
//! two rows into one node. NeedsReview leaves the entity untouched
//! (so a new node will be created) and surfaces the candidate hit list
//! in `SoftMergeReport.review_candidates` for the caller to consume.
//! NoMerge is a silent skip.
//!
//! Failure is loud by design. Soft-merge without an embedder, or
//! without a working `GraphClient`, is treated as a configuration
//! error rather than silently regressing to exact-string MERGE —
//! callers who don't want similarity merging should not set
//! `PrimaryKey::Soft` in the first place.

mod candidates;
mod decision;
mod lexical;
mod query;

#[cfg(test)]
mod integration_tests;

use std::collections::BTreeMap;

use serde_json::Value;

use crate::config::SoftMergeConfig;
use crate::db::GraphClient;
use crate::embeddings::Embedder;
use crate::graph::{Graph, PrimaryKey};
use crate::ingest::IngestError;
use crate::types::handlers::semantic_text;

use candidates::{collect_candidates, deduplicate_in_batch, EmbeddedCandidate};
use decision::{classify, incoming_name, CandidateInfo, Decision};
use query::{build_search_query, field_as_i64, parse_hits};

pub use decision::{GateReason, ReviewHit};

/// Per-call telemetry. Cheap to construct; surfaced to callers (and to
/// tests) that want to inspect how the resolver routed each candidate.
#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct SoftMergeReport {
    /// Number of soft-merge candidates considered (entities with
    /// `PrimaryKey::Soft` and a non-empty key property). Counted
    /// post-in-batch-dedup, so duplicates collapsed onto an in-batch
    /// representative are excluded here — see
    /// `in_batch_dedup_collapsed`.
    pub candidates: usize,
    /// Number of candidates auto-merged onto an existing graph node
    /// (via Qdrant) by rewriting the primary-key property.
    pub auto_merges: usize,
    /// Total number of review records emitted (in-batch + against
    /// existing graph). A single entity can contribute up to two
    /// records if it triggers reviews in both stages, so this counter
    /// matches `review_candidates.len()` rather than "unique entities".
    pub needs_review: usize,
    /// Number of candidates with no Qdrant hit (or hits all below
    /// `review_threshold`). The entity passes through unchanged and
    /// the standard MERGE creates a new node.
    pub no_merge: usize,
    /// Number of in-batch candidates auto-merged onto another
    /// candidate in the same ingest. The rewritten primary-key
    /// property still flows through the standard MERGE — these are
    /// effectively in-flight auto-merges with no Qdrant round-trip.
    pub in_batch_dedup_collapsed: usize,
    /// Audit records for every NeedsReview decision, both in-batch
    /// and against the existing graph. Empty when
    /// `cfg.emit_review_candidates == false`. Each record carries a
    /// `source` field that distinguishes the two kinds.
    pub review_candidates: Vec<ReviewCandidate>,
}

/// Where a `ReviewCandidate`'s top hit came from. Lets audit
/// consumers tell apart "two in-flight LLM extractions look like
/// duplicates" from "this extraction looks similar to a pre-existing
/// graph node" — the two cases usually have different remediation
/// paths.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ReviewSource {
    /// Match against another in-flight candidate in the same ingest.
    /// `top.hit_id` is its index in `Graph::entities`.
    InBatch,
    /// Match against a pre-existing graph node via Qdrant.
    /// `top.hit_id` is the Memgraph internal node id of the match.
    Existing,
}

/// Audit record for a candidate that passed the consideration floor
/// but failed the AutoMerge gates. Serialisable so callers can log /
/// persist it through whatever audit pipeline they own.
#[derive(Debug, Clone, serde::Serialize)]
pub struct ReviewCandidate {
    /// Index into `Graph::entities` — lets a caller cross-reference
    /// the original entity without re-walking the graph.
    pub entity_index: usize,
    pub label: String,
    pub field: String,
    /// The canonical (or raw) key text of the incoming entity.
    pub incoming_value: String,
    /// Whether the match was against another in-flight candidate
    /// (in-batch) or a pre-existing graph node (existing). See
    /// `ReviewSource` for the semantics.
    pub source: ReviewSource,
    /// Top hit by embedding score.
    pub top: ReviewHit,
    /// Runner-up hits, capped at `cfg.review_max_candidates - 1`.
    pub runners_up: Vec<ReviewHit>,
    /// All gates that rejected the AutoMerge decision, in evaluation
    /// order. Each variant carries the numbers that drove the
    /// rejection so reviewers can tune thresholds without re-running.
    pub rejected_by: Vec<GateReason>,
}

/// Resolve `PrimaryKey::Soft` entities in `graph` against the existing
/// graph. Returns a report; mutates `graph` in place so the standard
/// ingest path sees the rewritten primary-key values for any
/// candidate routed to AutoMerge.
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

    // Group by (label, field) so every row in a group hits the same
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
    for ((label, field), mut group) in groups {
        // In-batch dedup: run the same staged decision pipeline we
        // run against Qdrant, but with the other in-flight
        // candidates as the "hits". AutoMerge collapses the
        // duplicate onto a representative; NeedsReview keeps both
        // entities and emits an InBatch-flavoured review record;
        // NoMerge promotes the candidate to a new representative.
        let in_batch = deduplicate_in_batch(graph, &mut group, &label, &field, cfg)?;
        report.in_batch_dedup_collapsed += in_batch.collapsed;
        report.needs_review += in_batch.needs_review;
        report.review_candidates.extend(in_batch.review_candidates);
        report.candidates += group.len();

        let collection = semantic_text::with_prefix_index(
            prefix_index,
            &format!("{semantic_collection_base}__{field}"),
        );

        let query = build_search_query(&collection, &label, &field, cfg, &group);
        let result = client
            .execute(&query)
            .await
            .map_err(|e| IngestError::SoftMerge(format!("client.execute: {e}")))?;

        // Map idx → hits for the rows the query returned. Candidates
        // without any row (no hit above the consideration floor) land
        // in NoMerge.
        let mut idx_to_hits: BTreeMap<usize, Vec<query::Hit>> = BTreeMap::new();
        for row in result.rows {
            let Some(idx) = row.fields.get("idx").and_then(field_as_i64) else {
                if let Some(other) = row.fields.get("idx") {
                    return Err(IngestError::SoftMerge(format!(
                        "soft-merge query returned non-integer idx: {other:?}"
                    )));
                }
                continue;
            };
            let hits = match row.fields.get("hits") {
                Some(cell) => parse_hits(cell)?,
                None => continue,
            };
            idx_to_hits.insert(idx as usize, hits);
        }

        for emb in &group {
            let entity_index = emb.entity_index;
            // Take ownership of the cached entity facts we need; the
            // mutable borrow during AutoMerge requires we not hold an
            // immutable borrow at the same time.
            let (canonical_text, props_snapshot, entity_label, entity_field) = {
                let entity = graph.entities().get(entity_index).ok_or_else(|| {
                    IngestError::SoftMerge(format!("entity idx {entity_index} out of bounds"))
                })?;
                let text = entity
                    .properties
                    .get(&field)
                    .map(|p| match &p.value {
                        Value::String(s) => s.clone(),
                        other => other.to_string(),
                    })
                    .unwrap_or_default();
                (
                    text,
                    entity.properties.clone(),
                    entity.r#type.clone(),
                    field.clone(),
                )
            };

            let hits = idx_to_hits.remove(&entity_index).unwrap_or_default();
            let info = CandidateInfo {
                canonical_text: &canonical_text,
                props: &props_snapshot,
            };
            let decision = classify(&info, &hits, cfg);
            match decision {
                Decision::AutoMerge {
                    canonical,
                    hit_id,
                    top_score,
                } => {
                    tracing::debug!(
                        target: "linguagraph::soft_merge",
                        entity_index,
                        label = %entity_label,
                        field = %entity_field,
                        hit_id,
                        top_score,
                        canonical = %canonical,
                        "auto-merge: rewriting soft key to existing canonical value",
                    );
                    let entity = graph.entities_mut().get_mut(entity_index).ok_or_else(|| {
                        IngestError::SoftMerge(format!(
                            "entity idx {entity_index} out of bounds at rewrite"
                        ))
                    })?;
                    if let Some(prop) = entity.properties.get_mut(&entity_field) {
                        prop.value = Value::String(canonical);
                        report.auto_merges += 1;
                    }
                }
                Decision::NeedsReview {
                    top,
                    runners_up,
                    rejected_by,
                } => {
                    report.needs_review += 1;
                    if cfg.emit_review_candidates {
                        report.review_candidates.push(ReviewCandidate {
                            entity_index,
                            label: entity_label,
                            field: entity_field,
                            incoming_value: incoming_name(&canonical_text),
                            source: ReviewSource::Existing,
                            top,
                            runners_up,
                            rejected_by,
                        });
                    }
                }
                Decision::NoMerge => {
                    report.no_merge += 1;
                }
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
        .any(|e| matches!(e.primary_key, Some(PrimaryKey::Soft)))
}
