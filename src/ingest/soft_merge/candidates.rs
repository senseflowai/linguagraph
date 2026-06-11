//! Soft-merge candidate collection + in-batch staged-decision dedup.
//!

use std::collections::{BTreeSet, HashMap};

use serde_json::{Map as JsonMap, Value};

use crate::config::SoftMergeConfig;
use crate::embeddings::cosine_similarity;
use crate::graph::{Graph, PrimaryKey, Property, CANONICAL_FIELD};

use crate::ingest::IngestError;

use super::decision::{classify, incoming_name, CandidateInfo, Decision};
use super::query::Hit;
use super::{ReviewCandidate, ReviewSource};

#[derive(Debug)]
pub(super) struct Candidate {
    pub entity_index: usize,
    pub label: String,
    pub field: String,
    pub text: String,
}

#[derive(Debug)]
pub(super) struct EmbeddedCandidate {
    pub entity_index: usize,
    pub vec: Vec<f32>,
}

/// What `deduplicate_in_batch` produced for one group. Mirrors the
/// counters in `SoftMergeReport` but scoped to a single
/// `(label, field)` group so the orchestrator can accumulate.
#[derive(Debug, Default)]
pub(super) struct InBatchDedupOutcome {
    /// Number of in-batch candidates whose primary-key property was
    /// rewritten to a representative's value. The standard MERGE
    /// later folds them into the same node.
    pub collapsed: usize,
    /// Number of in-batch candidates routed to NeedsReview. Each one
    /// also contributes an entry to `review_candidates` when
    /// `cfg.emit_review_candidates` is true.
    pub needs_review: usize,
    /// Review records for the in-batch NeedsReview candidates above.
    /// Empty when `cfg.emit_review_candidates == false`.
    pub review_candidates: Vec<ReviewCandidate>,
}

pub(super) fn collect_candidates(graph: &Graph) -> Result<Vec<Candidate>, IngestError> {
    let mut out = Vec::new();
    for (idx, entity) in graph.entities().iter().enumerate() {
        let field = match &entity.primary_key {
            Some(PrimaryKey::Soft) => CANONICAL_FIELD.to_string(),
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

/// In-batch deduplication using the staged decision pipeline.
///
/// For each candidate in `group` (in iteration order), collect every
/// existing representative whose cosine similarity is at or above
/// `cfg.similarity_threshold` and present them as `Hit`s to
/// `classify`. The same gates that govern Qdrant matches govern
/// in-batch matches:
///   * **AutoMerge** — rewrite the duplicate's primary-key property
///     to the representative's value; remove the duplicate from
///     `group` so only representatives go to Qdrant search.
///   * **NeedsReview** — keep both entities as separate
///     representatives (the standard MERGE will create two nodes);
///     emit a `ReviewCandidate` with the matched representative as
///     the top hit so a reviewer can later collapse them by hand.
///   * **NoMerge** — keep the candidate as a new representative.
///
/// This is intentionally more conservative than the old plain-cosine
/// dedup. Two near-identical embeddings can still hide distinct
/// entities (e.g. two people with the same first name but different
/// emails), and false in-batch merges destroy data the same way
/// false Qdrant merges do.
pub(super) fn deduplicate_in_batch(
    graph: &mut Graph,
    group: &mut Vec<EmbeddedCandidate>,
    label: &str,
    field: &str,
    cfg: &SoftMergeConfig,
) -> Result<InBatchDedupOutcome, IngestError> {
    let mut outcome = InBatchDedupOutcome::default();
    if group.len() < 2 {
        return Ok(outcome);
    }
    let threshold = cfg.similarity_threshold;

    // Indices into `group` of cluster representatives chosen so far.
    let mut representatives: Vec<usize> = Vec::with_capacity(group.len());
    // For each index in `group`, the in-batch decision we made for it.
    // Reps and NeedsReview entries get `None`; AutoMerged duplicates
    // record the representative-group-index they were folded onto.
    let mut assignment: Vec<Option<usize>> = vec![None; group.len()];

    for i in 0..group.len() {
        // Snapshot candidate i's text + props. We need them for both
        // lexical comparison and hard-conflict detection.
        let (cand_text, cand_props) = entity_snapshot(graph, group[i].entity_index, field)?;

        // Build hits from representatives whose cosine clears the
        // consideration floor. Each carries the matched rep's
        // canonical field value and full property map so the same
        // `classify` we run on Qdrant hits works here verbatim.
        let mut hits: Vec<Hit> = Vec::new();
        for &rep_group_idx in &representatives {
            let cos = cosine_similarity(&group[i].vec, &group[rep_group_idx].vec) as f64;
            if cos < threshold {
                continue;
            }
            let rep_entity_index = group[rep_group_idx].entity_index;
            let (rep_text, rep_props) = entity_snapshot(graph, rep_entity_index, field)?;
            hits.push(Hit {
                // Encode the rep's entity_index so AutoMerge can find
                // it again, and so review records can cross-reference
                // the other in-flight entity.
                id: rep_entity_index as i64,
                score: cos,
                canonical: rep_text,
                props: properties_to_json_map(&rep_props),
            });
        }
        // classify expects hits sorted by score descending.
        hits.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));

        let info = CandidateInfo {
            canonical_text: &cand_text,
            props: &cand_props,
        };
        let decision = classify(&info, &hits, cfg);
        match decision {
            Decision::AutoMerge {
                canonical,
                hit_id,
                top_score,
            } => {
                let rep_entity_index = hit_id as usize;
                let dup_entity_index = group[i].entity_index;
                tracing::debug!(
                    target: "linguagraph::soft_merge",
                    label = %label,
                    field = %field,
                    duplicate_index = dup_entity_index,
                    representative_index = rep_entity_index,
                    top_score,
                    "in-batch dedup: collapsing duplicate onto representative",
                );
                let dup_entity = graph.entities_mut().get_mut(dup_entity_index).ok_or_else(|| {
                    IngestError::SoftMerge(format!(
                        "in-batch dedup: duplicate idx {dup_entity_index} out of bounds"
                    ))
                })?;
                if let Some(prop) = dup_entity.properties.get_mut(field) {
                    prop.value = Value::String(canonical);
                    outcome.collapsed += 1;
                }
                // Record which representative i was folded onto so the
                // filter at the bottom of this function drops it.
                let rep_group_idx = representatives
                    .iter()
                    .position(|&r| group[r].entity_index == rep_entity_index)
                    .ok_or_else(|| {
                        IngestError::SoftMerge(format!(
                            "in-batch dedup: representative entity_index {rep_entity_index} \
                             not present in representatives"
                        ))
                    })?;
                assignment[i] = Some(rep_group_idx);
            }
            Decision::NeedsReview {
                top,
                runners_up,
                rejected_by,
            } => {
                outcome.needs_review += 1;
                if cfg.emit_review_candidates {
                    outcome.review_candidates.push(ReviewCandidate {
                        entity_index: group[i].entity_index,
                        label: label.to_string(),
                        field: field.to_string(),
                        incoming_value: incoming_name(&cand_text),
                        source: ReviewSource::InBatch,
                        top,
                        runners_up,
                        rejected_by,
                    });
                }
                // Keep i as a new representative so subsequent
                // candidates that match i (but not the existing reps)
                // get their own evaluation.
                representatives.push(i);
            }
            Decision::NoMerge => {
                // Either no rep cleared the consideration floor, or
                // all hits sat below the review band — i is clearly
                // distinct. Promote it to representative.
                representatives.push(i);
            }
        }
    }

    // Filter group down to representatives only. Order preserved.
    let keep: BTreeSet<usize> = representatives.into_iter().collect();
    let mut idx = 0usize;
    group.retain(|_| {
        let k = keep.contains(&idx);
        idx += 1;
        k
    });
    let _ = assignment; // kept above for clarity; not needed post-filter

    Ok(outcome)
}

/// Snapshot `(text, properties)` of the entity at `entity_index`
/// without holding the borrow past the call site. The text is the
/// stringified value of the soft-merge `field` property.
fn entity_snapshot(
    graph: &Graph,
    entity_index: usize,
    field: &str,
) -> Result<(String, HashMap<String, Property>), IngestError> {
    let entity = graph.entities().get(entity_index).ok_or_else(|| {
        IngestError::SoftMerge(format!(
            "in-batch dedup: entity idx {entity_index} out of bounds"
        ))
    })?;
    let text = entity
        .properties
        .get(field)
        .map(|p| match &p.value {
            Value::String(s) => s.clone(),
            other => other.to_string(),
        })
        .unwrap_or_default();
    Ok((text, entity.properties.clone()))
}

/// Convert an in-memory property bag to the same `serde_json::Map`
/// shape `Hit::props` carries, so `detect_hard_conflict` can compare
/// in-batch representatives the same way it compares Qdrant hits.
fn properties_to_json_map(props: &HashMap<String, Property>) -> JsonMap<String, Value> {
    props
        .iter()
        .map(|(k, v)| (k.clone(), v.value.clone()))
        .collect()
}

fn json_to_text(value: &Value) -> String {
    match value {
        Value::String(s) => s.clone(),
        Value::Null => String::new(),
        other => other.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{EntityGraph, GraphBuilder, PropertyType};

    #[test]
    fn picks_soft_entities_only() {
        let mut b = GraphBuilder::new();
        b.add_entity(
            EntityGraph::new("LegalConcept")
                .soft_primary_key()
                .property("name", PropertyType::Text, "общественное согласие"),
        );
        b.add_entity(
            EntityGraph::new("Person")
                .strict_primary_key("id")
                .property("id", PropertyType::Keyword, "p1"),
        );
        let graph = b.build();

        let got = collect_candidates(&graph).unwrap();
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].entity_index, 0);
        assert_eq!(got[0].label, "LegalConcept");
        assert_eq!(got[0].field, CANONICAL_FIELD);
    }

    #[test]
    fn soft_entity_uses_builder_created_canonical_field() {
        let mut b = GraphBuilder::new();
        b.add_entity(EntityGraph::new("LegalConcept").soft_primary_key());
        let graph = b.build();
        let got = collect_candidates(&graph).unwrap();
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].field, CANONICAL_FIELD);
        assert_eq!(got[0].text, "type: LegalConcept");
    }
}
