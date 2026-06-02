//! Soft-merge candidate collection + in-batch deduplication.
//!
//! Candidates are entities with `PrimaryKey::Soft` and a non-empty
//! `_canonical` property. In-batch deduplication collapses
//! near-identical embeddings within one ingest call onto a single
//! representative — duplicates have their primary-key property
//! rewritten so the standard Cypher MERGE later folds them into the
//! same node.

use std::collections::BTreeSet;

use serde_json::Value;

use crate::config::SoftMergeConfig;
use crate::embeddings::cosine_similarity;
use crate::graph::{Graph, PrimaryKey, CANONICAL_FIELD};
use crate::ingest::IngestError;

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

/// Collapse near-identical embeddings within `group` onto a single
/// representative. For every duplicate, rewrite the entity's
/// soft-merge property (`field`) to match the representative's value
/// in `graph` — the standard Cypher MERGE then folds the two rows
/// into one node. The duplicate is also removed from `group` so the
/// downstream Qdrant query only carries representatives.
///
/// Single-link clustering by cosine similarity against
/// `cfg.similarity_threshold`. Within a single ingest, identical or
/// near-identical LLM extractions ("Microsoft" vs "Microsoft Corp.")
/// collapse here instead of producing two separate nodes that may or
/// may not merge at the database level depending on whether one of
/// them happens to match a pre-existing node first.
pub(super) fn deduplicate_in_batch(
    graph: &mut Graph,
    group: &mut Vec<EmbeddedCandidate>,
    field: &str,
    cfg: &SoftMergeConfig,
) -> Result<usize, IngestError> {
    if group.len() < 2 {
        return Ok(0);
    }
    let threshold = cfg.similarity_threshold as f32;

    let mut representatives: Vec<usize> = Vec::with_capacity(group.len());
    let mut assignment: Vec<Option<usize>> = vec![None; group.len()];

    for i in 0..group.len() {
        let mut best: Option<(usize, f32)> = None;
        for &rep in &representatives {
            let sim = cosine_similarity(&group[i].vec, &group[rep].vec);
            if sim >= threshold {
                match best {
                    Some((_, current)) if sim <= current => {}
                    _ => best = Some((rep, sim)),
                }
            }
        }
        match best {
            Some((rep, _)) => assignment[i] = Some(rep),
            None => representatives.push(i),
        }
    }

    let mut collapsed = 0usize;
    for (i, maybe_rep) in assignment.iter().enumerate() {
        let Some(rep_index) = *maybe_rep else { continue };
        let rep_entity_index = group[rep_index].entity_index;
        let dup_entity_index = group[i].entity_index;
        let canonical_value = {
            let rep_entity = graph.entities_mut().get(rep_entity_index).ok_or_else(|| {
                IngestError::SoftMerge(format!(
                    "in-batch dedup: representative idx {rep_entity_index} out of bounds"
                ))
            })?;
            rep_entity
                .properties
                .get(field)
                .map(|p| p.value.clone())
                .ok_or_else(|| {
                    IngestError::SoftMerge(format!(
                        "in-batch dedup: representative missing field `{field}`"
                    ))
                })?
        };
        let dup_entity = graph.entities_mut().get_mut(dup_entity_index).ok_or_else(|| {
            IngestError::SoftMerge(format!(
                "in-batch dedup: duplicate idx {dup_entity_index} out of bounds"
            ))
        })?;
        if let Some(prop) = dup_entity.properties.get_mut(field) {
            prop.value = canonical_value;
            collapsed += 1;
        }
    }

    let keep: BTreeSet<usize> = representatives.into_iter().collect();
    let mut idx = 0usize;
    group.retain(|_| {
        let k = keep.contains(&idx);
        idx += 1;
        k
    });

    Ok(collapsed)
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
                .property("id", PropertyType::String, "p1"),
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
