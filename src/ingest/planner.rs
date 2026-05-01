//! Turn extracted rows into a [`crate::ast::query::InsertQuery`].
//!
//! Two responsibilities:
//!
//! 1. **Deduplication.** The same primary key may be hit many times
//!    (e.g. several cameras share a Place). We keep the *last* property
//!    map for a given id — `MERGE … SET n += row.props` is order-agnostic
//!    in our data, but order-stable here makes tests deterministic.
//!
//! 2. **Implicit relationship resolution.** A relationship `(Camera, Place)`
//!    is materialised by pairing each Camera row with the Place rows whose
//!    extraction context shares a prefix. That mirrors how nested data
//!    naturally encodes parent/child links — Camera at `[i]` and Place at
//!    `[i]` are the same camera; a module at `[i, j]` belongs to camera
//!    `[i]`.

use std::collections::{BTreeMap, HashMap, HashSet};

use crate::ast::query::{
    InsertQuery, Literal, NodeBatch, NodeRow, RelationBatch, RelationRow,
};
use crate::mapper::{Extracted, ExtractedEntity, Mapping};

use super::dsl::{InsertPlan, NodeData, NodePlan, RelationData, RelationPlan};
use super::IngestError;

/// Configurable planner knobs.
///
/// `max_batch_size` caps the number of rows in a single Cypher batch so
/// huge ingests don't blow past Memgraph's parameter or memory limits.
#[derive(Debug, Clone, Copy)]
pub struct PlannerOptions {
    pub max_batch_size: usize,
}

impl Default for PlannerOptions {
    fn default() -> Self {
        Self { max_batch_size: 1000 }
    }
}

/// Convenience: extract → plan with default options.
pub fn plan(mapping: &Mapping, extracted: Extracted) -> Result<InsertQuery, IngestError> {
    plan_with_options(mapping, extracted, PlannerOptions::default())
}

/// Build the [`InsertPlan`] (internal DSL) and lower it directly into an
/// [`InsertQuery`] in one pass. We expose both so callers that want to
/// inspect or persist the intermediate plan can do so.
pub fn plan_with_options(
    mapping: &Mapping,
    extracted: Extracted,
    opts: PlannerOptions,
) -> Result<InsertQuery, IngestError> {
    if opts.max_batch_size == 0 {
        return Err(IngestError::InvalidBatchSize);
    }

    let plan = build_plan(mapping, &extracted)?;
    Ok(lower_plan(plan, opts))
}

/// Assemble the internal DSL from the extracted rows.
pub fn build_plan(mapping: &Mapping, extracted: &Extracted) -> Result<InsertPlan, IngestError> {
    let by_label: HashMap<&str, &ExtractedEntity> =
        extracted.entities.iter().map(|e| (e.label.as_str(), e)).collect();

    // ── Node batches ────────────────────────────────────────────────────────
    let mut nodes = Vec::with_capacity(extracted.entities.len());
    for ent in &extracted.entities {
        // Dedup by id, last-write-wins on properties. We use a HashMap
        // (Literal isn't Ord) and sort the resulting rows for stable
        // output — handy for snapshot tests and deterministic CI logs.
        let mut seen: HashMap<Literal, NodeData> = HashMap::new();
        let mut order: Vec<Literal> = Vec::new();
        for row in &ent.rows {
            if !seen.contains_key(&row.id) {
                order.push(row.id.clone());
            }
            let entry = seen.entry(row.id.clone()).or_insert_with(|| NodeData {
                id: row.id.clone(),
                props: BTreeMap::new(),
            });
            for (k, v) in &row.properties {
                entry.props.insert(k.clone(), v.clone());
            }
        }
        let mut rows: Vec<NodeData> = order
            .into_iter()
            .map(|id| seen.remove(&id).expect("present"))
            .collect();
        rows.sort_by(|a, b| literal_cmp(&a.id, &b.id));
        nodes.push(NodePlan {
            label: ent.label.clone(),
            merge_on: ent.primary_key_field.clone(),
            rows,
        });
    }

    // ── Relationship batches ────────────────────────────────────────────────
    let mut relations = Vec::with_capacity(mapping.relationships.len());
    for rel in &mapping.relationships {
        let from = by_label
            .get(rel.from.as_str())
            .ok_or_else(|| IngestError::UnknownEntityType(rel.from.clone()))?;
        let to = by_label
            .get(rel.to.as_str())
            .ok_or_else(|| IngestError::UnknownEntityType(rel.to.clone()))?;

        let mut pairs: HashSet<(Literal, Literal)> = HashSet::new();
        for from_row in &from.rows {
            for to_row in &to.rows {
                if contexts_align(&from_row.context, &to_row.context) {
                    pairs.insert((from_row.id.clone(), to_row.id.clone()));
                }
            }
        }

        // Sort so the rendered Cypher is deterministic for snapshot tests.
        let mut rows: Vec<(Literal, Literal)> = pairs.into_iter().collect();
        rows.sort_by(|a, b| literal_cmp(&a.0, &b.0).then(literal_cmp(&a.1, &b.1)));

        relations.push(RelationPlan {
            rel_type: rel.kind.clone(),
            from_label: from.label.clone(),
            from_key: from.primary_key_field.clone(),
            to_label: to.label.clone(),
            to_key: to.primary_key_field.clone(),
            rows: rows
                .into_iter()
                .map(|(from_id, to_id)| RelationData { from_id, to_id })
                .collect(),
        });
    }

    Ok(InsertPlan { action: "insert".to_string(), nodes, relations })
}

/// Two contexts align when one is a prefix of the other. That covers both
/// "siblings at the same depth" (equal contexts) and "parent ↔ descendant"
/// (one strictly extends the other). Empty contexts (no `[*]` in the
/// source path) match anything — useful for singleton entities.
fn contexts_align(a: &[usize], b: &[usize]) -> bool {
    let n = a.len().min(b.len());
    a[..n] == b[..n]
}

fn lower_plan(plan: InsertPlan, opts: PlannerOptions) -> InsertQuery {
    let mut node_batches = Vec::new();
    for n in plan.nodes {
        if n.rows.is_empty() {
            continue;
        }
        for chunk in n.rows.chunks(opts.max_batch_size) {
            node_batches.push(NodeBatch {
                label: n.label.clone(),
                merge_on: n.merge_on.clone(),
                rows: chunk
                    .iter()
                    .map(|r| NodeRow { id: r.id.clone(), props: r.props.clone() })
                    .collect(),
            });
        }
    }

    let mut relation_batches = Vec::new();
    for r in plan.relations {
        if r.rows.is_empty() {
            continue;
        }
        for chunk in r.rows.chunks(opts.max_batch_size) {
            relation_batches.push(RelationBatch {
                rel_type: r.rel_type.clone(),
                from_label: r.from_label.clone(),
                from_key: r.from_key.clone(),
                to_label: r.to_label.clone(),
                to_key: r.to_key.clone(),
                rows: chunk
                    .iter()
                    .map(|d| RelationRow {
                        from_id: d.from_id.clone(),
                        to_id: d.to_id.clone(),
                    })
                    .collect(),
            });
        }
    }

    InsertQuery { node_batches, relation_batches }
}

/// Total order over the literal types used as ids.
fn literal_cmp(a: &Literal, b: &Literal) -> std::cmp::Ordering {
    use std::cmp::Ordering::*;
    match (a, b) {
        (Literal::String(x), Literal::String(y)) => x.cmp(y),
        (Literal::Int(x), Literal::Int(y)) => x.cmp(y),
        (Literal::Bool(x), Literal::Bool(y)) => x.cmp(y),
        (Literal::Null, Literal::Null) => Equal,
        // Mixed-type ids would be a planner bug, but fall back to a
        // string-based ordering rather than panic.
        (a, b) => format!("{a:?}").cmp(&format!("{b:?}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mapper;
    use serde_json::json;

    fn make_mapping() -> Mapping {
        serde_json::from_value(json!({
            "entities": [
                {
                    "type": "Camera",
                    "source_path": "$.cameras[*]",
                    "primary_key": "$.cameras[*].id",
                    "properties": [
                        {"name": "id", "source_path": "$.cameras[*].id"},
                        {"name": "state", "source_path": "$.cameras[*].state"}
                    ]
                },
                {
                    "type": "Place",
                    "source_path": "$.cameras[*].origin",
                    "primary_key": "$.cameras[*].origin.place_id",
                    "properties": [
                        {"name": "id", "source_path": "$.cameras[*].origin.place_id"}
                    ]
                }
            ],
            "relationships": [
                {"type": "LOCATED_IN", "from": "Camera", "to": "Place"}
            ]
        }))
        .unwrap()
    }

    #[test]
    fn plans_nodes_and_one_to_one_relationships() {
        let m = make_mapping();
        let data = json!({
            "cameras": [
                {"id": "c1", "state": "active",   "origin": {"place_id": "p1"}},
                {"id": "c2", "state": "inactive", "origin": {"place_id": "p2"}}
            ]
        });
        let extracted = mapper::extract(&m, &data).unwrap();
        let q = plan(&m, extracted).unwrap();

        assert_eq!(q.node_batches.len(), 2);
        assert_eq!(q.node_batches[0].label, "Camera");
        assert_eq!(q.node_batches[0].rows.len(), 2);
        assert_eq!(q.node_batches[1].label, "Place");
        assert_eq!(q.node_batches[1].rows.len(), 2);

        assert_eq!(q.relation_batches.len(), 1);
        let r = &q.relation_batches[0];
        assert_eq!(r.rel_type, "LOCATED_IN");
        assert_eq!(r.rows.len(), 2);
    }

    #[test]
    fn deduplicates_shared_destinations() {
        let m = make_mapping();
        // Two cameras pointing at the same place.
        let data = json!({
            "cameras": [
                {"id": "c1", "origin": {"place_id": "shared"}},
                {"id": "c2", "origin": {"place_id": "shared"}}
            ]
        });
        let extracted = mapper::extract(&m, &data).unwrap();
        let q = plan(&m, extracted).unwrap();

        let place_batch = q.node_batches.iter().find(|b| b.label == "Place").unwrap();
        assert_eq!(place_batch.rows.len(), 1, "shared place should dedupe");

        let rel_rows = &q.relation_batches[0].rows;
        assert_eq!(rel_rows.len(), 2, "two cameras both link to the shared place");
    }

    #[test]
    fn parent_child_context_alignment() {
        // Modules belong to their parent camera by context prefix.
        let mapping: Mapping = serde_json::from_value(json!({
            "entities": [
                {
                    "type": "Camera",
                    "source_path": "$.cameras[*]",
                    "primary_key": "$.cameras[*].id"
                },
                {
                    "type": "Module",
                    "source_path": "$.cameras[*].modules[*]",
                    "primary_key": "$.cameras[*].modules[*].name"
                }
            ],
            "relationships": [
                {"type": "HAS", "from": "Camera", "to": "Module"}
            ]
        }))
        .unwrap();

        let data = json!({
            "cameras": [
                {"id": "c1", "modules": [{"name": "m1"}, {"name": "m2"}]},
                {"id": "c2", "modules": [{"name": "m3"}]}
            ]
        });
        let extracted = mapper::extract(&mapping, &data).unwrap();
        let q = plan(&mapping, extracted).unwrap();

        let rel = &q.relation_batches[0];
        assert_eq!(rel.rows.len(), 3);
        // c1 must NOT link to m3 (different camera context).
        let bad = rel.rows.iter().any(|r| {
            r.from_id == Literal::String("c1".into()) && r.to_id == Literal::String("m3".into())
        });
        assert!(!bad, "context-prefix alignment must not cross siblings");
    }

    #[test]
    fn batch_size_is_respected() {
        let m = make_mapping();
        let cameras: Vec<_> = (0..2500)
            .map(|i| json!({"id": format!("c{i}"), "origin": {"place_id": format!("p{i}")}}))
            .collect();
        let data = json!({"cameras": cameras});
        let extracted = mapper::extract(&m, &data).unwrap();
        let q = plan_with_options(
            &m,
            extracted,
            PlannerOptions { max_batch_size: 1000 },
        )
        .unwrap();
        // 2500 cameras → 3 batches; 2500 places → 3 batches.
        let camera_batches: Vec<_> = q
            .node_batches
            .iter()
            .filter(|b| b.label == "Camera")
            .collect();
        assert_eq!(camera_batches.len(), 3);
        assert_eq!(camera_batches[0].rows.len(), 1000);
        assert_eq!(camera_batches[2].rows.len(), 500);
    }

    #[test]
    fn unknown_relationship_endpoint_is_rejected() {
        let mapping: Mapping = serde_json::from_value(json!({
            "entities": [{
                "type": "Camera",
                "source_path": "$.cameras[*]",
                "primary_key": "$.cameras[*].id"
            }],
            "relationships": [
                {"type": "X", "from": "Camera", "to": "Ghost"}
            ]
        }))
        .unwrap();
        let data = json!({"cameras": [{"id": "c1"}]});
        let extracted = mapper::extract(&mapping, &data).unwrap();
        assert!(matches!(
            plan(&mapping, extracted),
            Err(IngestError::UnknownEntityType(_))
        ));
    }
}
