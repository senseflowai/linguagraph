//! Document-shaped ingestion: lift a `{document, chunks, entities, relations}`
//! JSON straight into an [`InsertPlan`] without going through the mapping
//! layer.
//!
//! The mapping-driven path (see [`super::planner`]) is right for ETL of
//! structured data with static labels and JSONPath-resolvable relations.
//! Documents are different: LLM-generated entity types are dynamic, relation
//! types are dynamic, and the `from`/`to` ids in a chunk's `relations` array
//! are *local* handles — they only mean something within the same chunk.
//!
//! So we build the plan directly. Document/Chunk are first-class built-in
//! labels (`Document`, `Chunk`); user-emitted entity types become Cypher
//! labels after sanitization; user-emitted relation types become Cypher
//! relationship types after sanitization + upper-casing. Entity nodes are
//! merged on a deterministic UUID v5 derived from
//! `(document.path, chunk.id, entity.local_id)` so re-ingesting the same
//! input is idempotent. There is *no* cross-chunk deduplication — two
//! mentions of "Alice" in different chunks are two distinct `(:Person)`
//! nodes — by design (see the plan file).
//!
//! ## Embeddings
//!
//! Every string field on a chunk (`text`) and every string field on an
//! entity (`name`) is treated as `SemanticText`. The registered
//! [`crate::types::handlers::SemanticTextHandler`] runs against each one
//! via [`IngestCtx`] just like a mapping-declared typed property would
//! (mirroring [`super::planner::apply_type_handlers`]). It stores the
//! raw text on the node *and* queues an
//! [`crate::types::SideEffect::EmbedAndStore`] so the pipeline embeds
//! the value after the Memgraph batch lands.
//!
//! The handler MUST be registered. Without it `build_document_plan`
//! returns [`IngestError::Type`] — silently ingesting without
//! embeddings would make `c.text` / `<Label>.name` SemanticText
//! searches miss without any signal to the caller.

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::ast::query::Literal;
use crate::types::context::IngestCtx;
use crate::types::handlers::SemanticTextHandler;
use crate::types::{SideEffectQueue, TypeHandler, TypeRegistry};

use super::dsl::{InsertPlan, NodeData, NodePlan, RelationData, RelationPlan};
use super::IngestError;

/// Built-in label for the document root node.
pub const DOCUMENT_LABEL: &str = "Document";
/// Built-in label for chunk nodes.
pub const CHUNK_LABEL: &str = "Chunk";
/// Built-in relation: `(:Document)-[:HAS_CHUNK]->(:Chunk)`.
pub const HAS_CHUNK_REL: &str = "HAS_CHUNK";
/// Built-in relation: `(:Chunk)-[:MENTIONS]->(:<EntityLabel>)`.
pub const MENTIONS_REL: &str = "MENTIONS";

/// Stable namespace for UUID v5 derivation. Picked once and never changed
/// so re-ingesting the same `(document.path, chunk.id, local_id)` triple
/// always yields the same node id.
const UUID_NAMESPACE: Uuid = Uuid::from_u128(0x6c69_6e67_7561_6772_6170_6864_6f63_5f31);

/// Root of the user-supplied JSON.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentInput {
    pub document: DocumentBody,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentBody {
    pub name: String,
    pub path: String,
    #[serde(default)]
    pub chunks: Vec<ChunkInput>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkInput {
    pub id: String,
    pub text: String,
    #[serde(default)]
    pub entities: Vec<EntityInput>,
    #[serde(default)]
    pub relations: Vec<RelationInput>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityInput {
    /// Local handle; scoped to the parent chunk's `relations` array.
    pub id: String,
    #[serde(rename = "type")]
    pub kind: String,
    pub name: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelationInput {
    pub from: String,
    pub to: String,
    #[serde(rename = "type")]
    pub kind: String,
}

/// Knobs for [`build_document_plan`].
///
/// The Qdrant collection name is **no longer** carried here — it lives
/// on the registered [`SemanticTextHandler`]'s config (`config.collection`)
/// so chunk-text and entity-name embeddings end up in the same
/// collections the query path consults (`{collection}__text`,
/// `{collection}__name`).
#[derive(Debug, Clone)]
pub struct DocumentIngestOptions {
    pub max_batch_size: usize,
}

impl Default for DocumentIngestOptions {
    fn default() -> Self {
        Self { max_batch_size: 1000 }
    }
}

/// Build an [`InsertPlan`] + side-effect queue from a document JSON.
///
/// Pure — no I/O, no DB calls. The caller (typically
/// `Pipeline::ingest_document`) lowers the plan, renders Cypher, executes
/// the batches, and drains the side effects.
///
/// `registry` MUST contain a `SemanticText` handler. Every string field
/// on chunks (`text`) and entities (`name`) is routed through that
/// handler so embeddings are queued in `effects` exactly like a typed
/// mapping property would do via [`super::planner::apply_type_handlers`].
pub fn build_document_plan(
    doc: &DocumentInput,
    opts: &DocumentIngestOptions,
    registry: &TypeRegistry,
    effects: &mut SideEffectQueue,
) -> Result<InsertPlan, IngestError> {
    if opts.max_batch_size == 0 {
        return Err(IngestError::InvalidBatchSize);
    }

    // Look up the SemanticText handler. We *require* it: every entity
    // `name` and every chunk `text` is assumed to be SemanticText, and
    // silently degrading to plain strings would make downstream semantic
    // search miss without any signal.
    let semantic_handler = registry
        .get_by_name(SemanticTextHandler::TYPE_ID)
        .map_err(|_| {
            IngestError::Type(format!(
                "document ingest requires a `{}` handler to be registered \
                 (it embeds chunk text and entity names for semantic search)",
                SemanticTextHandler::TYPE_ID,
            ))
        })?
        .clone();

    let doc_path = &doc.document.path;

    // ── 1. Document node. ──────────────────────────────────────────────
    let mut doc_props: BTreeMap<String, Literal> = BTreeMap::new();
    doc_props.insert("path".to_string(), Literal::String(doc_path.clone()));
    doc_props.insert("name".to_string(), Literal::String(doc.document.name.clone()));
    let document_plan = NodePlan {
        label: DOCUMENT_LABEL.to_string(),
        merge_on: "path".to_string(),
        rows: vec![NodeData {
            id: Literal::String(doc_path.clone()),
            props: doc_props,
        }],
    };

    // ── 2. Walk chunks. Accumulate: chunk rows, entity rows (keyed by
    //    sanitized label), MENTIONS rows (keyed by entity label), and
    //    user-relation rows (keyed by `(rel_type, from_label, to_label)`).
    let mut chunk_rows: Vec<NodeData> = Vec::with_capacity(doc.document.chunks.len());
    let mut entities_by_label: BTreeMap<String, Vec<NodeData>> = BTreeMap::new();
    let mut mentions_by_label: BTreeMap<String, Vec<RelationData>> = BTreeMap::new();
    type UserRelKey = (String, String, String); // (rel_type, from_label, to_label)
    let mut user_rels: BTreeMap<UserRelKey, Vec<RelationData>> = BTreeMap::new();
    let mut has_chunk_rows: Vec<RelationData> = Vec::with_capacity(doc.document.chunks.len());

    for (chunk_idx, chunk) in doc.document.chunks.iter().enumerate() {
        let composite_chunk_id = chunk_uuid(doc_path, &chunk.id);
        let chunk_key = Literal::String(composite_chunk_id.clone());

        // Chunk node. The `text` field is routed through the SemanticText
        // handler so the raw text lands on the node *and* an embedding
        // side effect is queued.
        let mut props: BTreeMap<String, Literal> = BTreeMap::new();
        let text_lit = apply_semantic_text(
            &semantic_handler,
            CHUNK_LABEL,
            "id",
            &chunk_key,
            "text",
            &chunk.text,
            effects,
        )?;
        if let Some(lit) = text_lit {
            props.insert("text".to_string(), lit);
        }
        props.insert("index".to_string(), Literal::Int(chunk_idx as i64));
        props.insert(
            "document_path".to_string(),
            Literal::String(doc_path.clone()),
        );
        // Keep the original chunk id (within-document handle) as a property
        // for human inspection; the merge key is the composite uuid.
        props.insert(
            "local_id".to_string(),
            Literal::String(chunk.id.clone()),
        );
        chunk_rows.push(NodeData {
            id: chunk_key.clone(),
            props,
        });

        // HAS_CHUNK edge.
        has_chunk_rows.push(RelationData {
            from_id: Literal::String(doc_path.clone()),
            to_id: chunk_key.clone(),
        });

        // Per-chunk symbol table: local id → (sanitized label, uuid).
        // We need both pieces to:
        //  - assemble MENTIONS rows (need label, since RelationPlan is per
        //    (chunk -> label) pair),
        //  - resolve `from`/`to` in user relations (need the uuid and the
        //    label of each endpoint).
        let mut local: HashMap<String, ResolvedEntity> =
            HashMap::with_capacity(chunk.entities.len());

        for ent in &chunk.entities {
            let sanitized = sanitize_label(&ent.kind).ok_or_else(|| {
                IngestError::InvalidLabel(ent.kind.clone())
            })?;
            if is_reserved_label(&sanitized) {
                return Err(IngestError::ReservedLabel(sanitized));
            }

            let uuid = entity_uuid(doc_path, &chunk.id, &ent.id);
            let entity_key = Literal::String(uuid.clone());

            // Entity node row. Store both the cleaned label and the
            // original (pre-sanitize) `type` string for display. The
            // `name` field is routed through the SemanticText handler
            // so it's embedded into the `{cfg.collection}__name`
            // Qdrant collection alongside chunk-text embeddings.
            let mut ent_props: BTreeMap<String, Literal> = BTreeMap::new();
            ent_props.insert("id".to_string(), entity_key.clone());
            let name_lit = apply_semantic_text(
                &semantic_handler,
                &sanitized,
                "id",
                &entity_key,
                "name",
                &ent.name,
                effects,
            )?;
            if let Some(lit) = name_lit {
                ent_props.insert("name".to_string(), lit);
            }
            // Preserve LLM-original type wording (may include spaces/etc).
            ent_props.insert(
                "type".to_string(),
                Literal::String(ent.kind.clone()),
            );
            entities_by_label
                .entry(sanitized.clone())
                .or_default()
                .push(NodeData {
                    id: entity_key.clone(),
                    props: ent_props,
                });

            // MENTIONS edge: (:Chunk {id: <composite>})-[:MENTIONS]->(:<label> {id: <uuid>}).
            mentions_by_label
                .entry(sanitized.clone())
                .or_default()
                .push(RelationData {
                    from_id: chunk_key.clone(),
                    to_id: entity_key.clone(),
                });

            // Symbol-table entry.
            if local
                .insert(
                    ent.id.clone(),
                    ResolvedEntity {
                        label: sanitized,
                        uuid,
                    },
                )
                .is_some()
            {
                tracing::debug!(
                    chunk = %chunk.id,
                    local_id = %ent.id,
                    "duplicate local entity id in same chunk; later entry wins"
                );
            }
        }

        for rel in &chunk.relations {
            let from = local.get(&rel.from).ok_or_else(|| {
                IngestError::UnknownLocalId {
                    chunk: chunk.id.clone(),
                    local_id: rel.from.clone(),
                }
            })?;
            let to = local.get(&rel.to).ok_or_else(|| {
                IngestError::UnknownLocalId {
                    chunk: chunk.id.clone(),
                    local_id: rel.to.clone(),
                }
            })?;
            let rel_type = sanitize_rel(&rel.kind).ok_or_else(|| {
                IngestError::InvalidLabel(rel.kind.clone())
            })?;
            if is_reserved_relation(&rel_type) {
                return Err(IngestError::ReservedRelation(rel_type));
            }
            user_rels
                .entry((rel_type, from.label.clone(), to.label.clone()))
                .or_default()
                .push(RelationData {
                    from_id: Literal::String(from.uuid.clone()),
                    to_id: Literal::String(to.uuid.clone()),
                });
        }
    }

    // ── 3. Assemble the final InsertPlan. Sort rows deterministically. ─
    let mut nodes: Vec<NodePlan> = Vec::new();
    nodes.push(document_plan);
    if !chunk_rows.is_empty() {
        sort_node_rows(&mut chunk_rows);
        nodes.push(NodePlan {
            label: CHUNK_LABEL.to_string(),
            merge_on: "id".to_string(),
            rows: chunk_rows,
        });
    }
    for (label, mut rows) in entities_by_label {
        sort_node_rows(&mut rows);
        nodes.push(NodePlan {
            label,
            merge_on: "id".to_string(),
            rows,
        });
    }

    let mut relations: Vec<RelationPlan> = Vec::new();
    if !has_chunk_rows.is_empty() {
        sort_relation_rows(&mut has_chunk_rows);
        relations.push(RelationPlan {
            rel_type: HAS_CHUNK_REL.to_string(),
            from_label: DOCUMENT_LABEL.to_string(),
            from_key: "path".to_string(),
            to_label: CHUNK_LABEL.to_string(),
            to_key: "id".to_string(),
            rows: has_chunk_rows,
        });
    }
    for (label, mut rows) in mentions_by_label {
        sort_relation_rows(&mut rows);
        relations.push(RelationPlan {
            rel_type: MENTIONS_REL.to_string(),
            from_label: CHUNK_LABEL.to_string(),
            from_key: "id".to_string(),
            to_label: label,
            to_key: "id".to_string(),
            rows,
        });
    }
    for ((rel_type, from_label, to_label), mut rows) in user_rels {
        sort_relation_rows(&mut rows);
        relations.push(RelationPlan {
            rel_type,
            from_label,
            from_key: "id".to_string(),
            to_label,
            to_key: "id".to_string(),
            rows,
        });
    }

    Ok(InsertPlan {
        action: "insert".to_string(),
        nodes,
        relations,
    })
}

/// Run `value` through the SemanticText handler exactly like
/// [`super::planner::apply_type_handlers`] does for a typed mapping
/// property. Returns the literal to store on the node (or `None` when
/// the handler chose to skip the property entirely — which the bundled
/// SemanticText never does, but other implementations could).
fn apply_semantic_text(
    handler: &Arc<dyn TypeHandler>,
    label: &str,
    key_field: &str,
    key_value: &Literal,
    field_name: &str,
    value: &str,
    effects: &mut SideEffectQueue,
) -> Result<Option<Literal>, IngestError> {
    let raw = serde_json::Value::String(value.to_string());
    let mut ctx = IngestCtx::new(label, key_field, key_value, field_name, &raw, effects);
    handler
        .on_ingest(&mut ctx)
        .map_err(|e| IngestError::Type(e.to_string()))?;
    Ok(match ctx.finish() {
        None => Some(Literal::String(value.to_string())),
        Some(Some(lit)) => Some(lit),
        Some(None) => None,
    })
}

#[derive(Debug, Clone)]
struct ResolvedEntity {
    label: String,
    uuid: String,
}

fn sort_node_rows(rows: &mut [NodeData]) {
    rows.sort_by(|a, b| literal_cmp(&a.id, &b.id));
}

fn sort_relation_rows(rows: &mut [RelationData]) {
    rows.sort_by(|a, b| {
        literal_cmp(&a.from_id, &b.from_id).then(literal_cmp(&a.to_id, &b.to_id))
    });
}

fn literal_cmp(a: &Literal, b: &Literal) -> std::cmp::Ordering {
    use std::cmp::Ordering::*;
    match (a, b) {
        (Literal::String(x), Literal::String(y)) => x.cmp(y),
        (Literal::Int(x), Literal::Int(y)) => x.cmp(y),
        (Literal::Bool(x), Literal::Bool(y)) => x.cmp(y),
        (Literal::Null, Literal::Null) => Equal,
        (a, b) => format!("{a:?}").cmp(&format!("{b:?}")),
    }
}

/// Sanitize an LLM-emitted label so it satisfies the Cypher identifier
/// grammar `[A-Za-z_][A-Za-z0-9_]*` (see `src/builder/insert.rs:119`).
/// Returns `None` if the result would be empty.
fn sanitize_label(s: &str) -> Option<String> {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        if c.is_ascii_alphanumeric() || c == '_' {
            out.push(c);
        } else {
            out.push('_');
        }
    }
    // Trim a leading/trailing run of underscores produced from spaces or
    // punctuation. Keeps interior underscores untouched.
    let trimmed = out.trim_matches('_');
    if trimmed.is_empty() {
        return None;
    }
    let mut out = trimmed.to_string();
    // Prefix `_` if the first char is now a digit.
    if out.chars().next().map(|c| c.is_ascii_digit()).unwrap_or(false) {
        out.insert(0, '_');
    }
    if out != s {
        tracing::debug!(original = %s, sanitized = %out, "label sanitized");
    }
    Some(out)
}

/// Same as [`sanitize_label`] but additionally upper-cases the result so
/// relation types follow the conventional `WORKS_AT` casing.
fn sanitize_rel(s: &str) -> Option<String> {
    let label = sanitize_label(s)?;
    let upper = label.to_ascii_uppercase();
    if upper != s {
        tracing::debug!(original = %s, sanitized = %upper, "relation type sanitized");
    }
    Some(upper)
}

fn is_reserved_label(s: &str) -> bool {
    matches!(s, DOCUMENT_LABEL | CHUNK_LABEL)
}

fn is_reserved_relation(s: &str) -> bool {
    matches!(s, HAS_CHUNK_REL | MENTIONS_REL)
}

/// Deterministic id for a chunk node.
fn chunk_uuid(doc_path: &str, chunk_id: &str) -> String {
    let key = format!("chunk:{doc_path}#{chunk_id}");
    Uuid::new_v5(&UUID_NAMESPACE, key.as_bytes())
        .hyphenated()
        .to_string()
}

/// Deterministic id for an entity node.
fn entity_uuid(doc_path: &str, chunk_id: &str, local_id: &str) -> String {
    let key = format!("entity:{doc_path}#{chunk_id}#{local_id}");
    Uuid::new_v5(&UUID_NAMESPACE, key.as_bytes())
        .hyphenated()
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embeddings::MockEmbedder;
    use crate::types::handlers::{SemanticTextConfig, SemanticTextHandler};
    use crate::types::{RegistryBuilder, SideEffect};
    use serde_json::json;
    use std::sync::Arc;

    fn parse_doc(v: serde_json::Value) -> DocumentInput {
        serde_json::from_value(v).expect("doc parses")
    }

    fn opts() -> DocumentIngestOptions {
        DocumentIngestOptions::default()
    }

    /// Registry pre-populated with a SemanticText handler backed by the
    /// deterministic mock embedder — required by `build_document_plan`.
    fn test_registry() -> TypeRegistry {
        let cfg = SemanticTextConfig {
            embedding_model: None,
            collection: "test".into(),
            top_k: 10,
            search_threshold: 0.8,
            reranker_threshold: 0.3,
        };
        RegistryBuilder::new()
            .register(SemanticTextHandler::new(cfg, Arc::new(MockEmbedder::new(8))))
            .build()
    }

    fn plan(doc: &DocumentInput) -> Result<(InsertPlan, SideEffectQueue), IngestError> {
        let reg = test_registry();
        let mut effects = SideEffectQueue::new();
        let p = build_document_plan(doc, &opts(), &reg, &mut effects)?;
        Ok((p, effects))
    }

    #[test]
    fn happy_path_builds_expected_plan() {
        let doc = parse_doc(json!({
            "document": {
                "name": "doc",
                "path": "/d.txt",
                "chunks": [
                    {
                        "id": "c1",
                        "text": "Alice works at Acme.",
                        "entities": [
                            {"id": "e1", "type": "Person", "name": "Alice"},
                            {"id": "e2", "type": "Company", "name": "Acme"}
                        ],
                        "relations": [
                            {"from": "e1", "to": "e2", "type": "WORKS_AT"}
                        ]
                    }
                ]
            }
        }));
        let (plan, effects) = plan(&doc).unwrap();

        // Nodes: Document + Chunk + Person + Company.
        let labels: Vec<&str> = plan.nodes.iter().map(|n| n.label.as_str()).collect();
        assert!(labels.contains(&"Document"));
        assert!(labels.contains(&"Chunk"));
        assert!(labels.contains(&"Person"));
        assert!(labels.contains(&"Company"));

        // Relations: HAS_CHUNK + MENTIONS (Person + Company) + WORKS_AT.
        let rel_types: Vec<&str> =
            plan.relations.iter().map(|r| r.rel_type.as_str()).collect();
        assert!(rel_types.contains(&"HAS_CHUNK"));
        assert_eq!(
            rel_types.iter().filter(|t| **t == "MENTIONS").count(),
            2,
            "one MENTIONS plan per entity label"
        );
        assert!(rel_types.contains(&"WORKS_AT"));

        // Side effects: 1 chunk text + 2 entity names = 3 EmbedAndStore.
        assert_eq!(effects.len(), 3);
    }

    #[test]
    fn entity_names_are_embedded_per_label() {
        // Each entity `name` flows through the SemanticText handler, so
        // we expect one EmbedAndStore per entity, tagged with the entity
        // label as payload_label and `name` as the field meta.
        let doc = parse_doc(json!({
            "document": {
                "name": "d", "path": "/d",
                "chunks": [{
                    "id": "c1", "text": "...",
                    "entities": [
                        {"id": "e1", "type": "Person",  "name": "Alice"},
                        {"id": "e2", "type": "Company", "name": "Acme"}
                    ],
                    "relations": []
                }]
            }
        }));
        let (_p, effects) = plan(&doc).unwrap();

        let by_label: std::collections::BTreeMap<String, Vec<String>> = effects
            .iter()
            .map(|e| match e {
                SideEffect::EmbedAndStore { label, text, .. } => {
                    (label.clone(), text.clone())
                }
            })
            .fold(Default::default(), |mut acc, (label, text)| {
                acc.entry(label).or_default().push(text);
                acc
            });
        // 1 chunk + 1 person + 1 company.
        assert_eq!(by_label.get("Chunk").map(|v| v.len()), Some(1));
        assert_eq!(by_label.get("Person").map(|v| v.len()), Some(1));
        assert_eq!(by_label.get("Company").map(|v| v.len()), Some(1));
        assert!(by_label.get("Person").unwrap().contains(&"Alice".to_string()));
        assert!(by_label.get("Company").unwrap().contains(&"Acme".to_string()));
    }

    #[test]
    fn requires_semantic_text_handler() {
        // A registry without SemanticText must produce a clear error.
        let reg = RegistryBuilder::new().build();
        let mut effects = SideEffectQueue::new();
        let doc = parse_doc(json!({
            "document": {"name": "d", "path": "/d",
                "chunks": [{"id": "c1", "text": "x", "entities": [], "relations": []}]}
        }));
        let err = build_document_plan(&doc, &opts(), &reg, &mut effects).unwrap_err();
        match err {
            IngestError::Type(msg) => assert!(
                msg.contains("SemanticText"),
                "expected SemanticText-required message, got: {msg}"
            ),
            other => panic!("expected IngestError::Type, got {other:?}"),
        }
    }

    #[test]
    fn idempotent_uuids() {
        let doc1 = parse_doc(json!({
            "document": {
                "name": "d",
                "path": "/d.txt",
                "chunks": [{
                    "id": "c1",
                    "text": "x",
                    "entities": [{"id": "e1", "type": "Person", "name": "A"}],
                    "relations": []
                }]
            }
        }));
        let doc2 = doc1.clone();
        let (p1, _) = plan(&doc1).unwrap();
        let (p2, _) = plan(&doc2).unwrap();
        let id1 = match &p1.nodes.iter().find(|n| n.label == "Person").unwrap().rows[0].id {
            Literal::String(s) => s.clone(),
            _ => unreachable!(),
        };
        let id2 = match &p2.nodes.iter().find(|n| n.label == "Person").unwrap().rows[0].id {
            Literal::String(s) => s.clone(),
            _ => unreachable!(),
        };
        assert_eq!(id1, id2);
    }

    #[test]
    fn sanitizes_dynamic_labels_and_relations() {
        let doc = parse_doc(json!({
            "document": {
                "name": "d",
                "path": "/d.txt",
                "chunks": [{
                    "id": "c1",
                    "text": "x",
                    "entities": [
                        {"id": "e1", "type": "Music Group", "name": "Beatles"},
                        {"id": "e2", "type": "Person", "name": "John"}
                    ],
                    "relations": [
                        {"from": "e2", "to": "e1", "type": "member-of"}
                    ]
                }]
            }
        }));
        let (plan, _) = plan(&doc).unwrap();
        assert!(
            plan.nodes.iter().any(|n| n.label == "Music_Group"),
            "dynamic label with space should be sanitized; got {:?}",
            plan.nodes.iter().map(|n| &n.label).collect::<Vec<_>>()
        );
        assert!(
            plan.relations.iter().any(|r| r.rel_type == "MEMBER_OF"),
            "relation with hyphen should be sanitized + uppercased; got {:?}",
            plan.relations.iter().map(|r| &r.rel_type).collect::<Vec<_>>()
        );
    }

    #[test]
    fn reserved_label_rejected() {
        let doc = parse_doc(json!({
            "document": {
                "name": "d", "path": "/d",
                "chunks": [{
                    "id": "c1", "text": "x",
                    "entities": [{"id": "e1", "type": "Document", "name": "x"}],
                    "relations": []
                }]
            }
        }));
        let err = plan(&doc).unwrap_err();
        assert!(matches!(err, IngestError::ReservedLabel(s) if s == "Document"));
    }

    #[test]
    fn reserved_relation_rejected() {
        let doc = parse_doc(json!({
            "document": {
                "name": "d", "path": "/d",
                "chunks": [{
                    "id": "c1", "text": "x",
                    "entities": [
                        {"id": "e1", "type": "Person", "name": "A"},
                        {"id": "e2", "type": "Person", "name": "B"}
                    ],
                    "relations": [
                        {"from": "e1", "to": "e2", "type": "MENTIONS"}
                    ]
                }]
            }
        }));
        let err = plan(&doc).unwrap_err();
        assert!(matches!(err, IngestError::ReservedRelation(s) if s == "MENTIONS"));
    }

    #[test]
    fn dangling_local_id_rejected() {
        let doc = parse_doc(json!({
            "document": {
                "name": "d", "path": "/d",
                "chunks": [{
                    "id": "c1", "text": "x",
                    "entities": [{"id": "e1", "type": "Person", "name": "A"}],
                    "relations": [
                        {"from": "e1", "to": "e9", "type": "KNOWS"}
                    ]
                }]
            }
        }));
        let err = plan(&doc).unwrap_err();
        match err {
            IngestError::UnknownLocalId { chunk, local_id } => {
                assert_eq!(chunk, "c1");
                assert_eq!(local_id, "e9");
            }
            other => panic!("expected UnknownLocalId, got {other:?}"),
        }
    }

    #[test]
    fn label_starting_with_digit_gets_underscore_prefix() {
        let doc = parse_doc(json!({
            "document": {
                "name": "d", "path": "/d",
                "chunks": [{
                    "id": "c1", "text": "x",
                    "entities": [{"id": "e1", "type": "3D-Model", "name": "cube"}],
                    "relations": []
                }]
            }
        }));
        let (plan, _) = plan(&doc).unwrap();
        // "3D-Model" → "3D_Model" → "_3D_Model" (underscore-prefixed).
        assert!(
            plan.nodes.iter().any(|n| n.label == "_3D_Model"),
            "labels starting with digits must be prefixed with `_`; got {:?}",
            plan.nodes.iter().map(|n| &n.label).collect::<Vec<_>>()
        );
    }

    #[test]
    fn pure_symbol_label_rejected() {
        let doc = parse_doc(json!({
            "document": {
                "name": "d", "path": "/d",
                "chunks": [{
                    "id": "c1", "text": "x",
                    "entities": [{"id": "e1", "type": "---", "name": "x"}],
                    "relations": []
                }]
            }
        }));
        let err = plan(&doc).unwrap_err();
        assert!(matches!(err, IngestError::InvalidLabel(_)));
    }

    #[test]
    fn cross_chunk_entities_are_distinct() {
        // Same (type, name) in two chunks → two distinct nodes (no dedup).
        let doc = parse_doc(json!({
            "document": {
                "name": "d", "path": "/d",
                "chunks": [
                    {
                        "id": "c1", "text": "Alice 1",
                        "entities": [{"id": "e1", "type": "Person", "name": "Alice"}],
                        "relations": []
                    },
                    {
                        "id": "c2", "text": "Alice 2",
                        "entities": [{"id": "e1", "type": "Person", "name": "Alice"}],
                        "relations": []
                    }
                ]
            }
        }));
        let (plan, effects) = plan(&doc).unwrap();
        let person = plan.nodes.iter().find(|n| n.label == "Person").unwrap();
        assert_eq!(person.rows.len(), 2);
        let ids: Vec<&Literal> = person.rows.iter().map(|r| &r.id).collect();
        assert_ne!(ids[0], ids[1]);
        // 2 chunk-text + 2 entity-name embeddings.
        assert_eq!(effects.len(), 4);
    }
}
