//! `SemanticText` — free-text fields searchable via embeddings + qlink.
//!
//! There are exactly two ways to match a `SemanticText` field, and they
//! are kept deliberately separate:
//!
//! * **Exact / substring** (`eq` / `neq` / `contains`) — plain Cypher
//!   against the raw string kept on the node. No embeddings, no Qdrant,
//!   no fuzziness: `eq` means equality.
//! * **Semantic** (`search` / `search_reranked` / `hybrid_search`, and
//!   the resolver-synthesised `entity_search`) — a single per-entity
//!   hybrid retrieval (dense ⊕ BM25, RRF-fused) followed by one
//!   cross-encoder rerank, always against the field-agnostic
//!   `_canonical` collection.
//!
//! Responsibilities:
//!
//! 1. **Ingest**: keep the raw string on the node (so exact matches
//!    work) and — only for the field that backs vector retrieval (the
//!    per-entity `_canonical` document, or `text` on a `Chunk`) — queue
//!    a [`super::super::SideEffect::EmbedAndStore`]. Every other text
//!    field's value already lives inside `_canonical`, so embedding it
//!    separately would only duplicate the vector.
//! 2. **Lower**: for semantic ops, embed the query once and stash the
//!    vector in the predicate's `params` so emission stays pure; for
//!    exact ops, just carry the literal.
//! 3. **Emit**: render `libqlink.search_hybrid_reranked` over
//!    `_canonical` for semantic ops, or a plain `=`/`<>`/`CONTAINS`
//!    clause for exact ops.
//!
//! The handler is configured by a `[types.SemanticText]` block:
//!
//! ```toml
//! [types.SemanticText]
//! embedding_model    = "models/bge-small.gguf"
//! collection         = "companies"
//! top_k              = 20
//! threshold          = 0.8   # cosine cutoff for stage-1 KNN retrieval
//! reranker_threshold = 0.3   # final reranker score cutoff
//! ```
//!
//! ## qlink procedures used
//!
//! * **Ingest** — `libqlink.insert_labeled` so each vector carries the
//!   originating Cypher node label as a Qdrant payload tag. That lets a
//!   single embedding collection host multiple node labels safely while
//!   still being addressable by label at query time.
//! * **Search** — `libqlink.search_reranked` does a label-filtered KNN
//!   pre-filter (cosine ≥ `search_threshold`), looks up each surviving
//!   id as a Memgraph node, runs a cross-encoder reranker locally,
//!   and emits hits whose reranker score is ≥ `reranker_threshold`,
//!   sorted descending. We hand it the raw natural-language query
//!   (the DSL filter `value`) and the embedded vector — qlink does
//!   the rest.

use std::collections::BTreeMap;

use serde_json::Value;

use crate::ast::query::{Literal, PropertyRef};
use crate::config::Config;
use crate::embeddings::SharedEmbedder;
use crate::graph::{CANONICAL_FIELD, CHUNK_LABEL};
use crate::types::context::OrderDir;
use crate::types::{
    Capabilities, EmitCtx, IngestCtx, LowerCtx, PromptHint, SideEffect, TypeError, TypeHandler,
    TypeId, TypedOp, TypedPredicate,
};

/// Default cosine cutoff for stage-1 retrieval inside
/// `libqlink.search_reranked`. A modest 0.8 keeps obvious near-
/// duplicates in and aggressively prunes the long tail; raise it
/// for small corpora, lower it for noisy ones.
pub const DEFAULT_SEARCH_THRESHOLD: f64 = 0.8;

/// Default reranker score cutoff for stage-2 of
/// `libqlink.search_reranked`. Reranker scores are sigmoid-bounded
/// to `[0, 1]`; values around 0.3 keep recall sane on out-of-the-
/// box BGE rerankers.
pub const DEFAULT_RERANKER_THRESHOLD: f64 = 0.5;

/// Default number of RRF-fused candidates handed to the cross-encoder by
/// the consolidated [`TypedOp::EntitySearch`] path. This is the main
/// rerank-cost knob: the cross-encoder runs once over `candidate_k`
/// documents per entity alias, so keep it modest.
pub const DEFAULT_CANDIDATE_K: i64 = 40;

/// Configuration for [`SemanticTextHandler`].
#[derive(Debug, Clone)]
pub struct SemanticTextConfig {
    /// Path to the GGUF embedding model. Optional — if `None` the
    /// handler defers to the embedder it was built with (typically a
    /// [`crate::embeddings::MockEmbedder`] in tests).
    pub embedding_model: Option<String>,
    /// Default Qdrant collection used by ingestion and search. Each
    /// SemanticText property may override this in its mapping by
    /// providing `collection: <str>` in the type params.
    pub collection: String,
    /// Number of results to fan out from stage-1 KNN. Currently
    /// informational only — `libqlink.search_reranked` hard-codes
    /// the stage-1 fan-out internally — but kept on the config so
    /// the field-types prompt block can still advertise it.
    pub top_k: u32,
    /// Cosine threshold for stage-1 KNN retrieval. Defaults to
    /// [`DEFAULT_SEARCH_THRESHOLD`].
    pub search_threshold: f64,
    /// Reranker threshold applied to the cross-encoder score in
    /// stage 2. Defaults to [`DEFAULT_RERANKER_THRESHOLD`].
    pub reranker_threshold: f64,
}

impl SemanticTextConfig {
    pub fn from_config(cfg: &Config) -> Option<Self> {
        cfg.types.get("SemanticText").map(|t| Self {
            embedding_model: t.embedding_model.clone(),
            collection: t
                .collection
                .clone()
                .unwrap_or_else(|| "semantic_text".into()),
            top_k: t.top_k.unwrap_or(20),
            // `threshold` in TOML refers to the cosine cutoff —
            // matches what was historically the only knob.
            search_threshold: t.threshold.unwrap_or(DEFAULT_SEARCH_THRESHOLD),
            reranker_threshold: t.reranker_threshold.unwrap_or(DEFAULT_RERANKER_THRESHOLD),
        })
    }
}

/// Handler that embeds string fields and exposes them via qlink.
pub struct SemanticTextHandler {
    config: SemanticTextConfig,
    embedder: SharedEmbedder,
}

impl std::fmt::Debug for SemanticTextHandler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SemanticTextHandler")
            .field("config", &self.config)
            .field("embedder_dim", &self.embedder.dim())
            .finish()
    }
}

impl SemanticTextHandler {
    pub const TYPE_ID: &'static str = "SemanticText";

    pub fn new(config: SemanticTextConfig, embedder: SharedEmbedder) -> Self {
        Self { config, embedder }
    }

    pub fn config(&self) -> &SemanticTextConfig {
        &self.config
    }
}

impl TypeHandler for SemanticTextHandler {
    fn type_id(&self) -> TypeId {
        TypeId::new(Self::TYPE_ID)
    }

    fn capabilities(&self) -> Capabilities {
        Capabilities::INGEST
            | Capabilities::EXACT_MATCH
            | Capabilities::SEMANTIC_SEARCH
            | Capabilities::HYBRID_SEARCH
            | Capabilities::CONTAINS
    }

    fn supported_ops(&self) -> Vec<TypedOp> {
        vec![
            TypedOp::Eq,
            TypedOp::Neq,
            TypedOp::Contains,
            TypedOp::Search,
            TypedOp::SearchReranked,
            TypedOp::HybridSearch,
        ]
    }

    fn on_ingest(&self, ctx: &mut IngestCtx<'_>) -> Result<(), TypeError> {
        let text = match ctx.value() {
            Value::String(s) => s.clone(),
            Value::Null => return Ok(()),
            other => {
                return Err(TypeError::InvalidValue {
                    ty: Self::TYPE_ID.into(),
                    reason: format!("SemanticText expects string, got {}", json_kind(other)),
                });
            }
        };

        // Keep the raw text on the node — useful for exact match,
        // contains, and human inspection. This happens for *every*
        // SemanticText field, even the ones we don't embed.
        ctx.set_value(Literal::String(text.clone()));

        // Only one field per node backs vector retrieval: the per-entity
        // `_canonical` document (which already concatenates every text
        // property), or `text` on a `Chunk`. Embedding any other field
        // would just duplicate a vector that `_canonical` already covers,
        // so we keep its value on the node but queue no side effect.
        let should_embed = if ctx.node_label == CHUNK_LABEL {
            ctx.field_name == "text"
        } else {
            ctx.field_name == CANONICAL_FIELD
        };
        if !should_embed {
            return Ok(());
        }

        // Queue the embed-and-store side effect. The collection name is
        // derived from the configured default plus the field name so
        // `_canonical` and `Chunk.text` land in distinct collections.
        // When a `prefix_index` is set, it's folded in as the outermost
        // segment so the same field in different prefixes lands in
        // separate Qdrant collections.
        let collection = with_prefix_index(
            ctx.prefix_index,
            &format!("{}__{}", self.config.collection, ctx.field_name),
        );
        ctx.push_side_effect(SideEffect::EmbedAndStore {
            collection,
            label: ctx.node_label.to_string(),
            key_field: ctx.node_key_field.to_string(),
            key_value: ctx.node_key.clone(),
            text,
            payload_label: Some(ctx.node_label.to_string()),
            meta: {
                let mut m = BTreeMap::new();
                m.insert("type".into(), Self::TYPE_ID.into());
                m.insert("field".into(), ctx.field_name.into());
                m
            },
        });

        Ok(())
    }

    fn lower(&self, ctx: &mut LowerCtx<'_>) -> Result<TypedPredicate, TypeError> {
        let text = ctx
            .raw
            .value
            .as_str()
            .ok_or_else(|| TypeError::InvalidValue {
                ty: Self::TYPE_ID.into(),
                reason: format!(
                    "expected string value for op {}, got {}",
                    ctx.raw.op,
                    json_kind(ctx.raw.value)
                ),
            })?;

        // Exact / substring ops never touch the vector store: emit()
        // renders plain Cypher against the raw string `on_ingest` kept on
        // the node, so `eq`/`neq`/`contains` stay precise. No embedding,
        // no params.
        if matches!(
            ctx.raw.op,
            TypedOp::Eq | TypedOp::Neq | TypedOp::Contains
        ) {
            return Ok(TypedPredicate {
                type_id: ctx.type_id.clone(),
                field: ctx.raw.field.clone(),
                op: ctx.raw.op,
                value: Literal::String(text.to_string()),
                params: BTreeMap::new(),
            });
        }

        // Every other (semantic) op resolves to the same retrieval: a
        // single per-entity hybrid search (dense ⊕ BM25, RRF-fused) over
        // the field-agnostic `_canonical` collection, then one
        // cross-encoder rerank. We embed the query once here so emit()
        // stays pure (no I/O) and the builder is testable in isolation.
        let vec = self
            .embedder
            .embed(text)
            .map_err(|e| TypeError::Embedder(e.to_string()))?;
        let lit_vec = Literal::List(vec.into_iter().map(|f| Literal::Float(f as f64)).collect());

        // The reranker needs both the textual query (used to build the
        // cross-encoder prompt) AND its embedding (used for stage-1
        // KNN). The Cypher node label is the qlink payload filter,
        // matching what `on_ingest` wrote via `insert_hybrid`.
        let label = ctx.field_label.ok_or_else(|| {
            TypeError::Handler(
                "SemanticText: cannot resolve graph label for field; \
             alias is not bound to a node/edge in the AST"
                    .into(),
            )
        })?;

        // Always the per-entity `_canonical` collection. The base +
        // prefix are identical to ingest, so query and ingest always
        // address the same Qdrant collection.
        let collection = canonical_collection_for(self, ctx.prefix_index);

        let mut params = BTreeMap::new();
        params.insert("embedding".to_string(), lit_vec);
        params.insert("collection".to_string(), Literal::String(collection));
        params.insert("query_str".to_string(), Literal::String(text.to_string()));
        params.insert("candidate_k".to_string(), Literal::Int(DEFAULT_CANDIDATE_K));
        params.insert("label".to_string(), Literal::String(label.to_string()));
        params.insert(
            "reranker_threshold".to_string(),
            Literal::Float(self.config.reranker_threshold),
        );

        Ok(TypedPredicate {
            type_id: ctx.type_id.clone(),
            field: ctx.raw.field.clone(),
            op: ctx.raw.op,
            value: Literal::String(text.to_string()),
            params,
        })
    }

    fn emit(&self, ctx: &mut EmitCtx<'_>, pred: &TypedPredicate) -> Result<(), TypeError> {
        match pred.op {
            // ── Exact / substring: precise plain Cypher. ──────────────
            //
            // The raw value lives on the node (see `on_ingest`), so these
            // never touch the vector store. `eq` is equality, `contains`
            // is substring — no embeddings, no fuzziness.
            TypedOp::Eq | TypedOp::Neq | TypedOp::Contains => {
                let placeholder = ctx.bind(pred.value.clone());
                let sym = match pred.op {
                    TypedOp::Eq => "=",
                    TypedOp::Neq => "<>",
                    TypedOp::Contains => "CONTAINS",
                    _ => unreachable!("guarded by the match arm"),
                };
                ctx.set_where(format!(
                    "{lhs} {sym} {placeholder}",
                    lhs = render_field(&pred.field)
                ));
                Ok(())
            }

            // ── Semantic: one per-entity hybrid search + single rerank. ─
            //
            // Every semantic op (plain `search`, `search_reranked`,
            // `hybrid_search`, and the resolver-synthesised
            // `entity_search`) shares this one implementation. One CALL
            // per alias against the `_canonical` collection: qlink fuses a
            // dense and a BM25-sparse branch with RRF, label-filters by
            // entity type, then reranks the top `candidate_k` candidates
            // over their canonical text. The yield is the surviving
            // (id, reranker_score) pairs descending; the MATCH then joins
            // by id, so we need no Cypher-side threshold or WITH gate.
            TypedOp::Search
            | TypedOp::SearchReranked
            | TypedOp::HybridSearch
            | TypedOp::EntitySearch => {
                let alias = pred.field.alias.as_str();
                let coll = pred
                    .params
                    .get("collection")
                    .cloned()
                    .ok_or_else(|| TypeError::Handler("missing 'collection' param".into()))?;
                let emb = pred
                    .params
                    .get("embedding")
                    .cloned()
                    .ok_or_else(|| TypeError::Handler("missing 'embedding' param".into()))?;
                let query_str = pred
                    .params
                    .get("query_str")
                    .cloned()
                    .ok_or_else(|| TypeError::Handler("missing 'query_str' param".into()))?;
                let label = pred
                    .params
                    .get("label")
                    .cloned()
                    .ok_or_else(|| TypeError::Handler("missing 'label' param".into()))?;
                let rerank_thr = pred
                    .params
                    .get("reranker_threshold")
                    .cloned()
                    .unwrap_or(Literal::Float(DEFAULT_RERANKER_THRESHOLD));
                let candidate_k = pred
                    .params
                    .get("candidate_k")
                    .cloned()
                    .unwrap_or(Literal::Int(DEFAULT_CANDIDATE_K));

                let coll_p = ctx.bind(coll);
                let q_p = ctx.bind(query_str);
                let emb_p = ctx.bind(emb);
                let label_p = ctx.bind(label);
                let thr_p = ctx.bind(rerank_thr);
                let k_p = ctx.bind(candidate_k);
                // Each call gets a unique suffix so multiple semantic
                // searches against the same alias don't collide on
                // `<alias>__qid` / `<alias>__score`.
                let n = ctx.fresh_id();
                let qid = format!("{alias}__qid_{n}");
                let score = format!("{alias}__score_{n}");
                ctx.push_pre_match(format!(
                    "CALL libqlink.search_hybrid_reranked({coll_p}, {q_p}, {emb_p}, {label_p}, {thr_p}, {k_p}) \
                     YIELD id AS {qid}, score AS {score}"
                ));
                ctx.set_where(format!("id({alias}) = {qid}"));
                ctx.contribution_mut()
                    .order_by
                    .push((score, OrderDir::Desc));
                Ok(())
            }

            other => Err(TypeError::UnsupportedOp {
                ty: Self::TYPE_ID.into(),
                op: other.to_string(),
            }),
        }
    }

    fn prompt_hint(&self) -> PromptHint {
        PromptHint {
            type_id: self.type_id(),
            capabilities: self.capabilities(),
            ops: self.supported_ops(),
            doc: Some(
                "Free-text field. Use `eq`/`neq`/`contains` for precise (non-fuzzy) lookups, \
                 and `search` for natural-language matches (a hybrid dense+keyword retrieval \
                 over the whole entity, with cross-encoder reranking)."
                    .into(),
            ),
            example: Some(
                r#"{"field":"c.name","type":"SemanticText","op":"search","value":"apple"}"#.into(),
            ),
        }
    }
}

fn json_kind(v: &Value) -> &'static str {
    match v {
        Value::Null => "null",
        Value::Bool(_) => "bool",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}

/// Render the left-hand side of an exact-match clause: `alias.property`
/// when the filter names a property, or just `alias` for an entity-level
/// reference.
fn render_field(field: &PropertyRef) -> String {
    match &field.property {
        Some(prop) => format!("{}.{}", field.alias, prop),
        None => field.alias.to_string(),
    }
}

/// Per-entity `_canonical` collection name. Mirrors how the ingest path
/// (`Pipeline::semantic_collection`) derives it — `{base}__{_canonical}`,
/// which renders with the literal triple underscore (e.g.
/// `semantic_text___canonical`) — so the consolidated query searches the
/// very collection the canonical embeddings were written to.
fn canonical_collection_for(h: &SemanticTextHandler, prefix_index: Option<&str>) -> String {
    with_prefix_index(
        prefix_index,
        &format!("{}__{}", h.config.collection, "_canonical"),
    )
}

/// Build the combined query string for a consolidated
/// [`TypedOp::EntitySearch`] from all the SemanticText filter terms on one
/// entity alias.
///
/// When `field_aware` is set and every term carries a property name, the
/// string mirrors the indexed `_canonical` document format
/// (`crate::graph::canonical::build_canonical_text`) — `type: {label}`
/// followed by `prop: value` lines in sorted order — so the cross-encoder
/// sees the query and the documents in the same shape. Otherwise (or when
/// a term has no property) it falls back to a values-only string
/// `"{label}: v1 | v2"`, which is robust to the DSL model attributing a
/// value to the wrong field.
pub fn build_canonical_query(
    label: &str,
    terms: &[(Option<String>, String)],
    field_aware: bool,
) -> String {
    let all_named = !terms.is_empty() && terms.iter().all(|(p, _)| p.is_some());
    if field_aware && all_named {
        let mut lines: Vec<(&str, &str)> = terms
            .iter()
            .map(|(p, v)| (p.as_deref().unwrap_or(""), v.as_str()))
            .collect();
        lines.sort_by(|a, b| a.0.cmp(b.0));
        let mut out = format!("type: {label}");
        for (p, v) in lines {
            out.push('\n');
            out.push_str(p);
            out.push_str(": ");
            out.push_str(v);
        }
        out
    } else {
        let values: Vec<&str> = terms.iter().map(|(_, v)| v.as_str()).collect();
        format!("{label}: {}", values.join(" | "))
    }
}

/// Fold an optional prefix into a Qdrant collection name. Empty
/// prefixes are normalised to "no prefix" so call sites don't need to
/// distinguish `Some("")` from `None`.
pub(crate) fn with_prefix_index(prefix_index: Option<&str>, base: &str) -> String {
    match prefix_index {
        Some(p) if !p.is_empty() => format!("{p}__{base}"),
        _ => base.to_string(),
    }
}

/// Errors produced by [`build_embed_insert_batch`]. Kept separate from
/// `TypeError` because the only failure modes are static-identifier
/// validation on labels and key fields — i.e. malformed input from the
/// side-effect queue, not handler logic.
#[derive(Debug, thiserror::Error)]
pub enum SideEffectEmitError {
    #[error("invalid label '{0}' in side effect")]
    InvalidLabel(String),

    #[error("invalid key field '{0}' in side effect")]
    InvalidKeyField(String),
}

/// Render an `UNWIND $rows AS row | MATCH ... CALL libqlink.insert_labeled
/// ...` Cypher batch for one homogeneous group of [`SideEffect::EmbedAndStore`]
/// side effects.
///
/// All effects in `group` must share the same Cypher `label`, the same
/// `key_field`, the same `collection`, and the same `payload_label`
/// (the caller groups by exactly these). The MATCH pattern is therefore
/// consistent across rows; only `key`/`vec` varies per row.
///
/// When the bucket has a `payload_label`, we use
/// `libqlink.insert_labeled` so each vector lands in Qdrant tagged
/// with the originating Cypher node label — that's what
/// `libqlink.search_hybrid_reranked` filters by at query time. When the
/// bucket has no label we fall back to plain `libqlink.insert`.
///
/// This Cypher renderer belongs to the SemanticText handler because
/// only this handler knows what shape an embedding side effect takes;
/// keeping it in `core::Pipeline` would couple the orchestration
/// layer to qlink-specific procedures.
pub fn build_embed_insert_batch(
    group: &[(SideEffect, Vec<f32>)],
) -> Result<crate::builder::CypherQuery, SideEffectEmitError> {
    use std::collections::BTreeMap;
    debug_assert!(!group.is_empty(), "callers must not pass an empty group");

    // All rows in `group` share these — see `Pipeline::drain_side_effects`.
    let (collection, payload_label, label, key_field) = match &group[0].0 {
        SideEffect::EmbedAndStore {
            collection,
            label,
            key_field,
            payload_label,
            ..
        } => (
            collection.clone(),
            payload_label.clone(),
            label.clone(),
            key_field.clone(),
        ),
    };

    if !is_valid_ident(&label) {
        return Err(SideEffectEmitError::InvalidLabel(label));
    }
    if !is_valid_ident(&key_field) {
        return Err(SideEffectEmitError::InvalidKeyField(key_field));
    }

    // Build the row payload. Each row is `{key: <pk>, vec: <embedding>,
    // text: <source text>}`. The text feeds `libqlink.insert_hybrid`,
    // which derives the BM25 sparse vector from it and stores it in the
    // point payload for reranking.
    let mut rows: Vec<Literal> = Vec::with_capacity(group.len());
    for (eff, vec) in group {
        let SideEffect::EmbedAndStore {
            key_value, text, ..
        } = eff;
        let mut row: BTreeMap<String, Literal> = BTreeMap::new();
        row.insert("key".to_string(), key_value.clone());
        row.insert(
            "vec".to_string(),
            Literal::List(vec.iter().map(|f| Literal::Float(*f as f64)).collect()),
        );
        row.insert("text".to_string(), Literal::String(text.clone()));
        rows.push(Literal::Object(row));
    }

    let mut params: BTreeMap<String, Literal> = BTreeMap::new();
    params.insert("coll".to_string(), Literal::String(collection));
    params.insert("rows".to_string(), Literal::List(rows));

    // Labeled inserts (every SemanticText field, including `_canonical`)
    // go through `insert_hybrid` so the collection carries both a dense
    // and a BM25 sparse vector plus the canonical text in its payload.
    // The unlabeled fallback stays dense-only (`insert`) — no handler
    // queues a label-less embedding today, so it never carries BM25.
    let cypher = if let Some(plabel) = payload_label {
        params.insert("label".to_string(), Literal::String(plabel));
        format!(
            "UNWIND $rows AS row\n\
             MATCH (n:{label} {{{key_field}: row.key}})\n\
             CALL libqlink.insert_hybrid($coll, id(n), row.vec, row.text, $label) YIELD success\n\
             RETURN count(success) AS inserted",
        )
    } else {
        format!(
            "UNWIND $rows AS row\n\
             MATCH (n:{label} {{{key_field}: row.key}})\n\
             CALL libqlink.insert($coll, id(n), row.vec) YIELD success\n\
             RETURN count(success) AS inserted",
        )
    };
    Ok(crate::builder::CypherQuery::new(cypher, params))
}

fn is_valid_ident(s: &str) -> bool {
    let mut chars = s.chars();
    let first = chars.next();
    matches!(first, Some(c) if c.is_ascii_alphabetic() || c == '_')
        && chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::query::{Alias, PropertyRef};
    use crate::embeddings::MockEmbedder;
    use crate::types::context::{
        CypherContribution, IngestCtx as IC, LowerCtx as LC, ParamBinder, RawTypedFilter,
    };
    use crate::types::SideEffectQueue;
    use std::sync::Arc;

    fn handler() -> SemanticTextHandler {
        handler_with_thresholds(DEFAULT_SEARCH_THRESHOLD, DEFAULT_RERANKER_THRESHOLD)
    }

    fn handler_with_thresholds(
        search_threshold: f64,
        reranker_threshold: f64,
    ) -> SemanticTextHandler {
        SemanticTextHandler::new(
            SemanticTextConfig {
                embedding_model: None,
                collection: "test".into(),
                top_k: 10,
                search_threshold,
                reranker_threshold,
            },
            Arc::new(MockEmbedder::new(8)),
        )
    }

    /// Helper to build a `LowerCtx` for the unit tests. The
    /// production path always populates `field_label` (the AST
    /// resolves the alias before the handler runs); the tests do the
    /// same so they exercise the same code path.
    fn lc<'a>(field: &'a PropertyRef, op: TypedOp, value: &'a Value, label: &'a str) -> LC<'a> {
        LC {
            raw: RawTypedFilter { field, op, value },
            type_id: TypeId::new(SemanticTextHandler::TYPE_ID),
            field_label: Some(label),
            prefix_index: None,
        }
    }

    fn pref(alias: &str, prop: &str) -> PropertyRef {
        PropertyRef {
            alias: Alias::new(alias),
            property: Some(prop.into()),
        }
    }

    struct CountingBinder {
        next: usize,
        next_var: usize,
        params: std::collections::BTreeMap<String, Literal>,
    }
    impl ParamBinder for CountingBinder {
        fn bind(&mut self, v: Literal) -> String {
            let n = format!("p{}", self.next);
            self.next += 1;
            self.params.insert(n.clone(), v);
            format!("${n}")
        }
        fn fresh_id(&mut self) -> usize {
            let id = self.next_var;
            self.next_var += 1;
            id
        }
    }

    #[test]
    fn ingest_embeds_canonical_field() {
        // The `_canonical` field is the one that backs vector retrieval,
        // so it both stores its value and queues an embedding.
        let h = handler();
        let mut q = SideEffectQueue::new();
        let key = Literal::String("type: Company\nname: Hello".into());
        let raw = serde_json::json!("type: Company\nname: Hello");
        let mut ctx = IC::new("Company", CANONICAL_FIELD, &key, CANONICAL_FIELD, &raw, &mut q);
        h.on_ingest(&mut ctx).unwrap();
        let stored = ctx.finish();
        assert_eq!(
            stored,
            Some(Some(Literal::String("type: Company\nname: Hello".into())))
        );
        assert_eq!(q.len(), 1);
        match &q.into_vec()[0] {
            SideEffect::EmbedAndStore {
                label, collection, ..
            } => {
                assert_eq!(label, "Company");
                // Without a prefix_index the collection name is just
                // `<base>__<field>`; `_canonical` keeps its leading
                // underscore, so the separator renders as `___`.
                assert_eq!(collection, "test___canonical");
            }
        }
    }

    #[test]
    fn ingest_keeps_value_but_does_not_embed_non_canonical_fields() {
        // A plain Text field (e.g. `name`) keeps its raw value on the node
        // for exact match, but is NOT embedded — its value already lives
        // inside `_canonical`, so a per-field vector would just duplicate.
        let h = handler();
        let mut q = SideEffectQueue::new();
        let key = Literal::String("c1".into());
        let raw = serde_json::json!("Hello world");
        let mut ctx = IC::new("Company", "id", &key, "name", &raw, &mut q);
        h.on_ingest(&mut ctx).unwrap();
        let stored = ctx.finish();
        assert_eq!(stored, Some(Some(Literal::String("Hello world".into()))));
        assert_eq!(q.len(), 0, "non-canonical fields must not queue an embedding");
    }

    #[test]
    fn ingest_embeds_chunk_text_field() {
        // Chunks are the one exception: their `text` field is the
        // retrieval key (the traversal pipeline searches the `text`
        // collection), so it is embedded while `_canonical` is not.
        let h = handler();
        let mut q = SideEffectQueue::new();
        let key = Literal::String("chunk-1".into());
        let raw = serde_json::json!("a fragment");
        let mut ctx = IC::new(CHUNK_LABEL, "id", &key, "text", &raw, &mut q);
        h.on_ingest(&mut ctx).unwrap();
        ctx.finish();
        assert_eq!(q.len(), 1);
        match &q.into_vec()[0] {
            SideEffect::EmbedAndStore { collection, .. } => {
                assert_eq!(collection, "test__text");
            }
        }
    }

    #[test]
    fn ingest_prefix_index_scopes_embedding_collection() {
        let h = handler();
        let mut q = SideEffectQueue::new();
        let key = Literal::String("type: Company".into());
        let raw = serde_json::json!("type: Company");
        let mut ctx = IC::new("Company", CANONICAL_FIELD, &key, CANONICAL_FIELD, &raw, &mut q)
            .with_prefix_index(Some("Tenant1"));
        h.on_ingest(&mut ctx).unwrap();
        ctx.finish();
        match &q.into_vec()[0] {
            SideEffect::EmbedAndStore { collection, .. } => {
                assert_eq!(collection, "Tenant1__test___canonical");
            }
        }
    }

    #[test]
    fn lower_prefix_index_propagates_into_collection_param() {
        let h = handler();
        let field = pref("c", "name");
        let value = serde_json::json!("apple");
        let mut ctx = LC {
            raw: RawTypedFilter {
                field: &field,
                op: TypedOp::Search,
                value: &value,
            },
            type_id: TypeId::new(SemanticTextHandler::TYPE_ID),
            field_label: Some("Company"),
            prefix_index: Some("Tenant1"),
        };
        let pred = h.lower(&mut ctx).unwrap();
        // Semantic ops always target the per-entity `_canonical` collection.
        assert_eq!(
            pred.params.get("collection").unwrap(),
            &Literal::String("Tenant1__test___canonical".into())
        );
    }

    #[test]
    fn lower_search_embeds_query_and_records_canonical_params() {
        let h = handler();
        let field = pref("c", "name");
        let value = serde_json::json!("apple");
        let mut ctx = lc(&field, TypedOp::Search, &value, "Company");
        let pred = h.lower(&mut ctx).unwrap();

        // Everything the canonical hybrid search needs is in params.
        for key in [
            "embedding",
            "collection",
            "query_str",
            "candidate_k",
            "label",
            "reranker_threshold",
        ] {
            assert!(pred.params.contains_key(key), "missing '{key}' in params");
        }
        // The cosine `search_threshold`/`top_k` knobs aren't used by
        // search_hybrid_reranked, so they're no longer carried.
        assert!(!pred.params.contains_key("search_threshold"));
        assert!(!pred.params.contains_key("top_k"));
        match pred.params.get("embedding").unwrap() {
            Literal::List(items) => assert_eq!(items.len(), 8),
            _ => panic!("embedding should be a List"),
        }
        assert_eq!(
            pred.params.get("query_str").unwrap(),
            &Literal::String("apple".into())
        );
        assert_eq!(
            pred.params.get("collection").unwrap(),
            &Literal::String("test___canonical".into())
        );
        assert_eq!(
            pred.params.get("label").unwrap(),
            &Literal::String("Company".into())
        );
        assert_eq!(
            pred.params.get("reranker_threshold").unwrap(),
            &Literal::Float(DEFAULT_RERANKER_THRESHOLD)
        );
    }

    #[test]
    fn lower_search_without_field_label_errors_loudly() {
        let h = handler();
        let field = pref("c", "name");
        let value = serde_json::json!("apple");
        let mut ctx = LC {
            raw: RawTypedFilter {
                field: &field,
                op: TypedOp::Search,
                value: &value,
            },
            type_id: TypeId::new(SemanticTextHandler::TYPE_ID),
            field_label: None,
            prefix_index: None,
        };
        let err = h.lower(&mut ctx).unwrap_err();
        assert!(
            matches!(err, TypeError::Handler(msg) if msg.contains("graph label")),
            "expected handler-error about missing label"
        );
    }

    #[test]
    fn lower_eq_is_exact_with_no_embedding() {
        // `eq`/`neq`/`contains` are exact ops: they carry only the literal
        // value and queue no embedding work — emit() renders plain Cypher.
        let h = handler();
        let field = pref("c", "name");
        let value = serde_json::json!("apple");
        let mut ctx = lc(&field, TypedOp::Eq, &value, "Company");
        let pred = h.lower(&mut ctx).unwrap();
        assert_eq!(pred.value, Literal::String("apple".into()));
        assert!(
            pred.params.is_empty(),
            "exact ops must not embed or carry search params; got {:?}",
            pred.params
        );
    }

    #[test]
    fn emit_eq_renders_plain_cypher_equality() {
        let h = handler();
        let field = pref("c", "name");
        let value = serde_json::json!("apple");
        let mut lower = lc(&field, TypedOp::Eq, &value, "Company");
        let pred = h.lower(&mut lower).unwrap();

        let mut contrib = CypherContribution::default();
        let mut binder = CountingBinder {
            next: 0,
            next_var: 0,
            params: Default::default(),
        };
        let mut emit = EmitCtx::new(&mut contrib, &mut binder);
        h.emit(&mut emit, &pred).unwrap();

        // Plain WHERE clause against the raw property — no qlink, no
        // pre_match prelude.
        assert_eq!(contrib.where_inline.as_deref(), Some("c.name = $p0"));
        assert!(contrib.pre_match.is_empty());
        assert!(contrib.order_by.is_empty());
        assert_eq!(binder.params.get("p0"), Some(&Literal::String("apple".into())));
    }

    #[test]
    fn emit_contains_renders_plain_cypher_contains() {
        let h = handler();
        let field = pref("c", "name");
        let value = serde_json::json!("app");
        let mut lower = lc(&field, TypedOp::Contains, &value, "Company");
        let pred = h.lower(&mut lower).unwrap();

        let mut contrib = CypherContribution::default();
        let mut binder = CountingBinder {
            next: 0,
            next_var: 0,
            params: Default::default(),
        };
        let mut emit = EmitCtx::new(&mut contrib, &mut binder);
        h.emit(&mut emit, &pred).unwrap();
        assert_eq!(contrib.where_inline.as_deref(), Some("c.name CONTAINS $p0"));
        assert!(contrib.pre_match.is_empty());
    }

    #[test]
    fn emit_search_calls_hybrid_reranked_and_orders_by_score() {
        // Every semantic op compiles to the same per-entity hybrid
        // retrieval + rerank over `_canonical`, exposing the surviving
        // (id, score) pairs descending.
        let h = handler();
        let field = pref("c", "name");
        let value = serde_json::json!("apple");
        let mut lower = lc(&field, TypedOp::Search, &value, "Company");
        let pred = h.lower(&mut lower).unwrap();

        let mut contrib = CypherContribution::default();
        let mut binder = CountingBinder {
            next: 0,
            next_var: 0,
            params: Default::default(),
        };
        let mut emit = EmitCtx::new(&mut contrib, &mut binder);
        h.emit(&mut emit, &pred).unwrap();

        let pre = contrib.pre_match.join("\n");
        assert!(
            pre.contains("CALL libqlink.search_hybrid_reranked("),
            "pre_match should call libqlink.search_hybrid_reranked; got {pre}"
        );
        assert!(pre.contains("YIELD id AS c__qid_0, score AS c__score_0"));
        // No post-yield WHERE — qlink filters internally.
        assert!(
            !pre.contains("WHERE c__score"),
            "must not emit a duplicate score-filter clause; got {pre}"
        );
        assert_eq!(contrib.where_inline.as_deref(), Some("id(c) = c__qid_0"));
        assert_eq!(contrib.order_by.len(), 1);
        assert_eq!(contrib.order_by[0].0, "c__score_0");
    }

    #[test]
    fn emit_search_reranked_threshold_is_bound_as_parameter() {
        // The reranker threshold is bound as a Cypher parameter, never
        // inlined into the `search_hybrid_reranked` call site.
        let h = handler_with_thresholds(0.42, 0.17);
        let field = pref("c", "name");
        let value = serde_json::json!("apple");
        let mut lower = lc(&field, TypedOp::SearchReranked, &value, "Company");
        let pred = h.lower(&mut lower).unwrap();

        let mut contrib = CypherContribution::default();
        let mut binder = CountingBinder {
            next: 0,
            next_var: 0,
            params: Default::default(),
        };
        let mut emit = EmitCtx::new(&mut contrib, &mut binder);
        h.emit(&mut emit, &pred).unwrap();

        let floats: Vec<f64> = binder
            .params
            .values()
            .filter_map(|v| {
                if let Literal::Float(f) = v {
                    Some(*f)
                } else {
                    None
                }
            })
            .collect();
        assert!(
            floats.iter().any(|f| (f - 0.17).abs() < 1e-9),
            "reranker_threshold 0.17 not bound; floats={floats:?}"
        );
        let pre = contrib.pre_match.join("\n");
        assert!(pre.contains("libqlink.search_hybrid_reranked"));
        assert!(
            !pre.contains("0.17"),
            "reranker_threshold leaked inline: {pre}"
        );
    }

    #[test]
    fn emit_hybrid_search_routes_to_canonical_hybrid() {
        // `hybrid_search` is now an alias for the unified semantic path:
        // it compiles to `search_hybrid_reranked` over `_canonical`, not
        // the old inline exact + score_batch_node scheme.
        let h = handler();
        let field = pref("c", "name");
        let value = serde_json::json!("apple");
        let mut lower = lc(&field, TypedOp::HybridSearch, &value, "Company");
        let pred = h.lower(&mut lower).unwrap();
        let mut contrib = CypherContribution::default();
        let mut binder = CountingBinder {
            next: 0,
            next_var: 0,
            params: Default::default(),
        };
        let mut emit = EmitCtx::new(&mut contrib, &mut binder);
        h.emit(&mut emit, &pred).unwrap();

        let pre = contrib.pre_match.join("\n");
        assert!(pre.contains("CALL libqlink.search_hybrid_reranked("));
        assert!(contrib.post_match.is_empty());
        assert!(
            !pre.contains("score_batch_node"),
            "hybrid_search must no longer emit score_batch_node; got {pre}"
        );
        assert_eq!(contrib.order_by.len(), 1);
        assert_eq!(contrib.order_by[0].0, "c__score_0");
    }

    #[test]
    fn build_canonical_query_mirrors_canonical_when_field_aware() {
        // Field-aware: mirrors `_canonical` — `type: <Label>` then sorted
        // `prop: value` lines (address before name).
        let terms = vec![
            (Some("name".to_string()), "office".to_string()),
            (Some("address".to_string()), "Астана мангелик".to_string()),
        ];
        assert_eq!(
            build_canonical_query("Place", &terms, true),
            "type: Place\naddress: Астана мангелик\nname: office"
        );
    }

    #[test]
    fn build_canonical_query_falls_back_to_values_only() {
        // A term without a property name forces the values-only fallback,
        // which is robust to the DSL model attributing a value to the
        // wrong field.
        let mixed = vec![
            (None, "office".to_string()),
            (Some("address".to_string()), "Астана".to_string()),
        ];
        assert_eq!(
            build_canonical_query("Place", &mixed, true),
            "Place: office | Астана"
        );
        // Disabling field_aware also yields values-only.
        let named = vec![(Some("name".to_string()), "office".to_string())];
        assert_eq!(
            build_canonical_query("Place", &named, false),
            "Place: office"
        );
    }

    #[test]
    fn lower_entity_search_targets_canonical_collection() {
        let h = handler();
        let field = pref("p", "name");
        let value = serde_json::json!("type: Place\nname: office");
        let mut ctx = lc(&field, TypedOp::EntitySearch, &value, "Place");
        let pred = h.lower(&mut ctx).unwrap();
        // EntitySearch searches the `_canonical` collection (triple
        // underscore: base `test` + `_canonical`), not the per-field one.
        assert_eq!(
            pred.params.get("collection").unwrap(),
            &Literal::String("test___canonical".into())
        );
        assert!(pred.params.contains_key("candidate_k"));
        // The query string round-trips as `query_str` for the reranker.
        assert_eq!(
            pred.params.get("query_str").unwrap(),
            &Literal::String("type: Place\nname: office".into())
        );
    }

    #[test]
    fn emit_entity_search_renders_hybrid_reranked_call() {
        let h = handler();
        let field = pref("p", "name");
        let value = serde_json::json!("type: Place\nname: office");
        let mut lower = lc(&field, TypedOp::EntitySearch, &value, "Place");
        let pred = h.lower(&mut lower).unwrap();

        let mut contrib = CypherContribution::default();
        let mut binder = CountingBinder {
            next: 0,
            next_var: 0,
            params: Default::default(),
        };
        let mut emit = EmitCtx::new(&mut contrib, &mut binder);
        h.emit(&mut emit, &pred).unwrap();

        let pre = contrib.pre_match.join("\n");
        assert!(
            pre.contains("CALL libqlink.search_hybrid_reranked("),
            "pre_match should call search_hybrid_reranked; got {pre}"
        );
        assert!(pre.contains("YIELD id AS p__qid_0, score AS p__score_0"));
        assert_eq!(contrib.where_inline.as_deref(), Some("id(p) = p__qid_0"));
        assert_eq!(contrib.order_by.len(), 1);
        assert_eq!(contrib.order_by[0].0, "p__score_0");
    }
}
