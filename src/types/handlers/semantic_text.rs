//! `SemanticText` — free-text fields searchable via embeddings + qlink.
//!
//! Responsibilities:
//!
//! 1. **Ingest**: keep the raw string on the node (so exact matches
//!    still work) and queue an [`super::super::SideEffect::EmbedAndStore`]
//!    so the pipeline can compute an embedding once the node id is
//!    known.
//! 2. **Lower**: validate the DSL fragment and stash the *embedded*
//!    query vector in the predicate's `params`, so the builder doesn't
//!    have to embed inside Cypher emission (which is expected to be
//!    pure).
//! 3. **Emit**: render the appropriate `qlink.*` Cypher fragment for
//!    `search` (pure vector) or `hybrid_search` (vector + exact) and
//!    bind the embedding as a parameter.
//!
//! The handler is configured by a `[types.SemanticText]` block:
//!
//! ```toml
//! [types.SemanticText]
//! embedding_model = "models/bge-small.gguf"
//! collection      = "companies"
//! top_k           = 20
//! ```

use std::collections::BTreeMap;

use serde_json::Value;

use crate::ast::query::{Literal, PropertyRef};
use crate::config::Config;
use crate::embeddings::SharedEmbedder;
use crate::types::context::OrderDir;
use crate::types::{
    Capabilities, EmitCtx, IngestCtx, LowerCtx, PromptHint, SideEffect, TypeError, TypeHandler,
    TypeId, TypedOp, TypedPredicate,
};

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
    /// Number of results returned by `qlink.search`.
    pub top_k: u32,
}

impl SemanticTextConfig {
    pub fn from_config(cfg: &Config) -> Option<Self> {
        cfg.types.get("SemanticText").map(|t| Self {
            embedding_model: t.embedding_model.clone(),
            collection: t.collection.clone().unwrap_or_else(|| "semantic_text".into()),
            top_k: t.top_k.unwrap_or(20),
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
                    reason: format!(
                        "SemanticText expects string, got {}",
                        json_kind(other)
                    ),
                });
            }
        };

        // Keep the raw text on the node — useful for exact match,
        // contains, and human inspection.
        ctx.set_value(Literal::String(text.clone()));

        // Queue the embed-and-store side effect. The collection name is
        // derived from the configured default plus the field name so
        // multiple SemanticText fields don't collide.
        let collection = format!("{}__{}", self.config.collection, ctx.field_name);
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

        // For non-vector ops we can short-circuit to the plain Cypher
        // path with no embedding work.
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

        // Embed the query once at lowering time — emit() must remain
        // pure (no I/O) so the builder is testable in isolation.
        let vec = self
            .embedder
            .embed(text)
            .map_err(|e| TypeError::Embedder(e.to_string()))?;
        let lit_vec = Literal::List(vec.into_iter().map(|f| Literal::Float(f as f64)).collect());

        let mut params = BTreeMap::new();
        params.insert("embedding".to_string(), lit_vec);
        params.insert(
            "collection".to_string(),
            Literal::String(collection_for(self, &ctx.raw.field)),
        );
        params.insert(
            "top_k".to_string(),
            Literal::Int(self.config.top_k as i64),
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
        let render_field = |p: &PropertyRef| match &p.property {
            Some(prop) => format!("{}.{}", p.alias, prop),
            None => p.alias.to_string(),
        };

        match pred.op {
            // ── Plain text ops route through standard Cypher. ─────────
            TypedOp::Eq | TypedOp::Neq | TypedOp::Contains => {
                let value = pred.value.clone();
                let placeholder = ctx.bind(value);
                let sym = match pred.op {
                    TypedOp::Eq => "=",
                    TypedOp::Neq => "<>",
                    TypedOp::Contains => "CONTAINS",
                    _ => unreachable!(),
                };
                ctx.set_where(format!(
                    "{lhs} {sym} {placeholder}",
                    lhs = render_field(&pred.field)
                ));
                Ok(())
            }
            // ── Pure vector search ─────────────────────────────────────
            //
            // Layout: a CALL qlink.search runs *before* MATCH and yields
            // `(qid, score)`. The MATCH then constrains to nodes whose
            // id equals that qid, and we ORDER BY the score.
            TypedOp::Search => {
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
                let top_k = pred
                    .params
                    .get("top_k")
                    .cloned()
                    .unwrap_or(Literal::Int(20));

                let coll_p = ctx.bind(coll);
                let emb_p = ctx.bind(emb);
                let topk_p = ctx.bind(top_k);

                ctx.push_pre_match(format!(
                    "CALL libqlink.search({coll_p}, {emb_p}, {topk_p}) YIELD id AS {alias}__qid, score AS {alias}__score"
                ));
                ctx.set_where(format!("id({alias}) = {alias}__qid"));
                ctx.contribution_mut()
                    .order_by
                    .push((format!("{alias}__score"), OrderDir::Desc));
                Ok(())
            }
            // ── Hybrid (exact OR semantic, weighted by score) ─────────
            //
            // Layout: after the user's MATCH/WHERE, compute an exact-
            // match column inline, then call qlink.score_batch_node to
            // attach the semantic score, then re-bind so the final
            // score = exact + semantic.
            TypedOp::HybridSearch => {
                let alias = pred.field.alias.as_str();
                let prop_name = pred.field.property.as_deref().ok_or_else(|| {
                    TypeError::Handler("hybrid_search requires <alias>.<property>".into())
                })?;
                let query = pred.value.clone();
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

                let q_p = ctx.bind(query);
                let coll_p = ctx.bind(coll);
                let emb_p = ctx.bind(emb);

                ctx.push_post_match(format!(
                    "WITH {alias},\n\
                          CASE WHEN {alias}.{prop_name} = {q_p} THEN 1.0 ELSE 0.0 END AS {alias}__exact\n\
                     WITH collect({{ n: {alias}, e: {alias}__exact }}) AS {alias}__rows\n\
                     CALL libqlink.score_batch_node({coll_p}, [{emb_p}],\n\
                          [r IN {alias}__rows | r.n], 0.0) YIELD node AS {alias}__n, score AS {alias}__sem\n\
                     WITH {alias}__rows, {alias}__n AS {alias}, {alias}__sem,\n\
                          [r IN {alias}__rows WHERE r.n = {alias}__n | r.e][0] AS {alias}__exact"
                ));
                ctx.contribution_mut()
                    .order_by
                    .push((
                        format!("({alias}__exact + {alias}__sem)"),
                        OrderDir::Desc,
                    ));
                // No WHERE addendum: the post_match clauses replace the
                // node binding, so further filtering happens against
                // the rebound `alias`.
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
                "Free-text field with vector search. Supports `eq`/`neq`/`contains` for exact \
                 lookups, `search` for natural-language matches, and `hybrid_search` to combine \
                 the two."
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

/// Per-field collection name. The handler's configured `collection` is
/// the prefix; the field name is appended so `Person.bio` and
/// `Company.description` end up in distinct Qdrant collections (a
/// requirement for vector-dim sanity).
fn collection_for(h: &SemanticTextHandler, field: &PropertyRef) -> String {
    let prop = field.property.as_deref().unwrap_or(field.alias.as_str());
    format!("{}__{}", h.config.collection, prop)
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
        SemanticTextHandler::new(
            SemanticTextConfig {
                embedding_model: None,
                collection: "test".into(),
                top_k: 10,
            },
            Arc::new(MockEmbedder::new(8)),
        )
    }

    fn pref(alias: &str, prop: &str) -> PropertyRef {
        PropertyRef {
            alias: Alias::new(alias),
            property: Some(prop.into()),
        }
    }

    struct CountingBinder {
        next: usize,
        params: std::collections::BTreeMap<String, Literal>,
    }
    impl ParamBinder for CountingBinder {
        fn bind(&mut self, v: Literal) -> String {
            let n = format!("p{}", self.next);
            self.next += 1;
            self.params.insert(n.clone(), v);
            format!("${n}")
        }
    }

    #[test]
    fn ingest_keeps_text_and_queues_side_effect() {
        let h = handler();
        let mut q = SideEffectQueue::new();
        let key = Literal::String("c1".into());
        let raw = serde_json::json!("Hello world");
        let mut ctx = IC::new("Company", "id", &key, "name", &raw, &mut q);
        h.on_ingest(&mut ctx).unwrap();
        let stored = ctx.finish();
        assert_eq!(stored, Some(Some(Literal::String("Hello world".into()))));
        assert_eq!(q.len(), 1);
        match &q.into_vec()[0] {
            SideEffect::EmbedAndStore { text, label, .. } => {
                assert_eq!(text, "Hello world");
                assert_eq!(label, "Company");
            }
        }
    }

    #[test]
    fn lower_search_embeds_query_into_params() {
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
        };
        let pred = h.lower(&mut ctx).unwrap();
        assert!(pred.params.contains_key("embedding"));
        assert!(pred.params.contains_key("collection"));
        assert!(pred.params.contains_key("top_k"));
        match pred.params.get("embedding").unwrap() {
            Literal::List(items) => assert_eq!(items.len(), 8),
            _ => panic!("embedding should be a List"),
        }
    }

    #[test]
    fn lower_eq_does_not_embed() {
        let h = handler();
        let field = pref("c", "name");
        let value = serde_json::json!("apple");
        let mut ctx = LC {
            raw: RawTypedFilter {
                field: &field,
                op: TypedOp::Eq,
                value: &value,
            },
            type_id: TypeId::new(SemanticTextHandler::TYPE_ID),
        };
        let pred = h.lower(&mut ctx).unwrap();
        assert!(pred.params.is_empty());
        assert_eq!(pred.value, Literal::String("apple".into()));
    }

    #[test]
    fn emit_search_renders_qlink_call_and_orders_by_score() {
        let h = handler();
        let field = pref("c", "name");
        let value = serde_json::json!("apple");
        let mut lower = LC {
            raw: RawTypedFilter {
                field: &field,
                op: TypedOp::Search,
                value: &value,
            },
            type_id: TypeId::new(SemanticTextHandler::TYPE_ID),
        };
        let pred = h.lower(&mut lower).unwrap();

        let mut contrib = CypherContribution::default();
        let mut binder = CountingBinder { next: 0, params: Default::default() };
        let mut emit = EmitCtx::new(&mut contrib, &mut binder);
        h.emit(&mut emit, &pred).unwrap();

        let pre = contrib.pre_match.join("\n");
        assert!(
            pre.contains("CALL libqlink.search"),
            "pre_match should contain qlink.search; got {pre}"
        );
        assert!(pre.contains("c__qid"));
        assert!(pre.contains("c__score"));
        assert_eq!(contrib.where_inline.as_deref(), Some("id(c) = c__qid"));
        assert_eq!(contrib.order_by.len(), 1);
        assert_eq!(contrib.order_by[0].0, "c__score");
    }

    #[test]
    fn emit_eq_uses_inline_where() {
        let h = handler();
        let field = pref("c", "name");
        let pred = TypedPredicate {
            type_id: TypeId::new(SemanticTextHandler::TYPE_ID),
            field,
            op: TypedOp::Eq,
            value: Literal::String("apple".into()),
            params: BTreeMap::new(),
        };
        let mut contrib = CypherContribution::default();
        let mut binder = CountingBinder { next: 0, params: Default::default() };
        let mut emit = EmitCtx::new(&mut contrib, &mut binder);
        h.emit(&mut emit, &pred).unwrap();
        let w = contrib.where_inline.unwrap();
        assert!(w.contains("c.name = $p0"));
    }

    #[test]
    fn emit_hybrid_renders_both_signals() {
        let h = handler();
        let field = pref("c", "name");
        let value = serde_json::json!("apple");
        let mut lower = LC {
            raw: RawTypedFilter {
                field: &field,
                op: TypedOp::HybridSearch,
                value: &value,
            },
            type_id: TypeId::new(SemanticTextHandler::TYPE_ID),
        };
        let pred = h.lower(&mut lower).unwrap();
        let mut contrib = CypherContribution::default();
        let mut binder = CountingBinder { next: 0, params: Default::default() };
        let mut emit = EmitCtx::new(&mut contrib, &mut binder);
        h.emit(&mut emit, &pred).unwrap();

        let post = contrib.post_match.join("\n");
        assert!(post.contains("c__exact"));
        assert!(post.contains("libqlink.score_batch_node"));
        assert!(post.contains("c__sem"));
        assert!(contrib.order_by.iter().any(|(k, _)| k.contains("c__exact + c__sem")));
    }
}
