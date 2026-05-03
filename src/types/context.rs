//! Per-stage contexts handed to [`super::TypeHandler`] methods.
//!
//! Each stage has its own context so handlers see exactly what they need —
//! no globals, no surprise dependencies. The contexts are built by the
//! caller (planner, AST lowering, builder), passed by `&mut`, and
//! discarded.

use std::collections::BTreeMap;

use serde_json::Value;

use crate::ast::query::{Literal, PropertyRef};

use super::{Capabilities, SideEffect, SideEffectQueue, TypeId, TypedOp};

// ─── Stage 1: ingestion ─────────────────────────────────────────────

/// Context handed to [`super::TypeHandler::on_ingest`].
///
/// The handler reads `value()`, decides what to *store* on the node, and
/// pushes any required side effects (embeddings, …). It must not mutate
/// the node's identity — `node_label`, `node_key_field`, and `node_key`
/// are read-only references to the row being ingested.
#[derive(Debug)]
pub struct IngestCtx<'a> {
    pub node_label: &'a str,
    pub node_key_field: &'a str,
    pub node_key: &'a Literal,
    /// Property name being ingested (the field's *name*, not its value).
    pub field_name: &'a str,
    /// Raw extracted JSON value. Some types (SemanticText) read the
    /// string and don't store it; others store the typed view.
    raw: &'a Value,
    /// Replacement value the node should store. Defaults to a faithful
    /// JSON-to-[`Literal`] conversion of the raw value. Handlers can
    /// override or take it offline.
    output: Option<Literal>,
    /// Whether the property should be stored at all. Handlers that
    /// keep the data entirely outside Memgraph (e.g. an opaque image
    /// embedding) call [`Self::skip`].
    skip: bool,
    /// Side-effect queue shared across the entire ingestion run.
    side_effects: &'a mut SideEffectQueue,
}

impl<'a> IngestCtx<'a> {
    pub fn new(
        node_label: &'a str,
        node_key_field: &'a str,
        node_key: &'a Literal,
        field_name: &'a str,
        raw: &'a Value,
        side_effects: &'a mut SideEffectQueue,
    ) -> Self {
        Self {
            node_label,
            node_key_field,
            node_key,
            field_name,
            raw,
            output: None,
            skip: false,
            side_effects,
        }
    }

    /// Raw JSON value as it came out of the mapper.
    pub fn value(&self) -> &Value {
        self.raw
    }

    /// Override the value the node should store.
    pub fn set_value(&mut self, lit: Literal) {
        self.output = Some(lit);
        self.skip = false;
    }

    /// Discard the property — nothing about it lands in Memgraph.
    pub fn skip(&mut self) {
        self.skip = true;
        self.output = None;
    }

    /// Push a side effect onto the queue (executed by the pipeline
    /// after the Memgraph batch lands).
    pub fn push_side_effect(&mut self, eff: SideEffect) {
        self.side_effects.push(eff);
    }

    /// Final disposition of the property:
    /// * `Some(Some(lit))` — store `lit`
    /// * `Some(None)` — handler said skip
    /// * `None` — handler did nothing; caller falls back to the default
    ///   JSON conversion.
    pub fn finish(self) -> Option<Option<Literal>> {
        if self.skip {
            Some(None)
        } else {
            self.output.map(Some)
        }
    }
}

// ─── Stage 2: DSL → AST ─────────────────────────────────────────────

/// Raw DSL filter (with `type` set) as seen by the lowering stage.
///
/// We don't reuse `dsl::schema::Filter` directly so the lowering
/// signature stays stable when the DSL surface evolves — adding fields
/// here is cheap.
#[derive(Debug, Clone)]
pub struct RawTypedFilter<'a> {
    pub field: &'a PropertyRef,
    pub op: TypedOp,
    pub value: &'a Value,
}

/// Context for [`super::TypeHandler::lower`].
#[derive(Debug)]
pub struct LowerCtx<'a> {
    pub raw: RawTypedFilter<'a>,
    pub type_id: TypeId,
}

// ─── Stage 3: AST → Cypher ──────────────────────────────────────────

/// Cypher fragments a handler may contribute around the user's main
/// MATCH/WHERE/RETURN frame.
///
/// Layout the builder splices everything into:
///
/// ```text
///   [pre_match]               ← handler-contributed (runs FIRST)
///   MATCH ...                 ← user's pattern
///   [WHERE inline ...]        ← user's predicates AND-ed with handler `where_inline`
///   [post_match]              ← handler-contributed (runs AFTER WHERE)
///   RETURN ...                ← user's projection
///   ORDER BY user-keys, extras...   ← user's sort + handler `order_by`
///   LIMIT ...
/// ```
///
/// * `pre_match` is for handlers that *seed* the MATCH (e.g. a pure
///   semantic search that runs `qlink.search` first, then MATCH joins
///   to the result).
/// * `post_match` is for handlers that need to compute over already-
///   matched rows (e.g. hybrid scoring: WITH c, CASE WHEN ... → CALL
///   qlink.score_batch_node → WITH c, c__exact + c__sem AS …).
/// * `where_inline` is the boolean expression to splice into WHERE.
/// * `order_by` keys are appended after any user sort keys.
#[derive(Debug, Default, Clone)]
pub struct CypherContribution {
    pub pre_match: Vec<String>,
    pub post_match: Vec<String>,
    pub where_inline: Option<String>,
    pub order_by: Vec<(String, OrderDir)>,
}

#[derive(Debug, Clone, Copy)]
pub enum OrderDir {
    Asc,
    Desc,
}

impl OrderDir {
    pub fn as_str(&self) -> &'static str {
        match self {
            OrderDir::Asc => "ASC",
            OrderDir::Desc => "DESC",
        }
    }
}

/// Context for [`super::TypeHandler::emit`].
///
/// The handler:
/// 1. Calls [`Self::bind`] to register parameters (never inline literals).
/// 2. Pushes any prelude/with fragments via [`Self::contribution_mut`].
/// 3. Calls [`Self::set_where`] with the boolean expression to splice
///    into WHERE (or leaves it unset if scoring happens elsewhere).
pub struct EmitCtx<'a> {
    contribution: &'a mut CypherContribution,
    binder: &'a mut dyn ParamBinder,
}

impl std::fmt::Debug for EmitCtx<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EmitCtx")
            .field("contribution", &self.contribution)
            .finish_non_exhaustive()
    }
}

impl<'a> EmitCtx<'a> {
    pub fn new(contribution: &'a mut CypherContribution, binder: &'a mut dyn ParamBinder) -> Self {
        Self { contribution, binder }
    }

    /// Register a parameter and return its `$name` placeholder.
    pub fn bind(&mut self, value: Literal) -> String {
        self.binder.bind(value)
    }

    pub fn contribution_mut(&mut self) -> &mut CypherContribution {
        self.contribution
    }

    pub fn set_where(&mut self, expr: impl Into<String>) {
        self.contribution.where_inline = Some(expr.into());
    }

    /// Add a Cypher fragment that runs *before* the user's MATCH.
    pub fn push_pre_match(&mut self, line: impl Into<String>) {
        self.contribution.pre_match.push(line.into());
    }

    /// Add a Cypher fragment that runs *after* the user's WHERE.
    pub fn push_post_match(&mut self, line: impl Into<String>) {
        self.contribution.post_match.push(line.into());
    }
}

/// Indirection so `EmitCtx` doesn't need to know about the builder's
/// `Cursor`. Anything that can hand out fresh `$pN` placeholders fits.
pub trait ParamBinder {
    fn bind(&mut self, value: Literal) -> String;
}

// ─── Stage 4: prompt advertisement ──────────────────────────────────

/// One-line description of a type, exposed to the LLM in the system
/// prompt.
#[derive(Debug, Clone)]
pub struct PromptHint {
    pub type_id: TypeId,
    pub capabilities: Capabilities,
    pub ops: Vec<TypedOp>,
    /// Human-readable explanation, e.g. "free-text natural-language
    /// search backed by a local embedder". Optional.
    pub doc: Option<String>,
    /// One-line DSL example the LLM can crib from.
    pub example: Option<String>,
}

impl PromptHint {
    pub fn from_capabilities(type_id: TypeId, caps: Capabilities) -> Self {
        let ops = caps.default_ops();
        Self { type_id, capabilities: caps, ops, doc: None, example: None }
    }
}

impl Default for PromptHint {
    fn default() -> Self {
        Self {
            type_id: TypeId::new(""),
            capabilities: Capabilities::empty(),
            ops: vec![],
            doc: None,
            example: None,
        }
    }
}

// ─── Helpers ────────────────────────────────────────────────────────

/// Trivially construct a property metadata map for tests / examples.
pub fn _unused_param_kept_for_doc(_: BTreeMap<String, String>) {}
