//! Pluggable field-type system.
//!
//! Each field type owns its behavior across four stages — *ingestion*,
//! *DSL → AST lowering*, *AST → Cypher emission*, and *prompt
//! advertisement* — and is registered once at startup. Core modules
//! (DSL parser, AST lowering, Cypher builder) **must not branch on type
//! names**; they always go through a [`TypeRegistry`].
//!
//! # Design
//!
//! The four stages have *different* state requirements. A single fat trait
//! with a method per stage works only if each method is handed a structured
//! `*Context` that is the only thing it touches. Otherwise the trait grows
//! ad-hoc parameters every time a type wants something new.
//!
//! Hence: a thin trait, four contexts, one registry.
//!
//! ```text
//!   ┌───────────┐     IngestCtx ───► on_ingest    (planner)
//!   │           │     LowerCtx  ───► lower        (ast::from_dsl)
//!   │  Handler  │     EmitCtx   ───► emit         (builder::where_part)
//!   │           │                ►   prompt_hint  (prompt::generator)
//!   └───────────┘
//! ```
//!
//! ## Why a thin trait + four contexts
//!
//! 1. **Composable & testable.** Each stage's context owns only what that
//!    stage needs. A unit test can exercise [`TypeHandler::on_ingest`]
//!    against a fixture context with no DB, no Cypher, no embedder.
//! 2. **Extensible.** Adding a new stage means adding a new method with
//!    its own context — existing handlers stay correct.
//! 3. **No stringly-typed dispatch.** Handlers register against a
//!    [`TypeId`] newtype; the registry resolves once and hands the trait
//!    object back.
//! 4. **Capability-driven validation.** Whether a type supports `search`
//!    is asked through [`TypeHandler::capabilities`] /
//!    [`TypeHandler::supported_ops`], not by trying to call `emit` and
//!    catching an error.
//! 5. **Side effects are first-class.** [`IngestCtx::push_side_effect`]
//!    queues an embedding/insert task; the pipeline drains the queue
//!    *after* the Memgraph batch lands and runs *one* batched
//!    `qlink.insert_batch` per (collection, label) — never per-row.
//! 6. **Cypher contributions are bounded.** [`EmitCtx`] exposes only
//!    `bind`, `prelude.push`, `where_inline.push`, etc. The builder
//!    still owns the overall query shape — types can extend, never
//!    replace, the structure.
//!
//! ## Public surface
//!
//! - [`TypeHandler`] — trait every type implements
//! - [`TypeRegistry`] / [`RegistryBuilder`] — registration & lookup
//! - [`Capabilities`] / [`TypedOp`] — capability declaration
//! - [`IngestCtx`] / [`LowerCtx`] / [`EmitCtx`] — stage contexts
//! - [`TypedPredicate`] / [`SideEffect`] — value types crossing stage
//!   boundaries
//! - [`handlers`] — bundled handler implementations (currently
//!   [`handlers::SemanticTextHandler`])

pub mod builtin;
pub mod capability;
pub mod context;
pub mod handlers;
pub mod op;
pub mod registry;
pub mod side_effect;

use std::fmt::Debug;
use std::sync::Arc;

use thiserror::Error;

pub use builtin::BuiltinType;
pub use capability::Capabilities;
pub use context::{EmitCtx, IngestCtx, LowerCtx, PrepareCtx, PromptHint};
pub use op::TypedOp;
pub use registry::{RegistryBuilder, TypeRegistry};
pub use side_effect::{SideEffect, SideEffectQueue};

use crate::ast::query::Literal;

/// A registered type's identifier — the value of `"type": "..."` in DSL
/// and mapping documents.
///
/// Wrapping the string in a newtype lets us derive [`Eq`] / [`Hash`] for
/// O(1) registry lookup and keeps callers from accidentally substituting
/// any old `String` where a registered type id is expected.
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(transparent)]
pub struct TypeId(pub String);

impl TypeId {
    pub fn new(s: impl Into<String>) -> Self {
        Self(s.into())
    }
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for TypeId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl From<&str> for TypeId {
    fn from(s: &str) -> Self {
        Self(s.to_string())
    }
}

/// Error type used by handlers and the registry.
#[derive(Debug, Error)]
pub enum TypeError {
    #[error("type '{0}' is not registered")]
    UnknownType(String),

    #[error("type '{ty}' does not support op '{op}'")]
    UnsupportedOp { ty: String, op: String },

    #[error("invalid value for type '{ty}': {reason}")]
    InvalidValue { ty: String, reason: String },

    #[error("missing configuration for type '{0}'")]
    MissingConfig(String),

    #[error("embedder error: {0}")]
    Embedder(String),

    #[error("handler error: {0}")]
    Handler(String),
}

/// A fully-resolved typed predicate carried in the AST.
///
/// `params` is a free-form scratchpad the handler fills during
/// [`TypeHandler::lower`] and reads during [`TypeHandler::emit`]. The
/// builder treats it opaquely, so handlers can store anything from a
/// pre-computed embedding to a kNN limit without touching core code.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TypedPredicate {
    pub type_id: TypeId,
    /// Field reference the predicate applies to (e.g. `c.name`). Stored
    /// as a property reference so the builder can render aliases without
    /// re-parsing strings.
    pub field: crate::ast::query::PropertyRef,
    pub op: TypedOp,
    pub value: Literal,
    /// Handler-private scratchpad. Survives across the AST → builder
    /// boundary so `lower` can hand information forward to `emit`.
    #[serde(default)]
    pub params: std::collections::BTreeMap<String, Literal>,
}

/// Trait every field type implements.
///
/// Handlers are registered into a [`TypeRegistry`] and cloned cheaply
/// (they live behind `Arc`). Implementations must be `Send + Sync` so
/// the registry can be threaded through async pipelines.
pub trait TypeHandler: Send + Sync + Debug {
    /// Stable identifier used in DSL (`"type": "..."`) and config
    /// (`[types.<ID>]`).
    fn type_id(&self) -> TypeId;

    /// What this type supports. Used for prompt generation and to
    /// validate DSL early — handlers do **not** need to re-check.
    fn capabilities(&self) -> Capabilities;

    /// DSL ops legal for this type. Defaults to inferring from
    /// [`Self::capabilities`]; override for finer control.
    fn supported_ops(&self) -> Vec<TypedOp> {
        self.capabilities().default_ops()
    }

    /// Stage 1: ingestion.
    ///
    /// Read the raw value from `ctx.value()`, optionally rewrite the
    /// stored property via `ctx.set_value(...)` or `ctx.skip()`, and
    /// queue any side effects (embedding tasks, geo indexing, …) on the
    /// context.
    fn on_ingest(&self, ctx: &mut IngestCtx<'_>) -> Result<(), TypeError>;

    /// Stage 2: DSL → AST lowering.
    ///
    /// Validate the DSL fragment for this type and produce a
    /// [`TypedPredicate`]. The DSL parser has already verified the type
    /// is registered and the op is in [`Self::supported_ops`]; the
    /// handler only has to check value shape and stash anything it
    /// needs in `params`.
    ///
    /// `lower` is documented as pure in [`crate::ast`] / [`crate::resolve`].
    /// Handlers that need I/O (e.g. computing embeddings) should
    /// stash a *symbolic* request in `params` here and do the actual
    /// batched I/O in [`Self::prepare`], keeping this method
    /// synchronous and testable.
    fn lower(&self, ctx: &mut LowerCtx<'_>) -> Result<TypedPredicate, TypeError>;

    /// Stage 2.5: batched preparation, run between lowering and emit.
    ///
    /// The pipeline collects all `TypedPredicate`s belonging to this
    /// handler in a single query (or across a batch of queries) and
    /// hands them here through `ctx.predicates_mut()`. Handlers can
    /// then do one batched I/O call (embedder, geocoder, …) and
    /// mutate the predicates' `params` in place.
    ///
    /// Default impl is a no-op. Handlers whose `lower` is already
    /// pure don't need to override.
    ///
    /// Kept synchronous so handlers that don't need async work don't
    /// pay the `async_trait` allocation cost. Pipelines that need
    /// async batching can use [`crate::types::PrepareCtx::take_pending`]
    /// + a custom driver — added as a follow-up when a handler
    /// actually needs it.
    fn prepare(&self, ctx: &mut PrepareCtx<'_>) -> Result<(), TypeError> {
        let _ = ctx;
        Ok(())
    }

    /// Stage 3: AST → Cypher.
    ///
    /// Contribute fragments to the cursor. Handlers must use
    /// [`EmitCtx::bind`] for every value — never inline literals into
    /// the buffer.
    fn emit(&self, ctx: &mut EmitCtx<'_>, pred: &TypedPredicate) -> Result<(), TypeError>;

    /// Stage 4: prompt advertisement. Default implementation derives a
    /// generic hint from the capabilities; types with bespoke behavior
    /// (e.g. SemanticText) override.
    fn prompt_hint(&self) -> PromptHint {
        PromptHint::from_capabilities(self.type_id(), self.capabilities())
    }
}

/// Convenience alias used everywhere a registry crosses async/Send
/// boundaries.
pub type SharedRegistry = Arc<TypeRegistry>;
