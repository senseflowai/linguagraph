//! The read/write capability traits the REST service depends on.
//!
//! The service is written against `dyn GraphRead` / `dyn GraphWrite`,
//! never against a concrete type. That buys three things:
//!
//! * **Least privilege.** The read path is handed a `GraphRead` only, so
//!   it cannot mutate the graph and can be routed to a read replica.
//! * **Testability.** Both traits are trivially mockable.
//! * **Backend freedom.** Memgraph is one implementation; nothing in the
//!   service contract names it.
//!
//! Methods are `async` via [`async_trait`]. Because `async_trait`
//! erases the future behind a `Box`, the traits are object-safe — which
//! is exactly why `dyn GraphRead` works. The streaming entry point
//! (`ask_stream`) returns an `impl Stream` and therefore lives on the
//! concrete [`ReadScope`](crate::api::ReadScope) rather than on the
//! object-safe trait.

use async_trait::async_trait;

use super::error::Result;
use super::model::{
    Entity, EntityId, Filters, Relation, RelationId, SourceId,
};
use super::options::{
    Answer, AskOptions, DeleteReport, EntityHit, EntityTypeInfo, Extraction, ExtractOptions,
    GraphBatch, MergeCandidate, MergeDecision, NeighborsPage, Path, PathOptions, QueryPlan,
    ResolveReport, ReviewOptions, RunOptions, SearchOptions, TraversalOptions, TypeMatch,
    UpsertOptions, UpsertReport,
};
use super::model::Page;
use super::model::{FacetDim, Facets};

/// Read-only graph capabilities. Safe to hand to the REST read path and
/// to route to a read replica.
#[async_trait]
pub trait GraphRead: Send + Sync {
    // ── ask ───────────────────────────────────────────────────────────

    /// Full cycle: natural language → DSL → Cypher → execute → answer.
    async fn ask(&self, question: &str, opts: AskOptions) -> Result<Answer>;

    /// Compile only: natural language → plan, no execution. Powers the
    /// "how this answer was produced" panel and pre-flight validation.
    async fn compile(&self, question: &str, opts: AskOptions) -> Result<QueryPlan>;

    /// Execute a previously compiled plan. The basis for share links.
    async fn run(&self, plan: &QueryPlan, opts: RunOptions) -> Result<super::model::Subgraph>;

    // ── search & discovery ────────────────────────────────────────────

    /// Hybrid (keyword + vector) entity search, cursor-paginated.
    async fn search(&self, query: &str, opts: SearchOptions) -> Result<Page<EntityHit>>;

    /// Type catalog for legends, facets, and the chips under the ask bar.
    async fn entity_types(&self, filter: Filters) -> Result<Vec<EntityTypeInfo>>;

    /// Which entity types are relevant to a specific question.
    async fn discover_types(&self, question: &str) -> Result<Vec<TypeMatch>>;

    // ── inspector ─────────────────────────────────────────────────────

    /// Fetch one entity by id.
    async fn entity(&self, id: &EntityId) -> Result<Option<Entity>>;

    /// Batch-fetch entities by id (avoids N+1 from the UI).
    async fn entities(&self, ids: &[EntityId]) -> Result<Vec<Entity>>;

    /// Fetch one relation by id.
    async fn relation(&self, id: &RelationId) -> Result<Option<Relation>>;

    // ── traversal ─────────────────────────────────────────────────────

    /// Expand a node's neighbours. Returns a subgraph in one response;
    /// cursor-paginated for high-degree nodes.
    async fn neighbors(&self, id: &EntityId, opts: TraversalOptions) -> Result<NeighborsPage>;

    /// Find paths between two entities ("show the link between A and B").
    async fn paths(&self, from: &EntityId, to: &EntityId, opts: PathOptions) -> Result<Vec<Path>>;

    // ── facets ────────────────────────────────────────────────────────

    /// Bucket counts across the given dimensions within a filter. Feeds
    /// the left rail and the completeness indicator.
    async fn facets(&self, scope: Filters, dims: &[FacetDim]) -> Result<Facets>;
}

/// Write / ingestion capabilities. Handed to the ingestion and review
/// paths only.
#[async_trait]
pub trait GraphWrite: Send + Sync {
    /// Text → `{entities, relations}` without writing. The service can
    /// show this as a preview.
    async fn extract(&self, text: &str, opts: ExtractOptions) -> Result<Extraction>;

    /// Write a batch with soft-merge deduplication. Idempotent by key.
    async fn upsert(&self, batch: GraphBatch, opts: UpsertOptions) -> Result<UpsertReport>;

    /// Delete everything attributed to a source.
    async fn delete_by_source(&self, source: &SourceId) -> Result<DeleteReport>;

    // ── review ────────────────────────────────────────────────────────

    /// The "needs review" duplicate queue, cursor-paginated.
    async fn review_queue(&self, opts: ReviewOptions) -> Result<Page<MergeCandidate>>;

    /// Apply a reviewer's merge decision.
    async fn resolve_merge(&self, decision: MergeDecision) -> Result<ResolveReport>;
}

/// Pluggable text embedder. Re-exported from the crate's embeddings
/// layer so the builder's `with_embedder` and the internals agree on one
/// type. Embeddings are typically computed by an external service.
pub use crate::embeddings::Embedder;

/// Shared embedder handle used by [`crate::api::LinguaGraphBuilder`].
pub use crate::embeddings::SharedEmbedder;

/// Pluggable cache for the type catalog (e.g. Redis) so autocomplete and
/// facet chips don't hit the database on every keystroke.
///
/// Kept minimal and byte-oriented: the API stores serialized catalog
/// snapshots under stable keys. A `None` from [`get`](Cache::get) is a
/// miss, not an error.
#[async_trait]
pub trait Cache: Send + Sync + std::fmt::Debug {
    /// Fetch a cached value. `Ok(None)` is a miss.
    async fn get(&self, key: &str) -> Result<Option<Vec<u8>>>;

    /// Store a value under `key` with an optional TTL in seconds.
    async fn set(&self, key: &str, value: Vec<u8>, ttl_secs: Option<u64>) -> Result<()>;
}
