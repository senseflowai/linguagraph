//! Request options and per-capability response payloads.
//!
//! These are the argument and return shapes for the [`GraphRead`] and
//! [`GraphWrite`] traits. They are deliberately plain data — no methods
//! beyond a `Default`/constructor here and there — so the REST service
//! can map query strings and JSON bodies straight onto them.
//!
//! [`GraphRead`]: crate::api::GraphRead
//! [`GraphWrite`]: crate::api::GraphWrite

use std::time::Duration;

use serde::{Deserialize, Serialize};

use super::model::{
    Cost, Cursor, Diagnostics, Domain, Entity, EntityId, EntitySummary, EntityType, Filters,
    Locale, Relation, RelationId, RelationType, SourceRef, Subgraph,
};

// ── ask ─────────────────────────────────────────────────────────────────

/// How much natural-language answer text to synthesise alongside the
/// evidence subgraph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum AnswerMode {
    /// Return only the subgraph, no prose.
    #[default]
    None,
    /// Quote spans from the sources (no free generation).
    Extractive,
    /// Generate a short narrative answer from the subgraph.
    Generative,
}

/// Options for [`ask`](crate::api::GraphRead::ask) and
/// [`compile`](crate::api::GraphRead::compile).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AskOptions {
    /// Scope/facet filter.
    pub filters: Filters,
    /// Node budget for the returned subgraph (keeps the screen legible).
    pub max_nodes: u32,
    /// Traversal-depth ceiling.
    pub max_depth: u8,
    /// Whether/how to synthesise answer prose.
    pub answer: AnswerMode,
    /// Include the compiled [`QueryPlan`] (Cypher/DSL) in the answer.
    pub include_plan: bool,
    /// Wall-clock budget for the whole call.
    pub timeout: Duration,
    /// Language for the synthesised answer.
    pub locale: Locale,
}

impl Default for AskOptions {
    fn default() -> Self {
        Self {
            filters: Filters::default(),
            max_nodes: 40,
            max_depth: 3,
            answer: AnswerMode::default(),
            include_plan: false,
            timeout: Duration::from_secs(30),
            locale: Locale::default(),
        }
    }
}

/// Options for [`run`](crate::api::GraphRead::run).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunOptions {
    /// Node budget for the returned subgraph.
    pub max_nodes: u32,
    /// Wall-clock budget.
    pub timeout: Duration,
}

impl Default for RunOptions {
    fn default() -> Self {
        Self {
            max_nodes: 40,
            timeout: Duration::from_secs(30),
        }
    }
}

/// A compiled, safe-to-execute query. Detached from the question so the
/// service can cache it, authorise it, and re-run it (share links).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryPlan {
    /// linguagraph JSON-DSL representation.
    pub dsl: serde_json::Value,
    /// Parameterised Cypher — never string-interpolated user input.
    pub cypher: String,
    /// Cypher parameters.
    pub params: serde_json::Value,
    /// Estimated cost so the service can reject an expensive plan before
    /// it hits the database.
    pub estimated_cost: Cost,
}

/// The result of [`ask`](crate::api::GraphRead::ask).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Answer {
    /// Synthesised answer text, when [`AnswerMode`] requested it.
    pub text: Option<String>,
    /// The evidence subgraph.
    pub subgraph: Subgraph,
    /// Root node for the UI's initial selection.
    pub focus: Option<EntityId>,
    /// The plan, when `include_plan` was set.
    pub plan: Option<QueryPlan>,
    /// Timings / row counts for observability.
    pub diagnostics: Diagnostics,
}

// ── streaming ─────────────────────────────────────────────────────────────

/// Metadata delivered on the final [`AnswerChunk::Done`] frame.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AnswerMeta {
    /// Root node for initial selection.
    pub focus: Option<EntityId>,
    /// Whether a limit clipped the subgraph.
    pub truncated: bool,
    /// Timings / row counts.
    pub diagnostics: Diagnostics,
}

/// One frame of a streamed answer. The stream interleaves answer tokens
/// with subgraph fragments so the UI can paint progressively.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AnswerChunk {
    /// A piece of the answer text.
    Token(String),
    /// A batch of nodes.
    Nodes(Vec<Entity>),
    /// A batch of edges.
    Edges(Vec<Relation>),
    /// Terminal frame with the answer metadata.
    Done(AnswerMeta),
}

// ── search / discovery ────────────────────────────────────────────────────

/// Retrieval strategy for [`search`](crate::api::GraphRead::search).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum SearchMode {
    /// Exact keyword matching only.
    Keyword,
    /// Vector / semantic matching only.
    Semantic,
    /// Both channels fused.
    #[default]
    Hybrid,
}

/// Options for [`search`](crate::api::GraphRead::search).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchOptions {
    /// Scope/facet filter.
    pub filters: Filters,
    /// Page size.
    pub limit: u32,
    /// Forward cursor, or `None` for the first page.
    pub cursor: Option<Cursor>,
    /// Retrieval strategy.
    pub mode: SearchMode,
}

impl Default for SearchOptions {
    fn default() -> Self {
        Self {
            filters: Filters::default(),
            limit: 20,
            cursor: None,
            mode: SearchMode::default(),
        }
    }
}

/// One search hit: an entity plus why it matched.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityHit {
    /// The matched entity, projected to a summary.
    pub entity: EntitySummary,
    /// Fused relevance score.
    pub score: f32,
    /// Field names that contributed to the match.
    pub matched_on: Vec<String>,
}

/// Catalog entry for a type, used to render legends / facet chips.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityTypeInfo {
    /// The type.
    pub entity_type: EntityType,
    /// How many entities of this type exist in scope.
    pub count: u64,
    /// The type's domain, when known.
    pub domain: Option<Domain>,
}

/// A type the question-analysis step judged relevant to a question.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TypeMatch {
    /// The relevant type.
    pub entity_type: EntityType,
    /// Relevance score.
    pub score: f32,
}

// ── traversal ─────────────────────────────────────────────────────────────

/// Edge-following direction relative to the anchor node.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum Direction {
    /// Outgoing edges (`anchor → other`).
    Out,
    /// Incoming edges (`other → anchor`).
    In,
    /// Both directions.
    #[default]
    Both,
}

/// Options for [`neighbors`](crate::api::GraphRead::neighbors). Both
/// `depth` and `limit` are mandatory — an unbounded traversal is
/// rejected with [`GraphError::UnboundedTraversal`].
///
/// [`GraphError::UnboundedTraversal`]: crate::api::GraphError::UnboundedTraversal
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraversalOptions {
    /// Which way to walk.
    pub direction: Direction,
    /// Restrict to these relation types.
    pub relation_types: Option<Vec<RelationType>>,
    /// Restrict neighbour nodes to these entity types.
    pub entity_types: Option<Vec<EntityType>>,
    /// Hops to expand (`1..=max_depth`). Mandatory.
    pub depth: u8,
    /// Max nodes to return. Mandatory.
    pub limit: u32,
    /// Forward cursor for high-degree nodes.
    pub cursor: Option<Cursor>,
    /// Drop neighbours below this confidence.
    pub min_confidence: Option<f32>,
}

impl TraversalOptions {
    /// A minimal 1-hop traversal in both directions with the given node
    /// limit. Convenience for the common "expand neighbours" button.
    pub fn one_hop(limit: u32) -> Self {
        Self {
            direction: Direction::Both,
            relation_types: None,
            entity_types: None,
            depth: 1,
            limit,
            cursor: None,
            min_confidence: None,
        }
    }

    /// Whether the options describe a bounded traversal. `false` means
    /// the library must reject the request.
    pub fn is_bounded(&self) -> bool {
        self.depth >= 1 && self.limit >= 1
    }
}

/// One page of a neighbour expansion.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct NeighborsPage {
    /// Nodes + edges for this page.
    pub subgraph: Subgraph,
    /// Cursor for the next page, or `None` at the end.
    pub next: Option<Cursor>,
}

/// Options for [`paths`](crate::api::GraphRead::paths).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PathOptions {
    /// Max hops between the endpoints.
    pub max_depth: u8,
    /// Max distinct paths to return.
    pub limit: u32,
    /// Restrict to these relation types.
    pub relation_types: Option<Vec<RelationType>>,
}

impl Default for PathOptions {
    fn default() -> Self {
        Self {
            max_depth: 4,
            limit: 10,
            relation_types: None,
        }
    }
}

/// A single path between two entities.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Path {
    /// Node ids along the path, in order.
    pub nodes: Vec<EntityId>,
    /// Edge ids along the path, in order.
    pub edges: Vec<RelationId>,
    /// Path weight (lower = shorter/stronger; backend-defined).
    pub weight: f32,
}

// ── ingestion ─────────────────────────────────────────────────────────────

/// Options for [`extract`](crate::api::GraphWrite::extract).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractOptions {
    /// Domain hint for the extractor.
    pub domain: Option<Domain>,
    /// Source the extracted facts are attributed to.
    pub source: SourceRef,
}

/// Result of [`extract`](crate::api::GraphWrite::extract) — a preview
/// the service can show before committing.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Extraction {
    /// Extracted entities.
    pub entities: Vec<Entity>,
    /// Extracted relations.
    pub relations: Vec<Relation>,
}

/// A batch of facts to write, all attributed to one source.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphBatch {
    /// Entities to upsert.
    pub entities: Vec<Entity>,
    /// Relations to upsert.
    pub relations: Vec<Relation>,
    /// Source the batch is attributed to.
    pub source: SourceRef,
}

/// How an [`upsert`](crate::api::GraphWrite::upsert) reconciles incoming
/// facts with existing ones.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum MergePolicy {
    /// Soft-merge by similarity; ambiguous matches go to review.
    #[default]
    SoftMerge,
    /// Never merge; always create new nodes.
    KeepSeparate,
    /// Overwrite matching nodes' properties from the incoming batch.
    Replace,
}

/// Options for [`upsert`](crate::api::GraphWrite::upsert).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct UpsertOptions {
    /// Idempotency key so a retried ingestion is a no-op.
    pub idempotency_key: Option<String>,
    /// Deduplication policy.
    pub merge: MergePolicy,
}

/// Result of an [`upsert`](crate::api::GraphWrite::upsert).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct UpsertReport {
    /// Nodes/edges created.
    pub created: u64,
    /// Nodes/edges updated in place.
    pub updated: u64,
    /// Nodes merged into existing ones.
    pub merged: u64,
    /// Ambiguous matches routed to the review queue.
    pub needs_review: Vec<MergeCandidate>,
}

/// Result of [`delete_by_source`](crate::api::GraphWrite::delete_by_source).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DeleteReport {
    /// Whether the source existed at all.
    pub source_found: bool,
    /// Entities removed (orphaned by the deletion).
    pub entities_deleted: u64,
    /// Relations removed.
    pub relations_deleted: u64,
}

// ── review ────────────────────────────────────────────────────────────────

/// Options for [`review_queue`](crate::api::GraphWrite::review_queue).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReviewOptions {
    /// Page size.
    pub limit: u32,
    /// Forward cursor.
    pub cursor: Option<Cursor>,
    /// Only surface candidates at/under this score (most ambiguous
    /// first), when set.
    pub max_score: Option<f32>,
}

impl Default for ReviewOptions {
    fn default() -> Self {
        Self {
            limit: 20,
            cursor: None,
            max_score: None,
        }
    }
}

/// One signal explaining why two entities look like (or unlike) a match.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MergeSignal {
    /// The field compared.
    pub field: String,
    /// Whether the two sides agree on it.
    pub agree: bool,
    /// Human-readable detail for the review card.
    pub detail: String,
}

/// A pair of entities the deduper thinks might be the same.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MergeCandidate {
    /// Left side.
    pub left: EntitySummary,
    /// Right side.
    pub right: EntitySummary,
    /// Similarity score.
    pub score: f32,
    /// Per-field agreement signals.
    pub signals: Vec<MergeSignal>,
}

/// A reviewer's decision on a [`MergeCandidate`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MergeDecision {
    /// Merge the pair, keeping `keep` as the surviving node.
    Merge {
        /// Left entity.
        left: EntityId,
        /// Right entity.
        right: EntityId,
        /// The id to keep.
        keep: EntityId,
    },
    /// Record that the pair are distinct so they stop resurfacing.
    KeepSeparate {
        /// Left entity.
        left: EntityId,
        /// Right entity.
        right: EntityId,
    },
}

/// Result of [`resolve_merge`](crate::api::GraphWrite::resolve_merge).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ResolveReport {
    /// Whether the decision changed the graph.
    pub applied: bool,
    /// Surviving entity id (present on a merge).
    pub surviving: Option<EntityId>,
}
