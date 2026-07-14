//! Serializable response types of the graph explorer.
//!
//! Every type here is a plain JSON-friendly DTO: `serde` on both sides,
//! optional `utoipa::ToSchema` for downstream OpenAPI generation. The
//! shapes are designed for a business-facing graph browser: entities are
//! `NodeView`s with classified properties, relations are `EdgeView`s, and
//! query answers carry their own provenance and trace.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;

/// A graph node prepared for display.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
pub struct NodeView {
    /// Public handle: the node's `id` property when present, otherwise a
    /// session-scoped `"_nid:<internal-id>"` fallback (see
    /// [`NodeView::ephemeral_handle`]).
    pub id: String,
    /// Display name: `name` → `title` → the id.
    pub name: String,
    /// Primary (business) type — the node label minus tenant-prefix,
    /// scope and domain labels.
    pub entity_type: String,
    /// All Cypher labels as stored.
    pub labels: Vec<String>,
    pub properties: PropertyGroups,
    /// Surfaced verbatim from a `confidence` property when one exists.
    /// The pipeline never computes confidence — this is a data convention.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub confidence: Option<f64>,
    /// True when [`NodeView::id`] is the `"_nid:…"` fallback, which is
    /// only stable within the current database session.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub ephemeral_handle: bool,
}

/// Node properties grouped by their ontology property type, so a UI can
/// render identifiers, prose and dates differently without re-deriving
/// the classification.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
pub struct PropertyGroups {
    /// `Keyword` / `List` properties — codes, tags, exact identifiers.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub identifiers: BTreeMap<String, JsonValue>,
    /// `Text` (semantic) properties — free-form descriptions.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub descriptions: BTreeMap<String, JsonValue>,
    /// `Number` / `Bool` properties.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub facts: BTreeMap<String, JsonValue>,
    /// `Datetime` properties.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub dates: BTreeMap<String, JsonValue>,
    /// Properties the ontology doesn't describe and value-shape inference
    /// couldn't place.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub other: BTreeMap<String, JsonValue>,
}

impl PropertyGroups {
    /// Iterate every property across all groups.
    pub fn iter(&self) -> impl Iterator<Item = (&String, &JsonValue)> {
        self.identifiers
            .iter()
            .chain(self.descriptions.iter())
            .chain(self.facts.iter())
            .chain(self.dates.iter())
            .chain(self.other.iter())
    }

    pub fn is_empty(&self) -> bool {
        self.iter().next().is_none()
    }
}

/// A relation prepared for display.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
pub struct EdgeView {
    /// Synthetic stable handle: `"{from}:{edge_type}:{to}"`. Parallel
    /// edges of the same type between the same pair collapse into one.
    pub id: String,
    pub edge_type: String,
    /// [`NodeView::id`] of the source endpoint.
    pub from: String,
    /// [`NodeView::id`] of the target endpoint.
    pub to: String,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub properties: BTreeMap<String, JsonValue>,
    /// Same convention as [`NodeView::confidence`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub confidence: Option<f64>,
}

/// A displayable fragment of the graph: nodes plus the edges among them.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
pub struct Subgraph {
    pub nodes: Vec<NodeView>,
    pub edges: Vec<EdgeView>,
    /// True when limits cut the fragment short (more data exists).
    #[serde(default)]
    pub truncated: bool,
}

impl Subgraph {
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty() && self.edges.is_empty()
    }
}

/// Pointer to a provenance `Source` node (an ingested document).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
pub struct SourceRef {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

/// Direction of a relation relative to the inspected entity.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
#[serde(rename_all = "lowercase")]
pub enum RelDirection {
    Out,
    In,
}

/// One row of an entity's relation summary: "N `edge_type` edges
/// (out|in) to `neighbor_type` entities".
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
pub struct RelationSummary {
    pub edge_type: String,
    pub direction: RelDirection,
    pub neighbor_type: String,
    pub count: u64,
}

/// Full inspector card for one entity: the node, where it came from, and
/// how it connects.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
pub struct EntityCard {
    pub node: NodeView,
    /// Documents this entity was extracted from (`mention`/`part_of`).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub sources: Vec<SourceRef>,
    /// Relation groups, built-in provenance edges excluded.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub relations: Vec<RelationSummary>,
}

/// "How was this answered": the full query trace of an [`AskResult`].
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
pub struct QueryTrace {
    /// The original natural-language question (absent for direct DSL runs).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub question: Option<String>,
    /// The executed DSL document as JSON.
    pub dsl: JsonValue,
    /// Human-readable one-line summary of the DSL.
    pub dsl_summary: String,
    /// The exact grounded Cypher that ran.
    pub cypher: String,
    /// Bound parameters, embedding vectors masked.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub cypher_params: BTreeMap<String, JsonValue>,
    /// LLM completions consumed to produce the DSL (0 for direct runs).
    pub llm_attempts: usize,
    pub elapsed_ms: u64,
}

/// Tabular slice of a query result (hidden system columns stripped).
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
pub struct TableSlice {
    pub columns: Vec<String>,
    pub rows: Vec<BTreeMap<String, JsonValue>>,
}

/// Everything the explorer produces for one question / query run.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
pub struct AskResult {
    pub trace: QueryTrace,
    pub table: TableSlice,
    /// Entity subgraph materialized from the result rows. Empty for
    /// aggregate queries (they have no per-entity bindings).
    #[serde(default, skip_serializing_if = "Subgraph::is_empty")]
    pub subgraph: Subgraph,
    /// Distinct provenance documents referenced by the result rows.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub sources: Vec<SourceRef>,
    /// LLM-synthesized natural-language answer, when requested.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub answer: Option<String>,
}

/// Which channel produced a [`SearchHit`].
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
#[serde(rename_all = "lowercase")]
pub enum SearchChannel {
    Keyword,
    Semantic,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
pub struct SearchHit {
    pub node: NodeView,
    /// Similarity score for semantic hits; keyword hits carry none.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub score: Option<f64>,
    pub channel: SearchChannel,
}

/// An entity type with its instance count.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
pub struct TypeCount {
    pub name: String,
    pub count: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
pub struct SearchResult {
    pub hits: Vec<SearchHit>,
    /// Entity types semantically related to the query (semantic channel
    /// only; empty otherwise).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub related_types: Vec<TypeCount>,
}

/// A relation type with count and (when known) endpoint labels.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
pub struct RelTypeCount {
    pub name: String,
    pub count: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub from: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub to: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
}

/// Dataset overview: what's in the graph, at a glance. Drives type
/// filters and legends.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
pub struct OverviewReport {
    /// Business entity types (built-in `Source`/`Chunk` excluded).
    pub entity_types: Vec<TypeCount>,
    /// Business relation types (built-in `mention`/`part_of` excluded).
    pub relation_types: Vec<RelTypeCount>,
    pub total_entities: u64,
    pub total_relations: u64,
    /// Ingested documents.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub sources: Vec<SourceRef>,
}

/// One page of entities of a single type.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
pub struct EntityTable {
    pub entity_type: String,
    /// Suggested display columns, most useful first.
    pub key_columns: Vec<String>,
    pub rows: Vec<NodeView>,
    /// Total instance count (across all pages).
    pub total: u64,
    pub offset: u32,
}

/// A dated fact extracted from a `Datetime` property.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
pub struct TimelineEvent {
    /// The property value as stored (normalized to a string).
    pub date: String,
    /// Seconds since the Unix epoch when the date parses; used for sorting.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub epoch_seconds: Option<i64>,
    /// Name of the `Datetime` property the event came from.
    pub property: String,
    pub entity_id: String,
    pub entity_name: String,
    pub entity_type: String,
}

/// A subgraph exported as `GraphBuilder::from_json`-compatible JSON
/// (`{entities, relations}`).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
pub struct ExportDoc(pub JsonValue);

// ── Options ─────────────────────────────────────────────────────────────

/// Options for [`crate::explore::Explorer::ask`] /
/// [`crate::explore::Explorer::run_dsl`].
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
pub struct AskOptions {
    /// Synthesize a natural-language answer with the LLM (requires a
    /// configured translator).
    #[serde(default)]
    pub synthesize_answer: bool,
    /// Materialize the entity subgraph from the result rows.
    #[serde(default = "default_true")]
    pub include_subgraph: bool,
    /// Auto-project fields referenced by the query's own filters when the
    /// filtered entity is already part of the response (e.g. a
    /// `price < 100` filter paired with `return: [name]` gains a `price`
    /// column). Lets an answer-synthesis LLM show the value that made a
    /// row match instead of only being told about the filter in prose.
    #[serde(default = "default_true")]
    pub include_filter_context: bool,
    /// Cap on rows kept in [`AskResult::table`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_rows: Option<u32>,
}

impl Default for AskOptions {
    fn default() -> Self {
        Self {
            synthesize_answer: false,
            include_subgraph: true,
            include_filter_context: true,
            max_rows: None,
        }
    }
}

fn default_true() -> bool {
    true
}

/// Filters and pagination for [`crate::explore::Explorer::neighbors`].
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
pub struct NeighborOptions {
    /// Keep only these edge types. `None` keeps every user edge type
    /// (built-in `mention`/`part_of` excluded); name them explicitly to
    /// walk provenance edges.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub edge_types: Option<Vec<String>>,
    /// Keep only neighbors carrying at least one of these labels.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target_labels: Option<Vec<String>>,
    /// `None` = both directions.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub direction: Option<RelDirection>,
    /// `None` = the explorer's default page size.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub limit: Option<u32>,
    #[serde(default)]
    pub offset: u32,
}

/// Search strategy for [`crate::explore::Explorer::search`].
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
#[serde(rename_all = "lowercase")]
pub enum SearchMode {
    /// Semantic when an embedder is configured, keyword otherwise.
    #[default]
    Auto,
    Keyword,
    Semantic,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
pub struct SearchOptions {
    /// Restrict hits to one entity type.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub entity_type: Option<String>,
    #[serde(default)]
    pub mode: SearchMode,
    /// Keyword channel: exact value match instead of case-insensitive
    /// substring.
    #[serde(default)]
    pub exact: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub limit: Option<u32>,
}

/// Pagination for entity tables and timelines.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
pub struct PageOptions {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub limit: Option<u32>,
    #[serde(default)]
    pub offset: u32,
    /// Property to sort by (default `name`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sort_by: Option<String>,
}
