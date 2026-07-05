//! Data model exposed over the public API.
//!
//! Every type here is `serde`-serializable so the REST service can hand
//! them to its transport layer almost verbatim. Identifiers are opaque,
//! stable newtypes: they survive process restarts, which is what makes
//! share links and cursors durable.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

// ── identifiers ───────────────────────────────────────────────────────────
//
// Opaque, stable newtypes. `serde(transparent)` keeps the wire form a
// bare string so the REST layer sees `"acme"`, not `{"0":"acme"}`.

macro_rules! id_newtype {
    ($(#[$meta:meta])* $name:ident) => {
        $(#[$meta])*
        #[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
        #[serde(transparent)]
        pub struct $name(pub String);

        impl $name {
            /// Borrow the underlying string.
            pub fn as_str(&self) -> &str {
                &self.0
            }
        }

        impl From<String> for $name {
            fn from(s: String) -> Self {
                Self(s)
            }
        }

        impl From<&str> for $name {
            fn from(s: &str) -> Self {
                Self(s.to_owned())
            }
        }

        impl std::fmt::Display for $name {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.write_str(&self.0)
            }
        }
    };
}

id_newtype!(
    /// Tenant / workspace identifier. Everything the API does is scoped to
    /// one of these; see [`crate::api::LinguaGraph::read`].
    TenantId
);
id_newtype!(
    /// Stable identifier for an [`Entity`].
    EntityId
);
id_newtype!(
    /// Stable identifier for a [`Relation`]. Edges are addressed
    /// explicitly because a pair of nodes may be joined by several
    /// relations.
    RelationId
);
id_newtype!(
    /// Identifier for an ingestion source (a document, feed, upload …).
    SourceId
);

// ── ontology-derived type tags ──────────────────────────────────────────

id_newtype!(
    /// Entity type from the ontology, e.g. `"Company"`, `"Person"`.
    EntityType
);
id_newtype!(
    /// Relation type from the ontology, e.g. `"OWNS"`, `"SUBJECT_TO"`.
    RelationType
);
id_newtype!(
    /// Ontology domain, e.g. `"legal"`, `"corporate"`.
    Domain
);

/// UI language tag for generated answers, e.g. `"ru"`, `"en"`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Locale(pub String);

impl Default for Locale {
    fn default() -> Self {
        Locale("en".to_owned())
    }
}

/// Millisecond Unix timestamp. Kept as a plain integer so the model has
/// no external date dependency; the REST service formats it for display.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Timestamp(pub i64);

// ── property values ───────────────────────────────────────────────────────

/// A scalar property value. `serde(untagged)` keeps the JSON natural
/// (`42`, `"text"`, `true`) rather than externally tagged.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Value {
    /// Absent / explicit null.
    Null,
    /// Boolean.
    Bool(bool),
    /// Integer.
    Int(i64),
    /// Floating point.
    Float(f64),
    /// String (keyword or free text — see [`PropertyKind`]).
    Text(String),
}

/// Whether a property is matched exactly (`Keyword`) or semantically
/// (`Text`). The inspector renders these as two distinct blocks, so the
/// distinction is part of the model rather than a rendering detail.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PropertyKind {
    /// Exact-match property (ids, enums, codes).
    Keyword,
    /// Semantic / free-text property (embedded for similarity search).
    Text,
}

/// A single entity or relation property.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Property {
    /// Property name.
    pub key: String,
    /// Property value.
    pub value: Value,
    /// Keyword vs. text semantics.
    pub kind: PropertyKind,
}

// ── confidence & provenance ───────────────────────────────────────────────

/// Derived confidence band used for the badge shown in the UI.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConfidenceLevel {
    /// High confidence — safe to present without caveat.
    High,
    /// Medium confidence — present with a soft signal.
    Medium,
    /// Needs review — flag for a human.
    Review,
}

/// Confidence attached to an extracted fact.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Confidence {
    /// Raw score in `0.0..=1.0`.
    pub score: f32,
    /// Bucketed level derived from `score`.
    pub level: ConfidenceLevel,
}

impl Confidence {
    /// Build a [`Confidence`], deriving the [`ConfidenceLevel`] from the
    /// score with the library's default thresholds
    /// (`>= 0.8` high, `>= 0.5` medium, else review).
    pub fn from_score(score: f32) -> Self {
        let level = if score >= 0.8 {
            ConfidenceLevel::High
        } else if score >= 0.5 {
            ConfidenceLevel::Medium
        } else {
            ConfidenceLevel::Review
        };
        Self { score, level }
    }
}

/// One source reference behind a fact — powers the trust panel and the
/// "open source document" affordance.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SourceRef {
    /// Source identifier.
    pub source_id: SourceId,
    /// Human-readable document title, e.g. `"Annual report 2024"`.
    pub document: String,
    /// Optional finer locator inside the document (page, chunk id …).
    pub locator: Option<String>,
    /// When the fact was extracted.
    pub extracted_at: Timestamp,
}

/// All sources backing an entity or relation.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Provenance {
    /// The individual source references.
    pub sources: Vec<SourceRef>,
}

// ── entities & relations ──────────────────────────────────────────────────

/// A graph node with its full inspector payload.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Entity {
    /// Stable id.
    pub id: EntityId,
    /// Display name.
    pub name: String,
    /// Ontology type.
    pub entity_type: EntityType,
    /// Ontology domain, when known.
    pub domain: Option<Domain>,
    /// Keyword + text properties.
    pub properties: Vec<Property>,
    /// Aggregate confidence.
    pub confidence: Confidence,
    /// Where the entity's facts came from.
    pub provenance: Provenance,
}

/// A directed graph edge. Direction is `from → to`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Relation {
    /// Stable id (edges are addressed individually).
    pub id: RelationId,
    /// Ontology relation type.
    pub rel_type: RelationType,
    /// Source endpoint.
    pub from: EntityId,
    /// Target endpoint.
    pub to: EntityId,
    /// Edge properties.
    pub properties: Vec<Property>,
    /// Aggregate confidence.
    pub confidence: Confidence,
    /// Where the relation's facts came from.
    pub provenance: Provenance,
}

/// A compact entity projection used in list/search results, where the
/// full [`Entity`] payload would be wasteful.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EntitySummary {
    /// Stable id.
    pub id: EntityId,
    /// Display name.
    pub name: String,
    /// Ontology type.
    pub entity_type: EntityType,
    /// Ontology domain, when known.
    pub domain: Option<Domain>,
    /// Aggregate confidence.
    pub confidence: Confidence,
}

/// Nodes and edges returned by any traversal or query in one response —
/// no N+1 round-trips from the caller.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Subgraph {
    /// Nodes in the result.
    pub nodes: Vec<Entity>,
    /// Edges in the result.
    pub edges: Vec<Relation>,
    /// `true` when a limit clipped the result — the UI shows "N of M".
    pub truncated: bool,
}

// ── pagination ────────────────────────────────────────────────────────────

/// Opaque forward cursor. Its contents are an implementation detail; the
/// service round-trips it verbatim.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Cursor(pub String);

/// One page of a cursor-paginated list.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Page<T> {
    /// Items on this page.
    pub items: Vec<T>,
    /// Cursor for the next page, or `None` at the end.
    pub next: Option<Cursor>,
    /// Best-effort total, when the backend can estimate it cheaply.
    pub total_estimate: Option<u64>,
}

impl<T> Default for Page<T> {
    fn default() -> Self {
        Self {
            items: Vec::new(),
            next: None,
            total_estimate: None,
        }
    }
}

// ── filters / facets ──────────────────────────────────────────────────────

/// Shared filter applied by `ask`, `search`, `neighbors`, and `facets`.
#[derive(Debug, Default, Clone, PartialEq, Serialize, Deserialize)]
pub struct Filters {
    /// Restrict to these entity types.
    pub entity_types: Option<Vec<EntityType>>,
    /// Restrict to these relation types.
    pub relation_types: Option<Vec<RelationType>>,
    /// Restrict to these domains.
    pub domains: Option<Vec<Domain>>,
    /// Restrict to these sources.
    pub sources: Option<Vec<SourceId>>,
    /// Inclusive `[from, to]` extraction-time window.
    pub time_range: Option<(Timestamp, Timestamp)>,
    /// Drop facts below this confidence score.
    pub min_confidence: Option<f32>,
}

/// A dimension the caller can bucket facet counts by.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FacetDim {
    /// Bucket by entity type.
    EntityType,
    /// Bucket by relation type.
    RelationType,
    /// Bucket by domain.
    Domain,
    /// Bucket by source.
    Source,
    /// Bucket by confidence level.
    ConfidenceLevel,
}

/// Facet counts keyed by dimension. Each value is a `(bucket, count)`
/// list, already sorted by the backend.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Facets {
    /// Per-dimension `(bucket label, count)` pairs.
    pub buckets: HashMap<FacetDim, Vec<(String, u64)>>,
}

// ── cost / diagnostics ────────────────────────────────────────────────────

/// A coarse cost estimate for a compiled plan. The service can reject a
/// plan before it runs (see [`crate::api::GraphError::CostExceeded`]).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Cost {
    /// Estimated rows the plan will scan/expand.
    pub estimated_rows: u64,
    /// Estimated traversal depth reached.
    pub estimated_depth: u8,
    /// Relative unit cost (higher = heavier); backend-defined scale.
    pub units: f64,
}

/// Observability payload attached to read responses: timings and row
/// counts, plus the executed Cypher when the caller asked for it.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Diagnostics {
    /// Total wall-clock time in milliseconds.
    pub elapsed_ms: u64,
    /// Rows returned by the backend before shaping.
    pub rows: u64,
    /// The Cypher that ran, when `include_plan` was set.
    pub cypher: Option<String>,
}
