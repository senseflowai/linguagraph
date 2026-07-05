//! Orchestration layer.
//!
//! [`Pipeline`] wires the layers (DSL → AST → Cypher → DB) together. Callers
//! provide their own [`GraphClient`] implementation, which keeps the core
//! testable without a live Memgraph and ready for future driver swaps.

pub mod entity_type_search;
pub mod factory;
pub mod inspect;
pub mod nl;
pub mod pipeline;

pub use entity_type_search::{
    EntityTypeHit, EntityTypeSearchQuery, EntityTypeSearchResult, DEFAULT_CATALOG_THRESHOLD,
    DEFAULT_SCORE_THRESHOLD, DEFAULT_TOP_K, MAX_SAMPLE_NODE_IDS,
};
pub use pipeline::{GraphRun, IngestSummary, Pipeline};
