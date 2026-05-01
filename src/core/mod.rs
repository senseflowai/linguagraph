//! Orchestration layer.
//!
//! [`Pipeline`] wires the layers (DSL → AST → Cypher → DB) together. Callers
//! provide their own [`GraphClient`] implementation, which keeps the core
//! testable without a live Memgraph and ready for future driver swaps.

pub mod pipeline;

pub use pipeline::{IngestSummary, Pipeline};
