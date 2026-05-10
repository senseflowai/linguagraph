//! Ingestion orchestration.
//!
//! This is the layer between the [`crate::mapper`] (raw data → normalised
//! rows) and the [`crate::ast`] (typed query). It owns the *internal*
//! ingestion DSL — a JSON-shaped intermediate that callers and tests can
//! inspect — and the planner that turns extracted rows into
//! deterministic [`crate::ast::query::InsertQuery`] batches.

pub mod dsl;
pub mod planner;

use thiserror::Error;

pub use dsl::{InsertPlan, NodePlan, RelationPlan};
pub use planner::{plan, plan_with_options, plan_with_registry, PlannerOptions};

use crate::ast::AstError;
use crate::mapper::MapperError;

#[derive(Debug, Error)]
pub enum IngestError {
    #[error(transparent)]
    Mapper(#[from] MapperError),

    #[error(transparent)]
    Ast(#[from] AstError),

    #[error("relationship references unknown entity type '{0}'")]
    UnknownEntityType(String),

    #[error("max_batch_size must be greater than zero")]
    InvalidBatchSize,

    #[error("type handler error during ingestion: {0}")]
    Type(String),
}
