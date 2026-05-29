//! Ingestion orchestration.
//!
//! This is the layer between the [`crate::mapper`] (raw data → normalised
//! rows) and the [`crate::ast`] (typed query). It owns the *internal*
//! ingestion DSL — a JSON-shaped intermediate that callers and tests can
//! inspect — and the planner that turns extracted rows into
//! deterministic [`crate::ast::query::InsertQuery`] batches.

pub mod delete;
pub mod dsl;
pub mod graph;
pub mod planner;
pub mod soft_merge;

use thiserror::Error;

pub use delete::{DeletePlan, DeletePlanError, DiscoveredNodes};
pub use dsl::{InsertPlan, NodePlan, RelationPlan};
pub use graph::{
    plan_graph_with_registry, plan_graph_with_registry_and_prefix,
    plan_graph_with_registry_and_prefixes,
};
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

    #[error("'{0}' is a reserved entity label and cannot be used for user entities")]
    ReservedLabel(String),

    #[error("'{0}' is a reserved relation type and cannot be used for user relations")]
    ReservedRelation(String),

    #[error("identifier '{0}' is not a valid Cypher label/relation name even after sanitization")]
    InvalidLabel(String),

    #[error("relation in chunk '{chunk}' references unknown local entity id '{local_id}'")]
    UnknownLocalId { chunk: String, local_id: String },

    #[error("graph entity '{0}' is missing a primary key")]
    MissingGraphPrimaryKey(String),

    #[error("graph entity '{label}' is missing primary-key property '{field}'")]
    MissingGraphPrimaryKeyValue { label: String, field: String },

    #[error("relationship references unknown graph entity ref {0}")]
    UnknownGraphEntityRef(usize),

    #[error(
        "soft-merge resolver requires an embedder + GraphClient but the pipeline has none: \
         entity '{0}' uses PrimaryKey::Soft"
    )]
    SoftMergeBackendUnavailable(String),

    #[error("soft-merge resolver failed: {0}")]
    SoftMerge(String),
}
