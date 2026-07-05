//! Public error type for the [`crate::api`] surface.
//!
//! Everything the REST service can observe funnels through
//! [`GraphError`]. The internal, layer-specific [`crate::Error`] is kept
//! out of the public contract on purpose — a downstream service should
//! match on a small, stable set of failure modes (not found, cost
//! budget, tenant isolation, …) rather than on the crate's internal
//! DSL/AST/builder taxonomy. Internal errors collapse into
//! [`GraphError::Backend`] with their `Display` string preserved for
//! diagnostics.

use thiserror::Error;

use super::model::Cost;

/// Result alias used throughout the public API.
pub type Result<T> = std::result::Result<T, GraphError>;

/// Stable, service-facing error surface.
///
/// `#[non_exhaustive]` so new failure modes can be added without a
/// breaking change; downstream `match`es must carry a wildcard arm.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum GraphError {
    /// The requested entity, relation or plan does not exist (or is not
    /// visible under the current tenant scope).
    #[error("not found")]
    NotFound,

    /// The natural-language question could not be compiled into a plan.
    /// Carries a human-readable reason suitable for surfacing to the UI.
    #[error("invalid question: {0}")]
    InvalidQuestion(String),

    /// A traversal was requested without the mandatory `depth`/`limit`
    /// bounds. The library refuses unbounded traversals to protect the
    /// database from runaway "hairball" expansions.
    #[error("unbounded traversal: depth and limit are required")]
    UnboundedTraversal,

    /// The compiled plan's estimated cost exceeds the caller's budget.
    /// The rejected estimate is attached so the service can report it.
    #[error("cost exceeded: estimated {0:?}")]
    CostExceeded(Cost),

    /// The operation did not complete within its `timeout`.
    #[error("operation timed out")]
    Timeout,

    /// A cross-tenant access was attempted and blocked. Surfaced
    /// explicitly (rather than as a silent empty result) so misconfigured
    /// callers fail loudly.
    #[error("tenant isolation violation")]
    TenantIsolation,

    /// The backend (Memgraph, the embedder, the network …) failed. The
    /// wrapped string is the underlying error's `Display` form.
    #[error("backend error: {0}")]
    Backend(String),

    /// Configuration supplied to the builder was invalid or incomplete.
    #[error("configuration error: {0}")]
    Config(String),

    /// The capability is part of the public contract but not yet wired
    /// to a backend in this build. Distinct from [`GraphError::Backend`]
    /// so callers can tell "the graph failed" from "this endpoint is not
    /// implemented here yet".
    #[error("unsupported: {0}")]
    Unsupported(String),
}

impl From<crate::Error> for GraphError {
    fn from(e: crate::Error) -> Self {
        // Preserve the internal error's message; collapse the internal
        // taxonomy into the stable public `Backend` bucket.
        GraphError::Backend(e.to_string())
    }
}

impl From<crate::db::DbError> for GraphError {
    fn from(e: crate::db::DbError) -> Self {
        GraphError::Backend(e.to_string())
    }
}
