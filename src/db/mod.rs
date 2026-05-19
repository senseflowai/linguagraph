//! Database access layer.
//!
//! All consumers of the database go through the [`GraphClient`] trait. The
//! production implementation in [`memgraph`] wraps `neo4rs`; the test
//! implementation in [`mock`] returns canned results without a network.

pub mod introspect;
pub mod memgraph;
pub mod mock;
pub mod result;

use async_trait::async_trait;
use thiserror::Error;

pub use introspect::{introspect_schema, IntrospectOptions};
pub use memgraph::MemgraphClient;
pub use mock::MockClient;
pub use result::{Column, NodeType, QueryResult, Row, Value};

use crate::builder::CypherQuery;

#[derive(Debug, Error)]
pub enum DbError {
    #[error("connection failed: {0}")]
    Connection(String),

    #[error("query failed: {0}")]
    Query(String),

    #[error("unsupported parameter type: {0}")]
    UnsupportedParameter(String),

    #[error("result decoding failed: {0}")]
    Decode(String),
}

/// Abstract async client. The CLI and pipeline only depend on this — never
/// on a concrete database driver.
#[async_trait]
pub trait GraphClient: Send + Sync {
    async fn execute(&self, query: &CypherQuery) -> Result<QueryResult, DbError>;

    /// Optional: introspect labels, edge types, and properties. Default impl
    /// returns an empty schema so backends can opt in incrementally.
    async fn schema(&self) -> Result<crate::prompt::GraphSchema, DbError> {
        Ok(crate::prompt::GraphSchema::default())
    }
}
