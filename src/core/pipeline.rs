//! End-to-end orchestration: DSL → AST → Cypher → DB.

use std::sync::Arc;

use crate::ast::{from_dsl, query::Query};
use crate::builder::{self, CypherQuery};
use crate::config::Config;
use crate::db::{GraphClient, QueryResult};
use crate::dsl::DslQuery;
use crate::error::Result;

/// High-level entrypoint used by the CLI and library consumers.
///
/// The pipeline is cheap to clone — its only state is an `Arc<dyn GraphClient>`
/// and a snapshot of the relevant config knobs.
#[derive(Clone)]
pub struct Pipeline {
    client: Arc<dyn GraphClient>,
    max_depth: u32,
    default_limit: u32,
}

impl std::fmt::Debug for Pipeline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Pipeline")
            .field("max_depth", &self.max_depth)
            .field("default_limit", &self.default_limit)
            .finish_non_exhaustive()
    }
}

impl Pipeline {
    pub fn new(client: Arc<dyn GraphClient>, config: &Config) -> Self {
        Self {
            client,
            max_depth: config.query.max_traversal_depth,
            default_limit: config.query.default_limit,
        }
    }

    /// Lower a DSL document to the typed AST. Pure; no I/O.
    pub fn lower(&self, dsl: DslQuery) -> Result<Query> {
        let mut q = from_dsl::lower(dsl, self.max_depth)?;
        if q.limit.is_none() {
            q.limit = Some(self.default_limit);
        }
        Ok(q)
    }

    /// Compile a DSL document all the way to a parameterized Cypher query.
    pub fn compile(&self, dsl: DslQuery) -> Result<CypherQuery> {
        let ast = self.lower(dsl)?;
        Ok(builder::build(&ast)?)
    }

    /// Compile and execute against the configured graph client.
    pub async fn run(&self, dsl: DslQuery) -> Result<QueryResult> {
        let cypher = self.compile(dsl)?;
        Ok(self.client.execute(&cypher).await?)
    }
}
