//! Postgres-backed storage for [`GraphSpecification`].
//!
//! Stores one specification per `scope` (e.g. one row per workspace,
//! tenant, or dataset) in a single JSONB column. The trait contract is
//! unchanged from the file backend — each storage instance is bound to a
//! single scope at construction time, so `load` / `save` operate on the
//! one row that scope owns.
//!
//! Enable with the `postgres` cargo feature. The implementation uses
//! `sqlx::PgPool` so the caller controls connection pooling and shares
//! the same pool across instances when many scopes coexist.

use std::sync::Arc;

use async_trait::async_trait;
use sqlx::PgPool;

use super::{GraphSpecification, GraphSpecificationStorage, GraphSpecificationStorageError};

/// Default Postgres table holding per-scope graph specifications.
pub const DEFAULT_GRAPH_SPECIFICATION_TABLE: &str = "linguagraph_graph_specifications";

/// Postgres-backed [`GraphSpecificationStorage`].
///
/// One instance addresses exactly one row in the configured table,
/// identified by `scope`. A missing row is treated as an empty
/// specification — matching the file backend's behaviour — so callers
/// can blindly call `load` after construction without having to seed.
#[derive(Clone)]
pub struct PostgresGraphSpecificationStorage {
    pool: Arc<PgPool>,
    table: String,
    scope: String,
}

impl PostgresGraphSpecificationStorage {
    /// Build a storage instance bound to a specific `scope`.
    pub fn new(pool: Arc<PgPool>, scope: impl Into<String>) -> Self {
        Self {
            pool,
            table: DEFAULT_GRAPH_SPECIFICATION_TABLE.to_string(),
            scope: scope.into(),
        }
    }

    /// Override the table name. The default
    /// (`linguagraph_graph_specifications`) is fine for most embedders;
    /// callers who share a database with other tenants can pick a
    /// schema-qualified name (e.g. `"app.lg_specs"`) here.
    pub fn with_table(mut self, table: impl Into<String>) -> Self {
        self.table = table.into();
        self
    }

    pub fn pool(&self) -> &Arc<PgPool> {
        &self.pool
    }

    pub fn table(&self) -> &str {
        &self.table
    }

    pub fn scope(&self) -> &str {
        &self.scope
    }

    /// Create the backing table if it doesn't already exist. Safe to
    /// call from every process startup; the `IF NOT EXISTS` clause
    /// keeps it idempotent.
    pub async fn ensure_table(&self) -> Result<(), GraphSpecificationStorageError> {
        let sql = format!(
            "CREATE TABLE IF NOT EXISTS {table} ( \
                scope TEXT PRIMARY KEY, \
                specification JSONB NOT NULL, \
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW() \
            )",
            table = self.table,
        );
        sqlx::query(&sql)
            .execute(self.pool.as_ref())
            .await
            .map_err(map_sqlx_err)?;
        Ok(())
    }
}

impl std::fmt::Debug for PostgresGraphSpecificationStorage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PostgresGraphSpecificationStorage")
            .field("table", &self.table)
            .field("scope", &self.scope)
            .finish_non_exhaustive()
    }
}

#[async_trait]
impl GraphSpecificationStorage for PostgresGraphSpecificationStorage {
    async fn load(&self) -> Result<GraphSpecification, GraphSpecificationStorageError> {
        let sql = format!(
            "SELECT specification FROM {table} WHERE scope = $1",
            table = self.table,
        );
        let row: Option<(serde_json::Value,)> = sqlx::query_as(&sql)
            .bind(&self.scope)
            .fetch_optional(self.pool.as_ref())
            .await
            .map_err(map_sqlx_err)?;

        match row {
            Some((value,)) => Ok(serde_json::from_value(value)?),
            None => Ok(GraphSpecification::new()),
        }
    }

    async fn save(
        &self,
        specification: &GraphSpecification,
    ) -> Result<(), GraphSpecificationStorageError> {
        let body = serde_json::to_value(specification)?;
        let sql = format!(
            "INSERT INTO {table} (scope, specification, updated_at) \
             VALUES ($1, $2, NOW()) \
             ON CONFLICT (scope) DO UPDATE SET \
                specification = EXCLUDED.specification, \
                updated_at = NOW()",
            table = self.table,
        );
        sqlx::query(&sql)
            .bind(&self.scope)
            .bind(&body)
            .execute(self.pool.as_ref())
            .await
            .map_err(map_sqlx_err)?;
        Ok(())
    }
}

fn map_sqlx_err(e: sqlx::Error) -> GraphSpecificationStorageError {
    GraphSpecificationStorageError::Backend(e.to_string())
}
