//! Memgraph (Bolt) implementation of [`GraphClient`].
//!
//! Implemented on top of `neo4rs`, which speaks the same Bolt dialect as
//! Memgraph 2.x. Connection pooling is handled by the driver
//! (`max_connections`).

use std::time::Duration;

use async_trait::async_trait;
use neo4rs::{ConfigBuilder, Graph};
use serde_json::Value as Json;

use crate::ast::query::Literal;
use crate::builder::CypherQuery;
use crate::config::DatabaseConfig;
use crate::prompt::GraphSchema;

use super::result::{QueryResult, Row, Value};
use super::{DbError, GraphClient};

pub struct MemgraphClient {
    graph: Graph,
    timeout: Duration,
}

impl std::fmt::Debug for MemgraphClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MemgraphClient")
            .field("timeout", &self.timeout)
            .finish_non_exhaustive()
    }
}

impl MemgraphClient {
    /// Open a pooled bolt connection.
    pub async fn connect(cfg: &DatabaseConfig) -> Result<Self, DbError> {
        let config = ConfigBuilder::default()
            .uri(cfg.uri.clone())
            .user(cfg.user.clone())
            .password(cfg.password.clone())
            .max_connections(cfg.max_connections as usize)
            .build()
            .map_err(|e| DbError::Connection(e.to_string()))?;
        let graph = Graph::connect(config)
            .await
            .map_err(|e| DbError::Connection(e.to_string()))?;
        Ok(Self {
            graph,
            timeout: Duration::from_secs(cfg.query_timeout_secs),
        })
    }
}

#[async_trait]
impl GraphClient for MemgraphClient {
    async fn execute(&self, q: &CypherQuery) -> Result<QueryResult, DbError> {
        let mut bolt = neo4rs::query(&q.text);
        for (name, value) in &q.params {
            bolt = bind_param(bolt, name, value)?;
        }

        let mut stream = tokio::time::timeout(self.timeout, self.graph.execute(bolt))
            .await
            .map_err(|_| DbError::Query("query timeout".into()))?
            .map_err(|e| DbError::Query(e.to_string()))?;

        let mut rows: Vec<Row> = Vec::new();
        let mut columns: Vec<String> = Vec::new();
        loop {
            let next = tokio::time::timeout(self.timeout, stream.next())
                .await
                .map_err(|_| DbError::Query("row read timeout".into()))?
                .map_err(|e| DbError::Query(e.to_string()))?;
            let Some(record) = next else {
                break;
            };

            // BoltMap → JSON object via the row's strict deserializer.
            // We never reach for typed BoltType values directly so the rest
            // of the pipeline stays driver-agnostic.
            let json: Json = record
                .to_strict()
                .map_err(|e| DbError::Decode(e.to_string()))?;

            let Json::Object(map) = json else {
                return Err(DbError::Decode(
                    "expected object row from strict decoder".into(),
                ));
            };

            if columns.is_empty() {
                columns = map.keys().cloned().collect();
            }
            let mut row = Row::default();
            for (k, v) in map {
                row.fields.insert(k, Value::Json(v));
            }
            rows.push(row);
        }

        Ok(QueryResult { columns, rows })
    }

    async fn schema(&self) -> Result<GraphSchema, DbError> {
        // A real schema crawl needs `SHOW SCHEMA INFO` (Memgraph) or
        // `CALL db.schema.visualization()` (Neo4j-compatible). Both are
        // backend-specific, so the default impl returns an empty schema and
        // operators are expected to provide one via `--schema <file>`.
        Ok(GraphSchema::default())
    }
}

/// Bind one [`Literal`] to a Bolt parameter. The match is exhaustive so any
/// new variant added to [`Literal`] will fail to compile here.
fn bind_param(q: neo4rs::Query, name: &str, lit: &Literal) -> Result<neo4rs::Query, DbError> {
    Ok(match lit {
        Literal::String(s) => q.param(name, s.clone()),
        Literal::Bool(b) => q.param(name, *b),
        Literal::Int(i) => q.param(name, *i),
        Literal::Float(f) => q.param(name, *f),
        Literal::Null => q.param(name, Option::<String>::None),
        Literal::List(items) => bind_list(q, name, items)?,
    })
}

fn bind_list(q: neo4rs::Query, name: &str, items: &[Literal]) -> Result<neo4rs::Query, DbError> {
    if items.iter().all(|x| matches!(x, Literal::Int(_))) {
        let v: Vec<i64> = items
            .iter()
            .map(|x| match x {
                Literal::Int(i) => *i,
                _ => unreachable!(),
            })
            .collect();
        Ok(q.param(name, v))
    } else if items.iter().all(|x| matches!(x, Literal::String(_))) {
        let v: Vec<String> = items
            .iter()
            .map(|x| match x {
                Literal::String(s) => s.clone(),
                _ => unreachable!(),
            })
            .collect();
        Ok(q.param(name, v))
    } else if items.iter().all(|x| matches!(x, Literal::Float(_))) {
        let v: Vec<f64> = items
            .iter()
            .map(|x| match x {
                Literal::Float(f) => *f,
                _ => unreachable!(),
            })
            .collect();
        Ok(q.param(name, v))
    } else if items.iter().all(|x| matches!(x, Literal::Bool(_))) {
        let v: Vec<bool> = items
            .iter()
            .map(|x| match x {
                Literal::Bool(b) => *b,
                _ => unreachable!(),
            })
            .collect();
        Ok(q.param(name, v))
    } else {
        Err(DbError::UnsupportedParameter(
            "heterogeneous or nested list literals are not supported as parameters".into(),
        ))
    }
}
