//! Memgraph (Bolt) implementation of [`GraphClient`].
//!
//! Implemented on top of `neo4rs`, which speaks the same Bolt dialect as
//! Memgraph 2.x. Connection pooling is handled by the driver
//! (`max_connections`).

use std::time::Duration;

use async_trait::async_trait;
use neo4rs::{
    BoltBoolean, BoltFloat, BoltInteger, BoltList, BoltMap, BoltNull, BoltString, BoltType,
    ConfigBuilder, Graph,
};
use serde_json::Value as Json;

use crate::ast::query::Literal;
use crate::builder::CypherQuery;
use crate::config::DatabaseConfig;
use crate::prompt::GraphSchema;

use super::result::{Column, QueryResult, Row, Value};
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
            .db(cfg.database.clone())
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
            bolt = bolt.param(name, literal_to_bolt(value));
        }

        let mut stream = tokio::time::timeout(self.timeout, self.graph.execute(bolt))
            .await
            .map_err(|_| DbError::Query("query timeout".into()))?
            .map_err(|e| DbError::Query(e.to_string()))?;

        let mut rows: Vec<Row> = Vec::new();
        let mut column_names: Vec<String> = Vec::new();
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

            if column_names.is_empty() {
                column_names = map.keys().cloned().collect();
            }
            let mut row = Row::default();
            for (k, v) in map {
                row.fields.insert(k, Value::Json(v));
            }
            rows.push(row);
        }

        let columns = merge_columns(&column_names, &q.columns);
        Ok(QueryResult { columns, rows })
    }

    async fn schema(&self) -> Result<GraphSchema, DbError> {
        // Portable Cypher introspection — works on Memgraph community
        // and any Neo4j-compatible backend. Sample size defaults to 100;
        // callers that need finer control can call
        // [`crate::db::introspect_schema`] directly.
        super::introspect::introspect_schema(self, super::introspect::IntrospectOptions::default())
            .await
    }
}

/// Merge the column ordering observed on the wire with the typed
/// [`Column`] list the builder computed from the AST. The wire order
/// wins so the driver's actual response stays authoritative; types are
/// looked up by name. Columns that appear in the response but not in
/// the builder's list (raw Cypher, schema introspection, etc.) keep
/// the default `node_type: None`.
fn merge_columns(observed: &[String], typed: &[Column]) -> Vec<Column> {
    if typed.is_empty() {
        return observed.iter().map(|name| Column::new(name.clone())).collect();
    }
    let type_map: std::collections::BTreeMap<&str, Option<super::result::NodeType>> = typed
        .iter()
        .map(|c| (c.name.as_str(), c.node_type))
        .collect();
    observed
        .iter()
        .map(|name| Column {
            name: name.clone(),
            node_type: type_map.get(name.as_str()).copied().flatten(),
        })
        .collect()
}

/// Recursively translate a [`Literal`] into a [`BoltType`]. The mapping
/// is total: every variant of `Literal` corresponds to a Bolt primitive
/// that the driver knows how to serialise.
pub(crate) fn literal_to_bolt(lit: &Literal) -> BoltType {
    match lit {
        Literal::Null => BoltType::Null(BoltNull),
        Literal::Bool(b) => BoltType::Boolean(BoltBoolean::new(*b)),
        Literal::Int(i) => BoltType::Integer(BoltInteger::new(*i)),
        Literal::Float(f) => BoltType::Float(BoltFloat::new(*f)),
        Literal::String(s) => BoltType::String(BoltString::new(s)),
        Literal::List(items) => {
            let mut out = BoltList::with_capacity(items.len());
            for it in items {
                out.push(literal_to_bolt(it));
            }
            BoltType::List(out)
        }
        Literal::Object(map) => {
            let mut out = BoltMap::with_capacity(map.len());
            for (k, v) in map {
                out.put(BoltString::new(k), literal_to_bolt(v));
            }
            BoltType::Map(out)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    #[test]
    fn literal_object_round_trips_to_bolt_map() {
        let mut inner = BTreeMap::new();
        inner.insert("name".to_string(), Literal::String("cam-1".into()));
        let lit = Literal::List(vec![Literal::Object(inner)]);
        match literal_to_bolt(&lit) {
            BoltType::List(list) => {
                assert_eq!(list.len(), 1);
                match list.get(0).unwrap() {
                    BoltType::Map(_) => {}
                    other => panic!("expected map, got {other:?}"),
                }
            }
            other => panic!("expected list, got {other:?}"),
        }
    }
}
