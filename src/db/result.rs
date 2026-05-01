//! Driver-agnostic query result types.
//!
//! We deliberately do not re-export `neo4rs` types here so the rest of the
//! codebase (and downstream consumers) stay decoupled from the driver.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

/// A scalar cell value. Nested structures coming from the database are
/// flattened to JSON — that's the common shape every consumer can handle.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum Value {
    Null,
    Bool(bool),
    Int(i64),
    Float(f64),
    String(String),
    Json(serde_json::Value),
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Row {
    pub fields: BTreeMap<String, Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct QueryResult {
    pub columns: Vec<String>,
    pub rows: Vec<Row>,
}

impl QueryResult {
    pub fn empty() -> Self {
        Self::default()
    }

    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }
}
