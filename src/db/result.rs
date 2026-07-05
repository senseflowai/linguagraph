//! Driver-agnostic query result types.
//!
//! We deliberately do not re-export `neo4rs` types here so the rest of the
//! codebase (and downstream consumers) stay decoupled from the driver.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::graph::{CHUNK_LABEL, SOURCE_LABEL};

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

/// High-level kind of a graph node referenced by a result column.
///
/// The Cypher builder tags each [`Column`] it emits with the kind of the
/// underlying binding: `c.id` with `c` matched as `:Chunk` becomes a
/// [`NodeType::Chunk`] column. Anything that doesn't match the built-in
/// `Source` or `Chunk` labels falls into [`NodeType::Entity`].
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
pub enum NodeType {
    Entity,
    Chunk,
    Source,
}

impl NodeType {
    /// Map a Cypher node label to its [`NodeType`]. The built-in labels
    /// `Source` and `Chunk` get their own variants; everything else is
    /// treated as a user-defined entity.
    pub fn from_label(label: &str) -> Self {
        match label {
            SOURCE_LABEL => NodeType::Source,
            CHUNK_LABEL => NodeType::Chunk,
            _ => NodeType::Entity,
        }
    }
}

/// A single projected column. Beyond the name it can carry the
/// [`NodeType`] of the underlying graph binding so downstream consumers
/// can interpret values without reparsing the query.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Column {
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub node_type: Option<NodeType>,
}

impl Column {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            node_type: None,
        }
    }

    pub fn with_type(name: impl Into<String>, node_type: NodeType) -> Self {
        Self {
            name: name.into(),
            node_type: Some(node_type),
        }
    }
}

impl From<&str> for Column {
    fn from(name: &str) -> Self {
        Column::new(name)
    }
}

impl From<String> for Column {
    fn from(name: String) -> Self {
        Column::new(name)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Row {
    pub fields: BTreeMap<String, Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct QueryResult {
    pub columns: Vec<Column>,
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
