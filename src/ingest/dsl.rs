//! Internal ingestion DSL.
//!
//! `InsertPlan` is a pre-AST representation — JSON-shaped, easy to print
//! and snapshot in tests, and one-to-one with the [`crate::ast::query::InsertQuery`]
//! it lowers into. Keeping this layer separate from the AST lets us evolve
//! either side independently (e.g. add a streaming planner that emits
//! `InsertPlan`s incrementally) without touching the builder.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::ast::query::Literal;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InsertPlan {
    /// Discriminator field — always `"insert"`. Makes the JSON shape
    /// self-describing if ever serialised next to a `"read"` plan.
    #[serde(default = "default_action")]
    pub action: String,
    pub nodes: Vec<NodePlan>,
    pub relations: Vec<RelationPlan>,
}

fn default_action() -> String {
    "insert".to_string()
}

impl InsertPlan {
    pub fn new() -> Self {
        Self {
            action: default_action(),
            nodes: Vec::new(),
            relations: Vec::new(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.iter().all(|n| n.rows.is_empty())
            && self.relations.iter().all(|r| r.rows.is_empty())
    }

    pub fn total_node_rows(&self) -> usize {
        self.nodes.iter().map(|n| n.rows.len()).sum()
    }

    pub fn total_relation_rows(&self) -> usize {
        self.relations.iter().map(|r| r.rows.len()).sum()
    }
}

impl Default for InsertPlan {
    fn default() -> Self {
        Self::new()
    }
}

/// A homogeneous batch of nodes to MERGE.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodePlan {
    pub label: String,
    /// Property name that uniquely identifies a node of this label.
    pub merge_on: String,
    pub rows: Vec<NodeData>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeData {
    pub id: Literal,
    pub props: BTreeMap<String, Literal>,
}

/// A homogeneous batch of relationships to MERGE between two labels.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelationPlan {
    pub rel_type: String,
    pub from_label: String,
    pub from_key: String,
    pub to_label: String,
    pub to_key: String,
    pub rows: Vec<RelationData>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelationData {
    pub from_id: Literal,
    pub to_id: Literal,
}
