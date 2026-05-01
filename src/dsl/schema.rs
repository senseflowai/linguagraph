//! Strongly-typed mirror of the JSON DSL.
//!
//! No business logic lives here — these types only describe what is legal
//! syntactically. Anything semantic (alias must exist, depth must be > 0,
//! aggregate may not appear with `find`, …) is enforced in [`super::parser`]
//! or in [`crate::ast`].

use serde::{Deserialize, Serialize};

/// Top-level DSL document.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DslQuery {
    pub action: Action,
    pub start: NodePattern,
    #[serde(default)]
    pub traversals: Vec<Traversal>,
    #[serde(default)]
    pub filters: Vec<Filter>,
    #[serde(default, rename = "return")]
    pub return_: Vec<ReturnItem>,
    #[serde(default)]
    pub group_by: Vec<String>,
    #[serde(default)]
    pub sort: Vec<SortItem>,
    #[serde(default)]
    pub limit: Option<u32>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Action {
    Find,
    Aggregate,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct NodePattern {
    pub label: String,
    pub alias: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Traversal {
    pub edge: EdgePattern,
    pub target: NodePattern,
    #[serde(default)]
    pub depth: Option<DepthRange>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EdgePattern {
    pub label: String,
    pub alias: String,
    pub direction: Direction,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Direction {
    Out,
    In,
    Both,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub struct DepthRange {
    pub min: u32,
    pub max: u32,
}

/// A single filter predicate. The DSL keeps this flat — boolean composition
/// across multiple filters is implicit AND. This is enough for the queries an
/// LLM emits in practice and keeps the schema small.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Filter {
    /// Qualified property reference, e.g. `"p.age"`.
    pub field: String,
    pub op: FilterOp,
    pub value: serde_json::Value,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum FilterOp {
    Eq,
    Neq,
    Gt,
    Gte,
    Lt,
    Lte,
    In,
    Contains,
    StartsWith,
    EndsWith,
}

/// One projected column. Either a plain field or an aggregation.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum ReturnItem {
    Aggregate {
        aggregate: AggregateFn,
        field: String,
        #[serde(default)]
        alias: Option<String>,
    },
    Field {
        field: String,
        #[serde(default)]
        alias: Option<String>,
    },
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum AggregateFn {
    Count,
    Sum,
    Avg,
    Min,
    Max,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SortItem {
    pub field: String,
    #[serde(default)]
    pub order: SortOrder,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "lowercase")]
pub enum SortOrder {
    #[default]
    Asc,
    Desc,
}
