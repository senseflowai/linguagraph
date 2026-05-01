//! Strongly-typed query model.
//!
//! Every node references aliases by [`Alias`] (a thin newtype) rather than
//! by raw `String`, so a typo can't slip from one stage to the next.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub struct Alias(pub String);

impl Alias {
    pub fn new(s: impl Into<String>) -> Self {
        Alias(s.into())
    }
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for Alias {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// Reference to either a node's property or the bound entity itself.
#[derive(Debug, Clone, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub struct PropertyRef {
    pub alias: Alias,
    pub property: Option<String>,
}

impl PropertyRef {
    pub fn parse(s: &str) -> Option<Self> {
        let mut it = s.split('.');
        let alias = it.next()?.to_string();
        let prop = it.next().map(str::to_string);
        if it.next().is_some() || alias.is_empty() {
            return None;
        }
        Some(Self { alias: Alias(alias), property: prop })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Query {
    pub action: Action,
    pub start: Node,
    pub traversals: Vec<EdgeTraversal>,
    pub filter: Option<FilterExpression>,
    pub returns: Vec<ReturnClause>,
    pub group_by: Vec<PropertyRef>,
    pub sort: Vec<SortKey>,
    pub limit: Option<u32>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum Action {
    Find,
    Aggregate,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Node {
    pub label: String,
    pub alias: Alias,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EdgeTraversal {
    pub edge_label: String,
    pub edge_alias: Alias,
    pub direction: Direction,
    pub target: Node,
    pub depth: Option<Depth>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum Direction {
    Out,
    In,
    Both,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub struct Depth {
    pub min: u32,
    pub max: u32,
}

/// Boolean expression tree. Simple enough today; ready to grow into NOT/OR.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FilterExpression {
    Predicate(Predicate),
    And(Vec<FilterExpression>),
    Or(Vec<FilterExpression>),
    Not(Box<FilterExpression>),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Predicate {
    pub field: PropertyRef,
    pub op: ComparisonOp,
    pub value: Literal,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum ComparisonOp {
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

/// Restricted literal set. We only let through what Memgraph parameters
/// can actually carry safely — no arbitrary nested objects.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum Literal {
    String(String),
    Bool(bool),
    Int(i64),
    Float(f64),
    List(Vec<Literal>),
    Null,
}

impl Literal {
    pub fn from_json(v: &serde_json::Value) -> Option<Self> {
        Some(match v {
            serde_json::Value::Null => Literal::Null,
            serde_json::Value::Bool(b) => Literal::Bool(*b),
            serde_json::Value::Number(n) => {
                if let Some(i) = n.as_i64() {
                    Literal::Int(i)
                } else if let Some(f) = n.as_f64() {
                    Literal::Float(f)
                } else {
                    return None;
                }
            }
            serde_json::Value::String(s) => Literal::String(s.clone()),
            serde_json::Value::Array(items) => {
                let mut out = Vec::with_capacity(items.len());
                for it in items {
                    out.push(Literal::from_json(it)?);
                }
                Literal::List(out)
            }
            serde_json::Value::Object(_) => return None,
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReturnClause {
    Field {
        field: PropertyRef,
        alias: Option<String>,
    },
    Aggregate {
        func: AggregateFn,
        field: PropertyRef,
        alias: Option<String>,
    },
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum AggregateFn {
    Count,
    Sum,
    Avg,
    Min,
    Max,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SortKey {
    /// Either an alias defined in the RETURN list or a qualified property.
    pub key: SortRef,
    pub order: SortOrder,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SortRef {
    /// Alias in the projection (e.g. "total_spent").
    Projected(String),
    /// Qualified entity property.
    Property(PropertyRef),
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum SortOrder {
    Asc,
    Desc,
}
