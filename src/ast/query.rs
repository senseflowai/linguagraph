//! Strongly-typed query model.
//!
//! The AST is a closed sum: every operation the rest of the stack can perform
//! is one variant of [`Query`]. Today there are two — read traversals and
//! batched ingestion — and adding a third (e.g. schema migration) means
//! adding a variant here and a builder for it.

use std::collections::BTreeMap;

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
        Some(Self {
            alias: Alias(alias),
            property: prop,
        })
    }
}

/// Top-level AST node.
///
/// Read and insert queries are deliberately kept in the same enum so that
/// callers (CLI, pipeline, future planners) can hold a single value and
/// dispatch on its kind. Each variant carries a self-contained, validated
/// structure — by the time it exists, the builder may compile it without
/// further checks.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Query {
    Read(ReadQuery),
    Insert(InsertQuery),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReadQuery {
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
    /// Already-bound alias the edge starts from. Defaults to the query's
    /// start node when the DSL omits `from`; the builder uses this to
    /// decide whether to chain the traversal onto the previous MATCH or
    /// emit a new MATCH clause.
    pub from_alias: Alias,
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
///
/// `Typed` carries a fully-resolved predicate produced by a
/// [`crate::types::TypeHandler`] during lowering. The Cypher builder
/// dispatches `Typed` back through the handler at emit time — it never
/// inspects the type id itself.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FilterExpression {
    Predicate(Predicate),
    Typed(crate::types::TypedPredicate),
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
/// can actually carry safely — no non-finite numbers, no foreign types.
///
/// `Object` was added to support batched ingestion, where we ship a
/// `List(Object(...))` as the parameter to an `UNWIND` clause. The DSL
/// front-end still rejects objects (see [`Literal::from_json`]) — they
/// only enter the system through the ingestion planner, which constructs
/// them directly.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum Literal {
    String(String),
    Bool(bool),
    Int(i64),
    Float(f64),
    List(Vec<Literal>),
    Object(BTreeMap<String, Literal>),
    Null,
}

impl Literal {
    /// Lower a JSON scalar/array/null into a [`Literal`].
    ///
    /// Returns `None` for objects (the DSL filter language forbids them) or
    /// for non-finite numbers. The ingestion path bypasses this and builds
    /// objects directly.
    pub fn from_json(v: &serde_json::Value) -> Option<Self> {
        Some(match v {
            serde_json::Value::Null => Literal::Null,
            serde_json::Value::Bool(b) => Literal::Bool(*b),
            serde_json::Value::Number(n) => {
                if let Some(i) = n.as_i64() {
                    Literal::Int(i)
                } else if let Some(f) = n.as_f64() {
                    if !f.is_finite() {
                        return None;
                    }
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

    /// Lower any JSON value, *including* objects. Used by the ingestion
    /// pipeline where arbitrary nested data is acceptable as a parameter.
    pub fn from_json_any(v: &serde_json::Value) -> Option<Self> {
        Some(match v {
            serde_json::Value::Object(map) => {
                let mut out = BTreeMap::new();
                for (k, vv) in map {
                    out.insert(k.clone(), Literal::from_json_any(vv)?);
                }
                Literal::Object(out)
            }
            serde_json::Value::Array(items) => {
                let mut out = Vec::with_capacity(items.len());
                for it in items {
                    out.push(Literal::from_json_any(it)?);
                }
                Literal::List(out)
            }
            other => Literal::from_json(other)?,
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

// ─── Insert ──────────────────────────────────────────────────────────────────

/// Batched, MERGE-semantic insert.
///
/// The query is precomputed: every batch is a fully resolved set of rows
/// whose values are already [`Literal`]s. The builder only has to render
/// `UNWIND $rows AS row …` template strings; it never re-checks identity
/// or types.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InsertQuery {
    pub node_batches: Vec<NodeBatch>,
    pub relation_batches: Vec<RelationBatch>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeBatch {
    /// Cypher node label (must be a valid identifier).
    pub label: String,
    /// Property name used as the merge key (typically `id`).
    pub merge_on: String,
    /// Each row contributes one MERGE.
    pub rows: Vec<NodeRow>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeRow {
    /// Stable identifier; the value of the `merge_on` property.
    pub id: Literal,
    /// All other properties (the merge key is added automatically by the
    /// builder so callers never have to remember to include it).
    pub props: BTreeMap<String, Literal>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelationBatch {
    pub rel_type: String,
    pub from_label: String,
    pub from_key: String,
    pub to_label: String,
    pub to_key: String,
    pub rows: Vec<RelationRow>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct RelationRow {
    /// The merge-key value of the source node.
    pub from_id: Literal,
    /// The merge-key value of the target node.
    pub to_id: Literal,
    /// Relationship properties set after MERGE.
    #[serde(default)]
    pub props: BTreeMap<String, Literal>,
}

// `Literal` derives `PartialEq`; we additionally need `Eq` and `Hash` so
// `RelationRow` can live inside hash sets for deduplication. Floats break
// `Eq`/`Hash`, but identifiers produced by the planner are scalars
// (string/int/null in practice) so the manual impls hold.
impl Eq for Literal {}

impl std::hash::Hash for Literal {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        std::mem::discriminant(self).hash(state);
        match self {
            Literal::String(s) => s.hash(state),
            Literal::Bool(b) => b.hash(state),
            Literal::Int(i) => i.hash(state),
            Literal::Float(f) => f.to_bits().hash(state),
            Literal::List(items) => {
                items.len().hash(state);
                for it in items {
                    it.hash(state);
                }
            }
            Literal::Object(map) => {
                map.len().hash(state);
                for (k, v) in map {
                    k.hash(state);
                    v.hash(state);
                }
            }
            Literal::Null => {}
        }
    }
}
