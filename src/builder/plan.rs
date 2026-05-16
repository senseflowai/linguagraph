//! Logical plan: an explicit, ordered list of Cypher clauses.
//!
//! Today the Cypher emitter in [`super::cypher::build_read_with`]
//! orders MATCH/WHERE/post-match/sources/RETURN/ORDER BY/LIMIT/pre-match
//! by procedural call order, and prepends pre-match fragments at the
//! end via a string concat. The implicit phase order is a code smell
//! — the cure is to make it explicit data.
//!
//! [`LogicalPlan`] is that data. A plan is a flat `Vec<Clause>`; the
//! emitter is then a single forward pass with no buffering hacks. The
//! current emitter is **not** rewritten in this step — a follow-up
//! makes [`super::cypher::build_read_with`] go through `LogicalPlan`
//! when handler contributions are introduced as structured plan
//! fragments. For now this module is a typed contract that other
//! layers (tests, snapshots, future passes) can build against.

use crate::ast::query::{
    AggregateFn, Direction, EdgeTraversal, Literal, Predicate, PropertyRef, ReadQuery,
    ReturnClause, SortKey,
};
use crate::types::TypedPredicate;

/// A single emitted clause. Order in [`LogicalPlan::clauses`] is the
/// order in which the emitter writes them — no implicit phase
/// reordering.
#[derive(Debug, Clone)]
pub enum Clause {
    /// `CALL libqlink.search_reranked(...) YIELD id AS X, score AS Y`
    /// — handler-provided prelude that runs before MATCH.
    Prelude(PreludeClause),

    /// `MATCH (p:Person)-[r:KNOWS]->(p2:Person)`.
    Match(MatchClause),

    /// `OPTIONAL MATCH (p)-[w:WORKS_AT]->(c:Company)`.
    OptionalMatch(MatchClause),

    /// `WHERE p.age > $p0 AND ...`. Stored as a tree so emit-time
    /// normalization (predicate pushdown, dead-code elimination) is
    /// possible later.
    Where(BoolExpr),

    /// `WITH p, ...` projection / re-binding step.
    With(WithClause),

    /// `RETURN p.name AS name, sum(o.total) AS total`.
    Project(ProjectClause),

    /// `ORDER BY k1 ASC, k2 DESC`.
    OrderBy(Vec<OrderEntry>),

    /// `LIMIT n`.
    Limit(u32),
}

/// A handler-contributed prelude. Kept as both the raw text and the
/// list of *yielded* aliases the rest of the plan can reference.
///
/// Today's `EmitCtx::push_pre_match(String)` becomes a constructor for
/// this type; the rest of the plan now *knows* what variables a
/// prelude binds, which is what makes WITH propagation correct
/// without ad-hoc scans of `extra_order_by`.
#[derive(Debug, Clone)]
pub struct PreludeClause {
    pub text: String,
    pub yields: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct MatchClause {
    pub start: NodePattern,
    pub tail: Vec<TraversalPattern>,
}

#[derive(Debug, Clone)]
pub struct NodePattern {
    pub alias: String,
    /// Empty string means label-less (any-node).
    pub label: String,
}

#[derive(Debug, Clone)]
pub struct TraversalPattern {
    pub edge_alias: String,
    pub edge_label: String,
    pub direction: Direction,
    pub depth: Option<(u32, u32)>,
    pub target: NodePattern,
}

/// Boolean predicate tree at plan level. Carries the resolver-time
/// scratchpad (`Params`) and the handler dispatch (`Typed`) so the
/// emitter never has to look up a registry by stringly-typed name
/// again.
#[derive(Debug, Clone)]
pub enum BoolExpr {
    Scalar(Predicate),
    Typed(TypedPredicate),
    And(Vec<BoolExpr>),
    Or(Vec<BoolExpr>),
    Not(Box<BoolExpr>),
}

#[derive(Debug, Clone)]
pub struct WithClause {
    pub text: String,
}

#[derive(Debug, Clone)]
pub struct ProjectClause {
    pub items: Vec<ReturnItem>,
}

#[derive(Debug, Clone)]
pub enum ReturnItem {
    Field {
        field: PropertyRef,
        alias: Option<String>,
    },
    Aggregate {
        func: AggregateFn,
        field: PropertyRef,
        alias: Option<String>,
    },
    /// Pre-rendered expression, e.g. `__sources__ AS sources`. Used
    /// by normalization passes (sources/score injection) that compute
    /// projections from already-resolved variables.
    Expr { expr: String, alias: Option<String> },
}

#[derive(Debug, Clone)]
pub struct OrderEntry {
    pub key: OrderKey,
    pub dir: OrderDir,
}

#[derive(Debug, Clone)]
pub enum OrderKey {
    Projected(String),
    Property(PropertyRef),
    /// Pre-rendered expression key, e.g. `c__score_0`.
    Expr(String),
}

#[derive(Debug, Clone, Copy)]
pub enum OrderDir {
    Asc,
    Desc,
}

#[derive(Debug, Clone, Default)]
pub struct LogicalPlan {
    pub clauses: Vec<Clause>,
}

impl LogicalPlan {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push(&mut self, clause: Clause) -> &mut Self {
        self.clauses.push(clause);
        self
    }

    /// Project a [`ReadQuery`] into its baseline logical plan: one
    /// MATCH for the start pattern + required traversals, OPTIONAL
    /// MATCHes for optional ones, the user's WHERE filter, the user's
    /// RETURN, ORDER BY, LIMIT.
    ///
    /// This is the *bare-bones* plan — it carries no handler
    /// contributions yet. The current `build_read_with` does more
    /// (sources injection, post-match WITH chains, pre-match
    /// preludes); those become normalization passes that *transform*
    /// the plan in follow-up work.
    pub fn from_read_query(q: &ReadQuery) -> Self {
        let mut plan = Self::new();
        plan.push(Clause::Match(match_clause_for(
            &q.start.alias,
            &q.start.label,
            &q.traversals,
            false,
        )));
        // OPTIONAL MATCH per optional traversal.
        for t in q.traversals.iter().filter(|t| t.optional) {
            plan.push(Clause::OptionalMatch(MatchClause {
                start: NodePattern {
                    alias: t.from_alias.as_str().to_string(),
                    // re-reference; no label re-applied
                    label: String::new(),
                },
                tail: vec![traversal_pattern_of(t)],
            }));
        }
        if let Some(expr) = &q.filter {
            plan.push(Clause::Where(bool_expr_of(expr)));
        }
        plan.push(Clause::Project(ProjectClause {
            items: q.returns.iter().map(return_item_of).collect(),
        }));
        if !q.sort.is_empty() {
            plan.push(Clause::OrderBy(q.sort.iter().map(order_entry_of).collect()));
        }
        if let Some(n) = q.limit {
            plan.push(Clause::Limit(n));
        }
        plan
    }
}

fn match_clause_for(
    start_alias: &crate::ast::query::Alias,
    start_label: &str,
    traversals: &[EdgeTraversal],
    include_optional: bool,
) -> MatchClause {
    let start = NodePattern {
        alias: start_alias.as_str().to_string(),
        label: start_label.to_string(),
    };
    let mut tail = Vec::new();
    let mut current_endpoint = start_alias.clone();
    for t in traversals {
        if !include_optional && t.optional {
            continue;
        }
        if t.from_alias == current_endpoint {
            tail.push(traversal_pattern_of(t));
            current_endpoint = t.target.alias.clone();
        }
        // Traversals whose `from_alias` doesn't match the running
        // endpoint become their own MATCH clauses in the real
        // emitter; the simple LogicalPlan::from_read_query path keeps
        // them in `tail` so callers can detect and split.
    }
    MatchClause { start, tail }
}

fn traversal_pattern_of(t: &EdgeTraversal) -> TraversalPattern {
    TraversalPattern {
        edge_alias: t.edge_alias.as_str().to_string(),
        edge_label: t.edge_label.clone(),
        direction: t.direction,
        depth: t.depth.map(|d| (d.min, d.max)),
        target: NodePattern {
            alias: t.target.alias.as_str().to_string(),
            label: t.target.label.clone(),
        },
    }
}

fn bool_expr_of(e: &crate::ast::query::FilterExpression) -> BoolExpr {
    use crate::ast::query::FilterExpression as FE;
    match e {
        FE::Predicate(p) => BoolExpr::Scalar(p.clone()),
        FE::Typed(t) => BoolExpr::Typed(t.clone()),
        FE::And(parts) => BoolExpr::And(parts.iter().map(bool_expr_of).collect()),
        FE::Or(parts) => BoolExpr::Or(parts.iter().map(bool_expr_of).collect()),
        FE::Not(inner) => BoolExpr::Not(Box::new(bool_expr_of(inner))),
    }
}

fn return_item_of(r: &ReturnClause) -> ReturnItem {
    match r {
        ReturnClause::Field { field, alias } => ReturnItem::Field {
            field: field.clone(),
            alias: alias.clone(),
        },
        ReturnClause::Aggregate { func, field, alias } => ReturnItem::Aggregate {
            func: *func,
            field: field.clone(),
            alias: alias.clone(),
        },
    }
}

fn order_entry_of(s: &SortKey) -> OrderEntry {
    use crate::ast::query::{SortOrder, SortRef};
    OrderEntry {
        key: match &s.key {
            SortRef::Projected(name) => OrderKey::Projected(name.clone()),
            SortRef::Property(p) => OrderKey::Property(p.clone()),
        },
        dir: match s.order {
            SortOrder::Asc => OrderDir::Asc,
            SortOrder::Desc => OrderDir::Desc,
        },
    }
}

/// Marker so dead-code lint stays quiet while consumers migrate.
#[allow(dead_code)]
fn _literal_ref(_: &Literal) {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::query::*;

    fn alias(s: &str) -> Alias {
        Alias::new(s)
    }

    fn pref(a: &str, p: Option<&str>) -> PropertyRef {
        PropertyRef {
            alias: alias(a),
            property: p.map(str::to_string),
        }
    }

    fn rq() -> ReadQuery {
        ReadQuery {
            action: Action::Find,
            start: Node {
                label: "Person".into(),
                alias: alias("p"),
                prefix_label: None,
            },
            traversals: vec![
                EdgeTraversal {
                    from_alias: alias("p"),
                    edge_label: "KNOWS".into(),
                    edge_alias: alias("r"),
                    direction: Direction::Out,
                    target: Node {
                        label: "Person".into(),
                        alias: alias("p2"),
                        prefix_label: None,
                    },
                    depth: Some(Depth { min: 1, max: 1 }),
                    optional: false,
                },
                EdgeTraversal {
                    from_alias: alias("p"),
                    edge_label: "WORKS_AT".into(),
                    edge_alias: alias("w"),
                    direction: Direction::Out,
                    target: Node {
                        label: "Company".into(),
                        alias: alias("c"),
                        prefix_label: None,
                    },
                    depth: None,
                    optional: true,
                },
            ],
            filter: Some(FilterExpression::Predicate(Predicate {
                field: pref("p", Some("active")),
                op: ComparisonOp::Eq,
                value: Literal::Bool(true),
            })),
            returns: vec![ReturnClause::Field {
                field: pref("p", Some("name")),
                alias: Some("name".into()),
            }],
            group_by: vec![],
            sort: vec![],
            limit: Some(10),
        }
    }

    #[test]
    fn baseline_plan_clauses_appear_in_canonical_order() {
        let plan = LogicalPlan::from_read_query(&rq());
        let kinds: Vec<&'static str> = plan
            .clauses
            .iter()
            .map(|c| match c {
                Clause::Prelude(_) => "prelude",
                Clause::Match(_) => "match",
                Clause::OptionalMatch(_) => "optional_match",
                Clause::Where(_) => "where",
                Clause::With(_) => "with",
                Clause::Project(_) => "project",
                Clause::OrderBy(_) => "order_by",
                Clause::Limit(_) => "limit",
            })
            .collect();
        assert_eq!(
            kinds,
            vec!["match", "optional_match", "where", "project", "limit"]
        );
    }

    #[test]
    fn match_clause_chains_required_traversal_only() {
        let plan = LogicalPlan::from_read_query(&rq());
        let Clause::Match(m) = &plan.clauses[0] else {
            panic!("first clause should be MATCH");
        };
        assert_eq!(m.start.alias, "p");
        assert_eq!(m.start.label, "Person");
        // Required `KNOWS` is in the tail; optional `WORKS_AT` is a
        // separate OPTIONAL MATCH clause.
        assert_eq!(m.tail.len(), 1);
        assert_eq!(m.tail[0].edge_label, "KNOWS");
    }

    #[test]
    fn optional_traversal_becomes_optional_match_clause() {
        let plan = LogicalPlan::from_read_query(&rq());
        let Clause::OptionalMatch(om) = &plan.clauses[1] else {
            panic!("second clause should be OPTIONAL MATCH");
        };
        assert_eq!(om.start.alias, "p");
        assert_eq!(om.tail[0].edge_label, "WORKS_AT");
    }
}
