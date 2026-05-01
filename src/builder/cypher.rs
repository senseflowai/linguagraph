//! Public entrypoint for the Cypher builder.

use thiserror::Error;

use crate::ast::query::*;

use super::cursor::{Cursor, CypherQuery};
use super::insert::{build_insert, InsertError};
use super::{match_part, return_part, where_part};

#[derive(Debug, Error)]
pub enum BuilderError {
    #[error("query has no projection (RETURN list is empty)")]
    EmptyReturn,

    #[error("insert builder error: {0}")]
    Insert(#[from] InsertError),
}

/// Compile a [`ReadQuery`] into a single parameterized [`CypherQuery`].
pub fn build_read(query: &ReadQuery) -> Result<CypherQuery, BuilderError> {
    if query.returns.is_empty() {
        return Err(BuilderError::EmptyReturn);
    }

    let mut cur = Cursor::new();

    match_part::write_match(&mut cur, query);
    if let Some(filter) = &query.filter {
        where_part::write_where(&mut cur, filter);
    }
    return_part::write_return(&mut cur, query);
    return_part::write_order_by(&mut cur, &query.sort);
    if let Some(limit) = query.limit {
        return_part::write_limit(&mut cur, limit);
    }

    Ok(cur.finish())
}

/// Backwards-compatible alias for [`build_read`].
///
/// Existing call sites that operate on the read AST continue to compile;
/// new code that holds a [`Query`] enum should use [`build`].
pub fn build(query: &ReadQuery) -> Result<CypherQuery, BuilderError> {
    build_read(query)
}

/// Compile any [`Query`] variant.
///
/// Reads return a single batch; inserts return one batch per node-/relation-
/// batch in the [`InsertQuery`].
pub fn compile(query: &Query) -> Result<Vec<CypherQuery>, BuilderError> {
    match query {
        Query::Read(r) => Ok(vec![build_read(r)?]),
        Query::Insert(i) => Ok(build_insert(i)?),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn alias(s: &str) -> Alias {
        Alias::new(s)
    }

    fn pref(a: &str, p: Option<&str>) -> PropertyRef {
        PropertyRef { alias: alias(a), property: p.map(str::to_string) }
    }

    #[test]
    fn builds_basic_match_with_filter() {
        let q = ReadQuery {
            action: Action::Find,
            start: Node { label: "Person".into(), alias: alias("p") },
            traversals: vec![],
            filter: Some(FilterExpression::Predicate(Predicate {
                field: pref("p", Some("age")),
                op: ComparisonOp::Gt,
                value: Literal::Int(30),
            })),
            returns: vec![ReturnClause::Field {
                field: pref("p", Some("name")),
                alias: Some("name".into()),
            }],
            group_by: vec![],
            sort: vec![],
            limit: Some(10),
        };
        let q = build_read(&q).unwrap();
        assert!(q.text.starts_with("MATCH (p:Person)"));
        assert!(q.text.contains("WHERE p.age > $p0"));
        assert!(q.text.contains("RETURN p.name AS name"));
        assert!(q.text.trim_end().ends_with("LIMIT 10"));
        assert_eq!(q.params.get("p0"), Some(&Literal::Int(30)));
    }

    #[test]
    fn renders_traversal_with_depth_and_direction() {
        let q = ReadQuery {
            action: Action::Find,
            start: Node { label: "Person".into(), alias: alias("p") },
            traversals: vec![EdgeTraversal {
                from_alias: alias("p"),
                edge_label: "KNOWS".into(),
                edge_alias: alias("r"),
                direction: Direction::Out,
                target: Node { label: "Person".into(), alias: alias("p2") },
                depth: Some(Depth { min: 1, max: 3 }),
            }],
            filter: None,
            returns: vec![ReturnClause::Field {
                field: pref("p2", Some("name")),
                alias: None,
            }],
            group_by: vec![],
            sort: vec![],
            limit: None,
        };
        let out = build_read(&q).unwrap().text;
        assert!(out.contains("(p:Person)-[r:KNOWS*1..3]->(p2:Person)"), "got: {out}");
    }

    #[test]
    fn builds_aggregate_with_order_by_alias() {
        let q = ReadQuery {
            action: Action::Aggregate,
            start: Node { label: "Customer".into(), alias: alias("c") },
            traversals: vec![EdgeTraversal {
                from_alias: alias("c"),
                edge_label: "PLACED".into(),
                edge_alias: alias("po"),
                direction: Direction::Out,
                target: Node { label: "Order".into(), alias: alias("o") },
                depth: None,
            }],
            filter: None,
            returns: vec![
                ReturnClause::Field {
                    field: pref("c", Some("name")),
                    alias: Some("customer".into()),
                },
                ReturnClause::Aggregate {
                    func: AggregateFn::Sum,
                    field: pref("o", Some("total")),
                    alias: Some("total_spent".into()),
                },
            ],
            group_by: vec![pref("c", Some("name"))],
            sort: vec![SortKey {
                key: SortRef::Projected("total_spent".into()),
                order: SortOrder::Desc,
            }],
            limit: Some(5),
        };
        let out = build_read(&q).unwrap().text;
        assert!(out.contains("RETURN c.name AS customer, sum(o.total) AS total_spent"));
        assert!(out.contains("ORDER BY total_spent DESC"));
    }

    #[test]
    fn parameter_indices_increment() {
        let q = ReadQuery {
            action: Action::Find,
            start: Node { label: "Person".into(), alias: alias("p") },
            traversals: vec![],
            filter: Some(FilterExpression::And(vec![
                FilterExpression::Predicate(Predicate {
                    field: pref("p", Some("age")),
                    op: ComparisonOp::Gt,
                    value: Literal::Int(18),
                }),
                FilterExpression::Predicate(Predicate {
                    field: pref("p", Some("city")),
                    op: ComparisonOp::Eq,
                    value: Literal::String("Berlin".into()),
                }),
            ])),
            returns: vec![ReturnClause::Field {
                field: pref("p", Some("name")),
                alias: None,
            }],
            group_by: vec![],
            sort: vec![],
            limit: None,
        };
        let q = build_read(&q).unwrap();
        assert!(q.text.contains("p.age > $p0"));
        assert!(q.text.contains("p.city = $p1"));
        assert_eq!(q.params.len(), 2);
    }
}
