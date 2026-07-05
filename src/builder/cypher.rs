//! Public entrypoint for the Cypher builder.

use std::collections::BTreeMap;

use thiserror::Error;

use crate::ast::query::*;
use crate::db::result::{Column, NodeType};
use crate::graph::{MENTION_REL, PART_OF_REL, SOURCE_LABEL};
use crate::types::{TypeError, TypeRegistry};

use super::cursor::{Cursor, CypherQuery};
use super::insert::InsertError;
use super::{match_part, return_part, where_part};

/// Column name used for the always-on Sources projection added by
/// [`build_read_with`]. Consumers can look this up in
/// [`crate::db::result::QueryResult`] to render source context.
pub const SOURCES_COLUMN: &str = "sources";
pub const SCORE_COLUMN: &str = "score";

#[derive(Debug, Error)]
pub enum BuilderError {
    #[error("query has no projection (RETURN list is empty)")]
    EmptyReturn,

    #[error("insert builder error: {0}")]
    Insert(#[from] InsertError),

    #[error("type system error: {0}")]
    Type(#[from] TypeError),
}

/// Compile a [`ReadQuery`] into a single parameterized [`CypherQuery`]
/// without any registered type handlers. Suitable for queries that are
/// known to be untyped.
pub fn build_read(query: &ReadQuery) -> Result<CypherQuery, BuilderError> {
    build_read_with(query, &TypeRegistry::empty())
}

/// Compile a [`ReadQuery`] using `registry` to resolve typed
/// predicates. The registry is consulted exactly once per
/// [`FilterExpression::Typed`] node.
pub fn build_read_with(
    query: &ReadQuery,
    registry: &TypeRegistry,
) -> Result<CypherQuery, BuilderError> {
    if query.returns.is_empty() {
        return Err(BuilderError::EmptyReturn);
    }

    let mut cur = Cursor::new();

    // ── Phase 1: MATCH + WHERE (collects type contributions). ─────────
    match_part::write_match(&mut cur, query);
    if let Some(filter) = &query.filter {
        where_part::write_where(&mut cur, filter, registry)?;
    }
    match_part::write_optional_matches(&mut cur, query);

    // ── Phase 2: post-match handler fragments. Spliced after WHERE
    //    so they can reference the matched aliases (e.g. CASE WHEN
    //    n.name = $q THEN 1.0 ELSE 0.0 ; CALL qlink.score_batch_node).
    if !cur.post_match.is_empty() {
        let frags = cur.post_match.drain(..).collect::<Vec<_>>();
        cur.buf.push('\n');
        cur.buf.push_str(&frags.join("\n"));
    }

    // ── Phase 2.5: gather the per-entity Source projection. ──────────
    //
    // Every Find query returns a `sources` column listing the
    // built-in `:Source` nodes reachable via `:mention` (user
    // entities) or `:part_of` (Chunks) from any of the matched node
    // aliases. Aggregate queries collapse rows into summary statistics
    // — `sources` is a per-row list there has no well-defined
    // aggregation, so we deliberately skip the projection for them.
    let inject_sources = matches!(query.action, Action::Find) && !is_sources_aliased_already(query);
    if inject_sources {
        write_sources_stage(&mut cur, query);
    }

    // ── Phase 3: RETURN. ──────────────────────────────────────────────
    return_part::write_return(&mut cur, query);
    if matches!(query.action, Action::Find) && !is_score_aliased_already(query) {
        if let Some(score_expr) = score_projection_expr(&cur) {
            cur.buf.push_str(", ");
            cur.buf.push_str(&score_expr);
            cur.buf.push_str(" AS ");
            cur.buf.push_str(SCORE_COLUMN);
        }
    }
    if inject_sources {
        cur.buf.push_str(", ");
        cur.buf.push_str("__sources__ AS ");
        cur.buf.push_str(SOURCES_COLUMN);
    }

    // ── Phase 4: ORDER BY (user's keys first, then handler extras). ──
    //
    // Handler-contributed ORDER BY keys (e.g. SemanticText's
    // `<alias>__score`) are dropped for aggregate queries: those
    // collapse rows via `count`/`sum`/etc., and the score column is
    // neither aggregated nor part of `group_by`, so referencing it
    // here is illegal Cypher. The vector candidate set is already
    // pruned by `libqlink.search`'s `top_k` and threshold, so the
    // ordering is implicit anyway.
    if matches!(query.action, Action::Aggregate) {
        cur.extra_order_by.clear();
    }
    return_part::write_order_by_with_extra(&mut cur, &query.sort);

    // ── Phase 5: LIMIT. ───────────────────────────────────────────────
    if let Some(limit) = query.limit {
        return_part::write_limit(&mut cur, limit);
    }

    // ── Phase 6: pre-match. Spliced at the very top so it runs
    //    before MATCH (e.g. `CALL qlink.search(...) YIELD id, score`).
    if !cur.pre_match.is_empty() {
        let mut pre = cur.pre_match.drain(..).collect::<Vec<_>>().join("\n");
        pre.push('\n');
        cur.buf = format!("{pre}{}", cur.buf);
    }

    let columns = projected_columns(query, inject_sources, has_score_projection(&cur));
    Ok(cur.finish().with_columns(columns))
}

/// Column name carrying the list of node maps emitted by
/// [`build_read_graph`].
pub const GRAPH_NODES_COLUMN: &str = "nodes";
/// Column name carrying the list of edge maps emitted by
/// [`build_read_graph`].
pub const GRAPH_EDGES_COLUMN: &str = "edges";

/// Compile a [`ReadQuery`] into a Cypher query whose projection is
/// **graph-shaped** rather than tabular: it returns two columns,
/// [`GRAPH_NODES_COLUMN`] and [`GRAPH_EDGES_COLUMN`], each a list of
/// maps describing the bound nodes and the relationships between them.
///
/// This exists because the ordinary [`build_read_with`] projection is a
/// list of *scalars* (`RETURN p.name AS name`), from which node identity
/// and relationship endpoints cannot be recovered. Rendering the result
/// as an entity/relationship graph (the shape a `{nodes, edges}` API or
/// UI consumes) needs the identity, labels, and endpoint ids that only a
/// dedicated projection preserves.
///
/// Design notes:
/// * The MATCH / WHERE / OPTIONAL MATCH / post-match stages are reused
///   verbatim from the tabular builder, so prefix scoping, typed-filter
///   handler contributions, and traversal shape all behave identically.
/// * Every node alias survives to the RETURN (the tabular builder relies
///   on the same guarantee for its `sources` stage), so each is projected
///   as a map `{alias, id, labels, props, sources}`. `props`/`sources`
///   use Cypher map projection / pattern comprehension, which the strict
///   row decoder renders straight into JSON — no `src/db` change needed.
/// * Edges are derived from each traversal via a pattern comprehension
///   over the two already-bound node aliases, so edge identity does not
///   depend on the edge variable staying in scope. Variable-length
///   traversals bind a list (not a single relationship), so their per-edge
///   identity/props are skipped; the endpoint connectivity is still
///   emitted.
pub fn build_read_graph(
    query: &ReadQuery,
    registry: &TypeRegistry,
) -> Result<CypherQuery, BuilderError> {
    let mut cur = Cursor::new();

    // ── Phase 1: MATCH + WHERE + OPTIONAL MATCH (identical to tabular). ─
    match_part::write_match(&mut cur, query);
    if let Some(filter) = &query.filter {
        where_part::write_where(&mut cur, filter, registry)?;
    }
    match_part::write_optional_matches(&mut cur, query);

    // ── Phase 2: post-match handler fragments. ─────────────────────────
    if !cur.post_match.is_empty() {
        let frags = cur.post_match.drain(..).collect::<Vec<_>>();
        cur.buf.push('\n');
        cur.buf.push_str(&frags.join("\n"));
    }

    // ── Phase 3: graph-shaped RETURN. ─────────────────────────────────
    let node_maps: Vec<String> = collect_node_aliases(query)
        .into_iter()
        .map(|a| graph_node_map(&a))
        .collect();
    let edge_lists: Vec<String> = query
        .traversals
        .iter()
        .filter_map(graph_edge_list)
        .collect();
    let edges_expr = if edge_lists.is_empty() {
        "[]".to_string()
    } else {
        edge_lists.join(" + ")
    };
    cur.buf.push_str(&format!(
        "\nRETURN [{}] AS {GRAPH_NODES_COLUMN}, {edges_expr} AS {GRAPH_EDGES_COLUMN}",
        node_maps.join(", ")
    ));

    // ── Phase 4: LIMIT (ORDER BY is meaningless for the aggregated
    //    single-row graph projection, so it is skipped). ───────────────
    if let Some(limit) = query.limit {
        return_part::write_limit(&mut cur, limit);
    }

    // ── Phase 5: pre-match (e.g. a vector-search CALL). ────────────────
    if !cur.pre_match.is_empty() {
        let mut pre = cur.pre_match.drain(..).collect::<Vec<_>>().join("\n");
        pre.push('\n');
        cur.buf = format!("{pre}{}", cur.buf);
    }

    let columns = vec![
        Column::new(GRAPH_NODES_COLUMN),
        Column::new(GRAPH_EDGES_COLUMN),
    ];
    Ok(cur.finish().with_columns(columns))
}

/// Render the per-node map projection used by [`build_read_graph`]:
/// `{alias, id, labels, props, sources}`. `sources` walks `:mention` /
/// `:part_of` to the built-in `:Source` nodes, mirroring the tabular
/// builder's sources stage but scoped to this one node.
fn graph_node_map(alias: &str) -> String {
    format!(
        "{{alias: '{alias}', id: id({alias}), labels: labels({alias}), \
         props: {alias} {{.*}}, \
         sources: [({alias})-[:{MENTION_REL}|{PART_OF_REL}]->(__s:{SOURCE_LABEL}) | __s {{.*}}]}}"
    )
}

/// Render the per-traversal edge list comprehension used by
/// [`build_read_graph`]. Returns `None` for variable-length traversals,
/// whose edge variable binds a list rather than a single relationship.
fn graph_edge_list(t: &EdgeTraversal) -> Option<String> {
    let is_var_length = matches!(t.depth, Some(d) if !(d.min == 1 && d.max == 1));
    if is_var_length {
        return None;
    }
    let from = t.from_alias.as_str();
    let to = t.target.alias.as_str();
    let edge = &t.edge_label;
    let pattern = match t.direction {
        Direction::Out => format!("({from})-[__e:{edge}]->({to})"),
        Direction::In => format!("({from})<-[__e:{edge}]-({to})"),
        Direction::Both => format!("({from})-[__e:{edge}]-({to})"),
    };
    // `startNode`/`endNode` recover the stored direction so `from`/`to`
    // reflect the actual relationship regardless of the DSL's traversal
    // orientation.
    Some(format!(
        "[{pattern} | {{id: id(__e), rel: type(__e), \
         from: id(startNode(__e)), to: id(endNode(__e)), props: __e {{.*}}}}]"
    ))
}

/// True when the post-match handler chain emitted a sort key that
/// [`build_read_with`] would surface as an aggregated `score` column.
/// Mirrors the condition that drives the `__score__ AS score`
/// projection above so the column metadata stays in sync.
fn has_score_projection(cur: &Cursor) -> bool {
    score_projection_expr(cur).is_some()
}

/// Compute the typed [`Column`] list for a [`ReadQuery`]'s projection,
/// matching the order in which the builder emits the RETURN list. Each
/// `ReturnClause::Field` column is tagged with the [`NodeType`] of the
/// underlying alias when that alias is bound by the MATCH (so `c.id`
/// where `c` is `:Chunk` yields a column named `id` of type Chunk).
/// Aggregates and the synthesised `score` / `sources` columns are left
/// untyped.
fn projected_columns(query: &ReadQuery, inject_sources: bool, inject_score: bool) -> Vec<Column> {
    let mut alias_to_label: BTreeMap<&str, &str> = BTreeMap::new();
    alias_to_label.insert(query.start.alias.as_str(), query.start.label.as_str());
    for t in &query.traversals {
        alias_to_label.insert(t.target.alias.as_str(), t.target.label.as_str());
    }

    let mut cols = Vec::with_capacity(query.returns.len() + 2);
    for clause in &query.returns {
        match clause {
            ReturnClause::Field { field, alias } => {
                let name = alias.clone().unwrap_or_else(|| render_field_name(field));
                let node_type = alias_to_label
                    .get(field.alias.as_str())
                    .map(|label| NodeType::from_label(label));
                cols.push(Column { name, node_type });
            }
            ReturnClause::GroupKey { key, alias } => {
                let node_type = alias_to_label
                    .get(key.field.alias.as_str())
                    .map(|label| NodeType::from_label(label));
                cols.push(Column {
                    name: alias.clone(),
                    node_type,
                });
            }
            ReturnClause::Aggregate { func, field, alias } => {
                let name = alias
                    .clone()
                    .unwrap_or_else(|| render_aggregate_name(func, field));
                cols.push(Column::new(name));
            }
        }
    }
    if matches!(query.action, Action::Find) {
        if inject_score && !is_score_aliased_already(query) {
            cols.push(Column::new(SCORE_COLUMN));
        }
        if inject_sources {
            cols.push(Column::new(SOURCES_COLUMN));
        }
    }
    cols
}

fn render_field_name(p: &PropertyRef) -> String {
    match &p.property {
        Some(prop) => format!("{}.{}", p.alias, prop),
        None => p.alias.to_string(),
    }
}

fn render_aggregate_name(func: &AggregateFn, field: &PropertyRef) -> String {
    let inner = render_field_name(field);
    match func {
        AggregateFn::Count => {
            let v = inner.split('.').next().unwrap_or(&inner);
            format!("count({v})")
        }
        AggregateFn::Sum => format!("sum({inner})"),
        AggregateFn::Avg => format!("avg({inner})"),
        AggregateFn::Min => format!("min({inner})"),
        AggregateFn::Max => format!("max({inner})"),
    }
}

/// Emit the WITH / OPTIONAL MATCH stage that gathers the unique set
/// of `:Source` nodes reachable from any of the query's matched node
/// aliases, exposing them as `__sources__`.
///
/// The stage walks both `:mention` (any user entity) and `:part_of`
/// (chunks) edges and de-duplicates with `collect(DISTINCT ...)`.
/// Edges in the user's traversals are intentionally excluded — we
/// carry node aliases only because edge variables can't be sources.
fn write_sources_stage(cur: &mut Cursor, query: &ReadQuery) {
    let aliases = collect_node_aliases(query);
    if aliases.is_empty() {
        return;
    }
    let source_alias_carry = aliases.join(", ");
    let carry = carry_names_for_sources(query, cur);
    let carry = carry.join(", ");
    let list = format!("[{source_alias_carry}]");
    cur.buf.push_str(&format!(
        "\nWITH {carry}\n\
         OPTIONAL MATCH (__src__:{SOURCE_LABEL})<-[:{MENTION_REL}|{PART_OF_REL}]-(__sn__)\n\
         WHERE __sn__ IN {list}\n\
         WITH {carry}, collect(DISTINCT __src__) AS __sources__"
    ));
}

fn carry_names_for_sources(query: &ReadQuery, cur: &Cursor) -> Vec<String> {
    let mut carry = collect_node_aliases(query);
    let mut seen: std::collections::BTreeSet<String> = carry.iter().cloned().collect();
    for (key, _) in &cur.extra_order_by {
        if is_plain_cypher_ident(key) && seen.insert(key.clone()) {
            carry.push(key.clone());
        }
    }
    carry
}

/// Names of every node alias bound by the query's MATCH clauses.
/// Edge aliases are intentionally excluded.
fn collect_node_aliases(query: &ReadQuery) -> Vec<String> {
    let mut out = Vec::with_capacity(1 + query.traversals.len());
    let mut seen: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    let push =
        |alias: &Alias, out: &mut Vec<String>, seen: &mut std::collections::BTreeSet<String>| {
            let s = alias.as_str().to_string();
            if seen.insert(s.clone()) {
                out.push(s);
            }
        };
    push(&query.start.alias, &mut out, &mut seen);
    for t in &query.traversals {
        push(&t.target.alias, &mut out, &mut seen);
    }
    out
}

/// True when the caller already projects a `sources` column. Used as
/// a defensive guard so the auto-projection doesn't collide with a
/// user-supplied one.
fn is_sources_aliased_already(query: &ReadQuery) -> bool {
    query.returns.iter().any(|clause| match clause {
        ReturnClause::Field { alias, .. } | ReturnClause::Aggregate { alias, .. } => {
            alias.as_deref() == Some(SOURCES_COLUMN)
        }
        ReturnClause::GroupKey { alias, .. } => alias == SOURCES_COLUMN,
    })
}

fn is_score_aliased_already(query: &ReadQuery) -> bool {
    query.returns.iter().any(|clause| match clause {
        ReturnClause::Field { alias, .. } | ReturnClause::Aggregate { alias, .. } => {
            alias.as_deref() == Some(SCORE_COLUMN)
        }
        ReturnClause::GroupKey { alias, .. } => alias == SCORE_COLUMN,
    })
}

fn score_projection_expr(cur: &Cursor) -> Option<String> {
    let mut scores = Vec::new();
    let mut seen = std::collections::BTreeSet::new();
    for (key, _) in &cur.extra_order_by {
        if is_plain_cypher_ident(key) && seen.insert(key.clone()) {
            scores.push(key.clone());
        }
    }
    match scores.len() {
        0 => None,
        1 => scores.into_iter().next(),
        _ => Some(format!("({})", scores.join(" + "))),
    }
}

fn is_plain_cypher_ident(s: &str) -> bool {
    let mut chars = s.chars();
    let Some(first) = chars.next() else {
        return false;
    };
    (first == '_' || first.is_ascii_alphabetic())
        && chars.all(|c| c == '_' || c.is_ascii_alphanumeric())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn alias(s: &str) -> Alias {
        Alias::new(s)
    }

    fn pref(a: &str, p: Option<&str>) -> PropertyRef {
        PropertyRef {
            alias: alias(a),
            property: p.map(str::to_string),
        }
    }

    fn gkey(a: &str, p: Option<&str>) -> GroupByKey {
        GroupByKey {
            field: pref(a, p),
            transform: None,
            alias: None,
        }
    }

    #[test]
    fn builds_basic_match_with_filter() {
        let q = ReadQuery {
            action: Action::Find,
            start: Node {
                label: "Person".into(),
                alias: alias("p"),
                prefix_label: None,
            },
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
            start: Node {
                label: "Person".into(),
                alias: alias("p"),
                prefix_label: None,
            },
            traversals: vec![EdgeTraversal {
                from_alias: alias("p"),
                edge_label: "KNOWS".into(),
                edge_alias: alias("r"),
                direction: Direction::Out,
                target: Node {
                    label: "Person".into(),
                    alias: alias("p2"),
                    prefix_label: None,
                },
                depth: Some(Depth { min: 1, max: 3 }),
                optional: false,
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
        assert!(
            out.contains("(p:Person)-[r:KNOWS*1..3]->(p2:Person)"),
            "got: {out}"
        );
    }

    #[test]
    fn renders_optional_traversal_as_optional_match() {
        let q = ReadQuery {
            action: Action::Find,
            start: Node {
                label: "Person".into(),
                alias: alias("p"),
                prefix_label: None,
            },
            traversals: vec![EdgeTraversal {
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
            }],
            filter: Some(FilterExpression::Predicate(Predicate {
                field: pref("p", Some("active")),
                op: ComparisonOp::Eq,
                value: Literal::Bool(true),
            })),
            returns: vec![ReturnClause::Field {
                field: pref("c", Some("name")),
                alias: None,
            }],
            group_by: vec![],
            sort: vec![],
            limit: None,
        };

        let out = build_read(&q).unwrap().text;
        assert!(
            out.contains(
                "MATCH (p:Person)\nWHERE p.active = $p0\nOPTIONAL MATCH (p)-[w:WORKS_AT]->(c:Company)"
            ),
            "got: {out}"
        );
    }

    #[test]
    fn builds_aggregate_with_order_by_alias() {
        let q = ReadQuery {
            action: Action::Aggregate,
            start: Node {
                label: "Customer".into(),
                alias: alias("c"),
                prefix_label: None,
            },
            traversals: vec![EdgeTraversal {
                from_alias: alias("c"),
                edge_label: "PLACED".into(),
                edge_alias: alias("po"),
                direction: Direction::Out,
                target: Node {
                    label: "Order".into(),
                    alias: alias("o"),
                    prefix_label: None,
                },
                depth: None,
                optional: false,
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
            group_by: vec![gkey("c", Some("name"))],
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
    fn find_queries_always_project_sources_column() {
        let q = ReadQuery {
            action: Action::Find,
            start: Node {
                label: "Person".into(),
                alias: alias("p"),
                prefix_label: None,
            },
            traversals: vec![],
            filter: None,
            returns: vec![ReturnClause::Field {
                field: pref("p", Some("name")),
                alias: Some("name".into()),
            }],
            group_by: vec![],
            sort: vec![],
            limit: None,
        };
        let q = build_read(&q).unwrap();

        assert!(
            q.text
                .contains("OPTIONAL MATCH (__src__:Source)<-[:mention|part_of]-(__sn__)"),
            "expected source-gathering OPTIONAL MATCH, got:\n{}",
            q.text
        );
        assert!(
            q.text.contains("collect(DISTINCT __src__) AS __sources__"),
            "expected source de-duplication via collect(DISTINCT), got:\n{}",
            q.text
        );
        assert!(
            q.text.contains("__sources__ AS sources"),
            "expected `sources` to appear in the projection, got:\n{}",
            q.text
        );
    }

    #[test]
    fn aggregate_queries_skip_sources_projection() {
        let q = ReadQuery {
            action: Action::Aggregate,
            start: Node {
                label: "Order".into(),
                alias: alias("o"),
                prefix_label: None,
            },
            traversals: vec![],
            filter: None,
            returns: vec![ReturnClause::Aggregate {
                func: AggregateFn::Count,
                field: pref("o", None),
                alias: Some("n".into()),
            }],
            group_by: vec![],
            sort: vec![],
            limit: None,
        };
        let q = build_read(&q).unwrap();

        assert!(
            !q.text.contains("__sources__"),
            "aggregate queries must not gather sources (per-row list with no \
             well-defined aggregation); got:\n{}",
            q.text
        );
    }

    #[test]
    fn columns_carry_node_type_from_binding_alias() {
        let q = ReadQuery {
            action: Action::Find,
            start: Node {
                label: "Chunk".into(),
                alias: alias("c"),
                prefix_label: None,
            },
            traversals: vec![EdgeTraversal {
                from_alias: alias("c"),
                edge_label: "mention".into(),
                edge_alias: alias("m"),
                direction: Direction::Out,
                target: Node {
                    label: "Person".into(),
                    alias: alias("p"),
                    prefix_label: None,
                },
                depth: None,
                optional: false,
            }],
            filter: None,
            returns: vec![
                ReturnClause::Field {
                    field: pref("c", Some("id")),
                    alias: Some("id".into()),
                },
                ReturnClause::Field {
                    field: pref("p", Some("name")),
                    alias: None,
                },
                ReturnClause::Aggregate {
                    func: AggregateFn::Count,
                    field: pref("p", None),
                    alias: Some("n".into()),
                },
            ],
            group_by: vec![],
            sort: vec![],
            limit: None,
        };
        let q = build_read(&q).unwrap();

        // `c.id` projects from a :Chunk binding → Chunk
        // `p.name` projects from a :Person binding → Entity (default)
        // count(p) is an aggregate → no node type
        // synthetic `sources` column auto-projected for Find queries → no type
        let found: std::collections::BTreeMap<_, _> = q
            .columns
            .iter()
            .map(|c| (c.name.clone(), c.node_type))
            .collect();
        assert_eq!(found.get("id"), Some(&Some(NodeType::Chunk)));
        assert_eq!(found.get("p.name"), Some(&Some(NodeType::Entity)));
        assert_eq!(found.get("n"), Some(&None));
        assert_eq!(found.get("sources"), Some(&None));
    }

    #[test]
    fn source_label_maps_to_source_node_type() {
        let q = ReadQuery {
            action: Action::Find,
            start: Node {
                label: "Source".into(),
                alias: alias("s"),
                prefix_label: None,
            },
            traversals: vec![],
            filter: None,
            returns: vec![ReturnClause::Field {
                field: pref("s", Some("name")),
                alias: Some("source_name".into()),
            }],
            group_by: vec![],
            sort: vec![],
            limit: None,
        };
        let q = build_read(&q).unwrap();
        let col = q
            .columns
            .iter()
            .find(|c| c.name == "source_name")
            .expect("source_name column exists");
        assert_eq!(col.node_type, Some(NodeType::Source));
    }

    #[test]
    fn graph_projection_emits_node_maps_and_edge_comprehension() {
        let q = ReadQuery {
            action: Action::Find,
            start: Node {
                label: "Person".into(),
                alias: alias("p"),
                prefix_label: None,
            },
            traversals: vec![EdgeTraversal {
                from_alias: alias("p"),
                edge_label: "OWNS".into(),
                edge_alias: alias("r"),
                direction: Direction::Out,
                target: Node {
                    label: "Company".into(),
                    alias: alias("c"),
                    prefix_label: None,
                },
                depth: None,
                optional: false,
            }],
            filter: None,
            returns: vec![ReturnClause::Field {
                field: pref("p", Some("name")),
                alias: None,
            }],
            group_by: vec![],
            sort: vec![],
            limit: Some(25),
        };
        let out = build_read_graph(&q, &TypeRegistry::empty()).unwrap();

        // MATCH is reused verbatim from the tabular builder.
        assert!(out.text.starts_with("MATCH (p:Person)"), "got: {}", out.text);
        assert!(out.text.contains("-[r:OWNS]->(c:Company)"), "got: {}", out.text);
        // Both bound node aliases are projected as identity maps.
        assert!(out.text.contains("id: id(p)"), "got: {}", out.text);
        assert!(out.text.contains("props: p {.*}"), "got: {}", out.text);
        assert!(out.text.contains("id: id(c)"), "got: {}", out.text);
        // Sources are gathered per node via pattern comprehension.
        assert!(
            out.text.contains("[(p)-[:mention|part_of]->(__s:Source) | __s {.*}]"),
            "got: {}",
            out.text
        );
        // The edge is derived from the traversal, with true direction.
        assert!(
            out.text.contains("[(p)-[__e:OWNS]->(c) | {id: id(__e), rel: type(__e)"),
            "got: {}",
            out.text
        );
        assert!(out.text.contains("from: id(startNode(__e))"), "got: {}", out.text);
        assert!(out.text.trim_end().ends_with("LIMIT 25"), "got: {}", out.text);

        let names: Vec<_> = out.columns.iter().map(|c| c.name.as_str()).collect();
        assert_eq!(names, vec!["nodes", "edges"]);
    }

    #[test]
    fn graph_projection_skips_variable_length_edges() {
        let q = ReadQuery {
            action: Action::Find,
            start: Node {
                label: "Person".into(),
                alias: alias("p"),
                prefix_label: None,
            },
            traversals: vec![EdgeTraversal {
                from_alias: alias("p"),
                edge_label: "KNOWS".into(),
                edge_alias: alias("r"),
                direction: Direction::Out,
                target: Node {
                    label: "Person".into(),
                    alias: alias("p2"),
                    prefix_label: None,
                },
                depth: Some(Depth { min: 1, max: 3 }),
                optional: false,
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
        let out = build_read_graph(&q, &TypeRegistry::empty()).unwrap();
        // Nodes still project; the var-length edge is not turned into a
        // per-edge comprehension (it would bind a list, not one edge).
        assert!(out.text.contains("id: id(p2)"), "got: {}", out.text);
        assert!(out.text.contains("[] AS edges"), "got: {}", out.text);
    }

    #[test]
    fn parameter_indices_increment() {
        let q = ReadQuery {
            action: Action::Find,
            start: Node {
                label: "Person".into(),
                alias: alias("p"),
                prefix_label: None,
            },
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
