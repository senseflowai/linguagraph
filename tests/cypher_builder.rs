//! End-to-end DSL → Cypher tests using only the public API.

use linguagraph::ast::from_dsl;
use linguagraph::ast::query::Literal;
use linguagraph::builder;
use linguagraph::dsl;

fn compile(json: &str) -> linguagraph::builder::CypherQuery {
    let dsl = dsl::parse_str(json).expect("dsl parses");
    let ast = from_dsl::lower(dsl, /* max_depth */ 6).expect("lowers");
    builder::build(&ast).expect("builds")
}

#[test]
fn find_example_round_trip() {
    let cypher = compile(include_str!("../examples/find_people.json"));
    assert!(cypher.text.contains("MATCH (p:Person)"));
    assert!(cypher.text.contains("[r:KNOWS*1..2]"));
    assert!(cypher.text.contains("WHERE"));
    assert!(cypher.text.contains("ORDER BY name ASC"));
    assert!(cypher.text.trim_end().ends_with("LIMIT 25"));
    // Two filter values bound as parameters.
    assert_eq!(cypher.params.len(), 2);
    assert!(cypher
        .params
        .values()
        .any(|v| matches!(v, Literal::Int(30))));
    assert!(cypher
        .params
        .values()
        .any(|v| matches!(v, Literal::String(s) if s == "Berlin")));
}

#[test]
fn aggregate_example_round_trip() {
    let cypher = compile(include_str!("../examples/aggregate_orders.json"));
    assert!(cypher.text.contains("count(o) AS order_count"));
    assert!(cypher.text.contains("sum(o.total) AS total_spent"));
    assert!(cypher.text.contains("ORDER BY total_spent DESC"));
}

#[test]
fn aggregate_projects_group_by_key_for_order_by() {
    // The model grouped and sorted by `sv.work_start` but listed the
    // field only in `group_by`/`sort`, not `return`. Cypher derives the
    // grouping keys from the RETURN projection, so the key must be
    // projected with an alias — and `ORDER BY` must target that alias.
    // A bare `ORDER BY sv.work_start` references `sv`, which is out of
    // scope after the aggregating RETURN ("Unbound variable: sv").
    let cypher = compile(
        r#"{
            "action": "aggregate",
            "start": { "label": "Client", "alias": "c" },
            "traversals": [
                { "from": "c",
                  "edge": { "label": "VISITED", "alias": "sv_rel", "direction": "out" },
                  "target": { "label": "ServiceVisit", "alias": "sv" } }
            ],
            "filters": [
                { "field": "sv.work_start", "op": "gte", "value": "2026-05-10T00:00:00" },
                { "field": "sv.work_start", "op": "lte", "value": "2026-05-15T23:59:59" }
            ],
            "return": [
                { "aggregate": "count", "field": "c.id", "alias": "client_count" }
            ],
            "group_by": ["sv.work_start"],
            "sort": [{ "field": "sv.work_start", "order": "asc" }]
        }"#,
    );

    let return_line = cypher
        .text
        .lines()
        .find(|l| l.starts_with("RETURN "))
        .expect("query has a RETURN clause");
    // The aggregate is still projected ...
    assert!(
        return_line.contains("count(c) AS client_count"),
        "got: {return_line}"
    );
    // ... and the grouping key is projected with an alias.
    assert!(
        return_line.contains("sv.work_start AS sv_work_start"),
        "RETURN must project the group_by key with an alias: {return_line}"
    );
    // ORDER BY targets the projected alias, never the raw property.
    assert!(
        cypher.text.contains("ORDER BY sv_work_start ASC"),
        "got: {}",
        cypher.text
    );
    let order_line = cypher
        .text
        .lines()
        .find(|l| l.starts_with("ORDER BY "))
        .expect("query has an ORDER BY clause");
    assert!(
        !order_line.contains('.'),
        "ORDER BY must reference an alias, not a property expression: {order_line}"
    );
}

#[test]
fn aggregate_with_multiple_aggregates_sorts_by_group_key_alias() {
    // Reported case: two aggregates, plus group_by/sort on a field that
    // is not in `return`. The grouping key must be projected with an
    // alias and the ORDER BY rewritten to that alias.
    let cypher = compile(
        r#"{
            "action": "aggregate",
            "start": { "label": "ServiceVisit", "alias": "sv" },
            "traversals": [
                { "from": "sv",
                  "edge": { "label": "INCLUDES_WORK", "alias": "wi", "direction": "out" },
                  "target": { "label": "WorkItem", "alias": "w" } }
            ],
            "filters": [
                { "field": "sv.work_start", "op": "gte", "value": "2026-05-10T00:00:00" },
                { "field": "sv.work_start", "op": "lte", "value": "2026-05-15T23:59:59" }
            ],
            "return": [
                { "aggregate": "count", "field": "sv.id", "alias": "visits_count" },
                { "aggregate": "sum", "field": "w.cost", "alias": "total_revenue" }
            ],
            "group_by": ["sv.work_start"],
            "sort": [{ "field": "sv.work_start", "order": "asc" }]
        }"#,
    );

    let return_line = cypher
        .text
        .lines()
        .find(|l| l.starts_with("RETURN "))
        .expect("query has a RETURN clause");
    assert!(
        return_line.contains("count(sv) AS visits_count"),
        "got: {return_line}"
    );
    assert!(
        return_line.contains("sum(w.cost) AS total_revenue"),
        "got: {return_line}"
    );
    assert!(
        return_line.contains("sv.work_start AS sv_work_start"),
        "RETURN must project the group_by key with an alias: {return_line}"
    );
    assert!(
        cypher.text.contains("ORDER BY sv_work_start ASC"),
        "ORDER BY must target the projected alias: {}",
        cypher.text
    );
}

#[test]
fn aggregate_can_group_datetime_by_year() {
    let cypher = compile(
        r#"{
            "action": "aggregate",
            "start": { "label": "Client", "alias": "c" },
            "traversals": [],
            "filters": [
                { "field": "c.created_at", "op": "gte", "value": "2023-01-01" },
                { "field": "c.created_at", "op": "lte", "value": "2027-12-31" }
            ],
            "return": [
                { "aggregate": "count", "field": "c.id", "alias": "client_count" }
            ],
            "group_by": [
                { "field": "c.created_at", "date_part": "year", "alias": "created_year" }
            ],
            "sort": [
                { "field": "created_year", "order": "asc" }
            ],
            "limit": 100
        }"#,
    );

    assert!(
        cypher
            .text
            .contains("RETURN count(c) AS client_count, c.created_at.year AS created_year"),
        "got: {}",
        cypher.text
    );
    assert!(
        cypher.text.contains("ORDER BY created_year ASC"),
        "got: {}",
        cypher.text
    );
}

#[test]
fn infers_aggregate_when_find_action_contains_aggregation() {
    let json = r#"{
        "action": "find",
        "start": { "label": "Person", "alias": "p" },
        "return": [{ "aggregate": "count", "field": "p", "alias": "n" }]
    }"#;
    let dsl = dsl::parse_str(json).unwrap();
    let query = from_dsl::lower(dsl, 6).expect("aggregate projection should determine action");
    assert_eq!(query.action, linguagraph::ast::query::Action::Aggregate);
}

#[test]
fn infers_aggregate_when_action_is_omitted() {
    let json = r#"{
        "start": { "label": "Person", "alias": "p" },
        "return": [{ "aggregate": "count", "field": "p", "alias": "n" }]
    }"#;
    let dsl = dsl::parse_str(json).unwrap();
    let query =
        from_dsl::lower(dsl, 6).expect("action should be optional for aggregate projections");
    assert_eq!(query.action, linguagraph::ast::query::Action::Aggregate);
}

#[test]
fn rejects_excessive_depth() {
    let json = r#"{
        "action": "find",
        "start": { "label": "Person", "alias": "p" },
        "traversals": [{
            "edge": { "label": "KNOWS", "alias": "r", "direction": "out" },
            "target": { "label": "Person", "alias": "p2" },
            "depth": { "min": 1, "max": 99 }
        }],
        "return": [{ "field": "p.name" }]
    }"#;
    let dsl = dsl::parse_str(json).unwrap();
    let err = from_dsl::lower(dsl, 6).unwrap_err();
    assert!(matches!(
        err,
        linguagraph::ast::AstError::DepthTooLarge { .. }
    ));
}

#[test]
fn rejects_unknown_alias_in_filter() {
    let json = r#"{
        "action": "find",
        "start": { "label": "Person", "alias": "p" },
        "filters": [{ "field": "ghost.age", "op": "gt", "value": 30 }],
        "return": [{ "field": "p.name" }]
    }"#;
    let dsl = dsl::parse_str(json).unwrap();
    let err = from_dsl::lower(dsl, 6).unwrap_err();
    assert!(matches!(err, linguagraph::ast::AstError::UnknownAlias(_)));
}

#[test]
fn parameters_never_appear_inline() {
    // Defense-in-depth: the value side of every predicate must be a `$pN`
    // placeholder, never the literal value rendered into the string.
    let cypher = compile(
        r#"{
            "action": "find",
            "start": { "label": "Person", "alias": "p" },
            "filters": [{ "field": "p.name", "op": "eq", "value": "Robert'); DROP TABLE Students;--" }],
            "return": [{ "field": "p.name" }]
        }"#,
    );
    assert!(!cypher.text.contains("DROP"));
    assert!(cypher.text.contains("p.name = $p0"));
}

#[test]
fn sibling_traversals_emit_separate_match_clauses() {
    // Two unrelated children of the start node — Storage and Place — must
    // not be chained as `(c)-[..]->(st)-[..]->(p)`. They are independent
    // children, each branching off `c`.
    let cypher = compile(
        r#"{
            "action": "aggregate",
            "start": { "label": "Camera", "alias": "c" },
            "traversals": [
                {
                    "edge": { "label": "HAS_STORAGE", "alias": "s_rel", "direction": "out" },
                    "target": { "label": "Storage", "alias": "st" }
                },
                {
                    "edge": { "label": "LOCATED_IN", "alias": "loc_rel", "direction": "out" },
                    "target": { "label": "Place", "alias": "p" }
                }
            ],
            "filters": [
                { "field": "st.depth", "op": "eq", "value": 30 },
                { "field": "p.name", "op": "eq", "value": "Office" }
            ],
            "return": [
                { "aggregate": "count", "field": "c.id", "alias": "n" }
            ]
        }"#,
    );

    // First MATCH carries the start node and the first traversal.
    assert!(
        cypher
            .text
            .contains("MATCH (c:Camera)-[s_rel:HAS_STORAGE]->(st:Storage)"),
        "got: {}",
        cypher.text
    );
    // Second MATCH branches back from `c` (already bound — no label needed).
    assert!(
        cypher
            .text
            .contains("MATCH (c)-[loc_rel:LOCATED_IN]->(p:Place)"),
        "got: {}",
        cypher.text
    );
    // It must NOT have collapsed into one chained pattern.
    assert!(
        !cypher.text.contains("(st:Storage)-[loc_rel:LOCATED_IN]"),
        "sibling traversals must not chain: {}",
        cypher.text
    );
}

#[test]
fn optional_traversal_emits_optional_match_clause() {
    let cypher = compile(
        r#"{
            "action": "find",
            "start": { "label": "Person", "alias": "p" },
            "traversals": [
                {
                    "optional": true,
                    "edge": { "label": "WORKS_AT", "alias": "w", "direction": "out" },
                    "target": { "label": "Company", "alias": "c" }
                }
            ],
            "filters": [{ "field": "p.active", "op": "eq", "value": true }],
            "return": [{ "field": "c.name" }]
        }"#,
    );

    assert!(
        cypher.text.contains(
            "MATCH (p:Person)\nWHERE p.active = $p0\nOPTIONAL MATCH (p)-[w:WORKS_AT]->(c:Company)"
        ),
        "got: {}",
        cypher.text
    );
}

#[test]
fn explicit_from_chains_traversals() {
    // `from` lets the author keep the legacy chained behavior on demand:
    // (p)-[k1]->(f1)-[k2]->(f2).
    let cypher = compile(
        r#"{
            "action": "find",
            "start": { "label": "Person", "alias": "p" },
            "traversals": [
                {
                    "edge": { "label": "KNOWS", "alias": "k1", "direction": "out" },
                    "target": { "label": "Person", "alias": "f1" }
                },
                {
                    "from": "f1",
                    "edge": { "label": "KNOWS", "alias": "k2", "direction": "out" },
                    "target": { "label": "Person", "alias": "f2" }
                }
            ],
            "return": [{ "field": "f2.name" }]
        }"#,
    );
    assert!(
        cypher
            .text
            .contains("MATCH (p:Person)-[k1:KNOWS]->(f1:Person)-[k2:KNOWS]->(f2:Person)"),
        "got: {}",
        cypher.text
    );
    // Single user MATCH clause — chain not split. The compiler
    // additionally emits an `OPTIONAL MATCH` for the always-on
    // sources projection, so we count standalone MATCH lines only.
    let user_match_lines = cypher
        .text
        .lines()
        .filter(|line| line.starts_with("MATCH "))
        .count();
    assert_eq!(user_match_lines, 1);
}

#[test]
fn from_must_reference_a_bound_alias() {
    let json = r#"{
        "action": "find",
        "start": { "label": "Person", "alias": "p" },
        "traversals": [
            {
                "from": "ghost",
                "edge": { "label": "KNOWS", "alias": "r", "direction": "out" },
                "target": { "label": "Person", "alias": "p2" }
            }
        ],
        "return": [{ "field": "p2.name" }]
    }"#;
    let dsl = dsl::parse_str(json).unwrap();
    let err = from_dsl::lower(dsl, 6).unwrap_err();
    assert!(matches!(err, linguagraph::ast::AstError::UnknownAlias(_)));
}
