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
    assert!(cypher.params.values().any(|v| matches!(v, Literal::Int(30))));
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
fn rejects_aggregation_in_find() {
    let json = r#"{
        "action": "find",
        "start": { "label": "Person", "alias": "p" },
        "return": [{ "aggregate": "count", "field": "p", "alias": "n" }]
    }"#;
    let dsl = dsl::parse_str(json).unwrap();
    let err = from_dsl::lower(dsl, 6).unwrap_err();
    assert!(matches!(err, linguagraph::ast::AstError::AggregateInFind));
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
    // Single MATCH clause — chain not split.
    assert_eq!(cypher.text.matches("MATCH ").count(), 1);
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
