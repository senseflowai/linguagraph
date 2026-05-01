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
