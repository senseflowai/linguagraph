//! Parse-stage integration tests. Hits the public `dsl::parse_str` API the
//! CLI uses, not internals.

use linguagraph::dsl::{self, Action, DslError};

#[test]
fn accepts_example_find() {
    let json = include_str!("../examples/find_people.json");
    let q = dsl::parse_str(json).expect("example must parse");
    assert_eq!(q.action, Action::Find);
    assert_eq!(q.start.alias, "p");
    assert_eq!(q.traversals.len(), 1);
}

#[test]
fn accepts_example_aggregate() {
    let json = include_str!("../examples/aggregate_orders.json");
    let q = dsl::parse_str(json).expect("aggregate example must parse");
    assert_eq!(q.action, Action::Aggregate);
    assert_eq!(q.group_by, vec!["c.name".to_string()]);
}

#[test]
fn accepts_missing_action() {
    let json = r#"{
        "start": { "label": "Person", "alias": "p" },
        "return": [{ "field": "p.name" }]
    }"#;
    let q = dsl::parse_str(json).expect("action should be optional");
    assert_eq!(q.action, Action::Find);
}

#[test]
fn rejects_unknown_top_level_field() {
    // serde_json by default ignores unknown fields, but we can still detect
    // syntactic problems via empty `return`, which the validator rejects.
    let json = r#"{
        "action": "find",
        "start": { "label": "Person", "alias": "p" },
        "return": []
    }"#;
    assert!(matches!(dsl::parse_str(json), Err(DslError::EmptyReturn)));
}

#[test]
fn rejects_zero_limit() {
    let json = r#"{
        "action": "find",
        "start": { "label": "Person", "alias": "p" },
        "return": [{ "field": "p.name" }],
        "limit": 0
    }"#;
    assert!(matches!(dsl::parse_str(json), Err(DslError::InvalidLimit)));
}

#[test]
fn rejects_invalid_field_reference() {
    let json = r#"{
        "action": "find",
        "start": { "label": "Person", "alias": "p" },
        "return": [{ "field": "p.name.extra" }]
    }"#;
    assert!(matches!(
        dsl::parse_str(json),
        Err(DslError::InvalidFieldRef(_))
    ));
}
