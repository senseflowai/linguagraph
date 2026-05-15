//! End-to-end pipeline test using [`MockClient`] in place of Memgraph.

use std::collections::BTreeMap;
use std::sync::Arc;

use linguagraph::config::{
    Config, DatabaseConfig, GraphSpecificationConfig, LlmConfig, QueryConfig,
};
use linguagraph::core::Pipeline;
use linguagraph::db::{MockClient, QueryResult, Row, Value};
use linguagraph::dsl;
use linguagraph::graph::{GraphBuilder, PropertyType};

fn test_config() -> Config {
    Config {
        database: DatabaseConfig {
            uri: "bolt://test".into(),
            user: "u".into(),
            password: "p".into(),
            database: "memgraph".into(),
            max_connections: 1,
            query_timeout_secs: 5,
        },
        llm: LlmConfig::default(),
        query: QueryConfig {
            max_traversal_depth: 4,
            default_limit: 50,
        },
        graph_specification: GraphSpecificationConfig::default(),
        types: Default::default(),
    }
}

#[tokio::test]
async fn pipeline_compiles_and_dispatches_to_client() {
    let mock = Arc::new(MockClient::new());
    let mut row = Row::default();
    row.fields
        .insert("name".into(), Value::String("Ada".into()));
    mock.enqueue(QueryResult {
        columns: vec!["name".into()],
        rows: vec![row],
    });

    let cfg = test_config();
    let pipeline = Pipeline::new(mock.clone(), &cfg);

    let dsl = dsl::parse_str(include_str!("../examples/find_people.json")).unwrap();
    let result = pipeline.run(dsl).await.expect("pipeline runs");
    assert_eq!(result.rows.len(), 1);

    let captured = mock.captured.lock().unwrap();
    assert_eq!(captured.len(), 1);
    assert!(captured[0].text.contains("MATCH (p:Person)"));
    assert!(captured[0].params.len() > 0);
    let _ = BTreeMap::<String, Value>::new(); // touch import
}

#[tokio::test]
async fn prefix_label_scopes_inserts_and_queries() {
    // Same Pipeline configuration drives both ingest and read: the
    // prefix label must appear in every MERGE pattern (nodes + edges)
    // and in every MATCH pattern produced from a plain DSL document.
    let mock = Arc::new(MockClient::new());
    let cfg = test_config();
    let pipeline = Pipeline::new(mock.clone(), &cfg).with_prefix_label(Some("Tenant1"));

    // Ingest path.
    let mut g = GraphBuilder::new();
    let a = g
        .entity("Person")
        .strict_primary_key("id")
        .property("id", PropertyType::String, "a")
        .add();
    let b = g
        .entity("Person")
        .strict_primary_key("id")
        .property("id", PropertyType::String, "b")
        .add();
    g.relationship(a, "KNOWS", b).add().unwrap();
    let summary = pipeline.ingest(&g.build()).await.unwrap();
    assert_eq!(summary.node_rows, 2);
    assert_eq!(summary.relation_rows, 1);

    {
        let captured = mock.captured.lock().unwrap();
        let texts: Vec<&str> = captured.iter().map(|c| c.text.as_str()).collect();
        assert!(
            texts
                .iter()
                .any(|t| t.contains("MERGE (n:Person:Tenant1 {id: row.id})")),
            "node MERGE missing prefix label; got: {texts:?}"
        );
        assert!(
            texts.iter().any(|t| {
                t.contains("MATCH (a:Person:Tenant1 {id: rel.from})")
                    && t.contains("MATCH (b:Person:Tenant1 {id: rel.to})")
            }),
            "relation MATCH missing prefix label; got: {texts:?}"
        );
    }

    // Read path. The DSL doesn't mention the prefix — the pipeline's
    // configured prefix must flow through into the rendered MATCH.
    mock.enqueue(QueryResult {
        columns: vec!["name".into()],
        rows: vec![],
    });
    let dsl = dsl::parse_str(
        r#"{
            "action": "find",
            "start": { "label": "Person", "alias": "p" },
            "return": [{ "field": "p.name" }]
        }"#,
    )
    .unwrap();
    let _ = pipeline.run(dsl).await.unwrap();
    let captured = mock.captured.lock().unwrap();
    let last = captured.last().unwrap();
    assert!(
        last.text.starts_with("MATCH (p:Person:Tenant1)"),
        "query MATCH missing prefix label; got: {}",
        last.text
    );
}

#[tokio::test]
async fn dsl_prefix_label_overrides_pipeline_default() {
    let mock = Arc::new(MockClient::new());
    let cfg = test_config();
    let pipeline = Pipeline::new(mock.clone(), &cfg).with_prefix_label(Some("DefaultTenant"));
    mock.enqueue(QueryResult {
        columns: vec![],
        rows: vec![],
    });

    let dsl = dsl::parse_str(
        r#"{
            "action": "find",
            "start": { "label": "Person", "alias": "p" },
            "return": [{ "field": "p.name" }],
            "prefix_label": "OverrideTenant"
        }"#,
    )
    .unwrap();
    let _ = pipeline.run(dsl).await.unwrap();
    let captured = mock.captured.lock().unwrap();
    assert!(
        captured[0].text.starts_with("MATCH (p:Person:OverrideTenant)"),
        "DSL prefix_label should override pipeline default; got: {}",
        captured[0].text
    );
}

#[tokio::test]
async fn default_limit_is_applied_when_omitted() {
    let mock = Arc::new(MockClient::new());
    let cfg = test_config();
    let pipeline = Pipeline::new(mock.clone(), &cfg);

    let dsl = dsl::parse_str(
        r#"{
            "action": "find",
            "start": { "label": "Person", "alias": "p" },
            "return": [{ "field": "p.name" }]
        }"#,
    )
    .unwrap();
    let _ = pipeline.run(dsl).await.unwrap();
    let captured = mock.captured.lock().unwrap();
    assert!(captured[0].text.contains("LIMIT 50"));
}
