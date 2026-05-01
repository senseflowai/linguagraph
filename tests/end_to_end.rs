//! End-to-end pipeline test using [`MockClient`] in place of Memgraph.

use std::collections::BTreeMap;
use std::sync::Arc;

use linguagraph::config::{Config, DatabaseConfig, LlmConfig, QueryConfig};
use linguagraph::core::Pipeline;
use linguagraph::db::{MockClient, QueryResult, Row, Value};
use linguagraph::dsl;

fn test_config() -> Config {
    Config {
        database: DatabaseConfig {
            uri: "bolt://test".into(),
            user: "u".into(),
            password: "p".into(),
            max_connections: 1,
            query_timeout_secs: 5,
        },
        llm: LlmConfig::default(),
        query: QueryConfig {
            max_traversal_depth: 4,
            default_limit: 50,
        },
    }
}

#[tokio::test]
async fn pipeline_compiles_and_dispatches_to_client() {
    let mock = Arc::new(MockClient::new());
    let mut row = Row::default();
    row.fields.insert("name".into(), Value::String("Ada".into()));
    mock.enqueue(QueryResult { columns: vec!["name".into()], rows: vec![row] });

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
