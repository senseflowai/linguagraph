//! Integration tests for the [`GraphService`] facade using the in-memory
//! [`MockClient`] and [`MockLlmClient`] — no live Memgraph or LLM.

use std::sync::Arc;

use linguagraph::config::{
    Config, DatabaseConfig, LlmConfig, OntologyCatalogConfig, QueryConfig,
};
use linguagraph::core::Pipeline;
use linguagraph::db::{Column, MockClient, NodeType, QueryResult, Row, Value};
use linguagraph::dsl;
use linguagraph::llm::{LlmClient, MockLlmClient};
use linguagraph::prompt::{GraphSchema, NodeKind, RelKind};
use linguagraph::service::{AskRequest, GraphService};
use serde_json::json;

const DSL: &str = r#"{
  "action":"find",
  "start":{"label":"Person","alias":"p"},
  "traversals":[{"edge":{"label":"OWNS","alias":"r","direction":"out"},
                 "target":{"label":"Company","alias":"c"}}],
  "return":[{"field":"c.name","alias":"name"}]
}"#;

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
        ontology_catalog: OntologyCatalogConfig::default(),
        prompt: Default::default(),
        ingest: Default::default(),
        types: Default::default(),
    }
}

/// A graph-shaped result the mock returns for the compiled `run_graph`.
fn graph_result() -> QueryResult {
    let mut row = Row::default();
    row.fields.insert(
        "nodes".into(),
        Value::Json(json!([
            {"alias":"p","id":1,"labels":["Person"],"props":{"name":"Elena"},"sources":[]},
            {"alias":"c","id":2,"labels":["Company"],"props":{"name":"Acme"},"sources":[]}
        ])),
    );
    row.fields.insert(
        "edges".into(),
        Value::Json(json!([{"id":10,"rel":"OWNS","from":1,"to":2,"props":{"share":80}}])),
    );
    QueryResult {
        columns: vec![Column::new("nodes"), Column::new("edges")],
        rows: vec![row],
    }
}

fn dyn_llm(mock: &Arc<MockLlmClient>) -> Arc<dyn LlmClient> {
    mock.clone()
}

#[tokio::test]
async fn ask_translates_and_shapes_a_graph() {
    let db = Arc::new(MockClient::new());
    db.enqueue(graph_result());

    let llm = Arc::new(MockLlmClient::single(DSL));
    let svc = GraphService::new(Pipeline::new(db.clone(), &test_config()), Some(dyn_llm(&llm)));

    let view = svc
        .ask(AskRequest {
            question: "Who owns Acme?".into(),
            ..Default::default()
        })
        .await
        .expect("ask succeeds");

    assert_eq!(view.nodes.len(), 2);
    assert_eq!(view.edges.len(), 1);
    assert_eq!(view.edges[0].rel, "OWNS");
    assert!(view.cypher.contains("RETURN ["), "cypher: {}", view.cypher);
    // The model was consulted exactly once (valid DSL on the first try),
    // and the prompt was schema-derived (system prompt is non-empty).
    assert_eq!(llm.call_count(), 1);
    assert!(!llm.calls()[0].0.is_empty());
}

#[tokio::test]
async fn ask_repairs_then_gives_up_on_junk_json() {
    let db = Arc::new(MockClient::new());
    let llm = Arc::new(MockLlmClient::single("not json at all"));
    let svc = GraphService::new(Pipeline::new(db, &test_config()), Some(dyn_llm(&llm)));

    let err = svc
        .ask(AskRequest {
            question: "anything".into(),
            ..Default::default()
        })
        .await
        .expect_err("junk JSON must fail");

    assert!(matches!(err, linguagraph::Error::Nl(_)), "got: {err}");
    // One initial attempt + DEFAULT_MAX_REPAIRS (2) repairs.
    assert_eq!(llm.call_count(), 3);
}

#[tokio::test]
async fn ask_without_llm_errors_clearly() {
    let db = Arc::new(MockClient::new());
    let svc = GraphService::new(Pipeline::new(db, &test_config()), None);
    let err = svc
        .ask(AskRequest {
            question: "q".into(),
            ..Default::default()
        })
        .await
        .expect_err("no LLM → error");
    assert!(matches!(err, linguagraph::Error::Nl(_)));
}

#[tokio::test]
async fn run_dsl_shapes_a_graph_without_an_llm() {
    let db = Arc::new(MockClient::new());
    db.enqueue(graph_result());
    let svc = GraphService::new(Pipeline::new(db, &test_config()), None);

    let dsl = dsl::parse_str(DSL).unwrap();
    let view = svc.run_dsl(dsl).await.expect("run_dsl succeeds");
    assert_eq!(view.nodes.len(), 2);
    assert_eq!(view.edges.len(), 1);
}

#[tokio::test]
async fn schema_reshapes_into_entity_and_relation_types() {
    let db = Arc::new(MockClient::new());
    db.set_schema(GraphSchema {
        nodes: vec![NodeKind {
            label: "Person".into(),
            domain: None,
            extra_labels: vec![],
            scopes: vec![],
            description: Some("A human".into()),
            properties: vec![],
        }],
        relationships: vec![RelKind {
            label: "OWNS".into(),
            domain: None,
            description: None,
            from: Some("Person".into()),
            to: Some("Company".into()),
            properties: vec![],
        }],
    });
    let svc = GraphService::new(Pipeline::new(db, &test_config()), None);

    let view = svc.schema().await.expect("schema succeeds");
    assert_eq!(view.entity_types.len(), 1);
    assert_eq!(view.entity_types[0].label, "Person");
    assert_eq!(view.relation_types.len(), 1);
    assert_eq!(view.relation_types[0].to.as_deref(), Some("Company"));
}

#[tokio::test]
async fn entity_detail_shapes_the_first_row() {
    let db = Arc::new(MockClient::new());
    let mut row = Row::default();
    row.fields.insert("id".into(), Value::Json(json!(1)));
    row.fields.insert("labels".into(), Value::Json(json!(["Person"])));
    row.fields
        .insert("props".into(), Value::Json(json!({"name":"Elena"})));
    row.fields.insert("sources".into(), Value::Json(json!([])));
    row.fields.insert("relations".into(), Value::Json(json!([])));
    db.enqueue(QueryResult {
        columns: vec![],
        rows: vec![row],
    });
    let svc = GraphService::new(Pipeline::new(db, &test_config()), None);

    let detail = svc.entity(1).await.expect("entity query").expect("found");
    assert_eq!(detail.id, "1");
    assert_eq!(detail.kind, NodeType::Entity);
    assert_eq!(detail.name.as_deref(), Some("Elena"));

    // Unknown id → empty result → None.
    let db2 = Arc::new(MockClient::new());
    let svc2 = GraphService::new(Pipeline::new(db2, &test_config()), None);
    assert!(svc2.entity(999).await.expect("query ok").is_none());
}
