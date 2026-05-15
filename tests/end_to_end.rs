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
async fn prefix_index_scopes_embedding_collections() {
    // The pipeline's prefix_index must reach both the ingest-side
    // embedding side effects (qlink.insert_labeled) and the read-side
    // collection parameter passed to qlink.search_*.
    use linguagraph::embeddings::MockEmbedder;
    use linguagraph::types::handlers::{SemanticTextConfig, SemanticTextHandler};
    use linguagraph::types::{handlers, RegistryBuilder, SharedRegistry};

    let registry: SharedRegistry = std::sync::Arc::new(
        handlers::register_core(RegistryBuilder::new())
            .register(SemanticTextHandler::new(
                SemanticTextConfig {
                    embedding_model: None,
                    collection: "docs".into(),
                    top_k: 10,
                    search_threshold: 0.1,
                    reranker_threshold: 0.2,
                },
                std::sync::Arc::new(MockEmbedder::new(8)),
            ))
            .build(),
    );
    let embedder: linguagraph::embeddings::SharedEmbedder =
        std::sync::Arc::new(MockEmbedder::new(8));

    let mock = Arc::new(MockClient::new());
    let cfg = test_config();
    let pipeline = Pipeline::new(mock.clone(), &cfg)
        .with_registry(registry)
        .with_embedder(embedder)
        .with_prefix_index(Some("Tenant1"));

    // Ingest a SemanticText property.
    let mut g = GraphBuilder::new();
    g.entity("Person")
        .strict_primary_key("id")
        .property("id", PropertyType::String, "a")
        .property("name", PropertyType::Text, "Alice")
        .add();
    pipeline.ingest(&g.build()).await.unwrap();

    {
        let captured = mock.captured.lock().unwrap();
        // The last batch is the qlink insert; its `coll` parameter must
        // be the prefixed collection name.
        let insert = captured
            .iter()
            .find(|c| c.text.contains("libqlink.insert_labeled"))
            .expect("expected a qlink insert batch");
        let coll = insert
            .params
            .get("coll")
            .expect("qlink insert must bind 'coll'");
        let coll = match coll {
            linguagraph::ast::query::Literal::String(s) => s.clone(),
            other => panic!("coll should be a String, got {other:?}"),
        };
        assert!(
            coll.starts_with("Tenant1__"),
            "ingest collection missing prefix; got {coll}"
        );
    }

    // Query: typed SemanticText search must also use the prefixed
    // collection.
    mock.enqueue(QueryResult {
        columns: vec![],
        rows: vec![],
    });
    let dsl = dsl::parse_str(
        r#"{
            "action": "find",
            "start": { "label": "Person", "alias": "p" },
            "filters": [
                {"field": "p.name", "type": "SemanticText", "op": "search", "value": "alice"}
            ],
            "return": [{ "field": "p.name" }]
        }"#,
    )
    .unwrap();
    let _ = pipeline.run(dsl).await.unwrap();
    let captured = mock.captured.lock().unwrap();
    let last = captured.last().unwrap();
    let coll = match last.params.get("p0").or_else(|| last.params.get("p1")) {
        Some(linguagraph::ast::query::Literal::String(s)) => Some(s.clone()),
        _ => None,
    };
    let has_prefixed_coll = last
        .params
        .values()
        .any(|v| matches!(v, linguagraph::ast::query::Literal::String(s) if s.starts_with("Tenant1__")));
    assert!(
        has_prefixed_coll,
        "query collection param missing prefix; params={:?}; cypher={}",
        last.params, last.text
    );
    let _ = coll;
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
