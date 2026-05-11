//! Integration tests for [`Pipeline::ingest_document`].
//!
//! Exercises the full bypass path: JSON → InsertPlan → InsertQuery →
//! Cypher → MockClient + MockEmbedder.

use std::sync::Arc;

use linguagraph::config::{Config, DatabaseConfig, LlmConfig, MetadataConfig, QueryConfig};
use linguagraph::core::Pipeline;
use linguagraph::db::MockClient;
use linguagraph::embeddings::MockEmbedder;
use linguagraph::ingest::DocumentInput;
use linguagraph::types::handlers::{SemanticTextConfig, SemanticTextHandler};
use linguagraph::types::{RegistryBuilder, SharedRegistry};

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
        metadata: MetadataConfig::default(),
        types: Default::default(),
    }
}

/// Registry with a SemanticText handler backed by the mock embedder —
/// required by [`Pipeline::ingest_document`].
fn test_registry() -> SharedRegistry {
    let cfg = SemanticTextConfig {
        embedding_model: None,
        collection: "test".into(),
        top_k: 10,
        search_threshold: 0.8,
        reranker_threshold: 0.3,
    };
    Arc::new(
        RegistryBuilder::new()
            .register(SemanticTextHandler::new(
                cfg,
                Arc::new(MockEmbedder::new(8)),
            ))
            .build(),
    )
}

/// Wire up a Pipeline that satisfies [`Pipeline::ingest_document`]'s
/// pre-conditions (SemanticText handler + embedder).
fn test_pipeline(mock: Arc<MockClient>) -> Pipeline {
    Pipeline::new(mock, &test_config())
        .with_embedder(Arc::new(MockEmbedder::new(8)))
        .with_registry(test_registry())
}

fn doc_from_json(v: serde_json::Value) -> DocumentInput {
    serde_json::from_value(v).expect("doc parses")
}

#[tokio::test]
async fn ingest_document_writes_expected_batches() {
    let mock = Arc::new(MockClient::new());
    let pipeline = test_pipeline(mock.clone());

    let doc = doc_from_json(serde_json::json!({
        "document": {
            "name": "users.txt",
            "path": "/docs/users.txt",
            "chunks": [
                {
                    "id": "c1",
                    "text": "Alice works at Acme.",
                    "entities": [
                        {"id": "e1", "type": "Person", "name": "Alice"},
                        {"id": "e2", "type": "Company", "name": "Acme"}
                    ],
                    "relations": [
                        {"from": "e1", "to": "e2", "type": "WORKS_AT"}
                    ]
                },
                {
                    "id": "c2",
                    "text": "Bob also works at Acme.",
                    "entities": [
                        {"id": "e1", "type": "Person", "name": "Bob"},
                        {"id": "e2", "type": "Company", "name": "Acme"}
                    ],
                    "relations": [
                        {"from": "e1", "to": "e2", "type": "WORKS_AT"}
                    ]
                }
            ]
        }
    }));

    let summary = pipeline
        .ingest_document(doc)
        .await
        .expect("ingest succeeds");

    // 1 Document row + 2 Chunk rows + 2 Person rows + 2 Company rows.
    assert_eq!(summary.node_rows, 7, "expected 7 node rows total");
    // 2 HAS_CHUNK + 2 MENTIONS(Person) + 2 MENTIONS(Company) + 2 WORKS_AT.
    assert_eq!(summary.relation_rows, 8, "expected 8 relation rows total");
    // Side effects: chunks bucket together (1 batch, 2 rows); Person
    // names bucket together (1 batch, 2 rows); Company names bucket
    // together (1 batch, 2 rows). 3 batches, 6 rows.
    assert_eq!(summary.side_effect_batches, 3);
    assert_eq!(summary.side_effect_rows, 6);

    // Inspect the captured Cypher.
    let captured = mock.captured.lock().unwrap();
    let texts: Vec<String> = captured.iter().map(|q| q.text.clone()).collect();

    // Built-in labels appear.
    assert!(texts.iter().any(|t| t.contains("MERGE (n:Document")));
    assert!(texts.iter().any(|t| t.contains("MERGE (n:Chunk")));
    // Dynamic labels appear.
    assert!(texts.iter().any(|t| t.contains("MERGE (n:Person")));
    assert!(texts.iter().any(|t| t.contains("MERGE (n:Company")));
    // Built-in relations appear.
    assert!(texts.iter().any(|t| t.contains("MERGE (a)-[:HAS_CHUNK]->")));
    assert!(texts.iter().any(|t| t.contains("MERGE (a)-[:MENTIONS]->")));
    // User relations appear.
    assert!(texts.iter().any(|t| t.contains("MERGE (a)-[:WORKS_AT]->")));
    // Side effect Qdrant calls appear — one per (collection, label)
    // bucket: Chunk__text, Person__name, Company__name.
    let qdrant_calls = texts
        .iter()
        .filter(|t| t.contains("libqlink.insert_labeled"))
        .count();
    assert_eq!(
        qdrant_calls, 3,
        "expected one libqlink.insert_labeled call per (label, field) bucket; got {qdrant_calls}"
    );
}

#[tokio::test]
async fn ingest_document_sanitizes_dynamic_identifiers() {
    let mock = Arc::new(MockClient::new());
    let pipeline = test_pipeline(mock.clone());

    let doc = doc_from_json(serde_json::json!({
        "document": {
            "name": "music.txt",
            "path": "/d",
            "chunks": [{
                "id": "c1",
                "text": "John of the Beatles.",
                "entities": [
                    {"id": "e1", "type": "Music Group", "name": "Beatles"},
                    {"id": "e2", "type": "Person", "name": "John"}
                ],
                "relations": [
                    {"from": "e2", "to": "e1", "type": "member-of"}
                ]
            }]
        }
    }));

    pipeline.ingest_document(doc).await.expect("ingest succeeds");

    let captured = mock.captured.lock().unwrap();
    let texts: Vec<String> = captured.iter().map(|q| q.text.clone()).collect();

    assert!(
        texts.iter().any(|t| t.contains(":Music_Group")),
        "expected 'Music Group' label to be sanitized to 'Music_Group'; got {texts:?}"
    );
    assert!(
        texts.iter().any(|t| t.contains("[:MEMBER_OF]")),
        "expected 'member-of' relation to be sanitized + uppercased to 'MEMBER_OF'; got {texts:?}"
    );
}

#[tokio::test]
async fn ingest_document_requires_semantic_text_handler() {
    // Pipeline missing the SemanticText handler — registry contains
    // only core scalar parsers.
    let mock = Arc::new(MockClient::new());
    let pipeline = Pipeline::new(mock.clone(), &test_config())
        .with_embedder(Arc::new(MockEmbedder::new(8)));

    let doc = doc_from_json(serde_json::json!({
        "document": {
            "name": "d",
            "path": "/d",
            "chunks": [{
                "id": "c1",
                "text": "x",
                "entities": [],
                "relations": []
            }]
        }
    }));

    let err = pipeline.ingest_document(doc).await.unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("SemanticText"),
        "expected SemanticText-required error, got: {msg}"
    );
}

#[tokio::test]
async fn ingest_document_requires_embedder() {
    // Registry has SemanticText, but no embedder is configured on the
    // pipeline → the drain step fails after the Memgraph batches land.
    let mock = Arc::new(MockClient::new());
    let pipeline = Pipeline::new(mock.clone(), &test_config())
        .with_registry(test_registry());

    let doc = doc_from_json(serde_json::json!({
        "document": {
            "name": "d",
            "path": "/d",
            "chunks": [{
                "id": "c1",
                "text": "x",
                "entities": [],
                "relations": []
            }]
        }
    }));

    let res = pipeline.ingest_document(doc).await;
    assert!(res.is_err(), "expected ingest to fail without an embedder");
}

#[tokio::test]
async fn ingest_document_is_idempotent_via_uuid_v5() {
    // Re-ingesting the same document should produce identical Cypher
    // batches because UUID v5 is deterministic.
    let mock = Arc::new(MockClient::new());
    let pipeline = test_pipeline(mock.clone());

    let make_doc = || {
        doc_from_json(serde_json::json!({
            "document": {
                "name": "d",
                "path": "/idempotent",
                "chunks": [{
                    "id": "c1",
                    "text": "stable",
                    "entities": [{"id": "e1", "type": "Person", "name": "Alice"}],
                    "relations": []
                }]
            }
        }))
    };

    pipeline.ingest_document(make_doc()).await.unwrap();
    let first: Vec<String> = mock
        .captured
        .lock()
        .unwrap()
        .iter()
        .map(|q| q.text.clone())
        .collect();

    // Reset capture state.
    mock.captured.lock().unwrap().clear();
    pipeline.ingest_document(make_doc()).await.unwrap();
    let second: Vec<String> = mock
        .captured
        .lock()
        .unwrap()
        .iter()
        .map(|q| q.text.clone())
        .collect();

    assert_eq!(first, second, "re-ingesting same doc must produce same Cypher");
}

#[tokio::test]
async fn ingest_document_rejects_reserved_labels() {
    let mock = Arc::new(MockClient::new());
    let pipeline = test_pipeline(mock.clone());

    let doc = doc_from_json(serde_json::json!({
        "document": {
            "name": "d",
            "path": "/d",
            "chunks": [{
                "id": "c1",
                "text": "x",
                "entities": [{"id": "e1", "type": "Document", "name": "x"}],
                "relations": []
            }]
        }
    }));

    let err = pipeline.ingest_document(doc).await.unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("reserved entity label"),
        "expected reserved-label error, got: {msg}"
    );
}

#[tokio::test]
async fn ingest_document_rejects_dangling_local_ids() {
    let mock = Arc::new(MockClient::new());
    let pipeline = test_pipeline(mock.clone());

    let doc = doc_from_json(serde_json::json!({
        "document": {
            "name": "d",
            "path": "/d",
            "chunks": [{
                "id": "c1",
                "text": "x",
                "entities": [{"id": "e1", "type": "Person", "name": "Alice"}],
                "relations": [
                    {"from": "e1", "to": "e9", "type": "KNOWS"}
                ]
            }]
        }
    }));

    let err = pipeline.ingest_document(doc).await.unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("unknown local entity id"),
        "expected dangling-id error, got: {msg}"
    );
}
