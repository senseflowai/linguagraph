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

fn doc_from_json(v: serde_json::Value) -> DocumentInput {
    serde_json::from_value(v).expect("doc parses")
}

#[tokio::test]
async fn ingest_document_writes_expected_batches() {
    let mock = Arc::new(MockClient::new());
    let pipeline = Pipeline::new(mock.clone(), &test_config())
        .with_embedder(Arc::new(MockEmbedder::new(8)));

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
    // Side effects: one Qdrant batch (both chunks bucket together), 2 rows.
    assert_eq!(summary.side_effect_batches, 1);
    assert_eq!(summary.side_effect_rows, 2);

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
    // Side effect Qdrant call appears.
    assert!(
        texts
            .iter()
            .any(|t| t.contains("libqlink.insert_labeled")),
        "expected libqlink.insert_labeled call in captured Cypher"
    );
}

#[tokio::test]
async fn ingest_document_sanitizes_dynamic_identifiers() {
    let mock = Arc::new(MockClient::new());
    let pipeline = Pipeline::new(mock.clone(), &test_config())
        .with_embedder(Arc::new(MockEmbedder::new(8)));

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
async fn ingest_document_requires_embedder() {
    // No embedder configured.
    let mock = Arc::new(MockClient::new());
    let pipeline = Pipeline::new(mock.clone(), &test_config());

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

    // The node batches will land first; the side-effect drain step then
    // fails because no embedder is configured. We don't care which error
    // shape comes back — only that the call surfaces it.
    let res = pipeline.ingest_document(doc).await;
    assert!(res.is_err(), "expected ingest to fail without an embedder");
}

#[tokio::test]
async fn ingest_document_is_idempotent_via_uuid_v5() {
    // Re-ingesting the same document should produce identical Cypher
    // batches because UUID v5 is deterministic.
    let mock = Arc::new(MockClient::new());
    let pipeline = Pipeline::new(mock.clone(), &test_config())
        .with_embedder(Arc::new(MockEmbedder::new(8)));

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
    let pipeline = Pipeline::new(mock.clone(), &test_config())
        .with_embedder(Arc::new(MockEmbedder::new(8)));

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
    let pipeline = Pipeline::new(mock.clone(), &test_config())
        .with_embedder(Arc::new(MockEmbedder::new(8)));

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
