//! End-to-end ingestion tests: data + mapping → Cypher batches → mock client.
//!
//! These tests exercise the full new pipeline (mapper → ingest planner →
//! insert builder → DB client) using fixtures from `examples/`.

use std::sync::Arc;

use linguagraph::ast::query::Literal;
use linguagraph::config::{Config, DatabaseConfig, LlmConfig, MetadataConfig, QueryConfig};
use linguagraph::core::Pipeline;
use linguagraph::db::MockClient;
use linguagraph::ingest;
use linguagraph::mapper::{self, Mapping};

fn cfg() -> Config {
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
        metadata: Default::default(),
        types: Default::default(),
    }
}

fn load_mapping() -> Mapping {
    Mapping::from_str(include_str!("../examples/data_mapping.json")).unwrap()
}

fn load_data() -> serde_json::Value {
    serde_json::from_str(include_str!("../examples/cameras_data.json")).unwrap()
}

#[test]
fn extracts_expected_entity_counts_from_example_fixtures() {
    let mapping = load_mapping();
    let data = load_data();
    let extracted = mapper::extract(&mapping, &data).unwrap();

    let by_label: std::collections::HashMap<_, _> = extracted
        .entities
        .iter()
        .map(|e| (e.label.as_str(), e.rows.len()))
        .collect();

    // Three cameras → 3 Camera rows. Two distinct storage depths but each
    // camera contributes one Storage row pre-dedup (3). Three Sources
    // (one per camera). Two distinct places; pre-dedup Place is 3 rows.
    // Four module instances total (2 + 1 + 1).
    assert_eq!(by_label.get("Camera"), Some(&3));
    assert_eq!(by_label.get("Storage"), Some(&3));
    assert_eq!(by_label.get("Source"), Some(&3));
    assert_eq!(by_label.get("Place"), Some(&3));
    assert_eq!(by_label.get("Location"), Some(&3));
    assert_eq!(by_label.get("VideoAnalyticsModule"), Some(&4));
}

#[test]
fn planner_dedupes_shared_places_and_modules() {
    let mapping = load_mapping();
    let data = load_data();
    let extracted = mapper::extract(&mapping, &data).unwrap();
    let q = ingest::plan(&mapping, extracted).unwrap();

    // Two distinct places (plc-7 shared by cam-001 and cam-002, plc-12 only
    // for cam-003) → one Place batch with 2 deduped rows.
    let place = q
        .node_batches
        .iter()
        .find(|b| b.label == "Place")
        .expect("Place batch present");
    assert_eq!(place.rows.len(), 2, "Place rows should dedupe to 2");

    // Three distinct video-analytics module names: PeopleCounting,
    // LicensePlateRecognition, IntrusionDetection.
    let modules = q
        .node_batches
        .iter()
        .find(|b| b.label == "VideoAnalyticsModule")
        .expect("module batch present");
    assert_eq!(modules.rows.len(), 3);

    // LOCATED_IN: Camera→Place — 3 cameras, each linked to its parent
    // Place by context prefix, so 3 distinct edges.
    let cam_place = q
        .relation_batches
        .iter()
        .find(|r| r.rel_type == "LOCATED_IN" && r.from_label == "Camera")
        .expect("Camera→Place edges present");
    assert_eq!(cam_place.rows.len(), 3);
}

#[test]
fn relationships_align_by_context_prefix() {
    let mapping = load_mapping();
    let data = load_data();
    let extracted = mapper::extract(&mapping, &data).unwrap();
    let q = ingest::plan(&mapping, extracted).unwrap();

    // HAS_ANALYTICS_MODULE: cam-001 → 2 modules, cam-002 → 1, cam-003 → 1.
    // Total 4 edges, BUT MERGE semantics means duplicates collapse: only
    // cam-001 has both PeopleCounting and LicensePlateRecognition, while
    // cam-002 also references PeopleCounting. So unique pairs:
    //   (cam-001, PeopleCounting)
    //   (cam-001, LicensePlateRecognition)
    //   (cam-002, PeopleCounting)
    //   (cam-003, IntrusionDetection)
    let analytics = q
        .relation_batches
        .iter()
        .find(|r| r.rel_type == "HAS_ANALYTICS_MODULE")
        .unwrap();
    assert_eq!(analytics.rows.len(), 4);

    // Sanity: cam-003 must NOT link to a module from cam-001 or cam-002.
    let cross = analytics.rows.iter().any(|r| {
        r.from_id == Literal::String("cam-003".into())
            && (r.to_id == Literal::String("PeopleCounting".into())
                || r.to_id == Literal::String("LicensePlateRecognition".into()))
    });
    assert!(!cross, "context prefix must isolate sibling subtrees");
}

#[test]
fn pipeline_compile_insert_yields_unwind_batches() {
    let mapping = load_mapping();
    let data = load_data();
    let pipeline = Pipeline::new(Arc::new(MockClient::new()), &cfg());
    let batches = pipeline.compile_insert(&mapping, &data).unwrap();

    // Six entity types + five relationships = 11 batches max (some may be
    // empty if dedup eliminates everything; nothing here is empty).
    assert_eq!(batches.len(), 6 + 5);

    // Every batch must use UNWIND, must MERGE, and must keep all data
    // off the template (in $rows / $rels parameter).
    for b in &batches {
        assert!(b.text.contains("UNWIND"), "batch missing UNWIND: {}", b.text);
        assert!(b.text.contains("MERGE"));
        let has_param = b.params.contains_key("rows") || b.params.contains_key("rels");
        assert!(has_param, "batch must carry parameter list");
    }
}

#[tokio::test]
async fn pipeline_ingest_runs_each_batch_through_client() {
    let mock = Arc::new(MockClient::new());
    let pipeline = Pipeline::new(mock.clone(), &cfg());
    let mapping = load_mapping();
    let data = load_data();

    let summary = pipeline.ingest(&mapping, &data).await.expect("ingest runs");

    let captured = mock.captured.lock().unwrap();
    assert_eq!(captured.len(), summary.batches_executed);
    assert!(summary.node_rows > 0);
    assert!(summary.relation_rows > 0);

    // Node batches always come before relation batches, so the first few
    // captured queries must MERGE nodes (no MATCH).
    assert!(captured[0].text.contains("MERGE (n:"));
}

#[test]
fn batch_size_splits_large_inserts() {
    use serde_json::json;

    let mapping: Mapping = serde_json::from_value(json!({
        "entities": [{
            "type": "Item",
            "source_path": "$.items[*]",
            "primary_key": "$.items[*].id"
        }]
    }))
    .unwrap();

    let items: Vec<_> = (0..2500).map(|i| json!({"id": format!("i{i}")})).collect();
    let data = json!({"items": items});

    let pipeline = Pipeline::new(Arc::new(MockClient::new()), &cfg())
        .with_ingest_batch_size(1000);
    let batches = pipeline.compile_insert(&mapping, &data).unwrap();
    // 2500 / 1000 = 3 batches.
    assert_eq!(batches.len(), 3);
}

#[test]
fn duplicate_rows_dedupe_in_node_batches() {
    use serde_json::json;
    let mapping: Mapping = serde_json::from_value(json!({
        "entities": [{
            "type": "Tag",
            "source_path": "$.tags[*]",
            "primary_key": "$.tags[*].name"
        }]
    }))
    .unwrap();
    let data = json!({"tags": [{"name": "a"}, {"name": "a"}, {"name": "b"}]});
    let pipeline = Pipeline::new(Arc::new(MockClient::new()), &cfg());
    let batches = pipeline.compile_insert(&mapping, &data).unwrap();
    assert_eq!(batches.len(), 1);
    // The single Tag batch has two rows after dedup.
    let rows_param = batches[0].params.get("rows").unwrap();
    match rows_param {
        Literal::List(items) => assert_eq!(items.len(), 2),
        _ => panic!("rows param must be a list"),
    }
}

#[test]
fn missing_primary_key_in_data_is_fatal() {
    use serde_json::json;
    let mapping: Mapping = serde_json::from_value(json!({
        "entities": [{
            "type": "Camera",
            "source_path": "$.cameras[*]",
            "primary_key": "$.cameras[*].id"
        }]
    }))
    .unwrap();
    let data = json!({"cameras": [{"name": "no id"}]});
    let pipeline = Pipeline::new(Arc::new(MockClient::new()), &cfg());
    let err = pipeline.compile_insert(&mapping, &data).unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("primary key"), "got: {msg}");
}

#[test]
fn invalid_jsonpath_in_mapping_is_fatal() {
    use serde_json::json;
    let mapping: Mapping = serde_json::from_value(json!({
        "entities": [{
            "type": "X",
            "source_path": "cameras[*]",
            "primary_key": "$.cameras[*].id"
        }]
    }))
    .unwrap();
    let pipeline = Pipeline::new(Arc::new(MockClient::new()), &cfg());
    let err = pipeline.compile_insert(&mapping, &json!({})).unwrap_err();
    assert!(err.to_string().contains("invalid JSONPath"));
}

#[test]
fn cypher_template_rejects_label_with_special_chars() {
    use linguagraph::ast::query::{InsertQuery, NodeBatch, NodeRow};
    use linguagraph::builder;
    use std::collections::BTreeMap;

    let q = InsertQuery {
        node_batches: vec![NodeBatch {
            label: "Camera) DETACH DELETE n //".into(),
            merge_on: "id".into(),
            rows: vec![NodeRow {
                id: Literal::String("x".into()),
                props: BTreeMap::new(),
            }],
        }],
        relation_batches: vec![],
    };
    let err = builder::build_insert(&q).unwrap_err();
    assert!(err.to_string().contains("invalid Cypher identifier"));
}
