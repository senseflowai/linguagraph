//! End-to-end tests for the pluggable type system.
//!
//! These tests exercise:
//!
//! * **Registry composition** — registering handlers, resolving by id,
//!   and the “core does not branch on type names” invariant.
//! * **Semantic ingestion** — typed properties are routed through their
//!   handler, which queues an `EmbedAndStore` side effect; the pipeline
//!   drains the queue into a `qlink.insert` Cypher batch after the
//!   Memgraph batches succeed.
//! * **Semantic query compilation** — DSL filters tagged with
//!   `"type": "SemanticText"` and `"op": "search"` lower into Cypher
//!   that calls `qlink.search`.
//! * **Hybrid query compilation** — `"op": "hybrid_search"` lowers into
//!   Cypher that combines an exact-match score with `qlink.score_batch_node`.

use std::collections::BTreeMap;
use std::sync::Arc;

use serde_json::json;

use linguagraph::ast::query::Literal;
use linguagraph::config::{Config, DatabaseConfig, LlmConfig, MetadataConfig, QueryConfig, TypeConfig};
use linguagraph::core::Pipeline;
use linguagraph::db::MockClient;
use linguagraph::dsl;
use linguagraph::embeddings::{MockEmbedder, SharedEmbedder};
use linguagraph::mapper::Mapping;
use linguagraph::metadata::{collect_from_mapping, PropertyMetadata};
use linguagraph::types::{
    handlers::{SemanticTextConfig, SemanticTextHandler},
    RegistryBuilder, SharedRegistry, TypeId, TypeRegistry,
};

fn cfg_with_semantic_text() -> Config {
    let mut types = BTreeMap::new();
    types.insert(
        "SemanticText".to_string(),
        TypeConfig {
            embedding_model: None,
            collection: Some("companies".into()),
            top_k: Some(10),
            embedding_dim: Some(8),
            extra: Default::default(),
        },
    );
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
        query: QueryConfig::default(),
        metadata: MetadataConfig::default(),
        types,
    }
}

fn registry_and_embedder() -> (SharedRegistry, SharedEmbedder) {
    let embedder: SharedEmbedder = Arc::new(MockEmbedder::new(8));
    let cfg = cfg_with_semantic_text();
    let st_cfg = SemanticTextConfig::from_config(&cfg).expect("config block present");
    let registry = RegistryBuilder::new()
        .register(SemanticTextHandler::new(st_cfg, embedder.clone()))
        .build();
    (Arc::new(registry), embedder)
}

#[test]
fn registry_resolves_handler_by_id() {
    let (reg, _) = registry_and_embedder();
    let h = reg.get(&TypeId::new("SemanticText")).expect("registered");
    assert_eq!(h.type_id().as_str(), "SemanticText");
    assert!(reg.get(&TypeId::new("DoesNotExist")).is_err());
}

#[test]
fn registry_advertises_capabilities_via_prompt_hints() {
    let (reg, _) = registry_and_embedder();
    let opts = linguagraph::prompt::PromptOptions {
        type_registry: Some((*reg).clone()),
        include_examples: false,
        ..Default::default()
    };
    let schema = linguagraph::prompt::GraphSchema::default();
    let prompt = linguagraph::prompt::generate_system_prompt(&schema, &opts);
    assert!(prompt.contains("# Field types"));
    assert!(prompt.contains("SemanticText"));
    assert!(prompt.contains("search"));
}

#[tokio::test]
async fn semantic_ingest_runs_qlink_insert_after_memgraph_batches() {
    // Mapping declares `Company.name` as a SemanticText field.
    let mapping: Mapping = serde_json::from_value(json!({
        "entities": [{
            "type": "Company",
            "source_path": "$.companies[*]",
            "primary_key": "$.companies[*].id",
            "properties": [
                {"name": "id",   "source_path": "$.companies[*].id"},
                {"name": "name", "source_path": "$.companies[*].name", "type": "SemanticText"}
            ]
        }]
    }))
    .unwrap();

    let data = json!({
        "companies": [
            {"id": "c1", "name": "Apple Inc."},
            {"id": "c2", "name": "Banana Republic"}
        ]
    });

    let client = Arc::new(MockClient::new());
    let (registry, embedder) = registry_and_embedder();
    let pipeline = Pipeline::new(client.clone(), &cfg_with_semantic_text())
        .with_registry(registry)
        .with_embedder(embedder);

    let summary = pipeline.ingest(&mapping, &data).await.unwrap();

    // Two companies → 1 node MERGE batch, 0 relationship batches.
    assert_eq!(summary.batches_executed, 1);
    assert_eq!(summary.node_rows, 2);
    assert_eq!(summary.relation_rows, 0);
    // Then exactly one qlink.insert batch (one collection).
    assert_eq!(summary.side_effect_batches, 1);
    assert_eq!(summary.side_effect_rows, 2);

    let captured = client.captured.lock().unwrap();
    // [0] = Company MERGE, [1] = qlink.insert batch.
    assert_eq!(captured.len(), 2);
    let qlink_batch = &captured[1];
    assert!(
        qlink_batch.text.contains("CALL libqlink.insert($coll, id(n), row.vec)"),
        "expected qlink.insert in side-effect batch; got:\n{}",
        qlink_batch.text
    );
    assert!(qlink_batch.text.contains("MATCH (n:Company {id: row.key})"));
    let rows = qlink_batch.params.get("rows").expect("rows param");
    match rows {
        Literal::List(items) => assert_eq!(items.len(), 2),
        _ => panic!("rows should be a list"),
    }
    let coll = qlink_batch.params.get("coll").expect("coll param");
    // Per-field collection scope: <configured>__<field_name>.
    assert_eq!(coll, &Literal::String("companies__name".into()));
}

#[tokio::test]
async fn ingest_without_embedder_fails_loudly_when_side_effects_arise() {
    let mapping: Mapping = serde_json::from_value(json!({
        "entities": [{
            "type": "Company",
            "source_path": "$.companies[*]",
            "primary_key": "$.companies[*].id",
            "properties": [
                {"name": "id",   "source_path": "$.companies[*].id"},
                {"name": "name", "source_path": "$.companies[*].name", "type": "SemanticText"}
            ]
        }]
    }))
    .unwrap();
    let data = json!({"companies": [{"id": "c1", "name": "Apple Inc."}]});

    let client = Arc::new(MockClient::new());
    let (registry, _) = registry_and_embedder();
    // Notice: no `.with_embedder(...)` call.
    let pipeline = Pipeline::new(client, &cfg_with_semantic_text()).with_registry(registry);

    let err = pipeline.ingest(&mapping, &data).await.unwrap_err();
    let msg = format!("{err}");
    assert!(msg.contains("embedder is configured") || msg.contains("no embedder"));
}

#[test]
fn semantic_search_compiles_to_qlink_search_call() {
    let cfg = cfg_with_semantic_text();
    let (registry, embedder) = registry_and_embedder();
    let pipeline = Pipeline::new(Arc::new(MockClient::new()), &cfg)
        .with_registry(registry)
        .with_embedder(embedder);

    let dsl_query = dsl::parse_str(
        r#"{
            "action": "find",
            "start": { "label": "Company", "alias": "c" },
            "filters": [
                { "field": "c.name", "type": "SemanticText",
                  "op": "search", "value": "apple" }
            ],
            "return": [{ "field": "c.name", "alias": "name" }],
            "limit": 5
        }"#,
    )
    .unwrap();
    let cypher = pipeline.compile(dsl_query).unwrap();

    // Prelude must come before MATCH and call qlink.search.
    let lines: Vec<&str> = cypher.text.lines().collect();
    let qlink_idx = lines
        .iter()
        .position(|l| l.contains("libqlink.search"))
        .expect("expected libqlink.search in cypher");
    let match_idx = lines
        .iter()
        .position(|l| l.starts_with("MATCH"))
        .expect("expected MATCH");
    assert!(
        qlink_idx < match_idx,
        "libqlink.search prelude must run before the MATCH; got:\n{}",
        cypher.text
    );

    // ORDER BY surfaces the score so closer hits come first.
    assert!(
        cypher.text.contains("ORDER BY") && cypher.text.contains("c__score DESC"),
        "expected ORDER BY c__score DESC; got:\n{}",
        cypher.text
    );

    // The query embedding lives in a parameter, never inline.
    let has_embedding = cypher
        .params
        .values()
        .any(|v| matches!(v, Literal::List(items) if items.len() == 8));
    assert!(has_embedding, "expected an 8-dim embedding parameter");
    assert!(!cypher.text.contains("[0."), "embedding leaked into cypher text");
}

#[test]
fn hybrid_search_combines_exact_and_semantic_signals() {
    let cfg = cfg_with_semantic_text();
    let (registry, embedder) = registry_and_embedder();
    let pipeline = Pipeline::new(Arc::new(MockClient::new()), &cfg)
        .with_registry(registry)
        .with_embedder(embedder);

    let dsl_query = dsl::parse_str(
        r#"{
            "action": "find",
            "start": { "label": "Company", "alias": "c" },
            "filters": [
                { "field": "c.name", "type": "SemanticText",
                  "op": "hybrid_search", "value": "apple" }
            ],
            "return": [{ "field": "c.name", "alias": "name" }]
        }"#,
    )
    .unwrap();
    let cypher = pipeline.compile(dsl_query).unwrap();
    assert!(
        cypher.text.contains("qlink.score_batch_node"),
        "hybrid should call score_batch_node; got:\n{}",
        cypher.text
    );
    assert!(
        cypher.text.contains("c__exact"),
        "hybrid should expose the exact-match signal; got:\n{}",
        cypher.text
    );
    assert!(cypher.text.contains("ORDER BY"));
}

#[test]
fn unknown_type_in_dsl_is_rejected_at_lowering() {
    // Empty registry — no types registered.
    let cfg = cfg_with_semantic_text();
    let pipeline = Pipeline::new(Arc::new(MockClient::new()), &cfg)
        .with_registry(Arc::new(TypeRegistry::empty()));

    let dsl_query = dsl::parse_str(
        r#"{
            "action": "find",
            "start": { "label": "Company", "alias": "c" },
            "filters": [
                { "field": "c.name", "type": "GhostType",
                  "op": "search", "value": "x" }
            ],
            "return": [{ "field": "c.name" }]
        }"#,
    )
    .unwrap();
    let err = pipeline.compile(dsl_query).unwrap_err();
    let msg = format!("{err}");
    assert!(msg.contains("GhostType") || msg.contains("UnknownType"));
}

#[test]
fn unsupported_op_for_type_is_rejected() {
    let cfg = cfg_with_semantic_text();
    let (registry, _) = registry_and_embedder();
    let pipeline = Pipeline::new(Arc::new(MockClient::new()), &cfg).with_registry(registry);

    let dsl_query = dsl::parse_str(
        r#"{
            "action": "find",
            "start": { "label": "Company", "alias": "c" },
            "filters": [
                { "field": "c.location", "type": "SemanticText",
                  "op": "near", "value": [0, 0] }
            ],
            "return": [{ "field": "c.name" }]
        }"#,
    )
    .unwrap();
    let err = pipeline.compile(dsl_query).unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("near") || msg.contains("UnsupportedTypedOp"),
        "expected unsupported-op error, got: {msg}"
    );
}

#[test]
fn plain_filters_remain_untyped_and_compile_without_registry() {
    let cfg = cfg_with_semantic_text();
    let pipeline = Pipeline::new(Arc::new(MockClient::new()), &cfg);
    let dsl_query = dsl::parse_str(
        r#"{
            "action": "find",
            "start": { "label": "Company", "alias": "c" },
            "filters": [
                { "field": "c.industry", "op": "eq", "value": "tech" }
            ],
            "return": [{ "field": "c.name" }]
        }"#,
    )
    .unwrap();
    let cypher = pipeline.compile(dsl_query).unwrap();
    assert!(cypher.text.contains("WHERE c.industry = $p0"));
    assert!(!cypher.text.contains("qlink"));
}

// ─── Auto-resolution from PropertyMetadata ──────────────────────────────
//
// When the DSL omits `"type"` but the property metadata declares one,
// the lowering step should pick up the type from the metadata snapshot
// and route the filter through the matching handler.

fn semantic_mapping() -> Mapping {
    serde_json::from_value(json!({
        "entities": [{
            "type": "Company",
            "source_path": "$.companies[*]",
            "primary_key": "$.companies[*].id",
            "properties": [
                {"name": "id",   "source_path": "$.companies[*].id"},
                {
                    "name": "name",
                    "source_path": "$.companies[*].name",
                    "type": "SemanticText",
                    "description": "the company name"
                },
                {"name": "industry", "source_path": "$.companies[*].industry"}
            ]
        }]
    }))
    .unwrap()
}

#[test]
fn metadata_round_trips_field_types() {
    let mapping = semantic_mapping();
    let meta = collect_from_mapping(&mapping);
    assert_eq!(meta.get_type("Company.name"), Some("SemanticText"));
    assert_eq!(meta.get("Company.name"), Some("the company name"));
    assert_eq!(meta.get_type("Company.industry"), None);
}

#[test]
fn untyped_dsl_filter_auto_resolves_to_semantic_text_via_metadata() {
    let cfg = cfg_with_semantic_text();
    let (registry, embedder) = registry_and_embedder();
    let meta = Arc::new(collect_from_mapping(&semantic_mapping()));
    let pipeline = Pipeline::new(Arc::new(MockClient::new()), &cfg)
        .with_registry(registry)
        .with_embedder(embedder)
        .with_metadata(meta);

    // Notice the DSL has NO `"type"` field — the handler is selected
    // from PropertyMetadata.
    let dsl_query = dsl::parse_str(
        r#"{
            "action": "find",
            "start": { "label": "Company", "alias": "c" },
            "filters": [
                { "field": "c.name", "op": "search", "value": "apple" }
            ],
            "return": [{ "field": "c.name", "alias": "name" }]
        }"#,
    )
    .unwrap();
    let cypher = pipeline.compile(dsl_query).unwrap();
    assert!(
        cypher.text.contains("CALL qlink.search"),
        "auto-resolved SemanticText should compile to qlink.search; got:\n{}",
        cypher.text
    );
}

#[test]
fn explicit_dsl_type_overrides_metadata() {
    // The mapping doesn't tag `c.industry` with any type, but the DSL
    // does — explicit always wins over the inferred metadata value.
    // Conversely, when an explicit type *is* set we must not silently
    // fall back to the metadata's type for the same field.
    let cfg = cfg_with_semantic_text();
    let (registry, embedder) = registry_and_embedder();
    let mut meta = collect_from_mapping(&semantic_mapping());
    // Pretend metadata thought industry was Keyword (a non-registered
    // type) — DSL explicit `SemanticText` should win.
    meta.insert_type("Company.industry", "Keyword");
    let pipeline = Pipeline::new(Arc::new(MockClient::new()), &cfg)
        .with_registry(registry)
        .with_embedder(embedder)
        .with_metadata(Arc::new(meta));

    let dsl_query = dsl::parse_str(
        r#"{
            "action": "find",
            "start": { "label": "Company", "alias": "c" },
            "filters": [
                { "field": "c.industry", "type": "SemanticText",
                  "op": "eq", "value": "tech" }
            ],
            "return": [{ "field": "c.industry" }]
        }"#,
    )
    .unwrap();
    // Compiles cleanly via the explicit SemanticText handler.
    let cypher = pipeline.compile(dsl_query).unwrap();
    assert!(cypher.text.contains("c.industry = $p0"));
}

#[test]
fn untyped_field_without_metadata_stays_plain() {
    let cfg = cfg_with_semantic_text();
    let (registry, embedder) = registry_and_embedder();
    let meta = Arc::new(collect_from_mapping(&semantic_mapping()));
    let pipeline = Pipeline::new(Arc::new(MockClient::new()), &cfg)
        .with_registry(registry)
        .with_embedder(embedder)
        .with_metadata(meta);

    // industry has no type tag in the mapping — should compile as a
    // plain WHERE clause, never touch qlink.
    let dsl_query = dsl::parse_str(
        r#"{
            "action": "find",
            "start": { "label": "Company", "alias": "c" },
            "filters": [
                { "field": "c.industry", "op": "eq", "value": "tech" }
            ],
            "return": [{ "field": "c.name" }]
        }"#,
    )
    .unwrap();
    let cypher = pipeline.compile(dsl_query).unwrap();
    assert!(cypher.text.contains("WHERE c.industry = $p0"));
    assert!(!cypher.text.contains("qlink"));
}

#[test]
fn metadata_lookup_keys_off_label_not_alias() {
    // Same property name on different labels must resolve independently.
    // Here `c` is bound to `Company` and `p` to `Person`. Only
    // `Company.name` is SemanticText.
    let cfg = cfg_with_semantic_text();
    let (registry, embedder) = registry_and_embedder();
    let mut meta = PropertyMetadata::new();
    meta.insert_type("Company.name", "SemanticText");
    // Person.name is left plain.
    let pipeline = Pipeline::new(Arc::new(MockClient::new()), &cfg)
        .with_registry(registry)
        .with_embedder(embedder)
        .with_metadata(Arc::new(meta));

    // Filter on Company.name -> auto SemanticText.
    let q = dsl::parse_str(
        r#"{
            "action": "find",
            "start": { "label": "Company", "alias": "c" },
            "filters": [
                { "field": "c.name", "op": "search", "value": "apple" }
            ],
            "return": [{ "field": "c.name" }]
        }"#,
    )
    .unwrap();
    assert!(pipeline.compile(q).unwrap().text.contains("qlink.search"));

    // Filter on Person.name -> plain (and `search` is not a valid plain
    // op, so this must error rather than silently routing to a wrong
    // handler).
    let q = dsl::parse_str(
        r#"{
            "action": "find",
            "start": { "label": "Person", "alias": "p" },
            "filters": [
                { "field": "p.name", "op": "search", "value": "apple" }
            ],
            "return": [{ "field": "p.name" }]
        }"#,
    )
    .unwrap();
    let err = pipeline.compile(q).unwrap_err();
    let msg = format!("{err}");
    assert!(msg.contains("UnknownPlainOp") || msg.contains("search"));
}

#[tokio::test]
async fn ingest_refreshes_in_memory_metadata_snapshot() {
    // Before ingest the pipeline has no metadata snapshot — so a typed
    // DSL without `"type"` falls through to plain ops. After ingest it
    // *does* have one, and the same DSL auto-resolves.
    let cfg = cfg_with_semantic_text();
    let (registry, embedder) = registry_and_embedder();
    let pipeline = Pipeline::new(Arc::new(MockClient::new()), &cfg)
        .with_registry(registry)
        .with_embedder(embedder);

    let mapping = semantic_mapping();
    let data = json!({
        "companies": [{"id": "c1", "name": "Apple Inc.", "industry": "tech"}]
    });
    pipeline.ingest(&mapping, &data).await.unwrap();

    // Now the in-memory snapshot has `Company.name → SemanticText`.
    let meta = pipeline.metadata().expect("snapshot refreshed by ingest");
    assert_eq!(meta.get_type("Company.name"), Some("SemanticText"));

    let q = dsl::parse_str(
        r#"{
            "action": "find",
            "start": { "label": "Company", "alias": "c" },
            "filters": [
                { "field": "c.name", "op": "search", "value": "apple" }
            ],
            "return": [{ "field": "c.name" }]
        }"#,
    )
    .unwrap();
    assert!(pipeline.compile(q).unwrap().text.contains("qlink.search"));
}

#[test]
fn prompt_surfaces_field_type_marker() {
    use linguagraph::prompt::{
        generate_system_prompt, GraphSchema, NodeKind, Property, PromptOptions, PropertyType,
    };
    let schema = GraphSchema {
        nodes: vec![NodeKind {
            label: "Company".into(),
            properties: vec![
                Property { name: "id".into(), ty: PropertyType::String },
                Property { name: "name".into(), ty: PropertyType::String },
            ],
        }],
        relationships: vec![],
    };
    let mut meta = PropertyMetadata::new();
    meta.insert("Company.name", "the company name");
    meta.insert_type("Company.name", "SemanticText");
    let prompt = generate_system_prompt(
        &schema,
        &PromptOptions {
            property_metadata: Some(meta),
            include_examples: false,
            ..Default::default()
        },
    );
    assert!(
        prompt.contains("name: string @SemanticText /* the company name */"),
        "prompt should annotate typed properties; got:\n{prompt}"
    );
}
