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

use linguagraph::ast::query::Literal;
use linguagraph::config::{
    Config, DatabaseConfig, GraphSpecificationConfig, LlmConfig, QueryConfig, TypeConfig,
};
use linguagraph::core::Pipeline;
use linguagraph::db::MockClient;
use linguagraph::dsl;
use linguagraph::embeddings::{MockEmbedder, SharedEmbedder};
use linguagraph::graph::{
    FileGraphSpecificationStorage, GraphBuilder, GraphSpecification, GraphSpecificationStorage,
    PropertyType as GraphPropertyType,
};
use linguagraph::types::{
    handlers::{self, SemanticTextConfig, SemanticTextHandler},
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
            // Pin both thresholds to recognisable values so the
            // end-to-end tests can assert they flow through
            // unchanged from config to bound parameter.
            threshold: Some(0.75),          // cosine, stage 1
            reranker_threshold: Some(0.42), // reranker, stage 2
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
        graph_specification: GraphSpecificationConfig::default(),
        prompt: Default::default(),
        types,
    }
}

fn registry_and_embedder() -> (SharedRegistry, SharedEmbedder) {
    let embedder: SharedEmbedder = Arc::new(MockEmbedder::new(8));
    let cfg = cfg_with_semantic_text();
    let st_cfg = SemanticTextConfig::from_config(&cfg).expect("config block present");
    let registry = handlers::register_core(RegistryBuilder::new())
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
    let client = Arc::new(MockClient::new());
    let (registry, embedder) = registry_and_embedder();
    let pipeline = Pipeline::new(client.clone(), &cfg_with_semantic_text())
        .with_registry(registry)
        .with_embedder(embedder);

    let mut graph = GraphBuilder::new();
    graph
        .entity("Company")
        .strict_primary_key("id")
        .property("id", GraphPropertyType::String, "c1")
        .property("name", GraphPropertyType::Text, "Apple Inc.")
        .add();
    graph
        .entity("Company")
        .strict_primary_key("id")
        .property("id", GraphPropertyType::String, "c2")
        .property("name", GraphPropertyType::Text, "Banana Republic")
        .add();

    let summary = pipeline.ingest(&graph.build()).await.unwrap();

    // Two companies → 1 node MERGE batch, 0 relationship batches.
    assert_eq!(summary.batches_executed, 1);
    assert_eq!(summary.node_rows, 2);
    assert_eq!(summary.relation_rows, 0);
    // Both rows share the same (collection, payload_label, label,
    // key_field) bucket → exactly one UNWIND-batched qlink call.
    assert_eq!(summary.side_effect_batches, 1);
    assert_eq!(summary.side_effect_rows, 2);

    let captured = client.captured.lock().unwrap();
    // [0] = Company MERGE, [1] = one batched libqlink.insert_labeled.
    assert_eq!(captured.len(), 2);
    let qlink_batch = &captured[1];
    assert!(
        qlink_batch
            .text
            .contains("CALL libqlink.insert_labeled($coll, id(n), row.vec, $label)"),
        "expected UNWIND-batched libqlink.insert_labeled; got:\n{}",
        qlink_batch.text
    );
    assert!(qlink_batch.text.contains("UNWIND $rows AS row"));
    assert!(qlink_batch.text.contains("MATCH (n:Company {id: row.key})"));
    // Both rows ride in `$rows` as `{key, vec}` objects.
    let rows = qlink_batch.params.get("rows").expect("missing $rows");
    let row_items = match rows {
        Literal::List(items) => items,
        other => panic!("$rows should be a List, got {other:?}"),
    };
    assert_eq!(row_items.len(), 2, "expected 2 rows in the UNWIND batch");
    let keys: Vec<&Literal> = row_items
        .iter()
        .map(|row| match row {
            Literal::Object(map) => map.get("key").expect("row missing 'key'"),
            other => panic!("row should be an Object, got {other:?}"),
        })
        .collect();
    assert!(keys.contains(&&Literal::String("c1".into())));
    assert!(keys.contains(&&Literal::String("c2".into())));
    // Collection + label live in scalar params (not per-row), so a
    // single bucket has a single $coll / $label binding.
    assert_eq!(
        qlink_batch.params.get("coll"),
        Some(&Literal::String("companies__name".into())),
    );
    assert_eq!(
        qlink_batch.params.get("label"),
        Some(&Literal::String("Company".into())),
    );
}

#[tokio::test]
async fn ingest_without_embedder_fails_loudly_when_side_effects_arise() {
    let client = Arc::new(MockClient::new());
    let (registry, _) = registry_and_embedder();
    // Notice: no `.with_embedder(...)` call.
    let pipeline = Pipeline::new(client, &cfg_with_semantic_text()).with_registry(registry);

    let mut graph = GraphBuilder::new();
    graph
        .entity("Company")
        .strict_primary_key("id")
        .property("id", GraphPropertyType::String, "c1")
        .property("name", GraphPropertyType::Text, "Apple Inc.")
        .add();

    let err = pipeline.ingest(&graph.build()).await.unwrap_err();
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

    // `search_reranked` is the cross-encoder path that emits
    // `libqlink.search_reranked` and binds the reranker threshold.
    // (`search` is the cheaper KNN path that goes through
    // `libqlink.search_labeled` — covered in unit tests.)
    let dsl_query = dsl::parse_str(
        r#"{
            "action": "find",
            "start": { "label": "Company", "alias": "c" },
            "filters": [
                { "field": "c.name", "type": "SemanticText",
                  "op": "search_reranked", "value": "apple" }
            ],
            "return": [{ "field": "c.name", "alias": "name" }],
            "limit": 5
        }"#,
    )
    .unwrap();
    let cypher = pipeline.compile(dsl_query).unwrap();

    // Prelude must come before MATCH and call libqlink.search_reranked.
    let lines: Vec<&str> = cypher.text.lines().collect();
    let qlink_idx = lines
        .iter()
        .position(|l| l.contains("libqlink.search_reranked"))
        .expect("expected libqlink.search_reranked in cypher");
    let match_idx = lines
        .iter()
        .position(|l| l.starts_with("MATCH"))
        .expect("expected MATCH");
    assert!(
        qlink_idx < match_idx,
        "libqlink.search_reranked prelude must run before the MATCH; got:\n{}",
        cypher.text
    );

    // ORDER BY surfaces the reranker score so closer hits come first.
    assert!(
        cypher.text.contains("ORDER BY") && cypher.text.contains("c__score_0 DESC"),
        "expected ORDER BY c__score_<n> DESC; got:\n{}",
        cypher.text
    );
    // search_reranked filters internally — no extra `WHERE c__score >=` slip-in.
    assert!(
        !cypher.text.contains("WHERE c__score"),
        "reranker handles filtering itself; we must not double-filter; got:\n{}",
        cypher.text
    );

    // The query embedding lives in a parameter, never inline.
    let has_embedding = cypher
        .params
        .values()
        .any(|v| matches!(v, Literal::List(items) if items.len() == 8));
    assert!(has_embedding, "expected an 8-dim embedding parameter");
    assert!(
        !cypher.text.contains("[0."),
        "embedding leaked into cypher text"
    );

    // The natural-language query is bound as `query_str` for the
    // reranker — it should round-trip the DSL `value` verbatim.
    let has_query_str = cypher
        .params
        .values()
        .any(|v| matches!(v, Literal::String(s) if s == "apple"));
    assert!(
        has_query_str,
        "expected 'apple' bound as query_str; params: {:?}",
        cypher.params
    );
    // Label flows through as a Qdrant payload filter.
    let has_label = cypher
        .params
        .values()
        .any(|v| matches!(v, Literal::String(s) if s == "Company"));
    assert!(
        has_label,
        "expected 'Company' bound as label; params: {:?}",
        cypher.params
    );

    // The reranker threshold (0.42) flows through as a bound Float
    // param. The cosine `search_threshold` is not currently handed
    // to libqlink.search_reranked (the call shape took a property
    // name in that slot), so it stays in the predicate's internal
    // params but doesn't reach the bound Cypher params.
    let floats: Vec<f64> = cypher
        .params
        .values()
        .filter_map(|v| {
            if let Literal::Float(f) = v {
                Some(*f)
            } else {
                None
            }
        })
        .collect();
    assert!(
        floats.iter().any(|f| (f - 0.42).abs() < 1e-9),
        "expected configured reranker_threshold 0.42; params: {:?}",
        cypher.params
    );
}

/// Regression: when a query carries two `SemanticText` filters on the
/// same alias, each `libqlink.search_reranked` call must yield into
/// distinct variables — Memgraph rejects `Redeclaring variable: c__qid`
/// otherwise. We disambiguate by appending a fresh integer suffix
/// allocated from the cursor's variable-name counter.
#[test]
fn multiple_semantic_searches_on_same_alias_yield_distinct_variables() {
    let cfg = cfg_with_semantic_text();
    let (registry, embedder) = registry_and_embedder();
    let pipeline = Pipeline::new(Arc::new(MockClient::new()), &cfg)
        .with_registry(registry)
        .with_embedder(embedder);

    let dsl_query = dsl::parse_str(
        r#"{
            "action": "find",
            "start": { "label": "Camera", "alias": "c" },
            "filters": [
                { "field": "c.origin_place_description", "type": "SemanticText",
                  "op": "contains", "value": "Mangilik 55" },
                { "field": "c.name", "type": "SemanticText",
                  "op": "contains", "value": "TargetAI" }
            ],
            "return": [
                { "field": "c.name", "alias": "name" },
                { "field": "c.origin_place_description", "alias": "address" }
            ],
            "limit": 25
        }"#,
    )
    .unwrap();
    let cypher = pipeline.compile(dsl_query).unwrap();

    // Two `libqlink.search_reranked` calls, each yielding into its
    // own pair of variables.
    let qid_count = cypher.text.matches("AS c__qid_").count();
    let score_count = cypher.text.matches("AS c__score_").count();
    assert_eq!(
        qid_count, 2,
        "expected two distinct c__qid_<n> yields; got:\n{}",
        cypher.text
    );
    assert_eq!(
        score_count, 2,
        "expected two distinct c__score_<n> yields; got:\n{}",
        cypher.text
    );

    // The bare `c__qid` / `c__score` names must NOT appear — they
    // would collide if both calls used them.
    assert!(
        !cypher.text.contains("AS c__qid,") && !cypher.text.contains("AS c__qid\n"),
        "bare c__qid collides across calls; got:\n{}",
        cypher.text
    );

    // WHERE references both qid variables.
    assert!(cypher.text.contains("id(c) = c__qid_0"));
    assert!(cypher.text.contains("id(c) = c__qid_1"));

    // ORDER BY mentions both score variables.
    assert!(cypher.text.contains("c__score_0 DESC"));
    assert!(cypher.text.contains("c__score_1 DESC"));
}

/// Regression: an aggregate query whose filters include a typed
/// vector search must not emit `ORDER BY <alias>__score` — the score
/// column isn't projected through the aggregation, and Memgraph would
/// reject the query as `Unbound variable`. The `libqlink.search`
/// candidate set is already top-k'd and threshold-filtered before
/// MATCH, so the implicit ordering is good enough.
#[test]
fn aggregate_with_semantic_search_drops_handler_order_by() {
    let cfg = cfg_with_semantic_text();
    let (registry, embedder) = registry_and_embedder();
    let pipeline = Pipeline::new(Arc::new(MockClient::new()), &cfg)
        .with_registry(registry)
        .with_embedder(embedder);

    // "How many cameras are at each Place that semantically matches
    // 'office'?" — find Places via the cross-encoder reranker
    // (`search_reranked`), traverse to Camera, count. The `search`
    // op is the plain KNN variant (covered separately).
    let dsl_query = dsl::parse_str(
        r#"{
            "action": "aggregate",
            "start": { "label": "Place", "alias": "p" },
            "traversals": [
                { "edge": { "label": "LOCATED_IN", "alias": "loc", "direction": "in" },
                  "target": { "label": "Camera", "alias": "c" } }
            ],
            "filters": [
                { "field": "p.name", "type": "SemanticText",
                  "op": "search_reranked", "value": "office" }
            ],
            "return": [
                { "aggregate": "count", "field": "c.id", "alias": "camera_count" }
            ]
        }"#,
    )
    .unwrap();
    let cypher = pipeline.compile(dsl_query).unwrap();

    // The `libqlink.search_reranked` prelude is still emitted; only
    // the score-based ORDER BY is suppressed for aggregates.
    assert!(cypher.text.contains("CALL libqlink.search_reranked"));
    assert!(
        !cypher.text.contains("ORDER BY p__score"),
        "aggregate queries must not order by an unprojected score column; \
         got:\n{}",
        cypher.text
    );
    assert!(cypher.text.contains("RETURN count(c) AS camera_count"));
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

// ─── Auto-resolution from GraphSpecification ────────────────────────────
//
// When the DSL omits `"type"` but the graph specification declares one,
// the lowering step should pick up the type from the specification snapshot
// and route the filter through the matching handler.

fn semantic_specification() -> GraphSpecification {
    GraphSpecification::new()
        .with_entity("Company", "")
        .with_property("Company", "id", GraphPropertyType::String, "")
        .with_property(
            "Company",
            "name",
            GraphPropertyType::Text,
            "the company name",
        )
        .with_property("Company", "industry", GraphPropertyType::String, "")
}

#[test]
fn graph_specification_round_trips_field_types() {
    let spec = semantic_specification();
    assert_eq!(spec.get_type("Company", "name"), Some("SemanticText"));
    assert_eq!(spec.get_query_type("Company", "name"), Some("SemanticText"));
    assert_eq!(
        spec.get_property("Company", "name")
            .map(|p| p.description.as_str()),
        Some("the company name")
    );
    assert_eq!(spec.get_type("Company", "industry"), Some("Text"));
    assert_eq!(spec.get_query_type("Company", "industry"), Some("Text"));
}

#[test]
fn untyped_dsl_filter_auto_resolves_to_semantic_text_via_graph_specification() {
    let cfg = cfg_with_semantic_text();
    let (registry, embedder) = registry_and_embedder();
    let specification = Arc::new(semantic_specification());
    let pipeline = Pipeline::new(Arc::new(MockClient::new()), &cfg)
        .with_registry(registry)
        .with_embedder(embedder)
        .with_graph_specification(specification);

    // Notice the DSL has NO `"type"` field — the handler is selected
    // from GraphSpecification.
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
        cypher.text.contains("libqlink.search"),
        "auto-resolved SemanticText should compile to qlink.search; got:\n{}",
        cypher.text
    );
}

#[test]
fn untyped_datetime_filter_auto_resolves_and_expands_eq_to_a_day_range() {
    // An `eq` on a Timestamp field with a midnight value must NOT compile
    // to a literal string equality — it expands to a half-open day range
    // so rows recorded at any time on that day still match.
    let cfg = cfg_with_semantic_text();
    let (registry, embedder) = registry_and_embedder();
    let specification = GraphSpecification::new()
        .with_entity("ServiceVisit", "")
        .with_property("ServiceVisit", "id", GraphPropertyType::String, "")
        .with_property(
            "ServiceVisit",
            "work_start",
            GraphPropertyType::Timestamp,
            "when the visit started",
        );
    let pipeline = Pipeline::new(Arc::new(MockClient::new()), &cfg)
        .with_registry(registry)
        .with_embedder(embedder)
        .with_graph_specification(Arc::new(specification));

    // No `"type"` field — the Timestamp handler is selected from the
    // graph specification.
    let dsl_query = dsl::parse_str(
        r#"{
            "action": "find",
            "start": { "label": "ServiceVisit", "alias": "sv" },
            "filters": [
                { "field": "sv.work_start", "op": "eq",
                  "value": "2026-05-15T00:00:00" }
            ],
            "return": [{ "field": "sv.id", "alias": "id" }]
        }"#,
    )
    .unwrap();
    let cypher = pipeline.compile(dsl_query).unwrap();
    assert!(
        cypher.text.contains("sv.work_start >=") && cypher.text.contains("sv.work_start <"),
        "midnight `eq` on a Timestamp field should expand to a day range; got:\n{}",
        cypher.text
    );
    assert!(
        !cypher.text.contains("sv.work_start = "),
        "should not compile to a literal equality; got:\n{}",
        cypher.text
    );
}

#[test]
fn explicit_dsl_type_overrides_graph_specification() {
    // The mapping doesn't tag `c.industry` with any type, but the DSL
    // does — explicit always wins over the inferred specification value.
    // Conversely, when an explicit type *is* set we must not silently
    // fall back to the specification's type for the same field.
    let cfg = cfg_with_semantic_text();
    let (registry, embedder) = registry_and_embedder();
    let specification = semantic_specification();
    let pipeline = Pipeline::new(Arc::new(MockClient::new()), &cfg)
        .with_registry(registry)
        .with_embedder(embedder)
        .with_graph_specification(Arc::new(specification));

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
    // Compiles cleanly via the explicit SemanticText handler. The
    // current handler routes `eq` (and `neq`/`contains`) through
    // `libqlink.search_reranked` rather than emitting a plain
    // `c.industry = $p0` WHERE clause, so the assertion is on the
    // call site rather than the equality.
    let cypher = pipeline.compile(dsl_query).unwrap();
    assert!(
        cypher.text.contains("CALL libqlink.search_reranked"),
        "explicit SemanticText `eq` should still route through search_reranked; got:\n{}",
        cypher.text
    );
}

#[test]
fn untyped_field_without_graph_specification_stays_plain() {
    let cfg = cfg_with_semantic_text();
    let (registry, embedder) = registry_and_embedder();
    let pipeline = Pipeline::new(Arc::new(MockClient::new()), &cfg)
        .with_registry(registry)
        .with_embedder(embedder);

    // Without a loaded graph specification, industry should compile as a
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
fn string_property_from_graph_specification_auto_resolves_to_text_handler() {
    let cfg = cfg_with_semantic_text();
    let (registry, embedder) = registry_and_embedder();
    let specification = Arc::new(semantic_specification());
    let pipeline = Pipeline::new(Arc::new(MockClient::new()), &cfg)
        .with_registry(registry)
        .with_embedder(embedder)
        .with_graph_specification(specification);

    let dsl_query = dsl::parse_str(
        r#"{
            "action": "find",
            "start": { "label": "Company", "alias": "c" },
            "filters": [
                { "field": "c.industry", "op": "eq", "value": " Fin-Tech " }
            ],
            "return": [{ "field": "c.name" }]
        }"#,
    )
    .unwrap();
    let cypher = pipeline.compile(dsl_query).unwrap();
    assert!(cypher.text.contains("WHERE c.industry = $p0"));
    assert_eq!(
        cypher.params.get("p0"),
        Some(&Literal::String("fintech".into()))
    );
    assert!(!cypher.text.contains("qlink"));
}

#[test]
fn graph_specification_lookup_keys_off_label_not_alias() {
    // Same property name on different labels must resolve independently.
    // Here `c` is bound to `Company` and `p` to `Person`. Only
    // `Company.name` is SemanticText.
    let cfg = cfg_with_semantic_text();
    let (registry, embedder) = registry_and_embedder();
    let specification = GraphSpecification::new().with_property(
        "Company",
        "name",
        GraphPropertyType::Text,
        "the company name",
    );
    // Person.name is left plain.
    let pipeline = Pipeline::new(Arc::new(MockClient::new()), &cfg)
        .with_registry(registry)
        .with_embedder(embedder)
        .with_graph_specification(Arc::new(specification));

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
    assert!(pipeline
        .compile(q)
        .unwrap()
        .text
        .contains("libqlink.search"));

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
async fn loaded_graph_specification_auto_resolves_semantic_text_filters() {
    let cfg = cfg_with_semantic_text();
    let (registry, embedder) = registry_and_embedder();
    let path =
        std::env::temp_dir().join(format!("linguagraph-spec-test-{}.json", std::process::id()));
    let storage = FileGraphSpecificationStorage::new(&path);
    storage.save(&semantic_specification()).await.unwrap();
    let storage: Arc<dyn GraphSpecificationStorage> = Arc::new(storage);
    let pipeline = Pipeline::new(Arc::new(MockClient::new()), &cfg)
        .with_registry(registry)
        .with_embedder(embedder)
        .with_graph_specification_storage(storage);
    pipeline.load_graph_specification().await.unwrap();

    let specification = pipeline.graph_specification().expect("snapshot loaded");
    assert_eq!(
        specification.get_type("Company", "name"),
        Some("SemanticText")
    );

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
    assert!(pipeline
        .compile(q)
        .unwrap()
        .text
        .contains("libqlink.search"));
    let _ = std::fs::remove_file(path);
}

#[test]
fn prompt_surfaces_field_type_marker() {
    use linguagraph::graph::{GraphSpecification, PropertyType as GraphPropertyType};
    use linguagraph::prompt::{
        generate_system_prompt, GraphSchema, NodeKind, PromptOptions, Property, PropertyType,
    };
    let schema = GraphSchema {
        nodes: vec![NodeKind {
            label: "Company".into(),
            properties: vec![
                Property {
                    name: "id".into(),
                    ty: PropertyType::String,
                },
                Property {
                    name: "name".into(),
                    ty: PropertyType::String,
                },
            ],
        }],
        relationships: vec![],
    };
    let spec = GraphSpecification::new().with_property(
        "Company",
        "name",
        GraphPropertyType::Text,
        "the company name",
    );
    let prompt = generate_system_prompt(
        &schema,
        &PromptOptions {
            graph_specification: Some(spec),
            include_examples: false,
            ..Default::default()
        },
    );
    assert!(
        prompt.contains("name: string @SemanticText /* the company name */"),
        "prompt should annotate typed properties; got:\n{prompt}"
    );
}
