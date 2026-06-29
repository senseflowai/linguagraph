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
//!   `"type": "SemanticText"` and a semantic op (`search` /
//!   `search_reranked` / `hybrid_search`) lower into one per-entity
//!   hybrid search via `qlink.search_hybrid_reranked` over `_canonical`.
//! * **Exact query compilation** — `eq` / `neq` / `contains` on a
//!   SemanticText field lower into plain Cypher (`=` / `<>` /
//!   `CONTAINS`) against the raw value, never a vector search.

use std::collections::BTreeMap;
use std::sync::Arc;

use linguagraph::ast::query::Literal;
use linguagraph::config::{
    Config, DatabaseConfig, LlmConfig, OntologyCatalogConfig, QueryConfig, TypeConfig,
};
use linguagraph::core::Pipeline;
use linguagraph::db::MockClient;
use linguagraph::dsl;
use linguagraph::embeddings::{MockEmbedder, SharedEmbedder};
use linguagraph::graph::{
    DomainOntology, EntityTypeSpec, GraphBuilder, JsonFileOntologyCatalogStorage, OntologyCatalog,
    OntologyCatalogStorage, OntologyPropertyType, PropertySpec, PropertyType as GraphPropertyType,
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
        ontology_catalog: OntologyCatalogConfig::default(),
        prompt: Default::default(),
        ingest: Default::default(),
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
        .property("id", GraphPropertyType::Keyword, "c1")
        .property("name", GraphPropertyType::Text, "Apple Inc.")
        .add();
    graph
        .entity("Company")
        .strict_primary_key("id")
        .property("id", GraphPropertyType::Keyword, "c2")
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
    // [0] = Company MERGE, [1] = one batched libqlink.insert_hybrid.
    assert_eq!(captured.len(), 2);
    let qlink_batch = &captured[1];
    assert!(
        qlink_batch
            .text
            .contains("CALL libqlink.insert_hybrid($coll, id(n), row.vec, row.text, $label)"),
        "expected UNWIND-batched libqlink.insert_hybrid; got:\n{}",
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
    // single bucket has a single $coll / $label binding. Only the
    // per-entity `_canonical` document is embedded (the `name` value
    // already lives inside it), so the collection is `companies___canonical`.
    assert_eq!(
        qlink_batch.params.get("coll"),
        Some(&Literal::String("companies___canonical".into())),
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
        .property("id", GraphPropertyType::Keyword, "c1")
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

    // A SemanticText `search_reranked` is a semantic op: the resolver
    // consolidates it into a single per-entity hybrid search that emits
    // `libqlink.search_hybrid_reranked` against the `_canonical`
    // collection and binds the reranker threshold. (Plain `search` folds
    // the same way; exact `eq`/`neq`/`contains` route to plain Cypher.)
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

    // Prelude must come before MATCH and call libqlink.search_hybrid_reranked.
    let lines: Vec<&str> = cypher.text.lines().collect();
    let qlink_idx = lines
        .iter()
        .position(|l| l.contains("libqlink.search_hybrid_reranked"))
        .expect("expected libqlink.search_hybrid_reranked in cypher");
    let match_idx = lines
        .iter()
        .position(|l| l.starts_with("MATCH"))
        .expect("expected MATCH");
    assert!(
        qlink_idx < match_idx,
        "libqlink.search_hybrid_reranked prelude must run before the MATCH; got:\n{}",
        cypher.text
    );

    // ORDER BY surfaces the reranker score so closer hits come first.
    assert!(
        cypher.text.contains("ORDER BY") && cypher.text.contains("c__score_0 DESC"),
        "expected ORDER BY c__score_<n> DESC; got:\n{}",
        cypher.text
    );
    // qlink filters internally — no extra `WHERE c__score >=` slip-in.
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

    // The consolidated query string is bound for the reranker. It mirrors
    // the `_canonical` document format — `type: <Label>` then `prop: value`
    // lines — so the cross-encoder sees query and documents in the same
    // shape. For a single `c.name = apple` term that's
    // "type: Company\nname: apple".
    let has_query_str = cypher
        .params
        .values()
        .any(|v| matches!(v, Literal::String(s) if s == "type: Company\nname: apple"));
    assert!(
        has_query_str,
        "expected the canonical-mirrored query string bound as query_str; params: {:?}",
        cypher.params
    );

    // The search targets the per-entity `_canonical` collection, not the
    // per-field one.
    let has_canonical_collection = cypher
        .params
        .values()
        .any(|v| matches!(v, Literal::String(s) if s == "companies___canonical"));
    assert!(
        has_canonical_collection,
        "expected the `companies___canonical` collection bound; params: {:?}",
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
    // param. The cosine `search_threshold` is not handed to
    // libqlink.search_hybrid_reranked (its arg list takes candidate_k,
    // not a cosine cutoff), so it stays in the predicate's internal
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

/// Consolidation: when a query carries several `SemanticText` semantic
/// filters on the *same* alias, they collapse into a single per-entity
/// hybrid search over `_canonical` — one `search_hybrid_reranked` call,
/// one cross-encoder pass — instead of one call per field. The combined
/// query string carries every term, field-agnostic, so the DSL model
/// attributing a value to the "wrong" field no longer drops the hit.
#[test]
fn multiple_semantic_filters_on_same_alias_consolidate_into_one_call() {
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
                  "op": "search", "value": "Mangilik 55" },
                { "field": "c.name", "type": "SemanticText",
                  "op": "search", "value": "TargetAI" }
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

    // Exactly ONE consolidated call, yielding into a single pair of vars.
    assert_eq!(
        cypher.text.matches("CALL libqlink.search_hybrid_reranked").count(),
        1,
        "the two same-alias filters must collapse into one call; got:\n{}",
        cypher.text
    );
    assert_eq!(
        cypher.text.matches("AS c__qid_").count(),
        1,
        "expected a single c__qid_<n> yield; got:\n{}",
        cypher.text
    );

    // The combined query string mirrors `_canonical`: `type: Camera` then
    // the two `prop: value` lines in sorted-key order (name before
    // origin_place_description). Both values survive — that's the
    // field-agnostic win.
    let expected_query = "type: Camera\nname: TargetAI\norigin_place_description: Mangilik 55";
    assert!(
        cypher
            .params
            .values()
            .any(|v| matches!(v, Literal::String(s) if s == expected_query)),
        "expected combined canonical query string; params: {:?}",
        cypher.params
    );

    // Joined by id and ordered by the single reranker score.
    assert!(cypher.text.contains("id(c) = c__qid_0"));
    assert!(cypher.text.contains("c__score_0 DESC"));
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
    // (`search_reranked`), traverse to Camera, count.
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

    // The `libqlink.search_hybrid_reranked` prelude is still emitted; only
    // the score-based ORDER BY is suppressed for aggregates.
    assert!(cypher.text.contains("CALL libqlink.search_hybrid_reranked"));
    assert!(
        !cypher.text.contains("ORDER BY p__score"),
        "aggregate queries must not order by an unprojected score column; \
         got:\n{}",
        cypher.text
    );
    assert!(cypher.text.contains("RETURN count(c) AS camera_count"));
}

#[test]
fn hybrid_search_routes_to_canonical_hybrid_retrieval() {
    // `hybrid_search` is now an alias for the unified semantic path: it
    // compiles to one `search_hybrid_reranked` over `_canonical` (dense ⊕
    // BM25 + rerank), not the old inline exact + `score_batch_node`
    // scheme. The keyword signal now comes from BM25 inside qlink rather
    // than a separate Cypher equality column.
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
        cypher.text.contains("CALL libqlink.search_hybrid_reranked"),
        "hybrid_search should compile to the canonical hybrid call; got:\n{}",
        cypher.text
    );
    assert!(
        !cypher.text.contains("score_batch_node"),
        "hybrid_search must no longer emit score_batch_node; got:\n{}",
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

// ─── Auto-resolution from OntologyCatalog ───────────────────────────────
//
// When the DSL omits `"type"` but the ontology catalog declares one,
// the lowering step should pick up the type from the catalog snapshot
// and route the filter through the matching handler.

fn prop(name: &str, property_type: OntologyPropertyType, description: Option<&str>) -> PropertySpec {
    PropertySpec {
        name: name.into(),
        description: description.map(str::to_string),
        property_type,
        required: false,
    }
}

#[test]
fn ontology_catalog_round_trips_field_types() {
    let catalog = semantic_catalog();
    assert_eq!(
        catalog.get_query_type("Company", "name"),
        Some("SemanticText")
    );
    assert_eq!(
        catalog
            .get_property("Company", "name")
            .and_then(|p| p.description.as_deref()),
        Some("the company name")
    );
    assert_eq!(catalog.get_query_type("Company", "industry"), Some("Keyword"));
}

fn semantic_catalog() -> OntologyCatalog {
    let mut catalog = OntologyCatalog::default();
    catalog.insert(
        "test",
        DomainOntology {
            entity_types: vec![EntityTypeSpec {
                name: "Company".into(),
                description: None,
                properties: vec![
                    prop("id", OntologyPropertyType::Keyword, None),
                    prop("name", OntologyPropertyType::Text, Some("the company name")),
                    prop("industry", OntologyPropertyType::Keyword, None),
                ],
                embedding: None,
            }],
            relation_types: vec![],
        },
    );
    catalog
}

#[test]
fn untyped_dsl_filter_auto_resolves_to_semantic_text_via_ontology_catalog() {
    let cfg = cfg_with_semantic_text();
    let (registry, embedder) = registry_and_embedder();
    let catalog = Arc::new(semantic_catalog());
    let pipeline = Pipeline::new(Arc::new(MockClient::new()), &cfg)
        .with_registry(registry)
        .with_embedder(embedder)
        .with_ontology_catalog(catalog);

    // Notice the DSL has NO `"type"` field — the handler is selected
    // from OntologyCatalog.
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
    let mut catalog = OntologyCatalog::default();
    catalog.insert(
        "test",
        DomainOntology {
            entity_types: vec![EntityTypeSpec {
                name: "ServiceVisit".into(),
                description: None,
                properties: vec![
                    prop("id", OntologyPropertyType::Keyword, None),
                    prop(
                        "work_start",
                        OntologyPropertyType::Datetime,
                        Some("when the visit started"),
                    ),
                ],
                embedding: None,
            }],
            relation_types: vec![],
        },
    );
    let pipeline = Pipeline::new(Arc::new(MockClient::new()), &cfg)
        .with_registry(registry)
        .with_embedder(embedder)
        .with_ontology_catalog(Arc::new(catalog));

    // No `"type"` field — the Timestamp handler is selected from the
    // ontology catalog.
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
fn explicit_dsl_type_overrides_ontology_catalog() {
    // The mapping doesn't tag `c.industry` with any type, but the DSL
    // does — explicit always wins over the inferred specification value.
    // Conversely, when an explicit type *is* set we must not silently
    // fall back to the catalog's type for the same field.
    let cfg = cfg_with_semantic_text();
    let (registry, embedder) = registry_and_embedder();
    let catalog = semantic_catalog();
    let pipeline = Pipeline::new(Arc::new(MockClient::new()), &cfg)
        .with_registry(registry)
        .with_embedder(embedder)
        .with_ontology_catalog(Arc::new(catalog));

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
    // Compiles cleanly via the explicit SemanticText handler. `eq` is an
    // exact op, so it lowers to a plain `c.industry = $p0` WHERE clause
    // against the raw value — precise, never a fuzzy vector search.
    let cypher = pipeline.compile(dsl_query).unwrap();
    assert!(
        cypher.text.contains("WHERE c.industry = $p0"),
        "explicit SemanticText `eq` should be an exact WHERE clause; got:\n{}",
        cypher.text
    );
    assert!(
        !cypher.text.contains("qlink"),
        "exact SemanticText `eq` must not touch qlink; got:\n{}",
        cypher.text
    );
}

#[test]
fn untyped_field_without_ontology_catalog_stays_plain() {
    let cfg = cfg_with_semantic_text();
    let (registry, embedder) = registry_and_embedder();
    let pipeline = Pipeline::new(Arc::new(MockClient::new()), &cfg)
        .with_registry(registry)
        .with_embedder(embedder);

    // Without a loaded ontology catalog, industry should compile as a
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
fn keyword_property_from_ontology_catalog_auto_resolves_to_keyword_handler() {
    let cfg = cfg_with_semantic_text();
    let (registry, embedder) = registry_and_embedder();
    let catalog = Arc::new(semantic_catalog());
    let pipeline = Pipeline::new(Arc::new(MockClient::new()), &cfg)
        .with_registry(registry)
        .with_embedder(embedder)
        .with_ontology_catalog(catalog);

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
    // Keyword stores/compares the value verbatim — no normalization.
    assert_eq!(
        cypher.params.get("p0"),
        Some(&Literal::String(" Fin-Tech ".into()))
    );
    assert!(!cypher.text.contains("qlink"));
}

#[test]
fn ontology_catalog_lookup_keys_off_label_not_alias() {
    // Same property name on different labels must resolve independently.
    // Here `c` is bound to `Company` and `p` to `Person`. Only
    // `Company.name` is SemanticText.
    let cfg = cfg_with_semantic_text();
    let (registry, embedder) = registry_and_embedder();
    let mut catalog = OntologyCatalog::default();
    catalog.insert(
        "test",
        DomainOntology {
            entity_types: vec![EntityTypeSpec {
                name: "Company".into(),
                description: None,
                properties: vec![prop(
                    "name",
                    OntologyPropertyType::Text,
                    Some("the company name"),
                )],
                embedding: None,
            }],
            relation_types: vec![],
        },
    );
    // Person.name is left plain.
    let pipeline = Pipeline::new(Arc::new(MockClient::new()), &cfg)
        .with_registry(registry)
        .with_embedder(embedder)
        .with_ontology_catalog(Arc::new(catalog));

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
async fn loaded_ontology_catalog_auto_resolves_semantic_text_filters() {
    let cfg = cfg_with_semantic_text();
    let (registry, embedder) = registry_and_embedder();
    let path =
        std::env::temp_dir().join(format!("linguagraph-catalog-test-{}.json", std::process::id()));
    let storage = JsonFileOntologyCatalogStorage::new(&path);
    storage.save(&semantic_catalog()).await.unwrap();
    let storage: Arc<dyn OntologyCatalogStorage> = Arc::new(storage);
    let pipeline = Pipeline::new(Arc::new(MockClient::new()), &cfg)
        .with_registry(registry)
        .with_embedder(embedder)
        .with_ontology_catalog_storage(storage);
    pipeline.load_ontology_catalog().await.unwrap();

    let catalog = pipeline.ontology_catalog().expect("snapshot loaded");
    assert_eq!(
        catalog.get_query_type("Company", "name"),
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
    use linguagraph::prompt::{
        generate_system_prompt, GraphSchema, NodeKind, PromptOptions, Property, PropertyType,
    };
    let schema = GraphSchema {
        nodes: vec![NodeKind {
            label: "Company".into(),
            domain: None,
            extra_labels: Vec::new(),
            scopes: Vec::new(),
            description: None,
            properties: vec![
                Property {
                    name: "id".into(),
                    ty: PropertyType::String,
                    description: None,
                },
                Property {
                    name: "name".into(),
                    ty: PropertyType::String,
                    description: None,
                },
            ],
        }],
        relationships: vec![],
    };
    let catalog = semantic_catalog();
    let prompt = generate_system_prompt(
        &schema,
        &PromptOptions {
            ontology_catalog: Some(catalog),
            include_examples: false,
            ..Default::default()
        },
    );
    assert!(
        prompt.contains("name: keyword @Text /* the company name */"),
        "prompt should annotate typed properties; got:\n{prompt}"
    );
}
