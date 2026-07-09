//! End-to-end pipeline test using [`MockClient`] in place of Memgraph.

use std::collections::BTreeMap;
use std::sync::Arc;

use linguagraph::config::{Config, DatabaseConfig, LlmConfig, OntologyCatalogConfig, QueryConfig};
use linguagraph::core::Pipeline;
use linguagraph::db::{MockClient, QueryResult, Row, Value};
use linguagraph::dsl;
use linguagraph::graph::{GraphBuilder, PropertyType};
use linguagraph::prompt::{GraphSchema, RelKind};
use linguagraph::types::handlers::{SemanticTextConfig, SemanticTextHandler};
use linguagraph::types::{handlers, RegistryBuilder, SharedRegistry};

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
async fn run_corrects_relationship_direction_from_live_schema() {
    let mock = Arc::new(MockClient::new());
    mock.set_schema(GraphSchema {
        nodes: Vec::new(),
        relationships: vec![RelKind {
            label: "WORKS_AT".into(),
            domain: None,
            description: None,
            from: Some("Company".into()),
            to: Some("Person".into()),
            properties: Vec::new(),
        }],
    });
    mock.enqueue(QueryResult::default());

    let cfg = test_config();
    let pipeline = Pipeline::new(mock.clone(), &cfg);
    let dsl = dsl::parse_str(
        r#"{
            "action": "find",
            "start": { "label": "Person", "alias": "p" },
            "traversals": [
                {
                    "edge": { "label": "WORKS_AT", "alias": "w", "direction": "out" },
                    "target": { "label": "Company", "alias": "c" }
                }
            ],
            "return": [{ "field": "c.name" }]
        }"#,
    )
    .unwrap();

    let _ = pipeline.run(dsl).await.unwrap();

    let captured = mock.captured.lock().unwrap();
    assert_eq!(captured.len(), 1);
    assert!(
        captured[0]
            .text
            .contains("MATCH (p:Person)<-[w:WORKS_AT]-(c:Company)"),
        "relationship direction should be corrected from live schema; got: {}",
        captured[0].text
    );
}

#[tokio::test]
async fn prefix_label_scopes_inserts_and_queries() {
    // Same Pipeline configuration drives both ingest and read: the
    // prefix label must appear in every MERGE pattern (nodes + edges)
    // and in every MATCH pattern produced from a plain DSL document.
    use linguagraph::embeddings::MockEmbedder;
    use linguagraph::types::handlers::{SemanticTextConfig, SemanticTextHandler};
    use linguagraph::types::{handlers, RegistryBuilder, SharedRegistry};

    // Every user entity carries an embedded `_canonical` document, so the
    // ingest path needs a registered SemanticText handler even when the
    // node itself has no free-text property.
    let registry: SharedRegistry = std::sync::Arc::new(
        handlers::register_core(RegistryBuilder::new())
            .register(SemanticTextHandler::new(
                SemanticTextConfig {
                    embedding_model: None,
                    collection: "docs".into(),
                    top_k: 10,
                    search_threshold: 0.1,
                    reranker_threshold: 0.2,
                    chunk_multivector: false,
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
        .with_prefix_label(Some("Tenant1"));

    // Ingest path.
    let mut g = GraphBuilder::new();
    let a = g
        .entity("Person")
        .strict_primary_key("id")
        .property("id", PropertyType::Keyword, "a")
        .add();
    let b = g
        .entity("Person")
        .strict_primary_key("id")
        .property("id", PropertyType::Keyword, "b")
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
    // embedding side effects (qlink.insert_hybrid) and the read-side
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
                    chunk_multivector: false,
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
        .property("id", PropertyType::Keyword, "a")
        .property("name", PropertyType::Text, "Alice")
        .add();
    pipeline.ingest(&g.build()).await.unwrap();

    {
        let captured = mock.captured.lock().unwrap();
        // The last batch is the qlink insert; its `coll` parameter must
        // be the prefixed collection name.
        let insert = captured
            .iter()
            .find(|c| c.text.contains("libqlink.insert_hybrid"))
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
    let has_prefixed_coll = last.params.values().any(
        |v| matches!(v, linguagraph::ast::query::Literal::String(s) if s.starts_with("Tenant1__")),
    );
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
        captured[0]
            .text
            .starts_with("MATCH (p:Person:OverrideTenant)"),
        "DSL prefix_label should override pipeline default; got: {}",
        captured[0].text
    );
}

#[tokio::test]
async fn soft_merge_rewrites_primary_key_to_existing_canonical() {
    // Knowledge-extraction payloads omit `primary_key`; the JSON
    // builder synthesises `_canonical` from the entity's properties
    // and defaults to `Soft`. `Pipeline::ingest` then
    // runs the soft-merge resolver against the existing graph
    // before issuing the MERGE. We seed the mock client with a
    // hits-list response that pretends Qdrant + Memgraph found a
    // near-duplicate and assert the subsequent MERGE keys off the
    // canonical value, not the incoming variant.
    use linguagraph::embeddings::MockEmbedder;
    use linguagraph::graph::GraphBuilder;

    let registry: SharedRegistry = std::sync::Arc::new(
        handlers::register_core(RegistryBuilder::new())
            .register(SemanticTextHandler::new(
                SemanticTextConfig {
                    embedding_model: None,
                    collection: "docs".into(),
                    top_k: 10,
                    search_threshold: 0.1,
                    reranker_threshold: 0.2,
                    chunk_multivector: false,
                },
                std::sync::Arc::new(MockEmbedder::new(8)),
            ))
            .build(),
    );

    let mock = Arc::new(MockClient::new());
    let cfg = test_config();
    let pipeline = Pipeline::new(mock.clone(), &cfg)
        .with_embedder(std::sync::Arc::new(MockEmbedder::new(8)))
        .with_registry(registry);

    // MockClient pops responses LIFO. Queue the MERGE responses first
    // (any empty result is fine for execution) and the resolver
    // response *last* so it's popped first — when the resolver
    // runs, it pulls a per-row `hits` list. The new staged resolver
    // returns top-K hits per row (each carrying score, canonical
    // value and the matched node's properties); a high score with
    // strong lexical and no competing candidates → AutoMerge.
    mock.enqueue(QueryResult::default()); // MERGE batch result
    let mut canonical_row = Row::default();
    canonical_row.fields.insert("idx".into(), Value::Int(0));
    canonical_row.fields.insert(
        "hits".into(),
        Value::Json(serde_json::json!([{
            "id": 7, "score": 0.99,
            "canonical": "общественное согласие",
            "props": {"name": "общественное согласие"}
        }])),
    );
    mock.enqueue(QueryResult {
        columns: vec!["idx".into(), "hits".into()],
        rows: vec![canonical_row],
    });

    // Use a raw `name` string so type inference picks
    // `PropertyType::Keyword` and the test doesn't depend on a
    // registered SemanticText handler. Soft-merge is orthogonal to
    // SemanticText: the resolver embeds the property text itself
    // and only consults Qdrant for the lookup; the on-node value
    // can be a plain string. Pin the soft key to `name` so the
    // resolver rewrites the visible name (rather than the
    // synthesised `_canonical`) — that way the MERGE-rows assertion
    // can check the name directly.
    let graph = GraphBuilder::from_json(
        r#"{
            "entities": [
                {
                    "id": "e1",
                    "type": "LegalConcept",
                    "name": "общественное соглас.",
                    "primary_key": {"soft": "name"}
                }
            ],
            "relations": []
        }"#,
    )
    .unwrap();

    pipeline.ingest(&graph).await.unwrap();

    let captured = mock.captured.lock().unwrap();
    // Resolver round-trip first, then the standard MERGE batch.
    assert!(
        captured[0].text.contains("libqlink.search_labeled"),
        "first Cypher must be the soft-merge resolver search; got: {}",
        captured[0].text
    );
    let merge = captured
        .iter()
        .find(|c| c.text.contains("MERGE (n:LegalConcept"))
        .expect("expected a MERGE batch against LegalConcept");
    // The MERGE rows must carry the canonical hit as their id. The
    // original extracted `name` is preserved as a normal property; only
    // the `_canonical` merge key is rewritten.
    let rows = merge
        .params
        .get("rows")
        .expect("MERGE batch must bind a 'rows' param");
    let linguagraph::ast::query::Literal::List(items) = rows else {
        panic!("MERGE rows should be a list, got {rows:?}");
    };
    let linguagraph::ast::query::Literal::Object(row) = &items[0] else {
        panic!("MERGE row should be an object, got {:?}", items[0]);
    };
    assert_eq!(
        row.get("id"),
        Some(&linguagraph::ast::query::Literal::String(
            "общественное согласие".into()
        ))
    );
}

#[tokio::test]
async fn soft_merge_without_embedder_errors_loudly() {
    // PrimaryKey::Soft without a configured embedder is treated as a
    // misconfiguration — the resolver would silently regress to
    // exact-string MERGE otherwise, which is exactly what soft-merge
    // is supposed to avoid.
    use linguagraph::graph::GraphBuilder;

    let mock = Arc::new(MockClient::new());
    let cfg = test_config();
    let pipeline = Pipeline::new(mock.clone(), &cfg);

    let graph = GraphBuilder::from_json(
        r#"{
            "entities": [{"id": "e1", "type": "LegalConcept", "name": "x"}],
            "relations": []
        }"#,
    )
    .unwrap();

    let err = pipeline
        .ingest(&graph)
        .await
        .expect_err("must error without embedder");
    let msg = err.to_string();
    assert!(
        msg.contains("soft-merge resolver requires an embedder"),
        "expected SoftMergeBackendUnavailable, got: {msg}"
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

#[tokio::test]
async fn entity_type_search_returns_unique_types_with_domains_and_scopes() {
    use linguagraph::core::EntityTypeSearchQuery;
    use linguagraph::embeddings::{MockEmbedder, SharedEmbedder};
    use linguagraph::graph::{
        DomainOntology, EntityTypeSpec, OntologyCatalog, OntologyPropertyType, PropertySpec,
    };
    use std::sync::Arc as StdArc;

    let mut catalog = OntologyCatalog::default();
    catalog.insert(
        "legal",
        DomainOntology {
            name: None,
            description: None,
            entity_types: vec![
                EntityTypeSpec {
                    name: "Person".into(),
                    description: Some("a legal person".into()),
                    properties: vec![PropertySpec {
                        name: "bio".into(),
                        description: None,
                        property_type: OntologyPropertyType::Text,
                        required: false,
                        allowed_values: Vec::new(),
                    }],
                    embedding: None,
                },
                EntityTypeSpec::with_description("Company", "a legal entity"),
            ],
            relation_types: vec![],
            embedding: None,
        },
    );

    let mock = Arc::new(MockClient::new());
    // Mock returns LIFO. Enqueue the neighbour leg first (popped last),
    // then the multi-collection vector leg (popped first).
    //
    // Neighbour leg returns one Company node tagged scope_structured.
    mock.enqueue(QueryResult {
        columns: vec!["nid".into(), "labs".into()],
        rows: vec![Row {
            fields: BTreeMap::from([
                ("nid".to_string(), Value::Json(serde_json::json!(99))),
                (
                    "labs".to_string(),
                    Value::Json(serde_json::json!(["Company", "legal", "scope_structured"])),
                ),
            ]),
        }],
    });
    // Vector leg: one Person hit from semantic_text___canonical with a
    // high score and scope_text.
    mock.enqueue(QueryResult {
        columns: vec!["nid".into(), "labs".into(), "score".into(), "coll".into()],
        rows: vec![Row {
            fields: BTreeMap::from([
                ("nid".to_string(), Value::Json(serde_json::json!(7))),
                (
                    "labs".to_string(),
                    Value::Json(serde_json::json!(["Person", "legal", "scope_text"])),
                ),
                ("score".to_string(), Value::Json(serde_json::json!(0.82))),
                (
                    "coll".to_string(),
                    Value::Json(serde_json::json!("semantic_text___canonical")),
                ),
            ]),
        }],
    });

    let cfg = test_config();
    let embedder: SharedEmbedder = StdArc::new(MockEmbedder::new(8));
    let pipeline = Pipeline::new(mock.clone(), &cfg)
        .with_embedder(embedder)
        .with_ontology_catalog(StdArc::new(catalog));

    let mut query = EntityTypeSearchQuery::new("who founded ACME?");
    query.include_neighbors = true;
    let result = pipeline
        .run_entity_type_search(query)
        .await
        .expect("entity-type search runs");

    assert_eq!(result.matches.len(), 1, "{:?}", result.matches);
    let person = &result.matches[0];
    assert_eq!(person.entity_type, "Person");
    assert_eq!(person.domain.as_deref(), Some("legal"));
    assert!(person
        .scopes
        .iter()
        .any(|s| { matches!(s, linguagraph::graph::Scope::Text) }));
    assert_eq!(
        person.per_collection.get("semantic_text___canonical"),
        Some(&0.82_f32)
    );
    assert_eq!(person.sample_node_ids, vec![7]);

    assert_eq!(result.neighbors.len(), 1);
    let company = &result.neighbors[0];
    assert_eq!(company.entity_type, "Company");
    assert_eq!(company.domain.as_deref(), Some("legal"));
    assert!(company
        .scopes
        .iter()
        .any(|s| { matches!(s, linguagraph::graph::Scope::Structured) }));
    // Neighbours carry no vector signal.
    assert!(company.vector_score.is_none());
    assert!(company.per_collection.is_empty());

    // The vector leg searches just the two collections that back every
    // node: `_canonical` (whole-entity documents) and `text` (chunks).
    // Per-field fan-out is gone — `_canonical` already covers every Text
    // property, so `…__name` / `…__bio` are no longer separate indexes.
    assert_eq!(result.collections_searched.len(), 2);
    assert!(result
        .collections_searched
        .iter()
        .any(|c| c == "semantic_text___canonical"));
    assert!(result
        .collections_searched
        .iter()
        .any(|c| c == "semantic_text__text"));

    // Cypher inspection: the vector leg should be one UNION ALL'd
    // batch of libqlink.search calls plus the neighbour leg. (Label
    // filtering moved into the Cypher MATCH, so the leg calls the plain
    // `libqlink.search`, not `search_labeled`.)
    let captured = mock.captured.lock().unwrap();
    assert_eq!(captured.len(), 2);
    let vector_cypher = &captured[0].text;
    assert!(
        vector_cypher.contains("libqlink.search("),
        "vector cypher missing qlink call:\n{vector_cypher}"
    );
    assert!(vector_cypher.contains("UNION ALL"));
    // Each UNION ALL branch must be a plain top-level query (CALL +
    // MATCH + RETURN), NOT wrapped in `CALL { ... }`. Memgraph rejects
    // a top-level query that consists solely of CALL subqueries with
    // an internal RETURN.
    assert!(
        !vector_cypher.contains("CALL {"),
        "vector cypher must not wrap branches in CALL {{ ... }} — \
         Memgraph rejects that shape:\n{vector_cypher}"
    );
    // Every branch ends with its own RETURN so the union has a
    // consistent column projection.
    let branches = vector_cypher.split("UNION ALL").count();
    let returns = vector_cypher.matches("RETURN").count();
    assert_eq!(
        branches, returns,
        "expected one RETURN per UNION ALL branch:\n{vector_cypher}"
    );
    let neighbour_cypher = &captured[1].text;
    assert!(
        neighbour_cypher.contains("MATCH (n)-[]-(m)"),
        "neighbour cypher missing pattern:\n{neighbour_cypher}"
    );
}

#[tokio::test]
async fn entity_type_search_requires_an_embedder() {
    use linguagraph::core::EntityTypeSearchQuery;

    let mock = Arc::new(MockClient::new());
    let cfg = test_config();
    let pipeline = Pipeline::new(mock.clone(), &cfg);

    let err = pipeline
        .run_entity_type_search(EntityTypeSearchQuery::new("anything"))
        .await
        .expect_err("missing embedder must fail");
    let msg = format!("{err}");
    assert!(
        msg.contains("embedder"),
        "expected embedder-missing error, got {msg}"
    );
}

#[tokio::test]
async fn entity_type_search_empty_text_short_circuits() {
    use linguagraph::core::EntityTypeSearchQuery;
    use linguagraph::embeddings::{MockEmbedder, SharedEmbedder};
    use std::sync::Arc as StdArc;

    let mock = Arc::new(MockClient::new());
    let cfg = test_config();
    let embedder: SharedEmbedder = StdArc::new(MockEmbedder::new(8));
    let pipeline = Pipeline::new(mock.clone(), &cfg).with_embedder(embedder);

    let result = pipeline
        .run_entity_type_search(EntityTypeSearchQuery::new("   "))
        .await
        .expect("empty text is not an error");
    assert!(result.matches.is_empty());
    assert!(result.neighbors.is_empty());
    assert!(result.collections_searched.is_empty());
    let captured = mock.captured.lock().unwrap();
    assert!(captured.is_empty(), "no DB call should have been made");
}
