//! Explorer integration tests using [`MockClient`] in place of Memgraph.
//!
//! The mock pops enqueued results LIFO — multi-query tests enqueue in
//! **reverse call order** — and captures every executed Cypher query, so
//! each test asserts both the decoded DTOs and the generated query text.

use std::collections::BTreeMap;
use std::sync::Arc;

use linguagraph::config::{Config, DatabaseConfig, LlmConfig, OntologyCatalogConfig, QueryConfig};
use linguagraph::core::Pipeline;
use linguagraph::db::{MockClient, QueryResult, Row, Value};
use linguagraph::explore::{
    AskOptions, EntityCard, Explorer, NeighborOptions, PageOptions, RelDirection, SearchOptions,
};
use linguagraph::graph::{
    DomainOntology, EntityTypeSpec, OntologyCatalog, OntologyPropertyType, PropertySpec,
};
use linguagraph::prompt::{GraphSchema, NodeKind, Property, PropertyType};
use serde_json::json;

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
            max_limit: 5000,
            grounding: Default::default(),
        },
        ontology_catalog: OntologyCatalogConfig::default(),
        qdrant: Default::default(),
        prompt: Default::default(),
        ingest: Default::default(),
        types: Default::default(),
    }
}

fn movie_catalog() -> Arc<OntologyCatalog> {
    let property = |name: &str, ty: OntologyPropertyType| PropertySpec {
        name: name.to_string(),
        description: None,
        property_type: ty,
        required: false,
        allowed_values: Vec::new(),
    };
    let mut catalog = OntologyCatalog::default();
    catalog.insert(
        "movies",
        DomainOntology {
            name: None,
            description: None,
            entity_types: vec![
                EntityTypeSpec {
                    name: "Movie".to_string(),
                    description: None,
                    properties: vec![
                        property("id", OntologyPropertyType::Keyword),
                        property("title", OntologyPropertyType::Keyword),
                        property("tagline", OntologyPropertyType::Text),
                        property("released", OntologyPropertyType::Datetime),
                        property("votes", OntologyPropertyType::Number),
                    ],
                },
                EntityTypeSpec {
                    name: "Person".to_string(),
                    description: None,
                    properties: vec![
                        property("id", OntologyPropertyType::Keyword),
                        property("name", OntologyPropertyType::Keyword),
                    ],
                },
            ],
            relation_types: Vec::new(),
        },
    );
    Arc::new(catalog)
}

fn explorer_with(mock: Arc<MockClient>, prefix: Option<&str>) -> Explorer {
    let cfg = test_config();
    let pipeline = Pipeline::new(mock, &cfg)
        .with_prefix_label(prefix)
        .with_ontology_catalog(movie_catalog());
    Explorer::new(pipeline)
}

fn row(fields: Vec<(&str, Value)>) -> Row {
    let mut row = Row::default();
    for (name, value) in fields {
        row.fields.insert(name.to_string(), value);
    }
    row
}

fn result(columns: Vec<&str>, rows: Vec<Row>) -> QueryResult {
    QueryResult {
        columns: columns.into_iter().map(Into::into).collect(),
        rows,
    }
}

fn matrix_node_row() -> Row {
    row(vec![
        ("nid", Value::Int(7)),
        ("id", Value::String("m1".into())),
        ("labels", Value::Json(json!(["E2E", "movies", "Movie"]))),
        (
            "props",
            Value::Json(json!({
                "id": "m1",
                "title": "The Matrix",
                "tagline": "Welcome to the Real World",
                "released": "1999-03-31",
                "votes": 4500,
                "_canonical": "hidden system text"
            })),
        ),
        (
            "sources",
            Value::Json(json!([
                {"id": "s1", "name": "movies.json"},
                {"id": null, "name": null}
            ])),
        ),
    ])
}

async fn fetch_matrix_card(explorer: &Explorer, mock: &MockClient) -> EntityCard {
    let _ = mock;
    explorer
        .entity("m1")
        .await
        .expect("entity lookup runs")
        .expect("m1 exists")
}

#[tokio::test]
async fn entity_card_scopes_prefix_classifies_props_and_merges_relations() {
    let mock = Arc::new(MockClient::new());
    // LIFO: enqueue the relation summary (2nd call) before the node (1st).
    // Relation summary: two raw rows for the same reduced group merge.
    mock.enqueue(result(
        vec!["rel", "dir", "neighbor_labels", "cnt"],
        vec![
            row(vec![
                ("rel", Value::String("ACTED_IN".into())),
                ("dir", Value::String("in".into())),
                ("neighbor_labels", Value::Json(json!(["E2E", "Person"]))),
                ("cnt", Value::Int(4)),
            ]),
            row(vec![
                ("rel", Value::String("ACTED_IN".into())),
                ("dir", Value::String("in".into())),
                (
                    "neighbor_labels",
                    Value::Json(json!(["E2E", "movies", "Person"])),
                ),
                ("cnt", Value::Int(1)),
            ]),
        ],
    ));
    mock.enqueue(result(
        vec!["nid", "id", "labels", "props", "sources"],
        vec![matrix_node_row()],
    ));

    let explorer = explorer_with(mock.clone(), Some("E2E"));
    let card = fetch_matrix_card(&explorer, &mock).await;

    // Node identity + classification.
    assert_eq!(card.node.id, "m1");
    assert_eq!(card.node.name, "The Matrix");
    assert_eq!(card.node.entity_type, "Movie");
    assert!(!card.node.ephemeral_handle);
    assert_eq!(
        card.node.properties.identifiers.get("title"),
        Some(&json!("The Matrix"))
    );
    assert!(card.node.properties.descriptions.contains_key("tagline"));
    assert!(card.node.properties.dates.contains_key("released"));
    assert_eq!(card.node.properties.facts.get("votes"), Some(&json!(4500)));
    assert!(
        card.node.properties.iter().all(|(k, _)| k != "_canonical"),
        "system fields must never surface"
    );

    // Provenance: the all-null OPTIONAL MATCH artifact is dropped.
    assert_eq!(card.sources.len(), 1);
    assert_eq!(card.sources[0].name.as_deref(), Some("movies.json"));

    // Relations: both raw rows reduce to Person and merge counts.
    assert_eq!(card.relations.len(), 1);
    let relation = &card.relations[0];
    assert_eq!(relation.edge_type, "ACTED_IN");
    assert_eq!(relation.direction, RelDirection::In);
    assert_eq!(relation.neighbor_type, "Person");
    assert_eq!(relation.count, 5);

    // Generated Cypher: prefix-scoped, id-parameterized, provenance join.
    let captured = mock.captured.lock().unwrap();
    assert_eq!(captured.len(), 2);
    assert!(captured[0].text.contains("MATCH (n:E2E)"));
    assert!(captured[0].text.contains("n.id = $id"));
    assert!(captured[0].text.contains(":mention|part_of"));
    assert!(captured[1]
        .text
        .contains("NOT type(r) IN [\"mention\", \"part_of\"]"));
}

#[tokio::test]
async fn entity_returns_none_for_missing_handle() {
    let mock = Arc::new(MockClient::new());
    mock.enqueue(QueryResult::default());
    let explorer = explorer_with(mock, Some("E2E"));
    assert!(explorer.entity("ghost").await.unwrap().is_none());
}

#[tokio::test]
async fn node_without_app_id_gets_ephemeral_handle() {
    let mock = Arc::new(MockClient::new());
    // LIFO: relation summary (2nd call) first, node lookup (1st) second.
    mock.enqueue(QueryResult::default());
    mock.enqueue(result(
        vec!["nid", "id", "labels", "props", "sources"],
        vec![row(vec![
            ("nid", Value::Int(42)),
            ("id", Value::Null),
            ("labels", Value::Json(json!(["Movie"]))),
            ("props", Value::Json(json!({"title": "Untitled"}))),
            ("sources", Value::Json(json!([]))),
        ])],
    ));

    let explorer = explorer_with(mock.clone(), None);
    let card = explorer.entity("_nid:42").await.unwrap().unwrap();
    assert_eq!(card.node.id, "_nid:42");
    assert!(card.node.ephemeral_handle);

    let captured = mock.captured.lock().unwrap();
    assert!(captured[0].text.contains("id(n) = $nid"));
}

#[tokio::test]
async fn neighbors_applies_filters_pagination_and_edge_direction() {
    let mock = Arc::new(MockClient::new());
    // LIFO: neighbor hop (2nd call) first, origin hydration (1st) second.
    // One incoming ACTED_IN neighbor.
    mock.enqueue(result(
        vec![
            "nid",
            "id",
            "labels",
            "props",
            "rel",
            "rel_props",
            "outgoing",
        ],
        vec![row(vec![
            ("nid", Value::Int(8)),
            ("id", Value::String("p1".into())),
            ("labels", Value::Json(json!(["E2E", "Person"]))),
            (
                "props",
                Value::Json(json!({"id": "p1", "name": "Keanu Reeves"})),
            ),
            ("rel", Value::String("ACTED_IN".into())),
            ("rel_props", Value::Json(json!({"roles": ["Neo"]}))),
            // The Memgraph client flattens bools to Json on the wire.
            ("outgoing", Value::Json(json!(false))),
        ])],
    ));
    // Origin hydration.
    mock.enqueue(result(
        vec!["nid", "id", "labels", "props", "sources"],
        vec![matrix_node_row()],
    ));

    let explorer = explorer_with(mock.clone(), Some("E2E"));
    let subgraph = explorer
        .neighbors(
            "m1",
            &NeighborOptions {
                edge_types: Some(vec!["ACTED_IN".into()]),
                target_labels: Some(vec!["Person".into()]),
                direction: None,
                limit: Some(10),
                offset: 20,
            },
        )
        .await
        .expect("neighbors run");

    // Origin + neighbor, edge oriented neighbor→origin (incoming).
    assert_eq!(subgraph.nodes.len(), 2);
    assert_eq!(subgraph.nodes[0].id, "m1");
    assert_eq!(subgraph.nodes[1].name, "Keanu Reeves");
    assert_eq!(subgraph.edges.len(), 1);
    let edge = &subgraph.edges[0];
    assert_eq!(edge.from, "p1");
    assert_eq!(edge.to, "m1");
    assert_eq!(edge.id, "p1:ACTED_IN:m1");
    assert_eq!(edge.properties.get("roles"), Some(&json!(["Neo"])));

    let captured = mock.captured.lock().unwrap();
    assert_eq!(captured.len(), 2);
    let neighbors_cypher = &captured[1].text;
    assert!(neighbors_cypher.contains("type(r) IN $edge_types"));
    assert!(neighbors_cypher.contains("any(l IN labels(m) WHERE l IN $target_labels)"));
    assert!(neighbors_cypher.contains("SKIP 20 LIMIT 10"));
}

#[tokio::test]
async fn neighbors_errors_on_unknown_origin() {
    let mock = Arc::new(MockClient::new());
    mock.enqueue(QueryResult::default());
    let explorer = explorer_with(mock, None);
    let err = explorer
        .neighbors("ghost", &NeighborOptions::default())
        .await
        .expect_err("missing origin must error");
    assert!(err.to_string().contains("unknown entity"), "got: {err}");
}

#[tokio::test]
async fn overview_reduces_label_sets_and_excludes_builtins() {
    let mock = Arc::new(MockClient::new());
    // LIFO: enqueue in reverse call order — sources, relation counts,
    // label-set counts.
    mock.enqueue(result(
        vec!["id", "name"],
        vec![row(vec![
            ("id", Value::String("s1".into())),
            ("name", Value::String("movies.json".into())),
        ])],
    ));
    // Relation counts (mention is plumbing, excluded).
    mock.enqueue(result(
        vec!["rel", "cnt"],
        vec![
            row(vec![
                ("rel", Value::String("ACTED_IN".into())),
                ("cnt", Value::Int(172)),
            ]),
            row(vec![
                ("rel", Value::String("mention".into())),
                ("cnt", Value::Int(143)),
            ]),
        ],
    ));
    // Label-set counts (reduced per business label client-side).
    mock.enqueue(result(
        vec!["labels", "cnt"],
        vec![
            row(vec![
                ("labels", Value::Json(json!(["E2E", "movies", "Movie"]))),
                ("cnt", Value::Int(38)),
            ]),
            row(vec![
                ("labels", Value::Json(json!(["E2E", "Person"]))),
                ("cnt", Value::Int(100)),
            ]),
            row(vec![
                ("labels", Value::Json(json!(["E2E", "Source"]))),
                ("cnt", Value::Int(1)),
            ]),
            row(vec![
                ("labels", Value::Json(json!(["E2E", "Chunk"]))),
                ("cnt", Value::Int(5)),
            ]),
        ],
    ));

    let explorer = explorer_with(mock.clone(), Some("E2E"));
    let overview = explorer.overview().await.expect("overview runs");

    assert_eq!(overview.total_entities, 138, "Source/Chunk excluded");
    assert_eq!(overview.entity_types.len(), 2);
    assert_eq!(overview.entity_types[0].name, "Person");
    assert_eq!(overview.entity_types[0].count, 100);
    assert_eq!(overview.entity_types[1].name, "Movie");
    assert_eq!(overview.relation_types.len(), 1);
    assert_eq!(overview.relation_types[0].name, "ACTED_IN");
    assert_eq!(overview.total_relations, 172);
    assert_eq!(overview.sources.len(), 1);

    let captured = mock.captured.lock().unwrap();
    assert!(captured.iter().all(|q| q.text.contains(":E2E")));
}

#[tokio::test]
async fn entities_of_type_pages_sorts_and_counts() {
    let mock = Arc::new(MockClient::new());
    // LIFO: count (2nd call) first, page (1st call) second.
    mock.enqueue(result(
        vec!["total"],
        vec![row(vec![("total", Value::Int(38))])],
    ));
    mock.enqueue(result(
        vec!["nid", "id", "labels", "props"],
        vec![matrix_node_row()],
    ));

    let explorer = explorer_with(mock.clone(), Some("E2E"));
    let table = explorer
        .entities_of_type(
            "Movie",
            &PageOptions {
                limit: Some(10),
                offset: 20,
                sort_by: Some("title".into()),
            },
        )
        .await
        .expect("table runs");

    assert_eq!(table.entity_type, "Movie");
    assert_eq!(table.total, 38);
    assert_eq!(table.offset, 20);
    assert_eq!(table.rows.len(), 1);
    assert!(table.key_columns.contains(&"id".to_string()));

    let captured = mock.captured.lock().unwrap();
    assert!(captured[0].text.contains("MATCH (n:Movie:E2E)"));
    assert!(captured[0].text.contains("ORDER BY n.title ASC"));
    assert!(captured[0].text.contains("SKIP 20 LIMIT 10"));
    assert!(captured[1].text.contains("count(n) AS total"));
}

#[tokio::test]
async fn keyword_search_scans_schema_string_props_only() {
    let mock = Arc::new(MockClient::new());
    mock.set_schema(GraphSchema {
        nodes: vec![NodeKind {
            label: "Movie".into(),
            domain: None,
            extra_labels: Vec::new(),
            scopes: Vec::new(),
            description: None,
            properties: vec![
                Property {
                    name: "tagline".into(),
                    ty: PropertyType::String,
                    description: None,
                    allowed_values: Vec::new(),
                },
                Property {
                    name: "roles".into(),
                    ty: PropertyType::List,
                    description: None,
                    allowed_values: Vec::new(),
                },
            ],
        }],
        relationships: Vec::new(),
    });
    mock.enqueue(result(
        vec!["nid", "id", "labels", "props"],
        vec![matrix_node_row()],
    ));

    let explorer = explorer_with(mock.clone(), Some("E2E"));
    let found = explorer
        .search("matrix", &SearchOptions::default())
        .await
        .expect("search runs");

    assert_eq!(found.hits.len(), 1);
    assert_eq!(found.hits[0].node.id, "m1");
    assert!(found.hits[0].score.is_none());

    let captured = mock.captured.lock().unwrap();
    let cypher = &captured[0].text;
    assert!(
        cypher.contains("n._lg_norm_tagline"),
        "string prop scanned: {cypher}"
    );
    assert!(
        !cypher.contains("n.roles"),
        "list prop must not be scanned (toString on lists errors): {cypher}"
    );
    assert!(cypher.contains("NOT n:Chunk"));
    assert!(
        cypher.contains("n._lg_norm_id")
            && cypher.contains("n._lg_norm_name")
            && cypher.contains("n._lg_norm_title")
    );
}

#[tokio::test]
async fn invalid_identifiers_are_rejected_not_interpolated() {
    let mock = Arc::new(MockClient::new());
    let explorer = explorer_with(mock.clone(), Some("E2E"));

    let err = explorer
        .entities_of_type("Movie) DETACH DELETE (n", &PageOptions::default())
        .await
        .expect_err("label injection rejected");
    assert!(err.to_string().contains("not a valid Cypher identifier"));

    let err = explorer
        .entities_of_type(
            "Movie",
            &PageOptions {
                sort_by: Some("name ASC; MATCH (m)".into()),
                ..Default::default()
            },
        )
        .await
        .expect_err("sort injection rejected");
    assert!(err.to_string().contains("not a valid Cypher identifier"));

    assert!(
        mock.captured.lock().unwrap().is_empty(),
        "nothing may reach the database"
    );

    // A malicious prefix label is rejected at query-build time too.
    let hostile = explorer_with(Arc::new(MockClient::new()), Some("E2E) DETACH DELETE (n"));
    let err = hostile
        .entity("m1")
        .await
        .expect_err("prefix injection rejected");
    assert!(err.to_string().contains("not a valid Cypher identifier"));
}

#[tokio::test]
async fn run_dsl_injects_ids_hydrates_subgraph_and_strips_columns() {
    let mock = Arc::new(MockClient::new());
    // LIFO: edges (4th call), nodes (3rd), main query (2nd); the 1st
    // pipeline step (direction validation) reads schema, not execute().
    mock.enqueue(result(
        vec!["from_id", "to_id", "rel", "props"],
        vec![
            row(vec![
                ("from_id", Value::String("p1".into())),
                ("to_id", Value::String("m1".into())),
                ("rel", Value::String("ACTED_IN".into())),
                ("props", Value::Json(json!({"roles": ["Neo"]}))),
            ]),
            row(vec![
                ("from_id", Value::String("p1".into())),
                ("to_id", Value::String("m2".into())),
                ("rel", Value::String("ACTED_IN".into())),
                ("props", Value::Json(json!({}))),
            ]),
        ],
    ));
    mock.enqueue(result(
        vec!["nid", "id", "labels", "props"],
        vec![
            row(vec![
                ("nid", Value::Int(1)),
                ("id", Value::String("p1".into())),
                ("labels", Value::Json(json!(["E2E", "Person"]))),
                (
                    "props",
                    Value::Json(json!({"id": "p1", "name": "Keanu Reeves"})),
                ),
            ]),
            row(vec![
                ("nid", Value::Int(2)),
                ("id", Value::String("m1".into())),
                ("labels", Value::Json(json!(["E2E", "Movie"]))),
                (
                    "props",
                    Value::Json(json!({"id": "m1", "title": "The Matrix"})),
                ),
            ]),
            row(vec![
                ("nid", Value::Int(3)),
                ("id", Value::String("m2".into())),
                ("labels", Value::Json(json!(["E2E", "Movie"]))),
                (
                    "props",
                    Value::Json(json!({"id": "m2", "title": "John Wick"})),
                ),
            ]),
        ],
    ));
    // Main query: two raw rows whose *visible* fields coincide (distinct
    // is restored client-side after the hidden id columns widen it).
    mock.enqueue(result(
        vec!["name", "__id_p", "__id_m", "score", "sources"],
        vec![
            row(vec![
                ("name", Value::String("Keanu Reeves".into())),
                ("__id_p", Value::String("p1".into())),
                ("__id_m", Value::String("m1".into())),
                ("score", Value::Float(0.9)),
                (
                    "sources",
                    Value::Json(json!([{"id": "s1", "name": "movies.json"}])),
                ),
            ]),
            row(vec![
                ("name", Value::String("Keanu Reeves".into())),
                ("__id_p", Value::String("p1".into())),
                ("__id_m", Value::String("m2".into())),
                (
                    "sources",
                    Value::Json(json!([{"id": "s1", "name": "movies.json"}])),
                ),
            ]),
        ],
    ));

    let explorer = explorer_with(mock.clone(), Some("E2E"));
    let dsl = linguagraph::dsl::parse_str(
        r#"{
            "start": { "label": "Person", "alias": "p" },
            "traversals": [
                { "edge": { "label": "ACTED_IN", "alias": "r", "direction": "out" },
                  "target": { "label": "Movie", "alias": "m" } }
            ],
            "filters": [ { "field": "p.name", "op": "eq", "value": "Keanu Reeves" } ],
            "return": [ { "field": "p.name", "alias": "name" } ],
            "distinct": true
        }"#,
    )
    .unwrap();

    let answer = explorer
        .run_dsl(dsl, &Default::default())
        .await
        .expect("run_dsl succeeds");

    // Table: hidden columns stripped, distinct restored.
    assert_eq!(answer.table.columns, vec!["name".to_string()]);
    assert_eq!(answer.table.rows.len(), 1);
    assert_eq!(
        answer.table.rows[0].get("name"),
        Some(&json!("Keanu Reeves"))
    );

    // Subgraph: 3 hydrated nodes, 2 edges among them.
    assert_eq!(answer.subgraph.nodes.len(), 3);
    assert_eq!(answer.subgraph.edges.len(), 2);
    assert_eq!(answer.subgraph.edges[0].id, "p1:ACTED_IN:m1");
    assert!(!answer.subgraph.truncated);

    // Provenance from the auto-injected sources column.
    assert_eq!(answer.sources.len(), 1);
    assert_eq!(answer.sources[0].id.as_deref(), Some("s1"));

    // Trace: the executed query, prefixed and with injected projections.
    assert_eq!(answer.trace.llm_attempts, 0);
    assert!(answer.trace.question.is_none());
    assert!(answer.trace.cypher.contains("__id_p"));
    assert!(answer.trace.cypher.contains("__id_m"));
    assert!(
        !answer.trace.dsl_summary.contains("__id_"),
        "summary stays clean"
    );
    assert!(answer.answer.is_none());

    let captured = mock.captured.lock().unwrap();
    assert_eq!(captured.len(), 3);
    assert!(
        captured[0].text.contains(":E2E"),
        "prefix forced: {}",
        captured[0].text
    );
    assert!(captured[1].text.contains("n.id IN $ids"));
    assert!(captured[1].text.contains("(n:E2E)"));
    assert!(captured[2].text.contains("a.id IN $ids"));
}

#[tokio::test]
async fn run_dsl_projects_filtered_field_when_entity_already_returned() {
    let mock = Arc::new(MockClient::new());
    mock.enqueue(result(
        vec!["title", "votes"],
        vec![row(vec![
            ("title", Value::String("The Matrix".into())),
            ("votes", Value::Int(80)),
        ])],
    ));

    let explorer = explorer_with(mock.clone(), Some("E2E"));
    let dsl = linguagraph::dsl::parse_str(
        r#"{
            "start": { "label": "Movie", "alias": "m" },
            "filters": [ { "field": "m.votes", "op": "lt", "value": 100 } ],
            "return": [ { "field": "m.title" } ]
        }"#,
    )
    .unwrap();

    let answer = explorer
        .run_dsl(
            dsl,
            &AskOptions {
                include_subgraph: false,
                ..Default::default()
            },
        )
        .await
        .expect("run_dsl succeeds");

    // The filter's own field (votes) is auto-projected because `m` is
    // already part of the response via `title` — the answer LLM (and the
    // table itself) can now see the value that made the row match.
    assert_eq!(
        answer.table.columns,
        vec!["title".to_string(), "votes".to_string()]
    );
    assert_eq!(answer.table.rows[0].get("votes"), Some(&json!(80)));

    let captured = mock.captured.lock().unwrap();
    assert_eq!(captured.len(), 1);
    assert!(
        captured[0].text.contains("m.votes"),
        "filter field projected: {}",
        captured[0].text
    );
}

#[tokio::test]
async fn run_dsl_skips_filter_context_when_disabled() {
    let mock = Arc::new(MockClient::new());
    mock.enqueue(result(
        vec!["title"],
        vec![row(vec![("title", Value::String("The Matrix".into()))])],
    ));

    let explorer = explorer_with(mock.clone(), Some("E2E"));
    let dsl = linguagraph::dsl::parse_str(
        r#"{
            "start": { "label": "Movie", "alias": "m" },
            "filters": [ { "field": "m.votes", "op": "lt", "value": 100 } ],
            "return": [ { "field": "m.title" } ]
        }"#,
    )
    .unwrap();

    let answer = explorer
        .run_dsl(
            dsl,
            &AskOptions {
                include_subgraph: false,
                include_filter_context: false,
                ..Default::default()
            },
        )
        .await
        .expect("run_dsl succeeds");

    assert_eq!(answer.table.columns, vec!["title".to_string()]);
    let captured = mock.captured.lock().unwrap();
    // `m.votes` still appears in the WHERE clause (the filter itself);
    // what must stay absent is a `votes` projection in RETURN.
    assert!(!captured[0].text.contains("RETURN m.title, m.votes"));
}

#[tokio::test]
async fn run_dsl_exposes_entity_refs_without_subgraph() {
    let mock = Arc::new(MockClient::new());
    mock.enqueue(result(
        vec!["title", "__id_m"],
        vec![
            row(vec![
                ("title", Value::String("The Matrix".into())),
                ("__id_m", Value::String("m1".into())),
            ]),
            row(vec![
                ("title", Value::String("John Wick".into())),
                ("__id_m", Value::String("m2".into())),
            ]),
        ],
    ));

    let explorer = explorer_with(mock.clone(), Some("E2E"));
    let dsl = linguagraph::dsl::parse_str(
        r#"{
            "start": { "label": "Movie", "alias": "m" },
            "return": [ { "field": "m.title", "alias": "title" } ]
        }"#,
    )
    .unwrap();

    let answer = explorer
        .run_dsl(
            dsl,
            &AskOptions {
                include_subgraph: false,
                ..Default::default()
            },
        )
        .await
        .expect("run_dsl succeeds");

    // No subgraph hydration round trips — just the one main query — yet
    // the table still carries enough to navigate from a cell to its node.
    assert!(answer.subgraph.is_empty());
    {
        let captured = mock.captured.lock().unwrap();
        assert_eq!(
            captured.len(),
            1,
            "no hydration queries without include_subgraph"
        );
        assert!(captured[0].text.contains("m.id AS __id_m"));
    }

    assert_eq!(answer.table.columns, vec!["title".to_string()]);
    assert_eq!(
        answer.table.entity_columns.get("title").map(String::as_str),
        Some("m"),
        "the `title` column is traced back to alias `m`"
    );
    assert_eq!(
        answer.table.row_entities,
        vec![
            BTreeMap::from([("m".to_string(), "m1".to_string())]),
            BTreeMap::from([("m".to_string(), "m2".to_string())]),
        ],
        "row_entities stays index-aligned with table.rows"
    );
}

#[tokio::test]
async fn run_dsl_disabling_entity_refs_omits_them() {
    let mock = Arc::new(MockClient::new());
    mock.enqueue(result(
        vec!["title"],
        vec![row(vec![("title", Value::String("The Matrix".into()))])],
    ));

    let explorer = explorer_with(mock.clone(), Some("E2E"));
    let dsl = linguagraph::dsl::parse_str(
        r#"{
            "start": { "label": "Movie", "alias": "m" },
            "return": [ { "field": "m.title", "alias": "title" } ]
        }"#,
    )
    .unwrap();

    let answer = explorer
        .run_dsl(
            dsl,
            &AskOptions {
                include_subgraph: false,
                include_entity_refs: false,
                ..Default::default()
            },
        )
        .await
        .expect("run_dsl succeeds");

    assert!(answer.table.entity_columns.is_empty());
    assert!(answer.table.row_entities.is_empty());
    let captured = mock.captured.lock().unwrap();
    assert!(!captured[0].text.contains("__id_"));
}

#[tokio::test]
async fn aggregate_dsl_has_no_subgraph_but_full_trace() {
    let mock = Arc::new(MockClient::new());
    mock.enqueue(result(
        vec!["total"],
        vec![row(vec![("total", Value::Int(38))])],
    ));

    let explorer = explorer_with(mock.clone(), Some("E2E"));
    let dsl = linguagraph::dsl::parse_str(
        r#"{
            "start": { "label": "Movie", "alias": "m" },
            "return": [ { "aggregate": "count", "field": "m", "alias": "total" } ]
        }"#,
    )
    .unwrap();

    let answer = explorer
        .run_dsl(dsl, &Default::default())
        .await
        .expect("aggregate runs");

    assert!(
        answer.subgraph.is_empty(),
        "aggregates have no entity bindings"
    );
    assert_eq!(answer.table.rows.len(), 1);
    assert_eq!(answer.table.rows[0].get("total"), Some(&json!(38)));
    assert!(!answer.trace.cypher.is_empty());

    let captured = mock.captured.lock().unwrap();
    assert_eq!(captured.len(), 1, "no hydration queries for aggregates");
    assert!(
        !captured[0].text.contains("__id_"),
        "no injection: {}",
        captured[0].text
    );
}

#[tokio::test]
async fn semantic_search_ranks_hits_and_hydrates_by_internal_id() {
    use linguagraph::embeddings::MockEmbedder;
    use linguagraph::explore::{SearchChannel, SearchMode};

    let mock = Arc::new(MockClient::new());
    // LIFO: hydration (2nd call) first, vector retrieval (1st) second.
    mock.enqueue(result(
        vec!["nid", "id", "labels", "props"],
        vec![
            row(vec![
                ("nid", Value::Int(1)),
                ("id", Value::String("p1".into())),
                ("labels", Value::Json(json!(["E2E", "Person"]))),
                (
                    "props",
                    Value::Json(json!({"id": "p1", "name": "Keanu Reeves"})),
                ),
            ]),
            row(vec![
                ("nid", Value::Int(2)),
                ("id", Value::String("m1".into())),
                ("labels", Value::Json(json!(["E2E", "Movie"]))),
                (
                    "props",
                    Value::Json(json!({"id": "m1", "title": "The Matrix"})),
                ),
            ]),
        ],
    ));
    mock.enqueue(result(
        vec!["nid", "score", "leg"],
        vec![
            row(vec![
                ("nid", Value::Int(2)),
                ("score", Value::Float(0.72)),
                ("leg", Value::String("entity".into())),
            ]),
            row(vec![
                ("nid", Value::Int(1)),
                ("score", Value::Float(0.91)),
                ("leg", Value::String("entity".into())),
            ]),
        ],
    ));

    let cfg = test_config();
    let pipeline = Pipeline::new(mock.clone(), &cfg)
        .with_prefix_label(Some("E2E"))
        .with_ontology_catalog(movie_catalog())
        .with_embedder(Arc::new(MockEmbedder::new(8)));
    let explorer = Explorer::new(pipeline);

    let found = explorer
        .search(
            "who played neo",
            &SearchOptions {
                mode: SearchMode::Semantic,
                ..Default::default()
            },
        )
        .await
        .expect("semantic search runs");

    // Ranked by score descending, hydrated through internal ids.
    assert_eq!(found.hits.len(), 2);
    assert_eq!(found.hits[0].node.id, "p1");
    assert_eq!(found.hits[0].score, Some(0.91));
    assert!(matches!(found.hits[0].channel, SearchChannel::Semantic));
    assert_eq!(found.hits[1].node.id, "m1");

    let captured = mock.captured.lock().unwrap();
    assert_eq!(captured.len(), 2);
    assert!(captured[0].text.contains("libqlink.search"));
    assert!(captured[0].text.contains("$prefix_label IN labels(e)"));
    assert!(captured[1].text.contains("id(n) IN $nids"));
}

#[tokio::test]
async fn semantic_mode_without_embedder_errors_but_auto_degrades() {
    use linguagraph::explore::SearchMode;

    let mock = Arc::new(MockClient::new());
    mock.enqueue(QueryResult::default());
    let explorer = explorer_with(mock.clone(), Some("E2E")); // no embedder

    let err = explorer
        .search(
            "neo",
            &SearchOptions {
                mode: SearchMode::Semantic,
                ..Default::default()
            },
        )
        .await
        .expect_err("semantic without embedder must fail");
    assert!(
        err.to_string().contains("semantic search unavailable"),
        "got: {err}"
    );

    // Auto silently uses the keyword channel instead.
    let found = explorer
        .search("neo", &SearchOptions::default())
        .await
        .expect("auto degrades to keyword");
    assert!(found.hits.is_empty());
    let captured = mock.captured.lock().unwrap();
    assert_eq!(captured.len(), 1);
    assert!(!captured[0].text.contains("libqlink"));
}

#[tokio::test]
async fn all_digit_handle_falls_back_to_internal_id() {
    let mock = Arc::new(MockClient::new());
    // LIFO: relation summary (3rd call), nid-fallback lookup (2nd),
    // empty app-id lookup (1st).
    mock.enqueue(QueryResult::default());
    mock.enqueue(result(
        vec!["nid", "id", "labels", "props", "sources"],
        vec![row(vec![
            ("nid", Value::Int(1000086)),
            ("id", Value::String("listing-42".into())),
            ("labels", Value::Json(json!(["E2E", "Listing"]))),
            (
                "props",
                Value::Json(json!({"id": "listing-42", "name": "Shop"})),
            ),
            ("sources", Value::Json(json!([]))),
        ])],
    ));
    mock.enqueue(QueryResult::default());

    let explorer = explorer_with(mock.clone(), Some("E2E"));
    let card = explorer
        .entity("1000086")
        .await
        .expect("lookup runs")
        .expect("resolved via internal id");
    // The stable property handle wins over the internal id the user typed.
    assert_eq!(card.node.id, "listing-42");
    assert!(!card.node.ephemeral_handle);

    let captured = mock.captured.lock().unwrap();
    assert_eq!(captured.len(), 3);
    // 1st try: id property, string AND integer forms.
    assert!(captured[0].text.contains("(n.id = $id OR n.id = $id_int)"));
    // 2nd try: internal id.
    assert!(captured[1].text.contains("id(n) = $nid"));
    // Relation summary must reuse the handle that actually matched.
    assert!(captured[2].text.contains("id(n) = $nid"));
}

#[tokio::test]
async fn integer_stored_id_property_resolves_and_stringifies() {
    let mock = Arc::new(MockClient::new());
    // Found on the first (property) lookup — the node stores id as an int.
    mock.enqueue(QueryResult::default()); // relation summary
    mock.enqueue(result(
        vec!["nid", "id", "labels", "props", "sources"],
        vec![row(vec![
            ("nid", Value::Int(7)),
            // Wire shape: Json(Number), as the Memgraph client emits.
            ("id", Value::Json(json!(1000086))),
            ("labels", Value::Json(json!(["Listing"]))),
            ("props", Value::Json(json!({"id": 1000086, "name": "Shop"}))),
            ("sources", Value::Json(json!([]))),
        ])],
    ));

    let explorer = explorer_with(mock.clone(), None);
    let card = explorer.entity("1000086").await.unwrap().unwrap();
    assert_eq!(card.node.id, "1000086", "numeric id stringified");
    assert!(!card.node.ephemeral_handle);

    let captured = mock.captured.lock().unwrap();
    assert_eq!(captured.len(), 2, "no fallback needed");
}
