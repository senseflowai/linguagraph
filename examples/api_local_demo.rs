//! Runnable tour of the `linguagraph::api` surface — no live database
//! required. It drives a [`LinguaGraph`] built over the in-memory
//! [`MockClient`], so everything here executes offline.
//!
//! Run it with:
//!
//! ```sh
//! cargo run --example api_local_demo --no-default-features
//! ```
//!
//! It shows the parts of the contract that are wired today
//! (construction, tenant scoping, `serde` round-tripping, the
//! bounded-traversal guard, and `delete_by_source`) and prints the
//! `GraphError` returned by capabilities whose backend mapping is still
//! pending — so the output doubles as a status map of the surface.

use std::sync::Arc;

use linguagraph::api::{
    AskOptions, Confidence, Entity, EntityId, EntityType, GraphError, GraphRead, GraphWrite,
    LinguaGraph, Limits, Property, PropertyKind, Provenance, SourceId, SourceRef, TenantId,
    Timestamp, TraversalOptions, Value,
};
use linguagraph::config;
use linguagraph::core::Pipeline;
use linguagraph::db::MockClient;

/// Build a handle over a mock client — the same pattern the unit tests
/// use. A real deployment would use `LinguaGraph::builder().memgraph(..)`
/// instead (see `examples/api_rest_handler.rs`).
fn local_graph() -> LinguaGraph {
    let cfg = config::load_from_str(
        r#"
        [database]
        uri = "bolt://localhost:7687"
        user = ""
        password = ""
        "#,
    )
    .expect("valid config");
    let pipeline = Pipeline::new(Arc::new(MockClient::new()), &cfg);
    LinguaGraph::from_pipeline(pipeline, Limits::default())
}

#[tokio::main]
async fn main() {
    let graph = local_graph();
    println!("handle limits: {:?}\n", graph.limits());

    // ── 1. Tenant scoping ───────────────────────────────────────────
    // Nothing can touch the graph without selecting a tenant. Two
    // scopes for two tenants never see each other's data.
    let acme = graph.read(TenantId("acme".into()));
    let globex = graph.read(TenantId("globex".into()));
    println!(
        "read scopes bound to tenants: {:?} / {:?}\n",
        acme.tenant(),
        globex.tenant()
    );

    // ── 2. The model is plain serde ─────────────────────────────────
    // The REST service hands these straight to its transport layer.
    let entity = Entity {
        id: EntityId("company:nordwind".into()),
        name: "Nordwind Holding".into(),
        entity_type: EntityType("Company".into()),
        domain: None,
        properties: vec![
            Property {
                key: "ticker".into(),
                value: Value::Text("NWH".into()),
                kind: PropertyKind::Keyword,
            },
            Property {
                key: "summary".into(),
                value: Value::Text("Controls three regulated assets.".into()),
                kind: PropertyKind::Text,
            },
        ],
        confidence: Confidence::from_score(0.92),
        provenance: Provenance {
            sources: vec![SourceRef {
                source_id: SourceId("doc:annual-2024".into()),
                document: "Annual report 2024".into(),
                locator: Some("p. 12".into()),
                extracted_at: Timestamp(1_720_000_000_000),
            }],
        },
    };
    let json = serde_json::to_string_pretty(&entity).unwrap();
    println!("an Entity as the UI would receive it:\n{json}\n");

    // ── 3. Bounded traversals are enforced ──────────────────────────
    // depth/limit are mandatory; drop one and the library refuses.
    let mut unbounded = TraversalOptions::one_hop(25);
    unbounded.limit = 0;
    match acme.neighbors(&entity.id, unbounded).await {
        Err(GraphError::UnboundedTraversal) => {
            println!("unbounded traversal correctly rejected\n");
        }
        other => println!("unexpected: {other:?}\n"),
    }

    // A properly bounded 1-hop request passes validation and reaches the
    // (not-yet-wired) traversal backend.
    let bounded = acme.neighbors(&entity.id, TraversalOptions::one_hop(25)).await;
    println!("bounded neighbors() -> {bounded:?}\n");

    // ── 4. A wired write path: delete_by_source ─────────────────────
    // Against the mock the source is unknown, so this is a clean no-op
    // report rather than an error.
    let writer = graph.write(TenantId("acme".into()));
    let report = writer
        .delete_by_source(&SourceId("doc:annual-2024".into()))
        .await
        .expect("delete_by_source runs against the mock");
    println!("delete_by_source report: {report:?}\n");

    // ── 5. Capabilities awaiting a backend report Unsupported ───────
    // They still validate inputs first (note the empty-question guard).
    match acme.ask("  ", AskOptions::default()).await {
        Err(GraphError::InvalidQuestion(why)) => {
            println!("ask(\"  \") rejected before compile: {why}");
        }
        other => println!("unexpected: {other:?}"),
    }
    if let Err(e) = acme.ask("Who controls Nordwind?", AskOptions::default()).await {
        println!("ask(...) -> {e}");
    }

    // ── 6. Least privilege by type ──────────────────────────────────
    // The read path can be handed a `dyn GraphRead` with no way to write.
    let read_only: &dyn GraphRead = &acme;
    println!(
        "\nread-only handle can search/traverse but not ingest; \
         tenant = {:?}",
        read_only.discover_types("assets").await.err().map(|_| "needs embedder")
    );
}
