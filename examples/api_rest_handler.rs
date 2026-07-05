//! How a REST service wires an endpoint onto the `linguagraph::api`
//! surface — the pattern from §6 of the public-API design.
//!
//! The service owns transport, auth, and caching; the library owns the
//! graph. A handler therefore boils down to: resolve the tenant, take a
//! (read-only) scope, translate query params into typed options, call
//! the capability, and map [`GraphError`] onto an HTTP status.
//!
//! Run it with:
//!
//! ```sh
//! cargo run --example api_rest_handler --no-default-features
//! ```
//!
//! It runs against a mock client so it executes offline; the commented
//! `builder()` block shows the real Memgraph wiring.

use std::sync::Arc;

use linguagraph::api::{
    Cursor, Direction, EntityId, GraphError, GraphRead, LinguaGraph, Limits, NeighborsPage,
    RelationType, TenantId, TraversalOptions,
};
use linguagraph::config;
use linguagraph::core::Pipeline;
use linguagraph::db::MockClient;

/// Query parameters for `GET /graph/entities/{id}/neighbors`, as the web
/// framework would deserialize them.
#[derive(Debug, Default)]
struct NeighborsQuery {
    direction: Option<Direction>,
    rel: Option<String>,
    depth: Option<u8>,
    limit: Option<u32>,
    cursor: Option<String>,
}

/// The service's own error envelope. In a real app this implements the
/// framework's `IntoResponse`; here we just carry a status + message.
#[derive(Debug)]
struct ApiError {
    status: u16,
    message: String,
}

/// The one place the library's error taxonomy meets HTTP. Because
/// [`GraphError`] is small and stable, this mapping is exhaustive and
/// easy to keep correct.
impl From<GraphError> for ApiError {
    fn from(e: GraphError) -> Self {
        let status = match &e {
            GraphError::NotFound => 404,
            GraphError::InvalidQuestion(_) | GraphError::UnboundedTraversal => 400,
            GraphError::CostExceeded(_) => 413,   // payload/So-much-work too large
            GraphError::Timeout => 504,
            GraphError::TenantIsolation => 403,
            GraphError::Config(_) => 500,
            GraphError::Backend(_) => 502,
            GraphError::Unsupported(_) => 501,
            _ => 500,
        };
        ApiError {
            status,
            message: e.to_string(),
        }
    }
}

/// `GET /graph/entities/{id}/neighbors?rel=OWNS&depth=1&limit=25&cursor=…`
///
/// The handler depends only on `&dyn GraphRead`, so it is trivially
/// unit-testable with a mock and can be routed to a read replica.
async fn neighbors_handler(
    graph: &LinguaGraph,
    tenant: TenantId,
    id: String,
    q: NeighborsQuery,
) -> Result<NeighborsPage, ApiError> {
    let scope = graph.read(tenant); // read-only; safe for a replica
    let page = scope
        .neighbors(
            &EntityId(id),
            TraversalOptions {
                direction: q.direction.unwrap_or(Direction::Both),
                relation_types: q.rel.map(|r| vec![RelationType(r)]),
                entity_types: None,
                depth: q.depth.unwrap_or(1), // mandatory — default supplied here
                limit: q.limit.unwrap_or(25),
                cursor: q.cursor.map(Cursor),
                min_confidence: None,
            },
        )
        .await?; // GraphError -> ApiError via the `From` above
    Ok(page)
}

fn service_graph() -> LinguaGraph {
    // Real deployment:
    //
    //     let graph = LinguaGraph::builder()
    //         .memgraph("bolt://localhost:7687", PoolConfig::default())
    //         .with_embedder(embedder)
    //         .default_limits(Limits::default())
    //         .build()
    //         .await?;
    //
    // Offline demo: a handle over the in-memory mock client.
    let cfg = config::load_from_str(
        "[database]\nuri = \"bolt://localhost:7687\"\nuser = \"\"\npassword = \"\"\n",
    )
    .expect("valid config");
    let pipeline = Pipeline::new(Arc::new(MockClient::new()), &cfg);
    LinguaGraph::from_pipeline(pipeline, Limits::default())
}

#[tokio::main]
async fn main() {
    // One handle per process; clone it into each request.
    let graph = service_graph();

    // A well-formed request. Against the mock the traversal backend isn't
    // wired, so the handler surfaces 501 — exactly what the error mapping
    // is there to produce.
    let ok_query = NeighborsQuery {
        rel: Some("OWNS".into()),
        depth: Some(1),
        limit: Some(25),
        ..Default::default()
    };
    match neighbors_handler(
        &graph,
        TenantId("acme".into()),
        "company:nordwind".into(),
        ok_query,
    )
    .await
    {
        Ok(page) => println!("200 OK\n{}", serde_json::to_string_pretty(&page).unwrap()),
        Err(err) => println!("{} {}", err.status, err.message),
    }

    // A malformed request: limit=0 makes the traversal unbounded, which
    // the library rejects and the mapping turns into 400.
    let bad_query = NeighborsQuery {
        depth: Some(1),
        limit: Some(0),
        ..Default::default()
    };
    match neighbors_handler(
        &graph,
        TenantId("acme".into()),
        "company:nordwind".into(),
        bad_query,
    )
    .await
    {
        Ok(_) => println!("unexpected success"),
        Err(err) => println!("{} {}", err.status, err.message),
    }
}
