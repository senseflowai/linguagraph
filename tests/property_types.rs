//! Property-type handling through the graph ingestion planner.
//!
//! Entities built with [`GraphBuilder`] carry typed properties
//! ([`PropertyType`]). At plan time the planner resolves each property's
//! type handler from the registry and runs its `on_ingest`, so the value
//! is normalised on the rendered node row:
//!
//! * **Number** strips thousands separators, understands a trailing `%`,
//!   and keeps whole numbers typed as `Int`.
//! * **Boolean** accepts the friendly string forms (`"yes"`/`"no"`, …).
//! * **Timestamp** renders an epoch integer as ISO-8601 UTC.
//! * A malformed value for a typed property surfaces an ingestion error
//!   rather than silently storing the raw string.
//! * A `null` (or absent) value drops the property from the row.

use std::sync::Arc;

use linguagraph::ast::query::{Literal, NodeRow};
use linguagraph::embeddings::MockEmbedder;
use linguagraph::graph::{GraphBuilder, PropertyType};
use linguagraph::ingest::{self, PlannerOptions};
use linguagraph::types::handlers::{self, SemanticTextConfig, SemanticTextHandler};
use linguagraph::types::{RegistryBuilder, SideEffectQueue, TypeRegistry};

fn pick<'a>(rows: &'a [NodeRow], id: &str) -> &'a NodeRow {
    rows.iter()
        .find(|r| matches!(&r.id, Literal::String(s) if s == id))
        .expect("row missing")
}

/// Core scalar handlers plus a `SemanticText` handler (backed by a mock
/// embedder) — the latter is needed because `GraphBuilder` synthesizes a
/// `_canonical` SemanticText property on every entity.
fn registry() -> TypeRegistry {
    handlers::register_core(RegistryBuilder::new())
        .register(SemanticTextHandler::new(
            SemanticTextConfig {
                embedding_model: None,
                collection: "docs".into(),
                top_k: 10,
                search_threshold: 0.1,
                reranker_threshold: 0.2,
            },
            Arc::new(MockEmbedder::new(8)),
        ))
        .build()
}

fn plan_single_reading(build: impl FnOnce(&mut GraphBuilder)) -> Result<Vec<NodeRow>, String> {
    let mut builder = GraphBuilder::new();
    build(&mut builder);
    let graph = builder.build();

    let mut effects = SideEffectQueue::new();
    ingest::plan_graph_with_registry(&graph, PlannerOptions::default(), &registry(), &mut effects)
        .map(|insert| insert.node_batches[0].rows.clone())
        .map_err(|e| e.to_string())
}

#[test]
fn core_scalar_types_parse_through_the_planner() {
    let rows = plan_single_reading(|b| {
        b.entity("Reading")
            .strict_primary_key("id")
            .property("id", PropertyType::Keyword, "r1")
            .property("count", PropertyType::Number, "1,234")
            .property("ratio", PropertyType::Number, "3.5")
            .property("share", PropertyType::Number, "12.5%")
            .property("active", PropertyType::Bool, "yes")
            .property("approved", PropertyType::Bool, false)
            .property("recorded_at", PropertyType::Datetime, 1_704_067_200i64)
            .add();
    })
    .unwrap();
    let row = pick(&rows, "r1");

    // Number: thousands separators stripped, kept as Int.
    assert_eq!(row.props.get("count"), Some(&Literal::Int(1234)));
    // Number: parsed float.
    assert_eq!(row.props.get("ratio"), Some(&Literal::Float(3.5)));
    // Number with %: divided by 100.
    assert_eq!(row.props.get("share"), Some(&Literal::Float(0.125)));
    // Boolean: friendly string forms.
    assert_eq!(row.props.get("active"), Some(&Literal::Bool(true)));
    assert_eq!(row.props.get("approved"), Some(&Literal::Bool(false)));
    // Timestamp: epoch integer rendered as ISO-8601 UTC.
    assert_eq!(
        row.props.get("recorded_at"),
        Some(&Literal::String("2024-01-01T00:00:00Z".into()))
    );
}

#[test]
fn malformed_value_for_typed_property_is_an_error() {
    let err = plan_single_reading(|b| {
        b.entity("Reading")
            .strict_primary_key("id")
            .property("id", PropertyType::Keyword, "r1")
            .property("count", PropertyType::Number, "garbage")
            .add();
    })
    .unwrap_err();
    assert!(
        err.contains("Number") && err.contains("garbage"),
        "expected error to mention the offending value; got {err}"
    );
}

#[test]
fn null_value_drops_the_property() {
    let rows = plan_single_reading(|b| {
        b.entity("Reading")
            .strict_primary_key("id")
            .property("id", PropertyType::Keyword, "r1")
            .property("count", PropertyType::Number, serde_json::Value::Null)
            .add();
    })
    .unwrap();
    let row = pick(&rows, "r1");
    assert!(!row.props.contains_key("count"));
}

#[test]
fn absent_property_is_simply_not_on_the_row() {
    let rows = plan_single_reading(|b| {
        b.entity("Reading")
            .strict_primary_key("id")
            .property("id", PropertyType::Keyword, "r1")
            .add();
    })
    .unwrap();
    let row = pick(&rows, "r1");
    assert!(!row.props.contains_key("count"));
}
