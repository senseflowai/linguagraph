//! End-to-end tests for the property type system in the mapper.
//!
//! These tests exercise:
//!
//! * **Required type tag.** Mappings whose properties don't declare a
//!   `type` are rejected at load time with a precise
//!   [`MapperError::MissingPropertyType`] error.
//! * **Built-in scalar parsing.** Each of `Text`, `Number`, `Boolean`,
//!   `Date`, `Timestamp` is exercised through `extract` + planner so we
//!   verify the value is rewritten on the rendered node row.
//! * **Custom registered types.** A user-defined parser (here:
//!   `Percentage`) is registered alongside the core handlers and used
//!   by a mapping. Confirms the "scalable" registration path advertised
//!   in the docs.
//! * **Loud failures on bad data.** A `Number` field receiving a junk
//!   string surfaces a ingestion error rather than silently storing the
//!   string.

use std::sync::Arc;

use serde_json::json;

use linguagraph::ast::query::Literal;
use linguagraph::ingest;
use linguagraph::mapper::{self, Mapping, MapperError};
use linguagraph::types::context::IngestCtx;
use linguagraph::types::handlers::{self, ScalarParser, ScalarTypeHandler};
use linguagraph::types::{
    Capabilities, RegistryBuilder, SideEffectQueue, TypeError, TypeHandler, TypeId,
};

fn pick<'a>(rows: &'a [linguagraph::ast::query::NodeRow], id: &str) -> &'a linguagraph::ast::query::NodeRow {
    rows.iter()
        .find(|r| matches!(&r.id, Literal::String(s) if s == id))
        .expect("row missing")
}

#[test]
fn missing_type_tag_is_rejected_at_load() {
    let raw = r#"{
        "entities": [{
            "type": "Camera",
            "source_path": "$.cameras[*]",
            "primary_key": "$.cameras[*].id",
            "properties": [
                {"name": "name", "source_path": "$.cameras[*].name"}
            ]
        }]
    }"#;
    let err = Mapping::from_str(raw).unwrap_err();
    match err {
        MapperError::MissingPropertyType { entity, property } => {
            assert_eq!(entity, "Camera");
            assert_eq!(property, "name");
        }
        other => panic!("expected MissingPropertyType, got {other:?}"),
    }
}

#[test]
fn missing_type_message_names_the_offending_property() {
    let raw = r#"{
        "entities": [{
            "type": "Place",
            "source_path": "$.places[*]",
            "primary_key": "$.places[*].id",
            "properties": [
                {"name": "id", "source_path": "$.places[*].id", "type": "Text"},
                {"name": "lat", "source_path": "$.places[*].lat"}
            ]
        }]
    }"#;
    let err = Mapping::from_str(raw).unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("lat"), "error must mention the property name; got {msg}");
    assert!(msg.contains("Place"), "error must mention the entity; got {msg}");
}

#[test]
fn core_scalar_types_parse_during_extraction() {
    let mapping = Mapping::from_str(
        r#"{
            "entities": [{
                "type": "Reading",
                "source_path": "$.readings[*]",
                "primary_key": "$.readings[*].id",
                "properties": [
                    {"name": "id",         "source_path": "$.readings[*].id",         "type": "Text"},
                    {"name": "label",      "source_path": "$.readings[*].label",      "type": "Text"},
                    {"name": "count",      "source_path": "$.readings[*].count",      "type": "Number"},
                    {"name": "ratio",      "source_path": "$.readings[*].ratio",      "type": "Number"},
                    {"name": "share",      "source_path": "$.readings[*].share",      "type": "Number"},
                    {"name": "active",     "source_path": "$.readings[*].active",     "type": "Boolean"},
                    {"name": "approved",   "source_path": "$.readings[*].approved",   "type": "Boolean"},
                    {"name": "born_on",    "source_path": "$.readings[*].born_on",    "type": "Date"},
                    {"name": "recorded_at","source_path": "$.readings[*].recorded_at","type": "Timestamp"}
                ]
            }]
        }"#,
    )
    .unwrap();

    let data = json!({
        "readings": [{
            "id": "r1",
            "label": 42,
            "count": "1,234",
            "ratio": "3.5",
            "share": "12.5%",
            "active": "yes",
            "approved": false,
            "born_on": "2024-05-06T08:30:00Z",
            "recorded_at": 1_704_067_200i64
        }]
    });

    let extracted = mapper::extract(&mapping, &data).unwrap();
    let q = ingest::plan(&mapping, extracted).unwrap();
    let row = pick(&q.node_batches[0].rows, "r1");

    // Text: integer pulled from JSON gets stringified.
    assert_eq!(row.props.get("label"), Some(&Literal::String("42".into())));
    // Number: thousands separators stripped, kept as Int.
    assert_eq!(row.props.get("count"), Some(&Literal::Int(1234)));
    // Number: parsed float.
    assert_eq!(row.props.get("ratio"), Some(&Literal::Float(3.5)));
    // Number with %: divided by 100.
    assert_eq!(row.props.get("share"), Some(&Literal::Float(0.125)));
    // Boolean: friendly string forms.
    assert_eq!(row.props.get("active"), Some(&Literal::Bool(true)));
    assert_eq!(row.props.get("approved"), Some(&Literal::Bool(false)));
    // Date: time component stripped.
    assert_eq!(
        row.props.get("born_on"),
        Some(&Literal::String("2024-05-06".into()))
    );
    // Timestamp: epoch integer rendered as ISO-8601 UTC.
    assert_eq!(
        row.props.get("recorded_at"),
        Some(&Literal::String("2024-01-01T00:00:00Z".into()))
    );
}

#[test]
fn malformed_value_for_typed_property_is_an_error() {
    let mapping = Mapping::from_str(
        r#"{
            "entities": [{
                "type": "Reading",
                "source_path": "$.readings[*]",
                "primary_key": "$.readings[*].id",
                "properties": [
                    {"name": "id",    "source_path": "$.readings[*].id",    "type": "Text"},
                    {"name": "count", "source_path": "$.readings[*].count", "type": "Number"}
                ]
            }]
        }"#,
    )
    .unwrap();
    let data = json!({"readings": [{"id": "r1", "count": "garbage"}]});
    let extracted = mapper::extract(&mapping, &data).unwrap();
    let err = ingest::plan(&mapping, extracted).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("Number") && msg.contains("garbage"),
        "expected error to mention the offending value; got {msg}"
    );
}

#[test]
fn missing_typed_property_is_still_tolerated() {
    // Same contract as before: a property whose JSONPath yields no
    // match is left off the row. The type tag does not change that.
    let mapping = Mapping::from_str(
        r#"{
            "entities": [{
                "type": "Reading",
                "source_path": "$.readings[*]",
                "primary_key": "$.readings[*].id",
                "properties": [
                    {"name": "id",    "source_path": "$.readings[*].id",    "type": "Text"},
                    {"name": "count", "source_path": "$.readings[*].count", "type": "Number"}
                ]
            }]
        }"#,
    )
    .unwrap();
    let data = json!({"readings": [{"id": "r1"}]});
    let extracted = mapper::extract(&mapping, &data).unwrap();
    let q = ingest::plan(&mapping, extracted).unwrap();
    let row = pick(&q.node_batches[0].rows, "r1");
    assert!(!row.props.contains_key("count"));
}

#[test]
fn null_value_drops_the_property() {
    let mapping = Mapping::from_str(
        r#"{
            "entities": [{
                "type": "Reading",
                "source_path": "$.readings[*]",
                "primary_key": "$.readings[*].id",
                "properties": [
                    {"name": "id",    "source_path": "$.readings[*].id",    "type": "Text"},
                    {"name": "count", "source_path": "$.readings[*].count", "type": "Number"}
                ]
            }]
        }"#,
    )
    .unwrap();
    let data = json!({"readings": [{"id": "r1", "count": null}]});
    let extracted = mapper::extract(&mapping, &data).unwrap();
    let q = ingest::plan(&mapping, extracted).unwrap();
    let row = pick(&q.node_batches[0].rows, "r1");
    assert!(!row.props.contains_key("count"));
}

#[test]
fn unknown_type_tag_is_an_ingest_error() {
    let mapping = Mapping::from_str(
        r#"{
            "entities": [{
                "type": "Reading",
                "source_path": "$.readings[*]",
                "primary_key": "$.readings[*].id",
                "properties": [
                    {"name": "id",  "source_path": "$.readings[*].id",  "type": "Text"},
                    {"name": "val", "source_path": "$.readings[*].val", "type": "GhostType"}
                ]
            }]
        }"#,
    )
    .unwrap();
    let data = json!({"readings": [{"id": "r1", "val": 1}]});
    let extracted = mapper::extract(&mapping, &data).unwrap();
    let err = ingest::plan(&mapping, extracted).unwrap_err();
    assert!(
        err.to_string().contains("GhostType"),
        "error must name the unknown type; got {err}"
    );
}

// ─── Custom registered type ────────────────────────────────────────────

/// A toy parser that mirrors the kind of extension users will write in
/// downstream crates. Stores a percentage as a `[0, 100]` integer
/// regardless of whether the source supplied `0.5`, `"50%"`, or `50`.
#[derive(Debug, Default)]
struct PercentageParser;

impl ScalarParser for PercentageParser {
    fn parse(
        &self,
        raw: &serde_json::Value,
    ) -> Result<Option<Literal>, TypeError> {
        match raw {
            serde_json::Value::Null => Ok(None),
            serde_json::Value::Number(n) => {
                let f = n.as_f64().ok_or_else(|| TypeError::InvalidValue {
                    ty: "Percentage".into(),
                    reason: format!("non-finite number: {n}"),
                })?;
                let pct = if (0.0..=1.0).contains(&f) { f * 100.0 } else { f };
                Ok(Some(Literal::Int(pct.round() as i64)))
            }
            serde_json::Value::String(s) => {
                let body = s.trim().trim_end_matches('%');
                let f: f64 = body.parse().map_err(|_| TypeError::InvalidValue {
                    ty: "Percentage".into(),
                    reason: format!("not a percentage: {s:?}"),
                })?;
                let pct = if !s.contains('%') && (0.0..=1.0).contains(&f) {
                    f * 100.0
                } else {
                    f
                };
                Ok(Some(Literal::Int(pct.round() as i64)))
            }
            other => Err(TypeError::InvalidValue {
                ty: "Percentage".into(),
                reason: format!("unsupported value: {other}"),
            }),
        }
    }
}

#[test]
fn custom_registered_type_runs_alongside_core_types() {
    // Mapping uses the core `Text` type plus a user-registered
    // `Percentage` type. The registry must resolve both.
    let mapping = Mapping::from_str(
        r#"{
            "entities": [{
                "type": "Loader",
                "source_path": "$.loaders[*]",
                "primary_key": "$.loaders[*].id",
                "properties": [
                    {"name": "id",       "source_path": "$.loaders[*].id",       "type": "Text"},
                    {"name": "progress", "source_path": "$.loaders[*].progress", "type": "Percentage"}
                ]
            }]
        }"#,
    )
    .unwrap();

    let data = json!({
        "loaders": [
            {"id": "a", "progress": 0.5},
            {"id": "b", "progress": "75%"},
            {"id": "c", "progress": 30}
        ]
    });

    let registry = handlers::register_core(RegistryBuilder::new())
        .register(ScalarTypeHandler::new("Percentage", Box::new(PercentageParser)))
        .build();

    let extracted = mapper::extract(&mapping, &data).unwrap();
    let mut effects = SideEffectQueue::new();
    let q = ingest::plan_with_registry(
        &mapping,
        extracted,
        ingest::PlannerOptions::default(),
        &registry,
        &mut effects,
    )
    .unwrap();

    let rows = &q.node_batches[0].rows;
    assert_eq!(pick(rows, "a").props.get("progress"), Some(&Literal::Int(50)));
    assert_eq!(pick(rows, "b").props.get("progress"), Some(&Literal::Int(75)));
    assert_eq!(pick(rows, "c").props.get("progress"), Some(&Literal::Int(30)));
}

#[test]
fn custom_handler_with_full_typehandler_trait_also_works() {
    // For users who need more than a parser — e.g. a handler that
    // queues side effects — the regular `TypeHandler` path is still
    // available. Here we register a stub that just stamps a constant.
    #[derive(Debug)]
    struct Stamper;
    impl TypeHandler for Stamper {
        fn type_id(&self) -> TypeId {
            TypeId::new("Stamp")
        }
        fn capabilities(&self) -> Capabilities {
            Capabilities::INGEST | Capabilities::EXACT_MATCH
        }
        fn on_ingest(
            &self,
            ctx: &mut IngestCtx<'_>,
        ) -> Result<(), TypeError> {
            ctx.set_value(Literal::String("stamped".into()));
            Ok(())
        }
        fn lower(
            &self,
            _: &mut linguagraph::types::context::LowerCtx<'_>,
        ) -> Result<linguagraph::types::TypedPredicate, TypeError> {
            unreachable!()
        }
        fn emit(
            &self,
            _: &mut linguagraph::types::context::EmitCtx<'_>,
            _: &linguagraph::types::TypedPredicate,
        ) -> Result<(), TypeError> {
            unreachable!()
        }
    }

    let mapping = Mapping::from_str(
        r#"{
            "entities": [{
                "type": "Item",
                "source_path": "$.items[*]",
                "primary_key": "$.items[*].id",
                "properties": [
                    {"name": "id",   "source_path": "$.items[*].id",   "type": "Text"},
                    {"name": "tag",  "source_path": "$.items[*].tag",  "type": "Stamp"}
                ]
            }]
        }"#,
    )
    .unwrap();
    let data = json!({"items": [{"id": "x", "tag": "anything"}]});

    let registry = handlers::register_core(RegistryBuilder::new())
        .register_arc(Arc::new(Stamper))
        .build();
    let extracted = mapper::extract(&mapping, &data).unwrap();
    let mut effects = SideEffectQueue::new();
    let q = ingest::plan_with_registry(
        &mapping,
        extracted,
        ingest::PlannerOptions::default(),
        &registry,
        &mut effects,
    )
    .unwrap();
    let row = pick(&q.node_batches[0].rows, "x");
    assert_eq!(row.props.get("tag"), Some(&Literal::String("stamped".into())));
}
