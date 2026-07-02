//! Lift raw JSON into entity rows according to a [`Mapping`].
//!
//! Each entity declares a `source_path`. Evaluating it against the input
//! gives the *rows* of that entity, each tagged with the array indices it
//! traversed (its **context**).
//!
//! Property and primary-key paths come in two flavours:
//!
//! 1. **Inside the source.** When a property path is a strict suffix of the
//!    source path, we evaluate the suffix against the row's sub-value.
//!    This is the common case, and fastest.
//!
//! 2. **Outside the source.** When the path lives elsewhere in the
//!    document (e.g. a primary key that is the *parent* camera's `id`
//!    rather than the source object's), we evaluate the path against the
//!    root and align matches with the row by **context-prefix
//!    compatibility**: two contexts are compatible iff one is a prefix of
//!    the other. The unique compatible match becomes the property value.

use std::collections::BTreeMap;

use serde_json::Value;

use crate::ast::query::Literal;

use super::path::{walk_segments, JsonPath, Match, Segment};
use super::schema::{EntityMapping, Mapping};
use super::MapperError;

/// All entities pulled out of one input document, in declaration order.
#[derive(Debug, Clone)]
pub struct Extracted {
    pub entities: Vec<ExtractedEntity>,
}

#[derive(Debug, Clone)]
pub struct ExtractedEntity {
    pub label: String,
    pub primary_key_field: String,
    pub rows: Vec<EntityRow>,
}

#[derive(Debug, Clone)]
pub struct EntityRow {
    /// Resolved primary-key value.
    pub id: Literal,
    /// Properties (the planner adds the merge key automatically).
    pub properties: BTreeMap<String, Literal>,
    /// Raw JSON values kept alongside [`Self::properties`] for properties
    /// whose mapping has a `type` tag. The planner needs the original
    /// value (not the lossy `Literal` view) so the handler can run.
    /// Untyped properties are absent from this map.
    pub raw_typed: BTreeMap<String, serde_json::Value>,
    /// Indices captured at every `[*]` segment in the source path.
    /// The planner uses this to align relationships across entity types.
    pub context: Vec<usize>,
    /// Resolved values for relationship foreign-key paths that target
    /// this entity, keyed by the raw JSONPath string declared in the
    /// relationship's `from_key`/`to_key`. Populated only for paths that
    /// some relationship actually references; empty otherwise. Used by
    /// [`super::graph`] to resolve value-joined (FK) relationships.
    pub join_keys: BTreeMap<String, Literal>,
}

/// Two contexts are compatible iff one is a prefix of the other.
fn contexts_compatible(a: &[usize], b: &[usize]) -> bool {
    let n = a.len().min(b.len());
    a[..n] == b[..n]
}

/// One value resolved for a row — either via suffix evaluation or via
/// global lookup.
fn resolve_value<'a>(
    path: &JsonPath,
    source: &JsonPath,
    row_value: &'a Value,
    row_ctx: &[usize],
    root: &'a Value,
) -> Vec<Match<'a>> {
    if path.starts_with(source) {
        // Fast path: walk the suffix against this row's sub-tree.
        let suffix: Vec<Segment> = path.relative_to(source);
        walk_segments(&suffix, row_value, row_ctx.to_vec())
    } else {
        // Slow path: evaluate against the root and keep matches whose
        // context aligns with this row's. The mapping author is asserting
        // a relationship by sibling/parent context, not by structural
        // containment.
        path.evaluate(root)
            .into_iter()
            .filter(|m| contexts_compatible(&m.context, row_ctx))
            .collect()
    }
}

#[derive(Debug, Clone)]
enum PrimaryKeySpec {
    Path(JsonPath),
    Composite(Vec<PrimaryKeyPart>),
}

#[derive(Debug, Clone)]
enum PrimaryKeyPart {
    Literal(String),
    Path(JsonPath),
}

/// Public entry-point — pure function over data + mapping.
pub fn extract(mapping: &Mapping, data: &Value) -> Result<Extracted, MapperError> {
    let join_key_paths = join_key_paths_by_entity(mapping);
    let mut entities = Vec::with_capacity(mapping.entities.len());
    for ent in &mapping.entities {
        let keys = join_key_paths
            .get(ent.kind.as_str())
            .map(Vec::as_slice)
            .unwrap_or(&[]);
        entities.push(extract_entity(ent, data, keys)?);
    }
    Ok(Extracted { entities })
}

/// Collect, per entity `kind`, the distinct foreign-key JSONPath strings
/// that relationships reference for that entity: `from_key` when the
/// entity is the relationship's `from`, `to_key` when it is the `to`.
/// These are resolved per row so [`super::graph`] can value-join.
fn join_key_paths_by_entity(mapping: &Mapping) -> std::collections::HashMap<&str, Vec<String>> {
    let mut out: std::collections::HashMap<&str, Vec<String>> = std::collections::HashMap::new();
    for rel in &mapping.relationships {
        if let Some(from_key) = &rel.from_key {
            let entry = out.entry(rel.from.as_str()).or_default();
            if !entry.contains(from_key) {
                entry.push(from_key.clone());
            }
        }
        if let Some(to_key) = &rel.to_key {
            let entry = out.entry(rel.to.as_str()).or_default();
            if !entry.contains(to_key) {
                entry.push(to_key.clone());
            }
        }
    }
    out
}

fn extract_entity(
    ent: &EntityMapping,
    data: &Value,
    join_key_paths: &[String],
) -> Result<ExtractedEntity, MapperError> {
    let source = parse(&ent.source_path)?;
    let pk_spec = parse_primary_key_spec(&ent.primary_key)?;

    // Pre-parse relationship foreign-key paths once, mirroring how
    // property paths are pre-parsed below.
    let join_paths: Vec<(String, JsonPath)> = join_key_paths
        .iter()
        .map(|raw| Ok::<_, MapperError>((raw.clone(), parse(raw)?)))
        .collect::<Result<_, _>>()?;

    // Pre-parse property paths so a bad mapping fails before we walk the
    // potentially huge input. We *do not* require property paths to be
    // children of the source — see module docs. The third tuple slot
    // marks the property as `Some(_)`-typed so the extractor knows to
    // also stash the raw JSON value for it.
    let props: Vec<(String, JsonPath, Option<String>)> = ent
        .properties
        .iter()
        .map(|p| {
            Ok::<_, MapperError>((p.name.clone(), parse(&p.source_path)?, p.field_type.clone()))
        })
        .collect::<Result<_, _>>()?;

    let pk_field_name = derive_pk_field_name(ent);
    let source_matches = source.evaluate(data);
    let mut rows = Vec::with_capacity(source_matches.len());

    for src in source_matches {
        let id = resolve_primary_key(&pk_spec, ent, &source, &src, data)?;
        if matches!(id, Literal::Null) {
            return Err(MapperError::MissingPrimaryKey {
                label: ent.kind.clone(),
                context: src.context.clone(),
            });
        }

        let mut properties: BTreeMap<String, Literal> = BTreeMap::new();
        let mut raw_typed: BTreeMap<String, serde_json::Value> = BTreeMap::new();
        for (name, path, field_type) in &props {
            let matches = resolve_value(path, &source, src.value, &src.context, data);
            // Missing values are tolerated (the data may legitimately omit
            // a field); MERGE leaves the existing property untouched.
            // Multiple matches at the same context are rare but legal —
            // we collect them as a list literal.
            match matches.len() {
                0 => {}
                1 => {
                    let raw = normalize_property_raw_value(matches[0].value.clone(), field_type);
                    if let Some(v) = Literal::from_json_any(&raw) {
                        properties.insert(name.clone(), v);
                    }
                    if field_type.is_some() {
                        raw_typed.insert(name.clone(), raw);
                    }
                }
                _ => {
                    let raw_list: Vec<serde_json::Value> =
                        matches.iter().map(|m| m.value.clone()).collect();
                    let raw = normalize_property_raw_value(
                        serde_json::Value::Array(raw_list.clone()),
                        field_type,
                    );
                    if let Some(v) = Literal::from_json_any(&raw) {
                        properties.insert(name.clone(), v);
                    }
                    if field_type.is_some() {
                        raw_typed.insert(name.clone(), raw);
                    }
                }
            }
        }

        // Resolve relationship foreign-key paths for this row. We take
        // the first match (FK fields are single-valued); missing values
        // simply yield no join key, so such rows contribute no FK edge.
        let mut join_keys: BTreeMap<String, Literal> = BTreeMap::new();
        for (raw, path) in &join_paths {
            let matches = resolve_value(path, &source, src.value, &src.context, data);
            if let Some(first) = matches.first() {
                if let Some(v) = Literal::from_json_any(first.value) {
                    join_keys.insert(raw.clone(), v);
                }
            }
        }

        rows.push(EntityRow {
            id,
            properties,
            raw_typed,
            context: src.context,
            join_keys,
        });
    }

    Ok(ExtractedEntity {
        label: ent.kind.clone(),
        primary_key_field: pk_field_name,
        rows,
    })
}

fn normalize_property_raw_value(
    raw: serde_json::Value,
    field_type: &Option<String>,
) -> serde_json::Value {
    if !is_list_string_type(field_type.as_deref()) {
        return raw;
    }

    match raw {
        serde_json::Value::Array(items) => serde_json::Value::String(
            items
                .iter()
                .map(json_list_item_to_string)
                .collect::<Vec<_>>()
                .join(","),
        ),
        other => other,
    }
}

fn is_list_string_type(field_type: Option<&str>) -> bool {
    matches!(field_type, Some("String" | "SemanticText"))
}

fn json_list_item_to_string(value: &serde_json::Value) -> String {
    match value {
        serde_json::Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

fn resolve_primary_key(
    spec: &PrimaryKeySpec,
    ent: &EntityMapping,
    source: &JsonPath,
    src: &Match<'_>,
    data: &Value,
) -> Result<Literal, MapperError> {
    match spec {
        PrimaryKeySpec::Path(path) => {
            let pk_matches = resolve_value(path, source, src.value, &src.context, data);
            resolve_single_primary_key_value(ent, &src.context, pk_matches)
        }
        PrimaryKeySpec::Composite(parts) => {
            let mut out = String::new();
            for part in parts {
                match part {
                    PrimaryKeyPart::Literal(s) => out.push_str(s),
                    PrimaryKeyPart::Path(path) => {
                        let pk_matches = resolve_value(path, source, src.value, &src.context, data);
                        let value =
                            resolve_single_primary_key_value(ent, &src.context, pk_matches)?;
                        out.push_str(&primary_key_part_to_string(&value).ok_or_else(|| {
                            MapperError::MissingPrimaryKey {
                                label: ent.kind.clone(),
                                context: src.context.clone(),
                            }
                        })?);
                    }
                }
            }
            if out.is_empty() {
                return Err(MapperError::MissingPrimaryKey {
                    label: ent.kind.clone(),
                    context: src.context.clone(),
                });
            }
            Ok(Literal::String(out))
        }
    }
}

fn resolve_single_primary_key_value(
    ent: &EntityMapping,
    context: &[usize],
    pk_matches: Vec<Match<'_>>,
) -> Result<Literal, MapperError> {
    match pk_matches.len() {
        0 => Err(MapperError::MissingPrimaryKey {
            label: ent.kind.clone(),
            context: context.to_vec(),
        }),
        1 => Ok(Literal::String(primary_key_json_to_string(
            pk_matches[0].value,
        ))),
        n => Err(MapperError::AmbiguousPrimaryKey {
            label: ent.kind.clone(),
            count: n,
        }),
    }
}

fn primary_key_json_to_string(value: &serde_json::Value) -> String {
    match value {
        serde_json::Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

pub(crate) fn primary_key_part_to_string(value: &Literal) -> Option<String> {
    match value {
        Literal::String(s) => Some(s.clone()),
        Literal::Bool(b) => Some(b.to_string()),
        Literal::Int(i) => Some(i.to_string()),
        Literal::Float(f) if f.is_finite() => Some(f.to_string()),
        Literal::Null | Literal::List(_) | Literal::Object(_) | Literal::Float(_) => None,
    }
}

/// We use `id` as the merge-key field by default. If a property mapping
/// already uses the same JSONPath as the primary key, reuse its declared
/// name so the user's choice survives the round-trip.
fn derive_pk_field_name(ent: &EntityMapping) -> String {
    for p in &ent.properties {
        if p.source_path == ent.primary_key {
            return p.name.clone();
        }
    }
    "id".to_string()
}

fn parse_primary_key_spec(raw: &str) -> Result<PrimaryKeySpec, MapperError> {
    if !raw.contains('{') && !raw.contains('}') {
        return parse(raw).map(PrimaryKeySpec::Path);
    }

    let mut parts = Vec::new();
    let mut rest = raw;
    let mut placeholders = 0usize;

    while let Some(open) = rest.find('{') {
        if let Some(close_before_open) = rest[..open].find('}') {
            return Err(invalid_pk_format(
                raw,
                format!("unmatched '}}' at position {close_before_open}"),
            ));
        }

        if open > 0 {
            parts.push(PrimaryKeyPart::Literal(rest[..open].to_string()));
        }

        let after_open = &rest[open + 1..];
        let close = after_open
            .find('}')
            .ok_or_else(|| invalid_pk_format(raw, "unmatched '{'".to_string()))?;
        let path_raw = &after_open[..close];
        if path_raw.trim().is_empty() {
            return Err(invalid_pk_format(raw, "empty placeholder".to_string()));
        }
        if path_raw.contains('{') {
            return Err(invalid_pk_format(
                raw,
                "nested placeholders are not supported".to_string(),
            ));
        }

        parts.push(PrimaryKeyPart::Path(parse(path_raw)?));
        placeholders += 1;
        rest = &after_open[close + 1..];
    }

    if let Some(pos) = rest.find('}') {
        return Err(invalid_pk_format(
            raw,
            format!(
                "unmatched '}}' at position {}",
                raw.len() - rest.len() + pos
            ),
        ));
    }
    if !rest.is_empty() {
        parts.push(PrimaryKeyPart::Literal(rest.to_string()));
    }
    if placeholders == 0 {
        return Err(invalid_pk_format(
            raw,
            "expected at least one '{JSONPath}' placeholder".to_string(),
        ));
    }

    Ok(PrimaryKeySpec::Composite(parts))
}

fn invalid_pk_format(primary_key: &str, reason: String) -> MapperError {
    MapperError::InvalidPrimaryKeyFormat {
        primary_key: primary_key.to_string(),
        reason,
    }
}

fn parse(s: &str) -> Result<JsonPath, MapperError> {
    JsonPath::parse(s).map_err(|e| MapperError::Path {
        path: s.to_string(),
        source: e,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mapper::schema::{Mapping, PropertyMapping};
    use serde_json::json;

    fn ent(kind: &str, src: &str, pk: &str, props: &[(&str, &str)]) -> EntityMapping {
        EntityMapping {
            kind: kind.into(),
            source_path: src.into(),
            primary_key: pk.into(),
            properties: props
                .iter()
                .map(|(n, p)| PropertyMapping {
                    name: (*n).into(),
                    source_path: (*p).into(),
                    description: None,
                    field_type: None,
                })
                .collect(),
            name: None,
            description: None,
        }
    }

    fn typed_ent(kind: &str, src: &str, pk: &str, props: &[(&str, &str, &str)]) -> EntityMapping {
        EntityMapping {
            kind: kind.into(),
            source_path: src.into(),
            primary_key: pk.into(),
            properties: props
                .iter()
                .map(|(n, p, t)| PropertyMapping {
                    name: (*n).into(),
                    source_path: (*p).into(),
                    description: None,
                    field_type: Some((*t).into()),
                })
                .collect(),
            name: None,
            description: None,
        }
    }

    #[test]
    fn extracts_one_row_per_array_element() {
        let mapping = Mapping {
            source: None,
            domain: None,
            entities: vec![ent(
                "Camera",
                "$.cameras[*]",
                "$.cameras[*].id",
                &[("state", "$.cameras[*].state")],
            )],
            relationships: vec![],
        };
        let data = json!({
            "cameras": [
                {"id": "c1", "state": "active"},
                {"id": "c2", "state": "inactive"}
            ]
        });
        let out = extract(&mapping, &data).unwrap();
        assert_eq!(out.entities.len(), 1);
        let e = &out.entities[0];
        assert_eq!(e.rows.len(), 2);
        assert_eq!(e.rows[0].id, Literal::String("c1".into()));
        assert_eq!(e.rows[1].context, vec![1]);
        assert_eq!(
            e.rows[1].properties.get("state"),
            Some(&Literal::String("inactive".into()))
        );
    }

    #[test]
    fn extract_resolves_relationship_join_keys() {
        // Foreign-key paths declared on relationships are resolved per row
        // into `join_keys`, including nested paths and the `to_key` target.
        let mapping: Mapping = serde_json::from_value(json!({
            "entities": [
                {"type": "Camera", "source_path": "$.cameras[*]", "primary_key": "$.cameras[*].id",
                 "properties": [{"name": "name", "source_path": "$.cameras[*].name", "type": "Text"}]},
                {"type": "Place", "source_path": "$.places[*]", "primary_key": "$.places[*].id",
                 "properties": [{"name": "name", "source_path": "$.places[*].name", "type": "Text"}]},
                {"type": "Event", "source_path": "$.events[*]", "primary_key": "$.events[*].event_id",
                 "properties": [{"name": "etype", "source_path": "$.events[*].event_type", "type": "Text"}]}
            ],
            "relationships": [
                {"type": "INSTALLED_AT", "from": "Camera", "to": "Place",
                 "from_key": "$.cameras[*].place_id", "to_key": "$.places[*].id"},
                {"type": "CAPTURED_BY", "from": "Event", "to": "Camera",
                 "from_key": "$.events[*].origin.camera_id", "to_key": "$.cameras[*].id"}
            ]
        }))
        .unwrap();
        mapping.validate().unwrap();
        let data = json!({
            "places": [{"id": 72, "name": "Office"}],
            "cameras": [{"id": "cam-1", "name": "A", "place_id": 72}],
            "events": [{"event_id": "ev-1", "event_type": "fr",
                        "origin": {"camera_id": "cam-1", "place_id": 72}}]
        });
        let out = extract(&mapping, &data).unwrap();
        let by_label = |label: &str| out.entities.iter().find(|e| e.label == label).unwrap();

        let cam = &by_label("Camera").rows[0];
        // from_key (FK to place) and to_key (own id, target of CAPTURED_BY).
        assert_eq!(
            cam.join_keys.get("$.cameras[*].place_id"),
            Some(&Literal::Int(72))
        );
        assert_eq!(
            cam.join_keys.get("$.cameras[*].id"),
            Some(&Literal::String("cam-1".into()))
        );

        let place = &by_label("Place").rows[0];
        assert_eq!(
            place.join_keys.get("$.places[*].id"),
            Some(&Literal::Int(72))
        );

        // Nested foreign-key path resolves too.
        let ev = &by_label("Event").rows[0];
        assert_eq!(
            ev.join_keys.get("$.events[*].origin.camera_id"),
            Some(&Literal::String("cam-1".into()))
        );
    }

    #[test]
    fn primary_key_outside_source_resolves_via_context() {
        // Source is the inner `source` object; PK references the parent
        // camera's id. This is the "Source" pattern from the bundled
        // example mapping.
        let mapping = Mapping {
            source: None,
            domain: None,
            entities: vec![ent(
                "Source",
                "$.cameras[*].source",
                "$.cameras[*].id",
                &[("url", "$.cameras[*].source.url")],
            )],
            relationships: vec![],
        };
        let data = json!({
            "cameras": [
                {"id": "c1", "source": {"url": "rtsp://1"}},
                {"id": "c2", "source": {"url": "rtsp://2"}}
            ]
        });
        let out = extract(&mapping, &data).unwrap();
        let e = &out.entities[0];
        assert_eq!(e.rows.len(), 2);
        assert_eq!(e.rows[0].id, Literal::String("c1".into()));
        assert_eq!(e.rows[1].id, Literal::String("c2".into()));
    }

    #[test]
    fn numeric_primary_key_is_converted_to_string() {
        let mapping = Mapping {
            source: None,
            domain: None,
            entities: vec![ent("Camera", "$.cameras[*]", "$.cameras[*].id", &[])],
            relationships: vec![],
        };
        let data = json!({"cameras": [{"id": 100}]});

        let out = extract(&mapping, &data).unwrap();
        assert_eq!(out.entities[0].rows[0].id, Literal::String("100".into()));
    }

    #[test]
    fn nested_wildcards_carry_full_context() {
        let mapping = Mapping {
            source: None,
            domain: None,
            entities: vec![ent(
                "Module",
                "$.cameras[*].modules[*]",
                "$.cameras[*].modules[*].name",
                &[],
            )],
            relationships: vec![],
        };
        let data = json!({
            "cameras": [
                {"modules": [{"name": "a"}, {"name": "b"}]},
                {"modules": [{"name": "c"}]}
            ]
        });
        let out = extract(&mapping, &data).unwrap();
        let e = &out.entities[0];
        assert_eq!(e.rows.len(), 3);
        assert_eq!(e.rows[0].context, vec![0, 0]);
        assert_eq!(e.rows[2].context, vec![1, 0]);
    }

    #[test]
    fn composite_primary_key_formats_multiple_contextual_paths() {
        let mapping = Mapping {
            source: None,
            domain: None,
            entities: vec![ent(
                "WorkItem",
                "$[*].stationWorks[*]",
                "{$[*].id}-{$[*].stationWorks[*].id}",
                &[],
            )],
            relationships: vec![],
        };
        let data = json!([
            {
                "id": 100,
                "stationWorks": [
                    {"id": 2, "displayName": "diagnostics"},
                    {"id": 3, "displayName": "repair"}
                ]
            },
            {
                "id": 101,
                "stationWorks": [
                    {"id": 1, "displayName": "wash"}
                ]
            }
        ]);

        let out = extract(&mapping, &data).unwrap();
        let rows = &out.entities[0].rows;
        assert_eq!(rows.len(), 3);
        assert_eq!(rows[0].id, Literal::String("100-2".into()));
        assert_eq!(rows[1].id, Literal::String("100-3".into()));
        assert_eq!(rows[2].id, Literal::String("101-1".into()));
    }

    #[test]
    fn malformed_composite_primary_key_is_rejected() {
        let mapping = Mapping {
            source: None,
            domain: None,
            entities: vec![ent(
                "WorkItem",
                "$[*].stationWorks[*]",
                "{$[*].id-{$[*].stationWorks[*].id}",
                &[],
            )],
            relationships: vec![],
        };
        let data = json!([]);

        let err = extract(&mapping, &data).unwrap_err();
        assert!(matches!(err, MapperError::InvalidPrimaryKeyFormat { .. }));
    }

    #[test]
    fn missing_primary_key_is_fatal() {
        let mapping = Mapping {
            source: None,
            domain: None,
            entities: vec![ent("Camera", "$.cameras[*]", "$.cameras[*].id", &[])],
            relationships: vec![],
        };
        let data = json!({"cameras": [{"name": "no-id"}]});
        let err = extract(&mapping, &data).unwrap_err();
        assert!(matches!(err, MapperError::MissingPrimaryKey { .. }));
    }

    #[test]
    fn missing_property_is_tolerated() {
        let mapping = Mapping {
            source: None,
            domain: None,
            entities: vec![ent(
                "Camera",
                "$.cameras[*]",
                "$.cameras[*].id",
                &[("state", "$.cameras[*].state")],
            )],
            relationships: vec![],
        };
        let data = json!({"cameras": [{"id": "c1"}]});
        let out = extract(&mapping, &data).unwrap();
        assert_eq!(out.entities[0].rows[0].properties.len(), 0);
    }

    #[test]
    fn string_typed_array_property_is_joined_with_commas() {
        let mapping = Mapping {
            source: None,
            domain: None,
            entities: vec![typed_ent(
                "Item",
                "$.items[*]",
                "$.items[*].id",
                &[("tags", "$.items[*].tags", "String")],
            )],
            relationships: vec![],
        };
        let data = json!({
            "items": [
                {"id": "i1", "tags": ["red", "large", 7]}
            ]
        });

        let out = extract(&mapping, &data).unwrap();
        let row = &out.entities[0].rows[0];
        assert_eq!(
            row.properties.get("tags"),
            Some(&Literal::String("red,large,7".into()))
        );
        assert_eq!(
            row.raw_typed.get("tags"),
            Some(&serde_json::Value::String("red,large,7".into()))
        );
    }

    #[test]
    fn semantic_text_multiple_matches_are_joined_with_commas() {
        let mapping = Mapping {
            source: None,
            domain: None,
            entities: vec![typed_ent(
                "Item",
                "$.items[*]",
                "$.items[*].id",
                &[("notes", "$.items[*].notes[*]", "SemanticText")],
            )],
            relationships: vec![],
        };
        let data = json!({
            "items": [
                {"id": "i1", "notes": ["first", "second"]}
            ]
        });

        let out = extract(&mapping, &data).unwrap();
        let row = &out.entities[0].rows[0];
        assert_eq!(
            row.properties.get("notes"),
            Some(&Literal::String("first,second".into()))
        );
        assert_eq!(
            row.raw_typed.get("notes"),
            Some(&serde_json::Value::String("first,second".into()))
        );
    }
}
