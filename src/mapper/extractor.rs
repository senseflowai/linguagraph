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

/// Public entry-point — pure function over data + mapping.
pub fn extract(mapping: &Mapping, data: &Value) -> Result<Extracted, MapperError> {
    let mut entities = Vec::with_capacity(mapping.entities.len());
    for ent in &mapping.entities {
        entities.push(extract_entity(ent, data)?);
    }
    Ok(Extracted { entities })
}

fn extract_entity(ent: &EntityMapping, data: &Value) -> Result<ExtractedEntity, MapperError> {
    let source = parse(&ent.source_path)?;
    let pk_path = parse(&ent.primary_key)?;

    // Pre-parse property paths so a bad mapping fails before we walk the
    // potentially huge input. We *do not* require property paths to be
    // children of the source — see module docs. The third tuple slot
    // marks the property as `Some(_)`-typed so the extractor knows to
    // also stash the raw JSON value for it.
    let props: Vec<(String, JsonPath, Option<String>)> = ent
        .properties
        .iter()
        .map(|p| {
            Ok::<_, MapperError>((
                p.name.clone(),
                parse(&p.source_path)?,
                p.field_type.clone(),
            ))
        })
        .collect::<Result<_, _>>()?;

    let pk_field_name = derive_pk_field_name(ent);
    let source_matches = source.evaluate(data);
    let mut rows = Vec::with_capacity(source_matches.len());

    for src in source_matches {
        let pk_matches =
            resolve_value(&pk_path, &source, src.value, &src.context, data);
        let id = match pk_matches.len() {
            0 => {
                return Err(MapperError::MissingPrimaryKey {
                    label: ent.kind.clone(),
                    context: src.context.clone(),
                })
            }
            1 => Literal::from_json_any(pk_matches[0].value).ok_or_else(|| {
                MapperError::MissingPrimaryKey {
                    label: ent.kind.clone(),
                    context: src.context.clone(),
                }
            })?,
            n => {
                return Err(MapperError::AmbiguousPrimaryKey {
                    label: ent.kind.clone(),
                    count: n,
                })
            }
        };
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
                    let raw = matches[0].value.clone();
                    if let Some(v) = Literal::from_json_any(matches[0].value) {
                        properties.insert(name.clone(), v);
                    }
                    if field_type.is_some() {
                        raw_typed.insert(name.clone(), raw);
                    }
                }
                _ => {
                    let collected: Vec<Literal> = matches
                        .iter()
                        .filter_map(|m| Literal::from_json_any(m.value))
                        .collect();
                    properties.insert(name.clone(), Literal::List(collected));
                    if field_type.is_some() {
                        let raw_list: Vec<serde_json::Value> =
                            matches.iter().map(|m| m.value.clone()).collect();
                        raw_typed.insert(name.clone(), serde_json::Value::Array(raw_list));
                    }
                }
            }
        }

        rows.push(EntityRow {
            id,
            properties,
            raw_typed,
            context: src.context,
        });
    }

    Ok(ExtractedEntity {
        label: ent.kind.clone(),
        primary_key_field: pk_field_name,
        rows,
    })
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

fn parse(s: &str) -> Result<JsonPath, MapperError> {
    JsonPath::parse(s).map_err(|e| MapperError::Path { path: s.to_string(), source: e })
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

    #[test]
    fn extracts_one_row_per_array_element() {
        let mapping = Mapping {
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
    fn primary_key_outside_source_resolves_via_context() {
        // Source is the inner `source` object; PK references the parent
        // camera's id. This is the "Source" pattern from the bundled
        // example mapping.
        let mapping = Mapping {
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
    fn nested_wildcards_carry_full_context() {
        let mapping = Mapping {
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
    fn missing_primary_key_is_fatal() {
        let mapping = Mapping {
            entities: vec![ent(
                "Camera",
                "$.cameras[*]",
                "$.cameras[*].id",
                &[],
            )],
            relationships: vec![],
        };
        let data = json!({"cameras": [{"name": "no-id"}]});
        let err = extract(&mapping, &data).unwrap_err();
        assert!(matches!(err, MapperError::MissingPrimaryKey { .. }));
    }

    #[test]
    fn missing_property_is_tolerated() {
        let mapping = Mapping {
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
}
