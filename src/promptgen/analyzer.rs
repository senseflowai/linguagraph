//! Walk arbitrary JSON, propose a [`JsonSchemaSummary`] the prompt
//! builder consumes.
//!
//! The walk is recursive but state-light: every leaf updates a
//! [`LeafStats`](super::inference::LeafStats) entry keyed by JSONPath,
//! and every array of objects becomes an [`EntitySummary`]. A
//! singleton root object that has scalar fields is also surfaced as
//! an entity so flat documents still get a mapping suggestion.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use super::inference::{infer, is_id_field, InferredType, LeafStats};

/// Top-level summary of the input document.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct JsonSchemaSummary {
    /// One entry per detected candidate entity (an array of objects
    /// or a top-level singleton object). Order is the document order
    /// in which the analyser encountered the entities — stable across
    /// runs so prompt output is deterministic.
    pub entities: Vec<EntitySummary>,
    /// Hints for relationships across detected entities. The prompt
    /// surfaces these as suggestions; the LLM has the final say.
    pub relationships: Vec<RelationshipHint>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntitySummary {
    /// Suggested entity label (PascalCase, singular).
    pub name: String,
    /// JSONPath that yields one row per match, e.g. `$.companies[*]`.
    pub source_path: String,
    /// JSONPath of the chosen primary key, e.g. `$.companies[*].id`.
    /// `None` when the analyzer couldn't pick one — the prompt then
    /// asks the LLM to choose.
    pub primary_key: Option<String>,
    /// Fields below the entity, ordered by JSON key for stability.
    pub fields: Vec<FieldSummary>,
    /// Number of array elements seen (or 1 for a singleton).
    pub samples: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldSummary {
    /// Property name as it should appear in the mapping.
    pub name: String,
    /// JSONPath to the value, e.g. `$.companies[*].id`.
    pub source_path: String,
    /// Inferred type. The prompt only renders the user-visible types
    /// (everything except `Identifier`, which becomes the entity's
    /// `primary_key` rather than a property).
    pub inferred_type: InferredType,
    /// Up to a handful of stringified samples so the LLM can sanity-
    /// check the heuristic (capped at 3).
    pub samples: Vec<String>,
    /// Number of distinct values across this field's observations.
    pub distinct: usize,
    /// Number of non-null values across observations.
    pub non_null: usize,
}

/// A hint the prompt surfaces under the `# Relationship hints`
/// section. Categorical: nested arrays produce `NestedEntity` hints,
/// foreign-key-like fields produce `ForeignKey` hints.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RelationshipHint {
    /// `parent` contains an array of `child` entities under
    /// `nested_path` — parent owns child.
    NestedEntity {
        parent: String,
        child: String,
        nested_path: String,
    },
    /// A field on `from` looks like it references another entity
    /// (e.g. `customer_id`).
    ForeignKey { from: String, field: String },
}

/// Public entry point.
pub fn analyze(value: &Value) -> JsonSchemaSummary {
    let mut acc = Acc::default();
    walk(&mut acc, "$", value, /*parent=*/ None);
    acc.finish()
}

// ─── Internal accumulator ──────────────────────────────────────────

#[derive(Default)]
struct Acc {
    /// Entities collected in document order, keyed by source_path so
    /// we can update the same entity if its array is encountered
    /// across multiple roots in some pathological document.
    by_path: BTreeMap<String, EntityAcc>,
    /// Insertion order of source_paths, since BTreeMap iterates by
    /// key but we want document order.
    order: Vec<String>,
    relationships: Vec<RelationshipHint>,
}

struct EntityAcc {
    name: String,
    source_path: String,
    samples: usize,
    /// Field stats keyed by the *trailing field name* (not full path),
    /// so multiple array elements feed into the same accumulator.
    fields: BTreeMap<String, FieldAcc>,
}

#[derive(Default)]
struct FieldAcc {
    source_path: String,
    stats: LeafStats,
    distinct_seen: std::collections::HashSet<String>,
    samples: Vec<String>,
}

impl Acc {
    fn finish(self) -> JsonSchemaSummary {
        let Acc {
            by_path,
            order,
            mut relationships,
        } = self;
        let mut entities = Vec::with_capacity(order.len());
        for path in order {
            let Some(mut ent) = by_path.get(&path).cloned() else {
                continue;
            };
            // Pick a primary key: prefer literal id-named fields.
            // When there's none, leave it None — the prompt nudges the
            // LLM to choose. We deliberately don't try to pick a "first
            // unique scalar" here because uniqueness across a single
            // sample document is meaningless on small inputs.
            let pk = ent
                .fields
                .iter()
                .find(|(name, _)| is_id_field(name))
                .map(|(_, fa)| fa.source_path.clone());

            // Materialise field summaries (sorted by name for
            // determinism). The Identifier-typed field is dropped
            // from the property list when it's also the chosen pk
            // — the mapping carries it via primary_key.
            let mut fields: Vec<FieldSummary> = ent
                .fields
                .iter_mut()
                .map(|(name, fa)| {
                    fa.stats.distinct = fa.distinct_seen.len();
                    let inferred = infer(name, &fa.stats);
                    FieldSummary {
                        name: name.clone(),
                        source_path: fa.source_path.clone(),
                        inferred_type: inferred,
                        samples: fa.samples.clone(),
                        distinct: fa.stats.distinct,
                        non_null: fa.stats.non_null,
                    }
                })
                .collect();
            // Drop the chosen primary-key field from `fields` — the
            // mapping already carries it as `primary_key`.
            if let Some(pk_path) = &pk {
                fields.retain(|f| &f.source_path != pk_path);
            }

            entities.push(EntitySummary {
                name: std::mem::take(&mut ent.name),
                source_path: ent.source_path.clone(),
                primary_key: pk,
                fields,
                samples: ent.samples,
            });
        }
        // Stable order for relationships too.
        relationships.sort_by(|a, b| format!("{a:?}").cmp(&format!("{b:?}")));
        JsonSchemaSummary {
            entities,
            relationships,
        }
    }
}

impl Clone for EntityAcc {
    fn clone(&self) -> Self {
        Self {
            name: self.name.clone(),
            source_path: self.source_path.clone(),
            samples: self.samples,
            fields: self.fields.clone(),
        }
    }
}

impl Clone for FieldAcc {
    fn clone(&self) -> Self {
        Self {
            source_path: self.source_path.clone(),
            stats: self.stats.clone(),
            distinct_seen: self.distinct_seen.clone(),
            samples: self.samples.clone(),
        }
    }
}

// ─── Walk ──────────────────────────────────────────────────────────

fn walk<'a>(acc: &mut Acc, path: &str, value: &'a Value, parent_entity: Option<&str>) {
    match value {
        Value::Array(items) => walk_array(acc, path, items, parent_entity),
        Value::Object(map) => walk_object(acc, path, map, parent_entity),
        _ => {} // root-scalar: nothing to map.
    }
}

fn walk_array(acc: &mut Acc, path: &str, items: &[Value], parent_entity: Option<&str>) {
    // Heuristic: an array of objects is an entity candidate. An array
    // of scalars is just a list-typed leaf at the parent (we don't
    // currently materialise list-typed fields; the prompt asks the
    // LLM to handle them).
    let entity_path = format!("{path}[*]");
    let key = path
        .rsplit('.')
        .next()
        .unwrap_or("entries")
        .trim_end_matches("[*]");
    let entity_name = entity_name_from_key(key);

    let any_object = items.iter().any(|v| v.is_object());
    if !any_object {
        return;
    }

    // Allocate or fetch the entity accumulator.
    if !acc.by_path.contains_key(&entity_path) {
        acc.by_path.insert(
            entity_path.clone(),
            EntityAcc {
                name: entity_name.clone(),
                source_path: entity_path.clone(),
                samples: 0,
                fields: BTreeMap::new(),
            },
        );
        acc.order.push(entity_path.clone());
        if let Some(parent) = parent_entity {
            acc.relationships.push(RelationshipHint::NestedEntity {
                parent: parent.to_string(),
                child: entity_name.clone(),
                nested_path: entity_path.clone(),
            });
        }
    }

    for item in items {
        if let Some(map) = item.as_object() {
            // Bump the sample count.
            if let Some(ent) = acc.by_path.get_mut(&entity_path) {
                ent.samples += 1;
            }
            walk_entity_fields(acc, &entity_path, &entity_name, map);
        }
    }
}

fn walk_object(acc: &mut Acc, path: &str, map: &Map<String, Value>, _parent: Option<&str>) {
    // A root object becomes a singleton entity; nested objects don't
    // (they'd usually duplicate a field set the parent already
    // captured). Detect "root" by checking whether any field is an
    // array — if all fields are scalars and we're at `$`, it's a
    // singleton record.
    if path == "$" {
        // Top level — recurse into arrays first; they're the
        // primary entity candidates.
        for (k, v) in map {
            walk(acc, &format!("$.{k}"), v, None);
        }
        // If after recursion no entities exist, treat the root as
        // a singleton entity.
        if acc.order.is_empty() {
            let entity_name = "Document".to_string();
            let entity_path = "$".to_string();
            acc.by_path.insert(
                entity_path.clone(),
                EntityAcc {
                    name: entity_name.clone(),
                    source_path: entity_path.clone(),
                    samples: 1,
                    fields: BTreeMap::new(),
                },
            );
            acc.order.push(entity_path.clone());
            walk_entity_fields(acc, &entity_path, &entity_name, map);
        }
        return;
    }
    // Non-root object that isn't part of an array: walk into it,
    // letting any nested arrays surface as entities of their own.
    for (k, v) in map {
        walk(acc, &format!("{path}.{k}"), v, None);
    }
}

fn walk_entity_fields(
    acc: &mut Acc,
    entity_path: &str,
    entity_name: &str,
    map: &Map<String, Value>,
) {
    for (k, v) in map {
        let field_path = format!("{}.{k}", entity_path.trim_end_matches("[*]"));
        // Strip the `[*]` suffix when the entity comes from an array
        // so the field path still references the parent row, e.g.
        // `$.companies[*].name`.
        let field_path = if entity_path.ends_with("[*]") {
            format!("{entity_path}.{k}")
        } else {
            field_path
        };

        match v {
            Value::Array(items) => {
                // Array → potential nested entity. Recurse.
                walk(acc, &format!("{entity_path}.{k}"), v, Some(entity_name));
                // Also note the array as a list-leaf so the prompt
                // mentions it; we don't observe individual scalar
                // values here.
                if items.iter().all(|x| !x.is_object()) {
                    let _ = items; // intentionally drop scalars
                }
            }
            Value::Object(_) => {
                // Nested object: recurse to capture nested arrays.
                walk(acc, &format!("{entity_path}.{k}"), v, Some(entity_name));
                // Nested objects without arrays become flattened
                // scalar fields so the LLM can still address them
                // (e.g. `address.city`). We don't currently flatten
                // — the prompt encourages the LLM to do so.
            }
            _ => {
                // Scalar leaf: feed the field accumulator.
                let ent = acc.by_path.get_mut(entity_path).expect("entity allocated");
                let fa = ent.fields.entry(k.clone()).or_insert_with(|| FieldAcc {
                    source_path: field_path.clone(),
                    ..Default::default()
                });
                fa.stats.observe(v);
                let s = scalar_repr(v);
                fa.distinct_seen.insert(s.clone());
                if fa.samples.len() < 3 {
                    fa.samples.push(s);
                }
                // Foreign-key hint.
                if k != "id" && is_id_field(k) {
                    acc.relationships.push(RelationshipHint::ForeignKey {
                        from: entity_name.to_string(),
                        field: k.clone(),
                    });
                }
            }
        }
    }
}

fn scalar_repr(v: &Value) -> String {
    match v {
        Value::Null => "null".into(),
        Value::Bool(b) => b.to_string(),
        Value::Number(n) => n.to_string(),
        Value::String(s) => s.clone(),
        _ => v.to_string(),
    }
}

/// Convert an array key like `companies` / `users` / `cameras` into a
/// best-effort PascalCase singular entity label.
///
/// We don't ship a full inflector; the prompt is allowed to override
/// our guess. The transformation only handles the boring cases (drop
/// trailing `s`, drop `ies` → `y`, `es` → ``).
fn entity_name_from_key(key: &str) -> String {
    let s = key
        .split(|c: char| c == '_' || c == '-' || c == '.')
        .filter(|p| !p.is_empty())
        .map(|p| {
            let mut chars = p.chars();
            match chars.next() {
                Some(c) => c.to_ascii_uppercase().to_string() + chars.as_str(),
                None => String::new(),
            }
        })
        .collect::<String>();
    singularize(&s)
}

fn singularize(s: &str) -> String {
    // Order matters: check the most specific suffix first.
    // "ies" → "y" (Companies → Company).
    if let Some(stem) = s.strip_suffix("ies") {
        return format!("{stem}y");
    }
    // "es" → "" when the stem ends with s/x/z/ch/sh
    // (Addresses → Address, Boxes → Box, Buzzes → Buzz, Patches → Patch,
    // Brushes → Brush). The plain "ses" branch is intentionally absent
    // — `addresses` ends in `sses`, not `ses` over `addres`.
    if let Some(stem) = s.strip_suffix("es") {
        let ends_with_chsh = stem.ends_with("ch") || stem.ends_with("sh");
        if stem.ends_with('s') || stem.ends_with('x') || stem.ends_with('z') || ends_with_chsh {
            return stem.to_string();
        }
    }
    // Trailing "s" → "" (Users → User), but not "ss" (Class → Class).
    if s.ends_with('s') && !s.ends_with("ss") {
        return s[..s.len() - 1].to_string();
    }
    s.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn flat_array_of_companies_yields_one_entity() {
        let v = json!({
            "companies": [
                {"id": 1, "name": "Stripe", "description": "Payments API",
                 "industry": "Fintech"},
                {"id": 2, "name": "Acme", "description": "Anvils that always work",
                 "industry": "Manufacturing"}
            ]
        });
        let s = analyze(&v);
        assert_eq!(s.entities.len(), 1);
        let e = &s.entities[0];
        assert_eq!(e.name, "Company");
        assert_eq!(e.source_path, "$.companies[*]");
        assert_eq!(e.primary_key.as_deref(), Some("$.companies[*].id"));
        assert_eq!(e.samples, 2);

        let by_name: BTreeMap<&str, &FieldSummary> =
            e.fields.iter().map(|f| (f.name.as_str(), f)).collect();
        // `id` was promoted to primary_key, so it's not in fields.
        assert!(!by_name.contains_key("id"));
        assert_eq!(
            by_name["description"].inferred_type,
            InferredType::SemanticText
        );
        assert_eq!(by_name["industry"].inferred_type, InferredType::Keyword);
        // `name` is short and not in any name-hint list — stays as
        // Text rather than SemanticText.
        assert_eq!(by_name["name"].inferred_type, InferredType::Text);
    }

    #[test]
    fn singleton_root_object_becomes_document_entity() {
        let v = json!({"id": "x", "title": "hello"});
        let s = analyze(&v);
        assert_eq!(s.entities.len(), 1);
        assert_eq!(s.entities[0].name, "Document");
        assert_eq!(s.entities[0].source_path, "$");
        assert_eq!(s.entities[0].primary_key.as_deref(), Some("$.id"));
    }

    #[test]
    fn nested_arrays_produce_relationship_hints() {
        let v = json!({
            "users": [
                {"id": 1, "name": "a", "posts": [{"id": 10, "title": "t"}]}
            ]
        });
        let s = analyze(&v);
        assert_eq!(s.entities.len(), 2);
        let names: Vec<&str> = s.entities.iter().map(|e| e.name.as_str()).collect();
        assert!(names.contains(&"User"));
        assert!(names.contains(&"Post"));
        assert!(s.relationships.iter().any(|r| matches!(r,
            RelationshipHint::NestedEntity { parent, child, .. }
                if parent == "User" && child == "Post"
        )));
    }

    #[test]
    fn foreign_key_field_emits_relationship_hint() {
        let v = json!({
            "orders": [
                {"id": 1, "customer_id": 7, "total": 100},
                {"id": 2, "customer_id": 8, "total": 50}
            ]
        });
        let s = analyze(&v);
        assert!(s.relationships.iter().any(|r| matches!(r,
            RelationshipHint::ForeignKey { from, field }
                if from == "Order" && field == "customer_id"
        )));
    }

    #[test]
    fn singularization_basics() {
        assert_eq!(entity_name_from_key("companies"), "Company");
        assert_eq!(entity_name_from_key("users"), "User");
        assert_eq!(entity_name_from_key("addresses"), "Address");
        assert_eq!(entity_name_from_key("data"), "Data"); // safe no-op
        assert_eq!(entity_name_from_key("video_streams"), "VideoStream");
    }

    #[test]
    fn no_panic_on_empty_or_scalar_root() {
        let s = analyze(&json!(null));
        assert!(s.entities.is_empty());
        let s = analyze(&json!(42));
        assert!(s.entities.is_empty());
        let s = analyze(&json!([]));
        assert!(s.entities.is_empty());
    }
}
