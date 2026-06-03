//! Lower an owned [`crate::graph::Graph`] into insert batches.

use std::collections::{BTreeMap, HashMap};

use serde_json::Value;

use crate::ast::query::{InsertQuery, Literal, NodeBatch, NodeRow, RelationBatch, RelationRow};
use crate::graph::{
    EntityGraph, EntityRef, Graph, PrimaryKey, Property, PropertyType, CANONICAL_FIELD,
};
use crate::types::context::IngestCtx;
use crate::types::handlers::SemanticTextHandler;
use crate::types::{SideEffectQueue, TypeId, TypeRegistry};

use super::{IngestError, PlannerOptions};

pub fn plan_graph_with_registry(
    graph: &Graph,
    opts: PlannerOptions,
    registry: &TypeRegistry,
    effects: &mut SideEffectQueue,
) -> Result<InsertQuery, IngestError> {
    plan_graph_with_registry_and_prefix(graph, opts, registry, effects, None)
}

/// Like [`plan_graph_with_registry`] but also stamps every emitted
/// node and relation batch with `prefix_label`. When set, the resulting
/// Cypher carries the prefix as an extra Cypher label on every MERGE
/// pattern, so entities only merge with their same-prefix siblings.
pub fn plan_graph_with_registry_and_prefix(
    graph: &Graph,
    opts: PlannerOptions,
    registry: &TypeRegistry,
    effects: &mut SideEffectQueue,
    prefix_label: Option<&str>,
) -> Result<InsertQuery, IngestError> {
    plan_graph_with_registry_and_prefixes(graph, opts, registry, effects, prefix_label, None)
}

/// Like [`plan_graph_with_registry_and_prefix`] but additionally folds
/// `prefix_index` into every embedding-index / Qdrant collection name a
/// type handler emits during ingestion. The two prefixes are
/// independent: `prefix_label` scopes the Memgraph data; `prefix_index`
/// scopes the vector store. Most callers pass the same value for both.
pub fn plan_graph_with_registry_and_prefixes(
    graph: &Graph,
    opts: PlannerOptions,
    registry: &TypeRegistry,
    effects: &mut SideEffectQueue,
    prefix_label: Option<&str>,
    prefix_index: Option<&str>,
) -> Result<InsertQuery, IngestError> {
    if opts.max_batch_size == 0 {
        return Err(IngestError::InvalidBatchSize);
    }
    let prefix_label = prefix_label
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_string);
    let prefix_index = prefix_index
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_string);

    let mut entity_keys: HashMap<EntityRef, EntityKey> =
        HashMap::with_capacity(graph.entities().len());
    let mut nodes_by_shape: BTreeMap<NodeShape, Vec<NodeRow>> = BTreeMap::new();

    for (idx, entity) in graph.entities().iter().enumerate() {
        let entity_ref = EntityRef::from_index(idx);
        let shape = node_shape(entity)?;
        let id = entity_id(entity, &shape.merge_on, idx)?;
        let props = lower_entity_properties(
            entity,
            &shape,
            &id,
            registry,
            effects,
            prefix_index.as_deref(),
        )?;

        entity_keys.insert(
            entity_ref,
            EntityKey {
                label: shape.label.clone(),
                key_field: shape.merge_on.clone(),
                domain: shape.domain.clone(),
                id: id.clone(),
            },
        );
        nodes_by_shape
            .entry(shape)
            .or_default()
            .push(NodeRow { id, props });
    }

    let mut relation_batches: BTreeMap<RelationShape, Vec<RelationRow>> = BTreeMap::new();
    for relation in graph.relations() {
        let from = entity_keys
            .get(&relation.from)
            .ok_or_else(|| IngestError::UnknownGraphEntityRef(relation.from.index()))?;
        let to = entity_keys
            .get(&relation.to)
            .ok_or_else(|| IngestError::UnknownGraphEntityRef(relation.to.index()))?;
        let props = lower_relation_properties(&relation.properties, registry)?;

        // A relation gets a domain label only when both endpoints share
        // the same domain — otherwise we can't safely scope the MATCH.
        let rel_domain = match (&from.domain, &to.domain) {
            (Some(a), Some(b)) if a == b => Some(a.clone()),
            _ => None,
        };

        relation_batches
            .entry(RelationShape {
                rel_type: relation.r#type.clone(),
                from_label: from.label.clone(),
                from_key: from.key_field.clone(),
                to_label: to.label.clone(),
                to_key: to.key_field.clone(),
                domain: rel_domain,
            })
            .or_default()
            .push(RelationRow {
                from_id: from.id.clone(),
                to_id: to.id.clone(),
                props,
            });
    }

    let node_batches = nodes_by_shape
        .into_iter()
        .flat_map(|(shape, mut rows)| {
            rows.sort_by(|a, b| literal_cmp(&a.id, &b.id));
            rows.chunks(opts.max_batch_size)
                .map(|chunk| NodeBatch {
                    label: shape.label.clone(),
                    merge_on: shape.merge_on.clone(),
                    prefix_label: prefix_label.clone(),
                    domain_label: shape.domain.clone(),
                    rows: chunk.to_vec(),
                })
                .collect::<Vec<_>>()
        })
        .collect();

    let relation_batches = relation_batches
        .into_iter()
        .flat_map(|(shape, mut rows)| {
            rows.sort_by(|a, b| {
                literal_cmp(&a.from_id, &b.from_id).then(literal_cmp(&a.to_id, &b.to_id))
            });
            rows.chunks(opts.max_batch_size)
                .map(|chunk| RelationBatch {
                    rel_type: shape.rel_type.clone(),
                    from_label: shape.from_label.clone(),
                    from_key: shape.from_key.clone(),
                    to_label: shape.to_label.clone(),
                    to_key: shape.to_key.clone(),
                    prefix_label: prefix_label.clone(),
                    domain_label: shape.domain.clone(),
                    rows: chunk.to_vec(),
                })
                .collect::<Vec<_>>()
        })
        .collect();

    Ok(InsertQuery {
        node_batches,
        relation_batches,
    })
}

#[derive(Debug, Clone)]
struct EntityKey {
    label: String,
    key_field: String,
    domain: Option<String>,
    id: Literal,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct NodeShape {
    label: String,
    merge_on: String,
    domain: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct RelationShape {
    rel_type: String,
    from_label: String,
    from_key: String,
    to_label: String,
    to_key: String,
    domain: Option<String>,
}

fn node_shape(entity: &EntityGraph) -> Result<NodeShape, IngestError> {
    let merge_on = match &entity.primary_key {
        Some(PrimaryKey::Strict(field)) => field.clone(),
        Some(PrimaryKey::Soft) => CANONICAL_FIELD.to_string(),
        None => return Err(IngestError::MissingGraphPrimaryKey(entity.r#type.clone())),
    };

    Ok(NodeShape {
        label: entity.r#type.clone(),
        merge_on,
        domain: entity.domain.clone(),
    })
}

fn entity_id(entity: &EntityGraph, key_field: &str, _index: usize) -> Result<Literal, IngestError> {
    match &entity.primary_key {
        Some(PrimaryKey::Strict(_)) => {
            let property = entity.properties.get(key_field).ok_or_else(|| {
                IngestError::MissingGraphPrimaryKeyValue {
                    label: entity.r#type.clone(),
                    field: key_field.to_string(),
                }
            })?;
            literal_from_json(
                &property.value,
                &property.name,
                Some(property.property_type),
            )
        }
        // Soft used to fall back to a synthetic "<Type>:<index>" id
        // when the key field was missing. The soft-merge resolver
        // (src/ingest/soft_merge.rs) now runs before this planner and
        // either rewrites the property to a canonical value or leaves
        // it untouched; either way the property must be present by
        // the time we reach the planner. Treat a missing value the
        // same way as Strict so callers get a clean, typed error
        // instead of a silent placeholder id that would never merge
        // with anything.
        Some(PrimaryKey::Soft) => {
            let property = entity.properties.get(key_field).ok_or_else(|| {
                IngestError::MissingGraphPrimaryKeyValue {
                    label: entity.r#type.clone(),
                    field: key_field.to_string(),
                }
            })?;
            literal_from_json(
                &property.value,
                &property.name,
                Some(property.property_type),
            )
        }
        None => Err(IngestError::MissingGraphPrimaryKey(entity.r#type.clone())),
    }
}

fn lower_entity_properties(
    entity: &EntityGraph,
    shape: &NodeShape,
    id: &Literal,
    registry: &TypeRegistry,
    effects: &mut SideEffectQueue,
    prefix_index: Option<&str>,
) -> Result<BTreeMap<String, Literal>, IngestError> {
    let mut out = BTreeMap::new();
    for property in entity.properties.values() {
        if let Some(lit) = lower_node_property(
            &shape.label,
            &shape.merge_on,
            id,
            property,
            registry,
            effects,
            prefix_index,
        )? {
            out.insert(property.name.clone(), lit);
        }
    }
    Ok(out)
}

fn lower_node_property(
    label: &str,
    key_field: &str,
    key_value: &Literal,
    property: &Property,
    registry: &TypeRegistry,
    effects: &mut SideEffectQueue,
    prefix_index: Option<&str>,
) -> Result<Option<Literal>, IngestError> {
    let type_id = node_type_id(property.property_type);
    let handler = registry
        .get(&TypeId::new(type_id))
        .map_err(|e| IngestError::Type(e.to_string()))?;

    let mut ctx = IngestCtx::new(
        label,
        key_field,
        key_value,
        &property.name,
        &property.value,
        effects,
    )
    .with_prefix_index(prefix_index);
    handler
        .on_ingest(&mut ctx)
        .map_err(|e| IngestError::Type(e.to_string()))?;

    Ok(match ctx.finish() {
        None => Some(literal_from_json(
            &property.value,
            &property.name,
            Some(property.property_type),
        )?),
        Some(Some(lit)) => Some(lit),
        Some(None) => None,
    })
}

fn lower_relation_properties(
    properties: &HashMap<String, Property>,
    registry: &TypeRegistry,
) -> Result<BTreeMap<String, Literal>, IngestError> {
    let mut out = BTreeMap::new();
    for property in properties.values() {
        let lit = lower_relation_property(property, registry)?;
        out.insert(property.name.clone(), lit);
    }
    Ok(out)
}

fn lower_relation_property(
    property: &Property,
    registry: &TypeRegistry,
) -> Result<Literal, IngestError> {
    let type_id = relation_type_id(property.property_type);
    let handler = registry
        .get(&TypeId::new(type_id))
        .map_err(|e| IngestError::Type(e.to_string()))?;
    let mut effects = SideEffectQueue::new();
    let key = Literal::String("__relationship__".to_string());
    let mut ctx = IngestCtx::new(
        "__Relationship",
        "id",
        &key,
        &property.name,
        &property.value,
        &mut effects,
    );
    handler
        .on_ingest(&mut ctx)
        .map_err(|e| IngestError::Type(e.to_string()))?;
    let lowered = ctx.finish();
    if !effects.is_empty() {
        return Err(IngestError::Type(
            "relationship properties cannot produce node-scoped side effects".into(),
        ));
    }
    Ok(match lowered {
        None => literal_from_json(
            &property.value,
            &property.name,
            Some(property.property_type),
        )?,
        Some(Some(lit)) => lit,
        Some(None) => Literal::Null,
    })
}

fn node_type_id(property_type: PropertyType) -> &'static str {
    match property_type {
        PropertyType::String => "Text",
        PropertyType::Text => SemanticTextHandler::TYPE_ID,
        PropertyType::Number => "Number",
        PropertyType::Boolean => "Boolean",
        PropertyType::DateTime | PropertyType::Timestamp => "Timestamp",
    }
}

fn relation_type_id(property_type: PropertyType) -> &'static str {
    match property_type {
        PropertyType::String | PropertyType::Text => "Text",
        PropertyType::Number => "Number",
        PropertyType::Boolean => "Boolean",
        PropertyType::DateTime | PropertyType::Timestamp => "Timestamp",
    }
}

fn literal_from_json(
    value: &Value,
    property_name: &str,
    property_type: Option<PropertyType>,
) -> Result<Literal, IngestError> {
    if let Some(property_type) = property_type {
        return literal_from_json_as_type(value, property_name, property_type);
    }

    literal_from_json_untyped(value, property_name)
}

fn literal_from_json_as_type(
    value: &Value,
    property_name: &str,
    property_type: PropertyType,
) -> Result<Literal, IngestError> {
    match property_type {
        PropertyType::String | PropertyType::Text => Ok(Literal::String(json_to_string(value))),
        PropertyType::Number => json_to_number(value, property_name),
        PropertyType::Boolean => json_to_bool(value, property_name).map(Literal::Bool),
        PropertyType::DateTime | PropertyType::Timestamp => match value {
            Value::Null => Ok(Literal::Null),
            Value::String(s) => Ok(Literal::String(s.clone())),
            Value::Number(n) => Ok(Literal::String(n.to_string())),
            other => Err(type_conversion_error(
                property_name,
                property_type,
                json_kind(other),
            )),
        },
    }
}

fn literal_from_json_untyped(value: &Value, property_name: &str) -> Result<Literal, IngestError> {
    Literal::from_json_any(value).ok_or_else(|| {
        IngestError::Type(format!(
            "property '{property_name}' contains a value that cannot be represented as a Cypher parameter"
        ))
    })
}

fn json_to_string(value: &Value) -> String {
    match value {
        Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

fn json_to_number(value: &Value, property_name: &str) -> Result<Literal, IngestError> {
    match value {
        Value::Null => Ok(Literal::Null),
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Ok(Literal::Int(i))
            } else if let Some(f) = n.as_f64() {
                if f.is_finite() {
                    Ok(Literal::Float(f))
                } else {
                    Err(type_conversion_error(
                        property_name,
                        PropertyType::Number,
                        "non-finite number",
                    ))
                }
            } else {
                Err(type_conversion_error(
                    property_name,
                    PropertyType::Number,
                    "unsupported number",
                ))
            }
        }
        Value::String(s) => {
            let trimmed = s.trim();
            if let Ok(i) = trimmed.parse::<i64>() {
                Ok(Literal::Int(i))
            } else if let Ok(f) = trimmed.parse::<f64>() {
                if f.is_finite() {
                    Ok(Literal::Float(f))
                } else {
                    Err(type_conversion_error(
                        property_name,
                        PropertyType::Number,
                        "non-finite number",
                    ))
                }
            } else {
                Err(type_conversion_error(
                    property_name,
                    PropertyType::Number,
                    "string",
                ))
            }
        }
        other => Err(type_conversion_error(
            property_name,
            PropertyType::Number,
            json_kind(other),
        )),
    }
}

fn json_to_bool(value: &Value, property_name: &str) -> Result<bool, IngestError> {
    match value {
        Value::Bool(b) => Ok(*b),
        Value::String(s) => match s.trim().to_ascii_lowercase().as_str() {
            "true" | "yes" | "y" | "1" | "on" => Ok(true),
            "false" | "no" | "n" | "0" | "off" => Ok(false),
            _ => Err(type_conversion_error(
                property_name,
                PropertyType::Boolean,
                "string",
            )),
        },
        other => Err(type_conversion_error(
            property_name,
            PropertyType::Boolean,
            json_kind(other),
        )),
    }
}

fn type_conversion_error(
    property_name: &str,
    property_type: PropertyType,
    actual: &str,
) -> IngestError {
    IngestError::Type(format!(
        "property '{property_name}' with type {property_type:?} cannot be converted from {actual}"
    ))
}

fn json_kind(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "bool",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}

fn literal_cmp(a: &Literal, b: &Literal) -> std::cmp::Ordering {
    use std::cmp::Ordering::*;
    match (a, b) {
        (Literal::String(x), Literal::String(y)) => x.cmp(y),
        (Literal::Int(x), Literal::Int(y)) => x.cmp(y),
        (Literal::Bool(x), Literal::Bool(y)) => x.cmp(y),
        (Literal::Null, Literal::Null) => Equal,
        (a, b) => format!("{a:?}").cmp(&format!("{b:?}")),
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use crate::embeddings::mock::MockEmbedder;
    use crate::graph::{GraphBuilder, PropertyType};
    use crate::types::handlers::{self, SemanticTextConfig, SemanticTextHandler};
    use crate::types::RegistryBuilder;
    use serde_json::json;

    use super::*;

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

    #[test]
    fn graph_text_properties_are_semantic_text() {
        let mut graph = GraphBuilder::new();
        let alice = graph
            .entity("Person")
            .strict_primary_key("id")
            .property("id", PropertyType::String, "a")
            .property("name", PropertyType::Text, "Alice")
            .add();
        let bob = graph
            .entity("Person")
            .strict_primary_key("id")
            .property("id", PropertyType::String, "b")
            .property("name", PropertyType::Text, "Bob")
            .add();
        graph
            .relationship(alice, "KNOWS", bob)
            .property("since", PropertyType::Number, 2024)
            .add()
            .unwrap();

        let mut effects = SideEffectQueue::new();
        let insert = plan_graph_with_registry(
            &graph.build(),
            PlannerOptions {
                max_batch_size: 1000,
            },
            &registry(),
            &mut effects,
        )
        .unwrap();

        assert_eq!(insert.node_batches.len(), 1);
        assert_eq!(insert.node_batches[0].rows.len(), 2);
        assert_eq!(insert.relation_batches.len(), 1);
        assert_eq!(
            insert.relation_batches[0].rows[0].props["since"],
            Literal::Int(2024)
        );
        assert_eq!(effects.len(), 2);
    }

    #[test]
    fn source_rooted_builder_lowers_with_mention_edges() {
        // GraphBuilder::with_source minted Source + auto-attached
        // `:mention` edges should land in the InsertQuery as a Source
        // node batch and a MENTION relation batch.
        use crate::graph::{MENTION_REL, PART_OF_REL, SOURCE_LABEL};
        let mut graph = GraphBuilder::with_source("Manual.pdf");
        graph
            .entity("Person")
            .strict_primary_key("id")
            .property("id", PropertyType::String, "alice")
            .property("name", PropertyType::Text, "Alice")
            .add();
        graph.chunk("Alice met Bob.").add().unwrap();

        let mut effects = SideEffectQueue::new();
        let insert = plan_graph_with_registry(
            &graph.build(),
            PlannerOptions::default(),
            &registry(),
            &mut effects,
        )
        .unwrap();

        let node_labels: Vec<&str> = insert
            .node_batches
            .iter()
            .map(|b| b.label.as_str())
            .collect();
        assert!(node_labels.contains(&SOURCE_LABEL));
        assert!(node_labels.contains(&"Person"));
        assert!(node_labels.contains(&"Chunk"));

        let rel_types: Vec<&str> = insert
            .relation_batches
            .iter()
            .map(|b| b.rel_type.as_str())
            .collect();
        assert!(rel_types.contains(&MENTION_REL));
        assert!(rel_types.contains(&PART_OF_REL));

        // 3 PropertyType::Text fields → 3 embedding side effects:
        // Source.name, Person.name, Chunk.text.
        assert_eq!(effects.len(), 3);
    }

    #[test]
    fn prefix_label_is_propagated_to_every_batch() {
        let mut graph = GraphBuilder::new();
        let a = graph
            .entity("Person")
            .strict_primary_key("id")
            .property("id", PropertyType::String, "a")
            .add();
        let b = graph
            .entity("Person")
            .strict_primary_key("id")
            .property("id", PropertyType::String, "b")
            .add();
        graph.relationship(a, "KNOWS", b).add().unwrap();

        let mut effects = SideEffectQueue::new();
        let insert = plan_graph_with_registry_and_prefix(
            &graph.build(),
            PlannerOptions::default(),
            &registry(),
            &mut effects,
            Some("Tenant1"),
        )
        .unwrap();

        for batch in &insert.node_batches {
            assert_eq!(batch.prefix_label.as_deref(), Some("Tenant1"));
        }
        for batch in &insert.relation_batches {
            assert_eq!(batch.prefix_label.as_deref(), Some("Tenant1"));
        }
    }

    #[test]
    fn prefix_index_scopes_embedding_side_effects() {
        // SemanticText fields produce embedding side effects; the
        // ingest-side prefix_index must land in every queued
        // collection name so vectors don't collide across prefixes.
        use crate::types::SideEffect;
        let mut graph = GraphBuilder::new();
        graph
            .entity("Person")
            .strict_primary_key("id")
            .property("id", PropertyType::String, "a")
            .property("name", PropertyType::Text, "Alice")
            .add();

        let mut effects = SideEffectQueue::new();
        plan_graph_with_registry_and_prefixes(
            &graph.build(),
            PlannerOptions::default(),
            &registry(),
            &mut effects,
            None,
            Some("Tenant1"),
        )
        .unwrap();

        assert_eq!(effects.len(), 1);
        match &effects.into_vec()[0] {
            SideEffect::EmbedAndStore { collection, .. } => {
                assert!(
                    collection.starts_with("Tenant1__"),
                    "expected prefix in collection name, got {collection}"
                );
            }
        }
    }

    #[test]
    fn empty_prefix_label_is_normalised_to_none() {
        let mut graph = GraphBuilder::new();
        graph
            .entity("Person")
            .strict_primary_key("id")
            .property("id", PropertyType::String, "a")
            .add();

        let insert = plan_graph_with_registry_and_prefix(
            &graph.build(),
            PlannerOptions::default(),
            &registry(),
            &mut SideEffectQueue::new(),
            Some("   "),
        )
        .unwrap();
        assert!(insert.node_batches[0].prefix_label.is_none());
    }

    #[test]
    fn strict_primary_key_value_is_required() {
        let mut graph = GraphBuilder::new();
        graph.entity("Person").strict_primary_key("id").add();

        let err = plan_graph_with_registry(
            &graph.build(),
            PlannerOptions::default(),
            &registry(),
            &mut SideEffectQueue::new(),
        )
        .unwrap_err();

        assert!(matches!(
            err,
            IngestError::MissingGraphPrimaryKeyValue { .. }
        ));
    }

    #[test]
    fn string_primary_key_converts_numeric_json_to_string() {
        let mut graph = GraphBuilder::new();
        graph
            .entity("Person")
            .strict_primary_key("id")
            .property("id", PropertyType::String, 100)
            .add();

        let insert = plan_graph_with_registry(
            &graph.build(),
            PlannerOptions::default(),
            &registry(),
            &mut SideEffectQueue::new(),
        )
        .unwrap();

        assert_eq!(
            insert.node_batches[0].rows[0].id,
            Literal::String("100".into())
        );
    }

    #[test]
    fn literal_from_json_uses_requested_property_type() {
        assert_eq!(
            literal_from_json(&json!("42"), "count", Some(PropertyType::Number)).unwrap(),
            Literal::Int(42)
        );
        assert_eq!(
            literal_from_json(&json!("yes"), "active", Some(PropertyType::Boolean)).unwrap(),
            Literal::Bool(true)
        );
        assert_eq!(
            literal_from_json(&json!(["a", 1]), "label", Some(PropertyType::String)).unwrap(),
            Literal::String(r#"["a",1]"#.into())
        );
    }
}
