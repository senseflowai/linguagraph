//! Lower an owned [`crate::graph::Graph`] into insert batches.

use std::collections::{BTreeMap, HashMap};

use serde_json::Value;

use crate::ast::query::{InsertQuery, Literal, NodeBatch, NodeRow, RelationBatch, RelationRow};
use crate::graph::{EntityGraph, EntityRef, Graph, PrimaryKey, Property, PropertyType};
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
    if opts.max_batch_size == 0 {
        return Err(IngestError::InvalidBatchSize);
    }

    let mut entity_keys: HashMap<EntityRef, EntityKey> =
        HashMap::with_capacity(graph.entities().len());
    let mut nodes_by_shape: BTreeMap<NodeShape, Vec<NodeRow>> = BTreeMap::new();

    for (idx, entity) in graph.entities().iter().enumerate() {
        let entity_ref = EntityRef::from_index(idx);
        let shape = node_shape(entity)?;
        let id = entity_id(entity, &shape.merge_on, idx)?;
        let props = lower_entity_properties(entity, &shape, &id, registry, effects)?;

        entity_keys.insert(
            entity_ref,
            EntityKey {
                label: shape.label.clone(),
                key_field: shape.merge_on.clone(),
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

        relation_batches
            .entry(RelationShape {
                rel_type: relation.r#type.clone(),
                from_label: from.label.clone(),
                from_key: from.key_field.clone(),
                to_label: to.label.clone(),
                to_key: to.key_field.clone(),
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
    id: Literal,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct NodeShape {
    label: String,
    merge_on: String,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct RelationShape {
    rel_type: String,
    from_label: String,
    from_key: String,
    to_label: String,
    to_key: String,
}

fn node_shape(entity: &EntityGraph) -> Result<NodeShape, IngestError> {
    let merge_on = match &entity.primary_key {
        Some(PrimaryKey::Strict(field)) | Some(PrimaryKey::Soft(field)) => field.clone(),
        None => return Err(IngestError::MissingGraphPrimaryKey(entity.r#type.clone())),
    };

    Ok(NodeShape {
        label: entity.r#type.clone(),
        merge_on,
    })
}

fn entity_id(entity: &EntityGraph, key_field: &str, index: usize) -> Result<Literal, IngestError> {
    match &entity.primary_key {
        Some(PrimaryKey::Strict(_)) => {
            let property = entity.properties.get(key_field).ok_or_else(|| {
                IngestError::MissingGraphPrimaryKeyValue {
                    label: entity.r#type.clone(),
                    field: key_field.to_string(),
                }
            })?;
            literal_from_json(&property.value, &property.name)
        }
        Some(PrimaryKey::Soft(_)) => match entity.properties.get(key_field) {
            Some(property) => literal_from_json(&property.value, &property.name),
            None => Ok(Literal::String(format!("{}:{index}", entity.r#type))),
        },
        None => Err(IngestError::MissingGraphPrimaryKey(entity.r#type.clone())),
    }
}

fn lower_entity_properties(
    entity: &EntityGraph,
    shape: &NodeShape,
    id: &Literal,
    registry: &TypeRegistry,
    effects: &mut SideEffectQueue,
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
    );
    handler
        .on_ingest(&mut ctx)
        .map_err(|e| IngestError::Type(e.to_string()))?;

    Ok(match ctx.finish() {
        None => Some(literal_from_json(&property.value, &property.name)?),
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
        None => literal_from_json(&property.value, &property.name)?,
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

fn literal_from_json(value: &Value, property_name: &str) -> Result<Literal, IngestError> {
    Literal::from_json_any(value).ok_or_else(|| {
        IngestError::Type(format!(
            "property '{property_name}' contains a value that cannot be represented as a Cypher parameter"
        ))
    })
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
}
