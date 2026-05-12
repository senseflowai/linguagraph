//! Convert mapping extraction output into an owned [`crate::graph::Graph`].

use std::collections::{BTreeMap, HashMap};

use serde_json::Value;

use crate::ast::query::Literal;
use crate::graph::{EntityGraph, EntityRef, Graph, GraphBuilder, GraphSpecification, PropertyType};

use super::{extract, Extracted, MapperError, Mapping};

/// Result of mapping raw JSON into graph-native structures.
#[derive(Debug, Clone, PartialEq)]
pub struct MappedGraph {
    pub graph: Graph,
    pub specification: GraphSpecification,
}

impl MappedGraph {
    pub fn into_parts(self) -> (Graph, GraphSpecification) {
        (self.graph, self.specification)
    }
}

/// Build an ingestion [`Graph`] and [`GraphSpecification`] from raw JSON
/// plus a mapping document.
///
/// This is the bridge from the mapper layer to the graph-only ingestion
/// pipeline. It preserves the mapper's existing extraction behavior:
///
/// * entity rows are selected from each mapping `source_path`,
/// * duplicate primary keys are collapsed with last-write-wins properties,
/// * missing primary-key properties are added to each graph entity,
/// * relationship endpoints are inferred from extraction context prefixes.
pub fn to_graph(mapping: &Mapping, data: &Value) -> Result<MappedGraph, MapperError> {
    let extracted = extract(mapping, data)?;
    let graph = graph_from_extracted(mapping, &extracted)?;
    let specification = specification_from_mapping(mapping)?;
    Ok(MappedGraph {
        graph,
        specification,
    })
}

fn graph_from_extracted(mapping: &Mapping, extracted: &Extracted) -> Result<Graph, MapperError> {
    let mut builder = GraphBuilder::new();
    let mut refs: HashMap<(String, Literal), EntityRef> = HashMap::new();

    for ent in &extracted.entities {
        let entity_mapping = mapping
            .entities
            .iter()
            .find(|candidate| candidate.kind == ent.label)
            .ok_or_else(|| MapperError::UnknownEntityType(ent.label.clone()))?;
        let property_types = property_types_by_name(entity_mapping)?;

        let mut order: Vec<Literal> = Vec::new();
        let mut rows: HashMap<Literal, BTreeMap<String, (PropertyType, Value)>> = HashMap::new();

        for row in &ent.rows {
            if !rows.contains_key(&row.id) {
                order.push(row.id.clone());
            }
            let properties = rows.entry(row.id.clone()).or_default();
            for property_mapping in &entity_mapping.properties {
                let Some(property_type) = property_types.get(&property_mapping.name).copied()
                else {
                    continue;
                };
                if let Some(raw) = row.raw_typed.get(&property_mapping.name) {
                    properties.insert(property_mapping.name.clone(), (property_type, raw.clone()));
                } else if let Some(literal) = row.properties.get(&property_mapping.name) {
                    properties.insert(
                        property_mapping.name.clone(),
                        (property_type, literal_to_json(literal)),
                    );
                }
            }

            properties
                .entry(ent.primary_key_field.clone())
                .or_insert_with(|| (PropertyType::String, literal_to_json(&row.id)));
        }

        for id in order {
            let mut entity = EntityGraph::new(ent.label.clone())
                .strict_primary_key(ent.primary_key_field.clone());
            if let Some(properties) = rows.remove(&id) {
                for (name, (property_type, value)) in properties {
                    entity = entity.property(name, property_type, value);
                }
            }

            let entity_ref = builder.add_entity(entity);
            refs.insert((ent.label.clone(), id), entity_ref);
        }
    }

    let by_label: HashMap<&str, _> = extracted
        .entities
        .iter()
        .map(|entity| (entity.label.as_str(), entity))
        .collect();

    for rel in &mapping.relationships {
        let from = by_label.get(rel.from.as_str()).ok_or_else(|| {
            MapperError::UnknownRelationshipEndpoint {
                label: rel.kind.clone(),
                missing: rel.from.clone(),
            }
        })?;
        let to = by_label.get(rel.to.as_str()).ok_or_else(|| {
            MapperError::UnknownRelationshipEndpoint {
                label: rel.kind.clone(),
                missing: rel.to.clone(),
            }
        })?;

        for from_row in &from.rows {
            for to_row in &to.rows {
                if !contexts_align(&from_row.context, &to_row.context) {
                    continue;
                }
                let from_ref = *refs
                    .get(&(from.label.clone(), from_row.id.clone()))
                    .ok_or_else(|| MapperError::UnknownRelationshipEndpoint {
                        label: rel.kind.clone(),
                        missing: from.label.clone(),
                    })?;
                let to_ref = *refs
                    .get(&(to.label.clone(), to_row.id.clone()))
                    .ok_or_else(|| MapperError::UnknownRelationshipEndpoint {
                        label: rel.kind.clone(),
                        missing: to.label.clone(),
                    })?;
                builder
                    .relationship(from_ref, rel.kind.clone(), to_ref)
                    .add()
                    .map_err(|e| MapperError::Graph(e.to_string()))?;
            }
        }
    }

    Ok(builder.build())
}

fn property_types_by_name(
    entity_mapping: &super::EntityMapping,
) -> Result<HashMap<String, PropertyType>, MapperError> {
    entity_mapping
        .properties
        .iter()
        .map(|property| {
            Ok((
                property.name.clone(),
                graph_property_type(property.type_name())?,
            ))
        })
        .collect()
}

fn specification_from_mapping(mapping: &Mapping) -> Result<GraphSpecification, MapperError> {
    let mut spec = GraphSpecification::new();
    for entity in &mapping.entities {
        spec.add_entity(
            entity.kind.clone(),
            entity
                .description
                .clone()
                .or_else(|| entity.name.clone())
                .unwrap_or_default(),
        );

        let pk_name = primary_key_property_name(entity);
        if !entity
            .properties
            .iter()
            .any(|property| property.name == pk_name)
        {
            spec.add_property(
                entity.kind.clone(),
                pk_name,
                PropertyType::String,
                "Primary key.",
            );
        }

        for property in &entity.properties {
            spec.add_property(
                entity.kind.clone(),
                property.name.clone(),
                graph_property_type(property.type_name())?,
                property.description.clone().unwrap_or_default(),
            );
        }
    }
    Ok(spec)
}

fn primary_key_property_name(entity: &super::EntityMapping) -> String {
    entity
        .properties
        .iter()
        .find(|property| property.source_path == entity.primary_key)
        .map(|property| property.name.clone())
        .unwrap_or_else(|| "id".to_string())
}

fn graph_property_type(type_name: &str) -> Result<PropertyType, MapperError> {
    match type_name {
        "String" => Ok(PropertyType::String),
        "Text" | "SemanticText" => Ok(PropertyType::Text),
        "Number" => Ok(PropertyType::Number),
        "Boolean" => Ok(PropertyType::Boolean),
        "DateTime" => Ok(PropertyType::DateTime),
        "Date" | "Timestamp" => Ok(PropertyType::Timestamp),
        other => Err(MapperError::UnknownPropertyType(other.to_string())),
    }
}

fn contexts_align(a: &[usize], b: &[usize]) -> bool {
    let n = a.len().min(b.len());
    a[..n] == b[..n]
}

fn literal_to_json(literal: &Literal) -> Value {
    match literal {
        Literal::String(s) => Value::String(s.clone()),
        Literal::Bool(b) => Value::Bool(*b),
        Literal::Int(i) => Value::Number((*i).into()),
        Literal::Float(f) => serde_json::Number::from_f64(*f)
            .map(Value::Number)
            .unwrap_or(Value::Null),
        Literal::List(items) => Value::Array(items.iter().map(literal_to_json).collect()),
        Literal::Object(map) => Value::Object(
            map.iter()
                .map(|(key, value)| (key.clone(), literal_to_json(value)))
                .collect(),
        ),
        Literal::Null => Value::Null,
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::mapper::Mapping;

    #[test]
    fn builds_graph_from_mapping_and_data() {
        let mapping: Mapping = serde_json::from_value(json!({
            "entities": [
                {
                    "type": "Company",
                    "source_path": "$.companies[*]",
                    "primary_key": "$.companies[*].id",
                    "properties": [
                        {"name": "name", "source_path": "$.companies[*].name", "type": "SemanticText"}
                    ]
                },
                {
                    "type": "Employee",
                    "source_path": "$.companies[*].employees[*]",
                    "primary_key": "$.companies[*].employees[*].id",
                    "properties": [
                        {"name": "name", "source_path": "$.companies[*].employees[*].name", "type": "Text"},
                        {"name": "age", "source_path": "$.companies[*].employees[*].age", "type": "Number"}
                    ]
                }
            ],
            "relationships": [
                {"type": "EMPLOYS", "from": "Company", "to": "Employee"}
            ]
        }))
        .unwrap();
        mapping.validate().unwrap();
        let data = json!({
            "companies": [{
                "id": "c1",
                "name": "Acme",
                "employees": [
                    {"id": "e1", "name": "Ada", "age": 37},
                    {"id": "e2", "name": "Grace", "age": 42}
                ]
            }]
        });

        let mapped = to_graph(&mapping, &data).unwrap();
        let graph = mapped.graph;

        assert_eq!(graph.entities().len(), 3);
        assert_eq!(graph.relations().len(), 2);
        let company = graph
            .entities()
            .iter()
            .find(|entity| entity.r#type == "Company")
            .unwrap();
        assert_eq!(
            company.properties["id"].value,
            json!("c1"),
            "primary key property should be injected when absent from mapping properties"
        );
        assert_eq!(company.properties["name"].property_type, PropertyType::Text);
    }

    #[test]
    fn returns_graph_specification_from_mapping() {
        let mapping: Mapping = serde_json::from_value(json!({
            "entities": [{
                "type": "Company",
                "source_path": "$.companies[*]",
                "primary_key": "$.companies[*].id",
                "description": "A legal organization.",
                "properties": [
                    {
                        "name": "name",
                        "source_path": "$.companies[*].name",
                        "type": "SemanticText",
                        "description": "Display name."
                    }
                ]
            }]
        }))
        .unwrap();
        mapping.validate().unwrap();
        let data = json!({"companies": [{"id": "c1", "name": "Acme"}]});

        let mapped = to_graph(&mapping, &data).unwrap();

        assert_eq!(
            mapped
                .specification
                .get_entity("Company")
                .unwrap()
                .description,
            "A legal organization."
        );
        assert_eq!(
            mapped
                .specification
                .get_property("Company", "name")
                .unwrap()
                .r#type,
            PropertyType::Text
        );
        assert_eq!(
            mapped
                .specification
                .get_property("Company", "id")
                .unwrap()
                .r#type,
            PropertyType::String
        );
    }

    #[test]
    fn duplicate_primary_keys_use_last_properties() {
        let mapping: Mapping = serde_json::from_value(json!({
            "entities": [{
                "type": "Company",
                "source_path": "$.companies[*]",
                "primary_key": "$.companies[*].id",
                "properties": [
                    {"name": "name", "source_path": "$.companies[*].name", "type": "String"}
                ]
            }]
        }))
        .unwrap();
        mapping.validate().unwrap();
        let data = json!({
            "companies": [
                {"id": "c1", "name": "Old"},
                {"id": "c1", "name": "New"}
            ]
        });

        let mapped = to_graph(&mapping, &data).unwrap();
        let graph = mapped.graph;

        assert_eq!(graph.entities().len(), 1);
        assert_eq!(graph.entities()[0].properties["name"].value, json!("New"));
    }
}
