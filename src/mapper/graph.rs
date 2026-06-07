//! Convert mapping extraction output into an owned [`crate::graph::Graph`].

use std::collections::{BTreeMap, HashMap};

use serde_json::Value;

use crate::ast::query::Literal;
use crate::graph::{
    DomainOntology, EntityGraph, EntityRef, EntityTypeSpec, Graph, GraphBuilder, OntologyCatalog,
    OntologyPropertyType, PropertySpec, PropertyType, RelationTypeSpec, Scope,
};

use super::extractor::primary_key_part_to_string;
use super::{extract, EntityRow, Extracted, MapperError, Mapping};

/// Default domain used by [`catalog_from_mapping`] when the mapping
/// document doesn't carry its own. Picked to match what senseflowai
/// historically used for raw JSON ingest paths.
pub const DEFAULT_MAPPING_DOMAIN: &str = "mapping";

/// Result of mapping raw JSON into graph-native structures.
#[derive(Debug, Clone, PartialEq)]
pub struct MappedGraph {
    pub graph: Graph,
    pub catalog: OntologyCatalog,
    pub domain: String,
}

impl MappedGraph {
    pub fn into_parts(self) -> (Graph, OntologyCatalog, String) {
        (self.graph, self.catalog, self.domain)
    }
}

/// Build an ingestion [`Graph`] and [`OntologyCatalog`] entries from raw
/// JSON plus a mapping document. The catalog is built under the
/// mapping's `domain` (or [`DEFAULT_MAPPING_DOMAIN`] when unset).
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
    let domain = mapping
        .domain
        .clone()
        .unwrap_or_else(|| DEFAULT_MAPPING_DOMAIN.to_string());
    let graph = graph_from_extracted(mapping, &extracted, &domain)?;
    let catalog = catalog_from_mapping(mapping, &domain)?;
    Ok(MappedGraph {
        graph,
        catalog,
        domain,
    })
}

fn graph_from_extracted(
    mapping: &Mapping,
    extracted: &Extracted,
    domain: &str,
) -> Result<Graph, MapperError> {
    // The mapper pipeline only sees structured input (JSON, DB rows
    // shaped through a mapping doc). Stamp every user entity it
    // produces with `Scope::Structured` so the QA service can pick a
    // DSL/Cypher strategy for these types without further annotation.
    // Builtin Source/Chunk are exempt via `is_builtin_entity` inside
    // `GraphBuilder::add_entity`.
    let mut builder = match mapping.source.as_deref() {
        Some(source) if !source.trim().is_empty() => GraphBuilder::with_source(source),
        _ => GraphBuilder::new(),
    }
    .with_default_scope(Scope::Structured);
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
                .domain(domain)
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

    // Stub endpoints synthesised for foreign keys whose target isn't in
    // the current document, deduplicated by (label, normalized id) so N
    // referrers share one id-only node. Shared across relationships.
    let mut stub_refs: HashMap<(String, String), EntityRef> = HashMap::new();

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

        match &rel.from_key {
            // Foreign-key value join: link a `from` row to every `to`
            // row whose `to_key` (default: primary key) value equals the
            // `from` row's `from_key` value. This is what connects
            // entities that live in separate top-level arrays, where
            // array-context alignment is meaningless.
            Some(from_key) => {
                let mut index: HashMap<String, Vec<&EntityRow>> = HashMap::new();
                for to_row in &to.rows {
                    if let Some(key) = target_key_string(to_row, rel.to_key.as_deref()) {
                        index.entry(key).or_default().push(to_row);
                    }
                }
                // A stub can only be upserted when `to_key` targets the
                // destination entity's primary key — then the target PK
                // equals the from-row's foreign-key value.
                let to_mapping = mapping.entities.iter().find(|e| e.kind == to.label);
                let to_key_is_pk = match (&rel.to_key, to_mapping) {
                    (None, _) => true,
                    (Some(k), Some(m)) => k == &m.primary_key,
                    (Some(_), None) => false,
                };
                for from_row in &from.rows {
                    let Some(value) = from_row.join_keys.get(from_key) else {
                        continue;
                    };
                    let Some(key) = primary_key_part_to_string(value) else {
                        continue;
                    };
                    let from_ref = resolve_ref(&refs, &from.label, &from_row.id, &rel.kind)?;
                    if let Some(matches) = index.get(&key) {
                        // Target present in this document → link the real
                        // entities (unchanged behavior).
                        for to_row in matches {
                            let to_ref = resolve_ref(&refs, &to.label, &to_row.id, &rel.kind)?;
                            builder
                                .relationship(from_ref, rel.kind.clone(), to_ref)
                                .add()
                                .map_err(|e| MapperError::Graph(e.to_string()))?;
                        }
                    } else if to_key_is_pk {
                        // Target not in this document → upsert an id-only
                        // stub. At ingest the stub's `MERGE (n:To {id})`
                        // matches an existing graph node (preserving its
                        // properties) or creates a skeleton to be enriched
                        // when the real entity is ingested later.
                        let to_pk_field = to_mapping
                            .map(primary_key_property_name)
                            .unwrap_or_else(|| "id".to_string());
                        let stub_ref = *stub_refs
                            .entry((to.label.clone(), key.clone()))
                            .or_insert_with(|| {
                                let stub = EntityGraph::new(to.label.clone())
                                    .domain(domain)
                                    .strict_primary_key(to_pk_field.clone())
                                    .property(
                                        to_pk_field.clone(),
                                        PropertyType::String,
                                        literal_to_json(value),
                                    );
                                builder.add_reference_entity(stub)
                            });
                        builder
                            .relationship(from_ref, rel.kind.clone(), stub_ref)
                            .add()
                            .map_err(|e| MapperError::Graph(e.to_string()))?;
                    }
                    // else: non-PK `to_key` with no in-document match →
                    // target PK is unknown, so no edge is emitted.
                }
            }
            // Default: array-context alignment (unchanged behavior).
            None => {
                for from_row in &from.rows {
                    for to_row in &to.rows {
                        if !contexts_align(&from_row.context, &to_row.context) {
                            continue;
                        }
                        let from_ref = resolve_ref(&refs, &from.label, &from_row.id, &rel.kind)?;
                        let to_ref = resolve_ref(&refs, &to.label, &to_row.id, &rel.kind)?;
                        builder
                            .relationship(from_ref, rel.kind.clone(), to_ref)
                            .add()
                            .map_err(|e| MapperError::Graph(e.to_string()))?;
                    }
                }
            }
        }
    }

    Ok(builder.build())
}

/// Resolve the [`EntityRef`] of an extracted row by `(label, id)`.
fn resolve_ref(
    refs: &HashMap<(String, Literal), EntityRef>,
    label: &str,
    id: &Literal,
    kind: &str,
) -> Result<EntityRef, MapperError> {
    refs.get(&(label.to_string(), id.clone()))
        .copied()
        .ok_or_else(|| MapperError::UnknownRelationshipEndpoint {
            label: kind.to_string(),
            missing: label.to_string(),
        })
}

/// Normalized join value for a `to` row: the resolved `to_key` value when
/// one is declared, otherwise the row's (already stringified) primary key.
fn target_key_string(to_row: &EntityRow, to_key: Option<&str>) -> Option<String> {
    match to_key {
        Some(key) => to_row.join_keys.get(key).and_then(primary_key_part_to_string),
        None => primary_key_part_to_string(&to_row.id),
    }
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

fn catalog_from_mapping(mapping: &Mapping, domain: &str) -> Result<OntologyCatalog, MapperError> {
    let mut entity_types: Vec<EntityTypeSpec> = Vec::with_capacity(mapping.entities.len());
    for entity in &mapping.entities {
        let description = entity.description.clone().or_else(|| entity.name.clone());

        let mut props: Vec<PropertySpec> = Vec::with_capacity(entity.properties.len() + 1);

        let pk_name = primary_key_property_name(entity);
        if !entity
            .properties
            .iter()
            .any(|property| property.name == pk_name)
        {
            props.push(PropertySpec {
                name: pk_name,
                description: Some("Primary key.".into()),
                property_type: OntologyPropertyType::String,
                required: true,
            });
        }

        for property in &entity.properties {
            props.push(PropertySpec {
                name: property.name.clone(),
                description: property.description.clone(),
                property_type: ontology_property_type(property.type_name())?,
                required: false,
            });
        }

        entity_types.push(EntityTypeSpec {
            name: entity.kind.clone(),
            description,
            properties: props,
            embedding: None,
        });
    }

    let mut relation_types: Vec<RelationTypeSpec> = Vec::with_capacity(mapping.relationships.len());
    for rel in &mapping.relationships {
        relation_types.push(RelationTypeSpec {
            name: rel.kind.clone(),
            description: None,
        });
    }

    let mut catalog = OntologyCatalog::default();
    catalog.insert(
        domain,
        DomainOntology {
            entity_types,
            relation_types,
        },
    );
    Ok(catalog)
}

fn ontology_property_type(type_name: &str) -> Result<OntologyPropertyType, MapperError> {
    match type_name {
        "String" => Ok(OntologyPropertyType::String),
        "Text" | "SemanticText" => Ok(OntologyPropertyType::Text),
        "Number" | "Int" => Ok(OntologyPropertyType::Int),
        "Float" => Ok(OntologyPropertyType::Float),
        "Boolean" | "Bool" => Ok(OntologyPropertyType::Bool),
        "Date" => Ok(OntologyPropertyType::Date),
        "DateTime" | "Datetime" | "Timestamp" => Ok(OntologyPropertyType::Datetime),
        "List" => Ok(OntologyPropertyType::List),
        other => Err(MapperError::UnknownPropertyType(other.to_string())),
    }
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

        // Mapper-emitted user entities inherit Scope::Structured
        // from the default scope set on the builder.
        assert!(company.has_scope(Scope::Structured));
        let employees: Vec<_> = graph
            .entities()
            .iter()
            .filter(|e| e.r#type == "Employee")
            .collect();
        assert_eq!(employees.len(), 2);
        for emp in employees {
            assert!(emp.has_scope(Scope::Structured));
        }
    }

    #[test]
    fn returns_ontology_catalog_from_mapping() {
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
                .catalog
                .get_entity("Company")
                .unwrap()
                .1
                .description,
            Some("A legal organization.".into())
        );
        assert_eq!(
            mapped
                .catalog
                .get_property("Company", "name")
                .unwrap()
                .property_type,
            OntologyPropertyType::Text
        );
        assert_eq!(
            mapped
                .catalog
                .get_property("Company", "id")
                .unwrap()
                .property_type,
            OntologyPropertyType::String
        );
    }

    #[test]
    fn fk_join_links_cross_array_entities_by_value_not_index() {
        // Cameras and places are sibling top-level arrays joined by
        // `place_id`. The array ordering is deliberately mismatched so
        // that the old context-alignment behavior (index pairing) would
        // give the WRONG answer — the FK join must pair by value.
        let mapping: Mapping = serde_json::from_value(json!({
            "entities": [
                {"type": "Place", "source_path": "$.places[*]", "primary_key": "$.places[*].id",
                 "properties": [{"name": "name", "source_path": "$.places[*].name", "type": "Text"}]},
                {"type": "Camera", "source_path": "$.cameras[*]", "primary_key": "$.cameras[*].id",
                 "properties": [{"name": "name", "source_path": "$.cameras[*].name", "type": "Text"}]}
            ],
            "relationships": [
                {"type": "INSTALLED_AT", "from": "Camera", "to": "Place",
                 "from_key": "$.cameras[*].place_id", "to_key": "$.places[*].id"}
            ]
        }))
        .unwrap();
        mapping.validate().unwrap();
        let data = json!({
            "places": [
                {"id": 72,   "name": "Office"},
                {"id": 5390, "name": "Sales"}
            ],
            "cameras": [
                {"id": "cam-a", "name": "A", "place_id": 5390},
                {"id": "cam-b", "name": "B", "place_id": 72}
            ]
        });

        let graph = to_graph(&mapping, &data).unwrap().graph;
        assert_eq!(graph.entities().len(), 4, "2 places + 2 cameras");

        let edges: Vec<_> = graph
            .relations()
            .iter()
            .filter(|r| r.r#type == "INSTALLED_AT")
            .collect();
        assert_eq!(edges.len(), 2);

        for edge in edges {
            let cam = &graph.entities()[edge.from.index()];
            let place = &graph.entities()[edge.to.index()];
            assert_eq!(cam.r#type, "Camera");
            assert_eq!(place.r#type, "Place");
            let cam_name = cam.properties["name"].value.as_str().unwrap();
            // Place PK is injected as a stringified `id` property.
            let place_id = place.properties["id"].value.as_str().unwrap();
            match cam_name {
                "A" => assert_eq!(place_id, "5390", "camera A's place_id is 5390"),
                "B" => assert_eq!(place_id, "72", "camera B's place_id is 72"),
                other => panic!("unexpected camera {other}"),
            }
        }
    }

    #[test]
    fn bundled_teye_example_builds_all_fk_relationships() {
        let load = |name: &str| {
            std::fs::read_to_string(format!(
                "{}/examples/teye/{name}",
                env!("CARGO_MANIFEST_DIR")
            ))
            .unwrap()
        };
        let mapping = Mapping::from_str(&load("teye_mapping.json")).unwrap();
        let data: Value = serde_json::from_str(&load("teye_data.json")).unwrap();

        let graph = to_graph(&mapping, &data).unwrap().graph;
        let count = |t: &str| {
            graph
                .relations()
                .iter()
                .filter(|r| r.r#type == t)
                .count()
        };
        // 2 cameras → places, 2 events → cameras, 2 events → places.
        assert_eq!(count("INSTALLED_AT"), 2);
        assert_eq!(count("CAPTURED_BY"), 2);
        assert_eq!(count("OCCURRED_AT"), 2);
    }

    #[test]
    fn fk_join_upserts_stub_for_target_not_in_document() {
        // The referenced place isn't in this document. Instead of dropping
        // the edge, an id-only stub Place is upserted so the edge links a
        // node that exists (or will exist) in the graph.
        let mapping: Mapping = serde_json::from_value(json!({
            "entities": [
                {"type": "Place", "source_path": "$.places[*]", "primary_key": "$.places[*].id",
                 "properties": [{"name": "name", "source_path": "$.places[*].name", "type": "Text"}]},
                {"type": "Camera", "source_path": "$.cameras[*]", "primary_key": "$.cameras[*].id",
                 "properties": [{"name": "name", "source_path": "$.cameras[*].name", "type": "Text"}]}
            ],
            "relationships": [
                {"type": "INSTALLED_AT", "from": "Camera", "to": "Place",
                 "from_key": "$.cameras[*].place_id", "to_key": "$.places[*].id"}
            ]
        }))
        .unwrap();
        mapping.validate().unwrap();
        let data = json!({
            "places": [{"id": 72, "name": "Office"}],
            "cameras": [{"id": "cam-x", "name": "X", "place_id": 999}]
        });

        let graph = to_graph(&mapping, &data).unwrap().graph;
        let edges: Vec<_> = graph
            .relations()
            .iter()
            .filter(|r| r.r#type == "INSTALLED_AT")
            .collect();
        assert_eq!(edges.len(), 1, "edge to the stubbed target is emitted");

        let stub = &graph.entities()[edges[0].to.index()];
        assert_eq!(stub.r#type, "Place");
        // Stub carries only its id (the raw FK value), no real properties.
        assert_eq!(stub.properties["id"].value, json!(999));
        assert!(
            !stub.properties.contains_key("name"),
            "stub must not fabricate real properties"
        );
    }

    #[test]
    fn fk_to_key_defaults_to_target_primary_key() {
        // Omitting `to_key` falls back to the target entity's primary key.
        let mapping: Mapping = serde_json::from_value(json!({
            "entities": [
                {"type": "Place", "source_path": "$.places[*]", "primary_key": "$.places[*].id",
                 "properties": [{"name": "name", "source_path": "$.places[*].name", "type": "Text"}]},
                {"type": "Camera", "source_path": "$.cameras[*]", "primary_key": "$.cameras[*].id",
                 "properties": [{"name": "name", "source_path": "$.cameras[*].name", "type": "Text"}]}
            ],
            "relationships": [
                {"type": "INSTALLED_AT", "from": "Camera", "to": "Place",
                 "from_key": "$.cameras[*].place_id"}
            ]
        }))
        .unwrap();
        mapping.validate().unwrap();
        let data = json!({
            "places": [{"id": 72, "name": "Office"}],
            "cameras": [{"id": "cam-b", "name": "B", "place_id": 72}]
        });

        let graph = to_graph(&mapping, &data).unwrap().graph;
        let edges = graph
            .relations()
            .iter()
            .filter(|r| r.r#type == "INSTALLED_AT")
            .count();
        assert_eq!(edges, 1);
    }

    /// Mapping used by the cross-ingest scenarios: Place/Camera/Event with
    /// FK relationships, matching the generated teye shape.
    fn surveillance_mapping() -> Mapping {
        let m: Mapping = serde_json::from_value(json!({
            "domain": "surveillance",
            "entities": [
                {"type": "Place", "source_path": "$.places[*]", "primary_key": "$.places[*].id",
                 "properties": [
                    {"name": "id", "source_path": "$.places[*].id", "type": "Text"},
                    {"name": "name", "source_path": "$.places[*].name", "type": "Text"}]},
                {"type": "Camera", "source_path": "$.cameras[*]", "primary_key": "$.cameras[*].id",
                 "properties": [
                    {"name": "id", "source_path": "$.cameras[*].id", "type": "Text"},
                    {"name": "name", "source_path": "$.cameras[*].name", "type": "Text"}]},
                {"type": "Event", "source_path": "$.events[*]", "primary_key": "$.events[*].event_id",
                 "properties": [
                    {"name": "event_id", "source_path": "$.events[*].event_id", "type": "Text"}]}
            ],
            "relationships": [
                {"type": "CAPTURED_BY", "from": "Event", "to": "Camera",
                 "from_key": "$.events[*].origin.camera_id", "to_key": "$.cameras[*].id"},
                {"type": "OCCURRED_AT", "from": "Event", "to": "Place",
                 "from_key": "$.events[*].origin.place_id", "to_key": "$.places[*].id"}
            ]
        }))
        .unwrap();
        m.validate().unwrap();
        m
    }

    #[test]
    fn event_links_to_camera_already_in_graph_via_stub() {
        // New batch: cameras[] is empty, but events reference a camera that
        // lives in the graph from a previous ingest. A stub Camera is
        // upserted so the CAPTURED_BY edge links it by id.
        let mapping = surveillance_mapping();
        let data = json!({
            "places": [{"id": 72, "name": "Office"}],
            "cameras": [],
            "events": [{
                "event_id": "ev-1",
                "origin": {"camera_id": "cam-in-graph", "place_id": 72}
            }]
        });

        let graph = to_graph(&mapping, &data).unwrap().graph;

        // CAPTURED_BY → stub Camera (id-only); OCCURRED_AT → real Place.
        let captured: Vec<_> = graph
            .relations()
            .iter()
            .filter(|r| r.r#type == "CAPTURED_BY")
            .collect();
        assert_eq!(captured.len(), 1);
        let cam = &graph.entities()[captured[0].to.index()];
        assert_eq!(cam.r#type, "Camera");
        assert_eq!(cam.properties["id"].value, json!("cam-in-graph"));
        assert!(!cam.properties.contains_key("name"), "stub has no real props");
        assert_eq!(cam.domain.as_deref(), Some("surveillance"));

        let occurred = graph
            .relations()
            .iter()
            .filter(|r| r.r#type == "OCCURRED_AT")
            .count();
        assert_eq!(occurred, 1);
    }

    #[test]
    fn stub_endpoints_lower_to_idonly_merge_and_relation() {
        // Confirm the ingest planner turns the stub into an id-only node
        // MERGE (no clobbering props) plus a CAPTURED_BY relation batch.
        // Uses plain `String` props so an empty type registry suffices.
        let mapping: Mapping = serde_json::from_value(json!({
            "domain": "surveillance",
            "entities": [
                {"type": "Camera", "source_path": "$.cameras[*]", "primary_key": "$.cameras[*].id",
                 "properties": [{"name": "id", "source_path": "$.cameras[*].id", "type": "String"}]},
                {"type": "Event", "source_path": "$.events[*]", "primary_key": "$.events[*].event_id",
                 "properties": [{"name": "event_id", "source_path": "$.events[*].event_id", "type": "String"}]}
            ],
            "relationships": [
                {"type": "CAPTURED_BY", "from": "Event", "to": "Camera",
                 "from_key": "$.events[*].origin.camera_id", "to_key": "$.cameras[*].id"}
            ]
        }))
        .unwrap();
        mapping.validate().unwrap();
        let data = json!({
            "cameras": [],
            "events": [{
                "event_id": "ev-1",
                "origin": {"camera_id": "cam-9"}
            }]
        });
        let graph = to_graph(&mapping, &data).unwrap().graph;

        // Real entities carry a `_canonical` Text property (SemanticText),
        // so the registry needs the core + SemanticText handlers. Stubs
        // skip `_canonical`, so they never require a handler.
        use crate::types::handlers::{self, SemanticTextConfig, SemanticTextHandler};
        let registry = handlers::register_core(crate::types::RegistryBuilder::new())
            .register(SemanticTextHandler::new(
                SemanticTextConfig {
                    embedding_model: None,
                    collection: "cams".into(),
                    top_k: 10,
                    search_threshold: 0.1,
                    reranker_threshold: 0.2,
                },
                std::sync::Arc::new(crate::embeddings::MockEmbedder::new(8)),
            ))
            .build();
        let mut effects = crate::types::SideEffectQueue::new();
        let insert = crate::ingest::plan_graph_with_registry(
            &graph,
            crate::ingest::PlannerOptions::default(),
            &registry,
            &mut effects,
        )
        .unwrap();

        // Camera stub node batch: merge_on `id`, single row carrying the id.
        let cam_batch = insert
            .node_batches
            .iter()
            .find(|b| b.label == "Camera")
            .expect("Camera stub node batch");
        assert_eq!(cam_batch.merge_on, "id");
        assert_eq!(cam_batch.rows.len(), 1);
        assert_eq!(cam_batch.rows[0].id, Literal::String("cam-9".into()));
        // Stub props must not carry fabricated fields beyond the id.
        assert!(
            cam_batch.rows[0]
                .props
                .keys()
                .all(|k| k == "id"),
            "stub node should not set non-id properties"
        );

        // CAPTURED_BY relation batch matches Event→Camera by id.
        let rel = insert
            .relation_batches
            .iter()
            .find(|b| b.rel_type == "CAPTURED_BY")
            .expect("CAPTURED_BY relation batch");
        assert_eq!(rel.to_label, "Camera");
        assert_eq!(rel.to_key, "id");
        assert_eq!(rel.rows[0].to_id, Literal::String("cam-9".into()));
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

    #[test]
    fn mapping_source_creates_source_node_and_mentions_entities() {
        let mapping: Mapping = serde_json::from_value(json!({
            "source": "companies.json",
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
        let data = json!({"companies": [{"id": "c1", "name": "Acme"}]});

        let mapped = to_graph(&mapping, &data).unwrap();
        let graph = mapped.graph;

        assert_eq!(graph.entities().len(), 2);
        let source_ref = graph
            .entities()
            .iter()
            .position(|entity| entity.r#type == crate::graph::SOURCE_LABEL)
            .unwrap();
        let company_ref = graph
            .entities()
            .iter()
            .position(|entity| entity.r#type == "Company")
            .unwrap();
        let source = &graph.entities()[source_ref];
        assert_eq!(source.properties["name"].value, json!("companies.json"));
        assert!(graph.relations().iter().any(|rel| {
            rel.r#type == crate::graph::MENTION_REL
                && rel.from.index() == company_ref
                && rel.to.index() == source_ref
        }));
    }
}
