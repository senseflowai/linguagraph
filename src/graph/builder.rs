use crate::graph::schema::{EntityGraph, PropertyType, RelationGraph};
use crate::graph::types::{EntityRef, GraphBuildError, RelationRef};

/// Owned graph assembled by [`GraphBuilder`].
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Graph {
    entities: Vec<EntityGraph>,
    relations: Vec<RelationGraph>,
}

impl Graph {
    pub fn entities(&self) -> &[EntityGraph] {
        &self.entities
    }

    pub fn relations(&self) -> &[RelationGraph] {
        &self.relations
    }

    pub fn entity(&self, entity_ref: EntityRef) -> Option<&EntityGraph> {
        self.entities.get(entity_ref.index())
    }

    pub fn relation(&self, relation_ref: RelationRef) -> Option<&RelationGraph> {
        self.relations.get(relation_ref.index())
    }
}

/// Incremental builder for an owned graph.
#[derive(Debug, Clone, Default)]
pub struct GraphBuilder {
    graph: Graph,
}

impl GraphBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn entity(&mut self, r#type: impl Into<String>) -> EntityBuilder<'_> {
        EntityBuilder {
            builder: self,
            entity: EntityGraph::new(r#type),
        }
    }

    pub fn add_entity(&mut self, entity: EntityGraph) -> EntityRef {
        let entity_ref = EntityRef(self.graph.entities.len());
        self.graph.entities.push(entity);
        entity_ref
    }

    pub fn relationship(
        &mut self,
        from: EntityRef,
        r#type: impl Into<String>,
        to: EntityRef,
    ) -> RelationshipBuilder<'_> {
        RelationshipBuilder {
            builder: self,
            relation: RelationGraph::new(from, r#type, to),
        }
    }

    pub fn add_relationship(
        &mut self,
        relation: RelationGraph,
    ) -> Result<RelationRef, GraphBuildError> {
        self.ensure_entity(relation.from)?;
        self.ensure_entity(relation.to)?;

        let relation_ref = RelationRef(self.graph.relations.len());
        self.graph.relations.push(relation);
        Ok(relation_ref)
    }

    pub fn build(self) -> Graph {
        self.graph
    }

    fn ensure_entity(&self, entity_ref: EntityRef) -> Result<(), GraphBuildError> {
        if entity_ref.index() < self.graph.entities.len() {
            Ok(())
        } else {
            Err(GraphBuildError::UnknownEntityRef(entity_ref.index()))
        }
    }
}

/// Fluent entity construction tied to a [`GraphBuilder`].
#[derive(Debug)]
pub struct EntityBuilder<'a> {
    builder: &'a mut GraphBuilder,
    entity: EntityGraph,
}

impl EntityBuilder<'_> {
    pub fn label(mut self, label: impl Into<String>) -> Self {
        self.entity = self.entity.label(label);
        self
    }

    pub fn labels(mut self, labels: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.entity = self.entity.labels(labels);
        self
    }

    pub fn strict_primary_key(mut self, field: impl Into<String>) -> Self {
        self.entity = self.entity.strict_primary_key(field);
        self
    }

    pub fn soft_primary_key(mut self, field: impl Into<String>) -> Self {
        self.entity = self.entity.soft_primary_key(field);
        self
    }

    pub fn property(
        mut self,
        name: impl Into<String>,
        property_type: PropertyType,
        value: impl Into<serde_json::Value>,
    ) -> Self {
        self.entity = self.entity.property(name, property_type, value);
        self
    }

    pub fn add(self) -> EntityRef {
        self.builder.add_entity(self.entity)
    }
}

/// Fluent relationship construction tied to a [`GraphBuilder`].
#[derive(Debug)]
pub struct RelationshipBuilder<'a> {
    builder: &'a mut GraphBuilder,
    relation: RelationGraph,
}

impl RelationshipBuilder<'_> {
    pub fn property(
        mut self,
        name: impl Into<String>,
        property_type: PropertyType,
        value: impl Into<serde_json::Value>,
    ) -> Self {
        self.relation = self.relation.property(name, property_type, value);
        self
    }

    pub fn add(self) -> Result<RelationRef, GraphBuildError> {
        self.builder.add_relationship(self.relation)
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::graph::{EntityGraph, PrimaryKey};

    #[test]
    fn creates_entity_values_conveniently() {
        let entity = EntityGraph::new("Person")
            .label("Human")
            .strict_primary_key("id")
            .property("id", PropertyType::String, "person-1")
            .property("age", PropertyType::Number, 42);

        assert_eq!(entity.r#type, "Person");
        assert_eq!(entity.labels, vec!["Human"]);
        assert_eq!(entity.primary_key, Some(PrimaryKey::Strict("id".into())));
        assert_eq!(entity.properties["id"].value, json!("person-1"));
        assert_eq!(entity.properties["age"].value, json!(42));
    }

    #[test]
    fn builds_entities_and_relationships_fluently() {
        let mut builder = GraphBuilder::new();

        let alice = builder
            .entity("Person")
            .label("User")
            .strict_primary_key("id")
            .property("id", PropertyType::String, "alice")
            .property("name", PropertyType::Text, "Alice")
            .add();

        let bob = builder
            .entity("Person")
            .label("User")
            .strict_primary_key("id")
            .property("id", PropertyType::String, "bob")
            .property("name", PropertyType::Text, "Bob")
            .add();

        let knows = builder
            .relationship(alice, "KNOWS", bob)
            .property("since", PropertyType::Number, 2024)
            .add()
            .unwrap();

        let graph = builder.build();

        assert_eq!(alice.index(), 0);
        assert_eq!(bob.index(), 1);
        assert_eq!(knows.index(), 0);
        assert_eq!(graph.entities().len(), 2);
        assert_eq!(graph.relations().len(), 1);
        assert_eq!(
            graph.entity(alice).unwrap().properties["name"].value,
            json!("Alice")
        );
        assert_eq!(graph.relation(knows).unwrap().r#type, "KNOWS");
        assert_eq!(graph.relation(knows).unwrap().from, alice);
        assert_eq!(graph.relation(knows).unwrap().to, bob);
    }

    #[test]
    fn also_accepts_prebuilt_relationships() {
        let mut builder = GraphBuilder::new();
        let article = builder.add_entity(EntityGraph::new("Article").strict_primary_key("id"));
        let author = builder.add_entity(EntityGraph::new("Author").strict_primary_key("id"));

        let relation = RelationGraph::new(article, "WRITTEN_BY", author).property(
            "confidence",
            PropertyType::Number,
            0.98,
        );
        let relation_ref = builder.add_relationship(relation).unwrap();
        let graph = builder.build();

        assert_eq!(graph.relation(relation_ref).unwrap().r#type, "WRITTEN_BY");
        assert_eq!(
            graph.relation(relation_ref).unwrap().properties["confidence"].value,
            json!(0.98)
        );
    }

    #[test]
    fn rejects_relationships_with_unknown_endpoints() {
        let mut builder = GraphBuilder::new();
        let known = builder.add_entity(EntityGraph::new("Known"));

        let err = builder
            .relationship(known, "POINTS_TO", EntityRef(99))
            .add()
            .unwrap_err();

        assert_eq!(err, GraphBuildError::UnknownEntityRef(99));
    }
}
