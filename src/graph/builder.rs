use crate::graph::builtins::{
    new_chunk, new_source, CHUNK_LABEL, MENTION_REL, PART_OF_REL, SOURCE_LABEL,
};
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
///
/// `GraphBuilder` is the canonical entrypoint for constructing a
/// [`Graph`] programmatically. It supports two flavors:
///
/// * [`GraphBuilder::new`] — bare builder with no built-in entities,
///   used by the mapping pipeline and other internal callers that
///   produce graphs whose lifecycle is driven by an external source.
/// * [`GraphBuilder::with_source`] — builds a source-rooted graph: a
///   [`SOURCE_LABEL`] entity is auto-created from the supplied name
///   (with a fresh UUID v4 id), every subsequently-added user entity
///   gets an auto `:mention` edge to it, and chunk construction via
///   [`GraphBuilder::chunk`] gets an auto `:part_of` edge.
#[derive(Debug, Clone, Default)]
pub struct GraphBuilder {
    graph: Graph,
    /// When `Some`, every user-added entity that is not itself a
    /// built-in [`SOURCE_LABEL`] / [`CHUNK_LABEL`] node automatically
    /// gets a `:mention` edge to this reference. Built-in chunks get a
    /// `:part_of` edge instead. Populated by [`with_source`] /
    /// [`add_source`].
    auto_source: Option<EntityRef>,
}

impl GraphBuilder {
    /// Bare builder — no built-in entities and no auto-edges. Reserved
    /// for callers (the mapping pipeline) that drive entity creation
    /// from an external schema.
    pub fn new() -> Self {
        Self::default()
    }

    /// Builder rooted at a fresh `Source` entity.
    ///
    /// The source is created with a UUID v4 `id` and a `Text` `name`
    /// property. After this call, every user entity added through the
    /// builder receives an automatic `:mention` edge to that source,
    /// and chunks added via [`GraphBuilder::chunk`] receive a
    /// `:part_of` edge.
    pub fn with_source(name: impl Into<String>) -> Self {
        let mut builder = Self::new();
        builder.add_source(name);
        builder
    }

    /// Add a `Source` entity to this builder and start auto-wiring
    /// `:mention` / `:part_of` edges against it.
    ///
    /// Calling this multiple times overwrites the auto-source target
    /// — every subsequent entity gets attached to the latest one.
    /// Earlier entities keep the edges they were created with.
    pub fn add_source(&mut self, name: impl Into<String>) -> EntityRef {
        let entity_ref = EntityRef(self.graph.entities.len());
        self.graph.entities.push(new_source(name));
        self.auto_source = Some(entity_ref);
        entity_ref
    }

    /// Reference to the currently-active auto-`Source`, if any.
    pub fn source(&self) -> Option<EntityRef> {
        self.auto_source
    }

    pub fn entity(&mut self, r#type: impl Into<String>) -> EntityBuilder<'_> {
        EntityBuilder {
            builder: self,
            entity: EntityGraph::new(r#type),
        }
    }

    /// Start building a [`Chunk`] entity.
    ///
    /// The chunk is initialised with a fresh UUID v4 `id` and the
    /// given text fragment stored as a `Text` property (so the
    /// `SemanticText` handler embeds it on ingestion). On
    /// [`ChunkBuilder::add`] the builder also emits an automatic
    /// `:part_of` edge to the active source — see [`with_source`] /
    /// [`add_source`].
    ///
    /// [`Chunk`]: CHUNK_LABEL
    /// [`with_source`]: Self::with_source
    /// [`add_source`]: Self::add_source
    pub fn chunk(&mut self, text: impl Into<String>) -> ChunkBuilder<'_> {
        ChunkBuilder {
            builder: self,
            entity: new_chunk(text),
        }
    }

    pub fn add_entity(&mut self, entity: EntityGraph) -> EntityRef {
        let label = entity.r#type.clone();
        let entity_ref = EntityRef(self.graph.entities.len());
        self.graph.entities.push(entity);
        self.attach_auto_edge(entity_ref, &label);
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

    /// Wire the auto-edge from `entity` to the active source (if any).
    /// Chunks get `:part_of`; every other non-built-in entity gets
    /// `:mention`. Sources never get auto-edges (we don't `:mention`
    /// ourselves), and an entity added before [`add_source`] / via
    /// [`new`] is left alone.
    ///
    /// [`add_source`]: Self::add_source
    /// [`new`]: Self::new
    fn attach_auto_edge(&mut self, entity_ref: EntityRef, label: &str) {
        let Some(source_ref) = self.auto_source else {
            return;
        };
        if entity_ref == source_ref {
            return;
        }
        let rel_label = if label == CHUNK_LABEL {
            PART_OF_REL
        } else if label == SOURCE_LABEL {
            return;
        } else {
            MENTION_REL
        };
        self.graph
            .relations
            .push(RelationGraph::new(entity_ref, rel_label, source_ref));
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

/// Fluent [`Chunk`] construction tied to a [`GraphBuilder`].
///
/// Forwards property additions and labels through to the underlying
/// [`EntityGraph`]. On [`ChunkBuilder::add`] the chunk is inserted
/// into the graph and a `:part_of` edge is emitted to the active
/// source.
///
/// [`Chunk`]: CHUNK_LABEL
#[derive(Debug)]
pub struct ChunkBuilder<'a> {
    builder: &'a mut GraphBuilder,
    entity: EntityGraph,
}

impl ChunkBuilder<'_> {
    pub fn label(mut self, label: impl Into<String>) -> Self {
        self.entity = self.entity.label(label);
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

    /// Insert the chunk into the graph. Returns the [`EntityRef`] of
    /// the inserted chunk. Fails if the builder has no active source —
    /// chunks must always be attached to one.
    pub fn add(self) -> Result<EntityRef, GraphBuildError> {
        if self.builder.auto_source.is_none() {
            return Err(GraphBuildError::ChunkWithoutSource);
        }
        Ok(self.builder.add_entity(self.entity))
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

    #[test]
    fn with_source_auto_creates_source_and_attaches_mention_edges() {
        let mut builder = GraphBuilder::with_source("My Doc");
        let source = builder.source().expect("source created");

        let alice = builder
            .entity("Person")
            .strict_primary_key("id")
            .property("id", PropertyType::String, "alice")
            .add();
        let bob = builder
            .entity("Person")
            .strict_primary_key("id")
            .property("id", PropertyType::String, "bob")
            .add();

        let graph = builder.build();

        // Source + Alice + Bob = 3 entities.
        assert_eq!(graph.entities().len(), 3);
        assert_eq!(graph.entity(source).unwrap().r#type, SOURCE_LABEL);

        // Each user entity has exactly one :mention edge to the source.
        let mentions: Vec<_> = graph
            .relations()
            .iter()
            .filter(|r| r.r#type == MENTION_REL)
            .collect();
        assert_eq!(mentions.len(), 2);
        for m in &mentions {
            assert_eq!(m.to, source);
            assert!(m.from == alice || m.from == bob);
        }
    }

    #[test]
    fn chunk_emits_part_of_edge_to_source() {
        let mut builder = GraphBuilder::with_source("Doc");
        let source = builder.source().unwrap();
        let chunk = builder.chunk("hello world").add().unwrap();

        let graph = builder.build();

        let chunk_entity = graph.entity(chunk).unwrap();
        assert_eq!(chunk_entity.r#type, CHUNK_LABEL);

        // Exactly one part_of edge from chunk → source, no :mention.
        let edges: Vec<_> = graph
            .relations()
            .iter()
            .filter(|r| r.from == chunk)
            .collect();
        assert_eq!(edges.len(), 1);
        assert_eq!(edges[0].r#type, PART_OF_REL);
        assert_eq!(edges[0].to, source);
    }

    #[test]
    fn chunk_requires_a_source() {
        let mut builder = GraphBuilder::new();
        let err = builder.chunk("orphan").add().unwrap_err();
        assert_eq!(err, GraphBuildError::ChunkWithoutSource);
    }

    #[test]
    fn bare_builder_does_not_attach_auto_edges() {
        // Without a source, GraphBuilder::new behaves exactly as before
        // — no auto-mention, no surprises for the mapping pipeline.
        let mut builder = GraphBuilder::new();
        builder
            .entity("Person")
            .strict_primary_key("id")
            .property("id", PropertyType::String, "alice")
            .add();
        let graph = builder.build();
        assert_eq!(graph.entities().len(), 1);
        assert!(graph.relations().is_empty());
    }
}
