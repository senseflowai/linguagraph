mod builder;
mod builtins;
mod canonical;
mod ontology;
mod ontology_storage;
mod schema;
mod types;

pub use builder::{ChunkBuilder, EntityBuilder, Graph, GraphBuilder, RelationshipBuilder};
pub use canonical::build_canonical_text;
pub use builtins::{
    is_builtin_entity, new_chunk, new_source, new_v4_id, CHUNK_LABEL, MENTION_REL, PART_OF_REL,
    SOURCE_LABEL,
};
pub use ontology::{
    DomainOntology, EntityTypeMatch, EntityTypeSpec, OntologyCatalog, OntologyError,
    OntologyPropertyType, PropertySpec, RelationTypeSpec,
};
pub use ontology_storage::{
    InMemoryOntologyCatalogStorage, JsonFileOntologyCatalogStorage, OntologyCatalogStorage,
    SharedOntologyCatalogStorage, DEFAULT_ONTOLOGY_CATALOG_CACHE_PATH,
};
pub use schema::{EntityGraph, PrimaryKey, Property, PropertyType, RelationGraph};
pub use types::{EntityRef, GraphBuildError, RelationRef};
