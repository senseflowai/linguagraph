mod builder;
mod builtins;
mod canonical;
mod ontology;
mod ontology_storage;
mod schema;
mod scope;
mod types;

pub use builder::{
    ChunkBuilder, EntityBuilder, Graph, GraphBuilder, RelationshipBuilder, CANONICAL_FIELD,
};
pub use builtins::{
    is_builtin_entity, new_chunk, new_source, new_v4_id, CHUNK_LABEL, MENTION_REL, PART_OF_REL,
    SOURCE_LABEL,
};
pub use canonical::build_canonical_text;
pub use ontology::{
    DomainOntology, DomainOntologyMatch, EntityTypeMatch, EntityTypeSpec, OntologyCatalog,
    OntologyError, OntologyPropertyType, PropertySpec, RelationTypeSpec,
    DEFAULT_DOMAIN_SELECTION_THRESHOLD, DEFAULT_DOMAIN_SELECTION_TOP_K,
};
// Crate-internal: the ontology service (`crate::ontology::service`) reuses
// the domain-level routing passage builder to compute the fresh embedding
// id-set it garbage-collects against.
pub(crate) use ontology::domain_embedding_text;
pub use ontology_storage::{
    InMemoryOntologyCatalogStorage, JsonFileOntologyCatalogStorage, OntologyCatalogStorage,
    DEFAULT_ONTOLOGY_CATALOG_CACHE_PATH,
};
pub use schema::{
    canonical_handler_id, EntityGraph, PrimaryKey, Property, PropertyType, RelationGraph,
};
pub use scope::Scope;
pub use types::{EntityRef, GraphBuildError, RelationRef};
