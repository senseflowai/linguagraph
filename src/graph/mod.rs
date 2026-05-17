mod builder;
mod builtins;
#[cfg(feature = "postgres")]
mod postgres_spec_storage;
mod schema;
mod spec;
mod spec_storage;
mod types;

pub use builder::{ChunkBuilder, EntityBuilder, Graph, GraphBuilder, RelationshipBuilder};
pub use builtins::{
    is_builtin_entity, new_chunk, new_source, new_v4_id, CHUNK_LABEL, MENTION_REL, PART_OF_REL,
    SOURCE_LABEL,
};
#[cfg(feature = "postgres")]
pub use postgres_spec_storage::{
    PostgresGraphSpecificationStorage, DEFAULT_GRAPH_SPECIFICATION_TABLE,
};
pub use schema::{EntityGraph, PrimaryKey, Property, PropertyType, RelationGraph};
pub use spec::{EntitySpecRecord, GraphSpecification, PropertySpecRecord, SpecRecord};
pub use spec_storage::{
    FileGraphSpecificationStorage, GraphSpecificationStorage, GraphSpecificationStorageError,
    DEFAULT_GRAPH_SPECIFICATION_CACHE_PATH,
};
pub use types::{EntityRef, GraphBuildError, RelationRef};
