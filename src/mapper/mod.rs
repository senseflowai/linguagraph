//! Declarative mapping from raw JSON to graph entities.
//!
//! The mapper is the front-end of the ingestion pipeline. It owns:
//!
//! * a typed mirror of the mapping document ([`schema`]),
//! * a small JSONPath evaluator restricted to what the mapping language
//!   actually uses ([`path`]),
//! * an extractor that combines the two to produce normalised rows
//!   ([`extractor`]).
//!
//! The output of this layer ([`Extracted`]) is *position-tagged*: every row
//! carries the array indices it came from, which is what makes implicit
//! relationship resolution deterministic in the [`crate::ingest`] planner.

pub mod extractor;
pub mod graph;
pub mod path;
pub mod schema;

use thiserror::Error;

pub use extractor::{extract, EntityRow, Extracted, ExtractedEntity};
pub use graph::{to_graph, MappedGraph};
pub use path::{JsonPath, PathError};
pub use schema::{EntityMapping, Mapping, PropertyMapping, RelationshipMapping};

#[derive(Debug, Error)]
pub enum MapperError {
    #[error("invalid mapping JSON: {0}")]
    Json(#[from] serde_json::Error),

    #[error("I/O error reading mapping: {0}")]
    Io(#[from] std::io::Error),

    #[error("invalid JSONPath '{path}': {source}")]
    Path { path: String, source: PathError },

    #[error("property path '{prop}' is not a child of entity source path '{src}'")]
    PropertyNotUnderSource { prop: String, src: String },

    #[error(
        "primary key for entity '{label}' resolved to {count} values for one row (expected 1)"
    )]
    AmbiguousPrimaryKey { label: String, count: usize },

    #[error("primary key missing for entity '{label}' at context {context:?}")]
    MissingPrimaryKey { label: String, context: Vec<usize> },

    #[error("entity '{label}' references unknown type in relationship: {missing}")]
    UnknownRelationshipEndpoint { label: String, missing: String },

    #[error("duplicate entity type '{0}' in mapping")]
    DuplicateEntityType(String),

    #[error("unknown entity type '{0}'")]
    UnknownEntityType(String),

    #[error("unknown mapping property type '{0}'")]
    UnknownPropertyType(String),

    #[error("graph construction error: {0}")]
    Graph(String),

    #[error(
        "property '{property}' in entity '{entity}' is missing a 'type' tag — \
         every property must declare its type (e.g. \"type\": \"Text\")"
    )]
    MissingPropertyType { entity: String, property: String },
}
