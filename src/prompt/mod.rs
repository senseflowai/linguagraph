//! Schema-aware LLM prompt generation.
//!
//! The prompt module is intentionally provider-agnostic: it takes a
//! [`GraphSchema`] and emits a single system-prompt string. Provider plumbing
//! (Anthropic, OpenAI, local models) lives outside the crate.
//!
//! # Public API
//!
//! - [`PromptGenerator`] — high-level facade. Holds an
//!   [`OntologyCatalog`] and exposes [`PromptGenerator::query_prompt`]
//!   and [`PromptGenerator::knowledge_extract_prompt`].
//! - [`OntologyCatalogStorage`] — pluggable backend trait for
//!   persisting the ontology catalog (default: JSON file via
//!   [`JsonFileOntologyCatalogStorage`]; custom backends can store in
//!   Postgres, S3, etc).
//! - Low-level free functions ([`generate_query_prompt`],
//!   [`generate_system_prompt`]) remain available for callers that don't
//!   need ontology handling.

mod builder;
mod generator;
mod knowledge;
mod ontology;
mod schema;
mod storage;

pub use builder::PromptGenerator;
pub use generator::{
    generate_query_prompt, generate_system_prompt, select_query_schema, PromptOptions,
    PromptSchemaSelection,
};
pub use knowledge::{render_knowledge_extract_prompt, DOMAIN_PLACEHOLDER};
pub use ontology::{
    DomainOntology, EntityTypeSpec, OntologyCatalog, OntologyError, RelationTypeSpec,
};
pub use schema::{GraphSchema, NodeKind, Property, PropertyType, RelKind};
pub use storage::{
    InMemoryOntologyCatalogStorage, JsonFileOntologyCatalogStorage, OntologyCatalogStorage,
    SharedOntologyCatalogStorage,
};
