//! Schema-aware LLM prompt generation.
//!
//! The prompt module is intentionally provider-agnostic: it takes a
//! [`GraphSchema`] and emits a single system-prompt string. Provider
//! plumbing (Anthropic, OpenAI, local models) lives outside the crate.
//!
//! The ontology catalog itself lives in
//! [`crate::graph`] — the prompt module is just one of its consumers.

mod generator;
mod ontology_suggest;
mod schema;

pub use generator::{
    generate_query_prompt, generate_system_prompt, select_query_schema, PromptOptions,
    PromptSchemaSelection,
};
pub use ontology_suggest::render_schema_suggest_prompt;
pub use schema::{GraphSchema, NodeKind, Property, PropertyType, RelKind};
