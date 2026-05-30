//! Schema-aware LLM prompt generation.
//!
//! The prompt module is intentionally provider-agnostic: it takes a
//! [`GraphSchema`] and emits a single system-prompt string. Provider
//! plumbing (Anthropic, OpenAI, local models) lives outside the crate.
//!
//! The ontology catalog itself lives in
//! [`crate::graph`] — the prompt module is just one of its consumers.

mod builder;
mod generator;
mod knowledge;
mod schema;

pub use builder::PromptGenerator;
pub use generator::{
    generate_query_prompt, generate_system_prompt, select_query_schema, PromptOptions,
    PromptSchemaSelection,
};
pub use knowledge::{render_knowledge_extract_prompt, DOMAIN_PLACEHOLDER};
pub use schema::{GraphSchema, NodeKind, Property, PropertyType, RelKind};
