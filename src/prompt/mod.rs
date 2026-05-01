//! Schema-aware LLM prompt generation.
//!
//! The prompt module is intentionally provider-agnostic: it takes a
//! [`GraphSchema`] and emits a single system-prompt string. Provider plumbing
//! (Anthropic, OpenAI, local models) lives outside the crate.

mod generator;
mod schema;

pub use generator::{generate_system_prompt, PromptOptions};
pub use schema::{GraphSchema, NodeKind, Property, RelKind};
