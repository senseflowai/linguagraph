//! Prompt-generation module: arbitrary JSON → prompt that asks an LLM
//! to produce a linguagraph mapping document.
//!
//! The pipeline is purely synchronous and pure-functional:
//!
//! ```text
//!   serde_json::Value
//!         │
//!         ▼
//!   analyzer ─┬─► JsonSchemaSummary  (entities, fields, type guesses)
//!         │   │
//!         │   └─► (snapshotted into the prompt as the "Inferred
//!         │        structure" section)
//!         ▼
//!   builder   ─► assembles sections from `template.rs` and the
//!                 schema summary into a single prompt string.
//! ```
//!
//! The summary is a pure function of the input JSON; the builder is a
//! pure function of (summary, options). Both are trivially testable.
//!
//! # Public API
//!
//! - [`generate_prompt`] — one-shot helper used by the CLI.
//! - [`analyzer::analyze`] — exposed for advanced callers that want to
//!   inspect/mutate the summary before rendering.
//! - [`builder::PromptBuilder`] — for callers that want fine-grained
//!   control over which sections go in.
//!
//! # Heuristic philosophy
//!
//! The analyzer's job is to *propose*, not to *decide*. The prompt
//! shows the LLM what the heuristics inferred and asks it to correct
//! anything that looks wrong (e.g. a name that isn't in fact prose, a
//! "category" field that's actually free text). That keeps us robust
//! to weird inputs without hardcoding specific shapes.

pub mod analyzer;
pub mod builder;
pub mod inference;
pub mod knowledge;
pub mod template;

pub use analyzer::{analyze, EntitySummary, FieldSummary, JsonSchemaSummary, RelationshipHint};
pub use builder::{PromptBuilder, PromptGenOptions};
pub use inference::InferredType;
pub use knowledge::{
    default_entity_types, default_relation_types, generate_knowledge_extract_prompt,
    EntityTypeSpec, KnowledgeExtractOptions, RelationTypeSpec,
};

use serde_json::Value;

/// One-shot helper: analyse `json`, build a prompt with `opts`,
/// return the rendered string.
pub fn generate_prompt(json: &Value, opts: &PromptGenOptions) -> String {
    let summary = analyze(json);
    PromptBuilder::new(summary, opts.clone()).build()
}
