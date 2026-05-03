//! Bundled type handlers.
//!
//! Each handler lives in its own submodule and is exported through
//! [`super::TypeRegistry`] via a small `register_*` helper. Adding a new
//! type means writing one file here and one line in
//! [`crate::types::handlers::register_default`].

pub mod semantic_text;

pub use semantic_text::{SemanticTextConfig, SemanticTextHandler};

use std::sync::Arc;

use super::{RegistryBuilder, TypeRegistry};
use crate::config::Config;
use crate::embeddings::SharedEmbedder;

/// Build the default registry from `config` and a shared embedder.
///
/// Currently registers:
///
/// * [`SemanticTextHandler`] — one instance per `[types.SemanticText]`
///   block in the configuration. (Only one is supported today; this
///   future-proofs the API for per-collection variants.)
pub fn register_default(
    config: &Config,
    embedder: SharedEmbedder,
) -> Result<TypeRegistry, super::TypeError> {
    let mut builder = RegistryBuilder::new();

    if let Some(cfg) = SemanticTextConfig::from_config(config) {
        builder = builder.register(SemanticTextHandler::new(cfg, embedder.clone()));
    }

    Ok(builder.build())
}

/// Convenience wrapper used by tests: build a registry from an explicit
/// list of handlers, no config involved.
pub fn registry_with(handlers: Vec<Arc<dyn super::TypeHandler>>) -> TypeRegistry {
    handlers
        .into_iter()
        .fold(RegistryBuilder::new(), |b, h| b.register_arc(h))
        .build()
}
