//! Bundled type handlers.
//!
//! Each handler lives in its own submodule and is exported through
//! [`super::TypeRegistry`] via a small `register_*` helper. Adding a new
//! type means writing one file here and one line in
//! [`crate::types::handlers::register_default`].
//!
//! ## Built-in catalogue
//!
//! Every registry produced by [`register_default`] / [`core_registry`]
//! contains the five scalar types `Text`, `Number`, `Boolean`, `Date`,
//! `Timestamp`. They are responsible for validating and converting raw
//! JSON values during ingestion (string `"50%"` → `0.5`, integer epoch
//! → ISO-8601 string, …) and they are *always* registered — a mapping
//! that uses any of them is portable across deployments without a
//! configuration block.
//!
//! Optional handlers (today: [`SemanticTextHandler`]) are registered
//! when their configuration block is present.

pub mod core;
pub mod semantic_text;

pub use core::{
    boolean_handler, date_handler, number_handler, text_handler, timestamp_handler,
    BooleanParser, DateParser, NumberParser, ScalarParser, ScalarTypeHandler, TextParser,
    TimestampParser,
};
pub use semantic_text::{SemanticTextConfig, SemanticTextHandler};

use std::sync::Arc;

use super::{RegistryBuilder, TypeRegistry};
use crate::config::Config;
use crate::embeddings::SharedEmbedder;

/// Build the default registry from `config` and a shared embedder.
///
/// Always registers the built-in scalar types (Text, Number, Boolean,
/// Date, Timestamp) plus any optional handlers whose configuration is
/// present:
///
/// * [`SemanticTextHandler`] — one instance per `[types.SemanticText]`
///   block in the configuration. (Only one is supported today; this
///   future-proofs the API for per-collection variants.)
pub fn register_default(
    config: &Config,
    embedder: SharedEmbedder,
) -> Result<TypeRegistry, super::TypeError> {
    let mut builder = register_core(RegistryBuilder::new());

    if let Some(cfg) = SemanticTextConfig::from_config(config) {
        builder = builder.register(SemanticTextHandler::new(cfg, embedder.clone()));
    }

    Ok(builder.build())
}

/// Build a registry containing only the built-in scalar types.
///
/// Useful when no embedder is available (tests, plain ingestion
/// pipelines) but mappings still need their values parsed.
pub fn core_registry() -> TypeRegistry {
    register_core(RegistryBuilder::new()).build()
}

/// Add the five built-in scalar types to an existing builder. Exposed
/// so callers that want a custom registry can still inherit the core
/// types in one line.
pub fn register_core(builder: RegistryBuilder) -> RegistryBuilder {
    builder
        .register(text_handler())
        .register(number_handler())
        .register(boolean_handler())
        .register(date_handler())
        .register(timestamp_handler())
}

/// Convenience wrapper used by tests: build a registry from an explicit
/// list of handlers, no config involved. Core types are included by
/// default; pass `false` for `include_core` if you need a registry that
/// genuinely contains only the handlers in `handlers`.
pub fn registry_with(handlers: Vec<Arc<dyn super::TypeHandler>>) -> TypeRegistry {
    let mut b = register_core(RegistryBuilder::new());
    for h in handlers {
        b = b.register_arc(h);
    }
    b.build()
}
