//! Shared runtime wiring.
//!
//! Building a fully-configured [`Pipeline`] from a [`Config`] — connecting
//! Memgraph, registering type handlers, attaching the embedder / reranker,
//! and loading the ontology catalog — is the same dance for every entry
//! point (the CLI, the [`crate::service::GraphService`], integration
//! tests). This module owns that construction so transports don't
//! re-implement it and can't drift apart.

use std::sync::Arc;

use crate::config::Config;
use crate::core::Pipeline;
use crate::db::MemgraphClient;
use crate::embeddings::{self, SharedEmbedder, SharedReranker};
use crate::error::Result;
use crate::graph::{JsonFileOntologyCatalogStorage, OntologyCatalogStorage};
use crate::llm::LlmClient;
use crate::types::{self, SharedRegistry};

/// Build a [`SharedRegistry`] of type handlers from `cfg`, plus the
/// embedder they share. Always returns a registry (possibly empty) so
/// callers can pass it through unconditionally.
pub fn build_registry(cfg: &Config) -> Result<(SharedRegistry, Option<SharedEmbedder>)> {
    let dim = cfg
        .types
        .get("SemanticText")
        .and_then(|t| t.embedding_dim)
        .unwrap_or(384);
    let model = cfg
        .types
        .get("SemanticText")
        .and_then(|t| t.embedding_model.clone());
    let embedder = embeddings::default_embedder(model.as_deref(), dim).map_err(|e| {
        crate::error::Error::Ingest(crate::ingest::IngestError::Type(format!(
            "embedder init: {e}"
        )))
    })?;
    let registry = types::handlers::register_default(cfg, embedder.clone()).map_err(|e| {
        crate::error::Error::Ingest(crate::ingest::IngestError::Type(format!(
            "registry init: {e}"
        )))
    })?;
    Ok((Arc::new(registry), Some(embedder)))
}

/// Build the SemanticText cross-encoder reranker from
/// `[types.SemanticText].reranking_model`, or `None` when unset (the
/// pipeline then defers reranking to qlink's `search_hybrid_reranked`).
pub fn build_semantic_text_reranker(cfg: &Config) -> Result<Option<SharedReranker>> {
    let Some(model) = cfg
        .types
        .get("SemanticText")
        .and_then(|t| t.reranking_model.clone())
    else {
        return Ok(None);
    };
    let dim = cfg
        .types
        .get("SemanticText")
        .and_then(|t| t.embedding_dim)
        .unwrap_or(384);
    let reranker = embeddings::default_reranker(Some(&model), dim).map_err(|e| {
        crate::error::Error::Ingest(crate::ingest::IngestError::Type(format!(
            "SemanticText reranker init: {e}"
        )))
    })?;
    Ok(Some(reranker))
}

/// Construct an [`LlmClient`] from `cfg` (an OpenAI-compatible endpoint,
/// e.g. a self-hosted vLLM server). Gated behind the `openai` feature;
/// without it, the natural-language `ask` path is unavailable.
#[cfg(feature = "openai")]
pub fn build_llm_client(cfg: &Config) -> Result<Arc<dyn LlmClient>> {
    Ok(Arc::new(crate::llm::OpenAiClient::from_config(&cfg.llm)))
}

#[cfg(not(feature = "openai"))]
pub fn build_llm_client(_cfg: &Config) -> Result<Arc<dyn LlmClient>> {
    Err(crate::error::Error::Nl(
        "the `openai` feature is disabled; rebuild with `--features openai`".to_string(),
    ))
}

/// Connect to Memgraph and assemble a read-ready [`Pipeline`] from `cfg`,
/// scoped to the optional `prefix_label` / `prefix_index`. Mirrors the
/// CLI's `run` wiring: registry + embedder, ontology-catalog storage,
/// SemanticText reranker, and a loaded ontology catalog.
pub async fn build_query_pipeline(
    cfg: &Config,
    prefix_label: Option<String>,
    prefix_index: Option<String>,
) -> Result<Pipeline> {
    let client = MemgraphClient::connect(&cfg.database).await?;
    let (registry, embedder) = build_registry(cfg)?;
    let spec_storage: Arc<dyn OntologyCatalogStorage> = Arc::new(
        JsonFileOntologyCatalogStorage::new(&cfg.ontology_catalog.cache_path),
    );
    let mut pipeline = Pipeline::new(Arc::new(client), cfg)
        .with_registry(registry)
        .with_ontology_catalog_storage(spec_storage)
        .with_prefix_label(prefix_label)
        .with_prefix_index(prefix_index);
    if let Some(e) = embedder {
        pipeline = pipeline.with_embedder(e);
    }
    if let Some(reranker) = build_semantic_text_reranker(cfg)? {
        pipeline = pipeline.with_reranker(reranker);
    }
    pipeline.load_ontology_catalog().await?;
    Ok(pipeline)
}
