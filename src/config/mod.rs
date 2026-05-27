//! TOML-backed configuration with optional environment overrides.

mod loader;

pub use loader::{load, load_from_str, ConfigError};

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub database: DatabaseConfig,
    #[serde(default)]
    pub llm: LlmConfig,
    #[serde(default)]
    pub query: QueryConfig,
    #[serde(default)]
    pub graph_specification: GraphSpecificationConfig,
    /// Prompt-generation settings (ontologies file, default domain).
    #[serde(default)]
    pub prompt: PromptConfig,
    /// Per-type configuration. Each `[types.<TypeId>]` block becomes one
    /// entry in this map and is read by the corresponding handler at
    /// registry-build time. The map is open-ended on purpose —
    /// adding a new type does not require touching this struct.
    #[serde(default)]
    pub types: BTreeMap<String, TypeConfig>,
}

/// Prompt-generation configuration block (`[prompt]` in TOML).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PromptConfig {
    /// Path to a JSON file with domain ontologies. When omitted, the
    /// built-in catalog (bundled `legal` vocabulary) is used.
    #[serde(default)]
    pub ontologies_path: Option<String>,
    /// Domain selected by knowledge-extract when the caller does not
    /// pass one explicitly.
    #[serde(default)]
    pub default_domain: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatabaseConfig {
    pub uri: String,
    pub user: String,
    pub password: String,
    #[serde(default = "default_database_name")]
    pub database: String,
    #[serde(default = "default_max_connections")]
    pub max_connections: u32,
    #[serde(default = "default_query_timeout")]
    pub query_timeout_secs: u64,
}

fn default_database_name() -> String {
    "memgraph".into()
}

fn default_max_connections() -> u32 {
    16
}
fn default_query_timeout() -> u64 {
    30
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmConfig {
    #[serde(default = "default_provider")]
    pub provider: String,
    #[serde(default = "default_model")]
    pub model: String,
    #[serde(default)]
    pub temperature: f32,
    #[serde(default)]
    pub max_tokens: Option<u32>,
}

impl Default for LlmConfig {
    fn default() -> Self {
        Self {
            provider: default_provider(),
            model: default_model(),
            temperature: 0.0,
            max_tokens: None,
        }
    }
}

fn default_provider() -> String {
    "anthropic".into()
}
fn default_model() -> String {
    "claude-opus-4-7".into()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryConfig {
    #[serde(default = "default_max_depth")]
    pub max_traversal_depth: u32,
    #[serde(default = "default_limit")]
    pub default_limit: u32,
}

impl Default for QueryConfig {
    fn default() -> Self {
        Self {
            max_traversal_depth: default_max_depth(),
            default_limit: default_limit(),
        }
    }
}

fn default_max_depth() -> u32 {
    6
}
fn default_limit() -> u32 {
    100
}

/// Graph specification settings. The specification is used to annotate
/// prompts and to select query-relevant entity types by embedding entity
/// descriptions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphSpecificationConfig {
    /// Path to the graph specification cache file.
    #[serde(default = "default_graph_specification_cache_path")]
    pub cache_path: String,
    /// Path to the embedding model used for graph-specification entity
    /// matching. When omitted, the configured embedder falls back to the
    /// default mock backend in builds without a concrete model.
    #[serde(default)]
    pub embedding_model: Option<String>,
    /// Path to the reranking model used after coarse embedding retrieval.
    /// With the `llama` feature this should be a GGUF reranker compatible
    /// with llama.cpp rank pooling.
    #[serde(default)]
    pub reranking_model: Option<String>,
    /// Embedding dimension hint, used by the mock embedder when no real
    /// model is configured.
    #[serde(default = "default_graph_specification_embedding_dim")]
    pub embedding_dim: usize,
    /// Minimum score required after reranking.
    #[serde(default = "default_graph_specification_reranking_threshold")]
    pub reranking_threshold: f64,
}

impl Default for GraphSpecificationConfig {
    fn default() -> Self {
        Self {
            cache_path: default_graph_specification_cache_path(),
            embedding_model: None,
            reranking_model: None,
            embedding_dim: default_graph_specification_embedding_dim(),
            reranking_threshold: default_graph_specification_reranking_threshold(),
        }
    }
}

fn default_graph_specification_cache_path() -> String {
    crate::graph::DEFAULT_GRAPH_SPECIFICATION_CACHE_PATH.into()
}

fn default_graph_specification_embedding_dim() -> usize {
    384
}

fn default_graph_specification_reranking_threshold() -> f64 {
    0.3
}

/// Open-ended per-type configuration block.
///
/// Handlers read whichever fields they care about and ignore the rest;
/// `extra` collects unknown fields so handlers added later can still
/// pick up values from the same TOML file without a config-schema bump.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TypeConfig {
    /// Path to a model file (used by the SemanticText handler for the
    /// embedding model, available to others as needed).
    #[serde(default)]
    pub embedding_model: Option<String>,
    /// Default qlink/Qdrant collection name.
    #[serde(default)]
    pub collection: Option<String>,
    /// Top-K to request from vector search.
    #[serde(default)]
    pub top_k: Option<u32>,
    /// Minimum cosine similarity required for a vector-search hit
    /// to survive stage 1 of `libqlink.search_reranked` (the KNN
    /// pre-filter). Used by the SemanticText handler.
    #[serde(default)]
    pub threshold: Option<f64>,
    /// Minimum reranker score required for a candidate to appear in
    /// the final result set of `libqlink.search_reranked`. Reranker
    /// scores are sigmoid-bounded to `[0, 1]`; sensible defaults sit
    /// around `0.3`.
    #[serde(default)]
    pub reranker_threshold: Option<f64>,
    /// Embedding dimension hint, used when no real model is loaded
    /// (e.g. for the mock embedder in tests).
    #[serde(default)]
    pub embedding_dim: Option<usize>,
    /// Anything else the handler wants to read.
    #[serde(flatten, default)]
    pub extra: BTreeMap<String, toml::Value>,
}
