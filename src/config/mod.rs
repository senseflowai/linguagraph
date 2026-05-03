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
    pub metadata: MetadataConfig,
    /// Per-type configuration. Each `[types.<TypeId>]` block becomes one
    /// entry in this map and is read by the corresponding handler at
    /// registry-build time. The map is open-ended on purpose —
    /// adding a new type does not require touching this struct.
    #[serde(default)]
    pub types: BTreeMap<String, TypeConfig>,
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

/// Property-metadata cache settings. The default backend is a JSON file;
/// future SQL or KV backends will plug in via [`crate::metadata::MetadataStore`]
/// without changing this config shape (add a `backend` field at that point).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetadataConfig {
    #[serde(default = "default_metadata_path")]
    pub cache_path: String,
}

impl Default for MetadataConfig {
    fn default() -> Self {
        Self { cache_path: default_metadata_path() }
    }
}

fn default_metadata_path() -> String {
    ".linguagraph/property_metadata.json".into()
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
    /// Minimum similarity score required for a vector-search hit to
    /// survive. Used by the SemanticText handler to filter
    /// `libqlink.search` results before they reach the MATCH.
    #[serde(default)]
    pub threshold: Option<f64>,
    /// Embedding dimension hint, used when no real model is loaded
    /// (e.g. for the mock embedder in tests).
    #[serde(default)]
    pub embedding_dim: Option<usize>,
    /// Anything else the handler wants to read.
    #[serde(flatten, default)]
    pub extra: BTreeMap<String, toml::Value>,
}
