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
    #[serde(default, alias = "graph_specification")]
    pub ontology_catalog: OntologyCatalogConfig,
    /// Prompt-generation settings (ontologies file, default domain).
    #[serde(default)]
    pub prompt: PromptConfig,
    /// Ingestion-time knobs. Currently holds the soft-merge similarity
    /// resolver configuration; future ingestion flags belong here too.
    #[serde(default)]
    pub ingest: IngestConfig,
    /// Per-type configuration. Each `[types.<TypeId>]` block becomes one
    /// entry in this map and is read by the corresponding handler at
    /// registry-build time. The map is open-ended on purpose —
    /// adding a new type does not require touching this struct.
    #[serde(default)]
    pub types: BTreeMap<String, TypeConfig>,
}

/// Ingestion-time configuration (`[ingest]` in TOML).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct IngestConfig {
    /// Soft primary-key similarity-merge resolver settings.
    #[serde(default)]
    pub soft_merge: SoftMergeConfig,
}

/// Configuration for the soft-merge resolver that runs before every
/// `Pipeline::ingest` and tries to dedupe `PrimaryKey::Soft` entities
/// against an existing graph by vector similarity.
///
/// The resolver runs a staged decision pipeline per candidate:
///   1. retrieve top-K hits above `similarity_threshold` (consideration floor),
///   2. score each candidate against the top hit on multiple signals,
///   3. route to AutoMerge / NeedsReview / NoMerge.
///
/// Defaults bias toward false-split-over-false-merge: an entity only
/// auto-merges when the embedding signal is strong AND consistent with
/// lexical similarity AND not ambiguous against runners-up AND has no
/// hard conflict on disambiguating properties.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SoftMergeConfig {
    /// Consideration floor: hits below this cosine score are dropped at
    /// the Cypher layer and never even surface for scoring. Above this
    /// they enter the decision pipeline.
    #[serde(default = "default_soft_merge_similarity_threshold")]
    pub similarity_threshold: f64,
    /// How many candidate vectors to fetch per entity from
    /// `libqlink.search_labeled`. Used in full now — the resolver looks
    /// at the whole top-K list (margin, ambiguity counts, runners-up)
    /// instead of just the best hit.
    #[serde(default = "default_soft_merge_top_k")]
    pub top_k: u32,
    /// Top-1 cosine score required for an automatic rewrite. Above
    /// `similarity_threshold` but below this, the candidate is routed
    /// to NeedsReview rather than AutoMerge.
    #[serde(default = "default_soft_merge_auto_merge_threshold")]
    pub auto_merge_threshold: f64,
    /// Top-1 floor for emitting a review record at all. Candidates
    /// whose top hit falls below this are quietly routed to NoMerge —
    /// they're obviously distinct and clutter reviews otherwise.
    #[serde(default = "default_soft_merge_review_threshold")]
    pub review_threshold: f64,
    /// Minimum `top1 - top2` gap required for AutoMerge. When there's
    /// only one hit, margin is treated as +∞ and the gate passes.
    #[serde(default = "default_soft_merge_min_margin")]
    pub min_margin: f64,
    /// Minimum Jaro-Winkler against the top hit's primary-name line
    /// required for AutoMerge. Pure dense-embedding similarity is too
    /// permissive — different entities embed close together when their
    /// canonical text shares prop patterns.
    #[serde(default = "default_soft_merge_min_lexical_similarity")]
    pub min_lexical_similarity: f64,
    /// At most this many runner-up hits may sit within
    /// `close_candidate_delta` of the top score for an AutoMerge.
    /// Above this we route to NeedsReview — too many close candidates
    /// is itself a hint the LLM extraction was ambiguous.
    #[serde(default = "default_soft_merge_max_close_candidates")]
    pub max_close_candidates: usize,
    /// Score-gap window for counting "close" runner-up candidates.
    #[serde(default = "default_soft_merge_close_candidate_delta")]
    pub close_candidate_delta: f64,
    /// When false (default), candidates whose `_canonical` text consists
    /// only of `type: X` (no other properties) are never auto-merged —
    /// they'd otherwise collapse onto whichever node of that type
    /// embeds nearest, which is almost never what users want.
    #[serde(default = "default_soft_merge_allow_type_only_auto_merge")]
    pub allow_type_only_auto_merge: bool,
    /// Populate `SoftMergeReport.review_candidates`. Off by default in
    /// principle, but practical experience says reviewers want them, so
    /// the default here is `true`.
    #[serde(default = "default_soft_merge_emit_review_candidates")]
    pub emit_review_candidates: bool,
    /// Cap on the size of the per-candidate hit list in a review record
    /// (top hit + runners-up combined).
    #[serde(default = "default_soft_merge_review_max_candidates")]
    pub review_max_candidates: usize,
    /// Disambiguating properties whose non-null values must agree before
    /// AutoMerge can fire. When both incoming and candidate have a
    /// value here and they differ (case-sensitive string compare),
    /// AutoMerge is blocked and the candidate is routed to NeedsReview
    /// with a `HardConflict` reason. Configurable per domain.
    #[serde(default = "default_soft_merge_conflict_properties")]
    pub conflict_properties: Vec<String>,
}

impl Default for SoftMergeConfig {
    fn default() -> Self {
        Self {
            similarity_threshold: default_soft_merge_similarity_threshold(),
            top_k: default_soft_merge_top_k(),
            auto_merge_threshold: default_soft_merge_auto_merge_threshold(),
            review_threshold: default_soft_merge_review_threshold(),
            min_margin: default_soft_merge_min_margin(),
            min_lexical_similarity: default_soft_merge_min_lexical_similarity(),
            max_close_candidates: default_soft_merge_max_close_candidates(),
            close_candidate_delta: default_soft_merge_close_candidate_delta(),
            allow_type_only_auto_merge: default_soft_merge_allow_type_only_auto_merge(),
            emit_review_candidates: default_soft_merge_emit_review_candidates(),
            review_max_candidates: default_soft_merge_review_max_candidates(),
            conflict_properties: default_soft_merge_conflict_properties(),
        }
    }
}

fn default_soft_merge_similarity_threshold() -> f64 {
    0.95
}

fn default_soft_merge_top_k() -> u32 {
    10
}

fn default_soft_merge_auto_merge_threshold() -> f64 {
    0.96
}

fn default_soft_merge_review_threshold() -> f64 {
    0.75
}

fn default_soft_merge_min_margin() -> f64 {
    0.08
}

fn default_soft_merge_min_lexical_similarity() -> f64 {
    0.70
}

fn default_soft_merge_max_close_candidates() -> usize {
    1
}

fn default_soft_merge_close_candidate_delta() -> f64 {
    0.03
}

fn default_soft_merge_allow_type_only_auto_merge() -> bool {
    false
}

fn default_soft_merge_emit_review_candidates() -> bool {
    true
}

fn default_soft_merge_review_max_candidates() -> usize {
    5
}

fn default_soft_merge_conflict_properties() -> Vec<String> {
    ["id", "email", "url", "isbn", "phone", "ssn", "doi", "ein"]
        .into_iter()
        .map(String::from)
        .collect()
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
    /// Base URL of an OpenAI-compatible chat-completions API. Defaults
    /// to a local vLLM server (`http://localhost:8000/v1`). The client
    /// POSTs to `{base_url}/chat/completions`.
    #[serde(default = "default_llm_base_url")]
    pub base_url: String,
    /// Name of the environment variable holding the API key. For a
    /// local vLLM server this can be left unset (the key is optional).
    #[serde(default = "default_llm_api_key_env")]
    pub api_key_env: String,
}

impl Default for LlmConfig {
    fn default() -> Self {
        Self {
            provider: default_provider(),
            model: default_model(),
            temperature: 0.0,
            max_tokens: None,
            base_url: default_llm_base_url(),
            api_key_env: default_llm_api_key_env(),
        }
    }
}

fn default_provider() -> String {
    "anthropic".into()
}
fn default_model() -> String {
    "claude-opus-4-7".into()
}
fn default_llm_base_url() -> String {
    "http://localhost:8000/v1".into()
}
fn default_llm_api_key_env() -> String {
    "OPENAI_API_KEY".into()
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

/// Ontology catalog settings. The catalog is used to annotate prompts,
/// to select query-relevant entity types by embedding entity
/// descriptions, and to enrich live-schema introspection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OntologyCatalogConfig {
    /// Path to the ontology catalog cache file (JSON backend).
    #[serde(default = "default_ontology_catalog_cache_path")]
    pub cache_path: String,
    /// Path to the embedding model used for entity matching. When
    /// omitted, the configured embedder falls back to the default mock
    /// backend in builds without a concrete model.
    #[serde(default)]
    pub embedding_model: Option<String>,
    /// Path to the reranking model used after coarse embedding retrieval.
    /// With the `llama` feature this should be a GGUF reranker compatible
    /// with llama.cpp rank pooling.
    #[serde(default)]
    pub reranking_model: Option<String>,
    /// Embedding dimension hint, used by the mock embedder when no real
    /// model is configured.
    #[serde(default = "default_ontology_catalog_embedding_dim")]
    pub embedding_dim: usize,
    /// Minimum score required after reranking.
    #[serde(default = "default_ontology_catalog_reranking_threshold")]
    pub reranking_threshold: f64,
}

impl Default for OntologyCatalogConfig {
    fn default() -> Self {
        Self {
            cache_path: default_ontology_catalog_cache_path(),
            embedding_model: None,
            reranking_model: None,
            embedding_dim: default_ontology_catalog_embedding_dim(),
            reranking_threshold: default_ontology_catalog_reranking_threshold(),
        }
    }
}

fn default_ontology_catalog_cache_path() -> String {
    crate::graph::DEFAULT_ONTOLOGY_CATALOG_CACHE_PATH.into()
}

fn default_ontology_catalog_embedding_dim() -> usize {
    384
}

fn default_ontology_catalog_reranking_threshold() -> f64 {
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
