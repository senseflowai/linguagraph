//! Pluggable embedding backend used by the [`crate::types::handlers::SemanticTextHandler`]
//! and any future type that turns text into vectors.
//!
//! The [`Embedder`] trait is deliberately small: a single `embed_batch`
//! call. That keeps the contract trivial to mock in tests and matches
//! how llama.cpp wants to be driven (one [`LlamaContext`] per batch is
//! cheaper than one per item).
//!
//! Concrete backends:
//!
//! * [`mock::MockEmbedder`] — deterministic hash-based vectors for unit
//!   tests and the default fallback when no model is configured.
//! * [`llama::LlamaEmbedder`] — feature-gated wrapper around
//!   `llama-cpp-2`. Loads a GGUF embedding model once and reuses it.

pub mod bm25;
pub mod mock;
pub mod store;

#[cfg(feature = "llama")]
pub mod llama;

use std::sync::Arc;

use thiserror::Error;

pub use mock::MockEmbedder;
pub use store::{
    ensure_indexed, point_id, EmbeddingFilter, EmbeddingIndex, EmbeddingKind, EmbeddingPayload,
    EmbeddingStore, InMemoryEmbeddingStore, RawScoredHit, ScoredHit, StoreError, StoredEmbedding,
};

#[cfg(feature = "llama")]
pub use llama::LlamaEmbedder;

#[derive(Debug, Error)]
pub enum EmbedError {
    #[error("embedder I/O error: {0}")]
    Io(String),
    #[error("embedder backend error: {0}")]
    Backend(String),
    #[error("embedder is not configured for model '{0}'")]
    Unconfigured(String),
}

/// Embedding backend.
///
/// Synchronous on purpose — embedders run on the same thread and the
/// pipeline drives concurrency above. Implementations that *must* be
/// async can wrap a `tokio::task::spawn_blocking` themselves.
pub trait Embedder: Send + Sync + std::fmt::Debug {
    /// Vector length the backend produces. The pipeline checks this to
    /// catch model/collection mismatches before talking to qlink.
    fn dim(&self) -> usize;

    /// Embed many inputs at once. Implementations should batch
    /// internally for efficiency. The output length must match the
    /// input length.
    fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbedError>;

    /// Convenience wrapper around [`Self::embed_batch`] for a single
    /// input.
    fn embed(&self, text: &str) -> Result<Vec<f32>, EmbedError> {
        let mut out = self.embed_batch(&[text])?;
        out.pop()
            .ok_or_else(|| EmbedError::Backend("empty embedder output".into()))
    }
}

/// Shared embedder reference threaded through the pipeline.
pub type SharedEmbedder = Arc<dyn Embedder>;

/// Shared embedding store reference (Qdrant or in-memory).
pub type SharedEmbeddingStore = Arc<dyn store::EmbeddingStore>;

/// Cross-encoder style reranker used after coarse embedding retrieval.
pub trait Reranker: Send + Sync + std::fmt::Debug {
    /// Return one score per document. Scores are expected to be normalized
    /// into a comparable range, typically `[0, 1]`.
    fn rerank(&self, query: &str, documents: &[String]) -> Result<Vec<f64>, EmbedError>;
}

/// Shared reranker reference.
pub type SharedReranker = Arc<dyn Reranker>;

#[derive(Debug)]
pub struct EmbeddingReranker {
    embedder: SharedEmbedder,
}

impl EmbeddingReranker {
    pub fn new(embedder: SharedEmbedder) -> Self {
        Self { embedder }
    }
}

impl Reranker for EmbeddingReranker {
    fn rerank(&self, query: &str, documents: &[String]) -> Result<Vec<f64>, EmbedError> {
        if documents.is_empty() {
            return Ok(Vec::new());
        }
        let query_embedding = self.embedder.embed(query)?;
        let refs: Vec<&str> = documents.iter().map(String::as_str).collect();
        let document_embeddings = self.embedder.embed_batch(&refs)?;
        if document_embeddings.len() != documents.len() {
            return Err(EmbedError::Backend(format!(
                "reranker embedder returned {} vectors for {} documents",
                document_embeddings.len(),
                documents.len()
            )));
        }
        document_embeddings
            .iter()
            .map(|embedding| {
                if embedding.len() != query_embedding.len() {
                    return Err(EmbedError::Backend(format!(
                        "reranker embedding dimension mismatch: document vector has {}, query vector has {}",
                        embedding.len(),
                        query_embedding.len()
                    )));
                }
                Ok(cosine_similarity(&query_embedding, embedding).max(0.0) as f64)
            })
            .collect()
    }
}

/// Build the default embedder for the configured profile.
///
/// Order of preference:
///
/// 1. The `llama` feature is on **and** `model_path` is `Some(path)` —
///    return a [`LlamaEmbedder`]. Errors load-side surface up the
///    stack so missing/corrupted models fail loud.
/// 2. Otherwise return a [`MockEmbedder`] sized to `dim`. This is
///    intentional: tests, dry-runs, and CI builds without GGUF assets
///    keep working end-to-end.
pub fn default_embedder(
    model_path: Option<&str>,
    dim: usize,
) -> Result<SharedEmbedder, EmbedError> {
    #[cfg(feature = "llama")]
    {
        if let Some(path) = model_path {
            return Ok(Arc::new(LlamaEmbedder::load(path)?));
        }
    }
    #[cfg(not(feature = "llama"))]
    {
        let _ = model_path;
    }
    Ok(Arc::new(MockEmbedder::new(dim)))
}

pub fn default_reranker(
    model_path: Option<&str>,
    dim: usize,
) -> Result<SharedReranker, EmbedError> {
    #[cfg(feature = "llama")]
    {
        if let Some(path) = model_path {
            return Ok(Arc::new(LlamaEmbedder::load(path)?));
        }
    }
    #[cfg(not(feature = "llama"))]
    {
        let _ = model_path;
    }
    Ok(Arc::new(EmbeddingReranker::new(Arc::new(
        MockEmbedder::new(dim),
    ))))
}

pub(crate) fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut dot = 0.0f32;
    let mut a_norm = 0.0f32;
    let mut b_norm = 0.0f32;

    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        a_norm += x * x;
        b_norm += y * y;
    }

    if a_norm == 0.0 || b_norm == 0.0 {
        0.0
    } else {
        dot / (a_norm.sqrt() * b_norm.sqrt())
    }
}
