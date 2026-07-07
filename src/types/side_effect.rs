//! Side effects produced during ingestion, drained by the pipeline.
//!
//! A side effect is anything that needs to happen *after* the Memgraph
//! batch lands — typically because it depends on the node's database id
//! (e.g. inserting an embedding into Qdrant). Handlers describe what
//! they want done; the pipeline collects, batches, and executes.

use std::collections::BTreeMap;

use crate::ast::query::Literal;

/// A side effect emitted by a [`TypeHandler`] during ingestion.
///
/// Generic by design: handlers describe *what* they want, the pipeline
/// owns *how* it gets done. New side-effect kinds can be added without
/// changing the handler trait.
#[derive(Debug, Clone)]
pub enum SideEffect {
    /// Embed `text` and upsert the resulting vector into a qlink/Qdrant
    /// collection, keyed by the Memgraph node id of the row identified
    /// by `(label, key_field, key_value)`.
    EmbedAndStore {
        collection: String,
        label: String,
        key_field: String,
        key_value: Literal,
        text: String,
        /// Optional payload label for `qlink.insert_labeled`.
        payload_label: Option<String>,
        /// Free-form metadata available to whoever drains the queue
        /// (e.g. which property name the embedding came from).
        meta: BTreeMap<String, String>,
    },

    /// Embed each string in `texts` *independently* and upsert the
    /// resulting per-sentence matrix as a single multivector Qdrant point
    /// (ColBERT-style late interaction, MaxSim at query time), keyed by
    /// the Memgraph node id of `(label, key_field, key_value)`.
    ///
    /// Used for long `Chunk.text` fragments: splitting the chunk into
    /// sentences and embedding each one keeps the retrieval signal sharp
    /// instead of averaging a whole 1000-token chunk into one noisy
    /// vector. `payload_text` is the full, unsplit chunk text — it backs
    /// the BM25 sparse branch and the reranker payload, exactly as `text`
    /// does for [`SideEffect::EmbedAndStore`].
    EmbedMultiAndStore {
        collection: String,
        label: String,
        key_field: String,
        key_value: Literal,
        /// Per-sentence texts, each embedded into one row of the point's
        /// multivector.
        texts: Vec<String>,
        /// Full fragment text stored for BM25 sparse + reranker payload.
        payload_text: String,
        /// Optional payload label for `qlink.insert_hybrid_multi`.
        payload_label: Option<String>,
        /// Free-form metadata available to whoever drains the queue.
        meta: BTreeMap<String, String>,
    },
}

impl SideEffect {
    /// Embedder-collection grouping key. Used by the pipeline to bucket
    /// effects so each collection's `qlink.insert_batch` is one call.
    pub fn group_key(&self) -> (&str, &str) {
        match self {
            SideEffect::EmbedAndStore {
                collection, label, ..
            }
            | SideEffect::EmbedMultiAndStore {
                collection, label, ..
            } => (collection.as_str(), label.as_str()),
        }
    }
}

/// FIFO queue of side effects threaded through ingestion.
///
/// Cheap to clone (it's a `Vec` wrapped in a newtype) — we keep it as a
/// concrete type rather than a trait so handlers can rely on stable
/// behavior across ingestion runs. Concurrency is the *pipeline's*
/// problem; handlers see only the per-row queue they were handed.
#[derive(Debug, Clone, Default)]
pub struct SideEffectQueue {
    items: Vec<SideEffect>,
}

impl SideEffectQueue {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push(&mut self, eff: SideEffect) {
        self.items.push(eff);
    }

    pub fn extend<I: IntoIterator<Item = SideEffect>>(&mut self, iter: I) {
        self.items.extend(iter);
    }

    pub fn drain(&mut self) -> impl Iterator<Item = SideEffect> + '_ {
        self.items.drain(..)
    }

    pub fn iter(&self) -> std::slice::Iter<'_, SideEffect> {
        self.items.iter()
    }

    pub fn len(&self) -> usize {
        self.items.len()
    }

    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    pub fn into_vec(self) -> Vec<SideEffect> {
        self.items
    }
}
