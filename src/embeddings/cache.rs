//! Content-addressed embedding cache.
//!
//! The prompt component embeds one short passage per ontology entity and
//! per property to select the query-relevant schema slice. Those passages
//! are stable across requests, so recomputing their vectors on every call
//! is pure waste. [`EmbeddingCache`] persists them to a single JSON file
//! and only invokes the embedder for entries it has never seen.
//!
//! Two levels of validation keep the cache honest without a manual
//! bust step:
//!
//! * **Header hash** — the file records the `{model, dim}` it was built
//!   with. On load, a mismatch against the active embedder discards every
//!   entry, so switching models never mixes incompatible vectors.
//! * **Content key** — each entry is keyed by a hash of the exact text
//!   that produced it (see [`content_key`]). If a property's description
//!   or value set changes, its text changes, its key changes, and the
//!   stale vector is simply never looked up again — the new text misses
//!   and is recomputed.

use std::path::Path;

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use tokio::fs;
use uuid::Uuid;

use super::{EmbedError, Embedder};

/// Stable namespace for [`Uuid::new_v5`] content keys. Bytes spell
/// `linguagraph-emb\0`; the value only needs to be fixed, not meaningful.
const CACHE_NAMESPACE: Uuid = Uuid::from_bytes([
    0x6c, 0x69, 0x6e, 0x67, 0x75, 0x61, 0x67, 0x72, 0x61, 0x70, 0x68, 0x2d, 0x65, 0x6d, 0x62, 0x00,
]);

/// A JSON-backed, content-addressed cache of text embeddings.
///
/// Cheap to construct empty; deserialized from disk via [`Self::load`].
/// Entries added through [`Self::embed_cached`] mark the cache dirty so
/// [`Self::save`] can skip a no-op write when nothing changed.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingCache {
    /// Embedding model identifier this cache was built with (a model
    /// path, or `"mock"` when no real model is configured).
    model: String,
    /// Vector length of every stored entry.
    dim: usize,
    /// `content_key(text) -> vector`.
    #[serde(default)]
    entries: BTreeMap<String, Vec<f32>>,
    /// Set when [`Self::embed_cached`] computed at least one new vector.
    /// Not serialized — it is a per-session write flag.
    #[serde(skip)]
    dirty: bool,
}

impl EmbeddingCache {
    /// Build an empty cache tagged with `model`/`dim`.
    pub fn new(model: impl Into<String>, dim: usize) -> Self {
        Self {
            model: model.into(),
            dim,
            entries: BTreeMap::new(),
            dirty: false,
        }
    }

    /// Load the cache from `path`, validating it against the active
    /// `{model, dim}`.
    ///
    /// Returns a fresh empty cache when the file is missing, empty,
    /// unparseable, or built for a different model/dimension — the cache
    /// is disposable, so a bad file is rebuilt rather than fatal.
    pub async fn load(path: impl AsRef<Path>, model: &str, dim: usize) -> Self {
        let bytes = match fs::read(path.as_ref()).await {
            Ok(bytes) => bytes,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Self::new(model, dim),
            Err(e) => {
                tracing::warn!(path = %path.as_ref().display(), error = %e, "embedding cache read failed; starting empty");
                return Self::new(model, dim);
            }
        };
        if bytes.is_empty() {
            return Self::new(model, dim);
        }
        match serde_json::from_slice::<EmbeddingCache>(&bytes) {
            Ok(cache) if cache.model == model && cache.dim == dim => EmbeddingCache {
                dirty: false,
                ..cache
            },
            Ok(_) => {
                tracing::info!(
                    path = %path.as_ref().display(),
                    "embedding cache model/dim changed; discarding stale entries"
                );
                Self::new(model, dim)
            }
            Err(e) => {
                tracing::warn!(path = %path.as_ref().display(), error = %e, "embedding cache parse failed; starting empty");
                Self::new(model, dim)
            }
        }
    }

    /// Persist the cache to `path` with an atomic write. No-op when
    /// nothing was added since the last load (`dirty == false`).
    pub async fn save(&self, path: impl AsRef<Path>) -> Result<(), EmbedError> {
        if !self.dirty {
            return Ok(());
        }
        let path = path.as_ref();
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent)
                    .await
                    .map_err(|e| EmbedError::Io(e.to_string()))?;
            }
        }
        let body =
            serde_json::to_vec_pretty(self).map_err(|e| EmbedError::Backend(e.to_string()))?;
        let tmp = path.with_extension("json.tmp");
        fs::write(&tmp, &body)
            .await
            .map_err(|e| EmbedError::Io(e.to_string()))?;
        fs::rename(&tmp, path)
            .await
            .map_err(|e| EmbedError::Io(e.to_string()))?;
        Ok(())
    }

    /// Number of cached vectors.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// True when no vectors are cached.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Embed `texts`, returning one vector per input in order, computing
    /// only the entries not already cached.
    ///
    /// Cache misses across the whole batch are embedded in a single
    /// [`Embedder::embed_batch`] call (deduplicated by content key), then
    /// stored for reuse. Hits never touch the embedder.
    pub fn embed_cached(
        &mut self,
        texts: &[String],
        embedder: &dyn Embedder,
    ) -> Result<Vec<Vec<f32>>, EmbedError> {
        let keys: Vec<String> = texts
            .iter()
            .map(|t| content_key(&self.model, self.dim, t))
            .collect();

        // Deduplicate misses by key so identical passages embed once.
        let mut misses: BTreeMap<String, &str> = BTreeMap::new();
        for (key, text) in keys.iter().zip(texts.iter()) {
            if !self.entries.contains_key(key) {
                misses.entry(key.clone()).or_insert(text.as_str());
            }
        }

        if !misses.is_empty() {
            let miss_keys: Vec<String> = misses.keys().cloned().collect();
            let miss_texts: Vec<&str> = misses.values().copied().collect();
            let vectors = embedder.embed_batch(&miss_texts)?;
            if vectors.len() != miss_texts.len() {
                return Err(EmbedError::Backend(format!(
                    "embedder returned {} vectors for {} texts",
                    vectors.len(),
                    miss_texts.len()
                )));
            }
            for (key, vector) in miss_keys.into_iter().zip(vectors.into_iter()) {
                self.entries.insert(key, vector);
            }
            self.dirty = true;
        }

        Ok(keys
            .iter()
            .map(|key| {
                self.entries
                    .get(key)
                    .cloned()
                    .expect("every key is cached after filling misses")
            })
            .collect())
    }
}

/// Deterministic content key for a passage under a given model/dim.
///
/// A UUIDv5 (SHA-1) over `"{model}|{dim}|{text}"` — stable across
/// processes and Rust versions, so the on-disk cache round-trips.
fn content_key(model: &str, dim: usize, text: &str) -> String {
    let payload = format!("{model}|{dim}|{text}");
    Uuid::new_v5(&CACHE_NAMESPACE, payload.as_bytes())
        .simple()
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embeddings::MockEmbedder;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};

    static COUNTER: AtomicU64 = AtomicU64::new(0);

    fn tmp_path(name: &str) -> std::path::PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!("linguagraph-embcache-{nanos}-{n}-{name}.json"))
    }

    /// A wrapper that counts how many texts were sent to the embedder,
    /// so tests can assert that cache hits skip recomputation.
    #[derive(Debug)]
    struct CountingEmbedder {
        inner: MockEmbedder,
        calls: std::sync::atomic::AtomicUsize,
    }

    impl CountingEmbedder {
        fn new(dim: usize) -> Self {
            Self {
                inner: MockEmbedder::new(dim),
                calls: std::sync::atomic::AtomicUsize::new(0),
            }
        }
        fn embedded(&self) -> usize {
            self.calls.load(Ordering::Relaxed)
        }
    }

    impl Embedder for CountingEmbedder {
        fn dim(&self) -> usize {
            self.inner.dim()
        }
        fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbedError> {
            self.calls.fetch_add(texts.len(), Ordering::Relaxed);
            self.inner.embed_batch(texts)
        }
    }

    #[test]
    fn miss_then_hit_skips_recompute() {
        let embedder = CountingEmbedder::new(16);
        let mut cache = EmbeddingCache::new("mock", 16);
        let texts = vec!["alpha".to_string(), "beta".to_string()];

        let first = cache.embed_cached(&texts, &embedder).unwrap();
        assert_eq!(embedder.embedded(), 2);
        assert_eq!(cache.len(), 2);

        let second = cache.embed_cached(&texts, &embedder).unwrap();
        // No new embedder calls: both were hits.
        assert_eq!(embedder.embedded(), 2);
        assert_eq!(first, second);
    }

    #[test]
    fn duplicate_texts_embed_once() {
        let embedder = CountingEmbedder::new(16);
        let mut cache = EmbeddingCache::new("mock", 16);
        let texts = vec!["same".to_string(), "same".to_string(), "same".to_string()];

        let out = cache.embed_cached(&texts, &embedder).unwrap();
        assert_eq!(embedder.embedded(), 1, "identical passages embed once");
        assert_eq!(out.len(), 3);
        assert_eq!(out[0], out[1]);
        assert_eq!(out[1], out[2]);
    }

    #[test]
    fn changed_text_recomputes_and_keeps_old_entry() {
        let embedder = CountingEmbedder::new(16);
        let mut cache = EmbeddingCache::new("mock", 16);

        cache
            .embed_cached(&["type: keyword".to_string()], &embedder)
            .unwrap();
        assert_eq!(cache.len(), 1);

        // A different passage is a different content key -> a miss.
        cache
            .embed_cached(&["type: enum".to_string()], &embedder)
            .unwrap();
        assert_eq!(embedder.embedded(), 2);
        assert_eq!(cache.len(), 2, "old and new entries coexist");
    }

    #[tokio::test]
    async fn save_then_load_round_trips() {
        let path = tmp_path("round-trip");
        let embedder = MockEmbedder::new(16);
        let mut cache = EmbeddingCache::new("mock", 16);
        let want = cache
            .embed_cached(&["one".to_string(), "two".to_string()], &embedder)
            .unwrap();
        cache.save(&path).await.unwrap();

        let loaded = EmbeddingCache::load(&path, "mock", 16).await;
        assert_eq!(loaded.len(), 2);
        // Loaded vectors are reused without hitting the embedder again.
        let mut loaded = loaded;
        let counting = tests_counting(16);
        let got = loaded
            .embed_cached(&["one".to_string(), "two".to_string()], &counting)
            .unwrap();
        assert_eq!(counting.embedded(), 0);
        assert_eq!(want, got);
        let _ = std::fs::remove_file(&path);
    }

    fn tests_counting(dim: usize) -> CountingEmbedder {
        CountingEmbedder::new(dim)
    }

    #[tokio::test]
    async fn header_mismatch_discards_entries() {
        let path = tmp_path("header-mismatch");
        let embedder = MockEmbedder::new(16);
        let mut cache = EmbeddingCache::new("model-a", 16);
        cache
            .embed_cached(&["x".to_string()], &embedder)
            .unwrap();
        cache.save(&path).await.unwrap();

        // Different model -> stale file is discarded.
        let reloaded = EmbeddingCache::load(&path, "model-b", 16).await;
        assert!(reloaded.is_empty());

        // Different dim -> also discarded.
        let reloaded_dim = EmbeddingCache::load(&path, "model-a", 32).await;
        assert!(reloaded_dim.is_empty());
        let _ = std::fs::remove_file(&path);
    }

    #[tokio::test]
    async fn missing_file_loads_empty() {
        let path = tmp_path("does-not-exist");
        let cache = EmbeddingCache::load(&path, "mock", 8).await;
        assert!(cache.is_empty());
    }

    #[tokio::test]
    async fn clean_cache_save_is_noop() {
        let path = tmp_path("noop");
        let cache = EmbeddingCache::new("mock", 8);
        cache.save(&path).await.unwrap();
        assert!(!path.exists(), "no write when nothing changed");
    }
}
