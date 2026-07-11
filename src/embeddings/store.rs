//! Pluggable vector store for ontology/entity embeddings.
//!
//! Query-driven prompt generation embeds one short passage per ontology
//! domain, entity, and property, then ranks them against the user's query.
//! [`EmbeddingStore`] abstracts where those vectors live and where the
//! ranking happens:
//!
//! * [`crate::db::QdrantClient`] (feature `qdrant`) — vectors are stored in
//!   Qdrant and similarity search runs **server-side**.
//! * [`InMemoryEmbeddingStore`] — vectors held in a process map, ranked
//!   with [`cosine_similarity`]. The test double and the fallback when no
//!   Qdrant endpoint is configured.
//!
//! Points are content-addressed: [`point_id`] hashes the model, dimension,
//! identity, and passage text into a stable UUIDv5, so a changed passage is
//! a fresh point (recomputed) while unchanged passages are reused across
//! requests — the same "cache" behaviour the old file cache provided, but
//! now server-side.

use std::collections::{BTreeMap, HashSet};
use std::sync::Mutex;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use super::{cosine_similarity, EmbedError, Embedder};

/// Stable namespace for [`Uuid::new_v5`] point ids. Bytes spell
/// `linguagraph-vec\0`; only needs to be fixed, not meaningful.
const POINT_NAMESPACE: Uuid = Uuid::from_bytes([
    0x6c, 0x69, 0x6e, 0x67, 0x75, 0x61, 0x67, 0x72, 0x61, 0x70, 0x68, 0x2d, 0x76, 0x65, 0x63, 0x00,
]);

/// What an embedding point describes. Serialized into the point payload so
/// searches can filter by it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum EmbeddingKind {
    Domain,
    Entity,
    Property,
}

impl EmbeddingKind {
    pub fn as_str(self) -> &'static str {
        match self {
            EmbeddingKind::Domain => "domain",
            EmbeddingKind::Entity => "entity",
            EmbeddingKind::Property => "property",
        }
    }
}

/// Payload stored alongside a vector, identifying the ontology element it
/// came from. `entity`/`property` are `None` for coarser kinds.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EmbeddingPayload {
    pub kind: EmbeddingKind,
    pub domain: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub entity: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub property: Option<String>,
}

impl EmbeddingPayload {
    pub fn domain(domain: impl Into<String>) -> Self {
        Self {
            kind: EmbeddingKind::Domain,
            domain: domain.into(),
            entity: None,
            property: None,
        }
    }
    pub fn entity(domain: impl Into<String>, entity: impl Into<String>) -> Self {
        Self {
            kind: EmbeddingKind::Entity,
            domain: domain.into(),
            entity: Some(entity.into()),
            property: None,
        }
    }
    pub fn property(
        domain: impl Into<String>,
        entity: impl Into<String>,
        property: impl Into<String>,
    ) -> Self {
        Self {
            kind: EmbeddingKind::Property,
            domain: domain.into(),
            entity: Some(entity.into()),
            property: Some(property.into()),
        }
    }
}

/// One vector to store.
#[derive(Debug, Clone)]
pub struct StoredEmbedding {
    pub id: Uuid,
    pub vector: Vec<f32>,
    pub payload: EmbeddingPayload,
}

/// Constraints applied to a search. Empty lists mean "no constraint".
#[derive(Debug, Clone, Default)]
pub struct EmbeddingFilter {
    pub kinds: Vec<EmbeddingKind>,
    pub domains: Vec<String>,
}

impl EmbeddingFilter {
    fn accepts(&self, payload: &EmbeddingPayload) -> bool {
        (self.kinds.is_empty() || self.kinds.contains(&payload.kind))
            && (self.domains.is_empty() || self.domains.iter().any(|d| *d == payload.domain))
    }
}

/// A scored search result.
#[derive(Debug, Clone)]
pub struct ScoredHit {
    pub score: f32,
    pub payload: EmbeddingPayload,
}

/// A scored search result carrying the backend point id and its *raw*
/// payload, without imposing the [`EmbeddingPayload`] shape.
///
/// [`ScoredHit`] assumes the ontology/entity payload schema this crate
/// writes through [`EmbeddingStore::upsert`]. That doesn't fit points
/// written by an *external* producer — in particular the qlink
/// `_canonical` collection, whose points are keyed by the Memgraph node
/// id and carry a qlink-defined `{text, label}` payload. `RawScoredHit`
/// hands those back verbatim so the caller (the grounding pass) can do
/// its own defensive extraction.
#[derive(Debug, Clone)]
pub struct RawScoredHit {
    /// Point id exactly as the backend returned it. For the qlink
    /// `_canonical` collection this is the stringified Memgraph node id
    /// (an integer); for this crate's ontology points it is a UUID.
    pub id: String,
    pub score: f32,
    /// Raw, backend-defined point payload. [`serde_json::Value::Null`]
    /// when the backend returned no payload.
    pub payload: serde_json::Value,
}

#[derive(Debug, thiserror::Error)]
pub enum StoreError {
    #[error("embedding store backend error: {0}")]
    Backend(String),
    #[error("embedding store transport error: {0}")]
    Transport(String),
}

impl From<StoreError> for EmbedError {
    fn from(e: StoreError) -> Self {
        EmbedError::Backend(e.to_string())
    }
}

/// A vector store for ontology/entity embeddings. All I/O is async; the
/// embedder that produces vectors stays synchronous (see [`Embedder`]).
#[async_trait]
pub trait EmbeddingStore: Send + Sync + std::fmt::Debug {
    /// Ensure `collection` exists with the given vector dimension (cosine
    /// distance). Idempotent.
    async fn ensure(&self, collection: &str, dim: usize) -> Result<(), StoreError>;

    /// Return the subset of `ids` not already present in `collection`.
    async fn missing(&self, collection: &str, ids: &[Uuid]) -> Result<Vec<Uuid>, StoreError>;

    /// Upsert points into `collection`.
    async fn upsert(
        &self,
        collection: &str,
        points: Vec<StoredEmbedding>,
    ) -> Result<(), StoreError>;

    /// Search `collection` for the nearest points to `vector`, filtered by
    /// `filter`, returning up to `limit` hits with score `>= score_threshold`
    /// (when set), highest score first.
    async fn search(
        &self,
        collection: &str,
        vector: &[f32],
        limit: usize,
        score_threshold: Option<f32>,
        filter: &EmbeddingFilter,
    ) -> Result<Vec<ScoredHit>, StoreError>;

    /// Nearest-neighbour search that returns the raw point id + payload
    /// (see [`RawScoredHit`]) rather than a typed [`EmbeddingPayload`].
    ///
    /// Used by the query-time *grounding* pass to search collections this
    /// crate does **not** own the payload schema of — notably the qlink
    /// `_canonical` collection, whose point ids are Memgraph node ids.
    /// Unlike [`Self::search`] it applies no [`EmbeddingFilter`]: the
    /// caller filters on the raw payload itself.
    ///
    /// The default implementation returns no hits, so a store that can't
    /// serve raw points simply opts out of grounding (the query then
    /// falls back to the server-side `libqlink` search). Backends that
    /// can serve it — [`crate::db::QdrantClient`] — override this.
    async fn search_raw(
        &self,
        collection: &str,
        vector: &[f32],
        limit: usize,
        score_threshold: Option<f32>,
    ) -> Result<Vec<RawScoredHit>, StoreError> {
        let _ = (collection, vector, limit, score_threshold);
        Ok(Vec::new())
    }

    /// Hybrid (dense ⊕ BM25) nearest-neighbour search over a qlink
    /// `_canonical` collection, fusing a dense branch (the default vector)
    /// and a lexical branch (the `text_bm25` sparse vector built from
    /// `query_text`) with Reciprocal Rank Fusion server-side. When `label`
    /// is set, both branches are filtered to points carrying that `label`
    /// payload. Returns up to `limit` hits, highest fused score first; the
    /// returned `score` is the RRF score (rank-based, not a cosine).
    ///
    /// The lexical branch is what recovers a *literal* term buried in a
    /// long text field — a pure dense KNN ([`Self::search_raw`]) on a short
    /// query against long `_canonical` docs ranks by overall semantic
    /// proximity and misses it. The default implementation has no sparse
    /// index, so it falls back to the dense-only path; backends that store
    /// the BM25 sparse vector — [`crate::db::QdrantClient`] — override this.
    async fn search_hybrid(
        &self,
        collection: &str,
        query_text: &str,
        vector: &[f32],
        label: Option<&str>,
        limit: usize,
    ) -> Result<Vec<RawScoredHit>, StoreError> {
        let _ = (query_text, label);
        self.search_raw(collection, vector, limit, None).await
    }

    /// Lexical (BM25) search over the `text_bm25` sparse vector built from
    /// `query_text`. Because a sparse query only returns points sharing at
    /// least one query term, a bare-value query (e.g. `"freedom"`) returns
    /// exactly the docs that contain that token — literal `contains` via
    /// the index. When `label` is set, results are filtered to that label.
    /// Returns up to `limit` hits, highest BM25 score first.
    ///
    /// The default implementation has no sparse index and returns nothing,
    /// so a store that can't serve BM25 opts out (the query then falls back
    /// to the server-side path); [`crate::db::QdrantClient`] overrides it.
    async fn search_bm25(
        &self,
        collection: &str,
        query_text: &str,
        label: Option<&str>,
        limit: usize,
    ) -> Result<Vec<RawScoredHit>, StoreError> {
        let _ = (collection, query_text, label, limit);
        Ok(Vec::new())
    }
}

/// A shared embedding store plus the collection/model it is addressed with.
/// Threaded through the query-driven selection functions.
#[derive(Debug, Clone, Copy)]
pub struct EmbeddingIndex<'a> {
    pub store: &'a dyn EmbeddingStore,
    pub collection: &'a str,
    /// Embedding-model identifier, folded into each point id so a model
    /// switch never reuses an incompatible vector.
    pub model: &'a str,
}

/// Deterministic point id for a passage: a UUIDv5 over model, dimension,
/// identity (kind/domain/entity/property), and the passage text.
pub fn point_id(model: &str, dim: usize, payload: &EmbeddingPayload, text: &str) -> Uuid {
    let key = format!(
        "{model}|{dim}|{}|{}|{}|{}|{text}",
        payload.kind.as_str(),
        payload.domain,
        payload.entity.as_deref().unwrap_or(""),
        payload.property.as_deref().unwrap_or(""),
    );
    Uuid::new_v5(&POINT_NAMESPACE, key.as_bytes())
}

/// Ensure every `(payload, text)` passage is present in the index,
/// embedding and upserting only the ones the store is missing.
///
/// This is the shared "index-or-reuse" step in front of a search: it keeps
/// the embedder call to the genuinely new passages, exactly like the old
/// file cache's miss-only embedding.
pub async fn ensure_indexed(
    index: &EmbeddingIndex<'_>,
    embedder: &dyn Embedder,
    passages: &[(EmbeddingPayload, String)],
) -> Result<(), EmbedError> {
    if passages.is_empty() {
        return Ok(());
    }
    let dim = embedder.dim();
    let ids: Vec<Uuid> = passages
        .iter()
        .map(|(payload, text)| point_id(index.model, dim, payload, text))
        .collect();

    let missing: HashSet<Uuid> = index
        .store
        .missing(index.collection, &ids)
        .await?
        .into_iter()
        .collect();
    if missing.is_empty() {
        return Ok(());
    }

    // Embed each missing point once (dedup by id).
    let mut seen: HashSet<Uuid> = HashSet::new();
    let mut texts: Vec<&str> = Vec::new();
    let mut coords: Vec<(Uuid, EmbeddingPayload)> = Vec::new();
    for ((payload, text), id) in passages.iter().zip(ids.iter()) {
        if missing.contains(id) && seen.insert(*id) {
            texts.push(text.as_str());
            coords.push((*id, payload.clone()));
        }
    }

    let vectors = embedder.embed_batch(&texts)?;
    if vectors.len() != coords.len() {
        return Err(EmbedError::Backend(format!(
            "embedder returned {} vectors for {} passages",
            vectors.len(),
            coords.len()
        )));
    }
    let points = coords
        .into_iter()
        .zip(vectors)
        .map(|((id, payload), vector)| StoredEmbedding {
            id,
            vector,
            payload,
        })
        .collect();
    index.store.upsert(index.collection, points).await?;
    Ok(())
}

/// Process-local embedding store. Ranks with [`cosine_similarity`]; used in
/// tests and as the fallback when no Qdrant endpoint is configured.
#[derive(Debug, Default)]
pub struct InMemoryEmbeddingStore {
    inner: Mutex<BTreeMap<String, BTreeMap<Uuid, (Vec<f32>, EmbeddingPayload)>>>,
}

impl InMemoryEmbeddingStore {
    pub fn new() -> Self {
        Self::default()
    }
}

#[async_trait]
impl EmbeddingStore for InMemoryEmbeddingStore {
    async fn ensure(&self, collection: &str, _dim: usize) -> Result<(), StoreError> {
        self.inner
            .lock()
            .unwrap()
            .entry(collection.to_string())
            .or_default();
        Ok(())
    }

    async fn missing(&self, collection: &str, ids: &[Uuid]) -> Result<Vec<Uuid>, StoreError> {
        let guard = self.inner.lock().unwrap();
        let present = guard.get(collection);
        Ok(ids
            .iter()
            .filter(|id| present.map(|m| !m.contains_key(id)).unwrap_or(true))
            .copied()
            .collect())
    }

    async fn upsert(
        &self,
        collection: &str,
        points: Vec<StoredEmbedding>,
    ) -> Result<(), StoreError> {
        let mut guard = self.inner.lock().unwrap();
        let map = guard.entry(collection.to_string()).or_default();
        for p in points {
            map.insert(p.id, (p.vector, p.payload));
        }
        Ok(())
    }

    async fn search(
        &self,
        collection: &str,
        vector: &[f32],
        limit: usize,
        score_threshold: Option<f32>,
        filter: &EmbeddingFilter,
    ) -> Result<Vec<ScoredHit>, StoreError> {
        let guard = self.inner.lock().unwrap();
        let Some(map) = guard.get(collection) else {
            return Ok(Vec::new());
        };
        let mut hits: Vec<ScoredHit> = map
            .values()
            .filter(|(_, payload)| filter.accepts(payload))
            .map(|(stored, payload)| ScoredHit {
                score: cosine_similarity(vector, stored),
                payload: payload.clone(),
            })
            .filter(|hit| score_threshold.map(|t| hit.score >= t).unwrap_or(true))
            .collect();
        hits.sort_by(|a, b| b.score.total_cmp(&a.score));
        hits.truncate(limit);
        Ok(hits)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vec3(a: f32, b: f32, c: f32) -> Vec<f32> {
        vec![a, b, c]
    }

    #[tokio::test]
    async fn upsert_missing_search_round_trip() {
        let store = InMemoryEmbeddingStore::new();
        store.ensure("c", 3).await.unwrap();

        let p = EmbeddingPayload::entity("flippa", "Listing");
        let id = point_id("m", 3, &p, "listing text");
        assert_eq!(
            store.missing("c", &[id]).await.unwrap(),
            vec![id],
            "absent before upsert"
        );

        store
            .upsert(
                "c",
                vec![StoredEmbedding {
                    id,
                    vector: vec3(1.0, 0.0, 0.0),
                    payload: p.clone(),
                }],
            )
            .await
            .unwrap();
        assert!(
            store.missing("c", &[id]).await.unwrap().is_empty(),
            "present after upsert (server-side cache)"
        );

        let hits = store
            .search(
                "c",
                &vec3(1.0, 0.0, 0.0),
                10,
                None,
                &EmbeddingFilter::default(),
            )
            .await
            .unwrap();
        assert_eq!(hits.len(), 1);
        assert!((hits[0].score - 1.0).abs() < 1e-5);
        assert_eq!(hits[0].payload, p);
    }

    #[tokio::test]
    async fn search_respects_filter_and_threshold() {
        let store = InMemoryEmbeddingStore::new();
        let listing = EmbeddingPayload::entity("flippa", "Listing");
        let patient = EmbeddingPayload::entity("clinic", "Patient");
        store
            .upsert(
                "c",
                vec![
                    StoredEmbedding {
                        id: point_id("m", 3, &listing, "a"),
                        vector: vec3(1.0, 0.0, 0.0),
                        payload: listing.clone(),
                    },
                    StoredEmbedding {
                        id: point_id("m", 3, &patient, "b"),
                        vector: vec3(0.0, 1.0, 0.0),
                        payload: patient.clone(),
                    },
                ],
            )
            .await
            .unwrap();

        // Domain filter keeps only flippa.
        let filter = EmbeddingFilter {
            kinds: vec![EmbeddingKind::Entity],
            domains: vec!["flippa".into()],
        };
        let hits = store
            .search("c", &vec3(1.0, 0.0, 0.0), 10, None, &filter)
            .await
            .unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].payload.domain, "flippa");

        // Threshold drops the orthogonal vector.
        let hits = store
            .search(
                "c",
                &vec3(1.0, 0.0, 0.0),
                10,
                Some(0.5),
                &EmbeddingFilter::default(),
            )
            .await
            .unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].payload.domain, "flippa");
    }

    #[test]
    fn point_id_is_content_addressed() {
        let p = EmbeddingPayload::property("flippa", "Listing", "sale_method");
        let a = point_id("m", 3, &p, "type: enum");
        let b = point_id("m", 3, &p, "type: keyword");
        let same = point_id("m", 3, &p, "type: enum");
        assert_ne!(a, b, "different text -> different id");
        assert_eq!(a, same, "same inputs -> same id");
        // Model change invalidates too.
        assert_ne!(a, point_id("m2", 3, &p, "type: enum"));
    }
}
