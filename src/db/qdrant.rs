//! Direct Qdrant REST client (feature `qdrant`).
//!
//! Stores ontology/entity embedding points and runs similarity search
//! **server-side**, implementing [`EmbeddingStore`]. A thin `reqwest`
//! wrapper mirroring [`crate::llm::openai::OpenAiClient`] — no gRPC stack.
//!
//! Unlike the `libqlink` path (which lets Memgraph create collections
//! implicitly), a direct client owns collection lifecycle: [`Self::ensure`]
//! creates the collection with the right vector size and cosine distance.

use std::collections::HashSet;
use std::time::Duration;

use async_trait::async_trait;
use serde_json::{json, Value as JsonValue};
use uuid::Uuid;

use crate::config::QdrantConfig;
use crate::embeddings::{
    EmbeddingFilter, EmbeddingPayload, EmbeddingStore, RawScoredHit, ScoredHit, StoreError,
    StoredEmbedding,
};

/// REST client for a Qdrant instance.
#[derive(Debug, Clone)]
pub struct QdrantClient {
    http: reqwest::Client,
    /// Base URL without a trailing slash, e.g. `http://127.0.0.1:6333`.
    base_url: String,
    /// Optional API key sent as the `api-key` header. Absent is fine for a
    /// local unauthenticated instance.
    api_key: Option<String>,
}

impl QdrantClient {
    /// Build a client from [`QdrantConfig`]. The API key is read from the
    /// environment variable named by `cfg.api_key_env` (absent tolerated).
    /// No network I/O happens here — `ensure`/`search`/… do that.
    pub fn connect(cfg: &QdrantConfig) -> Result<Self, StoreError> {
        let api_key = std::env::var(&cfg.api_key_env)
            .ok()
            .filter(|k| !k.trim().is_empty());
        let http = reqwest::Client::builder()
            .timeout(Duration::from_secs(cfg.timeout_secs.max(1)))
            .build()
            .map_err(|e| StoreError::Transport(e.to_string()))?;
        Ok(Self {
            http,
            base_url: cfg.url.trim_end_matches('/').to_string(),
            api_key,
        })
    }

    fn auth(&self, req: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        match &self.api_key {
            Some(k) => req.header("api-key", k),
            None => req,
        }
    }

    /// Send a request, returning the parsed JSON body on 2xx or a
    /// [`StoreError`] otherwise.
    async fn send(&self, req: reqwest::RequestBuilder) -> Result<JsonValue, StoreError> {
        let resp = self
            .auth(req)
            .send()
            .await
            .map_err(|e| StoreError::Transport(e.to_string()))?;
        let status = resp.status();
        let text = resp
            .text()
            .await
            .map_err(|e| StoreError::Transport(e.to_string()))?;
        if !status.is_success() {
            return Err(StoreError::Backend(format!(
                "qdrant {}: {}",
                status.as_u16(),
                text
            )));
        }
        if text.is_empty() {
            return Ok(JsonValue::Null);
        }
        serde_json::from_str(&text).map_err(|e| StoreError::Backend(e.to_string()))
    }
}

#[async_trait]
impl EmbeddingStore for QdrantClient {
    async fn ensure(&self, collection: &str, dim: usize) -> Result<(), StoreError> {
        let url = format!("{}/collections/{}", self.base_url, collection);
        // Existence check: a 404 is expected and means "create it".
        let resp = self
            .auth(self.http.get(&url))
            .send()
            .await
            .map_err(|e| StoreError::Transport(e.to_string()))?;
        if resp.status().is_success() {
            return Ok(());
        }
        if resp.status().as_u16() != 404 {
            let code = resp.status().as_u16();
            let text = resp.text().await.unwrap_or_default();
            return Err(StoreError::Backend(format!("qdrant {code}: {text}")));
        }
        // Create with cosine distance sized to the embedder.
        let body = json!({ "vectors": { "size": dim, "distance": "Cosine" } });
        match self.send(self.http.put(&url).json(&body)).await {
            Ok(_) => Ok(()),
            // A concurrent creator may have won the race (409 conflict);
            // re-check and treat an existing collection as success.
            Err(_) if self.collection_exists(collection).await? => Ok(()),
            Err(e) => Err(e),
        }
    }

    async fn missing(&self, collection: &str, ids: &[Uuid]) -> Result<Vec<Uuid>, StoreError> {
        if ids.is_empty() {
            return Ok(Vec::new());
        }
        let url = format!("{}/collections/{}/points", self.base_url, collection);
        let id_strs: Vec<String> = ids.iter().map(Uuid::to_string).collect();
        let body = json!({ "ids": id_strs, "with_payload": false, "with_vector": false });
        let resp = self.send(self.http.post(&url).json(&body)).await?;
        let present: HashSet<String> = resp
            .get("result")
            .and_then(JsonValue::as_array)
            .map(|arr| {
                arr.iter()
                    .filter_map(|p| point_id_string(&p["id"]))
                    .collect()
            })
            .unwrap_or_default();
        Ok(ids
            .iter()
            .filter(|id| !present.contains(&id.to_string()))
            .copied()
            .collect())
    }

    async fn upsert(
        &self,
        collection: &str,
        points: Vec<StoredEmbedding>,
    ) -> Result<(), StoreError> {
        if points.is_empty() {
            return Ok(());
        }
        let url = format!(
            "{}/collections/{}/points?wait=true",
            self.base_url, collection
        );
        let pts: Vec<JsonValue> = points
            .iter()
            .map(|p| {
                json!({
                    "id": p.id.to_string(),
                    "vector": p.vector,
                    "payload": payload_to_json(&p.payload),
                })
            })
            .collect();
        self.send(self.http.put(&url).json(&json!({ "points": pts })))
            .await?;
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
        let url = format!("{}/collections/{}/points/search", self.base_url, collection);
        let mut body = json!({
            "vector": vector,
            "limit": limit,
            "with_payload": true,
        });
        if let Some(t) = score_threshold {
            body["score_threshold"] = json!(t);
        }
        if let Some(f) = filter_to_json(filter) {
            body["filter"] = f;
        }
        let resp = self.send(self.http.post(&url).json(&body)).await?;
        let mut hits = Vec::new();
        if let Some(arr) = resp.get("result").and_then(JsonValue::as_array) {
            for item in arr {
                let score = item["score"].as_f64().unwrap_or(0.0) as f32;
                let payload = payload_from_json(&item["payload"])?;
                hits.push(ScoredHit { score, payload });
            }
        }
        Ok(hits)
    }

    async fn search_raw(
        &self,
        collection: &str,
        vector: &[f32],
        limit: usize,
        score_threshold: Option<f32>,
    ) -> Result<Vec<RawScoredHit>, StoreError> {
        let url = format!("{}/collections/{}/points/search", self.base_url, collection);
        // Ask for the payload but not the vector — the grounding pass only
        // needs the point id (a Memgraph node id for the `_canonical`
        // collection) plus whatever label/text the payload carries.
        let mut body = json!({
            "vector": vector,
            "limit": limit,
            "with_payload": true,
        });
        if let Some(t) = score_threshold {
            body["score_threshold"] = json!(t);
        }
        let resp = self.send(self.http.post(&url).json(&body)).await?;
        let mut hits = Vec::new();
        if let Some(arr) = resp.get("result").and_then(JsonValue::as_array) {
            for item in arr {
                let Some(id) = point_id_string(&item["id"]) else {
                    continue;
                };
                let score = item["score"].as_f64().unwrap_or(0.0) as f32;
                let payload = item.get("payload").cloned().unwrap_or(JsonValue::Null);
                hits.push(RawScoredHit { id, score, payload });
            }
        }
        Ok(hits)
    }
}

impl QdrantClient {
    async fn collection_exists(&self, collection: &str) -> Result<bool, StoreError> {
        let url = format!("{}/collections/{}", self.base_url, collection);
        let resp = self
            .auth(self.http.get(&url))
            .send()
            .await
            .map_err(|e| StoreError::Transport(e.to_string()))?;
        Ok(resp.status().is_success())
    }
}

/// Qdrant point ids come back as either a JSON string (UUID) or number.
fn point_id_string(v: &JsonValue) -> Option<String> {
    v.as_str()
        .map(str::to_string)
        .or_else(|| v.as_u64().map(|n| n.to_string()))
}

fn payload_to_json(payload: &EmbeddingPayload) -> JsonValue {
    serde_json::to_value(payload).unwrap_or(JsonValue::Null)
}

fn payload_from_json(v: &JsonValue) -> Result<EmbeddingPayload, StoreError> {
    serde_json::from_value(v.clone())
        .map_err(|e| StoreError::Backend(format!("bad point payload: {e}")))
}

/// Translate an [`EmbeddingFilter`] into a Qdrant `filter` object. Returns
/// `None` when there is nothing to constrain.
fn filter_to_json(filter: &EmbeddingFilter) -> Option<JsonValue> {
    let mut must = Vec::new();
    if !filter.kinds.is_empty() {
        let anys: Vec<&str> = filter.kinds.iter().map(|k| k.as_str()).collect();
        must.push(json!({ "key": "kind", "match": { "any": anys } }));
    }
    if !filter.domains.is_empty() {
        must.push(json!({ "key": "domain", "match": { "any": filter.domains } }));
    }
    if must.is_empty() {
        None
    } else {
        Some(json!({ "must": must }))
    }
}

#[cfg(test)]
impl QdrantClient {
    /// Delete a collection (test cleanup only).
    async fn delete_collection(&self, collection: &str) -> Result<(), StoreError> {
        let url = format!("{}/collections/{}", self.base_url, collection);
        self.send(self.http.delete(&url)).await?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embeddings::{EmbeddingKind, StoredEmbedding};

    #[test]
    fn payload_round_trips_through_json() {
        let p = EmbeddingPayload::property("flippa", "Listing", "sale_method");
        let j = payload_to_json(&p);
        assert_eq!(j["kind"], "property");
        assert_eq!(j["domain"], "flippa");
        assert_eq!(j["entity"], "Listing");
        assert_eq!(j["property"], "sale_method");
        assert_eq!(payload_from_json(&j).unwrap(), p);

        // Domain payload omits entity/property.
        let d = EmbeddingPayload::domain("flippa");
        let dj = payload_to_json(&d);
        assert_eq!(dj["kind"], "domain");
        assert!(dj.get("entity").is_none());
        assert_eq!(payload_from_json(&dj).unwrap(), d);
    }

    #[test]
    fn filter_builds_kind_and_domain_musts() {
        let f = EmbeddingFilter {
            kinds: vec![EmbeddingKind::Entity, EmbeddingKind::Property],
            domains: vec!["flippa".into(), "clinic".into()],
        };
        let j = filter_to_json(&f).unwrap();
        let must = j["must"].as_array().unwrap();
        assert_eq!(must.len(), 2);
        assert_eq!(must[0]["key"], "kind");
        assert_eq!(must[0]["match"]["any"][0], "entity");
        assert_eq!(must[1]["key"], "domain");
        assert_eq!(must[1]["match"]["any"][1], "clinic");
    }

    #[test]
    fn empty_filter_is_none() {
        assert!(filter_to_json(&EmbeddingFilter::default()).is_none());
    }

    #[test]
    fn point_id_string_accepts_uuid_and_number() {
        assert_eq!(point_id_string(&json!("6a...")).as_deref(), Some("6a..."));
        assert_eq!(point_id_string(&json!(42)).as_deref(), Some("42"));
    }

    /// Live round-trip against a real Qdrant. Skipped unless `QDRANT_URL`
    /// is set; run with `cargo test --features qdrant -- --ignored`.
    #[tokio::test]
    #[ignore = "requires a running Qdrant (set QDRANT_URL)"]
    async fn live_ensure_upsert_search_round_trip() {
        let Ok(url) = std::env::var("QDRANT_URL") else {
            return;
        };
        let cfg = QdrantConfig {
            url,
            ..QdrantConfig::default()
        };
        let client = QdrantClient::connect(&cfg).unwrap();
        let collection = format!("linguagraph_test_{}", uuid::Uuid::new_v4().simple());

        client.ensure(&collection, 3).await.unwrap();

        let payload = EmbeddingPayload::entity("flippa", "Listing");
        let id = crate::embeddings::point_id("m", 3, &payload, "listing");
        assert_eq!(client.missing(&collection, &[id]).await.unwrap(), vec![id]);

        client
            .upsert(
                &collection,
                vec![StoredEmbedding {
                    id,
                    vector: vec![1.0, 0.0, 0.0],
                    payload: payload.clone(),
                }],
            )
            .await
            .unwrap();
        assert!(client.missing(&collection, &[id]).await.unwrap().is_empty());

        let filter = EmbeddingFilter {
            kinds: vec![EmbeddingKind::Entity],
            domains: vec!["flippa".into()],
        };
        let hits = client
            .search(&collection, &[1.0, 0.0, 0.0], 5, None, &filter)
            .await
            .unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].payload, payload);

        client.delete_collection(&collection).await.unwrap();
    }
}
