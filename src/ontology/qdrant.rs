//! Qdrant-backed [`OntologyStore`] — the single-store backend.
//!
//! The authoritative schema lives **in Qdrant**, in the same
//! per-namespace collection (`{prefix}{token}`) as the routing embeddings
//! it drives. Each domain's schema is one *schema point*:
//!
//! * id — a deterministic UUIDv5 of `(namespace, domain)`, so a re-`put`
//!   overwrites in place;
//! * payload — `{ kind: "schema", domain, version, ontology: <full
//!   DomainOntology JSON> }`, the source of truth read back by `get` /
//!   `list` / `load_catalog`;
//! * vector — a fixed placeholder (a unit vector), never a meaningful
//!   search target. Every routing search filters by `kind ∈ {domain,
//!   entity, property}`, so schema points are structurally excluded from
//!   embedding results; the vector only exists because the collection has
//!   one dense-vector config and every point must carry one.
//!
//! Because schema and embeddings share a collection and both carry
//! `domain` in their payload, deleting/renaming a domain
//! garbage-collects *all* of its points in one filter delete — schema and
//! routing embeddings together — which is what closes the stale-point
//! leak the content-addressed embedding scheme left behind.
//!
//! This backend stays schema-authoritative only: it does **not** compute
//! embeddings (it has no embedder). Writing the routing embeddings that
//! accompany a schema `put`, and the finer content-addressed GC, is the
//! ontology *service*'s job (it composes this store with an embedder).

use async_trait::async_trait;
use serde_json::{json, Value as JsonValue};
use uuid::Uuid;

use super::namespace::Namespace;
use super::store::{now_ms, DomainSummary, OntologyStore, RenameOutcome, Version};
use crate::db::QdrantClient;
use crate::embeddings::{EmbeddingStore, StoreError};
use crate::graph::{DomainOntology, OntologyCatalog, OntologyError};

/// Payload discriminator for authoritative schema points, kept distinct
/// from the routing-embedding `kind`s (`domain`/`entity`/`property`) so
/// they never collide in a filter.
const SCHEMA_KIND: &str = "schema";

/// Qdrant-backed ontology schema store. Cheap to clone.
#[derive(Debug, Clone)]
pub struct QdrantOntologyStore {
    client: QdrantClient,
    /// Dense-vector dimension of the collection — the routing embedding
    /// dim. Schema points carry a placeholder vector of this size.
    dim: usize,
}

impl QdrantOntologyStore {
    /// `dim` must match the routing embedder's dimension so schema points
    /// and embeddings can share one collection.
    pub fn new(client: QdrantClient, dim: usize) -> Self {
        Self { client, dim }
    }

    /// The Qdrant collection backing `ns` — the single name authority is
    /// [`Namespace::ontology_collection`], so this store and the embedding
    /// index address the same collection by construction.
    pub fn collection(&self, ns: &Namespace) -> String {
        ns.ontology_collection()
    }

    async fn ensure(&self, collection: &str) -> Result<(), OntologyError> {
        self.client.ensure(collection, self.dim).await.map_err(be)
    }
}

fn be(e: StoreError) -> OntologyError {
    OntologyError::Backend(e.to_string())
}

/// Deterministic schema-point id for `(namespace, domain)`.
fn schema_point_id(ns_token: &str, domain: &str) -> String {
    Uuid::new_v5(
        &Uuid::NAMESPACE_OID,
        format!("lg-onto-schema|{ns_token}|{domain}").as_bytes(),
    )
    .to_string()
}

/// Placeholder unit vector: normalizable under Cosine (a zero vector is
/// not) and irrelevant to search because schema points are filtered out
/// by `kind`.
fn placeholder_vector(dim: usize) -> Vec<f32> {
    let mut v = vec![0.0f32; dim.max(1)];
    v[0] = 1.0;
    v
}

fn schema_payload(domain: &str, ontology: &DomainOntology, version: Version) -> JsonValue {
    json!({
        "kind": SCHEMA_KIND,
        "domain": domain,
        "version": version,
        "ontology": serde_json::to_value(ontology).unwrap_or(JsonValue::Null),
    })
}

/// Parse a schema point's payload back into `(domain, ontology, version)`,
/// hydrating the ontology's domain name. `None` for a malformed payload.
fn parse_schema(payload: &JsonValue) -> Option<(String, DomainOntology, Version)> {
    let domain = payload.get("domain")?.as_str()?.to_string();
    let version = payload
        .get("version")
        .and_then(JsonValue::as_u64)
        .unwrap_or(0);
    let mut ontology: DomainOntology =
        serde_json::from_value(payload.get("ontology")?.clone()).ok()?;
    ontology.name = Some(domain.clone());
    Some((domain, ontology, version))
}

fn schema_kind_filter() -> JsonValue {
    json!({ "must": [ { "key": "kind", "match": { "value": SCHEMA_KIND } } ] })
}

/// Matches every point (schema **and** routing embeddings) for one domain.
fn domain_filter(domain: &str) -> JsonValue {
    json!({ "must": [ { "key": "domain", "match": { "value": domain } } ] })
}

#[async_trait]
impl OntologyStore for QdrantOntologyStore {
    async fn list(&self, ns: &Namespace) -> Result<Vec<DomainSummary>, OntologyError> {
        let collection = self.collection(ns);
        let points = self
            .client
            .scroll_payloads(&collection, Some(schema_kind_filter()))
            .await
            .map_err(be)?;
        let mut out: Vec<DomainSummary> = points
            .iter()
            .filter_map(|(_, payload)| parse_schema(payload))
            .map(|(domain, ontology, version)| DomainSummary::of(&domain, &ontology, version))
            .collect();
        out.sort_by(|a, b| a.domain.cmp(&b.domain));
        Ok(out)
    }

    async fn get(
        &self,
        ns: &Namespace,
        domain: &str,
    ) -> Result<Option<DomainOntology>, OntologyError> {
        let collection = self.collection(ns);
        let id = schema_point_id(ns.token(), domain);
        let Some(payload) = self.client.get_payload(&collection, &id).await.map_err(be)? else {
            return Ok(None);
        };
        Ok(parse_schema(&payload).map(|(_, ontology, _)| ontology))
    }

    async fn put(
        &self,
        ns: &Namespace,
        domain: &str,
        mut ontology: DomainOntology,
    ) -> Result<DomainSummary, OntologyError> {
        let collection = self.collection(ns);
        self.ensure(&collection).await?;
        // `name` is derived from the key, not persisted in the payload
        // (it's `skip_serializing` anyway) — clear it before serializing so
        // the stored blob is canonical.
        ontology.name = None;
        let version = now_ms();
        let id = schema_point_id(ns.token(), domain);
        let payload = schema_payload(domain, &ontology, version);
        self.client
            .upsert_raw(&collection, &id, &placeholder_vector(self.dim), payload)
            .await
            .map_err(be)?;
        ontology.name = Some(domain.to_string());
        Ok(DomainSummary::of(domain, &ontology, version))
    }

    async fn delete(&self, ns: &Namespace, domain: &str) -> Result<bool, OntologyError> {
        let collection = self.collection(ns);
        let id = schema_point_id(ns.token(), domain);
        let existed = self
            .client
            .get_payload(&collection, &id)
            .await
            .map_err(be)?
            .is_some();
        if !existed {
            return Ok(false);
        }
        // GC: one filter delete removes the schema point *and* every
        // routing embedding for this domain (all carry `domain`).
        self.client
            .delete_by_filter(&collection, domain_filter(domain))
            .await
            .map_err(be)?;
        Ok(true)
    }

    async fn rename(
        &self,
        ns: &Namespace,
        from: &str,
        to: &str,
    ) -> Result<RenameOutcome, OntologyError> {
        let collection = self.collection(ns);
        let from_id = schema_point_id(ns.token(), from);
        let Some(payload) = self
            .client
            .get_payload(&collection, &from_id)
            .await
            .map_err(be)?
        else {
            return Ok(RenameOutcome::NotFound);
        };
        let to_id = schema_point_id(ns.token(), to);
        if self
            .client
            .get_payload(&collection, &to_id)
            .await
            .map_err(be)?
            .is_some()
        {
            return Ok(RenameOutcome::Conflict);
        }
        let Some((_, mut ontology, _)) = parse_schema(&payload) else {
            return Ok(RenameOutcome::NotFound);
        };

        self.ensure(&collection).await?;
        ontology.name = None;
        let version = now_ms();
        let new_payload = schema_payload(to, &ontology, version);
        // Write the new-name point first, then GC everything under the old
        // name. Ordering matters: the new point carries `domain == to`, so
        // the `domain == from` delete never touches it.
        self.client
            .upsert_raw(&collection, &to_id, &placeholder_vector(self.dim), new_payload)
            .await
            .map_err(be)?;
        self.client
            .delete_by_filter(&collection, domain_filter(from))
            .await
            .map_err(be)?;
        ontology.name = Some(to.to_string());
        Ok(RenameOutcome::Renamed(DomainSummary::of(
            to, &ontology, version,
        )))
    }

    async fn load_catalog(&self, ns: &Namespace) -> Result<OntologyCatalog, OntologyError> {
        let collection = self.collection(ns);
        let points = self
            .client
            .scroll_payloads(&collection, Some(schema_kind_filter()))
            .await
            .map_err(be)?;
        let mut catalog = OntologyCatalog::default();
        for (_, payload) in &points {
            if let Some((domain, ontology, _)) = parse_schema(payload) {
                catalog.insert(domain, ontology);
            }
        }
        Ok(catalog)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::QdrantConfig;
    use crate::graph::EntityTypeSpec;

    #[test]
    fn schema_point_id_is_deterministic_and_scoped() {
        let a = schema_point_id("ws_1", "coffeeline");
        assert_eq!(a, schema_point_id("ws_1", "coffeeline"));
        assert_ne!(a, schema_point_id("ws_2", "coffeeline"));
        assert_ne!(a, schema_point_id("ws_1", "retail"));
    }

    #[test]
    fn placeholder_vector_is_unit_and_sized() {
        let v = placeholder_vector(4);
        assert_eq!(v, vec![1.0, 0.0, 0.0, 0.0]);
        assert_eq!(placeholder_vector(0).len(), 1);
    }

    #[test]
    fn schema_payload_round_trips() {
        let onto = DomainOntology {
            name: Some("ignored".into()),
            description: Some("d".into()),
            entity_types: vec![EntityTypeSpec::new("Client")],
            relation_types: vec![],
            example: Some("ex".into()),
        };
        let payload = schema_payload("coffeeline", &onto, 42);
        assert_eq!(payload["kind"], "schema");
        assert_eq!(payload["domain"], "coffeeline");
        assert_eq!(payload["version"], 42);
        // `name` is skip_serializing, so the stored ontology omits it.
        assert!(payload["ontology"].get("name").is_none());

        let (domain, back, version) = parse_schema(&payload).unwrap();
        assert_eq!(domain, "coffeeline");
        assert_eq!(version, 42);
        assert_eq!(back.name.as_deref(), Some("coffeeline")); // hydrated
        assert_eq!(back.example.as_deref(), Some("ex"));
        assert_eq!(back.entity_types.len(), 1);
    }

    fn sample() -> DomainOntology {
        DomainOntology {
            name: None,
            description: Some("demo".into()),
            entity_types: vec![EntityTypeSpec::new("Client"), EntityTypeSpec::new("Order")],
            relation_types: vec![],
            example: None,
        }
    }

    /// Full CRUD round-trip against a live Qdrant. Skipped unless
    /// `QDRANT_URL` is set (`cargo test --features qdrant -- --ignored`).
    #[tokio::test]
    #[ignore = "requires a running Qdrant (set QDRANT_URL)"]
    async fn live_crud_round_trip() {
        let Ok(url) = std::env::var("QDRANT_URL") else {
            return;
        };
        let cfg = QdrantConfig {
            url,
            ..QdrantConfig::default()
        };
        let client = QdrantClient::connect(&cfg).unwrap();
        let store = QdrantOntologyStore::new(client.clone(), 4);
        // Isolate via a unique namespace token → unique collection.
        let ns = Namespace::new(format!("lg_onto_test_{}", Uuid::new_v4().simple()));
        let collection = store.collection(&ns);

        // empty namespace
        assert!(store.list(&ns).await.unwrap().is_empty());
        assert!(store.get(&ns, "demo").await.unwrap().is_none());
        assert!(!store.delete(&ns, "demo").await.unwrap());

        // put + get
        let s = store.put(&ns, "demo", sample()).await.unwrap();
        assert_eq!(s.domain, "demo");
        assert_eq!(s.entity_count, 2);
        let got = store.get(&ns, "demo").await.unwrap().unwrap();
        assert_eq!(got.name.as_deref(), Some("demo"));
        assert_eq!(got.entity_types.len(), 2);

        // put a second domain, list + load_catalog
        store.put(&ns, "retail", sample()).await.unwrap();
        let list = store.list(&ns).await.unwrap();
        assert_eq!(list.iter().map(|d| d.domain.as_str()).collect::<Vec<_>>(), vec!["demo", "retail"]);
        let cat = store.load_catalog(&ns).await.unwrap();
        assert!(cat.get("demo").is_some() && cat.get("retail").is_some());
        assert_eq!(cat.get("demo").unwrap().name.as_deref(), Some("demo"));

        // rename: conflict, not-found, ok
        assert_eq!(
            store.rename(&ns, "demo", "retail").await.unwrap(),
            RenameOutcome::Conflict
        );
        assert_eq!(
            store.rename(&ns, "missing", "x").await.unwrap(),
            RenameOutcome::NotFound
        );
        match store.rename(&ns, "demo", "renamed").await.unwrap() {
            RenameOutcome::Renamed(s) => assert_eq!(s.domain, "renamed"),
            other => panic!("expected Renamed, got {other:?}"),
        }
        assert!(store.get(&ns, "demo").await.unwrap().is_none());
        assert!(store.get(&ns, "renamed").await.unwrap().is_some());

        // GC check: a stray routing embedding for a domain is removed on
        // delete alongside the schema point.
        client
            .upsert_raw(
                &collection,
                &Uuid::new_v4().to_string(),
                &placeholder_vector(4),
                json!({ "kind": "entity", "domain": "renamed", "entity": "Client" }),
            )
            .await
            .unwrap();
        let all_before = client.scroll_payloads(&collection, None).await.unwrap();
        assert!(all_before.iter().any(|(_, p)| p["domain"] == "renamed" && p["kind"] == "entity"));

        assert!(store.delete(&ns, "renamed").await.unwrap());
        assert!(store.get(&ns, "renamed").await.unwrap().is_none());
        let all_after = client.scroll_payloads(&collection, None).await.unwrap();
        assert!(
            !all_after.iter().any(|(_, p)| p["domain"] == "renamed"),
            "delete must GC schema AND embedding points for the domain"
        );

        client.delete_collection(&collection).await.unwrap();
    }
}
