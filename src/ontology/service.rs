//! Ontology management service — the single entry point a host's HTTP
//! layer drives.
//!
//! [`OntologyService`] composes a schema [`OntologyStore`] with an
//! embedder and the routing [`EmbeddingStore`], and makes a schema edit
//! and its embeddings **one operation with one owner**: there is no
//! second store to synchronise and no "reindex status" to track across a
//! boundary. It also owns the edit *policy* the host used to carry —
//! create-vs-replace conflict, PATCH merge, optimistic concurrency, and
//! the additive-only validation used by the schema-suggest flow.
//!
//! ## Embedding lifecycle
//!
//! Schema writes ([`create`](OntologyService::create) /
//! [`replace`](OntologyService::replace) / [`patch`](OntologyService::patch))
//! are fast and return immediately — they touch only the schema store. The
//! caller then spawns [`refresh_embeddings`](OntologyService::refresh_embeddings)
//! off the request path (mirroring the host's existing "save now, warm
//! embeddings in the background" posture), which:
//!
//! 1. embeds the domain's routing passages — content-addressed, so
//!    unchanged passages are reused and only new/edited ones cost the
//!    embedder ([`ensure_indexed`]);
//! 2. **garbage-collects** the domain's now-stale points — the ones whose
//!    passage no longer exists after the edit — which `ensure_indexed`
//!    alone never removes. This closes the second half of the stale-point
//!    leak (the delete/rename half is handled inline below).
//!
//! [`delete`](OntologyService::delete) and [`rename`](OntologyService::rename)
//! GC the removed domain's embeddings inline (fast, no embedder needed).

use std::collections::HashSet;
use std::sync::Arc;

use uuid::Uuid;

use super::namespace::Namespace;
use super::store::{DomainSummary, OntologyStore, RenameOutcome, Version};
use crate::embeddings::{
    ensure_indexed, point_id, EmbeddingFilter, EmbeddingIndex, EmbeddingKind, SharedEmbedder,
    SharedEmbeddingStore,
};
use crate::graph::{DomainOntology, EntityTypeSpec, OntologyCatalog, OntologyError, RelationTypeSpec};
use crate::prompt::domain_routing_passages;

/// Result of [`OntologyService::create`]. `Conflict` ⇒ 409 upstream.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CreateOutcome {
    Created(DomainSummary),
    Conflict,
}

/// Result of [`OntologyService::replace`]. `NotFound` ⇒ 404 upstream.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReplaceOutcome {
    Replaced(DomainSummary),
    NotFound,
}

/// Result of [`OntologyService::patch`]. Maps to 200 / 404 / 409 / 400.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PatchOutcome {
    Patched(DomainSummary),
    NotFound,
    /// `expected_version` did not match the stored version (409).
    StaleVersion,
    /// A merge strategy rejected the patch (400); carries the reason.
    Invalid(String),
}

/// How an incoming patch is validated against the stored ontology.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MergeStrategy {
    /// Existing entity types and their existing properties (name + type)
    /// MUST stay unchanged; only **new** properties may be added and
    /// `relation_types` must match. Used by the schema-suggest wizard to
    /// enrich an ontology without ever silently mutating existing fields.
    AdditivePropertiesOnly,
}

/// A partial update to one domain. Every `Some` field is applied; the rest
/// stay as stored. The doubly-wrapped fields are triple-state: `None` =
/// leave, `Some(None)` = clear, `Some(Some(v))` = set.
#[derive(Debug, Clone, Default)]
pub struct DomainPatch {
    pub description: Option<Option<String>>,
    pub example: Option<Option<String>>,
    pub entity_types: Option<Vec<EntityTypeSpec>>,
    pub relation_types: Option<Vec<RelationTypeSpec>>,
    /// When set, the merged candidate is validated against this strategy.
    pub merge_strategy: Option<MergeStrategy>,
    /// Optimistic-concurrency guard: the [`Version`] the client last read.
    pub expected_version: Option<Version>,
}

/// Ontology management facade. Cheap to clone (all fields are `Arc`s).
#[derive(Clone)]
pub struct OntologyService {
    store: Arc<dyn OntologyStore>,
    embedder: SharedEmbedder,
    index: SharedEmbeddingStore,
    /// Embedding-model id folded into routing point ids so a model swap
    /// never reuses an incompatible vector.
    model_id: String,
}

impl std::fmt::Debug for OntologyService {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OntologyService")
            .field("model_id", &self.model_id)
            .finish_non_exhaustive()
    }
}

impl OntologyService {
    pub fn new(
        store: Arc<dyn OntologyStore>,
        embedder: SharedEmbedder,
        index: SharedEmbeddingStore,
        model_id: impl Into<String>,
    ) -> Self {
        Self {
            store,
            embedder,
            index,
            model_id: model_id.into(),
        }
    }

    /// The underlying schema store, for callers that need it directly.
    pub fn store(&self) -> &Arc<dyn OntologyStore> {
        &self.store
    }

    /// A [`crate::graph::OntologyCatalogStorage`] view of this service's
    /// store scoped to `ns` — the adapter that feeds a
    /// [`crate::core::Pipeline`] its catalog from the (namespace-scoped)
    /// single store, replacing whatever whole-catalog backend the host
    /// used to attach. Read-only (`save` is unsupported): the catalog is
    /// mutated through the service's own CRUD, not the pipeline.
    pub fn catalog_storage(
        &self,
        ns: &Namespace,
    ) -> Arc<dyn crate::graph::OntologyCatalogStorage> {
        Arc::new(NamespacedCatalog {
            store: self.store.clone(),
            ns: ns.clone(),
        })
    }

    // ── Reads ──────────────────────────────────────────────────────────

    pub async fn list(&self, ns: &Namespace) -> Result<Vec<DomainSummary>, OntologyError> {
        self.store.list(ns).await
    }

    pub async fn get(
        &self,
        ns: &Namespace,
        domain: &str,
    ) -> Result<Option<DomainOntology>, OntologyError> {
        self.store.get(ns, domain).await
    }

    /// Fetch a domain together with its current [`Version`] — the read a
    /// caller needs to seed an optimistic-lock guard for a later `patch`.
    pub async fn get_with_version(
        &self,
        ns: &Namespace,
        domain: &str,
    ) -> Result<Option<(DomainOntology, Version)>, OntologyError> {
        self.store.get_with_version(ns, domain).await
    }

    // ── Schema writes (fast; caller spawns `refresh_embeddings`) ────────

    /// Create a domain. `Conflict` when it already exists.
    ///
    /// The existence check and the write are not one transaction (Qdrant
    /// has none); at human edit cadence the TOCTOU window is not a
    /// concern, and a concurrent create simply last-writer-wins.
    pub async fn create(
        &self,
        ns: &Namespace,
        domain: &str,
        ontology: DomainOntology,
    ) -> Result<CreateOutcome, OntologyError> {
        if self.store.get(ns, domain).await?.is_some() {
            return Ok(CreateOutcome::Conflict);
        }
        Ok(CreateOutcome::Created(
            self.store.put(ns, domain, ontology).await?,
        ))
    }

    /// Replace a domain in full. `NotFound` when absent.
    pub async fn replace(
        &self,
        ns: &Namespace,
        domain: &str,
        ontology: DomainOntology,
    ) -> Result<ReplaceOutcome, OntologyError> {
        if self.store.get(ns, domain).await?.is_none() {
            return Ok(ReplaceOutcome::NotFound);
        }
        Ok(ReplaceOutcome::Replaced(
            self.store.put(ns, domain, ontology).await?,
        ))
    }

    /// Apply a partial update, honouring the optimistic-lock guard and
    /// merge-strategy validation.
    pub async fn patch(
        &self,
        ns: &Namespace,
        domain: &str,
        patch: DomainPatch,
    ) -> Result<PatchOutcome, OntologyError> {
        let Some((current, version)) = self.store.get_with_version(ns, domain).await? else {
            return Ok(PatchOutcome::NotFound);
        };
        if let Some(expected) = patch.expected_version {
            if expected != version {
                return Ok(PatchOutcome::StaleVersion);
            }
        }

        let mut candidate = current.clone();
        if let Some(description) = patch.description {
            candidate.description = description;
        }
        if let Some(example) = patch.example {
            candidate.example = example;
        }
        if let Some(entity_types) = patch.entity_types {
            candidate.entity_types = entity_types;
        }
        if let Some(relation_types) = patch.relation_types {
            candidate.relation_types = relation_types;
        }

        if let Some(MergeStrategy::AdditivePropertiesOnly) = patch.merge_strategy {
            if let Err(reason) = validate_additive_properties_only(&current, &candidate) {
                return Ok(PatchOutcome::Invalid(reason));
            }
        }

        Ok(PatchOutcome::Patched(
            self.store.put(ns, domain, candidate).await?,
        ))
    }

    /// Rename a domain's key. On success, GCs the old name's routing
    /// embeddings inline; the caller should spawn
    /// [`refresh_embeddings`](Self::refresh_embeddings) for the new name.
    pub async fn rename(
        &self,
        ns: &Namespace,
        from: &str,
        to: &str,
    ) -> Result<RenameOutcome, OntologyError> {
        let outcome = self.store.rename(ns, from, to).await?;
        if matches!(outcome, RenameOutcome::Renamed(_)) {
            self.gc_domain_embeddings(ns, from).await?;
        }
        Ok(outcome)
    }

    /// Delete a domain and GC all of its routing embeddings.
    pub async fn delete(&self, ns: &Namespace, domain: &str) -> Result<bool, OntologyError> {
        let deleted = self.store.delete(ns, domain).await?;
        if deleted {
            self.gc_domain_embeddings(ns, domain).await?;
        }
        Ok(deleted)
    }

    /// Seed linguagraph's built-in catalog into `ns`, skipping domains that
    /// already exist. Returns the number inserted. The caller should warm
    /// embeddings afterwards (per inserted domain, or the whole namespace).
    pub async fn import_builtin(&self, ns: &Namespace) -> Result<usize, OntologyError> {
        let builtin = OntologyCatalog::builtin();
        let mut imported = 0;
        for (domain, ontology) in builtin.domains_view() {
            if self.store.get(ns, domain).await?.is_none() {
                self.store.put(ns, domain, ontology.clone()).await?;
                imported += 1;
            }
        }
        Ok(imported)
    }

    // ── Embedding lifecycle ────────────────────────────────────────────

    /// Warm the domain's routing embeddings and garbage-collect its stale
    /// points. Idempotent and content-addressed: repeat calls for an
    /// unchanged domain are cheap no-ops. When the domain no longer exists
    /// this degrades to a pure GC of any leftovers.
    pub async fn refresh_embeddings(
        &self,
        ns: &Namespace,
        domain: &str,
    ) -> Result<(), OntologyError> {
        let Some(ontology) = self.store.get(ns, domain).await? else {
            return self.gc_domain_embeddings(ns, domain).await;
        };
        let collection = ns.ontology_collection();
        self.index
            .ensure(&collection, self.embedder.dim())
            .await
            .map_err(be)?;
        let index = EmbeddingIndex {
            store: self.index.as_ref(),
            collection: &collection,
            model: &self.model_id,
        };
        let passages = domain_routing_passages(domain, &ontology);
        let fresh: HashSet<Uuid> = passages
            .iter()
            .map(|(payload, text)| point_id(&self.model_id, self.embedder.dim(), payload, text))
            .collect();
        let existing = self
            .index
            .list_ids(&collection, &routing_filter(domain))
            .await
            .map_err(be)?;
        // Embed only what's missing (content-addressed reuse) …
        ensure_indexed(&index, self.embedder.as_ref(), &passages)
            .await
            .map_err(be)?;
        // … then delete whatever no longer belongs.
        let stale: Vec<Uuid> = existing.into_iter().filter(|id| !fresh.contains(id)).collect();
        self.index.delete(&collection, &stale).await.map_err(be)?;
        Ok(())
    }

    async fn gc_domain_embeddings(
        &self,
        ns: &Namespace,
        domain: &str,
    ) -> Result<(), OntologyError> {
        let collection = ns.ontology_collection();
        let ids = self
            .index
            .list_ids(&collection, &routing_filter(domain))
            .await
            .map_err(be)?;
        self.index.delete(&collection, &ids).await.map_err(be)?;
        Ok(())
    }
}

/// A namespace-scoped, read-only [`OntologyCatalogStorage`] view over an
/// [`OntologyStore`]. Bridges the new per-domain store to the old
/// whole-catalog trait a [`crate::core::Pipeline`] consumes, so the
/// pipeline reads its catalog from the single store without any change to
/// its own contract. `save` is intentionally unsupported — catalog
/// mutation flows through [`OntologyService`], never the pipeline.
#[derive(Debug, Clone)]
pub struct NamespacedCatalog {
    store: Arc<dyn OntologyStore>,
    ns: Namespace,
}

#[async_trait::async_trait]
impl crate::graph::OntologyCatalogStorage for NamespacedCatalog {
    async fn load(&self) -> Result<OntologyCatalog, OntologyError> {
        self.store.load_catalog(&self.ns).await
    }
    // `save` falls back to the trait default (Unsupported).
}

fn routing_filter(domain: &str) -> EmbeddingFilter {
    EmbeddingFilter {
        kinds: vec![
            EmbeddingKind::Domain,
            EmbeddingKind::Entity,
            EmbeddingKind::Property,
        ],
        domains: vec![domain.to_string()],
    }
}

fn be<E: std::fmt::Display>(e: E) -> OntologyError {
    OntologyError::Backend(e.to_string())
}

/// Reject any change beyond appending **new** properties to existing
/// entity types: the entity-type set must match, every existing property
/// must keep its name and `property_type`, and `relation_types` must be
/// unchanged. New property names on existing types are allowed.
pub fn validate_additive_properties_only(
    existing: &DomainOntology,
    candidate: &DomainOntology,
) -> Result<(), String> {
    let existing_names: HashSet<&str> =
        existing.entity_types.iter().map(|t| t.name.as_str()).collect();
    let candidate_names: HashSet<&str> = candidate
        .entity_types
        .iter()
        .map(|t| t.name.as_str())
        .collect();
    if existing_names != candidate_names {
        let added: Vec<&str> = candidate_names.difference(&existing_names).copied().collect();
        let removed: Vec<&str> = existing_names.difference(&candidate_names).copied().collect();
        return Err(format!(
            "entity_type set must be unchanged (added={added:?}, removed={removed:?})"
        ));
    }

    for cand in &candidate.entity_types {
        let exi = existing
            .entity_types
            .iter()
            .find(|t| t.name == cand.name)
            .expect("name set equality checked above");
        for exi_prop in &exi.properties {
            match cand.properties.iter().find(|p| p.name == exi_prop.name) {
                None => {
                    return Err(format!(
                        "property `{}` on `{}` was removed",
                        exi_prop.name, cand.name
                    ));
                }
                Some(c) if c.property_type != exi_prop.property_type => {
                    return Err(format!(
                        "property `{}` on `{}` changed type ({:?} → {:?})",
                        exi_prop.name, cand.name, exi_prop.property_type, c.property_type
                    ));
                }
                Some(_) => {}
            }
        }
    }

    if existing.relation_types != candidate.relation_types {
        return Err("relation_types must be unchanged".to_string());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embeddings::{InMemoryEmbeddingStore, MockEmbedder};
    use crate::graph::{OntologyPropertyType, PropertySpec};
    use crate::ontology::InMemoryOntologyStore;

    const DIM: usize = 16;

    fn service() -> OntologyService {
        OntologyService::new(
            Arc::new(InMemoryOntologyStore::new()),
            Arc::new(MockEmbedder::new(DIM)),
            Arc::new(InMemoryEmbeddingStore::new()),
            "mock",
        )
    }

    fn entity(name: &str, props: &[(&str, OntologyPropertyType)]) -> EntityTypeSpec {
        EntityTypeSpec {
            name: name.into(),
            description: Some(format!("the {name}")),
            properties: props
                .iter()
                .map(|(n, ty)| PropertySpec {
                    name: (*n).into(),
                    description: None,
                    property_type: *ty,
                    required: false,
                    allowed_values: vec![],
                })
                .collect(),
        }
    }

    fn ontology(entities: Vec<EntityTypeSpec>) -> DomainOntology {
        DomainOntology {
            name: None,
            description: Some("demo".into()),
            entity_types: entities,
            relation_types: vec![],
            example: None,
        }
    }

    #[tokio::test]
    async fn create_conflicts_then_replace_and_delete() {
        let svc = service();
        let ns = Namespace::new("ws_1");
        let onto = ontology(vec![entity("Client", &[("name", OntologyPropertyType::Keyword)])]);

        assert!(matches!(
            svc.create(&ns, "demo", onto.clone()).await.unwrap(),
            CreateOutcome::Created(_)
        ));
        assert_eq!(
            svc.create(&ns, "demo", onto.clone()).await.unwrap(),
            CreateOutcome::Conflict
        );
        assert!(matches!(
            svc.replace(&ns, "demo", onto.clone()).await.unwrap(),
            ReplaceOutcome::Replaced(_)
        ));
        assert_eq!(
            svc.replace(&ns, "missing", onto).await.unwrap(),
            ReplaceOutcome::NotFound
        );
        assert!(svc.delete(&ns, "demo").await.unwrap());
        assert!(!svc.delete(&ns, "demo").await.unwrap());
    }

    #[tokio::test]
    async fn patch_optimistic_lock_and_additive_validation() {
        let svc = service();
        let ns = Namespace::new("ws_1");
        let base = ontology(vec![entity("Client", &[("name", OntologyPropertyType::Keyword)])]);
        let CreateOutcome::Created(created) = svc.create(&ns, "demo", base).await.unwrap() else {
            panic!("created");
        };

        // Stale version → 409.
        assert_eq!(
            svc.patch(
                &ns,
                "demo",
                DomainPatch {
                    description: Some(Some("new".into())),
                    expected_version: Some(created.version.wrapping_sub(1)),
                    ..Default::default()
                }
            )
            .await
            .unwrap(),
            PatchOutcome::StaleVersion
        );

        // Additive strategy rejects removing an entity type.
        let invalid = svc
            .patch(
                &ns,
                "demo",
                DomainPatch {
                    entity_types: Some(vec![]),
                    merge_strategy: Some(MergeStrategy::AdditivePropertiesOnly),
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        assert!(matches!(invalid, PatchOutcome::Invalid(_)), "got {invalid:?}");

        // Additive strategy allows appending a new property.
        let ok = svc
            .patch(
                &ns,
                "demo",
                DomainPatch {
                    entity_types: Some(vec![entity(
                        "Client",
                        &[
                            ("name", OntologyPropertyType::Keyword),
                            ("segment", OntologyPropertyType::Keyword),
                        ],
                    )]),
                    merge_strategy: Some(MergeStrategy::AdditivePropertiesOnly),
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        assert!(matches!(ok, PatchOutcome::Patched(_)), "got {ok:?}");

        // Missing domain → 404.
        assert_eq!(
            svc.patch(&ns, "missing", DomainPatch::default())
                .await
                .unwrap(),
            PatchOutcome::NotFound
        );
    }

    /// The core payoff: after an edit that *shrinks* a domain, refresh
    /// embeds the new passages AND garbage-collects the points whose
    /// passage no longer exists — no stale leftovers.
    #[tokio::test]
    async fn refresh_embeds_and_gcs_stale_points() {
        let svc = service();
        let ns = Namespace::new("ws_1");
        let collection = ns.ontology_collection();

        // Two entities.
        svc.create(
            &ns,
            "demo",
            ontology(vec![
                entity("Client", &[("name", OntologyPropertyType::Keyword)]),
                entity("Order", &[("total", OntologyPropertyType::Number)]),
            ]),
        )
        .await
        .unwrap();
        svc.refresh_embeddings(&ns, "demo").await.unwrap();

        let after_two = svc
            .index
            .list_ids(&collection, &routing_filter("demo"))
            .await
            .unwrap()
            .len();
        // 1 domain + 2 entities + 2 properties.
        assert_eq!(after_two, 5, "expected all passages embedded");

        // Shrink to one entity, then refresh.
        svc.replace(
            &ns,
            "demo",
            ontology(vec![entity("Client", &[("name", OntologyPropertyType::Keyword)])]),
        )
        .await
        .unwrap();
        svc.refresh_embeddings(&ns, "demo").await.unwrap();

        let after_one = svc
            .index
            .list_ids(&collection, &routing_filter("demo"))
            .await
            .unwrap()
            .len();
        // 1 domain + 1 entity + 1 property — the Order entity/property GC'd.
        assert_eq!(after_one, 3, "stale points must be garbage-collected");
    }

    #[tokio::test]
    async fn delete_gcs_domain_embeddings() {
        let svc = service();
        let ns = Namespace::new("ws_1");
        let collection = ns.ontology_collection();

        svc.create(
            &ns,
            "demo",
            ontology(vec![entity("Client", &[("name", OntologyPropertyType::Keyword)])]),
        )
        .await
        .unwrap();
        svc.refresh_embeddings(&ns, "demo").await.unwrap();
        assert!(
            !svc.index
                .list_ids(&collection, &routing_filter("demo"))
                .await
                .unwrap()
                .is_empty()
        );

        assert!(svc.delete(&ns, "demo").await.unwrap());
        assert!(
            svc.index
                .list_ids(&collection, &routing_filter("demo"))
                .await
                .unwrap()
                .is_empty(),
            "delete must GC the domain's routing embeddings"
        );
    }

    #[tokio::test]
    async fn import_builtin_seeds_and_skips_existing() {
        let svc = service();
        let ns = Namespace::new("ws_1");
        let first = svc.import_builtin(&ns).await.unwrap();
        assert!(first > 0, "builtin catalog should seed at least one domain");
        let second = svc.import_builtin(&ns).await.unwrap();
        assert_eq!(second, 0, "re-import must skip existing domains");
    }

    /// End-to-end single-store proof against a real Qdrant: schema store
    /// (`QdrantOntologyStore`) and embedding index (the same
    /// `QdrantClient`) address ONE collection. Confirms schema points and
    /// routing embeddings coexist there, refresh GCs stale points, and
    /// delete removes both. Skipped unless `QDRANT_URL` is set.
    #[cfg(feature = "qdrant")]
    #[tokio::test]
    #[ignore = "requires a running Qdrant (set QDRANT_URL)"]
    async fn live_single_store_schema_and_embeddings_coexist_and_gc() {
        use crate::config::QdrantConfig;
        use crate::db::QdrantClient;
        use crate::ontology::QdrantOntologyStore;

        let Ok(url) = std::env::var("QDRANT_URL") else {
            return;
        };
        let client = QdrantClient::connect(&QdrantConfig {
            url,
            ..Default::default()
        })
        .unwrap();
        let svc = OntologyService::new(
            Arc::new(QdrantOntologyStore::new(client.clone(), DIM)),
            Arc::new(MockEmbedder::new(DIM)),
            Arc::new(client.clone()),
            "mock",
        );
        let ns = Namespace::new(format!("lg_svc_test_{}", Uuid::new_v4().simple()));
        let collection = ns.ontology_collection();

        svc.create(
            &ns,
            "demo",
            ontology(vec![
                entity("Client", &[("name", OntologyPropertyType::Keyword)]),
                entity("Order", &[("total", OntologyPropertyType::Number)]),
            ]),
        )
        .await
        .unwrap();
        svc.refresh_embeddings(&ns, "demo").await.unwrap();

        // ONE collection holds both the authoritative schema point and the
        // routing embeddings.
        let all = client.scroll_payloads(&collection, None).await.unwrap();
        let kinds: HashSet<&str> = all.iter().filter_map(|(_, p)| p["kind"].as_str()).collect();
        assert!(kinds.contains("schema"), "schema point missing: {kinds:?}");
        assert!(kinds.contains("domain") && kinds.contains("entity") && kinds.contains("property"));
        // Schema reads back through the service.
        assert_eq!(
            svc.get(&ns, "demo").await.unwrap().unwrap().entity_types.len(),
            2
        );

        // Shrink + refresh → the Order entity/property routing points GC'd.
        svc.replace(
            &ns,
            "demo",
            ontology(vec![entity("Client", &[("name", OntologyPropertyType::Keyword)])]),
        )
        .await
        .unwrap();
        svc.refresh_embeddings(&ns, "demo").await.unwrap();
        let routing = svc
            .index
            .list_ids(&collection, &routing_filter("demo"))
            .await
            .unwrap();
        assert_eq!(routing.len(), 3, "stale routing points must be GC'd");

        // Delete removes schema AND embeddings for the domain.
        assert!(svc.delete(&ns, "demo").await.unwrap());
        let after = client.scroll_payloads(&collection, None).await.unwrap();
        assert!(
            after.iter().all(|(_, p)| p["domain"] != "demo"),
            "delete must purge every point for the domain"
        );

        client.delete_collection(&collection).await.unwrap();
    }
}
