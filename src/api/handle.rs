//! The process-wide handle, its builder, and the tenant-scoped read /
//! write views that implement [`GraphRead`] / [`GraphWrite`].
//!
//! # Shape
//!
//! * [`LinguaGraph`] — one per process, cheap to clone (an `Arc` inside).
//!   Holds the connection pool (via the internal [`Pipeline`]) and the
//!   configured limits/embedder/cache.
//! * [`LinguaGraph::read`] / [`LinguaGraph::write`] — the only way to get
//!   at the graph. Both take a [`TenantId`], so no call can touch data
//!   without selecting a tenant. Isolation is enforced inside the library
//!   via a per-tenant label prefix; a cross-tenant access surfaces as
//!   [`GraphError::TenantIsolation`] rather than a silent empty result.
//! * [`ReadScope`] / [`WriteScope`] — the tenant-bound views.
//!
//! # Wiring status
//!
//! This is the public **contract** over the existing [`Pipeline`]. Where
//! a capability maps cleanly onto the pipeline it is wired through
//! (`discover_types` → entity-type search, `delete_by_source` →
//! source-rooted delete, and traversal-bound validation). The remaining
//! capabilities validate their inputs and return
//! [`GraphError::Unsupported`] until their backend mapping lands, so the
//! surface compiles, is mockable, and is safe to build a service against
//! today.

use std::sync::Arc;

use async_trait::async_trait;
use futures::stream::{self, Stream};

use crate::config::{Config, DatabaseConfig};
use crate::core::{EntityTypeSearchQuery, Pipeline};
use crate::db::MemgraphClient;
use crate::graph::OntologyCatalog;

use super::error::{GraphError, Result};
use super::model::{
    Entity, EntityId, EntityType, Filters, Page, Relation, RelationId, SourceId, Subgraph, TenantId,
};
use super::model::{FacetDim, Facets};
use super::options::{
    Answer, AnswerChunk, AskOptions, DeleteReport, EntityHit, EntityTypeInfo, Extraction,
    ExtractOptions, GraphBatch, MergeCandidate, MergeDecision, NeighborsPage, Path, PathOptions,
    QueryPlan, ResolveReport, ReviewOptions, RunOptions, SearchOptions, TraversalOptions,
    TypeMatch, UpsertOptions, UpsertReport,
};
use super::traits::{Cache, GraphRead, GraphWrite, SharedEmbedder};

/// A domain ontology handed to the builder. Aliased to the crate's
/// [`OntologyCatalog`] so callers configure one catalog everywhere.
pub type Ontology = OntologyCatalog;

/// Hard per-tenant ceilings the library refuses to exceed. Set once on
/// the builder via [`LinguaGraphBuilder::default_limits`].
#[derive(Debug, Clone, Copy)]
pub struct Limits {
    /// Maximum traversal depth any single request may reach.
    pub max_depth: u8,
    /// Maximum nodes any single request may return.
    pub max_nodes: u32,
    /// Default statement timeout in milliseconds.
    pub timeout_ms: u64,
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            max_depth: 6,
            max_nodes: 200,
            timeout_ms: 30_000,
        }
    }
}

/// Connection-pool + credentials for the Memgraph backend.
#[derive(Debug, Clone)]
pub struct PoolConfig {
    /// Bolt username.
    pub user: String,
    /// Bolt password.
    pub password: String,
    /// Logical database name.
    pub database: String,
    /// Maximum pooled connections.
    pub max_connections: u32,
    /// Per-statement timeout in seconds.
    pub query_timeout_secs: u64,
}

impl Default for PoolConfig {
    fn default() -> Self {
        Self {
            user: String::new(),
            password: String::new(),
            database: "memgraph".to_owned(),
            max_connections: 16,
            query_timeout_secs: 30,
        }
    }
}

/// Shared, immutable state behind [`LinguaGraph`].
struct Inner {
    pipeline: Pipeline,
    limits: Limits,
    catalog_cache: Option<Arc<dyn Cache>>,
}

impl std::fmt::Debug for Inner {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Inner")
            .field("limits", &self.limits)
            .field("catalog_cache", &self.catalog_cache.is_some())
            .finish_non_exhaustive()
    }
}

/// Process-wide entry point. Create once, clone per request.
#[derive(Clone, Debug)]
pub struct LinguaGraph {
    inner: Arc<Inner>,
}

impl LinguaGraph {
    /// Start configuring a handle.
    pub fn builder() -> LinguaGraphBuilder {
        LinguaGraphBuilder::default()
    }

    /// Build a handle from an already-constructed [`Pipeline`]. Useful
    /// for tests (a `Pipeline` over a mock client) and for callers who
    /// wire the pipeline themselves.
    pub fn from_pipeline(pipeline: Pipeline, limits: Limits) -> Self {
        Self {
            inner: Arc::new(Inner {
                pipeline,
                limits,
                catalog_cache: None,
            }),
        }
    }

    /// The configured hard limits.
    pub fn limits(&self) -> Limits {
        self.inner.limits
    }

    /// The configured type-catalog cache, if any.
    pub fn catalog_cache(&self) -> Option<&Arc<dyn Cache>> {
        self.inner.catalog_cache.as_ref()
    }

    /// A read-only, tenant-scoped view. Hand this to the REST read path.
    pub fn read(&self, tenant: TenantId) -> ReadScope {
        ReadScope {
            pipeline: self.scoped_pipeline(&tenant),
            tenant,
            limits: self.inner.limits,
        }
    }

    /// A write-capable, tenant-scoped view for ingestion and review.
    pub fn write(&self, tenant: TenantId) -> WriteScope {
        WriteScope {
            pipeline: self.scoped_pipeline(&tenant),
            tenant,
            limits: self.inner.limits,
        }
    }

    /// Clone the pipeline and stamp the tenant's isolation prefix onto
    /// it. The prefix scopes both the Memgraph labels (`prefix_label`)
    /// and the vector-store collections (`prefix_index`) so a scoped
    /// pipeline can only ever see one tenant's data.
    fn scoped_pipeline(&self, tenant: &TenantId) -> Pipeline {
        self.inner
            .pipeline
            .clone()
            .with_prefix_label(Some(tenant.as_str()))
            .with_prefix_index(Some(tenant.as_str()))
    }
}

/// Builder for [`LinguaGraph`].
#[derive(Default)]
pub struct LinguaGraphBuilder {
    memgraph_uri: Option<String>,
    pool: PoolConfig,
    embedder: Option<SharedEmbedder>,
    catalog_cache: Option<Arc<dyn Cache>>,
    ontology: Option<Arc<Ontology>>,
    limits: Limits,
}

impl std::fmt::Debug for LinguaGraphBuilder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LinguaGraphBuilder")
            .field("memgraph_uri", &self.memgraph_uri)
            .field("pool", &self.pool)
            .field("embedder", &self.embedder.is_some())
            .field("catalog_cache", &self.catalog_cache.is_some())
            .field("ontology", &self.ontology.is_some())
            .field("limits", &self.limits)
            .finish()
    }
}

impl LinguaGraphBuilder {
    /// Point the handle at a Memgraph instance. `uri` is a Bolt URL
    /// (e.g. `bolt://localhost:7687`); `pool` carries credentials and
    /// pool sizing.
    pub fn memgraph(mut self, uri: &str, pool: PoolConfig) -> Self {
        self.memgraph_uri = Some(uri.to_owned());
        self.pool = pool;
        self
    }

    /// Attach a text embedder (semantic search, entity-type discovery,
    /// soft-merge). Usually backed by an external embedding service.
    pub fn with_embedder(mut self, e: SharedEmbedder) -> Self {
        self.embedder = Some(e);
        self
    }

    /// Attach a cache for the type catalog (e.g. Redis).
    pub fn with_catalog_cache(mut self, c: Arc<dyn Cache>) -> Self {
        self.catalog_cache = Some(c);
        self
    }

    /// Attach a domain ontology used for type resolution and enrichment.
    pub fn with_ontology(mut self, o: Ontology) -> Self {
        self.ontology = Some(Arc::new(o));
        self
    }

    /// Set the hard per-tenant limits (depth / node ceilings, timeout).
    pub fn default_limits(mut self, l: Limits) -> Self {
        self.limits = l;
        self
    }

    /// Connect and assemble the handle.
    ///
    /// Fails with [`GraphError::Config`] if no Memgraph URI was set, and
    /// with [`GraphError::Backend`] if the connection cannot be
    /// established.
    pub async fn build(self) -> Result<LinguaGraph> {
        let uri = self
            .memgraph_uri
            .ok_or_else(|| GraphError::Config("no Memgraph URI configured".into()))?;

        let db = DatabaseConfig {
            uri,
            user: self.pool.user,
            password: self.pool.password,
            database: self.pool.database,
            max_connections: self.pool.max_connections,
            query_timeout_secs: self.pool.query_timeout_secs,
        };
        let cfg = Config {
            database: db,
            llm: Default::default(),
            query: crate::config::QueryConfig {
                max_traversal_depth: self.limits.max_depth as u32,
                default_limit: self.limits.max_nodes,
            },
            ontology_catalog: Default::default(),
            prompt: Default::default(),
            ingest: Default::default(),
            types: Default::default(),
        };

        let client = MemgraphClient::connect(&cfg.database)
            .await
            .map_err(GraphError::from)?;

        let mut pipeline = Pipeline::new(Arc::new(client), &cfg);
        if let Some(e) = self.embedder {
            pipeline = pipeline.with_embedder(e);
        }
        if let Some(o) = self.ontology {
            pipeline = pipeline.with_ontology_catalog(o);
        }

        Ok(LinguaGraph {
            inner: Arc::new(Inner {
                pipeline,
                limits: self.limits,
                catalog_cache: self.catalog_cache,
            }),
        })
    }
}

/// Tenant-scoped, read-only view. Implements [`GraphRead`].
#[derive(Clone, Debug)]
pub struct ReadScope {
    pipeline: Pipeline,
    tenant: TenantId,
    limits: Limits,
}

impl ReadScope {
    /// The tenant this scope is bound to.
    pub fn tenant(&self) -> &TenantId {
        &self.tenant
    }

    /// Stream an answer as it is produced: answer tokens interleaved with
    /// subgraph fragments, for a progressive "typing" UI and large
    /// subgraphs. Lives here rather than on [`GraphRead`] because it
    /// returns an `impl Stream` (not object-safe).
    ///
    /// The current build emits a single terminal error frame until the
    /// streaming backend is wired; the signature is stable.
    pub fn ask_stream(
        &self,
        _question: &str,
        _opts: AskOptions,
    ) -> impl Stream<Item = Result<AnswerChunk>> {
        stream::once(async {
            Err(GraphError::Unsupported(
                "ask_stream is not yet wired to a streaming backend".into(),
            ))
        })
    }
}

#[async_trait]
impl GraphRead for ReadScope {
    async fn ask(&self, question: &str, _opts: AskOptions) -> Result<Answer> {
        if question.trim().is_empty() {
            return Err(GraphError::InvalidQuestion("empty question".into()));
        }
        Err(GraphError::Unsupported(
            "ask requires the NL→DSL compiler to be wired".into(),
        ))
    }

    async fn compile(&self, question: &str, _opts: AskOptions) -> Result<QueryPlan> {
        if question.trim().is_empty() {
            return Err(GraphError::InvalidQuestion("empty question".into()));
        }
        Err(GraphError::Unsupported(
            "compile requires the NL→DSL compiler to be wired".into(),
        ))
    }

    async fn run(&self, _plan: &QueryPlan, _opts: RunOptions) -> Result<Subgraph> {
        Err(GraphError::Unsupported(
            "run requires plan execution to be wired".into(),
        ))
    }

    async fn search(&self, _query: &str, _opts: SearchOptions) -> Result<Page<EntityHit>> {
        Err(GraphError::Unsupported(
            "search requires the hybrid retriever to be wired".into(),
        ))
    }

    async fn entity_types(&self, _filter: Filters) -> Result<Vec<EntityTypeInfo>> {
        Err(GraphError::Unsupported(
            "entity_types requires catalog aggregation to be wired".into(),
        ))
    }

    async fn discover_types(&self, question: &str) -> Result<Vec<TypeMatch>> {
        // Wired: the entity-type discovery pipeline already exists.
        let result = self
            .pipeline
            .run_entity_type_search(EntityTypeSearchQuery::new(question))
            .await?;
        Ok(result
            .matches
            .into_iter()
            .map(|hit| TypeMatch {
                entity_type: EntityType(hit.entity_type),
                score: hit
                    .vector_score
                    .or(hit.catalog_score)
                    .unwrap_or(0.0),
            })
            .collect())
    }

    async fn entity(&self, _id: &EntityId) -> Result<Option<Entity>> {
        Err(GraphError::Unsupported(
            "entity lookup requires node projection to be wired".into(),
        ))
    }

    async fn entities(&self, _ids: &[EntityId]) -> Result<Vec<Entity>> {
        Err(GraphError::Unsupported(
            "batch entity lookup requires node projection to be wired".into(),
        ))
    }

    async fn relation(&self, _id: &RelationId) -> Result<Option<Relation>> {
        Err(GraphError::Unsupported(
            "relation lookup requires edge projection to be wired".into(),
        ))
    }

    async fn neighbors(&self, _id: &EntityId, opts: TraversalOptions) -> Result<NeighborsPage> {
        // Enforce the two invariants the library guarantees regardless of
        // backend: bounded traversal, and depth within the tenant ceiling.
        if !opts.is_bounded() {
            return Err(GraphError::UnboundedTraversal);
        }
        if opts.depth > self.limits.max_depth {
            return Err(GraphError::CostExceeded(super::model::Cost {
                estimated_rows: 0,
                estimated_depth: opts.depth,
                units: f64::from(opts.depth),
            }));
        }
        Err(GraphError::Unsupported(
            "neighbors requires the traversal projection to be wired".into(),
        ))
    }

    async fn paths(
        &self,
        _from: &EntityId,
        _to: &EntityId,
        opts: PathOptions,
    ) -> Result<Vec<Path>> {
        if opts.max_depth == 0 || opts.limit == 0 {
            return Err(GraphError::UnboundedTraversal);
        }
        Err(GraphError::Unsupported(
            "paths requires the path search to be wired".into(),
        ))
    }

    async fn facets(&self, _scope: Filters, _dims: &[FacetDim]) -> Result<Facets> {
        Err(GraphError::Unsupported(
            "facets requires the aggregation queries to be wired".into(),
        ))
    }
}

/// Tenant-scoped, write-capable view. Implements [`GraphWrite`].
#[derive(Clone, Debug)]
pub struct WriteScope {
    pipeline: Pipeline,
    tenant: TenantId,
    limits: Limits,
}

impl WriteScope {
    /// The tenant this scope is bound to.
    pub fn tenant(&self) -> &TenantId {
        &self.tenant
    }

    /// The hard limits in force for this scope.
    pub fn limits(&self) -> Limits {
        self.limits
    }
}

#[async_trait]
impl GraphWrite for WriteScope {
    async fn extract(&self, _text: &str, _opts: ExtractOptions) -> Result<Extraction> {
        Err(GraphError::Unsupported(
            "extract requires the knowledge-extraction pipeline to be wired".into(),
        ))
    }

    async fn upsert(&self, _batch: GraphBatch, _opts: UpsertOptions) -> Result<UpsertReport> {
        Err(GraphError::Unsupported(
            "upsert requires the batch → Graph mapping to be wired".into(),
        ))
    }

    async fn delete_by_source(&self, source: &SourceId) -> Result<DeleteReport> {
        // Wired: source-rooted delete already exists on the pipeline.
        let summary = self.pipeline.delete_by_source(source.as_str()).await?;
        Ok(DeleteReport {
            source_found: summary.source_found,
            entities_deleted: summary.orphan_entities as u64,
            relations_deleted: summary.chunks as u64,
        })
    }

    async fn review_queue(&self, _opts: ReviewOptions) -> Result<Page<MergeCandidate>> {
        Err(GraphError::Unsupported(
            "review_queue requires the merge-candidate store to be wired".into(),
        ))
    }

    async fn resolve_merge(&self, _decision: MergeDecision) -> Result<ResolveReport> {
        Err(GraphError::Unsupported(
            "resolve_merge requires the merge applier to be wired".into(),
        ))
    }
}
