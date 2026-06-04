//! End-to-end orchestration: DSL/mapping → AST → Cypher → DB.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use crate::ast::query::Literal;
use crate::ast::{from_dsl, query::InsertQuery, query::ReadQuery};
use crate::builder::{self, CypherQuery};
use crate::config::{Config, SoftMergeConfig};
use crate::db::{GraphClient, QueryResult, Row, Value as DbValue};
use crate::dsl::{Direction as DslDirection, DslQuery, TraversalQuery};
use crate::embeddings::SharedEmbedder;
use crate::error::Result;
use crate::graph::{EntityTypeMatch, Graph, OntologyCatalog, OntologyCatalogStorage};
use crate::ingest::{self, soft_merge, DeletePlan, DiscoveredNodes, IngestError, PlannerOptions};
use crate::types::{handlers, SharedRegistry, SideEffect, SideEffectQueue};

use crate::core::entity_type_search::{
    self, EntityTypeHit, EntityTypeSearchQuery, EntityTypeSearchResult, HitRow,
};

use std::sync::RwLock;

/// High-level entrypoint used by the CLI and library consumers.
///
/// The pipeline is cheap to clone — its only state is a few `Arc`s and a
/// snapshot of the relevant config knobs.
#[derive(Clone)]
pub struct Pipeline {
    client: Arc<dyn GraphClient>,
    max_depth: u32,
    default_limit: u32,
    ingest_batch_size: usize,
    ontology_catalog_storage: Option<Arc<dyn OntologyCatalogStorage>>,
    /// Registry of [`crate::types::TypeHandler`] instances. Defaults to
    /// an empty registry so plain (untyped) DSL queries don't need any
    /// configuration; the CLI / library callers register handlers via
    /// [`Self::with_registry`].
    registry: SharedRegistry,
    /// Embedder used by side-effect drainage (e.g. the SemanticText
    /// pipeline). Optional: when not configured, queries that reference
    /// types requiring an embedder fail at lowering time, not at ingest.
    embedder: Option<SharedEmbedder>,
    /// In-memory snapshot of [`OntologyCatalog`] consulted by
    /// [`Self::lower`] to auto-resolve filter types when the DSL omits
    /// `"type"`, and by [`Self::live_schema`] to enrich introspection
    /// results with descriptions.
    ontology_catalog: Arc<RwLock<Option<Arc<OntologyCatalog>>>>,
    /// Optional Cypher label stamped onto every ingested entity and
    /// onto every node in lowered queries. Lets a caller scope inserts
    /// and reads to a tenant, dataset or document without having to
    /// thread the label through the DSL/Graph by hand. `None` keeps the
    /// historic behaviour (no extra label).
    prefix_label: Option<String>,
    /// Optional prefix folded into the embedding-index / Qdrant
    /// collection names a type handler emits during ingestion and
    /// query. Scopes the vector store the way `prefix_label` scopes the
    /// Memgraph data; the two are configured independently so callers
    /// who only need one of them don't pay for the other.
    prefix_index: Option<String>,
    /// Base name of the SemanticText Qdrant collection, as read from
    /// `[types.SemanticText].collection`. Captured at construction time
    /// so `delete_by_source` can enumerate collections without having
    /// to downcast handlers out of the registry.
    semantic_collection: String,
    /// Soft-merge resolver configuration (similarity threshold, fan-out).
    /// Captured at construction time from `[ingest.soft_merge]` so
    /// `ingest()` can run the resolver without revisiting the full
    /// `Config`.
    soft_merge: SoftMergeConfig,
}

impl std::fmt::Debug for Pipeline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Pipeline")
            .field("max_depth", &self.max_depth)
            .field("default_limit", &self.default_limit)
            .field("ingest_batch_size", &self.ingest_batch_size)
            .field(
                "ontology_catalog_storage",
                &self.ontology_catalog_storage.is_some(),
            )
            .field("registry", &self.registry)
            .field("embedder", &self.embedder.is_some())
            .finish_non_exhaustive()
    }
}

/// Summary returned by [`Pipeline::ingest`].
#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct IngestSummary {
    pub batches_executed: usize,
    pub node_rows: usize,
    pub relation_rows: usize,
    /// Number of side-effect batches executed (e.g. embedding upserts).
    pub side_effect_batches: usize,
    /// Number of side-effect rows applied (vectors inserted).
    pub side_effect_rows: usize,
    /// Wall-clock time the full `Pipeline::ingest` call took, in
    /// milliseconds. Covers the soft-merge resolver, the MERGE batches,
    /// and the embedding-upsert side effects — i.e. everything between
    /// receiving the `Graph` and returning this summary.
    pub elapsed_ms: u64,
    /// Soft-merge resolver report — counts, decision routing, and any
    /// review candidates produced during this ingest. Default when the
    /// graph had no `PrimaryKey::Soft` entities. See
    /// [`soft_merge::SoftMergeReport`] for the field documentation.
    pub soft_merge: soft_merge::SoftMergeReport,
}

/// Summary returned by [`Pipeline::delete_by_source`].
///
/// `source_found = false` means the source name was unknown to the
/// database and the call was a no-op — every other counter will be
/// zero in that case.
#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct DeleteBySourceSummary {
    /// Whether the `Source {name: $source_name}` node existed at all.
    pub source_found: bool,
    /// Number of orphan user entities removed (entities mentioned only
    /// by the source being deleted).
    pub orphan_entities: usize,
    /// Number of chunks removed. Chunks are 1:1 with their source, so
    /// they always go.
    pub chunks: usize,
    /// 1 when the source node itself was deleted, 0 otherwise.
    pub sources: usize,
    /// Number of Qdrant collections we issued `delete_batch` calls
    /// against. Each call is a no-op for ids the collection doesn't
    /// know, so this is an upper bound on the work qlink actually did.
    pub qlink_collections: usize,
}

impl Pipeline {
    pub fn new(client: Arc<dyn GraphClient>, config: &Config) -> Self {
        let semantic_collection = config
            .types
            .get("SemanticText")
            .and_then(|t| t.collection.clone())
            .unwrap_or_else(|| "semantic_text".into());
        Self {
            client,
            max_depth: config.query.max_traversal_depth,
            default_limit: config.query.default_limit,
            ingest_batch_size: 1000,
            ontology_catalog_storage: None,
            // Default registry contains the built-in scalar parsers.
            // Graph `Text` properties route through `SemanticText`, so
            // callers ingesting text-rich graphs should register that
            // handler via [`Self::with_registry`].
            registry: Arc::new(handlers::core_registry()),
            embedder: None,
            ontology_catalog: Arc::new(RwLock::new(None)),
            prefix_label: None,
            prefix_index: None,
            semantic_collection,
            soft_merge: config.ingest.soft_merge.clone(),
        }
    }

    /// Set the prefix label that scopes both ingestion and query
    /// matching. Pass `None` (or an empty string) to disable. The same
    /// prefix is applied to every node in queries (start + traversal
    /// targets) and to every MERGE pattern at ingest, so entities only
    /// merge with — and queries only return — same-prefix nodes.
    pub fn with_prefix_label(mut self, prefix_label: Option<impl Into<String>>) -> Self {
        self.prefix_label = prefix_label
            .map(Into::into)
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty());
        self
    }

    pub fn prefix_label(&self) -> Option<&str> {
        self.prefix_label.as_deref()
    }

    /// Set the prefix folded into every embedding-index / Qdrant
    /// collection name. Applied at ingest (when handlers queue
    /// embedding side effects) and at query (when typed filters resolve
    /// their collection parameter). Empty / `None` disables the prefix.
    pub fn with_prefix_index(mut self, prefix_index: Option<impl Into<String>>) -> Self {
        self.prefix_index = prefix_index
            .map(Into::into)
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty());
        self
    }

    pub fn prefix_index(&self) -> Option<&str> {
        self.prefix_index.as_deref()
    }

    /// Override the ingestion batch size. Useful for tests and for callers
    /// who know their downstream system has stricter parameter limits.
    pub fn with_ingest_batch_size(mut self, n: usize) -> Self {
        self.ingest_batch_size = n;
        self
    }

    /// Attach an ontology-catalog storage for query-time type inference
    /// and live-schema enrichment.
    pub fn with_ontology_catalog_storage(
        mut self,
        storage: Arc<dyn OntologyCatalogStorage>,
    ) -> Self {
        self.ontology_catalog_storage = Some(storage);
        self
    }

    pub fn ontology_catalog_storage(&self) -> Option<&Arc<dyn OntologyCatalogStorage>> {
        self.ontology_catalog_storage.as_ref()
    }

    /// Attach a type-handler registry. Without one, typed DSL filters
    /// fail with `UnknownType` and properties tagged with a `type` are
    /// stored verbatim.
    pub fn with_registry(mut self, registry: SharedRegistry) -> Self {
        self.registry = registry;
        self
    }

    pub fn registry(&self) -> &SharedRegistry {
        &self.registry
    }

    /// Attach an embedder. Required only when ingestion produces
    /// embedding side effects (the SemanticText pipeline) — for
    /// query-only workloads the registry's handlers carry their own
    /// embedder reference.
    pub fn with_embedder(mut self, embedder: SharedEmbedder) -> Self {
        self.embedder = Some(embedder);
        self
    }

    pub fn embedder(&self) -> Option<SharedEmbedder> {
        self.embedder.clone()
    }

    /// Pre-load the ontology-catalog snapshot used to auto-resolve
    /// filter types and enrich live schema with descriptions.
    pub fn with_ontology_catalog(self, catalog: Arc<OntologyCatalog>) -> Self {
        *self
            .ontology_catalog
            .write()
            .expect("ontology catalog lock poisoned") = Some(catalog);
        self
    }

    /// Eagerly load the ontology-catalog snapshot from configured storage.
    /// No-op when no storage is set.
    pub async fn load_ontology_catalog(&self) -> Result<()> {
        if let Some(storage) = &self.ontology_catalog_storage {
            let catalog = storage.load().await?;
            *self
                .ontology_catalog
                .write()
                .expect("ontology catalog lock poisoned") = Some(Arc::new(catalog));
        }
        Ok(())
    }

    /// Snapshot of the ontology catalog currently informing query
    /// lowering and live-schema enrichment.
    pub fn ontology_catalog(&self) -> Option<Arc<OntologyCatalog>> {
        self.ontology_catalog
            .read()
            .expect("ontology catalog lock poisoned")
            .clone()
    }

    /// Fetch the live graph schema from the underlying client and, when
    /// an [`OntologyCatalog`] snapshot is loaded, enrich every node and
    /// relationship with descriptions (and domain labels resolved from
    /// the Cypher labels the planner stamps at ingest time).
    ///
    /// Convenience pass-through to [`GraphClient::schema`] so callers
    /// can drive [`crate::prompt::generate_system_prompt`] without
    /// having to keep their own handle on the client. `filter` contains
    /// node-label fragments to omit; each fragment is matched like Cypher
    /// `CONTAINS`, and relationships touching those labels are omitted too.
    pub async fn live_schema<S: AsRef<str>>(
        &self,
        filter: &[S],
    ) -> Result<crate::prompt::GraphSchema> {
        let mut schema = self
            .client
            .schema()
            .await?
            .filter_node_labels_containing(filter);
        if let Some(catalog) = self.ontology_catalog() {
            catalog.enrich(&mut schema);
        }
        Ok(schema)
    }

    // ── Read path ───────────────────────────────────────────────────────────

    /// Lower a DSL document to the typed AST. Pure; no I/O.
    ///
    /// When an [`OntologyCatalog`] snapshot is loaded, filters that
    /// omit `"type"` are auto-resolved against it: if the property's
    /// type is `Text` (SemanticText), the SemanticText handler is
    /// selected without any DSL change.
    pub fn lower(&self, dsl: DslQuery) -> Result<ReadQuery> {
        let catalog = self.ontology_catalog();
        // The DSL's own `prefix_label` / `prefix_index` win over the
        // Pipeline's configured defaults: a per-query override (e.g.
        // for a one-off search across all tenants) can drop or replace
        // the default.
        let mut dsl = dsl;
        if dsl.prefix_label.is_none() {
            if let Some(p) = &self.prefix_label {
                dsl.prefix_label = Some(p.clone());
            }
        }
        if dsl.prefix_index.is_none() {
            if let Some(p) = &self.prefix_index {
                dsl.prefix_index = Some(p.clone());
            }
        }
        let mut q = from_dsl::lower_full(dsl, self.max_depth, &self.registry, catalog.as_deref())?;
        if q.limit.is_none() {
            q.limit = Some(self.default_limit);
        }
        Ok(q)
    }

    /// Compile a DSL document all the way to a parameterized Cypher query.
    pub fn compile(&self, dsl: DslQuery) -> Result<CypherQuery> {
        let mut ast = self.lower(dsl)?;
        self.prepare(&mut ast)?;
        Ok(builder::build_read_with(&ast, &self.registry)?)
    }

    /// Run handler-level batched preparation over a lowered AST.
    ///
    /// Walks `ast` looking for typed predicates, groups them by
    /// `type_id`, and gives each handler one batched opportunity to
    /// rewrite its predicates' `params` (e.g. fold N independent
    /// `embed(...)` calls into one `embed_batch(...)`).
    ///
    /// Handlers that don't override `prepare` get a no-op pass — the
    /// only cost is the walk. Callers that want full control over
    /// staging can skip `compile` and call `lower → prepare →
    /// build_read_with` directly.
    pub fn prepare(&self, ast: &mut ReadQuery) -> Result<()> {
        use crate::types::PrepareCtx;

        // Group typed predicates by their handler's type id.
        let mut by_type: std::collections::HashMap<
            crate::types::TypeId,
            Vec<&mut crate::types::TypedPredicate>,
        > = std::collections::HashMap::new();
        if let Some(expr) = ast.filter.as_mut() {
            collect_typed_predicates_mut(expr, &mut by_type);
        }

        for (type_id, mut preds) in by_type {
            let handler = self
                .registry
                .get(&type_id)
                .map_err(|e| crate::error::Error::Ast(crate::ast::AstError::Type(e)))?;
            let mut ctx = PrepareCtx::new(&mut preds);
            handler
                .prepare(&mut ctx)
                .map_err(|e| crate::error::Error::Ast(crate::ast::AstError::Type(e)))?;
        }
        Ok(())
    }

    /// Compile and execute against the configured graph client.
    pub async fn run(&self, dsl: DslQuery) -> Result<QueryResult> {
        let mut dsl = dsl;
        self.validate_dsl_relationship_directions(&mut dsl).await?;
        let cypher = self.compile(dsl)?;
        Ok(self.client.execute(&cypher).await?)
    }

    /// Correct LLM-emitted relationship directions against the live schema.
    ///
    /// The DSL describes direction relative to the traversal's `from` alias:
    /// `out` means `(from)-[edge]->(target)`, `in` means
    /// `(from)<-[edge]-(target)`. When the live schema only contains the
    /// opposite endpoint order for the same relationship label, flip the DSL
    /// direction before compilation.
    async fn validate_dsl_relationship_directions(&self, dsl: &mut DslQuery) -> Result<()> {
        let schema = self.live_schema::<&str>(&[]).await?;
        if schema.relationships.is_empty() || dsl.traversals.is_empty() {
            return Ok(());
        }

        let mut alias_labels = BTreeMap::new();
        alias_labels.insert(dsl.start.alias.clone(), dsl.start.label.clone());

        for traversal in &mut dsl.traversals {
            let from_alias = traversal.from.as_deref().unwrap_or(&dsl.start.alias);
            let Some(from_label) = alias_labels.get(from_alias) else {
                alias_labels.insert(
                    traversal.target.alias.clone(),
                    traversal.target.label.clone(),
                );
                continue;
            };
            let target_label = traversal.target.label.as_str();
            if from_label.is_empty() || target_label.is_empty() {
                alias_labels.insert(
                    traversal.target.alias.clone(),
                    traversal.target.label.clone(),
                );
                continue;
            }

            let out_exists = schema.relationships.iter().any(|rel| {
                rel.label == traversal.edge.label
                    && rel.from.as_deref() == Some(from_label.as_str())
                    && rel.to.as_deref() == Some(target_label)
            });
            let in_exists = schema.relationships.iter().any(|rel| {
                rel.label == traversal.edge.label
                    && rel.from.as_deref() == Some(target_label)
                    && rel.to.as_deref() == Some(from_label.as_str())
            });

            traversal.edge.direction = match traversal.edge.direction {
                DslDirection::Out if !out_exists && in_exists => DslDirection::In,
                DslDirection::In if !in_exists && out_exists => DslDirection::Out,
                DslDirection::Both if !out_exists && in_exists => DslDirection::In,
                DslDirection::Both if out_exists && !in_exists => DslDirection::Out,
                direction => direction,
            };

            alias_labels.insert(
                traversal.target.alias.clone(),
                traversal.target.label.clone(),
            );
        }

        Ok(())
    }

    // ── Traversal path ──────────────────────────────────────────────────────
    //
    // [`TraversalQuery`] is a higher-level, traversal-oriented shape
    // for text retrieval. Compilation still exposes the goal-based
    // chunk-search leg as a normal DSL query; execution runs the full
    // retrieval pipeline: entity-name lookups, goal chunk search, then
    // chunk-level deduplication.

    /// Lower a [`TraversalQuery`] goal-search leg into the typed AST by
    /// first converting it to a [`DslQuery`].
    pub fn lower_traversal(&self, traversal: TraversalQuery) -> Result<ReadQuery> {
        self.lower(traversal.into_dsl())
    }

    /// Compile a [`TraversalQuery`] goal-search leg all the way to a
    /// parameterized Cypher query.
    pub fn compile_traversal(&self, traversal: TraversalQuery) -> Result<CypherQuery> {
        self.compile(traversal.into_dsl())
    }

    /// Execute the traversal retrieval pipeline:
    ///
    /// 1. Search every supplied entity name and collect chunks that
    ///    mention matching entities.
    /// 2. Search chunks by the goal text.
    /// 3. Return unique chunks with the union of associated entities.
    pub async fn run_traversal(&self, traversal: TraversalQuery) -> Result<QueryResult> {
        let limit = traversal.limit.or(Some(self.default_limit));
        let mut merged = TraversalMerge::default();

        for dsl in traversal.entity_dsls() {
            let cypher = self.compile(dsl)?;
            tracing::debug!(target: "linguagraph::traversal", cypher = %cypher.text, "entity leg");
            let result = self.client.execute(&cypher).await?;
            tracing::trace!(target: "linguagraph::traversal", rows = result.rows.len(), "entity leg result");
            merged.extend(result);
        }

        let cypher = self.compile_traversal(traversal)?;
        tracing::debug!(target: "linguagraph::traversal", cypher = %cypher.text, "goal leg");
        let result = self.client.execute(&cypher).await?;
        tracing::trace!(target: "linguagraph::traversal", rows = result.rows.len(), "goal leg result");
        merged.extend(result);

        Ok(merged.finish(limit))
    }

    // ── Entity-type discovery ───────────────────────────────────────────────

    /// Discover which entity types in the graph are semantically
    /// relevant to a free-text user query.
    ///
    /// See [`crate::core::entity_type_search`] for the result shape and
    /// the two-channel design (vector search + ontology catalog).
    ///
    /// Returns an empty result when:
    ///
    /// * the user text is empty,
    /// * no Qdrant collections survive after the `fields` /
    ///   `collections` filter (nothing to search).
    ///
    /// Errors when no embedder is configured on the pipeline.
    pub async fn run_entity_type_search(
        &self,
        q: EntityTypeSearchQuery,
    ) -> Result<EntityTypeSearchResult> {
        let started = std::time::Instant::now();

        if q.text.trim().is_empty() {
            return Ok(EntityTypeSearchResult {
                elapsed_ms: started.elapsed().as_millis() as u64,
                ..EntityTypeSearchResult::default()
            });
        }

        let embedder = self.embedder.as_ref().ok_or_else(|| {
            crate::error::Error::Ingest(IngestError::Type(
                "Pipeline has no embedder; run_entity_type_search needs one to embed the user \
                 query — call Pipeline::with_embedder before invoking this method"
                    .into(),
            ))
        })?;
        let mut embedded = embedder.embed_batch(&[q.text.as_str()]).map_err(|e| {
            crate::error::Error::Ingest(IngestError::Type(format!(
                "embed_batch failed for entity-type search: {e}"
            )))
        })?;
        let query_vector = embedded.pop().ok_or_else(|| {
            crate::error::Error::Ingest(IngestError::Type(
                "embedder produced no vectors for entity-type search".into(),
            ))
        })?;

        let catalog_snapshot = self.ontology_catalog();
        let catalog = catalog_snapshot.as_deref();

        let collections = resolve_collections(
            &q,
            catalog,
            self.prefix_index.as_deref(),
            &self.semantic_collection,
        );
        if collections.is_empty() {
            return Ok(EntityTypeSearchResult {
                elapsed_ms: started.elapsed().as_millis() as u64,
                ..EntityTypeSearchResult::default()
            });
        }

        let prefix_label = self.prefix_label.as_deref();
        if let Some(p) = prefix_label {
            if !crate::ingest::delete::is_valid_ident(p) {
                return Err(crate::error::Error::Ingest(IngestError::InvalidLabel(
                    p.to_string(),
                )));
            }
        }

        // ── Vector channel ──────────────────────────────────────────
        let cypher = build_entity_type_search_cypher(
            &collections,
            &query_vector,
            q.top_k,
            q.score_threshold,
            prefix_label,
        );
        tracing::debug!(
            target: "linguagraph::entity_type_search",
            cypher = %cypher.text,
            collections = collections.len(),
            "vector channel"
        );
        let vector_result = self.client.execute(&cypher).await?;
        let hit_rows = decode_hit_rows(&vector_result)?;
        tracing::trace!(
            target: "linguagraph::entity_type_search",
            rows = hit_rows.len(),
            "vector channel result"
        );
        let mut matches = entity_type_search::aggregate_hits(hit_rows, catalog, prefix_label);

        // ── Catalog channel ─────────────────────────────────────────
        if q.include_catalog_signal {
            if let Some(catalog) = catalog {
                match catalog.find(
                    &q.text,
                    q.catalog_threshold,
                    embedder.as_ref(),
                    None,
                    0.0,
                ) {
                    Ok(catalog_hits) => merge_catalog_signal(&mut matches, catalog_hits),
                    Err(err) => {
                        tracing::debug!(
                            target: "linguagraph::entity_type_search",
                            error = %err,
                            "catalog signal unavailable"
                        );
                    }
                }
            }
        }

        // ── Neighbour roll-up ──────────────────────────────────────
        let neighbors = if q.include_neighbors && !matches.is_empty() {
            let hit_ids: Vec<i64> = matches
                .iter()
                .flat_map(|m| m.sample_node_ids.iter().copied())
                .collect();
            if hit_ids.is_empty() {
                Vec::new()
            } else {
                let neighbor_cypher = build_neighbor_cypher(&hit_ids);
                tracing::debug!(
                    target: "linguagraph::entity_type_search",
                    cypher = %neighbor_cypher.text,
                    hit_ids = hit_ids.len(),
                    "neighbour leg"
                );
                let result = self.client.execute(&neighbor_cypher).await?;
                let rows = decode_neighbor_rows(&result)?;
                let mut hits = entity_type_search::aggregate_hits(rows, catalog, prefix_label);
                // Neighbour scores are synthetic; clear them so callers
                // never confuse them with a real cosine signal.
                let matched_types: std::collections::HashSet<&str> =
                    matches.iter().map(|m| m.entity_type.as_str()).collect();
                hits.retain(|h| !matched_types.contains(h.entity_type.as_str()));
                for h in &mut hits {
                    h.vector_score = None;
                    h.per_collection.clear();
                }
                hits.sort_by(|a, b| a.entity_type.cmp(&b.entity_type));
                hits
            }
        } else {
            Vec::new()
        };

        entity_type_search::sort_matches(&mut matches);

        Ok(EntityTypeSearchResult {
            matches,
            neighbors,
            collections_searched: collections,
            elapsed_ms: started.elapsed().as_millis() as u64,
        })
    }

    // ── Insert path ─────────────────────────────────────────────────────────

    /// Compile a graph into one Cypher batch per node/relation group.
    /// Pure; no I/O. Drops side effects on the floor — use
    /// [`Self::lower_insert_with_effects`] when you need them.
    pub fn compile_insert(&self, graph: &Graph) -> Result<Vec<CypherQuery>> {
        let (insert, _effects) = self.lower_insert_with_effects(graph)?;
        Ok(builder::build_insert(&insert)?)
    }

    /// Lower a graph into the typed [`InsertQuery`] AST, dropping any
    /// queued side effects.
    pub fn lower_insert(&self, graph: &Graph) -> Result<InsertQuery> {
        Ok(self.lower_insert_with_effects(graph)?.0)
    }

    /// Lower a graph, returning both the [`InsertQuery`] and the queue
    /// of side effects that must run after the Memgraph batches succeed.
    pub fn lower_insert_with_effects(
        &self,
        graph: &Graph,
    ) -> Result<(InsertQuery, SideEffectQueue)> {
        let opts = PlannerOptions {
            max_batch_size: self.ingest_batch_size,
        };
        let mut effects = SideEffectQueue::new();
        let insert = ingest::plan_graph_with_registry_and_prefixes(
            graph,
            opts,
            &self.registry,
            &mut effects,
            self.prefix_label.as_deref(),
            self.prefix_index.as_deref(),
        )?;
        Ok((insert, effects))
    }

    /// Ingest a JSON document through a mapping.
    ///
    /// Convenience wrapper around [`crate::mapper::to_graph`] followed by
    /// [`Self::ingest`]. The mapping describes how to lift raw JSON rows
    /// into typed graph entities and relations; the resulting
    /// [`Graph`] is then ingested via the standard path so all the same
    /// type handlers, side effects, and prefix scoping apply.
    ///
    /// `GraphSpecification` derived from the mapping is *not* persisted
    /// here — that lives in the optional `graph_specification_storage`
    /// path and is the caller's responsibility (see
    /// `cli::cmd_ingest_json` for the full file-backed variant).
    pub async fn ingest_json(
        &self,
        mapping: &crate::mapper::Mapping,
        value: &serde_json::Value,
    ) -> Result<IngestSummary> {
        let mapped = crate::mapper::to_graph(mapping, value)
            .map_err(|e| IngestError::Type(format!("mapper::to_graph: {e}")))?;
        self.ingest(&mapped.graph).await
    }

    /// Compile and execute the full graph ingestion pipeline.
    ///
    /// Each batch is executed sequentially so a partial failure leaves the
    /// graph in a well-defined intermediate state (already-MERGE'd batches
    /// stay; the failing one rolls back its own work). Every node MERGE
    /// runs before any relationship MERGE, so the planner's ordering
    /// guarantees that when relations execute, both endpoints exist.
    pub async fn ingest(&self, graph: &Graph) -> Result<IngestSummary> {
        let started = std::time::Instant::now();
        // Soft-merge resolver: rewrite `PrimaryKey::Soft` properties
        // in place before the planner generates its `MERGE` so the
        // standard MERGE deduplicates against existing nodes by
        // semantic similarity. Skipped when the graph has no soft
        // candidates — the common case for graphs built from
        // explicit schemas.
        let owned;
        let (resolved_graph, soft_merge_report): (&Graph, soft_merge::SoftMergeReport) =
            if soft_merge::has_soft_merge_candidates(graph) {
                let embedder = self.embedder.as_deref().ok_or_else(|| {
                    IngestError::SoftMergeBackendUnavailable(
                        "Pipeline has no embedder; call .with_embedder() before ingesting graphs \
                         that contain PrimaryKey::Soft entities"
                            .into(),
                    )
                })?;
                let mut cloned = graph.clone();
                let report = soft_merge::resolve_soft_keys(
                    &mut cloned,
                    embedder,
                    self.client.as_ref(),
                    &self.soft_merge,
                    &self.semantic_collection,
                    self.prefix_index.as_deref(),
                )
                .await?;
                owned = cloned;
                (&owned, report)
            } else {
                (graph, soft_merge::SoftMergeReport::default())
            };

        let (insert, effects) = self.lower_insert_with_effects(resolved_graph)?;
        let node_rows: usize = insert.node_batches.iter().map(|b| b.rows.len()).sum();
        let relation_rows: usize = insert.relation_batches.iter().map(|b| b.rows.len()).sum();

        let batches = builder::build_insert(&insert)?;
        let total = batches.len();
        for batch in &batches {
            let _ = self.client.execute(batch).await?;
        }

        // Side effects run after the Memgraph batches land. For
        // [`SideEffect::EmbedAndStore`], that means: embed all queued
        // texts in one call, then issue *one* `qlink.insert_batch` per
        // (collection, label) group — never per-row.
        let (se_batches, se_rows) = self.drain_side_effects(effects).await?;

        Ok(IngestSummary {
            batches_executed: total,
            node_rows,
            relation_rows,
            side_effect_batches: se_batches,
            side_effect_rows: se_rows,
            elapsed_ms: started.elapsed().as_millis() as u64,
            soft_merge: soft_merge_report,
        })
    }

    // ── Delete path ─────────────────────────────────────────────────────────

    /// Delete a source-rooted subgraph: the `Source {name: $source_name}`
    /// node, every `Chunk` attached to it via `:part_of`, and every
    /// user entity whose only `:mention` link was to that source.
    ///
    /// Chunks are removed unconditionally — by construction they belong
    /// to exactly one source. User entities are removed only when they
    /// have no remaining `:mention` edges to a different source, so
    /// shared entities survive when a single source is deleted.
    ///
    /// The matching Qdrant vectors are cleaned up via
    /// `libqlink.delete_batch` (no-op for ids the collection doesn't
    /// know, so we can safely fan a single id list across every
    /// collection enumerated from the cached graph specification + the
    /// two built-ins, `name` and `text`).
    ///
    /// The three phases — discover, qlink cleanup, `DETACH DELETE` —
    /// each run as a separate Cypher statement so a failure mid-flight
    /// leaves the graph in a well-defined intermediate state: a partial
    /// qlink cleanup is idempotent (running the whole call again is
    /// fine), and the final `DETACH DELETE` only fires once we know
    /// every relevant Qdrant point has been removed.
    pub async fn delete_by_source(
        &self,
        source_name: impl Into<String>,
    ) -> Result<DeleteBySourceSummary> {
        let plan = DeletePlan::new(source_name, self.semantic_collection_base())
            .map_err(|e| IngestError::Type(e.to_string()))?
            .with_prefix_label(self.prefix_label.clone())
            .map_err(|e| IngestError::Type(e.to_string()))?
            .with_prefix_index(self.prefix_index.clone());

        let discovered = self
            .client
            .execute(&plan.discover_query())
            .await
            .map_err(crate::error::Error::Db)
            .and_then(|r| parse_discovered_nodes(&r).map_err(crate::error::Error::Ingest))?;

        if discovered.is_empty() {
            return Ok(DeleteBySourceSummary::default());
        }

        let all_ids = discovered.all_ids();

        // ── qlink cleanup: one call per collection, full id list. ────
        let collections = plan.qlink_collections(self.ontology_catalog().as_deref());
        for coll in &collections {
            let q = plan.qlink_delete_batch_query(coll, &all_ids);
            let _ = self.client.execute(&q).await?;
        }

        // ── DETACH DELETE everything. ────────────────────────────────
        let _ = self
            .client
            .execute(&plan.detach_delete_query(&all_ids))
            .await?;

        Ok(DeleteBySourceSummary {
            source_found: true,
            orphan_entities: discovered.orphan_ids.len(),
            chunks: discovered.chunk_ids.len(),
            sources: usize::from(discovered.source_id.is_some()),
            qlink_collections: collections.len(),
        })
    }

    /// Base name of the SemanticText Qdrant collection used by this
    /// pipeline. Captured from `[types.SemanticText].collection` at
    /// construction time so we don't have to downcast through the
    /// type-handler registry.
    fn semantic_collection_base(&self) -> &str {
        &self.semantic_collection
    }

    /// Drain the side-effect queue. Currently handles
    /// [`SideEffect::EmbedAndStore`].
    ///
    /// Strategy:
    ///
    /// 1. Embed all texts in **one** `embed_batch` call — amortises
    ///    model warm-up and tokenizer cost.
    /// 2. Bucket effects by `(collection, payload_label, node_label,
    ///    key_field)`. The bucket key carries enough context that
    ///    every row inside a bucket shares the same MATCH pattern and
    ///    targets the same Qdrant collection with the same label tag.
    /// 3. For each bucket, emit **one** `UNWIND $rows … MATCH …
    ///    CALL libqlink.insert_labeled(…)` Cypher batch.
    ///
    /// Why grouping by `(collection, payload_label)` is now safe (it
    /// wasn't when we grouped by collection alone): with the label in
    /// the key, every row inside a bucket has the same Cypher node
    /// label. The MATCH pattern is therefore consistent across rows,
    /// and the `merge_on` key is unique-per-label-by-construction (the
    /// mapping author declared it as the entity's primary key). No
    /// duplicate match, no cross-label vector clobbering.
    ///
    /// Returns `(batches_run, rows_inserted)`.
    async fn drain_side_effects(&self, mut effects: SideEffectQueue) -> Result<(usize, usize)> {
        if effects.is_empty() {
            return Ok((0, 0));
        }

        // Snapshot the queue. We keep the original ordering so the
        // generated Cypher is deterministic for snapshot tests.
        let queue: Vec<SideEffect> = effects.drain().collect();

        let embedder = self.embedder.as_ref().ok_or_else(|| {
            crate::error::Error::Ingest(IngestError::Type(
                "ingestion produced embedding side effects but no embedder is configured \
                 (call Pipeline::with_embedder)"
                    .into(),
            ))
        })?;

        // ── 1. Embed everything in one shot. ─────────────────────────
        let texts: Vec<&str> = queue
            .iter()
            .map(|e| match e {
                SideEffect::EmbedAndStore { text, .. } => text.as_str(),
            })
            .collect();
        let vectors = embedder.embed_batch(&texts).map_err(|e| {
            crate::error::Error::Ingest(IngestError::Type(format!("embed_batch: {e}")))
        })?;
        if vectors.len() != queue.len() {
            return Err(crate::error::Error::Ingest(IngestError::Type(format!(
                "embedder returned {} vectors for {} inputs",
                vectors.len(),
                queue.len()
            ))));
        }

        // ── 2. Bucket. The 4-tuple key keeps every row in a bucket on
        //    the same MATCH pattern and the same Qdrant collection.
        //    `BTreeMap` keeps groups ordered for deterministic Cypher.
        type GroupKey = (String, Option<String>, String, String);
        let mut groups: std::collections::BTreeMap<GroupKey, Vec<(SideEffect, Vec<f32>)>> =
            std::collections::BTreeMap::new();
        for (eff, vec) in queue.into_iter().zip(vectors.into_iter()) {
            let key = match &eff {
                SideEffect::EmbedAndStore {
                    collection,
                    label,
                    key_field,
                    payload_label,
                    ..
                } => (
                    collection.clone(),
                    payload_label.clone(),
                    label.clone(),
                    key_field.clone(),
                ),
            };
            groups.entry(key).or_default().push((eff, vec));
        }

        // ── 3. One UNWIND batch per group. ───────────────────────────
        let mut batches_run = 0usize;
        let mut rows_inserted = 0usize;
        for ((_coll, _plabel, _nlabel, _kfield), group) in groups {
            let cypher = handlers::build_embed_insert_batch(&group)
                .map_err(|e| crate::error::Error::Ingest(IngestError::Type(e.to_string())))?;
            let _ = self.client.execute(&cypher).await?;
            batches_run += 1;
            rows_inserted += group.len();
        }
        let _ = embedder.dim(); // assert the embedder was usable
        Ok((batches_run, rows_inserted))
    }
}

#[derive(Default)]
struct TraversalMerge {
    order: Vec<String>,
    chunks: BTreeMap<String, ChunkHit>,
}

#[derive(Default)]
struct ChunkHit {
    chunk_id: Option<DbValue>,
    chunk_text: Option<DbValue>,
    source_id: Option<DbValue>,
    source_name: Option<DbValue>,
    score: f64,
    entities: Vec<serde_json::Value>,
    entity_keys: BTreeSet<String>,
}

impl TraversalMerge {
    fn extend(&mut self, result: QueryResult) {
        for row in result.rows {
            let Some(key) = chunk_key(&row) else {
                continue;
            };
            let hit = self.chunks.entry(key.clone()).or_insert_with(|| {
                self.order.push(key);
                ChunkHit::default()
            });

            fill_if_missing(&mut hit.chunk_id, row.fields.get("chunk_id"));
            fill_if_missing(&mut hit.chunk_text, row.fields.get("chunk_text"));
            fill_if_missing(&mut hit.source_id, row.fields.get("source_id"));
            fill_if_missing(&mut hit.source_name, row.fields.get("source_name"));
            hit.score += score_from_row(&row);

            if let Some((entity_key, entity)) = entity_from_row(&row) {
                if hit.entity_keys.insert(entity_key) {
                    hit.entities.push(entity);
                }
            }
        }
    }

    fn finish(self, limit: Option<u32>) -> QueryResult {
        let take = limit.map(|n| n as usize).unwrap_or(usize::MAX);
        let mut ordered = self
            .order
            .into_iter()
            .enumerate()
            .filter_map(|(idx, key)| self.chunks.get(&key).map(|hit| (idx, key, hit.score)))
            .collect::<Vec<_>>();
        ordered.sort_by(|a, b| b.2.total_cmp(&a.2).then_with(|| a.0.cmp(&b.0)));

        let rows = ordered
            .into_iter()
            .take(take)
            .filter_map(|(_, key, _)| self.chunks.get(&key))
            .map(|hit| {
                let mut fields = BTreeMap::new();
                fields.insert(
                    "chunk_id".into(),
                    hit.chunk_id
                        .clone()
                        .unwrap_or(DbValue::String(String::new())),
                );
                fields.insert(
                    "chunk_text".into(),
                    hit.chunk_text.clone().unwrap_or(DbValue::Null),
                );
                fields.insert(
                    "source_id".into(),
                    hit.source_id.clone().unwrap_or(DbValue::Null),
                );
                fields.insert(
                    "source_name".into(),
                    hit.source_name.clone().unwrap_or(DbValue::Null),
                );
                fields.insert("score".into(), DbValue::Float(hit.score));
                fields.insert(
                    "entities".into(),
                    DbValue::Json(serde_json::Value::Array(hit.entities.clone())),
                );
                Row { fields }
            })
            .collect();

        QueryResult {
            columns: vec![
                "chunk_id".into(),
                "chunk_text".into(),
                "source_id".into(),
                "source_name".into(),
                "score".into(),
                "entities".into(),
            ],
            rows,
        }
    }
}

/// Decode the single-row result of [`DeletePlan::discover_query`] into
/// a typed [`DiscoveredNodes`]. The query always returns one row with
/// three columns; an empty result (or a null `source_id`) means the
/// source did not exist.
fn parse_discovered_nodes(
    result: &QueryResult,
) -> std::result::Result<DiscoveredNodes, IngestError> {
    let Some(row) = result.rows.first() else {
        return Ok(DiscoveredNodes::default());
    };
    let source_id = row.fields.get("source_id").and_then(db_value_as_i64);
    if source_id.is_none() {
        return Ok(DiscoveredNodes::default());
    }
    let orphan_ids = id_list(row.fields.get("orphan_ids"))
        .ok_or_else(|| IngestError::Type("delete: orphan_ids missing or malformed".into()))?;
    let chunk_ids = id_list(row.fields.get("chunk_ids"))
        .ok_or_else(|| IngestError::Type("delete: chunk_ids missing or malformed".into()))?;
    Ok(DiscoveredNodes {
        source_id,
        orphan_ids,
        chunk_ids,
    })
}

fn id_list(value: Option<&DbValue>) -> Option<Vec<i64>> {
    match value? {
        DbValue::Null => Some(Vec::new()),
        DbValue::Json(serde_json::Value::Null) => Some(Vec::new()),
        DbValue::Json(serde_json::Value::Array(items)) => items
            .iter()
            .map(|v| match v {
                serde_json::Value::Number(n) => n.as_i64(),
                _ => None,
            })
            .collect(),
        _ => None,
    }
}

fn db_value_as_i64(value: &DbValue) -> Option<i64> {
    match value {
        DbValue::Int(v) => Some(*v),
        DbValue::Json(serde_json::Value::Number(n)) => n.as_i64(),
        DbValue::Null | DbValue::Json(serde_json::Value::Null) => None,
        _ => None,
    }
}

fn chunk_key(row: &Row) -> Option<String> {
    row.fields
        .get("chunk_id")
        .and_then(db_value_key)
        .or_else(|| row.fields.get("chunk_text").and_then(db_value_key))
}

fn entity_from_row(row: &Row) -> Option<(String, serde_json::Value)> {
    if let Some(entity) = row.fields.get("entity") {
        Some((
            db_value_key(entity).unwrap_or_else(|| "entity".into()),
            db_value_to_json(entity),
        ))
    } else if row.fields.contains_key("entity_id")
        || row.fields.contains_key("entity_name")
        || row.fields.contains_key("entity_type")
    {
        let key = row
            .fields
            .get("entity_id")
            .and_then(db_value_key)
            .or_else(|| row.fields.get("entity_name").and_then(db_value_key))
            .unwrap_or_else(|| "entity".into());
        let mut entity = serde_json::Map::new();
        if let Some(value) = row.fields.get("entity_id") {
            entity.insert("id".into(), db_value_to_json(value));
        }
        if let Some(value) = row.fields.get("entity_name") {
            entity.insert("name".into(), db_value_to_json(value));
        }
        if let Some(value) = row.fields.get("entity_type") {
            entity.insert("type".into(), db_value_to_json(value));
        }
        Some((key, serde_json::Value::Object(entity)))
    } else {
        None
    }
}

fn score_from_row(row: &Row) -> f64 {
    row.fields
        .iter()
        .filter(|(key, _)| {
            key.as_str() == "score"
                || key.as_str() == "chunk_score"
                || key.as_str() == "entity_score"
                || key.contains("__score")
        })
        .filter_map(|(_, value)| db_value_as_f64(value))
        .sum()
}

fn db_value_as_f64(value: &DbValue) -> Option<f64> {
    match value {
        DbValue::Int(v) => Some(*v as f64),
        DbValue::Float(v) => Some(*v),
        DbValue::String(v) => v.parse().ok(),
        DbValue::Json(v) => match v {
            serde_json::Value::Number(n) => n.as_f64(),
            serde_json::Value::String(s) => s.parse().ok(),
            _ => None,
        },
        DbValue::Null | DbValue::Bool(_) => None,
    }
}

fn fill_if_missing(slot: &mut Option<DbValue>, value: Option<&DbValue>) {
    if slot.is_none() {
        if let Some(value) = value {
            *slot = Some(value.clone());
        }
    }
}

fn db_value_key(value: &DbValue) -> Option<String> {
    match value {
        DbValue::Null => None,
        DbValue::Bool(v) => Some(v.to_string()),
        DbValue::Int(v) => Some(v.to_string()),
        DbValue::Float(v) => Some(v.to_string()),
        DbValue::String(v) => Some(v.clone()),
        DbValue::Json(v) => match v {
            serde_json::Value::Null => None,
            serde_json::Value::String(s) => Some(s.clone()),
            other => Some(other.to_string()),
        },
    }
}

fn db_value_to_json(value: &DbValue) -> serde_json::Value {
    match value {
        DbValue::Null => serde_json::Value::Null,
        DbValue::Bool(v) => serde_json::Value::Bool(*v),
        DbValue::Int(v) => serde_json::Value::Number((*v).into()),
        DbValue::Float(v) => serde_json::Number::from_f64(*v)
            .map(serde_json::Value::Number)
            .unwrap_or(serde_json::Value::Null),
        DbValue::String(v) => serde_json::Value::String(v.clone()),
        DbValue::Json(v) => v.clone(),
    }
}

/// Walk a [`crate::ast::query::FilterExpression`] tree and collect mutable
/// references to every `Typed` predicate, grouped by `type_id`.
fn collect_typed_predicates_mut<'a>(
    expr: &'a mut crate::ast::query::FilterExpression,
    out: &mut std::collections::HashMap<
        crate::types::TypeId,
        Vec<&'a mut crate::types::TypedPredicate>,
    >,
) {
    use crate::ast::query::FilterExpression;
    match expr {
        FilterExpression::Predicate(_) => {}
        FilterExpression::Typed(t) => {
            let key = t.type_id.clone();
            out.entry(key).or_default().push(t);
        }
        FilterExpression::And(parts) | FilterExpression::Or(parts) => {
            for p in parts.iter_mut() {
                collect_typed_predicates_mut(p, out);
            }
        }
        FilterExpression::Not(inner) => collect_typed_predicates_mut(inner, out),
    }
}

/// Marker re-export so the embedder trait can be referenced via
/// `pipeline::Embedder` in the README/tests without exposing the whole
/// `embeddings` path.
pub use crate::embeddings::Embedder as _Embedder;

// ── Entity-type search helpers ──────────────────────────────────────────────

/// Resolve the list of Qdrant collections to search.
fn resolve_collections(
    q: &EntityTypeSearchQuery,
    catalog: Option<&OntologyCatalog>,
    prefix_index: Option<&str>,
    semantic_collection: &str,
) -> Vec<String> {
    if let Some(explicit) = &q.collections {
        return explicit.clone();
    }
    let mut field_names = crate::ingest::delete::text_field_names(catalog);
    if let Some(filter) = &q.fields {
        let allowed: BTreeSet<&str> = filter.iter().map(String::as_str).collect();
        field_names.retain(|name| allowed.contains(name.as_str()));
    }
    field_names
        .into_iter()
        .map(|p| crate::ingest::delete::with_prefix_index(prefix_index, semantic_collection, &p))
        .collect()
}

/// Build the multi-collection UNION ALL Cypher used by the vector
/// channel of [`Pipeline::run_entity_type_search`].
///
/// Each collection becomes a standalone top-level query: a single
/// `CALL libqlink.search_labeled` + MATCH + RETURN, with the
/// collection name folded into the projection so downstream
/// aggregation can attribute the score back to its source. The
/// branches are stitched together with `UNION ALL`.
///
/// We deliberately avoid wrapping each branch in `CALL { ... }`:
/// Memgraph rejects a top-level query that consists only of CALL
/// subqueries with an internal RETURN ("Query should either create
/// or update something, or return results"). UNION ALL between plain
/// queries — each terminating in RETURN — is the supported shape.
fn build_entity_type_search_cypher(
    collections: &[String],
    query_vector: &[f32],
    top_k: u32,
    score_threshold: Option<f32>,
    prefix_label: Option<&str>,
) -> CypherQuery {
    let mut params: BTreeMap<String, Literal> = BTreeMap::new();
    params.insert(
        "emb".into(),
        Literal::List(
            query_vector
                .iter()
                .map(|f| Literal::Float(*f as f64))
                .collect(),
        ),
    );
    params.insert("top_k".into(), Literal::Int(top_k as i64));
    let has_threshold = score_threshold.is_some();
    if let Some(thr) = score_threshold {
        params.insert("score_thr".into(), Literal::Float(thr as f64));
    }
    let has_prefix = prefix_label.is_some();
    if let Some(p) = prefix_label {
        params.insert("prefix_label".into(), Literal::String(p.to_string()));
    }

    let mut branches: Vec<String> = Vec::with_capacity(collections.len());
    for (idx, collection) in collections.iter().enumerate() {
        let coll_param = format!("coll_{idx}");
        params.insert(coll_param.clone(), Literal::String(collection.clone()));

        let mut where_parts: Vec<String> = Vec::new();
        where_parts.push("id(n) = qid".into());
        if has_threshold {
            where_parts.push("sc >= $score_thr".into());
        }
        if has_prefix {
            where_parts.push("$prefix_label IN labels(n)".into());
        }

        branches.push(format!(
            "CALL libqlink.search([${coll_param}], $emb, $top_k) \
             YIELD id AS qid, score AS sc\n\
             MATCH (n) WHERE {where_clause}\n\
             RETURN id(n) AS nid, labels(n) AS labs, sc AS score, ${coll_param} AS coll",
            where_clause = where_parts.join(" AND "),
        ));
    }

    let text = branches.join("\nUNION ALL\n");
    CypherQuery::new(text, params)
}

/// Build the 1-hop neighbour Cypher.
fn build_neighbor_cypher(hit_ids: &[i64]) -> CypherQuery {
    let mut params: BTreeMap<String, Literal> = BTreeMap::new();
    params.insert(
        "hit_ids".into(),
        Literal::List(hit_ids.iter().map(|i| Literal::Int(*i)).collect()),
    );
    let text = "UNWIND $hit_ids AS h\n\
                MATCH (n)-[]-(m) WHERE id(n) = h\n\
                RETURN DISTINCT id(m) AS nid, labels(m) AS labs"
        .to_string();
    CypherQuery::new(text, params)
}

/// Decode rows from the vector channel into [`HitRow`]s.
fn decode_hit_rows(result: &QueryResult) -> Result<Vec<HitRow>> {
    let mut out = Vec::with_capacity(result.rows.len());
    for row in &result.rows {
        let Some(nid) = row.fields.get("nid").and_then(db_value_as_i64) else {
            continue;
        };
        let Some(labels) = row.fields.get("labs").and_then(db_value_as_string_list) else {
            continue;
        };
        let score = row
            .fields
            .get("score")
            .and_then(db_value_as_f64)
            .unwrap_or(0.0) as f32;
        let collection = row
            .fields
            .get("coll")
            .and_then(db_value_as_string)
            .unwrap_or_default();
        out.push(HitRow {
            nid,
            labels,
            score,
            collection,
        });
    }
    Ok(out)
}

/// Decode neighbour-leg rows into [`HitRow`]s with a synthetic score
/// and empty collection — `aggregate_hits` zeroes both out downstream.
fn decode_neighbor_rows(result: &QueryResult) -> Result<Vec<HitRow>> {
    let mut out = Vec::with_capacity(result.rows.len());
    for row in &result.rows {
        let Some(nid) = row.fields.get("nid").and_then(db_value_as_i64) else {
            continue;
        };
        let Some(labels) = row.fields.get("labs").and_then(db_value_as_string_list) else {
            continue;
        };
        out.push(HitRow {
            nid,
            labels,
            score: 0.0,
            collection: String::new(),
        });
    }
    Ok(out)
}

/// Merge the catalog signal into existing matches. Adds a new hit when
/// the type is not yet present, otherwise fills in the catalog score
/// (and the domain, when the vector channel could not infer one).
fn merge_catalog_signal(matches: &mut Vec<EntityTypeHit>, catalog_hits: Vec<EntityTypeMatch<'_>>) {
    for cat_hit in catalog_hits {
        let name = cat_hit.entity_type.name.clone();
        if let Some(existing) = matches.iter_mut().find(|h| h.entity_type == name) {
            existing.catalog_score = Some(cat_hit.score);
            if existing.domain.is_none() {
                existing.domain = Some(cat_hit.domain.to_string());
            }
        } else {
            matches.push(EntityTypeHit {
                entity_type: name,
                domain: Some(cat_hit.domain.to_string()),
                catalog_score: Some(cat_hit.score),
                ..EntityTypeHit::default()
            });
        }
    }
}

fn db_value_as_string(value: &DbValue) -> Option<String> {
    match value {
        DbValue::String(s) => Some(s.clone()),
        DbValue::Json(serde_json::Value::String(s)) => Some(s.clone()),
        _ => None,
    }
}

fn db_value_as_string_list(value: &DbValue) -> Option<Vec<String>> {
    match value {
        DbValue::Json(serde_json::Value::Array(items)) => items
            .iter()
            .map(|v| match v {
                serde_json::Value::String(s) => Some(s.clone()),
                _ => None,
            })
            .collect(),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{DatabaseConfig, LlmConfig, OntologyCatalogConfig, QueryConfig};
    use crate::db::MockClient;
    use crate::graph::{GraphBuilder, PropertyType};

    fn cfg() -> Config {
        Config {
            database: DatabaseConfig {
                uri: "bolt://t".into(),
                user: "u".into(),
                password: "p".into(),
                database: "memgraph".into(),
                max_connections: 1,
                query_timeout_secs: 5,
            },
            llm: LlmConfig::default(),
            query: QueryConfig::default(),
            ontology_catalog: OntologyCatalogConfig::default(),
            prompt: Default::default(),
            ingest: Default::default(),
            types: Default::default(),
        }
    }

    #[test]
    fn prepare_invokes_each_typed_handler_exactly_once_per_query() {
        use crate::ast::query::{
            Action, Alias, FilterExpression, Literal, Node, PropertyRef, ReadQuery, ReturnClause,
        };
        use crate::types::{
            Capabilities, EmitCtx, IngestCtx, LowerCtx, PrepareCtx, PromptHint, RegistryBuilder,
            TypeError, TypeHandler, TypeId, TypedOp, TypedPredicate,
        };
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Arc as StdArc;

        #[derive(Debug)]
        struct CountingHandler {
            id: &'static str,
            prepared_size: StdArc<AtomicUsize>,
            prepared_calls: StdArc<AtomicUsize>,
        }

        impl TypeHandler for CountingHandler {
            fn type_id(&self) -> TypeId {
                TypeId::new(self.id)
            }
            fn capabilities(&self) -> Capabilities {
                Capabilities::EXACT_MATCH
            }
            fn supported_ops(&self) -> Vec<TypedOp> {
                vec![TypedOp::Eq]
            }
            fn on_ingest(&self, _: &mut IngestCtx<'_>) -> std::result::Result<(), TypeError> {
                Ok(())
            }
            fn lower(
                &self,
                _: &mut LowerCtx<'_>,
            ) -> std::result::Result<TypedPredicate, TypeError> {
                unreachable!("test pre-builds the AST")
            }
            fn prepare(&self, ctx: &mut PrepareCtx<'_>) -> std::result::Result<(), TypeError> {
                self.prepared_size.store(ctx.len(), Ordering::SeqCst);
                self.prepared_calls.fetch_add(1, Ordering::SeqCst);
                // Mark each predicate so we can verify the mutation
                // crossed the lifetime boundary cleanly.
                for p in ctx.predicates_mut() {
                    p.params.insert("prepared".into(), Literal::Bool(true));
                }
                Ok(())
            }
            fn emit(
                &self,
                _: &mut EmitCtx<'_>,
                _: &TypedPredicate,
            ) -> std::result::Result<(), TypeError> {
                unreachable!("prepare-only test")
            }
            fn prompt_hint(&self) -> PromptHint {
                PromptHint::from_capabilities(self.type_id(), self.capabilities())
            }
        }

        let prepared_size = StdArc::new(AtomicUsize::new(0));
        let prepared_calls = StdArc::new(AtomicUsize::new(0));

        let registry: SharedRegistry = StdArc::new(
            RegistryBuilder::new()
                .register(CountingHandler {
                    id: "Counted",
                    prepared_size: prepared_size.clone(),
                    prepared_calls: prepared_calls.clone(),
                })
                .build(),
        );
        let pipeline = Pipeline::new(Arc::new(MockClient::new()), &cfg()).with_registry(registry);

        // Build an AST with two `Counted` predicates AND-ed together.
        let mk_pred = |alias: &str| {
            FilterExpression::Typed(TypedPredicate {
                type_id: TypeId::new("Counted"),
                field: PropertyRef {
                    alias: Alias::new(alias),
                    property: Some("x".into()),
                },
                op: TypedOp::Eq,
                value: Literal::Int(1),
                params: Default::default(),
            })
        };
        let mut ast = ReadQuery {
            action: Action::Find,
            start: Node {
                label: "T".into(),
                alias: Alias::new("p"),
                prefix_label: None,
            },
            traversals: vec![],
            filter: Some(FilterExpression::And(vec![mk_pred("p"), mk_pred("p")])),
            returns: vec![ReturnClause::Field {
                field: PropertyRef {
                    alias: Alias::new("p"),
                    property: Some("x".into()),
                },
                alias: None,
            }],
            group_by: vec![],
            sort: vec![],
            limit: None,
        };

        pipeline.prepare(&mut ast).unwrap();

        assert_eq!(prepared_calls.load(Ordering::SeqCst), 1, "one batched call");
        assert_eq!(
            prepared_size.load(Ordering::SeqCst),
            2,
            "both predicates batched"
        );

        // Both predicates carry the mutation written through PrepareCtx.
        fn assert_prepared(e: &FilterExpression) {
            match e {
                FilterExpression::Typed(t) => {
                    assert_eq!(t.params.get("prepared"), Some(&Literal::Bool(true)));
                }
                FilterExpression::And(parts) | FilterExpression::Or(parts) => {
                    for p in parts {
                        assert_prepared(p);
                    }
                }
                FilterExpression::Not(inner) => assert_prepared(inner),
                FilterExpression::Predicate(_) => {}
            }
        }
        assert_prepared(ast.filter.as_ref().unwrap());
    }

    #[tokio::test]
    async fn ingest_graph_executes_insert_batches() {
        let pipeline = Pipeline::new(Arc::new(MockClient::new()), &cfg());
        let mut graph = GraphBuilder::new();
        graph
            .entity("Camera")
            .strict_primary_key("id")
            .property("id", PropertyType::String, "c1")
            .property("state", PropertyType::String, "active")
            .add();

        let summary = pipeline.ingest(&graph.build()).await.unwrap();

        assert_eq!(summary.batches_executed, 1);
        assert_eq!(summary.node_rows, 1);
        assert_eq!(summary.relation_rows, 0);
    }

    #[tokio::test]
    async fn run_traversal_compiles_through_semantic_text_handler() {
        use crate::embeddings::MockEmbedder;
        use crate::types::handlers::{SemanticTextConfig, SemanticTextHandler};
        use crate::types::RegistryBuilder;
        use std::sync::Arc as StdArc;

        // A registry with a SemanticText handler so the typed
        // filter the TraversalQuery emits actually lowers cleanly.
        let registry: SharedRegistry = StdArc::new(
            crate::types::handlers::register_core(RegistryBuilder::new())
                .register(SemanticTextHandler::new(
                    SemanticTextConfig {
                        embedding_model: None,
                        collection: "test".into(),
                        top_k: 10,
                        search_threshold: 0.8,
                        reranker_threshold: 0.3,
                    },
                    StdArc::new(MockEmbedder::new(8)),
                ))
                .build(),
        );
        let pipeline = Pipeline::new(Arc::new(MockClient::new()), &cfg()).with_registry(registry);

        let cypher = pipeline
            .compile_traversal(crate::dsl::TraversalQuery::new(
                ["Elon Musk", "Company"],
                "Find companies founded by Elon Musk",
                "What companies did Elon Musk found?",
            ))
            .expect("traversal compiles to cypher");

        // Chunk is the start node and gets a label; the entity
        // target is label-less (any node).
        assert!(
            cypher.text.contains("MATCH (c:Chunk)"),
            "expected (c:Chunk) start; got: {}",
            cypher.text
        );
        // part_of is the required hop (rendered as MATCH).
        assert!(
            cypher.text.contains("[po:part_of]->(s:Source)"),
            "expected (s:Source) part_of hop; got: {}",
            cypher.text
        );
        // MENTIONS is optional (rendered as OPTIONAL MATCH) and
        // outgoing from the chunk to the mentioned entity. The
        // entity target is label-less.
        assert!(
            cypher.text.contains("OPTIONAL MATCH (c)-[m:MENTIONS]->(e)"),
            "expected optional MENTIONS hop with label-less entity; got: {}",
            cypher.text
        );
        // Semantic search routes through the labeled qlink search.
        assert!(
            cypher.text.contains("libqlink.search_labeled"),
            "expected libqlink.search_labeled; got: {}",
            cypher.text
        );
        // The carry order on the sources WITH chain follows the
        // bound-alias order: c (start), then part_of's target s,
        // then mentions' target e.
        assert!(
            cypher
                .text
                .contains("WITH c, s, e, c__score_0\nOPTIONAL MATCH"),
            "expected source projection to carry semantic score; got: {}",
            cypher.text
        );
        assert!(
            cypher.text.contains("c__score_0 AS score"),
            "expected RETURN projection to expose semantic score; got: {}",
            cypher.text
        );
    }

    #[tokio::test]
    async fn run_traversal_merges_unique_chunks_and_entities() {
        use crate::embeddings::MockEmbedder;
        use crate::types::handlers::{SemanticTextConfig, SemanticTextHandler};
        use crate::types::RegistryBuilder;
        use std::sync::Arc as StdArc;

        let registry: SharedRegistry = StdArc::new(
            crate::types::handlers::register_core(RegistryBuilder::new())
                .register(SemanticTextHandler::new(
                    SemanticTextConfig {
                        embedding_model: None,
                        collection: "test".into(),
                        top_k: 10,
                        search_threshold: 0.8,
                        reranker_threshold: 0.3,
                    },
                    StdArc::new(MockEmbedder::new(8)),
                ))
                .build(),
        );

        let mock = Arc::new(MockClient::new());
        // MockClient pops from the back. run_traversal executes entity
        // lookups first, then the goal-search query.
        mock.enqueue(QueryResult {
            columns: vec![],
            rows: vec![traversal_row("c1", "chunk one", "e2", "SpaceX", "Company")],
        });
        mock.enqueue(QueryResult {
            columns: vec![],
            rows: vec![
                traversal_row("c1", "chunk one", "e1", "Elon Musk", "Person"),
                traversal_row("c2", "chunk two", "e1", "Elon Musk", "Person"),
            ],
        });

        let pipeline = Pipeline::new(mock, &cfg()).with_registry(registry);
        let result = pipeline
            .run_traversal(crate::dsl::TraversalQuery::new(
                ["Elon Musk"],
                "find companies",
                "query",
            ))
            .await
            .expect("traversal runs");

        assert_eq!(
            result
                .columns
                .iter()
                .map(|c| c.name.as_str())
                .collect::<Vec<_>>(),
            vec![
                "chunk_id",
                "chunk_text",
                "source_id",
                "source_name",
                "score",
                "entities"
            ]
        );
        assert_eq!(result.rows.len(), 2);
        assert_eq!(
            result.rows[0].fields.get("chunk_id"),
            Some(&DbValue::String("c1".into()))
        );
        let DbValue::Json(serde_json::Value::Array(entities)) =
            result.rows[0].fields.get("entities").unwrap()
        else {
            panic!("entities should be a JSON array");
        };
        assert_eq!(entities.len(), 2);
    }

    #[tokio::test]
    async fn run_traversal_sums_scores_and_orders_chunks() {
        use crate::embeddings::MockEmbedder;
        use crate::types::handlers::{SemanticTextConfig, SemanticTextHandler};
        use crate::types::RegistryBuilder;
        use std::sync::Arc as StdArc;

        let registry: SharedRegistry = StdArc::new(
            crate::types::handlers::register_core(RegistryBuilder::new())
                .register(SemanticTextHandler::new(
                    SemanticTextConfig {
                        embedding_model: None,
                        collection: "test".into(),
                        top_k: 10,
                        search_threshold: 0.8,
                        reranker_threshold: 0.3,
                    },
                    StdArc::new(MockEmbedder::new(8)),
                ))
                .build(),
        );

        let mock = Arc::new(MockClient::new());
        mock.enqueue(QueryResult {
            columns: vec![],
            rows: vec![traversal_row_with_score(
                "c2",
                "chunk two",
                "e2",
                "SpaceX",
                "Company",
                "c__score_0",
                0.7,
            )],
        });
        mock.enqueue(QueryResult {
            columns: vec![],
            rows: vec![
                traversal_row_with_score(
                    "c1",
                    "chunk one",
                    "e1",
                    "Elon Musk",
                    "Person",
                    "seed__score_0",
                    0.4,
                ),
                traversal_row_with_score(
                    "c2",
                    "chunk two",
                    "e1",
                    "Elon Musk",
                    "Person",
                    "seed__score_0",
                    0.2,
                ),
            ],
        });

        let pipeline = Pipeline::new(mock, &cfg()).with_registry(registry);
        let result = pipeline
            .run_traversal(crate::dsl::TraversalQuery::new(
                ["Elon Musk"],
                "find companies",
                "query",
            ))
            .await
            .expect("traversal runs");

        assert_eq!(
            result.rows[0].fields.get("chunk_id"),
            Some(&DbValue::String("c2".into()))
        );
        assert_eq!(
            result.rows[0].fields.get("score"),
            Some(&DbValue::Float(0.8999999999999999))
        );
        assert_eq!(
            result.rows[1].fields.get("chunk_id"),
            Some(&DbValue::String("c1".into()))
        );
    }

    fn traversal_row(
        chunk_id: &str,
        chunk_text: &str,
        entity_id: &str,
        entity_name: &str,
        entity_type: &str,
    ) -> Row {
        let mut fields = BTreeMap::new();
        fields.insert("chunk_id".into(), DbValue::String(chunk_id.into()));
        fields.insert("chunk_text".into(), DbValue::String(chunk_text.into()));
        fields.insert("source_id".into(), DbValue::String("s1".into()));
        fields.insert("source_name".into(), DbValue::String("source".into()));
        fields.insert("entity_id".into(), DbValue::String(entity_id.into()));
        fields.insert("entity_name".into(), DbValue::String(entity_name.into()));
        fields.insert("entity_type".into(), DbValue::String(entity_type.into()));
        Row { fields }
    }

    fn traversal_row_with_score(
        chunk_id: &str,
        chunk_text: &str,
        entity_id: &str,
        entity_name: &str,
        entity_type: &str,
        score_field: &str,
        score: f64,
    ) -> Row {
        let mut row = traversal_row(chunk_id, chunk_text, entity_id, entity_name, entity_type);
        row.fields.insert(score_field.into(), DbValue::Float(score));
        row
    }

    #[tokio::test]
    async fn ingest_without_store_does_not_panic() {
        let pipeline = Pipeline::new(Arc::new(MockClient::new()), &cfg());
        let mut graph = GraphBuilder::new();
        graph
            .entity("X")
            .strict_primary_key("id")
            .property("id", PropertyType::String, "a")
            .add();

        pipeline.ingest(&graph.build()).await.unwrap();
    }

    #[tokio::test]
    async fn delete_by_source_runs_discover_qlink_and_detach_phases() {
        use crate::ast::query::Literal;
        // MockClient pops from the back, so the LAST enqueued result
        // is returned by the FIRST `execute` call. That first call is
        // the discover query, so its row goes in last.
        let mock = Arc::new(MockClient::new());
        // qlink + detach calls return empty results — fine.
        mock.enqueue(QueryResult::empty()); // detach delete
                                            // qlink deletes — we don't know how many collections, just
                                            // enqueue enough empty results. With no spec set there are
                                            // 3 collections (name, text, _canonical).
        mock.enqueue(QueryResult::empty());
        mock.enqueue(QueryResult::empty());
        mock.enqueue(QueryResult::empty());
        // discover query — returns one row with the source's id and
        // some orphan/chunk ids. Mock client builds Value::Json cells
        // by hand to mirror the production decode path.
        let mut fields = BTreeMap::new();
        fields.insert("source_id".into(), DbValue::Json(serde_json::json!(42)));
        fields.insert(
            "orphan_ids".into(),
            DbValue::Json(serde_json::json!([1, 2, 3])),
        );
        fields.insert(
            "chunk_ids".into(),
            DbValue::Json(serde_json::json!([10, 11])),
        );
        mock.enqueue(QueryResult {
            columns: vec!["source_id".into(), "orphan_ids".into(), "chunk_ids".into()],
            rows: vec![Row { fields }],
        });

        let pipeline = Pipeline::new(mock.clone(), &cfg());
        let summary = pipeline
            .delete_by_source("src-123")
            .await
            .expect("delete runs");

        assert!(summary.source_found);
        assert_eq!(summary.orphan_entities, 3);
        assert_eq!(summary.chunks, 2);
        assert_eq!(summary.sources, 1);
        // Three built-in collections: semantic_text__name + semantic_text__text
        // + semantic_text___canonical (the soft-merge canonical slot).
        assert_eq!(summary.qlink_collections, 3);

        let captured = mock.captured.lock().unwrap();
        // 1 discover + 3 qlink + 1 detach.
        assert_eq!(captured.len(), 5);
        assert!(captured[0].text.contains("MATCH (s:Source"));
        assert!(captured[0].text.contains("{name: $source_name}"));
        assert!(captured[1].text.contains("libqlink.delete_batch"));
        assert!(captured[4].text.contains("DETACH DELETE"));
        // The detach call must receive every doomed id: 3 orphans + 2 chunks + 1 source = 6.
        match captured[4].params.get("ids").unwrap() {
            Literal::List(items) => assert_eq!(items.len(), 6),
            other => panic!("expected list, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn delete_by_source_is_a_noop_when_source_missing() {
        let mock = Arc::new(MockClient::new());
        // Discover returns a row with null source_id — source not found.
        let mut fields = BTreeMap::new();
        fields.insert("source_id".into(), DbValue::Null);
        fields.insert("orphan_ids".into(), DbValue::Json(serde_json::json!([])));
        fields.insert("chunk_ids".into(), DbValue::Json(serde_json::json!([])));
        mock.enqueue(QueryResult {
            columns: vec![],
            rows: vec![Row { fields }],
        });

        let pipeline = Pipeline::new(mock.clone(), &cfg());
        let summary = pipeline.delete_by_source("nope").await.unwrap();

        assert!(!summary.source_found);
        assert_eq!(summary.orphan_entities, 0);
        assert_eq!(summary.chunks, 0);
        assert_eq!(summary.sources, 0);
        assert_eq!(summary.qlink_collections, 0);

        // Only the discover query should have run.
        let captured = mock.captured.lock().unwrap();
        assert_eq!(captured.len(), 1);
    }

    #[tokio::test]
    async fn delete_by_source_threads_prefix_label_into_cypher() {
        use crate::ast::query::Literal;
        let mock = Arc::new(MockClient::new());
        mock.enqueue(QueryResult::empty());
        mock.enqueue(QueryResult::empty());
        mock.enqueue(QueryResult::empty());
        let mut fields = BTreeMap::new();
        fields.insert("source_id".into(), DbValue::Json(serde_json::json!(1)));
        fields.insert("orphan_ids".into(), DbValue::Json(serde_json::json!([])));
        fields.insert("chunk_ids".into(), DbValue::Json(serde_json::json!([])));
        mock.enqueue(QueryResult {
            columns: vec![],
            rows: vec![Row { fields }],
        });

        let pipeline = Pipeline::new(mock.clone(), &cfg())
            .with_prefix_label(Some("Tenant1".to_string()))
            .with_prefix_index(Some("tenant1".to_string()));
        pipeline.delete_by_source("src-1").await.unwrap();

        let captured = mock.captured.lock().unwrap();
        // Discover query carries the prefix on every MATCH.
        assert!(captured[0].text.contains("(s:Source:Tenant1"));
        // qlink collection names carry the prefix_index.
        let coll = captured[1]
            .params
            .get("coll")
            .and_then(|v| match v {
                Literal::String(s) => Some(s.as_str()),
                _ => None,
            })
            .unwrap_or("");
        assert!(coll.starts_with("tenant1__semantic_text__"), "got {coll}");
    }
}
