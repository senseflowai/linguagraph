//! End-to-end orchestration: DSL → AST → Cypher → DB.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;
use std::time::Instant;

use crate::ast::query::Literal;
use crate::ast::{from_dsl, query::InsertQuery, query::ReadQuery};
use crate::builder::{self, CypherQuery};
use crate::config::{Config, SoftMergeConfig};
use crate::db::{GraphClient, QueryResult, Row, Value as DbValue};
use crate::dsl::{Direction as DslDirection, DslQuery, TraversalQuery};
use crate::embeddings::{SharedEmbedder, SharedReranker};
use crate::error::Result;
use crate::graph::{EntityTypeMatch, Graph, OntologyCatalog, OntologyCatalogStorage};
use crate::ingest::{self, soft_merge, DeletePlan, DiscoveredNodes, IngestError, PlannerOptions};
use crate::types::handlers::semantic_text::{with_prefix_index, DEFAULT_RERANKER_THRESHOLD};
use crate::types::{handlers, SharedRegistry, SideEffect, SideEffectQueue};

use crate::core::entity_type_search::{
    self, EntityTypeHit, EntityTypeSearchQuery, EntityTypeSearchResult, HitRow,
};

use std::sync::RwLock;

const DEFAULT_TRAVERSAL_RERANK_TOP_K: usize = 50;
const EMBEDDING_PROGRESS_CHUNK_SIZE: usize = 128;

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
    /// Optional cross-encoder reranker. When set, the final step of
    /// [`Self::run_traversal`] re-scores the top-N chunks with this
    /// reranker; when unset, the rerank step is skipped silently.
    reranker: Option<SharedReranker>,
    /// Minimum reranker score required for a reranked traversal hit to
    /// survive in the final result set.
    reranker_threshold: f64,
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
    /// Number of Qdrant collections the `libqlink.delete_batch_all`
    /// sweep touched. The sweep visits every existing collection and is
    /// a no-op for ids a collection doesn't hold, so this is an upper
    /// bound on the work qlink actually did.
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
            reranker: None,
            reranker_threshold: config
                .types
                .get("SemanticText")
                .and_then(|t| t.reranker_threshold)
                .unwrap_or(DEFAULT_RERANKER_THRESHOLD),
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

    /// Attach a cross-encoder reranker. When set, [`Self::run_traversal`]
    /// reranks the top-N chunks by `Reranker::rerank(query, texts)`
    /// after the score-aggregation step. Without one, the rerank step
    /// is silently skipped.
    pub fn with_reranker(mut self, reranker: SharedReranker) -> Self {
        self.reranker = Some(reranker);
        self
    }

    pub fn reranker(&self) -> Option<SharedReranker> {
        self.reranker.clone()
    }

    /// Override the traversal reranker cutoff. Scores below this value
    /// are dropped after `Reranker::rerank` returns.
    pub fn with_reranker_threshold(mut self, threshold: f64) -> Self {
        self.reranker_threshold = threshold;
        self
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
    // [`TraversalQuery`] is a high-level, doc-graph–oriented retrieval
    // request. The pipeline runs an explicit two-channel vector search
    // (entities via `_canonical`, chunks via `text`) plus a Cypher
    // traversal that walks chunk-level `mentions` from matched entities to their
    // chunks. Results are deduplicated by chunk, per-chunk scores are
    // summed, sorted, truncated to `limit`, and (optionally) reranked.
    //
    // The graph schema is fixed: `Chunk` / `Source` / `mentions` /
    // `part_of`, chunk text on `Chunk.text`, entity merge-key on
    // `<Entity>._canonical`. Graphs ingested with custom labels can't
    // use this entry point.

    /// Execute the traversal retrieval pipeline.
    ///
    /// Steps:
    ///
    /// 1. Embed `goal_search_text()` and each entity name.
    /// 2. Issue one Cypher query that calls `libqlink.search_labeled`
    ///    against the `_canonical` collection (one branch per entity
    ///    name) and against the `text` collection (one branch). Each
    ///    branch is filtered by `prefix_label` and, for the entity
    ///    channel, optionally by `entity_types`.
    /// 3. Run a graph-traversal Cypher that maps entity hits back to
    ///    the chunks they're mentioned in (and joins each chunk to
    ///    its source) and that also expands chunk hits to their
    ///    mentioned entities + source. Each row carries the vector
    ///    score from step 2 via a parameter map.
    /// 4. Deduplicate rows by chunk, sum per-channel contributions
    ///    into `total_score`.
    /// 5. Sort by `total_score`, truncate to `limit`.
    /// 6. Optionally rerank the survivors with the pipeline's
    ///    cross-encoder reranker (controlled by `traversal.rerank`
    ///    and `Pipeline::with_reranker`).
    pub async fn run_traversal(&self, traversal: TraversalQuery) -> Result<QueryResult> {
        let limit = traversal.limit.unwrap_or(self.default_limit) as usize;
        let entity_names = traversal.entity_names();
        let goal_text = traversal.goal_search_text();

        let prefix_label = traversal
            .prefix_label
            .as_deref()
            .or(self.prefix_label.as_deref());
        if let Some(p) = prefix_label {
            if !crate::ingest::delete::is_valid_ident(p) {
                return Err(crate::error::Error::Ingest(IngestError::InvalidLabel(
                    p.to_string(),
                )));
            }
        }
        if let Some(types) = traversal.entity_types.as_ref() {
            for t in types {
                if !crate::ingest::delete::is_valid_ident(t) {
                    return Err(crate::error::Error::Ingest(IngestError::InvalidLabel(
                        t.to_string(),
                    )));
                }
            }
        }

        let prefix_index = traversal
            .prefix_index
            .as_deref()
            .or(self.prefix_index.as_deref());

        // ── 1. Embed all query texts in a single batch. ──────────────
        let embedder = self.embedder.as_ref().ok_or_else(|| {
            crate::error::Error::Ingest(IngestError::Type(
                "Pipeline has no embedder; run_traversal needs one to embed the \
                 query — call Pipeline::with_embedder before invoking this method"
                    .into(),
            ))
        })?;

        let mut batch_inputs: Vec<&str> = Vec::with_capacity(entity_names.len() + 1);
        let have_goal = !goal_text.is_empty();
        if have_goal {
            batch_inputs.push(goal_text.as_str());
        }
        for n in &entity_names {
            batch_inputs.push(n.as_str());
        }
        let vectors = if batch_inputs.is_empty() {
            Vec::new()
        } else {
            embedder.embed_batch(&batch_inputs).map_err(|e| {
                crate::error::Error::Ingest(IngestError::Type(format!(
                    "embed_batch failed for traversal: {e}"
                )))
            })?
        };

        let mut vectors = vectors.into_iter();
        let goal_vec = if have_goal { vectors.next() } else { None };
        let entity_vecs: Vec<Vec<f32>> = vectors.collect();

        // Both retrieval channels share the same top_k policy: at least
        // 50, and at least twice the limit so dedup has slack.
        let retrieval_top_k = std::cmp::max(50, limit.saturating_mul(2)) as u32;

        // ── 2. Vector retrieval (entity + chunk channels). ──────────
        let entity_collection = with_prefix_index(
            prefix_index,
            &format!("{}__{}", self.semantic_collection, "_canonical"),
        );
        let chunk_collection = with_prefix_index(
            prefix_index,
            &format!("{}__{}", self.semantic_collection, "text"),
        );

        let mut entity_hits: BTreeMap<i64, f64> = BTreeMap::new();
        let mut chunk_hits: BTreeMap<i64, f64> = BTreeMap::new();

        if !entity_vecs.is_empty() || goal_vec.is_some() {
            let cypher = build_traversal_search_cypher(
                &entity_collection,
                &entity_vecs,
                &chunk_collection,
                goal_vec.as_deref(),
                retrieval_top_k,
                prefix_label,
                traversal.entity_types.as_deref(),
            );
            tracing::debug!(
                target: "linguagraph::traversal",
                cypher = %cypher.text,
                entity_terms = entity_vecs.len(),
                has_goal = goal_vec.is_some(),
                "vector retrieval"
            );
            let result = self.client.execute(&cypher).await?;
            tracing::trace!(
                target: "linguagraph::traversal",
                rows = result.rows.len(),
                "vector retrieval rows"
            );
            for row in &result.rows {
                let Some(nid) = row.fields.get("nid").and_then(db_value_as_i64) else {
                    continue;
                };
                let score = row
                    .fields
                    .get("score")
                    .and_then(db_value_as_f64)
                    .unwrap_or(0.0);
                let leg = row
                    .fields
                    .get("leg")
                    .and_then(db_value_as_string)
                    .unwrap_or_default();
                let bucket = match leg.as_str() {
                    "entity" => &mut entity_hits,
                    "chunk" => &mut chunk_hits,
                    _ => continue,
                };
                // Keep the best score we've seen for a node id.
                let slot = bucket.entry(nid).or_insert(score);
                if score > *slot {
                    *slot = score;
                }
            }
        }

        if entity_hits.is_empty() && chunk_hits.is_empty() {
            return Ok(QueryResult {
                columns: traversal_result_columns(false),
                rows: Vec::new(),
            });
        }

        // ── 3. Graph traversal: map hits → chunks (+ source + entities). ──
        let cypher = build_traversal_graph_cypher(&entity_hits, &chunk_hits, prefix_label);
        tracing::debug!(
            target: "linguagraph::traversal",
            cypher = %cypher.text,
            entity_hits = entity_hits.len(),
            chunk_hits = chunk_hits.len(),
            "graph traversal"
        );
        let result = self.client.execute(&cypher).await?;
        tracing::trace!(
            target: "linguagraph::traversal",
            rows = result.rows.len(),
            "graph traversal rows"
        );

        // ── 4. Dedup + total_score aggregation. ──────────────────────
        let mut merged = TraversalMerge::default();
        merged.extend(result);

        // ── 5. Sort + top-N candidate pool. ─────────────────────────
        let want_rerank = match traversal.rerank {
            Some(v) => v,
            None => self.reranker.is_some(),
        };
        let candidate_limit = if want_rerank {
            std::cmp::max(limit, DEFAULT_TRAVERSAL_RERANK_TOP_K)
        } else {
            limit
        };
        let mut top = merged.take_top(candidate_limit);

        // ── 6. Optional rerank. ──────────────────────────────────────
        let mut reranked = false;
        if want_rerank {
            let reranker = self.reranker.as_ref().ok_or_else(|| {
                crate::error::Error::Ingest(IngestError::Type(
                    "TraversalQuery.rerank = true but Pipeline has no reranker \
                     — call Pipeline::with_reranker first"
                        .into(),
                ))
            })?;
            if !top.is_empty() {
                let texts: Vec<String> = top
                    .iter()
                    .map(|hit| match &hit.chunk_text {
                        Some(DbValue::String(s)) => s.clone(),
                        Some(v) => db_value_to_json(v).to_string(),
                        None => String::new(),
                    })
                    .collect();
                let query_str = if traversal.query.trim().is_empty() {
                    goal_text.as_str()
                } else {
                    traversal.query.as_str()
                };
                let scores = reranker.rerank(query_str, &texts).map_err(|e| {
                    crate::error::Error::Ingest(IngestError::Type(format!("reranker failed: {e}")))
                })?;
                if scores.len() != top.len() {
                    return Err(crate::error::Error::Ingest(IngestError::Type(format!(
                        "reranker returned {} scores for {} docs",
                        scores.len(),
                        top.len()
                    ))));
                }
                for (hit, score) in top.iter_mut().zip(scores.into_iter()) {
                    hit.rerank_score = Some(score);
                }
                top.retain(|hit| {
                    hit.rerank_score
                        .is_some_and(|score| score >= self.reranker_threshold)
                });
                // Sort by rerank_score desc, tie-break with total_score.
                top.sort_by(|a, b| {
                    b.rerank_score
                        .unwrap_or(f64::NEG_INFINITY)
                        .total_cmp(&a.rerank_score.unwrap_or(f64::NEG_INFINITY))
                        .then_with(|| b.score.total_cmp(&a.score))
                });
                top.truncate(limit);
                reranked = true;
            }
        }

        // ── 7. Build QueryResult. ────────────────────────────────────
        let rows: Vec<Row> = top.into_iter().map(|hit| hit.into_row(reranked)).collect();
        Ok(QueryResult {
            columns: traversal_result_columns(reranked),
            rows,
        })
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
                match catalog.find(&q.text, q.catalog_threshold, embedder.as_ref(), None, 0.0) {
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

    /// Compile and execute the full graph ingestion pipeline.
    ///
    /// Each batch is executed sequentially so a partial failure leaves the
    /// graph in a well-defined intermediate state (already-MERGE'd batches
    /// stay; the failing one rolls back its own work). Every node MERGE
    /// runs before any relationship MERGE, so the planner's ordering
    /// guarantees that when relations execute, both endpoints exist.
    pub async fn ingest(&self, graph: &Graph) -> Result<IngestSummary> {
        let started = Instant::now();
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
        tracing::info!(
            target: "linguagraph::pipeline",
            node_rows,
            relation_rows,
            batches = total,
            "starting Memgraph insert stage"
        );
        let insert_started = Instant::now();
        for (idx, batch) in batches.iter().enumerate() {
            let batch_started = Instant::now();
            tracing::info!(
                target: "linguagraph::pipeline",
                batch = idx + 1,
                batches = total,
                "executing Memgraph insert batch"
            );
            let _ = self.client.execute(batch).await?;
            tracing::info!(
                target: "linguagraph::pipeline",
                batch = idx + 1,
                batches = total,
                elapsed_ms = batch_started.elapsed().as_millis() as u64,
                "finished Memgraph insert batch"
            );
        }
        tracing::info!(
            target: "linguagraph::pipeline",
            batches = total,
            elapsed_ms = insert_started.elapsed().as_millis() as u64,
            "finished Memgraph insert stage"
        );

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
    /// The matching Qdrant vectors are cleaned up via a single
    /// collection-agnostic `libqlink.delete_batch_all` call, which
    /// sweeps the doomed id list across *every* Qdrant collection. This
    /// is a no-op for ids a collection doesn't hold, and — unlike the
    /// older per-collection fan-out keyed off the ontology catalog — it
    /// reaches entity-property collections the catalog can't enumerate,
    /// which is what used to leave orphan-entity embeddings behind.
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

        // ── qlink cleanup: one collection-agnostic sweep. ────────────
        // Deleting the doomed node ids from *every* Qdrant collection
        // means entity vectors are removed even when their per-field
        // collection isn't enumerable from the ontology catalog — the
        // failure mode that left orphan-entity embeddings behind. Node
        // ids are globally unique and qlink no-ops on ids a collection
        // doesn't hold, so only the doomed points are touched.
        let qlink_result = self
            .client
            .execute(&plan.qlink_delete_all_query(&all_ids))
            .await?;
        let qlink_collections = parse_collections_swept(&qlink_result);

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
            qlink_collections,
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
        tracing::info!(
            target: "linguagraph::pipeline",
            queued = queue.len(),
            "starting embedding side effects"
        );

        let embedder = self.embedder.as_ref().ok_or_else(|| {
            crate::error::Error::Ingest(IngestError::Type(
                "ingestion produced embedding side effects but no embedder is configured \
                 (call Pipeline::with_embedder)"
                    .into(),
            ))
        })?;

        // ── 1. Embed everything in one shot. ─────────────────────────
        let embed_started = Instant::now();
        let texts: Vec<&str> = queue
            .iter()
            .map(|e| match e {
                SideEffect::EmbedAndStore { text, .. } => text.as_str(),
            })
            .collect();
        let vectors = embed_with_progress(embedder, &texts).map_err(|e| {
            crate::error::Error::Ingest(IngestError::Type(format!("embed_batch: {e}")))
        })?;
        tracing::info!(
            target: "linguagraph::pipeline",
            queued = queue.len(),
            vectors = vectors.len(),
            elapsed_ms = embed_started.elapsed().as_millis() as u64,
            "finished embedding side effects"
        );

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
        let group_total = groups.len();
        let side_effect_started = Instant::now();
        for (idx, ((_coll, _plabel, _nlabel, _kfield), group)) in groups.into_iter().enumerate() {
            let batch_started = Instant::now();
            tracing::info!(
                target: "linguagraph::pipeline",
                batch = idx + 1,
                batches = group_total,
                rows = group.len(),
                "executing embedding insert batch"
            );
            let cypher = handlers::build_embed_insert_batch(&group)
                .map_err(|e| crate::error::Error::Ingest(IngestError::Type(e.to_string())))?;
            let _ = self.client.execute(&cypher).await?;
            batches_run += 1;
            rows_inserted += group.len();
            tracing::info!(
                target: "linguagraph::pipeline",
                batch = idx + 1,
                batches = group_total,
                rows = group.len(),
                elapsed_ms = batch_started.elapsed().as_millis() as u64,
                "finished embedding insert batch"
            );
        }
        let _ = embedder.dim(); // assert the embedder was usable
        tracing::info!(
            target: "linguagraph::pipeline",
            batches = batches_run,
            rows = rows_inserted,
            elapsed_ms = side_effect_started.elapsed().as_millis() as u64,
            "finished embedding insert stage"
        );
        Ok((batches_run, rows_inserted))
    }
}

fn embed_with_progress(
    embedder: &SharedEmbedder,
    texts: &[&str],
) -> std::result::Result<Vec<Vec<f32>>, crate::embeddings::EmbedError> {
    if texts.is_empty() {
        return Ok(Vec::new());
    }

    let total = texts.len();
    let chunk_size = EMBEDDING_PROGRESS_CHUNK_SIZE.max(1);
    let mut vectors = Vec::with_capacity(total);
    let chunks = texts.len().div_ceil(chunk_size);

    for (idx, chunk) in texts.chunks(chunk_size).enumerate() {
        tracing::info!(
            target: "linguagraph::pipeline",
            chunk = idx + 1,
            chunks = chunks,
            rows = chunk.len(),
            processed = vectors.len(),
            total,
            "embedding progress"
        );
        let chunk_vectors = embedder.embed_batch(chunk)?;
        if chunk_vectors.len() != chunk.len() {
            return Err(crate::embeddings::EmbedError::Backend(format!(
                "embedder returned {} vectors for {} inputs in chunk {} of {}",
                chunk_vectors.len(),
                chunk.len(),
                idx + 1,
                chunks
            )));
        }
        vectors.extend(chunk_vectors);
        tracing::info!(
            target: "linguagraph::pipeline",
            chunk = idx + 1,
            chunks = chunks,
            processed = vectors.len(),
            total,
            "embedding progress advanced"
        );
    }

    Ok(vectors)
}

/// Accumulator for the chunk-leg of [`Pipeline::run_traversal`].
///
/// Rows arrive tagged with a `leg` (`"entity"` or `"chunk"`) and a
/// `contrib_score` already pre-bound from the vector-retrieval map.
/// Each chunk's `total_score` is `Σ entity_contrib(unique_entity_id) +
/// chunk_contrib (once)`.
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
    rerank_score: Option<f64>,
    entities: Vec<serde_json::Value>,
    entity_keys: BTreeSet<String>,
    /// Per-channel accounting so a chunk hit by the same entity in
    /// multiple rows only contributes once for that entity, and the
    /// chunk-channel direct hit contributes at most once.
    accounted_entity_ids: BTreeSet<i64>,
    chunk_leg_accounted: bool,
    first_seen: usize,
}

impl TraversalMerge {
    fn extend(&mut self, result: QueryResult) {
        for row in result.rows {
            let Some(key) = chunk_key(&row) else {
                continue;
            };
            let next_index = self.order.len();
            let hit = self.chunks.entry(key.clone()).or_insert_with(|| {
                self.order.push(key);
                ChunkHit {
                    first_seen: next_index,
                    ..ChunkHit::default()
                }
            });

            fill_if_missing(&mut hit.chunk_id, row.fields.get("chunk_id"));
            fill_if_missing(&mut hit.chunk_text, row.fields.get("chunk_text"));
            fill_if_missing(&mut hit.source_id, row.fields.get("source_id"));
            fill_if_missing(&mut hit.source_name, row.fields.get("source_name"));

            let leg = row
                .fields
                .get("leg")
                .and_then(db_value_as_string)
                .unwrap_or_default();
            let contrib = row
                .fields
                .get("contrib_score")
                .and_then(db_value_as_f64)
                .unwrap_or(0.0);

            match leg.as_str() {
                "entity" => {
                    if let Some(eid) = row.fields.get("entity_nid").and_then(db_value_as_i64) {
                        if hit.accounted_entity_ids.insert(eid) {
                            hit.score += contrib;
                        }
                    } else {
                        // Fallback: no internal id, count once.
                        hit.score += contrib;
                    }
                }
                "chunk" => {
                    if !hit.chunk_leg_accounted {
                        hit.score += contrib;
                        hit.chunk_leg_accounted = true;
                    }
                }
                _ => {
                    // Legacy / test rows without explicit leg — fall
                    // back to summing every score-like field.
                    hit.score += score_from_row(&row);
                }
            }

            if let Some((entity_key, entity)) = entity_from_row(&row) {
                if hit.entity_keys.insert(entity_key) {
                    hit.entities.push(entity);
                }
            }
        }
    }

    /// Sort all collected chunks by `total_score` desc (tie-break by
    /// insertion order) and return the top-N as owned [`ChunkHit`]s.
    fn take_top(mut self, limit: usize) -> Vec<ChunkHit> {
        let mut keys: Vec<String> = self.order.drain(..).collect();
        // Keep only keys that still have an entry (defensive).
        keys.retain(|k| self.chunks.contains_key(k));
        keys.sort_by(|a, b| {
            let ha = self.chunks.get(a).unwrap();
            let hb = self.chunks.get(b).unwrap();
            hb.score
                .total_cmp(&ha.score)
                .then_with(|| ha.first_seen.cmp(&hb.first_seen))
        });
        keys.truncate(limit);
        keys.into_iter()
            .filter_map(|k| self.chunks.remove(&k))
            .collect()
    }
}

impl ChunkHit {
    fn into_row(self, include_rerank: bool) -> Row {
        let mut fields = BTreeMap::new();
        fields.insert(
            "chunk_id".into(),
            self.chunk_id.unwrap_or(DbValue::String(String::new())),
        );
        fields.insert(
            "chunk_text".into(),
            self.chunk_text.unwrap_or(DbValue::Null),
        );
        fields.insert("source_id".into(), self.source_id.unwrap_or(DbValue::Null));
        fields.insert(
            "source_name".into(),
            self.source_name.unwrap_or(DbValue::Null),
        );
        fields.insert("score".into(), DbValue::Float(self.score));
        if include_rerank {
            fields.insert(
                "rerank_score".into(),
                self.rerank_score
                    .map(DbValue::Float)
                    .unwrap_or(DbValue::Null),
            );
        }
        fields.insert(
            "entities".into(),
            DbValue::Json(serde_json::Value::Array(self.entities)),
        );
        Row { fields }
    }
}

fn traversal_result_columns(include_rerank: bool) -> Vec<crate::db::Column> {
    let mut cols = vec![
        crate::db::Column::new("chunk_id"),
        crate::db::Column::new("chunk_text"),
        crate::db::Column::new("source_id"),
        crate::db::Column::new("source_name"),
        crate::db::Column::new("score"),
    ];
    if include_rerank {
        cols.push(crate::db::Column::new("rerank_score"));
    }
    cols.push(crate::db::Column::new("entities"));
    cols
}

/// Build the vector-retrieval Cypher: one UNION ALL branch per entity
/// name (against the `_canonical` collection) plus one branch for the
/// goal vector (against the `text` collection). Each branch yields
/// `(nid, score, leg)` rows.
fn build_traversal_search_cypher(
    entity_collection: &str,
    entity_vecs: &[Vec<f32>],
    chunk_collection: &str,
    goal_vec: Option<&[f32]>,
    top_k: u32,
    prefix_label: Option<&str>,
    entity_types: Option<&[String]>,
) -> CypherQuery {
    let mut params: BTreeMap<String, Literal> = BTreeMap::new();
    params.insert("top_k".into(), Literal::Int(top_k as i64));

    let has_prefix = prefix_label.is_some();
    if let Some(p) = prefix_label {
        params.insert("prefix_label".into(), Literal::String(p.to_string()));
    }
    let has_types = entity_types.map(|t| !t.is_empty()).unwrap_or(false);
    if let Some(types) = entity_types {
        if !types.is_empty() {
            params.insert(
                "entity_types".into(),
                Literal::List(types.iter().map(|s| Literal::String(s.clone())).collect()),
            );
        }
    }

    let mut branches: Vec<String> = Vec::new();

    if !entity_vecs.is_empty() {
        params.insert(
            "entity_collection".into(),
            Literal::String(entity_collection.to_string()),
        );
        for (idx, vec) in entity_vecs.iter().enumerate() {
            let emb_name = format!("emb_e_{idx}");
            params.insert(
                emb_name.clone(),
                Literal::List(vec.iter().map(|f| Literal::Float(*f as f64)).collect()),
            );
            let mut where_parts: Vec<String> = vec!["id(e) = qid".into()];
            if has_prefix {
                where_parts.push("$prefix_label IN labels(e)".into());
            }
            if has_types {
                where_parts.push("any(t IN $entity_types WHERE t IN labels(e))".into());
            }
            branches.push(format!(
                "CALL libqlink.search([$entity_collection], $\
                 {emb_name}, $top_k) YIELD id AS qid, score AS sc\n\
                 MATCH (e) WHERE {where_clause}\n\
                 RETURN id(e) AS nid, sc AS score, \"entity\" AS leg",
                where_clause = where_parts.join(" AND ")
            ));
        }
    }

    if let Some(vec) = goal_vec {
        params.insert(
            "chunk_collection".into(),
            Literal::String(chunk_collection.to_string()),
        );
        params.insert(
            "emb_goal".into(),
            Literal::List(vec.iter().map(|f| Literal::Float(*f as f64)).collect()),
        );
        let mut where_parts: Vec<String> =
            vec!["id(c) = qid".into(), "\"Chunk\" IN labels(c)".into()];
        if has_prefix {
            where_parts.push("$prefix_label IN labels(c)".into());
        }
        branches.push(format!(
            "CALL libqlink.search([$chunk_collection], $emb_goal, $top_k) \
             YIELD id AS qid, score AS sc\n\
             MATCH (c) WHERE {where_clause}\n\
             RETURN id(c) AS nid, sc AS score, \"chunk\" AS leg",
            where_clause = where_parts.join(" AND ")
        ));
    }

    let text = branches.join("\nUNION ALL\n");
    CypherQuery::new(text, params)
}

/// Build the graph-traversal Cypher. Entity hits → walk back through
/// `mentions` to chunks; chunk hits → fan out to mentioned entities.
/// Each row carries the originating contribution score so the merge
/// step can sum them per chunk.
fn build_traversal_graph_cypher(
    entity_hits: &BTreeMap<i64, f64>,
    chunk_hits: &BTreeMap<i64, f64>,
    prefix_label: Option<&str>,
) -> CypherQuery {
    const CHUNK_MENTION_REL: &str = "mentions";

    let mut params: BTreeMap<String, Literal> = BTreeMap::new();

    params.insert(
        "entity_hit_ids".into(),
        Literal::List(entity_hits.keys().copied().map(Literal::Int).collect()),
    );
    params.insert(
        "chunk_hit_ids".into(),
        Literal::List(chunk_hits.keys().copied().map(Literal::Int).collect()),
    );

    let entity_score_map: BTreeMap<String, Literal> = entity_hits
        .iter()
        .map(|(id, score)| (id.to_string(), Literal::Float(*score)))
        .collect();
    let chunk_score_map: BTreeMap<String, Literal> = chunk_hits
        .iter()
        .map(|(id, score)| (id.to_string(), Literal::Float(*score)))
        .collect();
    params.insert("entity_scores".into(), Literal::Object(entity_score_map));
    params.insert("chunk_scores".into(), Literal::Object(chunk_score_map));

    let plabel = prefix_label.map(|p| format!(":{p}")).unwrap_or_default();

    let mut branches: Vec<String> = Vec::new();

    if !entity_hits.is_empty() {
        branches.push(format!(
            "MATCH (e) WHERE id(e) IN $entity_hit_ids\n\
             MATCH (c:Chunk{plabel})-[:{chunk_mention_rel}]->(e)\n\
             OPTIONAL MATCH (c)-[:part_of]->(s:Source{plabel})\n\
             RETURN id(c) AS chunk_id, c.id AS chunk_pk, c.text AS chunk_text, \
                    s.id AS source_id, s.name AS source_name, \
                    id(e) AS entity_nid, e AS entity, \
                    coalesce($entity_scores[toString(id(e))], 0.0) AS contrib_score, \
                    \"entity\" AS leg",
            plabel = plabel,
            chunk_mention_rel = CHUNK_MENTION_REL,
        ));
    }

    if !chunk_hits.is_empty() {
        branches.push(format!(
            "MATCH (c:Chunk{plabel}) WHERE id(c) IN $chunk_hit_ids\n\
             OPTIONAL MATCH (c)-[:{chunk_mention_rel}]->(e)\n\
             OPTIONAL MATCH (c)-[:part_of]->(s:Source{plabel})\n\
             RETURN id(c) AS chunk_id, c.id AS chunk_pk, c.text AS chunk_text, \
                    s.id AS source_id, s.name AS source_name, \
                    id(e) AS entity_nid, e AS entity, \
                    coalesce($chunk_scores[toString(id(c))], 0.0) AS contrib_score, \
                    \"chunk\" AS leg",
            plabel = plabel,
            chunk_mention_rel = CHUNK_MENTION_REL,
        ));
    }

    let text = branches.join("\nUNION ALL\n");
    CypherQuery::new(text, params)
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

/// Read the `collections` count from a `libqlink.delete_batch_all` result
/// row. Used only to populate [`DeleteBySourceSummary::qlink_collections`];
/// a missing or malformed value degrades to `0` rather than failing the
/// delete, since the sweep itself already succeeded by the time we parse.
fn parse_collections_swept(result: &QueryResult) -> usize {
    result
        .rows
        .first()
        .and_then(|row| row.fields.get("collections"))
        .and_then(db_value_as_i64)
        .filter(|n| *n >= 0)
        .map(|n| n as usize)
        .unwrap_or(0)
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
///
/// Every entity's Text properties are embedded into its `_canonical`
/// document, and chunks embed their `text`; nothing else is embedded
/// per-field. So discovery only needs to fan out over those two
/// collections — searching `_canonical` reaches every entity type
/// regardless of which property carried the matching text.
fn resolve_collections(
    q: &EntityTypeSearchQuery,
    _catalog: Option<&OntologyCatalog>,
    prefix_index: Option<&str>,
    semantic_collection: &str,
) -> Vec<String> {
    if let Some(explicit) = &q.collections {
        return explicit.clone();
    }
    let mut field_names: Vec<String> = vec!["_canonical".to_string(), "text".to_string()];
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
        use crate::config::TypeConfig;
        use crate::embeddings::MockEmbedder;
        use crate::types::handlers::register_default;
        use std::sync::Arc as StdArc;

        let mut cfg = cfg();
        cfg.types.insert(
            "SemanticText".into(),
            TypeConfig {
                collection: Some("semantic_text".into()),
                embedding_dim: Some(8),
                ..TypeConfig::default()
            },
        );
        let embedder: StdArc<dyn crate::embeddings::Embedder> = StdArc::new(MockEmbedder::new(8));
        let pipeline = Pipeline::new(Arc::new(MockClient::new()), &cfg)
            .with_registry(StdArc::new(
                register_default(&cfg, embedder.clone()).unwrap(),
            ))
            .with_embedder(embedder);
        let mut graph = GraphBuilder::new();
        graph
            .entity("Camera")
            .strict_primary_key("id")
            .property("id", PropertyType::Keyword, "c1")
            .property("state", PropertyType::Keyword, "active")
            .add();

        let summary = pipeline.ingest(&graph.build()).await.unwrap();

        assert_eq!(summary.batches_executed, 1);
        assert_eq!(summary.node_rows, 1);
        assert_eq!(summary.relation_rows, 0);
    }

    #[tokio::test]
    async fn run_traversal_emits_search_calls_against_canonical_and_text_collections() {
        use crate::embeddings::MockEmbedder;
        use std::sync::Arc as StdArc;

        let mock = Arc::new(MockClient::new());
        // MockClient pops from the back. Two calls happen: vector
        // retrieval (entity + chunk) then the graph traversal.
        // Enqueue in reverse: traversal result first (= popped last),
        // retrieval result last (= popped first).
        mock.enqueue(QueryResult {
            columns: vec![],
            rows: vec![],
        });
        // Retrieval: yield one entity hit (id=1, score=0.6) and one
        // chunk hit (id=10, score=0.5).
        mock.enqueue(QueryResult {
            columns: vec![],
            rows: vec![
                retrieval_row(1, 0.6, "entity"),
                retrieval_row(10, 0.5, "chunk"),
            ],
        });

        let pipeline =
            Pipeline::new(mock.clone(), &cfg()).with_embedder(StdArc::new(MockEmbedder::new(8)));

        let _ = pipeline
            .run_traversal(crate::dsl::TraversalQuery {
                entities: vec!["Elon Musk".into()],
                goal: "find companies".into(),
                query: "query".into(),
                prefix_label: Some("Tenant1".into()),
                prefix_index: Some("tenant1".into()),
                limit: Some(10),
                entity_types: None,
                rerank: Some(false),
            })
            .await
            .expect("traversal runs");

        let captured = mock.captured.lock().unwrap();
        assert_eq!(captured.len(), 2, "expected two cypher calls");
        let retrieval = &captured[0];

        // Collection names are bound as parameters; assert by checking
        // the parameter literal so the test isn't coupled to the
        // surface Cypher syntax.
        let entity_collection = match retrieval.params.get("entity_collection") {
            Some(Literal::String(s)) => s.clone(),
            other => panic!("entity_collection param missing or wrong type: {other:?}"),
        };
        assert_eq!(entity_collection, "tenant1__semantic_text___canonical");
        let chunk_collection = match retrieval.params.get("chunk_collection") {
            Some(Literal::String(s)) => s.clone(),
            other => panic!("chunk_collection param missing or wrong type: {other:?}"),
        };
        assert_eq!(chunk_collection, "tenant1__semantic_text__text");

        assert!(
            retrieval.text.contains("$prefix_label IN labels(e)")
                && retrieval.text.contains("$prefix_label IN labels(c)"),
            "prefix-label filter should appear in both branches: {}",
            retrieval.text
        );
        assert!(retrieval.text.contains("\"Chunk\" IN labels(c)"));
        assert!(retrieval.text.contains("\"entity\" AS leg"));
        assert!(retrieval.text.contains("\"chunk\" AS leg"));
        assert!(retrieval.text.contains("libqlink.search"));

        let traversal_cypher = &captured[1];
        assert!(
            traversal_cypher
                .text
                .contains("(c:Chunk:Tenant1)-[:mentions]->(e)"),
            "expected prefix-stamped chunk traversal: {}",
            traversal_cypher.text
        );
        assert!(traversal_cypher.text.contains("$entity_scores"));
        assert!(traversal_cypher.text.contains("$chunk_scores"));
    }

    #[tokio::test]
    async fn run_traversal_entity_types_filter_threads_into_cypher() {
        use crate::embeddings::MockEmbedder;
        use std::sync::Arc as StdArc;

        let mock = Arc::new(MockClient::new());
        mock.enqueue(QueryResult {
            columns: vec![],
            rows: vec![],
        });
        mock.enqueue(QueryResult {
            columns: vec![],
            rows: vec![],
        });

        let pipeline =
            Pipeline::new(mock.clone(), &cfg()).with_embedder(StdArc::new(MockEmbedder::new(8)));

        let _ = pipeline
            .run_traversal(crate::dsl::TraversalQuery {
                entities: vec!["Acme".into()],
                goal: "g".into(),
                query: "q".into(),
                prefix_label: None,
                prefix_index: None,
                limit: Some(10),
                entity_types: Some(vec!["Person".into(), "Company".into()]),
                rerank: Some(false),
            })
            .await
            .expect("traversal runs");

        let captured = mock.captured.lock().unwrap();
        let retrieval = &captured[0];
        assert!(
            retrieval
                .text
                .contains("any(t IN $entity_types WHERE t IN labels(e))"),
            "expected entity_types filter clause in entity branch: {}",
            retrieval.text
        );
    }

    #[tokio::test]
    async fn run_traversal_sums_scores_and_deduplicates_chunks() {
        use crate::embeddings::MockEmbedder;
        use std::sync::Arc as StdArc;

        let mock = Arc::new(MockClient::new());
        // Graph traversal result (popped second): two entity-leg rows
        // for chunk c2 (entities e1 + e2) and one chunk-leg row for c1.
        let mut rows = Vec::new();
        rows.push(traversal_aggregated_row("c2", 1, "e1", "entity", 0.6));
        rows.push(traversal_aggregated_row("c2", 2, "e2", "entity", 0.4));
        rows.push(traversal_aggregated_row("c1", 0, "", "chunk", 0.5));
        mock.enqueue(QueryResult {
            columns: vec![],
            rows,
        });
        // Retrieval result (popped first): scores are already baked
        // into the traversal rows above, so the retrieval payload is
        // empty here — but the call still has to return *something*.
        mock.enqueue(QueryResult {
            columns: vec![],
            rows: vec![
                retrieval_row(1, 0.6, "entity"),
                retrieval_row(2, 0.4, "entity"),
                retrieval_row(20, 0.5, "chunk"),
            ],
        });

        let pipeline = Pipeline::new(mock, &cfg()).with_embedder(StdArc::new(MockEmbedder::new(8)));
        let result = pipeline
            .run_traversal(crate::dsl::TraversalQuery {
                entities: vec!["E1".into(), "E2".into()],
                goal: "find".into(),
                query: "q".into(),
                prefix_label: None,
                prefix_index: None,
                limit: Some(10),
                entity_types: None,
                rerank: Some(false),
            })
            .await
            .expect("traversal runs");

        let cols: Vec<&str> = result.columns.iter().map(|c| c.name.as_str()).collect();
        assert_eq!(
            cols,
            vec![
                "chunk_id",
                "chunk_text",
                "source_id",
                "source_name",
                "score",
                "entities",
            ]
        );
        assert_eq!(result.rows.len(), 2);
        // c2 wins (0.6 + 0.4 = 1.0 > 0.5).
        assert_eq!(
            result.rows[0].fields.get("chunk_id"),
            Some(&DbValue::String("c2".into()))
        );
        match result.rows[0].fields.get("score") {
            Some(DbValue::Float(f)) => assert!((*f - 1.0).abs() < 1e-9, "got {f}"),
            other => panic!("expected float score; got {other:?}"),
        }
        let DbValue::Json(serde_json::Value::Array(entities)) =
            result.rows[0].fields.get("entities").unwrap()
        else {
            panic!("entities must be a JSON array");
        };
        assert_eq!(entities.len(), 2);
    }

    #[tokio::test]
    async fn run_traversal_reranker_filters_hits_below_threshold() {
        use crate::embeddings::{EmbedError, MockEmbedder, Reranker};
        use std::sync::Arc as StdArc;

        #[derive(Debug)]
        struct FixedReranker {
            scores: Vec<f64>,
        }

        impl Reranker for FixedReranker {
            fn rerank(
                &self,
                _query: &str,
                documents: &[String],
            ) -> std::result::Result<Vec<f64>, EmbedError> {
                assert_eq!(documents.len(), self.scores.len());
                Ok(self.scores.clone())
            }
        }

        let mock = Arc::new(MockClient::new());
        mock.enqueue(QueryResult {
            columns: vec![],
            rows: vec![
                traversal_aggregated_row("c1", 0, "", "chunk", 0.9),
                traversal_aggregated_row("c2", 0, "", "chunk", 0.8),
                traversal_aggregated_row("c3", 0, "", "chunk", 0.7),
            ],
        });
        mock.enqueue(QueryResult {
            columns: vec![],
            rows: vec![
                retrieval_row(10, 0.9, "chunk"),
                retrieval_row(20, 0.8, "chunk"),
                retrieval_row(30, 0.7, "chunk"),
            ],
        });

        let pipeline = Pipeline::new(mock, &cfg())
            .with_embedder(StdArc::new(MockEmbedder::new(8)))
            .with_reranker(StdArc::new(FixedReranker {
                scores: vec![0.9, 0.2, 0.7],
            }))
            .with_reranker_threshold(0.5);

        let result = pipeline
            .run_traversal(crate::dsl::TraversalQuery {
                entities: Vec::new(),
                goal: "find".into(),
                query: "q".into(),
                prefix_label: None,
                prefix_index: None,
                limit: Some(10),
                entity_types: None,
                rerank: Some(true),
            })
            .await
            .expect("traversal runs");

        let ids: Vec<&str> = result
            .rows
            .iter()
            .filter_map(|row| match row.fields.get("chunk_id") {
                Some(DbValue::String(s)) => Some(s.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(ids, vec!["c1", "c3"]);
        assert_eq!(result.rows.len(), 2);
        assert!(result.columns.iter().any(|c| c.name == "rerank_score"));
    }

    #[tokio::test]
    async fn run_traversal_reranker_receives_minimum_top_50_candidate_pool() {
        use crate::config::TypeConfig;
        use crate::embeddings::{EmbedError, MockEmbedder, Reranker};
        use std::sync::Arc as StdArc;

        #[derive(Debug)]
        struct LengthAssertingReranker {
            expected_len: usize,
        }

        impl Reranker for LengthAssertingReranker {
            fn rerank(
                &self,
                _query: &str,
                documents: &[String],
            ) -> std::result::Result<Vec<f64>, EmbedError> {
                assert_eq!(documents.len(), self.expected_len);
                Ok(vec![1.0; documents.len()])
            }
        }

        let mock = Arc::new(MockClient::new());
        mock.enqueue(QueryResult {
            columns: vec![],
            rows: (1..=50)
                .map(|idx| traversal_aggregated_row(&format!("c{idx}"), 0, "", "chunk", 1.0))
                .collect(),
        });
        mock.enqueue(QueryResult {
            columns: vec![],
            rows: (1..=50)
                .map(|idx| retrieval_row(idx, 1.0, "chunk"))
                .collect(),
        });

        let mut cfg = cfg();
        cfg.query.default_limit = 30;
        cfg.types.insert(
            "SemanticText".into(),
            TypeConfig {
                reranker_threshold: Some(0.0),
                ..TypeConfig::default()
            },
        );
        let pipeline = Pipeline::new(mock, &cfg)
            .with_embedder(StdArc::new(MockEmbedder::new(8)))
            .with_reranker(StdArc::new(LengthAssertingReranker { expected_len: 50 }));

        let result = pipeline
            .run_traversal(crate::dsl::TraversalQuery {
                entities: Vec::new(),
                goal: "find".into(),
                query: "q".into(),
                prefix_label: None,
                prefix_index: None,
                limit: None,
                entity_types: None,
                rerank: None,
            })
            .await
            .expect("traversal runs");

        assert_eq!(result.rows.len(), 30);
    }

    fn retrieval_row(nid: i64, score: f64, leg: &str) -> Row {
        let mut fields = BTreeMap::new();
        fields.insert("nid".into(), DbValue::Int(nid));
        fields.insert("score".into(), DbValue::Float(score));
        fields.insert("leg".into(), DbValue::String(leg.into()));
        Row { fields }
    }

    fn traversal_aggregated_row(
        chunk_id: &str,
        entity_nid: i64,
        entity_name: &str,
        leg: &str,
        contrib: f64,
    ) -> Row {
        let mut fields = BTreeMap::new();
        fields.insert("chunk_id".into(), DbValue::String(chunk_id.into()));
        fields.insert(
            "chunk_text".into(),
            DbValue::String(format!("text for {chunk_id}")),
        );
        fields.insert("source_id".into(), DbValue::String("s1".into()));
        fields.insert("source_name".into(), DbValue::String("source".into()));
        fields.insert("contrib_score".into(), DbValue::Float(contrib));
        fields.insert("leg".into(), DbValue::String(leg.into()));
        if !entity_name.is_empty() {
            fields.insert("entity_nid".into(), DbValue::Int(entity_nid));
            fields.insert("entity_id".into(), DbValue::String(entity_name.into()));
            fields.insert("entity_name".into(), DbValue::String(entity_name.into()));
        }
        Row { fields }
    }

    #[tokio::test]
    async fn ingest_without_store_does_not_panic() {
        use crate::config::TypeConfig;
        use crate::embeddings::MockEmbedder;
        use crate::types::handlers::register_default;
        use std::sync::Arc as StdArc;

        let mut cfg = cfg();
        cfg.types.insert(
            "SemanticText".into(),
            TypeConfig {
                collection: Some("semantic_text".into()),
                embedding_dim: Some(8),
                ..TypeConfig::default()
            },
        );
        let embedder: StdArc<dyn crate::embeddings::Embedder> = StdArc::new(MockEmbedder::new(8));
        let pipeline = Pipeline::new(Arc::new(MockClient::new()), &cfg)
            .with_registry(StdArc::new(
                register_default(&cfg, embedder.clone()).unwrap(),
            ))
            .with_embedder(embedder);
        let mut graph = GraphBuilder::new();
        graph
            .entity("X")
            .strict_primary_key("id")
            .property("id", PropertyType::Keyword, "a")
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
        // MockClient pops from the back, so enqueue in reverse call order:
        // detach (last) first, then the qlink sweep, then discover (last in,
        // first out).
        mock.enqueue(QueryResult::empty()); // detach delete
                                            // qlink sweep — one collection-agnostic call. It reports how
                                            // many collections were swept via the `collections` column.
        let mut qlink_fields = BTreeMap::new();
        qlink_fields.insert("success".into(), DbValue::Bool(true));
        qlink_fields.insert("collections".into(), DbValue::Json(serde_json::json!(7)));
        mock.enqueue(QueryResult {
            columns: vec!["success".into(), "collections".into()],
            rows: vec![Row {
                fields: qlink_fields,
            }],
        });
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
        // The sweep reports the number of collections it cleaned, surfaced
        // verbatim in the summary.
        assert_eq!(summary.qlink_collections, 7);

        let captured = mock.captured.lock().unwrap();
        // 1 discover + 1 qlink sweep + 1 detach.
        assert_eq!(captured.len(), 3);
        assert!(captured[0].text.contains("MATCH (s:Source"));
        assert!(captured[0].text.contains("{name: $source_name}"));
        assert!(captured[1].text.contains("libqlink.delete_batch_all"));
        // The sweep must receive every doomed id: 3 orphans + 2 chunks + 1 source = 6.
        match captured[1].params.get("ids").unwrap() {
            Literal::List(items) => assert_eq!(items.len(), 6),
            other => panic!("expected list, got {other:?}"),
        }
        assert!(captured[2].text.contains("DETACH DELETE"));
        // The detach call must receive every doomed id too.
        match captured[2].params.get("ids").unwrap() {
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
        let mock = Arc::new(MockClient::new());
        mock.enqueue(QueryResult::empty()); // detach
        mock.enqueue(QueryResult::empty()); // qlink sweep
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
        // Discover query carries the prefix_label on every MATCH so it only
        // touches the tenant's partition.
        assert!(captured[0].text.contains("(s:Source:Tenant1"));
        // The qlink cleanup is collection-agnostic: it sweeps every Qdrant
        // collection by id, so the prefix_index no longer has to be baked
        // into a per-collection name (which is exactly what used to leak
        // entity vectors when the catalog was incomplete).
        assert!(captured[1].text.contains("libqlink.delete_batch_all"));
        assert!(!captured[1].params.contains_key("coll"));
    }

    #[test]
    fn parse_collections_swept_reads_count_and_degrades_to_zero() {
        // Happy path: the `collections` column is surfaced verbatim.
        let mut fields = BTreeMap::new();
        fields.insert("success".into(), DbValue::Bool(true));
        fields.insert("collections".into(), DbValue::Json(serde_json::json!(4)));
        let result = QueryResult {
            columns: vec!["success".into(), "collections".into()],
            rows: vec![Row { fields }],
        };
        assert_eq!(parse_collections_swept(&result), 4);

        // Missing column / empty result degrade to 0 rather than panicking.
        assert_eq!(parse_collections_swept(&QueryResult::empty()), 0);
    }
}
