//! Business-facing graph explorer.
//!
//! [`Explorer`] is a read-only facade over [`Pipeline`] for browsing a
//! knowledge graph the way a person would: look an entity up, inspect
//! its properties and provenance, walk relations hop by hop, filter by
//! type, and (with a configured [`NlTranslator`]) ask natural-language
//! questions that come back with the executed query, a result table and
//! a displayable subgraph.
//!
//! Everything the explorer returns is a serde-serializable DTO
//! ([`dto`]) so a downstream service can expose it over HTTP verbatim.
//!
//! ```no_run
//! # use linguagraph::{core::Pipeline, explore::Explorer};
//! # async fn demo(pipeline: Pipeline) -> linguagraph::Result<()> {
//! let explorer = Explorer::new(pipeline);
//! let overview = explorer.overview().await?;
//! if let Some(card) = explorer.entity("m1").await? {
//!     let neighbors = explorer.neighbors(&card.node.id, &Default::default()).await?;
//!     println!("{} connects to {} nodes", card.node.name, neighbors.nodes.len());
//! }
//! # Ok(())
//! # }
//! ```
//!
//! ## Identity
//!
//! The public node handle is the **`id` property** (a string): stable
//! across sessions and re-ingest, and what exports round-trip on. Nodes
//! ingested without one get a `"_nid:<internal-id>"` fallback handle
//! that is only valid against the current database instance
//! ([`dto::NodeView::ephemeral_handle`]). `id` values are not enforced
//! unique — lookups take the first match.
//!
//! ## Confidence
//!
//! The pipeline never computes confidence. [`dto::NodeView::confidence`]
//! / [`dto::EdgeView::confidence`] merely surface a `confidence`
//! property when the ingested data carries one.

use std::collections::BTreeMap;
use std::sync::Arc;

use serde_json::Value as JsonValue;
use thiserror::Error;

use crate::core::Pipeline;
use crate::db::{QueryResult, Row, Value as DbValue};
use crate::dsl::DslQuery;
use crate::error::Result;
use crate::graph::{OntologyCatalog, CHUNK_LABEL, SOURCE_LABEL};
use crate::nl::NlTranslator;
use crate::prompt::PropertyType as SchemaPropertyType;

pub mod dto;

mod classify;
mod export;
mod queries;
mod subgraph;
mod timeline;

pub use dto::*;

use queries::{NeighborQuery, NodeHandle, NID_HANDLE_PREFIX};

/// Errors specific to the explorer surface.
#[derive(Debug, Error)]
pub enum ExploreError {
    /// A label / property / prefix is not a valid Cypher identifier —
    /// rejected instead of interpolated.
    #[error("`{0}` is not a valid Cypher identifier")]
    InvalidIdentifier(String),

    /// A lookup required an entity that does not exist.
    #[error("unknown entity `{0}`")]
    UnknownEntity(String),

    /// [`Explorer::ask`] needs an [`NlTranslator`]; wire one with
    /// [`Explorer::with_translator`].
    #[error("NL translator is not configured; use Explorer::with_translator")]
    TranslatorMissing,

    /// The schema exposes no string-typed properties to search over.
    #[error("the graph schema has no searchable string properties")]
    NoSearchableProperties,

    /// Semantic search needs an embedder + vector store.
    #[error("semantic search unavailable: {0}")]
    SemanticSearchUnavailable(String),
}

/// Safety caps for explorer responses.
#[derive(Debug, Clone)]
pub struct ExplorerLimits {
    /// Page size when the caller doesn't pass one.
    pub default_page: u32,
    /// Node cap for materialized subgraphs.
    pub max_subgraph_nodes: usize,
    /// Edge cap for materialized subgraphs.
    pub max_subgraph_edges: usize,
    /// Cap on the overview's source listing.
    pub max_sources: u32,
}

impl Default for ExplorerLimits {
    fn default() -> Self {
        Self {
            default_page: 50,
            max_subgraph_nodes: 100,
            max_subgraph_edges: 500,
            max_sources: 200,
        }
    }
}

/// Read-only graph browser over a configured [`Pipeline`].
///
/// Construction mirrors the pipeline's builder style:
///
/// ```no_run
/// # use std::sync::Arc;
/// # use linguagraph::{core::Pipeline, explore::Explorer, nl::NlTranslator};
/// # fn demo(pipeline: Pipeline, translator: Arc<NlTranslator>) {
/// let explorer = Explorer::new(pipeline).with_translator(translator);
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct Explorer {
    pipeline: Pipeline,
    translator: Option<Arc<NlTranslator>>,
    limits: ExplorerLimits,
}

impl Explorer {
    /// Wrap a pipeline. Tenant scoping (`prefix_label` / `prefix_index`)
    /// is inherited from the pipeline configuration.
    pub fn new(pipeline: Pipeline) -> Self {
        Self {
            pipeline,
            translator: None,
            limits: ExplorerLimits::default(),
        }
    }

    /// Attach the NL front-end required by [`Explorer::ask`].
    pub fn with_translator(mut self, translator: Arc<NlTranslator>) -> Self {
        self.translator = Some(translator);
        self
    }

    pub fn with_limits(mut self, limits: ExplorerLimits) -> Self {
        self.limits = limits;
        self
    }

    /// The wrapped pipeline (e.g. for running raw DSL alongside).
    pub fn pipeline(&self) -> &Pipeline {
        &self.pipeline
    }

    pub(crate) fn translator(&self) -> Option<&Arc<NlTranslator>> {
        self.translator.as_ref()
    }

    /// Whether [`Explorer::ask`] is available.
    pub fn has_translator(&self) -> bool {
        self.translator.is_some()
    }

    pub(crate) fn limits(&self) -> &ExplorerLimits {
        &self.limits
    }

    fn prefix(&self) -> Option<&str> {
        self.pipeline.prefix_label()
    }

    fn catalog(&self) -> Option<Arc<OntologyCatalog>> {
        self.pipeline.ontology_catalog()
    }

    fn page_limit(&self, requested: Option<u32>) -> u32 {
        requested.unwrap_or(self.limits.default_page).max(1)
    }

    // ── Ask ─────────────────────────────────────────────────────────────

    /// Answer a natural-language question: translate it to DSL
    /// ([`NlTranslator`] required), execute, and return the result table,
    /// the materialized subgraph, provenance sources, the full query
    /// trace, and (optionally) an LLM-synthesized answer.
    pub async fn ask(&self, question: &str, opts: &AskOptions) -> Result<AskResult> {
        let translator = self
            .translator
            .clone()
            .ok_or(ExploreError::TranslatorMissing)?;
        let started = std::time::Instant::now();
        let schema = self.pipeline.live_schema::<&str>(&[]).await?;
        let catalog = self.catalog();
        let empty_catalog = OntologyCatalog::default();
        let generation = translator
            .question_to_dsl(
                question,
                &schema,
                catalog.as_deref().unwrap_or(&empty_catalog),
                self.pipeline.prefix_label(),
                self.pipeline.prefix_index(),
            )
            .await
            .map_err(crate::error::Error::from)?;
        self.run_query_flow(
            Some(question.to_string()),
            generation.dsl,
            generation.attempts,
            started,
            opts,
        )
        .await
    }

    /// Execute a hand-written [`DslQuery`] through the same flow as
    /// [`Explorer::ask`] (no LLM translation). The pipeline's tenant
    /// prefixes are forced onto the query.
    pub async fn run_dsl(&self, dsl: DslQuery, opts: &AskOptions) -> Result<AskResult> {
        let started = std::time::Instant::now();
        let mut dsl = dsl;
        dsl.prefix_label = self.pipeline.prefix_label().map(str::to_string);
        dsl.prefix_index = self.pipeline.prefix_index().map(str::to_string);
        self.run_query_flow(None, dsl, 0, started, opts).await
    }

    // ── Inspection ──────────────────────────────────────────────────────

    /// Full inspector card for one entity, or `None` when the handle
    /// doesn't resolve.
    pub async fn entity(&self, id: &str) -> Result<Option<EntityCard>> {
        let catalog = self.catalog();
        let Some((handle, row)) = self.lookup_entity_row(id).await? else {
            return Ok(None);
        };
        let Some(node) = self.node_view_from_row(&row, catalog.as_deref()) else {
            return Ok(None);
        };
        let sources = parse_source_refs(row.fields.get("sources"));

        let summary_query = queries::relation_summary(&handle, self.prefix())?;
        let summary = self.pipeline.execute(&summary_query).await?;
        let relations = self.reduce_relation_summary(&summary, catalog.as_deref());

        Ok(Some(EntityCard {
            node,
            sources,
            relations,
        }))
    }

    /// Resolve a handle to its entity row. All-digit inputs that miss as
    /// an `id` property are retried as Memgraph internal ids — that's
    /// what graph tools (Memgraph Lab, logs) put in front of users.
    async fn lookup_entity_row(&self, id: &str) -> Result<Option<(NodeHandle, Row)>> {
        let handle = NodeHandle::parse(id);
        let result = self
            .pipeline
            .execute(&queries::entity_by_id(&handle, self.prefix())?)
            .await?;
        if let Some(row) = result.rows.into_iter().next() {
            return Ok(Some((handle, row)));
        }
        if let Some(fallback) = handle.internal_id_fallback() {
            let result = self
                .pipeline
                .execute(&queries::entity_by_id(&fallback, self.prefix())?)
                .await?;
            if let Some(row) = result.rows.into_iter().next() {
                return Ok(Some((fallback, row)));
            }
        }
        Ok(None)
    }

    /// One hop from `id`: the origin, its (filtered) neighbors and the
    /// connecting edges. Errors with [`ExploreError::UnknownEntity`] when
    /// the origin doesn't resolve.
    pub async fn neighbors(&self, id: &str, opts: &NeighborOptions) -> Result<Subgraph> {
        let catalog = self.catalog();

        let (handle, origin_row) = self
            .lookup_entity_row(id)
            .await?
            .ok_or_else(|| ExploreError::UnknownEntity(id.to_string()))?;
        let origin = self
            .node_view_from_row(&origin_row, catalog.as_deref())
            .ok_or_else(|| ExploreError::UnknownEntity(id.to_string()))?;

        let limit = self.page_limit(opts.limit);
        let query = queries::neighbors(
            &NeighborQuery {
                handle: &handle,
                edge_types: opts.edge_types.as_deref(),
                target_labels: opts.target_labels.as_deref(),
                direction: opts.direction,
                limit,
                offset: opts.offset,
            },
            self.prefix(),
        )?;
        let result = self.pipeline.execute(&query).await?;

        let mut nodes: Vec<NodeView> = vec![origin.clone()];
        let mut edges: Vec<EdgeView> = Vec::new();
        for row in &result.rows {
            let Some(neighbor) = self.node_view_from_row(row, catalog.as_deref()) else {
                continue;
            };
            let edge_type = string_field(row, "rel").unwrap_or_default();
            let outgoing = bool_field(row, "outgoing").unwrap_or(false);
            let (from, to) = if outgoing {
                (origin.id.clone(), neighbor.id.clone())
            } else {
                (neighbor.id.clone(), origin.id.clone())
            };
            let properties = json_object_field(row, "rel_props");
            let confidence = properties.get("confidence").and_then(JsonValue::as_f64);
            let edge = EdgeView {
                id: format!("{from}:{edge_type}:{to}"),
                edge_type,
                from,
                to,
                properties,
                confidence,
            };
            if !nodes.iter().any(|n| n.id == neighbor.id) {
                nodes.push(neighbor);
            }
            if !edges.iter().any(|e| e.id == edge.id) {
                edges.push(edge);
            }
        }
        let truncated = result.rows.len() as u32 >= limit;
        Ok(Subgraph {
            nodes,
            edges,
            truncated,
        })
    }

    // ── Discovery ───────────────────────────────────────────────────────

    /// Dataset overview: entity/relation types with counts, totals and
    /// the ingested sources. Descriptions come from the ontology catalog
    /// via the (cached) live schema.
    pub async fn overview(&self) -> Result<OverviewReport> {
        let catalog = self.catalog();
        let schema = self.pipeline.live_schema::<&str>(&[]).await?;

        let label_counts = self
            .pipeline
            .execute(&queries::label_set_counts(self.prefix())?)
            .await?;
        let mut per_type: BTreeMap<String, u64> = BTreeMap::new();
        for row in &label_counts.rows {
            let labels = string_list_field(row, "labels");
            let count = int_field(row, "cnt").unwrap_or(0).max(0) as u64;
            let primary = classify::primary_label(&labels, self.prefix(), catalog.as_deref());
            *per_type.entry(primary).or_default() += count;
        }
        per_type.remove(SOURCE_LABEL);
        per_type.remove(CHUNK_LABEL);
        let total_entities: u64 = per_type.values().sum();
        let mut entity_types: Vec<TypeCount> = per_type
            .into_iter()
            .map(|(name, count)| {
                let description = schema
                    .nodes
                    .iter()
                    .find(|n| n.label == name)
                    .and_then(|n| n.description.clone());
                TypeCount {
                    name,
                    count,
                    description,
                }
            })
            .collect();
        entity_types.sort_by(|a, b| b.count.cmp(&a.count).then(a.name.cmp(&b.name)));

        let rel_counts = self
            .pipeline
            .execute(&queries::relation_type_counts(self.prefix())?)
            .await?;
        let mut relation_types = Vec::new();
        let mut total_relations = 0_u64;
        for row in &rel_counts.rows {
            let Some(name) = string_field(row, "rel") else {
                continue;
            };
            if name == crate::graph::MENTION_REL || name == crate::graph::PART_OF_REL {
                continue;
            }
            let count = int_field(row, "cnt").unwrap_or(0).max(0) as u64;
            total_relations += count;
            let kind = schema.relationships.iter().find(|r| r.label == name);
            relation_types.push(RelTypeCount {
                name,
                count,
                from: kind.and_then(|k| k.from.clone()),
                to: kind.and_then(|k| k.to.clone()),
                description: kind.and_then(|k| k.description.clone()),
            });
        }

        let sources_result = self
            .pipeline
            .execute(&queries::list_sources(self.limits.max_sources, self.prefix())?)
            .await?;
        let sources = sources_result
            .rows
            .iter()
            .map(|row| SourceRef {
                id: string_field(row, "id"),
                name: string_field(row, "name"),
            })
            .filter(|s| s.id.is_some() || s.name.is_some())
            .collect();

        Ok(OverviewReport {
            entity_types,
            relation_types,
            total_entities,
            total_relations,
            sources,
        })
    }

    /// One page of entities of `entity_type`, with suggested display
    /// columns.
    pub async fn entities_of_type(
        &self,
        entity_type: &str,
        page: &PageOptions,
    ) -> Result<EntityTable> {
        let catalog = self.catalog();
        let limit = self.page_limit(page.limit);
        let sort_by = page.sort_by.as_deref().unwrap_or("name");
        let page_query = queries::entity_table_page(
            entity_type,
            sort_by,
            limit,
            page.offset,
            self.prefix(),
        )?;
        let result = self.pipeline.execute(&page_query).await?;
        let rows: Vec<NodeView> = result
            .rows
            .iter()
            .filter_map(|row| self.node_view_from_row(row, catalog.as_deref()))
            .collect();

        let count_result = self
            .pipeline
            .execute(&queries::entity_count(entity_type, self.prefix())?)
            .await?;
        let total = count_result
            .rows
            .first()
            .and_then(|row| int_field(row, "total"))
            .unwrap_or(0)
            .max(0) as u64;

        Ok(EntityTable {
            entity_type: entity_type.to_string(),
            key_columns: self.key_columns_for(entity_type).await,
            rows,
            total,
            offset: page.offset,
        })
    }

    /// Find entities by text.
    ///
    /// * `Keyword` — case-insensitive scan over string-typed properties
    ///   (schema-driven).
    /// * `Semantic` — embed the query and search the entity `_canonical`
    ///   vectors through the graph server (`libqlink`); requires an
    ///   embedder on the pipeline and a vector-indexed graph.
    /// * `Auto` — semantic when an embedder is configured, falling back
    ///   to keyword when the semantic channel errors or finds nothing.
    pub async fn search(&self, query: &str, opts: &SearchOptions) -> Result<SearchResult> {
        let limit = self.page_limit(opts.limit);
        match opts.mode {
            SearchMode::Keyword => self.keyword_search(query, opts, limit).await,
            SearchMode::Semantic => self.semantic_search(query, opts, limit).await,
            SearchMode::Auto => {
                if self.pipeline.embedder().is_none() {
                    return self.keyword_search(query, opts, limit).await;
                }
                match self.semantic_search(query, opts, limit).await {
                    Ok(found) if !found.hits.is_empty() => Ok(found),
                    Ok(_) => self.keyword_search(query, opts, limit).await,
                    Err(err) => {
                        tracing::warn!(
                            target: "linguagraph::explore",
                            error = %err,
                            "semantic search failed; falling back to keyword"
                        );
                        self.keyword_search(query, opts, limit).await
                    }
                }
            }
        }
    }

    /// Semantic channel: query vector → `libqlink.search` over the
    /// entity `_canonical` collection → hydrate the hit node ids.
    async fn semantic_search(
        &self,
        needle: &str,
        opts: &SearchOptions,
        limit: u32,
    ) -> Result<SearchResult> {
        let embedder = self.pipeline.embedder().ok_or_else(|| {
            ExploreError::SemanticSearchUnavailable(
                "the pipeline has no embedder — configure an embedding model or use keyword mode"
                    .to_string(),
            )
        })?;
        let vector = embedder.embed(needle).map_err(|e| {
            ExploreError::SemanticSearchUnavailable(format!("query embedding failed: {e}"))
        })?;

        let entity_types: Option<Vec<String>> = opts.entity_type.clone().map(|t| vec![t]);
        let cypher = crate::core::pipeline::build_traversal_search_cypher(
            &self.pipeline.canonical_entity_collection(),
            &[vector],
            "", // no chunk channel — entity search only
            None,
            limit,
            self.prefix(),
            entity_types.as_deref(),
        );
        let result = self.pipeline.execute(&cypher).await?;

        // Best score per node id, then rank descending.
        let mut scored: Vec<(i64, f64)> = Vec::new();
        for row in &result.rows {
            let Some(nid) = int_field(row, "nid") else {
                continue;
            };
            let score = float_field(row, "score").unwrap_or(0.0);
            match scored.iter_mut().find(|(id, _)| *id == nid) {
                Some((_, best)) => *best = best.max(score),
                None => scored.push((nid, score)),
            }
        }
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(limit as usize);
        if scored.is_empty() {
            return Ok(SearchResult {
                hits: Vec::new(),
                related_types: Vec::new(),
            });
        }

        let catalog = self.catalog();
        let nids: Vec<i64> = scored.iter().map(|(nid, _)| *nid).collect();
        let hydrated = self
            .pipeline
            .execute(&queries::nodes_by_nids(&nids, self.prefix())?)
            .await?;
        let mut by_nid: BTreeMap<i64, NodeView> = BTreeMap::new();
        for row in &hydrated.rows {
            if let (Some(nid), Some(node)) = (
                int_field(row, "nid"),
                self.node_view_from_row(row, catalog.as_deref()),
            ) {
                by_nid.insert(nid, node);
            }
        }
        let hits = scored
            .into_iter()
            .filter_map(|(nid, score)| {
                by_nid.remove(&nid).map(|node| SearchHit {
                    node,
                    score: Some(score),
                    channel: SearchChannel::Semantic,
                })
            })
            .collect();
        Ok(SearchResult {
            hits,
            related_types: Vec::new(),
        })
    }

    async fn keyword_search(
        &self,
        needle: &str,
        opts: &SearchOptions,
        limit: u32,
    ) -> Result<SearchResult> {
        let catalog = self.catalog();
        let string_props = self.searchable_string_props(opts.entity_type.as_deref()).await?;
        let query = queries::keyword_search(
            needle,
            &string_props,
            opts.entity_type.as_deref(),
            opts.exact,
            limit,
            self.prefix(),
        )?;
        let result = self.pipeline.execute(&query).await?;
        let hits = result
            .rows
            .iter()
            .filter_map(|row| self.node_view_from_row(row, catalog.as_deref()))
            .map(|node| SearchHit {
                node,
                score: None,
                channel: SearchChannel::Keyword,
            })
            .collect();
        Ok(SearchResult {
            hits,
            related_types: Vec::new(),
        })
    }

    /// String-typed property names to scan in a keyword search —
    /// `toString()` on list values is a Cypher runtime error, so the
    /// predicate is built strictly from string-typed schema fields.
    async fn searchable_string_props(&self, entity_type: Option<&str>) -> Result<Vec<String>> {
        let schema = self.pipeline.live_schema::<&str>(&[]).await?;
        let mut props: Vec<String> = vec!["id".to_string(), "name".to_string(), "title".to_string()];
        for node in &schema.nodes {
            if entity_type.is_some_and(|t| t != node.label) {
                continue;
            }
            for property in &node.properties {
                if property.ty == SchemaPropertyType::String
                    && !classify::is_system_property(&property.name)
                    && !props.contains(&property.name)
                {
                    props.push(property.name.clone());
                }
            }
        }
        Ok(props)
    }

    /// Suggested table columns for a type: id/name first, then the
    /// schema's string properties.
    async fn key_columns_for(&self, entity_type: &str) -> Vec<String> {
        let mut columns = vec!["id".to_string(), "name".to_string()];
        if let Ok(schema) = self.pipeline.live_schema::<&str>(&[]).await {
            if let Some(node) = schema.nodes.iter().find(|n| n.label == entity_type) {
                for property in &node.properties {
                    if !classify::is_system_property(&property.name)
                        && !columns.contains(&property.name)
                    {
                        columns.push(property.name.clone());
                    }
                    if columns.len() >= 6 {
                        break;
                    }
                }
            }
        }
        columns
    }

    /// One page of entities of a type *plus* the edges among them —
    /// the subgraph behind a type-filtered graph view or export.
    pub async fn subgraph_of_type(
        &self,
        entity_type: &str,
        page: &PageOptions,
    ) -> Result<Subgraph> {
        let table = self.entities_of_type(entity_type, page).await?;
        let truncated = (table.rows.len() as u64) < table.total;
        let ids: Vec<String> = table
            .rows
            .iter()
            .filter(|n| !n.ephemeral_handle)
            .map(|n| n.id.clone())
            .collect();
        let mut edges = Vec::new();
        if !ids.is_empty() {
            let edge_result = self
                .pipeline
                .execute(&queries::edges_among(
                    &ids,
                    self.limits.max_subgraph_edges,
                    self.prefix(),
                )?)
                .await?;
            for row in &edge_result.rows {
                let (Some(from), Some(to), Some(edge_type)) = (
                    id_string_field(row, "from_id"),
                    id_string_field(row, "to_id"),
                    string_field(row, "rel"),
                ) else {
                    continue;
                };
                let properties = json_object_field(row, "props");
                let confidence = properties.get("confidence").and_then(JsonValue::as_f64);
                let edge = EdgeView {
                    id: format!("{from}:{edge_type}:{to}"),
                    edge_type,
                    from,
                    to,
                    properties,
                    confidence,
                };
                if !edges.iter().any(|e: &EdgeView| e.id == edge.id) {
                    edges.push(edge);
                }
            }
        }
        Ok(Subgraph {
            nodes: table.rows,
            edges,
            truncated,
        })
    }

    // ── Timeline & export ───────────────────────────────────────────────

    /// Dated events extracted from a subgraph's `Datetime` properties,
    /// sorted chronologically. Pure — no database round trip.
    pub fn timeline(&self, subgraph: &Subgraph) -> Vec<TimelineEvent> {
        timeline::subgraph_timeline(subgraph)
    }

    /// Timeline over one page of an entity type.
    pub async fn timeline_for_type(
        &self,
        entity_type: &str,
        page: &PageOptions,
    ) -> Result<Vec<TimelineEvent>> {
        let table = self.entities_of_type(entity_type, page).await?;
        Ok(timeline::nodes_timeline(&table.rows))
    }

    /// Render a subgraph as [`crate::graph::GraphBuilder::from_json`]-
    /// compatible JSON. Pure — no database round trip.
    pub fn export(&self, subgraph: &Subgraph) -> ExportDoc {
        export::export_subgraph(subgraph)
    }

    // ── Row decoding ────────────────────────────────────────────────────

    /// Build a [`NodeView`] from a row shaped `nid, id, labels, props`.
    fn node_view_from_row(
        &self,
        row: &Row,
        catalog: Option<&OntologyCatalog>,
    ) -> Option<NodeView> {
        let labels = string_list_field(row, "labels");
        let props = json_object_field(row, "props");
        let (id, ephemeral_handle) = match id_string_field(row, "id") {
            Some(id) => (id, false),
            None => {
                let nid = int_field(row, "nid")?;
                (format!("{NID_HANDLE_PREFIX}{nid}"), true)
            }
        };
        let entity_type = classify::primary_label(&labels, self.prefix(), catalog);
        let name = classify::display_name(&props, &id);
        let confidence = classify::confidence(&props);
        let properties = classify::classify_properties(&entity_type, props, catalog);
        Some(NodeView {
            id,
            name,
            entity_type,
            labels,
            properties,
            confidence,
            ephemeral_handle,
        })
    }

    /// Fold raw `rel/dir/neighbor_labels/cnt` rows into per-type relation
    /// summaries (labels reduced to business types, counts merged).
    fn reduce_relation_summary(
        &self,
        result: &QueryResult,
        catalog: Option<&OntologyCatalog>,
    ) -> Vec<RelationSummary> {
        let mut merged: BTreeMap<(String, RelDirection, String), u64> = BTreeMap::new();
        for row in &result.rows {
            let Some(edge_type) = string_field(row, "rel") else {
                continue;
            };
            let direction = match string_field(row, "dir").as_deref() {
                Some("out") => RelDirection::Out,
                _ => RelDirection::In,
            };
            let labels = string_list_field(row, "neighbor_labels");
            let neighbor_type = classify::primary_label(&labels, self.prefix(), catalog);
            let count = int_field(row, "cnt").unwrap_or(0).max(0) as u64;
            *merged.entry((edge_type, direction, neighbor_type)).or_default() += count;
        }
        merged
            .into_iter()
            .map(|((edge_type, direction, neighbor_type), count)| RelationSummary {
                edge_type,
                direction,
                neighbor_type,
                count,
            })
            .collect()
    }
}

// ── Row field helpers ───────────────────────────────────────────────────

pub(crate) fn string_field(row: &Row, name: &str) -> Option<String> {
    match row.fields.get(name) {
        Some(DbValue::String(s)) => Some(s.clone()),
        Some(DbValue::Json(JsonValue::String(s))) => Some(s.clone()),
        _ => None,
    }
}

fn int_field(row: &Row, name: &str) -> Option<i64> {
    match row.fields.get(name) {
        Some(DbValue::Int(v)) => Some(*v),
        Some(DbValue::Float(v)) => Some(*v as i64),
        Some(DbValue::Json(JsonValue::Number(n))) => n.as_i64(),
        _ => None,
    }
}

fn float_field(row: &Row, name: &str) -> Option<f64> {
    match row.fields.get(name) {
        Some(DbValue::Float(v)) => Some(*v),
        Some(DbValue::Int(v)) => Some(*v as f64),
        Some(DbValue::Json(JsonValue::Number(n))) => n.as_f64(),
        _ => None,
    }
}

/// The Memgraph client flattens every cell to `Value::Json`, so a bool
/// arrives as `Json(Bool)` on the wire and as `Bool` from test doubles —
/// accept both.
fn bool_field(row: &Row, name: &str) -> Option<bool> {
    match row.fields.get(name) {
        Some(DbValue::Bool(v)) => Some(*v),
        Some(DbValue::Json(JsonValue::Bool(v))) => Some(*v),
        _ => None,
    }
}

/// Decode an id cell: `id` properties may be stored as strings or
/// integers — integers are stringified so the public handle stays one
/// type.
pub(crate) fn id_string_field(row: &Row, name: &str) -> Option<String> {
    match row.fields.get(name) {
        Some(DbValue::String(s)) => Some(s.clone()),
        Some(DbValue::Int(v)) => Some(v.to_string()),
        Some(DbValue::Json(JsonValue::String(s))) => Some(s.clone()),
        Some(DbValue::Json(JsonValue::Number(n))) => Some(n.to_string()),
        _ => None,
    }
}

fn string_list_field(row: &Row, name: &str) -> Vec<String> {
    match row.fields.get(name) {
        Some(DbValue::Json(JsonValue::Array(items))) => items
            .iter()
            .filter_map(|v| v.as_str().map(str::to_string))
            .collect(),
        Some(DbValue::String(s)) => vec![s.clone()],
        _ => Vec::new(),
    }
}

pub(crate) fn json_object_field(row: &Row, name: &str) -> BTreeMap<String, JsonValue> {
    match row.fields.get(name) {
        Some(DbValue::Json(JsonValue::Object(map))) => {
            map.iter().map(|(k, v)| (k.clone(), v.clone())).collect()
        }
        _ => BTreeMap::new(),
    }
}

/// Decode the `sources` column (a collected list of `{id, name}` maps;
/// `OPTIONAL MATCH` misses produce all-null entries that are dropped).
fn parse_source_refs(value: Option<&DbValue>) -> Vec<SourceRef> {
    let Some(DbValue::Json(JsonValue::Array(items))) = value else {
        return Vec::new();
    };
    let mut refs: Vec<SourceRef> = items
        .iter()
        .filter_map(|item| {
            let map = item.as_object()?;
            let id = map.get("id").and_then(|v| v.as_str()).map(str::to_string);
            let name = map.get("name").and_then(|v| v.as_str()).map(str::to_string);
            (id.is_some() || name.is_some()).then_some(SourceRef { id, name })
        })
        .collect();
    refs.sort();
    refs.dedup();
    refs
}
