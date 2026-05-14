//! End-to-end orchestration: DSL/mapping → AST → Cypher → DB.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use crate::ast::{from_dsl, query::InsertQuery, query::Literal, query::ReadQuery};
use crate::builder::{self, CypherQuery};
use crate::config::Config;
use crate::db::{GraphClient, QueryResult, Row, Value as DbValue};
use crate::dsl::{DslQuery, TraversalQuery};
use crate::embeddings::SharedEmbedder;
use crate::error::Result;
use crate::graph::Graph;
use crate::ingest::{self, IngestError, PlannerOptions};
use crate::metadata::{MetadataStore, PropertyMetadata};
use crate::types::{handlers, SharedRegistry, SideEffect, SideEffectQueue};

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
    metadata_store: Option<Arc<dyn MetadataStore>>,
    /// Registry of [`crate::types::TypeHandler`] instances. Defaults to
    /// an empty registry so plain (untyped) DSL queries don't need any
    /// configuration; the CLI / library callers register handlers via
    /// [`Self::with_registry`].
    registry: SharedRegistry,
    /// Embedder used by side-effect drainage (e.g. the SemanticText
    /// pipeline). Optional: when not configured, queries that reference
    /// types requiring an embedder fail at lowering time, not at ingest.
    embedder: Option<SharedEmbedder>,
    /// In-memory snapshot of [`PropertyMetadata`] consulted by
    /// [`Self::lower`] to auto-resolve filter types when the DSL omits
    /// `"type"`.
    metadata: Arc<RwLock<Option<Arc<PropertyMetadata>>>>,
}

impl std::fmt::Debug for Pipeline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Pipeline")
            .field("max_depth", &self.max_depth)
            .field("default_limit", &self.default_limit)
            .field("ingest_batch_size", &self.ingest_batch_size)
            .field("metadata_store", &self.metadata_store.is_some())
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
}

impl Pipeline {
    pub fn new(client: Arc<dyn GraphClient>, config: &Config) -> Self {
        Self {
            client,
            max_depth: config.query.max_traversal_depth,
            default_limit: config.query.default_limit,
            ingest_batch_size: 1000,
            metadata_store: None,
            // Default registry contains the built-in scalar parsers.
            // Graph `Text` properties route through `SemanticText`, so
            // callers ingesting text-rich graphs should register that
            // handler via [`Self::with_registry`].
            registry: Arc::new(handlers::core_registry()),
            embedder: None,
            metadata: Arc::new(RwLock::new(None)),
        }
    }

    /// Override the ingestion batch size. Useful for tests and for callers
    /// who know their downstream system has stricter parameter limits.
    pub fn with_ingest_batch_size(mut self, n: usize) -> Self {
        self.ingest_batch_size = n;
        self
    }

    /// Attach a metadata store for query-time type metadata.
    pub fn with_metadata_store(mut self, store: Arc<dyn MetadataStore>) -> Self {
        self.metadata_store = Some(store);
        self
    }

    pub fn metadata_store(&self) -> Option<&Arc<dyn MetadataStore>> {
        self.metadata_store.as_ref()
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

    /// Pre-load the metadata snapshot used to auto-resolve filter
    /// types. Pass a snapshot you already have (typically obtained via
    /// `MetadataStore::load` at startup); pairs cleanly with
    /// [`Self::with_metadata_store`] for the write-back side.
    pub fn with_metadata(self, meta: Arc<PropertyMetadata>) -> Self {
        *self.metadata.write().expect("metadata lock poisoned") = Some(meta);
        self
    }

    /// Eagerly load the metadata snapshot from the configured store.
    /// No-op when no store is set. Intended to be called at startup so
    /// the first query benefits from auto-typed filters.
    pub async fn load_metadata(&self) -> Result<()> {
        if let Some(store) = &self.metadata_store {
            let m = store.load().await?;
            *self.metadata.write().expect("metadata lock poisoned") = Some(Arc::new(m));
        }
        Ok(())
    }

    /// Snapshot of the metadata currently informing query lowering.
    pub fn metadata(&self) -> Option<Arc<PropertyMetadata>> {
        self.metadata
            .read()
            .expect("metadata lock poisoned")
            .clone()
    }

    // ── Read path ───────────────────────────────────────────────────────────

    /// Lower a DSL document to the typed AST. Pure; no I/O.
    ///
    /// When a [`PropertyMetadata`] snapshot is loaded, filters that
    /// omit `"type"` are auto-resolved against it: if the property's
    /// type is `SemanticText`, the SemanticText handler is selected
    /// without any DSL change.
    pub fn lower(&self, dsl: DslQuery) -> Result<ReadQuery> {
        let meta_snapshot = self.metadata();
        let mut q = from_dsl::lower_full(
            dsl,
            self.max_depth,
            &self.registry,
            meta_snapshot.as_deref(),
        )?;
        if q.limit.is_none() {
            q.limit = Some(self.default_limit);
        }
        Ok(q)
    }

    /// Compile a DSL document all the way to a parameterized Cypher query.
    pub fn compile(&self, dsl: DslQuery) -> Result<CypherQuery> {
        let ast = self.lower(dsl)?;
        Ok(builder::build_read_with(&ast, &self.registry)?)
    }

    /// Compile and execute against the configured graph client.
    pub async fn run(&self, dsl: DslQuery) -> Result<QueryResult> {
        let cypher = self.compile(dsl)?;
        Ok(self.client.execute(&cypher).await?)
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
            println!("{}", cypher.text);
            let result = self.client.execute(&cypher).await?;
            println!("{:?}", result);
            merged.extend(result);
        }

        let cypher = self.compile_traversal(traversal)?;
        println!("{}", cypher.text);
        let result = self.client.execute(&cypher).await?;
        merged.extend(result);

        Ok(merged.finish(limit))
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
        let insert = ingest::plan_graph_with_registry(graph, opts, &self.registry, &mut effects)?;
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
        let (insert, effects) = self.lower_insert_with_effects(graph)?;
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
        })
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
            let cypher = build_qlink_insert_batch(&group)?;
            let _ = self.client.execute(&cypher).await?;
            batches_run += 1;
            rows_inserted += group.len();
        }
        let _ = embedder.dim(); // assert the embedder was usable
        Ok((batches_run, rows_inserted))
    }
}

/// Render an `UNWIND $rows AS row | MATCH ... CALL libqlink.insert_labeled
/// ...` Cypher batch for one homogeneous group of side effects.
///
/// All effects in `group` must share the same Cypher `label`, the same
/// `key_field`, the same `collection`, and the same `payload_label`
/// (the caller — `drain_side_effects` — keys the bucket by exactly
/// these). The MATCH pattern is therefore consistent across rows; the
/// only thing that varies per row is `key`/`vec` inside the row
/// payload.
///
/// When the bucket has a `payload_label`, we use
/// `libqlink.insert_labeled` so each vector lands in Qdrant tagged
/// with the originating Cypher node label — that's what
/// `libqlink.search_reranked` filters by at query time. When the
/// bucket has no label we fall back to plain `libqlink.insert` so
/// future handlers that don't care about labels still work.
fn build_qlink_insert_batch(group: &[(SideEffect, Vec<f32>)]) -> Result<CypherQuery> {
    use std::collections::BTreeMap;
    debug_assert!(!group.is_empty(), "callers must not pass an empty group");

    // All rows in `group` share these — see `drain_side_effects`.
    let (collection, payload_label, label, key_field) = match &group[0].0 {
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

    if !is_valid_ident(&label) {
        return Err(crate::error::Error::Ingest(IngestError::Type(format!(
            "invalid label '{label}' in side effect"
        ))));
    }
    if !is_valid_ident(&key_field) {
        return Err(crate::error::Error::Ingest(IngestError::Type(format!(
            "invalid key field '{key_field}' in side effect"
        ))));
    }

    // Build the row payload. Each row is `{key: <pk>, vec: <embedding>}`.
    let mut rows: Vec<Literal> = Vec::with_capacity(group.len());
    for (eff, vec) in group {
        let SideEffect::EmbedAndStore { key_value, .. } = eff;
        let mut row: BTreeMap<String, Literal> = BTreeMap::new();
        row.insert("key".to_string(), key_value.clone());
        row.insert(
            "vec".to_string(),
            Literal::List(vec.iter().map(|f| Literal::Float(*f as f64)).collect()),
        );
        rows.push(Literal::Object(row));
    }

    let mut params: BTreeMap<String, Literal> = BTreeMap::new();
    params.insert("coll".to_string(), Literal::String(collection));
    params.insert("rows".to_string(), Literal::List(rows));

    let text = if let Some(plabel) = payload_label {
        params.insert("label".to_string(), Literal::String(plabel));
        format!(
            "UNWIND $rows AS row\n\
             MATCH (n:{label} {{{key_field}: row.key}})\n\
             CALL libqlink.insert_labeled($coll, id(n), row.vec, $label) YIELD success\n\
             RETURN count(success) AS inserted",
        )
    } else {
        format!(
            "UNWIND $rows AS row\n\
             MATCH (n:{label} {{{key_field}: row.key}})\n\
             CALL libqlink.insert($coll, id(n), row.vec) YIELD success\n\
             RETURN count(success) AS inserted",
        )
    };
    Ok(CypherQuery::new(text, params))
}

fn is_valid_ident(s: &str) -> bool {
    let mut chars = s.chars();
    let first = chars.next();
    matches!(first, Some(c) if c.is_ascii_alphabetic() || c == '_')
        && chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
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

/// Marker re-export so the embedder trait can be referenced via
/// `pipeline::Embedder` in the README/tests without exposing the whole
/// `embeddings` path.
pub use crate::embeddings::Embedder as _Embedder;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{
        DatabaseConfig, GraphSpecificationConfig, LlmConfig, MetadataConfig, QueryConfig,
    };
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
            metadata: MetadataConfig::default(),
            graph_specification: GraphSpecificationConfig::default(),
            types: Default::default(),
        }
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
        // Mentions hop: chunk -[:MENTIONS]-> (e) — no label on e.
        assert!(
            cypher.text.contains("[m:MENTIONS]->(e)"),
            "expected label-less entity target; got: {}",
            cypher.text
        );
        // part_of hop: chunk -[:part_of]-> (s:Source).
        assert!(
            cypher.text.contains("[po:part_of]->(s:Source)"),
            "expected (s:Source) part_of hop; got: {}",
            cypher.text
        );
        // Semantic search routes through the labeled qlink search.
        assert!(
            cypher.text.contains("libqlink.search_labeled"),
            "expected libqlink.search_labeled; got: {}",
            cypher.text
        );
        assert!(
            cypher
                .text
                .contains("WITH c, e, s, c__score_0\nOPTIONAL MATCH"),
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
            result.columns,
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
}
