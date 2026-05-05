//! End-to-end orchestration: DSL/mapping → AST → Cypher → DB.

use std::sync::Arc;

use serde_json::Value;

use crate::ast::{from_dsl, query::ReadQuery, query::InsertQuery, query::Literal};
use crate::builder::{self, CypherQuery};
use crate::config::Config;
use crate::db::{GraphClient, QueryResult};
use crate::dsl::DslQuery;
use crate::embeddings::SharedEmbedder;
use crate::error::Result;
use crate::ingest::{self, IngestError, PlannerOptions};
use crate::mapper::{self, Mapping};
use crate::metadata::{self, MetadataStore, PropertyMetadata};
use crate::types::{SharedRegistry, SideEffect, SideEffectQueue, TypeRegistry};

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
    /// `"type"`. Wrapped in an `RwLock` so a successful [`Self::ingest`]
    /// can refresh it without taking `&mut self` (the pipeline is
    /// passed around as `&self` everywhere).
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
            registry: Arc::new(TypeRegistry::empty()),
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

    /// Attach a metadata store. When set, every [`Self::ingest`] call
    /// merges the mapping's property descriptions into it.
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
        self.metadata.read().expect("metadata lock poisoned").clone()
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

    // ── Insert path ─────────────────────────────────────────────────────────

    /// Compile a `(data, mapping)` pair into one Cypher batch per
    /// node/relation group. Pure; no I/O. Drops side effects on the
    /// floor — use [`Self::lower_insert_with_effects`] when you need
    /// them.
    pub fn compile_insert(&self, mapping: &Mapping, data: &Value) -> Result<Vec<CypherQuery>> {
        let (insert, _effects) = self.lower_insert_with_effects(mapping, data)?;
        Ok(builder::build_insert(&insert)?)
    }

    /// Lower a `(data, mapping)` pair into the typed [`InsertQuery`] AST,
    /// dropping any queued side effects.
    pub fn lower_insert(&self, mapping: &Mapping, data: &Value) -> Result<InsertQuery> {
        Ok(self.lower_insert_with_effects(mapping, data)?.0)
    }

    /// Lower a `(data, mapping)` pair, returning both the [`InsertQuery`]
    /// and the queue of side effects that must run after the Memgraph
    /// batches succeed.
    pub fn lower_insert_with_effects(
        &self,
        mapping: &Mapping,
        data: &Value,
    ) -> Result<(InsertQuery, SideEffectQueue)> {
        let extracted = mapper::extract(mapping, data)?;
        let opts = PlannerOptions { max_batch_size: self.ingest_batch_size };
        let mut effects = SideEffectQueue::new();
        let insert = ingest::plan_with_registry(
            mapping,
            extracted,
            opts,
            &self.registry,
            &mut effects,
        )?;
        Ok((insert, effects))
    }

    /// Compile and execute the full ingestion pipeline.
    ///
    /// Each batch is executed sequentially so a partial failure leaves the
    /// graph in a well-defined intermediate state (already-MERGE'd batches
    /// stay; the failing one rolls back its own work). Every node MERGE
    /// runs before any relationship MERGE, so the planner's ordering
    /// guarantees that when relations execute, both endpoints exist.
    pub async fn ingest(&self, mapping: &Mapping, data: &Value) -> Result<IngestSummary> {
        let (insert, effects) = self.lower_insert_with_effects(mapping, data)?;
        let node_rows: usize = insert
            .node_batches
            .iter()
            .map(|b| b.rows.len())
            .sum();
        let relation_rows: usize = insert
            .relation_batches
            .iter()
            .map(|b| b.rows.len())
            .sum();

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

        // Refresh property metadata from the mapping. This runs after
        // the graph writes succeed so a failed ingest doesn't leave the
        // cache describing data that never landed.
        //
        // The persisted snapshot is also re-mirrored into the
        // pipeline's in-memory cache so subsequent queries auto-resolve
        // typed filters against the freshest mapping without an extra
        // `load_metadata()` call.
        let incoming = metadata::collect_from_mapping(mapping);
        if !incoming.is_empty() {
            let merged = if let Some(store) = &self.metadata_store {
                store.update(&incoming).await?
            } else {
                // No persistent store: still keep an in-memory snapshot
                // by merging into whatever was previously loaded so
                // subsequent queries see typed-property hints.
                let mut current = self.metadata().map(|a| (*a).clone()).unwrap_or_default();
                current.merge(&incoming);
                current
            };
            *self.metadata.write().expect("metadata lock poisoned") =
                Some(Arc::new(merged));
        }

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
    async fn drain_side_effects(
        &self,
        mut effects: SideEffectQueue,
    ) -> Result<(usize, usize)> {
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

/// Marker re-export so the embedder trait can be referenced via
/// `pipeline::Embedder` in the README/tests without exposing the whole
/// `embeddings` path.
pub use crate::embeddings::Embedder as _Embedder;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{DatabaseConfig, LlmConfig, MetadataConfig, QueryConfig};
    use crate::db::MockClient;
    use crate::metadata::{MetadataError, PropertyMetadata};
    use async_trait::async_trait;
    use serde_json::json;
    use std::sync::Mutex;

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
            types: Default::default(),
        }
    }

    #[derive(Debug, Default)]
    struct InMemoryStore {
        inner: Mutex<PropertyMetadata>,
    }

    #[async_trait]
    impl MetadataStore for InMemoryStore {
        async fn load(&self) -> std::result::Result<PropertyMetadata, MetadataError> {
            Ok(self.inner.lock().unwrap().clone())
        }
        async fn save(
            &self,
            meta: &PropertyMetadata,
        ) -> std::result::Result<(), MetadataError> {
            *self.inner.lock().unwrap() = meta.clone();
            Ok(())
        }
    }

    #[tokio::test]
    async fn ingest_updates_metadata_store_with_descriptions() {
        let store = Arc::new(InMemoryStore::default());
        let pipeline = Pipeline::new(Arc::new(MockClient::new()), &cfg())
            .with_metadata_store(store.clone());

        let mapping: Mapping = serde_json::from_value(json!({
            "entities": [{
                "type": "Camera",
                "source_path": "$.cameras[*]",
                "primary_key": "$.cameras[*].id",
                "description": "An IP camera",
                "properties": [
                    {"name": "id", "source_path": "$.cameras[*].id"},
                    {
                        "name": "state",
                        "source_path": "$.cameras[*].state",
                        "description": "active or inactive"
                    }
                ]
            }]
        }))
        .unwrap();
        let data = json!({"cameras": [{"id": "c1", "state": "active"}]});

        pipeline.ingest(&mapping, &data).await.unwrap();

        let stored = store.inner.lock().unwrap().clone();
        assert_eq!(stored.get("Camera"), Some("An IP camera"));
        assert_eq!(stored.get("Camera.state"), Some("active or inactive"));
    }

    #[tokio::test]
    async fn ingest_without_store_does_not_panic() {
        let pipeline = Pipeline::new(Arc::new(MockClient::new()), &cfg());
        let mapping: Mapping = serde_json::from_value(json!({
            "entities": [{
                "type": "X",
                "source_path": "$.x[*]",
                "primary_key": "$.x[*].id",
                "description": "ignored without a store"
            }]
        }))
        .unwrap();
        pipeline
            .ingest(&mapping, &json!({"x": [{"id": "a"}]}))
            .await
            .unwrap();
    }
}
