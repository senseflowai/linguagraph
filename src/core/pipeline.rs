//! End-to-end orchestration: DSL/mapping → AST → Cypher → DB.

use std::sync::Arc;

use serde_json::Value;

use crate::ast::{from_dsl, query::ReadQuery, query::InsertQuery};
use crate::builder::{self, CypherQuery};
use crate::config::Config;
use crate::db::{GraphClient, QueryResult};
use crate::dsl::DslQuery;
use crate::error::Result;
use crate::ingest::{self, PlannerOptions};
use crate::mapper::{self, Mapping};
use crate::metadata::{self, MetadataStore};

/// High-level entrypoint used by the CLI and library consumers.
///
/// The pipeline is cheap to clone — its only state is an `Arc<dyn GraphClient>`
/// and a snapshot of the relevant config knobs.
#[derive(Clone)]
pub struct Pipeline {
    client: Arc<dyn GraphClient>,
    max_depth: u32,
    default_limit: u32,
    ingest_batch_size: usize,
    metadata_store: Option<Arc<dyn MetadataStore>>,
}

impl std::fmt::Debug for Pipeline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Pipeline")
            .field("max_depth", &self.max_depth)
            .field("default_limit", &self.default_limit)
            .field("ingest_batch_size", &self.ingest_batch_size)
            .field("metadata_store", &self.metadata_store.is_some())
            .finish_non_exhaustive()
    }
}

/// Summary returned by [`Pipeline::ingest`].
#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct IngestSummary {
    pub batches_executed: usize,
    pub node_rows: usize,
    pub relation_rows: usize,
}

impl Pipeline {
    pub fn new(client: Arc<dyn GraphClient>, config: &Config) -> Self {
        Self {
            client,
            max_depth: config.query.max_traversal_depth,
            default_limit: config.query.default_limit,
            ingest_batch_size: 1000,
            metadata_store: None,
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

    // ── Read path ───────────────────────────────────────────────────────────

    /// Lower a DSL document to the typed AST. Pure; no I/O.
    pub fn lower(&self, dsl: DslQuery) -> Result<ReadQuery> {
        let mut q = from_dsl::lower(dsl, self.max_depth)?;
        if q.limit.is_none() {
            q.limit = Some(self.default_limit);
        }
        Ok(q)
    }

    /// Compile a DSL document all the way to a parameterized Cypher query.
    pub fn compile(&self, dsl: DslQuery) -> Result<CypherQuery> {
        let ast = self.lower(dsl)?;
        Ok(builder::build_read(&ast)?)
    }

    /// Compile and execute against the configured graph client.
    pub async fn run(&self, dsl: DslQuery) -> Result<QueryResult> {
        let cypher = self.compile(dsl)?;
        Ok(self.client.execute(&cypher).await?)
    }

    // ── Insert path ─────────────────────────────────────────────────────────

    /// Compile a `(data, mapping)` pair into one Cypher batch per
    /// node/relation group. Pure; no I/O.
    pub fn compile_insert(&self, mapping: &Mapping, data: &Value) -> Result<Vec<CypherQuery>> {
        let extracted = mapper::extract(mapping, data)?;
        let opts = PlannerOptions { max_batch_size: self.ingest_batch_size };
        let insert = ingest::plan_with_options(mapping, extracted, opts)?;
        Ok(builder::build_insert(&insert)?)
    }

    /// Lower a `(data, mapping)` pair into the typed [`InsertQuery`] AST.
    pub fn lower_insert(&self, mapping: &Mapping, data: &Value) -> Result<InsertQuery> {
        let extracted = mapper::extract(mapping, data)?;
        let opts = PlannerOptions { max_batch_size: self.ingest_batch_size };
        Ok(ingest::plan_with_options(mapping, extracted, opts)?)
    }

    /// Compile and execute the full ingestion pipeline.
    ///
    /// Each batch is executed sequentially so a partial failure leaves the
    /// graph in a well-defined intermediate state (already-MERGE'd batches
    /// stay; the failing one rolls back its own work). Every node MERGE
    /// runs before any relationship MERGE, so the planner's ordering
    /// guarantees that when relations execute, both endpoints exist.
    pub async fn ingest(&self, mapping: &Mapping, data: &Value) -> Result<IngestSummary> {
        let insert = self.lower_insert(mapping, data)?;
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

        // Refresh property metadata from the mapping. This runs after the
        // graph writes succeed so a failed ingest doesn't leave the cache
        // describing data that never landed.
        if let Some(store) = &self.metadata_store {
            let incoming = metadata::collect_from_mapping(mapping);
            if !incoming.is_empty() {
                store.update(&incoming).await?;
            }
        }

        Ok(IngestSummary {
            batches_executed: total,
            node_rows,
            relation_rows,
        })
    }
}

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
