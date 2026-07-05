//! Programmatic interface — the API/CLI-ready facade.
//!
//! [`GraphService`] is the single entry point a transport wraps. It turns
//! the low-level [`Pipeline`] primitives into the handful of operations a
//! knowledge-graph front-end needs — ask a question, inspect an entity or
//! a relationship, list the schema — and returns plain `serde` DTOs
//! (see [`dto`]) that a REST handler, a CLI subcommand, or a gRPC service
//! can serialize directly. Nothing here depends on a web framework; the
//! HTTP (or CLI) plumbing stays with the caller.
//!
//! ```no_run
//! # async fn demo(cfg: &linguagraph::config::Config) -> linguagraph::Result<()> {
//! use linguagraph::service::{AskRequest, GraphService};
//!
//! let svc = GraphService::from_config(cfg).await?;
//! let view = svc.ask(AskRequest { question: "Who owns Acme?".into(), ..Default::default() }).await?;
//! println!("{} nodes, {} edges", view.nodes.len(), view.edges.len());
//! # Ok(())
//! # }
//! ```

use std::sync::Arc;

use crate::config::Config;
use crate::core::{factory, nl, Pipeline};
use crate::dsl::DslQuery;
use crate::error::{Error, Result};
use crate::llm::LlmClient;
use crate::prompt::{GraphSchema, PromptOptions};

pub mod convert;
pub mod dto;

pub use dto::*;

/// How many times `ask` re-prompts the model to repair invalid DSL.
const DEFAULT_MAX_REPAIRS: usize = 2;

/// High-level, transport-agnostic facade over a [`Pipeline`].
///
/// Cheap to clone (the pipeline is a bundle of `Arc`s) and safe to share
/// across concurrent requests behind an `Arc`.
#[derive(Clone)]
pub struct GraphService {
    pipeline: Pipeline,
    /// Optional LLM used by [`Self::ask`]. `None` (e.g. the `openai`
    /// feature is off) makes `ask` fail fast while `run_dsl` still works.
    llm: Option<Arc<dyn LlmClient>>,
    /// Node-label substring filter applied to schema introspection.
    schema_filter: Vec<String>,
}

impl std::fmt::Debug for GraphService {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GraphService")
            .field("pipeline", &self.pipeline)
            .field("has_llm", &self.llm.is_some())
            .field("schema_filter", &self.schema_filter)
            .finish()
    }
}

impl GraphService {
    /// Build a service from configuration: connect Memgraph, register
    /// handlers, load the ontology catalog, and (when the `openai`
    /// feature is on) attach an LLM for the natural-language path.
    pub async fn from_config(cfg: &Config) -> Result<Self> {
        let pipeline = factory::build_query_pipeline(cfg, None, None).await?;
        // An LLM is optional: without it, `ask` errors but the DSL and
        // inspection paths still work.
        let llm = factory::build_llm_client(cfg).ok();
        Ok(Self {
            pipeline,
            llm,
            schema_filter: Vec::new(),
        })
    }

    /// Construct directly from a [`Pipeline`] and an optional LLM. Used by
    /// tests and by embedders that build their own pipeline.
    pub fn new(pipeline: Pipeline, llm: Option<Arc<dyn LlmClient>>) -> Self {
        Self {
            pipeline,
            llm,
            schema_filter: Vec::new(),
        }
    }

    /// Return a copy of this service scoped to a tenant / dataset, without
    /// rebuilding the pipeline.
    pub fn scoped(&self, prefix_label: Option<String>, prefix_index: Option<String>) -> Self {
        Self {
            pipeline: self
                .pipeline
                .clone()
                .with_prefix_label(prefix_label)
                .with_prefix_index(prefix_index),
            llm: self.llm.clone(),
            schema_filter: self.schema_filter.clone(),
        }
    }

    /// Answer a natural-language question as an entity/relationship graph.
    ///
    /// Runs NL → DSL (via the configured LLM) → graph-shaped query, and
    /// returns the nodes, edges, and the generated Cypher. An empty result
    /// is a success (empty `nodes` / `edges`), not an error.
    pub async fn ask(&self, req: AskRequest) -> Result<GraphView> {
        let llm = self.llm.as_ref().ok_or_else(|| {
            Error::Nl(
                "no LLM configured; the `openai` feature is required for natural-language queries"
                    .to_string(),
            )
        })?;

        let schema = self.pipeline.live_schema(&self.schema_filter).await?;
        let opts = PromptOptions::default();
        let mut dsl =
            nl::generate_dsl(llm.as_ref(), &req.question, &schema, &opts, DEFAULT_MAX_REPAIRS)
                .await?;

        // Per-request overrides win over whatever the model emitted.
        if req.prefix_label.is_some() {
            dsl.prefix_label = req.prefix_label.clone();
        }
        if req.prefix_index.is_some() {
            dsl.prefix_index = req.prefix_index.clone();
        }
        if req.limit.is_some() {
            dsl.limit = req.limit;
        }

        self.run_dsl(dsl).await
    }

    /// Run a pre-built [`DslQuery`] with the graph-shaped projection — the
    /// LLM-free path for callers that already have a DSL document.
    pub async fn run_dsl(&self, dsl: DslQuery) -> Result<GraphView> {
        let run = self.pipeline.run_graph(dsl).await?;
        let (nodes, edges) = convert::query_result_to_graph(&run.result);
        Ok(GraphView {
            nodes,
            edges,
            cypher: run.cypher.text,
        })
    }

    /// The live graph schema, reshaped into entity-type and relation-type
    /// lists for a UI's filter chips and legend.
    pub async fn schema(&self) -> Result<SchemaView> {
        let schema = self.pipeline.live_schema(&self.schema_filter).await?;
        Ok(schema_view(&schema))
    }

    /// Fetch a single entity (properties, sources, relationships) by its
    /// database internal id. `None` when the id is unknown.
    pub async fn entity(&self, id: i64) -> Result<Option<EntityDetail>> {
        let result = self.pipeline.entity_detail(id).await?;
        Ok(convert::to_entity_detail(&result))
    }

    /// Fetch a single relationship (both endpoints, properties) by its
    /// database internal id. `None` when the id is unknown.
    pub async fn relation(&self, id: i64) -> Result<Option<RelationDetail>> {
        let result = self.pipeline.relation_detail(id).await?;
        Ok(convert::to_relation_detail(&result))
    }
}

fn schema_view(schema: &GraphSchema) -> SchemaView {
    let entity_types = schema
        .nodes
        .iter()
        .map(|n| EntityTypeInfo {
            label: n.label.clone(),
            description: n.description.clone(),
            properties: n
                .properties
                .iter()
                .map(|p| PropertyInfo {
                    name: p.name.clone(),
                    ty: format!("{:?}", p.ty).to_lowercase(),
                    description: p.description.clone(),
                })
                .collect(),
        })
        .collect();
    let relation_types = schema
        .relationships
        .iter()
        .map(|r| RelationTypeInfo {
            label: r.label.clone(),
            description: r.description.clone(),
            from: r.from.clone(),
            to: r.to.clone(),
        })
        .collect();
    SchemaView {
        entity_types,
        relation_types,
    }
}
