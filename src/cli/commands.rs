//! Clap-based CLI. Subcommands map 1:1 to pipeline stages so users can stop
//! at any layer for inspection.

use std::collections::BTreeSet;
use std::path::PathBuf;
use std::sync::Arc;

use crate::config::{self, Config};
use crate::core::Pipeline;
use crate::db::{introspect, Column, GraphClient, MemgraphClient, QueryResult, Value};
use crate::dsl;
use crate::embeddings::{self, SharedEmbedder};
use crate::error::Result;
use crate::graph::{
    GraphBuilder, JsonFileOntologyCatalogStorage, OntologyCatalogStorage,
    DEFAULT_ONTOLOGY_CATALOG_CACHE_PATH,
};
use crate::prompt::{self, GraphSchema, PromptOptions};
use crate::types::{self, SharedRegistry};
use clap::{Parser, Subcommand, ValueEnum};
use tabled::{builder::Builder, settings::Style};
use tokio::fs;

/// Output format for the `schema` subcommand.
#[derive(Debug, Clone, Copy, ValueEnum, Default)]
pub enum SchemaFormat {
    /// Pretty-printed JSON (machine-readable).
    #[default]
    Json,
    /// LLM-ready system prompt rendered from the schema.
    Prompt,
}

#[derive(Parser, Debug)]
#[command(
    name = "linguagraph",
    version,
    about = "Natural-language → graph-query pipeline backed by Memgraph"
)]
pub struct Cli {
    /// Path to the TOML config file.
    #[arg(long, short = 'c', global = true, default_value = "config.toml")]
    pub config: PathBuf,

    #[command(subcommand)]
    pub command: Command,
}

#[derive(Subcommand, Debug)]
pub enum Command {
    /// Validate a DSL JSON file and print the lowered AST.
    Dsl {
        /// Path to the DSL JSON file.
        path: PathBuf,
    },
    /// Compile a DSL JSON file into Cypher and print it with parameters.
    Cypher {
        path: PathBuf,
        /// Optional Cypher label appended to every node pattern so the
        /// query only matches entities ingested under the same
        /// `prefix_label`.
        #[arg(long)]
        prefix_label: Option<String>,
        /// Optional prefix folded into embedding-index / Qdrant
        /// collection names. Must match the ingest-side `prefix_index`,
        /// otherwise typed filters hit an empty collection.
        #[arg(long)]
        prefix_index: Option<String>,
    },
    /// Compile and execute a DSL file against the configured database.
    Run {
        path: PathBuf,
        /// Optional Cypher label appended to every node pattern so the
        /// query only matches entities ingested under the same
        /// `prefix_label`.
        #[arg(long)]
        prefix_label: Option<String>,
        /// Optional prefix folded into embedding-index / Qdrant
        /// collection names. Must match the ingest-side `prefix_index`.
        #[arg(long)]
        prefix_index: Option<String>,
    },
    /// Execute a doc-graph traversal-query JSON file against the configured database.
    ///
    /// The JSON shape is:
    ///
    /// ```json
    /// {
    ///   "entities": ["Article 365", "Code"],
    ///   "goal":     "Article 365",
    ///   "query":    "Article 365",
    ///   "prefix_label": "Entity_ws_1",
    ///   "prefix_index": "ws_1",
    ///   "limit": 30,
    ///   "entity_types": ["Person"]
    /// }
    /// ```
    ///
    /// Runs a two-channel vector search (entities in `_canonical`,
    /// chunks in `text`), follows `mentions` from matched entities
    /// to their chunks, deduplicates, aggregates per-chunk
    /// `total_score`, sorts, and optionally reranks.
    Traversal {
        /// Path to the traversal-query JSON file.
        path: PathBuf,
        /// Optional Cypher label appended to every node pattern so the
        /// traversal only matches entities ingested under the same
        /// `prefix_label`.
        #[arg(long)]
        prefix_label: Option<String>,
        /// Optional prefix folded into embedding-index / Qdrant
        /// collection names. Must match the ingest-side `prefix_index`.
        #[arg(long)]
        prefix_index: Option<String>,
    },
    /// Print a schema-aware system prompt for an LLM.
    Prompt {
        /// Natural-language query used to select relevant graph types.
        query: Option<String>,
        /// Path to a schema JSON file. If omitted, the live database is queried.
        #[arg(long)]
        schema: Option<PathBuf>,
        /// Skip the worked examples in the output.
        #[arg(long)]
        no_examples: bool,
        /// Skip annotating the prompt with the cached graph specification.
        #[arg(long = "no-specification", alias = "no-metadata")]
        no_specification: bool,
    },
    /// Fetch the live graph schema and print it (JSON by default).
    ///
    /// Use `--format prompt` to render the schema as a ready-to-use
    /// system prompt for an LLM.
    Schema {
        /// Maximum nodes/relationships sampled per type for property
        /// inference. Higher = more accurate, slower.
        #[arg(long, default_value_t = 100)]
        sample_size: u64,

        /// Output format.
        #[arg(long, value_enum, default_value_t = SchemaFormat::Json)]
        format: SchemaFormat,

        /// Write to this path instead of stdout.
        #[arg(long, short = 'o')]
        output: Option<PathBuf>,

        /// When `--format prompt`, omit the worked examples block.
        #[arg(long)]
        no_examples: bool,
    },
    /// Ingest a GraphBuilder JSON file directly into the configured database.
    ///
    /// The file must contain `entities` and `relations`/`relationships`
    /// arrays in the compact graph JSON shape accepted by
    /// `GraphBuilder::from_json`.
    IngestGraph {
        /// Path to the graph JSON file.
        path: PathBuf,
        /// Maximum rows per UNWIND batch.
        #[arg(long, default_value_t = 1000)]
        batch_size: usize,
        /// Optional Cypher label stamped onto every ingested entity.
        /// Entities only merge with same-prefix siblings.
        #[arg(long)]
        prefix_label: Option<String>,
        /// Optional prefix folded into embedding-index / Qdrant
        /// collection names so vectors from different prefixes don't
        /// share an index.
        #[arg(long)]
        prefix_index: Option<String>,
    },
    /// Delete every node belonging to a single ingest `Source`:
    /// chunks attached via `:part_of` (always — they're 1:1 with the
    /// source) and user entities whose only `:mention` link was to
    /// this source (orphans only — shared entities survive). Vectors
    /// in Qdrant are cleaned up via `libqlink.delete_batch` across
    /// every collection inferred from the cached graph specification.
    DeleteBySource {
        /// `name` property of the `Source` node to delete.
        #[arg(long = "source", alias = "source-name")]
        source: String,
        /// Optional Cypher prefix label, must match the one used at
        /// ingest time. Scopes the deletion to that tenant / dataset.
        #[arg(long)]
        prefix_label: Option<String>,
        /// Optional prefix folded into Qdrant collection names, must
        /// match the one used at ingest time.
        #[arg(long)]
        prefix_index: Option<String>,
        /// Path to the graph specification cache. Used to enumerate
        /// the Qdrant collections that may hold vectors for the
        /// doomed entities (one collection per `Text` property name).
        /// Defaults to the same path the ingest commands use.
        #[arg(long = "spec-cache", default_value = DEFAULT_ONTOLOGY_CATALOG_CACHE_PATH)]
        spec_cache: PathBuf,
    },
    /// Discover which entity types in the graph are semantically
    /// relevant to a free-form user query. Emits a JSON summary that a
    /// QA service can use to pick which types to probe with a DSL or
    /// traversal query.
    ///
    /// Backed by [`Pipeline::run_entity_type_search`]: embeds the
    /// `text` once with BGE-M3, fans the query across every Qdrant
    /// collection populated by the SemanticText handler, and rolls
    /// hits up by entity type with their domain and scopes.
    EntityTypeSearch {
        /// Free-form user text.
        text: String,
        /// `top_k` passed to each `libqlink.search_labeled` call.
        #[arg(long, default_value_t = crate::core::DEFAULT_TOP_K)]
        top_k: u32,
        /// Cosine cutoff for the vector channel. `--no-threshold` keeps
        /// every result inside `top_k`.
        #[arg(long, default_value_t = crate::core::DEFAULT_SCORE_THRESHOLD)]
        score_threshold: f32,
        /// Drop the cosine cutoff entirely.
        #[arg(long, conflicts_with = "score_threshold")]
        no_threshold: bool,
        /// Roll up 1-hop neighbours of the matched nodes into the
        /// `neighbors` array. Off by default.
        #[arg(long)]
        include_neighbors: bool,
        /// Skip the `OntologyCatalog::find` channel (catalog-side
        /// semantic match against type descriptions).
        #[arg(long)]
        no_catalog: bool,
        /// Cosine cutoff for the catalog channel.
        #[arg(long, default_value_t = crate::core::DEFAULT_CATALOG_THRESHOLD)]
        catalog_threshold: f32,
        /// Restrict the search to specific ontology field names (before
        /// prefixing). Repeatable. Defaults to every known SemanticText
        /// field plus the `name` / `text` / `_canonical` built-ins.
        #[arg(long = "field")]
        fields: Vec<String>,
        /// Optional Cypher label, must match the ingest-side
        /// `prefix_label` so only same-tenant nodes hit.
        #[arg(long)]
        prefix_label: Option<String>,
        /// Optional Qdrant collection prefix, must match the
        /// ingest-side `prefix_index`.
        #[arg(long)]
        prefix_index: Option<String>,
    },
}

pub async fn run(cli: Cli) -> Result<()> {
    match cli.command {
        Command::Dsl { path } => cmd_dsl(path).await,
        Command::Cypher {
            path,
            prefix_label,
            prefix_index,
        } => cmd_cypher(&cli.config, path, prefix_label, prefix_index).await,
        Command::Run {
            path,
            prefix_label,
            prefix_index,
        } => cmd_run(&cli.config, path, prefix_label, prefix_index).await,
        Command::Traversal {
            path,
            prefix_label,
            prefix_index,
        } => cmd_traversal(&cli.config, path, prefix_label, prefix_index).await,
        Command::Prompt {
            query,
            schema,
            no_examples,
            no_specification,
        } => cmd_prompt(&cli.config, query, schema, no_examples, no_specification).await,
        Command::Schema {
            sample_size,
            format,
            output,
            no_examples,
        } => cmd_schema(&cli.config, sample_size, format, output, no_examples).await,
        Command::IngestGraph {
            path,
            batch_size,
            prefix_label,
            prefix_index,
        } => cmd_ingest_graph(&cli.config, path, batch_size, prefix_label, prefix_index).await,
        Command::EntityTypeSearch {
            text,
            top_k,
            score_threshold,
            no_threshold,
            include_neighbors,
            no_catalog,
            catalog_threshold,
            fields,
            prefix_label,
            prefix_index,
        } => {
            cmd_entity_type_search(
                &cli.config,
                text,
                top_k,
                if no_threshold {
                    None
                } else {
                    Some(score_threshold)
                },
                include_neighbors,
                !no_catalog,
                catalog_threshold,
                fields,
                prefix_label,
                prefix_index,
            )
            .await
        }
        Command::DeleteBySource {
            source,
            prefix_label,
            prefix_index,
            spec_cache,
        } => {
            cmd_delete_by_source(&cli.config, source, prefix_label, prefix_index, spec_cache).await
        }
    }
}

/// Build a [`SharedRegistry`] from `cfg`. Always returns a registry
/// (possibly empty) so callers can pass it through unconditionally.
fn build_registry(cfg: &Config) -> Result<(SharedRegistry, Option<SharedEmbedder>)> {
    let dim = cfg
        .types
        .get("SemanticText")
        .and_then(|t| t.embedding_dim)
        .unwrap_or(384);
    let model = cfg
        .types
        .get("SemanticText")
        .and_then(|t| t.embedding_model.clone());
    let embedder = embeddings::default_embedder(model.as_deref(), dim).map_err(|e| {
        crate::error::Error::Ingest(crate::ingest::IngestError::Type(format!(
            "embedder init: {e}"
        )))
    })?;
    let registry = types::handlers::register_default(cfg, embedder.clone()).map_err(|e| {
        crate::error::Error::Ingest(crate::ingest::IngestError::Type(format!(
            "registry init: {e}"
        )))
    })?;
    Ok((std::sync::Arc::new(registry), Some(embedder)))
}

fn build_ontology_catalog_embedder(cfg: &Config) -> Result<SharedEmbedder> {
    embeddings::default_embedder(
        cfg.ontology_catalog.embedding_model.as_deref(),
        cfg.ontology_catalog.embedding_dim,
    )
    .map_err(|e| {
        crate::error::Error::Ingest(crate::ingest::IngestError::Type(format!(
            "graph specification embedder init: {e}"
        )))
    })
}

fn build_ontology_catalog_reranker(cfg: &Config) -> Result<embeddings::SharedReranker> {
    embeddings::default_reranker(
        cfg.ontology_catalog.reranking_model.as_deref(),
        cfg.ontology_catalog.embedding_dim,
    )
    .map_err(|e| {
        crate::error::Error::Ingest(crate::ingest::IngestError::Type(format!(
            "graph specification reranker init: {e}"
        )))
    })
}

async fn cmd_dsl(path: PathBuf) -> Result<()> {
    let q = dsl::parse(&path).await?;
    println!("{}", serde_json::to_string_pretty(&q)?);
    Ok(())
}

async fn cmd_cypher(
    config_path: &std::path::Path,
    path: PathBuf,
    prefix_label: Option<String>,
    prefix_index: Option<String>,
) -> Result<()> {
    let cfg = load_config_or_default(config_path).await;
    let (registry, embedder) = build_registry(&cfg)?;
    // Load the graph specification snapshot so a DSL filter like
    // `{"field": "c.name", "op": "search", ...}` resolves to the
    // SemanticText handler automatically when the cached mapping
    // tagged `Company.name` as such — no `"type"` needed in the DSL.
    let spec_storage: Arc<dyn OntologyCatalogStorage> = Arc::new(
        JsonFileOntologyCatalogStorage::new(&cfg.ontology_catalog.cache_path),
    );
    let mut pipeline = Pipeline::new(Arc::new(crate::db::MockClient::new()), &cfg)
        .with_registry(registry)
        .with_ontology_catalog_storage(spec_storage)
        .with_prefix_label(prefix_label)
        .with_prefix_index(prefix_index);
    if let Some(e) = embedder {
        pipeline = pipeline.with_embedder(e);
    }
    pipeline.load_ontology_catalog().await?;
    let dsl_query = dsl::parse(&path).await?;
    let cypher = pipeline.compile(dsl_query)?;
    println!("-- Cypher --\n{}", cypher.text);
    println!("\n-- Parameters --");
    for (k, v) in &cypher.params {
        println!("${k} = {}", serde_json::to_string(v)?);
    }
    Ok(())
}

async fn cmd_run(
    config_path: &std::path::Path,
    path: PathBuf,
    prefix_label: Option<String>,
    prefix_index: Option<String>,
) -> Result<()> {
    let cfg = config::load(config_path).await?;
    let client = MemgraphClient::connect(&cfg.database).await?;
    let (registry, embedder) = build_registry(&cfg)?;
    let spec_storage: Arc<dyn OntologyCatalogStorage> = Arc::new(
        JsonFileOntologyCatalogStorage::new(&cfg.ontology_catalog.cache_path),
    );
    let mut pipeline = Pipeline::new(Arc::new(client), &cfg)
        .with_registry(registry)
        .with_ontology_catalog_storage(spec_storage)
        .with_prefix_label(prefix_label)
        .with_prefix_index(prefix_index);
    if let Some(e) = embedder {
        pipeline = pipeline.with_embedder(e);
    }
    // When a cross-encoder reranker model is configured for SemanticText,
    // attach it so semantic search reranks candidates in-process instead
    // of inside Memgraph (`libqlink.search_hybrid_reranked`).
    if let Some(reranker) = build_semantic_text_reranker(&cfg)? {
        pipeline = pipeline.with_reranker(reranker);
    }
    pipeline.load_ontology_catalog().await?;
    let dsl_query = dsl::parse(&path).await?;
    let result = pipeline.run(dsl_query).await?;
    print_query_result_table(&result);
    Ok(())
}

/// Build the SemanticText cross-encoder reranker from
/// `[types.SemanticText].reranking_model`, or `None` when unset (the
/// pipeline then defers reranking to qlink's `search_hybrid_reranked`).
fn build_semantic_text_reranker(cfg: &Config) -> Result<Option<embeddings::SharedReranker>> {
    let Some(model) = cfg
        .types
        .get("SemanticText")
        .and_then(|t| t.reranking_model.clone())
    else {
        return Ok(None);
    };
    let dim = cfg
        .types
        .get("SemanticText")
        .and_then(|t| t.embedding_dim)
        .unwrap_or(384);
    let reranker = embeddings::default_reranker(Some(&model), dim).map_err(|e| {
        crate::error::Error::Ingest(crate::ingest::IngestError::Type(format!(
            "SemanticText reranker init: {e}"
        )))
    })?;
    Ok(Some(reranker))
}

async fn cmd_traversal(
    config_path: &std::path::Path,
    path: PathBuf,
    prefix_label: Option<String>,
    prefix_index: Option<String>,
) -> Result<()> {
    let cfg = config::load(config_path).await?;
    let client = MemgraphClient::connect(&cfg.database).await?;
    let (registry, embedder) = build_registry(&cfg)?;
    let spec_storage: Arc<dyn OntologyCatalogStorage> = Arc::new(
        JsonFileOntologyCatalogStorage::new(&cfg.ontology_catalog.cache_path),
    );
    let mut pipeline = Pipeline::new(Arc::new(client), &cfg)
        .with_registry(registry)
        .with_ontology_catalog_storage(spec_storage)
        .with_prefix_label(prefix_label.clone())
        .with_prefix_index(prefix_index.clone());
    if let Some(e) = embedder {
        pipeline = pipeline.with_embedder(e);
    }
    // Only attach a reranker when one is explicitly configured —
    // otherwise the traversal pipeline skips the rerank step.
    if cfg.ontology_catalog.reranking_model.is_some() {
        let reranker = build_ontology_catalog_reranker(&cfg)?;
        pipeline = pipeline.with_reranker(reranker);
    }
    pipeline.load_ontology_catalog().await?;

    let raw = fs::read_to_string(&path).await?;
    let mut traversal: dsl::TraversalQuery = serde_json::from_str(&raw)?;
    // The CLI flags win over what the traversal JSON declared, so a
    // user can scope an ad-hoc lookup without editing the file.
    if let Some(prefix) = prefix_label {
        let trimmed = prefix.trim().to_string();
        if !trimmed.is_empty() {
            traversal.prefix_label = Some(trimmed);
        }
    }
    if let Some(prefix) = prefix_index {
        let trimmed = prefix.trim().to_string();
        if !trimmed.is_empty() {
            traversal.prefix_index = Some(trimmed);
        }
    }
    let result = pipeline.run_traversal(traversal).await?;
    print_traversal_result(&result);
    Ok(())
}

/// Render a traversal result as readable text blocks instead of a
/// table: each chunk is printed as its score (and rerank score when
/// present), an optional source label, and the full chunk text on its
/// own lines. Long chunk text in a table is unreadable; this keeps the
/// newlines intact and surfaces only what callers care about.
fn print_traversal_result(result: &QueryResult) {
    if result.rows.is_empty() {
        println!("(no matching chunks)");
        return;
    }

    for (idx, row) in result.rows.iter().enumerate() {
        if idx > 0 {
            println!();
            println!("{}", "─".repeat(60));
        }

        let score = row.fields.get("score").map(traversal_score_string);
        let rerank = row.fields.get("rerank_score").map(traversal_score_string);

        let mut header = format!("#{}", idx + 1);
        if let Some(score) = score {
            header.push_str(&format!("  score={score}"));
        }
        if let Some(rerank) = rerank {
            header.push_str(&format!("  rerank={rerank}"));
        }
        if let Some(name) = row.fields.get("source_name").and_then(traversal_text_value) {
            if !name.is_empty() {
                header.push_str(&format!("  source={name}"));
            }
        }
        println!("{header}");

        let text = row
            .fields
            .get("chunk_text")
            .and_then(traversal_text_value)
            .unwrap_or_default();
        println!("{text}");
    }
    println!();
    println!("{} chunk(s)", result.rows.len());
}

/// Format a score cell with a stable 4-decimal precision; falls back to
/// the raw cell rendering for non-numeric values.
fn traversal_score_string(value: &Value) -> String {
    match value {
        Value::Float(v) => format!("{v:.4}"),
        Value::Int(v) => format!("{v}"),
        Value::Null => "—".into(),
        other => value_cell(other),
    }
}

/// Extract a plain string from a result cell, preserving newlines
/// (unlike [`value_cell`], which escapes them for table layout).
fn traversal_text_value(value: &Value) -> Option<String> {
    match value {
        Value::Null => None,
        Value::String(v) => Some(v.clone()),
        Value::Json(serde_json::Value::String(v)) => Some(v.clone()),
        other => Some(value_cell(other)),
    }
}

fn print_query_result_table(result: &QueryResult) {
    println!("{}", query_result_table(result));
}

fn query_result_table(result: &QueryResult) -> String {
    let columns = query_result_columns(result);
    if columns.is_empty() {
        return "(no rows)".into();
    }

    let mut builder = Builder::default();
    builder.push_record(columns.iter().map(column_header));
    for row in &result.rows {
        builder.push_record(columns.iter().map(|column| {
            row.fields
                .get(&column.name)
                .map(value_cell)
                .unwrap_or_default()
        }));
    }

    let mut out = builder.build().with(Style::ascii()).to_string();
    out.push_str(&format!("{} row(s)", result.rows.len()));
    out
}

fn column_header(column: &Column) -> String {
    match column.node_type {
        Some(t) => format!("{} ({:?})", column.name, t),
        None => column.name.clone(),
    }
}

fn query_result_columns(result: &QueryResult) -> Vec<Column> {
    if !result.columns.is_empty() {
        return result
            .columns
            .iter()
            .filter(|column| !is_hidden_result_column(&column.name))
            .cloned()
            .collect();
    }

    let mut names = BTreeSet::new();
    for row in &result.rows {
        names.extend(
            row.fields
                .keys()
                .filter(|name| !is_hidden_result_column(name))
                .cloned(),
        );
    }
    names.into_iter().map(Column::new).collect()
}

fn is_hidden_result_column(column: &str) -> bool {
    matches!(column, "score" | "sources")
}

fn value_cell(value: &Value) -> String {
    let raw = match value {
        Value::Null => String::new(),
        Value::Bool(v) => v.to_string(),
        Value::Int(v) => v.to_string(),
        Value::Float(v) => v.to_string(),
        Value::String(v) => v.clone(),
        Value::Json(v) => serde_json::to_string(v).unwrap_or_else(|_| v.to_string()),
    };
    raw.replace('\n', "\\n").replace('\r', "\\r")
}

async fn cmd_prompt(
    config_path: &std::path::Path,
    query: Option<String>,
    schema_path: Option<PathBuf>,
    no_examples: bool,
    no_specification: bool,
) -> Result<()> {
    let cfg = load_config_or_default(config_path).await;
    let schema = match schema_path {
        Some(p) => {
            let raw = fs::read_to_string(&p).await?;
            serde_json::from_str::<GraphSchema>(&raw)?
        }
        None => {
            let client: Arc<dyn GraphClient> =
                Arc::new(MemgraphClient::connect(&cfg.database).await?);
            client.schema().await?
        }
    };
    let (registry, _embedder) = build_registry(&cfg)?;
    let ontology_catalog_embedder = if query.is_some() {
        Some(build_ontology_catalog_embedder(&cfg)?)
    } else {
        None
    };
    let ontology_catalog_reranker = if query.is_some() {
        Some(build_ontology_catalog_reranker(&cfg)?)
    } else {
        None
    };
    let ontology_catalog = if no_specification {
        None
    } else {
        let store = JsonFileOntologyCatalogStorage::default();
        let mut catalog = store.load().await?;
        if catalog.is_empty() {
            None
        } else {
            if let Some(embedder) = ontology_catalog_embedder.as_ref() {
                catalog.compute(embedder.as_ref()).map_err(|e| {
                    crate::error::Error::Ingest(crate::ingest::IngestError::Type(format!(
                        "ontology catalog embedding: {e}"
                    )))
                })?;
            }
            Some(catalog)
        }
    };
    let registry_for_prompt = (*registry).clone();
    let opts = PromptOptions {
        include_examples: !no_examples,
        ontology_catalog,
        embedding_model: ontology_catalog_embedder,
        reranking_model: ontology_catalog_reranker,
        schema_selection: prompt::PromptSchemaSelection {
            reranking_threshold: cfg.ontology_catalog.reranking_threshold,
            ..Default::default()
        },
        type_registry: if registry_for_prompt.is_empty() {
            None
        } else {
            Some(registry_for_prompt)
        },
        ..PromptOptions::default()
    };
    let prompt = match query {
        Some(query) => prompt::generate_query_prompt(&query, &schema, &opts),
        None => prompt::generate_system_prompt(&schema, &opts),
    };
    println!("{prompt}");
    Ok(())
}

async fn cmd_schema(
    config_path: &std::path::Path,
    sample_size: u64,
    format: SchemaFormat,
    output: Option<PathBuf>,
    no_examples: bool,
) -> Result<()> {
    let cfg = config::load(config_path).await?;
    let client = MemgraphClient::connect(&cfg.database).await?;

    let schema = introspect::introspect_schema(
        &client,
        introspect::IntrospectOptions {
            sample_size,
            ..Default::default()
        },
    )
    .await?;

    let body = match format {
        SchemaFormat::Json => serde_json::to_string_pretty(&schema)?,
        SchemaFormat::Prompt => {
            let opts = PromptOptions {
                include_examples: !no_examples,
                ..PromptOptions::default()
            };
            prompt::generate_system_prompt(&schema, &opts)
        }
    };

    match output {
        Some(path) => {
            fs::write(&path, &body).await?;
            tracing::info!(target: "linguagraph::cli", path = %path.display(), "wrote schema");
        }
        None => println!("{body}"),
    }
    Ok(())
}

async fn cmd_ingest_graph(
    config_path: &std::path::Path,
    path: PathBuf,
    batch_size: usize,
    prefix_label: Option<String>,
    prefix_index: Option<String>,
) -> Result<()> {
    let cfg = config::load(config_path).await?;
    let raw = fs::read_to_string(&path).await?;
    let graph = GraphBuilder::from_json(&raw)?;
    let (registry, embedder) = build_registry(&cfg)?;

    let client = MemgraphClient::connect(&cfg.database).await?;
    let mut pipeline = Pipeline::new(Arc::new(client), &cfg)
        .with_ingest_batch_size(batch_size)
        .with_registry(registry)
        .with_prefix_label(prefix_label)
        .with_prefix_index(prefix_index);
    if let Some(e) = embedder {
        pipeline = pipeline.with_embedder(e);
    }

    let summary = pipeline.ingest(&graph).await?;
    println!("{}", serde_json::to_string_pretty(&summary)?);
    Ok(())
}

async fn cmd_delete_by_source(
    config_path: &std::path::Path,
    source: String,
    prefix_label: Option<String>,
    prefix_index: Option<String>,
    spec_cache: PathBuf,
) -> Result<()> {
    let cfg = config::load(config_path).await?;
    let (registry, embedder) = build_registry(&cfg)?;

    // Load the spec snapshot so the pipeline can enumerate per-property
    // Qdrant collections. Missing cache is fine — the deletion still
    // covers the two built-in collections (Source.name, Chunk.text).
    let spec_storage: Arc<dyn OntologyCatalogStorage> =
        Arc::new(JsonFileOntologyCatalogStorage::new(spec_cache));

    let client = MemgraphClient::connect(&cfg.database).await?;
    let mut pipeline = Pipeline::new(Arc::new(client), &cfg)
        .with_registry(registry)
        .with_ontology_catalog_storage(spec_storage)
        .with_prefix_label(prefix_label)
        .with_prefix_index(prefix_index);
    if let Some(e) = embedder {
        pipeline = pipeline.with_embedder(e);
    }
    pipeline.load_ontology_catalog().await?;

    let summary = pipeline.delete_by_source(source).await?;
    println!("{}", serde_json::to_string_pretty(&summary)?);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
async fn cmd_entity_type_search(
    config_path: &std::path::Path,
    text: String,
    top_k: u32,
    score_threshold: Option<f32>,
    include_neighbors: bool,
    include_catalog_signal: bool,
    catalog_threshold: f32,
    fields: Vec<String>,
    prefix_label: Option<String>,
    prefix_index: Option<String>,
) -> Result<()> {
    let cfg = config::load(config_path).await?;
    let client = MemgraphClient::connect(&cfg.database).await?;
    let (registry, embedder) = build_registry(&cfg)?;
    let spec_storage: Arc<dyn OntologyCatalogStorage> = Arc::new(
        JsonFileOntologyCatalogStorage::new(&cfg.ontology_catalog.cache_path),
    );
    let mut pipeline = Pipeline::new(Arc::new(client), &cfg)
        .with_registry(registry)
        .with_ontology_catalog_storage(spec_storage)
        .with_prefix_label(prefix_label)
        .with_prefix_index(prefix_index);
    if let Some(e) = embedder {
        pipeline = pipeline.with_embedder(e);
    }
    pipeline.load_ontology_catalog().await?;

    let query = crate::core::EntityTypeSearchQuery {
        text,
        top_k,
        score_threshold,
        include_neighbors,
        include_catalog_signal,
        catalog_threshold,
        fields: if fields.is_empty() {
            None
        } else {
            Some(fields)
        },
        collections: None,
    };
    let result = pipeline.run_entity_type_search(query).await?;
    println!("{}", serde_json::to_string_pretty(&result)?);
    Ok(())
}

/// For commands that don't need a live DB, missing config falls back to
/// safe defaults instead of failing.
async fn load_config_or_default(path: &std::path::Path) -> Config {
    match config::load(path).await {
        Ok(c) => c,
        Err(_) => Config {
            database: config::DatabaseConfig {
                uri: "bolt://localhost:7687".into(),
                user: String::new(),
                password: String::new(),
                max_connections: 1,
                query_timeout_secs: 30,
                database: "memgraph".to_string(),
            },
            llm: Default::default(),
            query: Default::default(),
            ontology_catalog: Default::default(),
            prompt: Default::default(),
            ingest: Default::default(),
            types: Default::default(),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    #[test]
    fn query_result_table_uses_result_columns_order() {
        let result = QueryResult {
            columns: vec!["name".into(), "score".into(), "sources".into()],
            rows: vec![row([
                ("score", Value::Float(0.75)),
                ("name", Value::String("Alice".into())),
                ("sources", Value::Json(serde_json::json!([{"id": "s1"}]))),
            ])],
        };

        let table = query_result_table(&result);

        assert!(table.contains("| name  |"));
        assert!(table.contains("| Alice |"));
        assert!(!table.contains("score"));
        assert!(!table.contains("sources"));
        assert!(table.ends_with("1 row(s)"));
    }

    #[test]
    fn query_result_table_falls_back_to_sorted_row_fields() {
        let result = QueryResult {
            columns: vec![],
            rows: vec![row([
                ("b", Value::Int(2)),
                ("a", Value::String("one".into())),
                ("score", Value::Float(0.75)),
                ("sources", Value::Json(serde_json::json!([{"id": "s1"}]))),
            ])],
        };

        let table = query_result_table(&result);

        assert!(table.contains("| a   | b |"));
        assert!(table.contains("| one | 2 |"));
        assert!(!table.contains("score"));
        assert!(!table.contains("sources"));
    }

    #[test]
    fn query_result_table_compacts_json_and_newlines() {
        let result = QueryResult {
            columns: vec!["chunk_text".into(), "entities".into()],
            rows: vec![row([
                ("chunk_text", Value::String("first\nsecond".into())),
                (
                    "entities",
                    Value::Json(serde_json::json!([{"id":"e1","name":"Alice"}])),
                ),
            ])],
        };

        let table = query_result_table(&result);

        assert!(table.contains("first\\nsecond"));
        assert!(table.contains(r#"[{"id":"e1","name":"Alice"}]"#));
    }

    #[test]
    fn query_result_table_handles_empty_result() {
        assert_eq!(query_result_table(&QueryResult::empty()), "(no rows)");
    }

    fn row(fields: impl IntoIterator<Item = (&'static str, Value)>) -> crate::db::Row {
        crate::db::Row {
            fields: fields
                .into_iter()
                .map(|(key, value)| (key.to_string(), value))
                .collect::<BTreeMap<_, _>>(),
        }
    }
}
