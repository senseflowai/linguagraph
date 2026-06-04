//! Clap-based CLI. Subcommands map 1:1 to pipeline stages so users can stop
//! at any layer for inspection.

use std::collections::BTreeSet;
use std::path::PathBuf;
use std::sync::Arc;

use crate::ast::Literal;
use crate::config::{self, Config};
use crate::core::Pipeline;
use crate::db::{introspect, Column, GraphClient, MemgraphClient, QueryResult, Value};
use crate::dsl;
use crate::embeddings::{self, SharedEmbedder};
use crate::error::Result;
use crate::graph::{
    DomainOntology, EntityTypeSpec, GraphBuilder, JsonFileOntologyCatalogStorage,
    OntologyCatalogStorage, RelationTypeSpec, DEFAULT_ONTOLOGY_CATALOG_CACHE_PATH,
};
use crate::mapper::{self, Mapping};
use crate::prompt::{self, GraphSchema, PromptGenerator, PromptOptions};
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
    /// chunks in `text`), follows `MENTIONS` from matched entities
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
    /// Compile a (data, mapper) pair into a graph, update the graph
    /// specification cache, and ingest the graph into the configured
    /// database.
    IngestJson {
        /// Path to the raw data JSON file.
        data: PathBuf,
        /// Path to the mapper JSON file.
        mapper: PathBuf,
        /// Path to the graph specification cache file.
        #[arg(long = "spec-cache", default_value = DEFAULT_ONTOLOGY_CATALOG_CACHE_PATH)]
        spec_cache: PathBuf,
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
    /// Compile a DSL JSON file with the configured type registry and
    /// print the generated Cypher (including any qlink fragments).
    /// Does not connect to the database.
    Query {
        /// Path to the DSL JSON file.
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
    /// Analyse an arbitrary JSON document and emit a prompt that
    /// instructs an LLM to produce a linguagraph mapping JSON for it.
    GeneratePrompt {
        /// Path to the input JSON file.
        path: PathBuf,
        /// Free-form domain hints (repeatable). Rendered verbatim
        /// under a "Domain hints" section.
        #[arg(long = "hint")]
        hints: Vec<String>,
        /// Preferred field types (repeatable; ordered).
        #[arg(long = "prefer")]
        prefer: Vec<String>,
        /// Skip the worked example block.
        #[arg(long)]
        no_examples: bool,
        /// Skip the inferred-structure section.
        #[arg(long)]
        no_summary: bool,
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
    /// Emit a system prompt instructing an LLM to extract entities and
    /// relations in the JSON shape consumed by `ingest-document`.
    KnowledgePrompt {
        /// Domain whose ontology should be used (e.g. `legal`).
        /// Falls back to `[prompt].default_domain` from config.
        /// Ignored when `--entity-type`/`--relation-type` are passed.
        #[arg(long)]
        domain: Option<String>,
        /// Allowed entity type (repeatable). When passed, fully
        /// overrides the domain ontology for this run.
        #[arg(long = "entity-type")]
        entity_types: Vec<String>,
        /// Allowed relation type (repeatable). When passed, fully
        /// overrides the domain ontology for this run.
        #[arg(long = "relation-type")]
        relation_types: Vec<String>,
        /// Write the prompt to this path instead of stdout.
        #[arg(long, short = 'o')]
        output: Option<PathBuf>,
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
        Command::IngestJson {
            data,
            mapper,
            spec_cache,
            batch_size,
            prefix_label,
            prefix_index,
        } => {
            cmd_ingest_json(
                &cli.config,
                data,
                mapper,
                spec_cache,
                batch_size,
                prefix_label,
                prefix_index,
            )
            .await
        }
        Command::IngestGraph {
            path,
            batch_size,
            prefix_label,
            prefix_index,
        } => cmd_ingest_graph(&cli.config, path, batch_size, prefix_label, prefix_index).await,
        Command::Query {
            path,
            prefix_label,
            prefix_index,
        } => cmd_query(&cli.config, path, prefix_label, prefix_index).await,
        Command::GeneratePrompt {
            path,
            hints,
            prefer,
            no_examples,
            no_summary,
        } => cmd_generate_prompt(&cli.config, path, hints, prefer, no_examples, no_summary).await,
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
                if no_threshold { None } else { Some(score_threshold) },
                include_neighbors,
                !no_catalog,
                catalog_threshold,
                fields,
                prefix_label,
                prefix_index,
            )
            .await
        }
        Command::KnowledgePrompt {
            domain,
            entity_types,
            relation_types,
            output,
        } => cmd_knowledge_prompt(&cli.config, domain, entity_types, relation_types, output).await,
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
        match v {
            Literal::List(vec) => {
                let emb: Vec<f32> = vec
                    .iter()
                    .filter_map(|value| match value {
                        Literal::Float(f) => Some(*f as f32),
                        _ => None,
                    })
                    .collect();
                println!("{:?}", emb);
            }
            _ => {}
        }
    }
    Ok(())
}

async fn cmd_query(
    config_path: &std::path::Path,
    path: PathBuf,
    prefix_label: Option<String>,
    prefix_index: Option<String>,
) -> Result<()> {
    // Same as `cypher` today; kept as a separate command so future
    // natural-language pipelines can hang off the more obvious name.
    cmd_cypher(config_path, path, prefix_label, prefix_index).await
}

async fn cmd_generate_prompt(
    config_path: &std::path::Path,
    path: PathBuf,
    hints: Vec<String>,
    prefer: Vec<String>,
    no_examples: bool,
    no_summary: bool,
) -> Result<()> {
    use crate::promptgen::{generate_prompt, PromptGenOptions};

    let raw = fs::read_to_string(&path).await?;
    let value: serde_json::Value = serde_json::from_str(&raw)?;

    // The registry is best-effort: when config is missing or wrong
    // we still want the CLI to work, falling back to the bundled
    // catalogue.
    let cfg = load_config_or_default(config_path).await;
    let registry = build_registry(&cfg).ok().map(|(r, _)| (*r).clone());

    let opts = PromptGenOptions {
        domain_hints: hints,
        preferred_types: prefer,
        include_examples: !no_examples,
        include_inferred_summary: !no_summary,
        registry,
        ..PromptGenOptions::default()
    };
    let prompt = generate_prompt(&value, &opts);
    print!("{prompt}");
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
    pipeline.load_ontology_catalog().await?;
    let dsl_query = dsl::parse(&path).await?;
    let result = pipeline.run(dsl_query).await?;
    print_query_result_table(&result);
    Ok(())
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
    print_query_result_table(&result);
    Ok(())
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

    let schema =
        introspect::introspect_schema(&client, introspect::IntrospectOptions { sample_size })
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

async fn cmd_ingest_json(
    config_path: &std::path::Path,
    data: PathBuf,
    mapper: PathBuf,
    spec_cache: PathBuf,
    batch_size: usize,
    prefix_label: Option<String>,
    prefix_index: Option<String>,
) -> Result<()> {
    let cfg = config::load(config_path).await?;
    let mapping = Mapping::load(&mapper).await?;
    let raw = fs::read_to_string(&data).await?;
    let value: serde_json::Value = serde_json::from_str(&raw)?;
    let mapped = mapper::to_graph(&mapping, &value)?;
    let (registry, embedder) = build_registry(&cfg)?;
    let ontology_catalog_embedder = build_ontology_catalog_embedder(&cfg)?;

    let catalog_storage = JsonFileOntologyCatalogStorage::new(spec_cache);
    let mut catalog = catalog_storage.load().await.unwrap_or_default();
    catalog.merge(&mapped.catalog);
    catalog
        .compute(ontology_catalog_embedder.as_ref())
        .map_err(|e| {
            crate::error::Error::Ingest(crate::ingest::IngestError::Type(format!(
                "ontology catalog embedding: {e}"
            )))
        })?;
    catalog_storage.save(&catalog).await?;

    let client = MemgraphClient::connect(&cfg.database).await?;
    let mut pipeline = Pipeline::new(Arc::new(client), &cfg)
        .with_ingest_batch_size(batch_size)
        .with_registry(registry)
        .with_prefix_label(prefix_label)
        .with_prefix_index(prefix_index);
    if let Some(e) = embedder {
        pipeline = pipeline.with_embedder(e);
    }

    let summary = pipeline.ingest(&mapped.graph).await?;
    println!("{}", serde_json::to_string_pretty(&summary)?);
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
        fields: if fields.is_empty() { None } else { Some(fields) },
        collections: None,
    };
    let result = pipeline.run_entity_type_search(query).await?;
    println!("{}", serde_json::to_string_pretty(&result)?);
    Ok(())
}

async fn cmd_knowledge_prompt(
    config_path: &std::path::Path,
    domain: Option<String>,
    entity_types: Vec<String>,
    relation_types: Vec<String>,
    output: Option<PathBuf>,
) -> Result<()> {
    let cfg = load_config_or_default(config_path).await;
    let generator = PromptGenerator::from_config(&cfg.prompt).await?;

    let prompt = if !entity_types.is_empty() || !relation_types.is_empty() {
        let ontology = DomainOntology {
            entity_types: entity_types.into_iter().map(EntityTypeSpec::new).collect(),
            relation_types: relation_types
                .into_iter()
                .map(RelationTypeSpec::new)
                .collect(),
        };
        // Use the explicit --domain when supplied so the prompt's
        // framing matches; otherwise fall back to the config default
        // or a neutral "custom" label.
        let label = domain
            .as_deref()
            .or(cfg.prompt.default_domain.as_deref())
            .unwrap_or("custom");
        generator.knowledge_extract_prompt_with(label, &ontology)
    } else {
        generator.knowledge_extract_prompt(domain.as_deref())?
    };

    match output {
        Some(p) => {
            fs::write(&p, &prompt).await?;
            tracing::info!(target: "linguagraph::cli", path = %p.display(), "wrote knowledge-extract prompt");
        }
        None => print!("{prompt}"),
    }
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
