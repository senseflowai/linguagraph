//! Clap-based CLI. Subcommands map 1:1 to pipeline stages so users can stop
//! at any layer for inspection.

use std::path::PathBuf;
use std::sync::Arc;

use crate::ast::Literal;
use crate::config::{self, Config};
use crate::core::Pipeline;
use crate::db::{introspect, GraphClient, MemgraphClient};
use crate::dsl;
use crate::embeddings::{self, SharedEmbedder};
use crate::error::Result;
use crate::graph::{
    FileGraphSpecificationStorage, GraphBuilder, GraphSpecificationStorage,
    DEFAULT_GRAPH_SPECIFICATION_CACHE_PATH,
};
use crate::mapper::{self, Mapping};
use crate::metadata::{FileMetadataStore, MetadataStore};
use crate::prompt::{self, GraphSchema, PromptOptions};
use crate::promptgen::knowledge::{EntityTypeSpec, KnowledgeExtractOptions, RelationTypeSpec};
use crate::types::{self, SharedRegistry};
use clap::{Parser, Subcommand, ValueEnum};
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
    Cypher { path: PathBuf },
    /// Compile and execute a DSL file against the configured database.
    Run { path: PathBuf },
    /// Compile and execute a traversal-query JSON file against the configured database.
    Traversal {
        /// Path to the traversal-query JSON file.
        path: PathBuf,
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
        #[arg(long = "spec-cache", default_value = DEFAULT_GRAPH_SPECIFICATION_CACHE_PATH)]
        spec_cache: PathBuf,
        /// Maximum rows per UNWIND batch.
        #[arg(long, default_value_t = 1000)]
        batch_size: usize,
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
    },
    /// Compile a DSL JSON file with the configured type registry and
    /// print the generated Cypher (including any qlink fragments).
    /// Does not connect to the database.
    Query {
        /// Path to the DSL JSON file.
        path: PathBuf,
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
    /// Ingest a document JSON (`{document: {name, path, chunks: [...]}}`)
    /// directly via the document-shaped ingestion path. Embeds chunk
    /// text and writes the Document/Chunk/Entity graph to the database.
    IngestDocument {
        /// Path to the document JSON file.
        path: PathBuf,
        /// Maximum rows per UNWIND batch.
        #[arg(long, default_value_t = 1000)]
        batch_size: usize,
    },
    /// Like `ingest-document` but prints the generated Cypher batches
    /// instead of executing them. Skips the embedding side-effects
    /// (which require a live embedder + Qdrant) so the output is
    /// purely declarative.
    IngestDocumentCypher {
        path: PathBuf,
        #[arg(long, default_value_t = 1000)]
        batch_size: usize,
    },
    /// Emit a prompt instructing an LLM to extract entities and
    /// relations from a legal text fragment, in the JSON shape
    /// consumed by `ingest-document`.
    KnowledgePrompt {
        /// Path to a UTF-8 text file containing the fragment to
        /// analyse. Use `-` to read from stdin.
        path: PathBuf,
        /// Allowed entity type (repeatable). Defaults to the bundled
        /// legal-domain vocabulary when none are passed.
        #[arg(long = "entity-type")]
        entity_types: Vec<String>,
        /// Allowed relation type (repeatable). Defaults to the
        /// bundled legal-domain vocabulary when none are passed.
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
        Command::Cypher { path } => cmd_cypher(&cli.config, path).await,
        Command::Run { path } => cmd_run(&cli.config, path).await,
        Command::Traversal { path } => cmd_traversal(&cli.config, path).await,
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
        } => cmd_ingest_json(&cli.config, data, mapper, spec_cache, batch_size).await,
        Command::IngestGraph { path, batch_size } => {
            cmd_ingest_graph(&cli.config, path, batch_size).await
        }
        Command::Query { path } => cmd_query(&cli.config, path).await,
        Command::GeneratePrompt {
            path,
            hints,
            prefer,
            no_examples,
            no_summary,
        } => cmd_generate_prompt(&cli.config, path, hints, prefer, no_examples, no_summary).await,
        Command::IngestDocument { path, batch_size } => {
            cmd_ingest_document(&cli.config, path, batch_size).await
        }
        Command::IngestDocumentCypher { path, batch_size } => {
            cmd_ingest_document_cypher(&cli.config, path, batch_size).await
        }
        Command::KnowledgePrompt {
            path,
            entity_types,
            relation_types,
            output,
        } => cmd_knowledge_prompt(path, entity_types, relation_types, output).await,
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

fn build_graph_specification_embedder(cfg: &Config) -> Result<SharedEmbedder> {
    embeddings::default_embedder(
        cfg.graph_specification.embedding_model.as_deref(),
        cfg.graph_specification.embedding_dim,
    )
    .map_err(|e| {
        crate::error::Error::Ingest(crate::ingest::IngestError::Type(format!(
            "graph specification embedder init: {e}"
        )))
    })
}

fn build_graph_specification_reranker(cfg: &Config) -> Result<embeddings::SharedReranker> {
    embeddings::default_reranker(
        cfg.graph_specification.reranking_model.as_deref(),
        cfg.graph_specification.embedding_dim,
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

async fn cmd_cypher(config_path: &std::path::Path, path: PathBuf) -> Result<()> {
    let cfg = load_config_or_default(config_path).await;
    let (registry, embedder) = build_registry(&cfg)?;
    // Load the metadata snapshot so a DSL filter like
    // `{"field": "c.name", "op": "search", ...}` resolves to the
    // SemanticText handler automatically when the cached mapping
    // tagged `Company.name` as such — no `"type"` needed in the DSL.
    let meta_store: Arc<dyn MetadataStore> =
        Arc::new(FileMetadataStore::new(&cfg.metadata.cache_path));
    let mut pipeline = Pipeline::new(Arc::new(crate::db::MockClient::new()), &cfg)
        .with_registry(registry)
        .with_metadata_store(meta_store);
    if let Some(e) = embedder {
        pipeline = pipeline.with_embedder(e);
    }
    pipeline.load_metadata().await?;
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

async fn cmd_query(config_path: &std::path::Path, path: PathBuf) -> Result<()> {
    // Same as `cypher` today; kept as a separate command so future
    // natural-language pipelines can hang off the more obvious name.
    cmd_cypher(config_path, path).await
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

async fn cmd_run(config_path: &std::path::Path, path: PathBuf) -> Result<()> {
    let cfg = config::load(config_path).await?;
    let client = MemgraphClient::connect(&cfg.database).await?;
    let (registry, embedder) = build_registry(&cfg)?;
    let meta_store: Arc<dyn MetadataStore> =
        Arc::new(FileMetadataStore::new(&cfg.metadata.cache_path));
    let mut pipeline = Pipeline::new(Arc::new(client), &cfg)
        .with_registry(registry)
        .with_metadata_store(meta_store);
    if let Some(e) = embedder {
        pipeline = pipeline.with_embedder(e);
    }
    pipeline.load_metadata().await?;
    let dsl_query = dsl::parse(&path).await?;
    let result = pipeline.run(dsl_query).await?;
    println!("{}", serde_json::to_string_pretty(&result)?);
    Ok(())
}

async fn cmd_traversal(config_path: &std::path::Path, path: PathBuf) -> Result<()> {
    let cfg = config::load(config_path).await?;
    let client = MemgraphClient::connect(&cfg.database).await?;
    let (registry, embedder) = build_registry(&cfg)?;
    let meta_store: Arc<dyn MetadataStore> =
        Arc::new(FileMetadataStore::new(&cfg.metadata.cache_path));
    let mut pipeline = Pipeline::new(Arc::new(client), &cfg)
        .with_registry(registry)
        .with_metadata_store(meta_store);
    if let Some(e) = embedder {
        pipeline = pipeline.with_embedder(e);
    }
    pipeline.load_metadata().await?;

    let raw = fs::read_to_string(&path).await?;
    let traversal: dsl::TraversalQuery = serde_json::from_str(&raw)?;
    let result = pipeline.run_traversal(traversal).await?;
    println!("{}", serde_json::to_string_pretty(&result)?);
    Ok(())
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
    let graph_specification_embedder = if query.is_some() {
        Some(build_graph_specification_embedder(&cfg)?)
    } else {
        None
    };
    let graph_specification_reranker = if query.is_some() {
        Some(build_graph_specification_reranker(&cfg)?)
    } else {
        None
    };
    let graph_specification = if no_specification {
        None
    } else {
        let store = FileGraphSpecificationStorage::default();
        let mut spec = store.load().await?;
        if spec.is_empty() {
            None
        } else {
            if let Some(embedder) = graph_specification_embedder.as_ref() {
                spec.compute(embedder.as_ref()).map_err(|e| {
                    crate::error::Error::Ingest(crate::ingest::IngestError::Type(format!(
                        "graph specification embedding: {e}"
                    )))
                })?;
            }
            Some(spec)
        }
    };
    let registry_for_prompt = (*registry).clone();
    let opts = PromptOptions {
        include_examples: !no_examples,
        graph_specification,
        embedding_model: graph_specification_embedder,
        reranking_model: graph_specification_reranker,
        schema_selection: prompt::PromptSchemaSelection {
            reranking_threshold: cfg.graph_specification.reranking_threshold,
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
) -> Result<()> {
    let cfg = config::load(config_path).await?;
    let mapping = Mapping::load(&mapper).await?;
    let raw = fs::read_to_string(&data).await?;
    let value: serde_json::Value = serde_json::from_str(&raw)?;
    let mapped = mapper::to_graph(&mapping, &value)?;
    let (registry, embedder) = build_registry(&cfg)?;
    let graph_specification_embedder = build_graph_specification_embedder(&cfg)?;

    let spec_storage = FileGraphSpecificationStorage::new(spec_cache);
    let mut specification = spec_storage.load().await?;
    specification.merge(&mapped.specification);
    specification
        .compute(graph_specification_embedder.as_ref())
        .map_err(|e| {
            crate::error::Error::Ingest(crate::ingest::IngestError::Type(format!(
                "graph specification embedding: {e}"
            )))
        })?;
    spec_storage.save(&specification).await?;

    let client = MemgraphClient::connect(&cfg.database).await?;
    let store: Arc<dyn MetadataStore> = Arc::new(FileMetadataStore::new(&cfg.metadata.cache_path));
    let mut pipeline = Pipeline::new(Arc::new(client), &cfg)
        .with_ingest_batch_size(batch_size)
        .with_metadata_store(store)
        .with_registry(registry);
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
) -> Result<()> {
    let cfg = config::load(config_path).await?;
    let raw = fs::read_to_string(&path).await?;
    let graph = GraphBuilder::from_json(&raw)?;
    let (registry, embedder) = build_registry(&cfg)?;

    let client = MemgraphClient::connect(&cfg.database).await?;
    let store: Arc<dyn MetadataStore> = Arc::new(FileMetadataStore::new(&cfg.metadata.cache_path));
    let mut pipeline = Pipeline::new(Arc::new(client), &cfg)
        .with_ingest_batch_size(batch_size)
        .with_metadata_store(store)
        .with_registry(registry);
    if let Some(e) = embedder {
        pipeline = pipeline.with_embedder(e);
    }

    let summary = pipeline.ingest(&graph).await?;
    println!("{}", serde_json::to_string_pretty(&summary)?);
    Ok(())
}

async fn cmd_ingest_document(
    config_path: &std::path::Path,
    path: PathBuf,
    batch_size: usize,
) -> Result<()> {
    let _ = (config_path, path, batch_size);
    Err(crate::error::Error::Ingest(
        crate::ingest::IngestError::Type(
            "document ingest was removed; build a graph::Graph and call Pipeline::ingest(&graph)"
                .into(),
        ),
    ))
}

async fn cmd_ingest_document_cypher(
    config_path: &std::path::Path,
    path: PathBuf,
    batch_size: usize,
) -> Result<()> {
    let _ = (config_path, path, batch_size);
    Err(crate::error::Error::Ingest(crate::ingest::IngestError::Type(
        "document ingest-cypher was removed; build a graph::Graph and call Pipeline::compile_insert(&graph)"
            .into(),
    )))
}

async fn cmd_knowledge_prompt(
    path: PathBuf,
    entity_types: Vec<String>,
    relation_types: Vec<String>,
    output: Option<PathBuf>,
) -> Result<()> {
    use crate::promptgen::knowledge::{
        default_entity_types, default_relation_types, generate_knowledge_extract_prompt,
    };
    use tokio::io::AsyncReadExt;

    // Read the fragment from a file or stdin. `-` means stdin so the
    // command composes cleanly with `cat doc.txt | linguagraph
    // knowledge-prompt -`.
    let fragment = if path == std::path::Path::new("-") {
        let mut buf = String::new();
        tokio::io::stdin().read_to_string(&mut buf).await?;
        buf
    } else {
        fs::read_to_string(&path).await?
    };

    let opts = KnowledgeExtractOptions {
        entity_types: if entity_types.is_empty() {
            default_entity_types()
        } else {
            entity_types.into_iter().map(EntityTypeSpec::new).collect()
        },
        relation_types: if relation_types.is_empty() {
            default_relation_types()
        } else {
            relation_types
                .into_iter()
                .map(RelationTypeSpec::new)
                .collect()
        },
    };
    let prompt = generate_knowledge_extract_prompt(&fragment, &opts);
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
            metadata: Default::default(),
            graph_specification: Default::default(),
            types: Default::default(),
        },
    }
}
