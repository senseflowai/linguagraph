//! Clap-based CLI. Subcommands map 1:1 to pipeline stages so users can stop
//! at any layer for inspection.

use std::path::PathBuf;
use std::sync::Arc;

use clap::{Parser, Subcommand, ValueEnum};
use tokio::fs;
use crate::ast::Literal;
use crate::config::{self, Config};
use crate::core::Pipeline;
use crate::db::{introspect, GraphClient, MemgraphClient};
use crate::dsl;
use crate::embeddings::{self, SharedEmbedder};
use crate::error::Result;
use crate::mapper::Mapping;
use crate::metadata::{FileMetadataStore, MetadataStore};
use crate::prompt::{self, GraphSchema, PromptOptions};
use crate::types::{self, SharedRegistry};

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
    },
    /// Compile and execute a DSL file against the configured database.
    Run {
        path: PathBuf,
    },
    /// Print a schema-aware system prompt for an LLM.
    Prompt {
        /// Path to a schema JSON file. If omitted, the live database is queried.
        #[arg(long)]
        schema: Option<PathBuf>,
        /// Skip the worked examples in the output.
        #[arg(long)]
        no_examples: bool,
        /// Skip annotating the prompt with cached property descriptions.
        #[arg(long)]
        no_metadata: bool,
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
    /// Compile a (data, mapping) pair and execute the ingestion against
    /// the configured database. Prints a summary of nodes/relationships
    /// MERGE'd.
    Ingest {
        /// Path to the raw data JSON file.
        data: PathBuf,
        /// Path to the mapping JSON file.
        mapping: PathBuf,
        /// Maximum rows per UNWIND batch.
        #[arg(long, default_value_t = 1000)]
        batch_size: usize,
    },
    /// Like `ingest` but prints the generated Cypher batches instead of
    /// executing them. Useful for inspection and CI snapshots.
    IngestCypher {
        data: PathBuf,
        mapping: PathBuf,
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
}

pub async fn run(cli: Cli) -> Result<()> {
    match cli.command {
        Command::Dsl { path } => cmd_dsl(path).await,
        Command::Cypher { path } => cmd_cypher(&cli.config, path).await,
        Command::Run { path } => cmd_run(&cli.config, path).await,
        Command::Prompt { schema, no_examples, no_metadata } => {
            cmd_prompt(&cli.config, schema, no_examples, no_metadata).await
        }
        Command::Schema { sample_size, format, output, no_examples } => {
            cmd_schema(&cli.config, sample_size, format, output, no_examples).await
        }
        Command::Ingest { data, mapping, batch_size } => {
            cmd_ingest(&cli.config, data, mapping, batch_size).await
        }
        Command::IngestCypher { data, mapping, batch_size } => {
            cmd_ingest_cypher(&cli.config, data, mapping, batch_size).await
        }
        Command::Query { path } => cmd_query(&cli.config, path).await,
        Command::GeneratePrompt { path, hints, prefer, no_examples, no_summary } => {
            cmd_generate_prompt(&cli.config, path, hints, prefer, no_examples, no_summary).await
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
                let emb: Vec<f32> = vec.iter()
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

async fn cmd_prompt(
    config_path: &std::path::Path,
    schema_path: Option<PathBuf>,
    no_examples: bool,
    no_metadata: bool,
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
    let property_metadata = if no_metadata {
        None
    } else {
        let store = FileMetadataStore::new(&cfg.metadata.cache_path);
        let m = store.load().await?;
        if m.is_empty() { None } else { Some(m) }
    };
    let (registry, _) = build_registry(&cfg)?;
    let registry_for_prompt = (*registry).clone();
    let opts = PromptOptions {
        include_examples: !no_examples,
        property_metadata,
        type_registry: if registry_for_prompt.is_empty() {
            None
        } else {
            Some(registry_for_prompt)
        },
        ..PromptOptions::default()
    };
    let prompt = prompt::generate_system_prompt(&schema, &opts);
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
        introspect::IntrospectOptions { sample_size },
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

async fn cmd_ingest(
    config_path: &std::path::Path,
    data: PathBuf,
    mapping: PathBuf,
    batch_size: usize,
) -> Result<()> {
    let cfg = config::load(config_path).await?;
    let mapping = Mapping::load(&mapping).await?;
    let raw = fs::read_to_string(&data).await?;
    let value: serde_json::Value = serde_json::from_str(&raw)?;

    let client = MemgraphClient::connect(&cfg.database).await?;
    let store: Arc<dyn MetadataStore> =
        Arc::new(FileMetadataStore::new(&cfg.metadata.cache_path));
    let (registry, embedder) = build_registry(&cfg)?;
    let mut pipeline = Pipeline::new(Arc::new(client), &cfg)
        .with_ingest_batch_size(batch_size)
        .with_metadata_store(store)
        .with_registry(registry);
    if let Some(e) = embedder {
        pipeline = pipeline.with_embedder(e);
    }
    let summary = pipeline.ingest(&mapping, &value).await?;
    println!("{}", serde_json::to_string_pretty(&summary)?);
    Ok(())
}

async fn cmd_ingest_cypher(
    config_path: &std::path::Path,
    data: PathBuf,
    mapping: PathBuf,
    batch_size: usize,
) -> Result<()> {
    let cfg = load_config_or_default(config_path).await;
    let mapping = Mapping::load(&mapping).await?;
    let raw = fs::read_to_string(&data).await?;
    let value: serde_json::Value = serde_json::from_str(&raw)?;

    let (registry, embedder) = build_registry(&cfg)?;
    let mut pipeline = Pipeline::new(Arc::new(crate::db::MockClient::new()), &cfg)
        .with_ingest_batch_size(batch_size)
        .with_registry(registry);
    if let Some(e) = embedder {
        pipeline = pipeline.with_embedder(e);
    }
    let batches = pipeline.compile_insert(&mapping, &value)?;
    for (i, q) in batches.iter().enumerate() {
        println!("-- Batch {i} --\n{}\n-- Parameters --", q.text);
        for (k, v) in &q.params {
            println!("${k} = {}", serde_json::to_string(v)?);
        }
        println!();
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
            types: Default::default(),
        },
    }
}
