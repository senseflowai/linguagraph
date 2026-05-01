//! Clap-based CLI. Subcommands map 1:1 to pipeline stages so users can stop
//! at any layer for inspection.

use std::path::PathBuf;
use std::sync::Arc;

use clap::{Parser, Subcommand};
use tokio::fs;

use crate::config::{self, Config};
use crate::core::Pipeline;
use crate::db::{GraphClient, MemgraphClient};
use crate::dsl;
use crate::error::Result;
use crate::prompt::{self, GraphSchema, PromptOptions};

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
    },
    /// Fetch and print the live graph schema as JSON.
    Schema,
}

pub async fn run(cli: Cli) -> Result<()> {
    match cli.command {
        Command::Dsl { path } => cmd_dsl(path).await,
        Command::Cypher { path } => cmd_cypher(&cli.config, path).await,
        Command::Run { path } => cmd_run(&cli.config, path).await,
        Command::Prompt { schema, no_examples } => cmd_prompt(&cli.config, schema, no_examples).await,
        Command::Schema => cmd_schema(&cli.config).await,
    }
}

async fn cmd_dsl(path: PathBuf) -> Result<()> {
    let q = dsl::parse(&path).await?;
    println!("{}", serde_json::to_string_pretty(&q)?);
    Ok(())
}

async fn cmd_cypher(config_path: &std::path::Path, path: PathBuf) -> Result<()> {
    let cfg = load_config_or_default(config_path).await;
    let pipeline = Pipeline::new(Arc::new(crate::db::MockClient::new()), &cfg);
    let dsl_query = dsl::parse(&path).await?;
    let cypher = pipeline.compile(dsl_query)?;
    println!("-- Cypher --\n{}", cypher.text);
    println!("\n-- Parameters --");
    for (k, v) in &cypher.params {
        println!("${k} = {}", serde_json::to_string(v)?);
    }
    Ok(())
}

async fn cmd_run(config_path: &std::path::Path, path: PathBuf) -> Result<()> {
    let cfg = config::load(config_path).await?;
    let client = MemgraphClient::connect(&cfg.database).await?;
    let pipeline = Pipeline::new(Arc::new(client), &cfg);
    let dsl_query = dsl::parse(&path).await?;
    let result = pipeline.run(dsl_query).await?;
    println!("{}", serde_json::to_string_pretty(&result)?);
    Ok(())
}

async fn cmd_prompt(
    config_path: &std::path::Path,
    schema_path: Option<PathBuf>,
    no_examples: bool,
) -> Result<()> {
    let schema = match schema_path {
        Some(p) => {
            let raw = fs::read_to_string(&p).await?;
            serde_json::from_str::<GraphSchema>(&raw)?
        }
        None => {
            let cfg = config::load(config_path).await?;
            let client: Arc<dyn GraphClient> = Arc::new(MemgraphClient::connect(&cfg.database).await?);
            client.schema().await?
        }
    };
    let opts = PromptOptions { include_examples: !no_examples, ..PromptOptions::default() };
    let prompt = prompt::generate_system_prompt(&schema, &opts);
    println!("{prompt}");
    Ok(())
}

async fn cmd_schema(config_path: &std::path::Path) -> Result<()> {
    let cfg = config::load(config_path).await?;
    let client = MemgraphClient::connect(&cfg.database).await?;
    let schema = client.schema().await?;
    println!("{}", serde_json::to_string_pretty(&schema)?);
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
            },
            llm: Default::default(),
            query: Default::default(),
        },
    }
}
