use std::path::PathBuf;
use std::time::Instant;

use anyhow::{anyhow, Context};
use clap::Parser;
use linguagraph::config;
use linguagraph::embeddings::Embedder;
use linguagraph::embeddings::LlamaEmbedder;

#[derive(Debug, Parser)]
#[command(
    name = "linguagraph-embed-bench",
    about = "Measure llama embedding latency on a small set of similar texts"
)]
struct Args {
    /// TOML config used to resolve the embedding model path.
    #[arg(long, short = 'c', default_value = "config.e2e.toml")]
    config: PathBuf,

    /// Comma-separated batch sizes to benchmark.
    #[arg(long, default_value = "1,2,5,10")]
    counts: String,

    /// Base text used to generate similar variants.
    #[arg(
        long,
        default_value = "Камера на проспекте Достык, Алматы, тестовый маршрут"
    )]
    base_text: String,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    init_tracing();
    let args = Args::parse();

    let cfg = config::load(&args.config)
        .await
        .with_context(|| format!("load config {}", args.config.display()))?;
    let model_path = cfg
        .types
        .get("SemanticText")
        .and_then(|t| t.embedding_model.clone())
        .or_else(|| cfg.ontology_catalog.embedding_model.clone())
        .ok_or_else(|| anyhow!("no embedding_model configured in {}", args.config.display()))?;

    println!("config={}", args.config.display());
    println!("model={model_path}");

    let load_started = Instant::now();
    let embedder = LlamaEmbedder::load(&model_path)?;
    println!(
        "model_loaded dim={} elapsed_ms={}",
        embedder.dim(),
        load_started.elapsed().as_millis()
    );

    let counts = parse_counts(&args.counts)?;
    println!("counts={counts:?}");
    println!("base_text={}", args.base_text);

    for count in counts {
        let texts = build_similar_texts(&args.base_text, count);
        let refs: Vec<&str> = texts.iter().map(String::as_str).collect();
        let started = Instant::now();
        let vecs = embedder.embed_batch(&refs)?;
        let elapsed = started.elapsed();
        println!(
            "count={count} vectors={} elapsed_ms={} per_item_ms={:.2}",
            vecs.len(),
            elapsed.as_millis(),
            elapsed.as_secs_f64() * 1000.0 / count as f64
        );
    }

    Ok(())
}

fn parse_counts(raw: &str) -> anyhow::Result<Vec<usize>> {
    let mut counts = Vec::new();
    for part in raw.split(',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        let value: usize = part
            .parse()
            .with_context(|| format!("invalid count `{part}`"))?;
        if value == 0 {
            anyhow::bail!("counts must be > 0");
        }
        counts.push(value);
    }
    if counts.is_empty() {
        anyhow::bail!("at least one count is required");
    }
    Ok(counts)
}

fn build_similar_texts(base: &str, count: usize) -> Vec<String> {
    (0..count)
        .map(|idx| format!("{base} | вариант {}", idx + 1))
        .collect()
}

fn init_tracing() {
    use tracing_subscriber::{fmt, EnvFilter};
    let filter =
        EnvFilter::try_from_env("LINGUAGRAPH_LOG").unwrap_or_else(|_| EnvFilter::new("info"));
    let _ = fmt().with_env_filter(filter).with_target(false).try_init();
}
