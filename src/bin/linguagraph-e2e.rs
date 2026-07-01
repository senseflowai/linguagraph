use std::path::PathBuf;

use clap::Parser;
use linguagraph::e2e::{run_suite, E2eRunOptions};

#[derive(Debug, Parser)]
#[command(
    name = "linguagraph-e2e",
    about = "Run full LLM -> DSL -> Memgraph e2e suites"
)]
struct Args {
    /// TOML configuration used for Memgraph, LLM, embedding, and reranking.
    #[arg(long, short = 'c', default_value = "config.e2e.toml")]
    config: PathBuf,

    /// Suite JSON. Relative graph/ontology/questions paths resolve from its directory.
    #[arg(long, short = 's', default_value = "examples/e2e/camera.suite.json")]
    suite: PathBuf,

    /// Override the graph JSON path declared by the suite.
    #[arg(long)]
    graph: Option<PathBuf>,

    /// Override the ontology JSON path declared by the suite.
    #[arg(long)]
    ontology: Option<PathBuf>,

    /// Override the questions JSON path declared by the suite.
    #[arg(long)]
    questions: Option<PathBuf>,

    /// Write a machine-readable JSON report.
    #[arg(long)]
    report: Option<PathBuf>,

    /// Override the suite prefix_label/prefix_index.
    #[arg(long)]
    prefix: Option<String>,

    /// Delete suite nodes/vectors after the run.
    #[arg(long)]
    cleanup_after: bool,

    /// Keep data after the run even if the suite enables cleanup_after.
    #[arg(long, conflicts_with = "cleanup_after")]
    keep_data: bool,

    /// Override [llm].base_url.
    #[arg(long)]
    llm_base_url: Option<String>,

    /// Override [llm].model.
    #[arg(long)]
    llm_model: Option<String>,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    init_tracing();
    let args = Args::parse();
    let report = run_suite(E2eRunOptions {
        config_path: args.config,
        suite_path: args.suite,
        graph_path: args.graph,
        ontology_path: args.ontology,
        questions_path: args.questions,
        report_path: args.report,
        prefix: args.prefix,
        cleanup_after: args.cleanup_after.then_some(true),
        keep_data: args.keep_data,
        llm_base_url: args.llm_base_url,
        llm_model: args.llm_model,
    })
    .await?;

    println!(
        "suite={} prefix={} passed={}/{} failed={}",
        report.suite, report.prefix_label, report.passed, report.total, report.failed
    );
    for case in &report.cases {
        if case.passed {
            println!("PASS {}", case.id);
        } else {
            println!("FAIL {}", case.id);
            for error in &case.errors {
                println!("  - {error}");
            }
        }
    }

    if report.is_success() {
        Ok(())
    } else {
        anyhow::bail!("{} e2e case(s) failed", report.failed)
    }
}

fn init_tracing() {
    use tracing_subscriber::{fmt, EnvFilter};
    let filter =
        EnvFilter::try_from_env("LINGUAGRAPH_LOG").unwrap_or_else(|_| EnvFilter::new("info"));
    let _ = fmt().with_env_filter(filter).with_target(false).try_init();
}
