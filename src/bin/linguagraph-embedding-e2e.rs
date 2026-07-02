use std::collections::BTreeMap;
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{anyhow, bail, Context};
use clap::Parser;
use linguagraph::embeddings::Embedder;
use linguagraph::embeddings::LlamaEmbedder;
use linguagraph::graph::{build_canonical_text, GraphBuilder};
use serde_json::Value;

#[derive(Debug, Parser)]
#[command(
    name = "linguagraph-embedding-e2e",
    about = "Isolated e2e check for canonical-text embeddings"
)]
struct Args {
    /// TOML config used to resolve the embedding model path.
    #[arg(long, short = 'c', default_value = "config.e2e.toml")]
    config: PathBuf,

    /// Graph JSON fixture produced from the cameras dump.
    #[arg(long, default_value = "target/e2e/cameras_1k.graph.json")]
    graph: PathBuf,

    /// Minimum cosine margin between similar and dissimilar pairs.
    #[arg(long, default_value_t = 0.05)]
    min_margin: f32,

    /// Allowed absolute deviation from unit length.
    #[arg(long, default_value_t = 0.05)]
    norm_epsilon: f32,

    /// Verify that batched and single-item embedding calls produce
    /// nearly identical vectors for the selected sample set.
    #[arg(long, default_value_t = true)]
    check_batch_consistency: bool,

    /// Maximum per-vector absolute difference allowed between batch
    /// and single-item embedding results.
    #[arg(long, default_value_t = 1e-2)]
    batch_abs_epsilon: f32,

    /// Minimum cosine similarity required between batch and single-item
    /// vectors for each sample.
    #[arg(long, default_value_t = 0.995)]
    batch_cosine_min: f32,

    /// Check a long synthetic camera-style text of roughly this many
    /// whitespace-delimited words.
    #[arg(long, default_value_t = 700)]
    long_text_words: usize,

    /// Disable the long-text regression case.
    #[arg(long, default_value_t = true)]
    check_long_text: bool,

    /// Minimum cosine similarity required between batch and single-item
    /// vectors for the long-text case.
    #[arg(long, default_value_t = 0.995)]
    long_text_cosine_min: f32,

    /// Maximum per-vector absolute difference allowed for the long-text
    /// batch/single comparison.
    #[arg(long, default_value_t = 1e-2)]
    long_text_abs_epsilon: f32,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    init_tracing();
    let args = Args::parse();

    let cfg = linguagraph::config::load(&args.config)
        .await
        .with_context(|| format!("load config {}", args.config.display()))?;
    let model_path = cfg
        .types
        .get("SemanticText")
        .and_then(|t| t.embedding_model.clone())
        .or_else(|| cfg.ontology_catalog.embedding_model.clone())
        .ok_or_else(|| anyhow!("no embedding_model configured in {}", args.config.display()))?;

    println!("config={}", args.config.display());
    println!("graph={}", args.graph.display());
    println!("model={model_path}");

    let graph_raw = tokio::fs::read_to_string(&args.graph)
        .await
        .with_context(|| format!("read graph {}", args.graph.display()))?;
    let graph = GraphBuilder::from_json(&graph_raw)
        .map_err(|e| anyhow!("parse graph {}: {e}", args.graph.display()))?;

    let samples = select_camera_samples(&graph)?;
    println!(
        "samples similar_left={} similar_right={} dissimilar={}",
        samples.similar_left.label, samples.similar_right.label, samples.dissimilar.label
    );
    println!(
        "place_id={} vs {}",
        samples.similar_left.place_id, samples.dissimilar.place_id
    );

    let load_started = Instant::now();
    let embedder = LlamaEmbedder::load(&model_path)?;
    println!(
        "model_loaded dim={} elapsed_ms={}",
        embedder.dim(),
        load_started.elapsed().as_millis()
    );

    let inputs = [
        samples.similar_left.canonical.as_str(),
        samples.similar_right.canonical.as_str(),
        samples.dissimilar.canonical.as_str(),
    ];
    println!("embedding_inputs={}", inputs.len());

    let started = Instant::now();
    let vectors = embedder.embed_batch(&inputs)?;
    let elapsed_ms = started.elapsed().as_millis();

    if vectors.len() != inputs.len() {
        bail!(
            "embedder returned {} vectors for {} inputs",
            vectors.len(),
            inputs.len()
        );
    }

    println!(
        "embedded vectors={} elapsed_ms={} per_item_ms={:.2}",
        vectors.len(),
        elapsed_ms,
        started.elapsed().as_secs_f64() * 1000.0 / vectors.len() as f64
    );

    validate_vector_shape(&vectors, embedder.dim(), args.norm_epsilon)?;

    if args.check_batch_consistency {
        let single_started = Instant::now();
        let mut single_vectors = Vec::with_capacity(inputs.len());
        for input in inputs {
            single_vectors.push(embedder.embed(input)?);
        }
        println!(
            "single_item_vectors={} elapsed_ms={}",
            single_vectors.len(),
            single_started.elapsed().as_millis()
        );
        validate_batch_consistency(
            &vectors,
            &single_vectors,
            args.batch_abs_epsilon,
            args.batch_cosine_min,
        )?;
    }

    if args.check_long_text {
        let long_text = build_long_text(
            samples.similar_left.canonical.as_str(),
            args.long_text_words,
        );
        let long_inputs = [long_text.as_str()];
        println!(
            "long_text words={} chars={}",
            long_text.split_whitespace().count(),
            long_text.len()
        );

        let long_started = Instant::now();
        let long_batch = embedder.embed_batch(&long_inputs)?;
        println!(
            "long_text_batch_vectors={} elapsed_ms={}",
            long_batch.len(),
            long_started.elapsed().as_millis()
        );
        validate_vector_shape(&long_batch, embedder.dim(), args.norm_epsilon)?;

        let long_single_started = Instant::now();
        let long_single = [embedder.embed(&long_text)?];
        println!(
            "long_text_single_vectors={} elapsed_ms={}",
            long_single.len(),
            long_single_started.elapsed().as_millis()
        );
        validate_batch_consistency(
            &long_batch,
            &long_single,
            args.long_text_abs_epsilon,
            args.long_text_cosine_min,
        )?;
    }

    let sim_same = cosine_similarity(&vectors[0], &vectors[1]);
    let sim_diff = cosine_similarity(&vectors[0], &vectors[2]);
    println!("cosine_similar={sim_same:.6}");
    println!("cosine_dissimilar={sim_diff:.6}");
    println!("margin={:.6}", sim_same - sim_diff);

    if sim_same <= sim_diff + args.min_margin {
        bail!(
            "expected similar pair to beat dissimilar pair by at least {:.4}, got {:.6} vs {:.6}",
            args.min_margin,
            sim_same,
            sim_diff
        );
    }

    println!("embedding_e2e=passed");
    Ok(())
}

#[derive(Debug, Clone)]
struct Sample {
    label: String,
    place_id: String,
    canonical: String,
}

struct Samples {
    similar_left: Sample,
    similar_right: Sample,
    dissimilar: Sample,
}

fn select_camera_samples(graph: &linguagraph::graph::Graph) -> anyhow::Result<Samples> {
    let mut by_place: BTreeMap<String, Vec<Sample>> = BTreeMap::new();
    let mut all_samples = Vec::new();

    for entity in graph.entities() {
        if entity.r#type != "Camera" {
            continue;
        }

        let Some(canonical_prop) = entity.properties.get("_canonical") else {
            continue;
        };
        let Some(name_prop) = entity.properties.get("name") else {
            continue;
        };
        let Some(place_prop) = entity.properties.get("place_id") else {
            continue;
        };

        let canonical = canonical_prop
            .value
            .as_str()
            .ok_or_else(|| anyhow!("_canonical must be a string"))?
            .to_string();

        let place_id = json_value_text(&place_prop.value);
        let name = json_value_text(&name_prop.value);

        let expected = {
            let mut props = BTreeMap::new();
            for (key, prop) in &entity.properties {
                if key == "_canonical" || key == "id" {
                    continue;
                }
                props.insert(key.clone(), prop.value.clone());
            }
            build_canonical_text(&entity.r#type, &props.into_iter().collect())
        };

        if expected != canonical {
            bail!(
                "canonical mismatch for {}: expected {:?}, got {:?}",
                name,
                expected,
                canonical
            );
        }

        let sample = Sample {
            label: name,
            place_id: place_id.clone(),
            canonical,
        };
        by_place.entry(place_id).or_default().push(sample.clone());
        all_samples.push(sample);
    }

    let Some((_, similar_group)) = by_place.iter().find(|(_, items)| items.len() >= 2) else {
        bail!("need at least one place with two camera records for a similarity check");
    };

    let similar_left = similar_group[0].clone();
    let similar_right = similar_group[1].clone();
    let dissimilar = all_samples
        .into_iter()
        .find(|sample| sample.place_id != similar_left.place_id)
        .ok_or_else(|| anyhow!("need a camera from a different place for a dissimilar check"))?;

    Ok(Samples {
        similar_left,
        similar_right,
        dissimilar,
    })
}

fn validate_vector_shape(
    vectors: &[Vec<f32>],
    dim: usize,
    norm_epsilon: f32,
) -> anyhow::Result<()> {
    for (idx, vec) in vectors.iter().enumerate() {
        if vec.len() != dim {
            bail!("vector {idx} has dim {}, expected {dim}", vec.len());
        }
        let norm = l2_norm(vec);
        if (norm - 1.0).abs() > norm_epsilon {
            bail!("vector {idx} has norm {norm:.6}, expected around 1.0");
        }
    }
    Ok(())
}

fn validate_batch_consistency(
    batched: &[Vec<f32>],
    single: &[Vec<f32>],
    abs_epsilon: f32,
    cosine_min: f32,
) -> anyhow::Result<()> {
    if batched.len() != single.len() {
        bail!(
            "batch/single length mismatch: batched={}, single={}",
            batched.len(),
            single.len()
        );
    }

    for (idx, (batch_vec, single_vec)) in batched.iter().zip(single.iter()).enumerate() {
        if batch_vec.len() != single_vec.len() {
            bail!(
                "vector {idx} dim mismatch: batched={}, single={}",
                batch_vec.len(),
                single_vec.len()
            );
        }

        let cosine = cosine_similarity(batch_vec, single_vec);
        let max_abs_diff = batch_vec
            .iter()
            .zip(single_vec.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);

        println!("batch_consistency idx={idx} cosine={cosine:.6} max_abs_diff={max_abs_diff:.6}");

        if cosine < cosine_min {
            bail!("vector {idx} batch/single cosine {cosine:.6} below minimum {cosine_min:.6}");
        }
        if max_abs_diff > abs_epsilon {
            bail!(
                "vector {idx} batch/single max abs diff {max_abs_diff:.6} above epsilon {abs_epsilon:.6}"
            );
        }
    }

    Ok(())
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn l2_norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

fn json_value_text(value: &Value) -> String {
    value
        .as_str()
        .map(ToOwned::to_owned)
        .unwrap_or_else(|| value.to_string())
}

fn build_long_text(base: &str, target_words: usize) -> String {
    let mut words = Vec::new();
    let mut idx = 0usize;
    while words.len() < target_words {
        idx += 1;
        words.extend(base.split_whitespace().map(str::to_owned));
        words.push("повтор".to_string());
        words.push(idx.to_string());
    }
    words.join(" ")
}

fn init_tracing() {
    use tracing_subscriber::{fmt, EnvFilter};
    let filter =
        EnvFilter::try_from_env("LINGUAGRAPH_LOG").unwrap_or_else(|_| EnvFilter::new("info"));
    let _ = fmt().with_env_filter(filter).with_target(false).try_init();
}
