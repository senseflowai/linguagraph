//! E2E test-kit for running the full LLM -> DSL -> Memgraph pipeline.
//!
//! This module intentionally sits above [`crate::core::Pipeline`]. It wires the
//! same production pieces the CLI uses, but adds fixture loading, isolated
//! prefixes, answer validation, and a machine-readable report.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{anyhow, bail, Context};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value as JsonValue};
use tokio::fs;

use crate::builder::CypherQuery;
use crate::config::{self, Config};
use crate::core::Pipeline;
use crate::db::{GraphClient, MemgraphClient, QueryResult, Value as DbValue};
use crate::dsl::{self, DslQuery, TraversalQuery};
use crate::embeddings::{
    self, EmbeddingIndex, InMemoryEmbeddingStore, SharedEmbedder, SharedEmbeddingStore,
};
use crate::graph::{Graph, GraphBuilder, OntologyCatalog, OntologyCatalogStorage, PrimaryKey};
use crate::llm::LlmClient;
use crate::prompt::{self, PromptOptions};
use crate::types::{self, SharedRegistry};

/// Runtime options supplied by the `linguagraph-e2e` binary.
#[derive(Debug, Clone)]
pub struct E2eRunOptions {
    pub config_path: PathBuf,
    pub suite_path: PathBuf,
    pub graph_path: Option<PathBuf>,
    pub ontology_path: Option<PathBuf>,
    pub questions_path: Option<PathBuf>,
    pub case_id: Option<String>,
    pub report_path: Option<PathBuf>,
    pub prefix: Option<String>,
    pub cleanup_after: Option<bool>,
    pub keep_data: bool,
    pub include_embeddings_in_report: Option<bool>,
    pub llm_base_url: Option<String>,
    pub llm_model: Option<String>,
}

/// Summary returned by [`run_suite`].
#[derive(Debug, Clone, Serialize)]
pub struct E2eReport {
    pub suite: String,
    pub prefix_label: String,
    pub prefix_index: String,
    pub graph_path: PathBuf,
    pub ontology_path: PathBuf,
    pub questions_path: PathBuf,
    pub total: usize,
    pub passed: usize,
    pub failed: usize,
    pub include_embeddings_in_report: bool,
    pub cases: Vec<E2eCaseReport>,
}

impl E2eReport {
    pub fn is_success(&self) -> bool {
        self.failed == 0
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct E2eCaseReport {
    pub id: String,
    pub question: String,
    pub passed: bool,
    pub errors: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dsl: Option<JsonValue>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub traversal: Option<JsonValue>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cypher: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cypher_params: Option<BTreeMap<String, JsonValue>>,
    pub row_count: usize,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub rows: Vec<BTreeMap<String, JsonValue>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub answer: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub judge: Option<JudgeReport>,
}

#[derive(Debug, Clone, Serialize)]
pub struct JudgeReport {
    pub passed: bool,
    pub reason: String,
}

#[derive(Debug, Clone, Deserialize)]
struct SuiteFile {
    #[serde(default)]
    name: Option<String>,
    graph: PathBuf,
    ontology: PathBuf,
    questions: PathBuf,
    #[serde(default)]
    settings: SuiteSettings,
}

#[derive(Debug, Clone, Deserialize)]
struct SuiteSettings {
    #[serde(default)]
    prefix: Option<String>,
    #[serde(default = "default_batch_size")]
    batch_size: usize,
    #[serde(default = "default_true")]
    cleanup_before: bool,
    #[serde(default)]
    cleanup_after: bool,
    #[serde(default = "default_true")]
    cleanup_vectors: bool,
    #[serde(default = "default_true")]
    answer_with_llm: bool,
    #[serde(default)]
    judge_with_llm: bool,
    /// Include raw embedding vectors in the machine-readable report. Disabled
    /// by default because vectors make reports huge and are rarely useful for
    /// pass/fail analysis.
    #[serde(
        default,
        alias = "report_embeddings",
        alias = "write_embeddings",
        alias = "include_embedding_in_report"
    )]
    include_embeddings_in_report: bool,
    #[serde(default = "default_max_repairs")]
    max_repairs: usize,
    #[serde(default)]
    llm_base_url: Option<String>,
    #[serde(default)]
    llm_model: Option<String>,
    #[serde(default)]
    indexes: Vec<IndexSpec>,
}

impl Default for SuiteSettings {
    fn default() -> Self {
        Self {
            prefix: None,
            batch_size: default_batch_size(),
            cleanup_before: true,
            cleanup_after: false,
            cleanup_vectors: true,
            answer_with_llm: true,
            judge_with_llm: false,
            include_embeddings_in_report: false,
            max_repairs: default_max_repairs(),
            llm_base_url: None,
            llm_model: None,
            indexes: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
struct IndexSpec {
    label: String,
    property: String,
}

#[derive(Debug, Clone, Deserialize)]
struct QuestionFile {
    questions: Vec<QuestionCase>,
}

#[derive(Debug, Clone, Deserialize)]
struct QuestionCase {
    id: String,
    question: String,
    /// Optional static DSL. When omitted, the LLM must produce DSL from
    /// `question` and the live schema prompt.
    #[serde(default)]
    dsl: Option<JsonValue>,
    /// Optional high-level text traversal query. When present, the case uses
    /// `Pipeline::run_traversal` instead of the natural-language-to-DSL path.
    #[serde(default)]
    traversal: Option<TraversalQuery>,
    #[serde(default)]
    validation: ValidationSpec,
}

#[derive(Debug, Clone, Default, Deserialize)]
struct ValidationSpec {
    #[serde(default)]
    row_count: Option<RowCountSpec>,
    #[serde(default)]
    rows: Vec<ExpectedRow>,
    #[serde(default)]
    columns: Vec<String>,
    #[serde(default)]
    contains: Vec<CellExpectation>,
    #[serde(default)]
    not_contains: Vec<CellExpectation>,
    #[serde(default)]
    numbers: Vec<NumericExpectation>,
    #[serde(default)]
    answer_contains: Vec<String>,
    #[serde(default)]
    dsl_expect: Option<DslValidationSpec>,
    #[serde(default)]
    judge: Option<JudgeSpec>,
}

#[derive(Debug, Clone, Default, Deserialize)]
struct RowCountSpec {
    #[serde(default)]
    exact: Option<usize>,
    #[serde(default)]
    min: Option<usize>,
    #[serde(default)]
    max: Option<usize>,
}

#[derive(Debug, Clone, Deserialize)]
struct CellExpectation {
    column: String,
    value: JsonValue,
    #[serde(default)]
    mode: MatchMode,
}

#[derive(Debug, Clone, Deserialize)]
struct ExpectedRow {
    #[serde(default)]
    fields: BTreeMap<String, JsonValue>,
}

#[derive(Debug, Clone, Copy, Default, Deserialize)]
#[serde(rename_all = "snake_case")]
enum MatchMode {
    #[default]
    Exact,
    Contains,
}

#[derive(Debug, Clone, Deserialize)]
struct NumericExpectation {
    column: String,
    op: NumericOp,
    value: f64,
}

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(rename_all = "snake_case")]
enum NumericOp {
    Eq,
    Neq,
    Gt,
    Gte,
    Lt,
    Lte,
}

#[derive(Debug, Clone, Deserialize)]
struct JudgeSpec {
    expected: String,
    #[serde(default = "default_true")]
    require_pass: bool,
}

#[derive(Debug, Clone, Default, Deserialize)]
struct DslValidationSpec {
    #[serde(default)]
    start_label: Option<String>,
    #[serde(default)]
    start_alias: Option<String>,
    #[serde(default)]
    required_filter_fields: Vec<String>,
    #[serde(default)]
    required_filter_ops: Vec<String>,
    #[serde(default)]
    forbidden_filter_ops: Vec<String>,
    #[serde(default)]
    required_return_fields: Vec<String>,
    #[serde(default)]
    required_traversal_labels: Vec<String>,
    #[serde(default)]
    min_filters: Option<usize>,
    #[serde(default)]
    max_filters: Option<usize>,
    #[serde(default)]
    min_returns: Option<usize>,
    #[serde(default)]
    max_returns: Option<usize>,
}

fn default_true() -> bool {
    true
}

fn default_batch_size() -> usize {
    1000
}

fn default_max_repairs() -> usize {
    1
}

/// Run one E2E suite and return a structured report. The function returns
/// `Ok(report)` even when individual cases fail; transport/setup errors are
/// returned as `Err`.
pub async fn run_suite(opts: E2eRunOptions) -> anyhow::Result<E2eReport> {
    let suite = load_suite(&opts.suite_path).await?;
    let suite_dir = opts
        .suite_path
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."));
    let suite_name = suite
        .name
        .clone()
        .unwrap_or_else(|| suite_name_from_path(&opts.suite_path));

    let graph_path = resolve_suite_path(&suite_dir, opts.graph_path.unwrap_or(suite.graph));
    let ontology_path =
        resolve_suite_path(&suite_dir, opts.ontology_path.unwrap_or(suite.ontology));
    let questions_path =
        resolve_suite_path(&suite_dir, opts.questions_path.unwrap_or(suite.questions));

    let mut settings = suite.settings;
    if let Some(prefix) = opts.prefix {
        settings.prefix = Some(prefix);
    }
    if let Some(cleanup_after) = opts.cleanup_after {
        settings.cleanup_after = cleanup_after;
    }
    if opts.keep_data {
        settings.cleanup_after = false;
    }
    if let Some(include_embeddings) = opts.include_embeddings_in_report {
        settings.include_embeddings_in_report = include_embeddings;
    }
    if let Some(base_url) = opts.llm_base_url {
        settings.llm_base_url = Some(base_url);
    }
    if let Some(model) = opts.llm_model {
        settings.llm_model = Some(model);
    }

    let prefix = settings
        .prefix
        .clone()
        .unwrap_or_else(|| unique_prefix(&suite_name));
    let prefix = sanitize_ident(&prefix)
        .ok_or_else(|| anyhow!("suite prefix `{prefix}` is not a valid Cypher identifier"))?;

    let mut cfg = config::load(&opts.config_path)
        .await
        .with_context(|| format!("load config {}", opts.config_path.display()))?;
    if let Some(base_url) = &settings.llm_base_url {
        cfg.llm.base_url = base_url.clone();
    }
    if let Some(model) = &settings.llm_model {
        cfg.llm.model = model.clone();
    }

    let catalog = load_catalog(&ontology_path).await?;
    let catalog = Arc::new(catalog);
    let questions = load_questions(&questions_path).await?;
    let questions = if let Some(case_id) = &opts.case_id {
        let filtered: Vec<_> = questions
            .into_iter()
            .filter(|case| case.id == *case_id)
            .collect();
        if filtered.is_empty() {
            return Err(anyhow!(
                "question id `{}` not found in {}",
                case_id,
                questions_path.display()
            ));
        }
        filtered
    } else {
        questions
    };
    let graph_raw = fs::read_to_string(&graph_path)
        .await
        .with_context(|| format!("read graph {}", graph_path.display()))?;
    let graph = GraphBuilder::from_json(&graph_raw)
        .map_err(|e| anyhow!("parse graph {}: {e}", graph_path.display()))?;

    let client: Arc<dyn GraphClient> = Arc::new(MemgraphClient::connect(&cfg.database).await?);
    let (registry, embedder) = build_registry(&cfg)?;
    let prompt_embedder = build_ontology_catalog_embedder(&cfg)?;
    let embedding_store = build_embedding_store(&cfg)?;
    let ontology_storage: Arc<dyn OntologyCatalogStorage> = Arc::new(
        crate::graph::InMemoryOntologyCatalogStorage::new((*catalog).clone()),
    );
    let mut pipeline = Pipeline::new(client.clone(), &cfg)
        .with_ingest_batch_size(settings.batch_size)
        .with_registry(registry.clone())
        .with_ontology_catalog_storage(ontology_storage)
        .with_ontology_catalog(catalog.clone())
        .with_prefix_label(Some(prefix.clone()))
        .with_prefix_index(Some(prefix.clone()));
    if let Some(embedder) = embedder {
        pipeline = pipeline.with_embedder(embedder);
    }
    if let Some(reranker) = build_semantic_text_reranker(&cfg)? {
        pipeline = pipeline.with_reranker(reranker);
    }
    if cfg.query.grounding.enabled {
        pipeline = pipeline.with_prefetch_store(embedding_store.clone());
    }
    pipeline.load_ontology_catalog().await?;

    ensure_indexes(client.as_ref(), &graph, &settings.indexes).await?;

    if settings.cleanup_before {
        tracing::info!(target: "linguagraph::e2e", prefix = %prefix, "cleanup before e2e suite");
        cleanup_prefix(client.as_ref(), &prefix, settings.cleanup_vectors).await?;
    }

    tracing::info!(
        target: "linguagraph::e2e",
        prefix = %prefix,
        "ingesting e2e graph"
    );
    let summary = pipeline.ingest(&graph).await?;
    tracing::info!(
        target: "linguagraph::e2e",
        nodes = summary.node_rows,
        relations = summary.relation_rows,
        batches = summary.batches_executed,
        side_effect_batches = summary.side_effect_batches,
        side_effect_rows = summary.side_effect_rows,
        elapsed_ms = summary.elapsed_ms,
        prefix = %prefix,
        "ingested e2e graph"
    );

    let llm = build_llm_client(&cfg)?;
    let prompt_reranker = pipeline.reranker();
    let schema = pipeline.live_schema(&[prefix.as_str()]).await?;
    let prompt_opts = PromptOptions {
        include_examples: true,
        reranking_model: prompt_reranker,
        type_registry: Some((*registry).clone()),
        ..PromptOptions::default()
    };

    let mut case_reports = Vec::with_capacity(questions.len());
    for case in questions {
        tracing::info!(target: "linguagraph::e2e", case = %case.id, "running e2e case");
        let report = run_case(
            &case,
            &pipeline,
            llm.clone(),
            &schema,
            &prompt_opts,
            catalog.as_ref(),
            prompt_embedder.as_ref(),
            embedding_store.as_ref(),
            &cfg.qdrant.collection,
            cfg.ontology_catalog
                .embedding_model
                .as_deref()
                .unwrap_or("mock"),
            &cfg.ontology_catalog,
            &prefix,
            &settings,
        )
        .await;
        case_reports.push(report);
    }

    if settings.cleanup_after {
        tracing::info!(target: "linguagraph::e2e", prefix = %prefix, "cleanup after e2e suite");
        cleanup_prefix(client.as_ref(), &prefix, settings.cleanup_vectors).await?;
    }

    let passed = case_reports.iter().filter(|c| c.passed).count();
    let failed = case_reports.len().saturating_sub(passed);
    let report = E2eReport {
        suite: suite_name,
        prefix_label: prefix.clone(),
        prefix_index: prefix,
        graph_path,
        ontology_path,
        questions_path,
        total: case_reports.len(),
        passed,
        failed,
        include_embeddings_in_report: settings.include_embeddings_in_report,
        cases: case_reports,
    };

    if let Some(path) = opts.report_path {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent).await?;
            }
        }
        fs::write(&path, serde_json::to_vec_pretty(&report)?).await?;
    }

    Ok(report)
}

async fn run_case(
    case: &QuestionCase,
    pipeline: &Pipeline,
    llm: Arc<dyn LlmClient>,
    schema: &prompt::GraphSchema,
    prompt_opts: &PromptOptions,
    catalog: &OntologyCatalog,
    prompt_embedder: &dyn embeddings::Embedder,
    embedding_store: &dyn embeddings::EmbeddingStore,
    prompt_collection: &str,
    prompt_model: &str,
    ontology_cfg: &config::OntologyCatalogConfig,
    prefix: &str,
    settings: &SuiteSettings,
) -> E2eCaseReport {
    let mut errors = Vec::new();
    let mut dsl_json = None;
    let mut traversal_json = None;
    let mut cypher = None;
    let mut cypher_params = None;
    let mut rows = Vec::new();
    let mut answer = None;
    let mut judge_report = None;

    if case.dsl.is_some() && case.traversal.is_some() {
        errors.push("case cannot define both `dsl` and `traversal`".to_string());
        return E2eCaseReport {
            id: case.id.clone(),
            question: case.question.clone(),
            passed: false,
            errors,
            dsl: dsl_json,
            traversal: traversal_json,
            cypher,
            cypher_params,
            row_count: 0,
            rows,
            answer,
            judge: judge_report,
        };
    }

    if let Some(traversal) = &case.traversal {
        let mut traversal = traversal.clone();
        force_traversal_prefix(&mut traversal, prefix);
        traversal_json = serde_json::to_value(&traversal).ok();

        tracing::debug!(
            target: "linguagraph::e2e",
            traversal = %serde_json::to_string(&traversal).unwrap_or_default(),
            "running e2e traversal"
        );

        if case.validation.dsl_expect.is_some() {
            errors.push("`dsl_expect` is not applicable to traversal cases".to_string());
        }

        let result = match pipeline.run_traversal(traversal.clone()).await {
            Ok(result) => result,
            Err(err) => {
                errors.push(format!("traversal query failed: {err}"));
                return E2eCaseReport {
                    id: case.id.clone(),
                    question: case.question.clone(),
                    passed: false,
                    errors,
                    dsl: dsl_json,
                    traversal: traversal_json,
                    cypher,
                    cypher_params,
                    row_count: 0,
                    rows,
                    answer,
                    judge: judge_report,
                };
            }
        };

        rows = result_rows_json(&result, settings.include_embeddings_in_report);
        errors.extend(validate_result(
            &case.validation,
            &result,
            answer.as_deref(),
        ));

        if settings.answer_with_llm || !case.validation.answer_contains.is_empty() {
            match synthesize_traversal_answer(llm.clone(), &case.question, &traversal, &rows).await
            {
                Ok(text) => {
                    errors.extend(validate_answer_contains(
                        &case.validation.answer_contains,
                        &text,
                    ));
                    answer = Some(text);
                }
                Err(err) => errors.push(format!("answer synthesis failed: {err}")),
            }
        }

        let judge_spec = case
            .validation
            .judge
            .as_ref()
            .filter(|_| settings.judge_with_llm || case.validation.judge.is_some());
        if let Some(spec) = judge_spec {
            match judge_answer(llm, &case.question, answer.as_deref(), &rows, spec).await {
                Ok(judge) => {
                    if spec.require_pass && !judge.passed {
                        errors.push(format!("LLM judge failed: {}", judge.reason));
                    }
                    judge_report = Some(judge);
                }
                Err(err) => errors.push(format!("LLM judge failed to run: {err}")),
            }
        }

        let passed = errors.is_empty();
        return E2eCaseReport {
            id: case.id.clone(),
            question: case.question.clone(),
            passed,
            errors,
            dsl: dsl_json,
            traversal: traversal_json,
            cypher,
            cypher_params,
            row_count: result.rows.len(),
            rows,
            answer,
            judge: judge_report,
        };
    }

    let dsl_result = match &case.dsl {
        Some(static_dsl) => parse_case_dsl(static_dsl.clone(), prefix),
        None => {
            generate_case_dsl(
                llm.clone(),
                &case.question,
                schema,
                prompt_opts,
                catalog,
                prompt_embedder,
                embedding_store,
                prompt_collection,
                prompt_model,
                ontology_cfg,
                prefix,
                settings.max_repairs,
            )
            .await
        }
    };

    let dsl = match dsl_result {
        Ok(dsl) => {
            dsl_json = serde_json::to_value(&dsl).ok();
            dsl
        }
        Err(err) => {
            errors.push(err.to_string());
            return E2eCaseReport {
                id: case.id.clone(),
                question: case.question.clone(),
                passed: false,
                errors,
                dsl: dsl_json,
                traversal: traversal_json,
                cypher,
                cypher_params,
                row_count: 0,
                rows,
                answer,
                judge: judge_report,
            };
        }
    };

    tracing::debug!(
        target: "linguagraph::e2e",
        dsl = %serde_json::to_string(&dsl).unwrap_or_default(),
        "compiled e2e DSL"
    );

    errors.extend(validate_dsl(&case.validation.dsl_expect, &dsl));

    match pipeline.compile(dsl.clone()) {
        Ok(query) => {
            cypher = Some(query.text.clone());
            cypher_params = Some(mask_cypher_params(
                &query.params,
                settings.include_embeddings_in_report,
            ));
        }
        Err(err) => {
            errors.push(format!("cypher compile failed: {err}"));
            return E2eCaseReport {
                id: case.id.clone(),
                question: case.question.clone(),
                passed: false,
                errors,
                dsl: dsl_json,
                traversal: traversal_json,
                cypher,
                cypher_params,
                row_count: 0,
                rows,
                answer,
                judge: judge_report,
            };
        }
    }

    let result = match pipeline.run(dsl.clone()).await {
        Ok(result) => result,
        Err(err) => {
            errors.push(format!("query failed: {err}"));
            return E2eCaseReport {
                id: case.id.clone(),
                question: case.question.clone(),
                passed: false,
                errors,
                dsl: dsl_json,
                traversal: traversal_json,
                cypher,
                cypher_params,
                row_count: 0,
                rows,
                answer,
                judge: judge_report,
            };
        }
    };

    rows = result_rows_json(&result, settings.include_embeddings_in_report);
    errors.extend(validate_result(
        &case.validation,
        &result,
        answer.as_deref(),
    ));

    if settings.answer_with_llm || !case.validation.answer_contains.is_empty() {
        match synthesize_answer(llm.clone(), &case.question, &dsl, &rows).await {
            Ok(text) => {
                errors.extend(validate_answer_contains(
                    &case.validation.answer_contains,
                    &text,
                ));
                answer = Some(text);
            }
            Err(err) => errors.push(format!("answer synthesis failed: {err}")),
        }
    }

    let judge_spec = case
        .validation
        .judge
        .as_ref()
        .filter(|_| settings.judge_with_llm || case.validation.judge.is_some());
    if let Some(spec) = judge_spec {
        match judge_answer(llm, &case.question, answer.as_deref(), &rows, spec).await {
            Ok(judge) => {
                if spec.require_pass && !judge.passed {
                    errors.push(format!("LLM judge failed: {}", judge.reason));
                }
                judge_report = Some(judge);
            }
            Err(err) => errors.push(format!("LLM judge failed to run: {err}")),
        }
    }

    let passed = errors.is_empty();
    E2eCaseReport {
        id: case.id.clone(),
        question: case.question.clone(),
        passed,
        errors,
        dsl: dsl_json,
        traversal: traversal_json,
        cypher,
        cypher_params,
        row_count: result.rows.len(),
        rows,
        answer,
        judge: judge_report,
    }
}

async fn load_suite(path: &Path) -> anyhow::Result<SuiteFile> {
    let raw = fs::read_to_string(path)
        .await
        .with_context(|| format!("read suite {}", path.display()))?;
    Ok(serde_json::from_str(&raw).with_context(|| format!("parse suite {}", path.display()))?)
}

async fn load_catalog(path: &Path) -> anyhow::Result<OntologyCatalog> {
    let raw = fs::read_to_string(path)
        .await
        .with_context(|| format!("read ontology {}", path.display()))?;
    Ok(OntologyCatalog::load_from_str(&raw)
        .map_err(|e| anyhow!("{e}"))
        .with_context(|| format!("parse ontology {}", path.display()))?)
}

async fn load_questions(path: &Path) -> anyhow::Result<Vec<QuestionCase>> {
    let raw = fs::read_to_string(path)
        .await
        .with_context(|| format!("read questions {}", path.display()))?;
    if let Ok(file) = serde_json::from_str::<QuestionFile>(&raw) {
        return Ok(file.questions);
    }
    let questions: Vec<QuestionCase> = serde_json::from_str(&raw)
        .with_context(|| format!("parse questions {}", path.display()))?;
    Ok(questions)
}

fn mask_cypher_params(
    params: &BTreeMap<String, crate::ast::query::Literal>,
    include_embeddings: bool,
) -> BTreeMap<String, JsonValue> {
    params
        .iter()
        .map(|(name, value)| {
            let json = literal_to_json(value);
            let masked = if include_embeddings {
                json
            } else if is_masked_embedding_name(name) {
                masked_embedding_value(&json)
            } else {
                sanitize_report_json(json, true)
            };
            (name.clone(), masked)
        })
        .collect()
}

fn is_masked_embedding_name(name: &str) -> bool {
    let name = name.to_ascii_lowercase();
    name.contains("embedding")
        || name == "emb"
        || name == "vec"
        || name == "vecs"
        || name.ends_with("_emb")
        || name.ends_with("_vec")
        || name.ends_with("_embedding")
}

fn masked_embedding_value(value: &JsonValue) -> JsonValue {
    match value {
        JsonValue::Array(items) if items.iter().all(JsonValue::is_number) => {
            JsonValue::String(format!("<masked embedding len={}>", items.len()))
        }
        JsonValue::Array(items) if items.iter().all(JsonValue::is_array) => {
            JsonValue::String(format!("<masked embedding matrix rows={}>", items.len()))
        }
        JsonValue::Array(items) => {
            JsonValue::String(format!("<masked embedding array len={}>", items.len()))
        }
        _ => JsonValue::String("<masked embedding>".to_string()),
    }
}

fn sanitize_report_json(value: JsonValue, mask_positional_numeric_vectors: bool) -> JsonValue {
    match value {
        JsonValue::Array(items) if mask_positional_numeric_vectors && is_numeric_vector(&items) => {
            masked_embedding_value(&JsonValue::Array(items))
        }
        JsonValue::Array(items) => JsonValue::Array(
            items
                .into_iter()
                .map(|value| sanitize_report_json(value, mask_positional_numeric_vectors))
                .collect(),
        ),
        JsonValue::Object(map) => JsonValue::Object(
            map.into_iter()
                .map(|(key, value)| {
                    let value = if is_masked_embedding_name(&key) {
                        masked_embedding_value(&value)
                    } else {
                        sanitize_report_json(value, mask_positional_numeric_vectors)
                    };
                    (key, value)
                })
                .collect(),
        ),
        other => other,
    }
}

fn is_numeric_vector(items: &[JsonValue]) -> bool {
    items.len() >= 16 && items.iter().all(JsonValue::is_number)
}

fn literal_to_json(literal: &crate::ast::query::Literal) -> JsonValue {
    match literal {
        crate::ast::query::Literal::String(s) => JsonValue::String(s.clone()),
        crate::ast::query::Literal::Bool(b) => JsonValue::Bool(*b),
        crate::ast::query::Literal::Int(i) => JsonValue::Number((*i).into()),
        crate::ast::query::Literal::Float(f) => serde_json::Number::from_f64(*f)
            .map(JsonValue::Number)
            .unwrap_or(JsonValue::Null),
        crate::ast::query::Literal::List(items) => {
            JsonValue::Array(items.iter().map(literal_to_json).collect())
        }
        crate::ast::query::Literal::Object(map) => JsonValue::Object(
            map.iter()
                .map(|(key, value)| (key.clone(), literal_to_json(value)))
                .collect(),
        ),
        crate::ast::query::Literal::Null => JsonValue::Null,
    }
}

fn resolve_suite_path(suite_dir: &Path, path: PathBuf) -> PathBuf {
    if path.is_absolute() {
        path
    } else {
        suite_dir.join(path)
    }
}

fn suite_name_from_path(path: &Path) -> String {
    path.file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("e2e")
        .to_string()
}

fn unique_prefix(suite_name: &str) -> String {
    let millis = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or_default();
    format!("E2E_{}_{}", suite_name, millis)
}

fn sanitize_ident(raw: &str) -> Option<String> {
    let mut out = String::with_capacity(raw.len());
    for ch in raw.chars() {
        if ch.is_ascii_alphanumeric() || ch == '_' {
            out.push(ch);
        } else {
            out.push('_');
        }
    }
    if out.is_empty() || out.as_bytes()[0].is_ascii_digit() {
        out.insert_str(0, "E2E_");
    }
    if crate::ingest::delete::is_valid_ident(&out) {
        Some(out)
    } else {
        None
    }
}

fn build_registry(cfg: &Config) -> anyhow::Result<(SharedRegistry, Option<SharedEmbedder>)> {
    let dim = cfg
        .types
        .get("SemanticText")
        .and_then(|t| t.embedding_dim)
        .unwrap_or(384);
    let model = cfg
        .types
        .get("SemanticText")
        .and_then(|t| t.embedding_model.clone());
    let embedder = embeddings::default_embedder(model.as_deref(), dim)
        .map_err(|e| anyhow!("embedder init: {e}"))?;
    let registry = types::handlers::register_default(cfg, embedder.clone())
        .map_err(|e| anyhow!("registry init: {e}"))?;
    Ok((Arc::new(registry), Some(embedder)))
}

fn build_semantic_text_reranker(
    cfg: &Config,
) -> anyhow::Result<Option<embeddings::SharedReranker>> {
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
    let reranker = embeddings::default_reranker(Some(&model), dim)
        .map_err(|e| anyhow!("SemanticText reranker init: {e}"))?;
    Ok(Some(reranker))
}

fn build_embedding_store(cfg: &Config) -> anyhow::Result<SharedEmbeddingStore> {
    #[cfg(feature = "qdrant")]
    if !cfg.qdrant.url.trim().is_empty() {
        let client = crate::db::QdrantClient::connect(&cfg.qdrant)
            .map_err(|e| anyhow!("qdrant connect: {e}"))?;
        return Ok(Arc::new(client));
    }
    #[cfg(not(feature = "qdrant"))]
    let _ = cfg;
    Ok(Arc::new(InMemoryEmbeddingStore::new()))
}

fn build_ontology_catalog_embedder(cfg: &Config) -> anyhow::Result<SharedEmbedder> {
    embeddings::default_embedder(
        cfg.ontology_catalog.embedding_model.as_deref(),
        cfg.ontology_catalog.embedding_dim,
    )
    .map_err(|e| anyhow!("ontology catalog embedder init: {e}"))
}

#[cfg(feature = "openai")]
fn build_llm_client(cfg: &Config) -> anyhow::Result<Arc<dyn LlmClient>> {
    Ok(Arc::new(crate::llm::OpenAiClient::from_config(&cfg.llm)))
}

#[cfg(not(feature = "openai"))]
fn build_llm_client(_cfg: &Config) -> anyhow::Result<Arc<dyn LlmClient>> {
    bail!("the `openai` feature is disabled; rebuild with `--features openai`")
}

async fn ensure_indexes(
    client: &dyn GraphClient,
    graph: &Graph,
    configured: &[IndexSpec],
) -> anyhow::Result<()> {
    let mut indexes: BTreeSet<(String, String)> = BTreeSet::new();
    for entity in graph.entities() {
        if let Some(PrimaryKey::Strict(field)) = &entity.primary_key {
            indexes.insert((entity.r#type.clone(), field.clone()));
        }
    }
    for idx in configured {
        indexes.insert((idx.label.clone(), idx.property.clone()));
    }

    for (label, property) in indexes {
        if !crate::ingest::delete::is_valid_ident(&label)
            || !crate::ingest::delete::is_valid_ident(&property)
        {
            bail!("invalid e2e index `{label}.{property}`");
        }
        let q = CypherQuery::new(
            format!("CREATE INDEX ON :{label}({property})"),
            BTreeMap::new(),
        );
        match client.execute(&q).await {
            Ok(_) => {
                tracing::info!(target: "linguagraph::e2e", label = %label, property = %property, "ensured e2e index");
            }
            Err(err) => {
                tracing::debug!(
                    target: "linguagraph::e2e",
                    label = %label,
                    property = %property,
                    error = %err,
                    "e2e index create skipped"
                );
            }
        }
    }
    Ok(())
}

const CLEANUP_BATCH_SIZE: usize = 1000;

async fn cleanup_prefix(
    client: &dyn GraphClient,
    prefix: &str,
    cleanup_vectors: bool,
) -> anyhow::Result<()> {
    let mut total = 0_usize;
    loop {
        let deleted = cleanup_prefix_batch(client, prefix, cleanup_vectors)
            .await
            .with_context(|| format!("cleanup prefix {prefix}"))?;
        if deleted == 0 {
            break;
        }
        total += deleted;
        tracing::info!(
            target: "linguagraph::e2e",
            prefix = %prefix,
            deleted,
            total,
            "cleaned e2e batch"
        );
    }
    Ok(())
}

async fn cleanup_prefix_batch(
    client: &dyn GraphClient,
    prefix: &str,
    cleanup_vectors: bool,
) -> anyhow::Result<usize> {
    let query = if cleanup_vectors {
        format!(
            "MATCH (n:{prefix}) \
             WITH n LIMIT {CLEANUP_BATCH_SIZE} \
             WITH collect(id(n)) AS ids \
             CALL libqlink.delete_batch_all(ids) YIELD success, collections \
             WITH ids \
             MATCH (n) WHERE id(n) IN ids \
             DETACH DELETE n \
             RETURN size(ids) AS nodes"
        )
    } else {
        format!(
            "MATCH (n:{prefix}) \
             WITH n LIMIT {CLEANUP_BATCH_SIZE} \
             DETACH DELETE n \
             RETURN count(*) AS nodes"
        )
    };
    let result = client
        .execute(&CypherQuery::new(query, BTreeMap::new()))
        .await?;
    Ok(first_usize(&result, "nodes"))
}

fn first_usize(result: &QueryResult, column: &str) -> usize {
    result
        .rows
        .first()
        .and_then(|row| row.fields.get(column))
        .and_then(db_value_as_f64)
        .map(|v| v.max(0.0) as usize)
        .unwrap_or(0)
}

fn parse_case_dsl(value: JsonValue, prefix: &str) -> anyhow::Result<DslQuery> {
    let mut dsl: DslQuery = serde_json::from_value(value).context("parse static DSL")?;
    force_prefix(&mut dsl, prefix);
    dsl::parse_str(&serde_json::to_string(&dsl)?).context("validate static DSL")
}

async fn generate_case_dsl(
    llm: Arc<dyn LlmClient>,
    question: &str,
    schema: &prompt::GraphSchema,
    prompt_opts: &PromptOptions,
    catalog: &OntologyCatalog,
    prompt_embedder: &dyn embeddings::Embedder,
    embedding_store: &dyn embeddings::EmbeddingStore,
    prompt_collection: &str,
    prompt_model: &str,
    ontology_cfg: &config::OntologyCatalogConfig,
    prefix: &str,
    max_repairs: usize,
) -> anyhow::Result<DslQuery> {
    embedding_store
        .ensure(prompt_collection, prompt_embedder.dim())
        .await
        .map_err(|e| anyhow!("embedding store: {e}"))?;
    let index = EmbeddingIndex {
        store: embedding_store,
        collection: prompt_collection,
        model: prompt_model,
    };
    let params = prompt::QueryPromptParams {
        domain_threshold: ontology_cfg.domain_selection_threshold,
        domain_top_k: ontology_cfg.domain_selection_top_k,
        selection: prompt::QuerySelectionParams {
            entity_threshold: ontology_cfg.entity_selection_threshold,
            property_threshold: ontology_cfg.property_selection_threshold,
            neighbor_hops: ontology_cfg.selection_neighbor_hops,
            ..prompt::QuerySelectionParams::default()
        },
        include_examples: prompt_opts.include_examples,
    };
    let mut schema = schema.clone();
    let system = prompt::generate_query_prompt(
        question,
        &mut schema,
        catalog,
        prompt_embedder,
        &index,
        &params,
    )
    .await
    .map_err(|e| anyhow!("query prompt generation failed: {e}"))?;
    let mut user = format!(
        "Question:\n{question}\n\nReturn only one JSON DSL object. Do not wrap it in Markdown."
    );
    let mut last_output = String::new();
    let mut last_error = String::new();

    for attempt in 0..=max_repairs {
        let raw = llm
            .complete(&system, &user)
            .await
            .map_err(|e| anyhow!("LLM DSL generation failed: {e}"))?;
        last_output = raw.clone();
        match extract_json_object(&raw)
            .and_then(|json| serde_json::from_str::<DslQuery>(&json).map_err(|e| e.into()))
            .and_then(|mut dsl| {
                force_prefix(&mut dsl, prefix);
                dsl::parse_str(&serde_json::to_string(&dsl)?).map_err(|e| e.into())
            }) {
            Ok(dsl) => return Ok(dsl),
            Err(err) => {
                last_error = err.to_string();
                if attempt < max_repairs {
                    user = format!(
                        "Repair the JSON DSL for this question.\n\nQuestion:\n{question}\n\n\
                         Previous output:\n{last_output}\n\nValidation error:\n{last_error}\n\n\
                         Return only the corrected JSON object."
                    );
                }
            }
        }
    }

    bail!("could not produce valid DSL: {last_error}; last output: {last_output}")
}

fn force_prefix(dsl: &mut DslQuery, prefix: &str) {
    dsl.prefix_label = Some(prefix.to_string());
    dsl.prefix_index = Some(prefix.to_string());
}

fn force_traversal_prefix(traversal: &mut TraversalQuery, prefix: &str) {
    traversal.prefix_label = Some(prefix.to_string());
    traversal.prefix_index = Some(prefix.to_string());
}

fn extract_json_object(raw: &str) -> anyhow::Result<String> {
    if serde_json::from_str::<JsonValue>(raw).is_ok() {
        return Ok(raw.trim().to_string());
    }

    let mut start = None;
    let mut depth = 0_i32;
    let mut in_string = false;
    let mut escape = false;
    for (idx, ch) in raw.char_indices() {
        if in_string {
            if escape {
                escape = false;
            } else if ch == '\\' {
                escape = true;
            } else if ch == '"' {
                in_string = false;
            }
            continue;
        }

        match ch {
            '"' => in_string = true,
            '{' => {
                if start.is_none() {
                    start = Some(idx);
                }
                depth += 1;
            }
            '}' if start.is_some() => {
                depth -= 1;
                if depth == 0 {
                    let s = &raw[start.unwrap()..=idx];
                    serde_json::from_str::<JsonValue>(s)
                        .with_context(|| format!("invalid extracted JSON: {s}"))?;
                    return Ok(s.to_string());
                }
            }
            _ => {}
        }
    }
    bail!("LLM output did not contain a JSON object")
}

fn validate_result(
    spec: &ValidationSpec,
    result: &QueryResult,
    answer: Option<&str>,
) -> Vec<String> {
    let mut errors = Vec::new();
    if let Some(row_count) = &spec.row_count {
        let n = result.rows.len();
        if row_count.exact.is_some_and(|want| n != want) {
            errors.push(format!(
                "row_count exact mismatch: expected {}, got {n}",
                row_count.exact.unwrap()
            ));
        }
        if row_count.min.is_some_and(|min| n < min) {
            errors.push(format!(
                "row_count min mismatch: expected at least {}, got {n}",
                row_count.min.unwrap()
            ));
        }
        if row_count.max.is_some_and(|max| n > max) {
            errors.push(format!(
                "row_count max mismatch: expected at most {}, got {n}",
                row_count.max.unwrap()
            ));
        }
    }

    if !spec.rows.is_empty() {
        errors.extend(validate_expected_rows(&spec.rows, result));
    }

    for column in &spec.columns {
        if !result_has_column(result, column) {
            errors.push(format!("missing expected column `{column}`"));
        }
    }

    for expected in &spec.contains {
        if !result_contains(result, expected) {
            errors.push(format!(
                "no row contains expected value in `{}`: {}",
                expected.column, expected.value
            ));
        }
    }

    for expected in &spec.not_contains {
        if result_contains(result, expected) {
            errors.push(format!(
                "unexpected value found in `{}`: {}",
                expected.column, expected.value
            ));
        }
    }

    for expected in &spec.numbers {
        if !result_satisfies_number(result, expected) {
            errors.push(format!(
                "no numeric value in `{}` satisfies {:?} {}",
                expected.column, expected.op, expected.value
            ));
        }
    }

    if let Some(answer) = answer {
        errors.extend(validate_answer_contains(&spec.answer_contains, answer));
    }

    errors
}

fn validate_dsl(spec: &Option<DslValidationSpec>, dsl: &DslQuery) -> Vec<String> {
    let Some(spec) = spec.as_ref() else {
        return Vec::new();
    };

    let mut errors = Vec::new();

    if let Some(expected) = &spec.start_label {
        if !dsl.start.label.eq_ignore_ascii_case(expected) {
            errors.push(format!(
                "dsl start label mismatch: expected `{expected}`, got `{}`",
                dsl.start.label
            ));
        }
    }
    if let Some(expected) = &spec.start_alias {
        if dsl.start.alias != *expected {
            errors.push(format!(
                "dsl start alias mismatch: expected `{expected}`, got `{}`",
                dsl.start.alias
            ));
        }
    }

    if spec.min_filters.is_some_and(|min| dsl.filters.len() < min) {
        errors.push(format!(
            "dsl filter count below minimum: expected at least {}, got {}",
            spec.min_filters.unwrap(),
            dsl.filters.len()
        ));
    }
    if spec.max_filters.is_some_and(|max| dsl.filters.len() > max) {
        errors.push(format!(
            "dsl filter count above maximum: expected at most {}, got {}",
            spec.max_filters.unwrap(),
            dsl.filters.len()
        ));
    }
    if spec.min_returns.is_some_and(|min| dsl.return_.len() < min) {
        errors.push(format!(
            "dsl return count below minimum: expected at least {}, got {}",
            spec.min_returns.unwrap(),
            dsl.return_.len()
        ));
    }
    if spec.max_returns.is_some_and(|max| dsl.return_.len() > max) {
        errors.push(format!(
            "dsl return count above maximum: expected at most {}, got {}",
            spec.max_returns.unwrap(),
            dsl.return_.len()
        ));
    }

    for expected in &spec.required_filter_fields {
        if !dsl
            .filters
            .iter()
            .any(|f| field_matches(&f.field, expected))
        {
            errors.push(format!("dsl missing filter field matching `{expected}`"));
        }
    }
    for expected in &spec.required_filter_ops {
        if !dsl
            .filters
            .iter()
            .any(|f| f.op.eq_ignore_ascii_case(expected))
        {
            errors.push(format!("dsl missing filter op `{expected}`"));
        }
    }
    for forbidden in &spec.forbidden_filter_ops {
        if dsl
            .filters
            .iter()
            .any(|f| f.op.eq_ignore_ascii_case(forbidden))
        {
            errors.push(format!("dsl contains forbidden filter op `{forbidden}`"));
        }
    }

    for expected in &spec.required_return_fields {
        if !dsl
            .return_
            .iter()
            .any(|item| return_item_matches(item, expected))
        {
            errors.push(format!("dsl missing return item matching `{expected}`"));
        }
    }

    for expected in &spec.required_traversal_labels {
        if !dsl.traversals.iter().any(|t| {
            t.edge.label.eq_ignore_ascii_case(expected)
                || t.target.label.eq_ignore_ascii_case(expected)
        }) {
            errors.push(format!(
                "dsl missing traversal edge or target matching `{expected}`"
            ));
        }
    }

    errors
}

fn validate_expected_rows(expected_rows: &[ExpectedRow], result: &QueryResult) -> Vec<String> {
    let actual_rows = result_rows_json(result, true);
    let mut unmatched = Vec::new();
    let mut used = vec![false; actual_rows.len()];

    for expected in expected_rows {
        let mut found = false;
        for (idx, actual) in actual_rows.iter().enumerate() {
            if used[idx] {
                continue;
            }
            if row_matches(actual, &expected.fields) {
                used[idx] = true;
                found = true;
                break;
            }
        }
        if !found {
            unmatched.push(format!(
                "no row matched expected subset: {}",
                serde_json::to_string(&expected.fields).unwrap_or_default()
            ));
        }
    }

    unmatched
}

fn row_matches(
    actual: &BTreeMap<String, JsonValue>,
    expected: &BTreeMap<String, JsonValue>,
) -> bool {
    expected.iter().all(|(column, expected_value)| {
        actual
            .get(column)
            .is_some_and(|actual_value| json_values_equal(actual_value, expected_value))
    })
}

fn field_matches(actual: &str, expected: &str) -> bool {
    actual == expected
        || actual
            .rsplit_once('.')
            .is_some_and(|(_, suffix)| suffix.eq_ignore_ascii_case(expected))
}

fn return_item_matches(item: &crate::dsl::schema::ReturnItem, expected: &str) -> bool {
    match item {
        crate::dsl::schema::ReturnItem::Field { field, alias } => {
            field_matches(field, expected) || alias.as_deref() == Some(expected)
        }
        crate::dsl::schema::ReturnItem::Aggregate { field, alias, .. } => {
            field_matches(field, expected) || alias.as_deref() == Some(expected)
        }
        crate::dsl::schema::ReturnItem::DatePart { field, alias, .. } => {
            field_matches(field, expected) || alias.as_deref() == Some(expected)
        }
    }
}

fn validate_answer_contains(expected: &[String], answer: &str) -> Vec<String> {
    let answer_lc = answer.to_lowercase();
    expected
        .iter()
        .filter(|needle| !answer_lc.contains(&needle.to_lowercase()))
        .map(|needle| format!("answer does not contain `{needle}`"))
        .collect()
}

fn result_has_column(result: &QueryResult, column: &str) -> bool {
    result.columns.iter().any(|c| c.name == column)
        || result
            .rows
            .iter()
            .any(|row| row.fields.contains_key(column))
}

fn result_contains(result: &QueryResult, expected: &CellExpectation) -> bool {
    if expected.column == "*" {
        return result.rows.iter().any(|row| {
            row.fields
                .values()
                .any(|cell| cell_matches(cell, &expected.value, expected.mode))
        });
    }
    result.rows.iter().any(|row| {
        row.fields
            .get(&expected.column)
            .is_some_and(|cell| cell_matches(cell, &expected.value, expected.mode))
    })
}

fn cell_matches(cell: &DbValue, expected: &JsonValue, mode: MatchMode) -> bool {
    let actual = db_value_to_json(cell);
    match mode {
        MatchMode::Exact => json_values_equal(&actual, expected),
        MatchMode::Contains => cell_text(cell)
            .to_lowercase()
            .contains(&json_value_text(expected).to_lowercase()),
    }
}

fn json_values_equal(actual: &JsonValue, expected: &JsonValue) -> bool {
    if actual == expected {
        return true;
    }
    match (actual, expected) {
        (JsonValue::Number(a), JsonValue::Number(e)) => {
            let Some(a) = a.as_f64() else { return false };
            let Some(e) = e.as_f64() else { return false };
            (a - e).abs() < 1e-9
        }
        _ => json_value_text(actual) == json_value_text(expected),
    }
}

fn result_satisfies_number(result: &QueryResult, expected: &NumericExpectation) -> bool {
    if expected.column == "*" {
        return result.rows.iter().any(|row| {
            row.fields
                .values()
                .filter_map(db_value_as_f64)
                .any(|actual| compare_number(actual, expected.op, expected.value))
        });
    }
    result.rows.iter().any(|row| {
        row.fields
            .get(&expected.column)
            .and_then(db_value_as_f64)
            .is_some_and(|actual| compare_number(actual, expected.op, expected.value))
    })
}

fn compare_number(actual: f64, op: NumericOp, expected: f64) -> bool {
    match op {
        NumericOp::Eq => (actual - expected).abs() < 1e-9,
        NumericOp::Neq => (actual - expected).abs() >= 1e-9,
        NumericOp::Gt => actual > expected,
        NumericOp::Gte => actual >= expected,
        NumericOp::Lt => actual < expected,
        NumericOp::Lte => actual <= expected,
    }
}

async fn synthesize_answer(
    llm: Arc<dyn LlmClient>,
    question: &str,
    dsl: &DslQuery,
    rows: &[BTreeMap<String, JsonValue>],
) -> anyhow::Result<String> {
    synthesize_answer_with_summary(llm, question, dsl.describe(), rows).await
}

async fn synthesize_traversal_answer(
    llm: Arc<dyn LlmClient>,
    question: &str,
    traversal: &TraversalQuery,
    rows: &[BTreeMap<String, JsonValue>],
) -> anyhow::Result<String> {
    let entity_names = traversal.entity_names();
    let entities = if entity_names.is_empty() {
        "none".to_string()
    } else {
        entity_names.join(", ")
    };
    let summary = format!(
        "Retrieving text chunks for goal: {}; entities: {}; limit: {:?}.",
        traversal.goal_search_text(),
        entities,
        traversal.limit
    );
    synthesize_answer_with_summary(llm, question, summary, rows).await
}

async fn synthesize_answer_with_summary(
    llm: Arc<dyn LlmClient>,
    question: &str,
    query_summary: String,
    rows: &[BTreeMap<String, JsonValue>],
) -> anyhow::Result<String> {
    let system = "Answer the user's question using only the supplied graph query rows. \
                  Be concise. If the rows are empty, say that the data does not contain the answer.";
    let user = json!({
        "question": question,
        "query_summary": query_summary,
        "rows": rows,
    });
    let answer = llm
        .complete(system, &serde_json::to_string_pretty(&user)?)
        .await?;
    Ok(answer.trim().to_string())
}

async fn judge_answer(
    llm: Arc<dyn LlmClient>,
    question: &str,
    answer: Option<&str>,
    rows: &[BTreeMap<String, JsonValue>],
    spec: &JudgeSpec,
) -> anyhow::Result<JudgeReport> {
    let system = "You are a strict E2E test judge. Decide whether the answer is fully supported \
                  by the provided rows and satisfies the expected answer. Return only JSON: \
                  {\"passed\": boolean, \"reason\": string}.";
    let user = json!({
        "question": question,
        "expected": spec.expected,
        "answer": answer.unwrap_or(""),
        "rows": rows,
    });
    let raw = llm
        .complete(system, &serde_json::to_string_pretty(&user)?)
        .await?;
    let json = extract_json_object(&raw)?;
    let value: JsonValue = serde_json::from_str(&json)?;
    let passed = value
        .get("passed")
        .and_then(JsonValue::as_bool)
        .unwrap_or(false);
    let reason = value
        .get("reason")
        .and_then(JsonValue::as_str)
        .unwrap_or("")
        .to_string();
    Ok(JudgeReport { passed, reason })
}

fn result_rows_json(
    result: &QueryResult,
    include_embeddings: bool,
) -> Vec<BTreeMap<String, JsonValue>> {
    result
        .rows
        .iter()
        .map(|row| {
            row.fields
                .iter()
                .filter(|(k, _)| !matches!(k.as_str(), "sources" | "score"))
                .map(|(k, v)| {
                    let value = db_value_to_json(v);
                    let value = if include_embeddings {
                        value
                    } else if is_masked_embedding_name(k) {
                        masked_embedding_value(&value)
                    } else {
                        sanitize_report_json(value, false)
                    };
                    (k.clone(), value)
                })
                .collect()
        })
        .collect()
}

fn db_value_to_json(value: &DbValue) -> JsonValue {
    match value {
        DbValue::Null => JsonValue::Null,
        DbValue::Bool(v) => JsonValue::Bool(*v),
        DbValue::Int(v) => json!(v),
        DbValue::Float(v) => json!(v),
        DbValue::String(v) => json!(v),
        DbValue::Json(v) => v.clone(),
    }
}

fn db_value_as_f64(value: &DbValue) -> Option<f64> {
    match value {
        DbValue::Int(v) => Some(*v as f64),
        DbValue::Float(v) => Some(*v),
        DbValue::Json(JsonValue::Number(n)) => n.as_f64(),
        DbValue::String(s) => s.parse().ok(),
        _ => None,
    }
}

fn cell_text(value: &DbValue) -> String {
    json_value_text(&db_value_to_json(value))
}

fn json_value_text(value: &JsonValue) -> String {
    match value {
        JsonValue::Null => "null".to_string(),
        JsonValue::Bool(v) => v.to_string(),
        JsonValue::Number(v) => v.to_string(),
        JsonValue::String(v) => v.clone(),
        JsonValue::Array(_) | JsonValue::Object(_) => value.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::query::Literal;
    use crate::dsl::{Action, DslQuery, Filter, NodePattern};
    use serde_json::json;

    #[test]
    fn validate_dsl_checks_structure_without_alias_exactness() {
        let dsl = DslQuery {
            action: Action::Find,
            start: NodePattern {
                label: "Account".into(),
                alias: "a".into(),
            },
            traversals: Vec::new(),
            filters: vec![Filter {
                field: "a.region".into(),
                op: "in".into(),
                value: json!(["CIS", "EMEA"]),
                field_type: None,
                cardinality: None,
            }],
            return_: Vec::new(),
            group_by: Vec::new(),
            sort: Vec::new(),
            limit: None,
            prefix_label: None,
            prefix_index: None,
        };

        let spec = DslValidationSpec {
            start_label: Some("account".into()),
            required_filter_fields: vec!["region".into()],
            required_filter_ops: vec!["in".into()],
            ..Default::default()
        };

        assert!(validate_dsl(&Some(spec), &dsl).is_empty());
    }

    #[test]
    fn validate_expected_rows_matches_subset_rows() {
        let result = QueryResult {
            columns: Vec::new(),
            rows: vec![
                crate::db::Row {
                    fields: BTreeMap::from([
                        (
                            "name".into(),
                            crate::db::Value::String("Eastline Logistics".into()),
                        ),
                        ("region".into(), crate::db::Value::String("EMEA".into())),
                    ]),
                },
                crate::db::Row {
                    fields: BTreeMap::from([
                        (
                            "name".into(),
                            crate::db::Value::String("Northwind Kazakhstan".into()),
                        ),
                        ("region".into(), crate::db::Value::String("CIS".into())),
                    ]),
                },
            ],
        };

        let spec = vec![ExpectedRow {
            fields: BTreeMap::from([("name".into(), json!("Eastline Logistics"))]),
        }];

        assert!(validate_expected_rows(&spec, &result).is_empty());
    }

    #[test]
    fn mask_cypher_params_hides_embedding_vector() {
        let params = BTreeMap::from([
            (
                "embedding".into(),
                Literal::List(vec![Literal::Float(1.0), Literal::Float(2.0)]),
            ),
            ("limit".into(), Literal::Int(10)),
        ]);

        let masked = mask_cypher_params(&params, false);
        assert_eq!(
            masked.get("embedding"),
            Some(&json!("<masked embedding len=2>"))
        );
        assert_eq!(masked.get("limit"), Some(&json!(10)));
    }

    #[test]
    fn mask_cypher_params_hides_positional_embedding_vectors() {
        let params = BTreeMap::from([
            (
                "p2".into(),
                Literal::List((0..32).map(|i| Literal::Float(i as f64)).collect()),
            ),
            (
                "ids".into(),
                Literal::List(vec![
                    Literal::String("a".into()),
                    Literal::String("b".into()),
                ]),
            ),
        ]);

        let masked = mask_cypher_params(&params, false);
        assert_eq!(masked.get("p2"), Some(&json!("<masked embedding len=32>")));
        assert_eq!(masked.get("ids"), Some(&json!(["a", "b"])));
    }

    #[test]
    fn result_rows_json_masks_nested_embedding_vectors_by_default() {
        let result = QueryResult {
            columns: Vec::new(),
            rows: vec![crate::db::Row {
                fields: BTreeMap::from([(
                    "entities".into(),
                    crate::db::Value::Json(json!([
                        {
                            "name": "Acme",
                            "embedding": [1.0, 2.0, 3.0],
                            "payload": {
                                "vec": [1.0, 2.0]
                            }
                        }
                    ])),
                )]),
            }],
        };

        let rows = result_rows_json(&result, false);
        assert_eq!(
            rows[0]["entities"][0]["embedding"],
            json!("<masked embedding len=3>")
        );
        assert_eq!(
            rows[0]["entities"][0]["payload"]["vec"],
            json!("<masked embedding len=2>")
        );

        let rows_with_embeddings = result_rows_json(&result, true);
        assert_eq!(
            rows_with_embeddings[0]["entities"][0]["embedding"],
            json!([1.0, 2.0, 3.0])
        );
    }
}
