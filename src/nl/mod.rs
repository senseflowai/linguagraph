//! Natural-language front-end: question → JSON DSL, rows → answer.
//!
//! [`NlTranslator`] bundles the pieces every NL consumer needs — an
//! [`LlmClient`], the prompt-selection embedder/store pair, and the
//! query-prompt tunables — behind two calls:
//!
//! * [`NlTranslator::question_to_dsl`] — build a query-tailored compact
//!   prompt ([`crate::prompt::generate_query_prompt`]), ask the LLM for a
//!   [`DslQuery`], validate with [`crate::dsl::parse_str`], and self-repair
//!   up to `max_repairs` times by feeding the validation error back.
//! * [`NlTranslator::synthesize_answer`] — turn result rows into a concise
//!   natural-language answer grounded in a query summary.
//!
//! The module is deliberately backend-free: it takes `Arc<dyn LlmClient>`
//! and never touches cargo features. Both the e2e harness
//! ([`crate::e2e`]) and the explorer ([`crate::explore`]) drive it.

use std::collections::BTreeMap;
use std::fmt;
use std::sync::Arc;

use serde_json::{json, Value as JsonValue};
use thiserror::Error;

use crate::ast::query::Literal;
use crate::config::Config;
use crate::dsl::{self, DslQuery};
use crate::embeddings::{EmbeddingIndex, SharedEmbedder, SharedEmbeddingStore};
use crate::graph::OntologyCatalog;
use crate::llm::{LlmClient, LlmError};
use crate::prompt::{self, GraphSchema};

/// Errors surfaced by the NL front-end.
#[derive(Debug, Error)]
pub enum NlError {
    /// The embedding store backing prompt selection failed.
    #[error("embedding store error: {0}")]
    Store(String),

    /// Query-tailored prompt generation failed.
    #[error("query prompt generation failed: {0}")]
    Prompt(String),

    /// The LLM request itself failed (transport, API, empty completion).
    #[error("LLM request failed: {0}")]
    Llm(#[from] LlmError),

    /// The completion contained no JSON object at all.
    #[error("LLM output did not contain a JSON object")]
    MissingJson,

    /// A JSON object was extracted but does not parse.
    #[error("invalid extracted JSON: {0}")]
    InvalidJson(String),

    /// Every attempt (initial + repairs) produced invalid DSL.
    #[error("could not produce valid DSL: {last_error}; last output: {last_output}")]
    InvalidDsl {
        last_error: String,
        last_output: String,
    },

    #[error("serialization error: {0}")]
    Serialize(#[from] serde_json::Error),
}

/// Successful outcome of [`NlTranslator::question_to_dsl`].
#[derive(Debug, Clone)]
pub struct DslGeneration {
    /// The validated query, with prefixes already forced.
    pub dsl: DslQuery,
    /// Number of LLM completions consumed (1 = no repair needed).
    pub attempts: usize,
    /// The system prompt that produced the DSL — kept for transparency
    /// ("how was this answered") and debugging.
    pub system_prompt: String,
}

/// Reusable NL→DSL translator + answer synthesizer.
///
/// Cheap to share: hold it behind an `Arc` and clone that.
pub struct NlTranslator {
    llm: Arc<dyn LlmClient>,
    embedder: SharedEmbedder,
    store: SharedEmbeddingStore,
    /// Qdrant/in-memory collection holding the prompt-selection vectors.
    collection: String,
    /// Embedding-model identifier folded into point ids.
    model_id: String,
    prompt_params: prompt::QueryPromptParams,
    max_repairs: usize,
}

impl fmt::Debug for NlTranslator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("NlTranslator")
            .field("collection", &self.collection)
            .field("model_id", &self.model_id)
            .field("max_repairs", &self.max_repairs)
            .finish_non_exhaustive()
    }
}

impl NlTranslator {
    pub fn new(
        llm: Arc<dyn LlmClient>,
        embedder: SharedEmbedder,
        store: SharedEmbeddingStore,
        collection: impl Into<String>,
        model_id: impl Into<String>,
    ) -> Self {
        Self {
            llm,
            embedder,
            store,
            collection: collection.into(),
            model_id: model_id.into(),
            prompt_params: prompt::QueryPromptParams::default(),
            max_repairs: 1,
        }
    }

    /// Wire the translator from configuration the same way the e2e harness
    /// does: prompt-selection thresholds from `[graph_specification]`, the
    /// vector collection from `[qdrant]`.
    pub fn from_config(
        cfg: &Config,
        llm: Arc<dyn LlmClient>,
        embedder: SharedEmbedder,
        store: SharedEmbeddingStore,
    ) -> Self {
        let ontology_cfg = &cfg.ontology_catalog;
        let prompt_params = prompt::QueryPromptParams {
            domain_threshold: ontology_cfg.domain_selection_threshold,
            domain_top_k: ontology_cfg.domain_selection_top_k,
            selection: prompt::QuerySelectionParams {
                entity_threshold: ontology_cfg.entity_selection_threshold,
                property_threshold: ontology_cfg.property_selection_threshold,
                neighbor_hops: ontology_cfg.selection_neighbor_hops,
                ..prompt::QuerySelectionParams::default()
            },
            ..prompt::QueryPromptParams::default()
        };
        Self {
            prompt_params,
            ..Self::new(
                llm,
                embedder,
                store,
                cfg.qdrant.collection.clone(),
                cfg.ontology_catalog
                    .embedding_model
                    .clone()
                    .unwrap_or_else(|| "mock".to_string()),
            )
        }
    }

    /// Maximum number of repair rounds after the first attempt.
    pub fn with_max_repairs(mut self, max_repairs: usize) -> Self {
        self.max_repairs = max_repairs;
        self
    }

    pub fn with_prompt_params(mut self, params: prompt::QueryPromptParams) -> Self {
        self.prompt_params = params;
        self
    }

    /// The underlying LLM client, for callers that need extra completions
    /// (e.g. the e2e judge).
    pub fn llm(&self) -> Arc<dyn LlmClient> {
        self.llm.clone()
    }

    /// Translate `question` into a validated [`DslQuery`].
    ///
    /// `prefix_label` / `prefix_index` are **forced** onto the result (and
    /// cleared when `None`) — the LLM is never trusted with tenant scoping.
    pub async fn question_to_dsl(
        &self,
        question: &str,
        schema: &GraphSchema,
        catalog: &OntologyCatalog,
        prefix_label: Option<&str>,
        prefix_index: Option<&str>,
    ) -> Result<DslGeneration, NlError> {
        self.store
            .ensure(&self.collection, self.embedder.dim())
            .await
            .map_err(|e| NlError::Store(e.to_string()))?;
        let index = EmbeddingIndex {
            store: self.store.as_ref(),
            collection: &self.collection,
            model: &self.model_id,
        };
        let mut schema = schema.clone();
        let system = prompt::generate_query_prompt(
            question,
            &mut schema,
            catalog,
            self.embedder.as_ref(),
            &index,
            &self.prompt_params,
        )
        .await
        .map_err(|e| NlError::Prompt(e.to_string()))?;

        let mut user = format!(
            "Question:\n{question}\n\nReturn only one JSON DSL object. Do not wrap it in Markdown."
        );
        let mut last_output = String::new();
        let mut last_error = String::new();

        for attempt in 0..=self.max_repairs {
            let raw = self.llm.complete(&system, &user).await?;
            last_output = raw.clone();
            let parsed = extract_json_object(&raw)
                .map_err(|e| e.to_string())
                .and_then(|json| {
                    serde_json::from_str::<DslQuery>(&json).map_err(|e| e.to_string())
                })
                .and_then(|mut dsl| {
                    dsl.prefix_label = prefix_label.map(Into::into);
                    dsl.prefix_index = prefix_index.map(Into::into);
                    let raw = serde_json::to_string(&dsl).map_err(|e| e.to_string())?;
                    dsl::parse_str(&raw).map_err(|e| e.to_string())
                });
            match parsed {
                Ok(dsl) => {
                    return Ok(DslGeneration {
                        dsl,
                        attempts: attempt + 1,
                        system_prompt: system,
                    })
                }
                Err(err) => {
                    last_error = err;
                    if attempt < self.max_repairs {
                        user = format!(
                            "Repair the JSON DSL for this question.\n\nQuestion:\n{question}\n\n\
                             Previous output:\n{last_output}\n\nValidation error:\n{last_error}\n\n\
                             Return only the corrected JSON object."
                        );
                    }
                }
            }
        }

        Err(NlError::InvalidDsl {
            last_error,
            last_output,
        })
    }

    /// Answer `question` using only the supplied result rows.
    /// `query_summary` tells the model the rows are already filtered
    /// (typically [`DslQuery::describe`]). The rows are assumed to already
    /// be shown to the user (e.g. as a table alongside the answer), so the
    /// prompt instructs the model to answer directly instead of
    /// restating/listing them back — that's pure wasted output tokens.
    pub async fn synthesize_answer(
        &self,
        question: &str,
        query_summary: &str,
        rows: &[BTreeMap<String, JsonValue>],
    ) -> Result<String, NlError> {
        let system = "Answer the user's question directly and concisely, using only the \
                      supplied graph query rows as your source of truth. The rows are already \
                      rendered as a table right next to your answer, so do not restate, list, \
                      or paraphrase them back to the user — that wastes tokens repeating what \
                      they can already see. State just the specific value, count, or judgment \
                      the question asks for. Name an individual row only when the question \
                      singles one out (e.g. \"which is cheapest\") — never enumerate the whole \
                      result set. If the rows are empty, say the data does not contain the \
                      answer.";
        let user = json!({
            "question": question,
            "query_summary": query_summary,
            "rows": rows,
        });
        let answer = self
            .llm
            .complete(system, &serde_json::to_string_pretty(&user)?)
            .await?;
        Ok(answer.trim().to_string())
    }
}

/// Extract the first balanced top-level JSON object from `raw`, tolerating
/// surrounding prose / markdown fences. A fully-JSON payload is returned
/// as-is.
pub fn extract_json_object(raw: &str) -> Result<String, NlError> {
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
                        .map_err(|e| NlError::InvalidJson(format!("{e}: {s}")))?;
                    return Ok(s.to_string());
                }
            }
            _ => {}
        }
    }
    Err(NlError::MissingJson)
}

/// Render Cypher params as JSON with embedding vectors masked (unless
/// `include_embeddings`), so query traces stay human-readable.
pub fn mask_cypher_params(
    params: &BTreeMap<String, Literal>,
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

pub(crate) fn is_masked_embedding_name(name: &str) -> bool {
    let name = name.to_ascii_lowercase();
    name.contains("embedding")
        || name == "emb"
        || name == "vec"
        || name == "vecs"
        || name.ends_with("_emb")
        || name.ends_with("_vec")
        || name.ends_with("_embedding")
}

pub(crate) fn masked_embedding_value(value: &JsonValue) -> JsonValue {
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

pub(crate) fn sanitize_report_json(
    value: JsonValue,
    mask_positional_numeric_vectors: bool,
) -> JsonValue {
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

/// Convert an AST literal into plain JSON.
pub fn literal_to_json(literal: &Literal) -> JsonValue {
    match literal {
        Literal::String(s) => JsonValue::String(s.clone()),
        Literal::Bool(b) => JsonValue::Bool(*b),
        Literal::Int(i) => JsonValue::Number((*i).into()),
        Literal::Float(f) => serde_json::Number::from_f64(*f)
            .map(JsonValue::Number)
            .unwrap_or(JsonValue::Null),
        Literal::List(items) => JsonValue::Array(items.iter().map(literal_to_json).collect()),
        Literal::Object(map) => JsonValue::Object(
            map.iter()
                .map(|(key, value)| (key.clone(), literal_to_json(value)))
                .collect(),
        ),
        Literal::Null => JsonValue::Null,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embeddings::{InMemoryEmbeddingStore, MockEmbedder};
    use crate::llm::MockLlmClient;
    use serde_json::json;

    fn translator(llm: MockLlmClient, max_repairs: usize) -> NlTranslator {
        NlTranslator::new(
            Arc::new(llm),
            Arc::new(MockEmbedder::new(32)),
            Arc::new(InMemoryEmbeddingStore::new()),
            "nl_test",
            "mock",
        )
        .with_max_repairs(max_repairs)
    }

    fn empty_schema() -> GraphSchema {
        GraphSchema {
            nodes: Vec::new(),
            relationships: Vec::new(),
        }
    }

    const VALID_DSL: &str = r#"{
        "start": { "label": "Person", "alias": "p" },
        "return": [ { "field": "p.name" } ]
    }"#;

    #[tokio::test]
    async fn question_to_dsl_happy_path_forces_prefixes() {
        let t = translator(MockLlmClient::single(VALID_DSL), 1);
        let generation = t
            .question_to_dsl(
                "who is there?",
                &empty_schema(),
                &OntologyCatalog::default(),
                Some("T1"),
                Some("t1_idx"),
            )
            .await
            .expect("valid DSL");
        assert_eq!(generation.attempts, 1);
        assert_eq!(generation.dsl.start.label, "Person");
        assert_eq!(generation.dsl.prefix_label.as_deref(), Some("T1"));
        assert_eq!(generation.dsl.prefix_index.as_deref(), Some("t1_idx"));
        assert!(!generation.system_prompt.is_empty());
    }

    #[tokio::test]
    async fn question_to_dsl_repairs_after_invalid_output() {
        let llm = MockLlmClient::new(["this is not json", VALID_DSL]);
        let t = translator(llm, 1);
        let generation = t
            .question_to_dsl(
                "who is there?",
                &empty_schema(),
                &OntologyCatalog::default(),
                None,
                None,
            )
            .await
            .expect("repaired DSL");
        assert_eq!(generation.attempts, 2);
        assert_eq!(generation.dsl.prefix_label, None);
    }

    #[tokio::test]
    async fn question_to_dsl_gives_up_after_max_repairs() {
        let t = translator(MockLlmClient::single("still not json"), 1);
        let err = t
            .question_to_dsl(
                "who is there?",
                &empty_schema(),
                &OntologyCatalog::default(),
                None,
                None,
            )
            .await
            .expect_err("must exhaust repairs");
        assert!(matches!(err, NlError::InvalidDsl { .. }), "got: {err}");
    }

    #[tokio::test]
    async fn synthesize_answer_trims_completion() {
        let t = translator(MockLlmClient::single("  The answer.  "), 0);
        let rows = vec![BTreeMap::from([("name".to_string(), json!("Alice"))])];
        let answer = t
            .synthesize_answer("who?", "Selecting Person entities.", &rows)
            .await
            .unwrap();
        assert_eq!(answer, "The answer.");
    }

    #[test]
    fn extract_json_object_from_prose() {
        let raw = "Sure! Here is the query:\n```json\n{\"a\": {\"b\": 1}}\n```\nDone.";
        assert_eq!(extract_json_object(raw).unwrap(), "{\"a\": {\"b\": 1}}");
    }

    #[test]
    fn extract_json_object_rejects_missing_object() {
        assert!(matches!(
            extract_json_object("no json here"),
            Err(NlError::MissingJson)
        ));
    }

    #[test]
    fn mask_cypher_params_hides_embedding_vector() {
        let params = BTreeMap::from([
            (
                "embedding".to_string(),
                Literal::List(vec![Literal::Float(1.0), Literal::Float(2.0)]),
            ),
            ("limit".to_string(), Literal::Int(10)),
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
                "p2".to_string(),
                Literal::List((0..32).map(|i| Literal::Float(i as f64)).collect()),
            ),
            (
                "ids".to_string(),
                Literal::List(vec![
                    Literal::String("a".to_string()),
                    Literal::String("b".to_string()),
                ]),
            ),
        ]);

        let masked = mask_cypher_params(&params, false);
        assert_eq!(masked.get("p2"), Some(&json!("<masked embedding len=32>")));
        assert_eq!(masked.get("ids"), Some(&json!(["a", "b"])));
    }
}
