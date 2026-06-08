//! LLM-driven generation of a linguagraph **mapping** document.
//!
//! Given an **ontology** (mandatory), a **JSON data** document, and an
//! optional **live graph schema**, this module asks an [`LlmClient`] to
//! emit a [`Mapping`] and drives it to a verified, ontology-conformant
//! result:
//!
//! ```text
//!   data ─┐
//!         ├─ promptgen::analyze ─► summary (PK / type heuristics)
//!   ontology (whitelist) ─┐
//!   live schema (opt.) ───┤
//!                         ▼
//!            build_mapping_prompt → (system, user)
//!                         │
//!                         ▼
//!              LlmClient::complete ──► raw text
//!                         │  strip fences
//!                         ▼
//!         Mapping::from_str (parse + validate)
//!                         │
//!         enforce_ontology (STRICT entity-type whitelist, canonicalise)
//!                         │
//!         mapper::extract  (verify primary keys resolve)
//!                         │  any failure ─► repair round-trip (bounded)
//!                         ▼
//!                    valid Mapping
//! ```
//!
//! Entity types are constrained to the ontology; relationships and extra
//! properties may be invented by the model.

mod describe;
mod prompt;

#[cfg(feature = "interactive")]
mod interactive;

pub use describe::{describe_properties, DescribeOptions};
pub use prompt::{build_mapping_prompt, MapGenPromptOptions};

#[cfg(feature = "interactive")]
pub use interactive::refine_interactively;

use serde_json::Value;
use thiserror::Error;

use crate::graph::DomainOntology;
use crate::llm::{LlmClient, LlmError};
use crate::mapper::{self, Mapping, MapperError};
use crate::prompt::GraphSchema;

/// Errors raised while generating a mapping.
#[derive(Debug, Error)]
pub enum MapGenError {
    #[error("LLM error: {0}")]
    Llm(#[from] LlmError),

    #[error("the ontology has no entity types — at least one is required")]
    EmptyOntology,

    #[error("could not parse/validate the generated mapping: {0}")]
    Parse(MapperError),

    #[error("entity type `{found}` is not in the ontology (allowed: {allowed})")]
    EntityTypeNotInOntology { found: String, allowed: String },

    #[error("the generated mapping failed extraction verification: {0}")]
    Verify(MapperError),

    #[error(
        "the LLM did not produce a valid mapping after {attempts} attempt(s); \
         last error: {last}"
    )]
    ExhaustedRepairs { attempts: usize, last: String },

    #[error("interactive prompt error: {0}")]
    Interactive(String),
}

/// Top-level options for [`generate_mapping`].
#[derive(Debug, Clone)]
pub struct MapGenOptions {
    /// How many *additional* attempts to make after the first failure.
    /// Total LLM calls = `max_repair_attempts + 1`. Default: 2.
    pub max_repair_attempts: usize,
    /// Prompt-assembly knobs.
    pub prompt: MapGenPromptOptions,
}

impl Default for MapGenOptions {
    fn default() -> Self {
        Self {
            max_repair_attempts: 2,
            prompt: MapGenPromptOptions::default(),
        }
    }
}

impl MapGenOptions {
    fn total_attempts(&self) -> usize {
        self.max_repair_attempts + 1
    }
}

/// Generate a verified, ontology-conformant [`Mapping`] for `data`.
///
/// * `ontology` — **required**; entity types are strictly whitelisted to
///   `ontology.entity_types`.
/// * `schema` — optional live graph schema the model is asked to reuse.
/// * `llm` — the backend that turns the prompt into a mapping.
///
/// On a parse / whitelist / verification failure the prompt is replayed
/// with the failed output and the error appended, up to
/// `opts.max_repair_attempts` extra times.
pub async fn generate_mapping(
    data: &Value,
    ontology: &DomainOntology,
    schema: Option<&GraphSchema>,
    llm: &dyn LlmClient,
    opts: &MapGenOptions,
) -> Result<Mapping, MapGenError> {
    if ontology.entity_types.is_empty() {
        return Err(MapGenError::EmptyOntology);
    }

    let summary = crate::promptgen::analyze(data);
    let (system, base_user) = build_mapping_prompt(data, &summary, ontology, schema, &opts.prompt);

    let mut user = base_user.clone();
    let mut last_err: Option<MapGenError> = None;

    for _ in 0..opts.total_attempts() {
        let raw = llm.complete(&system, &user).await?;
        let cleaned = strip_code_fences(&raw);
        match finalize(&cleaned, ontology, data) {
            Ok(mapping) => return Ok(mapping),
            Err(e) => {
                let msg = e.to_string();
                last_err = Some(e);
                user = format!(
                    "{base_user}\n\n# Your previous output was INVALID\n\
                     Previous output:\n```json\n{raw}\n```\n\n\
                     Error: {msg}\n\n\
                     Fix the problem and output ONLY the corrected JSON mapping.",
                );
            }
        }
    }

    Err(MapGenError::ExhaustedRepairs {
        attempts: opts.total_attempts(),
        last: last_err.map(|e| e.to_string()).unwrap_or_default(),
    })
}

/// Parse, validate, whitelist-check, and verify a candidate mapping.
fn finalize(cleaned: &str, ontology: &DomainOntology, data: &Value) -> Result<Mapping, MapGenError> {
    let mut mapping = Mapping::from_str(cleaned).map_err(MapGenError::Parse)?;
    enforce_ontology(&mut mapping, ontology)?;
    // Verify primary keys actually resolve against the data.
    mapper::extract(&mapping, data).map_err(MapGenError::Verify)?;
    Ok(mapping)
}

/// Enforce the STRICT entity-type whitelist and canonicalise spellings.
///
/// Entity `type` values are matched case-insensitively against the
/// ontology and rewritten to the ontology's canonical spelling. A type
/// with no match is rejected. Relationship endpoints are likewise
/// canonicalised to the emitted entity types so the downstream planner
/// resolves them.
fn enforce_ontology(mapping: &mut Mapping, ontology: &DomainOntology) -> Result<(), MapGenError> {
    let allowed: Vec<&str> = ontology
        .entity_types
        .iter()
        .map(|e| e.name.as_str())
        .collect();

    for ent in &mut mapping.entities {
        match allowed
            .iter()
            .find(|name| name.eq_ignore_ascii_case(ent.kind.trim()))
        {
            Some(canon) => ent.kind = (*canon).to_string(),
            None => {
                return Err(MapGenError::EntityTypeNotInOntology {
                    found: ent.kind.clone(),
                    allowed: allowed.join(", "),
                });
            }
        }
    }

    let emitted: Vec<String> = mapping.entities.iter().map(|e| e.kind.clone()).collect();
    for rel in &mut mapping.relationships {
        canonicalize_endpoint(&mut rel.from, &emitted);
        canonicalize_endpoint(&mut rel.to, &emitted);
    }
    Ok(())
}

fn canonicalize_endpoint(endpoint: &mut String, emitted: &[String]) {
    if let Some(canon) = emitted.iter().find(|e| e.eq_ignore_ascii_case(endpoint.trim())) {
        *endpoint = canon.clone();
    }
}

/// Pull a JSON object out of a model completion: unwrap a fenced code
/// block if present, otherwise slice from the first `{` to the last `}`.
fn strip_code_fences(s: &str) -> String {
    let t = s.trim();

    if let Some(rest) = t.strip_prefix("```") {
        // Drop an optional language tag on the fence's opening line.
        let after_lang = rest.split_once('\n').map(|x| x.1).unwrap_or(rest);
        let inner = match after_lang.rfind("```") {
            Some(end) => &after_lang[..end],
            None => after_lang,
        };
        return inner.trim().to_string();
    }

    if let (Some(start), Some(end)) = (t.find('{'), t.rfind('}')) {
        if end >= start {
            return t[start..=end].to_string();
        }
    }

    t.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{EntityTypeSpec, RelationTypeSpec};
    use crate::llm::MockLlmClient;
    use serde_json::json;

    fn data() -> Value {
        json!({
            "companies": [
                {"id": 1, "name": "Stripe", "industry": "Fintech"},
                {"id": 2, "name": "Acme", "industry": "Manufacturing"}
            ]
        })
    }

    fn ontology() -> DomainOntology {
        DomainOntology {
            entity_types: vec![EntityTypeSpec::with_description("Company", "A business.")],
            relation_types: vec![RelationTypeSpec::new("OWNS")],
        }
    }

    fn good_mapping() -> String {
        json!({
            "entities": [{
                "type": "Company",
                "source_path": "$.companies[*]",
                "primary_key": "$.companies[*].id",
                "properties": [
                    {"name": "name", "type": "Text", "source_path": "$.companies[*].name"},
                    {"name": "industry", "type": "Keyword", "source_path": "$.companies[*].industry"}
                ]
            }],
            "relationships": []
        })
        .to_string()
    }

    #[tokio::test]
    async fn happy_path_returns_validated_mapping() {
        let llm = MockLlmClient::single(good_mapping());
        let mapping = generate_mapping(&data(), &ontology(), None, &llm, &MapGenOptions::default())
            .await
            .unwrap();
        assert_eq!(mapping.entities.len(), 1);
        assert_eq!(mapping.entities[0].kind, "Company");
        assert_eq!(llm.call_count(), 1);
    }

    #[tokio::test]
    async fn empty_ontology_is_rejected() {
        let llm = MockLlmClient::single(good_mapping());
        let err = generate_mapping(
            &data(),
            &DomainOntology::default(),
            None,
            &llm,
            &MapGenOptions::default(),
        )
        .await
        .unwrap_err();
        assert!(matches!(err, MapGenError::EmptyOntology));
        // No LLM call should have happened.
        assert_eq!(llm.call_count(), 0);
    }

    #[tokio::test]
    async fn entity_type_outside_ontology_triggers_repair_then_succeeds() {
        // First a forbidden type, then a valid mapping.
        let bad = json!({
            "entities": [{
                "type": "Organization",
                "source_path": "$.companies[*]",
                "primary_key": "$.companies[*].id",
                "properties": [{"name": "name", "type": "Text", "source_path": "$.companies[*].name"}]
            }],
            "relationships": []
        })
        .to_string();
        let llm = MockLlmClient::new([bad, good_mapping()]);
        let mapping = generate_mapping(&data(), &ontology(), None, &llm, &MapGenOptions::default())
            .await
            .unwrap();
        assert_eq!(mapping.entities[0].kind, "Company");
        assert_eq!(llm.call_count(), 2);
        // The repair turn must carry the whitelist error.
        let repair_user = llm.calls()[1].1.clone();
        assert!(repair_user.contains("is not in the ontology"));
    }

    #[tokio::test]
    async fn case_insensitive_type_is_canonicalised() {
        let lower = json!({
            "entities": [{
                "type": "company",
                "source_path": "$.companies[*]",
                "primary_key": "$.companies[*].id",
                "properties": [{"name": "name", "type": "Text", "source_path": "$.companies[*].name"}]
            }],
            "relationships": []
        })
        .to_string();
        let llm = MockLlmClient::single(lower);
        let mapping = generate_mapping(&data(), &ontology(), None, &llm, &MapGenOptions::default())
            .await
            .unwrap();
        assert_eq!(mapping.entities[0].kind, "Company");
        assert_eq!(llm.call_count(), 1);
    }

    #[tokio::test]
    async fn invented_relationship_is_accepted() {
        let with_rel = json!({
            "entities": [
                {
                    "type": "Company",
                    "source_path": "$.companies[*]",
                    "primary_key": "$.companies[*].id",
                    "properties": [{"name": "name", "type": "Text", "source_path": "$.companies[*].name"}]
                }
            ],
            "relationships": [
                {"type": "COMPETES_WITH", "from": "company", "to": "Company"}
            ]
        })
        .to_string();
        let llm = MockLlmClient::single(with_rel);
        let mapping = generate_mapping(&data(), &ontology(), None, &llm, &MapGenOptions::default())
            .await
            .unwrap();
        assert_eq!(mapping.relationships.len(), 1);
        // Endpoints canonicalised to the emitted entity type.
        assert_eq!(mapping.relationships[0].from, "Company");
        assert_eq!(mapping.relationships[0].to, "Company");
    }

    #[tokio::test]
    async fn end_to_end_on_bundled_example_file() {
        // Exercise the full path against the real example document and a
        // mapping shaped like the bundled `companies_mapping.json`.
        let raw = std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/examples/companies_data.json"
        ))
        .expect("example data file present");
        let doc: Value = serde_json::from_str(&raw).unwrap();

        let mapping_json = json!({
            "entities": [{
                "type": "Company",
                "source_path": "$.companies[*]",
                "primary_key": "$.companies[*].id",
                "properties": [
                    {"name": "id", "type": "Text", "source_path": "$.companies[*].id"},
                    {"name": "name", "type": "SemanticText", "source_path": "$.companies[*].name"},
                    {"name": "description", "type": "SemanticText", "source_path": "$.companies[*].description"},
                    {"name": "industry", "type": "Keyword", "source_path": "$.companies[*].industry"}
                ]
            }],
            "relationships": []
        })
        .to_string();

        let llm = MockLlmClient::single(mapping_json);
        let mapping = generate_mapping(&doc, &ontology(), None, &llm, &MapGenOptions::default())
            .await
            .unwrap();
        assert_eq!(mapping.entities[0].kind, "Company");
        // Verification ran `mapper::extract`; confirm it resolves the
        // three example rows here too.
        let extracted = mapper::extract(&mapping, &doc).unwrap();
        assert_eq!(extracted.entities[0].rows.len(), 3);
    }

    #[tokio::test]
    async fn fenced_and_prose_wrapped_json_is_recovered() {
        let wrapped = format!("Here is your mapping:\n```json\n{}\n```\nDone!", good_mapping());
        let llm = MockLlmClient::single(wrapped);
        let mapping = generate_mapping(&data(), &ontology(), None, &llm, &MapGenOptions::default())
            .await
            .unwrap();
        assert_eq!(mapping.entities[0].kind, "Company");
    }

    #[tokio::test]
    async fn exhausts_repairs_on_persistent_garbage() {
        let llm = MockLlmClient::single("not json at all");
        let opts = MapGenOptions {
            max_repair_attempts: 1,
            ..MapGenOptions::default()
        };
        let err = generate_mapping(&data(), &ontology(), None, &llm, &opts)
            .await
            .unwrap_err();
        match err {
            MapGenError::ExhaustedRepairs { attempts, .. } => assert_eq!(attempts, 2),
            other => panic!("expected ExhaustedRepairs, got {other:?}"),
        }
        // 1 initial + 1 repair = 2 calls.
        assert_eq!(llm.call_count(), 2);
    }

    #[test]
    fn strip_fences_variants() {
        assert_eq!(strip_code_fences("```json\n{\"a\":1}\n```"), "{\"a\":1}");
        assert_eq!(strip_code_fences("```\n{\"a\":1}\n```"), "{\"a\":1}");
        assert_eq!(strip_code_fences("prefix {\"a\":1} suffix"), "{\"a\":1}");
        assert_eq!(strip_code_fences("  {\"a\":1}  "), "{\"a\":1}");
    }
}
