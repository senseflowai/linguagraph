//! Optional enrichment step: fill in concise property `description`s using
//! the LLM.
//!
//! Each property is described from two cheap, high-signal inputs:
//!
//! * the **entity type description** from the ontology (what kind of thing
//!   carries this property), and
//! * **1–2 real sample values** pulled from the source document at the
//!   property's `source_path`.
//!
//! Requests are issued **one per property and run concurrently** (bounded
//! by [`DescribeOptions::max_concurrency`]) so a wide mapping doesn't
//! serialize the round-trips. The step is best-effort: a failed or empty
//! completion leaves that property's description untouched.

use std::fmt::Write;

use futures::stream::{self, StreamExt};
use serde_json::Value;

use crate::graph::DomainOntology;
use crate::llm::LlmClient;
use crate::mapper::{JsonPath, Mapping};

use super::MapGenError;

/// Knobs for [`describe_properties`].
#[derive(Debug, Clone)]
pub struct DescribeOptions {
    /// Maximum number of in-flight LLM requests.
    pub max_concurrency: usize,
    /// How many distinct sample values to show the model per property.
    pub max_samples: usize,
    /// Re-describe properties that already carry a non-empty description.
    /// Off by default — only missing descriptions are filled.
    pub overwrite_existing: bool,
}

impl Default for DescribeOptions {
    fn default() -> Self {
        Self {
            max_concurrency: 8,
            max_samples: 2,
            overwrite_existing: false,
        }
    }
}

/// One queued description request, addressed by its location in the mapping.
struct Job {
    entity_idx: usize,
    prop_idx: usize,
    system: String,
    user: String,
}

/// Fill in `description` for the mapping's properties using the LLM.
///
/// Mutates `mapping` in place and returns how many descriptions were
/// written. Properties that already have a description are skipped unless
/// [`DescribeOptions::overwrite_existing`] is set.
pub async fn describe_properties(
    mapping: &mut Mapping,
    ontology: &DomainOntology,
    data: &Value,
    llm: &dyn LlmClient,
    opts: &DescribeOptions,
) -> Result<usize, MapGenError> {
    // 1. Collect jobs while we only need an immutable borrow of `mapping`.
    let mut jobs: Vec<Job> = Vec::new();
    for (entity_idx, ent) in mapping.entities.iter().enumerate() {
        let entity_desc = ontology
            .entity_types
            .iter()
            .find(|e| e.name.eq_ignore_ascii_case(ent.kind.trim()))
            .and_then(|e| e.description.clone());

        for (prop_idx, prop) in ent.properties.iter().enumerate() {
            let already = prop
                .description
                .as_deref()
                .map(|d| !d.trim().is_empty())
                .unwrap_or(false);
            if already && !opts.overwrite_existing {
                continue;
            }
            let samples = sample_values(data, &prop.source_path, opts.max_samples);
            let (system, user) = build_describe_prompt(
                &ent.kind,
                entity_desc.as_deref(),
                &prop.name,
                prop.field_type.as_deref(),
                &samples,
            );
            jobs.push(Job {
                entity_idx,
                prop_idx,
                system,
                user,
            });
        }
    }

    if jobs.is_empty() {
        return Ok(0);
    }

    // 2. Run the requests concurrently, bounded by `max_concurrency`.
    let concurrency = opts.max_concurrency.max(1);
    let results: Vec<(usize, usize, Option<String>)> = stream::iter(jobs.into_iter().map(|job| {
        async move {
            match llm.complete(&job.system, &job.user).await {
                Ok(raw) => (job.entity_idx, job.prop_idx, Some(clean_description(&raw))),
                Err(e) => {
                    tracing::warn!(
                        target: "linguagraph::mapgen",
                        error = %e,
                        "property description request failed; leaving it unset"
                    );
                    (job.entity_idx, job.prop_idx, None)
                }
            }
        }
    }))
    .buffer_unordered(concurrency)
    .collect()
    .await;

    // 3. Write the descriptions back (mutable borrow).
    let mut written = 0;
    for (entity_idx, prop_idx, desc) in results {
        if let Some(desc) = desc.filter(|d| !d.is_empty()) {
            mapping.entities[entity_idx].properties[prop_idx].description = Some(desc);
            written += 1;
        }
    }
    Ok(written)
}

/// Up to `max` distinct, non-null stringified sample values at `source_path`.
fn sample_values(data: &Value, source_path: &str, max: usize) -> Vec<String> {
    let Ok(path) = JsonPath::parse(source_path) else {
        return Vec::new();
    };
    let mut out: Vec<String> = Vec::new();
    for m in path.evaluate(data) {
        if matches!(m.value, Value::Null) {
            continue;
        }
        let s = sample_to_string(m.value);
        if s.trim().is_empty() || out.contains(&s) {
            continue;
        }
        out.push(s);
        if out.len() >= max {
            break;
        }
    }
    out
}

fn sample_to_string(value: &Value) -> String {
    let s = match value {
        Value::String(s) => s.clone(),
        other => other.to_string(),
    };
    if s.chars().count() > 80 {
        let cut: String = s.chars().take(77).collect();
        format!("{cut}...")
    } else {
        s
    }
}

fn build_describe_prompt(
    entity_kind: &str,
    entity_desc: Option<&str>,
    prop_name: &str,
    prop_type: Option<&str>,
    samples: &[String],
) -> (String, String) {
    let system = "You write concise, single-sentence descriptions of properties in a \
         knowledge graph. Describe what the property holds and means — not its data type. \
         Output ONLY the description text: one sentence, no surrounding quotes, no property \
         name prefix, no markdown, at most ~120 characters."
        .to_string();

    let mut user = String::new();
    let _ = writeln!(user, "Entity type: {entity_kind}");
    if let Some(d) = entity_desc {
        let _ = writeln!(user, "Entity description: {d}");
    }
    let _ = writeln!(user, "Property name: {prop_name}");
    if let Some(t) = prop_type {
        let _ = writeln!(user, "Property type: {t}");
    }
    if samples.is_empty() {
        let _ = writeln!(user, "Sample values: (none available)");
    } else {
        let rendered: Vec<String> = samples.iter().map(|s| format!("`{s}`")).collect();
        let _ = writeln!(user, "Sample value(s) from the data: {}", rendered.join(", "));
    }
    user.push_str("\nWrite the description now.");
    (system, user)
}

/// Normalize a model completion into a single clean line.
fn clean_description(raw: &str) -> String {
    let line = raw
        .lines()
        .map(str::trim)
        .find(|l| !l.is_empty())
        .unwrap_or("")
        .trim_matches(|c| c == '"' || c == '\'' || c == '`')
        .trim();
    if line.chars().count() > 160 {
        let cut: String = line.chars().take(157).collect();
        format!("{cut}...")
    } else {
        line.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{DomainOntology, EntityTypeSpec};
    use crate::llm::MockLlmClient;
    use serde_json::json;

    fn mapping() -> Mapping {
        serde_json::from_value(json!({
            "entities": [{
                "type": "Camera",
                "source_path": "$.cameras[*]",
                "primary_key": "$.cameras[*].id",
                "properties": [
                    {"name": "id", "source_path": "$.cameras[*].id", "type": "Text",
                     "description": "Existing description."},
                    {"name": "min_event_interval", "type": "Number",
                     "source_path": "$.cameras[*].video_analytics.modules[*].config.roi[*].min_event_interval"}
                ]
            }]
        }))
        .unwrap()
    }

    fn ontology() -> DomainOntology {
        DomainOntology {
            entity_types: vec![EntityTypeSpec::with_description(
                "Camera",
                "A video camera installed at a place.",
            )],
            relation_types: vec![],
        }
    }

    fn data() -> Value {
        json!({
            "cameras": [{
                "id": "cam-1",
                "video_analytics": {"modules": [
                    {"config": {"roi": [{"min_event_interval": 60}, {"min_event_interval": 1200}]}}
                ]}
            }]
        })
    }

    #[tokio::test]
    async fn fills_missing_descriptions_only() {
        let mut m = mapping();
        let llm = MockLlmClient::single("Minimum seconds between events for a region.");
        let n = describe_properties(&mut m, &ontology(), &data(), &llm, &DescribeOptions::default())
            .await
            .unwrap();
        assert_eq!(n, 1, "only the property without a description is filled");
        // Existing description preserved.
        assert_eq!(
            m.entities[0].properties[0].description.as_deref(),
            Some("Existing description.")
        );
        assert_eq!(
            m.entities[0].properties[1].description.as_deref(),
            Some("Minimum seconds between events for a region.")
        );
        // One request for the one missing property.
        assert_eq!(llm.call_count(), 1);
        // The prompt carried the ontology entity description and a sample.
        let (_, user) = &llm.calls()[0];
        assert!(user.contains("A video camera installed at a place."));
        assert!(user.contains("`60`"));
    }

    #[tokio::test]
    async fn overwrite_redescribes_everything() {
        let mut m = mapping();
        let llm = MockLlmClient::single("New text.");
        let opts = DescribeOptions {
            overwrite_existing: true,
            ..DescribeOptions::default()
        };
        let n = describe_properties(&mut m, &ontology(), &data(), &llm, &opts)
            .await
            .unwrap();
        assert_eq!(n, 2);
        assert_eq!(llm.call_count(), 2);
    }

    #[test]
    fn sampling_dedups_and_caps() {
        let s = sample_values(
            &data(),
            "$.cameras[*].video_analytics.modules[*].config.roi[*].min_event_interval",
            2,
        );
        assert_eq!(s, vec!["60".to_string(), "1200".to_string()]);
    }

    #[test]
    fn clean_strips_quotes_and_picks_first_line() {
        assert_eq!(clean_description("\"Hello there.\"\nextra"), "Hello there.");
        assert_eq!(clean_description("  `value`  "), "value");
    }
}
