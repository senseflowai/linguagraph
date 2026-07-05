//! Natural-language → DSL bridge.
//!
//! Turns a plain-language question into a validated [`DslQuery`] by
//! prompting an [`LlmClient`] with a schema-aware system prompt and
//! parsing (then, on failure, asking it to repair) the JSON it returns.
//! This is the one place the "constrain the model to emit our DSL" loop
//! lives, so every consumer — the CLI's `ask`, the
//! [`crate::service::GraphService`], the e2e harness — shares it.

use crate::dsl::{self, DslQuery};
use crate::error::{Error, Result};
use crate::llm::LlmClient;
use crate::prompt::{self, GraphSchema, PromptOptions};

/// Ask `llm` to translate `question` into a validated [`DslQuery`],
/// re-prompting up to `max_repairs` times when the output fails to parse
/// or validate.
///
/// The returned DSL has already passed [`dsl::parse_str`], so callers can
/// lower/compile it without re-validating. Errors:
/// * [`Error::Llm`] — the model call itself failed (transport, empty, …).
/// * [`Error::Nl`] — no valid DSL after `max_repairs` attempts (the
///   message carries the last validation error and raw output).
pub async fn generate_dsl(
    llm: &dyn LlmClient,
    question: &str,
    schema: &GraphSchema,
    opts: &PromptOptions,
    max_repairs: usize,
) -> Result<DslQuery> {
    let system = prompt::generate_query_prompt(question, schema, opts);
    let mut user = format!(
        "Question:\n{question}\n\nReturn only one JSON DSL object. Do not wrap it in Markdown."
    );
    let mut last_output = String::new();
    let mut last_error = String::new();

    for attempt in 0..=max_repairs {
        let raw = llm.complete(&system, &user).await?;
        last_output = raw.clone();
        match parse_and_validate(&raw) {
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

    Err(Error::Nl(format!(
        "could not produce valid DSL after {} attempt(s): {last_error}; last output: {last_output}",
        max_repairs + 1
    )))
}

fn parse_and_validate(raw: &str) -> Result<DslQuery> {
    let json = extract_json_object(raw)?;
    let dsl: DslQuery = serde_json::from_str(&json)?;
    let validated = dsl::parse_str(&serde_json::to_string(&dsl)?)?;
    Ok(validated)
}

/// Extract the first balanced JSON object from a model completion,
/// tolerating Markdown fences or prose around it. Returns the object
/// substring (still to be deserialized by the caller).
pub fn extract_json_object(raw: &str) -> Result<String> {
    if serde_json::from_str::<serde_json::Value>(raw).is_ok() {
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
                    // Validate the slice is real JSON before returning it.
                    serde_json::from_str::<serde_json::Value>(s)?;
                    return Ok(s.to_string());
                }
            }
            _ => {}
        }
    }
    Err(Error::Nl(
        "LLM output did not contain a JSON object".to_string(),
    ))
}
