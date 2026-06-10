//! End-to-end tests for the prompt-generation module.
//!
//! These tests live outside the crate so they exercise the public
//! API only. They cover:
//!
//! * Analyser-driven examples — the prompt that comes out of the
//!   bundled `examples/promptgen_input.json` is stable and contains
//!   the expected high-signal fragments.
//! * Behaviour around opt-in sections (`include_examples`,
//!   `include_inferred_summary`, `domain_hints`).
//! * Resilience to oddly-shaped or empty input.

use serde_json::json;

use linguagraph::promptgen::{analyze, generate_prompt, InferredType, PromptGenOptions};

fn opts() -> PromptGenOptions {
    PromptGenOptions::default()
}

#[test]
fn bundled_companies_input_yields_well_formed_prompt() {
    let raw = include_str!("../examples/promptgen_input.json");
    let value: serde_json::Value = serde_json::from_str(raw).unwrap();

    // Sanity-check the analyser before we look at the prompt: the
    // CLI's behaviour follows from these primitives.
    let summary = analyze(&value);
    assert_eq!(summary.entities.len(), 1);
    let company = &summary.entities[0];
    assert_eq!(company.name, "Company");
    assert_eq!(company.source_path, "$.companies[*]");
    assert_eq!(company.primary_key.as_deref(), Some("$.companies[*].id"));
    assert_eq!(company.samples, 3);

    // The four "interesting" fields each end up with a sensible type.
    let by_name: std::collections::BTreeMap<&str, InferredType> = company
        .fields
        .iter()
        .map(|f| (f.name.as_str(), f.inferred_type))
        .collect();
    assert_eq!(by_name["description"], InferredType::Text);
    assert_eq!(by_name["industry"], InferredType::Keyword);
    assert_eq!(by_name["founded_at"], InferredType::DateTime);
    assert_eq!(by_name["is_public"], InferredType::Boolean);
    assert_eq!(by_name["employee_count"], InferredType::Number);

    // The rendered prompt contains every section in order.
    let prompt = generate_prompt(&value, &opts());
    assert!(prompt.starts_with("You are a senior data engineer."));
    let sections = [
        "# Mapping schema",
        "# Available field types",
        "# Inferred structure",
        "# Rules",
        "# Example",
    ];
    let mut last = 0usize;
    for s in sections {
        let pos = prompt
            .find(s)
            .unwrap_or_else(|| panic!("missing section {s}"));
        assert!(pos >= last, "section {s} appears out of order");
        last = pos;
    }
    // Header for the inferred entity, with the path and sample count.
    assert!(prompt.contains("## Company (`$.companies[*]`, 3 samples)"));
    // Type recommendations bubble up correctly.
    assert!(prompt.contains("`description` → **Text**"));
    assert!(prompt.contains("`industry` → **Keyword**"));
    assert!(prompt.contains("`founded_at` → **DateTime**"));
    // Rules are present; few-shot example is on by default.
    assert!(prompt.contains("Output JSON only."));
    assert!(prompt.contains("Expected output:"));
}

#[test]
fn no_examples_and_no_summary_strip_those_sections() {
    let raw = include_str!("../examples/promptgen_input.json");
    let value: serde_json::Value = serde_json::from_str(raw).unwrap();

    let opts = PromptGenOptions {
        include_examples: false,
        include_inferred_summary: false,
        ..opts()
    };
    let prompt = generate_prompt(&value, &opts);
    assert!(!prompt.contains("# Example"));
    assert!(!prompt.contains("# Inferred structure"));
    assert!(prompt.contains("# Mapping schema"));
    assert!(prompt.contains("# Rules"));
}

#[test]
fn domain_and_preference_hints_render_when_supplied() {
    let opts = PromptGenOptions {
        domain_hints: vec![
            "this is an HR / people-management dataset".into(),
            "all timestamps are UTC".into(),
        ],
        preferred_types: vec!["SemanticText".into(), "Keyword".into()],
        constraints: vec!["entity names must be in English".into()],
        ..opts()
    };
    let prompt = generate_prompt(&json!({"x": 1}), &opts);
    assert!(prompt.contains("# Domain hints"));
    assert!(prompt.contains("HR / people-management"));
    assert!(prompt.contains("# Preferred types"));
    assert!(prompt.contains("SemanticText, Keyword"));
    assert!(prompt.contains("# Constraints"));
    assert!(prompt.contains("entity names must be in English"));
}

#[test]
fn nested_arrays_surface_as_separate_entities_in_the_prompt() {
    let v = json!({
        "users": [
            {"id": 1, "name": "alice",
             "posts": [{"id": 10, "title": "hello", "body": "yet another long enough body to satisfy the heuristic threshold." }]}
        ]
    });
    let prompt = generate_prompt(&v, &opts());
    // Both entities head sections.
    assert!(prompt.contains("## User (`$.users[*]`"));
    assert!(prompt.contains("## Post (`$.users[*].posts[*]`"));
    // And the relationship hint section calls out the parent/child link.
    assert!(prompt.contains("Relationship hints:"));
    assert!(prompt.contains("`User` HAS_MANY `Post`"));
}

#[test]
fn missing_primary_key_is_called_out() {
    let v = json!({"items": [{"name": "x"}, {"name": "y"}]});
    let prompt = generate_prompt(&v, &opts());
    assert!(prompt.contains("primary_key: **(not detected)**"));
}

#[test]
fn deeply_nested_input_does_not_panic_or_blow_up() {
    // A pathological shape: arrays inside objects inside arrays.
    let mut v = json!({"id": 1, "value": "a"});
    for _ in 0..10 {
        v = json!({"children": [v]});
    }
    let prompt = generate_prompt(&v, &opts());
    assert!(prompt.starts_with("You are a senior data engineer."));
    // Should produce *some* entity even if the analyser collapses
    // most of the depth into nested-children.
    assert!(prompt.contains("## "));
}

/// Golden-style test: when the bundled input feeds the prompt, the
/// generator's output is byte-for-byte stable across runs. Rather
/// than freezing the entire string (brittle to template edits), we
/// snapshot a list of *invariants* — anything tooling downstream
/// would rely on.
#[test]
fn golden_invariants_for_companies_input() {
    let raw = include_str!("../examples/promptgen_input.json");
    let value: serde_json::Value = serde_json::from_str(raw).unwrap();
    let prompt = generate_prompt(&value, &opts());

    let invariants = [
        "Output **only** the JSON mapping",
        "PascalCase, singular",
        "SCREAMING_SNAKE_CASE",
        "$.companies[*]",
        "$.companies[*].id",
        "$.companies[*].description",
        "**Text**",
        "**Keyword**",
        "**DateTime**",
        "Now produce the mapping JSON for the input below.",
    ];
    for needle in invariants {
        assert!(
            prompt.contains(needle),
            "expected invariant {needle:?} in prompt; got:\n{prompt}"
        );
    }
}
