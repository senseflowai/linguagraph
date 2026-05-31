//! Deterministic canonical-text representation for whole-entity
//! soft-merge embeddings.
//!
//! The `_canonical` property is a fixed-format multi-line string that
//! serves as the universal soft-merge key for every LLM-extracted
//! entity (Source and Chunk builtins are excluded — they use strict
//! primary keys). Embedding this string with a `SemanticText` handler
//! gives the soft-merge resolver a stable cosine-similarity signal
//! independent of which surface form the LLM chose.
//!
//! Format:
//!
//! ```text
//! type: {entity_type}
//! {prop_a}: {value_a}
//! {prop_b}: {value_b}
//! ...
//! ```
//!
//! Properties are emitted in alphabetical key order so two runs over
//! identical inputs produce byte-identical strings (and therefore
//! identical embeddings). `name`, when the ontology declares it,
//! appears as just another property line — no special-casing.
//!
//! When `props` is empty the output is just `type: {entity_type}`. The
//! resulting embedding then merges every type-only mention into a
//! single node, which is the intended behaviour for patological
//! property-less extractions (better to collapse than to scatter).

use std::collections::HashMap;

use serde_json::Value;

/// Build the canonical text for an entity. See module docs for the
/// exact format. Pure function — deterministic for a given input.
pub fn build_canonical_text(entity_type: &str, props: &HashMap<String, Value>) -> String {
    let mut lines = Vec::with_capacity(props.len() + 1);
    lines.push(format!("type: {entity_type}"));
    let mut keys: Vec<&String> = props.keys().collect();
    keys.sort();
    for k in keys {
        let v = match &props[k] {
            Value::String(s) => s.clone(),
            other => other.to_string(),
        };
        lines.push(format!("{k}: {v}"));
    }
    lines.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn empty_props_yields_type_only() {
        let props = HashMap::new();
        assert_eq!(build_canonical_text("Person", &props), "type: Person");
    }

    #[test]
    fn single_string_property_appears_in_canonical() {
        let mut props = HashMap::new();
        props.insert("name".to_string(), json!("Alice"));
        assert_eq!(
            build_canonical_text("Person", &props),
            "type: Person\nname: Alice"
        );
    }

    #[test]
    fn properties_sorted_alphabetically_for_determinism() {
        let mut props = HashMap::new();
        props.insert("role".to_string(), json!("CEO"));
        props.insert("age".to_string(), json!(42));
        props.insert("name".to_string(), json!("Elon Musk"));
        let got = build_canonical_text("Person", &props);
        assert_eq!(got, "type: Person\nage: 42\nname: Elon Musk\nrole: CEO");
    }

    #[test]
    fn non_string_values_rendered_via_json_to_string() {
        let mut props = HashMap::new();
        props.insert("count".to_string(), json!(7));
        props.insert("active".to_string(), json!(true));
        let got = build_canonical_text("Thing", &props);
        assert_eq!(got, "type: Thing\nactive: true\ncount: 7");
    }

    #[test]
    fn identical_inputs_produce_identical_outputs() {
        let mut a = HashMap::new();
        a.insert("name".to_string(), json!("X"));
        a.insert("kind".to_string(), json!("Y"));
        let mut b = HashMap::new();
        b.insert("kind".to_string(), json!("Y"));
        b.insert("name".to_string(), json!("X"));
        assert_eq!(build_canonical_text("T", &a), build_canonical_text("T", &b));
    }
}
