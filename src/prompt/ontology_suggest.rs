//! Prompt renderer for the LLM **ontology schema suggester**.
//!
//! Given a text fragment and the current [`DomainOntology`], produces a
//! prompt instructing the LLM to propose **new properties** to add to
//! existing entity types. The LLM MUST NOT introduce new entity types,
//! remove or retype existing properties, or touch relation types.
//!
//! Pure function — no I/O.

use std::fmt::Write;

use crate::graph::{DomainOntology, EntityTypeSpec, OntologyPropertyType};

/// Human-readable label for an ontology property type, used in the prompt
/// vocabulary shown to the LLM.
fn property_type_label(t: OntologyPropertyType) -> &'static str {
    match t {
        OntologyPropertyType::Keyword => "keyword",
        OntologyPropertyType::Text => "text",
        OntologyPropertyType::Number => "number",
        OntologyPropertyType::Bool => "bool",
        OntologyPropertyType::Datetime => "datetime",
        OntologyPropertyType::List => "list",
    }
}

/// Render the schema-suggestion prompt for `fragment` against the given
/// `ontology`. `domain` is the human label of the ontology (used only
/// for framing).
pub fn render_schema_suggest_prompt(
    fragment: &str,
    domain: &str,
    ontology: &DomainOntology,
) -> String {
    let mut out = String::with_capacity(4096);

    let _ = writeln!(
        out,
        "## 🔹 Role\n\nYou are an ontology designer for the **{domain}** domain.\n\
         Your sole task is to inspect a text fragment and propose **additional \
         properties** to attach to entity types that already exist in the ontology.\n\n\
         You enrich the schema so that downstream knowledge extraction captures \
         the attributes actually present in the text, instead of producing bare entities."
    );
    out.push('\n');

    out.push_str("## 🔹 Existing Ontology (READ-ONLY)\n\n");
    out.push_str(
        "These entity types exist. Listed under each are properties already \
                  defined for it — you MUST NOT propose any property whose name matches \
                  an existing one.\n\n",
    );
    render_existing_entity_types(&mut out, &ontology.entity_types);
    out.push('\n');

    out.push_str("## 🔹 Allowed Property Types\n\n");
    out.push_str(
        "Use ONLY these `property_type` values (lowercase):\n\n\
                  * `keyword` — short single-line string / identifier / code / category\n\
                  * `text`    — long free-form text (will be embedded for semantic search)\n\
                  * `number`  — integer or decimal number\n\
                  * `bool`    — true/false\n\
                  * `datetime`— calendar date (YYYY-MM-DD) or ISO-8601 timestamp\n\
                  * `list`    — JSON array\n\n",
    );

    out.push_str("## 🔹 Hard Constraints\n\n\
                  * DO NOT introduce new entity types\n\
                  * DO NOT remove, rename, or retype existing properties\n\
                  * DO NOT propose relation types or modify relations in any way\n\
                  * DO NOT propose properties that just restate `name` (e.g. `full_name` for `Person`)\n\
                  * Only propose properties whose values are **explicitly present** in the text fragment for at least one instance of that entity type\n\
                  * If the text adds no useful new properties for an entity type, output an empty array for it (or omit it)\n\n");

    out.push_str("## 🔹 Output Format (STRICT)\n\n");
    out.push_str("Return ONLY a JSON object of this exact shape:\n\n");
    out.push_str("```json\n");
    out.push_str(
        "{\n  \"entity_types\": {\n    \"<EntityTypeName>\": [\n      \
                  {\"name\":\"<snake_case>\",\"property_type\":\"<one of the allowed types>\",\
                  \"required\":false,\"description\":\"short why\"}\n    ]\n  }\n}\n",
    );
    out.push_str("```\n\n");
    out.push_str(
        "Rules:\n\
                  * Top-level key MUST be exactly `entity_types`\n\
                  * Property `name` MUST be `snake_case`, unique within its entity type, \
                    and MUST NOT collide with an existing property of that entity type\n\
                  * `required` should be `false` unless the text shows this attribute \
                    is mandatory for every instance\n\
                  * `description` ≤ 80 chars, no newlines\n\
                  * No comments, no markdown, no text outside the JSON object\n\n",
    );

    out.push_str("## 🔹 Text Fragment\n\n```\n");
    out.push_str(fragment);
    if !fragment.ends_with('\n') {
        out.push('\n');
    }
    out.push_str("```\n");

    out
}

fn render_existing_entity_types(out: &mut String, types: &[EntityTypeSpec]) {
    if types.is_empty() {
        out.push_str("_(no entity types defined — nothing to enrich)_\n");
        return;
    }
    for t in types {
        match &t.description {
            Some(d) => {
                let _ = writeln!(out, "* `{}` — {}", t.name, d);
            }
            None => {
                let _ = writeln!(out, "* `{}`", t.name);
            }
        }
        if t.properties.is_empty() {
            let _ = writeln!(out, "  _(no properties yet)_");
        } else {
            let _ = writeln!(out, "  Existing properties (DO NOT redefine):");
            for p in &t.properties {
                let req = if p.required {
                    " (required)"
                } else {
                    " (optional)"
                };
                let _ = writeln!(
                    out,
                    "  * `{}` ({}){}",
                    p.name,
                    property_type_label(p.property_type),
                    req
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{OntologyPropertyType, PropertySpec, RelationTypeSpec};

    fn sample_ontology() -> DomainOntology {
        DomainOntology {
            name: None,
            description: None,
            entity_types: vec![
                EntityTypeSpec {
                    name: "Person".to_string(),
                    description: Some("A natural person.".to_string()),
                    properties: vec![PropertySpec {
                        name: "full_name".to_string(),
                        description: None,
                        property_type: OntologyPropertyType::Keyword,
                        required: true,
                        allowed_values: Vec::new(),
                    }],
                    embedding: None,
                },
                EntityTypeSpec::new("Organization"),
            ],
            relation_types: vec![RelationTypeSpec::new("WORKS_FOR")],
            embedding: None,
        }
    }

    #[test]
    fn prompt_lists_all_entity_types() {
        let p = render_schema_suggest_prompt("any text", "core_business", &sample_ontology());
        assert!(p.contains("* `Person`"));
        assert!(p.contains("* `Organization`"));
    }

    #[test]
    fn prompt_lists_existing_properties_to_avoid_dup() {
        let p = render_schema_suggest_prompt("any", "core_business", &sample_ontology());
        assert!(p.contains("Existing properties"));
        assert!(p.contains("* `full_name` (keyword) (required)"));
    }

    #[test]
    fn prompt_forbids_entity_type_and_relation_changes() {
        let p = render_schema_suggest_prompt("any", "core_business", &sample_ontology());
        assert!(p.contains("DO NOT introduce new entity types"));
        assert!(p.contains("DO NOT propose relation types"));
        assert!(p.contains("DO NOT remove, rename, or retype existing properties"));
    }

    #[test]
    fn prompt_uses_allowed_type_vocabulary_matching_knowledge_prompt() {
        // Vocabulary MUST exactly match what render_knowledge_extract_prompt uses,
        // otherwise the suggest output won't deserialize back into PropertySpec.
        let p = render_schema_suggest_prompt("any", "demo", &sample_ontology());
        for label in [
            "`keyword`",
            "`text`",
            "`number`",
            "`bool`",
            "`datetime`",
            "`list`",
        ] {
            assert!(p.contains(label), "missing type label {label}");
        }
    }

    #[test]
    fn prompt_includes_required_json_shape() {
        let p = render_schema_suggest_prompt("any", "demo", &sample_ontology());
        assert!(p.contains("\"entity_types\""));
        assert!(p.contains("\"property_type\""));
    }

    #[test]
    fn fragment_appears_at_the_end_in_a_code_block() {
        let frag = "the buyer signed the contract on 2024-01-15";
        let p = render_schema_suggest_prompt(frag, "legal", &sample_ontology());
        let frag_idx = p.find(frag).unwrap();
        let rules_idx = p.find("Hard Constraints").unwrap();
        assert!(frag_idx > rules_idx);
        assert!(p.ends_with("```\n"));
    }

    #[test]
    fn empty_ontology_does_not_panic_and_says_so() {
        let onto = DomainOntology::default();
        let p = render_schema_suggest_prompt("text", "demo", &onto);
        assert!(p.contains("no entity types defined"));
    }
}
