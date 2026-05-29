//! Prompt renderer for the LLM **knowledge extractor**.
//!
//! Given a text fragment and a [`DomainOntology`], produces a complete
//! system+user prompt instructing the LLM to extract entities and relations
//! as JSON. The caller parses the JSON output, converts it to a
//! [`crate::graph::Graph`], and feeds it to [`crate::core::Pipeline::ingest`].
//!
//! Pure function — no I/O.

use std::fmt::Write;

use super::ontology::{DomainOntology, EntityTypeSpec, OntologyPropertyType, RelationTypeSpec};

/// Placeholder substituted with the active domain name when the prompt
/// is rendered. Present in the static section constants
/// ([`ROLE_SECTION`], [`INPUT_STRUCTURE_SECTION`], … ) so callers that
/// pull the sections directly can do the substitution themselves.
pub const DOMAIN_PLACEHOLDER: &str = "{domain}";

/// Render the knowledge-extraction prompt for `fragment` using the
/// entity/relation vocabulary defined in `ontology`.
///
/// `domain` is the human label substituted into the prompt's framing
/// (role, input structure, rules) — e.g. `"legal"`, `"medical"`. It is
/// only a label; the actual vocabulary comes from `ontology`.
///
/// The returned string is the *complete* prompt — system rules first,
/// then the text fragment under a clearly-labelled section. The LLM is
/// instructed to output **only** JSON. When entity types define `properties`,
/// the shape includes an optional `properties` dict:
///
/// ```json
/// {"entities":[{"id":"e1","type":"...","name":"...","properties":{"p1":"..."}}],
///  "relations":[{"from":"e1","to":"e2","type":"..."}]}
/// ```
pub fn render_knowledge_extract_prompt(
    fragment: &str,
    domain: &str,
    ontology: &DomainOntology,
) -> String {
    let mut out = String::with_capacity(4096);

    out.push_str(ROLE_SECTION);
    out.push_str("\n\n");
    out.push_str(INPUT_STRUCTURE_SECTION);
    out.push_str("\n\n");
    out.push_str(CORE_PRINCIPLES_SECTION);
    out.push_str("\n\n");

    // Ontology — dynamic.
    out.push_str("# 🔹 Ontology Specification (STRICT)\n\n");
    out.push_str("## Entity Types\n\n");
    out.push_str("You MUST use only the following types (no others):\n\n");
    render_entity_list(&mut out, &ontology.entity_types);
    out.push('\n');
    out.push_str("## Relation Types\n\n");
    out.push_str("You MUST use only the following relation types (no others):\n\n");
    render_relation_list(&mut out, &ontology.relation_types);
    out.push('\n');
    out.push_str(RELATION_NAME_RULES);
    out.push_str("\n\n");

    out.push_str(ENTITY_RULES_SECTION);
    out.push_str("\n\n");
    out.push_str(RELATION_RULES_SECTION);
    out.push_str("\n\n");
    out.push_str(COREFERENCE_SECTION);
    out.push_str("\n\n");
    out.push_str(ANTI_HALLUCINATION_SECTION);
    out.push_str("\n\n");
    render_output_section(&mut out, &ontology.entity_types);
    out.push_str("\n\n");
    out.push_str(SELF_VALIDATION_SECTION);
    out.push_str("\n\n");
    out.push_str(STRICT_JSON_RULES_SECTION);
    out.push_str("\n\n");
    out.push_str(FINAL_CHECKLIST_SECTION);
    out.push_str("\n\n");

    // Substitute the domain placeholder in the rules sections built so
    // far. The fragment is appended afterward so its content is never
    // touched by the substitution.
    let mut out = out.replace(DOMAIN_PLACEHOLDER, domain);

    // Text fragment — last so it's the freshest context window slice.
    out.push_str("# 🔹 Text Fragment\n\n");
    out.push_str("```\n");
    out.push_str(fragment);
    if !fragment.ends_with('\n') {
        out.push('\n');
    }
    out.push_str("```\n");

    out
}

fn render_entity_list(out: &mut String, types: &[EntityTypeSpec]) {
    for t in types {
        match &t.description {
            Some(d) => {
                let _ = writeln!(out, "* `{}` — {}", t.name, d);
            }
            None => {
                let _ = writeln!(out, "* `{}`", t.name);
            }
        }
        if !t.properties.is_empty() {
            let _ = writeln!(out, "  Properties:");
            for p in &t.properties {
                let type_str = match p.property_type {
                    OntologyPropertyType::String => "string",
                    OntologyPropertyType::Int => "int",
                    OntologyPropertyType::Float => "float",
                    OntologyPropertyType::Bool => "bool",
                };
                let req = if p.required { " (required)" } else { " (optional)" };
                match &p.description {
                    Some(d) => {
                        let _ = writeln!(out, "  * `{}` ({type_str}){req} — {d}", p.name);
                    }
                    None => {
                        let _ = writeln!(out, "  * `{}` ({type_str}){req}", p.name);
                    }
                }
            }
        }
    }
}

fn render_output_section(out: &mut String, entity_types: &[EntityTypeSpec]) {
    let has_props = entity_types.iter().any(|t| !t.properties.is_empty());
    if has_props {
        out.push_str(OUTPUT_SECTION_WITH_PROPERTIES);
    } else {
        out.push_str(OUTPUT_SECTION);
    }
}

fn render_relation_list(out: &mut String, types: &[RelationTypeSpec]) {
    for t in types {
        let upper = t.name.to_ascii_uppercase();
        match &t.description {
            Some(d) => {
                let _ = writeln!(out, "* `{upper}` — {d}");
            }
            None => {
                let _ = writeln!(out, "* `{upper}`");
            }
        }
    }
}

// ── Static prompt sections ──────────────────────────────────────────────
//
// Kept as separate constants so callers can also pull individual chunks
// (e.g. for testing or for a slimmer prompt variant) without parsing the
// final string.

pub const ROLE_SECTION: &str = "## 🔹 Role

You are an expert knowledge engineer specializing in **{domain} information extraction**, **ontology-driven semantic modeling**, and **embedding-optimized canonicalization**.

Your task is to extract structured knowledge from a {domain} text fragment and produce a **semantically consistent, embedding-ready representation**.

You must strictly distinguish between:

* **Normative content**
* **Structural elements**";

pub const INPUT_STRUCTURE_SECTION: &str = "## 🔹 Input Structure

The input contains:

* **Text Fragment** — authoritative {domain} content";

pub const CORE_PRINCIPLES_SECTION: &str = "## 🔹 Core Principles

* Extract **only explicitly stated information**
* Do **NOT infer**, interpret, or generalize
* Do **NOT add {domain} reasoning or doctrine**
* All outputs must be **deterministic and lexically consistent**
* The result must be optimized for **semantic embeddings**";

pub const RELATION_NAME_RULES: &str = "❗ Relation names MUST be:

* UPPERCASE
* EXACT (no variation)
* single token (no spaces)";

pub const ENTITY_RULES_SECTION: &str = "# 🔹 Entity Extraction Rules

## Canonicalization (CRITICAL)

Each entity MUST follow:

* Use **singular form**
* Use **nominative/base form** (important for inflected languages)
* Remove grammatical inflections (case, number)
* Remove leading articles (e.g., \"the\", \"a\")
* Remove unnecessary punctuation
* Preserve original wording, but normalized

## Name Consistency

* Each real-world/{domain} concept MUST appear **only once**
* Use original language!
* All references (including pronouns) MUST resolve to the same entity
* Do NOT create synonyms or alternate phrasings

❗ Example (forbidden):

* multiple surface forms referring to the same entity → MUST be ONE entity

## Granularity Rule

* Extract entities at the **smallest {domain}-meaningful unit**
* Do NOT split a concept unless explicitly separated in the text

## Deduplication Rule

Merge entities if:

* same meaning
* same referent
* same role in context";

pub const RELATION_RULES_SECTION: &str = "# 🔹 Relation Extraction Rules

* Extract ONLY **explicitly stated relations**
* Do NOT infer logical or domain-specific implications
* Each relation MUST connect two existing entities
* Relations MUST be **directional and consistent**

## Directionality

Use:

```
SOURCE → TARGET
```

Example:

```
SubjectType RELATION_NAME ObjectType
```

## Forbidden

* No implicit relations
* No bidirectional duplication
* No relation invention";

pub const COREFERENCE_SECTION: &str = "# 🔹 Coreference Resolution

* Resolve pronouns ONLY if unambiguous
* Merge all mentions of the same entity
* If ambiguity exists → DO NOT create entity";

pub const ANTI_HALLUCINATION_SECTION: &str = "# 🔹 Anti-Hallucination Rule

If something is:

* not clearly stated
* or ambiguous

👉 DO NOT extract it";

pub const OUTPUT_SECTION: &str = "# 🔹 OUTPUT MODE (CRITICAL)

You MUST output ONLY a valid JSON object.

- Output MUST start with `{` and end with `}`
- No text before or after JSON
- No markdown
- No code fences
- No explanations

Return ONLY valid JSON:

```json
{
  \"entities\": [
    {
      \"id\": \"e1\",
      \"type\": \"TypeName\",
      \"name\": \"string\"
    },
    {
      \"id\": \"e2\",
      \"type\": \"TypeName\",
      \"name\": \"string\"
    }
  ],
  \"relations\": [
    {
      \"from\": \"e1\",
      \"to\": \"e2\",
      \"type\": \"RELATION_TYPE\"
    }
  ]
}
```

## Constraints

* No extra fields
* No comments
* No markdown
* No explanations
* Keys MUST be exactly:

  * entities
  * relations";

pub const OUTPUT_SECTION_WITH_PROPERTIES: &str = "# 🔹 OUTPUT MODE (CRITICAL)

You MUST output ONLY a valid JSON object.

- Output MUST start with `{` and end with `}`
- No text before or after JSON
- No markdown
- No code fences
- No explanations

Return ONLY valid JSON:

```json
{
  \"entities\": [
    {
      \"id\": \"e1\",
      \"type\": \"TypeName\",
      \"name\": \"canonical entity name\",
      \"properties\": {
        \"string_prop\": \"value\",
        \"int_prop\": 42,
        \"float_prop\": 3.14,
        \"bool_prop\": true
      }
    }
  ],
  \"relations\": [
    {
      \"from\": \"e1\",
      \"to\": \"e2\",
      \"type\": \"RELATION_TYPE\"
    }
  ]
}
```

## Property Rules

* For entity types that define properties: populate a `\"properties\"` object with the declared keys
* Use `null` for optional properties you cannot find in the text
* Do NOT invent property keys that are not declared in the ontology
* Property value types MUST match the declared type (string → `\"...\"`, int → integer, float → number, bool → `true`/`false`)

## Constraints

* No extra fields outside declared properties
* No comments
* No markdown
* No explanations
* Top-level entity keys MUST be exactly: `id`, `type`, `name` (and `properties` when applicable)
* Top-level relation keys MUST be exactly: `from`, `to`, `type`";

pub const SELF_VALIDATION_SECTION: &str = "# 🔹 SELF-VALIDATION (MANDATORY)

Before output, you MUST:

1. Ensure JSON is syntactically valid
2. Ensure all brackets are closed
3. Ensure all commas are correct
4. Ensure all strings use double quotes
5. Ensure no trailing commas
6. Ensure all ids are unique
7. Ensure all referenced ids exist

If JSON is invalid → FIX it before output";

pub const STRICT_JSON_RULES_SECTION: &str = "# 🔹 STRICT JSON RULES

- Use ONLY double quotes \"
- Do NOT use trailing commas
- Do NOT use comments
- Do NOT use ellipsis (...)
- Do NOT omit required fields
- Do NOT output partial JSON";

pub const FINAL_CHECKLIST_SECTION: &str = "# 🔹 Final Checklist (MANDATORY)

Before producing output, ensure:

* No duplicate entities
* No synonym drift
* No invented relations
* All names canonicalized
* All types valid
* All relations valid
* JSON is valid and complete";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prompt_lists_supplied_types() {
        let onto = DomainOntology {
            entity_types: vec![EntityTypeSpec::new("Foo"), EntityTypeSpec::new("Bar")],
            relation_types: vec![RelationTypeSpec::new("KNOWS")],
        };
        let p = render_knowledge_extract_prompt("hello", "demo", &onto);
        assert!(p.contains("* `Foo`"));
        assert!(p.contains("* `Bar`"));
        assert!(p.contains("* `KNOWS`"));
        // Defaults must NOT leak in as list entries.
        assert!(!p.contains("* `LegalNorm`"));
        assert!(!p.contains("* `GRANTS`"));
    }

    #[test]
    fn relation_types_are_uppercased_in_output() {
        let onto = DomainOntology {
            entity_types: vec![EntityTypeSpec::new("X")],
            relation_types: vec![RelationTypeSpec::new("works-at")],
        };
        let p = render_knowledge_extract_prompt("t", "demo", &onto);
        assert!(p.contains("* `WORKS-AT`"));
    }

    #[test]
    fn fragment_appears_at_the_end() {
        let onto = DomainOntology::default();
        let p = render_knowledge_extract_prompt("the court grants the right", "legal", &onto);
        let idx_fragment = p.find("the court grants the right").unwrap();
        let idx_rules = p.find("# 🔹 Final Checklist").unwrap();
        assert!(idx_fragment > idx_rules);
    }

    #[test]
    fn descriptions_are_rendered_when_present() {
        let onto = DomainOntology {
            entity_types: vec![EntityTypeSpec::with_description(
                "Person",
                "A natural person.",
            )],
            relation_types: vec![RelationTypeSpec::with_description("KNOWS", "Acquaintance.")],
        };
        let p = render_knowledge_extract_prompt("x", "demo", &onto);
        assert!(p.contains("* `Person` — A natural person."));
        assert!(p.contains("* `KNOWS` — Acquaintance."));
    }

    #[test]
    fn output_section_specifies_exact_json_shape() {
        let p = render_knowledge_extract_prompt("x", "demo", &DomainOntology::default());
        assert!(p.contains("\"entities\""));
        assert!(p.contains("\"relations\""));
        assert!(p.contains("\"id\": \"e1\""));
        assert!(p.contains("\"from\": \"e1\""));
    }

    #[test]
    fn domain_placeholder_is_substituted_into_framing_sections() {
        let p = render_knowledge_extract_prompt("x", "medical", &DomainOntology::default());
        assert!(p.contains("**medical information extraction**"));
        assert!(p.contains("from a medical text fragment"));
        assert!(p.contains("authoritative medical content"));
        assert!(p.contains("real-world/medical concept"));
        assert!(p.contains("smallest medical-meaningful unit"));
        // No leftover placeholders.
        assert!(!p.contains(DOMAIN_PLACEHOLDER));
        // No "legal" framing left over either.
        assert!(!p.contains("legal information extraction"));
        assert!(!p.contains("legal text fragment"));
    }

    #[test]
    fn domain_substitution_does_not_touch_fragment() {
        // The fragment text MUST NOT be mutated even if it happens to
        // contain the placeholder token verbatim.
        let frag = "raw text including {domain} literally";
        let p = render_knowledge_extract_prompt(frag, "legal", &DomainOntology::default());
        assert!(p.contains("raw text including {domain} literally"));
    }

    #[test]
    fn property_specs_appear_in_entity_listing() {
        use super::super::ontology::{OntologyPropertyType, PropertySpec};
        let onto = DomainOntology {
            entity_types: vec![EntityTypeSpec {
                name: "Person".to_string(),
                description: None,
                properties: vec![
                    PropertySpec {
                        name: "first_name".to_string(),
                        description: None,
                        property_type: OntologyPropertyType::String,
                        required: true,
                    },
                    PropertySpec {
                        name: "age".to_string(),
                        description: Some("Age in years.".to_string()),
                        property_type: OntologyPropertyType::Int,
                        required: false,
                    },
                ],
            }],
            relation_types: vec![],
        };
        let p = render_knowledge_extract_prompt("text", "demo", &onto);
        assert!(p.contains("* `Person`"));
        assert!(p.contains("* `first_name` (string) (required)"));
        assert!(p.contains("* `age` (int) (optional) — Age in years."));
    }

    #[test]
    fn output_section_with_properties_used_when_specs_present() {
        use super::super::ontology::{OntologyPropertyType, PropertySpec};
        let onto = DomainOntology {
            entity_types: vec![EntityTypeSpec {
                name: "Invoice".to_string(),
                description: None,
                properties: vec![PropertySpec {
                    name: "amount".to_string(),
                    description: None,
                    property_type: OntologyPropertyType::Float,
                    required: true,
                }],
            }],
            relation_types: vec![],
        };
        let p = render_knowledge_extract_prompt("text", "demo", &onto);
        assert!(p.contains("\"properties\""));
        assert!(p.contains("Property Rules"));
    }

    #[test]
    fn output_section_without_properties_when_no_specs() {
        let onto = DomainOntology {
            entity_types: vec![EntityTypeSpec::new("Foo")],
            relation_types: vec![],
        };
        let p = render_knowledge_extract_prompt("text", "demo", &onto);
        assert!(!p.contains("Property Rules"));
        assert!(p.contains("\"name\": \"string\""));
    }
}
