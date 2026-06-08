//! Prompt renderer for the LLM **knowledge extractor**.
//!
//! Given a [`DomainOntology`], produces a complete system prompt instructing
//! the LLM to extract entities and relations as JSON. The caller parses the
//! JSON output, converts it to a
//! [`crate::graph::Graph`], and feeds it to [`crate::core::Pipeline::ingest`].
//!
//! Pure function — no I/O.

use std::fmt::Write;

use crate::graph::{DomainOntology, EntityTypeSpec, OntologyPropertyType, RelationTypeSpec};

/// Placeholder substituted with the active domain name when the prompt
/// is rendered. Present in the static section constants
/// ([`ROLE_SECTION`], [`INPUT_STRUCTURE_SECTION`], … ) so callers that
/// pull the sections directly can do the substitution themselves.
pub const DOMAIN_PLACEHOLDER: &str = "{domain}";

/// Render the knowledge-extraction system prompt using the entity/relation
/// vocabulary defined in `ontology`.
///
/// `domain` is the human label substituted into the prompt's framing
/// (role, input structure, rules) — e.g. `"legal"`, `"medical"`. It is
/// only a label; the actual vocabulary comes from `ontology`.
///
/// The returned string is the complete system prompt. The LLM is instructed to
/// output **only** JSON. When entity types define `properties`, the shape
/// includes inline property keys:
///
/// ```json
/// {"entities":[{"id":"e1","type":"...","name":"...","p1":"..."}],
///  "relations":[{"s":"e1","o":"e2","t":"..."}]}
/// ```
pub fn render_knowledge_extract_prompt(domain: &str, ontology: &DomainOntology) -> String {
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
    out.push_str(OUTPUT_SECTION);
    out.push_str("\n\n");
    out.push_str(SELF_VALIDATION_SECTION);
    out.push_str("\n\n");
    out.push_str(STRICT_JSON_RULES_SECTION);
    out.push_str("\n\n");
    out.push_str(FINAL_CHECKLIST_SECTION);
    out.push_str("\n\n");

    out.replace(DOMAIN_PLACEHOLDER, domain)
}

/// Lexical label for an ontology property type. Must match exactly the
/// vocabulary the LLM is asked to use in the ontology-suggest prompt.
pub fn property_type_label(t: OntologyPropertyType) -> &'static str {
    match t {
        OntologyPropertyType::String => "string",
        OntologyPropertyType::Text => "text",
        OntologyPropertyType::Int => "int",
        OntologyPropertyType::Float => "float",
        OntologyPropertyType::Bool => "bool",
        OntologyPropertyType::Date => "date",
        OntologyPropertyType::Datetime => "datetime",
        OntologyPropertyType::List => "list",
    }
}

pub(crate) fn render_entity_list(out: &mut String, types: &[EntityTypeSpec]) {
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
                let type_str = property_type_label(p.property_type);
                let req = if p.required {
                    " (required)"
                } else {
                    " (optional)"
                };
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

pub(crate) fn render_relation_list(out: &mut String, types: &[RelationTypeSpec]) {
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

Your task is to extract structured knowledge from {domain} text supplied by the user and produce a **semantically consistent, embedding-ready representation**.

You must strictly distinguish between:

* **Normative content**
* **Structural elements**";

pub const INPUT_STRUCTURE_SECTION: &str = "## 🔹 Input Structure

The input contains:

* authoritative {domain} content supplied by the user";

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

## Compact Shape

To minimize tokens, properties are placed **inline** in the entity object
(no wrapping `properties` key), and relations use short keys `s`/`o`/`t`:

```json
{
  \"entities\": [
    {\"id\":\"e1\",\"type\":\"TypeA\",\"name\":\"canonical name\",\"prop_a\":\"value\",\"prop_b\":42},
    {\"id\":\"e2\",\"type\":\"TypeB\",\"name\":\"another\",\"date_prop\":\"2024-01-15\"}
  ],
  \"relations\": [
    {\"s\":\"e1\",\"o\":\"e2\",\"t\":\"RELATION_TYPE\"}
  ]
}
```

In relations: `s` = subject (source entity id), `o` = object (target id),
`t` = relation type (UPPER_SNAKE).

## Reserved Entity Keys

These keys have fixed meaning and are NOT properties:

* `id` — local string id (referenced by relations)
* `type` — entity type from the ontology

**Every other key in the entity object is treated as an ontology property
for that entity type — including `name` if the ontology declares it.**
Use ONLY property names declared for the entity's type in the ontology
— never invent new keys.

## Property Value Types

Match the declared type exactly:

* `string` / `text` → `\"...\"`
* `int` → integer (no quotes), e.g. `42`
* `float` → number (no quotes), e.g. `3.14`
* `bool` → `true` or `false` (no quotes)
* `date` → `\"YYYY-MM-DD\"`
* `datetime` → ISO-8601 string, e.g. `\"2024-01-15T09:30:00Z\"`
* `list` → JSON array, e.g. `[\"a\",\"b\"]`

Omit the key entirely for optional properties you cannot find in the text.
Do NOT emit `null`.

## Constraints

* No comments, markdown, explanations
* No keys other than reserved + declared ontology properties
* Relations use ONLY keys `s`, `o`, `t`";

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
        let p = render_knowledge_extract_prompt("demo", &onto);
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
        let p = render_knowledge_extract_prompt("demo", &onto);
        assert!(p.contains("* `WORKS-AT`"));
    }

    #[test]
    fn prompt_does_not_include_a_fragment_section() {
        let onto = DomainOntology::default();
        let p = render_knowledge_extract_prompt("legal", &onto);
        assert!(!p.contains("# 🔹 Text Fragment"));
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
        let p = render_knowledge_extract_prompt("demo", &onto);
        assert!(p.contains("* `Person` — A natural person."));
        assert!(p.contains("* `KNOWS` — Acquaintance."));
    }

    #[test]
    fn output_section_specifies_exact_json_shape() {
        let p = render_knowledge_extract_prompt("demo", &DomainOntology::default());
        assert!(p.contains("\"entities\""));
        assert!(p.contains("\"relations\""));
        assert!(p.contains("\"id\":\"e1\""));
        // Relations use compact short keys.
        assert!(p.contains("\"s\":\"e1\""));
        assert!(p.contains("\"o\":\"e2\""));
        assert!(p.contains("\"t\":\"RELATION_TYPE\""));
        // Old verbose keys MUST NOT appear in the example.
        assert!(!p.contains("\"from\": \"e1\""));
        // Reserved keys list MUST NOT include `name` — it's an ordinary
        // property now.
        assert!(p.contains("`id` — local string id"));
        assert!(p.contains("`type` — entity type from the ontology"));
        assert!(!p.contains("`name` — canonical name"));
    }

    #[test]
    fn domain_placeholder_is_substituted_into_framing_sections() {
        let p = render_knowledge_extract_prompt("medical", &DomainOntology::default());
        assert!(p.contains("**medical information extraction**"));
        assert!(p.contains("from medical text supplied by the user"));
        assert!(p.contains("authoritative medical content"));
        assert!(p.contains("real-world/medical concept"));
        assert!(p.contains("smallest medical-meaningful unit"));
        // No leftover placeholders.
        assert!(!p.contains(DOMAIN_PLACEHOLDER));
        // No "legal" framing left over either.
        assert!(!p.contains("legal information extraction"));
        assert!(!p.contains("legal text supplied by the user"));
    }

    #[test]
    fn no_domain_placeholder_remains() {
        let p = render_knowledge_extract_prompt("legal", &DomainOntology::default());
        assert!(!p.contains(DOMAIN_PLACEHOLDER));
    }

    #[test]
    fn property_specs_appear_in_entity_listing() {
        use crate::graph::{OntologyPropertyType, PropertySpec};
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
                embedding: None,
            }],
            relation_types: vec![],
        };
        let p = render_knowledge_extract_prompt("demo", &onto);
        assert!(p.contains("* `Person`"));
        assert!(p.contains("* `first_name` (string) (required)"));
        assert!(p.contains("* `age` (int) (optional) — Age in years."));
    }

    #[test]
    fn output_section_is_unified_regardless_of_properties() {
        use crate::graph::{OntologyPropertyType, PropertySpec};
        // With properties.
        let with_props = DomainOntology {
            entity_types: vec![EntityTypeSpec {
                name: "Invoice".to_string(),
                description: None,
                properties: vec![PropertySpec {
                    name: "amount".to_string(),
                    description: None,
                    property_type: OntologyPropertyType::Float,
                    required: true,
                }],
                embedding: None,
            }],
            relation_types: vec![],
        };
        // Without properties.
        let no_props = DomainOntology {
            entity_types: vec![EntityTypeSpec::new("Foo")],
            relation_types: vec![],
        };
        let p_with = render_knowledge_extract_prompt("demo", &with_props);
        let p_no = render_knowledge_extract_prompt("demo", &no_props);
        // Both prompts contain the same OUTPUT sections.
        assert!(p_with.contains("Compact Shape"));
        assert!(p_no.contains("Compact Shape"));
        assert!(p_with.contains("Reserved Entity Keys"));
        assert!(p_no.contains("Reserved Entity Keys"));
        assert!(p_with.contains("Property Value Types"));
        assert!(p_no.contains("Property Value Types"));
        // The unified inline example MUST appear in both.
        assert!(p_with.contains("\"prop_a\":\"value\""));
        assert!(p_no.contains("\"prop_a\":\"value\""));
        // No wrapping `"properties": {` block in either.
        assert!(!p_with.contains("\"properties\": {"));
        assert!(!p_no.contains("\"properties\": {"));
    }

    #[test]
    fn text_property_type_uses_text_label_not_string() {
        use crate::graph::{OntologyPropertyType, PropertySpec};
        let onto = DomainOntology {
            entity_types: vec![EntityTypeSpec {
                name: "Doc".to_string(),
                description: None,
                properties: vec![PropertySpec {
                    name: "body".to_string(),
                    description: None,
                    property_type: OntologyPropertyType::Text,
                    required: false,
                }],
                embedding: None,
            }],
            relation_types: vec![],
        };
        let p = render_knowledge_extract_prompt("demo", &onto);
        // The label must be `text`, matching the suggest-prompt vocabulary.
        assert!(p.contains("* `body` (text) (optional)"));
        // And the old mis-label `(string)` for a Text property must be gone.
        assert!(!p.contains("* `body` (string)"));
    }
}
