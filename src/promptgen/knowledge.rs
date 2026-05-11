//! Prompt generator for the LLM **knowledge extractor**.
//!
//! Given a text fragment (e.g. one chunk of a document), this module
//! produces a system+user prompt instructing the LLM to extract a list
//! of entities and relations in the *exact* JSON shape consumed by
//! [`crate::ingest::ChunkInput::entities`] /
//! [`crate::ingest::ChunkInput::relations`]. The caller can then drop
//! the LLM output straight into a [`crate::ingest::DocumentInput`] and
//! feed it to [`crate::core::Pipeline::ingest_document`].
//!
//! ## Design
//!
//! * The prompt template is the legal-domain extraction prompt from the
//!   spec, with two placeholders for the **allowed entity types** and
//!   **allowed relation types**.
//! * Defaults cover the common legal vocabulary (`LegalNorm`,
//!   `StateBody`, `Person`, …; `GRANTS`, `REQUIRES`, …). Callers
//!   override them via [`KnowledgeExtractOptions`].
//! * The text fragment is appended at the end so the LLM sees the rules
//!   first.
//!
//! The function is pure — no I/O — so it's trivial to unit-test.

use std::fmt::Write;

/// Type spec for one entity class the LLM may emit.
#[derive(Debug, Clone)]
pub struct EntityTypeSpec {
    /// Canonical PascalCase name (e.g. `LegalNorm`).
    pub name: String,
    /// Optional one-line description shown alongside the name.
    pub description: Option<String>,
}

impl EntityTypeSpec {
    pub fn new(name: impl Into<String>) -> Self {
        Self { name: name.into(), description: None }
    }

    pub fn with_description(name: impl Into<String>, desc: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: Some(desc.into()),
        }
    }
}

/// Type spec for one relation class the LLM may emit.
#[derive(Debug, Clone)]
pub struct RelationTypeSpec {
    /// Canonical UPPER_SNAKE name (e.g. `GRANTS`).
    pub name: String,
    /// Optional one-line description.
    pub description: Option<String>,
}

impl RelationTypeSpec {
    pub fn new(name: impl Into<String>) -> Self {
        Self { name: name.into(), description: None }
    }

    pub fn with_description(name: impl Into<String>, desc: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: Some(desc.into()),
        }
    }
}

/// Knobs for [`generate_knowledge_extract_prompt`].
#[derive(Debug, Clone)]
pub struct KnowledgeExtractOptions {
    /// Allowed entity types. The LLM is instructed to use **only**
    /// types from this list. Defaults to a legal-domain vocabulary
    /// (see [`Self::default`]).
    pub entity_types: Vec<EntityTypeSpec>,
    /// Allowed relation types. Defaults to a legal-domain vocabulary.
    /// All names are uppercased before rendering to honour the
    /// "Relation names MUST be UPPERCASE" rule.
    pub relation_types: Vec<RelationTypeSpec>,
}

impl Default for KnowledgeExtractOptions {
    fn default() -> Self {
        Self {
            entity_types: default_entity_types(),
            relation_types: default_relation_types(),
        }
    }
}

/// Default legal-domain entity vocabulary. Covers the typical actors,
/// instruments, and objects of legal norms.
pub fn default_entity_types() -> Vec<EntityTypeSpec> {
    vec![
        EntityTypeSpec::with_description(
            "LegalNorm",
            "A rule, provision, article, or paragraph stated in the text.",
        ),
        EntityTypeSpec::with_description(
            "LegalAct",
            "A named statute, code, law, decree, or treaty.",
        ),
        EntityTypeSpec::with_description(
            "StateBody",
            "Any organ of public authority: court, ministry, agency, parliament.",
        ),
        EntityTypeSpec::with_description(
            "Person",
            "A natural person identified by role or name (judge, plaintiff, citizen).",
        ),
        EntityTypeSpec::with_description(
            "Organization",
            "A legal entity, association, or corporation other than a state body.",
        ),
        EntityTypeSpec::with_description(
            "LegalRight",
            "An entitlement or freedom granted to a subject.",
        ),
        EntityTypeSpec::with_description(
            "LegalObligation",
            "A duty, requirement, or prohibition imposed on a subject.",
        ),
        EntityTypeSpec::with_description(
            "Sanction",
            "A penalty, fine, or other consequence for a violation.",
        ),
        EntityTypeSpec::with_description(
            "LegalProcedure",
            "A formally described process (filing, appeal, hearing, registration).",
        ),
        EntityTypeSpec::with_description(
            "LegalConcept",
            "A defined legal term or category (property, citizenship, contract).",
        ),
        EntityTypeSpec::with_description(
            "Date",
            "A specific date or time period explicitly mentioned.",
        ),
        EntityTypeSpec::with_description(
            "Location",
            "A jurisdiction, territory, or named place.",
        ),
        EntityTypeSpec::with_description(
            "MonetaryAmount",
            "A specific sum of money referenced in the text.",
        ),
    ]
}

/// Default legal-domain relation vocabulary.
pub fn default_relation_types() -> Vec<RelationTypeSpec> {
    vec![
        RelationTypeSpec::with_description("GRANTS", "Subject confers a right or power on another."),
        RelationTypeSpec::with_description("REQUIRES", "Subject imposes a requirement on another."),
        RelationTypeSpec::with_description("PROHIBITS", "Subject forbids the target action or state."),
        RelationTypeSpec::with_description("REGULATES", "Subject sets rules over the target."),
        RelationTypeSpec::with_description("ESTABLISHES", "Subject creates, founds, or institutes the target."),
        RelationTypeSpec::with_description("ENFORCES", "Subject implements or compels compliance with the target."),
        RelationTypeSpec::with_description("REFERENCES", "Subject cites or invokes the target."),
        RelationTypeSpec::with_description("AMENDS", "Subject modifies the target legal act/norm."),
        RelationTypeSpec::with_description("REPEALS", "Subject revokes the target legal act/norm."),
        RelationTypeSpec::with_description("APPLIES_TO", "Subject's scope covers the target."),
        RelationTypeSpec::with_description("PART_OF", "Subject is structurally contained in the target."),
        RelationTypeSpec::with_description("HAS_SANCTION", "Subject (norm) attaches the target sanction."),
        RelationTypeSpec::with_description("ISSUED_BY", "Subject (act/decision) was issued by the target body."),
        RelationTypeSpec::with_description("DEFINED_AS", "Subject term is defined as the target concept."),
    ]
}

/// Render the knowledge-extraction prompt for `fragment`.
///
/// The returned string is the *complete* prompt — system rules first,
/// then the text fragment under a clearly-labelled section. The LLM is
/// instructed to output **only** JSON of the shape:
///
/// ```json
/// {"entities":[{"id":"e1","type":"...","name":"..."}],
///  "relations":[{"from":"e1","to":"e2","type":"..."}]}
/// ```
///
/// which parses directly as `(Vec<EntityInput>, Vec<RelationInput>)`
/// for a [`crate::ingest::ChunkInput`].
pub fn generate_knowledge_extract_prompt(
    fragment: &str,
    opts: &KnowledgeExtractOptions,
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
    render_entity_list(&mut out, &opts.entity_types);
    out.push('\n');
    out.push_str("## Relation Types\n\n");
    out.push_str("You MUST use only the following relation types (no others):\n\n");
    render_relation_list(&mut out, &opts.relation_types);
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

You are an expert knowledge engineer specializing in **legal information extraction**, **ontology-driven semantic modeling**, and **embedding-optimized canonicalization**.

Your task is to extract structured knowledge from a legal text fragment and produce a **semantically consistent, embedding-ready representation**.

You must strictly distinguish between:

* **Normative content**
* **Structural elements**";

pub const INPUT_STRUCTURE_SECTION: &str = "## 🔹 Input Structure

The input contains:

* **Text Fragment** — authoritative legal content";

pub const CORE_PRINCIPLES_SECTION: &str = "## 🔹 Core Principles

* Extract **only explicitly stated information**
* Do **NOT infer**, interpret, or generalize
* Do **NOT add legal reasoning or doctrine**
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

* Each real-world/legal concept MUST appear **only once**
* Use original language!
* All references (including pronouns) MUST resolve to the same entity
* Do NOT create synonyms or alternate phrasings

❗ Example (forbidden):

* \"court\", \"the court\", \"judicial authority\" → MUST be ONE entity

## Granularity Rule

* Extract entities at the **smallest legally meaningful unit**
* Do NOT split a concept unless explicitly separated in the text

## Deduplication Rule

Merge entities if:

* same meaning
* same referent
* same legal role in context";

pub const RELATION_RULES_SECTION: &str = "# 🔹 Relation Extraction Rules

* Extract ONLY **explicitly stated relations**
* Do NOT infer logical or legal implications
* Each relation MUST connect two existing entities
* Relations MUST be **directional and consistent**

## Directionality

Use:

```
SOURCE → TARGET
```

Example:

```
StateBody GRANTS LegalRight
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
    fn defaults_contain_legal_vocabulary() {
        let opts = KnowledgeExtractOptions::default();
        let names: Vec<&str> = opts.entity_types.iter().map(|e| e.name.as_str()).collect();
        assert!(names.contains(&"LegalNorm"));
        assert!(names.contains(&"StateBody"));
        let rels: Vec<&str> = opts.relation_types.iter().map(|r| r.name.as_str()).collect();
        assert!(rels.contains(&"GRANTS"));
        assert!(rels.contains(&"REGULATES"));
    }

    #[test]
    fn prompt_lists_supplied_types() {
        let opts = KnowledgeExtractOptions {
            entity_types: vec![EntityTypeSpec::new("Foo"), EntityTypeSpec::new("Bar")],
            relation_types: vec![RelationTypeSpec::new("KNOWS")],
        };
        let p = generate_knowledge_extract_prompt("hello", &opts);
        // Each user-supplied type appears as a bulleted list entry.
        assert!(p.contains("* `Foo`"));
        assert!(p.contains("* `Bar`"));
        assert!(p.contains("* `KNOWS`"));
        // The default vocabulary must NOT appear as list entries.
        // (We use the exact bullet prefix so we don't false-positive on
        // illustrative examples in the static rules sections, which
        // legitimately reference `StateBody`/`GRANTS`.)
        assert!(!p.contains("* `LegalNorm`"));
        assert!(!p.contains("* `GRANTS`"));
    }

    #[test]
    fn relation_types_are_uppercased_in_output() {
        let opts = KnowledgeExtractOptions {
            entity_types: vec![EntityTypeSpec::new("X")],
            relation_types: vec![RelationTypeSpec::new("works-at")],
        };
        let p = generate_knowledge_extract_prompt("t", &opts);
        // Sanitized at the *rendering* step: relation names are rendered uppercase.
        assert!(p.contains("* `WORKS-AT`"));
    }

    #[test]
    fn fragment_appears_at_the_end() {
        let opts = KnowledgeExtractOptions::default();
        let p = generate_knowledge_extract_prompt("the court grants the right", &opts);
        let idx_fragment = p.find("the court grants the right").unwrap();
        let idx_rules = p.find("# 🔹 Final Checklist").unwrap();
        assert!(idx_fragment > idx_rules);
    }

    #[test]
    fn descriptions_are_rendered_when_present() {
        let opts = KnowledgeExtractOptions {
            entity_types: vec![EntityTypeSpec::with_description(
                "Person",
                "A natural person.",
            )],
            relation_types: vec![RelationTypeSpec::with_description(
                "KNOWS",
                "Acquaintance.",
            )],
        };
        let p = generate_knowledge_extract_prompt("x", &opts);
        assert!(p.contains("* `Person` — A natural person."));
        assert!(p.contains("* `KNOWS` — Acquaintance."));
    }

    #[test]
    fn output_section_specifies_exact_json_shape() {
        let p = generate_knowledge_extract_prompt("x", &KnowledgeExtractOptions::default());
        assert!(p.contains("\"entities\""));
        assert!(p.contains("\"relations\""));
        assert!(p.contains("\"id\": \"e1\""));
        assert!(p.contains("\"from\": \"e1\""));
    }
}
