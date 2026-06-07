//! Assemble the system/user prompt pair that drives mapping generation.
//!
//! The system prompt reuses the [`crate::promptgen`] building blocks
//! (mapping spec, field-type catalogue, heuristic inferred-structure,
//! rules, few-shot example) and prepends two authoritative sections:
//!
//! 1. **Allowed entity types** — a STRICT whitelist taken from the
//!    ontology. The LLM may invent relationships and extra properties,
//!    but never entity types outside this list.
//! 2. **Existing graph schema** — the live `GraphSchema`, so the model
//!    reuses labels / properties / relationship types already present in
//!    the graph instead of coining new ones.
//!
//! The user prompt carries the actual input document.

use std::fmt::Write;

use serde_json::Value;

use crate::graph::DomainOntology;
use crate::prompt::knowledge::{render_entity_list, render_relation_list};
use crate::prompt::{GraphSchema, PropertyType};
use crate::promptgen::{JsonSchemaSummary, PromptBuilder, PromptGenOptions};
use crate::types::TypeRegistry;

/// Knobs for prompt assembly. Mirrors the relevant subset of
/// [`PromptGenOptions`] plus a cap on how much of the input document is
/// inlined into the user prompt.
#[derive(Debug, Clone)]
pub struct MapGenPromptOptions {
    /// Include the worked few-shot example from `promptgen`.
    pub include_examples: bool,
    /// Include the heuristic "Inferred structure" section (carries the
    /// analyser's primary-key / type guesses).
    pub include_inferred_summary: bool,
    /// Optional live type registry — drives the "Available field types"
    /// section when present, otherwise the bundled catalogue is used.
    pub registry: Option<TypeRegistry>,
    /// Upper bound on the number of characters of the pretty-printed
    /// input document inlined into the user prompt. Oversized documents
    /// are truncated with a marker so a huge file can't blow the context.
    pub max_data_chars: usize,
}

impl Default for MapGenPromptOptions {
    fn default() -> Self {
        Self {
            include_examples: true,
            include_inferred_summary: true,
            registry: None,
            max_data_chars: 16_000,
        }
    }
}

impl MapGenPromptOptions {
    fn to_promptgen_opts(&self) -> PromptGenOptions {
        PromptGenOptions {
            include_examples: self.include_examples,
            include_inferred_summary: self.include_inferred_summary,
            registry: self.registry.clone(),
            ..PromptGenOptions::default()
        }
    }
}

/// Build the `(system, user)` prompt pair for mapping generation.
///
/// `summary` is the precomputed [`crate::promptgen::analyze`] result for
/// `data` (passed in so callers can reuse it elsewhere, e.g. interactive
/// refinement).
pub fn build_mapping_prompt(
    data: &Value,
    summary: &JsonSchemaSummary,
    ontology: &DomainOntology,
    schema: Option<&GraphSchema>,
    opts: &MapGenPromptOptions,
) -> (String, String) {
    let base = PromptBuilder::new(summary.clone(), opts.to_promptgen_opts()).build();

    let mut system = String::with_capacity(base.len() + 2048);
    render_ontology_section(&mut system, ontology);
    if let Some(s) = schema {
        if !s.nodes.is_empty() || !s.relationships.is_empty() {
            render_schema_section(&mut system, s);
        }
    }
    system.push_str(&base);

    let user = render_user_payload(data, opts.max_data_chars);
    (system, user)
}

fn render_ontology_section(out: &mut String, ontology: &DomainOntology) {
    out.push_str("# Allowed entity types (STRICT — use ONLY these)\n\n");
    out.push_str(
        "Map the input into ONLY the entity types listed below. Do NOT invent new entity \
         `type` values. The \"Inferred structure\" section further down suggests names taken \
         from the raw JSON keys — you MUST map those onto the allowed types here, renaming \
         as needed.\n\n\
         For each entity type, use the declared properties when the data provides them; you \
         MAY additionally add properties you find in the data that are not listed.\n\n",
    );
    render_entity_list(out, &ontology.entity_types);
    out.push('\n');
    out.push_str(
        "In each property's mapping `type`, use the field-type vocabulary from \
         \"# Available field types\" — NOT the ontology type labels above. Translate: \
         ontology `text` → `SemanticText`; `string` → `Text` (or `Keyword` for short \
         categorical values); `int`/`float` → `Number`; `bool` → `Boolean`; \
         `date`/`datetime` → `DateTime`; `list` → `List`.\n\n",
    );

    out.push_str("# Relation types\n\n");
    if ontology.relation_types.is_empty() {
        out.push_str("No relation types are predefined.\n");
    } else {
        out.push_str("These relation types are suggested:\n\n");
        render_relation_list(out, &ontology.relation_types);
    }
    out.push_str(
        "\nYou MAY introduce additional relationships beyond this list when the data clearly \
         implies them. Each relationship's `from`/`to` MUST reference entity types you \
         actually emit.\n\n",
    );
}

fn render_schema_section(out: &mut String, schema: &GraphSchema) {
    out.push_str("# Existing graph schema (reuse where it matches)\n\n");
    out.push_str(
        "The target graph already contains the following. Reuse these labels, property \
         names, and relationship types when they fit the data instead of coining new ones.\n\n",
    );

    if !schema.nodes.is_empty() {
        out.push_str("Nodes:\n");
        for node in &schema.nodes {
            let _ = writeln!(out, "- `{}`", node.label);
            for p in &node.properties {
                let _ = writeln!(out, "    - `{}`: {}", p.name, property_type_label(p.ty));
            }
        }
        out.push('\n');
    }

    if !schema.relationships.is_empty() {
        out.push_str("Relationships:\n");
        for rel in &schema.relationships {
            match (&rel.from, &rel.to) {
                (Some(from), Some(to)) => {
                    let _ = writeln!(out, "- `{}` ({from} → {to})", rel.label);
                }
                _ => {
                    let _ = writeln!(out, "- `{}`", rel.label);
                }
            }
        }
        out.push('\n');
    }
}

fn property_type_label(ty: PropertyType) -> &'static str {
    match ty {
        PropertyType::String => "string",
        PropertyType::Int => "int",
        PropertyType::Float => "float",
        PropertyType::Bool => "bool",
        PropertyType::Date => "date",
        PropertyType::Datetime => "datetime",
        PropertyType::List => "list",
    }
}

fn render_user_payload(data: &Value, max_chars: usize) -> String {
    let mut body = serde_json::to_string_pretty(data).unwrap_or_else(|_| data.to_string());
    if body.chars().count() > max_chars {
        let cut: String = body.chars().take(max_chars).collect();
        body = format!("{cut}\n… (input truncated for length)");
    }
    format!(
        "# Input JSON\n\n```json\n{body}\n```\n\n\
         Produce the linguagraph mapping JSON for the document above now. \
         Output ONLY the JSON mapping.",
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{EntityTypeSpec, RelationTypeSpec};
    use crate::prompt::{NodeKind, Property, RelKind};
    use crate::promptgen::analyze;
    use serde_json::json;

    fn ontology() -> DomainOntology {
        DomainOntology {
            entity_types: vec![EntityTypeSpec::with_description("Company", "A business.")],
            relation_types: vec![RelationTypeSpec::new("OWNS")],
        }
    }

    #[test]
    fn system_prompt_lists_allowed_entity_types_and_relations() {
        let data = json!({"companies": [{"id": 1, "name": "Stripe"}]});
        let summary = analyze(&data);
        let (system, user) = build_mapping_prompt(
            &data,
            &summary,
            &ontology(),
            None,
            &MapGenPromptOptions::default(),
        );
        assert!(system.contains("# Allowed entity types (STRICT"));
        assert!(system.contains("* `Company` — A business."));
        assert!(system.contains("* `OWNS`"));
        // Authoritative whitelist comes before the reused promptgen base.
        let onto_idx = system.find("# Allowed entity types").unwrap();
        let spec_idx = system.find("# Mapping schema").unwrap();
        assert!(onto_idx < spec_idx);
        // User payload carries the document.
        assert!(user.contains("\"companies\""));
        assert!(user.contains("Output ONLY the JSON mapping"));
    }

    #[test]
    fn schema_section_rendered_when_present() {
        let data = json!({"companies": [{"id": 1}]});
        let summary = analyze(&data);
        let schema = GraphSchema {
            nodes: vec![NodeKind {
                label: "Company".into(),
                domain: None,
                extra_labels: vec![],
                scopes: vec![],
                description: None,
                properties: vec![Property {
                    name: "name".into(),
                    ty: PropertyType::String,
                    description: None,
                }],
            }],
            relationships: vec![RelKind {
                label: "OWNS".into(),
                domain: None,
                description: None,
                from: Some("Person".into()),
                to: Some("Company".into()),
                properties: vec![],
            }],
        };
        let (system, _) = build_mapping_prompt(
            &data,
            &summary,
            &ontology(),
            Some(&schema),
            &MapGenPromptOptions::default(),
        );
        assert!(system.contains("# Existing graph schema"));
        assert!(system.contains("- `Company`"));
        assert!(system.contains("- `name`: string"));
        assert!(system.contains("- `OWNS` (Person → Company)"));
    }
}
