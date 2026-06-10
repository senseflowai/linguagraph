//! Assemble a [`JsonSchemaSummary`] into a final prompt string.
//!
//! The builder is a thin orchestrator: it owns the section ordering
//! and the per-summary rendering of "Inferred structure" / "Available
//! field types". Everything else is a static `&str` from
//! [`super::template`].

use std::fmt::Write;

use crate::types::TypeRegistry;

use super::analyzer::{JsonSchemaSummary, RelationshipHint};
use super::inference::InferredType;
use super::template;

/// Caller-supplied knobs.
#[derive(Debug, Clone)]
pub struct PromptGenOptions {
    /// Bias the prompt towards a specific field type when the
    /// heuristic is uncertain. The LLM still gets the heuristic's
    /// guess, plus a "preferred when in doubt" line per entry.
    pub preferred_types: Vec<String>,
    /// Free-form domain hints rendered verbatim under a "Domain
    /// hints" section (e.g. `"this is a CRM dataset"`).
    pub domain_hints: Vec<String>,
    /// Hard constraints rendered verbatim before the rules.
    pub constraints: Vec<String>,
    /// Output language. Currently only English; the option is here so
    /// future locale support doesn't need a signature change.
    pub language: Language,
    /// Include the worked few-shot example.
    pub include_examples: bool,
    /// Include the analysed `Inferred structure` section.
    pub include_inferred_summary: bool,
    /// Optional live type registry — when supplied, the
    /// "Available field types" section is auto-rendered from the
    /// registered handlers. When `None`, the bundled catalogue is
    /// used.
    pub registry: Option<TypeRegistry>,
}

impl Default for PromptGenOptions {
    fn default() -> Self {
        Self {
            preferred_types: Vec::new(),
            domain_hints: Vec::new(),
            constraints: Vec::new(),
            language: Language::English,
            include_examples: true,
            include_inferred_summary: true,
            registry: None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Language {
    #[default]
    English,
}

/// Renders prompts.
#[derive(Debug, Clone)]
pub struct PromptBuilder {
    summary: JsonSchemaSummary,
    opts: PromptGenOptions,
}

impl PromptBuilder {
    pub fn new(summary: JsonSchemaSummary, opts: PromptGenOptions) -> Self {
        Self { summary, opts }
    }

    pub fn build(&self) -> String {
        let mut out = String::with_capacity(4096);

        // 1. Preamble.
        out.push_str(template::PREAMBLE);
        out.push_str("\n\n");

        // 2. Mapping schema spec.
        out.push_str(template::MAPPING_SPEC);
        out.push('\n');

        // 3. Available field types — registry-aware when configured.
        match &self.opts.registry {
            Some(reg) if !reg.is_empty() => {
                out.push_str("# Available field types\n\n");
                let mut handlers: Vec<_> = reg.iter().collect();
                handlers.sort_by(|a, b| a.type_id().0.cmp(&b.type_id().0));
                for h in handlers {
                    let hint = h.prompt_hint();
                    let _ = writeln!(out, "- **{}**", contract_type_name(&hint.type_id.0));
                    if let Some(doc) = hint.doc {
                        let _ = writeln!(out, "    {doc}");
                    }
                    if !hint.ops.is_empty() {
                        let ops: Vec<&str> = hint.ops.iter().map(|o| o.as_str()).collect();
                        let _ = writeln!(out, "    ops: {}", ops.join(", "));
                    }
                }
                // Always also list the analyser's vocabulary so the LLM
                // knows the full set even when a handler isn't registered
                // for every type in this deployment.
                out.push_str(
                    "\nThe full field-type vocabulary is: `Keyword` (plain string, \
                     exact / range / regex matching), `Text` (free-form text, \
                     semantic search), `Number`, `Boolean`, `DateTime`.\n",
                );
            }
            _ => out.push_str(template::DEFAULT_TYPES_CATALOGUE),
        }
        out.push('\n');

        // 4. Domain hints (optional, free-form).
        if !self.opts.domain_hints.is_empty() {
            out.push_str("# Domain hints\n\n");
            for h in &self.opts.domain_hints {
                let _ = writeln!(out, "- {h}");
            }
            out.push('\n');
        }

        // 5. Preferred types (optional bias).
        if !self.opts.preferred_types.is_empty() {
            out.push_str("# Preferred types\n\n");
            out.push_str("When the field type is ambiguous, prefer in this order: ");
            out.push_str(&self.opts.preferred_types.join(", "));
            out.push_str(".\n\n");
        }

        // 6. Constraints (optional hard rules).
        if !self.opts.constraints.is_empty() {
            out.push_str("# Constraints\n\n");
            for c in &self.opts.constraints {
                let _ = writeln!(out, "- {c}");
            }
            out.push('\n');
        }

        // 7. Inferred structure summary.
        if self.opts.include_inferred_summary && !self.summary.entities.is_empty() {
            out.push_str("# Inferred structure\n\n");
            out.push_str(
                "These are the analyser's *suggestions*. Correct any that look wrong; \
                 don't blindly copy them.\n\n",
            );
            for ent in &self.summary.entities {
                render_entity(&mut out, ent);
                out.push('\n');
            }
            if !self.summary.relationships.is_empty() {
                out.push_str("Relationship hints:\n");
                for rel in &self.summary.relationships {
                    render_relationship(&mut out, rel);
                }
                out.push('\n');
            }
        }

        // 8. Rules.
        out.push_str(template::RULES);
        out.push('\n');

        // 9. Few-shot example.
        if self.opts.include_examples {
            out.push_str(template::EXAMPLE);
            out.push('\n');
        }

        // 10. Tail.
        out.push_str(template::TAIL);
        out.push_str("\n\n");

        out
    }
}

fn render_entity(out: &mut String, ent: &super::EntitySummary) {
    let _ = writeln!(
        out,
        "## {} (`{}`, {} sample{})",
        ent.name,
        ent.source_path,
        ent.samples,
        if ent.samples == 1 { "" } else { "s" }
    );
    match &ent.primary_key {
        Some(pk) => {
            let _ = writeln!(out, "- primary_key: `{pk}`");
        }
        None => {
            out.push_str(
                "- primary_key: **(not detected)** — choose the most likely \
                 unique-looking scalar.\n",
            );
        }
    }
    if ent.fields.is_empty() {
        out.push_str("- (no scalar fields detected)\n");
        return;
    }
    out.push_str("- fields:\n");
    for f in &ent.fields {
        let suggested = render_suggested_type(f.inferred_type);
        let mut samples = if f.samples.is_empty() {
            String::new()
        } else {
            // Truncate each sample to keep the prompt compact.
            let truncated: Vec<String> = f
                .samples
                .iter()
                .map(|s| {
                    if s.chars().count() > 60 {
                        let cut: String = s.chars().take(57).collect();
                        format!("{cut}...")
                    } else {
                        s.clone()
                    }
                })
                .map(|s| format!("`{}`", s.replace('`', "'")))
                .collect();
            format!(" — samples: {}", truncated.join(", "))
        };
        // For the Identifier case we never get here (those are already
        // promoted to primary_key), so the branch is dead in practice
        // — left in for paranoia.
        if f.inferred_type == InferredType::Identifier {
            samples.push_str(" (looks like an identifier)");
        }
        let _ = writeln!(
            out,
            "    - `{}` → {}  (distinct={}, non-null={}){}",
            f.name, suggested, f.distinct, f.non_null, samples
        );
    }
}

fn render_suggested_type(t: InferredType) -> &'static str {
    match t {
        InferredType::Keyword => "**Keyword**",
        InferredType::Text => "**Text**",
        InferredType::DateTime => "**DateTime**",
        InferredType::Number => "Number",
        InferredType::Boolean => "Boolean",
        InferredType::Identifier => "Identifier",
        InferredType::Unknown => "Text *(uncertain — confirm)*",
    }
}

/// Map a registry handler id to the contract-facing type name shown to
/// the LLM. The semantic handler is registered under `SemanticText`, but
/// the contract exposes it as `Text` (everything textual that isn't a
/// `Keyword`).
fn contract_type_name(handler_id: &str) -> &str {
    match handler_id {
        "SemanticText" => "Text",
        other => other,
    }
}

fn render_relationship(out: &mut String, rel: &RelationshipHint) {
    match rel {
        RelationshipHint::NestedEntity {
            parent,
            child,
            nested_path,
        } => {
            let _ = writeln!(
                out,
                "- `{parent}` HAS_MANY `{child}` (nested at `{nested_path}`) \
                 → consider a `HAS_{child_upper}` relationship.",
                child_upper = child.to_ascii_uppercase()
            );
        }
        RelationshipHint::ForeignKey {
            from,
            field,
            source_path,
        } => {
            let _ = writeln!(
                out,
                "- `{from}` has a foreign key `{field}` at `{source_path}` → \
                 emit a relationship to the entity this id references, setting \
                 `from_key` to `{source_path}` and `to_key` to that entity's \
                 `primary_key`.",
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::promptgen::analyze;
    use serde_json::json;

    fn render(v: serde_json::Value, opts: PromptGenOptions) -> String {
        PromptBuilder::new(analyze(&v), opts).build()
    }

    #[test]
    fn always_starts_with_preamble_and_ends_with_tail() {
        let out = render(json!({"x": 1}), PromptGenOptions::default());
        assert!(out.starts_with(template::PREAMBLE));
        assert!(out.trim_end().ends_with(template::TAIL.trim_end()));
    }

    #[test]
    fn includes_mapping_spec_and_rules() {
        let out = render(json!({"x": 1}), PromptGenOptions::default());
        assert!(out.contains("# Mapping schema"));
        assert!(out.contains("# Rules"));
        assert!(out.contains("# Available field types"));
    }

    #[test]
    fn renders_inferred_entities_with_samples_and_types() {
        let v = json!({
            "companies": [
                {"id": 1, "name": "Stripe", "description": "Payments.",
                 "industry": "Fintech"}
            ]
        });
        let out = render(v, PromptGenOptions::default());
        assert!(out.contains("# Inferred structure"));
        assert!(out.contains("## Company (`$.companies[*]`"));
        assert!(out.contains("primary_key: `$.companies[*].id`"));
        // Description is short here so length-only would miss it; the
        // name hint kicks in → the semantic `Text` type.
        assert!(out.contains("`description` → **Text**"));
        assert!(out.contains("`industry` → **Keyword**"));
    }

    #[test]
    fn examples_can_be_disabled() {
        let mut opts = PromptGenOptions::default();
        opts.include_examples = false;
        let out = render(json!({"x": 1}), opts);
        assert!(!out.contains("# Example"));
        assert!(out.contains("# Rules"));
    }

    #[test]
    fn domain_hints_and_preferred_types_render() {
        let mut opts = PromptGenOptions::default();
        opts.domain_hints = vec!["this is a CRM dataset".into()];
        opts.preferred_types = vec!["Text".into(), "Keyword".into()];
        opts.constraints = vec!["entity names must be in English".into()];
        let out = render(json!({"x": 1}), opts);
        assert!(out.contains("# Domain hints"));
        assert!(out.contains("this is a CRM dataset"));
        assert!(out.contains("# Preferred types"));
        assert!(out.contains("Text, Keyword"));
        assert!(out.contains("# Constraints"));
        assert!(out.contains("entity names must be in English"));
    }

    #[test]
    fn missing_primary_key_is_called_out() {
        let v = json!({"items": [{"name": "x"}, {"name": "y"}]});
        let out = render(v, PromptGenOptions::default());
        assert!(out.contains("primary_key: **(not detected)**"));
    }

    #[test]
    fn registry_overrides_default_catalogue() {
        use crate::embeddings::MockEmbedder;
        use crate::types::handlers::{SemanticTextConfig, SemanticTextHandler};
        use crate::types::RegistryBuilder;
        use std::sync::Arc;

        let cfg = SemanticTextConfig {
            embedding_model: None,
            collection: "test".into(),
            top_k: 10,
            search_threshold: 0.8,
            reranker_threshold: 0.3,
        };
        let registry = RegistryBuilder::new()
            .register(SemanticTextHandler::new(
                cfg,
                Arc::new(MockEmbedder::new(8)),
            ))
            .build();
        let mut opts = PromptGenOptions::default();
        opts.registry = Some(registry);
        let out = render(json!({"x": 1}), opts);
        assert!(out.contains("# Available field types"));
        // The semantic handler is registered as `SemanticText` but shown
        // to the LLM under the contract name `Text`.
        assert!(out.contains("**Text**"));
        assert!(!out.contains("**SemanticText**"));
        // The full-vocabulary line keeps `Keyword` available too.
        assert!(out.contains("Keyword"));
    }
}
