//! High-level [`PromptGenerator`] facade.
//!
//! Holds an [`OntologyCatalog`] and exposes a small, consistent surface
//! for the two flavors of LLM prompt the project produces:
//!
//! * `query_prompt` — schema-aware DSL prompt (delegates to
//!   [`super::generator::generate_query_prompt`]).
//! * `knowledge_extract_prompt` — domain-scoped knowledge extraction
//!   prompt (delegates to [`super::knowledge::render_knowledge_extract_prompt`]).

use std::path::Path;

use crate::config::PromptConfig;

use super::generator::{self, PromptOptions};
use super::knowledge::render_knowledge_extract_prompt;
use super::ontology::{DomainOntology, OntologyCatalog, OntologyError};
use super::schema::GraphSchema;

/// Unified entry point for prompt generation.
#[derive(Debug, Clone, Default)]
pub struct PromptGenerator {
    catalog: OntologyCatalog,
    default_domain: Option<String>,
}

impl PromptGenerator {
    /// Construct from an existing catalog.
    pub fn new(catalog: OntologyCatalog) -> Self {
        Self {
            catalog,
            default_domain: None,
        }
    }

    /// Convenience: generator pre-loaded with the built-in catalog
    /// (currently the `legal` domain).
    pub fn with_builtin_catalog() -> Self {
        Self::new(OntologyCatalog::builtin())
    }

    pub fn with_default_domain(mut self, domain: impl Into<String>) -> Self {
        self.default_domain = Some(domain.into());
        self
    }

    /// Build from a [`PromptConfig`]. When `ontologies_path` is set the
    /// JSON file at that path is loaded; otherwise the built-in catalog
    /// is used.
    pub fn from_config(cfg: &PromptConfig) -> Result<Self, OntologyError> {
        let catalog = match &cfg.ontologies_path {
            Some(path) => OntologyCatalog::load_from_path(Path::new(path))?,
            None => OntologyCatalog::builtin(),
        };
        let mut g = Self::new(catalog);
        if let Some(d) = &cfg.default_domain {
            g.default_domain = Some(d.clone());
        }
        Ok(g)
    }

    /// The loaded catalog (for inspection / extending at runtime).
    pub fn catalog(&self) -> &OntologyCatalog {
        &self.catalog
    }

    pub fn catalog_mut(&mut self) -> &mut OntologyCatalog {
        &mut self.catalog
    }

    pub fn default_domain(&self) -> Option<&str> {
        self.default_domain.as_deref()
    }

    /// Render a DSL query prompt for `query` against `schema`.
    /// Pure delegation to the low-level renderer.
    pub fn query_prompt(
        &self,
        query: &str,
        schema: &GraphSchema,
        opts: &PromptOptions,
    ) -> String {
        generator::generate_query_prompt(query, schema, opts)
    }

    /// Render a knowledge-extraction prompt for `fragment`.
    ///
    /// `domain` selects the ontology from the catalog. When `None`, the
    /// generator's [`default_domain`](Self::default_domain) is used; if
    /// neither is set the call fails with [`OntologyError::UnknownDomain`].
    pub fn knowledge_extract_prompt(
        &self,
        fragment: &str,
        domain: Option<&str>,
    ) -> Result<String, OntologyError> {
        let name = domain
            .map(str::to_owned)
            .or_else(|| self.default_domain.clone())
            .ok_or_else(|| OntologyError::UnknownDomain("<unspecified>".into()))?;
        let ontology = self
            .catalog
            .get(&name)
            .ok_or_else(|| OntologyError::UnknownDomain(name.clone()))?;
        Ok(render_knowledge_extract_prompt(fragment, ontology))
    }

    /// Escape hatch: render with a caller-built ontology, bypassing
    /// the catalog. Useful for CLI overrides.
    pub fn knowledge_extract_prompt_with(
        &self,
        fragment: &str,
        ontology: &DomainOntology,
    ) -> String {
        render_knowledge_extract_prompt(fragment, ontology)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prompt::ontology::{EntityTypeSpec, RelationTypeSpec};

    #[test]
    fn knowledge_extract_uses_explicit_domain() {
        let g = PromptGenerator::with_builtin_catalog();
        let p = g
            .knowledge_extract_prompt("text", Some("legal"))
            .expect("legal domain present in builtin catalog");
        assert!(p.contains("* `LegalNorm`"));
        assert!(p.contains("* `GRANTS`"));
    }

    #[test]
    fn knowledge_extract_uses_default_domain_when_none() {
        let g = PromptGenerator::with_builtin_catalog().with_default_domain("legal");
        let p = g.knowledge_extract_prompt("text", None).unwrap();
        assert!(p.contains("* `LegalNorm`"));
    }

    #[test]
    fn unknown_domain_errors() {
        let g = PromptGenerator::with_builtin_catalog();
        let err = g.knowledge_extract_prompt("x", Some("medical")).unwrap_err();
        assert!(matches!(err, OntologyError::UnknownDomain(d) if d == "medical"));
    }

    #[test]
    fn missing_domain_and_default_errors() {
        let g = PromptGenerator::with_builtin_catalog();
        let err = g.knowledge_extract_prompt("x", None).unwrap_err();
        assert!(matches!(err, OntologyError::UnknownDomain(_)));
    }

    #[test]
    fn from_config_with_no_path_uses_builtin() {
        let cfg = PromptConfig {
            ontologies_path: None,
            default_domain: Some("legal".into()),
        };
        let g = PromptGenerator::from_config(&cfg).unwrap();
        assert_eq!(g.default_domain(), Some("legal"));
        assert!(g.catalog().get("legal").is_some());
    }

    #[test]
    fn override_ontology_bypasses_catalog() {
        let g = PromptGenerator::default();
        let onto = DomainOntology {
            entity_types: vec![EntityTypeSpec::new("X")],
            relation_types: vec![RelationTypeSpec::new("R")],
        };
        let p = g.knowledge_extract_prompt_with("frag", &onto);
        assert!(p.contains("* `X`"));
        assert!(p.contains("* `R`"));
    }
}
