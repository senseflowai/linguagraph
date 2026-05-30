//! High-level [`PromptGenerator`] facade.
//!
//! Holds an [`OntologyCatalog`] and exposes a small, consistent surface
//! for the two flavors of LLM prompt the project produces:
//!
//! * `query_prompt` — schema-aware DSL prompt (delegates to
//!   [`super::generator::generate_query_prompt`]).
//! * `knowledge_extract_prompt` — domain-scoped knowledge extraction
//!   prompt (delegates to [`super::knowledge::render_knowledge_extract_prompt`]).

use crate::config::PromptConfig;
use crate::graph::{
    DomainOntology, JsonFileOntologyCatalogStorage, OntologyCatalog, OntologyCatalogStorage,
    OntologyError,
};

use super::generator::{self, PromptOptions};
use super::knowledge::render_knowledge_extract_prompt;
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

    /// Load the catalog through a custom [`OntologyCatalogStorage`].
    /// Use this to plug in a Postgres / HTTP / S3 backend.
    pub async fn from_storage<S: OntologyCatalogStorage + ?Sized>(
        storage: &S,
    ) -> Result<Self, OntologyError> {
        let catalog = storage.load().await?;
        Ok(Self::new(catalog))
    }

    /// Build from a [`PromptConfig`].
    ///
    /// * When `ontologies_path` is set, the JSON file is loaded via the
    ///   default [`JsonFileOntologyCatalogStorage`].
    /// * Otherwise the built-in catalog is used.
    ///
    /// To inject a non-filesystem backend (Postgres, HTTP service)
    /// build it directly and use [`Self::from_storage`].
    pub async fn from_config(cfg: &PromptConfig) -> Result<Self, OntologyError> {
        let catalog = match &cfg.ontologies_path {
            Some(path) => {
                let storage = JsonFileOntologyCatalogStorage::new(path);
                storage.load().await?
            }
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
    /// The domain name is also substituted into the prompt's framing
    /// sections (role, input structure, rules).
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
        Ok(render_knowledge_extract_prompt(fragment, &name, ontology))
    }

    /// Escape hatch: render with a caller-built ontology, bypassing
    /// the catalog. Useful for CLI overrides. `domain` is still used
    /// for the prompt's framing — pass something descriptive
    /// (e.g. `"custom"`, `"ad-hoc"`).
    pub fn knowledge_extract_prompt_with(
        &self,
        fragment: &str,
        domain: &str,
        ontology: &DomainOntology,
    ) -> String {
        render_knowledge_extract_prompt(fragment, domain, ontology)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{EntityTypeSpec, RelationTypeSpec};
    use crate::prompt::storage::InMemoryOntologyCatalogStorage;

    #[test]
    fn knowledge_extract_uses_explicit_domain() {
        let g = PromptGenerator::with_builtin_catalog();
        let p = g
            .knowledge_extract_prompt("text", Some("legal"))
            .expect("legal domain present in builtin catalog");
        assert!(p.contains("* `LegalNorm`"));
        assert!(p.contains("* `GRANTS`"));
        // Framing is substituted with the selected domain.
        assert!(p.contains("**legal information extraction**"));
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

    #[tokio::test]
    async fn from_config_with_no_path_uses_builtin() {
        let cfg = PromptConfig {
            ontologies_path: None,
            default_domain: Some("legal".into()),
        };
        let g = PromptGenerator::from_config(&cfg).await.unwrap();
        assert_eq!(g.default_domain(), Some("legal"));
        assert!(g.catalog().get("legal").is_some());
    }

    #[tokio::test]
    async fn from_storage_loads_through_the_backend() {
        let storage = InMemoryOntologyCatalogStorage::new(OntologyCatalog::builtin());
        let g = PromptGenerator::from_storage(&storage).await.unwrap();
        assert!(g.catalog().get("legal").is_some());
    }

    #[test]
    fn override_ontology_bypasses_catalog() {
        let g = PromptGenerator::default();
        let onto = DomainOntology {
            entity_types: vec![EntityTypeSpec::new("X")],
            relation_types: vec![RelationTypeSpec::new("R")],
        };
        let p = g.knowledge_extract_prompt_with("frag", "custom", &onto);
        assert!(p.contains("* `X`"));
        assert!(p.contains("* `R`"));
        assert!(p.contains("**custom information extraction**"));
    }
}
