//! Domain ontologies for knowledge-extraction prompts.
//!
//! An [`OntologyCatalog`] is a `domain → DomainOntology` map deserialized
//! from a flat JSON object. The catalog ships with a built-in legal
//! vocabulary (compiled in via [`include_str!`]) and can be replaced or
//! extended at runtime by loading a user-provided JSON file (path is
//! taken from `[prompt].ontologies_path` in the TOML config).

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Type of a property defined in the ontology.
/// Maps to LinguaGraph's `PropertyType`: `String` → `Text` (embedded),
/// `Int`/`Float` → `Number`, `Bool` → `Boolean`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum OntologyPropertyType {
    String,
    Int,
    Float,
    Bool,
}

/// One typed property that an entity of a given type may carry.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PropertySpec {
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub property_type: OntologyPropertyType,
    #[serde(default)]
    pub required: bool,
}

/// One allowed entity type the LLM may emit.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct EntityTypeSpec {
    /// Canonical PascalCase name (e.g. `LegalNorm`).
    pub name: String,
    /// Optional one-line description shown alongside the name.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// Typed properties the LLM should extract for entities of this type.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub properties: Vec<PropertySpec>,
}

impl EntityTypeSpec {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: None,
            properties: vec![],
        }
    }

    pub fn with_description(name: impl Into<String>, desc: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: Some(desc.into()),
            properties: vec![],
        }
    }
}

/// One allowed relation type the LLM may emit.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RelationTypeSpec {
    /// Canonical UPPER_SNAKE name (e.g. `GRANTS`).
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
}

impl RelationTypeSpec {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: None,
        }
    }

    pub fn with_description(name: impl Into<String>, desc: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: Some(desc.into()),
        }
    }
}

/// All allowed types for a single domain (e.g. `legal`, `medical`).
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct DomainOntology {
    #[serde(default)]
    pub entity_types: Vec<EntityTypeSpec>,
    #[serde(default)]
    pub relation_types: Vec<RelationTypeSpec>,
}

/// Catalog of domain ontologies. Serializes as a flat JSON object
/// (`{ "legal": {...}, "medical": {...} }`) — no extra wrapper key.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(transparent)]
pub struct OntologyCatalog {
    pub domains: BTreeMap<String, DomainOntology>,
}

/// Embedded built-in catalog, compiled into the binary.
const BUILTIN_JSON: &str = include_str!("ontologies.default.json");

impl OntologyCatalog {
    /// Built-in catalog (currently: `legal` domain).
    ///
    /// # Panics
    /// Never — the embedded JSON is validated at compile time by the test
    /// suite. A panic here means a broken build artifact.
    pub fn builtin() -> Self {
        serde_json::from_str(BUILTIN_JSON).expect("builtin ontologies.default.json must parse")
    }

    pub fn load_from_str(raw: &str) -> Result<Self, OntologyError> {
        Ok(serde_json::from_str(raw)?)
    }

    pub fn load_from_path(path: &Path) -> Result<Self, OntologyError> {
        if !path.exists() {
            return Err(OntologyError::NotFound(path.to_path_buf()));
        }
        let raw = std::fs::read_to_string(path)?;
        Self::load_from_str(&raw)
    }

    pub fn get(&self, domain: &str) -> Option<&DomainOntology> {
        self.domains.get(domain)
    }

    pub fn domains(&self) -> impl Iterator<Item = &str> {
        self.domains.keys().map(String::as_str)
    }

    pub fn insert(&mut self, domain: impl Into<String>, ontology: DomainOntology) {
        self.domains.insert(domain.into(), ontology);
    }
}

#[derive(Debug, Error)]
pub enum OntologyError {
    #[error("ontology file not found: {0}")]
    NotFound(PathBuf),
    #[error("ontology parse error: {0}")]
    Parse(#[from] serde_json::Error),
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("unknown domain: {0}")]
    UnknownDomain(String),
    /// The backend does not support the requested operation
    /// (e.g. a read-only storage backend cannot serve `save`).
    #[error("unsupported storage operation: {0}")]
    Unsupported(String),
    /// Wraps an arbitrary backend-specific error (Postgres, HTTP, …).
    #[error("ontology storage backend error: {0}")]
    Backend(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loads_builtin_catalog_with_legal_domain() {
        let cat = OntologyCatalog::builtin();
        let legal = cat.get("legal").expect("legal domain must be present");
        let entity_names: Vec<&str> = legal
            .entity_types
            .iter()
            .map(|e| e.name.as_str())
            .collect();
        assert!(entity_names.contains(&"LegalNorm"));
        assert!(entity_names.contains(&"StateBody"));
        let rel_names: Vec<&str> = legal
            .relation_types
            .iter()
            .map(|r| r.name.as_str())
            .collect();
        assert!(rel_names.contains(&"GRANTS"));
        assert!(rel_names.contains(&"APPLIES_TO"));
    }

    #[test]
    fn round_trip_json_serialization() {
        let mut cat = OntologyCatalog::default();
        cat.insert(
            "demo",
            DomainOntology {
                entity_types: vec![EntityTypeSpec::with_description("Foo", "A foo.")],
                relation_types: vec![RelationTypeSpec::new("KNOWS")],
            },
        );
        let raw = serde_json::to_string(&cat).unwrap();
        // Flat shape — no "domains" wrapper.
        assert!(raw.starts_with(r#"{"demo":"#));
        let back: OntologyCatalog = serde_json::from_str(&raw).unwrap();
        assert_eq!(
            back.get("demo").unwrap().entity_types[0],
            EntityTypeSpec::with_description("Foo", "A foo.")
        );
    }

    #[test]
    fn entity_type_spec_with_properties_round_trips() {
        let spec = EntityTypeSpec {
            name: "Person".to_string(),
            description: Some("A human being.".to_string()),
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
        };
        let raw = serde_json::to_string(&spec).unwrap();
        let back: EntityTypeSpec = serde_json::from_str(&raw).unwrap();
        assert_eq!(back, spec);
    }

    #[test]
    fn entity_type_spec_without_properties_deserializes_empty_vec() {
        // Old JSON rows without "properties" must deserialize cleanly.
        let raw = r#"{"name":"Foo","description":"bar"}"#;
        let spec: EntityTypeSpec = serde_json::from_str(raw).unwrap();
        assert!(spec.properties.is_empty());
    }

    #[test]
    fn unknown_domain_returns_none() {
        let cat = OntologyCatalog::builtin();
        assert!(cat.get("nope").is_none());
    }

    #[test]
    fn load_missing_path_errors() {
        let err = OntologyCatalog::load_from_path(Path::new("/nonexistent/ontologies.json"))
            .unwrap_err();
        assert!(matches!(err, OntologyError::NotFound(_)));
    }

    #[test]
    fn load_from_str_parses_minimal_catalog() {
        let raw = r#"{"x":{"entity_types":[{"name":"A"}],"relation_types":[{"name":"R"}]}}"#;
        let cat = OntologyCatalog::load_from_str(raw).unwrap();
        let d = cat.get("x").unwrap();
        assert_eq!(d.entity_types[0].name, "A");
        assert_eq!(d.relation_types[0].name, "R");
    }
}
