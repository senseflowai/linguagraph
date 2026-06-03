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

use crate::embeddings::{EmbedError, Embedder, Reranker};
use crate::prompt::GraphSchema;

/// Type of a property defined in the ontology.
///
/// Covers both the lexical vocabulary the LLM emits during knowledge
/// extraction and the storage shape ingested into the graph. `Text` is
/// the semantic-text variant: properties of this type are routed
/// through the `SemanticTextHandler` (embedded + vector-searchable).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum OntologyPropertyType {
    String,
    Text,
    Int,
    Float,
    Bool,
    Date,
    Datetime,
    List,
}

impl OntologyPropertyType {
    /// Stable type id used by the type registry to look up handlers.
    /// Mirrors the historical mapping baked into the ingest pipeline.
    pub fn type_id(self) -> &'static str {
        match self {
            Self::String => "Text",
            Self::Text => "SemanticText",
            Self::Int | Self::Float => "Number",
            Self::Bool => "Boolean",
            Self::Date | Self::Datetime => "Timestamp",
            Self::List => "List",
        }
    }

    /// Query-side type id: `Some(...)` for types whose DSL filters need
    /// a specialised handler (semantic search, timestamp interval
    /// lowering), `None` for plain comparisons.
    pub fn query_type_id(self) -> Option<&'static str> {
        match self {
            Self::Text => Some("SemanticText"),
            Self::String | Self::Date | Self::Datetime => Some(self.type_id()),
            Self::Int | Self::Float | Self::Bool | Self::List => None,
        }
    }
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
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct EntityTypeSpec {
    /// Canonical PascalCase name (e.g. `LegalNorm`).
    pub name: String,
    /// Optional one-line description shown alongside the name.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// Typed properties the LLM should extract for entities of this type.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub properties: Vec<PropertySpec>,
    /// Cached embedding of the entity description, used by
    /// [`OntologyCatalog::find`] for semantic entity matching. Recomputed
    /// from `name + description + properties` whenever the catalog is
    /// saved through storage.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub embedding: Option<Vec<f32>>,
}

impl EntityTypeSpec {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: None,
            properties: vec![],
            embedding: None,
        }
    }

    pub fn with_description(name: impl Into<String>, desc: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: Some(desc.into()),
            properties: vec![],
            embedding: None,
        }
    }

    pub fn embedding(&self) -> Option<&[f32]> {
        self.embedding.as_deref()
    }

    /// Find a property of this entity by name.
    pub fn property(&self, name: &str) -> Option<&PropertySpec> {
        self.properties.iter().find(|p| p.name == name)
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
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct DomainOntology {
    #[serde(default)]
    pub entity_types: Vec<EntityTypeSpec>,
    #[serde(default)]
    pub relation_types: Vec<RelationTypeSpec>,
}

/// Catalog of domain ontologies. Serializes as a flat JSON object
/// (`{ "legal": {...}, "medical": {...} }`) — no extra wrapper key.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
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

    pub fn is_empty(&self) -> bool {
        self.domains.is_empty()
    }

    /// Borrow the full domain map.
    pub fn domains_view(&self) -> &BTreeMap<String, DomainOntology> {
        &self.domains
    }

    // ── Cross-domain entity / property / relation lookup ───────────────

    /// Look up an entity type by name in `domain`.
    pub fn get_entity_in(&self, domain: &str, name: &str) -> Option<&EntityTypeSpec> {
        self.domains
            .get(domain)?
            .entity_types
            .iter()
            .find(|e| e.name == name)
    }

    /// Look up an entity type by name across all domains, returning the
    /// first match (BTreeMap iteration order is deterministic on domain
    /// name). Returns `(domain, entity_type)`.
    pub fn get_entity(&self, name: &str) -> Option<(&str, &EntityTypeSpec)> {
        for (domain, ontology) in &self.domains {
            if let Some(spec) = ontology.entity_types.iter().find(|e| e.name == name) {
                return Some((domain.as_str(), spec));
            }
        }
        None
    }

    /// Look up a property by `(entity, name)` in `domain`.
    pub fn get_property_in(
        &self,
        domain: &str,
        entity: &str,
        property: &str,
    ) -> Option<&PropertySpec> {
        self.get_entity_in(domain, entity)?.property(property)
    }

    /// Look up a property by `(entity, name)` across all domains.
    pub fn get_property(&self, entity: &str, property: &str) -> Option<&PropertySpec> {
        self.get_entity(entity)?.1.property(property)
    }

    /// Look up a relation type by name in `domain`.
    pub fn get_relation_in(&self, domain: &str, name: &str) -> Option<&RelationTypeSpec> {
        self.domains
            .get(domain)?
            .relation_types
            .iter()
            .find(|r| r.name == name)
    }

    /// Look up a relation type by name across all domains.
    pub fn get_relation(&self, name: &str) -> Option<(&str, &RelationTypeSpec)> {
        for (domain, ontology) in &self.domains {
            if let Some(spec) = ontology.relation_types.iter().find(|r| r.name == name) {
                return Some((domain.as_str(), spec));
            }
        }
        None
    }

    /// Query-side type id for an `(entity, property)` pair, used by the
    /// DSL lowering to auto-resolve filter handlers (e.g. SemanticText).
    pub fn get_query_type(&self, entity: &str, property: &str) -> Option<&'static str> {
        self.get_property(entity, property)?
            .property_type
            .query_type_id()
    }

    // ── Mutation ───────────────────────────────────────────────────────

    /// Fill in `description` for an entity, leaving an existing
    /// description and embedding untouched. Creates the domain and entry
    /// when missing. Useful when bootstrapping a workspace catalog from
    /// pre-existing graph labels.
    pub fn insert_entity_description(&mut self, domain: &str, name: &str, description: &str) {
        let ontology = self.domains.entry(domain.to_string()).or_default();
        match ontology.entity_types.iter_mut().find(|e| e.name == name) {
            Some(existing) => {
                if existing.description.as_deref().unwrap_or("").is_empty() {
                    existing.description = Some(description.to_string());
                }
            }
            None => ontology.entity_types.push(EntityTypeSpec {
                name: name.to_string(),
                description: Some(description.to_string()),
                properties: Vec::new(),
                embedding: None,
            }),
        }
    }

    /// Merge entries from `other` into `self`. For each `(domain, name)`
    /// pair, the incoming description / properties / relation list win,
    /// but an existing embedding is kept when the incoming record has
    /// none — so a catalog merge after `find()` doesn't blow away the
    /// vectors that were already computed.
    pub fn merge(&mut self, other: &OntologyCatalog) {
        for (domain, incoming) in &other.domains {
            let target = self.domains.entry(domain.clone()).or_default();
            for spec in &incoming.entity_types {
                let preserved_embedding = target
                    .entity_types
                    .iter()
                    .find(|e| e.name == spec.name)
                    .and_then(|e| e.embedding.clone());
                target.entity_types.retain(|e| e.name != spec.name);
                let mut copy = spec.clone();
                if copy.embedding.is_none() {
                    copy.embedding = preserved_embedding;
                }
                target.entity_types.push(copy);
            }
            for rel in &incoming.relation_types {
                target.relation_types.retain(|r| r.name != rel.name);
                target.relation_types.push(rel.clone());
            }
        }
    }

    /// Recompute embeddings for every entity type whose `embedding` is
    /// `None`. Existing embeddings are preserved.
    pub fn compute(&mut self, embedder: &dyn Embedder) -> Result<(), EmbedError> {
        // Collect texts and their (domain, name) coordinates so we can
        // batch the embedder call once across the whole catalog.
        let mut jobs: Vec<(String, String, String)> = Vec::new();
        for (domain, ontology) in &self.domains {
            for entity in &ontology.entity_types {
                if entity.embedding.is_none() {
                    jobs.push((
                        domain.clone(),
                        entity.name.clone(),
                        entity_embedding_text(entity),
                    ));
                }
            }
        }
        if jobs.is_empty() {
            return Ok(());
        }
        let texts: Vec<&str> = jobs.iter().map(|(_, _, t)| t.as_str()).collect();
        let embeddings = embedder.embed_batch(&texts)?;
        if embeddings.len() != jobs.len() {
            return Err(EmbedError::Backend(format!(
                "embedder returned {} vectors for {} entity specs",
                embeddings.len(),
                jobs.len()
            )));
        }
        for ((domain, name, _), vec) in jobs.into_iter().zip(embeddings.into_iter()) {
            if let Some(ontology) = self.domains.get_mut(&domain) {
                if let Some(spec) = ontology.entity_types.iter_mut().find(|e| e.name == name) {
                    spec.embedding = Some(vec);
                }
            }
        }
        Ok(())
    }

    /// Semantic match: embed `text`, compare to every entity type's
    /// embedding, and return the ones above `threshold` (optionally
    /// rerank-filtered). Mirrors the historical `GraphSpecification::find`.
    pub fn find(
        &self,
        text: impl AsRef<str>,
        threshold: f32,
        embedder: &dyn Embedder,
        reranker: Option<&dyn Reranker>,
        reranking_threshold: f64,
    ) -> Result<Vec<EntityTypeMatch<'_>>, EmbedError> {
        let prompt = format!(
            "User query:{}\nTask: Identify database schema elements need for answering this query",
            text.as_ref()
        );
        let query = embedder.embed(prompt.as_str())?;
        let mut matches = Vec::new();
        for (domain, ontology) in &self.domains {
            for spec in &ontology.entity_types {
                let Some(embedding) = spec.embedding.as_deref() else {
                    continue;
                };
                if embedding.len() != query.len() {
                    return Err(EmbedError::Backend(format!(
                        "embedding dimension mismatch for entity '{}': entity vector has {}, query vector has {}",
                        spec.name,
                        embedding.len(),
                        query.len()
                    )));
                }
                let score = cosine_similarity(&query, embedding);
                if score >= threshold {
                    matches.push(EntityTypeMatch {
                        domain: domain.as_str(),
                        entity_type: spec,
                        score,
                    });
                }
            }
        }
        if let Some(reranker) = reranker {
            if !matches.is_empty() {
                let documents: Vec<String> = matches
                    .iter()
                    .map(|m| entity_embedding_text(m.entity_type))
                    .collect();
                let scores = reranker.rerank(prompt.as_str(), &documents)?;
                if scores.len() != matches.len() {
                    return Err(EmbedError::Backend(format!(
                        "reranker returned {} scores for {} entity specs",
                        scores.len(),
                        matches.len()
                    )));
                }
                matches = matches
                    .into_iter()
                    .zip(scores.into_iter())
                    .filter_map(|(mut m, score)| {
                        if score >= reranking_threshold {
                            m.score = score as f32;
                            Some(m)
                        } else {
                            None
                        }
                    })
                    .collect();
            }
        }
        matches.sort_by(|a, b| {
            b.score
                .total_cmp(&a.score)
                .then_with(|| a.entity_type.name.cmp(&b.entity_type.name))
        });
        Ok(matches)
    }

    /// Resolve descriptions and domain labels on a live [`GraphSchema`]
    /// in place. Node / relationship / property descriptions are pulled
    /// from this catalog using the node's `domain` label or one of its
    /// `extra_labels` (when one matches a known domain), and otherwise
    /// via a cross-domain first-match scan.
    pub fn enrich(&self, schema: &mut GraphSchema) {
        for node in &mut schema.nodes {
            // Pick a domain hint: an explicit `domain` first, then the
            // first `extra_label` that names a known domain in this
            // catalog (this is what the planner stamps at ingest time).
            let domain_hint = node.domain.clone().or_else(|| {
                node.extra_labels
                    .iter()
                    .find(|l| self.domains.contains_key(l.as_str()))
                    .cloned()
            });
            let (domain, spec) = self.resolve_entity_view(&node.label, domain_hint.as_deref());
            node.domain = domain;
            node.description = spec.and_then(|s| s.description.clone());
            for prop in &mut node.properties {
                prop.description = spec
                    .and_then(|s| s.property(&prop.name))
                    .and_then(|p| p.description.clone());
            }
        }
        for rel in &mut schema.relationships {
            let domain_hint = rel.domain.clone();
            let from = rel.from.clone();
            let to = rel.to.clone();
            let (domain, spec) = self.resolve_relation_view(
                &rel.label,
                domain_hint.as_deref(),
                from.as_deref(),
                to.as_deref(),
            );
            rel.domain = domain;
            rel.description = spec.and_then(|s| s.description.clone());
        }
    }

    fn resolve_entity_view(
        &self,
        label: &str,
        domain_hint: Option<&str>,
    ) -> (Option<String>, Option<&EntityTypeSpec>) {
        if let Some(domain) = domain_hint {
            if let Some(spec) = self.get_entity_in(domain, label) {
                return (Some(domain.to_string()), Some(spec));
            }
        }
        match self.get_entity(label) {
            Some((domain, spec)) => (Some(domain.to_string()), Some(spec)),
            None => (domain_hint.map(str::to_string), None),
        }
    }

    fn resolve_relation_view(
        &self,
        label: &str,
        domain_hint: Option<&str>,
        from: Option<&str>,
        to: Option<&str>,
    ) -> (Option<String>, Option<&RelationTypeSpec>) {
        if let Some(domain) = domain_hint {
            if let Some(spec) = self.get_relation_in(domain, label) {
                return (Some(domain.to_string()), Some(spec));
            }
        }
        // Try to use endpoint node labels as a hint: a relation usually
        // lives in the same domain as its endpoints.
        for endpoint in [from, to].into_iter().flatten() {
            if let Some((domain, _)) = self.get_entity(endpoint) {
                if let Some(spec) = self.get_relation_in(domain, label) {
                    return (Some(domain.to_string()), Some(spec));
                }
            }
        }
        match self.get_relation(label) {
            Some((domain, spec)) => (Some(domain.to_string()), Some(spec)),
            None => (domain_hint.map(str::to_string), None),
        }
    }
}

/// A semantic-search match returned by [`OntologyCatalog::find`].
#[derive(Debug, Clone, Copy)]
pub struct EntityTypeMatch<'a> {
    pub domain: &'a str,
    pub entity_type: &'a EntityTypeSpec,
    pub score: f32,
}

fn entity_embedding_text(spec: &EntityTypeSpec) -> String {
    let description = spec.description.as_deref().unwrap_or("");
    let mut out = format!("{} - {}\n", spec.name, description);
    if !spec.properties.is_empty() {
        let mut props: Vec<&PropertySpec> = spec.properties.iter().collect();
        props.sort_by(|a, b| a.name.cmp(&b.name));
        out.push_str("Properties:");
        for property in props {
            let desc = property.description.as_deref().unwrap_or("");
            out.push_str(&format!(
                "\n- {} ({:?}): {}",
                property.name, property.property_type, desc
            ));
        }
    }
    out
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut dot = 0.0f32;
    let mut a_norm = 0.0f32;
    let mut b_norm = 0.0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        a_norm += x * x;
        b_norm += y * y;
    }
    if a_norm == 0.0 || b_norm == 0.0 {
        0.0
    } else {
        dot / (a_norm.sqrt() * b_norm.sqrt())
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
        let entity_names: Vec<&str> = legal.entity_types.iter().map(|e| e.name.as_str()).collect();
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
            embedding: None,
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
        let err =
            OntologyCatalog::load_from_path(Path::new("/nonexistent/ontologies.json")).unwrap_err();
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
