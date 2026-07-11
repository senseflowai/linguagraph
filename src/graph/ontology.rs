//! Domain ontologies for knowledge-extraction prompts.
//!
//! An [`OntologyCatalog`] is a `domain → DomainOntology` map deserialized
//! from a flat JSON object. The catalog ships with a built-in legal
//! vocabulary (compiled in via [`include_str!`]) and can be replaced or
//! extended at runtime by loading a user-provided JSON file (path is
//! taken from `[prompt].ontologies_path` in the TOML config).

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::embeddings::{
    ensure_indexed, EmbedError, Embedder, EmbeddingFilter, EmbeddingIndex, EmbeddingKind,
    EmbeddingPayload,
};
use crate::prompt::{GraphSchema, Property};
use crate::types::BuiltinType;

/// Type of a property defined in the ontology.
///
/// This is the **single source of truth** for the property-type
/// vocabulary across the whole system: the types a user picks in the
/// ontology UI, the types the LLM emits during knowledge extraction, and
/// the types handed to the [`crate::graph::GraphBuilder`] when writing a
/// graph. The graph builder's [`crate::graph::PropertyType`] is a
/// re-export of this very enum, so the ontology layer and the storage
/// layer can never drift apart.
///
/// There are exactly six canonical types:
///
/// * `Keyword` — a plain string matched by standard Cypher operators
///   (`=`, `!=`, `<`, `>`, `=~`, `CONTAINS`, …); identifiers, codes,
///   statuses, categorical labels.
/// * `Text` — free-form text; always routed through the
///   `SemanticTextHandler` (embedded + vector-searchable).
/// * `Number` — an integer or a float.
/// * `Bool` — a boolean.
/// * `Datetime` — a calendar date or an instant.
/// * `List` — a JSON array.
///
/// Two string representations co-exist by design:
///
/// * **serde / prompt** — lowercase (`"keyword"`, `"number"`, …), the
///   JSON shape used in stored ontologies and prompt rendering.
/// * **mapping `type` names** — PascalCase (`"Text"`, `"Number"`, …),
///   parsed via the derived [`std::str::FromStr`].
///
/// The `#[strum]` / `#[serde]` aliases below are the single place that
/// records which historical spellings collapse onto each canonical
/// variant (e.g. `int` / `float` → `Number`, `string` → `Keyword`,
/// `semantictext` → `Text`, `date` / `timestamp` → `Datetime`). They keep
/// ontologies stored under the old vocabulary readable.
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    Hash,
    Serialize,
    Deserialize,
    strum::EnumString,
    strum::Display,
)]
#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
#[serde(rename_all = "lowercase")]
pub enum OntologyPropertyType {
    /// Plain string, standard Cypher matching (`=`, `<`, `>`, `=~`, …).
    /// `string` is the legacy serialized spelling, still accepted.
    #[strum(serialize = "Keyword", serialize = "String")]
    #[serde(alias = "Keyword", alias = "string", alias = "String")]
    Keyword,
    /// Free-form text, always embedded via the `SemanticText` handler.
    #[strum(serialize = "Text", serialize = "SemanticText")]
    #[serde(alias = "Text", alias = "semantictext", alias = "SemanticText")]
    Text,
    /// Numeric scalar (integer or float). `int` / `float` are legacy.
    #[strum(serialize = "Number", serialize = "Int", serialize = "Float")]
    #[serde(
        alias = "Number",
        alias = "int",
        alias = "float",
        alias = "Int",
        alias = "Float"
    )]
    Number,
    /// Boolean. `boolean` is the legacy spelling.
    #[strum(serialize = "Bool", serialize = "Boolean")]
    #[serde(alias = "Bool", alias = "boolean", alias = "Boolean")]
    Bool,
    /// Calendar date or instant. `date` / `timestamp` are legacy.
    #[strum(
        serialize = "Datetime",
        serialize = "DateTime",
        serialize = "Date",
        serialize = "Timestamp"
    )]
    #[serde(
        alias = "Datetime",
        alias = "date",
        alias = "timestamp",
        alias = "Date",
        alias = "DateTime",
        alias = "Timestamp"
    )]
    Datetime,
    /// JSON array.
    #[serde(alias = "List")]
    List,
}

impl OntologyPropertyType {
    /// Registry handler id used to ingest and store a value of this type.
    /// Sourced from [`BuiltinType`] so the registry vocabulary has a
    /// single definition. `List` has no dedicated handler and is stored
    /// plainly via the `Keyword` handler.
    pub fn handler_id(self) -> &'static str {
        match self {
            Self::Keyword | Self::List => BuiltinType::Keyword.id(),
            Self::Text => BuiltinType::SemanticText.id(),
            Self::Number => BuiltinType::Number.id(),
            Self::Bool => BuiltinType::Boolean.id(),
            Self::Datetime => BuiltinType::Timestamp.id(),
        }
    }

    /// Query-side type id: `Some(...)` for types whose DSL filters need
    /// a specialised handler (semantic search, timestamp interval
    /// lowering), `None` for plain comparisons.
    pub fn query_type_id(self) -> Option<&'static str> {
        match self {
            Self::Text => Some(BuiltinType::SemanticText.id()),
            Self::Keyword | Self::Datetime => Some(self.handler_id()),
            Self::List => Some(BuiltinType::List.id()),
            Self::Number | Self::Bool => None,
        }
    }
}

/// One typed property that an entity of a given type may carry.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
pub struct PropertySpec {
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub property_type: OntologyPropertyType,
    #[serde(default)]
    pub required: bool,
    /// Hand-declared closed vocabulary for an enum-like keyword field, in
    /// canonical (lowercase) form. Merged with the value set discovered
    /// by live introspection during schema projection (see
    /// [`OntologyCatalog::project_schema`]). Empty when the field is
    /// free-form or its vocabulary is left to introspection.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub allowed_values: Vec<String>,
}

/// One allowed entity type the LLM may emit.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
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

    /// Find a property of this entity by name.
    pub fn property(&self, name: &str) -> Option<&PropertySpec> {
        self.properties.iter().find(|p| p.name == name)
    }
}

/// One allowed relation type the LLM may emit.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
pub struct RelationTypeSpec {
    /// Canonical UPPER_SNAKE name (e.g. `GRANTS`).
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// Typed properties carried on relationships of this type (e.g.
    /// `ACTED_IN.roles`, `REVIEWED.rating`). Optional and empty by
    /// default — most relation types carry no properties, and filters on
    /// undeclared ones simply fall back to the untyped comparison path.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub properties: Vec<PropertySpec>,
}

impl RelationTypeSpec {
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

    /// Find a property of this relation type by name.
    pub fn property(&self, name: &str) -> Option<&PropertySpec> {
        self.properties.iter().find(|p| p.name == name)
    }
}

/// All allowed types for a single domain (e.g. `legal`, `medical`).
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
pub struct DomainOntology {
    /// Domain key from [`OntologyCatalog::domains`]. Hydrated by the
    /// catalog when loaded or inserted; skipped on save because the flat
    /// JSON catalog already stores this as the object key.
    #[serde(default, skip_serializing)]
    pub name: Option<String>,

    #[serde(default)]
    pub description: Option<String>,

    #[serde(default)]
    pub entity_types: Vec<EntityTypeSpec>,
    #[serde(default)]
    pub relation_types: Vec<RelationTypeSpec>,
}

impl DomainOntology {
    /// Domain key, when this ontology came from an [`OntologyCatalog`].
    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    /// Look up an entity type by name inside this domain.
    pub fn get_entity(&self, name: &str) -> Option<&EntityTypeSpec> {
        self.entity_types.iter().find(|e| e.name == name)
    }

    /// Look up a property by `(type, property)` inside this domain. `type`
    /// is checked against entity types first, then relation types — so a
    /// traversal's edge alias (whose "label" is the relationship type
    /// name, e.g. `ACTED_IN`) resolves `roles` via `RelationTypeSpec`.
    pub fn get_property(&self, entity: &str, property: &str) -> Option<&PropertySpec> {
        if let Some(p) = self.get_entity(entity).and_then(|e| e.property(property)) {
            return Some(p);
        }
        self.get_relation(entity)?.property(property)
    }

    /// Look up a relation type by name inside this domain.
    pub fn get_relation(&self, name: &str) -> Option<&RelationTypeSpec> {
        self.relation_types.iter().find(|r| r.name == name)
    }

    /// Query-side type id for an `(entity, property)` pair inside this domain.
    pub fn get_query_type(&self, entity: &str, property: &str) -> Option<&'static str> {
        self.get_property(entity, property)?
            .property_type
            .query_type_id()
    }
}

/// Catalog of domain ontologies. Serializes as a flat JSON object
/// (`{ "legal": {...}, "medical": {...} }`) — no extra wrapper key.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct OntologyCatalog {
    pub domains: BTreeMap<String, DomainOntology>,
}

/// Default cosine cutoff for domain selection.
pub const DEFAULT_DOMAIN_SELECTION_THRESHOLD: f32 = 0.45;
/// Default maximum number of domains returned by domain selection.
pub const DEFAULT_DOMAIN_SELECTION_TOP_K: usize = 3;

/// Embedded built-in catalog, compiled into the binary.
const BUILTIN_JSON: &str = include_str!("ontologies.default.json");

impl OntologyCatalog {
    /// Built-in catalog (currently: `legal` domain).
    ///
    /// # Panics
    /// Never — the embedded JSON is validated at compile time by the test
    /// suite. A panic here means a broken build artifact.
    pub fn builtin() -> Self {
        Self::load_from_str(BUILTIN_JSON).expect("builtin ontologies.default.json must parse")
    }

    pub fn load_from_str(raw: &str) -> Result<Self, OntologyError> {
        let mut catalog: Self = serde_json::from_str(raw)?;
        catalog.hydrate_domain_names();
        Ok(catalog)
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

    pub fn insert(&mut self, domain: impl Into<String>, mut ontology: DomainOntology) {
        let domain = domain.into();
        ontology.name = Some(domain.clone());
        self.domains.insert(domain, ontology);
    }

    pub fn is_empty(&self) -> bool {
        self.domains.is_empty()
    }

    /// Borrow the full domain map.
    pub fn domains_view(&self) -> &BTreeMap<String, DomainOntology> {
        &self.domains
    }

    /// Borrow every ontology domain in deterministic catalog order.
    pub fn all_domains(&self) -> Vec<&DomainOntology> {
        self.domains.values().collect()
    }

    pub(crate) fn hydrate_domain_names(&mut self) {
        for (name, ontology) in &mut self.domains {
            ontology.name = Some(name.clone());
        }
    }

    // ── Cross-domain entity / property / relation lookup ───────────────

    /// Look up an entity type by name in `domain`.
    pub fn get_entity_in(&self, domain: &str, name: &str) -> Option<&EntityTypeSpec> {
        self.domains.get(domain)?.get_entity(name)
    }

    /// Look up an entity type by name across all domains, returning the
    /// first match (BTreeMap iteration order is deterministic on domain
    /// name). Returns `(domain, entity_type)`.
    pub fn get_entity(&self, name: &str) -> Option<(&str, &EntityTypeSpec)> {
        for (domain, ontology) in &self.domains {
            if let Some(spec) = ontology.get_entity(name) {
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
        self.domains.get(domain)?.get_property(entity, property)
    }

    /// Look up a property by `(entity, name)` across all domains. `entity`
    /// is checked against entity types first, then relation types (see
    /// [`DomainOntology::get_property`]).
    pub fn get_property(&self, entity: &str, property: &str) -> Option<&PropertySpec> {
        if let Some(p) = self.get_entity(entity).and_then(|(_, e)| e.property(property)) {
            return Some(p);
        }
        self.get_relation(entity)?.1.property(property)
    }

    /// Look up a relation type by name in `domain`.
    pub fn get_relation_in(&self, domain: &str, name: &str) -> Option<&RelationTypeSpec> {
        self.domains.get(domain)?.get_relation(name)
    }

    /// Look up a relation type by name across all domains.
    pub fn get_relation(&self, name: &str) -> Option<(&str, &RelationTypeSpec)> {
        for (domain, ontology) in &self.domains {
            if let Some(spec) = ontology.get_relation(name) {
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
            }),
        }
    }

    /// Merge entries from `other` into `self`. For each `(domain, name)`
    /// pair, the incoming description / properties / relation list win.
    pub fn merge(&mut self, other: &OntologyCatalog) {
        for (domain, incoming) in &other.domains {
            let target = self.domains.entry(domain.clone()).or_default();
            target.name = Some(domain.clone());
            target.description = incoming.description.clone();
            for spec in &incoming.entity_types {
                target.entity_types.retain(|e| e.name != spec.name);
                target.entity_types.push(spec.clone());
            }
            for rel in &incoming.relation_types {
                target.relation_types.retain(|r| r.name != rel.name);
                target.relation_types.push(rel.clone());
            }
        }
    }

    pub fn split_schema_by_domain(schema: &GraphSchema) -> BTreeMap<String, GraphSchema> {
        let mut schemas = BTreeMap::<String, GraphSchema>::new();

        for node in &schema.nodes {
            let Some(domain) = node.domain.as_deref() else {
                continue;
            };
            schemas
                .entry(domain.to_string())
                .or_default()
                .nodes
                .push(node.clone());
        }

        for rel in &schema.relationships {
            let Some(domain) = rel.domain.as_deref() else {
                continue;
            };
            schemas
                .entry(domain.to_string())
                .or_default()
                .relationships
                .push(rel.clone());
        }

        schemas
    }

    /// Route `query` to its most relevant domains by embedding similarity,
    /// searched **server-side** through `index`.
    ///
    /// Domain passages are lazily embedded and upserted into the store (only
    /// the ones it is missing), then a single filtered vector search returns
    /// the top `top_k` domains scoring `>= threshold`, highest first.
    /// `candidates`, when set, restricts routing to those domains (the ones
    /// actually present in the live graph).
    pub async fn select_domains(
        &self,
        query: &str,
        threshold: f32,
        top_k: usize,
        candidates: Option<&BTreeSet<String>>,
        embedder: &dyn Embedder,
        index: &EmbeddingIndex<'_>,
    ) -> Result<Vec<DomainOntologyMatch<'_>>, EmbedError> {
        let passages: Vec<(EmbeddingPayload, String)> = self
            .domains
            .iter()
            .filter(|(domain, _)| candidates.is_none_or(|set| set.contains(*domain)))
            .map(|(domain, ontology)| {
                (
                    EmbeddingPayload::domain(domain.clone()),
                    domain_embedding_text(domain, ontology),
                )
            })
            .collect();
        if passages.is_empty() {
            return Ok(Vec::new());
        }

        ensure_indexed(index, embedder, &passages).await?;

        let query_embedding = embedder.embed(domain_query_text(query).as_str())?;
        let filter = EmbeddingFilter {
            kinds: vec![EmbeddingKind::Domain],
            domains: passages.iter().map(|(p, _)| p.domain.clone()).collect(),
        };
        let hits = index
            .store
            .search(
                index.collection,
                &query_embedding,
                top_k,
                Some(threshold),
                &filter,
            )
            .await?;

        let mut matches = Vec::new();
        for hit in hits {
            if let Some((key, ontology)) = self.domains.get_key_value(&hit.payload.domain) {
                matches.push(DomainOntologyMatch {
                    domain: key.as_str(),
                    ontology,
                    score: hit.score,
                });
            }
        }
        Ok(matches)
    }

    /// Project a live [`GraphSchema`] onto this catalog in place.
    ///
    /// The resulting schema contains only entity types, entity
    /// properties and relation types declared in the ontology. Retained
    /// entries receive ontology descriptions and domain labels. Property
    /// enum vocabularies discovered by introspection are merged with
    /// ontology-declared values.
    pub fn project_schema(schema: &mut GraphSchema, domains: Vec<&DomainOntology>) {
        let source_nodes = schema.nodes.clone();
        let source_relationships = schema.relationships.clone();
        let domain_names: BTreeSet<String> = domains
            .iter()
            .filter_map(|domain| domain.name().map(str::to_string))
            .collect();
        let mut projected_nodes = Vec::new();
        let mut kept_node_keys = BTreeSet::new();

        for domain in &domains {
            let Some(domain_name) = domain.name() else {
                continue;
            };
            for spec in &domain.entity_types {
                let Some(source_node) = source_nodes.iter().find(|node| {
                    node.label == spec.name && node_matches_domain(node, domain_name, &domain_names)
                }) else {
                    continue;
                };

                let mut node = source_node.clone();
                node.domain = Some(domain_name.to_string());
                node.description = spec.description.clone();
                retain_declared_properties(&mut node.properties, spec);

                if kept_node_keys.insert((domain_name.to_string(), node.label.clone())) {
                    projected_nodes.push(node);
                }
            }
        }

        let mut projected_relationships = Vec::new();
        let mut kept_relationship_keys = BTreeSet::new();

        for domain in &domains {
            let Some(domain_name) = domain.name() else {
                continue;
            };
            for spec in &domain.relation_types {
                for source_rel in source_relationships.iter().filter(|rel| {
                    rel.label == spec.name
                        && relationship_matches_domain(rel, domain_name)
                        && endpoint_is_kept_in_domain(
                            rel.from.as_deref(),
                            domain_name,
                            &kept_node_keys,
                        )
                        && endpoint_is_kept_in_domain(
                            rel.to.as_deref(),
                            domain_name,
                            &kept_node_keys,
                        )
                }) {
                    let mut rel = source_rel.clone();
                    rel.domain = Some(domain_name.to_string());
                    rel.description = spec.description.clone();
                    // Keep only the relationship properties the ontology
                    // declares for this relation type (e.g. `ACTED_IN.roles`,
                    // `REVIEWED.rating`), enriched with their descriptions —
                    // mirroring node-property projection. Introspected
                    // properties the ontology doesn't declare are dropped so
                    // strict projection doesn't leak internal fields.
                    retain_declared_rel_properties(&mut rel.properties, spec);

                    let key = (
                        domain_name.to_string(),
                        rel.label.clone(),
                        rel.from.clone(),
                        rel.to.clone(),
                    );
                    if kept_relationship_keys.insert(key) {
                        projected_relationships.push(rel);
                    }
                }
            }
        }

        schema.nodes = projected_nodes;
        schema.relationships = projected_relationships;
    }
}

fn node_matches_domain(
    node: &crate::prompt::NodeKind,
    domain_name: &str,
    known_domains: &BTreeSet<String>,
) -> bool {
    match node.domain.as_deref() {
        Some(domain) => domain == domain_name,
        None => {
            if node.extra_labels.iter().any(|label| label == domain_name) {
                return true;
            }
            !node
                .extra_labels
                .iter()
                .any(|label| known_domains.contains(label))
        }
    }
}

fn relationship_matches_domain(rel: &crate::prompt::RelKind, domain_name: &str) -> bool {
    rel.domain
        .as_deref()
        .map_or(true, |domain| domain == domain_name)
}

fn retain_declared_properties(props: &mut Vec<Property>, spec: &EntityTypeSpec) {
    props.retain_mut(|prop| {
        let Some(declared) = spec.property(&prop.name) else {
            return false;
        };
        prop.description = declared.description.clone();
        prop.allowed_values = merged_allowed_values(&prop.allowed_values, &declared.allowed_values);
        true
    });
}

fn retain_declared_rel_properties(props: &mut Vec<Property>, spec: &RelationTypeSpec) {
    props.retain_mut(|prop| {
        let Some(declared) = spec.property(&prop.name) else {
            return false;
        };
        prop.description = declared.description.clone();
        prop.allowed_values = merged_allowed_values(&prop.allowed_values, &declared.allowed_values);
        true
    });
}

fn merged_allowed_values(introspected: &[String], declared: &[String]) -> Vec<String> {
    let mut values: Vec<String> = introspected
        .iter()
        .chain(declared.iter())
        .map(|v| v.to_lowercase())
        .collect();
    values.sort();
    values.dedup();
    values
}

fn endpoint_is_kept_in_domain(
    endpoint: Option<&str>,
    domain_name: &str,
    kept_node_keys: &BTreeSet<(String, String)>,
) -> bool {
    match endpoint {
        Some(label) => kept_node_keys.contains(&(domain_name.to_string(), label.to_string())),
        None => true,
    }
}

/// A semantic-search match returned by [`OntologyCatalog::find`].
#[derive(Debug, Clone, Copy)]
pub struct EntityTypeMatch<'a> {
    pub domain: &'a str,
    pub entity_type: &'a EntityTypeSpec,
    pub score: f32,
}

/// A semantic-search domain match returned by
/// [`OntologyCatalog::select_domain_matches`].
#[derive(Debug, Clone, Copy)]
pub struct DomainOntologyMatch<'a> {
    pub domain: &'a str,
    pub ontology: &'a DomainOntology,
    pub score: f32,
}

fn domain_query_text(query: &str) -> String {
    format!("User query:{query}\nTask: Identify the most relevant ontology domains for this query")
}

fn domain_embedding_text(domain: &str, ontology: &DomainOntology) -> String {
    let mut out = format!("Domain: {domain}\n");
    if let Some(description) = ontology.description.as_deref().filter(|d| !d.is_empty()) {
        out.push_str("Description: ");
        out.push_str(description);
        out.push('\n');
    }
    if !ontology.entity_types.is_empty() {
        let mut entities: Vec<&EntityTypeSpec> = ontology.entity_types.iter().collect();
        entities.sort_by(|a, b| a.name.cmp(&b.name));
        out.push_str("Entities:");
        for entity in entities {
            out.push('\n');
            out.push_str("- ");
            out.push_str(entity_embedding_text(entity).trim());
        }
        out.push('\n');
    }
    if !ontology.relation_types.is_empty() {
        let mut relations: Vec<&RelationTypeSpec> = ontology.relation_types.iter().collect();
        relations.sort_by(|a, b| a.name.cmp(&b.name));
        out.push_str("Relations:");
        for relation in relations {
            out.push('\n');
            out.push_str("- ");
            out.push_str(&relation.name);
        }
        out.push('\n');
    }
    out
}

fn entity_embedding_text(spec: &EntityTypeSpec) -> String {
    let description = spec.description.as_deref().unwrap_or("");
    format!("{} - {}\n", spec.name, description)
    //let mut out = format!("{} - {}\n", spec.name, description);
    // if !spec.properties.is_empty() {
    //     let mut props: Vec<&PropertySpec> = spec.properties.iter().collect();
    //     props.sort_by(|a, b| a.name.cmp(&b.name));
    //     out.push_str("Properties:");
    //     for property in props {
    //         let desc = property.description.as_deref().unwrap_or("");
    //         out.push_str(&format!(
    //             "\n- {} ({:?}): {}",
    //             property.name, property.property_type, desc
    //         ));
    //     }
    // }
    //out
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
                name: None,
                description: None,
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
                    property_type: OntologyPropertyType::Keyword,
                    required: true,
                    allowed_values: Vec::new(),
                },
                PropertySpec {
                    name: "age".to_string(),
                    description: Some("Age in years.".to_string()),
                    property_type: OntologyPropertyType::Number,
                    required: false,
                    allowed_values: Vec::new(),
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
        let err =
            OntologyCatalog::load_from_path(Path::new("/nonexistent/ontologies.json")).unwrap_err();
        assert!(matches!(err, OntologyError::NotFound(_)));
    }

    #[test]
    fn project_schema_keeps_only_ontology_declared_schema_and_enriches_descriptions() {
        use crate::prompt::{NodeKind, Property, PropertyType, RelKind};

        let mut cat = OntologyCatalog::default();
        cat.insert(
            "shop",
            DomainOntology {
                name: None,
                description: None,
                entity_types: vec![
                    EntityTypeSpec {
                        name: "Order".into(),
                        description: Some("A customer order".into()),
                        properties: vec![PropertySpec {
                            name: "status".into(),
                            description: Some("lifecycle state".into()),
                            property_type: OntologyPropertyType::Keyword,
                            required: false,
                            // Hand-declared value present only in the ontology.
                            allowed_values: vec!["Refunded".into()],
                        }],
                    },
                    EntityTypeSpec {
                        name: "Customer".into(),
                        description: Some("A buyer".into()),
                        properties: vec![PropertySpec {
                            name: "name".into(),
                            description: Some("Customer display name".into()),
                            property_type: OntologyPropertyType::Text,
                            required: true,
                            allowed_values: Vec::new(),
                        }],
                    },
                ],
                relation_types: vec![
                    RelationTypeSpec::with_description("PLACED", "Customer placed an order"),
                    RelationTypeSpec::with_description("AUDITED", "Audit relation"),
                ],
            },
        );
        cat.insert(
            "debug",
            DomainOntology {
                name: None,
                description: None,
                entity_types: vec![EntityTypeSpec {
                    name: "DebugNode".into(),
                    description: Some("Debug-only node".into()),
                    properties: Vec::new(),
                }],
                relation_types: vec![RelationTypeSpec::with_description(
                    "AUDITED",
                    "Audit relation",
                )],
            },
        );

        let mut schema = GraphSchema {
            nodes: vec![
                NodeKind {
                    label: "Order".into(),
                    domain: None,
                    extra_labels: vec!["shop".into()],
                    scopes: Vec::new(),
                    description: None,
                    properties: vec![
                        Property {
                            name: "status".into(),
                            ty: PropertyType::String,
                            description: None,
                            // Introspected values.
                            allowed_values: vec!["pending".into(), "completed".into()],
                        },
                        Property {
                            name: "internal_note".into(),
                            ty: PropertyType::String,
                            description: Some("not from ontology".into()),
                            allowed_values: Vec::new(),
                        },
                    ],
                },
                NodeKind {
                    label: "Customer".into(),
                    domain: None,
                    extra_labels: vec!["shop".into()],
                    scopes: Vec::new(),
                    description: None,
                    properties: vec![Property {
                        name: "name".into(),
                        ty: PropertyType::String,
                        description: None,
                        allowed_values: Vec::new(),
                    }],
                },
                NodeKind {
                    label: "DebugNode".into(),
                    domain: None,
                    extra_labels: vec!["debug".into()],
                    scopes: Vec::new(),
                    description: Some("live only".into()),
                    properties: Vec::new(),
                },
            ],
            relationships: vec![
                RelKind {
                    label: "PLACED".into(),
                    domain: None,
                    description: None,
                    from: Some("Customer".into()),
                    to: Some("Order".into()),
                    properties: vec![Property {
                        name: "since".into(),
                        ty: PropertyType::Date,
                        description: None,
                        allowed_values: Vec::new(),
                    }],
                },
                RelKind {
                    label: "IGNORED".into(),
                    domain: None,
                    description: None,
                    from: Some("Customer".into()),
                    to: Some("Order".into()),
                    properties: Vec::new(),
                },
                RelKind {
                    label: "AUDITED".into(),
                    domain: None,
                    description: None,
                    from: Some("Order".into()),
                    to: Some("DebugNode".into()),
                    properties: Vec::new(),
                },
            ],
        };

        let selected_domains = vec![cat.get("shop").unwrap()];
        OntologyCatalog::project_schema(&mut schema, selected_domains);

        let node_labels: Vec<&str> = schema
            .nodes
            .iter()
            .map(|node| node.label.as_str())
            .collect();
        assert_eq!(node_labels, vec!["Order", "Customer"]);
        assert_eq!(
            schema.nodes[0].description.as_deref(),
            Some("A customer order")
        );
        assert_eq!(schema.nodes[0].properties.len(), 1);
        let prop = &schema.nodes[0].properties[0];
        // Description resolved and the two sources unioned + lowercased + sorted.
        assert_eq!(prop.name, "status");
        assert_eq!(prop.description.as_deref(), Some("lifecycle state"));
        assert_eq!(
            prop.allowed_values,
            vec![
                "completed".to_string(),
                "pending".to_string(),
                "refunded".to_string()
            ]
        );

        assert_eq!(schema.nodes[1].description.as_deref(), Some("A buyer"));
        assert_eq!(schema.nodes[1].properties.len(), 1);
        assert_eq!(
            schema.nodes[1].properties[0].description.as_deref(),
            Some("Customer display name")
        );

        assert_eq!(schema.relationships.len(), 1);
        let rel = &schema.relationships[0];
        assert_eq!(rel.label, "PLACED");
        assert_eq!(rel.description.as_deref(), Some("Customer placed an order"));
        assert!(rel.properties.is_empty());
    }

    #[test]
    fn project_schema_keeps_relationships_inside_their_projected_domain() {
        use crate::prompt::{NodeKind, RelKind};

        let mut cat = OntologyCatalog::default();
        cat.insert(
            "core",
            DomainOntology {
                name: None,
                description: None,
                entity_types: vec![
                    EntityTypeSpec::new("Person"),
                    EntityTypeSpec::new("Location"),
                ],
                relation_types: vec![RelationTypeSpec::with_description(
                    "LOCATED_AT",
                    "Generic location relation",
                )],
            },
        );
        cat.insert(
            "camera",
            DomainOntology {
                name: None,
                description: None,
                entity_types: vec![EntityTypeSpec::new("Camera"), EntityTypeSpec::new("Place")],
                relation_types: vec![RelationTypeSpec::new("LOCATED_AT")],
            },
        );

        let mut schema = GraphSchema {
            nodes: vec![
                NodeKind {
                    label: "Camera".into(),
                    domain: None,
                    extra_labels: vec!["camera".into()],
                    scopes: Vec::new(),
                    description: None,
                    properties: Vec::new(),
                },
                NodeKind {
                    label: "Place".into(),
                    domain: None,
                    extra_labels: vec!["camera".into()],
                    scopes: Vec::new(),
                    description: None,
                    properties: Vec::new(),
                },
            ],
            relationships: vec![RelKind {
                label: "LOCATED_AT".into(),
                domain: None,
                description: None,
                from: Some("Camera".into()),
                to: Some("Place".into()),
                properties: Vec::new(),
            }],
        };

        OntologyCatalog::project_schema(&mut schema, cat.all_domains());

        assert_eq!(schema.relationships.len(), 1);
        let rel = &schema.relationships[0];
        assert_eq!(rel.label, "LOCATED_AT");
        assert_eq!(rel.domain.as_deref(), Some("camera"));
        assert_eq!(rel.description, None);
    }

    #[test]
    fn project_schema_keeps_unscoped_nodes_for_e2e_prefix_labels() {
        use crate::prompt::{NodeKind, RelKind};

        let mut cat = OntologyCatalog::default();
        cat.insert(
            "camera_domain",
            DomainOntology {
                name: None,
                description: None,
                entity_types: vec![EntityTypeSpec::new("Camera"), EntityTypeSpec::new("Place")],
                relation_types: vec![RelationTypeSpec::with_description(
                    "LOCATED_AT",
                    "Camera to place",
                )],
            },
        );

        let mut schema = GraphSchema {
            nodes: vec![
                NodeKind {
                    label: "Camera".into(),
                    domain: None,
                    extra_labels: vec!["E2E_CAMERAS_BASIC".into()],
                    scopes: Vec::new(),
                    description: None,
                    properties: Vec::new(),
                },
                NodeKind {
                    label: "Place".into(),
                    domain: None,
                    extra_labels: vec!["E2E_CAMERAS_BASIC".into()],
                    scopes: Vec::new(),
                    description: None,
                    properties: Vec::new(),
                },
            ],
            relationships: vec![RelKind {
                label: "LOCATED_AT".into(),
                domain: None,
                description: None,
                from: Some("Camera".into()),
                to: Some("Place".into()),
                properties: Vec::new(),
            }],
        };

        OntologyCatalog::project_schema(&mut schema, cat.all_domains());

        let labels: Vec<&str> = schema
            .nodes
            .iter()
            .map(|node| node.label.as_str())
            .collect();
        assert_eq!(labels, vec!["Camera", "Place"]);
        assert_eq!(schema.relationships.len(), 1);
        assert_eq!(schema.relationships[0].label, "LOCATED_AT");
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
