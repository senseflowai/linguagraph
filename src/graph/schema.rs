use serde_json::Value;
use std::collections::{BTreeSet, HashMap};

use crate::graph::scope::Scope;

/// Storage-shape tag carried by an ingested [`Property`].
///
/// Parsed from a mapping document's PascalCase `type` name via the
/// derived [`std::str::FromStr`]; the `#[strum]` aliases are the single
/// place recording which spellings collapse onto each variant (e.g.
/// `Date` / `Timestamp`). The ingest planner maps these to registry
/// handler ids in [`crate::ingest`].
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize,
    strum::EnumString, strum::Display,
)]
pub enum PropertyType {
    /// Plain string, standard Cypher matching. `String` is the legacy
    /// spelling, still accepted on input.
    #[strum(serialize = "Keyword", serialize = "String")]
    #[serde(alias = "String")]
    Keyword,
    /// Free-form text; always handled as `SemanticText` (embedded).
    #[strum(serialize = "Text", serialize = "SemanticText")]
    #[serde(alias = "SemanticText")]
    Text,
    Number,
    Boolean,
    DateTime,
    #[strum(serialize = "Timestamp", serialize = "Date")]
    Timestamp,
}

impl PropertyType {
    /// Registry handler id this storage type resolves to. This is the
    /// single mapping from the mapping/storage vocabulary to the type
    /// registry: `Keyword`→`Keyword`, `Text`→`SemanticText` (semantic),
    /// numbers/bools/dates to their scalar handlers.
    pub fn handler_id(self) -> &'static str {
        use crate::types::BuiltinType;
        match self {
            Self::Keyword => BuiltinType::Keyword.id(),
            Self::Text => BuiltinType::SemanticText.id(),
            Self::Number => BuiltinType::Number.id(),
            Self::Boolean => BuiltinType::Boolean.id(),
            Self::DateTime | Self::Timestamp => BuiltinType::Timestamp.id(),
        }
    }
}

/// Canonicalize a property-type spelling from the mapping / DSL vocabulary
/// to a registry handler id. This is the single trust-boundary translation
/// that lets the contract names (`Keyword`, `Text`) and their legacy
/// aliases (`String`, `SemanticText`, `Int`, `Date`, …) all resolve to the
/// right registered handler. Spellings that aren't built-in property types
/// (e.g. a custom registered type) pass through unchanged.
pub fn canonical_handler_id(raw: &str) -> String {
    use std::str::FromStr;
    PropertyType::from_str(raw)
        .map(|pt| pt.handler_id().to_string())
        .unwrap_or_else(|_| raw.to_string())
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum PrimaryKey {
    Strict(String),
    Soft,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Property {
    pub name: String,
    pub property_type: PropertyType,
    pub value: Value,
}

impl Property {
    pub fn new(
        name: impl Into<String>,
        property_type: PropertyType,
        value: impl Into<Value>,
    ) -> Self {
        Self {
            name: name.into(),
            property_type,
            value: value.into(),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct EntityGraph {
    pub r#type: String,
    pub labels: Vec<String>,
    /// Optional ontology domain ("legal", "medical", ...). When set, the
    /// planner emits an extra Cypher label so live schema introspection
    /// can later resolve descriptions for this node from the catalog.
    pub domain: Option<String>,
    /// Origins this entity was extracted from. See [`Scope`]. Multiple
    /// scopes accumulate naturally when entities from different sources
    /// merge onto the same Memgraph node — every scope is materialised
    /// as an extra Cypher label by the planner, and Cypher labels are
    /// idempotent sets, so the union is automatic.
    pub scopes: BTreeSet<Scope>,
    pub primary_key: Option<PrimaryKey>,
    pub properties: HashMap<String, Property>,
}

impl EntityGraph {
    pub fn new(r#type: impl Into<String>) -> Self {
        Self {
            r#type: r#type.into(),
            labels: Vec::new(),
            domain: None,
            scopes: BTreeSet::new(),
            primary_key: None,
            properties: HashMap::new(),
        }
    }

    pub fn label(mut self, label: impl Into<String>) -> Self {
        self.labels.push(label.into());
        self
    }

    pub fn labels(mut self, labels: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.labels.extend(labels.into_iter().map(Into::into));
        self
    }

    pub fn domain(mut self, domain: impl Into<String>) -> Self {
        self.domain = Some(domain.into());
        self
    }

    /// Tag this entity with a single origin [`Scope`]. Repeated calls
    /// accumulate (the underlying set deduplicates), so chaining
    /// `.scope(Text).scope(Table)` is equivalent to `.scopes([Text,
    /// Table])`.
    pub fn scope(mut self, scope: Scope) -> Self {
        self.scopes.insert(scope);
        self
    }

    /// Tag this entity with multiple origin [`Scope`]s in one call.
    pub fn scopes(mut self, scopes: impl IntoIterator<Item = Scope>) -> Self {
        self.scopes.extend(scopes);
        self
    }

    /// Check whether this entity carries a given [`Scope`].
    pub fn has_scope(&self, scope: Scope) -> bool {
        self.scopes.contains(&scope)
    }

    pub fn primary_key(mut self, primary_key: PrimaryKey) -> Self {
        self.primary_key = Some(primary_key);
        self
    }

    pub fn strict_primary_key(self, field: impl Into<String>) -> Self {
        self.primary_key(PrimaryKey::Strict(field.into()))
    }

    pub fn soft_primary_key(self) -> Self {
        self.primary_key(PrimaryKey::Soft)
    }

    pub fn property(
        mut self,
        name: impl Into<String>,
        property_type: PropertyType,
        value: impl Into<Value>,
    ) -> Self {
        let property = Property::new(name, property_type, value);
        self.properties.insert(property.name.clone(), property);
        self
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct RelationGraph {
    pub r#type: String,
    pub from: crate::graph::EntityRef,
    pub to: crate::graph::EntityRef,
    pub properties: HashMap<String, Property>,
}

impl RelationGraph {
    pub fn new(
        from: crate::graph::EntityRef,
        r#type: impl Into<String>,
        to: crate::graph::EntityRef,
    ) -> Self {
        Self {
            r#type: r#type.into(),
            from,
            to,
            properties: HashMap::new(),
        }
    }

    pub fn property(
        mut self,
        name: impl Into<String>,
        property_type: PropertyType,
        value: impl Into<Value>,
    ) -> Self {
        let property = Property::new(name, property_type, value);
        self.properties.insert(property.name.clone(), property);
        self
    }
}
