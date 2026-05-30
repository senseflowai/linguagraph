use serde_json::Value;
use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum PropertyType {
    String,
    Text,
    Number,
    Boolean,
    DateTime,
    Timestamp,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum PrimaryKey {
    Strict(String),
    Soft(String),
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
    pub primary_key: Option<PrimaryKey>,
    pub properties: HashMap<String, Property>,
}

impl EntityGraph {
    pub fn new(r#type: impl Into<String>) -> Self {
        Self {
            r#type: r#type.into(),
            labels: Vec::new(),
            domain: None,
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

    pub fn primary_key(mut self, primary_key: PrimaryKey) -> Self {
        self.primary_key = Some(primary_key);
        self
    }

    pub fn strict_primary_key(self, field: impl Into<String>) -> Self {
        self.primary_key(PrimaryKey::Strict(field.into()))
    }

    pub fn soft_primary_key(self, field: impl Into<String>) -> Self {
        self.primary_key(PrimaryKey::Soft(field.into()))
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
