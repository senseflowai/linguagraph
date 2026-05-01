//! Graph schema description fed into the prompt generator.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GraphSchema {
    pub nodes: Vec<NodeKind>,
    pub relationships: Vec<RelKind>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeKind {
    pub label: String,
    #[serde(default)]
    pub properties: Vec<Property>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelKind {
    pub label: String,
    #[serde(default)]
    pub from: Option<String>,
    #[serde(default)]
    pub to: Option<String>,
    #[serde(default)]
    pub properties: Vec<Property>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Property {
    pub name: String,
    pub ty: PropertyType,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum PropertyType {
    String,
    Int,
    Float,
    Bool,
    Date,
    Datetime,
    List,
}
