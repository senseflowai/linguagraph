//! Transport-agnostic request / response types.
//!
//! Everything the [`GraphService`](super::GraphService) accepts or returns
//! is here: plain `serde` structs with no dependency on any web framework.
//! Each derives [`utoipa::ToSchema`] under the `utoipa` feature so a
//! downstream HTTP crate gets an OpenAPI document for free.

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::db::NodeType;

/// A natural-language query, optionally scoped to a tenant / dataset.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
pub struct AskRequest {
    /// The plain-language question to answer.
    pub question: String,
    /// Optional Cypher prefix label scoping the query to one tenant /
    /// dataset (must match the label used at ingest time).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prefix_label: Option<String>,
    /// Optional embedding-index prefix, paired with `prefix_label`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prefix_index: Option<String>,
    /// Optional row limit override.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub limit: Option<u32>,
}

/// An entity/relationship graph plus the Cypher that produced it — the
/// shape a `{nodes, edges}` UI renders.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
pub struct GraphView {
    pub nodes: Vec<GraphNode>,
    pub edges: Vec<GraphEdge>,
    /// The generated Cypher query (for a "how was this answered?" panel).
    pub cypher: String,
}

/// One entity in a [`GraphView`].
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
pub struct GraphNode {
    /// Database internal id, as a string. Stable within a graph; usable
    /// as the argument to the entity-detail lookup.
    pub id: String,
    /// Whether this is a user entity, a chunk, or a source document.
    pub kind: NodeType,
    /// Primary Cypher label (the entity type).
    pub label: String,
    /// Best-effort display name (`name` / `title` property when present).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// All node properties.
    #[cfg_attr(feature = "utoipa", schema(value_type = Object))]
    pub properties: Map<String, Value>,
    /// Source documents reachable from this node via `mention` / `part_of`.
    #[cfg_attr(feature = "utoipa", schema(value_type = Vec<Object>))]
    pub sources: Vec<Value>,
}

/// One relationship in a [`GraphView`].
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
pub struct GraphEdge {
    pub id: String,
    /// Id of the source node (relationship start).
    pub from: String,
    /// Id of the target node (relationship end).
    pub to: String,
    /// Relationship type (e.g. `OWNS`).
    pub rel: String,
    #[cfg_attr(feature = "utoipa", schema(value_type = Object))]
    pub properties: Map<String, Value>,
}

/// The graph schema, reshaped for a UI's entity-type filter and legend.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
pub struct SchemaView {
    pub entity_types: Vec<EntityTypeInfo>,
    pub relation_types: Vec<RelationTypeInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
pub struct EntityTypeInfo {
    pub label: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub properties: Vec<PropertyInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
pub struct PropertyInfo {
    pub name: String,
    /// Property type, lower-cased (e.g. `string`, `int`, `datetime`).
    pub ty: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
pub struct RelationTypeInfo {
    pub label: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub from: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub to: Option<String>,
}

/// A single entity with its properties, sources, and relationships.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
pub struct EntityDetail {
    pub id: String,
    pub kind: NodeType,
    pub label: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[cfg_attr(feature = "utoipa", schema(value_type = Object))]
    pub properties: Map<String, Value>,
    #[cfg_attr(feature = "utoipa", schema(value_type = Vec<Object>))]
    pub sources: Vec<Value>,
    pub relations: Vec<RelationSummary>,
}

/// One relationship as seen from an entity's perspective.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
pub struct RelationSummary {
    pub id: String,
    pub rel: String,
    pub from: String,
    pub to: String,
    /// The node on the other end of this relationship.
    pub other_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub other_label: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub other_name: Option<String>,
    #[cfg_attr(feature = "utoipa", schema(value_type = Object))]
    pub properties: Map<String, Value>,
}

/// A single relationship with both endpoints fully described.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
pub struct RelationDetail {
    pub id: String,
    pub rel: String,
    pub from: Endpoint,
    pub to: Endpoint,
    #[cfg_attr(feature = "utoipa", schema(value_type = Object))]
    pub properties: Map<String, Value>,
}

/// One end of a relationship.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
pub struct Endpoint {
    pub id: String,
    pub kind: NodeType,
    pub label: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[cfg_attr(feature = "utoipa", schema(value_type = Object))]
    pub properties: Map<String, Value>,
}
