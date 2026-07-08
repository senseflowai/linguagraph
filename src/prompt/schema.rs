//! Graph schema description fed into the prompt generator.

use serde::{Deserialize, Serialize};

use crate::graph::Scope;

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GraphSchema {
    pub nodes: Vec<NodeKind>,
    pub relationships: Vec<RelKind>,
}

impl GraphSchema {
    /// Drop node labels matching any filter by substring, plus every
    /// relationship whose known endpoints reference those labels.
    pub fn filter_node_labels_containing<S: AsRef<str>>(mut self, filter: &[S]) -> Self {
        if filter.is_empty() {
            return self;
        }

        self.nodes
            .retain(|node| !matches_label_filter(&node.label, filter));
        self.relationships.retain(|rel| {
            !rel.from
                .as_deref()
                .is_some_and(|label| matches_label_filter(label, filter))
                && !rel
                    .to
                    .as_deref()
                    .is_some_and(|label| matches_label_filter(label, filter))
        });
        self
    }
}

fn matches_label_filter<S: AsRef<str>>(label: &str, filter: &[S]) -> bool {
    filter.iter().any(|needle| label.contains(needle.as_ref()))
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeKind {
    pub label: String,
    /// Domain label resolved against an `OntologyCatalog` during
    /// enrichment. Set when one of `extra_labels` matches a known
    /// domain name.
    #[serde(default)]
    pub domain: Option<String>,
    /// All Cypher labels seen on sample nodes of this kind, minus
    /// `label`. Carries prefix (tenant) labels, domain labels, scope
    /// labels (`scope_text`, …) and any other labels stamped at
    /// ingestion. Consumed by enrichment; callers normally read
    /// `domain` / `scopes` instead.
    #[serde(default)]
    pub extra_labels: Vec<String>,
    /// Origin [`Scope`]s seen on sample nodes of this kind. Filled by
    /// introspection from the subset of `extra_labels` that decode as
    /// recognised scope labels. An empty list means the entity type
    /// was never stamped with a scope — QA consumers should treat
    /// this as "scope unknown" and consider the type visible to all
    /// query strategies.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub scopes: Vec<Scope>,
    /// Description resolved from an `OntologyCatalog`, when one is
    /// attached to the pipeline.
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub properties: Vec<Property>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelKind {
    pub label: String,
    /// Domain of the relation, inferred from its endpoint nodes or set
    /// explicitly during enrichment.
    #[serde(default)]
    pub domain: Option<String>,
    /// Description resolved from an `OntologyCatalog`.
    #[serde(default)]
    pub description: Option<String>,
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
    /// Description resolved from an `OntologyCatalog`.
    #[serde(default)]
    pub description: Option<String>,
    /// Closed set of allowed values for an enum-like keyword field, in
    /// canonical (lowercase) form. Populated by introspection when the
    /// field's cardinality is low enough (see
    /// [`IntrospectOptions::enum_cardinality_cap`](crate::db::IntrospectOptions)),
    /// and/or declared in an ontology `PropertySpec`. Empty for
    /// high-cardinality fields (ids, codes, VINs) and non-keyword types.
    /// When non-empty, the prompt renders an `enum` marker on the field
    /// and lists the values in a dedicated enumerations block.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub allowed_values: Vec<String>,
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

#[cfg(test)]
mod tests {
    use super::*;

    fn node(label: &str) -> NodeKind {
        NodeKind {
            label: label.to_string(),
            domain: None,
            extra_labels: Vec::new(),
            scopes: Vec::new(),
            description: None,
            properties: Vec::new(),
        }
    }

    fn rel(label: &str, from: Option<&str>, to: Option<&str>) -> RelKind {
        RelKind {
            label: label.to_string(),
            domain: None,
            description: None,
            from: from.map(str::to_string),
            to: to.map(str::to_string),
            properties: Vec::new(),
        }
    }

    #[test]
    fn label_filter_matches_by_contains_and_drops_participating_relationships() {
        let schema = GraphSchema {
            nodes: vec![node("TenantUser"), node("Company")],
            relationships: vec![
                rel("WORKS_AT", Some("TenantUser"), Some("Company")),
                rel("OWNS", Some("Company"), Some("Asset")),
            ],
        };

        let filtered = schema.filter_node_labels_containing(&["User"]);

        assert_eq!(filtered.nodes.len(), 1);
        assert_eq!(filtered.nodes[0].label, "Company");
        assert_eq!(filtered.relationships.len(), 1);
        assert_eq!(filtered.relationships[0].label, "OWNS");
    }

    #[test]
    fn label_filter_keeps_relationships_without_known_filtered_endpoint() {
        let schema = GraphSchema {
            nodes: vec![node("Person")],
            relationships: vec![
                rel("UNKNOWN", None, None),
                rel("SELF", Some("Person"), Some("Person")),
            ],
        };

        let filtered = schema.filter_node_labels_containing(&["Person"]);

        assert!(filtered.nodes.is_empty());
        assert_eq!(filtered.relationships.len(), 1);
        assert_eq!(filtered.relationships[0].label, "UNKNOWN");
    }
}
