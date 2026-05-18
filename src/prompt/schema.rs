//! Graph schema description fed into the prompt generator.

use serde::{Deserialize, Serialize};

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

#[cfg(test)]
mod tests {
    use super::*;

    fn node(label: &str) -> NodeKind {
        NodeKind {
            label: label.to_string(),
            properties: Vec::new(),
        }
    }

    fn rel(label: &str, from: Option<&str>, to: Option<&str>) -> RelKind {
        RelKind {
            label: label.to_string(),
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
