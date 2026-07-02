//! Origin scope for graph entities.
//!
//! [`Scope`] tags an [`EntityGraph`](crate::graph::EntityGraph) with the
//! kind of source it was extracted from. The QA service that reads the
//! ingested graph uses this to pick a query strategy per entity type —
//! DSL/Cypher when properties are predictable
//! ([`Scope::Structured`]), embedding retrieval when they come from
//! free text ([`Scope::Text`]), or a hybrid path
//! ([`Scope::Table`]).
//!
//! A single entity can carry multiple scopes. When two entities from
//! different sources merge onto the same Memgraph node (by primary key
//! or by soft-merge), the planner emits `SET n:<scope_label>` for
//! every scope on the incoming entity; Cypher labels are sets, so the
//! union is automatic and idempotent across re-ingestion.
//!
//! Scopes round-trip through three representations:
//!
//! * **Rust**: the [`Scope`] enum.
//! * **JSON**: lowercase strings (`"text" | "table" | "structured"`),
//!   accepted by the [`from_json`](crate::graph::GraphBuilder::from_json)
//!   builder under the `scopes` field.
//! * **Cypher**: snake_case labels (`scope_text`, `scope_table`,
//!   `scope_structured`) attached alongside the entity's type label.

use serde::{Deserialize, Serialize};
use strum::VariantNames;

/// Origin of an [`EntityGraph`](crate::graph::EntityGraph): the kind of
/// source it was extracted from. See the [module docs](self) for
/// semantics and the round-trip representations.
///
/// The Cypher-label spelling (`scope_text`, …) lives in the `#[strum]`
/// attributes — the single source for both [`cypher_label`](Self::cypher_label)
/// and its inverse, with `serde` keeping the lowercase JSON form.
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    Hash,
    PartialOrd,
    Ord,
    Serialize,
    Deserialize,
    strum::EnumString,
    strum::IntoStaticStr,
    strum::VariantNames,
)]
#[serde(rename_all = "lowercase")]
pub enum Scope {
    /// Extracted from free-form markdown text (NER, LLM extraction).
    #[strum(serialize = "scope_text")]
    Text,
    /// Extracted from markdown tables.
    #[strum(serialize = "scope_table")]
    Table,
    /// Sourced from JSON, databases, or other structured input.
    #[strum(serialize = "scope_structured")]
    Structured,
}

impl Scope {
    /// Cypher label used to materialise this scope on a graph node.
    pub fn cypher_label(self) -> &'static str {
        self.into()
    }

    /// Inverse of [`cypher_label`]. Returns `None` for any label that
    /// isn't one of the three reserved scope identifiers.
    ///
    /// [`cypher_label`]: Self::cypher_label
    pub fn from_cypher_label(label: &str) -> Option<Self> {
        label.parse().ok()
    }
}

/// Every Cypher label reserved by [`Scope`], in enum-declaration order.
/// Consumers (e.g. introspection) can use this to partition raw label
/// strings into scope vs non-scope labels without re-deriving the set.
pub const SCOPE_LABELS: &[&str] = Scope::VARIANTS;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serde_round_trip_uses_lowercase() {
        assert_eq!(serde_json::to_string(&Scope::Text).unwrap(), "\"text\"");
        assert_eq!(serde_json::to_string(&Scope::Table).unwrap(), "\"table\"");
        assert_eq!(
            serde_json::to_string(&Scope::Structured).unwrap(),
            "\"structured\""
        );

        assert_eq!(
            serde_json::from_str::<Scope>("\"text\"").unwrap(),
            Scope::Text
        );
        assert_eq!(
            serde_json::from_str::<Scope>("\"table\"").unwrap(),
            Scope::Table
        );
        assert_eq!(
            serde_json::from_str::<Scope>("\"structured\"").unwrap(),
            Scope::Structured
        );
    }

    #[test]
    fn cypher_label_inverse() {
        for scope in [Scope::Text, Scope::Table, Scope::Structured] {
            assert_eq!(Scope::from_cypher_label(scope.cypher_label()), Some(scope));
        }
        assert_eq!(Scope::from_cypher_label("scope_unknown"), None);
        assert_eq!(Scope::from_cypher_label("Person"), None);
    }

    #[test]
    fn scope_labels_covers_every_variant() {
        // Defence against forgetting to update SCOPE_LABELS when a new
        // variant is added: every label must round-trip to Some(_).
        for label in SCOPE_LABELS {
            assert!(Scope::from_cypher_label(label).is_some(), "{label}");
        }
        assert_eq!(SCOPE_LABELS.len(), 3);
    }
}
