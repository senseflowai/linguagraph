//! Tenant scoping token for the ontology + graph subsystem.
//!
//! A [`Namespace`] is the single place that owns the naming conventions
//! that used to live, by informal agreement, in the *consuming* service
//! (senseflow's `linguagraph_bridge`): the Cypher label every node in a
//! tenant carries, the index/collection prefix, and — via the ontology
//! store — the Qdrant collection its schema and routing embeddings live
//! in. Making these derivations a first-class linguagraph type means the
//! contract (schema ↔ embeddings ↔ graph prefixes) is enforced by one
//! owner instead of kept in sync across a repo boundary.
//!
//! The token is opaque to linguagraph. A multi-tenant host passes
//! something like `"ws_1"`; a single-tenant CLI can use any stable label.

use std::fmt;

/// An opaque tenant scope. Cheap to clone.
///
/// The derivations are pure functions of the token, chosen to match the
/// labels senseflow already ingests under (`"ws_1"` →
/// `prefix_label = "Entity_ws_1"`, `prefix_index = "ws_1"`), so adopting
/// this type is a drop-in for the existing convention.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Namespace {
    token: String,
}

impl Namespace {
    /// Wrap a scope token. The token must be a valid Cypher identifier
    /// fragment (it is interpolated into `prefix_label`) — callers scope
    /// per tenant, e.g. `Namespace::new(format!("ws_{workspace_id}"))`.
    pub fn new(token: impl Into<String>) -> Self {
        Self {
            token: token.into(),
        }
    }

    /// The raw scope token (e.g. `"ws_1"`).
    pub fn token(&self) -> &str {
        &self.token
    }

    /// Cypher label appended to every node ingested under this scope, so
    /// tenants never collide and can be filtered cheaply at query time.
    /// `"ws_1"` → `"Entity_ws_1"`.
    pub fn prefix_label(&self) -> String {
        format!("Entity_{}", self.token)
    }

    /// Prefix folded into index / vector-collection names for this scope.
    /// `"ws_1"` → `"ws_1"` (the token itself; a distinct method so callers
    /// read as intent, not as "reuse the token string").
    pub fn prefix_index(&self) -> String {
        self.token.clone()
    }

    /// Qdrant collection holding this scope's ontology — **both** the
    /// authoritative schema points and the routing embeddings, in one
    /// place (that co-location is the whole point of the single-store
    /// design). `"ws_1"` → `"onto__ws_1"`. The single authority for this
    /// name: the schema store and the embedding index both derive it from
    /// here, so they can never disagree.
    pub fn ontology_collection(&self) -> String {
        format!("onto__{}", self.token)
    }
}

impl fmt::Display for Namespace {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.token)
    }
}

impl From<&str> for Namespace {
    fn from(s: &str) -> Self {
        Namespace::new(s)
    }
}

impl From<String> for Namespace {
    fn from(s: String) -> Self {
        Namespace::new(s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn derivations_match_the_existing_convention() {
        let ns = Namespace::new("ws_1");
        assert_eq!(ns.token(), "ws_1");
        assert_eq!(ns.prefix_label(), "Entity_ws_1");
        assert_eq!(ns.prefix_index(), "ws_1");
        assert_eq!(ns.ontology_collection(), "onto__ws_1");
        assert_eq!(ns.to_string(), "ws_1");
    }

    #[test]
    fn is_cloneable_and_comparable() {
        let a = Namespace::from("ws_2");
        let b = a.clone();
        assert_eq!(a, b);
    }
}
