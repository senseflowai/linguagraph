//! Ontology management subsystem — one owner for schema **and** its
//! embeddings.
//!
//! Historically the ontology was split across a consumer's relational
//! store (the authoritative schema) and linguagraph's Qdrant index (the
//! routing embeddings), kept in sync by hand. This module collapses that:
//! it owns schema persistence ([`OntologyStore`]) and, in the Qdrant
//! backend, stores the schema **in the same collection** as its
//! embeddings, so there is nothing to synchronise.
//!
//! Scoping is a first-class [`Namespace`] — the naming conventions
//! (Cypher label, index/collection prefix) that used to live by informal
//! agreement in the consuming service now belong here.
//!
//! The domain *types* ([`DomainOntology`] &c.) currently live in
//! [`crate::graph`] and are re-exported here, so consumers can depend on
//! the stable `linguagraph::ontology::*` path regardless of where the
//! types physically sit — a later refactor can relocate them without
//! breaking callers.

mod namespace;
#[cfg(feature = "qdrant")]
pub mod qdrant;
pub mod service;
pub mod store;

pub use namespace::Namespace;
pub use service::{
    validate_additive_properties_only, CreateOutcome, DomainPatch, MergeStrategy, NamespacedCatalog,
    OntologyService, PatchOutcome, ReplaceOutcome,
};
pub use store::{
    DomainSummary, InMemoryOntologyStore, JsonFileOntologyStore, OntologyStore, RenameOutcome,
    Version,
};

#[cfg(feature = "qdrant")]
pub use qdrant::QdrantOntologyStore;

// Stable facade over the domain types (physically in
// `crate::graph::ontology` for now).
pub use crate::graph::{
    DomainOntology, EntityTypeSpec, OntologyCatalog, OntologyError, OntologyPropertyType,
    PropertySpec, RelationTypeSpec,
};
