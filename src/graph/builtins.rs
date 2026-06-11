//! Built-in entity types and relation labels.
//!
//! Built-ins intentionally use strict primary keys (`name` for
//! [`Source`], `id` for [`Chunk`]) and therefore do NOT carry the
//! `_canonical` property that LLM-extracted user entities rely on for
//! cosine-similarity soft merge. They never enter the soft-merge
//! resolver path — [`Source`] is keyed by exact name, [`Chunk`] by
//! UUID v4.
//!
//! The [`crate::graph::GraphBuilder`] interface exposes two first-class
//! concepts that don't come from user mappings:
//!
//! * [`SOURCE_LABEL`] — `Source` is the document/origin a graph was
//!   built from. Every [`GraphBuilder::with_source`] graph has exactly
//!   one. Its `id` is a fresh UUID v4, its `name` is supplied by the
//!   caller and stored as a `Text` property (so it is embedded by the
//!   `SemanticText` handler when one is registered).
//! * [`CHUNK_LABEL`] — `Chunk` is a text fragment that belongs to a
//!   `Source`. Its `id` is a UUID v4, its `text` field is a `Text`
//!   property and therefore embedded for search.
//!
//! Two relation labels glue these together:
//!
//! * [`MENTION_REL`] (`mention`) — emitted automatically from every
//!   user-added entity to the surrounding [`Source`].
//! * [`PART_OF_REL`] (`part_of`) — emitted automatically from every
//!   [`Chunk`] to its [`Source`].
//!
//! The constructors here mint deterministic [`EntityGraph`] values so
//! callers (and the [`GraphBuilder`]) can produce them in a single line.
//!
//! [`GraphBuilder`]: crate::graph::GraphBuilder
//! [`Source`]: SOURCE_LABEL
//! [`Chunk`]: CHUNK_LABEL

use uuid::Uuid;

use crate::graph::schema::{EntityGraph, PropertyType};

/// Cypher label used for the built-in [`Source`] entity.
pub const SOURCE_LABEL: &str = "Source";

/// Cypher label used for the built-in [`Chunk`] entity.
pub const CHUNK_LABEL: &str = "Chunk";

/// Relation label connecting any user entity to its originating
/// [`Source`].
pub const MENTION_REL: &str = "mention";

/// Relation label connecting a [`Chunk`] to its parent [`Source`].
pub const PART_OF_REL: &str = "part_of";

/// Generate a fresh UUID v4 string suitable for use as a built-in
/// entity's primary key.
pub fn new_v4_id() -> String {
    Uuid::new_v4().hyphenated().to_string()
}

/// Construct a fully-initialised `Source` [`EntityGraph`] with a fresh
/// UUID v4 id and the supplied human-readable name. The `name` is
/// stored as a [`PropertyType::Text`] so the `SemanticText` handler
/// will embed it when one is registered.
pub fn new_source(name: impl Into<String>) -> EntityGraph {
    EntityGraph::new(SOURCE_LABEL)
        .strict_primary_key("name")
        .property("id", PropertyType::Keyword, new_v4_id())
        .property("name", PropertyType::Text, name.into())
}

/// Construct a fully-initialised `Chunk` [`EntityGraph`] with a fresh
/// UUID v4 id and the supplied text fragment. `text` is stored as a
/// [`PropertyType::Text`] so it is routed through the `SemanticText`
/// handler during ingestion, producing an embedding for vector search.
pub fn new_chunk(text: impl Into<String>) -> EntityGraph {
    EntityGraph::new(CHUNK_LABEL)
        .strict_primary_key("id")
        .property("id", PropertyType::Keyword, new_v4_id())
        .property("text", PropertyType::Text, text.into())
}

/// True when `label` names a reserved built-in entity type. Used by
/// the [`crate::graph::GraphBuilder`] to skip auto-attaching the
/// `:mention` edge from `Source` to itself.
pub fn is_builtin_entity(label: &str) -> bool {
    matches!(label, SOURCE_LABEL | CHUNK_LABEL)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_has_uuid_v4_and_named_text_property() {
        let s = new_source("My Doc");
        assert_eq!(s.r#type, SOURCE_LABEL);
        let id = match &s.properties["id"].value {
            serde_json::Value::String(v) => v.clone(),
            other => panic!("id should be a string, got {other:?}"),
        };
        let parsed = Uuid::parse_str(&id).expect("id parses as UUID");
        assert_eq!(parsed.get_version_num(), 4, "expected UUID v4");
        assert_eq!(s.properties["name"].property_type, PropertyType::Text);
    }

    #[test]
    fn chunk_has_uuid_v4_and_text_field() {
        let c = new_chunk("hello world");
        assert_eq!(c.r#type, CHUNK_LABEL);
        let id = match &c.properties["id"].value {
            serde_json::Value::String(v) => v.clone(),
            other => panic!("id should be a string, got {other:?}"),
        };
        let parsed = Uuid::parse_str(&id).expect("id parses as UUID");
        assert_eq!(parsed.get_version_num(), 4);
        assert_eq!(c.properties["text"].property_type, PropertyType::Text);
        assert_eq!(
            c.properties["text"].value,
            serde_json::Value::String("hello world".into())
        );
    }

    #[test]
    fn two_sources_have_distinct_ids() {
        let a = new_source("a");
        let b = new_source("b");
        assert_ne!(a.properties["id"].value, b.properties["id"].value);
    }
}
