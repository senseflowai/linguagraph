//! Delete-by-source planner.
//!
//! Produces the Cypher fragments needed to wipe a `Source` from the
//! graph along with:
//!
//! * every `Chunk` attached to the source via `:part_of` (chunks are
//!   1:1 with their source by construction, so they always go),
//! * every user `Entity` whose only `:mention` link was to this source
//!   (i.e. "orphans" — entities that survive in some other source's
//!   subgraph are left alone),
//! * the source node itself,
//! * the Qdrant points associated with every doomed Memgraph node,
//!   removed from *every* vector collection in one collection-agnostic
//!   `libqlink.delete_batch_all` call. Sweeping all collections (rather
//!   than fanning a per-collection `libqlink.delete_batch` across a
//!   catalog-derived name list) is what guarantees orphan-entity
//!   embeddings are cleaned up even when their per-field collection
//!   isn't enumerable from the ontology catalog.
//!
//! The planner stays pure: it renders [`CypherQuery`] values and does
//! not talk to Memgraph or qlink itself. [`crate::core::Pipeline`]
//! drives the three phases (discover → qlink delete → DETACH DELETE)
//! sequentially.

use std::collections::{BTreeMap, BTreeSet};

use thiserror::Error;

use crate::ast::query::Literal;
use crate::builder::CypherQuery;
use crate::graph::{
    OntologyCatalog, OntologyPropertyType, CHUNK_LABEL, MENTION_REL, PART_OF_REL, SOURCE_LABEL,
};

/// Errors produced when assembling a [`DeletePlan`].
#[derive(Debug, Error)]
pub enum DeletePlanError {
    #[error("source name is empty")]
    EmptySourceName,
    #[error("invalid Cypher identifier '{0}': prefix labels must match [A-Za-z_][A-Za-z0-9_]*")]
    InvalidPrefixLabel(String),
}

/// Inputs needed to plan a source-scoped deletion.
#[derive(Debug, Clone)]
pub struct DeletePlan {
    /// `name` property value of the `Source` node to remove.
    pub source_name: String,
    /// Optional extra Cypher label scoping the source and its orphan
    /// candidates to a tenant / dataset partition. Mirrors the
    /// ingest-side `prefix_label`.
    pub prefix_label: Option<String>,
    /// Optional prefix folded into Qdrant collection names so the same
    /// property in different partitions maps to different collections.
    /// Mirrors the ingest-side `prefix_index`.
    pub prefix_index: Option<String>,
    /// Base collection name used by the SemanticText handler
    /// (`[types.SemanticText].collection` in config, defaults to
    /// `semantic_text`).
    pub semantic_collection: String,
}

/// Output of the discovery phase. The Memgraph IDs in [`source_id`],
/// [`orphan_ids`] and [`chunk_ids`] are the exact set of nodes that the
/// second and third phases must clean up — there are no further
/// re-queries.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct DiscoveredNodes {
    /// Memgraph internal id of the `Source` node, if it was found.
    /// `None` means the source name is unknown to the database — the
    /// pipeline reports zero deletions and skips the rest of the phases.
    pub source_id: Option<i64>,
    /// Memgraph internal ids of user entities that were mentioned only
    /// by the source being deleted.
    pub orphan_ids: Vec<i64>,
    /// Memgraph internal ids of every chunk attached to the source.
    pub chunk_ids: Vec<i64>,
}

impl DiscoveredNodes {
    /// Flattened id list (source + chunks + orphans) used by both the
    /// qlink-cleanup calls and the final `DETACH DELETE`.
    pub fn all_ids(&self) -> Vec<i64> {
        let mut ids = Vec::with_capacity(self.orphan_ids.len() + self.chunk_ids.len() + 1);
        ids.extend_from_slice(&self.orphan_ids);
        ids.extend_from_slice(&self.chunk_ids);
        if let Some(src) = self.source_id {
            ids.push(src);
        }
        ids
    }

    /// Total number of Memgraph nodes the plan will detach-delete.
    pub fn total_nodes(&self) -> usize {
        self.orphan_ids.len() + self.chunk_ids.len() + usize::from(self.source_id.is_some())
    }

    /// True when there's nothing to do.
    pub fn is_empty(&self) -> bool {
        self.source_id.is_none()
    }
}

impl DeletePlan {
    /// Construct a new plan, validating the prefix label up front.
    pub fn new(
        source_name: impl Into<String>,
        semantic_collection: impl Into<String>,
    ) -> Result<Self, DeletePlanError> {
        let source_name = source_name.into();
        if source_name.is_empty() {
            return Err(DeletePlanError::EmptySourceName);
        }
        Ok(Self {
            source_name,
            prefix_label: None,
            prefix_index: None,
            semantic_collection: semantic_collection.into(),
        })
    }

    pub fn with_prefix_label(mut self, prefix: Option<String>) -> Result<Self, DeletePlanError> {
        let prefix = prefix
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty());
        if let Some(p) = &prefix {
            if !is_valid_ident(p) {
                return Err(DeletePlanError::InvalidPrefixLabel(p.clone()));
            }
        }
        self.prefix_label = prefix;
        Ok(self)
    }

    pub fn with_prefix_index(mut self, prefix: Option<String>) -> Self {
        self.prefix_index = prefix
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty());
        self
    }

    /// Phase 1: discover the source's Memgraph id, the ids of every
    /// orphan entity (mentioned only by this source) and every chunk
    /// attached to this source.
    ///
    /// Returns one row with three columns: `source_id`, `orphan_ids`,
    /// `chunk_ids`. The orphan / chunk lists may be empty; `source_id`
    /// is `null` when the source does not exist.
    pub fn discover_query(&self) -> CypherQuery {
        let prefix_suffix = self.prefix_suffix();
        let text = format!(
            "MATCH (s:{src}{prefix} {{name: $source_name}})\n\
             OPTIONAL MATCH (s)<-[:{mention}]-(e)\n\
             WHERE e IS NOT NULL AND NOT EXISTS {{\n\
             \x20\x20MATCH (e)-[:{mention}]->(other:{src}{prefix})\n\
             \x20\x20WHERE id(other) <> id(s)\n\
             }}\n\
             WITH s, collect(DISTINCT id(e)) AS orphan_ids\n\
             OPTIONAL MATCH (s)<-[:{part_of}]-(c:{chunk}{prefix})\n\
             WITH s, orphan_ids, collect(DISTINCT id(c)) AS chunk_ids\n\
             RETURN id(s) AS source_id, orphan_ids, chunk_ids",
            src = SOURCE_LABEL,
            chunk = CHUNK_LABEL,
            mention = MENTION_REL,
            part_of = PART_OF_REL,
            prefix = prefix_suffix,
        );
        let mut params = BTreeMap::new();
        params.insert(
            "source_name".into(),
            Literal::String(self.source_name.clone()),
        );
        CypherQuery::new(text, params)
    }

    /// Phase 2: render a `CALL libqlink.delete_batch` for one Qdrant
    /// collection. Pass the union of all doomed Memgraph ids — qlink's
    /// `delete_batch` is a no-op for ids it doesn't know, so over-
    /// asking is safe and cheaper than enumerating which ids landed in
    /// which collection.
    pub fn qlink_delete_batch_query(&self, collection: &str, ids: &[i64]) -> CypherQuery {
        let text = "CALL libqlink.delete_batch($coll, $ids) YIELD success\n\
                    RETURN success"
            .to_string();
        let mut params = BTreeMap::new();
        params.insert("coll".into(), Literal::String(collection.to_string()));
        params.insert(
            "ids".into(),
            Literal::List(ids.iter().map(|i| Literal::Int(*i)).collect()),
        );
        CypherQuery::new(text, params)
    }

    /// Phase 2 (collection-agnostic): render a single
    /// `CALL libqlink.delete_batch_all` that wipes the doomed point ids
    /// from *every* Qdrant collection.
    ///
    /// This is the robust counterpart to [`Self::qlink_delete_batch_query`].
    /// The per-collection fan-out only cleans collections it can name, and
    /// it derives those names from the ontology catalog
    /// ([`Self::qlink_collections`]). When the catalog is missing or out of
    /// sync with what was actually embedded at ingest — which is the common
    /// case for user-entity text properties — those collections are never
    /// enumerated and their vectors leak. Sweeping every collection avoids
    /// that: Memgraph node ids are globally unique and qlink no-ops on ids a
    /// collection doesn't hold, so only the doomed points are ever removed.
    pub fn qlink_delete_all_query(&self, ids: &[i64]) -> CypherQuery {
        let text = "CALL libqlink.delete_batch_all($ids) YIELD success, collections\n\
                    RETURN success, collections"
            .to_string();
        let mut params = BTreeMap::new();
        params.insert(
            "ids".into(),
            Literal::List(ids.iter().map(|i| Literal::Int(*i)).collect()),
        );
        CypherQuery::new(text, params)
    }

    /// Phase 3: `DETACH DELETE` every doomed node by Memgraph id. The
    /// `DETACH` keyword takes care of every adjacent edge, including
    /// `:RELATION` edges between user entities that the planner did not
    /// enumerate explicitly.
    pub fn detach_delete_query(&self, ids: &[i64]) -> CypherQuery {
        let text = "UNWIND $ids AS i\n\
                    MATCH (n) WHERE id(n) = i\n\
                    DETACH DELETE n"
            .to_string();
        let mut params = BTreeMap::new();
        params.insert(
            "ids".into(),
            Literal::List(ids.iter().map(|i| Literal::Int(*i)).collect()),
        );
        CypherQuery::new(text, params)
    }

    /// Enumerate the Qdrant collections that may hold vectors for the
    /// doomed nodes. Delegates to [`text_field_names`] for the field
    /// enumeration so the same source of truth drives both deletion and
    /// the entity-type discovery query in
    /// [`crate::core::Pipeline::run_entity_type_search`].
    pub fn qlink_collections(&self, catalog: Option<&OntologyCatalog>) -> Vec<String> {
        text_field_names(catalog)
            .into_iter()
            .map(|p| with_prefix_index(self.prefix_index.as_deref(), &self.semantic_collection, &p))
            .collect()
    }

    fn prefix_suffix(&self) -> String {
        match self.prefix_label.as_deref() {
            Some(p) if !p.is_empty() => format!(":{p}"),
            _ => String::new(),
        }
    }
}

/// Names of every entity field that the SemanticText handler may have
/// embedded into a Qdrant collection. Combines the two built-ins —
/// `name` (Source) and `text` (Chunk) — with the soft-merge
/// `_canonical` slot and every `OntologyPropertyType::Text` property
/// declared in `catalog`. The set is used by [`DeletePlan`] to clean up
/// vectors and by entity-type discovery to enumerate which collections
/// to search.
///
/// The `_canonical` entry deliberately keeps the literal leading
/// underscore; combined with the `__` separator that
/// [`with_prefix_index`] uses, the resulting collection name carries a
/// triple underscore (e.g. `semantic_text___canonical`). This is the
/// soft-merge resolver's existing naming convention — see
/// `src/ingest/soft_merge/mod.rs` and `tests/integration_tests.rs:226`.
pub fn text_field_names(catalog: Option<&OntologyCatalog>) -> BTreeSet<String> {
    let mut props: BTreeSet<String> = BTreeSet::new();
    // Built-ins. Source.name and Chunk.text are always embedded
    // when a SemanticText handler is registered.
    props.insert("name".into());
    props.insert("text".into());
    // Soft-merge canonical embeddings. The handler writes whenever
    // PrimaryKey::Soft is in play; cheap to include unconditionally.
    props.insert("_canonical".into());

    if let Some(catalog) = catalog {
        for ontology in catalog.domains_view().values() {
            for entity in &ontology.entity_types {
                for prop in &entity.properties {
                    if prop.property_type == OntologyPropertyType::Text {
                        props.insert(prop.name.clone());
                    }
                }
            }
        }
    }

    props
}

/// Render the fully-qualified Qdrant collection name for a field, the
/// same way the SemanticText handler does at ingest time. Public so
/// that entity-type discovery can share the convention with
/// [`DeletePlan::qlink_collections`].
pub fn with_prefix_index(prefix_index: Option<&str>, base: &str, property: &str) -> String {
    match prefix_index {
        Some(p) if !p.is_empty() => format!("{p}__{base}__{property}"),
        _ => format!("{base}__{property}"),
    }
}

/// Validate that `s` is a Cypher-safe identifier so callers can splice
/// it into queries as a literal label without exposing parameter-binding
/// limitations to injection.
pub fn is_valid_ident(s: &str) -> bool {
    let mut chars = s.chars();
    let first = chars.next();
    matches!(first, Some(c) if c.is_ascii_alphabetic() || c == '_')
        && chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{
        DomainOntology, EntityTypeSpec, OntologyCatalog, OntologyPropertyType, PropertySpec,
    };

    fn plan(source: &str) -> DeletePlan {
        DeletePlan::new(source, "semantic_text").unwrap()
    }

    #[test]
    fn discover_query_renders_with_no_prefix() {
        let q = plan("src-1").discover_query();
        assert!(q.text.contains("MATCH (s:Source {name: $source_name})"));
        assert!(q.text.contains("OPTIONAL MATCH (s)<-[:mention]-(e)"));
        assert!(q.text.contains("MATCH (e)-[:mention]->(other:Source)"));
        assert!(q.text.contains("WHERE id(other) <> id(s)"));
        assert!(q.text.contains("OPTIONAL MATCH (s)<-[:part_of]-(c:Chunk)"));
        assert!(q
            .text
            .contains("RETURN id(s) AS source_id, orphan_ids, chunk_ids"));
        assert_eq!(
            q.params.get("source_name"),
            Some(&Literal::String("src-1".into()))
        );
    }

    #[test]
    fn discover_query_threads_prefix_label_through_every_match() {
        let q = plan("src-1")
            .with_prefix_label(Some("Tenant1".into()))
            .unwrap()
            .discover_query();
        assert!(q.text.contains("(s:Source:Tenant1 {name: $source_name})"));
        assert!(q.text.contains("(other:Source:Tenant1)"));
        assert!(q.text.contains("(c:Chunk:Tenant1)"));
    }

    #[test]
    fn discover_rejects_bad_prefix_label() {
        let err = plan("src")
            .with_prefix_label(Some("1Bad".into()))
            .unwrap_err();
        assert!(matches!(err, DeletePlanError::InvalidPrefixLabel(_)));
    }

    #[test]
    fn empty_source_name_rejected() {
        assert!(matches!(
            DeletePlan::new("", "semantic_text"),
            Err(DeletePlanError::EmptySourceName)
        ));
    }

    #[test]
    fn qlink_delete_query_binds_collection_and_ids() {
        let q = plan("src").qlink_delete_batch_query("semantic_text__name", &[1, 2, 3]);
        assert!(q.text.contains("libqlink.delete_batch($coll, $ids)"));
        assert_eq!(
            q.params.get("coll"),
            Some(&Literal::String("semantic_text__name".into()))
        );
        match q.params.get("ids").unwrap() {
            Literal::List(items) => {
                assert_eq!(items.len(), 3);
                assert_eq!(items[0], Literal::Int(1));
            }
            other => panic!("expected list, got {other:?}"),
        }
    }

    #[test]
    fn qlink_delete_all_query_binds_ids_only() {
        let q = plan("src").qlink_delete_all_query(&[7, 8, 9]);
        assert!(q.text.contains("libqlink.delete_batch_all($ids)"));
        assert!(q.text.contains("YIELD success, collections"));
        // Collection-agnostic: no per-collection name is bound.
        assert!(!q.params.contains_key("coll"));
        match q.params.get("ids").unwrap() {
            Literal::List(items) => {
                assert_eq!(items.len(), 3);
                assert_eq!(items[0], Literal::Int(7));
            }
            other => panic!("expected list, got {other:?}"),
        }
    }

    #[test]
    fn detach_delete_renders_unwind_loop() {
        let q = plan("src").detach_delete_query(&[10, 20]);
        assert!(q.text.contains("UNWIND $ids AS i"));
        assert!(q.text.contains("MATCH (n) WHERE id(n) = i"));
        assert!(q.text.contains("DETACH DELETE n"));
    }

    #[test]
    fn qlink_collections_unions_spec_props_with_builtins() {
        let mut catalog = OntologyCatalog::default();
        catalog.insert(
            "test",
            DomainOntology {
                entity_types: vec![EntityTypeSpec {
                    name: "Person".into(),
                    description: Some("a human".into()),
                    properties: vec![
                        PropertySpec {
                            name: "bio".into(),
                            description: Some("biography".into()),
                            property_type: OntologyPropertyType::Text,
                            required: false,
                            allowed_values: Vec::new(),
                        },
                        PropertySpec {
                            name: "age".into(),
                            description: Some("age".into()),
                            property_type: OntologyPropertyType::Number,
                            required: false,
                            allowed_values: Vec::new(),
                        },
                    ],
                    embedding: None,
                }],
                relation_types: vec![],
            },
        );
        let cols = plan("src").qlink_collections(Some(&catalog));
        // bio (from catalog), name + text (builtins). age is Int — skipped.
        assert!(cols.contains(&"semantic_text__bio".into()));
        assert!(cols.contains(&"semantic_text__name".into()));
        assert!(cols.contains(&"semantic_text__text".into()));
        assert!(!cols.iter().any(|c| c.ends_with("__age")));
    }

    #[test]
    fn qlink_collections_applies_prefix_index() {
        let cols = plan("src")
            .with_prefix_index(Some("tenant1".into()))
            .qlink_collections(None);
        for c in &cols {
            assert!(c.starts_with("tenant1__semantic_text__"), "got {c}");
        }
    }

    #[test]
    fn qlink_collections_without_spec_still_covers_builtins() {
        let cols = plan("src").qlink_collections(None);
        assert!(cols.contains(&"semantic_text__name".into()));
        assert!(cols.contains(&"semantic_text__text".into()));
    }

    #[test]
    fn discovered_nodes_all_ids_appends_source_last() {
        let d = DiscoveredNodes {
            source_id: Some(42),
            orphan_ids: vec![1, 2],
            chunk_ids: vec![10],
        };
        assert_eq!(d.all_ids(), vec![1, 2, 10, 42]);
        assert_eq!(d.total_nodes(), 4);
        assert!(!d.is_empty());
    }

    #[test]
    fn discovered_nodes_empty_when_no_source() {
        let d = DiscoveredNodes::default();
        assert!(d.is_empty());
        assert_eq!(d.total_nodes(), 0);
        assert!(d.all_ids().is_empty());
    }
}
