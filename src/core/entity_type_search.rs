//! Entity-type discovery query for QA services.
//!
//! Given a piece of user text, [`Pipeline::run_entity_type_search`](crate::core::Pipeline::run_entity_type_search)
//! returns the unique entity types in the graph that are semantically
//! related to it, each annotated with its domain (from the ontology
//! catalog and the Cypher label the planner stamps at ingest) and its
//! scopes (the `scope_text` / `scope_table` / `scope_structured`
//! labels). The result is consumed by a QA front-end to decide which
//! types are worth probing with a DSL filter or a `TraversalQuery`.
//!
//! Two complementary signals feed the result:
//!
//! 1. **Vector search** — embeds the user text once and fans it out
//!    over every Qdrant collection populated by the SemanticText
//!    handler (`…__name`, `…__text`, `…___canonical`, plus one per
//!    `Text` property of every entity type in the ontology). The hits
//!    are resolved to nodes via `labels(n)` and aggregated by entity
//!    type. This is what actually exists in the graph.
//! 2. **Ontology catalog** — `OntologyCatalog::find` ranks the
//!    embeddings cached on every `EntityTypeSpec` against the same
//!    user text. This tells us which types are *appropriate* by their
//!    declared description, even when no data has landed yet.
//!
//! Optionally, the 1-hop graph neighbours of the hit nodes are rolled
//! up into a second list so the QA service can see adjacent types
//! that may carry the actual answer.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::graph::{OntologyCatalog, Scope};

/// Free-text discovery query.
#[derive(Debug, Clone, Deserialize)]
pub struct EntityTypeSearchQuery {
    /// User text the QA service received.
    pub text: String,

    /// `top_k` passed to `libqlink.search_labeled` for each
    /// collection. Defaults to [`DEFAULT_TOP_K`].
    #[serde(default = "default_top_k")]
    pub top_k: u32,

    /// Cosine cutoff for the vector channel. Hits below the threshold
    /// are dropped. `None` keeps every result inside `top_k`.
    /// Defaults to [`DEFAULT_SCORE_THRESHOLD`].
    #[serde(default = "default_score_threshold")]
    pub score_threshold: Option<f32>,

    /// Roll up 1-hop graph neighbours of the matched nodes into the
    /// `neighbors` result list. Off by default — opt in when the QA
    /// service wants the adjacent-types context.
    #[serde(default)]
    pub include_neighbors: bool,

    /// Also consult [`OntologyCatalog::find`] and surface the result
    /// in `catalog_score`.
    #[serde(default = "default_true")]
    pub include_catalog_signal: bool,

    /// Cosine cutoff used inside [`OntologyCatalog::find`]. Only
    /// applied when `include_catalog_signal` is `true`.
    /// Defaults to [`DEFAULT_CATALOG_THRESHOLD`].
    #[serde(default = "default_catalog_threshold")]
    pub catalog_threshold: f32,

    /// Subset of ontology field names (before prefixing) to search.
    /// `None` enumerates every known field from the ontology plus the
    /// `name` / `text` / `_canonical` built-ins.
    #[serde(default)]
    pub fields: Option<Vec<String>>,

    /// Fully-qualified collection names. When set, bypasses field
    /// enumeration and prefix folding entirely. Mostly useful for
    /// targeted debugging.
    #[serde(default)]
    pub collections: Option<Vec<String>>,
}

impl EntityTypeSearchQuery {
    /// Construct a query from the user text with all defaults.
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            top_k: default_top_k(),
            score_threshold: default_score_threshold(),
            include_neighbors: false,
            include_catalog_signal: default_true(),
            catalog_threshold: default_catalog_threshold(),
            fields: None,
            collections: None,
        }
    }
}

/// Discovery result.
#[derive(Debug, Clone, Default, Serialize)]
pub struct EntityTypeSearchResult {
    /// Types with actual data matching the query or appearing in the
    /// catalog signal. Sorted by score descending.
    pub matches: Vec<EntityTypeHit>,
    /// Types observed on 1-hop neighbours of `matches`. Empty when
    /// `include_neighbors` was false, or when no matches were found.
    pub neighbors: Vec<EntityTypeHit>,
    /// Collections that were searched. Useful for debugging and for
    /// matching `per_collection` keys.
    pub collections_searched: Vec<String>,
    /// Wall-clock time the call took, in milliseconds.
    pub elapsed_ms: u64,
}

/// One unique entity type in the result.
#[derive(Debug, Clone, Default, Serialize, PartialEq)]
pub struct EntityTypeHit {
    /// Cypher node label (e.g. `Person`).
    pub entity_type: String,
    /// Domain pulled from the catalog or, when the catalog has no
    /// matching entry, inferred from a sibling node label.
    pub domain: Option<String>,
    /// Scopes recovered from `scope_*` labels on the matched nodes.
    pub scopes: BTreeSet<Scope>,
    /// Best cosine across every collection that hit this type.
    /// `None` when the type came in solely through the catalog signal.
    pub vector_score: Option<f32>,
    /// Per-collection breakdown: `collection name → max(score)`. Empty
    /// for catalog-only hits.
    pub per_collection: BTreeMap<String, f32>,
    /// `OntologyCatalog::find` score, when the catalog signal is
    /// enabled and the type cleared `catalog_threshold`.
    pub catalog_score: Option<f32>,
    /// A few Memgraph internal ids of matched nodes, capped at
    /// [`MAX_SAMPLE_NODE_IDS`]. Useful for the optional neighbour
    /// roll-up and for follow-up DSL queries.
    pub sample_node_ids: Vec<i64>,
}

/// Default `top_k` passed to each `libqlink.search_labeled` call.
pub const DEFAULT_TOP_K: u32 = 32;

/// Default cosine cutoff for the vector channel.
///
/// Chosen lower than [`crate::types::handlers::DEFAULT_SEARCH_THRESHOLD`]
/// (0.8): discovery wants recall over precision so the QA service
/// gets a few extra candidates rather than missing relevant types.
pub const DEFAULT_SCORE_THRESHOLD: f32 = 0.5;

/// Default cosine cutoff for the catalog channel inside
/// [`OntologyCatalog::find`].
pub const DEFAULT_CATALOG_THRESHOLD: f32 = 0.45;

/// Per-type cap on how many node ids are surfaced in the result.
pub const MAX_SAMPLE_NODE_IDS: usize = 5;

fn default_top_k() -> u32 {
    DEFAULT_TOP_K
}

fn default_score_threshold() -> Option<f32> {
    Some(DEFAULT_SCORE_THRESHOLD)
}

fn default_catalog_threshold() -> f32 {
    DEFAULT_CATALOG_THRESHOLD
}

fn default_true() -> bool {
    true
}

/// One row of the vector-channel result, in the shape expected by
/// [`aggregate_hits`]. Constructed by `Pipeline::run_entity_type_search`
/// after decoding the Cypher response.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct HitRow {
    pub nid: i64,
    pub labels: Vec<String>,
    pub score: f32,
    pub collection: String,
}

/// Aggregate a flat list of vector-search hits into one
/// [`EntityTypeHit`] per unique entity type.
///
/// The classification of `labels` follows three rules in order:
///
/// 1. Drop the configured `prefix_label` if it's present on the node.
/// 2. Collect every recognised `scope_*` label into the `scopes` set.
/// 3. From the remainder, pick the entity type and domain:
///    a. First label whose name resolves through
///    [`OntologyCatalog::get_entity`] wins — that's the type and the
///    lookup also yields the domain.
///    b. If neither catalog hit, but one of the two remaining labels
///    matches a known domain name from the catalog, the other one is
///    the entity type.
///    c. Last-resort fallback: first remaining label is the entity
///    type, second is the domain.
pub(crate) fn aggregate_hits<I>(
    rows: I,
    catalog: Option<&OntologyCatalog>,
    prefix_label: Option<&str>,
) -> Vec<EntityTypeHit>
where
    I: IntoIterator<Item = HitRow>,
{
    let mut by_type: BTreeMap<String, EntityTypeHit> = BTreeMap::new();

    for row in rows {
        let (entity_type, domain, scopes) =
            classify_labels(&row.labels, catalog, prefix_label);
        let Some(entity_type) = entity_type else {
            continue;
        };

        let hit = by_type.entry(entity_type.clone()).or_insert_with(|| {
            EntityTypeHit {
                entity_type: entity_type.clone(),
                domain: domain.clone(),
                scopes: BTreeSet::new(),
                vector_score: None,
                per_collection: BTreeMap::new(),
                catalog_score: None,
                sample_node_ids: Vec::new(),
            }
        });
        // Domain is filled lazily; the catalog wins over later rows.
        if hit.domain.is_none() {
            hit.domain = domain;
        }
        for s in scopes {
            hit.scopes.insert(s);
        }
        hit.vector_score = Some(match hit.vector_score {
            Some(prev) if prev >= row.score => prev,
            _ => row.score,
        });
        if !row.collection.is_empty() {
            hit.per_collection
                .entry(row.collection.clone())
                .and_modify(|existing| {
                    if row.score > *existing {
                        *existing = row.score;
                    }
                })
                .or_insert(row.score);
        }
        if hit.sample_node_ids.len() < MAX_SAMPLE_NODE_IDS
            && !hit.sample_node_ids.contains(&row.nid)
        {
            hit.sample_node_ids.push(row.nid);
        }
    }

    by_type.into_values().collect()
}

/// Classify a node's raw label list. See [`aggregate_hits`] for the
/// rules.
pub(crate) fn classify_labels(
    labels: &[String],
    catalog: Option<&OntologyCatalog>,
    prefix_label: Option<&str>,
) -> (Option<String>, Option<String>, BTreeSet<Scope>) {
    let mut scopes = BTreeSet::new();
    let mut remaining: Vec<String> = Vec::new();
    for label in labels {
        if let Some(prefix) = prefix_label {
            if label == prefix {
                continue;
            }
        }
        if let Some(scope) = Scope::from_cypher_label(label) {
            scopes.insert(scope);
            continue;
        }
        remaining.push(label.clone());
    }

    if let Some(catalog) = catalog {
        for label in &remaining {
            if let Some((domain, _)) = catalog.get_entity(label) {
                return (Some(label.clone()), Some(domain.to_string()), scopes);
            }
        }
        if remaining.len() == 2 {
            let known: BTreeSet<&str> = catalog.domains().collect();
            let (a, b) = (&remaining[0], &remaining[1]);
            if known.contains(a.as_str()) {
                return (Some(b.clone()), Some(a.clone()), scopes);
            }
            if known.contains(b.as_str()) {
                return (Some(a.clone()), Some(b.clone()), scopes);
            }
        }
    }

    match remaining.len() {
        0 => (None, None, scopes),
        1 => (Some(remaining.remove(0)), None, scopes),
        _ => {
            let entity = remaining.remove(0);
            let domain = remaining.remove(0);
            (Some(entity), Some(domain), scopes)
        }
    }
}

/// Sort the hits the QA service receives: best signal first, with
/// alphabetical tie-break for determinism.
pub(crate) fn sort_matches(hits: &mut [EntityTypeHit]) {
    hits.sort_by(|a, b| {
        let score_a = a.vector_score.or(a.catalog_score).unwrap_or(0.0);
        let score_b = b.vector_score.or(b.catalog_score).unwrap_or(0.0);
        score_b
            .partial_cmp(&score_a)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.entity_type.cmp(&b.entity_type))
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{DomainOntology, EntityTypeSpec, OntologyCatalog};

    fn legal_catalog() -> OntologyCatalog {
        let mut cat = OntologyCatalog::default();
        cat.insert(
            "legal",
            DomainOntology {
                entity_types: vec![EntityTypeSpec::with_description("Person", "a human")],
                relation_types: vec![],
            },
        );
        cat
    }

    #[test]
    fn classify_uses_catalog_first_then_strips_scopes_and_prefix() {
        let cat = legal_catalog();
        let labels = vec![
            "Person".to_string(),
            "legal".to_string(),
            "scope_text".to_string(),
            "scope_table".to_string(),
            "tenantA".to_string(),
        ];
        let (entity, domain, scopes) =
            classify_labels(&labels, Some(&cat), Some("tenantA"));
        assert_eq!(entity.as_deref(), Some("Person"));
        assert_eq!(domain.as_deref(), Some("legal"));
        let mut expected = BTreeSet::new();
        expected.insert(Scope::Text);
        expected.insert(Scope::Table);
        assert_eq!(scopes, expected);
    }

    #[test]
    fn classify_falls_back_when_catalog_has_no_entry() {
        let labels = vec![
            "CustomType".to_string(),
            "scope_structured".to_string(),
            "custom_domain".to_string(),
        ];
        let (entity, domain, scopes) = classify_labels(&labels, None, None);
        assert_eq!(entity.as_deref(), Some("CustomType"));
        assert_eq!(domain.as_deref(), Some("custom_domain"));
        assert!(scopes.contains(&Scope::Structured));
    }

    #[test]
    fn classify_uses_known_domain_match_when_catalog_misses_type() {
        // catalog knows about "legal" as a domain but not "Receipt"
        let cat = legal_catalog();
        let labels = vec!["Receipt".to_string(), "legal".to_string()];
        let (entity, domain, _) = classify_labels(&labels, Some(&cat), None);
        assert_eq!(entity.as_deref(), Some("Receipt"));
        assert_eq!(domain.as_deref(), Some("legal"));
    }

    #[test]
    fn aggregate_groups_by_type_and_keeps_max_score_per_collection() {
        let rows = vec![
            HitRow {
                nid: 1,
                labels: vec!["Person".into(), "legal".into(), "scope_text".into()],
                score: 0.6,
                collection: "semantic_text__name".into(),
            },
            HitRow {
                nid: 2,
                labels: vec!["Person".into(), "legal".into(), "scope_text".into()],
                score: 0.8,
                collection: "semantic_text__name".into(),
            },
            HitRow {
                nid: 3,
                labels: vec!["Person".into(), "legal".into()],
                score: 0.55,
                collection: "semantic_text___canonical".into(),
            },
            HitRow {
                nid: 10,
                labels: vec!["Company".into(), "legal".into(), "scope_structured".into()],
                score: 0.7,
                collection: "semantic_text__name".into(),
            },
        ];
        let cat = legal_catalog();
        let hits = aggregate_hits(rows, Some(&cat), None);
        // Sorted by name (BTreeMap iteration); Company before Person.
        assert_eq!(hits.len(), 2);
        let person = hits.iter().find(|h| h.entity_type == "Person").unwrap();
        assert_eq!(person.domain.as_deref(), Some("legal"));
        assert_eq!(person.vector_score, Some(0.8));
        assert_eq!(
            person.per_collection.get("semantic_text__name"),
            Some(&0.8)
        );
        assert_eq!(
            person.per_collection.get("semantic_text___canonical"),
            Some(&0.55)
        );
        assert!(person.scopes.contains(&Scope::Text));
        assert_eq!(person.sample_node_ids, vec![1, 2, 3]);

        let company = hits.iter().find(|h| h.entity_type == "Company").unwrap();
        assert_eq!(company.vector_score, Some(0.7));
        assert!(company.scopes.contains(&Scope::Structured));
    }

    #[test]
    fn aggregate_skips_rows_without_a_classifiable_entity_type() {
        // Only scope + prefix labels — nothing to attribute the hit to.
        let rows = vec![HitRow {
            nid: 1,
            labels: vec!["scope_text".into(), "tenantA".into()],
            score: 0.7,
            collection: "semantic_text__name".into(),
        }];
        let hits = aggregate_hits(rows, None, Some("tenantA"));
        assert!(hits.is_empty());
    }

    #[test]
    fn sort_matches_orders_by_score_desc_then_name() {
        let mut hits = vec![
            EntityTypeHit {
                entity_type: "Zebra".into(),
                vector_score: Some(0.9),
                ..Default::default()
            },
            EntityTypeHit {
                entity_type: "Apple".into(),
                catalog_score: Some(0.6),
                ..Default::default()
            },
            EntityTypeHit {
                entity_type: "Banana".into(),
                vector_score: Some(0.6),
                ..Default::default()
            },
        ];
        sort_matches(&mut hits);
        assert_eq!(hits[0].entity_type, "Zebra");
        // Apple and Banana share 0.6; alphabetical tie-break.
        assert_eq!(hits[1].entity_type, "Apple");
        assert_eq!(hits[2].entity_type, "Banana");
    }

    #[test]
    fn query_defaults_neighbours_off_and_catalog_on() {
        let q = EntityTypeSearchQuery::new("hello");
        assert!(!q.include_neighbors);
        assert!(q.include_catalog_signal);
        // Same for the serde default path.
        let from_json: EntityTypeSearchQuery =
            serde_json::from_str(r#"{"text": "hello"}"#).unwrap();
        assert!(!from_json.include_neighbors);
        assert!(from_json.include_catalog_signal);
    }
}
