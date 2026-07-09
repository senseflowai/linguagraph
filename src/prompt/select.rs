//! Query-driven entity/property selection.
//!
//! Domain selection ([`crate::graph::OntologyCatalog::select_domains`])
//! narrows the graph to the handful of domains relevant to a user query.
//! This module narrows further, to the individual entities, properties,
//! and relationships needed to build the DSL, so the rendered prompt stays
//! compact.
//!
//! The signal is embedding similarity between the query and one short
//! passage per entity and per property (see [`entity_embedding_text`] /
//! [`property_embedding_text`]). Those passages are cached across requests
//! by [`EmbeddingCache`]; only the query embedding is recomputed each call.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write;

use super::generator::SCHEMA_HIDDEN_PROPS;
use super::schema::{GraphSchema, NodeKind, Property};
use crate::embeddings::{cosine_similarity, EmbedError, EmbeddingCache, Embedder};

/// Tunables for [`select_query_schema`].
#[derive(Debug, Clone)]
pub struct QuerySelectionParams {
    /// Minimum cosine score for an entity to be kept on its own merit.
    pub entity_threshold: f32,
    /// Minimum cosine score for a property to be kept (and to promote its
    /// entity into the selection).
    pub property_threshold: f32,
    /// Upper bound on the number of score-selected entities.
    pub max_entities: usize,
    /// When nothing clears the thresholds inside the selected domains,
    /// keep this many top-scored entities as a best effort so the prompt
    /// is never empty.
    pub min_entities: usize,
    /// Relationship hops to expand around the score-selected entities,
    /// pulling in their neighbours so the LLM can express traversals.
    pub neighbor_hops: usize,
}

impl Default for QuerySelectionParams {
    fn default() -> Self {
        Self {
            entity_threshold: 0.30,
            property_threshold: 0.28,
            max_entities: 12,
            min_entities: 3,
            neighbor_hops: 1,
        }
    }
}

/// Query framing for the entity/property retrieval embedding. Mirrors the
/// style of `domain_query_text` used for domain routing.
fn query_text(query: &str) -> String {
    format!(
        "User query:{query}\nTask: Identify the ontology entities and properties needed to build a graph query for this request"
    )
}

/// Embedding passage for one property, in a stable, self-describing form:
///
/// ```text
/// property: Listing.sale_method
/// domain: flippa
/// type: enum
/// description: Sale method, auction, or fixed price
/// values: auction, classified, instant_sale
/// ```
///
/// `type` is `enum` when the property carries a closed value set,
/// otherwise the scalar type. The `description` and `values` lines are
/// emitted only when present.
pub(crate) fn property_embedding_text(domain: &str, label: &str, prop: &Property) -> String {
    let is_enum = !prop.allowed_values.is_empty();
    let ty = if is_enum { "enum" } else { prop.ty.as_str() };
    let mut out = String::new();
    let _ = writeln!(out, "property: {label}.{}", prop.name);
    let _ = writeln!(out, "domain: {domain}");
    let _ = writeln!(out, "type: {ty}");
    if let Some(desc) = prop.description.as_deref().filter(|d| !d.is_empty()) {
        let _ = writeln!(out, "description: {desc}");
    }
    if is_enum {
        let _ = writeln!(out, "values: {}", prop.allowed_values.join(", "));
    }
    out
}

/// Embedding passage for one entity: its label, domain, description, and
/// the names of its (non-hidden) properties.
pub(crate) fn entity_embedding_text(domain: &str, node: &NodeKind) -> String {
    let mut out = String::new();
    let _ = writeln!(out, "entity: {}", node.label);
    let _ = writeln!(out, "domain: {domain}");
    if let Some(desc) = node.description.as_deref().filter(|d| !d.is_empty()) {
        let _ = writeln!(out, "description: {desc}");
    }
    let names: Vec<&str> = node
        .properties
        .iter()
        .filter(|p| !SCHEMA_HIDDEN_PROPS.contains(&p.name.as_str()))
        .map(|p| p.name.as_str())
        .collect();
    if !names.is_empty() {
        let _ = writeln!(out, "properties: {}", names.join(", "));
    }
    out
}

/// Per-node scoring state gathered before the selection decision.
struct NodeScore {
    node: NodeKind,
    entity_score: f32,
    /// `property index -> cosine score` for non-hidden properties.
    prop_scores: BTreeMap<usize, f32>,
}

/// Narrow the already domain-filtered `domain_schemas` down to the schema
/// slice relevant to `query`.
///
/// Returns a single merged [`GraphSchema`] containing the score-selected
/// entities (with their relevant properties), their `neighbor_hops`
/// neighbours, and every relationship whose endpoints both survive. Node
/// `domain` fields are preserved so the renderer can still resolve
/// per-domain catalog annotations.
pub(crate) fn select_query_schema(
    query: &str,
    domain_schemas: &BTreeMap<String, GraphSchema>,
    cache: &mut EmbeddingCache,
    embedder: &dyn Embedder,
    params: &QuerySelectionParams,
) -> Result<GraphSchema, EmbedError> {
    if domain_schemas.is_empty() {
        return Ok(GraphSchema::default());
    }

    let query_embedding = embedder.embed(&query_text(query))?;

    // Flatten every (domain, node) and build the embedding-text batch:
    // one entity passage followed by its property passages, tracking where
    // each landed so scores can be read back positionally.
    let mut texts: Vec<String> = Vec::new();
    let mut entity_idx: Vec<usize> = Vec::new();
    let mut prop_idx: Vec<Vec<(usize, usize)>> = Vec::new();
    let mut flat: Vec<NodeKind> = Vec::new();

    for (domain, schema) in domain_schemas {
        for node in &schema.nodes {
            entity_idx.push(texts.len());
            texts.push(entity_embedding_text(domain, node));
            let mut props = Vec::new();
            for (pi, prop) in node.properties.iter().enumerate() {
                if SCHEMA_HIDDEN_PROPS.contains(&prop.name.as_str()) {
                    continue;
                }
                props.push((pi, texts.len()));
                texts.push(property_embedding_text(domain, &node.label, prop));
            }
            prop_idx.push(props);
            flat.push(node.clone());
        }
    }

    if flat.is_empty() {
        return Ok(GraphSchema::default());
    }

    let embeddings = cache.embed_cached(&texts, embedder)?;
    let score_at = |i: usize| cosine_similarity(&query_embedding, &embeddings[i]);

    // Score each node (entity score = max of its own passage and its best
    // property). Keyed by label — labels are unique across the projected
    // schema because each node binds to exactly one domain.
    let mut scores: BTreeMap<String, NodeScore> = BTreeMap::new();
    for (i, node) in flat.into_iter().enumerate() {
        let mut entity_score = score_at(entity_idx[i]);
        let mut prop_scores = BTreeMap::new();
        for (pi, ti) in &prop_idx[i] {
            let s = score_at(*ti);
            prop_scores.insert(*pi, s);
            if s > entity_score {
                entity_score = s;
            }
        }
        scores.insert(
            node.label.clone(),
            NodeScore {
                node,
                entity_score,
                prop_scores,
            },
        );
    }

    // Decide the score-selected entities.
    let mut ranked: Vec<(&String, &NodeScore)> = scores.iter().collect();
    ranked.sort_by(|a, b| {
        b.1.entity_score
            .total_cmp(&a.1.entity_score)
            .then_with(|| a.0.cmp(b.0))
    });

    let mut selected: BTreeSet<String> = ranked
        .iter()
        .filter(|(_, ns)| {
            ns.entity_score >= params.entity_threshold
                || ns
                    .prop_scores
                    .values()
                    .any(|s| *s >= params.property_threshold)
        })
        .take(params.max_entities)
        .map(|(label, _)| (*label).clone())
        .collect();

    if selected.is_empty() {
        // Nothing cleared the bar inside the selected domains: keep the
        // top few by score so the prompt still carries a usable slice.
        selected = ranked
            .iter()
            .take(params.min_entities.max(1))
            .map(|(label, _)| (*label).clone())
            .collect();
    }

    // Expand neighbours over relationships across the selected domains.
    let rels: Vec<&super::schema::RelKind> = domain_schemas
        .values()
        .flat_map(|s| s.relationships.iter())
        .collect();

    let mut labels = selected.clone();
    let mut frontier = selected.clone();
    for _ in 0..params.neighbor_hops {
        let mut next = BTreeSet::new();
        for rel in &rels {
            let (Some(from), Some(to)) = (&rel.from, &rel.to) else {
                continue;
            };
            if frontier.contains(from) && labels.insert(to.clone()) {
                next.insert(to.clone());
            }
            if frontier.contains(to) && labels.insert(from.clone()) {
                next.insert(from.clone());
            }
        }
        if next.is_empty() {
            break;
        }
        frontier = next;
    }

    // Assemble the narrowed schema. Score-selected entities keep only
    // their relevant (or enum) properties; neighbour-only entities keep
    // their full property set so a traversal target stays queryable.
    let mut nodes: Vec<NodeKind> = Vec::new();
    for label in &labels {
        let Some(ns) = scores.get(label) else {
            continue; // referenced by a relationship but not a real node here
        };
        let mut node = ns.node.clone();
        if selected.contains(label) {
            node.properties = filter_properties(&ns.node, &ns.prop_scores, params);
        }
        nodes.push(node);
    }

    let relationships = rels
        .iter()
        .filter(|rel| match (&rel.from, &rel.to) {
            (Some(from), Some(to)) => labels.contains(from) && labels.contains(to),
            _ => false,
        })
        .map(|rel| (*rel).clone())
        .collect();

    Ok(GraphSchema {
        nodes,
        relationships,
    })
}

/// Keep the query-relevant properties of a score-selected entity: those
/// clearing `property_threshold`, plus every enum-like field (cheap,
/// high-signal for filters). Never returns empty for an entity that has
/// visible properties — falls back to the full (non-hidden) set.
fn filter_properties(
    node: &NodeKind,
    prop_scores: &BTreeMap<usize, f32>,
    params: &QuerySelectionParams,
) -> Vec<Property> {
    let kept: Vec<Property> = node
        .properties
        .iter()
        .enumerate()
        .filter(|(pi, prop)| {
            !SCHEMA_HIDDEN_PROPS.contains(&prop.name.as_str())
                && (!prop.allowed_values.is_empty()
                    || prop_scores.get(pi).copied().unwrap_or(0.0) >= params.property_threshold)
        })
        .map(|(_, prop)| prop.clone())
        .collect();

    if kept.is_empty() {
        return node
            .properties
            .iter()
            .filter(|p| !SCHEMA_HIDDEN_PROPS.contains(&p.name.as_str()))
            .cloned()
            .collect();
    }
    kept
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prompt::schema::{PropertyType as PT, RelKind};

    fn prop(name: &str, ty: PT, values: &[&str]) -> Property {
        Property {
            name: name.into(),
            ty,
            description: None,
            allowed_values: values.iter().map(|v| v.to_string()).collect(),
        }
    }

    fn node(domain_label: &str, props: Vec<Property>) -> NodeKind {
        NodeKind {
            label: domain_label.into(),
            domain: None,
            extra_labels: Vec::new(),
            scopes: Vec::new(),
            description: None,
            properties: props,
        }
    }

    #[test]
    fn property_text_has_expected_shape() {
        let p = Property {
            name: "sale_method".into(),
            ty: PT::String,
            description: Some("Sale method, auction, or fixed price".into()),
            allowed_values: vec!["auction".into(), "classified".into(), "instant_sale".into()],
        };
        let text = property_embedding_text("flippa", "Listing", &p);
        assert!(text.contains("property: Listing.sale_method"));
        assert!(text.contains("domain: flippa"));
        assert!(text.contains("type: enum"));
        assert!(text.contains("description: Sale method, auction, or fixed price"));
        assert!(text.contains("values: auction, classified, instant_sale"));
    }

    #[test]
    fn scalar_property_text_omits_enum_and_values() {
        let p = prop("vin", PT::String, &[]);
        let text = property_embedding_text("cars", "Car", &p);
        assert!(text.contains("type: keyword"));
        assert!(!text.contains("type: enum"));
        assert!(!text.contains("values:"));
    }

    /// Deterministic 3-axis stub so cosine similarity is meaningful in
    /// tests. Keys on entity-specific tokens only (never `flippa`, which
    /// appears in every passage's `domain:` line): axis 0 =
    /// "listing/auction/sale/title", axis 1 = "clinic/patient/visit",
    /// axis 2 = a small constant floor.
    #[derive(Debug)]
    struct StubEmbedder;

    impl Embedder for StubEmbedder {
        fn dim(&self) -> usize {
            3
        }
        fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbedError> {
            Ok(texts
                .iter()
                .map(|t| {
                    let t = t.to_lowercase();
                    let mut v = [0.0f32, 0.0, 0.1];
                    if ["listing", "auction", "sale", "title"]
                        .iter()
                        .any(|k| t.contains(k))
                    {
                        v[0] += 1.0;
                    }
                    if ["clinic", "patient", "visit"].iter().any(|k| t.contains(k)) {
                        v[1] += 1.0;
                    }
                    v.to_vec()
                })
                .collect())
        }
    }

    fn params() -> QuerySelectionParams {
        QuerySelectionParams {
            entity_threshold: 0.2,
            property_threshold: 0.2,
            ..Default::default()
        }
    }

    #[test]
    fn narrows_to_relevant_entity_and_drops_others() {
        let listing = node(
            "Listing",
            vec![
                prop("sale_method", PT::String, &["auction", "classified", "instant_sale"]),
                prop("title", PT::String, &[]),
            ],
        );
        // Unrelated entity in the same domain, no relationship to Listing.
        let visit = node("ClinicVisit", vec![prop("patient_name", PT::String, &[])]);

        let mut schemas = BTreeMap::new();
        schemas.insert(
            "flippa".to_string(),
            GraphSchema {
                nodes: vec![listing, visit],
                relationships: vec![],
            },
        );

        let embedder = StubEmbedder;
        let mut cache = EmbeddingCache::new("stub", embedder.dim());
        let narrowed = select_query_schema(
            "listings sold by auction on flippa",
            &schemas,
            &mut cache,
            &embedder,
            &params(),
        )
        .unwrap();

        let labels: Vec<&str> = narrowed.nodes.iter().map(|n| n.label.as_str()).collect();
        assert!(labels.contains(&"Listing"), "relevant entity kept");
        assert!(
            !labels.contains(&"ClinicVisit"),
            "irrelevant entity dropped: {labels:?}"
        );
    }

    #[test]
    fn pulls_in_one_hop_neighbor() {
        let listing = node(
            "Listing",
            vec![prop("sale_method", PT::String, &["auction", "instant_sale"])],
        );
        // Seller has no query-matching text, but is one hop from Listing.
        let seller = node("Seller", vec![prop("rating", PT::Float, &[])]);
        let rel = RelKind {
            label: "SOLD_BY".into(),
            domain: Some("flippa".into()),
            description: None,
            from: Some("Listing".into()),
            to: Some("Seller".into()),
            properties: vec![],
        };

        let mut schemas = BTreeMap::new();
        schemas.insert(
            "flippa".to_string(),
            GraphSchema {
                nodes: vec![listing, seller],
                relationships: vec![rel],
            },
        );

        let embedder = StubEmbedder;
        let mut cache = EmbeddingCache::new("stub", embedder.dim());
        let narrowed = select_query_schema(
            "auction listings",
            &schemas,
            &mut cache,
            &embedder,
            &params(),
        )
        .unwrap();

        let labels: Vec<&str> = narrowed.nodes.iter().map(|n| n.label.as_str()).collect();
        assert!(labels.contains(&"Listing"));
        assert!(labels.contains(&"Seller"), "1-hop neighbour pulled in");
        assert_eq!(narrowed.relationships.len(), 1, "connecting rel kept");
    }

    #[test]
    fn keeps_enum_property_even_when_off_topic() {
        // A selected entity always retains its enum fields (useful for
        // filters) plus any query-matching property.
        let listing = node(
            "Listing",
            vec![
                prop("sale_method", PT::String, &["auction", "instant_sale"]),
                prop("random_scalar", PT::Int, &[]),
            ],
        );
        let mut schemas = BTreeMap::new();
        schemas.insert(
            "flippa".to_string(),
            GraphSchema {
                nodes: vec![listing],
                relationships: vec![],
            },
        );
        let embedder = StubEmbedder;
        let mut cache = EmbeddingCache::new("stub", embedder.dim());
        let narrowed = select_query_schema("auction sale", &schemas, &mut cache, &embedder, &params())
            .unwrap();
        let listing = &narrowed.nodes[0];
        let names: Vec<&str> = listing.properties.iter().map(|p| p.name.as_str()).collect();
        assert!(names.contains(&"sale_method"), "enum property retained");
    }
}
