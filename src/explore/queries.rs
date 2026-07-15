//! Parameterized Cypher templates behind the explorer.
//!
//! Cypher cannot parameterize labels, relationship types in patterns, or
//! property names in `ORDER BY` — those are validated with
//! [`is_valid_ident`] and inlined (the same pattern the ingest/delete and
//! traversal paths use). Every *value* travels as a `$param`.
//! `SKIP`/`LIMIT` take validated integers inlined like the Cypher
//! builder does.

use std::collections::BTreeMap;

use crate::ast::query::Literal;
use crate::builder::CypherQuery;
use crate::graph::{MENTION_REL, PART_OF_REL, SOURCE_LABEL};
use crate::ingest::delete::is_valid_ident;
use crate::normalize::{normalize_for_match, normalized_property_name};

use super::dto::RelDirection;
use super::ExploreError;

/// Public node handle accepted by explorer lookups: the `id` property, or
/// the `"_nid:<internal-id>"` session fallback minted for nodes without
/// one.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum NodeHandle {
    AppId(String),
    Nid(i64),
}

/// Prefix a [`NodeHandle`] renders to when only the internal id exists.
pub(crate) const NID_HANDLE_PREFIX: &str = "_nid:";

impl NodeHandle {
    pub(crate) fn parse(raw: &str) -> Self {
        raw.strip_prefix(NID_HANDLE_PREFIX)
            .and_then(|rest| rest.parse::<i64>().ok())
            .map(NodeHandle::Nid)
            .unwrap_or_else(|| NodeHandle::AppId(raw.to_string()))
    }

    /// All-digit app-id handles get an internal-id retry when the
    /// property lookup misses (see [`super::Explorer::entity`]).
    pub(crate) fn internal_id_fallback(&self) -> Option<NodeHandle> {
        match self {
            NodeHandle::AppId(id) => id.parse::<i64>().ok().map(NodeHandle::Nid),
            NodeHandle::Nid(_) => None,
        }
    }

    /// `WHERE` clause selecting the node bound as `n`, plus its bound
    /// parameters. An all-digit app id also matches integer-stored `id`
    /// properties (Cypher never coerces `1 = "1"`).
    fn where_clause(&self) -> (String, BTreeMap<String, Literal>) {
        match self {
            NodeHandle::AppId(id) => {
                let mut params = BTreeMap::from([("id".to_string(), Literal::String(id.clone()))]);
                match id.parse::<i64>() {
                    Ok(numeric) => {
                        params.insert("id_int".to_string(), Literal::Int(numeric));
                        ("(n.id = $id OR n.id = $id_int)".to_string(), params)
                    }
                    Err(_) => ("n.id = $id".to_string(), params),
                }
            }
            NodeHandle::Nid(nid) => (
                "id(n) = $nid".to_string(),
                BTreeMap::from([("nid".to_string(), Literal::Int(*nid))]),
            ),
        }
    }
}

/// `":Prefix"` suffix appended to every node pattern, or `""`.
/// Rejects prefixes that are not valid Cypher identifiers.
pub(crate) fn label_suffix(prefix: Option<&str>) -> Result<String, ExploreError> {
    match prefix {
        None => Ok(String::new()),
        Some(p) if is_valid_ident(p) => Ok(format!(":{p}")),
        Some(p) => Err(ExploreError::InvalidIdentifier(p.to_string())),
    }
}

fn validated_ident(raw: &str) -> Result<&str, ExploreError> {
    if is_valid_ident(raw) {
        Ok(raw)
    } else {
        Err(ExploreError::InvalidIdentifier(raw.to_string()))
    }
}

fn builtin_rels_cypher_list() -> String {
    format!("[\"{MENTION_REL}\", \"{PART_OF_REL}\"]")
}

/// T1 — one entity by handle: labels, properties, provenance sources.
pub(crate) fn entity_by_id(
    handle: &NodeHandle,
    prefix: Option<&str>,
) -> Result<CypherQuery, ExploreError> {
    let p = label_suffix(prefix)?;
    let (where_clause, params) = handle.where_clause();
    let text = format!(
        "MATCH (n{p}) WHERE {where_clause} \
         WITH n LIMIT 1 \
         OPTIONAL MATCH (n)-[:{MENTION_REL}|{PART_OF_REL}]->(src:{SOURCE_LABEL}{p}) \
         WITH n, collect(DISTINCT {{id: src.id, name: src.name}}) AS sources \
         RETURN id(n) AS nid, n.id AS id, labels(n) AS labels, \
                properties(n) AS props, sources"
    );
    Ok(CypherQuery::new(text, params))
}

/// T2 — relation summary of one entity, grouped by type/direction/
/// neighbor labels; built-in provenance edges excluded.
pub(crate) fn relation_summary(
    handle: &NodeHandle,
    prefix: Option<&str>,
) -> Result<CypherQuery, ExploreError> {
    let p = label_suffix(prefix)?;
    let (where_clause, params) = handle.where_clause();
    let builtin = builtin_rels_cypher_list();
    let text = format!(
        "MATCH (n{p})-[r]->(m{p}) WHERE {where_clause} AND NOT type(r) IN {builtin} \
         RETURN type(r) AS rel, \"out\" AS dir, labels(m) AS neighbor_labels, count(*) AS cnt \
         UNION ALL \
         MATCH (n{p})<-[r]-(m{p}) WHERE {where_clause} AND NOT type(r) IN {builtin} \
         RETURN type(r) AS rel, \"in\" AS dir, labels(m) AS neighbor_labels, count(*) AS cnt"
    );
    Ok(CypherQuery::new(text, params))
}

/// Resolved neighbor filters (validated upstream where needed).
pub(crate) struct NeighborQuery<'a> {
    pub handle: &'a NodeHandle,
    pub edge_types: Option<&'a [String]>,
    pub target_labels: Option<&'a [String]>,
    pub direction: Option<RelDirection>,
    pub limit: u32,
    pub offset: u32,
}

/// T3 — one hop from an entity, with type/label filters and pagination.
pub(crate) fn neighbors(
    q: &NeighborQuery<'_>,
    prefix: Option<&str>,
) -> Result<CypherQuery, ExploreError> {
    let p = label_suffix(prefix)?;
    let arrow = match q.direction {
        Some(RelDirection::Out) => "-[r]->",
        Some(RelDirection::In) => "<-[r]-",
        None => "-[r]-",
    };
    let (where_clause, mut params) = q.handle.where_clause();
    let mut conditions = vec![where_clause];
    match q.edge_types {
        Some(types) => {
            conditions.push("type(r) IN $edge_types".to_string());
            params.insert(
                "edge_types".to_string(),
                Literal::List(types.iter().cloned().map(Literal::String).collect()),
            );
        }
        // Provenance edges stay out of the default walk; ask for them
        // explicitly via `edge_types`.
        None => conditions.push(format!("NOT type(r) IN {}", builtin_rels_cypher_list())),
    }
    if let Some(labels) = q.target_labels {
        conditions.push("any(l IN labels(m) WHERE l IN $target_labels)".to_string());
        params.insert(
            "target_labels".to_string(),
            Literal::List(labels.iter().cloned().map(Literal::String).collect()),
        );
    }
    let where_part = conditions.join(" AND ");
    let (offset, limit) = (q.offset, q.limit);
    let text = format!(
        "MATCH (n{p}){arrow}(m{p}) WHERE {where_part} \
         RETURN id(m) AS nid, m.id AS id, labels(m) AS labels, properties(m) AS props, \
                type(r) AS rel, properties(r) AS rel_props, startNode(r) = n AS outgoing \
         ORDER BY rel ASC, id ASC \
         SKIP {offset} LIMIT {limit}"
    );
    Ok(CypherQuery::new(text, params))
}

/// `(predicate, params)` matching `{field}` against `ids` as strings —
/// and, for all-digit ids, as integers too (numeric-stored `id`
/// properties never equal their string form in Cypher).
fn id_list_predicate(
    field: &str,
    ids: &[String],
    param_prefix: &str,
) -> (String, BTreeMap<String, Literal>) {
    let mut params = BTreeMap::from([(
        format!("{param_prefix}ids"),
        Literal::List(ids.iter().cloned().map(Literal::String).collect()),
    )]);
    let numeric: Vec<Literal> = ids
        .iter()
        .filter_map(|id| id.parse::<i64>().ok())
        .map(Literal::Int)
        .collect();
    if numeric.is_empty() {
        (format!("{field} IN ${param_prefix}ids"), params)
    } else {
        params.insert(format!("{param_prefix}ids_int"), Literal::List(numeric));
        (
            format!("({field} IN ${param_prefix}ids OR {field} IN ${param_prefix}ids_int)"),
            params,
        )
    }
}

/// T4 — hydrate nodes by public ids.
pub(crate) fn nodes_by_ids(
    ids: &[String],
    prefix: Option<&str>,
) -> Result<CypherQuery, ExploreError> {
    let p = label_suffix(prefix)?;
    let (predicate, params) = id_list_predicate("n.id", ids, "");
    let text = format!(
        "MATCH (n{p}) WHERE {predicate} \
         RETURN id(n) AS nid, n.id AS id, labels(n) AS labels, properties(n) AS props"
    );
    Ok(CypherQuery::new(text, params))
}

/// T4b — hydrate nodes by internal ids (bridge from vector search).
pub(crate) fn nodes_by_nids(
    nids: &[i64],
    prefix: Option<&str>,
) -> Result<CypherQuery, ExploreError> {
    let p = label_suffix(prefix)?;
    let text = format!(
        "MATCH (n{p}) WHERE id(n) IN $nids \
         RETURN id(n) AS nid, n.id AS id, labels(n) AS labels, properties(n) AS props"
    );
    let params = BTreeMap::from([(
        "nids".to_string(),
        Literal::List(nids.iter().copied().map(Literal::Int).collect()),
    )]);
    Ok(CypherQuery::new(text, params))
}

/// T5 — every user edge whose both endpoints are in the id set.
pub(crate) fn edges_among(
    ids: &[String],
    edge_cap: usize,
    prefix: Option<&str>,
) -> Result<CypherQuery, ExploreError> {
    let p = label_suffix(prefix)?;
    let builtin = builtin_rels_cypher_list();
    let (from_predicate, params) = id_list_predicate("a.id", ids, "");
    let (to_predicate, _) = id_list_predicate("b.id", ids, "");
    let text = format!(
        "MATCH (a{p})-[r]->(b{p}) \
         WHERE {from_predicate} AND {to_predicate} AND NOT type(r) IN {builtin} \
         RETURN a.id AS from_id, b.id AS to_id, type(r) AS rel, properties(r) AS props \
         LIMIT {edge_cap}"
    );
    Ok(CypherQuery::new(text, params))
}

/// T6a — node counts grouped by the full label set (reduced to business
/// types client-side).
pub(crate) fn label_set_counts(prefix: Option<&str>) -> Result<CypherQuery, ExploreError> {
    let p = label_suffix(prefix)?;
    let text = format!("MATCH (n{p}) RETURN labels(n) AS labels, count(*) AS cnt");
    Ok(CypherQuery::new(text, BTreeMap::new()))
}

/// T6b — relation counts by type.
pub(crate) fn relation_type_counts(prefix: Option<&str>) -> Result<CypherQuery, ExploreError> {
    let p = label_suffix(prefix)?;
    let text = format!(
        "MATCH (a{p})-[r]->(b{p}) RETURN type(r) AS rel, count(*) AS cnt ORDER BY cnt DESC"
    );
    Ok(CypherQuery::new(text, BTreeMap::new()))
}

/// T6c — ingested documents.
pub(crate) fn list_sources(limit: u32, prefix: Option<&str>) -> Result<CypherQuery, ExploreError> {
    let p = label_suffix(prefix)?;
    let text = format!(
        "MATCH (s:{SOURCE_LABEL}{p}) RETURN s.id AS id, s.name AS name \
         ORDER BY name ASC LIMIT {limit}"
    );
    Ok(CypherQuery::new(text, BTreeMap::new()))
}

/// T7 — keyword search across the given string-typed properties.
///
/// `string_props` must come from schema introspection (string-typed
/// only): `toString()` on a list value is a runtime error in Cypher, so
/// the predicate never touches non-string properties.
pub(crate) fn keyword_search(
    needle: &str,
    string_props: &[String],
    entity_type: Option<&str>,
    exact: bool,
    limit: u32,
    prefix: Option<&str>,
) -> Result<CypherQuery, ExploreError> {
    let p = label_suffix(prefix)?;
    let type_label = match entity_type {
        Some(t) => format!(":{}", validated_ident(t)?),
        None => String::new(),
    };
    let mut predicates = Vec::new();
    for prop in string_props {
        let Ok(prop) = validated_ident(prop) else {
            continue; // introspected names may contain exotic characters — skip, don't fail
        };
        if exact {
            predicates.push(format!("n.{prop} = $q"));
        } else {
            let normalized = normalized_property_name(prop);
            predicates.push(format!(
                "toString(coalesce(n.{normalized}, \"\")) CONTAINS $q"
            ));
        }
    }
    if predicates.is_empty() {
        return Err(ExploreError::NoSearchableProperties);
    }
    let value_part = predicates.join(" OR ");
    // Chunks are text fragments, not business entities; keep them out
    // unless the caller pins the type explicitly.
    let chunk_guard = if entity_type.is_none() {
        "NOT n:Chunk AND "
    } else {
        ""
    };
    let text = format!(
        "MATCH (n{type_label}{p}) WHERE {chunk_guard}({value_part}) \
         RETURN id(n) AS nid, n.id AS id, labels(n) AS labels, properties(n) AS props \
         LIMIT {limit}"
    );
    let needle = if exact {
        needle.to_string()
    } else {
        normalize_for_match(needle)
    };
    let params = BTreeMap::from([("q".to_string(), Literal::String(needle))]);
    Ok(CypherQuery::new(text, params))
}

/// T8a — one page of entities of a type.
pub(crate) fn entity_table_page(
    entity_type: &str,
    sort_prop: &str,
    limit: u32,
    offset: u32,
    prefix: Option<&str>,
) -> Result<CypherQuery, ExploreError> {
    let p = label_suffix(prefix)?;
    let entity_type = validated_ident(entity_type)?;
    let sort_prop = validated_ident(sort_prop)?;
    let text = format!(
        "MATCH (n:{entity_type}{p}) \
         RETURN id(n) AS nid, n.id AS id, labels(n) AS labels, properties(n) AS props \
         ORDER BY n.{sort_prop} ASC \
         SKIP {offset} LIMIT {limit}"
    );
    Ok(CypherQuery::new(text, BTreeMap::new()))
}

/// T8b — total count for the table's pager.
pub(crate) fn entity_count(
    entity_type: &str,
    prefix: Option<&str>,
) -> Result<CypherQuery, ExploreError> {
    let p = label_suffix(prefix)?;
    let entity_type = validated_ident(entity_type)?;
    let text = format!("MATCH (n:{entity_type}{p}) RETURN count(n) AS total");
    Ok(CypherQuery::new(text, BTreeMap::new()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn node_handle_parses_nid_fallback() {
        assert_eq!(NodeHandle::parse("_nid:42"), NodeHandle::Nid(42));
        assert_eq!(NodeHandle::parse("m1"), NodeHandle::AppId("m1".to_string()));
        // Malformed nid falls back to an app id, never panics.
        assert_eq!(
            NodeHandle::parse("_nid:abc"),
            NodeHandle::AppId("_nid:abc".to_string())
        );
    }

    #[test]
    fn label_suffix_rejects_injection() {
        assert_eq!(label_suffix(Some("T1")).unwrap(), ":T1");
        assert_eq!(label_suffix(None).unwrap(), "");
        assert!(matches!(
            label_suffix(Some("T1) DETACH DELETE (n")),
            Err(ExploreError::InvalidIdentifier(_))
        ));
    }

    #[test]
    fn entity_by_id_scopes_prefix_and_binds_id() {
        let q = entity_by_id(&NodeHandle::parse("m1"), Some("T1")).unwrap();
        assert!(q.text.contains("MATCH (n:T1)"));
        assert!(q.text.contains("n.id = $id"));
        assert!(q.text.contains(":mention|part_of"));
        assert_eq!(q.params.get("id"), Some(&Literal::String("m1".into())));
    }

    #[test]
    fn entity_by_nid_matches_internal_id() {
        let q = entity_by_id(&NodeHandle::parse("_nid:7"), None).unwrap();
        assert!(q.text.contains("id(n) = $nid"));
        assert_eq!(q.params.get("nid"), Some(&Literal::Int(7)));
    }

    #[test]
    fn neighbors_renders_filters_and_pagination() {
        let handle = NodeHandle::parse("m1");
        let edge_types = vec!["ACTED_IN".to_string()];
        let target_labels = vec!["Person".to_string()];
        let q = neighbors(
            &NeighborQuery {
                handle: &handle,
                edge_types: Some(&edge_types),
                target_labels: Some(&target_labels),
                direction: Some(RelDirection::In),
                limit: 10,
                offset: 20,
            },
            Some("T1"),
        )
        .unwrap();
        assert!(q.text.contains("<-[r]-"));
        assert!(q.text.contains("type(r) IN $edge_types"));
        assert!(q
            .text
            .contains("any(l IN labels(m) WHERE l IN $target_labels)"));
        assert!(q.text.contains("SKIP 20 LIMIT 10"));
        assert!(
            !q.text.contains("NOT type(r) IN"),
            "explicit edge types disable the builtin guard"
        );
    }

    #[test]
    fn neighbors_excludes_builtin_edges_by_default() {
        let handle = NodeHandle::parse("m1");
        let q = neighbors(
            &NeighborQuery {
                handle: &handle,
                edge_types: None,
                target_labels: None,
                direction: None,
                limit: 50,
                offset: 0,
            },
            None,
        )
        .unwrap();
        assert!(q.text.contains("-[r]-("));
        assert!(q.text.contains("NOT type(r) IN [\"mention\", \"part_of\"]"));
    }

    #[test]
    fn keyword_search_builds_safe_predicates() {
        let props = vec![
            "name".to_string(),
            "bad prop".to_string(), // skipped, not an ident
        ];
        let q = keyword_search("Keanu", &props, None, false, 25, Some("T1")).unwrap();
        assert!(q
            .text
            .contains("toString(coalesce(n._lg_norm_name, \"\")) CONTAINS $q"));
        assert!(!q.text.contains("bad prop"));
        assert!(q.text.contains("NOT n:Chunk"));
        assert_eq!(q.params.get("q"), Some(&Literal::String("keanu".into())));
    }

    #[test]
    fn keyword_search_exact_uses_equality_and_original_case() {
        let props = vec!["name".to_string()];
        let q = keyword_search("Keanu", &props, Some("Person"), true, 5, None).unwrap();
        assert!(q.text.contains("MATCH (n:Person)"));
        assert!(q.text.contains("n.name = $q"));
        assert!(!q.text.contains("NOT n:Chunk"));
        assert_eq!(q.params.get("q"), Some(&Literal::String("Keanu".into())));
    }

    #[test]
    fn table_rejects_invalid_type_and_sort_idents() {
        assert!(matches!(
            entity_table_page("Movie) DETACH DELETE (n", "name", 10, 0, None),
            Err(ExploreError::InvalidIdentifier(_))
        ));
        assert!(matches!(
            entity_table_page("Movie", "name ASC; MATCH", 10, 0, None),
            Err(ExploreError::InvalidIdentifier(_))
        ));
    }
}
