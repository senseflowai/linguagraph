//! Strongly-typed mirror of the JSON DSL.
//!
//! No business logic lives here — these types only describe what is legal
//! syntactically. Anything semantic (alias must exist, depth must be > 0,
//! aggregate may not appear with `find`, …) is enforced in [`super::parser`]
//! or in [`crate::ast`].

use serde::{Deserialize, Serialize};

/// Top-level DSL document.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DslQuery {
    pub action: Action,
    pub start: NodePattern,
    #[serde(default)]
    pub traversals: Vec<Traversal>,
    #[serde(default)]
    pub filters: Vec<Filter>,
    #[serde(default, rename = "return")]
    pub return_: Vec<ReturnItem>,
    #[serde(default)]
    pub group_by: Vec<String>,
    #[serde(default)]
    pub sort: Vec<SortItem>,
    #[serde(default)]
    pub limit: Option<u32>,
    /// Optional Cypher label to apply to every node in the query
    /// (start + traversal targets). When set, only entities that
    /// carry this label alongside their declared type label are
    /// matched.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prefix_label: Option<String>,
    /// Optional prefix folded into the embedding-index / Qdrant
    /// collection names used by typed filters (e.g. `SemanticText`
    /// `search` / `hybrid_search`). Must match the prefix used at
    /// ingest time, otherwise the typed query hits an empty index.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prefix_index: Option<String>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Action {
    Find,
    Aggregate,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct NodePattern {
    pub label: String,
    pub alias: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Traversal {
    /// Alias the traversal starts from. Must reference a previously
    /// bound alias (the start node or an earlier traversal's target).
    /// When omitted, defaults to the start node — so `traversals: [..., ...]`
    /// reads as "from `start`, also traverse …, also traverse …" rather
    /// than as a single chained path.
    #[serde(default)]
    pub from: Option<String>,
    pub edge: EdgePattern,
    pub target: NodePattern,
    #[serde(default)]
    pub depth: Option<DepthRange>,
    #[serde(default)]
    pub optional: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EdgePattern {
    pub label: String,
    pub alias: String,
    pub direction: Direction,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Direction {
    Out,
    In,
    Both,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub struct DepthRange {
    pub min: u32,
    pub max: u32,
}

/// A single filter predicate. The DSL keeps this flat — boolean composition
/// across multiple filters is implicit AND. This is enough for the queries an
/// LLM emits in practice and keeps the schema small.
///
/// Filters come in two shapes:
///
/// * **Plain**: untyped equality/range/containment over scalar properties.
/// * **Typed**: tagged with a registered `type` (e.g. `"SemanticText"`),
///   in which case the operator + value semantics are delegated to the
///   matching [`crate::types::TypeHandler`]. This is how custom field
///   types plug new ops (`search`, `hybrid_search`, `near`, …) in
///   without touching the core parser.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Filter {
    /// Qualified property reference, e.g. `"p.age"`.
    pub field: String,
    /// Operator name. For plain filters this maps onto [`FilterOp`]; for
    /// typed filters it is whatever the type handler accepts (kept as a
    /// string here so the DSL surface stays open-ended).
    pub op: String,
    pub value: serde_json::Value,
    /// Optional field-type tag. When present, the type handler decides
    /// how to validate the op + value and how to compile the predicate.
    #[serde(default, rename = "type", skip_serializing_if = "Option::is_none")]
    pub field_type: Option<String>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum FilterOp {
    Eq,
    Neq,
    Gt,
    Gte,
    Lt,
    Lte,
    In,
    Contains,
    StartsWith,
    EndsWith,
}

impl FilterOp {
    /// Parse the string form used in the DSL `op` field. Returns `None`
    /// when the op is not one of the built-in plain ops; the caller
    /// then tries to interpret it as a typed op.
    pub fn parse(s: &str) -> Option<Self> {
        Some(match s {
            "eq" => FilterOp::Eq,
            "neq" => FilterOp::Neq,
            "gt" => FilterOp::Gt,
            "gte" => FilterOp::Gte,
            "lt" => FilterOp::Lt,
            "lte" => FilterOp::Lte,
            "in" => FilterOp::In,
            "contains" => FilterOp::Contains,
            "starts_with" => FilterOp::StartsWith,
            "ends_with" => FilterOp::EndsWith,
            _ => return None,
        })
    }
}

/// One projected column. Either a plain field or an aggregation.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum ReturnItem {
    Aggregate {
        aggregate: AggregateFn,
        field: String,
        #[serde(default)]
        alias: Option<String>,
    },
    Field {
        field: String,
        #[serde(default)]
        alias: Option<String>,
    },
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum AggregateFn {
    Count,
    Sum,
    Avg,
    Min,
    Max,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SortItem {
    pub field: String,
    #[serde(default)]
    pub order: SortOrder,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "lowercase")]
pub enum SortOrder {
    #[default]
    Asc,
    Desc,
}

/// High-level, traversal-oriented query for text-chunk retrieval.
///
/// A [`TraversalQuery`] is the minimal structured request a client
/// emits when its goal is text search over the document graph. The
/// pipeline runs a two-channel retrieval against the vector store
/// (entities in the `_canonical` collection, chunks in the `text`
/// collection), follows `MENTIONS` from matched entities back to
/// the chunks that contain them, deduplicates chunks, aggregates
/// per-chunk scores, sorts, and (optionally) reranks the top hits
/// with a cross-encoder. See [`crate::core::Pipeline::run_traversal`]
/// for the full pipeline.
///
/// The schema labels and relations are fixed (`Chunk` / `Source` /
/// `MENTIONS` / `part_of`, with the text on `Chunk.text` and the
/// entity-merge key on `<Entity>._canonical`). Graphs with a
/// different ingest schema can't use this endpoint directly.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TraversalQuery {
    /// Entity names the caller is searching for, e.g.
    /// `["Elon Musk", "SpaceX"]`. Each non-empty name becomes one
    /// vector search against the entity `_canonical` collection.
    #[serde(default)]
    pub entities: Vec<String>,
    /// High-level search goal, e.g. "Find companies founded by Elon
    /// Musk". Used as the chunk-channel semantic query.
    pub goal: String,
    /// Raw verbatim client query. Kept for the cross-encoder
    /// reranker prompt; falls back to `goal` for the chunk-search
    /// vector query when `goal` is empty.
    pub query: String,
    /// Optional Cypher label applied to every node match in the
    /// traversal (entity, chunk, source). Must match the label
    /// stamped at ingest time.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prefix_label: Option<String>,
    /// Optional prefix folded into the Qdrant collection names
    /// used for the entity (`_canonical`) and chunk (`text`)
    /// retrieval channels. Must match the prefix used at ingest.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prefix_index: Option<String>,
    /// Maximum number of chunks returned (after dedup, scoring,
    /// and optional rerank). Defaults to the pipeline's
    /// `default_limit` when `None`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub limit: Option<u32>,
    /// Optional whitelist of Cypher labels for entity matches. When
    /// set, an entity hit only counts if at least one of these
    /// labels is present on the node. `None` accepts any entity
    /// type.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub entity_types: Option<Vec<String>>,
    /// Override the pipeline's default reranking behaviour:
    /// `Some(true)` forces a rerank step on the top-N hits (errors
    /// when no reranker is configured), `Some(false)` skips it,
    /// `None` defers to the pipeline (reranker runs iff
    /// `Pipeline::with_reranker` was used).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rerank: Option<bool>,
}

impl TraversalQuery {
    /// Construct a [`TraversalQuery`] with the minimum required
    /// fields. Prefix / limit / filters default to `None`.
    pub fn new(
        entities: impl IntoIterator<Item = impl Into<String>>,
        goal: impl Into<String>,
        query: impl Into<String>,
    ) -> Self {
        Self {
            entities: entities.into_iter().map(Into::into).collect(),
            goal: goal.into(),
            query: query.into(),
            prefix_label: None,
            prefix_index: None,
            limit: None,
            entity_types: None,
            rerank: None,
        }
    }

    /// Concatenate `query`, `goal`, and entity names into a single
    /// string. Used for legacy callers / display; the retrieval
    /// pipeline uses [`Self::goal_search_text`] and the entity
    /// names directly.
    pub fn search_text(&self) -> String {
        let mut s = self.query.trim().to_string();
        if !self.goal.trim().is_empty() {
            if !s.is_empty() {
                s.push('\n');
            }
            s.push_str(self.goal.trim());
        }
        let names: Vec<&str> = self
            .entities
            .iter()
            .map(|e| e.trim())
            .filter(|e| !e.is_empty())
            .collect();
        if !names.is_empty() {
            if !s.is_empty() {
                s.push('\n');
            }
            s.push_str("Entities: ");
            s.push_str(&names.join(", "));
        }
        s
    }

    /// Text used for the chunk-search channel. Prefers `goal`,
    /// falls back to `query` when `goal` is empty.
    pub fn goal_search_text(&self) -> String {
        let goal = self.goal.trim();
        if goal.is_empty() {
            self.query.trim().to_string()
        } else {
            goal.to_string()
        }
    }

    /// Non-empty trimmed entity names, in input order.
    pub fn entity_names(&self) -> Vec<String> {
        self.entities
            .iter()
            .map(|e| e.trim().to_string())
            .filter(|e| !e.is_empty())
            .collect()
    }
}

#[cfg(test)]
mod traversal_tests {
    use super::*;

    #[test]
    fn search_text_combines_query_goal_and_entities() {
        let t = TraversalQuery::new(
            ["Elon Musk", "Company"],
            "Find companies founded by Elon Musk",
            "What companies did Elon Musk found?",
        );
        let s = t.search_text();
        assert!(s.contains("What companies did Elon Musk found?"));
        assert!(s.contains("Find companies founded by Elon Musk"));
        assert!(s.contains("Elon Musk"));
        assert!(s.contains("Company"));
    }

    #[test]
    fn traversal_optional_defaults_false_and_deserializes() {
        let raw = r#"
        {
          "from": "p",
          "edge": {"label": "WORKS_AT", "alias": "w", "direction": "out"},
          "target": {"label": "Company", "alias": "c"}
        }
        "#;
        let traversal: Traversal = serde_json::from_str(raw).unwrap();
        assert!(!traversal.optional);

        let raw = r#"
        {
          "from": "p",
          "edge": {"label": "WORKS_AT", "alias": "w", "direction": "out"},
          "target": {"label": "Company", "alias": "c"},
          "optional": true
        }
        "#;
        let traversal: Traversal = serde_json::from_str(raw).unwrap();
        assert!(traversal.optional);
    }

    #[test]
    fn search_text_handles_empty_entities_and_goal() {
        let t = TraversalQuery::new(Vec::<String>::new(), "", "hello world");
        assert_eq!(t.search_text(), "hello world");
        assert_eq!(t.goal_search_text(), "hello world");
    }

    #[test]
    fn entity_names_trims_and_drops_blanks() {
        let t = TraversalQuery::new(
            ["  Elon Musk ", "", "  ", "SpaceX"],
            "goal",
            "query",
        );
        assert_eq!(t.entity_names(), vec!["Elon Musk", "SpaceX"]);
    }

    #[test]
    fn doc_find_json_shape_round_trips() {
        // Matches examples/doc_find.json verbatim.
        let raw = r#"
        {
          "entities": ["Article 365", "Code"],
          "goal": "Article 365",
          "query": "Article 365",
          "prefix_label": "Entity_ws_1",
          "prefix_index": "ws_1",
          "limit": 30
        }
        "#;
        let t: TraversalQuery = serde_json::from_str(raw).unwrap();
        assert_eq!(t.entities, vec!["Article 365", "Code"]);
        assert_eq!(t.goal, "Article 365");
        assert_eq!(t.query, "Article 365");
        assert_eq!(t.prefix_label.as_deref(), Some("Entity_ws_1"));
        assert_eq!(t.prefix_index.as_deref(), Some("ws_1"));
        assert_eq!(t.limit, Some(30));
        assert!(t.entity_types.is_none());
        assert!(t.rerank.is_none());
    }

    #[test]
    fn entity_types_and_rerank_deserialize_when_provided() {
        let raw = r#"
        {
          "entities": ["Acme"],
          "goal": "g",
          "query": "q",
          "entity_types": ["Person", "Company"],
          "rerank": true
        }
        "#;
        let t: TraversalQuery = serde_json::from_str(raw).unwrap();
        assert_eq!(
            t.entity_types.as_deref(),
            Some(["Person".to_string(), "Company".to_string()].as_slice())
        );
        assert_eq!(t.rerank, Some(true));
    }
}
