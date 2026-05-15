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
/// A [`TraversalQuery`] is a structured request that the LLM (or
/// any client) emits when its goal is text search rather than a
/// hand-built MATCH pattern: instead of describing the Cypher
/// shape, the client lists the entities it cares about, the
/// search goal, and the verbatim user query. The pipeline turns
/// that into a small retrieval pipeline:
///
/// 1. Search entities by each supplied name and traverse back to
///    the chunks that mention them.
/// 2. Search chunks by the goal text.
/// 3. Merge both result streams into unique chunks with their
///    associated entities.
///
/// Defaults match the built-ins used by document ingestion:
/// `Chunk`/`text` for the chunk node, `MENTIONS` for chunk→entity,
/// `part_of` for chunk→source, and `Source` for the source node.
/// Override any of them for graphs that use different labels.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TraversalQuery {
    /// Entity names the caller is searching for, e.g.
    /// `["Elon Musk", "SpaceX"]`. Each non-empty name becomes one
    /// entity lookup query in the traversal pipeline.
    #[serde(default)]
    pub entities: Vec<String>,

    #[serde(default = "default_entity_field")]
    pub entity_field: String,
    /// High-level search goal, e.g. "Find companies founded by Elon
    /// Musk". Used as part of the semantic search query.
    pub goal: String,
    /// Raw verbatim client query, e.g. "What companies did Elon
    /// Musk found?". Kept for callers that want the original user
    /// text alongside the more structured goal. The traversal
    /// pipeline uses it as a fallback when `goal` is empty.
    pub query: String,
    /// Cypher label of the searchable chunk node. Defaults to
    /// `"Chunk"`.
    #[serde(default = "default_chunk_label")]
    pub chunk_label: String,
    /// Property on the chunk node carrying the searchable text.
    /// Defaults to `"text"`.
    #[serde(default = "default_chunk_text_field")]
    pub chunk_text_field: String,
    /// Relation label connecting a chunk to a mentioned entity.
    /// Defaults to `"MENTIONS"`.
    #[serde(default = "default_mentions_rel")]
    pub mentions_rel: String,
    /// Relation label connecting a chunk to its source.
    /// Defaults to `"part_of"`.
    #[serde(default = "default_part_of_rel")]
    pub part_of_rel: String,
    /// Cypher label of the source node. Defaults to `"Source"`.
    #[serde(default = "default_source_label")]
    pub source_label: String,
    /// Optional label restriction for the entity target of the
    /// `MENTIONS` traversal. When `None`, the target is label-less
    /// and matches entities of any type.
    #[serde(default)]
    pub entity_label: Option<String>,
    /// Optional max number of result rows.
    #[serde(default)]
    pub limit: Option<u32>,
    /// Optional Cypher label applied to every node in the lowered
    /// queries (start + traversal targets). Propagated identically
    /// into both the entity-search leg and the goal-search leg.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prefix_label: Option<String>,
    /// Optional prefix folded into the embedding-index / Qdrant
    /// collection names used by the goal-search and entity-search
    /// legs. Must match the prefix used at ingest time.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prefix_index: Option<String>,
}

fn default_chunk_label() -> String {
    "Chunk".into()
}
fn default_chunk_text_field() -> String {
    "text".into()
}

fn default_entity_field() -> String {
    "name".into()
}
fn default_mentions_rel() -> String {
    "MENTIONS".into()
}
fn default_part_of_rel() -> String {
    "part_of".into()
}
fn default_source_label() -> String {
    "Source".into()
}

impl TraversalQuery {
    /// Construct a [`TraversalQuery`] from the three caller-supplied
    /// fields. All other knobs take their default values (matching
    /// the built-in document-ingestion labels).
    pub fn new(
        entities: impl IntoIterator<Item = impl Into<String>>,
        goal: impl Into<String>,
        query: impl Into<String>,
    ) -> Self {
        Self {
            entities: entities.into_iter().map(Into::into).collect(),
            goal: goal.into(),
            query: query.into(),
            chunk_label: default_chunk_label(),
            chunk_text_field: default_chunk_text_field(),
            mentions_rel: default_mentions_rel(),
            part_of_rel: default_part_of_rel(),
            source_label: default_source_label(),
            entity_field: default_entity_field(),
            entity_label: None,
            limit: None,
            prefix_label: None,
            prefix_index: None,
        }
    }

    /// Concatenate `query`, `goal`, and entity names into a single
    /// string. Kept for backwards compatibility with callers/tests
    /// that inspect the old combined search text; traversal execution
    /// now uses [`Self::goal_search_text`] instead.
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

    /// Text used for the chunk-search leg of traversal retrieval.
    /// The explicit goal is preferred; the raw query is a fallback for
    /// older callers that only populated `query`.
    pub fn goal_search_text(&self) -> String {
        let goal = self.goal.trim();
        if goal.is_empty() {
            self.query.trim().to_string()
        } else {
            goal.to_string()
        }
    }

    /// Build one DSL query per supplied entity name. Each query starts
    /// from a seed entity, filters by `name`, walks incoming
    /// `MENTIONS` edges to chunks, then fans back out from each chunk
    /// to all mentioned entities and its source.
    pub fn entity_dsls(&self) -> Vec<DslQuery> {
        self.entities
            .iter()
            .map(|name| name.trim())
            .filter(|name| !name.is_empty())
            .map(|name| self.entity_dsl(&self.entity_field, name))
            .collect()
    }

    /// Lower the traversal query into the equivalent [`DslQuery`].
    ///
    /// The resulting DSL has:
    ///
    /// This is the goal-search leg of the full traversal pipeline:
    ///
    /// * `start = (c:<chunk_label>)`
    /// * traversal 1: `c -[m:<mentions_rel>]-> (e[:<entity_label>])`
    ///   — at most one hop, target label optional.
    /// * traversal 2: `c -[po:<part_of_rel>]-> (s:<source_label>)`
    ///   — at most one hop.
    /// * one typed `SemanticText` filter on `c.<chunk_text_field>`
    ///   carrying the goal text.
    /// * returns: chunk text + id, source name + id, entity name.
    ///
    /// The total hop count is bounded at two by construction —
    /// both traversals start from the same chunk, so no path is
    /// ever longer than one edge from `c`.
    pub fn into_dsl(self) -> DslQuery {
        let search_text = self.goal_search_text();
        let text_field = format!("c.{}", self.chunk_text_field);
        let prefix_label = self.prefix_label.clone();
        let prefix_index = self.prefix_index.clone();

        let mut traversals = Vec::with_capacity(2);

        // Hop 2: chunk → source.
        traversals.push(Traversal {
            from: Some("c".into()),
            edge: EdgePattern {
                label: self.part_of_rel,
                alias: "po".into(),
                direction: Direction::Out,
            },
            target: NodePattern {
                label: self.source_label,
                alias: "s".into(),
            },
            depth: None,
            optional: false,
        });

        // Hop 1: chunk → mentioned entity.
        traversals.push(Traversal {
            from: Some("c".into()),
            edge: EdgePattern {
                label: self.mentions_rel.clone(),
                alias: "m".into(),
                direction: Direction::In,
            },
            target: NodePattern {
                label: self.entity_label.clone().unwrap_or_default(),
                alias: "e".into(),
            },
            depth: None,
            optional: true,
        });

        let filters = vec![Filter {
            field: text_field,
            op: "search".into(),
            value: serde_json::Value::String(search_text),
            field_type: Some("SemanticText".into()),
        }];

        let return_ = traversal_return_items(&self.chunk_text_field);

        DslQuery {
            action: Action::Find,
            start: NodePattern {
                label: self.chunk_label,
                alias: "c".into(),
            },
            traversals,
            filters,
            return_,
            group_by: Vec::new(),
            sort: Vec::new(),
            limit: self.limit,
            prefix_label,
            prefix_index,
        }
    }

    fn entity_dsl(&self, field: &str, value: &str) -> DslQuery {
        let entity_label = self.entity_label.clone().unwrap_or_default();
        DslQuery {
            action: Action::Find,
            start: NodePattern {
                label: entity_label,
                alias: "e".into(),
            },
            traversals: vec![
                Traversal {
                    from: Some("e".into()),
                    edge: EdgePattern {
                        label: self.mentions_rel.clone(),
                        alias: "seed_m".into(),
                        direction: Direction::Out,
                    },
                    target: NodePattern {
                        label: self.chunk_label.clone(),
                        alias: "c".into(),
                    },
                    depth: None,
                    optional: false,
                },
                Traversal {
                    from: Some("c".into()),
                    edge: EdgePattern {
                        label: self.part_of_rel.clone(),
                        alias: "po".into(),
                        direction: Direction::Out,
                    },
                    target: NodePattern {
                        label: self.source_label.clone(),
                        alias: "s".into(),
                    },
                    depth: None,
                    optional: false,
                },
            ],
            filters: vec![Filter {
                field: format!("e.{}", field).into(),
                op: "search_reranked".into(),
                value: serde_json::Value::String(value.to_string()),
                field_type: Some("SemanticText".into()),
            }],
            return_: traversal_return_items(&self.chunk_text_field),
            group_by: Vec::new(),
            sort: Vec::new(),
            limit: self.limit,
            prefix_label: self.prefix_label.clone(),
            prefix_index: self.prefix_index.clone(),
        }
    }
}

fn traversal_return_items(chunk_text_field: &str) -> Vec<ReturnItem> {
    vec![
        ReturnItem::Field {
            field: format!("c.{chunk_text_field}"),
            alias: Some("chunk_text".into()),
        },
        ReturnItem::Field {
            field: "c.id".into(),
            alias: Some("chunk_id".into()),
        },
        ReturnItem::Field {
            field: "s.name".into(),
            alias: Some("source_name".into()),
        },
        ReturnItem::Field {
            field: "s.id".into(),
            alias: Some("source_id".into()),
        },
        ReturnItem::Field {
            field: "e".into(),
            alias: Some("entity".into()),
        },
    ]
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
    fn into_dsl_produces_two_one_hop_traversals_from_chunk() {
        let t = TraversalQuery::new(
            ["Elon Musk", "Company"],
            "Find companies founded by Elon Musk",
            "What companies did Elon Musk found?",
        );
        let dsl = t.into_dsl();

        assert_eq!(dsl.action, Action::Find);
        assert_eq!(dsl.start.label, "Chunk");
        assert_eq!(dsl.start.alias, "c");
        assert_eq!(dsl.traversals.len(), 2);

        // `part_of` is the *required* hop (chunk → source) and goes
        // first; `MENTIONS` is the optional fan-out (chunk ← entity)
        // and comes second. The match_part emitter splits required
        // and optional traversals into separate MATCH / OPTIONAL
        // MATCH clauses, so this order is what lets a chunk with no
        // mentions still return its source.
        let part_of = &dsl.traversals[0];
        assert_eq!(part_of.from.as_deref(), Some("c"));
        assert_eq!(part_of.edge.label, "part_of");
        assert_eq!(part_of.edge.direction, Direction::Out);
        assert_eq!(part_of.target.label, "Source");
        assert_eq!(part_of.target.alias, "s");
        assert!(part_of.depth.is_none());
        assert!(!part_of.optional);

        let mentions = &dsl.traversals[1];
        assert_eq!(mentions.from.as_deref(), Some("c"));
        assert_eq!(mentions.edge.label, "MENTIONS");
        // Incoming MENTIONS: entities point at chunks they mention.
        assert_eq!(mentions.edge.direction, Direction::In);
        assert_eq!(mentions.target.label, "");
        assert_eq!(mentions.target.alias, "e");
        assert!(mentions.depth.is_none());
        assert!(mentions.optional);
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
    fn into_dsl_emits_semantic_text_filter_with_combined_query() {
        let t = TraversalQuery::new(
            ["Elon Musk"],
            "Find companies founded by Elon Musk",
            "What companies did Elon Musk found?",
        );
        let dsl = t.into_dsl();

        assert_eq!(dsl.filters.len(), 1);
        let f = &dsl.filters[0];
        assert_eq!(f.field, "c.text");
        assert_eq!(f.op, "search");
        assert_eq!(f.field_type.as_deref(), Some("SemanticText"));
        let v = f.value.as_str().expect("filter value is a string");
        assert!(v.contains("Find companies founded by Elon Musk"));
        assert!(!v.contains("What companies did Elon Musk found?"));
    }

    #[test]
    fn into_dsl_returns_chunks_sources_and_entities() {
        let dsl = TraversalQuery::new(["X"], "g", "q").into_dsl();
        let aliases: Vec<&str> = dsl
            .return_
            .iter()
            .filter_map(|r| match r {
                ReturnItem::Field { alias, .. } => alias.as_deref(),
                _ => None,
            })
            .collect();
        assert!(aliases.contains(&"chunk_text"));
        assert!(aliases.contains(&"source_name"));
        assert!(aliases.contains(&"entity"));
    }

    #[test]
    fn entity_label_pins_the_target_label() {
        let mut t = TraversalQuery::new(["Acme"], "find acme", "what is acme?");
        t.entity_label = Some("Company".into());
        let dsl = t.into_dsl();
        // traversals[0] is the `part_of` hop (chunk → source);
        // the entity target lives in the second (optional) hop.
        assert_eq!(dsl.traversals[1].target.label, "Company");
    }

    #[test]
    fn search_text_handles_empty_entities_and_goal() {
        let t = TraversalQuery::new(Vec::<String>::new(), "", "hello world");
        assert_eq!(t.search_text(), "hello world");
        assert_eq!(t.goal_search_text(), "hello world");
    }

    #[test]
    fn entity_name_dsls_search_entities_then_chunks() {
        let t = TraversalQuery::new(["Elon Musk"], "find companies", "query");
        let dsls = t.entity_dsls();
        assert_eq!(dsls.len(), 1);
        let dsl = &dsls[0];
        // Entity-name leg starts at the (possibly label-less) entity
        // bound to alias `e`, then walks outward to chunks and on to
        // each chunk's source.
        assert_eq!(dsl.start.alias, "e");
        assert_eq!(dsl.start.label, "");
        assert_eq!(dsl.filters[0].field, "e.name");
        assert_eq!(dsl.filters[0].op, "search_reranked");
        assert_eq!(dsl.traversals[0].from.as_deref(), Some("e"));
        assert_eq!(dsl.traversals[0].edge.direction, Direction::Out);
        assert_eq!(dsl.traversals[0].target.label, "Chunk");
        assert_eq!(dsl.traversals[1].from.as_deref(), Some("c"));
        assert_eq!(dsl.traversals[1].edge.direction, Direction::Out);
        assert_eq!(dsl.traversals[1].target.alias, "s");
        assert_eq!(dsl.traversals[1].target.label, "Source");
    }
}
