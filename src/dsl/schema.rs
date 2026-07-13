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
    /// Legacy action hint. The lowerer derives the effective read kind from
    /// the projection (`return`) so generated DSL remains valid when an LLM
    /// forgets to switch this from `find` to `aggregate`. The field is optional
    /// on input and defaults to `find` for backwards-compatible Rust callers.
    #[serde(default)]
    pub action: Action,
    pub start: NodePattern,
    #[serde(default)]
    pub traversals: Vec<Traversal>,
    #[serde(default)]
    pub filters: Vec<Filter>,
    #[serde(default, rename = "return")]
    pub return_: Vec<ReturnItem>,
    #[serde(default)]
    pub group_by: Vec<GroupByItem>,
    #[serde(default)]
    pub sort: Vec<SortItem>,
    #[serde(default)]
    pub limit: Option<u32>,
    /// Emit `RETURN DISTINCT` — deduplicate the projected rows. Useful when
    /// a traversal fans out (e.g. one person per movie they acted in) but
    /// the projection only keeps a column that then repeats.
    #[serde(default)]
    pub distinct: bool,
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

impl DslQuery {
    /// Render a concise, human-readable natural-language summary of what
    /// this query selects and returns. Intended for downstream LLM context
    /// so a synthesizer can be told the rows it sees are already filtered.
    ///
    /// This is purely descriptive — it is NOT Cypher and does not
    /// round-trip back into a [`DslQuery`].
    ///
    /// Example: `Selecting Camera entities where state = "active"; returning: name.`
    pub fn describe(&self) -> String {
        let mut out = format!("Selecting {} entities", self.start.label);

        if !self.filters.is_empty() {
            out.push_str(" where ");
            let parts: Vec<String> = self.filters.iter().map(describe_filter).collect();
            out.push_str(&parts.join(" and "));
        }

        for traversal in &self.traversals {
            out.push_str(&format!(
                "; traversing: {}→{}",
                traversal.edge.label, traversal.target.label
            ));
        }

        if !self.return_.is_empty() {
            out.push_str("; returning: ");
            let parts: Vec<String> = self.return_.iter().map(describe_return_item).collect();
            out.push_str(&parts.join(", "));
        }

        if let Some(limit) = self.limit {
            out.push_str(&format!("; limit {limit}"));
        }

        out.push('.');
        out
    }
}

/// One filter predicate as `{field} {op} {value}`. Typed filters (and any
/// op the plain [`FilterOp`] parser doesn't recognise) degrade to a
/// neutral `matches` phrasing rather than inventing an operator symbol.
fn describe_filter(filter: &Filter) -> String {
    let field = strip_alias_prefix(&filter.field);
    let value = describe_value(&filter.value);
    match (filter.field_type.as_deref(), FilterOp::parse(&filter.op)) {
        (None, Some(op)) => format!("{} {} {}", field, describe_op(op), value),
        (field_type, _) => {
            let mut s = format!("{field} matches {value}");
            if let Some(ft) = field_type {
                s.push_str(&format!(" (typed: {ft})"));
            }
            s
        }
    }
}

fn describe_op(op: FilterOp) -> &'static str {
    match op {
        FilterOp::Eq => "=",
        FilterOp::Neq => "≠",
        FilterOp::Gt => ">",
        FilterOp::Gte => "≥",
        FilterOp::Lt => "<",
        FilterOp::Lte => "≤",
        FilterOp::In => "in",
        FilterOp::Contains => "contains",
        FilterOp::StartsWith => "starts with",
        FilterOp::EndsWith => "ends with",
    }
}

/// Human-readable rendering of a JSON filter value: strings quoted,
/// scalars verbatim, arrays comma-joined in brackets, objects as compact
/// JSON.
fn describe_value(value: &serde_json::Value) -> String {
    match value {
        serde_json::Value::String(s) => format!("\"{s}\""),
        serde_json::Value::Array(items) => {
            let parts: Vec<String> = items.iter().map(describe_value).collect();
            format!("[{}]", parts.join(", "))
        }
        serde_json::Value::Null => "null".to_string(),
        serde_json::Value::Object(_) => value.to_string(),
        serde_json::Value::Bool(b) => b.to_string(),
        serde_json::Value::Number(n) => n.to_string(),
    }
}

/// Short label for one projected column. Plain fields are stripped of
/// their alias prefix (`c.name` → `name`); aggregates and date-parts keep
/// a function-call shape.
fn describe_return_item(item: &ReturnItem) -> String {
    match item {
        ReturnItem::Field { field, alias } => alias
            .clone()
            .unwrap_or_else(|| strip_alias_prefix(field).to_string()),
        ReturnItem::Aggregate {
            aggregate,
            field,
            alias,
        } => alias
            .clone()
            .unwrap_or_else(|| format!("{}({})", aggregate_name(*aggregate), field)),
        ReturnItem::DatePart {
            field,
            date_part,
            alias,
        } => alias
            .clone()
            .unwrap_or_else(|| format!("{}({})", date_part_name(*date_part), field)),
    }
}

fn strip_alias_prefix(field: &str) -> &str {
    field.rsplit('.').next().unwrap_or(field)
}

fn aggregate_name(f: AggregateFn) -> &'static str {
    match f {
        AggregateFn::Count => "count",
        AggregateFn::Sum => "sum",
        AggregateFn::Avg => "avg",
        AggregateFn::Min => "min",
        AggregateFn::Max => "max",
        AggregateFn::Collect => "collect",
    }
}

fn date_part_name(d: DatePart) -> &'static str {
    match d {
        DatePart::Year => "year",
        DatePart::Quarter => "quarter",
        DatePart::Month => "month",
        DatePart::Day => "day",
        DatePart::Hour => "hour",
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Action {
    Find,
    Aggregate,
}

impl Default for Action {
    fn default() -> Self {
        Self::Find
    }
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
    /// Optional cardinality hint (`"one"` | `"many"`) for `SemanticText`
    /// filters. Disambiguates what the `op` alone cannot: whether the
    /// question names one specific entity (even via a fuzzy text/
    /// description match) or asks for every matching item. Ignored by
    /// filters that aren't folded into a `SemanticText` search; absent
    /// means "infer from `op`" (`eq`/`neq` → one, everything else →
    /// many) for backward compatibility.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cardinality: Option<String>,
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
    DatePart {
        field: String,
        date_part: DatePart,
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
    /// Gather the values into a list per group (Cypher `collect(...)`).
    Collect,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SortItem {
    pub field: String,
    #[serde(default)]
    pub order: SortOrder,
}

/// A grouping key. Keep the common case tiny (`"c.name"`), but allow
/// date/time bucketing when a timestamp should be grouped by a component.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum GroupByItem {
    Field(String),
    DatePart {
        field: String,
        date_part: DatePart,
        #[serde(default)]
        alias: Option<String>,
    },
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum DatePart {
    Year,
    Quarter,
    Month,
    Day,
    Hour,
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
/// collection), follows `mentions` from matched entities back to
/// the chunks that contain them, deduplicates chunks, aggregates
/// per-chunk scores, sorts, and (optionally) reranks the top hits
/// with a cross-encoder. See [`crate::core::Pipeline::run_traversal`]
/// for the full pipeline.
///
/// The schema labels and relations are fixed (`Chunk` / `Source` /
/// `mentions` / `part_of`, with the text on `Chunk.text` and the
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
mod describe_tests {
    use super::*;
    use serde_json::json;

    fn camera_filter() -> Filter {
        Filter {
            field: "c.state".into(),
            op: "eq".into(),
            value: json!("active"),
            field_type: None,
            cardinality: None,
        }
    }

    fn start(label: &str, alias: &str) -> NodePattern {
        NodePattern {
            label: label.into(),
            alias: alias.into(),
        }
    }

    fn field_return(field: &str) -> ReturnItem {
        ReturnItem::Field {
            field: field.into(),
            alias: None,
        }
    }

    #[test]
    fn describe_eq_filter_and_single_field() {
        let q = DslQuery {
            action: Action::Find,
            start: start("Camera", "c"),
            traversals: vec![],
            filters: vec![camera_filter()],
            return_: vec![field_return("c.name")],
            group_by: vec![],
            sort: vec![],
            limit: None,
            distinct: false,
            prefix_label: None,
            prefix_index: None,
        };
        assert_eq!(
            q.describe(),
            "Selecting Camera entities where state = \"active\"; returning: name."
        );
    }

    #[test]
    fn describe_multiple_filters_joined_with_and() {
        let q = DslQuery {
            action: Action::Find,
            start: start("Order", "o"),
            traversals: vec![],
            filters: vec![
                Filter {
                    field: "o.status".into(),
                    op: "eq".into(),
                    value: json!("completed"),
                    field_type: None,
                    cardinality: None,
                },
                Filter {
                    field: "o.total".into(),
                    op: "gte".into(),
                    value: json!(100),
                    field_type: None,
                    cardinality: None,
                },
            ],
            return_: vec![field_return("o.id")],
            group_by: vec![],
            sort: vec![],
            limit: None,
            distinct: false,
            prefix_label: None,
            prefix_index: None,
        };
        assert_eq!(
            q.describe(),
            "Selecting Order entities where status = \"completed\" and total ≥ 100; returning: id."
        );
    }

    #[test]
    fn describe_typed_filter_uses_matches_branch() {
        let q = DslQuery {
            action: Action::Find,
            start: start("Document", "d"),
            traversals: vec![],
            filters: vec![Filter {
                field: "d.body".into(),
                op: "search".into(),
                value: json!("invoice"),
                field_type: Some("SemanticText".into()),
                cardinality: None,
            }],
            return_: vec![field_return("d.title")],
            group_by: vec![],
            sort: vec![],
            limit: None,
            distinct: false,
            prefix_label: None,
            prefix_index: None,
        };
        assert_eq!(
            q.describe(),
            "Selecting Document entities where body matches \"invoice\" (typed: SemanticText); returning: title."
        );
    }

    #[test]
    fn describe_aggregate_projection() {
        let q = DslQuery {
            action: Action::Aggregate,
            start: start("Customer", "c"),
            traversals: vec![],
            filters: vec![],
            return_: vec![ReturnItem::Aggregate {
                aggregate: AggregateFn::Count,
                field: "o".into(),
                alias: None,
            }],
            group_by: vec![],
            sort: vec![],
            limit: None,
            distinct: false,
            prefix_label: None,
            prefix_index: None,
        };
        assert_eq!(
            q.describe(),
            "Selecting Customer entities; returning: count(o)."
        );
    }

    #[test]
    fn describe_without_filters_omits_where() {
        let q = DslQuery {
            action: Action::Find,
            start: start("Camera", "c"),
            traversals: vec![],
            filters: vec![],
            return_: vec![field_return("c.name")],
            group_by: vec![],
            sort: vec![],
            limit: Some(10),
            distinct: false,
            prefix_label: None,
            prefix_index: None,
        };
        let described = q.describe();
        assert!(!described.contains("where"));
        assert_eq!(
            described,
            "Selecting Camera entities; returning: name; limit 10."
        );
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
        let t = TraversalQuery::new(["  Elon Musk ", "", "  ", "SpaceX"], "goal", "query");
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
