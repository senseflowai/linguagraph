//! Render a system prompt from a [`GraphSchema`].
//!
//! The output is plain text suitable for any chat-completion endpoint. We
//! never embed examples that would leak provider-specific markers — the
//! prompt is a portable contract.

use std::fmt::Write;

use super::schema::{GraphSchema, NodeKind, Property, RelKind};
use crate::embeddings::{SharedEmbedder, SharedReranker};
use crate::graph::{GraphSpecification, PropertySpecRecord};
use crate::types::TypeRegistry;

#[derive(Debug, Clone)]
pub struct PromptSchemaSelection {
    /// Maximum graph-schema relationship hops to include around the
    /// entities matched by the query.
    pub related_entity_hops: usize,
    /// Minimum cosine score accepted from [`GraphSpecification::find`].
    pub entity_match_threshold: f32,
    /// Minimum score accepted after reranking.
    pub reranking_threshold: f64,
}

impl Default for PromptSchemaSelection {
    fn default() -> Self {
        Self {
            related_entity_hops: 2,
            entity_match_threshold: 0.4,
            reranking_threshold: 0.3,
        }
    }
}

#[derive(Debug, Clone)]
pub struct PromptOptions {
    /// Optional task framing line, e.g. "You are a graph query planner."
    pub preamble: Option<String>,
    /// If true, include 1-2 worked examples after the rules.
    pub include_examples: bool,
    /// Optional graph specification. When provided, each schema entry whose
    /// key matches `<NodeLabel>.<property>` (or `<NodeLabel>` for the node
    /// itself) is annotated inline.
    pub graph_specification: Option<GraphSpecification>,
    /// Embedder used to match the user's query against the graph
    /// specification. When omitted, the full schema is rendered.
    pub embedding_model: Option<SharedEmbedder>,
    /// Reranker applied after embedding retrieval. When omitted, embedding
    /// retrieval scores are used directly.
    pub reranking_model: Option<SharedReranker>,
    /// Query-specific schema selection controls.
    pub schema_selection: PromptSchemaSelection,
    /// Registered field types whose capabilities should be advertised
    /// to the LLM. When `None`, the prompt only describes plain ops.
    pub type_registry: Option<TypeRegistry>,
}

impl Default for PromptOptions {
    fn default() -> Self {
        Self {
            preamble: Some(
                "You translate user questions about a graph into a JSON DSL. \
                 Emit only valid JSON, no prose."
                    .into(),
            ),
            include_examples: true,
            graph_specification: None,
            embedding_model: None,
            reranking_model: None,
            schema_selection: PromptSchemaSelection::default(),
            type_registry: None,
        }
    }
}

pub fn generate_system_prompt(schema: &GraphSchema, opts: &PromptOptions) -> String {
    render_prompt(schema, opts)
}

pub fn generate_query_prompt(query: &str, schema: &GraphSchema, opts: &PromptOptions) -> String {
    //let selected_schema = select_query_schema(query, schema, opts);
    render_prompt(&schema, opts)
}

fn render_prompt(schema: &GraphSchema, opts: &PromptOptions) -> String {
    let mut out = String::with_capacity(2048);

    if let Some(p) = &opts.preamble {
        out.push_str(p);
        out.push_str("\n\n");
    }

    out.push_str("# Graph schema\n");
    write_nodes(&mut out, &schema.nodes, opts.graph_specification.as_ref());
    write_rels(
        &mut out,
        &schema.relationships,
        opts.graph_specification.as_ref(),
    );

    if let Some(reg) = &opts.type_registry {
        write_field_types(&mut out, reg);
    }

    out.push_str("\n# DSL rules\n");
    out.push_str(DSL_RULES);

    if opts.include_examples {
        out.push_str("\n# Examples\n");
        out.push_str(EXAMPLES);
    }

    out
}

/// Select the schema slice relevant to `query`.
///
/// The default strategy uses [`GraphSpecification::find`] to seed relevant
/// entity labels, then expands through schema relationships according to
/// [`PromptSchemaSelection`]. Callers can use this directly when they need
/// the selected schema separately from prompt rendering.
pub fn select_query_schema(query: &str, schema: &GraphSchema, opts: &PromptOptions) -> GraphSchema {
    let Some(spec) = opts.graph_specification.as_ref() else {
        return schema.clone();
    };
    let Some(embedder) = opts.embedding_model.as_deref() else {
        return schema.clone();
    };
    if query.trim().is_empty() {
        return schema.clone();
    }

    let Ok(matches) = spec.find(
        query,
        opts.schema_selection.entity_match_threshold,
        embedder,
        opts.reranking_model.as_deref(),
        opts.schema_selection.reranking_threshold,
    ) else {
        return schema.clone();
    };
    if matches.is_empty() {
        return GraphSchema::default();
    }

    let mut labels: std::collections::BTreeSet<String> =
        matches.into_iter().map(|m| m.record.name.clone()).collect();
    let mut frontier = labels.clone();
    for _ in 0..opts.schema_selection.related_entity_hops {
        let mut next = std::collections::BTreeSet::new();
        for rel in &schema.relationships {
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

    let nodes = schema
        .nodes
        .iter()
        .filter(|node| labels.contains(&node.label))
        .cloned()
        .collect();
    let relationships = schema
        .relationships
        .iter()
        .filter(|rel| match (&rel.from, &rel.to) {
            (Some(from), Some(to)) => labels.contains(from) && labels.contains(to),
            _ => false,
        })
        .cloned()
        .collect();

    GraphSchema {
        nodes,
        relationships,
    }
}

/// Render a `# Field types` section enumerating registered handlers,
/// their capabilities, supported ops, and an example DSL fragment.
///
/// The LLM uses this to decide when to attach `"type"` to a filter.
fn write_field_types(out: &mut String, registry: &TypeRegistry) {
    if registry.is_empty() {
        return;
    }
    out.push_str("\n# Field types\n");
    out.push_str(
        "Filters may be tagged with `\"type\"` to opt into specialised behaviour. \
         Each registered type lists the ops it supports.\n",
    );
    let mut handlers: Vec<_> = registry.iter().collect();
    handlers.sort_by(|a, b| a.type_id().0.cmp(&b.type_id().0));
    for h in handlers {
        let hint = h.prompt_hint();
        let _ = writeln!(
            out,
            "  - {}  [capabilities: {}]",
            hint.type_id, hint.capabilities
        );
        if let Some(doc) = hint.doc {
            let _ = writeln!(out, "      {doc}");
        }
        if !hint.ops.is_empty() {
            let ops: Vec<&str> = hint.ops.iter().map(|o| o.as_str()).collect();
            let _ = writeln!(out, "      ops: {}", ops.join(", "));
        }
        if let Some(ex) = hint.example {
            let _ = writeln!(out, "      example: {ex}");
        }
    }
}

fn write_nodes(out: &mut String, nodes: &[NodeKind], spec: Option<&GraphSpecification>) {
    if nodes.is_empty() {
        out.push_str("(no node labels declared)\n");
        return;
    }
    out.push_str("Nodes:\n");
    for n in nodes {
        let header_desc = spec.and_then(|s| s.get_entity(&n.label).map(|e| e.description.as_str()));
        let _ = match header_desc {
            Some(d) => writeln!(
                out,
                "  - {} — {}{}",
                n.label,
                d,
                render_props(&n.label, &n.properties, spec)
            ),
            None => writeln!(
                out,
                "  - {}{}",
                n.label,
                render_props(&n.label, &n.properties, spec)
            ),
        };
    }
}

fn write_rels(out: &mut String, rels: &[RelKind], spec: Option<&GraphSpecification>) {
    if rels.is_empty() {
        out.push_str("Relationships: (none declared)\n");
        return;
    }
    out.push_str("Relationships:\n");
    for r in rels {
        let endpoints = match (&r.from, &r.to) {
            (Some(f), Some(t)) => format!("({f})-[:{}]->({t})", r.label),
            _ => format!("[:{}]", r.label),
        };
        let _ = writeln!(
            out,
            "  - {}{}",
            endpoints,
            render_props(&r.label, &r.properties, spec)
        );
    }
}

fn render_props(owner: &str, props: &[Property], spec: Option<&GraphSpecification>) -> String {
    if props.is_empty() {
        return String::new();
    }
    let inner: Vec<String> = props
        .iter()
        .map(|p| {
            let property_spec = spec.and_then(|s| s.get_property(owner, &p.name));
            // Property header shape:
            //   <name>: <scalar-ty>                       (untyped, undocumented)
            //   <name>: <scalar-ty> @<FieldType>           (typed, e.g. SemanticText)
            //   <name>: <scalar-ty> /* description */      (documented only)
            //   <name>: <scalar-ty> @<FieldType> /* … */   (both)
            //
            // Graph Text fields route through SemanticText ingestion and
            // query handlers, so surface that type marker in the prompt.
            let mut base = format!("{}: {}", p.name, format_ty(p.ty));
            if let Some(ty) = property_spec.and_then(field_type_marker) {
                base = format!("{base} @{ty}");
            }
            if let Some(desc) = property_spec
                .map(|p| p.description.as_str())
                .filter(|desc| !desc.is_empty())
            {
                base = format!("{base} /* {desc} */");
            }
            base
        })
        .collect();
    format!(" {{ {} }}", inner.join(", "))
}

fn field_type_marker(spec: &PropertySpecRecord) -> Option<&'static str> {
    match spec.r#type {
        crate::graph::PropertyType::Text => Some("SemanticText"),
        _ => None,
    }
}

fn format_ty(t: super::schema::PropertyType) -> &'static str {
    use super::schema::PropertyType::*;
    match t {
        String => "string",
        Int => "int",
        Float => "float",
        Bool => "bool",
        Date => "date",
        Datetime => "datetime",
        List => "list",
    }
}

const DSL_RULES: &str = r#"Output a single JSON object with this shape:
{
  "action": "find" | "aggregate",
  "start":  { "label": <NodeLabel>, "alias": <ident> },
  "traversals": [
    {
      "from":   <ident>,
      "edge":   { "label": <RelLabel>, "alias": <ident>, "direction": "out"|"in"|"both" },
      "target": { "label": <NodeLabel>, "alias": <ident> },
      "depth":  { "min": <int>, "max": <int> }   // optional
    }
  ],
  "filters": [
    { "field": "<alias>.<prop>", "op": "eq|neq|gt|gte|lt|lte|in|contains|starts_with|ends_with",
      "value": <json scalar or array> }
  ],
  "return": [
    { "field": "<alias>.<prop>", "alias": <ident> },
    { "aggregate": "count|sum|avg|min|max", "field": "<alias>[.<prop>]", "alias": <ident> }
  ],
  "group_by": [ "<alias>.<prop>" ],
  "sort":     [ { "field": <alias-or-projected>, "order": "asc"|"desc" } ],
  "limit":    <int>
}

Constraints:
- Use only labels and properties listed in the schema above.
- Aliases must be unique across the whole query.
- "find" queries must NOT contain aggregations.
- "aggregate" queries that mix aggregated and non-aggregated columns must list the
  non-aggregated columns in `group_by`.
- Never embed user-supplied values in identifiers; values go in `filters[*].value`.
- Explicitly specify in `traversals[*].from` where the relation should be created .
"#;

const EXAMPLES: &str = r#"User: "Show me people over 30 who know someone in Berlin."
Assistant:
{
  "action": "find",
  "start": { "label": "Person", "alias": "p" },
  "traversals": [
    { "edge": { "label": "KNOWS", "alias": "r", "direction": "out" },
      "target": { "label": "Person", "alias": "friend" } }
  ],
  "filters": [
    { "field": "p.age", "op": "gt", "value": 30 },
    { "field": "friend.city", "op": "eq", "value": "Berlin" }
  ],
  "return": [{ "field": "p.name", "alias": "name" }],
  "limit": 25
}

User: "Total spend per customer for completed orders, top 10."
Assistant:
{
  "action": "aggregate",
  "start": { "label": "Customer", "alias": "c" },
  "traversals": [
    { "from": "c",
      "edge": { "label": "PLACED", "alias": "po", "direction": "out" },
      "target": { "label": "Order", "alias": "o" } }
  ],
  "filters": [{ "field": "o.status", "op": "eq", "value": "completed" }],
  "return": [
    { "field": "c.name", "alias": "customer" },
    { "aggregate": "sum", "field": "o.total", "alias": "total_spent" }
  ],
  "group_by": ["c.name"],
  "sort": [{ "field": "total_spent", "order": "desc" }],
  "limit": 10
}
"#;

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    use crate::embeddings::{EmbedError, Embedder};
    use crate::prompt::schema::PropertyType as PT;

    #[test]
    fn includes_schema_and_rules() {
        let schema = GraphSchema {
            nodes: vec![NodeKind {
                label: "Person".into(),
                properties: vec![
                    Property {
                        name: "name".into(),
                        ty: PT::String,
                    },
                    Property {
                        name: "age".into(),
                        ty: PT::Int,
                    },
                ],
            }],
            relationships: vec![RelKind {
                label: "KNOWS".into(),
                from: Some("Person".into()),
                to: Some("Person".into()),
                properties: vec![],
            }],
        };
        let prompt = generate_system_prompt(&schema, &PromptOptions::default());
        assert!(prompt.contains("Person"));
        assert!(prompt.contains("name: string"));
        assert!(prompt.contains("(Person)-[:KNOWS]->(Person)"));
        assert!(prompt.contains("\"action\": \"find\""));
    }

    #[test]
    fn graph_specification_annotates_schema_block() {
        let schema = GraphSchema {
            nodes: vec![NodeKind {
                label: "Camera".into(),
                properties: vec![
                    Property {
                        name: "id".into(),
                        ty: PT::String,
                    },
                    Property {
                        name: "state".into(),
                        ty: PT::String,
                    },
                ],
            }],
            relationships: vec![],
        };
        let spec = crate::graph::GraphSpecification::new()
            .with_entity("Camera", "An IP surveillance camera")
            .with_property(
                "Camera",
                "state",
                crate::graph::PropertyType::String,
                "active or inactive",
            );
        let prompt = generate_system_prompt(
            &schema,
            &PromptOptions {
                graph_specification: Some(spec),
                include_examples: false,
                ..PromptOptions::default()
            },
        );
        assert!(prompt.contains("Camera — An IP surveillance camera"));
        assert!(prompt.contains("state: string /* active or inactive */"));
        assert!(prompt.contains("id: string"));
        assert!(!prompt.contains("id: string /*"));
    }

    #[test]
    fn query_prompt_keeps_found_entities_and_two_hop_neighbors() {
        #[derive(Debug)]
        struct KeywordEmbedder;

        impl Embedder for KeywordEmbedder {
            fn dim(&self) -> usize {
                2
            }

            fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbedError> {
                Ok(texts
                    .iter()
                    .map(|text| {
                        let lower = text.to_ascii_lowercase();
                        if lower.contains("camera") || lower.contains("surveillance") {
                            vec![1.0, 0.0]
                        } else if lower.contains("invoice") || lower.contains("billing") {
                            vec![0.0, 1.0]
                        } else {
                            vec![0.0, 0.0]
                        }
                    })
                    .collect())
            }
        }

        let schema = GraphSchema {
            nodes: vec![
                NodeKind {
                    label: "Camera".into(),
                    properties: vec![],
                },
                NodeKind {
                    label: "Site".into(),
                    properties: vec![],
                },
                NodeKind {
                    label: "Company".into(),
                    properties: vec![],
                },
                NodeKind {
                    label: "User".into(),
                    properties: vec![],
                },
                NodeKind {
                    label: "Invoice".into(),
                    properties: vec![],
                },
            ],
            relationships: vec![
                RelKind {
                    label: "INSTALLED_AT".into(),
                    from: Some("Camera".into()),
                    to: Some("Site".into()),
                    properties: vec![],
                },
                RelKind {
                    label: "OWNED_BY".into(),
                    from: Some("Site".into()),
                    to: Some("Company".into()),
                    properties: vec![],
                },
                RelKind {
                    label: "HAS_USER".into(),
                    from: Some("Company".into()),
                    to: Some("User".into()),
                    properties: vec![],
                },
                RelKind {
                    label: "BILLED_BY".into(),
                    from: Some("Invoice".into()),
                    to: Some("Company".into()),
                    properties: vec![],
                },
            ],
        };
        let embedder = Arc::new(KeywordEmbedder);
        let mut spec = crate::graph::GraphSpecification::new()
            .with_entity("Camera", "A surveillance camera")
            .with_entity("Invoice", "A billing invoice");
        spec.compute(embedder.as_ref()).unwrap();

        let prompt = generate_query_prompt(
            "camera status",
            &schema,
            &PromptOptions {
                graph_specification: Some(spec),
                embedding_model: Some(embedder),
                schema_selection: PromptSchemaSelection {
                    entity_match_threshold: 0.9,
                    related_entity_hops: 2,
                    reranking_threshold: 0.0,
                },
                include_examples: false,
                ..PromptOptions::default()
            },
        );

        assert!(prompt.contains("Camera"));
        assert!(prompt.contains("Site"));
        assert!(prompt.contains("Company"));
        assert!(prompt.contains("(Camera)-[:INSTALLED_AT]->(Site)"));
        assert!(prompt.contains("(Site)-[:OWNED_BY]->(Company)"));
        assert!(!prompt.contains("User"));
        assert!(!prompt.contains("Invoice"));
        assert!(!prompt.contains("HAS_USER"));
        assert!(!prompt.contains("BILLED_BY"));
    }
}
