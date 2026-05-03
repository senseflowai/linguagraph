//! Render a system prompt from a [`GraphSchema`].
//!
//! The output is plain text suitable for any chat-completion endpoint. We
//! never embed examples that would leak provider-specific markers — the
//! prompt is a portable contract.

use std::fmt::Write;

use super::schema::{GraphSchema, NodeKind, Property, RelKind};
use crate::metadata::PropertyMetadata;
use crate::types::TypeRegistry;

#[derive(Debug, Clone)]
pub struct PromptOptions {
    /// Optional task framing line, e.g. "You are a graph query planner."
    pub preamble: Option<String>,
    /// If true, include 1-2 worked examples after the rules.
    pub include_examples: bool,
    /// Optional per-property descriptions. When provided, each schema entry
    /// whose key matches `<NodeLabel>.<property>` (or `<NodeLabel>` for the
    /// node itself) is annotated inline.
    pub property_metadata: Option<PropertyMetadata>,
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
            property_metadata: None,
            type_registry: None,
        }
    }
}

pub fn generate_system_prompt(schema: &GraphSchema, opts: &PromptOptions) -> String {
    let mut out = String::with_capacity(2048);

    if let Some(p) = &opts.preamble {
        out.push_str(p);
        out.push_str("\n\n");
    }

    out.push_str("# Graph schema\n");
    write_nodes(&mut out, &schema.nodes, opts.property_metadata.as_ref());
    write_rels(&mut out, &schema.relationships, opts.property_metadata.as_ref());

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

fn write_nodes(out: &mut String, nodes: &[NodeKind], meta: Option<&PropertyMetadata>) {
    if nodes.is_empty() {
        out.push_str("(no node labels declared)\n");
        return;
    }
    out.push_str("Nodes:\n");
    for n in nodes {
        let header_desc = meta.and_then(|m| m.get(&n.label));
        let _ = match header_desc {
            Some(d) => writeln!(
                out,
                "  - {} — {}{}",
                n.label,
                d,
                render_props(&n.label, &n.properties, meta)
            ),
            None => writeln!(
                out,
                "  - {}{}",
                n.label,
                render_props(&n.label, &n.properties, meta)
            ),
        };
    }
}

fn write_rels(out: &mut String, rels: &[RelKind], meta: Option<&PropertyMetadata>) {
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
            render_props(&r.label, &r.properties, meta)
        );
    }
}

fn render_props(
    owner: &str,
    props: &[Property],
    meta: Option<&PropertyMetadata>,
) -> String {
    if props.is_empty() {
        return String::new();
    }
    let inner: Vec<String> = props
        .iter()
        .map(|p| {
            let base = format!("{}: {}", p.name, format_ty(p.ty));
            match meta.and_then(|m| m.get(&format!("{owner}.{}", p.name))) {
                Some(d) => format!("{base} /* {d} */"),
                None => base,
            }
        })
        .collect();
    format!(" {{ {} }}", inner.join(", "))
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
    use crate::prompt::schema::PropertyType as PT;

    #[test]
    fn includes_schema_and_rules() {
        let schema = GraphSchema {
            nodes: vec![NodeKind {
                label: "Person".into(),
                properties: vec![
                    Property { name: "name".into(), ty: PT::String },
                    Property { name: "age".into(), ty: PT::Int },
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
    fn property_metadata_annotates_schema_block() {
        let schema = GraphSchema {
            nodes: vec![NodeKind {
                label: "Camera".into(),
                properties: vec![
                    Property { name: "id".into(), ty: PT::String },
                    Property { name: "state".into(), ty: PT::String },
                ],
            }],
            relationships: vec![],
        };
        let mut meta = PropertyMetadata::new();
        meta.insert("Camera", "An IP surveillance camera");
        meta.insert("Camera.state", "active or inactive");
        let prompt = generate_system_prompt(
            &schema,
            &PromptOptions {
                property_metadata: Some(meta),
                include_examples: false,
                ..PromptOptions::default()
            },
        );
        assert!(prompt.contains("Camera — An IP surveillance camera"));
        assert!(prompt.contains("state: string /* active or inactive */"));
        assert!(prompt.contains("id: string"));
        assert!(!prompt.contains("id: string /*"));
    }
}
