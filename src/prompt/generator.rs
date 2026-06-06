//! Render a system prompt from a [`GraphSchema`].
//!
//! The output is plain text suitable for any chat-completion endpoint. We
//! never embed examples that would leak provider-specific markers — the
//! prompt is a portable contract.

use std::fmt::Write;

use super::schema::{GraphSchema, NodeKind, Property, RelKind};
use crate::embeddings::{SharedEmbedder, SharedReranker};
use crate::graph::{OntologyCatalog, OntologyPropertyType, PropertySpec};
use crate::types::TypeRegistry;

/// Properties added by the senseflow ingestion pipeline that carry no
/// semantic meaning for DSL query construction and should be hidden from
/// the schema block shown to the LLM.
const SCHEMA_HIDDEN_PROPS: &[&str] = &["entity_id", "primary_key"];

#[derive(Debug, Clone)]
pub struct PromptSchemaSelection {
    /// Maximum graph-schema relationship hops to include around the
    /// entities matched by the query.
    pub related_entity_hops: usize,
    /// Minimum cosine score accepted from [`OntologyCatalog::find`].
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
    /// Optional ontology catalog. When provided, descriptions and
    /// SemanticText markers are emitted next to each node, relationship,
    /// and property in the rendered schema block.
    pub ontology_catalog: Option<OntologyCatalog>,
    /// Embedder used to match the user's query against the catalog.
    /// When omitted, the full schema is rendered.
    pub embedding_model: Option<SharedEmbedder>,
    /// Reranker applied after embedding retrieval. When omitted, embedding
    /// retrieval scores are used directly.
    pub reranking_model: Option<SharedReranker>,
    /// Query-specific schema selection controls.
    pub schema_selection: PromptSchemaSelection,
    /// Registered field types whose capabilities should be advertised
    /// to the LLM. When `None`, the prompt only describes plain ops.
    pub type_registry: Option<TypeRegistry>,
    /// Authoritative entity labels to seed schema selection. When set,
    /// [`select_query_schema`] skips the [`OntologyCatalog::find`] hop
    /// entirely and expands relationships around these labels instead.
    /// Use this when the caller has out-of-band evidence (e.g. a vector
    /// search against actual entity data) for which labels are relevant.
    pub pinned_labels: Option<Vec<String>>,
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
            ontology_catalog: None,
            embedding_model: None,
            reranking_model: None,
            schema_selection: PromptSchemaSelection::default(),
            type_registry: None,
            pinned_labels: None,
        }
    }
}

pub fn generate_system_prompt(schema: &GraphSchema, opts: &PromptOptions) -> String {
    render_prompt(schema, opts)
}

pub fn generate_query_prompt(query: &str, schema: &GraphSchema, opts: &PromptOptions) -> String {
    let selected_schema = select_query_schema(query, schema, opts);
    render_prompt(&selected_schema, opts)
}

fn render_prompt(schema: &GraphSchema, opts: &PromptOptions) -> String {
    let mut out = String::with_capacity(2048);

    if let Some(p) = &opts.preamble {
        out.push_str(p);
        out.push_str("\n\n");
    }

    out.push_str("# Graph schema\n");
    write_nodes(&mut out, &schema.nodes, opts.ontology_catalog.as_ref());
    write_rels(
        &mut out,
        &schema.relationships,
        opts.ontology_catalog.as_ref(),
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
/// The default strategy uses [`OntologyCatalog::find`] to seed relevant
/// entity labels, then expands through schema relationships according to
/// [`PromptSchemaSelection`]. Callers can use this directly when they need
/// the selected schema separately from prompt rendering.
pub fn select_query_schema(query: &str, schema: &GraphSchema, opts: &PromptOptions) -> GraphSchema {
    // Caller-pinned labels bypass the catalog-find hop entirely. We
    // still run the same relationship-expansion loop so the prompt
    // includes the natural neighborhood of the pinned types.
    let seed_labels: Option<std::collections::BTreeSet<String>> = match &opts.pinned_labels {
        Some(labels) if !labels.is_empty() => Some(labels.iter().cloned().collect()),
        _ => None,
    };

    let mut labels = match seed_labels {
        Some(seed) => seed,
        None => {
            let Some(catalog) = opts.ontology_catalog.as_ref() else {
                return schema.clone();
            };
            let Some(embedder) = opts.embedding_model.as_deref() else {
                return schema.clone();
            };
            if query.trim().is_empty() {
                return schema.clone();
            }

            let Ok(matches) = catalog.find(
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

            matches
                .into_iter()
                .map(|m| m.entity_type.name.clone())
                .collect()
        }
    };

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

fn write_nodes(out: &mut String, nodes: &[NodeKind], catalog: Option<&OntologyCatalog>) {
    if nodes.is_empty() {
        out.push_str("(no node labels declared)\n");
        return;
    }
    out.push_str("Nodes:\n");
    for n in nodes {
        let header_desc = n.description.as_deref().or_else(|| {
            catalog.and_then(|c| {
                c.get_entity(&n.label)
                    .and_then(|(_, e)| e.description.as_deref())
            })
        });
        let _ = match header_desc {
            Some(d) => writeln!(
                out,
                "  - {} — {}{}",
                n.label,
                d,
                render_props(&n.label, n.domain.as_deref(), &n.properties, catalog)
            ),
            None => writeln!(
                out,
                "  - {}{}",
                n.label,
                render_props(&n.label, n.domain.as_deref(), &n.properties, catalog)
            ),
        };
    }
}

fn write_rels(out: &mut String, rels: &[RelKind], catalog: Option<&OntologyCatalog>) {
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
        let header_desc = r.description.as_deref().or_else(|| {
            catalog.and_then(|c| {
                c.get_relation(&r.label)
                    .and_then(|(_, spec)| spec.description.as_deref())
            })
        });
        let tail = render_props(&r.label, r.domain.as_deref(), &r.properties, catalog);
        let _ = match header_desc {
            Some(d) => writeln!(out, "  - {} — {}{}", endpoints, d, tail),
            None => writeln!(out, "  - {}{}", endpoints, tail),
        };
    }
}

fn render_props(
    owner: &str,
    domain: Option<&str>,
    props: &[Property],
    catalog: Option<&OntologyCatalog>,
) -> String {
    if props.is_empty() {
        return String::new();
    }
    let inner: Vec<String> = props
        .iter()
        .filter(|p| !SCHEMA_HIDDEN_PROPS.contains(&p.name.as_str()))
        .map(|p| {
            let property_spec: Option<&PropertySpec> = catalog.and_then(|c| match domain {
                Some(d) => c.get_property_in(d, owner, &p.name),
                None => c.get_property(owner, &p.name),
            });
            // Property header shape:
            //   <name>: <scalar-ty>                       (untyped, undocumented)
            //   <name>: <scalar-ty> @<FieldType>           (typed, e.g. SemanticText)
            //   <name>: <scalar-ty> /* description */      (documented only)
            //   <name>: <scalar-ty> @<FieldType> /* … */   (both)
            let mut base = format!("{}: {}", p.name, format_ty(p.ty));
            if let Some(ty) = property_spec.and_then(field_type_marker) {
                base = format!("{base} @{ty}");
            }
            let desc = p
                .description
                .as_deref()
                .or_else(|| property_spec.and_then(|p| p.description.as_deref()))
                .filter(|d| !d.is_empty());
            if let Some(d) = desc {
                base = format!("{base} /* {d} */");
            }
            base
        })
        .collect();
    if inner.is_empty() {
        return String::new();
    }
    format!(" {{ {} }}", inner.join(", "))
}

fn field_type_marker(spec: &PropertySpec) -> Option<&'static str> {
    match spec.property_type {
        OntologyPropertyType::Text => Some("SemanticText"),
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
  "action": "find" | "aggregate",  // optional; inferred from `return`
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
    { "field": "<alias>.<prop>", "alias": <ident>, "date_part": "year|quarter|month|day|hour" },
    { "aggregate": "count|sum|avg|min|max", "field": "<alias>[.<prop>]", "alias": <ident> }
  ],
  "group_by": [ "<alias>.<prop>" | { "field": "<alias>.<datetime_prop>", "date_part": "year|quarter|month|day|hour", "alias": <ident> } ],
  "sort":     [ { "field": <alias-or-projected>, "order": "asc"|"desc" } ],
  "limit":    <int>
}

Constraints:
- Use only labels and properties listed in the schema above.
- Every alias — the start node, each traversal edge, and each traversal target —
  must be unique across the whole query. Nodes and edges share one namespace, so
  an edge and the node it points at can never reuse the same alias.
- `action` is optional and only a legacy hint; the engine infers aggregate queries
  from aggregate items in `return`, so prefer omitting `action` if unsure.
- Queries that mix aggregated and non-aggregated columns must list the
  non-aggregated columns in `group_by`.
- For timestamp/date aggregation like "by year" or "monthly", do not group by
  the raw datetime field. Use the object form, e.g.
  `{ "field": "c.created_at", "date_part": "year", "alias": "created_year" }`,
  and use the same object form in `return` when the bucket should be visible; sort by that alias.
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
                domain: None,
                extra_labels: Vec::new(),
                scopes: Vec::new(),
                description: None,
                properties: vec![
                    Property {
                        name: "name".into(),
                        ty: PT::String,
                        description: None,
                    },
                    Property {
                        name: "age".into(),
                        ty: PT::Int,
                        description: None,
                    },
                ],
            }],
            relationships: vec![RelKind {
                label: "KNOWS".into(),
                domain: None,
                description: None,
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
                domain: None,
                extra_labels: Vec::new(),
                scopes: Vec::new(),
                description: None,
                properties: vec![
                    Property {
                        name: "id".into(),
                        ty: PT::String,
                        description: None,
                    },
                    Property {
                        name: "state".into(),
                        ty: PT::String,
                        description: None,
                    },
                ],
            }],
            relationships: vec![],
        };
        let mut catalog = crate::graph::OntologyCatalog::default();
        catalog.insert(
            "test",
            crate::graph::DomainOntology {
                entity_types: vec![crate::graph::EntityTypeSpec {
                    name: "Camera".into(),
                    description: Some("An IP surveillance camera".into()),
                    properties: vec![crate::graph::PropertySpec {
                        name: "state".into(),
                        description: Some("active or inactive".into()),
                        property_type: crate::graph::OntologyPropertyType::String,
                        required: false,
                    }],
                    embedding: None,
                }],
                relation_types: vec![],
            },
        );
        let prompt = generate_system_prompt(
            &schema,
            &PromptOptions {
                ontology_catalog: Some(catalog),
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
                    domain: None,
                    extra_labels: Vec::new(),
                    scopes: Vec::new(),
                    description: None,
                    properties: vec![],
                },
                NodeKind {
                    label: "Site".into(),
                    domain: None,
                    extra_labels: Vec::new(),
                    scopes: Vec::new(),
                    description: None,
                    properties: vec![],
                },
                NodeKind {
                    label: "Company".into(),
                    domain: None,
                    extra_labels: Vec::new(),
                    scopes: Vec::new(),
                    description: None,
                    properties: vec![],
                },
                NodeKind {
                    label: "User".into(),
                    domain: None,
                    extra_labels: Vec::new(),
                    scopes: Vec::new(),
                    description: None,
                    properties: vec![],
                },
                NodeKind {
                    label: "Invoice".into(),
                    domain: None,
                    extra_labels: Vec::new(),
                    scopes: Vec::new(),
                    description: None,
                    properties: vec![],
                },
            ],
            relationships: vec![
                RelKind {
                    label: "INSTALLED_AT".into(),
                    domain: None,
                    description: None,
                    from: Some("Camera".into()),
                    to: Some("Site".into()),
                    properties: vec![],
                },
                RelKind {
                    label: "OWNED_BY".into(),
                    domain: None,
                    description: None,
                    from: Some("Site".into()),
                    to: Some("Company".into()),
                    properties: vec![],
                },
                RelKind {
                    label: "HAS_USER".into(),
                    domain: None,
                    description: None,
                    from: Some("Company".into()),
                    to: Some("User".into()),
                    properties: vec![],
                },
                RelKind {
                    label: "BILLED_BY".into(),
                    domain: None,
                    description: None,
                    from: Some("Invoice".into()),
                    to: Some("Company".into()),
                    properties: vec![],
                },
            ],
        };
        let embedder = Arc::new(KeywordEmbedder);
        let mut catalog = crate::graph::OntologyCatalog::default();
        catalog.insert(
            "test",
            crate::graph::DomainOntology {
                entity_types: vec![
                    crate::graph::EntityTypeSpec::with_description(
                        "Camera",
                        "A surveillance camera",
                    ),
                    crate::graph::EntityTypeSpec::with_description("Invoice", "A billing invoice"),
                ],
                relation_types: vec![],
            },
        );
        catalog.compute(embedder.as_ref()).unwrap();

        let prompt = generate_query_prompt(
            "camera status",
            &schema,
            &PromptOptions {
                ontology_catalog: Some(catalog),
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

    #[test]
    fn system_properties_are_excluded_from_schema_prompt() {
        let schema = GraphSchema {
            nodes: vec![NodeKind {
                label: "Document".into(),
                domain: None,
                extra_labels: Vec::new(),
                scopes: Vec::new(),
                description: None,
                properties: vec![
                    Property {
                        name: "entity_id".into(),
                        ty: PT::String,
                        description: None,
                    },
                    Property {
                        name: "primary_key".into(),
                        ty: PT::String,
                        description: None,
                    },
                    Property {
                        name: "title".into(),
                        ty: PT::String,
                        description: None,
                    },
                    Property {
                        name: "created_at".into(),
                        ty: PT::Datetime,
                        description: None,
                    },
                    // Generic "id" field from a user-defined schema is NOT hidden.
                    Property {
                        name: "doc_number".into(),
                        ty: PT::String,
                        description: None,
                    },
                ],
            }],
            relationships: vec![],
        };
        let prompt = generate_system_prompt(&schema, &PromptOptions::default());
        assert!(prompt.contains("title"), "user property must appear");
        assert!(prompt.contains("created_at"), "user property must appear");
        assert!(prompt.contains("doc_number"), "user property must appear");
        assert!(!prompt.contains("entity_id"), "entity_id must be excluded");
        assert!(
            !prompt.contains("primary_key"),
            "primary_key must be excluded"
        );
    }
}
