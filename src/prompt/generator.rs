//! Render a system prompt from a [`GraphSchema`].
//!
//! The output is plain text suitable for any chat-completion endpoint. We
//! never embed examples that would leak provider-specific markers — the
//! prompt is a portable contract.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write;

use super::is_enum_candidate_property_name;
use super::schema::{GraphSchema, NodeKind, Property, RelKind};
use super::select::{select_query_schema, QuerySelectionParams};
use crate::embeddings::{EmbedError, Embedder, EmbeddingIndex, SharedEmbedder, SharedReranker};
use crate::graph::{
    OntologyCatalog, OntologyPropertyType, PropertySpec, DEFAULT_DOMAIN_SELECTION_THRESHOLD,
    DEFAULT_DOMAIN_SELECTION_TOP_K,
};
use crate::types::TypeRegistry;

/// Properties added by the senseflow ingestion pipeline that carry no
/// semantic meaning for DSL query construction and should be hidden from
/// the schema block shown to the LLM. Also excluded from the per-property
/// embedding text built during query-driven schema selection.
pub(crate) const SCHEMA_HIDDEN_PROPS: &[&str] = &["_canonical", "entity_id", "primary_key"];

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
                "You translate user questions into Linguagraph JSON DSL.\n\
                 Output exactly one JSON object. No prose, no markdown, no code fences."
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

/// Controls for the query-driven compact prompt built by
/// [`generate_query_prompt`].
#[derive(Debug, Clone)]
pub struct QueryPromptParams {
    /// Minimum cosine score for a domain to be routed to (see
    /// [`OntologyCatalog::select_domains`]).
    pub domain_threshold: f32,
    /// Maximum number of domains kept by routing.
    pub domain_top_k: usize,
    /// Entity/property-level selection tunables.
    pub selection: QuerySelectionParams,
    /// Include worked DSL examples after the rules.
    pub include_examples: bool,
}

impl Default for QueryPromptParams {
    fn default() -> Self {
        Self {
            domain_threshold: DEFAULT_DOMAIN_SELECTION_THRESHOLD,
            domain_top_k: DEFAULT_DOMAIN_SELECTION_TOP_K,
            selection: QuerySelectionParams::default(),
            include_examples: true,
        }
    }
}

/// Build a **compact** DSL-generation prompt tailored to `query`.
///
/// The full pipeline, behind one call:
/// 1. project the live `schema` onto `catalog` (in place) and split it per
///    domain;
/// 2. route to the top-k domains relevant to `query` by embedding
///    similarity;
/// 3. within those domains, select the entities, properties (with their
///    value sets) and 1-hop neighbours the query needs;
/// 4. render the prompt from that narrowed schema.
///
/// Domain and entity/property embeddings are stored in and searched from
/// `index` (Qdrant in production, in-memory in tests); only the query
/// embedding is computed here. Reuses [`OntologyCatalog::project_schema`],
/// [`OntologyCatalog::split_schema_by_domain`],
/// [`OntologyCatalog::select_domains`] and [`select_query_schema`].
pub async fn generate_query_prompt(
    query: &str,
    schema: &mut GraphSchema,
    catalog: &OntologyCatalog,
    embedder: &dyn Embedder,
    index: &EmbeddingIndex<'_>,
    params: &QueryPromptParams,
) -> Result<String, EmbedError> {
    OntologyCatalog::project_schema(schema, catalog.all_domains());
    let domain_schemas = OntologyCatalog::split_schema_by_domain(schema);

    let candidates: BTreeSet<String> = domain_schemas.keys().cloned().collect();
    let matches = catalog
        .select_domains(
            query,
            params.domain_threshold,
            params.domain_top_k,
            Some(&candidates),
            embedder,
            index,
        )
        .await?;
    let selected: BTreeMap<String, GraphSchema> = matches
        .iter()
        .filter_map(|matched| {
            domain_schemas
                .get(matched.domain)
                .map(|schema| (matched.domain.to_string(), schema.clone()))
        })
        .collect();

    let narrowed =
        select_query_schema(query, &selected, embedder, index, &params.selection).await?;

    let opts = PromptOptions {
        include_examples: params.include_examples,
        ..PromptOptions::default()
    };
    Ok(render_prompt(&narrowed, &opts))
}

fn render_prompt(schema: &GraphSchema, opts: &PromptOptions) -> String {
    let mut out = String::with_capacity(2048);

    if let Some(p) = &opts.preamble {
        out.push_str(p);
        out.push_str("\n\n");
    }

    let aliases = assign_node_aliases(&schema.nodes);
    write_schema(
        &mut out,
        &schema.nodes,
        opts.ontology_catalog.as_ref(),
        &aliases,
    );
    write_paths(
        &mut out,
        &schema.relationships,
        opts.ontology_catalog.as_ref(),
        &aliases,
    );

    if let Some(reg) = &opts.type_registry {
        write_field_types(&mut out, reg);
    }

    out.push_str("\n# OUTPUT SHAPE\n");
    out.push_str(OUTPUT_SHAPE);

    out.push_str("\n# RULES\n");
    out.push_str(RULES);

    if opts.include_examples {
        out.push_str("\n# Examples\n");
        out.push_str(EXAMPLES);
    }

    out
}

/// Assign a stable, short, unique alias to each node, threaded through the
/// schema and paths blocks so the model copies them verbatim. The alias is
/// the shortest lowercase alphanumeric prefix of the label that is still
/// free (`Listing`→`l`, `Category`→`c`, `Country`→`co`, `AssetType`→`a`).
fn assign_node_aliases(nodes: &[NodeKind]) -> BTreeMap<String, String> {
    let mut used: BTreeSet<String> = BTreeSet::new();
    let mut map = BTreeMap::new();
    for (i, node) in nodes.iter().enumerate() {
        let letters: Vec<char> = node
            .label
            .chars()
            .filter(|c| c.is_alphanumeric())
            .flat_map(|c| c.to_lowercase())
            .collect();
        let mut alias = None;
        for len in 1..=letters.len() {
            let cand: String = letters[..len].iter().collect();
            if !used.contains(&cand) {
                alias = Some(cand);
                break;
            }
        }
        // Empty/degenerate label, or every prefix taken: fall back to n{i}.
        let mut alias = alias.unwrap_or_else(|| format!("n{i}"));
        while used.contains(&alias) {
            alias.push('_');
        }
        used.insert(alias.clone());
        map.insert(node.label.clone(), alias);
    }
    map
}

/// The `# SCHEMA` block: a header, the flags legend, and one block per node
/// (`Label (alias) — description`, aligned property rows, and inline enum
/// values under each enum property).
fn write_schema(
    out: &mut String,
    nodes: &[NodeKind],
    catalog: Option<&OntologyCatalog>,
    aliases: &BTreeMap<String, String>,
) {
    out.push_str("# SCHEMA\n\n");
    out.push_str("Nodes retrieved for this question. Aliases are pre-assigned — use as printed.\n");
    out.push_str(
        "Flags: [enum] closed value set — use only a listed value, copied as printed\n       \
         [free-text] not normalized — never sort or compare it directly. `eq`, `contains`, \
         and `search` all retrieve by meaning, not by literal string match; see rule 8 for \
         which one to pick and when to add `cardinality`\n\n",
    );
    if nodes.is_empty() {
        out.push_str("(no node labels declared)\n");
        return;
    }

    for node in nodes {
        let alias = aliases.get(&node.label).map(String::as_str).unwrap_or("?");
        let header_desc = node.description.as_deref().or_else(|| {
            catalog.and_then(|c| {
                c.get_entity(&node.label)
                    .and_then(|(_, e)| e.description.as_deref())
            })
        });
        match header_desc {
            Some(d) => {
                let _ = writeln!(out, "{} ({}) — {}", node.label, alias, d);
            }
            None => {
                let _ = writeln!(out, "{} ({})", node.label, alias);
            }
        }

        // Collect visible property rows, then align the name/type columns.
        let visible: Vec<&Property> = node
            .properties
            .iter()
            .filter(|p| !SCHEMA_HIDDEN_PROPS.contains(&p.name.as_str()))
            .collect();
        let name_w = visible.iter().map(|p| p.name.len()).max().unwrap_or(0);
        let type_w = visible
            .iter()
            .map(|p| {
                render_type(
                    p,
                    property_spec(catalog, node.domain.as_deref(), &node.label, &p.name),
                )
                .len()
            })
            .max()
            .unwrap_or(0);

        for prop in &visible {
            let spec = property_spec(catalog, node.domain.as_deref(), &node.label, &prop.name);
            let ty = render_type(prop, spec);
            let flags = property_flags(prop, spec);
            let desc = prop
                .description
                .as_deref()
                .or_else(|| spec.and_then(|s| s.description.as_deref()))
                .filter(|d| !d.is_empty())
                .unwrap_or("");
            let mut tail = String::new();
            if !flags.is_empty() {
                tail.push_str(&flags);
                tail.push(' ');
            }
            tail.push_str(desc);
            let _ = writeln!(
                out,
                "  {:<name_w$}  {:<type_w$}  {}",
                prop.name,
                ty,
                tail.trim_end()
            );
            let values = effective_allowed_values(prop, spec);
            if !values.is_empty() {
                let _ = writeln!(out, "    {}", values.join(" | "));
            }
        }
        out.push('\n');
    }
}

/// The `# PATHS` block: one pre-aliased, pre-directed path per relationship
/// with known endpoints. The model copies these verbatim instead of
/// composing a traversal.
fn write_paths(
    out: &mut String,
    rels: &[RelKind],
    catalog: Option<&OntologyCatalog>,
    aliases: &BTreeMap<String, String>,
) {
    // Edge aliases live in the same namespace as node aliases.
    let mut used: BTreeSet<String> = aliases.values().cloned().collect();
    let mut lines: Vec<String> = Vec::new();
    for rel in rels {
        let (Some(from), Some(to)) = (rel.from.as_deref(), rel.to.as_deref()) else {
            continue;
        };
        let (Some(fa), Some(ta)) = (aliases.get(from), aliases.get(to)) else {
            continue;
        };
        let base = format!("e_{ta}");
        let mut edge = base.clone();
        let mut n = 2;
        while used.contains(&edge) {
            edge = format!("{base}_{n}");
            n += 1;
        }
        used.insert(edge.clone());

        let desc = rel.description.as_deref().or_else(|| {
            catalog.and_then(|c| {
                c.get_relation(&rel.label)
                    .and_then(|(_, spec)| spec.description.as_deref())
            })
        });
        let mut line = format!("  ({fa})-[{edge}:{}]->({ta})", rel.label);
        // Advertise the relationship's own properties (e.g. `ACTED_IN` has
        // `roles`, `REVIEWED` has `rating`/`summary`) so the model filters /
        // projects them on the edge alias instead of guessing they live on
        // an endpoint node.
        let edge_props: Vec<String> = rel
            .properties
            .iter()
            .filter(|p| !SCHEMA_HIDDEN_PROPS.contains(&p.name.as_str()))
            .map(|p| {
                let spec =
                    property_spec(catalog, rel.domain.as_deref(), &rel.label, &p.name);
                format!("{}: {}", p.name, render_type(p, spec))
            })
            .collect();
        if !edge_props.is_empty() {
            line.push_str(&format!("  {{{}}}", edge_props.join(", ")));
        }
        if let Some(d) = desc.filter(|d| !d.is_empty()) {
            line.push_str(&format!("  — {d}"));
        }
        lines.push(line);
    }

    out.push_str("\n# PATHS\n\n");
    if lines.is_empty() {
        out.push_str("(no relationships between the selected nodes)\n");
        return;
    }
    out.push_str("Copy verbatim. Aliases are pre-bound. Never re-derive direction.\n");
    out.push_str("Include a path only if you filter on it or return from it.\n\n");
    for line in lines {
        out.push_str(&line);
        out.push('\n');
    }
}

/// Look up the ontology `PropertySpec` for a property, honouring the node's
/// domain when known.
fn property_spec<'a>(
    catalog: Option<&'a OntologyCatalog>,
    domain: Option<&str>,
    owner: &str,
    prop: &str,
) -> Option<&'a PropertySpec> {
    catalog.and_then(|c| match domain {
        Some(d) => c.get_property_in(d, owner, prop),
        None => c.get_property(owner, prop),
    })
}

/// Rendered type for a property: `text` for an ontology free-text field,
/// otherwise the introspected scalar type (`string`/`number`/…).
fn render_type(prop: &Property, spec: Option<&PropertySpec>) -> &'static str {
    if matches!(
        spec.map(|s| s.property_type),
        Some(OntologyPropertyType::Text)
    ) {
        "text"
    } else {
        prop.ty.as_str()
    }
}

/// Space-separated flag markers for a property: `[free-text]` for an
/// ontology `Text` field, `[enum]` for a field with a closed value set.
fn property_flags(prop: &Property, spec: Option<&PropertySpec>) -> String {
    let mut flags: Vec<&str> = Vec::new();
    if matches!(
        spec.map(|s| s.property_type),
        Some(OntologyPropertyType::Text)
    ) {
        flags.push("[free-text]");
    }
    if !effective_allowed_values(prop, spec).is_empty() {
        flags.push("[enum]");
    }
    flags.join(" ")
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

/// Effective enum vocabulary for a property: the union of the value set
/// discovered by introspection (carried on the [`Property`]) and any
/// hand-declared vocabulary on the ontology [`PropertySpec`]. Canonical
/// (lowercase), sorted, deduped. Empty ⇒ the field is not enum-like and
/// gets neither the `enum` marker nor an entry in the enumerations block.
fn effective_allowed_values(prop: &Property, spec: Option<&PropertySpec>) -> Vec<String> {
    if !is_enum_candidate_property_name(&prop.name) {
        return Vec::new();
    }
    let mut out: Vec<String> = prop
        .allowed_values
        .iter()
        .map(|s| s.to_lowercase())
        .collect();
    if let Some(spec) = spec {
        out.extend(spec.allowed_values.iter().map(|s| s.to_lowercase()));
    }
    out.sort();
    out.dedup();
    out
}

const OUTPUT_SHAPE: &str = r#"{
  "start":  { "label": "<Label>", "alias": "<alias>" },
  "traversals": [
    { "from":   "<alias>",
      "edge":   { "label": "<REL>", "alias": "<alias>", "direction": "out" },
      "target": { "label": "<Label>", "alias": "<alias>" },
      "depth":  { "min": <int>, "max": <int> } }        // depth optional
  ],
  "filters": [
    { "field": "<alias>.<prop>", "op": "eq|neq|gt|gte|lt|lte|in|contains|starts_with|ends_with",
      "value": <scalar or array>,
      "cardinality": "one|many" }               // [free-text] fields only, see rule 8; omit elsewhere
  ],
  "return": [
    { "field": "<alias>.<prop>", "alias": "<name>" },
    { "field": "<alias>.<datetime_prop>", "date_part": "year|quarter|month|day|hour", "alias": "<name>" },
    { "aggregate": "count|sum|avg|min|max", "field": "<alias>[.<prop>]", "alias": "<name>" }
  ],
  "group_by": [ "<alias>.<prop>" | { "field": "<alias>.<datetime_prop>", "date_part": "year|quarter|month|day|hour", "alias": "<name>" } ],
  "sort":     [ { "field": "<alias>.<prop> or projected alias", "order": "asc|desc" } ],
  "limit":    <int>
}

Omit any key you don't use. To traverse, copy a path from # PATHS verbatim — its
aliases and direction are already correct; then set traversals[*].from to the alias
you start from.
"#;

const RULES: &str = r#"1. Use only the labels, properties, paths, and enum values printed above.
   Never invent a field, label, path, or enum value.
2. Copy enum values character-for-character, lowercase, as printed. If a user term
   has no exact match and no obvious synonym in the list, do not substitute the
   nearest value — omit that filter.
3. User-supplied values go only into filters[*].value, never into a field name.
4. Prefer a direct property over a traversal when both express the same thing.
5. Never sort or range-compare a [free-text] field or a `list` field. For a
   [free-text] field use `contains` (semantic match); for a `list` field use
   `contains` (element membership). A `list` value has no order, so it must
   never appear in `sort` or in `eq`/`gt`/`lt`-style comparisons.
6. Superlatives -> sort, never a fabricated filter. "top", "highest", "best",
   "largest", "most", "newest", "latest", "youngest" -> desc; "cheapest",
   "lowest", "smallest", "fewest", "oldest", "earliest" -> asc. For a person's
   birth year, "youngest" = largest year -> desc, "oldest" = smallest -> asc.
   Never invent a filter to express a superlative (e.g. do NOT write
   `born = 0` for "oldest") — sort alone.
   Set `limit` only when the question names a count: "top N" / "first N" /
   "N X" -> limit N; a lone superlative ("the most", "which X has the highest")
   -> limit 1. Otherwise OMIT `limit` — every matching row is returned (a
   server-side safety cap bounds the result). Never add a default limit.
7. Aggregation: every non-aggregated column in `return` must also appear in
   `group_by`; sort by the projected alias. For "by year"/"monthly", use the
   date_part object form in both `return` and `group_by`, not the raw datetime.
8. [free-text] filters retrieve by meaning regardless of `op` — `op` only picks
   the matching style, it does NOT say how many rows you expect. State that
   separately with `cardinality`:
   - The question names ONE specific thing — a person, a company, an id, or a
     whole description that identifies a single record ("which warehouse's
     description mentions cold chain and retail shipments") -> `cardinality: "one"`.
   - The question asks for a set — "show/find/list all/every X that mentions
     Y", "which reviews say Z" -> `cardinality: "many"`.
   - Genuinely unsure -> omit `cardinality`; it then falls back to `eq`/`neq` ->
     one, anything else -> many. Prefer stating it explicitly when the
     question's own wording (singular "which one" vs plural "all/every") makes
     the answer obvious — don't rely on the fallback for those.
"#;

const EXAMPLES: &str = r#"User: "Show me people over 30 who know someone in Berlin."
Assistant:
{
  "start": { "label": "Person", "alias": "p" },
  "traversals": [
    { "edge": { "label": "KNOWS", "alias": "r", "direction": "out" },
      "target": { "label": "Person", "alias": "friend" } }
  ],
  "filters": [
    { "field": "p.age", "op": "gt", "value": 30 },
    { "field": "friend.city", "op": "eq", "value": "Berlin" }
  ],
  "return": [{ "field": "p.name", "alias": "name" }]
}
// No count named -> no `limit`; all matching people are returned.

User: "Total spend per customer for completed orders, top 10."
Assistant:
{
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

User: "Which warehouse's description mentions cold chain and retail shipments?"
Assistant:
{
  "start": { "label": "Warehouse", "alias": "w" },
  "filters": [
    { "field": "w.description", "op": "contains",
      "value": "cold chain and retail shipments", "cardinality": "one" }
  ],
  "return": [{ "field": "w.name", "alias": "name" }],
  "limit": 1
}
// One specific warehouse is being identified by its description, even
// though the match is fuzzy — `cardinality: "one"` says so explicitly.

User: "Show me every review that mentions bad product quality."
Assistant:
{
  "start": { "label": "Review", "alias": "r" },
  "filters": [
    { "field": "r.text", "op": "contains",
      "value": "bad product quality", "cardinality": "many" }
  ],
  "return": [{ "field": "r.text", "alias": "text" }]
}
// Same op (`contains`) as above, opposite cardinality: the question asks
// for every matching row, not the single best match — and "every" names no
// count, so no `limit`.

User: "Найди клиента по описанию про складскую инфраструктуру."
Assistant:
{
  "start": { "label": "Customer", "alias": "c" },
  "filters": [
    { "field": "c.description", "op": "contains",
      "value": "складскую инфраструктуру", "cardinality": "one" }
  ],
  "return": [{ "field": "c.name", "alias": "name" }],
  "limit": 1
}
// "Найди клиента" (singular "the customer") names one specific record by
// its description — `cardinality: "one"`, not "many", even though the
// filter op is `contains`. Contrast with "Найди всех клиентов, у которых
// в описании..." (plural "all customers") — that would be "many".
"#;

#[cfg(test)]
mod tests {
    use super::*;

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
                        allowed_values: Vec::new(),
                    },
                    Property {
                        name: "age".into(),
                        ty: PT::Int,
                        description: None,
                        allowed_values: Vec::new(),
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
        assert!(prompt.contains("# SCHEMA"), "{prompt}");
        assert!(prompt.contains("Person (p)"), "{prompt}");
        assert!(prompt.contains("string"), "keyword is gone");
        assert!(!prompt.contains("keyword"), "keyword vocabulary removed");
        // A pre-aliased, pre-directed path the model copies verbatim.
        assert!(prompt.contains("(p)-[e_p:KNOWS]->(p)"), "{prompt}");
        assert!(prompt.contains("# RULES"), "{prompt}");
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
                        allowed_values: Vec::new(),
                    },
                    Property {
                        name: "state".into(),
                        ty: PT::String,
                        description: None,
                        allowed_values: Vec::new(),
                    },
                ],
            }],
            relationships: vec![],
        };
        let mut catalog = crate::graph::OntologyCatalog::default();
        catalog.insert(
            "test",
            crate::graph::DomainOntology {
                name: None,
                description: None,
                entity_types: vec![crate::graph::EntityTypeSpec {
                    name: "Camera".into(),
                    description: Some("An IP surveillance camera".into()),
                    properties: vec![crate::graph::PropertySpec {
                        name: "state".into(),
                        description: Some("active or inactive".into()),
                        property_type: crate::graph::OntologyPropertyType::Keyword,
                        required: false,
                        allowed_values: Vec::new(),
                    }],
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
        assert!(
            prompt.contains("Camera (c) — An IP surveillance camera"),
            "{prompt}"
        );
        // Description resolved from the ontology; type rendered as `string`.
        assert!(prompt.contains("active or inactive"), "{prompt}");
        assert!(prompt.contains("string"), "{prompt}");
        assert!(!prompt.contains("keyword"), "{prompt}");
    }

    #[test]
    fn enum_fields_render_inline_with_marker() {
        let schema = GraphSchema {
            nodes: vec![NodeKind {
                label: "Order".into(),
                domain: None,
                extra_labels: Vec::new(),
                scopes: Vec::new(),
                description: None,
                properties: vec![
                    Property {
                        name: "status".into(),
                        ty: PT::String,
                        description: None,
                        allowed_values: vec![
                            "pending".into(),
                            "completed".into(),
                            "cancelled".into(),
                        ],
                    },
                    // High-cardinality field: no dictionary, no marker.
                    Property {
                        name: "vin".into(),
                        ty: PT::String,
                        description: None,
                        allowed_values: Vec::new(),
                    },
                ],
            }],
            relationships: vec![],
        };
        let prompt = generate_system_prompt(
            &schema,
            &PromptOptions {
                include_examples: false,
                ..PromptOptions::default()
            },
        );

        // Enum field carries an `[enum]` marker and its values inline …
        let status_line = prompt
            .lines()
            .find(|l| l.trim_start().starts_with("status "))
            .unwrap_or("");
        assert!(status_line.contains("[enum]"), "{prompt}");
        assert!(
            prompt.contains("cancelled | completed | pending"),
            "{prompt}"
        );
        // … while the high-cardinality field is not flagged.
        let vin_line = prompt
            .lines()
            .find(|l| l.trim_start().starts_with("vin "))
            .unwrap_or("");
        assert!(!vin_line.contains("[enum]"), "{prompt}");
        // No separate `# Enum field values` block any more.
        assert!(!prompt.contains("# Enum field values"), "{prompt}");
    }

    #[test]
    fn identifier_values_do_not_render_as_enum() {
        let schema = GraphSchema {
            nodes: vec![NodeKind {
                label: "Category".into(),
                domain: None,
                extra_labels: Vec::new(),
                scopes: Vec::new(),
                description: None,
                properties: vec![
                    Property {
                        name: "id".into(),
                        ty: PT::String,
                        description: None,
                        allowed_values: vec!["cat-1".into(), "cat-2".into()],
                    },
                    Property {
                        name: "status".into(),
                        ty: PT::String,
                        description: None,
                        allowed_values: vec!["active".into(), "archived".into()],
                    },
                ],
            }],
            relationships: vec![],
        };
        let prompt = generate_system_prompt(
            &schema,
            &PromptOptions {
                include_examples: false,
                ..PromptOptions::default()
            },
        );

        let id_line = prompt
            .lines()
            .find(|l| l.trim_start().starts_with("id "))
            .unwrap_or("");
        assert!(id_line.contains("string"), "{prompt}");
        assert!(!id_line.contains("[enum]"), "{prompt}");
        assert!(!prompt.contains("Category.id:"));
        let status_line = prompt
            .lines()
            .find(|l| l.trim_start().starts_with("status "))
            .unwrap_or("");
        assert!(status_line.contains("string"), "{prompt}");
        assert!(status_line.contains("[enum]"), "{prompt}");
        assert!(prompt.contains("active | archived"), "{prompt}");
    }

    #[test]
    fn no_enum_block_when_no_enum_fields() {
        let schema = GraphSchema {
            nodes: vec![NodeKind {
                label: "Person".into(),
                domain: None,
                extra_labels: Vec::new(),
                scopes: Vec::new(),
                description: None,
                properties: vec![Property {
                    name: "name".into(),
                    ty: PT::String,
                    description: None,
                    allowed_values: Vec::new(),
                }],
            }],
            relationships: vec![],
        };
        let prompt = generate_system_prompt(&schema, &PromptOptions::default());
        assert!(!prompt.contains("# Enum field values"));
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
                        allowed_values: Vec::new(),
                    },
                    Property {
                        name: "primary_key".into(),
                        ty: PT::String,
                        description: None,
                        allowed_values: Vec::new(),
                    },
                    Property {
                        name: "_canonical".into(),
                        ty: PT::String,
                        description: None,
                        allowed_values: Vec::new(),
                    },
                    Property {
                        name: "title".into(),
                        ty: PT::String,
                        description: None,
                        allowed_values: Vec::new(),
                    },
                    Property {
                        name: "created_at".into(),
                        ty: PT::Datetime,
                        description: None,
                        allowed_values: Vec::new(),
                    },
                    // Generic "id" field from a user-defined schema is NOT hidden.
                    Property {
                        name: "doc_number".into(),
                        ty: PT::String,
                        description: None,
                        allowed_values: Vec::new(),
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
        assert!(
            !prompt.contains("_canonical"),
            "_canonical must be excluded"
        );
    }

    /// Deterministic 3-axis stub (mirrors the one in `select`): axis 0 =
    /// "listing/auction/sale/title", axis 1 = "clinic/patient/visit".
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

    fn spec(name: &str, desc: &str, prop: &str, values: &[&str]) -> crate::graph::EntityTypeSpec {
        crate::graph::EntityTypeSpec {
            name: name.into(),
            description: Some(desc.into()),
            properties: vec![crate::graph::PropertySpec {
                name: prop.into(),
                description: None,
                property_type: OntologyPropertyType::Keyword,
                required: false,
                allowed_values: values.iter().map(|v| v.to_string()).collect(),
            }],
        }
    }

    fn domain(desc: &str, entity: crate::graph::EntityTypeSpec) -> crate::graph::DomainOntology {
        crate::graph::DomainOntology {
            name: None,
            description: Some(desc.into()),
            entity_types: vec![entity],
            relation_types: vec![],
        }
    }

    fn live_node(label: &str, extra: &str, prop: &str, values: &[&str]) -> NodeKind {
        NodeKind {
            label: label.into(),
            domain: None,
            extra_labels: vec![extra.into()],
            scopes: Vec::new(),
            description: None,
            properties: vec![Property {
                name: prop.into(),
                ty: PT::String,
                description: None,
                allowed_values: values.iter().map(|v| v.to_string()).collect(),
            }],
        }
    }

    #[tokio::test]
    async fn query_prompt_is_compact_and_domain_scoped() {
        let mut catalog = OntologyCatalog::default();
        catalog.insert(
            "flippa",
            domain(
                "Online marketplace for buying and selling websites",
                spec(
                    "Listing",
                    "A marketplace listing for sale",
                    "sale_method",
                    &["auction", "classified"],
                ),
            ),
        );
        catalog.insert(
            "clinic",
            domain(
                "Healthcare clinic operations",
                spec("Patient", "A clinic patient", "visit_reason", &[]),
            ),
        );

        // Live schema carries both entities; projection binds them by the
        // domain label in `extra_labels`.
        let mut schema = GraphSchema {
            nodes: vec![
                live_node(
                    "Listing",
                    "flippa",
                    "sale_method",
                    &["auction", "classified"],
                ),
                live_node("Patient", "clinic", "visit_reason", &[]),
            ],
            relationships: vec![],
        };

        let embedder = StubEmbedder;
        let store = crate::embeddings::InMemoryEmbeddingStore::new();
        let index = EmbeddingIndex {
            store: &store,
            collection: "test",
            model: "stub",
        };
        let params = QueryPromptParams {
            domain_threshold: 0.2,
            domain_top_k: 3,
            selection: QuerySelectionParams {
                entity_threshold: 0.2,
                property_threshold: 0.2,
                ..QuerySelectionParams::default()
            },
            include_examples: false,
        };

        let prompt = generate_query_prompt(
            "auction listings for sale",
            &mut schema,
            &catalog,
            &embedder,
            &index,
            &params,
        )
        .await
        .unwrap();

        // The relevant domain's entity (with a pre-assigned alias), its
        // enum property + inline values, and the DSL contract are present …
        assert!(prompt.contains("Listing (l)"), "{prompt}");
        let sale_line = prompt
            .lines()
            .find(|l| l.trim_start().starts_with("sale_method "))
            .unwrap_or("");
        assert!(sale_line.contains("string"), "{prompt}");
        assert!(sale_line.contains("[enum]"), "{prompt}");
        assert!(prompt.contains("auction | classified"), "{prompt}");
        assert!(prompt.contains("# RULES"), "{prompt}");
        assert!(!prompt.contains("keyword"), "{prompt}");
        // … while the unrelated domain is excluded entirely.
        assert!(
            !prompt.contains("Patient"),
            "unrelated domain leaked: {prompt}"
        );
    }
}
