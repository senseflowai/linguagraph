//! `linguagraph explore` — the graph-browser CLI.
//!
//! Scriptable subcommands over [`crate::explore::Explorer`]. Every
//! subcommand supports `--format json` (the exact DTO the library
//! returns — the contract a downstream UI consumes) and a human `table`
//! rendering (default).

use std::path::PathBuf;
use std::sync::Arc;

use clap::{Args, Subcommand, ValueEnum};
use tabled::{builder::Builder, settings::Style};
use tokio::fs;

use crate::config;
use crate::error::Result;
use crate::explore::{
    AskOptions, AskResult, EntityCard, EntityTable, ExploreError, Explorer, NeighborOptions,
    OverviewReport, PageOptions, RelDirection, SearchMode, SearchOptions, SearchResult, Subgraph,
    TimelineEvent,
};
use crate::llm::LlmClient;
use crate::nl::NlTranslator;

use super::commands::{
    build_embedding_store, build_ontology_catalog_embedder, build_query_pipeline,
};

#[derive(Debug, Clone, Copy, ValueEnum, Default, PartialEq, Eq)]
pub enum OutputFormat {
    /// Human-readable tables and cards.
    #[default]
    Table,
    /// The raw explorer DTO as pretty JSON.
    Json,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum DirectionArg {
    Out,
    In,
    Both,
}

impl DirectionArg {
    fn into_dto(self) -> Option<RelDirection> {
        match self {
            DirectionArg::Out => Some(RelDirection::Out),
            DirectionArg::In => Some(RelDirection::In),
            DirectionArg::Both => None,
        }
    }
}

#[derive(Debug, Clone, Copy, ValueEnum, Default)]
pub enum SearchModeArg {
    #[default]
    Auto,
    Keyword,
    Semantic,
}

impl SearchModeArg {
    fn into_dto(self) -> SearchMode {
        match self {
            SearchModeArg::Auto => SearchMode::Auto,
            SearchModeArg::Keyword => SearchMode::Keyword,
            SearchModeArg::Semantic => SearchMode::Semantic,
        }
    }
}

#[derive(Args, Debug)]
pub struct ExploreArgs {
    /// Optional Cypher label appended to every node pattern so the
    /// explorer only sees entities ingested under the same
    /// `prefix_label`.
    #[arg(long, global = true)]
    pub prefix_label: Option<String>,
    /// Optional prefix folded into embedding-index / Qdrant collection
    /// names. Must match the ingest-side `prefix_index`.
    #[arg(long, global = true)]
    pub prefix_index: Option<String>,
    /// Output format.
    #[arg(long, global = true, value_enum, default_value_t)]
    pub format: OutputFormat,

    #[command(subcommand)]
    pub command: ExploreCommand,
}

#[derive(Subcommand, Debug)]
pub enum ExploreCommand {
    /// Ask a natural-language question (requires the `openai` feature
    /// and `[llm]` config).
    Ask {
        question: String,
        /// Also synthesize a natural-language answer with the LLM.
        #[arg(long)]
        answer: bool,
        /// Skip subgraph materialization (rows + trace only).
        #[arg(long)]
        no_subgraph: bool,
        /// Print the generated DSL and Cypher ("how was this answered").
        #[arg(long)]
        show_cypher: bool,
    },
    /// Execute a hand-written DSL JSON file through the ask flow
    /// (subgraph + trace + sources, no LLM translation).
    RunDsl {
        path: PathBuf,
        /// Print the executed DSL and Cypher.
        #[arg(long)]
        show_cypher: bool,
    },
    /// Inspect one entity: properties, provenance, relation summary.
    Entity {
        /// Entity id (the `id` property, or a `_nid:<n>` handle).
        id: String,
    },
    /// Walk one hop from an entity.
    Neighbors {
        id: String,
        /// Keep only these edge types (repeatable). Built-in
        /// `mention`/`part_of` edges are excluded unless named.
        #[arg(long = "edge-type")]
        edge_types: Vec<String>,
        /// Keep only neighbors carrying one of these labels (repeatable).
        #[arg(long = "target-label")]
        target_labels: Vec<String>,
        #[arg(long, value_enum, default_value_t = DirectionArg::Both)]
        direction: DirectionArg,
        #[arg(long)]
        limit: Option<u32>,
        #[arg(long, default_value_t = 0)]
        offset: u32,
    },
    /// Find entities by text.
    Search {
        text: String,
        /// Restrict to one entity type.
        #[arg(long = "type")]
        entity_type: Option<String>,
        #[arg(long, value_enum, default_value_t)]
        mode: SearchModeArg,
        /// Exact value match instead of substring.
        #[arg(long)]
        exact: bool,
        #[arg(long)]
        limit: Option<u32>,
    },
    /// Entity/relation types with counts, totals, and sources.
    Overview,
    /// One page of entities of a type.
    Table {
        entity_type: String,
        #[arg(long)]
        limit: Option<u32>,
        #[arg(long, default_value_t = 0)]
        offset: u32,
        /// Property to sort by (default `name`).
        #[arg(long)]
        sort: Option<String>,
    },
    /// Dated events extracted from a type's Datetime properties.
    Timeline {
        entity_type: String,
        #[arg(long)]
        limit: Option<u32>,
    },
    /// Export a subgraph as GraphBuilder-compatible JSON.
    Export {
        /// Export every entity of this type (plus edges among them).
        #[arg(long = "type", conflicts_with = "entity")]
        entity_type: Option<String>,
        /// Export one entity's 1-hop neighborhood.
        #[arg(long)]
        entity: Option<String>,
        #[arg(long)]
        limit: Option<u32>,
        /// Write to a file instead of stdout.
        #[arg(short, long)]
        output: Option<PathBuf>,
    },
    /// Interactive shell with navigation state (current node, trail).
    Repl,
}

pub async fn cmd_explore(config_path: &std::path::Path, args: ExploreArgs) -> Result<()> {
    let cfg = config::load(config_path).await?;
    let pipeline =
        build_query_pipeline(&cfg, args.prefix_label.clone(), args.prefix_index.clone()).await?;
    let mut explorer = Explorer::new(pipeline);
    if needs_translator(&args.command) {
        match build_translator(&cfg) {
            Ok(translator) => explorer = explorer.with_translator(Arc::new(translator)),
            // The REPL is still useful for browsing without an LLM;
            // `ask` inside it will explain what's missing.
            Err(err) if matches!(args.command, ExploreCommand::Repl) => {
                eprintln!("warning: natural-language `ask` disabled: {err}");
            }
            Err(err) => return Err(err),
        }
    }
    let format = args.format;

    match args.command {
        ExploreCommand::Ask {
            question,
            answer,
            no_subgraph,
            show_cypher,
        } => {
            let opts = AskOptions {
                synthesize_answer: answer,
                include_subgraph: !no_subgraph,
                ..Default::default()
            };
            let result = explorer.ask(&question, &opts).await?;
            emit(format, &result, |r| render_ask(r, show_cypher))
        }
        ExploreCommand::RunDsl { path, show_cypher } => {
            let dsl = crate::dsl::parse(&path).await?;
            let result = explorer.run_dsl(dsl, &AskOptions::default()).await?;
            emit(format, &result, |r| render_ask(r, show_cypher))
        }
        ExploreCommand::Entity { id } => match explorer.entity(&id).await? {
            Some(card) => emit(format, &card, render_entity_card),
            None => {
                println!("entity `{id}` not found");
                Ok(())
            }
        },
        ExploreCommand::Neighbors {
            id,
            edge_types,
            target_labels,
            direction,
            limit,
            offset,
        } => {
            let opts = NeighborOptions {
                edge_types: (!edge_types.is_empty()).then_some(edge_types),
                target_labels: (!target_labels.is_empty()).then_some(target_labels),
                direction: direction.into_dto(),
                limit,
                offset,
            };
            let subgraph = explorer.neighbors(&id, &opts).await?;
            emit(format, &subgraph, render_subgraph)
        }
        ExploreCommand::Search {
            text,
            entity_type,
            mode,
            exact,
            limit,
        } => {
            let opts = SearchOptions {
                entity_type,
                mode: mode.into_dto(),
                exact,
                limit,
            };
            let found = explorer.search(&text, &opts).await?;
            emit(format, &found, render_search)
        }
        ExploreCommand::Overview => {
            let overview = explorer.overview().await?;
            emit(format, &overview, render_overview)
        }
        ExploreCommand::Table {
            entity_type,
            limit,
            offset,
            sort,
        } => {
            let page = PageOptions {
                limit,
                offset,
                sort_by: sort,
            };
            let table = explorer.entities_of_type(&entity_type, &page).await?;
            emit(format, &table, render_entity_table)
        }
        ExploreCommand::Timeline { entity_type, limit } => {
            let page = PageOptions {
                limit,
                ..Default::default()
            };
            let events = explorer.timeline_for_type(&entity_type, &page).await?;
            emit(format, &events, |e| render_timeline(e))
        }
        ExploreCommand::Export {
            entity_type,
            entity,
            limit,
            output,
        } => {
            let subgraph = match (entity, entity_type) {
                (Some(id), None) => {
                    explorer
                        .neighbors(
                            &id,
                            &NeighborOptions {
                                limit,
                                ..Default::default()
                            },
                        )
                        .await?
                }
                (None, Some(t)) => {
                    explorer
                        .subgraph_of_type(
                            &t,
                            &PageOptions {
                                limit,
                                ..Default::default()
                            },
                        )
                        .await?
                }
                _ => {
                    return Err(ExploreError::UnknownEntity(
                        "pass exactly one of --entity <ID> or --type <TYPE>".to_string(),
                    )
                    .into())
                }
            };
            let doc = explorer.export(&subgraph);
            let rendered = serde_json::to_string_pretty(&doc.0)?;
            match output {
                Some(path) => {
                    fs::write(&path, rendered).await?;
                    println!(
                        "exported {} node(s), {} edge(s) to {}",
                        subgraph.nodes.len(),
                        subgraph.edges.len(),
                        path.display()
                    );
                }
                None => println!("{rendered}"),
            }
            Ok(())
        }
        ExploreCommand::Repl => super::repl::run_repl(explorer, format).await,
    }
}

fn needs_translator(command: &ExploreCommand) -> bool {
    matches!(
        command,
        ExploreCommand::Ask { .. } | ExploreCommand::Repl
    )
}

#[cfg(feature = "openai")]
fn build_translator(cfg: &config::Config) -> Result<NlTranslator> {
    let llm: Arc<dyn LlmClient> = Arc::new(crate::llm::OpenAiClient::from_config(&cfg.llm));
    let embedder = build_ontology_catalog_embedder(cfg)?;
    let store = build_embedding_store(cfg)?;
    Ok(NlTranslator::from_config(cfg, llm, embedder, store))
}

#[cfg(not(feature = "openai"))]
fn build_translator(_cfg: &config::Config) -> Result<NlTranslator> {
    Err(crate::error::Error::Llm(crate::llm::LlmError::Http(
        "the `openai` feature is disabled; rebuild with `--features openai` to use `explore ask`"
            .to_string(),
    )))
}

/// Print `value` as JSON or via the human renderer.
fn emit<T: serde::Serialize>(
    format: OutputFormat,
    value: &T,
    render: impl Fn(&T) -> String,
) -> Result<()> {
    match format {
        OutputFormat::Json => println!("{}", serde_json::to_string_pretty(value)?),
        OutputFormat::Table => println!("{}", render(value)),
    }
    Ok(())
}

// ── Human renderers ─────────────────────────────────────────────────────

fn ascii_table(header: Vec<&str>, rows: Vec<Vec<String>>) -> String {
    let mut builder = Builder::default();
    builder.push_record(header);
    for row in rows {
        builder.push_record(row);
    }
    builder.build().with(Style::ascii()).to_string()
}

fn compact(value: &serde_json::Value) -> String {
    match value {
        serde_json::Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

pub(super) fn render_entity_card(card: &EntityCard) -> String {
    let node = &card.node;
    let mut out = format!("{} [{}]  id={}\n", node.name, node.entity_type, node.id);
    if node.ephemeral_handle {
        out.push_str("(session-scoped handle — the node has no `id` property)\n");
    }
    if let Some(confidence) = node.confidence {
        out.push_str(&format!("confidence: {confidence}\n"));
    }

    let mut section = |title: &str, entries: &std::collections::BTreeMap<String, serde_json::Value>| {
        if entries.is_empty() {
            return;
        }
        out.push_str(&format!("\n{title}:\n"));
        for (name, value) in entries {
            out.push_str(&format!("  {name} = {}\n", compact(value)));
        }
    };
    section("identifiers", &node.properties.identifiers);
    section("descriptions", &node.properties.descriptions);
    section("facts", &node.properties.facts);
    section("dates", &node.properties.dates);
    section("other", &node.properties.other);

    if !card.sources.is_empty() {
        out.push_str("\nsources:\n");
        for source in &card.sources {
            out.push_str(&format!(
                "  {}{}\n",
                source.name.as_deref().unwrap_or("(unnamed)"),
                source
                    .id
                    .as_deref()
                    .map(|id| format!("  [{id}]"))
                    .unwrap_or_default()
            ));
        }
    }

    if !card.relations.is_empty() {
        out.push_str(&format!("\nrelations ({}):\n", card.relations.len()));
        let rows = card
            .relations
            .iter()
            .map(|r| {
                let arrow = match r.direction {
                    RelDirection::Out => "→",
                    RelDirection::In => "←",
                };
                vec![
                    format!("{arrow} {}", r.edge_type),
                    r.neighbor_type.clone(),
                    r.count.to_string(),
                ]
            })
            .collect();
        out.push_str(&ascii_table(vec!["relation", "neighbor type", "count"], rows));
    }
    out
}

pub(super) fn render_subgraph(subgraph: &Subgraph) -> String {
    let mut out = String::new();
    let node_rows = subgraph
        .nodes
        .iter()
        .map(|n| {
            vec![
                n.id.clone(),
                n.name.clone(),
                n.entity_type.clone(),
                n.confidence.map(|c| c.to_string()).unwrap_or_default(),
            ]
        })
        .collect();
    out.push_str(&format!("nodes ({}):\n", subgraph.nodes.len()));
    out.push_str(&ascii_table(vec!["id", "name", "type", "confidence"], node_rows));

    if !subgraph.edges.is_empty() {
        let edge_rows = subgraph
            .edges
            .iter()
            .map(|e| {
                let props = if e.properties.is_empty() {
                    String::new()
                } else {
                    serde_json::to_string(&e.properties).unwrap_or_default()
                };
                vec![e.from.clone(), e.edge_type.clone(), e.to.clone(), props]
            })
            .collect();
        out.push_str(&format!("\nedges ({}):\n", subgraph.edges.len()));
        out.push_str(&ascii_table(vec!["from", "relation", "to", "properties"], edge_rows));
    }
    if subgraph.truncated {
        out.push_str("\n(truncated — more data exists beyond the configured limits)");
    }
    out
}

pub(super) fn render_search(found: &SearchResult) -> String {
    if found.hits.is_empty() {
        return "no matches".to_string();
    }
    let rows = found
        .hits
        .iter()
        .map(|hit| {
            vec![
                hit.node.id.clone(),
                hit.node.name.clone(),
                hit.node.entity_type.clone(),
                hit.score.map(|s| format!("{s:.3}")).unwrap_or_default(),
                format!("{:?}", hit.channel).to_lowercase(),
            ]
        })
        .collect();
    let mut out = ascii_table(vec!["id", "name", "type", "score", "channel"], rows);
    if !found.related_types.is_empty() {
        out.push_str("\nrelated types: ");
        out.push_str(
            &found
                .related_types
                .iter()
                .map(|t| format!("{} ({})", t.name, t.count))
                .collect::<Vec<_>>()
                .join(", "),
        );
    }
    out
}

pub(super) fn render_overview(overview: &OverviewReport) -> String {
    let mut out = format!(
        "{} entities, {} relations\n\nentity types:\n",
        overview.total_entities, overview.total_relations
    );
    let entity_rows = overview
        .entity_types
        .iter()
        .map(|t| {
            vec![
                t.name.clone(),
                t.count.to_string(),
                t.description.clone().unwrap_or_default(),
            ]
        })
        .collect();
    out.push_str(&ascii_table(vec!["type", "count", "description"], entity_rows));

    if !overview.relation_types.is_empty() {
        let rel_rows = overview
            .relation_types
            .iter()
            .map(|r| {
                let endpoints = match (&r.from, &r.to) {
                    (Some(from), Some(to)) => format!("{from} → {to}"),
                    _ => String::new(),
                };
                vec![r.name.clone(), r.count.to_string(), endpoints]
            })
            .collect();
        out.push_str("\nrelation types:\n");
        out.push_str(&ascii_table(vec!["type", "count", "endpoints"], rel_rows));
    }

    if !overview.sources.is_empty() {
        out.push_str(&format!("\nsources ({}):\n", overview.sources.len()));
        for source in &overview.sources {
            out.push_str(&format!(
                "  {}\n",
                source.name.as_deref().unwrap_or("(unnamed)")
            ));
        }
    }
    out
}

pub(super) fn render_entity_table(table: &EntityTable) -> String {
    let columns: Vec<&str> = table.key_columns.iter().map(String::as_str).collect();
    let rows = table
        .rows
        .iter()
        .map(|node| {
            columns
                .iter()
                .map(|column| match *column {
                    "id" => node.id.clone(),
                    "name" => node.name.clone(),
                    other => node
                        .properties
                        .iter()
                        .find(|(k, _)| k.as_str() == other)
                        .map(|(_, v)| compact(v))
                        .unwrap_or_default(),
                })
                .collect()
        })
        .collect();
    let mut out = ascii_table(columns.clone(), rows);
    out.push_str(&format!(
        "{} of {} {} row(s), offset {}",
        table.rows.len(),
        table.total,
        table.entity_type,
        table.offset
    ));
    out
}

pub(super) fn render_timeline(events: &Vec<TimelineEvent>) -> String {
    if events.is_empty() {
        return "no dated events".to_string();
    }
    let rows = events
        .iter()
        .map(|e| {
            vec![
                e.date.clone(),
                e.property.clone(),
                e.entity_name.clone(),
                e.entity_type.clone(),
                e.entity_id.clone(),
            ]
        })
        .collect();
    ascii_table(vec!["date", "property", "entity", "type", "id"], rows)
}

pub(super) fn render_ask(result: &AskResult, show_cypher: bool) -> String {
    let mut out = String::new();
    if let Some(answer) = &result.answer {
        out.push_str(&format!("answer:\n{answer}\n\n"));
    }

    out.push_str(&format!("rows ({}):\n", result.table.rows.len()));
    if result.table.rows.is_empty() {
        out.push_str("(no rows)\n");
    } else {
        let columns: Vec<&str> = result.table.columns.iter().map(String::as_str).collect();
        let rows = result
            .table
            .rows
            .iter()
            .map(|row| {
                columns
                    .iter()
                    .map(|c| row.get(*c).map(compact).unwrap_or_default())
                    .collect()
            })
            .collect();
        out.push_str(&ascii_table(columns, rows));
        out.push('\n');
    }

    if !result.subgraph.is_empty() {
        out.push('\n');
        out.push_str(&render_subgraph(&result.subgraph));
        out.push('\n');
    }

    if !result.sources.is_empty() {
        out.push_str(&format!(
            "\nsources: {}\n",
            result
                .sources
                .iter()
                .map(|s| s.name.clone().or_else(|| s.id.clone()).unwrap_or_default())
                .collect::<Vec<_>>()
                .join(", ")
        ));
    }

    out.push_str(&format!(
        "\nquery: {} ({} ms{})",
        result.trace.dsl_summary,
        result.trace.elapsed_ms,
        if result.trace.llm_attempts > 0 {
            format!(", {} LLM attempt(s)", result.trace.llm_attempts)
        } else {
            String::new()
        }
    ));
    if show_cypher {
        out.push_str(&format!(
            "\n\n-- DSL --\n{}\n\n-- Cypher --\n{}\n\n-- Parameters --\n{}",
            serde_json::to_string_pretty(&result.trace.dsl).unwrap_or_default(),
            result.trace.cypher,
            serde_json::to_string_pretty(&result.trace.cypher_params).unwrap_or_default(),
        ));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::explore::{EdgeView, NodeView, PropertyGroups, QueryTrace, TableSlice};
    use serde_json::json;
    use std::collections::BTreeMap;

    fn node(id: &str, name: &str, ty: &str) -> NodeView {
        NodeView {
            id: id.into(),
            name: name.into(),
            entity_type: ty.into(),
            labels: vec![ty.into()],
            properties: PropertyGroups::default(),
            confidence: None,
            ephemeral_handle: false,
        }
    }

    #[test]
    fn render_subgraph_lists_nodes_edges_and_truncation() {
        let subgraph = Subgraph {
            nodes: vec![node("p1", "Keanu Reeves", "Person"), node("m1", "The Matrix", "Movie")],
            edges: vec![EdgeView {
                id: "p1:ACTED_IN:m1".into(),
                edge_type: "ACTED_IN".into(),
                from: "p1".into(),
                to: "m1".into(),
                properties: BTreeMap::new(),
                confidence: None,
            }],
            truncated: true,
        };
        let out = render_subgraph(&subgraph);
        assert!(out.contains("nodes (2)"));
        assert!(out.contains("Keanu Reeves"));
        assert!(out.contains("edges (1)"));
        assert!(out.contains("ACTED_IN"));
        assert!(out.contains("truncated"));
    }

    #[test]
    fn render_ask_shows_answer_rows_and_trace() {
        let result = AskResult {
            trace: QueryTrace {
                question: Some("who?".into()),
                dsl: json!({}),
                dsl_summary: "Selecting Person entities.".into(),
                cypher: "MATCH (p:Person) RETURN p.name".into(),
                cypher_params: BTreeMap::new(),
                llm_attempts: 1,
                elapsed_ms: 12,
            },
            table: TableSlice {
                columns: vec!["name".into()],
                rows: vec![BTreeMap::from([("name".into(), json!("Keanu Reeves"))])],
            },
            subgraph: Subgraph::default(),
            sources: Vec::new(),
            answer: Some("Keanu Reeves.".into()),
        };
        let plain = render_ask(&result, false);
        assert!(plain.contains("answer:\nKeanu Reeves."));
        assert!(plain.contains("Selecting Person entities."));
        assert!(plain.contains("1 LLM attempt(s)"));
        assert!(!plain.contains("-- Cypher --"));

        let verbose = render_ask(&result, true);
        assert!(verbose.contains("-- Cypher --"));
        assert!(verbose.contains("MATCH (p:Person)"));
    }
}
