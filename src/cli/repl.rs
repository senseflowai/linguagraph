//! Interactive graph-walking shell behind `linguagraph explore repl`.
//!
//! Keeps navigation state between commands — the current entity, a
//! breadcrumb trail, the last numbered listing (so `open #2` works), the
//! last ask result (for `show dsl|cypher|…`) and active type/edge
//! filters — turning the Explorer API into a walkable graph browser.
//!
//! Line editing and history need the `repl` cargo feature (rustyline);
//! without it the subcommand explains how to rebuild.

#[cfg(not(feature = "repl"))]
pub(super) async fn run_repl(
    _explorer: crate::explore::Explorer,
    _format: super::explore::OutputFormat,
) -> crate::error::Result<()> {
    Err(crate::error::Error::Io(std::io::Error::other(
        "the `repl` feature is disabled; rebuild with `--features repl`",
    )))
}

#[cfg(feature = "repl")]
pub(super) use imp::run_repl;

#[cfg(feature = "repl")]
mod imp {
    use std::path::PathBuf;

    use rustyline::error::ReadlineError;
    use rustyline::DefaultEditor;

    use crate::error::Result;
    use crate::explore::{
        AskOptions, AskResult, Explorer, NeighborOptions, NodeView, PageOptions, RelDirection,
        RelationSummary, SearchOptions, Subgraph,
    };

    use super::super::explore::{
        render_ask, render_entity_card, render_entity_table, render_overview, render_search,
        render_subgraph, render_timeline, OutputFormat,
    };

    const HELP: &str = "\
commands:
  ask <question>          ask in natural language (needs LLM config)
  open <id | #n>          inspect an entity (by id or last listing number)
  ls                      relations of the current entity, numbered
  go <n | relation | type>  walk from the current entity: a group number
                          from `ls`, an edge type (LOCATED_IN), or a
                          neighbor entity type (Listing)
  back                    return to the previous entity on the trail
  trail                   show the breadcrumb path
  search <text>           find entities by text
  types                   dataset overview (entity/relation types)
  table <Type> [offset]   list entities of a type
  timeline                dated events of the last shown subgraph
  filter type <A,B|clear> restrict `go`/`search` to entity types
  filter edge <X,Y|clear> restrict `go` to edge types
  filter                  show active filters
  show dsl|cypher|params|sources|answer   inspect the last ask
  export <path>           write the last subgraph as GraphBuilder JSON
  format table|json       switch output format
  help                    this help
  quit / exit             leave";

    /// One parsed input line. Pure data so the parser is unit-testable.
    #[derive(Debug, Clone, PartialEq)]
    pub(crate) enum ReplCommand {
        Ask(String),
        Open(String),
        Ls,
        Go(String),
        Back,
        Trail,
        Search(String),
        Types,
        Table { entity_type: String, offset: u32 },
        Timeline,
        FilterShow,
        FilterType(Option<Vec<String>>),
        FilterEdge(Option<Vec<String>>),
        FilterClear,
        Show(String),
        Export(PathBuf),
        Format(OutputFormat),
        Help,
        Quit,
        Empty,
        Unknown(String),
    }

    /// Strip one pair of matching surrounding quotes. There is no shell
    /// in front of the REPL, so `search "Fishing-Stuff"` would otherwise
    /// look for the quote characters literally.
    fn unquote(raw: &str) -> &str {
        let bytes = raw.as_bytes();
        if raw.len() >= 2 {
            let (first, last) = (bytes[0], bytes[raw.len() - 1]);
            if first == last && (first == b'"' || first == b'\'') {
                return &raw[1..raw.len() - 1];
            }
        }
        raw
    }

    pub(crate) fn parse(line: &str) -> ReplCommand {
        let line = line.trim();
        if line.is_empty() {
            return ReplCommand::Empty;
        }
        let (head, rest) = match line.split_once(char::is_whitespace) {
            Some((head, rest)) => (head, rest.trim()),
            None => (line, ""),
        };
        let arg = || unquote(rest).to_string();
        match head.to_ascii_lowercase().as_str() {
            "ask" if !rest.is_empty() => ReplCommand::Ask(arg()),
            "open" if !rest.is_empty() => ReplCommand::Open(arg()),
            "ls" => ReplCommand::Ls,
            // Bare `go` walks every (filter-permitted) relation.
            "go" => ReplCommand::Go(arg()),
            "back" => ReplCommand::Back,
            "trail" => ReplCommand::Trail,
            "search" if !rest.is_empty() => ReplCommand::Search(arg()),
            "types" | "overview" => ReplCommand::Types,
            "table" if !rest.is_empty() => {
                let mut parts = rest.split_whitespace();
                let entity_type = parts.next().unwrap_or_default().to_string();
                let offset = parts.next().and_then(|p| p.parse().ok()).unwrap_or(0);
                ReplCommand::Table {
                    entity_type,
                    offset,
                }
            }
            "timeline" => ReplCommand::Timeline,
            "filter" => parse_filter(rest),
            "show" if !rest.is_empty() => ReplCommand::Show(rest.to_ascii_lowercase()),
            "export" if !rest.is_empty() => ReplCommand::Export(PathBuf::from(unquote(rest))),
            "format" => match rest.to_ascii_lowercase().as_str() {
                "json" => ReplCommand::Format(OutputFormat::Json),
                "table" => ReplCommand::Format(OutputFormat::Table),
                _ => ReplCommand::Unknown(format!("format {rest}")),
            },
            "help" | "?" => ReplCommand::Help,
            "quit" | "exit" | "q" => ReplCommand::Quit,
            _ => ReplCommand::Unknown(line.to_string()),
        }
    }

    fn parse_filter(rest: &str) -> ReplCommand {
        let (kind, values) = match rest.split_once(char::is_whitespace) {
            Some((kind, values)) => (kind, values.trim()),
            None => (rest, ""),
        };
        let list = |values: &str| -> Option<Vec<String>> {
            let items: Vec<String> = values
                .split([',', ' '])
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(str::to_string)
                .collect();
            (!items.is_empty()).then_some(items)
        };
        match kind.to_ascii_lowercase().as_str() {
            "" => ReplCommand::FilterShow,
            "clear" => ReplCommand::FilterClear,
            "type" if values.eq_ignore_ascii_case("clear") => ReplCommand::FilterType(None),
            "edge" if values.eq_ignore_ascii_case("clear") => ReplCommand::FilterEdge(None),
            "type" => ReplCommand::FilterType(list(values)),
            "edge" => ReplCommand::FilterEdge(list(values)),
            other => ReplCommand::Unknown(format!("filter {other}")),
        }
    }

    /// What a `go <arg>` argument names.
    #[derive(Debug, Clone, PartialEq)]
    enum GoIntent {
        /// Bare `go` — every relation the session filters permit.
        All,
        /// A number from the last `ls` listing.
        Group(RelationSummary),
        /// An edge type known from the current entity's relations.
        EdgeType(String),
        /// A neighbor entity type known from the current entity's
        /// relations ("go Listing" from a Country).
        NeighborType(String),
        /// Not recognizable from the relation summary — tried as an edge
        /// type first, retried as a label if that matches nothing.
        Unknown(String),
    }

    /// Classify a `go` argument against the current relation summary,
    /// case-insensitively; the canonical casing from the graph wins.
    fn resolve_go(target: &str, relations: &[RelationSummary]) -> GoIntent {
        if target.is_empty() {
            return GoIntent::All;
        }
        if let Some(relation) = target
            .parse::<usize>()
            .ok()
            .and_then(|n| relations.get(n.saturating_sub(1)))
        {
            return GoIntent::Group(relation.clone());
        }
        if let Some(relation) = relations
            .iter()
            .find(|r| r.edge_type.eq_ignore_ascii_case(target))
        {
            return GoIntent::EdgeType(relation.edge_type.clone());
        }
        if let Some(relation) = relations
            .iter()
            .find(|r| r.neighbor_type.eq_ignore_ascii_case(target))
        {
            return GoIntent::NeighborType(relation.neighbor_type.clone());
        }
        GoIntent::Unknown(target.to_string())
    }

    /// A breadcrumb step.
    #[derive(Debug, Clone)]
    struct TrailEntry {
        id: String,
        name: String,
        entity_type: String,
    }

    /// A numbered listing entry `open #n` resolves against.
    #[derive(Debug, Clone)]
    struct ListEntry {
        id: String,
        name: String,
    }

    #[derive(Debug, Default)]
    struct ReplState {
        current: Option<NodeView>,
        trail: Vec<TrailEntry>,
        last_ask: Option<AskResult>,
        last_list: Vec<ListEntry>,
        last_relations: Vec<RelationSummary>,
        last_subgraph: Option<Subgraph>,
        type_filter: Option<Vec<String>>,
        edge_filter: Option<Vec<String>>,
    }

    fn history_path() -> Option<PathBuf> {
        std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".linguagraph_history"))
    }

    pub(in super::super) async fn run_repl(explorer: Explorer, format: OutputFormat) -> Result<()> {
        let mut editor = DefaultEditor::new()
            .map_err(|e| std::io::Error::other(format!("readline init: {e}")))?;
        let history = history_path();
        if let Some(path) = &history {
            let _ = editor.load_history(path);
        }

        println!("linguagraph explorer — `help` for commands, `quit` to leave");
        let mut state = ReplState::default();
        let mut format = format;

        loop {
            let prompt = match &state.current {
                Some(node) => format!("{} [{}]> ", node.name, node.entity_type),
                None => "explore> ".to_string(),
            };
            let line = tokio::task::block_in_place(|| editor.readline(&prompt));
            let line = match line {
                Ok(line) => line,
                Err(ReadlineError::Interrupted) | Err(ReadlineError::Eof) => break,
                Err(e) => {
                    eprintln!("readline error: {e}");
                    break;
                }
            };
            if !line.trim().is_empty() {
                let _ = editor.add_history_entry(line.trim());
            }

            let command = parse(&line);
            if command == ReplCommand::Quit {
                break;
            }
            if let ReplCommand::Format(new_format) = command {
                format = new_format;
                println!("output format: {new_format:?}");
                continue;
            }
            if let Err(err) = step(&explorer, &mut state, format, command).await {
                eprintln!("error: {err}");
            }
        }

        if let Some(path) = &history {
            let _ = editor.save_history(path);
        }
        Ok(())
    }

    /// Execute one command against the explorer, mutating REPL state.
    async fn step(
        explorer: &Explorer,
        state: &mut ReplState,
        format: OutputFormat,
        command: ReplCommand,
    ) -> Result<()> {
        match command {
            ReplCommand::Empty | ReplCommand::Quit | ReplCommand::Format(_) => {}
            ReplCommand::Help => println!("{HELP}"),
            ReplCommand::Unknown(line) => {
                println!("unrecognized: `{line}` — `help` lists commands")
            }

            ReplCommand::Ask(question) => {
                let opts = AskOptions {
                    synthesize_answer: explorer_has_translator(explorer),
                    ..Default::default()
                };
                let result = explorer.ask(&question, &opts).await?;
                print_value(format, &result, |r| render_ask(r, false));
                state.last_list = result
                    .subgraph
                    .nodes
                    .iter()
                    .map(|n| ListEntry {
                        id: n.id.clone(),
                        name: n.name.clone(),
                    })
                    .collect();
                print_listing(&state.last_list);
                state.last_subgraph = Some(result.subgraph.clone());
                state.last_ask = Some(result);
            }

            ReplCommand::Open(target) => {
                let id = match resolve_target(&target, &state.last_list) {
                    Ok(id) => id,
                    Err(message) => {
                        println!("{message}");
                        return Ok(());
                    }
                };
                open_entity(explorer, state, format, &id, true).await?;
            }

            ReplCommand::Ls => match &state.current {
                None => println!("no current entity — `open <id>` first"),
                Some(_) => {
                    if state.last_relations.is_empty() {
                        println!("no relations");
                    }
                    for (i, relation) in state.last_relations.iter().enumerate() {
                        let arrow = match relation.direction {
                            RelDirection::Out => "→",
                            RelDirection::In => "←",
                        };
                        println!(
                            "  {}. {arrow} {}  {} ({})",
                            i + 1,
                            relation.edge_type,
                            relation.neighbor_type,
                            relation.count
                        );
                    }
                }
            },

            ReplCommand::Go(target) => {
                let Some(current) = state.current.clone() else {
                    println!("no current entity — `open <id>` first");
                    return Ok(());
                };
                let intent = resolve_go(&target, &state.last_relations);
                let opts = match &intent {
                    GoIntent::All => NeighborOptions {
                        edge_types: state.edge_filter.clone(),
                        target_labels: state.type_filter.clone(),
                        ..Default::default()
                    },
                    GoIntent::Group(relation) => NeighborOptions {
                        edge_types: Some(vec![relation.edge_type.clone()]),
                        target_labels: state.type_filter.clone(),
                        direction: Some(relation.direction),
                        ..Default::default()
                    },
                    GoIntent::EdgeType(name) | GoIntent::Unknown(name) => NeighborOptions {
                        edge_types: Some(vec![name.clone()]),
                        target_labels: state.type_filter.clone(),
                        ..Default::default()
                    },
                    GoIntent::NeighborType(label) => NeighborOptions {
                        edge_types: state.edge_filter.clone(),
                        target_labels: Some(vec![label.clone()]),
                        ..Default::default()
                    },
                };
                let mut subgraph = explorer.neighbors(&current.id, &opts).await?;
                // A name we couldn't classify that matched no edges may
                // be an entity type ("go Listing") — retry as a label.
                if let GoIntent::Unknown(name) = &intent {
                    if subgraph.nodes.len() <= 1 {
                        let retry = NeighborOptions {
                            edge_types: state.edge_filter.clone(),
                            target_labels: Some(vec![name.clone()]),
                            ..Default::default()
                        };
                        let by_label = explorer.neighbors(&current.id, &retry).await?;
                        if by_label.nodes.len() > 1 {
                            subgraph = by_label;
                        }
                    }
                }
                print_value(format, &subgraph, render_subgraph);
                state.last_list = subgraph
                    .nodes
                    .iter()
                    .filter(|n| n.id != current.id)
                    .map(|n| ListEntry {
                        id: n.id.clone(),
                        name: n.name.clone(),
                    })
                    .collect();
                print_listing(&state.last_list);
                state.last_subgraph = Some(subgraph);
            }

            ReplCommand::Back => {
                if state.trail.len() < 2 {
                    println!("trail is empty");
                } else {
                    state.trail.pop();
                    let previous = state.trail.last().cloned();
                    if let Some(previous) = previous {
                        open_entity(explorer, state, format, &previous.id, false).await?;
                    }
                }
            }

            ReplCommand::Trail => {
                if state.trail.is_empty() {
                    println!("trail is empty");
                } else {
                    let path = state
                        .trail
                        .iter()
                        .map(|t| format!("{} [{}]", t.name, t.entity_type))
                        .collect::<Vec<_>>()
                        .join(" > ");
                    println!("{path}");
                }
            }

            ReplCommand::Search(text) => {
                let opts = SearchOptions {
                    entity_type: state
                        .type_filter
                        .as_ref()
                        .and_then(|types| types.first().cloned()),
                    ..Default::default()
                };
                let found = explorer.search(&text, &opts).await?;
                print_value(format, &found, render_search);
                state.last_list = found
                    .hits
                    .iter()
                    .map(|hit| ListEntry {
                        id: hit.node.id.clone(),
                        name: hit.node.name.clone(),
                    })
                    .collect();
                print_listing(&state.last_list);
            }

            ReplCommand::Types => {
                let overview = explorer.overview().await?;
                print_value(format, &overview, render_overview);
            }

            ReplCommand::Table {
                entity_type,
                offset,
            } => {
                let page = PageOptions {
                    offset,
                    ..Default::default()
                };
                let table = explorer.entities_of_type(&entity_type, &page).await?;
                print_value(format, &table, render_entity_table);
                state.last_list = table
                    .rows
                    .iter()
                    .map(|n| ListEntry {
                        id: n.id.clone(),
                        name: n.name.clone(),
                    })
                    .collect();
                print_listing(&state.last_list);
            }

            ReplCommand::Timeline => match &state.last_subgraph {
                None => println!("no subgraph yet — `ask`, `go` or `table` first"),
                Some(subgraph) => {
                    let events = explorer.timeline(subgraph);
                    print_value(format, &events, render_timeline);
                }
            },

            ReplCommand::FilterShow => {
                println!(
                    "type filter: {}\nedge filter: {}",
                    state
                        .type_filter
                        .as_ref()
                        .map(|f| f.join(", "))
                        .unwrap_or_else(|| "(none)".to_string()),
                    state
                        .edge_filter
                        .as_ref()
                        .map(|f| f.join(", "))
                        .unwrap_or_else(|| "(none)".to_string()),
                );
            }
            ReplCommand::FilterType(filter) => {
                state.type_filter = filter;
                println!("ok");
            }
            ReplCommand::FilterEdge(filter) => {
                state.edge_filter = filter;
                println!("ok");
            }
            ReplCommand::FilterClear => {
                state.type_filter = None;
                state.edge_filter = None;
                println!("filters cleared");
            }

            ReplCommand::Show(what) => match &state.last_ask {
                None => println!("nothing asked yet"),
                Some(ask) => match what.as_str() {
                    "dsl" => println!(
                        "{}",
                        serde_json::to_string_pretty(&ask.trace.dsl).unwrap_or_default()
                    ),
                    "cypher" => println!("{}", ask.trace.cypher),
                    "params" => println!(
                        "{}",
                        serde_json::to_string_pretty(&ask.trace.cypher_params).unwrap_or_default()
                    ),
                    "sources" => {
                        for source in &ask.sources {
                            println!(
                                "  {}",
                                source
                                    .name
                                    .clone()
                                    .or_else(|| source.id.clone())
                                    .unwrap_or_default()
                            );
                        }
                    }
                    "answer" => println!("{}", ask.answer.as_deref().unwrap_or("(no answer)")),
                    other => println!("unknown `show {other}` — dsl|cypher|params|sources|answer"),
                },
            },

            ReplCommand::Export(path) => match &state.last_subgraph {
                None => println!("no subgraph yet — `ask`, `go` or `table` first"),
                Some(subgraph) => {
                    let doc = explorer.export(subgraph);
                    tokio::fs::write(&path, serde_json::to_string_pretty(&doc.0)?).await?;
                    println!(
                        "exported {} node(s), {} edge(s) to {}",
                        subgraph.nodes.len(),
                        subgraph.edges.len(),
                        path.display()
                    );
                }
            },
        }
        Ok(())
    }

    async fn open_entity(
        explorer: &Explorer,
        state: &mut ReplState,
        format: OutputFormat,
        id: &str,
        push_trail: bool,
    ) -> Result<()> {
        let Some(card) = explorer.entity(id).await? else {
            println!("entity `{id}` not found");
            return Ok(());
        };
        print_value(format, &card, render_entity_card);
        if push_trail {
            let same_tail = state
                .trail
                .last()
                .is_some_and(|last| last.id == card.node.id);
            if !same_tail {
                state.trail.push(TrailEntry {
                    id: card.node.id.clone(),
                    name: card.node.name.clone(),
                    entity_type: card.node.entity_type.clone(),
                });
            }
        }
        state.last_relations = card.relations.clone();
        state.current = Some(card.node);
        Ok(())
    }

    /// Resolve `#n` against the last numbered listing, otherwise treat
    /// the argument as an entity id.
    fn resolve_target(target: &str, listing: &[ListEntry]) -> std::result::Result<String, String> {
        let Some(number) = target.strip_prefix('#') else {
            return Ok(target.to_string());
        };
        let index: usize = number
            .parse()
            .map_err(|_| format!("`{target}` is not a listing number"))?;
        listing
            .get(index.saturating_sub(1))
            .map(|entry| entry.id.clone())
            .ok_or_else(|| format!("listing has {} item(s)", listing.len()))
    }

    fn print_listing(listing: &[ListEntry]) {
        if listing.is_empty() {
            return;
        }
        println!("(open #n to inspect)");
        for (i, entry) in listing.iter().enumerate() {
            println!("  {}. {}  [{}]", i + 1, entry.name, entry.id);
        }
    }

    fn print_value<T: serde::Serialize>(
        format: OutputFormat,
        value: &T,
        render: impl Fn(&T) -> String,
    ) {
        match format {
            OutputFormat::Json => println!(
                "{}",
                serde_json::to_string_pretty(value).unwrap_or_default()
            ),
            OutputFormat::Table => println!("{}", render(value)),
        }
    }

    fn explorer_has_translator(explorer: &Explorer) -> bool {
        explorer.has_translator()
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn parse_recognizes_the_full_grammar() {
            assert_eq!(
                parse("ask who owns the company?"),
                ReplCommand::Ask("who owns the company?".into())
            );
            assert_eq!(parse("open m1"), ReplCommand::Open("m1".into()));
            assert_eq!(parse("open #2"), ReplCommand::Open("#2".into()));
            assert_eq!(parse("ls"), ReplCommand::Ls);
            assert_eq!(parse("go 1"), ReplCommand::Go("1".into()));
            assert_eq!(parse("go ACTED_IN"), ReplCommand::Go("ACTED_IN".into()));
            assert_eq!(parse("back"), ReplCommand::Back);
            assert_eq!(parse("trail"), ReplCommand::Trail);
            assert_eq!(parse("search Keanu"), ReplCommand::Search("Keanu".into()));
            assert_eq!(parse("types"), ReplCommand::Types);
            assert_eq!(
                parse("table Movie 20"),
                ReplCommand::Table {
                    entity_type: "Movie".into(),
                    offset: 20
                }
            );
            assert_eq!(parse("timeline"), ReplCommand::Timeline);
            assert_eq!(
                parse("filter type Person,Movie"),
                ReplCommand::FilterType(Some(vec!["Person".into(), "Movie".into()]))
            );
            assert_eq!(parse("filter type clear"), ReplCommand::FilterType(None));
            assert_eq!(parse("filter clear"), ReplCommand::FilterClear);
            assert_eq!(parse("filter"), ReplCommand::FilterShow);
            assert_eq!(parse("show cypher"), ReplCommand::Show("cypher".into()));
            assert_eq!(
                parse("export /tmp/x.json"),
                ReplCommand::Export(PathBuf::from("/tmp/x.json"))
            );
            assert!(matches!(parse("format json"), ReplCommand::Format(OutputFormat::Json)));
            assert_eq!(parse("help"), ReplCommand::Help);
            assert_eq!(parse("quit"), ReplCommand::Quit);
            assert_eq!(parse(""), ReplCommand::Empty);
            assert_eq!(parse("bogus"), ReplCommand::Unknown("bogus".into()));
            assert_eq!(parse("ask"), ReplCommand::Unknown("ask".into()));
        }

        #[test]
        fn resolve_go_classifies_numbers_edges_and_neighbor_types() {
            let relations = vec![
                RelationSummary {
                    edge_type: "LOCATED_IN".into(),
                    direction: crate::explore::RelDirection::In,
                    neighbor_type: "Listing".into(),
                    count: 125,
                },
                RelationSummary {
                    edge_type: "PART_OF_REGION".into(),
                    direction: crate::explore::RelDirection::Out,
                    neighbor_type: "Region".into(),
                    count: 1,
                },
            ];
            assert_eq!(resolve_go("", &relations), GoIntent::All);
            assert!(matches!(
                resolve_go("1", &relations),
                GoIntent::Group(ref r) if r.edge_type == "LOCATED_IN"
            ));
            // Edge type, case-insensitive, canonical casing returned.
            assert_eq!(
                resolve_go("located_in", &relations),
                GoIntent::EdgeType("LOCATED_IN".into())
            );
            // Neighbor entity type ("go Listing" from a Country).
            assert_eq!(
                resolve_go("listing", &relations),
                GoIntent::NeighborType("Listing".into())
            );
            // Unknown names get the try-edge-then-label treatment.
            assert_eq!(
                resolve_go("Whatever", &relations),
                GoIntent::Unknown("Whatever".into())
            );
            // Out-of-range numbers are not groups.
            assert_eq!(
                resolve_go("9", &relations),
                GoIntent::Unknown("9".into())
            );
        }

        #[test]
        fn parse_strips_surrounding_quotes_from_free_text_args() {
            assert_eq!(
                parse("search \"Fishing-Stuff\""),
                ReplCommand::Search("Fishing-Stuff".into())
            );
            assert_eq!(
                parse("ask 'who owns it?'"),
                ReplCommand::Ask("who owns it?".into())
            );
            assert_eq!(
                parse("open \"id with spaces\""),
                ReplCommand::Open("id with spaces".into())
            );
            assert_eq!(
                parse("export \"/tmp/my graph.json\""),
                ReplCommand::Export(PathBuf::from("/tmp/my graph.json"))
            );
            // Interior or unbalanced quotes stay untouched.
            assert_eq!(
                parse("search Keanu \"The One\" Reeves"),
                ReplCommand::Search("Keanu \"The One\" Reeves".into())
            );
            assert_eq!(
                parse("search \"unbalanced"),
                ReplCommand::Search("\"unbalanced".into())
            );
        }

        #[test]
        fn resolve_target_handles_numbers_and_ids() {
            let listing = vec![
                ListEntry {
                    id: "p1".into(),
                    name: "Keanu".into(),
                },
                ListEntry {
                    id: "m1".into(),
                    name: "Matrix".into(),
                },
            ];
            assert_eq!(resolve_target("m1", &listing).unwrap(), "m1");
            assert_eq!(resolve_target("#2", &listing).unwrap(), "m1");
            assert!(resolve_target("#5", &listing).is_err());
            assert!(resolve_target("#x", &listing).is_err());
        }
    }
}
