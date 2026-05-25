# linguagraph

`linguagraph` turns natural-language questions about a graph into safe,
parameterized [Cypher](https://opencypher.org/) and runs them against
[Memgraph](https://memgraph.com/). Instead of asking a language model to write
Cypher directly — which invites string-interpolation bugs and injection — the
crate constrains the model to a small, strongly-shaped JSON DSL. That DSL is the
only thing the model produces; everything downstream is deterministic Rust:
structural validation, lowering to a typed AST, compilation to Cypher with bound
parameters, and execution through an `async` Bolt driver.

The crate is equal parts query compiler and graph toolkit. Alongside the read
path it ships data ingestion (mapping-driven and graph-JSON), live schema
introspection, schema-aware prompt generation, a pluggable field-type system
with built-in semantic and hybrid vector search, and a traversal retrieval
pipeline for RAG-style chunk lookups. The same layered design runs through all
of it: every stage has its own types, its own error type, and its own tests.

```
natural language ──▶ LLM ──▶ JSON DSL ──▶ AST ──▶ Cypher (+ params) ──▶ Memgraph
                          │                                                 │
                  prompt::generate_system_prompt                  db::GraphClient
```

## Philosophy

Letting a model emit Cypher directly is unsafe and hard to test. Free-form
Cypher means filter values are interpolated into the query string, which opens
the door to injection and makes the output impossible to validate without
effectively parsing Cypher yourself. A model that emits a whole query language
can also fail in unbounded ways.

`linguagraph` takes the opposite stance: **the model's job is to fill in a small
JSON shape; the crate's job is to produce correct Cypher.** The DSL is
deliberately tiny — a handful of actions, traversals, filters and projections —
so it can be validated in microseconds and every field reference checked before
a query is ever built. Once the DSL is valid, query construction is pure,
deterministic Rust that the project owns and tests.

Two principles fall out of that:

- **Safety by construction.** Filter values are never concatenated into Cypher.
  They are bound as Bolt parameters (`$p0`, `$p1`, …), so a malicious or
  malformed value can change *data*, never query *structure*.
- **Typed, layered boundaries.** DSL, AST and Cypher are distinct
  representations with distinct error types. A bug is contained to the layer
  that produced it, and each layer is testable in isolation — no database
  required.

## Project Overview

`linguagraph` is a Rust library (the `linguagraph` crate) and a command-line
tool (the `linguagraph` binary) built around one core pipeline plus the
supporting machinery a graph-backed LLM application needs.

The **query path** is the heart of the project. A JSON DSL document describes a
graph question — a start node, optional traversals, filters, projections,
grouping, sorting and a limit. `linguagraph` validates it structurally, lowers
it to a typed AST, compiles that AST to a parameterized Cypher query, and
executes it against Memgraph over Bolt. The CLI exposes each stage as its own
subcommand (`dsl`, `cypher`, `run`), so you can stop and inspect any
intermediate representation.

Around that core it can also:

- **Generate prompts.** From a graph schema it renders a portable,
  provider-agnostic system prompt — schema description, DSL rules and worked
  examples — that teaches any LLM to emit valid DSL. It can also tailor the
  prompt to a specific natural-language query.
- **Introspect a live graph.** The `schema` command samples a running Memgraph
  instance and reports node labels, relationship types and inferred property
  types as JSON or as a ready-to-use prompt.
- **Ingest data.** Two ingestion paths write into the graph: a *mapping-driven*
  path that lifts arbitrary structured JSON into typed entities and relations
  via a declarative JSONPath mapping, and a *graph-JSON* path that ingests a
  compact `{entities, relations}` document directly. Both run through the same
  planner, which emits deterministic, idempotent `MERGE` batches.
- **Search semantically.** The pluggable type system ships a `SemanticText`
  type that integrates with [qlink](https://github.com/senseflowai/qlink) for
  vector and hybrid (vector + exact) search. Text properties are embedded on
  ingestion and become searchable through two extra DSL operators, `search` and
  `hybrid_search`.
- **Retrieve for RAG.** A traversal retrieval pipeline searches text chunks by
  goal and by mentioned entities, then merges and ranks unique chunks — the
  building block for retrieval-augmented generation over a graph.
- **Stay multi-tenant.** Optional `prefix_label` / `prefix_index` scoping stamps
  every ingested node and every query so independent tenants or datasets can
  share one database and one vector store without colliding.
- **Clean up.** `delete-by-source` removes a source-rooted subgraph — its
  chunks, its orphaned entities and the matching vectors — in well-defined
  phases.

Everything runs without a live database where it can: the read and ingest paths
are exercised end-to-end in tests through an in-memory `MockClient`, and the
default build uses a deterministic mock embedder so nothing depends on native
model code.

## Features

- **Typed pipeline.** Every layer (DSL → AST → Cypher) has its own error type,
  its own tests, and its own boundary. Bugs don't slip across layers.
- **Parameterized output.** Filter values never enter the query string. They
  are bound as `$p0`, `$p1`, … and shipped to the driver as Bolt parameters.
- **Driver-agnostic.** The pipeline depends on a `GraphClient` trait. Production
  uses a `neo4rs`-backed `MemgraphClient`; tests use an in-memory `MockClient`.
  New backends plug in the same way.
- **Schema-aware prompting.** A `GraphSchema` description renders into a
  provider-agnostic system prompt with rules and worked examples.
- **Pluggable type system.** Field types own their own ingestion / lowering /
  Cypher emission. Bundled `SemanticText` integrates with
  [qlink](https://github.com/senseflowai/qlink) for vector + hybrid search;
  add your own (`GeoLocation`, `Keyword`, `ImageEmbedding`) without touching
  the core. See [Pluggable type system](#pluggable-type-system).
- **Two ingestion paths.** Mapping-driven (`ingest-json`) lifts structured JSON
  into the graph through a declarative mapping; graph-JSON (`ingest-graph`)
  ingests a compact `{entities, relations}` document directly. Both emit
  deterministic, idempotent `MERGE` batches.
- **TOML config + env overrides.** `LINGUAGRAPH__DATABASE__URI=...` overrides
  the file without templating.

## Project layout

```
src/
├── dsl/        JSON DSL types + structural parser/validator
├── ast/        typed query model
├── resolve/    DSL → AST resolution (the only stringly-typed → typed boundary)
├── builder/    AST → Cypher (split into match/where/return parts)
├── db/         GraphClient trait, Memgraph (neo4rs) impl, mock impl
├── config/     TOML loader with env overrides
├── prompt/     LLM prompt generation (query prompts + knowledge-extract prompts,
│               domain-scoped ontologies loaded from JSON)
├── promptgen/  JSON → mapping-authoring prompt
├── types/      pluggable field-type system (registry, handlers)
├── embeddings/ Embedder trait + mock + llama-cpp-2 backend
├── graph/      owned graph model, GraphBuilder, graph specification
├── mapper/     declarative JSON → entity-row extraction
├── ingest/     graph → InsertQuery planner with side-effect queue
├── core/       Pipeline orchestration (wires the layers together)
├── cli/        clap-based CLI
└── error.rs    crate-wide Error / Result
tests/          integration tests (no live DB required)
examples/       sample DSL JSON, mappings, usage notes
```

Anything user-facing — the CLI, the integration tests — goes through
`core::Pipeline`; the layers below it are reusable on their own.

## Pluggable type system

Each field type owns its behaviour across four stages — *ingestion*,
*DSL → AST lowering*, *AST → Cypher emission*, and *prompt advertisement*.
Core modules never branch on type names; they go through a `TypeRegistry`.

### Built-in type: `SemanticText`

Free-text fields searchable via embeddings + [qlink](https://github.com/senseflowai/qlink).
Configure once in `config.toml`:

```toml
[types.SemanticText]
embedding_model = "models/bge-small.gguf"
collection      = "companies"
top_k           = 20

[graph_specification]
embedding_model = "models/bge-small.gguf"
reranking_model = "models/bge-reranker.gguf"
reranking_threshold = 0.3
embedding_dim   = 384
```

Tag a property in the mapping:

```json
{
  "name": "name",
  "source_path": "$.companies[*].name",
  "type": "SemanticText"
}
```

Now the field is exact-match searchable *and* embedded into qlink/Qdrant.
The DSL grows two new ops, `search` and `hybrid_search`:

```json
{
  "action": "find",
  "start": { "label": "Company", "alias": "c" },
  "filters": [
    { "field": "c.name", "type": "SemanticText", "op": "search", "value": "apple" }
  ],
  "return": [{ "field": "c.name", "alias": "name" }],
  "limit": 5
}
```

compiles to:

```cypher
CALL qlink.search([$p0], $p1, $p2) YIELD id AS c__qid, score AS c__score
MATCH (c:Company)
WHERE id(c) = c__qid
RETURN c.name AS name
ORDER BY c__score DESC
LIMIT 5
```

`hybrid_search` adds an exact-match score to the vector score and orders
by their sum. See `examples/find_company_*.json` for both shapes.

### Adding a new type

```rust
struct GeoLocation;

impl TypeHandler for GeoLocation {
    fn type_id(&self) -> TypeId { TypeId::new("GeoLocation") }
    fn capabilities(&self) -> Capabilities { Capabilities::GEO_SEARCH }
    fn on_ingest(&self, ctx: &mut IngestCtx<'_>) -> Result<(), TypeError> { /* … */ }
    fn lower(&self, ctx: &mut LowerCtx<'_>) -> Result<TypedPredicate, TypeError> { /* … */ }
    fn emit(&self, ctx: &mut EmitCtx<'_>, p: &TypedPredicate) -> Result<(), TypeError> { /* … */ }
}

let registry = RegistryBuilder::new()
    .register(GeoLocation)
    .register(SemanticTextHandler::new(cfg, embedder))
    .build();
let pipeline = Pipeline::new(client, &cfg).with_registry(Arc::new(registry));
```

No changes to the DSL parser, AST, or Cypher builder are required.

## Getting started

### 1. Install Rust

Stable toolchain (≥ 1.75) via [rustup](https://rustup.rs/).

### 2. Run a Memgraph instance

```bash
docker run -p 7687:7687 memgraph/memgraph-platform
```

### 3. Configure

Create a `config.toml` in the working directory (see
[Configuration](#configuration) for every field) or override individual values
with `LINGUAGRAPH__SECTION__FIELD=...` environment variables. Commands that
don't touch the database — `dsl`, `cypher`, `generate-prompt`,
`knowledge-prompt` — fall back to safe defaults when no config file is present.

### 4. Build

```bash
cargo build --release
```

### 5. Try the examples

```bash
# Validate a DSL document and print the typed AST
cargo run -- dsl examples/find_people.json

# Compile DSL to Cypher (+ parameters) without touching the DB
cargo run -- cypher examples/aggregate_orders.json

# Compile and execute against the configured Memgraph
cargo run -- run examples/find_people.json --config config.toml

# Print a schema-aware system prompt for an LLM
cargo run -- prompt --schema schema.json
```

## The DSL

```json
{
  "action": "find" | "aggregate",
  "start":  { "label": "<NodeLabel>", "alias": "<ident>" },
  "traversals": [
    {
      "edge":   { "label": "<RelLabel>", "alias": "<ident>", "direction": "out|in|both" },
      "target": { "label": "<NodeLabel>", "alias": "<ident>" },
      "depth":  { "min": 1, "max": 3 }
    }
  ],
  "filters": [
    { "field": "<alias>.<prop>", "op": "eq|neq|gt|gte|lt|lte|in|contains|starts_with|ends_with",
      "value": <json scalar or array> }
  ],
  "return": [
    { "field": "<alias>.<prop>", "alias": "<ident>" },
    { "aggregate": "count|sum|avg|min|max", "field": "<alias>[.<prop>]", "alias": "<ident>" }
  ],
  "group_by": ["<alias>.<prop>"],
  "sort":     [{ "field": "<alias-or-projected>", "order": "asc|desc" }],
  "limit":    100
}
```

Validation rules enforced before any query is built:

- Aliases are unique and match `[A-Za-z_][A-Za-z0-9_]*`.
- Field references are exactly one `<alias>` or `<alias>.<property>`.
- `find` queries may not contain aggregations.
- `aggregate` queries that mix aggregated and plain projections must list the
  plain ones in `group_by`.
- Traversal depth is bounded by `query.max_traversal_depth` from config.
- Every filter value lands in a Bolt parameter — never in the query string.

### Example: `examples/find_people.json`

```json
{
  "action": "find",
  "start": { "label": "Person", "alias": "p" },
  "traversals": [
    {
      "edge": { "label": "KNOWS", "alias": "r", "direction": "out" },
      "target": { "label": "Person", "alias": "friend" },
      "depth": { "min": 1, "max": 2 }
    }
  ],
  "filters": [
    { "field": "p.age", "op": "gt", "value": 30 },
    { "field": "friend.city", "op": "eq", "value": "Berlin" }
  ],
  "return": [
    { "field": "p.name", "alias": "name" },
    { "field": "friend.name", "alias": "friend_name" }
  ],
  "sort": [{ "field": "name", "order": "asc" }],
  "limit": 25
}
```

Compiles to:

```cypher
MATCH (p:Person)-[r:KNOWS*1..2]->(friend:Person)
WHERE (p.age > $p0 AND friend.city = $p1)
RETURN p.name AS name, friend.name AS friend_name
ORDER BY name ASC
LIMIT 25
```

with `$p0 = 30`, `$p1 = "Berlin"`.

## Ingesting data

`linguagraph` writes into the graph through two front-ends that share one
planner. The planner emits deterministic, idempotent `MERGE` batches: every
node `MERGE` runs before any relationship `MERGE`, so endpoints always exist
when a relation lands, and re-ingesting the same input is a no-op.

### Mapping-driven ingest

The `ingest-json` command takes a raw data file and a *mapping* file. The
mapping is a declarative document — JSONPath expressions plus type tags — that
describes how to lift rows out of arbitrary JSON into typed graph entities and
relations. See `examples/companies_data.json` + `examples/companies_mapping.json`
for a worked pair.

```bash
linguagraph ingest-json examples/companies_data.json examples/companies_mapping.json
```

`generate-prompt` analyses an arbitrary JSON document and emits a prompt that
asks an LLM to author the mapping for it, so you don't have to write the
mapping by hand.

### Graph-JSON ingest

When you already have a graph in hand, `ingest-graph` ingests a compact
document directly — no mapping required:

```json
{
  "entities": [
    { "id": "alice", "type": "Person", "primary_key": "id", "name": "Alice" },
    { "id": "acme",  "type": "Company", "primary_key": "name", "name": "Acme" }
  ],
  "relations": [
    { "from": "alice", "to": "acme", "type": "WORKS_AT", "since": 2024 }
  ]
}
```

Entity `id` values are local handles used only to wire relations. A property
value may be typed (`{"type": "Text", "value": "..."}`) or raw, in which case a
graph property type is inferred from the JSON value. `relationships` is accepted
as an alias for `relations`. This is the shape produced by
`GraphBuilder::from_json` in the library.

### Knowledge-extraction prompt

`knowledge-prompt` emits a deterministic LLM prompt whose output is a
`{entities, relations}` document — exactly the graph-JSON shape `ingest-graph`
consumes. The entity/relation vocabulary is supplied by a *domain ontology*
loaded from a JSON catalog; the crate ships a built-in `legal` ontology and
additional domains can be added by pointing `[prompt].ontologies_path` at a
JSON file.

```bash
# Use the built-in legal ontology.
linguagraph knowledge-prompt fragment.txt --domain legal

# When [prompt].default_domain is set in config, --domain is optional.
linguagraph knowledge-prompt fragment.txt

# Ad-hoc override: ignore the catalog entirely for one run.
linguagraph knowledge-prompt fragment.txt \
    --entity-type Article \
    --entity-type Citation \
    --relation-type CITES \
    --relation-type CONTAINS \
    -o extract_prompt.md

# Pipe a fragment in from stdin.
cat fragment.txt | linguagraph knowledge-prompt - --domain legal
```

The built-in `legal` domain:

| Entity types | Relation types |
|---|---|
| `LegalNorm`, `LegalAct`, `StateBody`, `Person`, `Organization`, `LegalRight`, `LegalObligation`, `Sanction`, `LegalProcedure`, `LegalConcept`, `Date`, `Location`, `MonetaryAmount` | `GRANTS`, `REQUIRES`, `PROHIBITS`, `REGULATES`, `ESTABLISHES`, `ENFORCES`, `REFERENCES`, `AMENDS`, `REPEALS`, `APPLIES_TO`, `PART_OF`, `HAS_SANCTION`, `ISSUED_BY`, `DEFINED_AS` |

#### Ontology catalog format

An ontology catalog is a flat JSON object — keys are domain names, values
are `{entity_types, relation_types}` lists:

```json
{
  "legal": {
    "entity_types": [
      { "name": "LegalNorm", "description": "A rule, provision, article, or paragraph." },
      { "name": "StateBody", "description": "Any organ of public authority." }
    ],
    "relation_types": [
      { "name": "GRANTS", "description": "Subject confers a right or power on another." }
    ]
  },
  "medical": {
    "entity_types": [{ "name": "Disease" }, { "name": "Symptom" }],
    "relation_types": [{ "name": "CAUSES" }, { "name": "TREATS" }]
  }
}
```

Wire it up via `config.toml`:

```toml
[prompt]
ontologies_path = "config/ontologies.json"
default_domain  = "legal"
```

#### Programmatic use

`PromptGenerator` is the high-level facade — it owns the catalog and exposes
both prompt flavours:

```rust
use linguagraph::prompt::{
    DomainOntology, EntityTypeSpec, OntologyCatalog, PromptGenerator, RelationTypeSpec,
};

// Built-in catalog, or load your own with OntologyCatalog::load_from_path(...).
let generator = PromptGenerator::with_builtin_catalog()
    .with_default_domain("legal");

let prompt = generator.knowledge_extract_prompt(fragment, Some("legal"))?;
// or fall back to default_domain:
let prompt = generator.knowledge_extract_prompt(fragment, None)?;

// Extend a built-in domain at runtime.
let mut catalog = OntologyCatalog::builtin();
catalog.domains.get_mut("legal").unwrap()
    .entity_types.push(EntityTypeSpec::with_description(
        "Citation", "Reference to another legal act or article.",
    ));
let generator = PromptGenerator::new(catalog);

// Bypass the catalog with an ad-hoc ontology.
let ad_hoc = DomainOntology {
    entity_types: vec![EntityTypeSpec::new("Article")],
    relation_types: vec![RelationTypeSpec::new("CITES")],
};
let prompt = generator.knowledge_extract_prompt_with(fragment, &ad_hoc);
```

## Configuration

```toml
[database]
uri = "bolt://localhost:7687"
user = "memgraph"
password = "memgraph"
database = "memgraph"
max_connections = 16
query_timeout_secs = 30

[llm]
provider = "anthropic"
model = "claude-opus-4-7"
temperature = 0.0
max_tokens = 2048

[query]
max_traversal_depth = 6
default_limit = 100

[graph_specification]
cache_path = ".linguagraph/graph_specification.json"
embedding_model = "models/bge-small.gguf"
reranking_model = "models/bge-reranker.gguf"
embedding_dim = 384
reranking_threshold = 0.3

[prompt]
# Path to a domain-ontology JSON catalog (see "Ontology catalog format").
# When omitted, the built-in catalog (currently the `legal` domain) is used.
ontologies_path = "config/ontologies.json"
# Domain selected by `knowledge-prompt` when --domain is omitted.
default_domain  = "legal"

# One block per registered field type; the SemanticText handler reads this one.
[types.SemanticText]
embedding_model = "models/bge-small.gguf"
collection = "companies"
top_k = 20
```

`[llm]`, `[query]`, `[graph_specification]`, `[prompt]` and `[types.*]` are all
optional and fall back to the defaults shown above. Any field can be overridden via
`LINGUAGRAPH__SECTION__FIELD`, e.g.:

```bash
LINGUAGRAPH__DATABASE__URI=bolt://memgraph:7687 cargo run -- run query.json
```

## Library usage

```rust
use std::sync::Arc;
use linguagraph::{config, core::Pipeline, db::MemgraphClient, dsl};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cfg = config::load(std::path::Path::new("config.toml")).await?;
    let client = Arc::new(MemgraphClient::connect(&cfg.database).await?);
    let pipeline = Pipeline::new(client, &cfg);

    let query = dsl::parse_str(r#"{
        "action": "find",
        "start": { "label": "Person", "alias": "p" },
        "filters": [{ "field": "p.age", "op": "gt", "value": 30 }],
        "return": [{ "field": "p.name" }]
    }"#)?;

    let result = pipeline.run(query).await?;
    println!("{} rows", result.rows.len());
    Ok(())
}
```

You can also stop at any intermediate stage:

```rust
let cypher = pipeline.compile(query.clone())?; // CypherQuery { text, params }
let ast    = pipeline.lower(query)?;           // typed AST for inspection
```

Ingestion goes through the same `Pipeline`. Build a graph and call `ingest`, or
hand a mapping + JSON value to `ingest_json`:

```rust
use linguagraph::graph::GraphBuilder;

let graph = GraphBuilder::from_json(raw_json)?;
let summary = pipeline.ingest(&graph).await?;
println!("{} node rows, {} relation rows", summary.node_rows, summary.relation_rows);
```

## Prompt generation

```rust
use linguagraph::prompt::{generate_system_prompt, GraphSchema, PromptOptions};

let schema: GraphSchema = serde_json::from_str(SCHEMA_JSON)?;
let prompt = generate_system_prompt(&schema, &PromptOptions::default());
// hand `prompt` to any LLM provider as the system message.
```

The generator emits a portable string: schema → DSL rules → worked examples.
It never embeds provider-specific markers; plug it into Anthropic, OpenAI, a
local model, anything. `generate_query_prompt` produces the same prompt narrowed
to the entity types relevant to a specific natural-language query.

## CLI reference

| Command | Purpose |
|---|---|
| `linguagraph dsl <file.json>` | Validate a DSL file and print the lowered AST. |
| `linguagraph cypher <file.json>` | Compile to Cypher; print query + parameters (no DB). |
| `linguagraph query <file.json>` | Alias of `cypher`; reserved for future natural-language front-ends. |
| `linguagraph run <file.json>` | Compile and execute a DSL query against Memgraph. |
| `linguagraph traversal <file.json>` | Run the traversal retrieval pipeline (entity + goal chunk search). |
| `linguagraph prompt [query] [--schema <file>] [--no-examples]` | Print a schema-aware system prompt for an LLM. |
| `linguagraph schema [--format json\|prompt]` | Introspect the live graph schema. |
| `linguagraph ingest-json <data.json> <mapper.json>` | Mapping-driven ingest; execute against the configured DB. |
| `linguagraph ingest-graph <graph.json>` | Ingest a compact `{entities, relations}` graph JSON. |
| `linguagraph generate-prompt <data.json>` | Generate a mapping-authoring prompt for an LLM. |
| `linguagraph knowledge-prompt <fragment.txt> [--domain D] [--entity-type X] [--relation-type Y]` | Generate a knowledge-extraction prompt for a domain ontology. `--entity-type` / `--relation-type` override the catalog for one run. |
| `linguagraph delete-by-source --source <name>` | Delete a source-rooted subgraph and its vectors. |

Global flag: `--config <path>` (default `config.toml`). The ingest, `run`,
`cypher`, `traversal` and `query` commands also accept `--prefix-label` /
`--prefix-index` to scope reads and writes to a tenant or dataset.

## Testing

```bash
cargo test
```

The suite has unit tests next to each module plus integration suites in
`tests/` that exercise the public API end-to-end via `MockClient` — no live
Memgraph required:

- `tests/dsl_parsing.rs` — DSL parse + structural validation
- `tests/cypher_builder.rs` — DSL → Cypher round trips, including an injection
  attempt that confirms values never leak into the query string
- `tests/end_to_end.rs` — `Pipeline` dispatches to a `GraphClient` and applies
  default limits
- `tests/type_system.rs` / `tests/property_types.rs` — the pluggable type
  registry and per-type ingestion / lowering / emission
- `tests/promptgen.rs` / `tests/knowledge_prompt.rs` — prompt generators

## Extending linguagraph

The crate is built to be extended at its trait boundaries:

- **Alternate graph backends.** Everything talks to the database through the
  `db::GraphClient` trait. `MemgraphClient` (neo4rs) is one implementation;
  a different Cypher backend can replace it without touching the rest of the
  codebase, exactly as `MockClient` does for tests.
- **New field types.** Implement `TypeHandler` and register it — see
  [Adding a new type](#adding-a-new-type). The DSL parser, AST and Cypher
  builder never branch on type names, so they need no changes.
- **New embedding backends.** The `embeddings::Embedder` trait is a single
  `embed_batch` call. The default build ships a deterministic mock; the
  `llama` feature wires in a GGUF-backed embedder via `llama-cpp-2`.
- **Pluggable LLM providers.** The prompt module is provider-agnostic by
  design; callers own the HTTP plumbing.

## Tech stack

`serde`, `serde_json`, `toml`, `thiserror`, `anyhow`, `async-trait`, `tokio`,
`clap`, `tracing`, `tracing-subscriber`, `neo4rs`, `tabled`, `uuid`,
`once_cell`, `encoding_rs`, `llama-cpp-2` (optional, `llama` feature),
`pretty_assertions` (dev).

## License

Dual-licensed under MIT or Apache-2.0.
</content>
</invoke>
