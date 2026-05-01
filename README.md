# linguagraph

Translate natural-language graph questions into safe, parameterized Cypher and
run them against [Memgraph](https://memgraph.com/). The crate is structured so
an LLM emits a small JSON DSL, the DSL is validated, lowered to a typed AST,
compiled to Cypher with bound parameters, and executed by an `async` driver.

```
natural language ──▶ LLM ──▶ JSON DSL ──▶ AST ──▶ Cypher (+ params) ──▶ Memgraph
                          │                                                 │
                  prompt::generate_system_prompt                  db::GraphClient
```

## Why

Letting a model emit Cypher directly is unsafe (string interpolation, SQL-style
injection) and hard to test. `linguagraph` constrains the model to a tiny JSON
shape that we can validate in microseconds, then takes responsibility for
producing correct, parameterized Cypher ourselves.

## Features

- **Typed pipeline.** Every layer (DSL → AST → Cypher) has its own error type,
  its own tests, and its own boundary. Bugs don't slip across layers.
- **Parameterized output.** Filter values never enter the query string. They
  are bound as `$p0`, `$p1`, … and shipped to the driver as Bolt parameters.
- **Driver-agnostic.** The pipeline depends on a `GraphClient` trait. Production
  uses `neo4rs`; tests use an in-memory `MockClient`. Future drivers (a vector
  store, a different Cypher backend) plug in the same way.
- **Schema-aware prompting.** A `GraphSchema` description renders into a
  provider-agnostic system prompt with rules and worked examples.
- **TOML config + env overrides.** `LINGUAGRAPH__DATABASE__URI=...` overrides
  the file without templating.

## Project layout

```
src/
├── dsl/        JSON DSL types + structural parser/validator
├── ast/        typed query model + DSL → AST lowering
├── builder/    AST → Cypher (split into match/where/return parts)
├── db/         GraphClient trait, neo4rs impl, mock impl
├── config/     TOML loader with env overrides
├── prompt/     schema-aware system-prompt generator
├── core/       Pipeline orchestration (wires layers together)
├── cli/        clap-based CLI
└── error.rs    crate-wide Error / Result
tests/          integration tests (no live DB required)
examples/       sample DSL JSON, usage notes
```

## Getting started

### 1. Install Rust

Stable toolchain (≥ 1.75) via [rustup](https://rustup.rs/).

### 2. Run a Memgraph instance

```bash
docker run -p 7687:7687 memgraph/memgraph-platform
```

### 3. Configure

```bash
cp config.example.toml config.toml
# edit config.toml or override with LINGUAGRAPH__DATABASE__URI=...
```

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

## Configuration

```toml
[database]
uri = "bolt://localhost:7687"
user = "memgraph"
password = "memgraph"
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
```

Any field can be overridden via `LINGUAGRAPH__SECTION__FIELD`, e.g.:

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
let cypher = pipeline.compile(query)?;     // CypherQuery { text, params }
let ast    = pipeline.lower(query.clone())?; // typed AST for inspection
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
local model, anything.

## CLI reference

| Command | Purpose |
|---|---|
| `linguagraph dsl <file.json>` | Parse and print the typed AST. |
| `linguagraph cypher <file.json>` | Compile to Cypher; print query + parameters. |
| `linguagraph run <file.json>` | Compile and execute against Memgraph. |
| `linguagraph prompt [--schema <file>] [--no-examples]` | Print a system prompt. |
| `linguagraph schema` | Fetch the live graph schema as JSON. |

Global flag: `--config <path>` (default `config.toml`).

## Testing

```bash
cargo test
```

The suite has unit tests next to each module plus three integration suites in
`tests/` that exercise the public API end-to-end via `MockClient` — no live
Memgraph required:

- `tests/dsl_parsing.rs` — DSL parse + structural validation
- `tests/cypher_builder.rs` — DSL → Cypher round trips, including an injection
  attempt that confirms values never leak into the query string
- `tests/end_to_end.rs` — `Pipeline` dispatches to a `GraphClient` and applies
  default limits

## Extensibility

The trait boundary at `db::GraphClient` is deliberate. Future work:

- **Embeddings & vector search.** Add a `VectorStore` trait alongside
  `GraphClient`; introduce a `semantic` action in the DSL that the AST lowers
  into a `MATCH … WHERE … CALL vector.search(...)` plan.
- **Hybrid queries.** Extend `FilterExpression` with a `Semantic { field,
  query }` predicate that the builder compiles to a Memgraph MAGE call.
- **Pluggable LLM providers.** The prompt module is provider-agnostic by design;
  callers own the HTTP plumbing. A `PromptOptions::preamble` hook is already
  there to inject provider-specific framing.
- **Alternate Cypher backends.** `MemgraphClient` is one implementation of
  `GraphClient`. A `Neo4jClient` or `SqlxCypherClient` can replace it without
  touching the rest of the codebase.

## Tech stack

`serde`, `serde_json`, `toml`, `thiserror`, `anyhow`, `async-trait`, `tokio`,
`clap`, `tracing`, `neo4rs`, `pretty_assertions` (dev).

## License

Dual-licensed under MIT or Apache-2.0.
