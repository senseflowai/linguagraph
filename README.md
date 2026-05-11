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
- **Pluggable type system.** Field types own their own ingestion / lowering /
  Cypher emission. Bundled `SemanticText` integrates with
  [qlink](https://github.com/senseflowai/qlink) for vector + hybrid search;
  add your own (`GeoLocation`, `Keyword`, `ImageEmbedding`) without touching
  the core. See [`docs/type-system.md`](#pluggable-type-system).
- **Document ingestion.** Lift LLM-extracted `{document, chunks, entities,
  relations}` JSON straight into a graph of `Document → Chunk → Entity`
  nodes, with chunk text embedded for semantic retrieval. Ships with a
  knowledge-extraction prompt generator so the LLM emits the exact shape
  the ingester expects. See [Document ingestion](#document-ingestion).
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
├── promptgen/  JSON → mapping-authoring prompt; chunk → knowledge-extract prompt
├── types/      pluggable field-type system (registry, handlers)
├── embeddings/ Embedder trait + Mock + llama-cpp-2 backend
├── ingest/     mapping/document → InsertQuery planner with side-effect queue
├── mapper/     declarative JSON → entity-row extraction
├── core/       Pipeline orchestration (wires layers together)
├── cli/        clap-based CLI
└── error.rs    crate-wide Error / Result
tests/          integration tests (no live DB required)
examples/       sample DSL JSON, usage notes
```

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

## Document ingestion

For RAG-style workloads — where a document is split into chunks and an LLM
extracts entities/relations from each chunk — there's a second ingest path
that bypasses the mapping layer. Input JSON shape:

```json
{
  "document": {
    "name": "Article 5 of the Civil Code",
    "path": "/docs/civil_code/article_5.txt",
    "chunks": [
      {
        "id": "c1",
        "text": "The court grants the citizen the right to appeal.",
        "entities": [
          {"id": "e1", "type": "StateBody",  "name": "court"},
          {"id": "e2", "type": "Person",     "name": "citizen"},
          {"id": "e3", "type": "LegalRight", "name": "right to appeal"}
        ],
        "relations": [
          {"from": "e1", "to": "e3", "type": "GRANTS"},
          {"from": "e3", "to": "e2", "type": "APPLIES_TO"}
        ]
      }
    ]
  }
}
```

Pass it to `ingest-document` and you get the graph:

```
(:Document {path, name})
  -[:HAS_CHUNK]->
    (:Chunk {id, text, index, document_path})  // chunk text is embedded into Qdrant
      -[:MENTIONS]->
        (:StateBody | :Person | :LegalRight {id, name, type})
          -[:GRANTS | :APPLIES_TO | ...]->
            (other entity)
```

- `Document` and `Chunk` are reserved built-in labels; `HAS_CHUNK` and
  `MENTIONS` are reserved built-in relation types.
- Entity types and relation types are LLM-emitted strings; they are
  auto-sanitized to satisfy the Cypher identifier grammar (`Music Group`
  → `Music_Group`, `member-of` → `MEMBER_OF`).
- Entity nodes are merged on a deterministic UUID v5 keyed off
  `(document.path, chunk.id, local_id)` — re-ingest is idempotent. The
  local `e1`/`e2` ids only wire relations within their own chunk and are
  not stored as node keys.
- Chunk text is always embedded via the existing `SemanticText`
  side-effect machinery so chunks become queryable through the standard
  DSL:

  ```json
  {
    "action": "find",
    "start": {"label": "Chunk", "alias": "c"},
    "filters": [
      {"field": "c.text", "type": "SemanticText", "op": "search",
       "value": "right to appeal"}
    ],
    "return": [{"field": "c.id"}, {"field": "c.text"}],
    "limit": 5
  }
  ```

### Library

```rust
use linguagraph::{core::Pipeline, ingest::DocumentInput};

let doc: DocumentInput = serde_json::from_str(&raw_json)?;
let summary = pipeline.ingest_document(doc).await?;
println!("{}", serde_json::to_string_pretty(&summary)?);
```

### CLI

```bash
# Execute against the configured database.
linguagraph ingest-document path/to/doc.json

# Dry-run: print the rendered Cypher batches without connecting.
linguagraph ingest-document-cypher path/to/doc.json
```

### Knowledge-extraction prompt

`knowledge-prompt` emits a deterministic LLM prompt whose output JSON
plugs straight into `ingest-document`. Defaults are tuned for the legal
domain — pass `--entity-type` / `--relation-type` (repeatable) to
constrain the LLM to a custom ontology.

```bash
# Use the bundled legal-domain defaults (LegalNorm, StateBody, Person, ...;
# GRANTS, REGULATES, APPLIES_TO, ...).
linguagraph knowledge-prompt fragment.txt

# Constrain the LLM to a custom vocabulary.
linguagraph knowledge-prompt fragment.txt \
    --entity-type Article \
    --entity-type Citation \
    --entity-type Court \
    --relation-type CITES \
    --relation-type ISSUED_BY \
    --relation-type CONTAINS \
    -o extract_prompt.md

# Pipe a fragment in.
cat fragment.txt | linguagraph knowledge-prompt -
```

Example custom-types invocation and the JSON the LLM is told to emit:

```bash
linguagraph knowledge-prompt article5.txt \
    --entity-type LegalNorm \
    --entity-type StateBody \
    --entity-type LegalRight \
    --relation-type GRANTS \
    --relation-type APPLIES_TO
```

```json
{
  "entities": [
    {"id": "e1", "type": "StateBody",  "name": "court"},
    {"id": "e2", "type": "LegalRight", "name": "right to appeal"}
  ],
  "relations": [
    {"from": "e1", "to": "e2", "type": "GRANTS"}
  ]
}
```

Drop the LLM's `entities`/`relations` arrays straight into a chunk and feed
the document to `ingest-document`.

The default vocabularies bundled with `knowledge-prompt`:

| Default entity types | Default relation types |
|---|---|
| `LegalNorm`, `LegalAct`, `StateBody`, `Person`, `Organization`, `LegalRight`, `LegalObligation`, `Sanction`, `LegalProcedure`, `LegalConcept`, `Date`, `Location`, `MonetaryAmount` | `GRANTS`, `REQUIRES`, `PROHIBITS`, `REGULATES`, `ESTABLISHES`, `ENFORCES`, `REFERENCES`, `AMENDS`, `REPEALS`, `APPLIES_TO`, `PART_OF`, `HAS_SANCTION`, `ISSUED_BY`, `DEFINED_AS` |

For programmatic use the same lists are available as
`promptgen::knowledge::default_entity_types()` /
`default_relation_types()`, and you can extend them rather than replace:

```rust
use linguagraph::promptgen::knowledge::{
    default_entity_types, default_relation_types,
    generate_knowledge_extract_prompt,
    EntityTypeSpec, KnowledgeExtractOptions, RelationTypeSpec,
};

let mut ents = default_entity_types();
ents.push(EntityTypeSpec::with_description(
    "Citation", "Reference to another legal act or article.",
));
let mut rels = default_relation_types();
rels.push(RelationTypeSpec::new("CITES"));

let prompt = generate_knowledge_extract_prompt(
    fragment,
    &KnowledgeExtractOptions { entity_types: ents, relation_types: rels },
);
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
| `linguagraph ingest <data.json> <mapping.json>` | Mapping-driven ingest; execute against the configured DB. |
| `linguagraph ingest-cypher <data.json> <mapping.json>` | Mapping-driven ingest; print Cypher only. |
| `linguagraph ingest-document <doc.json>` | Document/chunk/entity ingest; execute against the configured DB. |
| `linguagraph ingest-document-cypher <doc.json>` | Document/chunk/entity ingest; print Cypher only. |
| `linguagraph generate-prompt <data.json>` | Generate a mapping-authoring prompt for an LLM. |
| `linguagraph knowledge-prompt <fragment.txt> [--entity-type X] [--relation-type Y]` | Generate a knowledge-extraction prompt; defaults to a legal vocabulary. |

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
