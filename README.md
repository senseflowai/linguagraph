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
path it ships graph-JSON data ingestion, live schema introspection,
schema-aware prompt generation, a reusable NL front-end (question → DSL,
rows → answer), a business-facing graph explorer with a CLI/REPL, a pluggable
field-type system with built-in semantic and hybrid vector search, and a
traversal retrieval pipeline for RAG-style chunk lookups. The same layered
design runs through all of it: every stage has its own types, its own error
type, and its own tests.

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
- **Ingest data.** A compact `{entities, relations}` graph-JSON document (or a
  programmatically built `Graph`) goes through a planner that emits
  deterministic, idempotent `MERGE` batches.
- **Explore the graph.** A business-facing explorer (`explore::Explorer`)
  browses entities, walks relations, filters by type, and answers
  natural-language questions with the executed query, a result table and a
  displayable subgraph — exposed 1:1 as `linguagraph explore` subcommands and
  an interactive REPL. See [Graph explorer](#graph-explorer).
- **Search semantically.** The pluggable type system ships a `SemanticText`
  type that integrates with [qlink](https://github.com/senseflowai/qlink). Each
  entity's text is embedded once into a per-entity `_canonical` document; the
  `search` operator runs a hybrid (dense + BM25) retrieval over it with
  cross-encoder reranking, while `eq` / `neq` / `contains` stay exact (plain
  Cypher against the raw value).
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
- **Graph-JSON ingestion.** `ingest-graph` ingests a compact
  `{entities, relations}` document directly, emitting deterministic,
  idempotent `MERGE` batches.
- **Graph explorer.** `explore::Explorer` + the `explore` CLI/REPL: NL
  questions with full query traces, entity inspection with classified
  properties and provenance, hop-by-hop traversal, keyword + semantic search,
  timelines and round-trippable subgraph export — all as JSON-serializable
  DTOs a downstream UI can consume verbatim.
- **Soft-merge by embedding similarity.** Knowledge-extraction payloads arrive
  without a stable primary key. `linguagraph` synthesises a deterministic
  `_canonical` text from each entity's type + properties and defaults to
  `PrimaryKey::Soft("_canonical")`. Before MERGE, a staged resolver embeds the
  canonical text, retrieves top-K nearest neighbours from Qdrant, and routes
  each candidate to `AutoMerge` / `NeedsReview` / `NoMerge` based on multiple
  signals — top-1 score, top-1/top-2 margin, lexical similarity on the primary
  name, close-candidate ambiguity, and disambiguating-property conflicts.
  Default thresholds bias toward false-split-over-false-merge. See
  [Soft-merge](#soft-merge-deduplicating-entities-by-similarity).
- **Entity-type discovery for QA.** Given raw user text,
  `Pipeline::run_entity_type_search` embeds it once and fans the query
  across every `SemanticText` collection in the graph, returning the
  unique entity types that actually carry relevant data, each annotated
  with its domain and its scopes (`text` / `table` / `structured`).
  A second channel surfaces the same answer from
  `OntologyCatalog::find` (catalog-side semantic match against type
  descriptions). Optional 1-hop neighbour roll-up lets a QA front-end
  see adjacent types that may carry the answer. See
  [Entity-type discovery](#entity-type-discovery).
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
├── prompt/     LLM prompt generation (full + query-tailored compact prompts)
├── llm/        provider-agnostic LlmClient trait (+ OpenAI-compatible client)
├── types/      pluggable field-type system (registry, handlers)
├── embeddings/ Embedder trait + mock + llama-cpp-2 backend
├── graph/      owned graph model, GraphBuilder, ontology catalog
├── ingest/     graph → InsertQuery planner with side-effect queue
├── nl/         NL front-end: question → DSL translation, answer synthesis
├── core/       Pipeline orchestration (wires the layers together)
├── explore/    business-facing graph browser (Explorer facade + JSON DTOs)
├── cli/        clap-based CLI (incl. `explore` subcommands + REPL)
├── e2e.rs      end-to-end test-kit (LLM → DSL → Memgraph suites)
└── error.rs    crate-wide Error / Result
tests/          integration tests (no live DB required)
examples/       sample DSL JSON, graph fixtures, e2e suites
```

Anything user-facing — the CLI, the integration tests — goes through
`core::Pipeline`; the layers below it are reusable on their own.

## Pluggable type system

Each field type owns its behaviour across four stages — *ingestion*,
*DSL → AST lowering*, *AST → Cypher emission*, and *prompt advertisement*.
Core modules never branch on type names; they go through a `TypeRegistry`.

### Textual field types: `Keyword` vs `Text`

There are exactly **two** textual property types, and the choice is the
whole contract:

- **`Keyword`** — a plain string stored **verbatim**. Cypher matches it with
  the standard operators (`=`, `!=`, `<`, `>`, `=~` regex, `CONTAINS`,
  `STARTS WITH`, `ENDS WITH`, `IN`). Use it for identifiers, codes, statuses,
  and short categorical / enum-like labels — anything you match exactly,
  filter, or compare. Never embedded.
- **`Text`** — everything else textual (names, descriptions, summaries,
  notes). On the linguagraph side it is **always** processed as
  `SemanticText`: stored on the node (so `eq` / `contains` stay exact) and
  folded into the entity's `_canonical` document, which is embedded once for
  semantic search.

The legacy spellings `String` (→ `Keyword`) and `SemanticText` (→ `Text`) are
still accepted on input for backward compatibility, but new fixtures and
ontologies should use `Keyword` / `Text`.

### `Text` under the hood: the `SemanticText` handler

`Text` is backed by the `SemanticText` type handler — free-text search via
embeddings + [qlink](https://github.com/senseflowai/qlink). Configure it once
in `config.toml`:

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

Tag a property as `Text` in graph-JSON (`{"type": "Text", "value": …}`) or
in the ontology catalog (`"property_type": "Text"`):

```json
{ "id": "acme", "type": "Company",
  "name": { "type": "Text", "value": "Acme Corp" } }
```

The field's raw value stays on the node (so `eq` / `neq` / `contains` are
exact), and its text is folded into the entity's `_canonical` document, which
is embedded into qlink/Qdrant. The `search` op runs a semantic match:

```json
{
  "action": "find",
  "start": { "label": "Company", "alias": "c" },
  "filters": [
    { "field": "c.name", "type": "Text", "op": "search", "value": "apple" }
  ],
  "return": [{ "field": "c.name", "alias": "name" }],
  "limit": 5
}
```

compiles to a single per-entity hybrid retrieval (dense ⊕ BM25, RRF-fused)
followed by a cross-encoder rerank, against the `_canonical` collection:

```cypher
CALL libqlink.search_hybrid_reranked($p0, $p1, $p2, $p3, $p4, $p5)
  YIELD id AS c__qid_0, score AS c__score_0
MATCH (c:Company)
WHERE id(c) = c__qid_0
RETURN c.name AS name
ORDER BY c__score_0 DESC
LIMIT 5
```

`search_reranked` and `hybrid_search` are aliases for the same semantic path.
Several semantic filters on one alias collapse into a single call. For exact
matching, use `eq` / `neq` / `contains`, which compile to a plain
`WHERE c.name = $p0` (no qlink). See `examples/find_company_*.json`.

#### Grounding (optional)

With `[query.grounding].enabled = true` and a Qdrant endpoint that holds the
qlink-populated `_canonical` points, the pipeline runs an extra *grounding*
pass before compiling: it resolves the filter directly against `_canonical`
(reusing the query embedding — no re-embed), and when a same-label hit clears
`threshold` (optionally after a cross-encoder rerank) it pins those node ids
into the Cypher instead of the server-side search:

```cypher
UNWIND $p0 AS c__pf_0        -- [{nid, score}, …] resolved client-side
MATCH (c:Company)
WHERE id(c) = c__pf_0.nid
RETURN c.name AS name
ORDER BY c__pf_0.score DESC
LIMIT 5
```

This trades a Qdrant round-trip for precision and control when the filter
maps to a specific known entity. Anything below the confidence bar (or of a
different label) is left untouched and falls back to
`libqlink.search_hybrid_reranked`. Off by default. See `[query.grounding]` in
`config.e2e.toml`.

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
don't touch the database — `dsl`, `cypher` — fall back to safe defaults when
no config file is present.

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

# Ingest a graph fixture, then browse it
cargo run -- ingest-graph examples/e2e/text2cypher-movies/graph.json --prefix-label DEMO
cargo run -- explore --prefix-label DEMO overview
cargo run -- explore --prefix-label DEMO repl
```

## The DSL

```json
{
  "action": "find" | "aggregate",  // optional legacy hint; inferred from `return`
  "start":  { "label": "<NodeLabel>", "alias": "<ident>" },
  "traversals": [
    {
      "edge":   { "label": "<RelLabel>[|<RelLabel>…]", "alias": "<ident>", "direction": "out|in|both" },
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
    { "aggregate": "count|sum|avg|min|max|collect", "field": "<alias>[.<prop>]", "alias": "<ident>" },
    { "field": "<alias>.<prop>", "date_part": "year|quarter|month|week|day", "alias": "<ident>" }
  ],
  "group_by": ["<alias>.<prop>"],
  "sort":     [{ "field": "<alias-or-projected>", "order": "asc|desc" }],
  "limit":    100,
  "distinct": false
}
```

Validation rules enforced before any query is built:

- Aliases are unique and match `[A-Za-z_][A-Za-z0-9_]*`.
- Field references are exactly one `<alias>` or `<alias>.<property>`.
- The effective query kind is inferred from `return`: any aggregate projection
  makes the query aggregate, so `action` can be omitted and a stale
  `"action": "find"` is ignored.
- Aggregate queries that mix aggregated and plain projections must list the
  plain ones in `group_by`.
- Traversal depth is bounded by `query.max_traversal_depth` from config.
- An edge label may be a `|`-union (`"ACTED_IN|DIRECTED"`); `distinct: true`
  emits `RETURN DISTINCT`; a `date_part` projection buckets a datetime field
  (also usable inside `group_by`).
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

`linguagraph` writes into the graph through one planner. It emits
deterministic, idempotent `MERGE` batches: every node `MERGE` runs before any
relationship `MERGE`, so endpoints always exist when a relation lands, and
re-ingesting the same input is a no-op. Feed it a graph-JSON document
(`ingest-graph` / `GraphBuilder::from_json`) or build a `Graph`
programmatically with `GraphBuilder` and call `Pipeline::ingest`.

### Graph-JSON ingest

`ingest-graph` ingests a compact document directly:

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

`primary_key` is the field MERGE keys off of. Three forms are accepted:

- `"primary_key": "id"` — equivalent to `{"strict": "id"}`. The named property
  is required; missing values are a hard ingest error.
- `{"soft": "name"}` — soft merge against the existing graph by embedding
  similarity (see below).
- *omitted entirely* — the builder synthesises a deterministic `_canonical`
  property from the entity's `type` + properties and defaults `primary_key`
  to `{"soft": "_canonical"}`. This is the natural shape for
  LLM-extracted payloads, where the model doesn't know any stable
  identifiers — only a `type` and a free-text `name`.

### Soft-merge: deduplicating entities by similarity

When an entity uses `PrimaryKey::Soft(field)` — either explicitly or because
the JSON omitted `primary_key` and the builder defaulted to
`Soft("_canonical")` — `Pipeline::ingest` runs a staged resolver before the
MERGE batches. The resolver retrieves top-K candidates from Qdrant and
classifies each one as **AutoMerge** (rewrite the soft key, collapse at MERGE
time), **NeedsReview** (surface in `IngestSummary.soft_merge.review_candidates`
for human triage, do NOT collapse), or **NoMerge** (leave alone, create a new
node).

The pipeline runs as follows:

1. **Embed.** The resolver gathers every soft entity and embeds its key
   property in one batch through the configured `Embedder`. For entities that
   default to `Soft("_canonical")` the key is a deterministic multi-line
   `type: X\nkey: value\n...` text (see [Canonical text format](#canonical-text-format)).
2. **In-batch dedup.** Near-identical embeddings within the same ingest are
   collapsed onto a single representative before talking to Qdrant.
3. **Retrieve top-K.** For every `(label, field)` group the resolver issues
   one Cypher round-trip that calls `libqlink.search_labeled`, then pulls
   each hit's canonical field value AND the matched node's full property map
   in the same query — no follow-up round-trip needed for downstream gates.
   The collection is `{semantic_collection_base}__{field}`, optionally folded
   with the configured `prefix_index`.
4. **Score & decide.** Each candidate is checked against six gates:
   * **AutoMerge threshold.** Top-1 cosine must reach `auto_merge_threshold`.
   * **Margin.** `top1 - top2` must be at least `min_margin` (single-hit
     cases pass automatically).
   * **Lexical.** Jaro-Winkler against the top hit's primary-name line must
     reach `min_lexical_similarity`. Embedding similarity alone is not enough.
   * **Ambiguity.** At most `max_close_candidates` runner-ups may sit within
     `close_candidate_delta` of the top score.
   * **Hard conflict.** None of the `conflict_properties` (e.g. `email`,
     `id`, `url`) may differ between incoming and candidate when both are
     non-null.
   * **Type-only guard.** Candidates whose `_canonical` text consists of just
     `type: X` (no other properties) are blocked from AutoMerge unless
     `allow_type_only_auto_merge = true`.
5. **Route.** All gates clean → AutoMerge: the resolver rewrites the entity's
   key property to the existing node's canonical value, and the standard
   MERGE collapses the two rows into one node. Any gate failure with top-1 ≥
   `review_threshold` → NeedsReview: the entity is left untouched and a
   `ReviewCandidate` (top hit + runners-up + every failing gate's reason) is
   appended to the report. Below `review_threshold` → NoMerge.
6. **Side effects.** The post-MERGE `qlink.insert_labeled` runs as usual so
   the vector index stays current.

Configure via `[ingest.soft_merge]` (defaults shown):

```toml
[ingest.soft_merge]
# Consideration floor — hits below this cosine score are dropped at the
# Cypher layer and never reach the decision pipeline.
similarity_threshold = 0.85
# Candidates fetched per entity. Used in full now (margin, ambiguity counts).
top_k = 10
# Top-1 cosine required for an automatic rewrite. Above the floor but below
# this routes to NeedsReview.
auto_merge_threshold = 0.96
# Top-1 floor for surfacing a review record at all. Below this is silently
# routed to NoMerge.
review_threshold = 0.75
# Minimum top1-top2 gap required for AutoMerge.
min_margin = 0.08
# Minimum Jaro-Winkler on the primary-name line required for AutoMerge.
min_lexical_similarity = 0.70
# At most this many runner-up hits may sit within `close_candidate_delta` of
# the top score; above blocks AutoMerge (ambiguity gate).
max_close_candidates = 1
close_candidate_delta = 0.03
# When false (default), type-only candidates (`_canonical` is just `type: X`)
# never auto-merge — they'd otherwise collapse onto whichever node of that
# type embeds nearest.
allow_type_only_auto_merge = false
# Populate `SoftMergeReport.review_candidates`.
emit_review_candidates = true
# Cap on the size of the per-candidate hit list in a review record
# (top hit + runners-up).
review_max_candidates = 5
# Disambiguating properties. When both incoming and candidate have a value
# here and they differ, AutoMerge is blocked.
conflict_properties = ["id", "email", "url", "isbn", "phone", "ssn", "doi", "ein"]
```

`IngestSummary.soft_merge` exposes counts (`candidates`, `auto_merges`,
`needs_review`, `no_merge`, `in_batch_dedup_collapsed`) and the
`review_candidates` list when `emit_review_candidates = true`. The list is
serialisable (`serde::Serialize`) so callers can log / persist it through
whatever audit pipeline they own. Each review record carries the incoming
key, the top hit + runners-up (with both embedding and lexical scores), and
every gate reason that rejected the AutoMerge.

Soft-merge is **fail-loud**: a graph that contains soft entities but lacks a
configured `Embedder` errors out with `SoftMergeBackendUnavailable` instead of
silently regressing to exact-string MERGE. Soft entities without a value for
the key field error with `MissingGraphPrimaryKeyValue`. Wire an embedder up via
`Pipeline::with_embedder` (the CLI does this when `[types.SemanticText]`
declares an `embedding_model`).

#### Canonical text format

When the JSON ingest path omits `primary_key`, the builder synthesises a
deterministic `_canonical` property and defaults `primary_key` to
`Soft("_canonical")`. The format is:

```text
type: {entity_type}
{prop_a}: {value_a}
{prop_b}: {value_b}
...
```

Properties are sorted alphabetically by key for byte-identical output across
runs. `id` is excluded. When the entity has no other properties the output is
just `type: {entity_type}` — those candidates are *type-only* and are blocked
from AutoMerge by default (see the `allow_type_only_auto_merge` knob above).
See `src/graph/canonical.rs` for the implementation.

## Ontology catalog

Query lowering, schema enrichment, prompt generation and the explorer's
property classification all consult an `OntologyCatalog` — a domain-keyed
vocabulary of entity types, relation types and typed properties. The crate
ships a built-in catalog (currently a `legal` domain,
`OntologyCatalog::builtin()`); deployments layer their own domains on top.

### Ontology catalog format

An ontology catalog is a flat JSON object — keys are domain names, values
are `{description, entity_types, relation_types}`. Entity and relation
types may declare typed properties (`Keyword`, `Text`, `Number`, `Bool`,
`Datetime`, `List` — see
[Textual field types](#textual-field-types-keyword-vs-text)):

```json
{
  "movies": {
    "description": "Movies, people and their credits.",
    "entity_types": [
      {
        "name": "Movie",
        "description": "A feature film.",
        "properties": [
          { "name": "id", "property_type": "Keyword" },
          { "name": "title", "property_type": "Keyword" },
          { "name": "tagline", "property_type": "Text" },
          { "name": "released", "property_type": "Datetime" }
        ]
      }
    ],
    "relation_types": [
      { "name": "ACTED_IN", "description": "Person played a role in a movie." }
    ]
  }
}
```

The catalog powers domain routing and schema narrowing in query-driven
prompt generation, `Text`-property auto-resolution during DSL lowering,
description enrichment in `schema`/`overview` output, and property
classification in explorer cards. See `examples/e2e/*/ontology.json` for
worked catalogs.

#### Wiring and storage

A pipeline gets its catalog snapshot through the `OntologyCatalogStorage`
trait, so deployments can keep ontologies in Postgres, an internal HTTP
service, S3, etc. instead of a JSON file. Bundled backends:

* `JsonFileOntologyCatalogStorage` — reads/atomically rewrites one JSON
  file. The CLI uses it with `[graph_specification].cache_path`
  (default `.linguagraph/ontology_catalog.json`).
* `InMemoryOntologyCatalogStorage` — for tests and programmatically-built
  catalogs.

```rust
use std::sync::Arc;
use linguagraph::graph::{JsonFileOntologyCatalogStorage, OntologyCatalogStorage};

let storage: Arc<dyn OntologyCatalogStorage> =
    Arc::new(JsonFileOntologyCatalogStorage::new(".linguagraph/ontology_catalog.json"));
let pipeline = pipeline.with_ontology_catalog_storage(storage);
pipeline.load_ontology_catalog().await?;
// …or inject a snapshot directly:
// pipeline.with_ontology_catalog(Arc::new(catalog))
```

The trait's `save` has a default returning `OntologyError::Unsupported`,
so read-only backends only implement `load`.

## Entity-type discovery

A QA service sitting in front of `linguagraph` rarely knows which
entity types are worth querying for a given user question.
`Pipeline::run_entity_type_search` answers that question against the
graph that's actually loaded.

The query has two complementary channels:

- **Vector** — embeds the user text once with BGE-M3 and fans the same
  vector across the two Qdrant collections ingest populates:
  `…___canonical` (every entity's whole-entity document) and `…__text`
  (chunk fragments). Because every Text property is folded into
  `_canonical`, those two collections reach every node — no per-field
  fan-out. Each hit is resolved to its node labels and rolled up by
  unique entity type. `vector_score` is the max cosine across
  collections; `per_collection` carries the breakdown.
- **Catalog** — runs the same text through `OntologyCatalog::find`,
  which ranks every entity type by the cosine similarity of its
  description embedding. Surfaces as a separate `catalog_score` so the
  QA service can distinguish "type definitionally matches" from "type
  has actual data". Enabled by default; opt out with
  `--no-catalog` / `include_catalog_signal = false`.

Each result carries the entity type's **domain** (from the catalog,
cross-checked against the Cypher label the planner stamps at ingest)
and its **scopes** — the subset of `text` / `table` / `structured`
the ingest pipeline recorded for that node. The QA service can then
pick a query strategy per type: a structured-scope type is DSL bait,
a text-scope type is best probed with a `TraversalQuery`.

Optionally (`--include-neighbors` / `include_neighbors = true`) the
result also lists the unique entity types of the 1-hop graph
neighbours of the matched nodes, so a follow-up query can hop one
step over.

### CLI

```bash
linguagraph entity-type-search "who founded ACME?" \
    --top-k 32 --score-threshold 0.5 \
    --include-neighbors \
    --prefix-label Tenant1 --prefix-index tenant1
```

The command prints the result as pretty-printed JSON:

```json
{
  "matches": [
    {
      "entity_type": "Company",
      "domain": "legal",
      "scopes": ["structured", "text"],
      "vector_score": 0.81,
      "per_collection": {
        "semantic_text___canonical": 0.81
      },
      "catalog_score": 0.62,
      "sample_node_ids": [12, 47]
    }
  ],
  "neighbors": [
    { "entity_type": "Person", "domain": "legal", "scopes": ["text"], ... }
  ],
  "collections_searched": ["semantic_text__name", "semantic_text__text", ...],
  "elapsed_ms": 18
}
```

### Library

```rust
use linguagraph::core::{EntityTypeSearchQuery, Pipeline};

let mut q = EntityTypeSearchQuery::new("who founded ACME?");
q.include_neighbors = true;
let result = pipeline.run_entity_type_search(q).await?;
for hit in &result.matches {
    println!(
        "{} [{}] scopes={:?} score={:?} catalog={:?}",
        hit.entity_type, hit.domain.as_deref().unwrap_or("-"),
        hit.scopes, hit.vector_score, hit.catalog_score,
    );
}
```

Defaults: `top_k = 32`, `score_threshold = Some(0.5)` (BGE-M3 cosine
sits around 0.4–0.9 for relevant hits; the cut-off is intentionally
lower than the soft-merge default of 0.8 so discovery favours recall),
`include_neighbors = false`, `include_catalog_signal = true`,
`catalog_threshold = 0.45`. The constants are exported as
`linguagraph::core::DEFAULT_TOP_K`, `DEFAULT_SCORE_THRESHOLD`,
`DEFAULT_CATALOG_THRESHOLD` and `MAX_SAMPLE_NODE_IDS`.

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
# OpenAI-compatible chat-completions endpoint used by the NL front-end
# (`explore ask`, the e2e harness) — e.g. a self-hosted vLLM server.
provider = "openai"
model = "vllm-qwen"
temperature = 0.0
max_tokens = 2048
base_url = "http://localhost:8000/v1"
# Env var holding the API key (optional for local servers).
api_key_env = "OPENAI_API_KEY"

[query]
max_traversal_depth = 6
default_limit = 100          # traversal (RAG chunk) retrieval default
max_limit = 5000             # read-path safety ceiling: a query that omits
                             # `limit` returns all matching rows up to this cap,
                             # and an explicit `limit` is capped here too

[ingest]
# Rows per embedding-insert query when draining side effects. Each row runs a
# MATCH plus a `libqlink.insert_hybrid` (Qdrant upsert + BM25), so this is
# capped well below the node/relation batch size to keep large ingests under
# `query_timeout_secs`.
embedding_insert_batch_size = 500

[ingest.soft_merge]
# See "Soft-merge: deduplicating entities by similarity" for the full
# semantics. Defaults bias toward false-split-over-false-merge.
similarity_threshold      = 0.85   # consideration floor; hits below this are dropped
top_k                     = 10     # candidates fetched from Qdrant per entity
auto_merge_threshold      = 0.96   # top-1 score required for an automatic rewrite
review_threshold          = 0.75   # below this routes silently to NoMerge
min_margin                = 0.08   # required top1-top2 gap
min_lexical_similarity    = 0.70   # Jaro-Winkler on the primary-name line
max_close_candidates      = 1      # tolerated runners-up within close_candidate_delta
close_candidate_delta     = 0.03
allow_type_only_auto_merge = false
emit_review_candidates    = true
review_max_candidates     = 5
conflict_properties       = ["id", "email", "url", "isbn", "phone", "ssn", "doi", "ein"]

# Ontology catalog cache + embedding models for query-driven prompt
# generation and entity-type discovery. `[ontology_catalog]` is accepted
# as an alias for this section name.
[graph_specification]
cache_path = ".linguagraph/ontology_catalog.json"
embedding_model = "models/bge-m3.gguf"
reranking_model = "models/bge-reranker.gguf"
embedding_dim = 1024
reranking_threshold = 0.3

[qdrant]
# Empty url disables Qdrant — prompt generation then uses an in-process store.
url = "http://127.0.0.1:6333"
collection = "linguagraph_ontology"

# One block per registered field type; the SemanticText handler reads this one.
[types.SemanticText]
embedding_model = "models/bge-m3.gguf"
collection = "companies"
top_k = 20
```

`[llm]`, `[query]`, `[graph_specification]`, `[qdrant]`, `[ingest.soft_merge]`
and `[types.*]` are all optional and fall back to the defaults shown above. Any field can be overridden via
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

Ingestion goes through the same `Pipeline`. Parse a graph-JSON document (or
assemble a `Graph` with the `GraphBuilder` API) and call `ingest`:

```rust
use linguagraph::graph::GraphBuilder;

let graph = GraphBuilder::from_json(raw_json)?;
let summary = pipeline.ingest(&graph).await?;
println!(
    "{} node rows, {} relation rows in {} ms",
    summary.node_rows, summary.relation_rows, summary.elapsed_ms,
);
```

## Graph explorer

`explore::Explorer` is the business-facing read surface: browse
entities, walk relations, filter by type, and ask natural-language
questions that come back with the executed query, a result table and a
displayable subgraph. Every response is a serde-serializable DTO
(`explore::dto`, optional `utoipa::ToSchema` behind the `utoipa`
feature), so a downstream service can expose it over HTTP verbatim.

```rust
use std::sync::Arc;
use linguagraph::explore::{AskOptions, Explorer, NeighborOptions};
use linguagraph::nl::NlTranslator;

let explorer = Explorer::new(pipeline)                    // wraps a configured Pipeline
    .with_translator(Arc::new(translator));               // optional: enables ask()

// Dataset overview: entity/relation types with counts, sources.
let overview = explorer.overview().await?;

// Inspect one entity: classified properties, provenance, relations.
if let Some(card) = explorer.entity("m1").await? {
    // Walk one hop, filtered.
    let hop = explorer.neighbors(&card.node.id, &NeighborOptions {
        edge_types: Some(vec!["ACTED_IN".into()]),
        ..Default::default()
    }).await?;
    let doc = explorer.export(&hop);                       // GraphBuilder-compatible JSON
}

// Search: keyword (schema-driven property scan) or semantic (_canonical vectors).
let hits = explorer.search("Keanu", &Default::default()).await?;

// NL question → DSL → Cypher → rows + subgraph + trace (+ LLM answer).
let answer = explorer.ask("Who acted in The Matrix?", &AskOptions {
    synthesize_answer: true,
    ..Default::default()
}).await?;
println!("{}", answer.trace.cypher);                       // "how was this answered"
```

Identity: the public node handle is the `id` property (stable across
sessions; not enforced unique — lookups take the first match).
Integer-stored `id` values match too and are stringified in responses.
Nodes without an `id` property get a session-scoped `_nid:<internal-id>`
handle; as a convenience, an all-digit handle that matches no `id`
property is retried as a Memgraph internal id (the number graph tools
like Memgraph Lab display), and the response carries the stable property
handle. Confidence is a *data convention*: `NodeView.confidence` surfaces
a `confidence` property when the ingested data carries one — the pipeline
never computes it. Everything respects the pipeline's
`prefix_label`/`prefix_index` tenant scoping.

The `explore` CLI mirrors the API 1:1 (`--format json` prints the exact
DTOs); `explore repl` adds an interactive shell with navigation state
(rustyline, `repl` feature, on by default):

```
explore> open Spain               # or a listing number: open #2
Spain [Country]  id=Spain
Spain [Country]> ls
  1. ← LOCATED_IN  Listing (125)
Spain [Country]> go Listing       # group number (`go 1`), edge type
                                  # (`go LOCATED_IN`) or neighbor entity type
Spain [Country]> trail
Spain [Country]
Spain [Country]> ask "which listings are located in Spain?"
Spain [Country]> show cypher      # how the last ask was answered
Spain [Country]> filter type Listing   # restrict go/search; `filter clear` resets
Spain [Country]> export /tmp/spain.json
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
| `linguagraph run <file.json>` | Compile and execute a DSL query against Memgraph. |
| `linguagraph traversal <file.json>` | Run the traversal retrieval pipeline (entity + goal chunk search). |
| `linguagraph prompt [query] [--schema <file>] [--no-examples]` | Print a schema-aware system prompt for an LLM. |
| `linguagraph schema [--format json\|prompt]` | Introspect the live graph schema. |
| `linguagraph ingest-graph <graph.json>` | Ingest a compact `{entities, relations}` graph JSON. |
| `linguagraph entity-type-search <text> [--top-k N] [--score-threshold X] [--include-neighbors] [--no-catalog] [--field NAME]...` | Discover which entity types are semantically relevant to a free-text user query. Emits a JSON summary with domains, scopes and per-collection scores. See [Entity-type discovery](#entity-type-discovery). |
| `linguagraph delete-by-source --source <name>` | Delete a source-rooted subgraph and its vectors. |
| `linguagraph explore overview` | Entity/relation types with counts, totals, sources. |
| `linguagraph explore entity <id>` | Inspect one entity: properties, provenance, relation summary. |
| `linguagraph explore neighbors <id> [--edge-type T] [--target-label L] [--direction in\|out\|both]` | Walk one hop from an entity. |
| `linguagraph explore search <text> [--type T] [--mode auto\|keyword\|semantic] [--exact]` | Find entities by text. |
| `linguagraph explore table <Type> [--sort P] [--offset N]` | One page of entities of a type. |
| `linguagraph explore timeline <Type>` | Dated events from `Datetime` properties. |
| `linguagraph explore ask "<question>" [--answer] [--show-cypher]` | NL question → DSL → rows + subgraph + query trace (needs `openai` feature + `[llm]` config). |
| `linguagraph explore run-dsl <file.json>` | Run a hand-written DSL file through the ask flow (subgraph + trace). |
| `linguagraph explore export (--entity ID \| --type T) [-o file]` | Export a subgraph as GraphBuilder-compatible JSON. |
| `linguagraph explore repl` | Interactive graph-walking shell (trail, numbered listings, filters). |

Global flag: `--config <path>` (default `config.toml`). The `ingest-graph`,
`run`, `cypher`, `traversal`, `entity-type-search`, `delete-by-source` and
`explore` commands also accept `--prefix-label` / `--prefix-index` to scope
reads and writes to a tenant or dataset. Every `explore` subcommand takes
`--format table|json`; JSON output is the exact DTO contract a downstream
UI consumes.

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
- `tests/explorer.rs` — the graph explorer against a mock client: generated
  Cypher, DTO decoding, subgraph materialization, injection guards

For live end-to-end coverage against local Memgraph, real GGUF embeddings and
an OpenAI-compatible LLM, use the e2e test-kit:

```bash
./scripts/run-e2e.sh
```

It loads a graph JSON fixture, an ontology catalog and natural-language
questions, then validates generated DSL results and optional final answers.
See [docs/e2e.md](docs/e2e.md) for the fixture format and validation rules.
For the product-level view of CRM, ERP and semantic sandboxes, see
[docs/e2e-whitepaper.md](docs/e2e-whitepaper.md). For a recall benchmark of the
NL→DSL→Cypher path on the public `neo4j/text2cypher-2024v1` movies slice —
dataset, methodology, results (≈87%) and how it compares to public text-to-Cypher
solutions — see [docs/text2cypher-benchmark.md](docs/text2cypher-benchmark.md).

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
- **Pluggable LLM providers.** Everything NL-facing depends on the
  `llm::LlmClient` trait (one `complete(system, user)` call). The bundled
  `OpenAiClient` (feature `openai`, on by default) targets any
  OpenAI-compatible `/v1/chat/completions` endpoint, e.g. a self-hosted
  vLLM server; other providers implement the trait.
- **Ontology storage.** Implement `graph::OntologyCatalogStorage` to load
  domain catalogs from Postgres, S3, an HTTP service — anything beyond the
  bundled JSON-file and in-memory backends.

## Tech stack

`serde`, `serde_json`, `toml`, `strum`, `thiserror`, `anyhow`, `async-trait`,
`tokio`, `futures`, `clap`, `tracing`, `tracing-subscriber`, `neo4rs`,
`tabled`, `uuid`, `once_cell`, `encoding_rs`; optional: `llama-cpp-2`
(`llama`), `reqwest` (`openai` / `qdrant`), `rustyline` (`repl`), `utoipa`
(`utoipa`); `pretty_assertions` (dev).

## License

Dual-licensed under MIT or Apache-2.0.
</content>
</invoke>
