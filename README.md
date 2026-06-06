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
  "action": "find" | "aggregate",  // optional legacy hint; inferred from `return`
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
- The effective query kind is inferred from `return`: any aggregate projection
  makes the query aggregate, so `action` can be omitted and a stale
  `"action": "find"` is ignored.
- Aggregate queries that mix aggregated and plain projections must list the
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

`primary_key` is the field MERGE keys off of. Three forms are accepted:

- `"primary_key": "id"` — equivalent to `{"strict": "id"}`. The named property
  is required; missing values are a hard ingest error.
- `{"soft": "name"}` — soft merge against the existing graph by embedding
  similarity (see below).
- *omitted entirely* — the builder synthesises a deterministic `_canonical`
  property from the entity's `type` + properties and defaults `primary_key`
  to `{"soft": "_canonical"}`. This is the shape emitted by
  `knowledge-prompt`, where the LLM doesn't know any stable identifiers —
  only a `type` and a free-text `name`.

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

### Knowledge-extraction prompt

`knowledge-prompt` emits a deterministic LLM system prompt whose output is a
`{entities, relations}` document — exactly the graph-JSON shape `ingest-graph`
consumes. The entity/relation vocabulary is supplied by a *domain ontology*
loaded from a JSON catalog; the crate ships a built-in `legal` ontology and
additional domains can be added by pointing `[prompt].ontologies_path` at a
JSON file.

```bash
# Use the built-in legal ontology.
linguagraph knowledge-prompt --domain legal

# When [prompt].default_domain is set in config, --domain is optional.
linguagraph knowledge-prompt

# Ad-hoc override: ignore the catalog entirely for one run.
linguagraph knowledge-prompt \
    --entity-type Article \
    --entity-type Citation \
    --relation-type CITES \
    --relation-type CONTAINS \
    -o extract_prompt.md
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
both prompt flavours. The domain name supplied at render time is also
substituted into the prompt's framing sections (role, input structure, rules),
so the LLM sees `"medical information extraction"` rather than a hardcoded
`"legal"` framing for non-legal domains.

```rust
use linguagraph::prompt::{
    DomainOntology, EntityTypeSpec, OntologyCatalog, PromptGenerator, RelationTypeSpec,
};

// Built-in catalog, or load your own via storage (see below).
let generator = PromptGenerator::with_builtin_catalog()
    .with_default_domain("legal");

let prompt = generator.knowledge_extract_prompt(Some("legal"))?;
// or fall back to default_domain:
let prompt = generator.knowledge_extract_prompt(None)?;

// Extend a built-in domain at runtime.
let mut catalog = OntologyCatalog::builtin();
catalog.domains.get_mut("legal").unwrap()
    .entity_types.push(EntityTypeSpec::with_description(
        "Citation", "Reference to another legal act or article.",
    ));
let generator = PromptGenerator::new(catalog);

// Bypass the catalog with an ad-hoc ontology. The second argument is
// the framing label that gets substituted into the prompt sections.
let ad_hoc = DomainOntology {
    entity_types: vec![EntityTypeSpec::new("Article")],
    relation_types: vec![RelationTypeSpec::new("CITES")],
};
let prompt = generator.knowledge_extract_prompt_with("custom", &ad_hoc);
```

#### Pluggable storage backend

The catalog is loaded through the [`OntologyCatalogStorage`] trait, so
real-world deployments can keep ontologies in Postgres, an internal
HTTP service, S3, etc. instead of a checked-in JSON file. The crate
ships two ready-to-use backends:

* `JsonFileOntologyCatalogStorage` — default; reads and atomically
  rewrites a single JSON file. Used by
  `PromptGenerator::from_config` when `[prompt].ontologies_path` is set.
* `InMemoryOntologyCatalogStorage` — read-only, useful for tests and
  programmatically-built catalogs.

```rust
use async_trait::async_trait;
use linguagraph::prompt::{
    OntologyCatalog, OntologyCatalogStorage, OntologyError, PromptGenerator,
};

#[derive(Debug)]
struct PostgresOntologyStorage { /* … pool, etc … */ }

#[async_trait]
impl OntologyCatalogStorage for PostgresOntologyStorage {
    async fn load(&self) -> Result<OntologyCatalog, OntologyError> {
        // SELECT domain, entity_types, relation_types FROM ontologies; …
        # unimplemented!()
    }

    async fn save(&self, catalog: &OntologyCatalog) -> Result<(), OntologyError> {
        // upsert into ontologies … 
        # let _ = catalog;
        # unimplemented!()
    }
}

let storage = PostgresOntologyStorage { /* … */ };
let generator = PromptGenerator::from_storage(&storage)
    .await?
    .with_default_domain("legal");
```

The trait's `save` method has a default that returns
`OntologyError::Unsupported`, so read-only backends only need to
implement `load`.

## Entity-type discovery

A QA service sitting in front of `linguagraph` rarely knows which
entity types are worth querying for a given user question.
`Pipeline::run_entity_type_search` answers that question against the
graph that's actually loaded.

The query has two complementary channels:

- **Vector** — embeds the user text once with BGE-M3, fans the same
  vector across every Qdrant collection populated by ingest
  (`…__name`, `…__text`, `…___canonical`, plus one per
  `OntologyPropertyType::Text` field declared in the ontology),
  resolves each hit to its node labels, and rolls everything up by
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
        "semantic_text__name": 0.81,
        "semantic_text___canonical": 0.74
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
provider = "anthropic"
model = "claude-opus-4-7"
temperature = 0.0
max_tokens = 2048

[query]
max_traversal_depth = 6
default_limit = 100

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

`[llm]`, `[query]`, `[graph_specification]`, `[prompt]`, `[ingest.soft_merge]`
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

Ingestion goes through the same `Pipeline`. Build a graph and call `ingest`, or
hand a mapping + JSON value to `ingest_json`:

```rust
use linguagraph::graph::GraphBuilder;

let graph = GraphBuilder::from_json(raw_json)?;
let summary = pipeline.ingest(&graph).await?;
println!(
    "{} node rows, {} relation rows in {} ms",
    summary.node_rows, summary.relation_rows, summary.elapsed_ms,
);
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
| `linguagraph knowledge-prompt [--domain D] [--entity-type X] [--relation-type Y]` | Generate a knowledge-extraction system prompt for a domain ontology. `--entity-type` / `--relation-type` override the catalog for one run. |
| `linguagraph entity-type-search <text> [--top-k N] [--score-threshold X] [--include-neighbors] [--no-catalog] [--field NAME]...` | Discover which entity types are semantically relevant to a free-text user query. Emits a JSON summary with domains, scopes and per-collection scores. See [Entity-type discovery](#entity-type-discovery). |
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
