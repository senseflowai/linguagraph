# Text2Cypher-Movies: a constrained-DSL NL→Cypher benchmark

A recall benchmark for `linguagraph`'s natural-language → DSL → Cypher path,
built from the public [`neo4j/text2cypher-2024v1`](https://huggingface.co/datasets/neo4j/text2cypher-2024v1)
dataset over Neo4j's official Movies graph. It measures whether the pipeline
turns a question into a query that returns the *right answer*, executed against a
live Memgraph — not whether it reproduces a reference query string.

- **Fixture:** [`examples/e2e/text2cypher-movies/`](../examples/e2e/text2cypher-movies)
- **Suite:** [`examples/e2e/text2cypher-movies.suite.json`](../examples/e2e/text2cypher-movies.suite.json)
- **Runner:** the e2e test-kit ([`docs/e2e.md`](e2e.md))

---

## 1. Why this benchmark

`linguagraph` never asks the model to write Cypher. The model fills a small,
strongly-typed JSON DSL; everything downstream — validation, lowering, Cypher
compilation, execution — is deterministic Rust (see the [README](../README.md)).
That trades expressive power for **safety by construction**: filter values are
always Bolt parameters, never string-interpolated, and every field reference is
checked before a query is built.

The question this benchmark answers is: *given that constraint, how often does
the pipeline still produce the correct answer to a real natural-language graph
question?* Public text-to-Cypher datasets are the natural yardstick, but they
assume free-form Cypher generation, so they need curation to fit a constrained
DSL (see §2.3).

## 2. Dataset

### 2.1 Source

`neo4j/text2cypher-2024v1` is 44,387 `(question, schema, cypher)` triplets from
12 sources. We take the **`neo4jlabs_demo_db_movies`** slice — **2,152 questions**
whose golden Cypher targets Neo4j's classic Movies schema — because, unlike most
of the dataset, that schema has a **public, executable data dump**
([neo4j-graph-examples/movies](https://github.com/neo4j-graph-examples/movies)),
so golden queries can actually be *run* to produce ground-truth answers.

The 220 selected questions span all five movies-schema sources:

| Source | Questions |
|---|---|
| `neo4jLabs_synthetic_gpt4turbo` | 72 |
| `neo4jLabs_synthetic_claudeopus` | 51 |
| `neo4jLabs_synthetic_gpt4o` | 42 |
| `neo4jLabs_synthetic_gemini` | 29 |
| `neo4j_crowdsourced` | 26 |

### 2.2 The graph

The official Movies dump, converted to `linguagraph`'s ingest JSON: **171 nodes**
(38 `Movie`, 133 `Person`) and **253 relationships**.

| Relationship | Count | Properties |
|---|---|---|
| `ACTED_IN` | 172 | `roles` (list) |
| `DIRECTED` | 44 | — |
| `PRODUCED` | 15 | — |
| `WROTE` | 10 | — |
| `REVIEWED` | 9 | `summary`, `rating` |
| `FOLLOWS` | 3 | — |

`Movie.title` / `Person.name` / `Movie.tagline` are typed **`Text`** (embedded
for semantic search **and** BM25-indexed); `id` is a `Keyword` primary key.

### 2.3 Curation: from 2,152 to 220

Two structural facts rule out using the slice verbatim:

1. **Not every golden query is DSL-expressible.** The DSL is one start node +
   bounded traversals + filters + projections + group-by/sort/limit. A golden
   query using `WITH`-chains, `UNWIND`, subqueries, multiple disconnected
   `MATCH`es, self-joins, arithmetic, or string functions cannot — by design —
   be produced by the DSL.
2. **Some golden queries reference data the dump lacks** (e.g. `Movie.votes`,
   which the text2cypher *schema* advertises but the real dump doesn't have).

A classifier filters the 2,152 down to a **DSL-expressible, data-compatible pool
of 957**. The dominant rejections:

| Reason | Rejected |
|---|---|
| `WITH` chains | 586 |
| references missing `votes` property | 250 |
| `DISTINCT` (later added; see §5.3) | 222 |
| Cypher function calls (`size()`, `toLower()`, …) | 166 |
| self-join / repeated alias (`(p)-[]->(m)<-[]-(p)`) | 155 |
| `NOT` / negation | 84 |
| `EXISTS {}` subqueries | 59 |
| field-to-field comparison (`a.x = b.y`) | 34 |

From the 957 we sample **220**, deduplicating by query *shape* (at most 2 per
structural template) and stratifying across sources, to avoid a benchmark
dominated by near-identical "list N people born before Y" clones.

### 2.4 Question tiers

Questions are labelled by what their golden filter touches — this is a
reporting axis, not a difficulty grade:

- **structural (146)** — no text-value filter: numbers, dates, traversals,
  counts. Type-agnostic; validated literally.
- **exact_named (49)** — filters on a *specific* named entity (`{name: 'Keanu
  Reeves'}`). Resolved via client-side entity grounding.
- **contains_thematic (25)** — a literal `CONTAINS 'word'` on a text field.
  Resolved via the BM25 lexical branch.

## 3. Task & methodology

### 3.1 System under test

For each question the runner: (1) renders a schema-aware system prompt, (2) asks
an LLM for DSL JSON, (3) lowers → compiles → executes it against Memgraph under
an isolated tenant prefix. No golden Cypher is shown to the model.

### 3.2 Execution-based validation

Every golden query is executed against the ingested graph to produce
ground-truth rows; the DSL result is checked against them. We validate **answers,
not query strings** (string match is meaningless for Cypher — many correct
queries differ textually). Primitives (`docs/e2e.md`): `row_count`
(exact/min/max), `contains` (a value appears in a column or in `*`),
`not_contains`, `numbers`, `rows`, `columns`, `dsl_expect`.

### 3.3 Validation fidelity

Golden execution and validation are made *fair to a correct-but-divergent DSL*:

- **Case-insensitive `CONTAINS`** on the golden side, matching the BM25 lexical
  branch's case-folding (so "tagline contains 'real'" counts *Welcome to the
  **Real** World* on both sides).
- **Projection-agnostic `contains`**: derived from the first golden column that
  isn't a question-supplied input, so "what roles did X play" is validated on
  the *roles*, not on X's name that the DSL needn't re-project.
- **`row_count`-only when the golden query truncates an unordered result**
  (`LIMIT` without `ORDER BY`) — the specific rows are arbitrary.

### 3.4 Models & setup

| Component | Model |
|---|---|
| LLM (DSL generation) | Qwen3 (served via vLLM, OpenAI-compatible endpoint) |
| Embedding | `bge-m3` (GGUF Q4_K_M, 1024-d) |
| Cross-encoder rerank (`cardinality:"one"`) | `bge-reranker-v2-m3` (GGUF Q4_K_M) |
| Graph DB | Memgraph (Bolt) |
| Vector store | Qdrant (dense + BM25 sparse) |

All embedding/reranking is local GGUF; only the DSL-generation LLM is a remote
endpoint. A full 220-question run is ~5–7 min.

## 4. How to run

```bash
# Prereqs: Memgraph + qlink module, Qdrant, and an OpenAI-compatible LLM
# endpoint, all configured in config.e2e.toml.
cargo build --release --bin linguagraph-e2e

./target/release/linguagraph-e2e \
  --config config.e2e.toml \
  --suite examples/e2e/text2cypher-movies.suite.json \
  --report target/e2e/text2cypher-movies.report.json
```

The suite ingests the graph under an isolated prefix, generates DSL per question,
executes it, and writes a machine-readable report (per-case DSL, compiled Cypher,
rows, pass/fail). `gold_reference.json` holds each question's golden Cypher and
tier for offline analysis.

## 5. Results

### 5.1 Headline

**192 / 220 = 87.3%** answer-level accuracy (Qwen3 + bge-m3).

> Run-to-run variance is **±2–3** questions — the DSL-generation LLM is sampled,
> so re-runs land in ~85–88%. Treat the headline as "≈87%", not a fixed point.

### 5.2 By tier

| Tier | Passed | Accuracy |
|---|---|---|
| structural | 127 / 146 | 87% |
| exact_named | 44 / 49 | **90%** |
| contains_thematic | 21 / 25 | 84% |

### 5.3 What moved the needle

The benchmark doubled as a driver for real fixes; the progression:

| Stage | Score | Change |
|---|---|---|
| Baseline (Keyword typing, literal validation) | 165 / 220 (75%) | — |
| **Phase A** — unify semantic retrieval on client-side Qdrant | **184** | +19 |
| **Phase B** — edge-type union `:A\|B\|C` (DISTINCT added but not prompted) | **188** | +4 |
| **Phase C** — validation fidelity + `collect()` | **192** | +4 |

Phase A is the architectural core. `Text`-field ops split by intent:

- **lexical** (`contains` / `eq` / …) → **BM25-only**. A bare-value sparse query
  returns *exactly* the docs containing the token — literal `CONTAINS` via the
  index — and a no-match pins an **empty** set (0 rows), not a semantic guess.
  This alone took `contains_thematic` from **0/25 to ~17/25**.
- **semantic** (`search`) → dense ⊕ BM25 hybrid (Qdrant RRF).

Both run client-side against Qdrant, so **all 75** text-value queries resolve by
pinning node ids (`UNWIND`), with **zero** server-side `libqlink` fallback in the
query path; the other 145 are plain Cypher.

The benchmark also surfaced and fixed three real query-builder bugs (edge alias
dropped across a `WITH`; two grounded filters colliding on `WITH`; list `contains`
compiling to string `CONTAINS`) and a 6× perf win (per-question full-DB schema
introspection → cached: ~30 min → ~5 min per run).

### 5.4 Where the remaining 28 fail

Increasingly *not* linguagraph correctness: LLM sampling variance (a capable
feature like `collect()` / edge-union not emitted where apt), genuinely hard
multi-hop / co-actor patterns, and residual golden-query quirks (arbitrary
`ORDER BY` under `LIMIT`, `FOLLOWS`-vs-`REVIEWED` phrasing). ≈87–88% is a
realistic ceiling for this literal-exact slice with a constrained DSL.

## 6. Comparison with public solutions

Neo4j's own [Text2Cypher (2024) benchmark](https://neo4j.com/blog/developer/benchmarking-neo4j-text2cypher-dataset/)
evaluates foundational and fine-tuned models on the **full 4,833-question test
set** (all ~16 schemas). Its execution-based metric is **ExactMatch**: the
generated and golden queries are run and their **entire result sets compared as
ordered, lexicographically-sorted strings**. Under that protocol the best models
— GPT-4o and the fine-tuned `HF/tomasonjo_text2cypher` — reach **≈30% ExactMatch**
([Neo4j blog](https://medium.com/neo4j/benchmarking-using-the-neo4j-text2cypher-2024-dataset-d77be96ab65a);
[Ozsoy et al., 2024](https://arxiv.org/html/2412.10064v1)).

**These numbers are not directly comparable to ours, and shouldn't be read as
"87% > 30%".** The tasks differ on every axis:

| | Neo4j text2cypher benchmark | This benchmark |
|---|---|---|
| **Task** | free-form Cypher generation | constrained JSON DSL → Cypher |
| **Scope** | full test set, all schemas, unbounded query shapes | 1 schema (movies), DSL-expressible subset |
| **Metric** | ExactMatch (full ordered result-set equality) | `row_count` + answer-`contains` (tier-adapted) |
| **Model** | GPT-4o / Gemini / fine-tuned | Qwen3 (local-ish, via vLLM) |

The strictness gap alone is large: ExactMatch demands the *exact* result set in
the *exact* order; our `row_count`+`contains` accepts a correct answer projected
or ordered differently. And our subset deliberately excludes the query shapes the
DSL can't express — which are also the shapes free-Cypher models most often get
wrong. So the honest reading is: **on the executable, DSL-expressible movies
slice, the constrained-DSL pipeline answers ≈87% of questions correctly** — a
statement about *this* system on *this* slice, not a leaderboard placement.

What the comparison *does* show: ExactMatch on the raw dataset is a punishing,
formatting-sensitive metric (Neo4j notes it is "very sensitive"), which is
exactly why we validate answers by execution rather than string/result-set
identity, and why direct number-vs-number claims across the two setups are
unsound.

## 7. Limitations

- **Single LLM.** Results are for Qwen3-via-vLLM; a stronger DSL generator would
  move the number. The pipeline is model-agnostic.
- **Curated subset.** By construction it excludes what the DSL can't express;
  it measures the constrained pipeline on its addressable surface, not the full
  dataset's difficulty.
- **Sampling variance** (±2–3) — report a range, not a point.
- **Literal-intent bias.** text2cypher questions were written for literal Cypher;
  `Text` fields therefore lean on the lexical (BM25) branch here. The semantic
  (`search`) branch is exercised more by the `crm` / `erp` / `canonical-semantic`
  suites.

## 8. Reproducibility

- Fixture (`graph.json`, `ontology.json`, `questions.json`, `gold_reference.json`)
  and suite are checked in under `examples/e2e/text2cypher-movies/`.
- The report JSON records, per question, the generated DSL, the executed Cypher,
  the rows, and the pass/fail with reasons — enough to audit any case.
- Regression-checked against the other e2e suites (`cameras` 6/6, `crm` 15/15,
  `erp` 11/11, `text-chunks` 7/7, `canonical-semantic` 2/2) to confirm the
  retrieval changes don't regress the semantic path.

---

*Fixture and pipeline: MIT/Apache-2.0, same as the crate. Golden questions:
`neo4j/text2cypher-2024v1` (Apache-2.0). Movies graph:
`neo4j-graph-examples/movies`.*
