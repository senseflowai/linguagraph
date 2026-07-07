# E2E Test-Kit

For the higher-level product and architecture view, see
[docs/e2e-whitepaper.md](e2e-whitepaper.md).

The e2e runner exercises the production path:

1. load a graph JSON fixture;
2. load an ontology catalog;
3. ingest the graph into local Memgraph under an isolated `prefix_label`;
4. ask an OpenAI-compatible LLM to produce linguagraph DSL for each question;
5. run the DSL through `Pipeline`;
6. validate rows and, optionally, the synthesized answer.

Run the bundled suite:

```bash
./scripts/run-e2e.sh
```

Equivalent explicit command:

```bash
cargo run --bin linguagraph-e2e -- \
  --config config.e2e.toml \
  --suite examples/e2e/cameras.suite.json \
  --report target/e2e/cameras.report.json
```

The default e2e config uses:

- LLM API: configured in `config.e2e.toml`
- LLM model: `vllm-qwen36`
- embedding model: `gpustack/bge-m3-GGUF` / `bge-m3-Q4_K_M.gguf`
- reranking model: `gpustack/bge-reranker-v2-m3-GGUF` / `bge-reranker-v2-m3-Q4_K_M.gguf`
- local Memgraph: `bolt://127.0.0.1:7687/memgraph`

## Suite Format

`cameras.suite.json`:

```json
{
  "name": "cameras-basic",
  "graph": "cameras/graph.json",
  "ontology": "cameras/ontology.json",
  "questions": "cameras/questions.json",
  "settings": {
    "prefix": "E2E_CAMERAS_BASIC",
    "cleanup_before": true,
    "cleanup_after": false,
    "answer_with_llm": true,
    "judge_with_llm": false,
    "include_embeddings_in_report": false,
    "max_repairs": 1
  }
}
```

Relative paths resolve from the suite file directory. The prefix is applied as
both `prefix_label` and `prefix_index`, so graph nodes and vector collections
stay isolated from other data.

`include_embeddings_in_report` controls whether raw embedding vectors are
written to the JSON report. It defaults to `false`; leave it off for normal
runs because vectors make reports very large. It can also be enabled from the
CLI with `--include-embeddings-in-report`.

## Graph Format

The graph file is the compact shape accepted by `GraphBuilder::from_json`:

```json
{
  "source": "source-name",
  "entities": [
    {
      "id": "camera-1",
      "type": "Camera",
      "primary_key": "id",
      "properties": {
        "id": { "type": "Keyword", "value": "camera-1" },
        "name": { "type": "Text", "value": "TargetAI - Face (RTSP)" }
      }
    }
  ],
  "relations": [
    { "from": "camera-1", "to": "place-1", "type": "LOCATED_AT" }
  ]
}
```

Use strict primary keys for deterministic e2e fixtures. Text properties are
embedded through `SemanticText`; identifiers and enum-like fields should be
`Keyword`.

## Ontology Format

The ontology file is an `OntologyCatalog`:

```json
{
  "camera_domain": {
    "entity_types": [
      {
        "name": "Camera",
        "description": "A video camera.",
        "properties": [
          { "name": "id", "property_type": "keyword", "required": true },
          { "name": "name", "property_type": "text", "required": true }
        ]
      }
    ],
    "relation_types": [
      { "name": "LOCATED_AT", "description": "Camera to Place." }
    ]
  }
}
```

## Questions Format

```json
{
  "questions": [
    {
      "id": "face_recognition_cameras",
      "question": "На каких камерах включена функция распознавания лиц?",
      "validation": {
        "row_count": { "exact": 3 },
        "dsl_expect": {
          "required_traversal_labels": ["USES_MODULE"]
        },
        "contains": [
          { "column": "*", "mode": "contains", "value": "AST Entrance FR-01" },
          { "column": "*", "mode": "contains", "value": "WH North Dock FR-01" },
          { "column": "*", "mode": "contains", "value": "AA Hall FR-01" }
        ],
        "not_contains": [
          { "column": "*", "mode": "contains", "value": "AST Entrance LPR-03" }
        ]
      }
    }
  ]
}
```

For text-ingest fixtures, a case can skip DSL generation entirely and run the
document traversal endpoint:

```json
{
  "id": "traversal_robot_line_context",
  "question": "Найди фрагменты отчета о Phoenix-12.",
  "traversal": {
    "entities": ["Phoenix-12", "Sokol WMS"],
    "goal": "robot line integration with Sokol WMS",
    "query": "How is Phoenix-12 connected to Sokol WMS?",
    "limit": 4,
    "entity_types": ["RobotLine", "SoftwareSystem"],
    "rerank": false
  },
  "validation": {
    "row_count": { "min": 1 },
    "contains": [
      { "column": "chunk_text", "mode": "contains", "value": "Phoenix-12" }
    ]
  }
}
```

`traversal` cases call `Pipeline::run_traversal` after graph ingest. The runner
forces the suite prefix into `prefix_label` and `prefix_index`, just as it does
for static or generated DSL. Do not combine `dsl` and `traversal` in one case.

### Chunk multivector coverage

`chunk_multivector` (config `[types.SemanticText]`) is **on by default**. Every
`Chunk.text` is split into sentences and stored as a per-sentence multivector
Qdrant point; the chunk channel of a traversal then scores each chunk by its
best-matching sentence (MaxSim) instead of one averaged, noisy vector. Set
`chunk_multivector = false` in the config only to fall back to the legacy
single-vector path.

The `text-chunks` suite exercises this end to end. `chunk-009` is a large
(~1000-token, ~30 sentence) appendix that buries two needles on deliberately
unrelated topics — a network root cause (`VLAN 318`, `switch stack DL-7`,
`build R429`) and an HR detail (`retention bonus`, `twelve thousand rubles`) —
that appear in no other chunk. The `multivector_root_cause_needle` and
`multivector_retention_bonus_needle` cases each run a goal-only traversal
(`"entities": []`, so only the chunk channel fires) and assert the needle text
comes back. Both queries retrieving the same long chunk is the signal that the
chunk was split into many embeddings and MaxSim surfaced whichever sentence
matched: a single averaged vector could not rank highly for two far-apart
queries.

`rows` entries are subset matches:

```json
{
  "validation": {
    "rows": [
      { "fields": { "name": "Eastline Logistics", "region": "EMEA" } }
    ]
  }
}
```

Validation is intentionally layered:

- `row_count`: deterministic cardinality check (`exact`, `min`, `max`);
- `rows`: order-insensitive expected row subsets, useful when aliases are stable enough to
  assert exact values but row order is not;
- `columns`: require projected column names when aliases matter;
- `contains`: require a value in a column, or in any column with `"column": "*"`;
- `not_contains`: inverse of `contains`;
- `numbers`: numeric comparison (`eq`, `neq`, `gt`, `gte`, `lt`, `lte`);
- `dsl_expect`: structural checks on the generated DSL, matched against the parsed query
  rather than raw text;
- `answer_contains`: substring checks on the final LLM-synthesized answer;
- `judge`: optional LLM-as-judge for freer answers, kept off by default.

Prefer deterministic row validation for core correctness. Use LLM judging only
for cases where the answer is naturally free-form and cannot be reduced to
stable row/cell expectations.
