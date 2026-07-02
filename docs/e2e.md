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
  --suite examples/e2e/camera.suite.json \
  --report target/e2e/camera.report.json
```

The default e2e config uses:

- LLM API: configured in `config.e2e.toml`
- LLM model: `vllm-qwen36`
- embedding model: `gpustack/bge-m3-GGUF` / `bge-m3-Q4_K_M.gguf`
- reranking model: `gpustack/bge-reranker-v2-m3-GGUF` / `bge-reranker-v2-m3-Q4_K_M.gguf`
- local Memgraph: `bolt://127.0.0.1:7687/memgraph`

## Suite Format

`camera.suite.json`:

```json
{
  "name": "camera-basic",
  "graph": "camera.graph.json",
  "ontology": "camera.ontology.json",
  "questions": "camera.questions.json",
  "settings": {
    "prefix": "E2E_CAMERA_BASIC",
    "cleanup_before": true,
    "cleanup_after": false,
    "answer_with_llm": true,
    "judge_with_llm": false,
    "max_repairs": 1
  }
}
```

Relative paths resolve from the suite file directory. The prefix is applied as
both `prefix_label` and `prefix_index`, so graph nodes and vector collections
stay isolated from other data.

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
      "id": "camera_at_sales",
      "question": "Какая камера установлена в месте Sales?",
      "validation": {
        "row_count": { "exact": 1 },
        "dsl_expect": {
          "start_label": "Place",
          "required_traversal_labels": ["LOCATED_AT"]
        },
        "contains": [
          { "column": "*", "mode": "contains", "value": "TargetAI - Face (RTSP)" }
        ],
        "numbers": [
          { "column": "*", "op": "eq", "value": 2 }
        ],
        "answer_contains": ["TargetAI"]
      }
    }
  ]
}
```

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
