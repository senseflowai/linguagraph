# linguagraph E2E White Paper

`linguagraph` is not just a query engine. It is a pipeline for turning natural
language into validated graph queries, executing them safely, and proving that
the full path works on realistic data.

This document explains the current e2e program in two layers:

1. what each domain sandbox demonstrates for business users and engineers;
2. how the system transforms a question from NL into DSL and then into Cypher.

The point is not “can the model write a query”. The point is that the model can
operate inside a constrained contract, while the library guarantees the query
shape, the parameterization, the retrieval strategy, and the validation.

## What The E2E Program Proves

Each suite follows the same production-like flow:

1. load a graph fixture;
2. load an ontology;
3. ingest into local Memgraph with isolated `prefix_label` / `prefix_index`;
4. ask an OpenAI-compatible LLM to generate linguagraph DSL;
5. lower DSL to typed AST and Cypher;
6. execute against Memgraph;
7. validate rows, values, and DSL structure;
8. optionally validate the synthesized answer.

The current default e2e stack uses:

- OpenAI-compatible LLM endpoint: configured in `config.e2e.toml`
- embedding model: `gpustack/bge-m3-GGUF` / `bge-m3-Q4_K_M.gguf`
- reranker model: `gpustack/bge-reranker-v2-m3-GGUF` / `bge-reranker-v2-m3-Q4_K_M.gguf`
- local Memgraph: `bolt://127.0.0.1:7687/memgraph`

## Domain Coverage

### CRM sandbox

The CRM suite is the best example of a compact operational sandbox.

Data shape:

- accounts
- contacts
- deals
- tickets
- activities
- products
- users

Ontology emphasis:

- `id` fields are `Keyword` for exact lookups;
- `name` / `description` fields are `Text` where semantic search matters;
- statuses, regions, segments, and stages are `Keyword`;
- relations encode account ownership, deal ownership, product linkage, primary
  contacts, and next activity.

Question classes:

- relationship navigation: who owns a deal, which product is tied to a deal,
  what is the next activity;
- exact and semantic lookup: by `id`, by location phrase, by account name;
- date-based retrieval: scheduled activity on a specific date;
- counting and aggregation: active accounts, open tickets, open deals.

Current result:

- `crm-basic`: `15/15` passed

Example:

```text
NL
Кто отвечает за сделку с id deal-5003?

DSL
find Deal where d.id = "deal-5003" traverse DEAL_OWNER -> User return owner_name

Cypher
MATCH (d:Deal:E2E_CRM_BASIC)-[o:DEAL_OWNER]->(u:User:E2E_CRM_BASIC)
WHERE d.id = $p0
RETURN u.name AS owner_name
```

What this shows business users: CRM questions can be asked in plain language,
including references to specific operational objects, while still returning
traceable rows.

What this shows engineers: the graph, ontology, and validation are all stable
enough to support a repeatable semantic layer for operational data.

### ERP sandbox

The ERP suite covers a broader back-office shape and a more mixed query space.

Data shape:

- customers
- employees
- products
- purchase orders
- sales orders
- invoices
- payments
- shipments
- warehouses
- suppliers

Ontology emphasis:

- natural-language fields such as descriptions stay `Text`;
- IDs, statuses, order states, and enumerations stay `Keyword`;
- dates stay date-typed;
- relations capture customer ownership, order lines, supplier ties, warehouse
  assignment, and payment linkage.

Question classes:

- semantic entity lookup by description;
- owner lookups through relations;
- product and supplier joins;
- order state filtering with multi-value statuses;
- date-based invoice lookup;
- sales-order counts.

Current result:

- `erp-basic`: `9/11` passed

Current known failures:

- `customer_by_description`
- `warehouse_by_description`

These are useful failures, not noise. They show where the semantic boundary is
still being tuned for multi-hop ERP data and where the ontology descriptions can
be made more explicit.

Example:

```text
NL
Какие заказы находятся в активной обработке или ожидают следующего действия?

DSL
find SalesOrder where so.status in ["открыт", "ожидает"] return id, name, status

Cypher
MATCH (so:SalesOrder:E2E_ERP_BASIC)
WHERE (so.status = $p0 OR so.status = $p1)
RETURN so.id AS id, so.name AS name, so.status AS status
```

What this shows business users: the system can normalize operational language
into a controlled query over a business vocabulary, including multi-value
statuses.

What this shows engineers: the DSL can carry structured filter intent (`in`,
`eq`, date ranges, aggregates) without exposing raw Cypher to the model.

### Canonical semantic sandbox

The canonical-semantic suite isolates the semantic retrieval path.

Data shape:

- places
- people
- cameras
- face events

Ontology emphasis:

- semantically meaningful text fields are exposed as `Text`;
- the `_canonical` document is the retrieval anchor;
- exact identifiers remain `Keyword`;
- the suite tests the pure semantic path separately from broader graph logic.

Question classes:

- place lookup by descriptive phrase;
- person lookup by list-based name phrase.

Current result:

- `canonical-semantic`: `2/2` passed

Example:

```text
NL
Найди место по фразе из его описания: проспект Мангилик Ел, 55/22, Астана

DSL
search Place.description ~ "проспект Мангилик Ел, 55/22, Астана"

Cypher
CALL libqlink.search_hybrid_reranked(...)
MATCH (p:Place:E2E_CANONICAL_SEMANTIC)
WHERE id(p) = p__qid_0
RETURN p.name, p.description
```

What this shows business users: semantic search can recover the right record
from natural phrasing, even when the query is not an exact key match.

What this shows engineers: the canonical embedding path is a distinct, testable
unit.

## Validation Philosophy

The e2e framework validates multiple things at once:

- `row_count`: cardinality expectations (`exact`, `min`, `max`);
- `rows`: exact field/value subsets when order is not relevant;
- `contains`: value presence in one or all columns;
- `numbers`: numeric comparisons for aggregates;
- `dsl_expect`: structural checks on the generated DSL;
- optional answer checks for synthesized natural-language responses.

This is deliberate. It avoids the usual trap where the test only checks that a
query ran. Here we check that:

- the model picked the right entity type;
- the right filter operator was used;
- the right traversal was chosen;
- the result set contains the expected business fact.

## Why This Is A Different Data Workflow

Traditional NL-to-SQL / NL-to-Cypher systems ask the model to write the query
language directly. That makes the output hard to validate and easy to break.

`linguagraph` uses a different contract:

- the model emits a compact DSL, not raw Cypher;
- the library validates the DSL structurally;
- the library compiles the DSL into parameterized Cypher;
- semantic fields are routed through a dedicated retrieval path;
- the e2e harness checks the entire stack on real fixtures.

That gives you a new operating model:

- business users ask questions in normal language;
- engineers define the ontology and the allowed graph behavior;
- the system preserves structure, safety, and repeatability;
- domain sandboxes can be added for CRM, ERP, cameras, or any other graph
  surface without changing the core pipeline.

## How To Run

Run the default suite:

```bash
./scripts/run-e2e.sh
```

Run a specific domain suite:

```bash
./scripts/run-e2e.sh examples/e2e/crm.suite.json
./scripts/run-e2e.sh examples/e2e/erp.suite.json
./scripts/run-e2e.sh examples/e2e/canonical-semantic.suite.json
```

Generate a report:

```bash
cargo run --bin linguagraph-e2e -- \
  --config config.e2e.toml \
  --suite examples/e2e/crm.suite.json \
  --report /tmp/crm.report.json
```

## Practical Takeaway

The main outcome is not just query correctness. It is the ability to turn
domain data into a stable semantic interface:

- ontology describes the business shape;
- questions exercise business language;
- the model chooses constrained query intent;
- the library executes and validates the result;
- the report captures both the generated DSL and the final Cypher.

That is what makes the system suitable as a real integration layer for CRM,
ERP, and other operational datasets.
