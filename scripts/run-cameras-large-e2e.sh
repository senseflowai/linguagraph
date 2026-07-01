#!/usr/bin/env bash
set -euo pipefail

input="${CAMERAS_DUMP:-/home/df/Downloads/cameras_dump.json}"
graph="${CAMERAS_LARGE_GRAPH:-target/e2e/cameras_1k.graph.json}"
stats="${CAMERAS_LARGE_STATS:-target/e2e/cameras_1k.stats.json}"
report="${CAMERAS_LARGE_REPORT:-target/e2e/cameras_1k.report.json}"

python3 scripts/prepare-cameras-large-e2e.py \
  --input "$input" \
  --limit "${CAMERAS_LARGE_LIMIT:-1000}" \
  --output "$graph" \
  --stats "$stats"

graph_abs="$(realpath "$graph")"

cargo run --release --bin linguagraph-e2e -- \
  --config config.e2e.toml \
  --suite examples/e2e/cameras-large.suite.json \
  --graph "$graph_abs" \
  --report "$report" \
  "$@"
