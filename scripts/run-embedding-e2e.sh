#!/usr/bin/env bash
set -euo pipefail

features=()

if [[ "${LINGUAGRAPH_EMBED_E2E_FORCE_CUDA:-0}" == "1" ]]; then
  features+=(--features cuda)
elif command -v nvcc >/dev/null 2>&1; then
  features+=(--features cuda)
fi

if [[ ${#features[@]} -eq 0 ]]; then
  echo "running embedding e2e without CUDA feature"
else
  echo "running embedding e2e with CUDA feature"
fi

./scripts/prepare-cameras-large-e2e.py \
  --input "${CAMERAS_EMBED_INPUT:-/home/df/Downloads/cameras_dump.json}" \
  --limit "${CAMERAS_EMBED_LIMIT:-1000}" \
  --output "${CAMERAS_EMBED_GRAPH:-target/e2e/cameras_1k.graph.json}" \
  --stats "${CAMERAS_EMBED_STATS:-target/e2e/cameras_1k.stats.json}"

cargo run --release "${features[@]}" --bin linguagraph-embedding-e2e -- "$@"
