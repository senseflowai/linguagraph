#!/usr/bin/env bash
set -euo pipefail

features=()

if [[ "${LINGUAGRAPH_EMBED_BENCH_FORCE_CUDA:-0}" == "1" ]]; then
  features+=(--features cuda)
elif command -v nvcc >/dev/null 2>&1; then
  features+=(--features cuda)
fi

if [[ ${#features[@]} -eq 0 ]]; then
  echo "running embedding bench without CUDA feature"
else
  echo "running embedding bench with CUDA feature"
fi

cargo run --release "${features[@]}" --bin linguagraph-embed-bench -- "$@"
