#!/usr/bin/env bash
set -euo pipefail

suite="${1:-examples/e2e/camera.suite.json}"
shift || true

cargo run --bin linguagraph-e2e -- --suite "$suite" "$@"
