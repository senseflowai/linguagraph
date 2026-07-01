#!/usr/bin/env bash
set -euo pipefail

suite="${1:-examples/e2e/camera.suite.json}"
shift || true

suite_name="$(basename "${suite%.suite.json}")"
report="${E2E_REPORT:-target/e2e/${suite_name}.report.json}"

cargo run --bin linguagraph-e2e -- --suite "$suite" --report "$report" "$@"
