#!/usr/bin/env bash
set -euo pipefail

./scripts/run-e2e.sh examples/e2e/canonical-semantic.suite.json "$@"
