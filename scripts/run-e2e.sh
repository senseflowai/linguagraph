#!/usr/bin/env bash
set -euo pipefail

suite="examples/e2e/cameras.suite.json"
args=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --suite|-s)
      if [[ $# -lt 2 ]]; then
        echo "missing value for $1" >&2
        exit 2
      fi
      suite="$2"
      shift 2
      ;;
    --suite=*)
      suite="${1#--suite=}"
      shift
      ;;
    -s=*)
      suite="${1#-s=}"
      shift
      ;;
    *.suite.json)
      suite="$1"
      shift
      ;;
    *)
      args+=("$1")
      shift
      ;;
  esac
done

suite_name="$(basename "${suite%.suite.json}")"
report="${E2E_REPORT:-target/e2e/${suite_name}.report.json}"

cargo run --bin linguagraph-e2e -- --suite "$suite" --report "$report" "${args[@]}"
