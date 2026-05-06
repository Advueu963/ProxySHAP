#!/usr/bin/env bash
set -euo pipefail

# Convenience wrapper for submit_until_limit.sh targeting HPO benchmark jobs.
#
# Defaults can be overridden via environment variables:
#   HPO_BENCHMARK_LIMIT, HPO_BENCHMARK_CLUSTER, HPO_BENCHMARK_PARTITION,
#   HPO_BENCHMARK_CMD_GLOB
#
# Examples:
#   bash scripts/submit_hpo_benchmarks_until_limit.sh
#   HPO_BENCHMARK_CLUSTER=serial bash scripts/submit_hpo_benchmarks_until_limit.sh --once --dry-run

cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1

HPO_BENCHMARK_LIMIT="${HPO_BENCHMARK_LIMIT:-200}"
HPO_BENCHMARK_CLUSTER="${HPO_BENCHMARK_CLUSTER:-}"
HPO_BENCHMARK_PARTITION="${HPO_BENCHMARK_PARTITION:-}"
HPO_BENCHMARK_CMD_GLOB="${HPO_BENCHMARK_CMD_GLOB:-benchmark_hpo*.cmd}"

declare -a WRAPPER_ARGS=(
  --limit "$HPO_BENCHMARK_LIMIT"
  --cmd-glob "$HPO_BENCHMARK_CMD_GLOB"
  --state-file ".submit_hpo_until_limit.state.tsv"
)

if [[ -n "$HPO_BENCHMARK_CLUSTER" ]]; then
  WRAPPER_ARGS+=(--cluster "$HPO_BENCHMARK_CLUSTER")
fi

if [[ -n "$HPO_BENCHMARK_PARTITION" ]]; then
  WRAPPER_ARGS+=(--partition "$HPO_BENCHMARK_PARTITION")
fi

exec bash scripts/submit_until_limit.sh "${WRAPPER_ARGS[@]}" "$@"