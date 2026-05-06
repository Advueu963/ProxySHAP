#!/usr/bin/env bash
set -euo pipefail

# Convenience wrapper for submit_until_limit.sh targeting HPO training jobs.
#
# Defaults can be overridden via environment variables:
#   HPO_TRAINING_LIMIT, HPO_TRAINING_CLUSTER, HPO_TRAINING_PARTITION,
#   HPO_TRAINING_CMD_GLOB
#
# The default target is benchmark_tabpfn_hpo_training.cmd.

cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1

HPO_TRAINING_LIMIT="${HPO_TRAINING_LIMIT:-1000000}"
HPO_TRAINING_CLUSTER="${HPO_TRAINING_CLUSTER:-hlai}"
HPO_TRAINING_PARTITION="${HPO_TRAINING_PARTITION:-}"
HPO_TRAINING_CMD_GLOB="${HPO_TRAINING_CMD_GLOB:-benchmark_tabpfn_hpo_training.cmd}"

declare -a WRAPPER_ARGS=(
  --limit "$HPO_TRAINING_LIMIT"
  --cluster "$HPO_TRAINING_CLUSTER"
  --cmd-glob "$HPO_TRAINING_CMD_GLOB"
  --state-file ".submit_tabpfn_hpo_training_until_limit.state.tsv"
)

if [[ -n "$HPO_TRAINING_PARTITION" ]]; then
  WRAPPER_ARGS+=(--partition "$HPO_TRAINING_PARTITION")
fi

exec bash scripts/submit_until_limit.sh "${WRAPPER_ARGS[@]}" "$@"