#!/usr/bin/env bash
set -euo pipefail

# Backward-compatible alias for HPO benchmark submission.
exec bash "$(dirname "${BASH_SOURCE[0]}")/submit_hpo_benchmarks_until_limit.sh" "$@"
