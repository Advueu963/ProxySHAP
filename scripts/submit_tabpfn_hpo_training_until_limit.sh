#!/usr/bin/env bash
set -euo pipefail

# Backward-compatible alias for HPO training submission.
exec bash "$(dirname "${BASH_SOURCE[0]}")/submit_hpo_training_until_limit.sh" "$@"
