#!/usr/bin/env bash
set -euo pipefail

# Convenience wrapper for submit_until_limit.sh targeting *_true.cmd scripts.
#
# Submits true-value jobs while respecting a queue cap, polling for free slots,
# and tracking progress in a dedicated state file.
# Defaults target the serial cluster/partition, but can be overridden via args.
#
# Examples:
#   bash scripts/submit_true_until_limit.sh
#   bash scripts/submit_true_until_limit.sh --once --dry-run
#   bash scripts/submit_true_until_limit.sh --limit 120

cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1

exec bash scripts/submit_until_limit.sh \
  --cluster serial \
  --partition serial_std \
  --cmd-glob '*_true.cmd' \
  --state-file ".submit_true_until_limit.state.tsv" \
  "$@"
