#!/usr/bin/env bash
set -euo pipefail

# Unified submit entrypoint.
#
# Usage:
#   bash submit.sh <target> [options]
#
# Targets:
#   benchmark      Submit benchmark_*.cmd via queue-filler logic (default)
#   compute        Submit compute_*.cmd via queue-filler logic
#   true           Submit *_true.cmd via queue-filler logic
#   hpo            Submit HPO benchmark cmd files
#   hpo-benchmark  Submit HPO benchmark cmd files
#   hpo-training   Submit HPO training cmd files

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

TARGET="${1:-benchmark}"
if [[ $# -gt 0 ]]; then
  shift
fi

case "$TARGET" in
  benchmark)
    exec bash scripts/submit_until_limit.sh \
      --cluster serial \
      --partition serial_std \
      --auto-array-from-info \
      --cmd-glob 'benchmark_*.cmd' \
      --state-file '.submit_until_limit.state.tsv' \
      "$@"
    ;;
  compute)
    exec bash scripts/submit_compute_until_limit.sh "$@"
    ;;
  true)
    exec bash scripts/submit_true_until_limit.sh "$@"
    ;;
  hpo|hpo-benchmark)
    exec bash scripts/submit_hpo_benchmarks_until_limit.sh "$@"
    ;;
  hpo-training)
    exec bash scripts/submit_hpo_training_until_limit.sh "$@"
    ;;
  -h|--help|help)
    cat <<'EOF'
Usage:
  bash submit.sh <target> [options]

Targets:
  benchmark      Submit benchmark_*.cmd jobs (default)
  compute        Submit compute_*.cmd jobs
  true           Submit *_true.cmd jobs
  hpo            Submit HPO benchmark cmd jobs
  hpo-benchmark  Submit HPO benchmark cmd jobs
  hpo-training   Submit HPO training cmd jobs

Examples:
  bash submit.sh compute --once --dry-run
  bash submit.sh benchmark --limit 150
  HPO_CLUSTER=hlai HPO_LIMIT=1000000 bash submit.sh hpo --once --dry-run
EOF
    ;;
  *)
    echo "Unknown target: $TARGET" >&2
    echo "Run 'bash submit.sh --help' for usage." >&2
    exit 2
    ;;
esac
