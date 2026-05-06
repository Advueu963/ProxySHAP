#!/usr/bin/env bash
set -euo pipefail

# Submit ground-truth jobs based on benchmark_*.cmd files.
#
# Example:
#   bash scripts/slurm/submit_ground_truth_from_benchmark_cmds.sh --dry-run
#   bash scripts/slurm/submit_ground_truth_from_benchmark_cmds.sh --glob 'benchmark_interventional_tabarena_*.cmd'

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

CMD_GLOB="benchmark_*.cmd"
DRY_RUN=0
OVERRIDE=0
DEFAULT_TIME="24:00:00"

usage() {
  cat <<'EOF'
Usage: submit_ground_truth_from_benchmark_cmds.sh [options]

Options:
  --glob <pattern>       Glob for benchmark command files (default: benchmark_*.cmd)
  --override             Pass --override to experiments/benchmark_slurm.py --mode true
  --time <HH:MM:SS>      Default walltime for ground-truth jobs (default: 24:00:00)
  --dry-run              Print sbatch commands without submitting
  -h, --help             Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --glob)
      CMD_GLOB="$2"
      shift 2
      ;;
    --override)
      OVERRIDE=1
      shift
      ;;
    --time)
      DEFAULT_TIME="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

shopt -s nullglob
cmd_files=( $CMD_GLOB )
shopt -u nullglob

if [[ ${#cmd_files[@]} -eq 0 ]]; then
  echo "No command files found for glob: $CMD_GLOB" >&2
  exit 1
fi

extract_benchmark_python_line() {
  local cmd_file="$1"
  grep -E 'uv run python[[:space:]]+experiments/benchmark_slurm.py' "$cmd_file" | head -n 1
}

extract_arg_value() {
  local line="$1"
  local flag="$2"
  awk -v f="$flag" '
    {
      for (i = 1; i <= NF; i++) {
        if ($i == f && i < NF) {
          print $(i + 1)
          exit
        }
      }
    }
  ' <<< "$line"
}

extract_sbatch_value() {
  local cmd_file="$1"
  local key="$2"
  local value
  value=$(sed -n "s/^#SBATCH[[:space:]]\+${key}=//p" "$cmd_file" | head -n 1 || true)
  echo "$value"
}

for cmd_file in "${cmd_files[@]}"; do
  py_line=$(extract_benchmark_python_line "$cmd_file" || true)
  if [[ -z "$py_line" ]]; then
    echo "[WARN] Skip $cmd_file (no benchmark_slurm.py line found)."
    continue
  fi

  config=$(extract_arg_value "$py_line" "--config")
  game_type=$(extract_arg_value "$py_line" "--game_type")
  config_approx=$(extract_arg_value "$py_line" "--config_approximators")

  if [[ -z "${config:-}" || -z "${game_type:-}" ]]; then
    echo "[WARN] Skip $cmd_file (cannot parse --config/--game_type)."
    continue
  fi
  if [[ -z "${config_approx:-}" ]]; then
    config_approx=37
  fi

  if [[ ! -f "$config" ]]; then
    echo "[WARN] Skip $cmd_file (config not found: $config)."
    continue
  fi

  base_name=$(basename "$cmd_file" .cmd)
  suffix="${base_name#benchmark_}"
  job_slug=$(echo "$suffix" | tr '[:lower:]' '[:upper:]')

  cluster=$(extract_sbatch_value "$cmd_file" "--clusters")
  partition=$(extract_sbatch_value "$cmd_file" "--partition")
  cpus=$(extract_sbatch_value "$cmd_file" "--cpus-per-task")
  mail_user=$(extract_sbatch_value "$cmd_file" "--mail-user")

  cluster_args=()
  partition_args=()
  cpus_args=()
  mail_args=()

  if [[ -n "$cluster" ]]; then
    cluster_args+=("--clusters=$cluster")
  fi
  if [[ -n "$partition" ]]; then
    partition_args+=("--partition=$partition")
  fi
  if [[ -n "$cpus" ]]; then
    cpus_args+=("--cpus-per-task=$cpus")
  else
    cpus_args+=("--cpus-per-task=1")
  fi
  if [[ -n "$mail_user" ]]; then
    mail_args+=("--mail-user=$mail_user")
  fi

  override_flag=""
  if [[ "$OVERRIDE" -eq 1 ]]; then
    override_flag=" --override"
  fi

  gt_wrap="module load slurm_setup; export OMP_NUM_THREADS=\$SLURM_CPUS_PER_TASK; uv run python experiments/benchmark_slurm.py --config $config --game_type $game_type --mode true --config_approximators $config_approx$override_flag"

  submit_cmd=(
    sbatch
    --parsable
    "${cluster_args[@]}"
    "${partition_args[@]}"
    "${cpus_args[@]}"
    --time="$DEFAULT_TIME"
    "${mail_args[@]}"
    --job-name="GT_${job_slug}"
    --output="./GT_${job_slug}.%j.%N.out"
    --error="./GT_${job_slug}.%j.%N.err"
    --wrap "$gt_wrap"
  )

  filtered_submit_cmd=()
  for x in "${submit_cmd[@]}"; do
    [[ -n "$x" ]] && filtered_submit_cmd+=("$x")
  done

  echo "[INFO] $cmd_file"
  echo "       config=$config game_type=$game_type"

  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "DRY-RUN submit: ${filtered_submit_cmd[*]}"
    echo
    continue
  fi

  job_id=$("${filtered_submit_cmd[@]}")
  echo "  submitted ground-truth job: $job_id"
  echo

done

echo "Done."
