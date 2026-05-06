#!/usr/bin/env bash
set -euo pipefail

# Fill a SLURM queue up to a user-defined limit by submitting .cmd files with sbatch.
#
# Features:
# - Respects a global queue cap (default 200 jobs)
# - Polls periodically and auto-submits when slots become free
# - Dynamically chunks SLURM array jobs based on currently free slots
# - Tracks per-file progress in a local state file
# - Avoids duplicate submissions across restarts
#
# Examples:
#   bash scripts/submit_until_limit.sh \
#     --cluster serial \
#     --partition serial_long \
#     --cmd-glob 'benchmark_*.cmd'
#
#   bash scripts/submit_until_limit.sh --once --limit 200 precompute_*.cmd

LIMIT=200
RESERVE=0
POLL_SECONDS=30
CLUSTER=""
PARTITION=""
STATE_FILE=".submit_until_limit.state.tsv"
CMD_GLOB="benchmark_*.cmd"
ONCE=0
DRY_RUN=0
STATE_VERSION="v2"
HLAI_EXCLUDE_NODE="${HLAI_EXCLUDE_NODE:-hpdar06c01s01}"
AUTO_ARRAY_FROM_INFO=0

usage() {
  cat <<'EOF'
Usage:
  submit_until_limit.sh [options] [cmd_file ...]

Options:
  --limit N            Max number of your submitted jobs in queue (default: 200)
  --reserve N          Keep N free slots as buffer (default: 0)
  --poll-seconds N     Poll interval in seconds (default: 30)
  --cluster NAME       SLURM cluster for squeue query (passed to -M)
  --partition NAME     SLURM partition for squeue query (passed to -p)
  --cmd-glob PATTERN   Glob if no positional files are given (default: benchmark_*.cmd)
  --state-file PATH    File that stores already submitted cmd files
  --auto-array-from-info
                      For benchmark_slurm cmd files: derive array size from
                      'experiments/benchmark_slurm.py --mode info' based on
                      the cmd's --parallel_dims (fallback to #SBATCH --array)
  --once               Run one cycle only (no waiting loop)
  --dry-run            Print what would be submitted without calling sbatch
  -h, --help           Show this help

Notes:
  - Positional cmd files override --cmd-glob.
  - State file stores per-file progress for dynamic array chunk submission.
  - If you intentionally want to re-submit from scratch, delete/reset the state file.
EOF
}

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*"
}

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    log "ERROR: required command not found: $1"
    exit 1
  fi
}

while (($# > 0)); do
  case "$1" in
    --limit)
      LIMIT="$2"
      shift 2
      ;;
    --reserve)
      RESERVE="$2"
      shift 2
      ;;
    --poll-seconds)
      POLL_SECONDS="$2"
      shift 2
      ;;
    --cluster)
      CLUSTER="$2"
      shift 2
      ;;
    --partition)
      PARTITION="$2"
      shift 2
      ;;
    --cmd-glob)
      CMD_GLOB="$2"
      shift 2
      ;;
    --state-file)
      STATE_FILE="$2"
      shift 2
      ;;
    --auto-array-from-info)
      AUTO_ARRAY_FROM_INFO=1
      shift
      ;;
    --once)
      ONCE=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      break
      ;;
    -*)
      log "ERROR: unknown option: $1"
      usage
      exit 2
      ;;
    *)
      break
      ;;
  esac
done

if ! [[ "$LIMIT" =~ ^[0-9]+$ && "$RESERVE" =~ ^[0-9]+$ && "$POLL_SECONDS" =~ ^[0-9]+$ ]]; then
  log "ERROR: --limit, --reserve and --poll-seconds must be non-negative integers"
  exit 2
fi

if (( RESERVE >= LIMIT )); then
  log "ERROR: --reserve must be smaller than --limit"
  exit 2
fi

EFFECTIVE_LIMIT="$LIMIT"

require_cmd squeue
require_cmd sbatch
require_cmd realpath

if (( AUTO_ARRAY_FROM_INFO == 1 )); then
  require_cmd uv
fi

SERIAL_LONG_AVAILABLE=""

is_partition_available() {
  local target_partition="$1"
  local -a cmd=(sinfo -h -o '%P')

  if [[ -z "$target_partition" ]]; then
    return 1
  fi

  if ! command -v sinfo >/dev/null 2>&1; then
    return 1
  fi

  [[ -n "$CLUSTER" ]] && cmd+=(-M "$CLUSTER")

  "${cmd[@]}" 2>/dev/null \
    | tr ',' '\n' \
    | sed 's/\*$//' \
    | awk '{print $1}' \
    | grep -Fxq "$target_partition"
}

serial_long_available() {
  if [[ -z "$SERIAL_LONG_AVAILABLE" ]]; then
    if is_partition_available "serial_long"; then
      SERIAL_LONG_AVAILABLE=1
    else
      SERIAL_LONG_AVAILABLE=0
    fi
  fi

  [[ "$SERIAL_LONG_AVAILABLE" == "1" ]]
}

append_cluster_scope_args() {
  local -n target="$1"
  local target_partition="$2"

  if [[ -n "$CLUSTER" ]]; then
    target+=(--clusters "$CLUSTER")
  fi

  if [[ "$CLUSTER" == "hlai" && -n "$HLAI_EXCLUDE_NODE" ]]; then
    target+=(--exclude "$HLAI_EXCLUDE_NODE")
  fi

  if [[ -n "$target_partition" ]]; then
    target+=(--partition "$target_partition")
  fi
}

# Build command list (positional files or glob fallback).
declare -a CMD_FILES=()
if (($# > 0)); then
  while (($# > 0)); do
    CMD_FILES+=("$1")
    shift
  done
else
  shopt -s nullglob
  for f in $CMD_GLOB; do
    CMD_FILES+=("$f")
  done
  shopt -u nullglob
fi

if ((${#CMD_FILES[@]} == 0)); then
  log "No cmd files found. Nothing to submit."
  exit 0
fi

# Normalize and validate files.
declare -a NORMALIZED=()
for f in "${CMD_FILES[@]}"; do
  if [[ ! -f "$f" ]]; then
    log "WARNING: skipping missing file: $f"
    continue
  fi
  if [[ "$f" != *.cmd ]]; then
    log "WARNING: skipping non-.cmd file: $f"
    continue
  fi
  NORMALIZED+=("$(realpath "$f")")
done

if ((${#NORMALIZED[@]} == 0)); then
  log "No valid .cmd files after filtering."
  exit 0
fi

IFS=$'\n' read -r -d '' -a ALL_CMDS < <(printf '%s\n' "${NORMALIZED[@]}" | sort -u && printf '\0')

mkdir -p "$(dirname "$STATE_FILE")"
if [[ ! -f "$STATE_FILE" ]]; then
  printf '# %s\n' "$STATE_VERSION" > "$STATE_FILE"
fi

declare -A IS_TARGET=()
for f in "${ALL_CMDS[@]}"; do
  IS_TARGET["$f"]=1
done

# Per-file progress and metadata.
declare -A MODE=()       # array | single
declare -A NEXT_IDX=()   # next index to submit (single: 0 -> not submitted, 1 -> done)
declare -A END_IDX=()    # inclusive end index for arrays (single: 0)
declare -A STEP_SZ=()    # array step (single: 1)
declare -A CONCURRENCY=() # optional array %limit
declare -A SEEN_IN_STATE=()

trim_ws() {
  local s="$1"
  s="${s#${s%%[![:space:]]*}}"
  s="${s%${s##*[![:space:]]}}"
  printf '%s' "$s"
}

parse_sbatch_value() {
  local cmd_file="$1"
  local key="$2"
  local line raw

  line="$(grep -m1 -E "^#SBATCH[[:space:]]+${key}(=|[[:space:]])" "$cmd_file" || true)"
  if [[ -z "$line" ]]; then
    return 1
  fi

  raw="${line#*${key}}"
  raw="${raw#=}"
  raw="$(trim_ws "$raw")"
  raw="${raw%%[[:space:]]*}"
  printf '%s' "$raw"
  return 0
}

parse_slurm_time_to_seconds() {
  local time_spec="$1"
  local days=0 hours=0 minutes=0 seconds=0
  local rest="$time_spec"
  local part_count

  if [[ "$time_spec" =~ ^([0-9]+)-(.+)$ ]]; then
    days="${BASH_REMATCH[1]}"
    rest="${BASH_REMATCH[2]}"
  fi

  part_count="$(awk -F: '{print NF}' <<<"$rest")"
  if (( days > 0 )); then
    if (( part_count == 1 )); then
      hours="$rest"
    elif (( part_count == 2 )); then
      IFS=':' read -r hours minutes <<<"$rest"
    elif (( part_count == 3 )); then
      IFS=':' read -r hours minutes seconds <<<"$rest"
    else
      return 1
    fi
  else
    if (( part_count == 1 )); then
      minutes="$rest"
    elif (( part_count == 2 )); then
      IFS=':' read -r minutes seconds <<<"$rest"
    elif (( part_count == 3 )); then
      IFS=':' read -r hours minutes seconds <<<"$rest"
    else
      return 1
    fi
  fi

  if ! [[ "$days" =~ ^[0-9]+$ && "$hours" =~ ^[0-9]+$ && "$minutes" =~ ^[0-9]+$ && "$seconds" =~ ^[0-9]+$ ]]; then
    return 1
  fi

  echo $((days * 86400 + hours * 3600 + minutes * 60 + seconds))
  return 0
}

resolve_partition_for_cmd() {
  local cmd_file="$1"
  local chosen_partition="$PARTITION"
  local cmd_partition=""
  local cmd_time=""
  local cmd_seconds=""

  if cmd_partition="$(parse_sbatch_value "$cmd_file" '--partition')"; then
    :
  fi

  if [[ -z "$chosen_partition" && -n "$cmd_partition" ]]; then
    chosen_partition="$cmd_partition"
  fi

  if cmd_time="$(parse_sbatch_value "$cmd_file" '--time')"; then
    if cmd_seconds="$(parse_slurm_time_to_seconds "$cmd_time")"; then
      if [[ "$CLUSTER" == "serial" && "$cmd_seconds" =~ ^[0-9]+$ && $cmd_seconds -gt 86400 ]]; then
        if serial_long_available; then
          if [[ "$chosen_partition" != "serial_long" ]]; then
            log "INFO: forcing partition serial_long for $cmd_file (time=$cmd_time > 24h)" >&2
          fi
          chosen_partition="serial_long"
        else
          log "WARNING: serial_long partition not available on cluster ${CLUSTER:-default}; keeping partition ${chosen_partition:-<none>} for $cmd_file" >&2
        fi
      fi
    fi
  fi

  printf '%s' "$chosen_partition"
}

parse_array_meta() {
  local cmd_file="$1"
  local line spec raw conc start end step
  local inferred_end=""

  if (( AUTO_ARRAY_FROM_INFO == 1 )); then
    if inferred_end="$(infer_array_end_from_info "$cmd_file")" && [[ "$inferred_end" =~ ^[0-9]+$ ]]; then
      echo "array 0 $inferred_end 1"
      return 0
    fi
  fi

  line="$(grep -m1 -E '^#SBATCH[[:space:]]+--array(=|[[:space:]])' "$cmd_file" || true)"
  if [[ -z "$line" ]]; then
    echo "single 0 0 1"
    return 0
  fi

  raw="${line#*--array}"
  raw="${raw#=}"
  raw="$(trim_ws "$raw")"
  raw="${raw%%[[:space:]]*}"

  conc=""
  spec="$raw"
  if [[ "$spec" == *%* ]]; then
    conc="${spec#*%}"
    spec="${spec%%%*}"
  fi

  if [[ "$spec" == *,* ]]; then
    log "ERROR: unsupported --array list syntax in $cmd_file ($raw). Use a single range."
    return 1
  fi

  if [[ "$spec" =~ ^([0-9]+)$ ]]; then
    start="${BASH_REMATCH[1]}"
    end="$start"
    step=1
    echo "array $start $end $step $conc"
    return 0
  fi

  if [[ "$spec" =~ ^([0-9]+)-([0-9]+)(:([0-9]+))?$ ]]; then
    start="${BASH_REMATCH[1]}"
    end="${BASH_REMATCH[2]}"
    step="${BASH_REMATCH[4]:-1}"
    if (( end < start || step <= 0 )); then
      log "ERROR: invalid --array range in $cmd_file ($raw)."
      return 1
    fi
    echo "array $start $end $step $conc"
    return 0
  fi

  log "ERROR: unsupported --array syntax in $cmd_file ($raw)."
  return 1
}

extract_arg_value_from_cmd_line() {
  local cmd_line="$1"
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
  ' <<<"$cmd_line"
}

infer_array_end_from_info() {
  local cmd_file="$1"
  local py_line=""
  local config=""
  local game_type=""
  local parallel_dims="game_approx"
  local n_budget_steps="20"
  local section_header=""
  local info_out=""
  local inferred_end=""

  py_line="$(grep -E 'uv run python[[:space:]]+experiments/benchmark_slurm.py' "$cmd_file" | head -n1 || true)"
  if [[ -z "$py_line" ]]; then
    return 1
  fi

  config="$(extract_arg_value_from_cmd_line "$py_line" "--config")"
  game_type="$(extract_arg_value_from_cmd_line "$py_line" "--game_type")"

  if [[ -z "$config" || -z "$game_type" ]]; then
    return 1
  fi

  if v="$(extract_arg_value_from_cmd_line "$py_line" "--parallel_dims")" && [[ -n "$v" ]]; then
    parallel_dims="$v"
  fi

  if v="$(extract_arg_value_from_cmd_line "$py_line" "--n_budget_steps")" && [[ -n "$v" ]]; then
    n_budget_steps="$v"
  fi

  case "$parallel_dims" in
    game)
      section_header="─── game mode"
      ;;
    game_approx)
      section_header="─── game_approx mode"
      ;;
    game_approx_explain)
      section_header="─── game_approx_explain mode"
      ;;
    *)
      return 1
      ;;
  esac

  if ! info_out="$(uv run python experiments/benchmark_slurm.py --config "$config" --game_type "$game_type" --mode info --parallel_dims "$parallel_dims" --n_budget_steps "$n_budget_steps" 2>/dev/null)"; then
    return 1
  fi

  inferred_end="$(awk -v header="$section_header" '
    $0 ~ header {in_section=1; next}
    /^─── / && in_section {in_section=0}
    in_section && match($0, /#SBATCH --array=0-([0-9]+)/, m) {print m[1]; exit}
  ' <<<"$info_out")"

  if [[ "$inferred_end" =~ ^[0-9]+$ ]]; then
    printf '%s' "$inferred_end"
    return 0
  fi

  return 1
}

append_state_entry() {
  local f="$1"
  local tmp
  tmp="${STATE_FILE}.tmp"

  if [[ ! -f "$STATE_FILE" ]]; then
    printf '# %s\n' "$STATE_VERSION" > "$STATE_FILE"
  fi

  {
    awk -v target="$f" 'BEGIN { header=0 }
      /^#/ && header == 0 { print; header=1; next }
      $1 != target { print }
    ' "$STATE_FILE"
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$f" \
      "${MODE[$f]}" \
      "${NEXT_IDX[$f]}" \
      "${END_IDX[$f]}" \
      "${STEP_SZ[$f]}" \
      "${CONCURRENCY[$f]}"
  } > "$tmp"

  mv "$tmp" "$STATE_FILE"
}

# Initialize metadata from cmd files.
for f in "${ALL_CMDS[@]}"; do
  if ! meta="$(parse_array_meta "$f")"; then
    exit 1
  fi
  read -r mode start end step conc <<<"$meta"
  MODE["$f"]="$mode"
  NEXT_IDX["$f"]="$start"
  END_IDX["$f"]="$end"
  STEP_SZ["$f"]="$step"
  CONCURRENCY["$f"]="${conc:-}"
done

# Load stored progress (if compatible) and override defaults.
while IFS=$'\t' read -r cmd_path mode next end step conc; do
  [[ -n "${cmd_path:-}" ]] || continue
  [[ "${cmd_path:0:1}" == "#" ]] && continue
  [[ -n "${IS_TARGET[$cmd_path]+x}" ]] || continue
  SEEN_IN_STATE["$cmd_path"]=1

  if [[ "$mode" != "array" && "$mode" != "single" ]]; then
    continue
  fi
  if ! [[ "$next" =~ ^[0-9]+$ && "$end" =~ ^[0-9]+$ && "$step" =~ ^[0-9]+$ ]]; then
    continue
  fi

  if (( AUTO_ARRAY_FROM_INFO == 1 )) && [[ "${MODE[$cmd_path]}" == "array" ]]; then
    # Keep dynamically inferred array end/step, only restore progress pointer.
    NEXT_IDX["$cmd_path"]="$next"
    CONCURRENCY["$cmd_path"]="${conc:-}"
  else
    MODE["$cmd_path"]="$mode"
    NEXT_IDX["$cmd_path"]="$next"
    END_IDX["$cmd_path"]="$end"
    STEP_SZ["$cmd_path"]="$step"
    CONCURRENCY["$cmd_path"]="${conc:-}"
  fi
done < "$STATE_FILE"

# Append initial entries for newly discovered cmd files.
for f in "${ALL_CMDS[@]}"; do
  if [[ -z "${SEEN_IN_STATE[$f]+x}" ]]; then
    append_state_entry "$f"
  fi
done

count_current_jobs() {
  # Use -r to expand array elements so each task counts toward submit limits.
  # Do not apply partition filtering here: most submit limits are account/user scoped.
  local -a cmd=(squeue -h -r -u "$USER" -o '%i')

  if [[ -n "$CLUSTER" ]]; then
    cmd+=(-M "$CLUSTER")
  else
    # When no cluster is specified, count jobs across all visible clusters.
    cmd+=(-M all)
  fi

  "${cmd[@]}" | wc -l | tr -d ' '
}

count_partition_jobs() {
  local -a cmd=(squeue -h -r -u "$USER" -o '%i')

  if [[ -n "$CLUSTER" ]]; then
    cmd+=(-M "$CLUSTER")
  else
    cmd+=(-M all)
  fi

  [[ -n "$PARTITION" ]] && cmd+=(-p "$PARTITION")

  "${cmd[@]}" | wc -l | tr -d ' '
}

submit_file() {
  local cmd_file="$1"
  local array_spec="${2:-}"
  local sbatch_out
  local tmp_cmd=""
  local effective_partition
  local -a sbatch_cmd=(sbatch)

  effective_partition="$(resolve_partition_for_cmd "$cmd_file")"
  append_cluster_scope_args sbatch_cmd "$effective_partition"

  if [[ -n "$array_spec" ]]; then
    sbatch_cmd+=(--array "$array_spec")
  fi
  sbatch_cmd+=("$cmd_file")

  if (( DRY_RUN == 1 )); then
    log "DRY-RUN: ${sbatch_cmd[*]}"
    return 0
  fi

  if ! sbatch_out="$("${sbatch_cmd[@]}" 2>&1)"; then
    log "ERROR: sbatch failed for $cmd_file"
    log "ERROR: $sbatch_out"

    if grep -q 'Invalid node name specified\|Invalid partition name specified\|invalid partition specified' <<<"$sbatch_out"; then
      local retry_partition="$effective_partition"

      if grep -qi 'Invalid partition name specified\|invalid partition specified' <<<"$sbatch_out"; then
        retry_partition=""
      fi

      # Some cmd files contain cluster-specific SBATCH directives that are invalid
      # for the currently forced target cluster.
      tmp_cmd="$(mktemp "${TMPDIR:-/tmp}/submit_until_limit.XXXXXX.cmd")"
      sed -E '/^#SBATCH[[:space:]]+--exclude(=|[[:space:]])/d; /^#SBATCH[[:space:]]+-x([[:space:]]|=)/d; /^#SBATCH[[:space:]]+--partition(=|[[:space:]])/d; /^#SBATCH[[:space:]]+-p([[:space:]]|=)/d; /^#SBATCH[[:space:]]+--clusters(=|[[:space:]])/d' "$cmd_file" > "$tmp_cmd"

      sbatch_cmd=(sbatch)
      append_cluster_scope_args sbatch_cmd "$retry_partition"
      if [[ -n "$array_spec" ]]; then
        sbatch_cmd+=(--array "$array_spec")
      fi
      sbatch_cmd+=("$tmp_cmd")

      if ! sbatch_out="$("${sbatch_cmd[@]}" 2>&1)"; then
        rm -f "$tmp_cmd"
        log "ERROR: retry with sanitized SBATCH directives failed for $cmd_file"
        log "ERROR: $sbatch_out"
      else
        local retry_job_id
        retry_job_id="$(awk '/Submitted batch job/{print $4}' <<<"$sbatch_out" | tail -n1)"
        rm -f "$tmp_cmd"
        log "Submitted (retry with sanitized SBATCH directives): ${sbatch_cmd[*]}${retry_job_id:+ (job_id=$retry_job_id)}"
        return 0
      fi
    fi

    if grep -q 'AssocMaxSubmitJobLimit' <<<"$sbatch_out"; then
      return 3
    fi
    return 1
  fi

  # Typical output: Submitted batch job 123456
  local job_id
  job_id="$(awk '/Submitted batch job/{print $4}' <<<"$sbatch_out" | tail -n1)"
  log "Submitted: ${sbatch_cmd[*]}${job_id:+ (job_id=$job_id)}"
  return 0
}

remaining_count() {
  local n=0
  local f
  for f in "${ALL_CMDS[@]}"; do
    if [[ "${MODE[$f]}" == "single" ]]; then
      if (( NEXT_IDX[$f] == 0 )); then
        ((n += 1))
      fi
      continue
    fi

    if (( NEXT_IDX[$f] <= END_IDX[$f] )); then
      # Count remaining array tasks: floor((end - next) / step) + 1
      ((n += ((END_IDX[$f] - NEXT_IDX[$f]) / STEP_SZ[$f]) + 1))
    fi
  done
  echo "$n"
}

log "Starting queue filler with limit=$LIMIT reserve=$RESERVE poll=${POLL_SECONDS}s"
[[ -n "$CLUSTER" ]] && log "Using cluster filter: $CLUSTER"
[[ -n "$PARTITION" ]] && log "Using partition filter: $PARTITION"
log "State file: $STATE_FILE"
log "Total cmd files: ${#ALL_CMDS[@]}"

while true; do
  remaining="$(remaining_count)"
  if (( remaining == 0 )); then
    log "All cmd files have been submitted according to state file. Exiting."
    exit 0
  fi

  current_jobs="$(count_current_jobs)"
  current_partition_jobs="$(count_partition_jobs)"
  free_slots=$((EFFECTIVE_LIMIT - RESERVE - current_jobs))
  if (( free_slots < 0 )); then
    free_slots=0
  fi

  log "Queue status: current_cluster=$current_jobs current_partition=$current_partition_jobs free=$free_slots remaining_to_submit=$remaining effective_limit=$EFFECTIVE_LIMIT"

  if (( free_slots > 0 )); then
    submitted_now=0
    hit_submit_limit=0
    for f in "${ALL_CMDS[@]}"; do
      if (( submitted_now >= free_slots )); then
        break
      fi

      slots_left=$((free_slots - submitted_now))
      if (( slots_left <= 0 )); then
        break
      fi

      if [[ "${MODE[$f]}" == "single" ]]; then
        if (( NEXT_IDX[$f] != 0 )); then
          continue
        fi
        if submit_file "$f"; then
          if (( DRY_RUN == 0 )); then
            NEXT_IDX["$f"]=1
            append_state_entry "$f"
          fi
          ((submitted_now += 1))
        else
          submit_rc=$?
          if (( submit_rc == 3 )); then
            hit_submit_limit=1
            learned_limit=$((current_jobs + submitted_now))
            if (( learned_limit < EFFECTIVE_LIMIT )); then
              EFFECTIVE_LIMIT="$learned_limit"
              log "Adjusted effective limit to $EFFECTIVE_LIMIT due to scheduler submit policy."
            fi
            log "Submit limit reached according to scheduler. Stopping submissions for this cycle."
            break
          fi
        fi
        continue
      fi

      # Array mode: submit only as many tasks as currently free.
      local_next="${NEXT_IDX[$f]}"
      local_end="${END_IDX[$f]}"
      local_step="${STEP_SZ[$f]}"
      local_conc="${CONCURRENCY[$f]}"

      if (( local_next > local_end )); then
        continue
      fi

      remaining_for_file=$((((local_end - local_next) / local_step) + 1))
      chunk_tasks="$slots_left"
      if (( chunk_tasks > remaining_for_file )); then
        chunk_tasks="$remaining_for_file"
      fi

      chunk_last=$((local_next + (chunk_tasks - 1) * local_step))
      if (( chunk_last > local_end )); then
        chunk_last="$local_end"
      fi

      if (( local_step == 1 )); then
        if (( local_next == chunk_last )); then
          chunk_spec="$local_next"
        else
          chunk_spec="${local_next}-${chunk_last}"
        fi
      else
        chunk_spec="${local_next}-${chunk_last}:${local_step}"
      fi
      if [[ -n "$local_conc" ]]; then
        chunk_spec+="%${local_conc}"
      fi

      if submit_file "$f" "$chunk_spec"; then
        if (( DRY_RUN == 0 )); then
          NEXT_IDX["$f"]=$((chunk_last + local_step))
          append_state_entry "$f"
        fi
        ((submitted_now += chunk_tasks))
      else
        submit_rc=$?
        if (( submit_rc == 3 )); then
          hit_submit_limit=1
          learned_limit=$((current_jobs + submitted_now))
          if (( learned_limit < EFFECTIVE_LIMIT )); then
            EFFECTIVE_LIMIT="$learned_limit"
            log "Adjusted effective limit to $EFFECTIVE_LIMIT due to scheduler submit policy."
          fi
          log "Submit limit reached according to scheduler. Stopping submissions for this cycle."
          break
        fi
      fi
    done
    log "Submitted this cycle: $submitted_now"
    if (( hit_submit_limit == 1 )); then
      log "Will retry after next poll interval."
    fi
  fi

  if (( ONCE == 1 )); then
    log "--once active, stopping after one cycle."
    exit 0
  fi

  sleep "$POLL_SECONDS"
done
