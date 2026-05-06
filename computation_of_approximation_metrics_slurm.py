"""SLURM-optimised metrics computation for Shapley interaction approximations.

Reads results directly from shard files produced by `experiments/benchmark_slurm.py`,
avoiding the need for a separate merge step.  Falls back to individual `.json` files
when no shards are present (backward-compatible with local runs).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 Parallelisation model
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
One SLURM array task = one game_id from the benchmark config JSON.
All (id_explain, approximator, budget) combinations for that game are
processed sequentially within the task — they are fast (I/O + math only).

  Array size = number of games in the config JSON.

Each task writes one partial CSV to {base}/_metrics_partial/.
A final --mode reduce step concatenates them into the standard
results_benchmark_{INDEX}_{ORDER}_{game_type}.csv that plot_approximation.py
already expects.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 Modes
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  compute  [default]
      Run metrics for one game (determined by SLURM_ARRAY_TASK_ID or --task_id).
      Reads shards from approximations/{game_type}/_shards/, falls back to
      individual files if no shards exist.  Writes a partial CSV.

  reduce
      Concatenate all partial CSVs → results_benchmark_{INDEX}_{ORDER}_{game_type}.csv.
      Run locally after all SLURM tasks finish.

  info
      Print how many tasks are needed (= number of games in the config).
      Does not load any game data.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 Typical workflow
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  # 1. Find array size
  uv run python computation_of_approximation_metrics_slurm.py \\
      --config shapiq-benchmark/benchmarks/exhaustive/configuration_exhaustive_shapley_order2.json \\
      --game_type exhaustive --index SII --order 2 --mode info

  # 2. Submit (reads shards directly — no merge step from benchmark_slurm.py needed)
  sbatch --array=0-6 slurm/run_metrics.sh

  # 3. Combine partial results
  uv run python computation_of_approximation_metrics_slurm.py \\
      --config ... --game_type exhaustive --index SII --order 2 --mode reduce

  # 4. Plot (unchanged)
  uv run python plot_approximation.py
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import pandas as pd
from shapiq import InteractionValues
from shapiq.game import Game
from shapiq.utils.saving import dict_to_lookup_and_values
from shapiq_benchmark.configuration import GAME_NAME_TO_CLASS_MAPPING
from shapiq_benchmark.load import GameFactory
from shapiq_benchmark.metrics import get_all_metrics


# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SLURM-optimised metrics computation for Shapley interaction approximations.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config", required=True,
        help="Path to JSON benchmark config file (same as benchmark_slurm.py).",
    )
    parser.add_argument(
        "--game_type", required=True,
        help="Game type / approximations subdirectory (e.g. exhaustive, interventional).",
    )
    parser.add_argument(
        "--mode", default="compute", choices=["compute", "reduce", "info"],
        help=(
            "compute: run metrics for one game (SLURM task). "
            "reduce: merge partial CSVs → final CSV. "
            "info: print array size. Default: compute."
        ),
    )
    parser.add_argument(
        "--config_approximators", type=int, default=37,
        help=(
            "Approximator configuration ID used during benchmark_slurm.py: "
            "37 (PAIRING=True, REPLACEMENT=False), "
            "38 (PAIRING=True, REPLACEMENT=True), "
            "39 (PAIRING=False, REPLACEMENT=False), "
            "40 (PAIRING=False, REPLACEMENT=True). Default: 37."
        ),
    )
    parser.add_argument(
        "--index", type=str, default="SII",
        help="Interaction index (e.g. SV, SII, FBII). Default: SII.",
    )
    parser.add_argument(
        "--order", type=int, default=2,
        help="Interaction order. Default: 2.",
    )
    parser.add_argument(
        "--output_path", type=str, default=None,
        help="Base output directory. Defaults to $SCRATCH_DSS/neurips_tree or cwd.",
    )
    parser.add_argument(
        "--override", action="store_true", default=False,
        help="Recompute partial CSVs that already exist.",
    )
    parser.add_argument(
        "--task_id", type=int, default=None,
        help="Override SLURM_ARRAY_TASK_ID for local testing.",
    )
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Path helpers
# ─────────────────────────────────────────────────────────────────────────────

def resolve_base_path(args: argparse.Namespace) -> Path:
    if args.output_path:
        return Path(args.output_path)
    if "SCRATCH_DSS" in os.environ:
        scratch_path = Path(os.environ["SCRATCH_DSS"])
        return scratch_path / "neurips_tree"
    return Path.cwd()


def get_task_id(args: argparse.Namespace) -> int | None:
    if args.task_id is not None:
        return args.task_id
    raw = os.environ.get("SLURM_ARRAY_TASK_ID")
    return int(raw) if raw is not None else None


# ─────────────────────────────────────────────────────────────────────────────
# InteractionValues reconstruction from a shard result dict
# ─────────────────────────────────────────────────────────────────────────────

def iv_from_result_dict(result_dict: dict) -> InteractionValues:
    """Reconstruct an InteractionValues object from a shard result dict.

    The dict has the same structure as what InteractionValues.to_json_file writes:
      {"metadata": {...}, "data": {"(i,j)": float, ...}, ...}

    Replicates the parsing logic of InteractionValues.from_json_file
    (shapiq_local/src/shapiq/interaction_values.py:209) without needing a file.
    """
    metadata = result_dict["metadata"]
    interaction_lookup, values = dict_to_lookup_and_values(result_dict["data"])
    return InteractionValues(
        values=values,
        index=metadata["index"],
        max_order=metadata["max_order"],
        n_players=metadata["n_players"],
        min_order=metadata["min_order"],
        interaction_lookup=interaction_lookup,
        estimated=metadata["estimated"],
        estimation_budget=metadata["estimation_budget"],
        baseline_value=metadata["baseline_value"],
    )


# ─────────────────────────────────────────────────────────────────────────────
# Metrics dict helper (reused from computation_of_approximation_metrics.py)
# ─────────────────────────────────────────────────────────────────────────────

def build_metrics_dict(all_metrics) -> dict:
    """Convert list of Metric namedtuples to a flat {key: float} dict."""
    metrics_dict = {}
    for m in all_metrics:
        key = str(getattr(m, "metric_id", repr(m)))
        if getattr(m, "computed_k", None) is not None:
            key = f"{key[:-2]}@{m.computed_k}" if "@" in key else f"{key}@{m.computed_k}"
        elif getattr(m, "order", None) is not None:
            key = f"{key}_order{m.order}"
        try:
            metrics_dict[key] = float(m.value)
        except Exception:
            metrics_dict[key] = m.value
    return metrics_dict


# ─────────────────────────────────────────────────────────────────────────────
# Underlying game loader (needed for faithfulness metric)
# ─────────────────────────────────────────────────────────────────────────────

def load_underlying_game(
    game_name: str,
    config_id: int,
    id_explain: int,
    game_type: str,
    game_class: type[Game],
    game_config_path: str,
) -> Game:
    """Create the underlying game instance for a given (game, model, id_explain).

    Replicates the logic in computation_of_approximation_metrics.py lines 136–166.
    Pre-load `configurations` from disk before calling this to avoid repeated I/O.
    """
    with open(game_config_path) as f:
        configurations = json.load(f)

    kwargs = dict(
        configuration=configurations["configurations"][int(config_id) - 1],
        game_should_be_precomputed=False,
        iteration_param=configurations["iteration_parameter"],
        iteration_param_values=[configurations["iteration_parameter_values"][id_explain]],
        iteration_param_values_names=[configurations["iteration_parameter_values_names"][id_explain]],
        game_class=game_class,
        n_players=configurations["n_players"],
        n_games=1,
    )
    try:
        return list(GameFactory.create_game_from_configs(**kwargs))[0]
    except Exception:
        kwargs["game_should_be_precomputed"] = True
        return list(GameFactory.create_game_from_configs(**kwargs))[0]


# ─────────────────────────────────────────────────────────────────────────────
# Result sources: shards and individual files
# ─────────────────────────────────────────────────────────────────────────────

def iter_results_from_shards(
    shard_dir: Path,
    game_id: str,
    config_approx: int,
    index: str,
    order: int,
):
    """Yield (id_explain, approx_name, budget, result_dict, runtime_params) from shard files.

    Handles both shard structures:
      game mode:        {game_id}_{config_approx}_{index}_{order}.shard.json
                        results = {explain: {approx_name: {budget: result}}}
      game_approx*:     {game_id}_{config_approx}_{approx_name}_{index}_{order}.shard.json
                        results = {explain: {budget: result}}
    """
    # game_approx / game_approx_explain shards
    pattern_ga = str(shard_dir / f"{game_id}_{config_approx}_*_{index}_{order}.shard.json")
    # game mode shard (no approx_name in filename)
    pattern_g = str(shard_dir / f"{game_id}_{config_approx}_{index}_{order}.shard.json")

    shard_paths = sorted(set(glob.glob(pattern_ga)) | set(glob.glob(pattern_g)))
    for shard_path in shard_paths:
        with open(shard_path) as f:
            shard = json.load(f)
        approx_name = shard["shard_meta"].get("approx_name")
        if approx_name is None:
            # game mode: results = {explain: {approx_name: {budget: result}}}
            for explain_str, approx_dict in shard["results"].items():
                id_explain = int(explain_str)
                for aname, budgets in approx_dict.items():
                    for budget_str, result_dict in budgets.items():
                        budget = int(budget_str)
                        runtime_params = result_dict.get("parameters", {})
                        yield id_explain, aname, budget, result_dict, runtime_params
        else:
            # game_approx / game_approx_explain: results = {explain: {budget: result}}
            for explain_str, budgets in shard["results"].items():
                id_explain = int(explain_str)
                for budget_str, result_dict in budgets.items():
                    budget = int(budget_str)
                    runtime_params = result_dict.get("parameters", {})
                    yield id_explain, approx_name, budget, result_dict, runtime_params


def iter_results_from_individual_files(
    approx_dir: Path,
    game_id: str,
    config_approx: int,
    index: str,
    order: int,
):
    """Yield (id_explain, approx_name, budget, iv_path, runtime_params) from individual files.

    Individual file pattern (from benchmark_local.py):
        {game_id}_{config_approx}_{id_explain}_{approx_name}_{budget}_{index}_{order}.json
    """
    pattern = str(approx_dir / f"{game_id}_{config_approx}_*_*_{index}_{order}.json")
    file_paths = sorted(glob.glob(pattern))
    for file_path in file_paths:
        fname = Path(file_path).name
        parts = fname.split("_")
        # parts: game parts... config_approx id_explain approx_name budget index order .json
        # The last 5 meaningful parts (before .json suffix): budget index order -> parts[-3], parts[-2], parts[-1][:-5]
        # approx_name = parts[-4], id_explain = parts[-5], config_approx = parts[-6]
        try:
            id_explain = int(parts[-5])
            approx_name = parts[-4]
            budget = int(parts[-3])
        except (ValueError, IndexError):
            print(f"  [WARN] Cannot parse filename: {fname} — skipping.")
            continue
        with open(file_path) as f:
            runtime_params = json.load(f).get("parameters", {})
        yield id_explain, approx_name, budget, file_path, runtime_params


# ─────────────────────────────────────────────────────────────────────────────
# Partial CSV naming
# ─────────────────────────────────────────────────────────────────────────────

def partial_csv_path(partial_dir: Path, game_id: str, index: str, order: int, game_type: str, config_approx: int) -> Path:
    return partial_dir / f"{game_id}_{index}_{order}_{game_type}_{config_approx}.metrics.csv"


# ─────────────────────────────────────────────────────────────────────────────
# Mode: info
# ─────────────────────────────────────────────────────────────────────────────

def run_info_mode(config_path: str) -> None:
    with open(config_path) as f:
        raw_config = json.load(f)
    n_games = len(raw_config)
    print(f"Config:     {config_path}")
    print(f"Games:      {n_games}")
    print()
    print(f"  #SBATCH --array=0-{n_games - 1}   ({n_games} tasks)")
    print()
    print("Per-game breakdown:")
    for game_name, info in raw_config.items():
        cfg_id = info.get("config_id", "?")
        n_approx = len(info.get("approximation_methods", []))
        n_explain = info.get("n_games", "?")
        print(f"  {game_name}_{cfg_id}: {n_approx} approx × {n_explain} x_explain")


# ─────────────────────────────────────────────────────────────────────────────
# Mode: compute
# ─────────────────────────────────────────────────────────────────────────────

def run_compute_mode(
    args: argparse.Namespace,
    raw_config: dict,
    task_id: int,
    approx_dir: Path,
    shard_dir: Path,
    truth_dir: Path,
    partial_dir: Path,
) -> None:
    game_ids = list(raw_config.keys())  # benchmark_name keys (without config_id suffix)

    if task_id >= len(game_ids):
        print(f"[SLURM] Task {task_id} out of range (total: {len(game_ids)}). Nothing to do.")
        return

    benchmark_name = game_ids[task_id]
    info = raw_config[benchmark_name]
    config_id = info["config_id"]
    game_id = f"{benchmark_name}_{config_id}"

    print(f"[SLURM] Task {task_id}: game_id={game_id}")

    # Check if partial CSV already done
    partial_csv = partial_csv_path(partial_dir, game_id, args.index, args.order, args.game_type, args.config_approximators)
    if partial_csv.exists() and not args.override:
        print(f"[SKIP] Partial CSV already exists: {partial_csv.name}  (use --override to recompute).")
        return

    # Load per-game config JSON once (for underlying game creation)
    game_config_path = f"shapiq-benchmark/configurations_{args.game_type}/{benchmark_name}.json"
    game_config_loaded = False
    try:
        with open(game_config_path) as f:
            game_configurations = json.load(f)
        game_config_loaded = True
    except Exception:
        print(f"  [WARN] Cannot load game config {game_config_path} — faithfulness metric will be skipped.")
        game_configurations = None

    game_class = GAME_NAME_TO_CLASS_MAPPING.get(benchmark_name)

    # ── Decide source: shards or individual files ──────────────────────────
    shard_pattern_ga = str(shard_dir / f"{game_id}_{args.config_approximators}_*_{args.index}_{args.order}.shard.json")
    shard_pattern_g = str(shard_dir / f"{game_id}_{args.config_approximators}_{args.index}_{args.order}.shard.json")
    has_shards = len(glob.glob(shard_pattern_ga)) > 0 or len(glob.glob(shard_pattern_g)) > 0

    if has_shards:
        print(f"  Source: shard files in {shard_dir}")
    else:
        print(f"  Source: individual files in {approx_dir}  (no shards found for {game_id})")

    results = []
    random_state = 40  # matches benchmark_slurm.py and existing metrics script
    stats = {
        "candidates_seen": 0,
        "missing_ground_truth": 0,
        "metrics_errors": 0,
    }

    # Primary location is the resolved base path. For convenience, also allow
    # local workspace ground truth as fallback if SCRATCH only contains approximations.
    truth_dirs = [truth_dir]
    local_truth_dir = Path.cwd() / "ground_truth" / args.game_type
    if local_truth_dir.resolve() != truth_dir.resolve():
        truth_dirs.append(local_truth_dir)

    # Cache ground truth per id_explain to avoid repeated loads
    gt_cache: dict[int, InteractionValues | None] = {}
    # Cache underlying games per id_explain
    game_cache: dict[int, Game | None] = {}

    def get_ground_truth(id_explain: int) -> InteractionValues | None:
        if id_explain not in gt_cache:
            gt_name = f"{game_id}_{random_state}_{id_explain}_{args.index}_{args.order}_exact_values.json"
            gt_obj = None
            for tdir in truth_dirs:
                gt_path = tdir / gt_name
                try:
                    gt_obj = InteractionValues.load(gt_path)
                    break
                except Exception:
                    continue
            if gt_obj is None:
                print(
                    f"  [WARN] Ground truth not found: {gt_name} "
                    f"(searched: {', '.join(str(p) for p in truth_dirs)})"
                )
            gt_cache[id_explain] = gt_obj
        return gt_cache[id_explain]

    def get_underlying_game(id_explain: int) -> Game | None:
        if id_explain not in game_cache:
            if not game_config_loaded or game_class is None:
                game_cache[id_explain] = None
            else:
                try:
                    kwargs = dict(
                        configuration=game_configurations["configurations"][int(config_id) - 1],
                        game_should_be_precomputed=False,
                        iteration_param=game_configurations["iteration_parameter"],
                        iteration_param_values=[game_configurations["iteration_parameter_values"][id_explain]],
                        iteration_param_values_names=[game_configurations["iteration_parameter_values_names"][id_explain]],
                        game_class=game_class,
                        n_players=game_configurations["n_players"],
                        n_games=1,
                    )
                    try:
                        game_cache[id_explain] = list(GameFactory.create_game_from_configs(**kwargs))[0]
                    except Exception:
                        kwargs["game_should_be_precomputed"] = True
                        game_cache[id_explain] = list(GameFactory.create_game_from_configs(**kwargs))[0]
                except Exception as exc:
                    print(f"  [WARN] Could not create underlying game for explain={id_explain}: {exc}")
                    game_cache[id_explain] = None
        return game_cache[id_explain]

    def process_one(id_explain: int, approx_name: str, budget: int, iv: InteractionValues, runtime_params: dict):
        ground_truth = get_ground_truth(id_explain)
        if ground_truth is None:
            stats["missing_ground_truth"] += 1
            return

        underlying_game = get_underlying_game(id_explain)

        try:
            all_metrics = get_all_metrics(ground_truth, iv, underlying_game)
        except Exception as exc:
            print(f"  [ERROR] Metrics failed: explain={id_explain} approx={approx_name} budget={budget}: {exc}")
            stats["metrics_errors"] += 1
            return

        result = {
            "game_type": args.game_type,
            "game": benchmark_name,
            "model": str(config_id),
            "game_id": game_id,
            "id_explain": id_explain,
            "n_players": ground_truth.n_players,
            "budget": budget,
            "budget_relative": round(budget / (2 ** ground_truth.n_players), 6),
            "approximator": approx_name,
            "used_budget": budget,
            "iteration": 1,
            "id_config_approximator": args.config_approximators,
        }
        result.update(build_metrics_dict(all_metrics))
        # Flatten runtime params (same logic as computation_of_approximation_metrics.py:245-249)
        for key, value in runtime_params.items():
            if isinstance(value, dict):
                result.update(value)
            else:
                result[key] = value

        results.append(result)

    # ── Iterate over results ───────────────────────────────────────────────
    if has_shards:
        for id_explain, approx_name, budget, result_dict, runtime_params in iter_results_from_shards(
            shard_dir, game_id, args.config_approximators, args.index, args.order
        ):
            stats["candidates_seen"] += 1
            try:
                iv = iv_from_result_dict(result_dict)
            except Exception as exc:
                print(f"  [ERROR] Cannot parse IV from shard: explain={id_explain} approx={approx_name} budget={budget}: {exc}")
                continue
            process_one(id_explain, approx_name, budget, iv, runtime_params)
            print(f"  ✓ explain={id_explain} approx={approx_name} budget={budget}")
    else:
        for id_explain, approx_name, budget, file_path, runtime_params in iter_results_from_individual_files(
            approx_dir, game_id, args.config_approximators, args.index, args.order
        ):
            stats["candidates_seen"] += 1
            try:
                iv = InteractionValues.load(file_path)
            except Exception as exc:
                print(f"  [ERROR] Cannot load IV from {file_path}: {exc}")
                continue
            process_one(id_explain, approx_name, budget, iv, runtime_params)
            print(f"  ✓ explain={id_explain} approx={approx_name} budget={budget}")

    if not results:
        print(
            f"  [WARN] No results collected for {game_id}. Partial CSV not written. "
            f"Candidates={stats['candidates_seen']}, "
            f"missing_ground_truth={stats['missing_ground_truth']}, "
            f"metrics_errors={stats['metrics_errors']}"
        )
        return

    df = pd.DataFrame(results)
    df.to_csv(partial_csv, index=False)
    print(f"[DONE] Partial CSV written: {partial_csv.name}  ({len(df)} rows)")


# ─────────────────────────────────────────────────────────────────────────────
# Mode: reduce
# ─────────────────────────────────────────────────────────────────────────────

def run_reduce_mode(
    partial_dir: Path,
    args: argparse.Namespace,
    base: Path,
) -> None:
    """Concatenate all partial CSVs into the final results CSV."""
    csv_files = sorted(partial_dir.glob(f"*_{args.index}_{args.order}_{args.game_type}_{args.config_approximators}.metrics.csv"))
    if not csv_files:
        print(f"[REDUCE] No partial CSVs found in {partial_dir}.")
        return

    print(f"[REDUCE] Merging {len(csv_files)} partial CSV(s)...")
    frames = [pd.read_csv(p) for p in csv_files]
    combined = pd.concat(frames, ignore_index=True)

    out_csv = Path.cwd() / f"results_benchmark_{args.index}_{args.order}_{args.game_type}.csv"
    combined.to_csv(out_csv, index=False)
    print(f"[REDUCE] Done: {len(combined)} total rows → {out_csv}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = parse_args()

    if args.mode == "info":
        run_info_mode(args.config)
        raise SystemExit(0)

    base = resolve_base_path(args)
    approx_dir = base / "approximations" / args.game_type
    shard_dir = approx_dir / "_shards"
    truth_dir = base / "ground_truth" / args.game_type
    partial_dir = base / "_metrics_partial"
    partial_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "reduce":
        run_reduce_mode(partial_dir, args, base)
        raise SystemExit(0)

    # compute mode
    with open(args.config) as f:
        raw_config = json.load(f)

    task_id = get_task_id(args)
    if task_id is None:
        print(
            "[ERROR] No SLURM_ARRAY_TASK_ID found and --task_id not set.\n"
            "  For SLURM: submit with --array=0-N\n"
            "  For local testing: pass --task_id <int>\n"
            "  For info on array size: --mode info"
        )
        raise SystemExit(1)

    run_compute_mode(args, raw_config, task_id, approx_dir, shard_dir, truth_dir, partial_dir)
