"""Precompute raw game coalition values for later use by benchmark scripts.

Instead of evaluating the real (expensive) game model during benchmarking, this
script pre-evaluates it once and caches the results.  The benchmark scripts
(benchmark_local.py, benchmark_slurm.py) will load the cache and serve values
from a lightweight dict lookup, skipping model inference entirely.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 How it works
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
For each game the script iterates over all x_explain instances and the same log-spaced
budget grid used by the benchmark scripts.  At every budget level it creates a
CoalitionSampler seeded with --random_state (default 40) — identical to what
each approximator does internally — samples the coalitions, evaluates the real
game, and stores the (coalition → value) mapping.  Because the sampler is
deterministic, the stored coalitions at budget B are exactly the ones the
approximator will request when called with that budget, making the cache safe
to use as a drop-in game replacement.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 Output
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
One shard file per game under {approx_dir}/_shards/:

    {game_id}_computed_game_values.shard.json

Shard structure:
    {
      "shard_version": 1,
      "shard_meta": { "game_id", "index", "order", "parallel_dims" },
      "results": {
        "<id_explain>": {
          "<budget>": {
            "metadata": { "n_players": int, "budget": int },
            "data": { "<bool_coalition_str>": float, ... }
          }
        }
      }
    }

Coalition keys in "data" use safe_tuple_to_str applied to the boolean
coalition row, e.g. "True,False,True,False" for a 4-player game.
Values are already normalized (game.__call__ subtracts normalization_value).

Shard files are written atomically (write to .tmp → rename).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 Parallelisation model
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
This script parallelises only over games via SLURM_ARRAY_TASK_ID:

  game
      Each task handles one game and loops internally over all x_explain
      instances and all budget steps.
      Array size = number of games

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 Typical workflow
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  # 1. Precompute game values (one task per game)
  uv run python experiments/precompute_games.py \\
      --config shapiq-benchmark/benchmarks/exhaustive/configuration_exhaustive_shapley_order2.json \\
      --game_type exhaustive --task_id 0

  # Or via SLURM (replace N with number of games minus 1)
  sbatch --array=0-N slurm/run_precompute.sh

  # 2. Run benchmark — will automatically load precomputed values
  uv run python experiments/benchmark_local.py \\
      --config shapiq-benchmark/benchmarks/exhaustive/configuration_exhaustive_shapley_order2.json \\
      --game_type exhaustive --mode approx
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from shapiq.approximator.sampling import CoalitionSampler
from shapiq_benchmark.approximators import get_approximators
from shapiq_benchmark.load import BenchmarkFactory
from shapiq_benchmark.tabpfn import TabPFNBenchmark
from shapiq_benchmark.tree import InterventionalTreeBenchmark, TreeSHAPIQBenchmark
from shapiq.utils.saving import interactions_to_dict, make_file_metadata, safe_tuple_to_str

warnings.filterwarnings("ignore", category=UserWarning)
pd.set_option("future.no_silent_downcasting", True)

SHARD_VERSION = 1


# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SLURM-optimised benchmark with shard-based output.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to JSON benchmark config file.",
    )
    parser.add_argument(
        "--game_type",
        required=True,
        help="Game type / output subdirectory (e.g. exhaustive, interventional).",
    )
    parser.add_argument(
        "--mode",
        default="approx",
        choices=["approx", "info"],
        help=(
            "Operation mode: "
            "approx (default): compute approximations and write shard files. "
            "info: print task counts for all parallelisation modes without loading games. "
        ),
    )
    parser.add_argument(
        "--config_approximators",
        type=int,
        default=37,
        help=(
            "Approximator configuration ID: "
            "37 (PAIRING=True, REPLACEMENT=False), "
            "38 (PAIRING=True, REPLACEMENT=True), "
            "39 (PAIRING=False, REPLACEMENT=False), "
            "40 (PAIRING=False, REPLACEMENT=True). Default: 37."
        ),
    )
    parser.add_argument(
        "--max_budget",
        type=int,
        default=35000,
        help="Upper bound for budget sweep. Default: 35000.",
    )
    parser.add_argument(
        "--n_budget_steps",
        type=int,
        default=20,
        help="Number of log-spaced budget points. Default: 20.",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=40,
        help="Global random seed. Default: 40.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Base output directory. Defaults to /dss/dssfs02/lwp-dss-0001/pn49je/pn49je-dss-0000/ra98xir2/neurips_tree or cwd.",
    )
    parser.add_argument(
        "--override",
        action="store_true",
        default=False,
        help="Recompute and overwrite existing shard/result files.",
    )
    parser.add_argument(
        "--task_id",
        type=int,
        default=None,
        help="Override SLURM_ARRAY_TASK_ID for local testing.",
    )
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def resolve_base_path(args: argparse.Namespace) -> Path:
    if args.output_path:
        return Path(args.output_path)
    if "SCRATCH_DSS" in os.environ:
        return Path(
            "/dss/dssfs02/lwp-dss-0001/pn49je/pn49je-dss-0000/ra98xir2/neurips_tree"
        )
    return Path.cwd()


def resolve_pairing(config_id: int) -> bool:
    pairing_map = {37: True, 38: True, 39: False, 40: False}
    if config_id not in pairing_map:
        raise ValueError(
            f"Unknown config_approximators id: {config_id}. Must be 37–40."
        )
    return pairing_map[config_id]


def get_task_id(args: argparse.Namespace) -> int | None:
    """Return the SLURM array task ID, or None if not in a SLURM environment."""
    if args.task_id is not None:
        return args.task_id
    raw = os.environ.get("SLURM_ARRAY_TASK_ID")
    return int(raw) if raw is not None else None


def create_benchmark(game_instance, game_type: str, random_state: int):
    """Create a Benchmark object for game types with specialised exact value computers.

    Returns an InterventionalTreeBenchmark, TreeSHAPIQBenchmark, or TabPFNBenchmark for the
    respective game types, or None for other game types where the raw game_instance is used directly.
    """
    if game_type == "interventional":
        rng = np.random.RandomState(random_state)
        ref_idx = rng.choice(
            game_instance.setup.x_train.shape[0], size=50, replace=False
        )
        reference_data = game_instance.setup.x_train[ref_idx].astype(np.float32)
        x_explain = game_instance.x.astype(np.float32)
        return InterventionalTreeBenchmark(
            tree_model=game_instance.setup.model,
            x_explain=x_explain,
            reference_data=reference_data,
        )
    if game_type == "pathdependent":
        x_explain = game_instance.x.astype(np.float32)
        return TreeSHAPIQBenchmark(
            tree_model=game_instance.setup.model,
            x_explain=x_explain,
            normalize=True,
        )
    if game_type == "exhaustive_tabpfn":
        return TabPFNBenchmark(
            tabpfn_model=game_instance.setup.model,
            data=game_instance.setup.x_train,
            labels=game_instance.setup.y_train,
            x_explain=game_instance.x.astype(np.float32),
        )
    return None


def build_runtime_kwargs(wall_clock: float) -> dict:
    return {
        "evaluations": wall_clock,
    }

def coal_matrix_and_values_to_dict(coalitions_matrix: np.ndarray, coalition_values: np.ndarray) -> dict[str, float]:
    """Convert coalitions matrix and values into a dictionary for saving."""
    return {
        safe_tuple_to_str(tuple(coalitions_matrix[i])): float(coalition_values[i])
        for i in range(coalitions_matrix.shape[0])
    }

def build_result_dict(coalitions_matrix, coalition_values, run_time_kwargs: dict, metadata: dict) -> dict:
    """Build the exact JSON dict that InteractionValues.to_json_file would write.

    This lets us store results inside a shard and later write them out as
    individual files during the merge step without any re-serialisation.
    """
    return {
        **make_file_metadata(
            coalitions_matrix,
            data_type="game_values",
            parameters=run_time_kwargs,
        ),
        "metadata": {
            "n_players": int(metadata.get("n_players")),
            "budget": int(metadata.get("budget")),
        },
        "data": coal_matrix_and_values_to_dict(coalitions_matrix, coalition_values),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Task list construction
# ─────────────────────────────────────────────────────────────────────────────


def build_task_list(benchmarks: dict, parallel_dims: str) -> list[tuple]:
    """Build an ordered flat list of tasks for array indexing.

    Returns a list of (game_id, approx_idx, explain_idx_or_None) tuples.
    When explain_idx is None the task covers all x_explain instances.
    """
    tasks: list[tuple] = []
    for game_id, info in benchmarks.items():
        tasks.append((game_id, None, None))
    return tasks


def build_task_list_from_raw_config(
    raw_config: dict, parallel_dims: str
) -> list[tuple]:
    """Like build_task_list but uses the raw JSON dict (no game loading needed).

    Used by --mode info so we don't pay the cost of loading games.
    explain_count comes from n_games field in the config JSON.
    """
    tasks: list[tuple] = []
    for game_id, info in raw_config.items():
        n_approx = len(info.get("approximation_methods", []))
        n_explain = info.get("n_games", 10)
        if parallel_dims == "game":
            tasks.append((game_id, None, None))
        else:
            for ai in range(n_approx):
                if parallel_dims == "game_approx":
                    tasks.append((game_id, ai, None))
                else:  # game_approx_explain
                    for ei in range(n_explain):
                        tasks.append((game_id, ai, ei))
    return tasks


# ─────────────────────────────────────────────────────────────────────────────
# Shard file helpers
# ─────────────────────────────────────────────────────────────────────────────


def shard_filename(
    game_id: str,
) -> str:
    # game mode: one shard per game, no approx or explain in filename
    return f"{game_id}_computed_game_values.shard.json"


def write_shard_atomic(path: Path, shard_data: dict) -> None:
    """Write a shard file atomically: write to .tmp then rename."""
    tmp = path.with_suffix(".tmp")
    with tmp.open("w") as f:
        json.dump(shard_data, f, indent=2)
    tmp.rename(path)


# ─────────────────────────────────────────────────────────────────────────────
# Individual filename (must match benchmark_local.py naming)
# ─────────────────────────────────────────────────────────────────────────────


def individual_filename(
    game_id: str,
    config_id: int,
    id_explain: int,
    approx_name: str,
    budget: int,
    index: str,
    order: int,
) -> str:
    return f"{game_id}_{config_id}_{id_explain}_{approx_name}_{budget}_{index}_{order}.json"


# ─────────────────────────────────────────────────────────────────────────────
# Mode: approx
# ─────────────────────────────────────────────────────────────────────────────


def run_game_precompute(
    args: argparse.Namespace,
    benchmarks: dict,
    pairing: bool,
    shard_dir: Path,
    task_id: int,
) -> None:
    tasks = build_task_list(benchmarks, "game")

    if task_id >= len(tasks):
        print(
            f"[SLURM] Task {task_id} is out of range (total tasks: {len(tasks)}). Nothing to do."
        )
        return

    game_id, approx_idx, fixed_explain_idx = tasks[task_id]
    info = benchmarks[game_id]
    all_games = list(enumerate(info["games"]))

    print(
        f"[SLURM] Task {task_id}: game={game_id}, approx_idx={approx_idx}, explain_idx={fixed_explain_idx}"
    )

    # game mode: one task covers the whole game (all approx + all explain)
    game_mode = approx_idx is None

    # Determine which x_explain instances this task covers
    if fixed_explain_idx is not None:
        explain_subset = [(fixed_explain_idx, all_games[fixed_explain_idx][1])]
    else:
        explain_subset = all_games  # all x_explain instances

    # Build approximator list using the first game instance to get n_players.
    first_game_raw = explain_subset[0][1]
    first_benchmark = create_benchmark(
        first_game_raw, args.game_type, args.random_state
    )
    first_game = first_benchmark.game if first_benchmark is not None else first_game_raw

    # Budget grid (same as benchmark_local.py)
    min_budget = first_game.n_players + 1
    max_budget = min(2**first_game.n_players, args.max_budget)
    budget_range = (
        np.ceil(
            np.logspace(np.log10(min_budget), np.log10(max_budget), args.n_budget_steps)
        )
        .clip(min_budget, max_budget)
        .astype(int)
    )

    # Shard skeleton
    # game mode results:       {str(id_explain): {approx_name: {str(budget): result_dict}}}
    # game_approx* results:    {str(id_explain): {str(budget): result_dict}}
    shard_data: dict = {
        "shard_version": SHARD_VERSION,
        "shard_meta": {
            "game_id": game_id,
            "index": info["index"],
            "order": info["order"],
            "parallel_dims": "game",
        },
        "results": {},
    }
    shard_path = shard_dir / shard_filename(
        game_id=game_id
    )
    if shard_path.exists() and not args.override:
        print(
            f"[SKIP] Shard already exists: {shard_path.name}. "
            "Use --override to recompute."
        )
        return

    for id_explain, game_raw in explain_subset:
        benchmark = create_benchmark(game_raw, args.game_type, args.random_state)
        game_to_run = benchmark.game if benchmark is not None else game_raw
        sampler = CoalitionSampler(
            n_players=game_to_run.n_players,
            sampling_weights=np.ones(game_to_run.n_players + 1),
            pairing_trick=pairing,
            random_state=args.random_state,
            replacement=False,
        )
        results_for_explain = {}
        for budget in budget_range:
            try:
                sampler.sample(budget)
                coalitions_matrix = sampler.coalitions_matrix
                t_start = time.time()
                coalition_values = game_to_run(coalitions_matrix)  # precompute values for sampled coalitions
                wall_clock = time.time() - t_start
                run_time_kwargs = build_runtime_kwargs(wall_clock)
                metadata = {
                    "n_players": game_to_run.n_players,
                    "budget": budget,
                }
                results_for_explain[str(budget)] = build_result_dict(
                    coalitions_matrix,coalition_values, run_time_kwargs, metadata
                )
                print(
                    f"  game={game_id} explain={id_explain} budget={budget}"
                    f" values computed | {wall_clock:.2f}s"
                )
            except Exception as exc:
                print(
                    f"  [ERROR] game={game_id} explain={id_explain} budget={budget}"
                    f" : {exc}"
                )
        shard_data["results"][str(id_explain)] = results_for_explain

    # Atomic write
    write_shard_atomic(shard_path, shard_data)
    if game_mode:
        n_results = sum(
            sum(len(ab) for ab in explain_dict.values())
            for explain_dict in shard_data["results"].values()
        )
    else:
        n_results = sum(len(v) for v in shard_data["results"].values())
    print(f"[DONE] Shard written: {shard_path.name}  ({n_results} results)")


# __________________________________________________________________________________
# Mode: info
# __________________________________________________________________________________

def run_info_mode(config_path: str, n_budget_steps: int) -> None:
    """Print task counts for game-level parallelisation without loading games."""
    with open(config_path) as f:
        raw_config = json.load(f)

    tasks_g = build_task_list_from_raw_config(raw_config, "game")

    total_individual = sum(
        len(info.get("approximation_methods", [])) * info.get("n_games", 10)
        for info in raw_config.values()
    ) * n_budget_steps

    print(f"Config:        {config_path}")
    print(f"Games:         {len(raw_config)}")
    print()
    print("─── game mode ──────────────────────────────────────────────────────")
    print(f"  #SBATCH --array=0-{len(tasks_g) - 1}   ({len(tasks_g)} tasks)")
    print(f"  Each task: 1 game × all approx × all x_explain × {n_budget_steps} budgets")
    print(f"  Shard files written: {len(tasks_g)}")
    print()
    print(f"Individual files without sharding: {total_individual}")
    print(f"File reduction (game):                {total_individual} → {len(tasks_g)}  ({total_individual // max(len(tasks_g), 1)}× fewer)")
    print()
    print("Per-game breakdown:")
    for game_id, info in raw_config.items():
        na = len(info.get("approximation_methods", []))
        ne = info.get("n_games", 10)
        print(f"  {game_id}: {na} approx × {ne} x_explain = {na * ne * n_budget_steps} individual files → 1 shard (game)")


def run_merge_mode(approx_dir: Path, shard_dir: Path, override: bool = False) -> None:
    """Unpack shard files into individual .json files for the metrics script.

    Individual file naming matches benchmark_local.py:
        {game_id}_{config_id}_{id_explain}_{approx_name}_{budget}_{index}_{order}.json
    """
    shard_files = sorted(shard_dir.glob("*.shard.json"))
    if not shard_files:
        print(f"[MERGE] No shard files found in {shard_dir}.")
        return

    print(f"[MERGE] Found {len(shard_files)} shard file(s) in {shard_dir}.")
    total_written = 0
    total_skipped = 0

    for shard_path in shard_files:
        with shard_path.open() as f:
            shard = json.load(f)

        meta = shard["shard_meta"]
        game_id = meta["game_id"]
        approx_name = meta.get("approx_name")  # None for game mode
        config_id = meta["config_approximators"]
        index = meta["index"]
        order = meta["order"]

        if approx_name is None:
            # game mode: results = {explain: {approx_name: {budget: result}}}
            for explain_str, approx_dict in shard["results"].items():
                id_explain = int(explain_str)
                for aname, budgets in approx_dict.items():
                    for budget_str, result_dict in budgets.items():
                        budget = int(budget_str)
                        fname = individual_filename(
                            game_id, config_id, id_explain, aname, budget, index, order
                        )
                        fpath = approx_dir / fname
                        if fpath.exists() and not override:
                            total_skipped += 1
                            continue
                        with fpath.open("w") as f:
                            json.dump(result_dict, f, indent=2)
                        total_written += 1
        else:
            # game_approx / game_approx_explain: results = {explain: {budget: result}}
            for explain_str, budgets in shard["results"].items():
                id_explain = int(explain_str)
                for budget_str, result_dict in budgets.items():
                    budget = int(budget_str)
                    fname = individual_filename(
                        game_id,
                        config_id,
                        id_explain,
                        approx_name,
                        budget,
                        index,
                        order,
                    )
                    fpath = approx_dir / fname
                    if fpath.exists() and not override:
                        total_skipped += 1
                        continue
                    with fpath.open("w") as f:
                        json.dump(result_dict, f, indent=2)
                    total_written += 1

    print(f"[MERGE] Done: {total_written} files written, {total_skipped} skipped.")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = parse_args()

    # --- directory setup ---
    base = resolve_base_path(args)
    approx_dir = base / "approximations" / args.game_type
    truth_dir = base / "ground_truth" / args.game_type
    shard_dir = approx_dir / "_shards"
    approx_dir.mkdir(parents=True, exist_ok=True)
    truth_dir.mkdir(parents=True, exist_ok=True)
    shard_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "info":
        run_info_mode(args.config, args.n_budget_steps)
        raise SystemExit(0)
    # --- load benchmarks (approx / true modes) ---
    benchmarks = BenchmarkFactory.load_benchmarks_from_json(config_path=args.config)

    task_id = get_task_id(args)
    if task_id is None:
        print(
            "[ERROR] No SLURM_ARRAY_TASK_ID found and --task_id not set.\n"
            "  For SLURM: submit with --array=0-N\n"
            "  For local testing: pass --task_id <int>\n"
            "  For a sequential local run: use experiments/benchmark_local.py instead."
        )
        raise SystemExit(1)

    pairing = resolve_pairing(args.config_approximators)
    run_game_precompute(args, benchmarks, pairing, shard_dir, task_id)
