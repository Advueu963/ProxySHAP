"""SLURM-optimised benchmark script for Shapley interaction approximation.

Designed to run efficiently as SLURM array jobs.  Unlike the local script,
results are collected into *shard files* (one per array task) rather than one
file per (game, explain, approximator, budget), drastically reducing the number
of files written — critical on partitions with strict inode / file-count limits.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 Parallelisation model
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Three dimensions can be parallelised via SLURM_ARRAY_TASK_ID:

  game
      Each task handles one game and loops internally over all approximators
      and all x_explain instances.
      Array size = number of games

  game_approx (default)
      Each task handles one (game, approximator) pair.
      Loops internally over all x_explain instances and all budget steps.
      Array size = Σ_game  n_approx(game)

  game_approx_explain
      Each task handles one (game, approximator, x_explain) triple.
      Loops internally over all budget steps.
      Array size = Σ_game  n_approx(game) × n_explain(game)

Budget steps are always sequential within a task — they are cheap and keeping
them together eliminates the need to synchronise partial results.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 Sharding
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Each task writes exactly ONE shard file under {approx_dir}/_shards/:

  game mode:
        {game_id}_{cfg}_{index}_{order}.shard.json
        contains all (x_explain, approx, budget) results for this game.

  game_approx mode:
      {game_id}_{cfg}_{approx}_{index}_{order}.shard.json
      contains all (x_explain, budget) results for this task.

  game_approx_explain mode:
      {game_id}_{cfg}_{eid}_{approx}_{index}_{order}.shard.json
      contains all budget results for this (game, approx, explain) task.

Shard files are written atomically (write to .tmp → rename).  If a shard
already exists and --override is not set, the task exits immediately.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 Modes
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  approx  [default]
      Run approximations and write shard files.  Reads SLURM_ARRAY_TASK_ID
      from the environment (or --task_id for local testing).

  merge
      Convert shard files → individual .json files that are compatible with
      computation_of_approximation_metrics.py.  Run this locally after all
      SLURM jobs finish.

  info
      Print array sizes for both parallelisation modes without loading any
      game data.  Use the output to set #SBATCH --array.

  true
      Compute exact ground truth values (no sharding — very few files).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 Typical workflow
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  # 1. Find out how many tasks you need
  uv run python experiments/benchmark_slurm.py \\
      --config shapiq-benchmark/benchmarks/exhaustive/configuration_exhaustive_shapley_order2.json \\
      --game_type exhaustive --mode info

  # 2. Submit (replace 48 with value from step 1 minus 1)
  sbatch --array=0-48 slurm/run_benchmark.sh

  # 3. After all tasks finish, merge shards → individual files
  uv run python experiments/benchmark_slurm.py \\
      --config shapiq-benchmark/benchmarks/exhaustive/configuration_exhaustive_shapley_order2.json \\
      --game_type exhaustive --mode merge
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
from shapiq import Game
from shapiq.utils.saving import interactions_to_dict, make_file_metadata, safe_tuple_to_str
from shapiq_benchmark.approximators import get_approximators
from shapiq_benchmark.load import BenchmarkFactory
from shapiq_benchmark.tabpfn import TabPFNBenchmark
from shapiq_benchmark.tree import InterventionalTreeBenchmark, TreeSHAPIQBenchmark
warnings.filterwarnings("ignore", category=UserWarning)


# ─────────────────────────────────────────────────────────────────────────────
# Precomputed game cache
# ─────────────────────────────────────────────────────────────────────────────


class PrecomputedGame(Game):
    """Lightweight lookup game backed by pre-evaluated coalition values.

    Values are stored already normalized (as returned by Game.__call__), so
    normalization_value is kept at 0.0 to avoid double-subtraction.
    """

    def __init__(self, n_players: int, lookup: dict[str, float]) -> None:
        super().__init__(n_players=n_players, normalize=False)
        self._lookup = lookup

    def value_function(self, coalitions: np.ndarray) -> np.ndarray:
        return np.array([self._lookup[safe_tuple_to_str(tuple(row))] for row in coalitions])


def try_load_precomputed_game(
    shard_dir: Path,
    game_id: str,
    id_explain: int,
    budget: int,
    n_players: int,
) -> PrecomputedGame | None:
    """Return a PrecomputedGame if a precomputed shard exists for (game_id, id_explain, budget).

    Returns None if the shard file is missing or does not cover this (explain, budget) pair,
    in which case the caller should fall back to the real game.
    """
    shard_path = shard_dir / f"{game_id}_computed_game_values.shard.json"
    if not shard_path.exists():
        return None
    with shard_path.open() as f:
        shard = json.load(f)
    budget_data = shard.get("results", {}).get(str(id_explain), {}).get(str(budget))
    if budget_data is None:
        return None
    return PrecomputedGame(n_players=n_players, lookup=budget_data["data"])
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
        "--config", required=True,
        help="Path to JSON benchmark config file.",
    )
    parser.add_argument(
        "--game_type", required=True,
        help="Game type / output subdirectory (e.g. exhaustive, interventional).",
    )
    parser.add_argument(
        "--mode", default="approx",
        choices=["approx", "merge", "info", "true"],
        help=(
            "approx: run approximations (SLURM). "
            "merge: unpack shards → individual files. "
            "info: print array sizes. "
            "true: compute ground truth. Default: approx."
        ),
    )
    parser.add_argument(
        "--parallel_dims", default="game_approx",
        choices=["game", "game_approx", "game_approx_explain"],
        help=(
            "Dimensions to parallelise. "
            "game: one task per game (loops all approx + x_explain internally). "
            "game_approx: one task per (game, approx). "
            "game_approx_explain: one task per (game, approx, x_explain). "
            "Default: game_approx."
        ),
    )
    parser.add_argument(
        "--config_approximators", type=int, default=37,
        help=(
            "Approximator configuration ID: "
            "37 (PAIRING=True, REPLACEMENT=False), "
            "38 (PAIRING=True, REPLACEMENT=True), "
            "39 (PAIRING=False, REPLACEMENT=False), "
            "40 (PAIRING=False, REPLACEMENT=True). Default: 37."
        ),
    )
    parser.add_argument(
        "--max_budget", type=int, default=35000,
        help="Upper bound for budget sweep. Default: 35000.",
    )
    parser.add_argument(
        "--n_budget_steps", type=int, default=20,
        help="Number of log-spaced budget points. Default: 20.",
    )
    parser.add_argument(
        "--random_state", type=int, default=40,
        help="Global random seed. Default: 40.",
    )
    parser.add_argument(
        "--n_estimators", type=int, default=None,
        help="Override n_estimators for tree-based approximators. Default: None.",
    )
    parser.add_argument(
        "--output_path", type=str, default=None,
        help="Base output directory. Defaults to /dss/dssfs02/lwp-dss-0001/pn49je/pn49je-dss-0000/ra98xir2/neurips_tree or cwd.",
    )
    parser.add_argument(
        "--override", action="store_true", default=False,
        help="Recompute and overwrite existing shard/result files.",
    )
    parser.add_argument(
        "--task_id", type=int, default=None,
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
        scratch_path = Path(os.environ["SCRATCH_DSS"])
        return scratch_path / "neurips_tree"
    return Path.cwd()


def resolve_pairing(config_id: int) -> bool:
    pairing_map = {37: True, 38: True, 39: False, 40: False}
    if config_id not in pairing_map:
        raise ValueError(f"Unknown config_approximators id: {config_id}. Must be 37–40.")
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
        ref_idx = rng.choice(game_instance.setup.x_train.shape[0], size=50, replace=False)
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


def build_runtime_kwargs(wall_clock: float, approximator) -> dict:
    detailed = approximator.runtime_last_approximate_run
    return {
        "total_runtime": wall_clock,
        **{"total_approximation" if k == "total" else k: v for k, v in detailed.items()},
    }


def build_result_dict(shap_approx, run_time_kwargs: dict) -> dict:
    """Build the exact JSON dict that InteractionValues.to_json_file would write.

    This lets us store results inside a shard and later write them out as
    individual files during the merge step without any re-serialisation.
    """
    return {
        **make_file_metadata(
            shap_approx,
            data_type="interaction_values",
            parameters=run_time_kwargs,
        ),
        "metadata": {
            "n_players": shap_approx.n_players,
            "index": shap_approx.index,
            "max_order": shap_approx.max_order,
            "min_order": shap_approx.min_order,
            "estimated": shap_approx.estimated,
            "estimation_budget": shap_approx.estimation_budget,
            "baseline_value": shap_approx.baseline_value,
        },
        "data": interactions_to_dict(interactions=shap_approx.dict_values),
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
        n_approx = len(info["approximation_methods"])
        n_explain = info["n_games"]
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


def build_task_list_from_raw_config(raw_config: dict, parallel_dims: str) -> list[tuple]:
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
    config_id: int,
    approx_name: str | None,
    index: str,
    order: int,
    explain_idx: int | None = None,
) -> str:
    if approx_name is None:
        # game mode: one shard per game, no approx or explain in filename
        return f"{game_id}_{config_id}_{index}_{order}.shard.json"
    if explain_idx is not None:
        return f"{game_id}_{config_id}_{explain_idx}_{approx_name}_{index}_{order}.shard.json"
    return f"{game_id}_{config_id}_{approx_name}_{index}_{order}.shard.json"


def write_shard_atomic(path: Path, shard_data: dict) -> None:
    """Write a shard file atomically: write to .tmp then rename."""
    tmp = path.with_suffix(".tmp")
    with tmp.open("w") as f:
        json.dump(shard_data, f, indent=2)
    tmp.rename(path)


def load_partial_shard(path: Path) -> dict | None:
    """Load a partial .tmp shard if it exists, return None otherwise."""
    tmp = path.with_suffix(".tmp")
    if not tmp.exists():
        return None
    try:
        with tmp.open() as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        print(f"[RESUME] Warning: .tmp shard {tmp.name} corrupted, will recompute.")
        return None


def get_completed_budgets_by_explain(partial_shard: dict, is_game_mode: bool) -> dict[int, set[int]] | dict[tuple[int, str], set[int]]:
    """Extract completed budgets from a partial shard.
    
    For game_mode:
        Returns: {(id_explain, approx_name): {budgets}}
    For game_approx modes:
        Returns: {id_explain: {budgets}}
    """
    completed: dict = {}
    for explain_str, explain_data in partial_shard.get("results", {}).items():
        id_explain = int(explain_str)
        if is_game_mode:
            # game mode: {approx_name: {budget: result}}
            for approx_name, budget_dict in explain_data.items():
                key = (id_explain, approx_name)
                completed[key] = set(int(b) for b in budget_dict.keys())
        else:
            # game_approx / game_approx_explain: {budget: result}
            completed[id_explain] = set(int(b) for b in explain_data.keys())
    return completed


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

def run_approx_mode(
    args: argparse.Namespace,
    benchmarks: dict,
    pairing: bool,
    shard_dir: Path,
    task_id: int,
) -> None:
    tasks = build_task_list(benchmarks, args.parallel_dims)

    if task_id >= len(tasks):
        print(f"[SLURM] Task {task_id} is out of range (total tasks: {len(tasks)}). Nothing to do.")
        return

    game_id, approx_idx, fixed_explain_idx = tasks[task_id]
    info = benchmarks[game_id]
    all_games = list(enumerate(info["games"]))

    print(f"[SLURM] Task {task_id}: game={game_id}, approx_idx={approx_idx}, explain_idx={fixed_explain_idx}")

    # game mode: one task covers the whole game (all approx + all explain)
    game_mode = approx_idx is None

    # Determine which x_explain instances this task covers
    if fixed_explain_idx is not None:
        explain_subset = [(fixed_explain_idx, all_games[fixed_explain_idx][1])]
    else:
        explain_subset = all_games  # all x_explain instances

    # Build approximator list using the first game instance to get n_players.
    first_game_raw = explain_subset[0][1]
    first_benchmark = create_benchmark(first_game_raw, args.game_type, args.random_state)
    first_game = first_benchmark.game if first_benchmark is not None else first_game_raw

    full_approx_list = get_approximators(
        info["approximation_methods"],
        first_game.n_players,
        args.random_state,
        pairing,
        info["index"],
        info["order"],
        n_estimators=args.n_estimators,
    )

    # For game_approx / game_approx_explain: single target approximator
    if not game_mode:
        target_approx = full_approx_list[approx_idx]
        # SVARMIQ is too slow for large games
        if target_approx.name == "SVARMIQ" and first_game.n_players > 20:
            print(f"[SKIP] SVARMIQ on {game_id}: n_players={first_game.n_players} > 20.")
            return
        shard_approx_name = target_approx.name
    else:
        shard_approx_name = None  # game mode: all approx stored in shard

    # Check shard already done
    shard_name = shard_filename(
        game_id, args.config_approximators, shard_approx_name,
        info["index"], info["order"], fixed_explain_idx,
    )
    shard_path = shard_dir / shard_name
    if shard_path.exists() and not args.override:
        print(f"[SKIP] Shard already exists: {shard_path.name}  (use --override to recompute).")
        return

    # Try to load partial .tmp shard for resuming
    partial_shard = load_partial_shard(shard_path)
    if partial_shard is not None:
        print(f"[RESUME] Found partial .tmp shard, will skip completed budgets.")
        completed_budgets_by_explain = get_completed_budgets_by_explain(partial_shard, game_mode)
    else:
        completed_budgets_by_explain = {}

    # Budget grid (same as benchmark_local.py)
    min_budget = first_game.n_players + 1
    max_budget = min(2 ** first_game.n_players, args.max_budget)
    budget_range = (
        np.ceil(np.logspace(np.log10(min_budget), np.log10(max_budget), args.n_budget_steps))
        .clip(min_budget, max_budget)
        .astype(int)
    )

    # Shard skeleton
    # game mode results:       {str(id_explain): {approx_name: {str(budget): result_dict}}}
    # game_approx* results:    {str(id_explain): {str(budget): result_dict}}
    if partial_shard is not None:
        shard_data = partial_shard
    else:
        shard_data: dict = {
            "shard_version": SHARD_VERSION,
            "shard_meta": {
                "game_id": game_id,
                "approx_name": shard_approx_name,
                "config_approximators": args.config_approximators,
                "index": info["index"],
                "order": info["order"],
                "parallel_dims": args.parallel_dims,
            },
            "results": {},
        }

    for id_explain, game_raw in explain_subset:
        benchmark = create_benchmark(game_raw, args.game_type, args.random_state)
        game_to_run = benchmark.game if benchmark is not None else game_raw

        # Rebuild approximator list for each explain instance (reset internal state)
        approx_list = get_approximators(
            info["approximation_methods"],
            game_to_run.n_players,
            args.random_state,
            pairing,
            info["index"],
            info["order"],
            n_estimators=args.n_estimators,
        )

        if game_mode:
            # Loop over all approximators; store results nested under approx name
            results_for_explain: dict = {}
            for approx in approx_list:
                if approx.name == "SVARMIQ" and game_to_run.n_players > 20:
                    print(f"[SKIP] SVARMIQ on {game_id} explain={id_explain}: n_players > 20.")
                    continue
                approx_results: dict = {}
                for budget in budget_range:
                    # Skip if already completed in partial shard
                    completion_key = (id_explain, approx.name)
                    if completion_key in completed_budgets_by_explain and budget in completed_budgets_by_explain[completion_key]:
                        existing = shard_data["results"].get(str(id_explain), {}).get(approx.name, {}).get(str(budget))
                        if existing is not None:
                            approx_results[str(budget)] = existing
                            print(f"  [RESUME] game={game_id} explain={id_explain} budget={budget} approx={approx.name} (skipped, already done)")
                            continue
                    try:
                        game_for_budget = try_load_precomputed_game(shard_dir, game_id, id_explain, int(budget), game_to_run.n_players)
                        if game_for_budget is None:
                            game_for_budget = game_to_run
                        t_start = time.time()
                        shap_approx = approx.approximate(
                            budget=int(budget),
                            game=game_for_budget,
                            game_id=game_id,
                            id_explain=id_explain,
                        )
                        wall_clock = time.time() - t_start
                        run_time_kwargs = build_runtime_kwargs(wall_clock, approx)
                        approx_results[str(budget)] = build_result_dict(shap_approx, run_time_kwargs)
                        print(
                            f"  game={game_id} explain={id_explain} budget={budget}"
                            f" approx={approx.name} | {wall_clock:.2f}s"
                        )
                    except Exception as exc:
                        print(
                            f"  [ERROR] game={game_id} explain={id_explain} budget={budget}"
                            f" approx={approx.name}: {exc}"
                        )
                results_for_explain[approx.name] = approx_results
            shard_data["results"][str(id_explain)] = results_for_explain
        else:
            approx = approx_list[approx_idx]
            results_for_explain = {}
            for budget in budget_range:
                # Skip if already completed in partial shard
                if id_explain in completed_budgets_by_explain and budget in completed_budgets_by_explain[id_explain]:
                    existing = shard_data["results"].get(str(id_explain), {}).get(str(budget))
                    if existing is not None:
                        results_for_explain[str(budget)] = existing
                        print(f"  [RESUME] game={game_id} explain={id_explain} budget={budget} approx={approx.name} (skipped, already done)")
                        continue
                try:
                    game_for_budget = try_load_precomputed_game(shard_dir, game_id, id_explain, int(budget), game_to_run.n_players)
                    if game_for_budget is None:
                        game_for_budget = game_to_run
                    t_start = time.time()
                    shap_approx = approx.approximate(
                        budget=int(budget),
                        game=game_for_budget,
                        game_id=game_id,
                        id_explain=id_explain,
                    )
                    wall_clock = time.time() - t_start
                    run_time_kwargs = build_runtime_kwargs(wall_clock, approx)
                    results_for_explain[str(budget)] = build_result_dict(shap_approx, run_time_kwargs)
                    print(
                        f"  game={game_id} explain={id_explain} budget={budget}"
                        f" approx={approx.name} | {wall_clock:.2f}s"
                    )
                except Exception as exc:
                    print(
                        f"  [ERROR] game={game_id} explain={id_explain} budget={budget}"
                        f" approx={approx.name}: {exc}"
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


# ─────────────────────────────────────────────────────────────────────────────
# Mode: merge
# ─────────────────────────────────────────────────────────────────────────────

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
                        fname = individual_filename(game_id, config_id, id_explain, aname, budget, index, order)
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
                    fname = individual_filename(game_id, config_id, id_explain, approx_name, budget, index, order)
                    fpath = approx_dir / fname
                    if fpath.exists() and not override:
                        total_skipped += 1
                        continue
                    with fpath.open("w") as f:
                        json.dump(result_dict, f, indent=2)
                    total_written += 1

    print(f"[MERGE] Done: {total_written} files written, {total_skipped} skipped.")


# ─────────────────────────────────────────────────────────────────────────────
# Mode: info
# ─────────────────────────────────────────────────────────────────────────────

def run_info_mode(config_path: str, n_budget_steps: int) -> None:
    """Print task counts for all parallelisation modes without loading games."""
    with open(config_path) as f:
        raw_config = json.load(f)

    tasks_g = build_task_list_from_raw_config(raw_config, "game")
    tasks_ga = build_task_list_from_raw_config(raw_config, "game_approx")
    tasks_gae = build_task_list_from_raw_config(raw_config, "game_approx_explain")

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
    print("─── game_approx mode ───────────────────────────────────────────────")
    print(f"  #SBATCH --array=0-{len(tasks_ga) - 1}   ({len(tasks_ga)} tasks)")
    print(f"  Each task: 1 game × 1 approx × all x_explain × {n_budget_steps} budgets")
    print(f"  Shard files written: {len(tasks_ga)}")
    print()
    print("─── game_approx_explain mode ───────────────────────────────────────")
    print(f"  #SBATCH --array=0-{len(tasks_gae) - 1}   ({len(tasks_gae)} tasks)")
    print(f"  Each task: 1 game × 1 approx × 1 x_explain × {n_budget_steps} budgets")
    print(f"  Shard files written: {len(tasks_gae)}")
    print()
    print(f"Individual files without sharding: {total_individual}")
    print(f"File reduction (game):                {total_individual} → {len(tasks_g)}  ({total_individual // max(len(tasks_g), 1)}× fewer)")
    print(f"File reduction (game_approx):         {total_individual} → {len(tasks_ga)}  ({total_individual // max(len(tasks_ga), 1)}× fewer)")
    print(f"File reduction (game_approx_explain): {total_individual} → {len(tasks_gae)}  ({total_individual // max(len(tasks_gae), 1)}× fewer)")
    print()
    print("Per-game breakdown:")
    for game_id, info in raw_config.items():
        na = len(info.get("approximation_methods", []))
        ne = info.get("n_games", 10)
        print(f"  {game_id}: {na} approx × {ne} x_explain = {na * ne * n_budget_steps} individual files → 1 shard (game) / {na} shards (game_approx)")


# ─────────────────────────────────────────────────────────────────────────────
# Mode: true (ground truth)
# ─────────────────────────────────────────────────────────────────────────────

def run_true_mode(args: argparse.Namespace, benchmarks: dict, truth_dir: Path) -> None:
    """Compute exact ground truth values (no sharding — very few output files)."""
    for game_id, info in benchmarks.items():
        for id_explain, game_instance in enumerate(info["games"]):
            benchmark = create_benchmark(game_instance, args.game_type, args.random_state)
            game_to_run = benchmark.game if benchmark is not None else game_instance
            save_path = truth_dir / (
                f"{game_id}_{args.random_state}_{id_explain}"
                f"_{info['index']}_{info['order']}_exact_values.json"
            )
            if save_path.exists() and not args.override:
                print(f"[SKIP] {save_path.name} already exists.")
                continue
            if args.game_type in ("exhaustive", "exhaustive_tabpfn") and game_to_run.n_players > 20:
                print(f"[SKIP] {game_id} id={id_explain}: n_players={game_to_run.n_players} > 20 (too large for exact).")
                continue
            if benchmark is not None:
                gt = benchmark.exact_values(index=info["index"], order=info["order"])
            else:
                gt = game_to_run.exact_values(index=info["index"], order=info["order"])
            gt.save(save_path)
            print(f"[TRUE] Saved: {save_path.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = parse_args()

    # --- info mode: no game loading needed ---
    if args.mode == "info":
        run_info_mode(args.config, args.n_budget_steps)
        raise SystemExit(0)

    # --- directory setup ---
    base = resolve_base_path(args)
    approx_dir = base / "approximations" / args.game_type
    truth_dir = base / "ground_truth" / args.game_type
    shard_dir = approx_dir / "_shards"
    approx_dir.mkdir(parents=True, exist_ok=True)
    truth_dir.mkdir(parents=True, exist_ok=True)
    shard_dir.mkdir(parents=True, exist_ok=True)

    # --- merge mode: no game loading needed ---
    if args.mode == "merge":
        run_merge_mode(approx_dir, shard_dir, override=args.override)
        raise SystemExit(0)

    # --- load benchmarks (approx / true modes) ---
    benchmarks = BenchmarkFactory.load_benchmarks_from_json(config_path=args.config)

    if args.mode == "true":
        run_true_mode(args, benchmarks, truth_dir)

    elif args.mode == "approx":
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
        run_approx_mode(args, benchmarks, pairing, shard_dir, task_id)