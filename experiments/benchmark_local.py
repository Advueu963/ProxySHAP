"""Local benchmark script — simple sequential approximation runner.

Runs all (game, x_explain, approximator, budget) combinations sequentially
and writes one .json file per result.  No SLURM logic, no sharding.

Usage examples:
    # Approximation mode
    uv run python experiments/benchmark_local.py \\
        --config shapiq-benchmark/benchmarks/exhaustive/configuration_exhaustive_shapley_order2.json \\
        --game_type exhaustive \\
        --mode approx \\
        --config_approximators 37 \\
        --max_budget 35000 \\
        --n_budget_steps 20

    # Ground truth mode
    uv run python experiments/benchmark_local.py \\
        --config shapiq-benchmark/benchmarks/exhaustive/configuration_exhaustive_shapley_order2.json \\
        --game_type exhaustive \\
        --mode true
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
from shapiq.utils.saving import safe_tuple_to_str
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Local sequential benchmark (no SLURM, no sharding)."
    )
    parser.add_argument("--config", required=True, help="Path to JSON benchmark config file.")
    parser.add_argument(
        "--game_type", required=True,
        help="Game type / output subdirectory (e.g. exhaustive, interventional).",
    )
    parser.add_argument(
        "--mode", default="approx", choices=["approx", "true"],
        help="approx: run approximators. true: compute exact ground truth. Default: approx.",
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
    parser.add_argument("--max_budget", type=int, default=35000, help="Max budget. Default: 35000.")
    parser.add_argument("--n_budget_steps", type=int, default=20, help="Budget grid size. Default: 20.")
    parser.add_argument("--random_state", type=int, default=40, help="Random seed. Default: 40.")
    parser.add_argument(
        "--n_estimators", type=int, default=None,
        help="Override n_estimators for tree-based approximators. Default: None.",
    )
    parser.add_argument(
        "--output_path", type=str, default=None,
        help="Base output directory. Defaults to $SCRATCH_DSS/msr_int_iq or cwd.",
    )
    parser.add_argument(
        "--override", action="store_true", default=False,
        help="Recompute and overwrite existing result files.",
    )
    parser.add_argument(
        "--parallel_dims", default=None,
        choices=["game", "game_approx", "game_approx_explain"],
        help=(
            "Run only one task instead of all. Requires --task_id. "
            "game: one task per game. "
            "game_approx: one task per (game, approx). "
            "game_approx_explain: one task per (game, approx, x_explain). "
            "Default: None (run all sequentially)."
        ),
    )
    parser.add_argument(
        "--task_id", type=int, default=None,
        help="Index of the task to run when --parallel_dims is set.",
    )
    return parser.parse_args()


def resolve_base_path(args: argparse.Namespace) -> Path:
    if args.output_path:
        return Path(args.output_path)
    if "SCRATCH_DSS" in os.environ:
        return Path(os.environ["SCRATCH_DSS"]) / "msr_int_iq"
    return Path.cwd()


def resolve_pairing(config_id: int) -> bool:
    pairing_map = {37: True, 38: True, 39: False, 40: False}
    if config_id not in pairing_map:
        raise ValueError(f"Unknown config_approximators id: {config_id}. Must be 37–40.")
    return pairing_map[config_id]


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


def build_task_list(benchmarks: dict, parallel_dims: str) -> list[tuple]:
    """Build an ordered flat list of (game_id, approx_idx_or_None, explain_idx_or_None) tasks."""
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


def build_runtime_kwargs(wall_clock: float, approximator) -> dict:
    detailed = approximator.runtime_last_approximate_run
    return {
        "total_runtime": wall_clock,
        **{"total_approximation" if k == "total" else k: v for k, v in detailed.items()},
    }


def _run_single_explain(
    game_id: str,
    id_explain: int,
    game_instance,
    approximator_list: list,
    approx_idx: int | None,
    budget_range,
    args: argparse.Namespace,
    approx_dir: Path,
    info: dict,
) -> None:
    """Run one (game, explain) slice for the given approximators."""
    if approx_idx is not None:
        approximator_list = [approximator_list[approx_idx]]

    for approximator in approximator_list:
        if approximator.name == "SVARMIQ" and game_instance.n_players > 20:
            print(f"[SKIP] SVARMIQ on {game_id} id={id_explain}: n_players > 20.")
            continue

        print(f"Running {approximator.name} on {game_id} (explain={id_explain})")

        print(f"  Budget range: {budget_range[0]} to {budget_range[-1]} ({len(budget_range)} steps)")
        for budget in budget_range:
            save_path = approx_dir / (
                f"{game_id}_{args.config_approximators}_{id_explain}"
                f"_{approximator.name}_{budget}_{info['index']}_{info['order']}.json"
            )
            if save_path.exists() and not args.override:
                print(f"  [SKIP] {save_path.name}")
                continue
            shard_dir = approx_dir / "_shards"
            if approximator.name == "PermutationSamplingSII":
                # PermutationSamplingSII does not draw coalitions via the shared
                # CoalitionSampler (it enumerates permutation-based coalitions),
                # so the precomputed shard does not cover its queries.
                game = game_instance
            else:
                game = try_load_precomputed_game(shard_dir, game_id, id_explain, int(budget), game_instance.n_players)
                if game is None:
                    print(f"  [SHARD MISS] No precomputed game values for budget={budget}. Running on the original game.")
                    game = game_instance
                else:
                    print(f"  [SHARD HIT] Loaded precomputed game values for budget={budget} from {shard_dir / f'{game_id}_computed_game_values.shard.json'}")
            try:
                t_start = time.time()
                shap_approx = approximator.approximate(
                    budget=int(budget),
                    game=game,
                    game_id=game_id,
                    id_explain=id_explain,
                )
                wall_clock = time.time() - t_start
                run_time_kwargs = build_runtime_kwargs(wall_clock, approximator)
                shap_approx.save(save_path, **run_time_kwargs)
                print(f"  Saved {save_path.name} | {wall_clock:.2f}s")
            except Exception as exc:
                print(f"  [ERROR] budget={budget}: {exc}")


def run_approx_mode(args: argparse.Namespace, benchmarks: dict, pairing: bool, approx_dir: Path) -> None:
    # Task-based execution: run only one task if --parallel_dims and --task_id are set
    if args.parallel_dims is not None and args.task_id is not None:
        tasks = build_task_list(benchmarks, args.parallel_dims)
        if args.task_id >= len(tasks):
            print(f"[ERROR] task_id={args.task_id} is out of range (total tasks: {len(tasks)}).")
            return
        game_id, approx_idx, fixed_explain_idx = tasks[args.task_id]
        print(f"[LOCAL] Task {args.task_id}: game={game_id}, approx_idx={approx_idx}, explain_idx={fixed_explain_idx}")
        info = benchmarks[game_id]
        all_games = list(enumerate(info["games"]))
        explain_subset = [(fixed_explain_idx, all_games[fixed_explain_idx][1])] if fixed_explain_idx is not None else all_games

        for id_explain, game_instance_raw in explain_subset:
            benchmark = create_benchmark(game_instance_raw, args.game_type, args.random_state)
            game_to_run = benchmark.game if benchmark is not None else game_instance_raw
            approximator_list = get_approximators(
                info["approximation_methods"],
                game_to_run.n_players,
                args.random_state,
                pairing,
                info["index"],
                info["order"],
                n_estimators=args.n_estimators,
            )
            min_budget = game_to_run.n_players + 1
            max_budget = min(2 ** game_to_run.n_players, args.max_budget)
            budget_range = (
                np.ceil(np.logspace(np.log10(min_budget), np.log10(max_budget), args.n_budget_steps))
                .clip(min_budget, max_budget)
                .astype(int)
            )
            _run_single_explain(game_id, id_explain, game_to_run, approximator_list, approx_idx, budget_range, args, approx_dir, info)
        return

    # Default: sequential over all games, explains, and approximators
    for game_id, info in benchmarks.items():
        for id_explain, game_instance in enumerate(info["games"]):
            benchmark = create_benchmark(game_instance, args.game_type, args.random_state)
            game_to_run = benchmark.game if benchmark is not None else game_instance

            approximator_list = get_approximators(
                info["approximation_methods"],
                game_to_run.n_players,
                args.random_state,
                pairing,
                info["index"],
                info["order"],
                n_estimators=args.n_estimators,
            )

            min_budget = game_to_run.n_players + 1
            max_budget = min(2 ** game_to_run.n_players, args.max_budget)
            budget_range = (
                np.ceil(
                    np.logspace(np.log10(min_budget), np.log10(max_budget), args.n_budget_steps)
                )
                .clip(min_budget, max_budget)
                .astype(int)
            )

            _run_single_explain(game_id, id_explain, game_to_run, approximator_list, None, budget_range, args, approx_dir, info)


def run_true_mode(args: argparse.Namespace, benchmarks: dict, truth_dir: Path) -> None:
    for game_id, info in benchmarks.items():
        for id_explain, game_instance in enumerate(info["games"]):
            benchmark = create_benchmark(game_instance, args.game_type, args.random_state)
            game_to_run = benchmark.game if benchmark is not None else game_instance
            save_path = truth_dir / (
                f"{game_id}_{args.random_state}_{id_explain}"
                f"_{info['index']}_{info['order']}_exact_values.json"
            )
            if save_path.exists() and not args.override:
                print(f"[SKIP] {save_path.name}")
                continue
            has_cheap_exact = game_to_run.__class__.__name__ == "SOUM"
            if (
                args.game_type in ("exhaustive", "exhaustive_tabpfn")
                and game_to_run.n_players > 20
                and not has_cheap_exact
            ):
                print(f"[SKIP] {game_id} id={id_explain}: n_players={game_to_run.n_players} > 20.")
                continue
            if benchmark is not None:
                gt = benchmark.exact_values(index=info["index"], order=info["order"])
            else:
                gt = game_to_run.exact_values(index=info["index"], order=info["order"])
            gt.save(save_path)
            print(f"[TRUE] Saved: {save_path.name}")


if __name__ == "__main__":
    args = parse_args()
    pairing = resolve_pairing(args.config_approximators)
    base = resolve_base_path(args)
    approx_dir = base / "approximations" / args.game_type
    truth_dir = base / "ground_truth" / args.game_type
    approx_dir.mkdir(parents=True, exist_ok=True)
    truth_dir.mkdir(parents=True, exist_ok=True)

    benchmarks = BenchmarkFactory.load_benchmarks_from_json(config_path=args.config)

    if args.mode == "approx":
        run_approx_mode(args, benchmarks, pairing, approx_dir)
    else:
        run_true_mode(args, benchmarks, truth_dir)
