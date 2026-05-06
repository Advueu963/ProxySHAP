from __future__ import annotations

import argparse
import csv
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator

import pandas as pd
from sklearn.model_selection import ParameterGrid

pd.set_option("future.no_silent_downcasting", True)

from shapiq_benchmark.load import BenchmarkFactory
from intiq.regressionMSR import RegressionMSR
from intiq.tree_recap import TreeRECAP
from shapiq.interaction_values import InteractionValues
from shapiq_benchmark.metrics import _remove_empty_value

POSSIBLE_MODELS = ["xgb", "rf", "dt", "lightgbm"]
SAMPLING_WEIGHTS = ["default", "leverage"]
RESIDUAL_APPROXIMATOR = ["shapiq", "kernelshapiq", "svarmiq"]
regression_msr_params = {
    "value_model": POSSIBLE_MODELS,
    "sampling_weight": SAMPLING_WEIGHTS,
    "residual_approximator": RESIDUAL_APPROXIMATOR,
}
tree_recap_params = {
    "value_model": POSSIBLE_MODELS,
    "sampling_weight": SAMPLING_WEIGHTS,
}
PARAM_GRIDS = {
    "xgb": {
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 10, 20],
        # "learning_rate": [0.01, 0.1, 0.2],
        # "subsample": [0.6, 0.8, 1.0],
        # "colsample_bytree": [0.6, 0.8, 1.0],
    },
    "rf": {
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 10, 20],
        # "min_samples_split": [2, 5, 10],
        # "min_samples_leaf": [1, 2, 4],
    },
    "dt": {
        "max_depth": [None, 10, 20],
        # "min_samples_split": [2, 5, 10],
        # "min_samples_leaf": [1, 2, 4],
    },
    "lightgbm": {
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 10, 20],
        # "learning_rate": [0.01, 0.1, 0.2],
        # "subsample": [0.6, 0.8, 1.0],
        # "colsample_bytree": [0.6, 0.8, 1.0],
    },
}
GAMES_PARAMS = {
    "exhaustive": BenchmarkFactory.load_benchmarks_from_json(
        "shapiq-benchmark/benchmarks/configuration_abalation_exhaustive.json"
    ),
    "pathdependent": BenchmarkFactory.load_benchmarks_from_json(
        "shapiq-benchmark/benchmarks/configuration_abalation_pathdependent.json"
    ),
    "interventional": BenchmarkFactory.load_benchmarks_from_json(
        "shapiq-benchmark/benchmarks/configuration_abalation_interventional.json"
    ),
    "exhaustive_tabpfn": BenchmarkFactory.load_benchmarks_from_json(
        "shapiq-benchmark/benchmarks/configuration_abalation_exhaustive_tabpfn.json"
    ),
}
DEFAULT_GAME_TYPES = tuple(GAMES_PARAMS.keys())


def _materialize_games() -> None:
    """Ensure all benchmark game iterables are materialized as lists.

    Some benchmark loaders may return generators/iterators for the "games" field.
    If those iterators are consumed in the parent process before forking workers,
    worker processes will see an exhausted iterator and indexing will fail.
    """

    for benchmarks in GAMES_PARAMS.values():
        for benchmark_info in benchmarks.values():
            games = benchmark_info.get("games")
            if games is None:
                continue
            if not isinstance(games, list):
                benchmark_info["games"] = list(games)


_materialize_games()


def _slurm_array_context() -> tuple[int | None, int | None]:
    """Return (shard_index, num_shards) from SLURM env, if present.

    SLURM_ARRAY_TASK_ID is 1-based; we convert it to 0-based.
    """

    task_id = os.environ.get("SLURM_ARRAY_TASK_ID")
    task_count = os.environ.get("SLURM_ARRAY_TASK_COUNT")
    if not task_id or not task_count:
        return None, None
    try:
        shard_index = int(task_id) - 1
        num_shards = int(task_count)
    except ValueError:
        return None, None
    if shard_index < 0 or num_shards <= 0:
        return None, None
    return shard_index, num_shards


def _infer_max_workers(user_value: int | None) -> int | None:
    """Infer a good per-node worker count.

    On SLURM, SLURM_CPUS_PER_TASK is typically set and should match
    --cpus-per-task. When absent, ProcessPoolExecutor will use os.cpu_count().
    """

    if user_value is not None:
        return user_value if user_value > 0 else None

    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm_cpus:
        try:
            inferred = int(slurm_cpus)
            return inferred if inferred > 0 else None
        except ValueError:
            return None
    return None


def _get_game_instance(*, game_type: str, game_identifier: str, id_explain: int):
    benchmark_info = GAMES_PARAMS[game_type][game_identifier]
    games = benchmark_info["games"]
    if not isinstance(games, list):
        games = list(games)
        benchmark_info["games"] = games
    if id_explain < 0 or id_explain >= len(games):
        raise IndexError(
            f"id_explain={id_explain} out of range for {game_type}/{game_identifier} "
            f"with {len(games)} games"
        )
    game_instance = games[id_explain]
    return game_instance, benchmark_info["index"], benchmark_info["order"]


def _shard_jobs(
    job_iter: Iterable[Dict[str, Any]], *, shard_index: int, num_shards: int
) -> Iterator[Dict[str, Any]]:
    """Deterministically split a job stream across shards."""

    for i, job in enumerate(job_iter):
        if (i % num_shards) == shard_index:
            yield job


def _resolve_output_path(
    output: str, *, shard_index: int | None, num_shards: int | None
) -> str:
    if shard_index is None or num_shards is None or num_shards <= 1:
        return output
    p = Path(output)
    suffix = "".join(p.suffixes) or ".csv"
    stem = p.name[: -len(suffix)] if p.name.endswith(suffix) else p.stem
    shard_tag = f".shard{shard_index:04d}-of-{num_shards:04d}"
    return str(p.with_name(f"{stem}{shard_tag}{suffix}"))


def _stream_to_csv(rows: Iterable[Dict[str, Any]], *, output_path: str) -> int:
    # Make the header stable across all model types.
    model_param_keys: set[str] = set()
    for grid in PARAM_GRIDS.values():
        model_param_keys.update(grid.keys())

    fieldnames = [
        "approximator",
        "game_type",
        "game_identifier",
        "id_explain",
        "value_model",
        "sampling_weight",
        "residual_approximator",
        "MSE",
        *sorted(model_param_keys),
    ]

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with output_file.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
            count += 1
    return count


def _build_ground_truth_path(
    *,
    game_type: str,
    game_identifier: str,
    id_explain: int,
    index: str,
    max_order: int,
) -> str:
    return str(
        Path("ground_truth")
        / game_type
        / f"{game_identifier}_40_{id_explain}_{index}_{max_order}_exact_values.json"
    )


def _iter_jobs(selected_game_types: Iterable[str]) -> Iterator[Dict[str, Any]]:
    for game_type in selected_game_types:
        benchmarks = GAMES_PARAMS.get(game_type)
        if benchmarks is None:
            raise ValueError(f"Unknown game type '{game_type}'.")
        for game_identifier, benchmark_info in benchmarks.items():
            games = benchmark_info["games"]
            if not isinstance(games, list):
                games = list(games)
                benchmark_info["games"] = games
            index = benchmark_info["index"]
            max_order = benchmark_info["order"]
            for id_explain in range(len(games)):
                ground_truth_path = _build_ground_truth_path(
                    game_type=game_type,
                    game_identifier=game_identifier,
                    id_explain=id_explain,
                    index=index,
                    max_order=max_order,
                )
                for msr_params in ParameterGrid(regression_msr_params):
                    value_model = msr_params["value_model"]
                    param_grid = PARAM_GRIDS[value_model]
                    for model_params in ParameterGrid(param_grid):
                        # Emit a fully-specified job so workers stay side-effect free.
                        yield {
                            "approximator": "RegressionMSR",
                            "game_type": game_type,
                            "game_identifier": game_identifier,
                            "id_explain": id_explain,
                            "index": index,
                            "max_order": max_order,
                            "msr_params": msr_params,
                            "model_params": model_params,
                            "ground_truth_path": ground_truth_path,
                        }
                for recap_params in ParameterGrid(tree_recap_params):
                    value_model = recap_params["value_model"]
                    param_grid = PARAM_GRIDS[value_model]
                    for model_params in ParameterGrid(param_grid):
                        # Emit a fully-specified job so workers stay side-effect free.
                        yield {
                            "approximator": "TreeRecap",
                            "game_type": game_type,
                            "game_identifier": game_identifier,
                            "id_explain": id_explain,
                            "index": index,
                            "max_order": max_order,
                            "msr_params": recap_params,
                            "model_params": model_params,
                            "ground_truth_path": ground_truth_path,
                        }


def _evaluate_job(job: Dict[str, Any]) -> Dict[str, Any]:
    game_instance, _, _ = _get_game_instance(
        game_type=job["game_type"],
        game_identifier=job["game_identifier"],
        id_explain=job["id_explain"],
    )
    budget = min(10_000, (2**game_instance.n_players) / 2)

    msr_params = job["msr_params"]
    model_params = job["model_params"]
    value_model = msr_params["value_model"]
    if job["approximator"] == "TreeRecap":

        tree_recap = TreeRECAP(
            n=game_instance.n_players,
            own_intervention=True,
            pairing_trick=True,
            replacement=False,
            random_state=40,
            index=job["index"],
            max_order=job["max_order"],
            ablation_kwargs={
                "value_model": value_model,
                "model_params": model_params,
                "sampling_weights": msr_params["sampling_weight"],
            },
        )

        approx_interactions = tree_recap.approximate(
            game=game_instance, budget=int(budget)
        )
    else:
        regression_msr = RegressionMSR(
            n=game_instance.n_players,
            own_intervention=True,
            pairing_trick=True,
            replacement=False,
            random_state=40,
            index=job["index"],
            max_order=job["max_order"],
            ablation_kwargs={
                "value_model": value_model,
                "model_params": model_params,
                "sampling_weights": msr_params["sampling_weight"],
                "residual_approximator": msr_params["residual_approximator"],
            },
        )
        approx_interactions = regression_msr.approximate(
            game=game_instance, budget=int(budget)
        )

    ground_truth = InteractionValues.load(job["ground_truth_path"])
    difference = ground_truth - approx_interactions
    diff_values = _remove_empty_value(difference).values
    mse = float((diff_values**2).mean())

    row = {
        "approximator": job["approximator"],
        "game_type": job["game_type"],
        "game_identifier": job["game_identifier"],
        "id_explain": job["id_explain"],
        "value_model": value_model,
        "sampling_weight": msr_params["sampling_weight"],
        "residual_approximator": msr_params.get("residual_approximator"),
        "MSE": mse,
    }
    row.update(model_params)
    return row


def _consume_jobs(
    *,
    job_iter: Iterable[Dict[str, Any]],
    max_workers: int | None,
    chunksize: int,
) -> Iterator[Dict[str, Any]]:
    if max_workers == 1:
        # Helpful for debugging or when resources are scarce.
        for job in job_iter:
            yield _evaluate_job(job)
        return

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # executor.map streams jobs lazily; chunksize trims IPC overhead.
        for row in executor.map(_evaluate_job, job_iter, chunksize=chunksize):
            yield row


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Search for the best RegressionMSR configuration using parallel workers."
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Number of processes to use. Defaults to os.cpu_count(). Set to 1 to disable parallelism.",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=1,
        help="Number of jobs dispatched to each worker at once when running in parallel.",
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=None,
        help="0-based shard index for splitting the sweep (useful for SLURM job arrays).",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=None,
        help="Total number of shards for splitting the sweep (useful for SLURM job arrays).",
    )
    parser.add_argument(
        "--game-types",
        nargs="+",
        choices=DEFAULT_GAME_TYPES,
        default=list(DEFAULT_GAME_TYPES),
        help="Subset of game types to evaluate.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="find_optimal_model_results.csv",
        help="Path of the CSV file that stores the evaluation results.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    chunksize = max(1, args.chunksize)

    # Sharding for SLURM arrays (multi-node parallelism).
    shard_index = args.shard_index
    num_shards = args.num_shards
    if shard_index is None or num_shards is None:
        slurm_shard_index, slurm_num_shards = _slurm_array_context()
        shard_index = shard_index if shard_index is not None else slurm_shard_index
        num_shards = num_shards if num_shards is not None else slurm_num_shards
    if (shard_index is None) != (num_shards is None):
        raise ValueError("Provide both --shard-index and --num-shards, or neither.")
    if shard_index is not None:
        if num_shards is None or num_shards <= 0:
            raise ValueError("--num-shards must be > 0")
        if shard_index < 0 or shard_index >= num_shards:
            raise ValueError("--shard-index must be in [0, num_shards)")

    max_workers = _infer_max_workers(args.max_workers)
    job_iter: Iterable[Dict[str, Any]] = _iter_jobs(args.game_types)
    if shard_index is not None and num_shards is not None and num_shards > 1:
        job_iter = _shard_jobs(job_iter, shard_index=shard_index, num_shards=num_shards)

    rows = _consume_jobs(
        job_iter=job_iter, max_workers=max_workers, chunksize=chunksize
    )
    output_path = _resolve_output_path(
        args.output, shard_index=shard_index, num_shards=num_shards
    )
    written = _stream_to_csv(rows, output_path=output_path)
    if written == 0:
        # This can happen when using many shards relative to the number of jobs.
        # In SLURM arrays, empty shards should exit cleanly to avoid spurious failures.
        if shard_index is not None and num_shards is not None and num_shards > 1:
            print(
                "No evaluation jobs were generated for this shard "
                f"(shard {shard_index + 1}/{num_shards})."
            )
            return
        raise RuntimeError(
            "No evaluation jobs were generated. Check the game configuration inputs."
        )
    print(f"Wrote {written} rows to {output_path}")


if __name__ == "__main__":
    main()
