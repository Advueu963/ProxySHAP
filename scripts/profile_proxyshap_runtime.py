"""Profile ProxySHAP runtime bottlenecks on benchmark games.

This script runs selected ProxySHAP approximators on selected benchmark games
and budgets, and writes two outputs:

1) Phase-level timings from `approximator.runtime_last_approximate_run`
2) Function-level profiling from `cProfile` (top cumulative time functions)

Example:
    uv run python scripts/profile_proxyshap_runtime.py \
      --config shapiq-benchmark/benchmarks/tabarena/configuration_interventional_tabarena_shapley_order2_large.json \
      --game-filter TabArenaBioresponseLocalXAI \
      --approximator "ProxySHAP (XGBoost)" \
      --explain-id 0 \
      --budgets 2000 5000 10000 \
      --out-dir results/proxyshap_runtime_profile
"""

from __future__ import annotations

import argparse
import cProfile
from dataclasses import dataclass
import json
from pathlib import Path
import pstats
import time
from typing import Any

import pandas as pd

from shapiq_benchmark.approximators import get_approximators
from shapiq_benchmark.load import BenchmarkFactory


PAIRING_MAP: dict[int, bool] = {37: True, 38: True, 39: False, 40: False}


@dataclass
class RunResult:
    game_key: str
    explain_id: int
    approximator: str
    budget: int
    wall_clock: float
    runtime_details: dict[str, Any]
    top_functions: list[dict[str, Any]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile ProxySHAP runtime bottlenecks.")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to benchmark json config (e.g. configuration_interventional_*.json).",
    )
    parser.add_argument(
        "--game-filter",
        default=None,
        help="Optional substring filter on benchmark key (e.g. TabArenaBioresponse).",
    )
    parser.add_argument(
        "--approximator",
        action="append",
        default=None,
        help=(
            "Approximator name to profile. Can be passed multiple times. "
            "Default: ProxySHAP (XGBoost)."
        ),
    )
    parser.add_argument(
        "--config-approximators",
        type=int,
        default=37,
        help="Configuration id in {37, 38, 39, 40} for pairing/replacement setup.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=40,
        help="Random seed for approximators.",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=None,
        help="Optional override for tree-based approximators.",
    )
    parser.add_argument(
        "--explain-id",
        type=int,
        default=0,
        help="Which explanation instance to profile (0-based).",
    )
    parser.add_argument(
        "--budgets",
        type=int,
        nargs="+",
        required=True,
        help="Budgets to profile.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=40,
        help="Top-K functions from cProfile sorted by cumulative time.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/proxyshap_runtime_profile"),
        help="Output directory for csv/json/txt reports.",
    )
    return parser.parse_args()


def _load_benchmarks(config_path: str) -> dict[str, Any]:
    return BenchmarkFactory.load_benchmarks_from_json(config_path)


def _build_approximators_for_game(
    benchmark_info: dict[str, Any],
    game,
    args: argparse.Namespace,
) -> list[Any]:
    pairing = PAIRING_MAP.get(args.config_approximators)
    if pairing is None:
        msg = "--config-approximators must be one of {37, 38, 39, 40}."
        raise ValueError(msg)

    return get_approximators(
        benchmark_info["approximation_methods"],
        game.n_players,
        args.random_state,
        pairing,
        benchmark_info["index"],
        benchmark_info["order"],
        n_estimators=args.n_estimators,
    )


def _profile_single_call(
    approximator: Any,
    game: Any,
    game_key: str,
    explain_id: int,
    budget: int,
    top_k: int,
) -> RunResult:
    profiler = cProfile.Profile()

    t_start = time.perf_counter()
    profiler.enable()
    approximator.approximate(
        budget=int(budget),
        game=game,
        game_id=game_key,
        id_explain=explain_id,
    )
    profiler.disable()
    wall_clock = time.perf_counter() - t_start

    stats = pstats.Stats(profiler)
    stats.sort_stats("cumulative")

    top_functions: list[dict[str, Any]] = []
    # pstats key: (filename, line_no, function_name)
    for func_key, stat_tuple in list(stats.stats.items()):
        ccalls, ncalls, tottime, cumtime, _ = stat_tuple
        filename, line_no, func_name = func_key
        top_functions.append(
            {
                "filename": filename,
                "line": line_no,
                "function": func_name,
                "primitive_calls": ccalls,
                "total_calls": ncalls,
                "tottime": float(tottime),
                "cumtime": float(cumtime),
            }
        )

    top_functions = sorted(top_functions, key=lambda x: x["cumtime"], reverse=True)[:top_k]

    runtime_details = dict(getattr(approximator, "runtime_last_approximate_run", {}))

    return RunResult(
        game_key=game_key,
        explain_id=explain_id,
        approximator=getattr(approximator, "name", approximator.__class__.__name__),
        budget=budget,
        wall_clock=wall_clock,
        runtime_details=runtime_details,
        top_functions=top_functions,
    )


def _result_to_flat_row(result: RunResult) -> dict[str, Any]:
    row: dict[str, Any] = {
        "game": result.game_key,
        "explain_id": result.explain_id,
        "approximator": result.approximator,
        "budget": result.budget,
        "wall_clock": result.wall_clock,
    }
    for key, value in result.runtime_details.items():
        row[f"runtime_{key}"] = value

    eval_time = float(result.runtime_details.get("evaluations", 0.0) or 0.0)
    total_time = float(result.runtime_details.get("total", result.wall_clock) or result.wall_clock)
    row["runtime_overhead_excl_eval"] = max(total_time - eval_time, 0.0)
    return row


def _write_reports(results: list[RunResult], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = [_result_to_flat_row(r) for r in results]
    summary_df = pd.DataFrame(summary_rows)
    summary_csv = out_dir / "runtime_phase_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    detailed_json = out_dir / "runtime_function_profile.json"
    with detailed_json.open("w", encoding="utf-8") as f:
        json.dump([r.__dict__ for r in results], f, indent=2)

    # Create a compact text report with top function hotspots per run.
    text_report = out_dir / "runtime_hotspots.txt"
    lines: list[str] = []
    for r in results:
        lines.append(
            f"Run: game={r.game_key}, explain_id={r.explain_id}, approx={r.approximator}, budget={r.budget}"
        )
        lines.append(
            f"  wall_clock={r.wall_clock:.4f}s, runtime_details={json.dumps(r.runtime_details, sort_keys=True)}"
        )
        lines.append("  Top functions by cumulative time:")
        for i, entry in enumerate(r.top_functions[:15], start=1):
            lines.append(
                "    "
                + f"{i:02d}. {entry['cumtime']:.4f}s cum | {entry['tottime']:.4f}s self | "
                + f"{entry['filename']}:{entry['line']}::{entry['function']}"
            )
        lines.append("")

    text_report.write_text("\n".join(lines), encoding="utf-8")

    print(f"Saved phase summary: {summary_csv}")
    print(f"Saved detailed function profile: {detailed_json}")
    print(f"Saved hotspot text report: {text_report}")


def main() -> None:
    args = parse_args()

    benchmarks = _load_benchmarks(args.config)
    selected_keys = sorted(benchmarks.keys())
    if args.game_filter:
        selected_keys = [k for k in selected_keys if args.game_filter in k]

    if not selected_keys:
        raise ValueError("No benchmark entries matched --game-filter.")

    all_results: list[RunResult] = []

    for game_key in selected_keys:
        info = benchmarks[game_key]
        games = list(info["games"])

        if args.explain_id < 0 or args.explain_id >= len(games):
            raise IndexError(
                f"--explain-id={args.explain_id} out of range for {game_key}; "
                f"available explanations: 0..{len(games) - 1}"
            )

        game = games[args.explain_id]
        approximators = _build_approximators_for_game(info, game, args)
        requested_approximators = args.approximator or ["ProxySHAP (XGBoost)"]
        wanted = set(requested_approximators)
        approximators = [a for a in approximators if getattr(a, "name", "") in wanted]

        if not approximators:
            available = [getattr(a, "name", a.__class__.__name__) for a in _build_approximators_for_game(info, game, args)]
            raise ValueError(
                f"No requested approximator found for {game_key}. "
                f"Requested={sorted(wanted)} Available={available}"
            )

        for approximator in approximators:
            for budget in args.budgets:
                print(
                    f"Profiling game={game_key}, explain_id={args.explain_id}, "
                    f"approx={approximator.name}, budget={budget}"
                )
                result = _profile_single_call(
                    approximator=approximator,
                    game=game,
                    game_key=game_key,
                    explain_id=args.explain_id,
                    budget=budget,
                    top_k=args.top_k,
                )
                all_results.append(result)

    _write_reports(all_results, args.out_dir)


if __name__ == "__main__":
    main()
