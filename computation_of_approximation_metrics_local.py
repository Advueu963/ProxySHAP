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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Local metrics computation for Shapley interaction approximations.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        required=False,
        default=None,
        help="Path to the benchmark config JSON file.",
    )
    parser.add_argument(
        "--game_type",
        required=True,
        help="Game type / approximations subdirectory (e.g. exhaustive, interventional).",
    )
    parser.add_argument(
        "--config_approximators",
        type=int,
        default=37,
        help=(
            "Approximator configuration ID used during benchmark generation: "
            "37 (PAIRING=True, REPLACEMENT=False), "
            "38 (PAIRING=True, REPLACEMENT=True), "
            "39 (PAIRING=False, REPLACEMENT=False), "
            "40 (PAIRING=False, REPLACEMENT=True). Default: 37."
        ),
    )
    parser.add_argument(
        "--index",
        type=str,
        default="SII",
        help="Interaction index (e.g. SV, SII, FBII). Default: SII.",
    )
    parser.add_argument(
        "--order",
        type=int,
        default=2,
        help="Interaction order. Default: 2.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Base output directory. Defaults to $SCRATCH_DSS/msr_int_iq or cwd.",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=40,
        help="Random state used in graph ground-truth filenames. Default: 40.",
    )
    return parser.parse_args()


def resolve_base_path(args: argparse.Namespace) -> Path:
    if args.output_path:
        return Path(args.output_path)
    if "SCRATCH_DSS" in os.environ:
        return Path("/dss/dssfs02/lwp-dss-0001/pn49je/pn49je-dss-0000/ra98xir2/neurips_tree")
    return Path.cwd()


def iv_from_result_dict(result_dict: dict) -> InteractionValues:
    """Reconstruct an InteractionValues object from a shard result dict."""
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


def build_metrics_dict(all_metrics) -> dict:
    metrics_dict = {}
    for metric in all_metrics:
        key = str(getattr(metric, "metric_id", repr(metric)))
        if getattr(metric, "computed_k", None) is not None:
            key = f"{key[:-2]}@{metric.computed_k}" if "@" in key else f"{key}@{metric.computed_k}"
        elif getattr(metric, "order", None) is not None:
            key = f"{key}_order{metric.order}"
        try:
            metrics_dict[key] = float(metric.value)
        except Exception:
            metrics_dict[key] = metric.value
    return metrics_dict


def iter_results_from_shards(
    shard_dir: Path,
    game_id: str,
    config_approx: int,
    index: str,
    order: int,
):
    pattern_ga = str(shard_dir / f"{game_id}_{config_approx}_*_{index}_{order}.shard.json")
    pattern_g = str(shard_dir / f"{game_id}_{config_approx}_{index}_{order}.shard.json")

    shard_paths = sorted(set(glob.glob(pattern_ga)) | set(glob.glob(pattern_g)))
    for shard_path in shard_paths:
        with open(shard_path) as f:
            shard = json.load(f)
        approx_name = shard["shard_meta"].get("approx_name")
        if approx_name is None:
            for explain_str, approx_dict in shard["results"].items():
                id_explain = int(explain_str)
                for aname, budgets in approx_dict.items():
                    for budget_str, result_dict in budgets.items():
                        budget = int(budget_str)
                        runtime_params = result_dict.get("parameters", {})
                        yield id_explain, aname, budget, result_dict, runtime_params
        else:
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
    pattern = str(approx_dir / f"{game_id}_{config_approx}_*_*_{index}_{order}.json")
    for file_path in sorted(glob.glob(pattern)):
        fname = Path(file_path).name
        parts = fname.split("_")
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


def iter_results_from_graph_files(
    approx_dir: Path,
    config_approx: int,
    index: str,
    order: int,
):
    pattern = str(approx_dir / f"*_{config_approx}_*_*_{index}_{order}.json")
    for file_path in sorted(glob.glob(pattern)):
        fname = Path(file_path).name
        parts = fname.rsplit(".", 1)[0].split("_")
        try:
            id_explain = int(parts[-5])
            approx_name = parts[-4]
            budget = int(parts[-3])
            game_id = "_".join(parts[:-6])
        except (ValueError, IndexError):
            print(f"  [WARN] Cannot parse graph filename: {fname} — skipping.")
            continue
        with open(file_path) as f:
            runtime_params = json.load(f).get("parameters", {})
        yield game_id, id_explain, approx_name, budget, file_path, runtime_params


def run_local_mode(args: argparse.Namespace) -> None:
    base = resolve_base_path(args)
    approx_dir = base / "approximations" / args.game_type
    shard_dir = approx_dir / "_shards"
    truth_dir = base / "ground_truth" / args.game_type

    if args.game_type == "graph":
        run_graph_mode(args, approx_dir, truth_dir)
        return

    if args.config is None:
        raise ValueError("--config is required for non-graph game types.")

    with open(args.config) as f:
        raw_config = json.load(f)

    results = []
    random_state = args.random_state

    for benchmark_name, info in raw_config.items():
        config_id = info["config_id"]
        game_id = f"{benchmark_name}_{config_id}"
        print(f"Loading {game_id}")

        game_config_path = f"shapiq-benchmark/configurations_{args.game_type}/{benchmark_name}.json"
        try:
            with open(game_config_path) as f:
                game_configurations = json.load(f)
        except Exception:
            print(f"  [WARN] Cannot load game config {game_config_path} — faithfulness metric will be skipped.")
            game_configurations = None

        game_class = GAME_NAME_TO_CLASS_MAPPING.get(benchmark_name)
        has_shards = bool(
            glob.glob(str(shard_dir / f"{game_id}_{args.config_approximators}_*_{args.index}_{args.order}.shard.json"))
            or glob.glob(str(shard_dir / f"{game_id}_{args.config_approximators}_{args.index}_{args.order}.shard.json"))
        )

        if has_shards:
            print(f"  Source: shard files in {shard_dir}")
        else:
            print(f"  Source: individual files in {approx_dir}  (no shards found for {game_id})")

        gt_cache: dict[int, InteractionValues | None] = {}
        game_cache: dict[int, Game | None] = {}

        def get_ground_truth(id_explain: int) -> InteractionValues | None:
            if id_explain not in gt_cache:
                gt_path = (
                    truth_dir
                    / f"{game_id}_{random_state}_{id_explain}_{args.index}_{args.order}_exact_values.json"
                )
                try:
                    gt_cache[id_explain] = InteractionValues.load(gt_path)
                except Exception:
                    print(f"  [WARN] Ground truth not found: {gt_path.name}")
                    gt_cache[id_explain] = None
            return gt_cache[id_explain]

        def get_underlying_game(id_explain: int) -> Game | None:
            if id_explain not in game_cache:
                if game_configurations is None or game_class is None:
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

        def process_one(id_explain: int, approx_name: str, budget: int, iv: InteractionValues, runtime_params: dict) -> None:
            ground_truth = get_ground_truth(id_explain)
            if ground_truth is None:
                return

            underlying_game = get_underlying_game(id_explain)

            try:
                all_metrics = get_all_metrics(ground_truth, iv, underlying_game)
            except Exception as exc:
                print(f"  [ERROR] Metrics failed: explain={id_explain} approx={approx_name} budget={budget}: {exc}")
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
            for key, value in runtime_params.items():
                if isinstance(value, dict):
                    result.update(value)
                else:
                    result[key] = value

            results.append(result)

        # Track results from both sources to avoid duplicates
        processed_keys = set()

        if has_shards:
            for id_explain, approx_name, budget, result_dict, runtime_params in iter_results_from_shards(
                shard_dir,
                game_id,
                args.config_approximators,
                args.index,
                args.order,
            ):
                try:
                    iv = iv_from_result_dict(result_dict)
                except Exception as exc:
                    print(
                        f"  [ERROR] Cannot parse IV from shard: explain={id_explain} "
                        f"approx={approx_name} budget={budget}: {exc}"
                    )
                    continue
                process_one(id_explain, approx_name, budget, iv, runtime_params)
                print(f"  ✓ explain={id_explain} approx={approx_name} budget={budget}")
                processed_keys.add((id_explain, approx_name, budget))

        # Always also check individual files as fallback or primary source
        for id_explain, approx_name, budget, file_path, runtime_params in iter_results_from_individual_files(
            approx_dir,
            game_id,
            args.config_approximators,
            args.index,
            args.order,
        ):
            key = (id_explain, approx_name, budget)
            if key in processed_keys:
                continue  # Skip if already processed from shards
            try:
                iv = InteractionValues.load(file_path)
            except Exception as exc:
                print(f"  [ERROR] Cannot load IV from {file_path}: {exc}")
                continue
            process_one(id_explain, approx_name, budget, iv, runtime_params)
            print(f"  ✓ explain={id_explain} approx={approx_name} budget={budget}")
            processed_keys.add(key)

    if not results:
        print("[WARN] No results collected. CSV not written.")
        return

    out_csv = Path.cwd() / f"results_benchmark_{args.index}_{args.order}_{args.game_type}_sii3_permutation_missing.csv"
    pd.DataFrame(results).to_csv(out_csv, index=False)
    print(f"[DONE] Wrote {len(results)} rows -> {out_csv}")


def run_graph_mode(args: argparse.Namespace, approx_dir: Path, truth_dir: Path) -> None:
    results = []

    def get_ground_truth(game_id: str, id_explain: int) -> InteractionValues | None:
        gt_path = (
            truth_dir
            / f"{game_id}_{args.random_state}_{id_explain}_{args.index}_{args.order}_exact_values.json"
        )
        try:
            return InteractionValues.load(gt_path)
        except Exception:
            print(f"  [WARN] Ground truth not found: {gt_path.name}")
            return None

    for game_id, id_explain, approx_name, budget, file_path, runtime_params in iter_results_from_graph_files(
        approx_dir,
        args.config_approximators,
        args.index,
        args.order,
    ):
        ground_truth = get_ground_truth(game_id, id_explain)
        if ground_truth is None:
            continue

        try:
            approximated_values = InteractionValues.load(file_path)
        except Exception as exc:
            print(f"  [ERROR] Cannot load IV from {file_path}: {exc}")
            continue

        try:
            all_metrics = get_all_metrics(ground_truth, approximated_values, None)
        except Exception as exc:
            print(
                f"  [ERROR] Metrics failed: game_id={game_id} explain={id_explain} "
                f"approx={approx_name} budget={budget}: {exc}"
            )
            continue

        result = {
            "game_type": args.game_type,
            "game": game_id,
            "model": game_id,
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
        for key, value in runtime_params.items():
            if isinstance(value, dict):
                result.update(value)
            else:
                result[key] = value
        results.append(result)
        print(f"  ✓ game_id={game_id} explain={id_explain} approx={approx_name} budget={budget}")

    if not results:
        print("[WARN] No results collected. CSV not written.")
        return

    out_csv = Path.cwd() / f"results_benchmark_{args.index}_{args.order}_{args.game_type}.csv"
    pd.DataFrame(results).to_csv(out_csv, index=False)
    print(f"[DONE] Wrote {len(results)} rows -> {out_csv}")


if __name__ == "__main__":
    run_local_mode(parse_args())