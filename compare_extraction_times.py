"""
Comparison of Fourier extraction vs Interventional Explainer extraction times
across multiple interventional datasets for SII and BII interaction indices.

This script:
1. Loads all datasets from interventional benchmark configs (order 2 & 3)
2. Creates interventional games and coalition samplers for each dataset
3. Trains XGBoost models with varying tree depths
4. Benchmarks both extraction methods (Fourier vs InterventionalTreeExplainer)
5. Generates plots showing relative runtime (InterventionalTreeExplainer / Fourier)
"""

from __future__ import annotations

import json
import time
import csv
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
from xgboost import XGBRegressor

from shapiq_games.benchmark.local_xai import (
    AdultCensus,
    BikeSharing,
    BreastCancer,
    CaliforniaHousing,
    CommunitiesAndCrime,
    Corrgroups60,
    ForestFires,
    Independent,
    Nhanesi,
    RealEstate,
)
from shapiq.approximator.sampling import CoalitionSampler
from shapiq.game_theory.moebius_converter import MoebiusConverter
from shapiq.interaction_values import InteractionValues
from shapiq.tree.interventional import InterventionalGame
from shapiq.tree.interventional.explainer import InterventionalTreeExplainer


# Mapping of dataset names to their corresponding classes
DATASET_CLASSES = {
    "AdultCensusLocalXAI": AdultCensus,
    "BikeSharingLocalXAI": BikeSharing,
    "BreastCancerLocalXAI": BreastCancer,
    "CaliforniaHousingLocalXAI": CaliforniaHousing,
    "CommunitiesAndCrimeLocalXAI": CommunitiesAndCrime,
    "Corrgroups60LocalXAI": Corrgroups60,
    "ForestFiresLocalXAI": ForestFires,
    "IndependentLinear60LocalXAI": Independent,
    "NHANESILocalXAI": Nhanesi,
    "RealEstateLocalXAI": RealEstate,
}

# Configuration
ORDER_TO_DEPTH_RANGE = {
    1: range(1, 9),
    2: range(2, 9),
    3: range(3, 9),
}
NUM_BENCHMARK_RUNS = 5
RANDOM_STATE = 42


def _xgboost_to_fourier(
    model_dict: list[dict[str, Any]],
) -> dict[tuple[int, ...], float]:
    """Extracts the aggregated Fourier coefficients from an XGBoost model dictionary."""
    aggregated_coeffs = defaultdict(float)

    for tree_info in model_dict:
        tree_coeffs = _xgboost_tree_to_fourier(json.loads(tree_info))
        for interaction, value in tree_coeffs.items():
            aggregated_coeffs[interaction] += value

    return {k: v for k, v in aggregated_coeffs.items() if v != 0.0}


def _xgboost_tree_to_fourier(tree_info: dict[str, Any]) -> dict[tuple[int, ...], float]:
    """Recursively strips the Fourier coefficients from a single XGBoost tree."""

    def _combine_coeffs(
        left_coeffs: dict[tuple[int, ...], float],
        right_coeffs: dict[tuple[int, ...], float],
        feature_idx: int,
    ) -> dict[tuple[int, ...], float]:
        """Combines Fourier coefficients from the left and right children of a split node."""
        combined_coeffs = {}
        all_interactions = set(left_coeffs.keys()) | set(right_coeffs.keys())

        for interaction in all_interactions:
            left_val = left_coeffs.get(interaction, 0.0)
            right_val = right_coeffs.get(interaction, 0.0)
            combined_coeffs[interaction] = (left_val + right_val) / 2

            new_interaction = tuple(sorted(set(interaction) | {feature_idx}))
            combined_coeffs[new_interaction] = (left_val - right_val) / 2
        return combined_coeffs

    def _dfs_traverse(node: dict[str, Any]) -> dict[tuple[int, ...], float]:
        """Performs a depth-first traversal of the tree to compute coefficients."""
        if "leaf" in node:
            return {(): node["leaf"]}

        left_coeffs = _dfs_traverse(node["children"][0])
        right_coeffs = _dfs_traverse(node["children"][1])
        feature_idx = int(node["split"][1:])

        return _combine_coeffs(left_coeffs, right_coeffs, feature_idx)

    return _dfs_traverse(tree_info)


def _fourier_to_sii(
    fourier_coeffs: dict[tuple[int, ...], float], n_players: int, order: int
) -> np.ndarray:
    """Convert Fourier coefficients to SII (Shapley Interaction Index)."""
    from shapiq.approximator.sparse.base import fourier_to_moebius

    moebius_transform = fourier_to_moebius(fourier_coeffs)
    moebius_interactions = InteractionValues(
        values=np.array([moebius_transform[key] for key in moebius_transform]),
        index="Moebius",
        min_order=0,
        max_order=order,
        n_players=n_players,
        interaction_lookup={key: i for i, key in enumerate(moebius_transform.keys())},
        estimated=True,
        baseline_value=moebius_transform.get((), 0.0),
    )
    autoconverter = MoebiusConverter(moebius_coefficients=moebius_interactions)
    converted = autoconverter(index="SII", order=order)
    return converted.values


def _fourier_to_fsii(
    fourier_coeffs: dict[tuple[int, ...], float], n_players: int, order: int
) -> np.ndarray:
    """Convert Fourier coefficients to SII (Shapley Interaction Index)."""
    from shapiq.approximator.sparse.base import fourier_to_moebius

    moebius_transform = fourier_to_moebius(fourier_coeffs)
    moebius_interactions = InteractionValues(
        values=np.array([moebius_transform[key] for key in moebius_transform]),
        index="Moebius",
        min_order=0,
        max_order=order,
        n_players=n_players,
        interaction_lookup={key: i for i, key in enumerate(moebius_transform.keys())},
        estimated=True,
        baseline_value=moebius_transform.get((), 0.0),
    )
    autoconverter = MoebiusConverter(moebius_coefficients=moebius_interactions)
    converted = autoconverter(index="FSII", order=order)
    return converted.values


def _fourier_to_bii(
    fourier_coeffs: dict[tuple[int, ...], float], n_players: int, order: int
) -> dict[tuple[int, ...], float]:
    """Convert Fourier coefficients to BII (Banzhaf Interaction Index)."""
    bii_dict = defaultdict(float)
    for key, value in fourier_coeffs.items():
        weighted_coeff = value * (-2)
        bii_dict[key] = weighted_coeff
    return dict(bii_dict)


def fourier_to_fbii(
    fourier_coeffs: dict[tuple[int, ...], float],
    n_players: int,
    order: int,
) -> dict[tuple[int, ...], float]:
    fbii = defaultdict(float)
    for S, coeff in fourier_coeffs.items():
        S = tuple(sorted(S))
        s = len(S)
        # Only subsets T of S with |T| <= order receive contribution from F(S)
        for t in range(1, min(s, order) + 1):
            for T in combinations(S, t):
                fbii[T] += ((-2) ** t) * coeff
    return dict(fbii)


def proxyspex_extraction_sii(tree_dump: str, n_players: int, order: int) -> np.ndarray:
    """Extract SII using Fourier-based method (ProxySPEX)."""
    fourier_coeffs = _xgboost_to_fourier(tree_dump)
    return _fourier_to_sii(fourier_coeffs, n_players, order)


def proxyspex_extraction_bii(
    tree_dump: str, n_players: int, order: int
) -> dict[tuple[int, ...], float]:
    """Extract BII using Fourier-based method (ProxySPEX)."""
    fourier_coeffs = _xgboost_to_fourier(tree_dump)
    return _fourier_to_bii(fourier_coeffs, n_players, order)


def proxyspex_extraction_fsii(tree_dump: str, n_players: int, order: int) -> np.ndarray:
    """Extract FSII using Fourier-based method (ProxySPEX)."""
    fourier_coeffs = _xgboost_to_fourier(tree_dump)
    return _fourier_to_fsii(fourier_coeffs, n_players, order)

def proxyspex_extraction_fbii(
    tree_dump: str, n_players: int, order: int
) -> dict[tuple[int, ...], float]:
    """Extract FBII using Fourier-based method (ProxySPEX)."""
    fourier_coeffs = _xgboost_to_fourier(tree_dump)
    return fourier_to_fbii(fourier_coeffs, n_players, order)

def create_interventional_explainer(
    boolean_tree, n_players: int, index: str, order: int
) -> InterventionalTreeExplainer:
    """Create InterventionalTreeExplainer once (preprocessing not timed)."""
    return InterventionalTreeExplainer(
        model=boolean_tree,
        data=np.zeros((1, n_players)),
        index=index,
        max_order=order,
        bool_tree=True,
    )


def interventional_extraction(
    explainer: InterventionalTreeExplainer, n_players: int
) -> np.ndarray:
    """Extract interaction values using a pre-built explainer."""
    interaction_values = explainer.explain_function(np.ones((n_players)))
    return interaction_values.values


def load_interventional_datasets() -> dict[str, dict[str, Any]]:
    """Load all interventional datasets from config."""
    config_file = Path(
        "shapiq-benchmark/benchmarks/standard/configuration_interventional_standard_shapley_order1.json"
    )
    with open(config_file, "r") as f:
        config_order1 = json.load(f)

    config_file = Path(
        "shapiq-benchmark/benchmarks/standard/configuration_interventional_standard_shapley_order2.json"
    )
    with open(config_file, "r") as f:
        config_order2 = json.load(f)

    config_file = Path(
        "shapiq-benchmark/benchmarks/standard/configuration_interventional_standard_shapley_order3.json"
    )
    with open(config_file, "r") as f:
        config_order3 = json.load(f)

    datasets = {}
    for dataset_name in config_order2.keys():
        if dataset_name in DATASET_CLASSES:
            datasets[dataset_name] = {
                "order1": config_order1[dataset_name],
                "order2": config_order2[dataset_name],
                "order3": config_order3[dataset_name],
                "class": DATASET_CLASSES[dataset_name],
            }

    return datasets


def setup_interventional_game(
    dataset_class, x: int = 0
) -> tuple[InterventionalGame, int, np.ndarray]:
    """Setup interventional game for a dataset."""
    game = dataset_class(
        x=x,
        model_name="gradient_boosting",
        imputer="baseline",
        random_state=RANDOM_STATE,
    )
    model = game.setup.model
    x_train = game.setup.x_train
    x_explain = game.x

    interventional_game = InterventionalGame(
        model=model,
        reference_data=x_train[0:50],
        target_instance=x_explain,
    )

    n_players = x_train.shape[1]
    return interventional_game, n_players, x_train


def create_coalitions(
    interventional_game, n_players: int, sampling_budget: int = 10_000
) -> tuple[np.ndarray, np.ndarray]:
    """Create and evaluate coalitions using sampling."""
    sampler = CoalitionSampler(
        n_players=n_players,
        sampling_weights=np.ones(n_players + 1),
        pairing_trick=True,
        replacement=False,
        random_state=RANDOM_STATE,
    )
    sampler.sample(sampling_budget=sampling_budget)
    coalitions_matrix = sampler.coalitions_matrix
    values = interventional_game(coalitions_matrix)
    return coalitions_matrix, values


def benchmark_extraction(
    dataset_name: str,
    dataset_info: dict[str, Any],
    index: str,
    order: int,
) -> dict[int, dict[str, list[float]]]:
    """Benchmark extraction times for a single dataset and index."""
    print(f"\n{'=' * 70}")
    print(f"Benchmarking: {dataset_name} | Index: {index} | Order: {order}")
    print(f"{'=' * 70}")

    # Setup game
    try:
        interventional_game, n_players, x_train = setup_interventional_game(
            dataset_info["class"]
        )
    except Exception as e:
        print(f"Error setting up game for {dataset_name}: {e}")
        return {}

    # Create coalitions
    try:
        coalitions_matrix, values = create_coalitions(interventional_game, n_players)
    except Exception as e:
        print(f"Error creating coalitions for {dataset_name}: {e}")
        return {}

    results = {}

    depth_range = ORDER_TO_DEPTH_RANGE.get(order)
    if depth_range is None:
        print(f"Unsupported order for depth scheduling: {order}")
        return {}

    for max_depth in depth_range:
        print(f"  Depth {max_depth}/{max(depth_range)}: ", end="", flush=True)

        # Train model
        xgb_model = XGBRegressor(
            n_estimators=100,
            max_depth=max_depth,
            random_state=RANDOM_STATE,
            min_child_weight=0,
            reg_lambda=0.0,
            reg_alpha=0.0,
        )
        xgb_model.fit(coalitions_matrix, values)
        score = xgb_model.score(coalitions_matrix, values)

        results[max_depth] = {
            "fourier": [],
            "interventional": [],
            "r2_score": score,
        }

        # Benchmark Fourier extraction
        for run in range(NUM_BENCHMARK_RUNS):
            tree_dump = xgb_model.get_booster().get_dump(dump_format="json")
            start = time.perf_counter()
            if index == "SII":
                _ = proxyspex_extraction_sii(tree_dump, n_players, order)
            elif index == "BII":  # BII
                _ = proxyspex_extraction_bii(tree_dump, n_players, order)
            elif index == "FBII":  # FBII
                _ = proxyspex_extraction_fbii(tree_dump, n_players, order)
            elif index == "FSII":  # FSII
                _ = proxyspex_extraction_fsii(tree_dump, n_players, order)
            end = time.perf_counter()
            results[max_depth]["fourier"].append(end - start)

        # Build explainer once (outside timed extraction loop).
        explainer = create_interventional_explainer(xgb_model, n_players, index, order)

        # Benchmark Interventional extraction (extraction only)
        for run in range(NUM_BENCHMARK_RUNS):
            try:
                start = time.perf_counter()
                _ = interventional_extraction(explainer, n_players)
                end = time.perf_counter()
                results[max_depth]["interventional"].append(end - start)
            except Exception as e:
                print(
                    f"\nError during interventional extraction at depth {max_depth}: {e}"
                )
                results[max_depth]["interventional"].append(np.nan)

        fourier_mean = np.mean(results[max_depth]["fourier"])
        interventional_mean = np.mean(results[max_depth]["interventional"])
        print(
            f"✓ R²={score:.4f} | Fourier={fourier_mean:.4f}s | Interv={interventional_mean:.4f}s"
        )

    return results


def save_results_to_csv(
    all_results: dict[str, dict[str, dict[int, dict[int, dict[str, list[float]]]]]],
    raw_output_path: Path,
    summary_output_path: Path,
) -> None:
    """Save benchmark results to raw and aggregated CSV files."""
    raw_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for dataset_name, dataset_results in all_results.items():
        for index, order_results in dataset_results.items():
            for order, depth_results in order_results.items():
                for depth, times in depth_results.items():
                    fourier_times = times["fourier"]
                    interventional_times = times["interventional"]

                    fourier_mean = (
                        float(np.nanmean(fourier_times)) if fourier_times else np.nan
                    )
                    interventional_mean = (
                        float(np.nanmean(interventional_times))
                        if interventional_times
                        else np.nan
                    )
                    relative_runtime = (
                        interventional_mean / fourier_mean
                        if not np.isnan(fourier_mean) and fourier_mean > 0
                        else np.nan
                    )

                    summary_rows.append(
                        {
                            "dataset": dataset_name,
                            "index": index,
                            "order": order,
                            "depth": depth,
                            "r2_score": times["r2_score"],
                            "fourier_time_mean": fourier_mean,
                            "interventional_time_mean": interventional_mean,
                            "relative_runtime": relative_runtime,
                        }
                    )

                    for run_id, run_time in enumerate(fourier_times, start=1):
                        raw_rows.append(
                            {
                                "dataset": dataset_name,
                                "index": index,
                                "order": order,
                                "depth": depth,
                                "method": "fourier",
                                "run": run_id,
                                "time_seconds": run_time,
                            }
                        )

                    for run_id, run_time in enumerate(interventional_times, start=1):
                        raw_rows.append(
                            {
                                "dataset": dataset_name,
                                "index": index,
                                "order": order,
                                "depth": depth,
                                "method": "interventional",
                                "run": run_id,
                                "time_seconds": run_time,
                            }
                        )

    raw_output_path.parent.mkdir(parents=True, exist_ok=True)

    with raw_output_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(
            csvfile,
            fieldnames=[
                "dataset",
                "index",
                "order",
                "depth",
                "method",
                "run",
                "time_seconds",
            ],
        )
        writer.writeheader()
        writer.writerows(raw_rows)

    with summary_output_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(
            csvfile,
            fieldnames=[
                "dataset",
                "index",
                "order",
                "depth",
                "r2_score",
                "fourier_time_mean",
                "interventional_time_mean",
                "relative_runtime",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"\n✓ Saved raw results: {raw_output_path}")
    print(f"✓ Saved summary results: {summary_output_path}")


def main():
    """Main execution function."""
    print("Loading datasets...")
    datasets = load_interventional_datasets()
    print(f"Loaded {len(datasets)} datasets")

    all_results = {name: {} for name in datasets.keys()}

    # Benchmark each dataset for SII and BII at orders 2 and 3
    for dataset_name, dataset_info in datasets.items():
        print(f"\n{'#' * 70}")
        print(f"# Dataset: {dataset_name}")
        print(f"{'#' * 70}")

        all_results[dataset_name]["SII"] = {}
        all_results[dataset_name]["BII"] = {}
        all_results[dataset_name]["FBII"] = {}
        all_results[dataset_name]["FSII"] = {}

        for order in [1, 2, 3]:
            # SII benchmarks
            all_results[dataset_name]["SII"][order] = benchmark_extraction(
                dataset_name, dataset_info, "SII", order
            )

            # BII benchmarks
            all_results[dataset_name]["BII"][order] = benchmark_extraction(
                dataset_name, dataset_info, "BII", order
            )
            
            # FBII benchmarks
            all_results[dataset_name]["FBII"][order] = benchmark_extraction(
                dataset_name, dataset_info, "FBII", order
            )
            
            # FSII benchmarks
            all_results[dataset_name]["FSII"][order] = benchmark_extraction(
                dataset_name, dataset_info, "FSII", order
            )

    # Save benchmark results for separate plotting script.
    print(f"\n{'=' * 70}")
    print("Saving CSV results...")
    print(f"{'=' * 70}")
    save_results_to_csv(
        all_results=all_results,
        raw_output_path=Path("results/extraction_times_raw.csv"),
        summary_output_path=Path("results/extraction_times_summary.csv"),
    )


if __name__ == "__main__":
    main()
