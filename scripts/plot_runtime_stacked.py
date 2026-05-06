from __future__ import annotations

import argparse
import ast
import math
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from shapiq_benchmark.plot import STYLE_DICT

DATA_NAMES: Dict[str, str] = {
    "BreastCancerLocalXAI": "Cancer ($d=30$)",
    "CommunitiesAndCrimeLocalXAI": "Crime ($d=101$)",
    "Corrgroups60LocalXAI": "CG60 ($d=60$)",
    "ForestFiresLocalXAI": "Forest ($d=13$)",
    "IndependentLinear60LocalXAI": "IL60 ($d=60$)",
    "NHANESILocalXAI": "NHANES ($d=79$)",
    "RealEstateLocalXAI": "Estate ($d=15$)",
    "wine_quality": "Wine ($d=11$)",
    "AdultCensusLocalXAI": "Adult ($d=14$)",
    "CaliforniaHousingLocalXAI": "Housing ($d=8$)",
    "BikeSharingLocalXAI": "Bike ($d=12$)",
    "ViT4by4Patches": "ViT16 ($d=16$)",
    "ViT3by3Patches": "ViT9 ($d=9$)",
    "ResNet18w14Superpixel": "ResNet18 ($d=14$)",
    "SentimentIMDBDistilBERT14": "DistilBERT ($d=14$)",
    "SOUM": "SOUM",
}

DEFAULT_COLORMAP = plt.get_cmap("tab20")


def parse_kwargs(raw: object) -> Dict[str, float]:
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            return {}
        if isinstance(parsed, dict):
            return parsed
    return {}


def target_budget(n_players: float) -> int:
    n = int(n_players)
    return min(int(math.pow(2, n)), 20_000)


def select_rows(df: pd.DataFrame) -> pd.DataFrame:
    records: List[pd.Series] = []
    for (game, approximator), group in df.groupby(["game", "approximator"]):
        if group.empty:
            continue
        target = target_budget(group["n_players"].iloc[0])
        match = group[group["used_budget"] == target]
        if match.empty:
            idx = group["used_budget"].idxmax()
            match = group.loc[[idx]]
        records.append(match.iloc[0])
    if not records:
        raise ValueError("No suitable rows found for the stacked bar plot.")
    return pd.DataFrame(records)


def extract_evaluation_baseline(df: pd.DataFrame) -> Dict[str, float]:
    """Collect shared evaluation times from RegressionMSRIQ rows per game."""

    eval_times: Dict[str, float] = {}
    for (game, approx), group in df.groupby(["game", "approximator"]):
        kwargs_dicts = group.get("kwargs_dict")
        if kwargs_dicts is None:
            continue
        for kwargs_dict in kwargs_dicts:
            if isinstance(kwargs_dict, dict) and "evaluations" in kwargs_dict:
                eval_times[game] = float(max(kwargs_dict["evaluations"], 0.0))
                break
    return eval_times


def build_color_map(approximators: Iterable[str]) -> Dict[str, object]:
    """Assign colors, preferring STYLE_DICT definitions."""

    approx_list = sorted(set(approximators))
    colors: Dict[str, object] = {}
    fallback_index = 0

    for approx in approx_list:
        style = STYLE_DICT.get(approx)
        color = None
        if style is not None:
            color = style.get("color")
        if color is None:
            color = DEFAULT_COLORMAP(fallback_index % DEFAULT_COLORMAP.N)
            fallback_index += 1
        colors[approx] = color

    return colors


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot stacked runtime bars per dataset."
    )
    parser.add_argument(
        "--input",
        type=Path,
        help="Benchmark CSV file.",
        default=Path("results_benchmark_BII_2.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("plots/runtime_stacked.png"),
        help="Destination for the generated figure.",
    )
    parser.add_argument(
        "--sort-datasets",
        choices=["alphabetical", "runtime"],
        default="alphabetical",
        help="Ordering of datasets on the x-axis (default: alphabetical).",
    )
    parser.add_argument(
        "--logy",
        action="store_true",
        help="Use logarithmic scale for runtimes.",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    df = df[df["approximator"] != "RegressionMSRIQ-DT"].copy()
    if "kwargs" not in df.columns:
        raise ValueError(
            "CSV file must contain a 'kwargs' column with runtime metadata."
        )

    df["kwargs_dict"] = df["kwargs"].apply(parse_kwargs)
    if "total_runtime" not in df.columns:
        df["total_runtime"] = df["kwargs_dict"].apply(lambda d: d.get("total_runtime"))

    df = df.dropna(subset=["total_runtime"]).copy()

    selected = select_rows(df)

    eval_baseline = extract_evaluation_baseline(selected)

    selected = selected.copy()
    selected["pure_runtime"] = selected.apply(
        lambda row: max(
            (
                row["total_runtime"]
                - eval_baseline.get(row["game"], 0.0)
                # if row["
                # else row["total_runtime"]
            ),
            0.0,
        ),
        axis=1,
    )

    pivot = selected.pivot_table(
        index="game",
        columns="approximator",
        values="pure_runtime",
        aggfunc="mean",
        fill_value=-1,
    )

    games = pivot.index.tolist()
    approx_all = list(pivot.columns)

    if args.sort_datasets == "runtime":
        runtime_sums = pivot.replace(-1, np.nan).sum(axis=1).fillna(np.inf).to_numpy()
        ordering = np.argsort(runtime_sums)
        games = [games[i] for i in ordering]
    else:
        games.sort()

    colors = build_color_map(approx_all)

    x = np.arange(len(games))
    width = 0.6
    fig, ax = plt.subplots(figsize=(max(10, len(games) * 0.8), 6))

    plotted_labels: set[str] = set()
    for idx, game in enumerate(games):
        series = pivot.loc[game].replace(-1, np.nan).dropna()
        if series.empty and eval_baseline.get(game, 0.0) == 0:
            continue

        series = series.sort_values()

        bottom = 0.0
        eval_height = eval_baseline.get(game, 0.0)
        if eval_height > 0:
            ax.bar(
                [x[idx]],
                [eval_height],
                width,
                bottom=[bottom],
                color="#bbbbbb",
                label=(
                    "Evaluations"
                    if "Evaluations" not in plotted_labels
                    else "_nolegend_"
                ),
            )
            plotted_labels.add("Evaluations")
            bottom += eval_height

        for approx, height in series.items():
            color = colors[approx]
            label = approx if approx not in plotted_labels else "_nolegend_"
            ax.bar(
                [x[idx]],
                [height],
                width,
                bottom=[bottom],
                color=color,
                label=label,
            )
            plotted_labels.add(approx)
            bottom += float(height)

    ax.set_xticks(x)
    ax.set_xticklabels(
        [str(DATA_NAMES.get(game, game)) for game in games], rotation=45, ha="right"
    )
    ax.set_ylabel("Total runtime [s] (log scale)" if args.logy else "Total runtime [s]")
    ax.set_xlabel("Dataset")
    ax.set_title("Runtime per dataset at maximum budget")
    if args.logy:
        ax.set_yscale("log")
    ax.legend(title="Approximator", bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.margins(x=0.02)
    fig.tight_layout()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
