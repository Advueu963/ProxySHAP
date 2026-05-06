from __future__ import annotations

import argparse
import ast
from pathlib import Path
from typing import Dict, Iterable

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import pandas as pd
import numpy as np

try:
    from shapiq_benchmark.plot import STYLE_DICT  # type: ignore
except ImportError:  # pragma: no cover - fallback for standalone usage
    STYLE_DICT: Dict[str, Dict[str, object]] = {}

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


def parse_kwargs(raw: object) -> Dict[str, float]:
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            return {}
        if isinstance(parsed, dict):
            return parsed
    return {}


COLOR_MAP = plt.get_cmap("tab20")
MARKERS = ["o", "s", "^", "D", "P", "X", "*", "h", "v", "<", ">", "p"]


def create_style_map(approximators: Iterable[str]) -> Dict[str, Dict[str, object]]:
    """Build deterministic style entries, preferring the shared STYLE_DICT."""

    approx_list = sorted({str(approx) for approx in approximators})
    style_map: Dict[str, Dict[str, object]] = {}
    color_index = 0
    marker_index = 0

    for approx in approx_list:
        base_style = STYLE_DICT.get(approx, {})
        color = base_style.get("color")
        marker = base_style.get("marker")
        linestyle = base_style.get("linestyle", "-")

        if color is None:
            color = COLOR_MAP(color_index % COLOR_MAP.N)
            color_index += 1
        if marker is None:
            marker = MARKERS[marker_index % len(MARKERS)]
            marker_index += 1

        style_map[approx] = {
            "color": color,
            "linestyle": linestyle,
            "marker": marker,
        }

    return style_map


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot runtime vs budget curves for each game."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("results_benchmark_BII_2.csv"),
        help="Benchmark CSV file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("plots/runtime_vs_budget"),
        help="Directory the figures are written to.",
    )
    parser.add_argument(
        "--logx",
        action="store_true",
        help="Log scale for the budget axis.",
    )
    parser.add_argument(
        "--logy",
        action="store_true",
        help="Log scale for the runtime axis.",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    if "kwargs" not in df.columns:
        raise ValueError("CSV must contain a 'kwargs' column with runtime metadata.")

    df = df[df["approximator"] != "RegressionMSRIQ-DT"].copy()
    df["kwargs_dict"] = df["kwargs"].apply(parse_kwargs)
    if "total_runtime" not in df.columns:
        df["total_runtime"] = df["kwargs_dict"].apply(lambda d: d.get("total_runtime"))
    df = df.dropna(subset=["total_runtime", "used_budget"])

    style_map = create_style_map(df["approximator"].unique())

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for game, game_df in df.groupby("game"):
        if game_df.empty:
            continue

        fig, ax = plt.subplots(figsize=(7, 5))

        stats_per_approx: Dict[str, pd.DataFrame] = {}
        eval_samples: Dict[float, list[float]] = {}

        for approximator, group in game_df.groupby("approximator"):
            group_sorted = group.sort_values("used_budget")
            if group_sorted.empty:
                continue

            stats = (
                group_sorted.groupby("used_budget")["total_runtime"]
                .agg(["mean", "std", "count"])
                .reset_index()
                .sort_values("used_budget")
            )
            budgets = stats["used_budget"].to_numpy()
            runtimes = stats["mean"].to_numpy()
            stds = stats["std"].fillna(0.0).to_numpy()

            approx_key = str(approximator)
            if approx_key not in style_map:
                style_map.update(create_style_map([approx_key]))
            style = style_map[approx_key]
            color = style.get("color")
            linestyle = style.get("linestyle", "-")
            marker = style.get("marker")

            ax.plot(
                budgets,
                runtimes,
                label=approximator,
                color=color,
                linestyle=linestyle,
                marker=marker,
                linewidth=1.6,
                markersize=5,
            )

            if np.any(stds > 0):
                lower = runtimes - stds
                upper = runtimes + stds
                if args.logy:
                    lower = np.maximum(lower, 1e-12)
                ax.fill_between(
                    budgets,
                    lower,
                    upper,
                    color=color,
                    alpha=0.2,
                    linewidth=0,
                )

            stats_per_approx[str(approximator)] = stats
            eval_series = group_sorted["kwargs_dict"].apply(
                lambda d: d.get("evaluations") if isinstance(d, dict) else None
            )
            for budget, eval_value in zip(
                group_sorted["used_budget"], eval_series, strict=False
            ):
                if pd.notna(eval_value):
                    eval_samples.setdefault(float(budget), []).append(float(eval_value))

        budgets_unique = sorted(game_df["used_budget"].unique())
        n_players = None
        if "n_players" in game_df:
            n_players_series = game_df["n_players"].dropna()
            if not n_players_series.empty:
                n_players = int(n_players_series.iloc[0])

        if eval_samples:
            eval_stats_df = pd.DataFrame(
                {
                    "used_budget": sorted(eval_samples.keys()),
                    "mean": [
                        np.mean(eval_samples[b]) for b in sorted(eval_samples.keys())
                    ],
                    "std": [
                        np.std(eval_samples[b], ddof=0)
                        for b in sorted(eval_samples.keys())
                    ],
                }
            )

            if args.logy:
                eval_stats_df["mean"] = eval_stats_df["mean"].clip(lower=1e-12)
                eval_stats_df["std"] = np.minimum(
                    eval_stats_df["std"], eval_stats_df["mean"] * 0.999
                )

            eval_mean = eval_stats_df["mean"].to_numpy()
            eval_std = eval_stats_df["std"].to_numpy()
            eval_lower = eval_mean - eval_std
            eval_upper = eval_mean + eval_std
            if args.logy:
                eval_lower = np.maximum(eval_lower, 1e-12)

            ax.plot(
                eval_stats_df["used_budget"],
                eval_mean,
                color="#888888",
                linestyle="--",
                linewidth=1.4,
                marker=None,
                label="Evaluations (mean)",
            )
            ax.fill_between(
                eval_stats_df["used_budget"],
                eval_lower,
                eval_upper,
                color="#b0b0b0",
                alpha=0.25,
                linewidth=0,
                label="Evaluations (std)",
            )

        if args.logx:
            ax.set_xscale("log")
        if args.logy:
            ax.set_yscale("log")

        ax.set_xlabel("Budget")
        ylabel = "Runtime [s]"
        if args.logy:
            ylabel += " (log scale)"
        ax.set_ylabel(ylabel)
        game_key = str(game)
        title = DATA_NAMES.get(game_key, game_key)
        ax.set_title(f"Runtime vs Budget – {title}")
        #ax.legend(show=False)
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)

        if args.logy:
            ymin, ymax = ax.get_ylim()
            eval_min = None
            if eval_samples:
                eval_min = min(
                    value
                    for values in eval_samples.values()
                    for value in values
                    if value > 0
                )
            if eval_min:
                target_min = max(eval_min / 5, ymin)
                if target_min < ymax:
                    ax.set_ylim(bottom=target_min)

        max_budget = int(max(budgets_unique)) if budgets_unique else 0
        if n_players is not None and max_budget > 0:
            _set_x_axis_ticks(ax, n_players, max_budget)
        else:
            ax.set_xticks(budgets_unique)
            ax.set_xticklabels(
                [str(int(b)) for b in budgets_unique], rotation=0, ha="center"
            )
        fig.tight_layout()

        output_path = args.output_dir / f"{game}_runtime_vs_budget.png"
        fig.savefig(output_path, dpi=300)
        plt.close(fig)


def _set_x_axis_ticks(ax: Axes, n_players: int, max_budget: int) -> None:
    """Set x-axis ticks at 25% intervals with percent annotations."""

    if n_players <= 16:
        budgets_relative = np.arange(0, 1.25, 0.25)
        budgets = budgets_relative * (2**n_players)
    else:
        budgets = ax.get_xticks()
        budgets = budgets[budgets >= 0]
        budgets = budgets[budgets <= max_budget * 1.05]
        budgets_relative = budgets / (2**n_players) if n_players < 20 else np.zeros_like(budgets)

    xtick_labels = []
    for bdgt, bdgt_rel in zip(budgets, budgets_relative, strict=False):
        bdgt_rel_str = f"{bdgt_rel:.0%}"
        if bdgt_rel <= 0.01 and bdgt_rel != 0:
            bdgt_rel_str = "<1%"
        if bdgt_rel == 0:
            xtick_labels.append("0")
        else:
            xtick_labels.append(f"{int(bdgt)}\n({bdgt_rel_str})")

    ax.set_xticks(budgets)
    ax.set_xticklabels(xtick_labels)


if __name__ == "__main__":
    main()
