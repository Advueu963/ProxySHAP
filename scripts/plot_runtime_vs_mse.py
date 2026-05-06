from __future__ import annotations

import argparse
import ast
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
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


DEFAULT_MARKERS = ["o", "s", "^", "D", "P", "X", "*", "h", "v", "<", ">", "p"]


def parse_kwargs(raw: object) -> Dict[str, float]:
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            return {}
        if isinstance(parsed, dict):
            return parsed
    return {}


def build_color_map(approximators: Iterable[str]) -> Dict[str, object]:
    approx_list = sorted({str(a) for a in approximators})
    colors: Dict[str, object] = {}
    fallback_index = 0

    for approx in approx_list:
        style = STYLE_DICT.get(approx)
        colors[approx] = style["color"]

    return colors


def build_marker_map(games: Iterable[str]) -> Dict[str, str]:
    game_list = sorted({str(g) for g in games})
    return {
        game: DEFAULT_MARKERS[idx % len(DEFAULT_MARKERS)]
        for idx, game in enumerate(game_list)
    }


def _first_non_null(series: pd.Series) -> Optional[object]:
    non_null = series.dropna()
    if non_null.empty:
        return None
    return non_null.iloc[0]


def _cross(o: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    return float((a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]))


def convex_hull(points: np.ndarray) -> np.ndarray:
    """Compute the convex hull of 2D points via monotone chain.

    Returns hull vertices in counterclockwise order. For < 3 unique points,
    returns the unique points.
    """

    if points.size == 0:
        return points
    pts = np.asarray(points, dtype=float)
    pts = np.unique(pts, axis=0)
    if pts.shape[0] < 3:
        return pts

    order = np.lexsort((pts[:, 1], pts[:, 0]))
    pts = pts[order]

    lower: List[np.ndarray] = []
    for p in pts:
        while len(lower) >= 2 and _cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    upper: List[np.ndarray] = []
    for p in pts[::-1]:
        while len(upper) >= 2 and _cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    hull = np.vstack((lower[:-1], upper[:-1]))
    return hull


def _flat_span(value: float, *, log_space: bool) -> float:
    """Half-thickness for flat-line rectangles.

    In log-space we use a fixed thickness in decades.
    In linear space we use a small relative thickness.
    """

    if log_space:
        return 0.5  # decades
    base = abs(value)
    return max(base * 0.02, 1e-12)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Plot runtime vs MSE at the maximum budget for each (dataset, approximator). "
            "Color encodes approximator; marker encodes dataset."
        )
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Benchmark CSV file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("plots/runtime_vs_mse.png"),
        help="Destination for the generated figure.",
    )
    parser.add_argument(
        "--budget-column",
        type=str,
        default="used_budget",
        help="Budget column used to pick the maximum budget (default: used_budget).",
    )
    parser.add_argument(
        "--budget-quantile",
        type=float,
        default=1.0,
        help=(
            "Quantile p in (0, 1] used to select the budget per (dataset, approximator). "
            "The plot uses the available budget value closest to the p-quantile. "
            "Use 1.0 for the maximum budget (default: 1.0)."
        ),
    )
    parser.add_argument(
        "--runtime-key",
        type=str,
        default="total_runtime",
        help=(
            "Runtime key to read from kwargs dict (default: total_runtime). "
            "Only used if the runtime column does not already exist."
        ),
    )
    parser.add_argument(
        "--without-game-call",
        action="store_true",
        help=(
            "If set, exclude data points where the 'game_call' in kwargs matches this value. "
            "Useful to filter out failed runs."
        ),
    )
    parser.add_argument(
        "--logx",
        action="store_true",
        help="Log scale for the runtime axis.",
    )
    parser.add_argument(
        "--logy",
        action="store_true",
        help="Log scale for the MSE axis.",
    )
    parser.add_argument(
        "--mse-floor",
        type=float,
        default=1e-7,
        help="Minimum MSE value used when --logy is set (default: 1e-7).",
    )
    parser.add_argument(
        "--exclude-approximator",
        action="append",
        default=["RegressionMSRIQ-DT"],
        help="Approximator name(s) to exclude (can be repeated).",
    )
    parser.add_argument(
        "--hull",
        dest="hull",
        action="store_true",
        default=True,
        help="Draw a convex-hull polygon per approximator (default: enabled).",
    )
    parser.add_argument(
        "--no-hull",
        dest="hull",
        action="store_false",
        help="Disable convex-hull polygons.",
    )
    parser.add_argument(
        "--hull-alpha",
        type=float,
        default=0.12,
        help="Fill opacity for hull polygons (default: 0.12).",
    )
    parser.add_argument(
        "--hull-line-alpha",
        type=float,
        default=0.4,
        help="Line opacity for hull outlines (default: 0.4).",
    )
    parser.add_argument(
        "--highlight-topk-players",
        type=int,
        default=0,
        help=(
            "Highlight only the K datasets with the most players (n_players) using full alpha; "
            "all other datasets are faded. Use 0 to disable (default: 0)."
        ),
    )
    parser.add_argument(
        "--highlight-alpha",
        type=float,
        default=0.9,
        help="Alpha used for highlighted datasets (default: 0.9).",
    )
    parser.add_argument(
        "--other-alpha",
        type=float,
        default=0.2,
        help="Alpha used for non-highlighted datasets (default: 0.2).",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    if "kwargs" not in df.columns:
        raise ValueError("CSV must contain a 'kwargs' column with runtime metadata.")
    if "MSE" not in df.columns:
        raise ValueError("CSV must contain an 'MSE' column.")
    if "game" not in df.columns or "approximator" not in df.columns:
        raise ValueError("CSV must contain 'game' and 'approximator' columns.")
    if args.budget_column not in df.columns:
        raise ValueError(
            f"CSV must contain the selected budget column '{args.budget_column}'."
        )

    if args.exclude_approximator:
        df = df[~df["approximator"].isin(args.exclude_approximator)].copy()

    df["kwargs_dict"] = df["kwargs"].apply(parse_kwargs)
    if "total_runtime" not in df.columns:
        df["total_runtime"] = df["kwargs_dict"].apply(
            lambda d: d.get(args.runtime_key) if isinstance(d, dict) else None
        )
        if args.without_game_call:
            df["total_runtime"] = df["total_runtime"] - df["kwargs_dict"].apply(
                lambda d: d.get("evaluations") if isinstance(d, dict) else None
            )

    df = df.dropna(subset=["total_runtime", "MSE", args.budget_column]).copy()
    df["total_runtime"] = pd.to_numeric(df["total_runtime"], errors="coerce")
    df["MSE"] = pd.to_numeric(df["MSE"], errors="coerce")
    df = df.dropna(subset=["total_runtime", "MSE"]).copy()

    if not (0.0 < float(args.budget_quantile) <= 1.0):
        raise ValueError("--budget-quantile must be in (0, 1].")

    if args.logy:
        df["MSE"] = df["MSE"].clip(lower=args.mse_floor)

    # Extract the budget quantile for each game based on those of ProxySpex
    proxy_spex_df = df[df["approximator"] == "ProxySpex"]
    game_to_proxy_spex_budget = {}
    for game, group in proxy_spex_df.groupby("game", dropna=False):
        budgets = pd.to_numeric(group[args.budget_column], errors="coerce")
        budgets = budgets.dropna()
        if budgets.empty:
            continue
        max_budget = float(budgets.max())
        if float(args.budget_quantile) >= 1.0:
            selected_budget = float(budgets.max())
        else:
            diffs = (budgets / max_budget - args.budget_quantile).abs()
            min_diff = float(diffs.min())
            candidate_budgets = budgets[diffs == min_diff]
            selected_budget = float(candidate_budgets.min())
        game_to_proxy_spex_budget[str(game)] = float(selected_budget)
        print("Selected budget for game", game, "is", selected_budget)

    # Extract per game quantile budget data points.
    game_to_used_budget: dict[str, float] = {}
    for game, group in df.groupby("game", dropna=False):
        budgets = pd.to_numeric(group[args.budget_column], errors="coerce")
        budgets = budgets.dropna()

        max_budget = float(budgets.max())

        if budgets.empty:
            continue

        selected_budget = game_to_proxy_spex_budget.get(str(game))
        if selected_budget is None:
            raise ValueError(
                f"No ProxySpex data for game {game}; cannot determine budget quantile."
            )

        game_to_used_budget[str(game)] = float(selected_budget)

    records: List[dict] = []
    for (game, approximator), group in df.groupby(
        ["game", "approximator"], dropna=False
    ):
        if group.empty:
            continue

        selected_budget = game_to_used_budget.get(str(game))
        if selected_budget is None:
            continue

        selected_rows = group[group[args.budget_column] == selected_budget]
        if selected_rows.empty:
            # Fallback for float/int mismatch: use numeric comparison.
            selected_rows = group[
                pd.to_numeric(group[args.budget_column], errors="coerce")
                == selected_budget
            ]
        if selected_rows.empty:
            continue

        records.append(
            {
                "game": str(game),
                "approximator": str(approximator),
                "selected_budget": float(selected_budget),
                "budget_quantile": float(args.budget_quantile),
                "runtime_mean": float(selected_rows["total_runtime"].mean()),
                "mse_mean": float(selected_rows["MSE"].mean()),
                "budget": selected_budget,
                "n": int(len(selected_rows)),
            }
        )

    if not records:
        raise ValueError("No data points found after filtering/aggregation.")

    agg = pd.DataFrame.from_records(records)

    dataset_players: Dict[str, float] = {}
    if "n_players" in df.columns:
        tmp = df[["game", "n_players"]].dropna()
        if not tmp.empty:
            tmp["n_players"] = pd.to_numeric(tmp["n_players"], errors="coerce")
            tmp = tmp.dropna(subset=["n_players"])
            if not tmp.empty:
                dataset_players = (
                    tmp.groupby("game")["n_players"].max().to_dict()  # type: ignore[assignment]
                )
                agg["n_players"] = agg["game"].map(dataset_players)

    highlight_games: set[str] = set()
    if (
        args.highlight_topk_players
        and args.highlight_topk_players > 0
        and dataset_players
    ):
        ordered = sorted(
            dataset_players.items(), key=lambda kv: (-float(kv[1]), str(kv[0]))
        )
        highlight_games = {
            name for name, _ in ordered[: int(args.highlight_topk_players)]
        }

    if args.logx:
        agg["runtime_mean"] = agg["runtime_mean"].clip(lower=1e-12)

    colors = build_color_map(agg["approximator"].unique())
    markers = build_marker_map(agg["game"].unique())

    fig, ax = plt.subplots(figsize=(12, 6.5))

    if args.hull:
        for approximator, group in agg.groupby("approximator", dropna=False):
            pts = group[["runtime_mean", "mse_mean"]].to_numpy(dtype=float)
            if pts.shape[0] < 3:
                continue

            pts_work = pts.copy()
            if args.logx:
                pts_work[:, 0] = np.log10(np.maximum(pts_work[:, 0], 1e-300))
            if args.logy:
                pts_work[:, 1] = np.log10(np.maximum(pts_work[:, 1], 1e-300))

            hull = convex_hull(pts_work)
            if hull.shape[0] < 3:
                # Degenerate case: points lie on a line in plotting space.
                # Common case: all MSE are identical across datasets -> draw a thin rectangle
                # spanning min/max runtime so the region is still visible.
                x_vals = pts_work[:, 0]
                y_vals = pts_work[:, 1]
                x_min = float(np.min(x_vals))
                x_max = float(np.max(x_vals))
                y_min = float(np.min(y_vals))
                y_max = float(np.max(y_vals))

                # Treat as flat if the range is effectively zero.
                flat_x = np.isclose(x_min, x_max, rtol=0.0, atol=1e-12)
                flat_y = np.isclose(y_min, y_max, rtol=0.0, atol=1e-12)
                if not (flat_x or flat_y):
                    continue

                if flat_y and not flat_x:
                    y_center = y_min
                    span = _flat_span(y_center, log_space=args.logy)
                    rect = np.array(
                        [
                            [x_min, y_center - span],
                            [x_max, y_center - span],
                            [x_max, y_center + span],
                            [x_min, y_center + span],
                        ],
                        dtype=float,
                    )
                elif flat_x and not flat_y:
                    x_center = x_min
                    span = _flat_span(x_center, log_space=args.logx)
                    rect = np.array(
                        [
                            [x_center - span, y_min],
                            [x_center + span, y_min],
                            [x_center + span, y_max],
                            [x_center - span, y_max],
                        ],
                        dtype=float,
                    )
                else:
                    # All points identical in plotting space; nothing meaningful to draw.
                    continue

                hull = rect

            if args.logx:
                hull_x = np.power(10.0, hull[:, 0])
            else:
                hull_x = hull[:, 0]
            if args.logy:
                hull_y = np.power(10.0, hull[:, 1])
            else:
                hull_y = hull[:, 1]

            hull_x = np.append(hull_x, hull_x[0])
            hull_y = np.append(hull_y, hull_y[0])

            color = colors[str(approximator)]
            ax.fill(
                hull_x,
                hull_y,
                color=color,
                alpha=float(args.hull_alpha),
                linewidth=0,
                zorder=1,
            )
            ax.plot(
                hull_x,
                hull_y,
                color=color,
                alpha=float(args.hull_line_alpha),
                linewidth=1.2,
                zorder=2,
            )

    for _, row in agg.iterrows():
        approx = str(row["approximator"])
        game = str(row["game"])
        if highlight_games:
            point_alpha = (
                float(args.highlight_alpha)
                if game in highlight_games
                else float(args.other_alpha)
            )
        else:
            point_alpha = float(args.highlight_alpha)
        ax.scatter(
            row["runtime_mean"],
            row["mse_mean"],
            c=[colors[approx]],
            marker=markers[game],
            s=70,
            alpha=point_alpha,
            edgecolors="black",
            linewidths=0.4,
            zorder=3,
        )

    if args.logx:
        ax.set_xscale("log")
    if args.logy:
        ax.set_yscale("log")

    ax.set_xlabel("Runtime [s]")
    ax.set_ylabel("MSE")
    game_type = None
    if "game_type" in df.columns:
        game_type = _first_non_null(df["game_type"])
    title_suffix = f" ({game_type})" if game_type else ""
    if float(args.budget_quantile) >= 1.0:
        budget_label = "maximum budget"
    else:
        budget_label = f"budget quantile p={float(args.budget_quantile):.2f}"
    ax.set_title(f"Runtime vs MSE at {budget_label}{title_suffix}")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)

    approx_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markerfacecolor=colors[a],
            markeredgecolor="black",
            markeredgewidth=0.4,
            markersize=8,
            label=a,
        )
        for a in sorted(colors.keys())
    ]
    dataset_handles = [
        Line2D(
            [0],
            [0],
            marker=markers[g],
            linestyle="None",
            color="black",
            markerfacecolor="white",
            markeredgecolor="black",
            markersize=8,
            label=DATA_NAMES.get(g, g),
        )
        for g in sorted(markers.keys())
    ]

    legend1 = ax.legend(
        handles=approx_handles,
        title="Approximator (color)",
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
        borderaxespad=0.0,
    )
    ax.add_artist(legend1)
    ax.legend(
        handles=dataset_handles,
        title="Dataset (marker)",
        loc="lower left",
        bbox_to_anchor=(1.02, 0),
        borderaxespad=0.0,
    )

    # Leave room on the right for long legend entries.
    fig.tight_layout(rect=(0, 0, 0.72, 1))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
