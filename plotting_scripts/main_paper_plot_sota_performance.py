"""Plot win-rate vs. budget across all datasets for a SOTA comparison.

For each dataset d and budget b, the winning method is the one with lowest MSE.
The win rate of method m at budget b is:

    WinRate_m(b) = #{d : b in B_d and m = winner(d, b)} / #{d : b in B_d}

Since budget grids differ across datasets, each point aggregates only over
datasets evaluated at that exact budget; the denominator N_b is annotated
below the x-axis.

Use --game-type merged to merge interventional, exhaustive, and tabpfn results
into a single view; dataset names are suffixed with (INT), (EXT), (TABPFN).

Example:
    uv run python special_plot_scripts/main_paper_plot_sota_performance.py \\
        --index SII --order 2 --game-type interventional

    uv run python special_plot_scripts/main_paper_plot_sota_performance.py \\
        --index SII --order 2 --game-type merged
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import transforms
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, NullFormatter
import numpy as np
import pandas as pd

from _plot_style import apply_tick_style, setup_fonts, W_REGULAR, W_SEMIBOLD

setup_fonts()

APPROXIMATOR_RENAMING: dict[str, str] = {
    "RegressionMSRIQ-NoAdjustment":             "ProxySHAP (XGBoost) [our]",
    "ProxySHAP (XGBoost)":                      "ProxySHAP (XGBoost) [our]",
    "RegressionMSRIQ":                           "ProxySHAP (XGBoost, MSR) [our]",
    "ProxySHAP (XGBoost, MSR)":                 "ProxySHAP (XGBoost, MSR) [our]",
    "ProxySPEX (XGBoost)":                       "ProxySPEX (XGBoost)",
    "ProxySPEX (XGBoost, NoRefinement)":         "ProxySPEX (XGBoost, NoRef.)",
    "ProxySPEX (XGBoost, NoTruncation, NoRefinement)": "ProxySPEX (XGBoost, NoTrunc.)",
}

APPROXIMATORS_TO_PLOT: list[str] = [
    "ProxySHAP (XGBoost) [our]",
    "ProxySHAP (XGBoost, MSR) [our]",
    "ProxySHAP (Linear) [our]",
    "ProxySHAP (Linear, MSR) [our]",
    "KernelSHAPIQ",
    "ProxySPEX (XGBoost)",
]

STYLE: dict[str, dict] = {
    "ProxySHAP (XGBoost) [our]":      {"color": "#61abec", "marker": "o", "linestyle": "-",  "linewidth": 1.5, "markersize": 4.0, "zorder": 5},
    "ProxySHAP (XGBoost, MSR) [our]": {"color": "#1e88e5", "marker": "s", "linestyle": "-",  "linewidth": 1.5, "markersize": 4.0, "zorder": 4},
    "ProxySHAP (Linear) [our]":      {"color": "#15B01A", "marker": "o", "linestyle": "-",  "linewidth": 1.5, "markersize": 4.0, "zorder": 3},
    "ProxySHAP (Linear, MSR) [our]": {"color": "#15B01A", "marker": "s", "linestyle": "-",  "linewidth": 1.5, "markersize": 4.0, "zorder": 2},
    "KernelSHAPIQ":                    {"color": "#ff6f00", "marker": "^", "linestyle": "--", "linewidth": 1.5, "markersize": 4.0, "zorder": 3},
    "ProxySPEX (XGBoost)":             {"color": "#ef27a6", "marker": "D", "linestyle": "--", "linewidth": 1.5, "markersize": 3.5, "zorder": 3},
}

SPACING = {
    "xlabel_pad":         -1,
    "ylabel_pad":         -4,
    "axis_margin":        0.05,
    "line_outline_size":  2,
    "marker_outline_size": 2,
}


def _plot_with_white_outline(
    ax,
    x_values,
    y_values,
    *,
    color: str,
    marker: str,
    linestyle: str,
    linewidth: float,
    markersize: float,
    line_outline_size: float,
    marker_outline_size: float,
    zorder: int,
):
    """Draw a benchmark-style white outline behind a colored line and markers."""
    ax.plot(
        x_values,
        y_values,
        color="white",
        linestyle=linestyle,
        marker=marker,
        linewidth=linewidth + line_outline_size,
        markersize=markersize + marker_outline_size,
        markeredgecolor="white",
        markerfacecolor="white",
        zorder=zorder,
    )
    return ax.plot(
        x_values,
        y_values,
        color=color,
        linestyle=linestyle,
        marker=marker,
        linewidth=linewidth,
        markersize=markersize,
        markerfacecolor=color,
        markeredgecolor=color,
        zorder=zorder,
    )[0]


MERGE_GAME_TYPES: dict[str, str] = {
    "interventional": "(INT)",
    "exhaustive":     "(EXT)",
    "tabpfn":         "(TABPFN)",
}


def _load_single_csv(*, index: str, order: int, game_type: str, input_path: str | None, suffix: str | None) -> pd.DataFrame:
    if input_path is not None:
        path = Path(input_path)
    else:
        path = Path(f"results_benchmark_{index}_{order}_{game_type}.csv")
    if not path.exists():
        raise FileNotFoundError(f"Could not find input CSV: {path}")
    df = pd.read_csv(path)
    df = df[df["n_players"] >= 12].copy()
    df = df.replace({"approximator": APPROXIMATOR_RENAMING})
    kernel_mask = df["approximator"] == "KernelSHAPIQ"
    min_budget = df["n_players"].apply(lambda n: math.comb(int(n) + 1, order))
    df = df[~(kernel_mask & (df["budget"] < min_budget))].copy()
    if suffix is not None:
        df["game"] = df["game"].astype(str) + f" {suffix}"
    return df


def _load_results(*, index: str, order: int, game_type: str, input_path: str | None) -> pd.DataFrame:
    if game_type == "merged":
        frames: list[pd.DataFrame] = []
        for gt, suffix in MERGE_GAME_TYPES.items():
            try:
                frames.append(_load_single_csv(index=index, order=order, game_type=gt, input_path=None, suffix=suffix))
            except FileNotFoundError as exc:
                print(f"Warning: skipping {gt} — {exc}")
        if not frames:
            raise FileNotFoundError("No CSV files found for any of the merged game types.")
        return pd.concat(frames, ignore_index=True)
    return _load_single_csv(index=index, order=order, game_type=game_type, input_path=input_path, suffix=None)


def _mean_mse_complete_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """Return mean MSE per (game, budget, approximator), restricted to
    (game, budget) pairs where every method in APPROXIMATORS_TO_PLOT is present."""
    df = df[df["approximator"].isin(APPROXIMATORS_TO_PLOT)].copy()
    mean_mse = (
        df.groupby(["game", "budget", "approximator"], as_index=False)["MSE"].mean()
    )
    counts = mean_mse.groupby(["game", "budget"])["approximator"].nunique()
    n_methods = len(APPROXIMATORS_TO_PLOT)
    valid_pairs = counts[counts == n_methods].reset_index()[["game", "budget"]]
    return mean_mse.merge(valid_pairs, on=["game", "budget"], how="inner")


def _compute_win_rates(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return (win_rate_df, n_b_series).

    win_rate_df has columns [budget, approximator, win_rate].
    n_b_series is indexed by budget, values = number of contributing datasets.
    """
    mean_mse = _mean_mse_complete_pairs(df)

    if mean_mse.empty:
        return pd.DataFrame(columns=["budget", "approximator", "win_rate"]), pd.Series(dtype=int)

    # Winner = approximator with lowest MSE for each (game, budget).
    winner_idx = mean_mse.groupby(["game", "budget"])["MSE"].idxmin()
    winners = mean_mse.loc[winner_idx, ["game", "budget", "approximator"]].copy()

    # N_b: number of distinct datasets per budget.
    n_b = winners.groupby("budget")["game"].nunique()

    # Win counts and rates.
    win_counts = winners.groupby(["budget", "approximator"])["game"].nunique().reset_index(name="wins")
    win_counts = win_counts.merge(n_b.rename("n_b"), on="budget")
    win_counts["win_rate"] = win_counts["wins"] / win_counts["n_b"]

    return win_counts[["budget", "approximator", "win_rate"]], n_b


def _compute_per_dataset_ratios(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return (ratio_df, n_b_series) with one row per (budget, approximator, game).

    ratio = MSE_m / min_{m'} MSE_{m'} at that (game, budget).
    Only (game, budget) pairs where every method is present are kept.
    """
    mean_mse = _mean_mse_complete_pairs(df)
    if mean_mse.empty:
        return pd.DataFrame(columns=["budget", "approximator", "game", "ratio"]), pd.Series(dtype=int)

    best = mean_mse.groupby(["game", "budget"])["MSE"].min().rename("best_mse").reset_index()
    merged = mean_mse.merge(best, on=["game", "budget"])
    merged["ratio"] = merged["MSE"] / merged["best_mse"].clip(lower=1e-300)
    merged = merged[np.isfinite(merged["ratio"]) & (merged["ratio"] > 0)]

    n_b = merged.groupby("budget")["game"].nunique()
    return merged[["budget", "approximator", "game", "ratio"]], n_b


def _draw_violin(
    ax,
    x_center: float,
    values: np.ndarray,
    *,
    color: str,
    half_width: float,
    y_min: float,
    y_max: float,
    zorder: int,
    alpha: float = 0.55,
) -> float:
    """Draw a vertical KDE violin at x_center on a log-y axis. Returns the median."""
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals) & (vals > 0) & (vals >= y_min) & (vals <= y_max)]
    if vals.size < 2:
        v = float(vals[0]) if vals.size == 1 else np.nan
        if np.isfinite(v):
            ax.scatter([x_center], [v], color=color, s=20, zorder=zorder + 2,
                       edgecolors="white", linewidths=0.5)
        return v

    # KDE in log(y) space so the violin is symmetric on the log-y axis.
    log_vals = np.log(vals)
    bw = max(1.06 * log_vals.std(ddof=1) * vals.size ** (-0.2), 1e-3)

    y_log_grid = np.linspace(np.log(y_min), np.log(y_max), 120)
    density = np.exp(
        -0.5 * ((y_log_grid[:, None] - log_vals[None, :]) / bw) ** 2
    ).sum(axis=1)
    density /= density.max()

    y_grid = np.exp(y_log_grid)
    x_left  = x_center - half_width * density
    x_right = x_center + half_width * density

    ax.fill_betweenx(y_grid, x_left, x_right, color=color, alpha=alpha, linewidth=0.0, zorder=zorder - 1)
    ax.plot(x_right, y_grid, color=color, linewidth=0.5, alpha=min(alpha * 1.6, 1.0), zorder=zorder)
    ax.plot(x_left,  y_grid, color=color, linewidth=0.5, alpha=min(alpha * 1.6, 1.0), zorder=zorder)

    median = float(np.exp(np.median(log_vals)))
    ax.scatter([x_center], [median], color=color, s=18, zorder=zorder + 2,
               edgecolors="white", linewidths=0.6)
    return median


def _setup_x_axis(ax, n_b: pd.Series, *, x_log_scale: bool) -> None:
    """Configure log x-axis ticks snapped to available budget values."""
    if n_b.empty:
        return
    all_budgets = np.array(sorted(n_b.index.tolist()), dtype=float)
    if all_budgets.size == 0:
        return
    if x_log_scale:
        ax.set_xscale("log")
    tick_candidates = np.unique(
        np.round(np.geomspace(all_budgets[0], all_budgets[-1], 6)).astype(int)
    )
    available = all_budgets.astype(int)
    ticks = sorted({int(min(available, key=lambda b, tc=tc: abs(b - tc))) for tc in tick_candidates})
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax.xaxis.set_minor_formatter(NullFormatter())


def _annotate_n_b(ax, n_b: pd.Series) -> None:
    """Place n=<count> labels below each budget tick."""
    if n_b.empty:
        return
    tf = transforms.blended_transform_factory(ax.transData, ax.transAxes)
    for budget, n in n_b.items():
        ax.text(
            budget, -0.16, f"n={n}",
            transform=tf, ha="center", va="top",
            fontsize=5.5, color="#888888", fontweight=W_REGULAR, clip_on=False,
        )


def _shared_params(spacing: dict | None) -> tuple[float, float, float, float, float]:
    _sp = spacing or {}
    return (
        _sp.get("xlabel_pad", 4.0),
        _sp.get("ylabel_pad", 4.0),
        _sp.get("axis_margin", 0.05),
        _sp.get("line_outline_size", 1.0),
        _sp.get("marker_outline_size", 1.0),
    )


def _save(fig, ax, out: Path) -> None:
    fig.tight_layout()
    apply_tick_style(ax)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", pad_inches=0.05, dpi=300)
    plt.close(fig)
    print(f"Saved plot to {out}")


def _place_legend(ax, handles: list, labels: list) -> None:
    ax.legend(
        handles, labels,
        loc="upper left", ncol=1, frameon=False,
        fontsize=8, handlelength=2.0, handletextpad=0.3, labelspacing=0.4,
        bbox_to_anchor=(1.01, 1.13), bbox_transform=ax.transAxes,
    )


def plot_win_rate_vs_budget(
    *,
    index: str,
    order: int,
    game_type: str,
    input_path: str | None,
    output_path: str,
    x_log_scale: bool = True,
    title: str | None = None,
    spacing: dict | None = None,
) -> None:
    xlabel_pad, ylabel_pad, axis_margin, line_outline_size, marker_outline_size = _shared_params(spacing)

    df = _load_results(index=index, order=order, game_type=game_type, input_path=input_path)
    win_rate_df, n_b = _compute_win_rates(df)

    fig, ax = plt.subplots(1, 1, figsize=(12, 3.2))
    ax.grid(True, which="both", axis="both", linestyle="--", linewidth=0.6, alpha=0.35)

    legend_handles: list = []
    legend_labels: list = []

    for method in APPROXIMATORS_TO_PLOT:
        style = STYLE.get(method, {"color": "#888888", "marker": "o", "linestyle": "-", "linewidth": 1.5, "markersize": 4.0, "zorder": 2})
        method_df = win_rate_df[win_rate_df["approximator"] == method].sort_values("budget")
        if method_df.empty:
            continue
        h = _plot_with_white_outline(
            ax,
            method_df["budget"].to_numpy(dtype=float),
            method_df["win_rate"].to_numpy(dtype=float),
            color=style["color"], marker=style["marker"], linestyle=style["linestyle"],
            linewidth=style["linewidth"], markersize=style["markersize"],
            line_outline_size=line_outline_size, marker_outline_size=marker_outline_size,
            zorder=style["zorder"],
        )
        legend_handles.append(h)
        legend_labels.append(method)

    _annotate_n_b(ax, n_b)
    _setup_x_axis(ax, n_b, x_log_scale=x_log_scale)
    ax.set_ylim(-0.02, 1.05)
    ax.yaxis.set_major_locator(FixedLocator([0, 0.25, 0.5, 0.75, 1.0]))
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.margins(axis_margin)
    ax.set_xlabel("Budget", fontsize=10, fontweight=W_SEMIBOLD, labelpad=xlabel_pad)
    ax.set_ylabel("Win Rate", fontsize=10, fontweight=W_SEMIBOLD, labelpad=ylabel_pad)
    if title is not None:
        fig.suptitle(title, fontsize=13, fontweight=W_REGULAR)
    _place_legend(ax, legend_handles, legend_labels)
    _save(fig, ax, Path(output_path))


def plot_performance_ratio_vs_budget(
    *,
    index: str,
    order: int,
    game_type: str,
    input_path: str | None,
    output_path: str,
    x_log_scale: bool = True,
    title: str | None = None,
    spacing: dict | None = None,
    n_violin_checkpoints: int = 5,
) -> None:
    """Plot per-dataset MSE ratio distributions as violins at budget checkpoints.

    At each checkpoint, one violin per method shows the full distribution of
    r = MSE_m / min_{m'} MSE_{m'} across datasets. Medians are connected by a
    thin line. Both axes are log-scaled.
    """
    xlabel_pad, ylabel_pad, axis_margin, _, _ = _shared_params(spacing)
    Y_MIN, Y_MAX = 0.9, 10.0

    df = _load_results(index=index, order=order, game_type=game_type, input_path=input_path)
    ratio_df, n_b = _compute_per_dataset_ratios(df)

    fig, ax = plt.subplots(1, 1, figsize=(12, 3.2))
    ax.axhline(1.0, color="#555555", linestyle="--", linewidth=1.0, zorder=1)
    ax.grid(True, which="both", axis="both", linestyle="--", linewidth=0.6, alpha=0.35)

    # Select budget checkpoints evenly spaced in log across available budgets.
    all_budgets = np.array(sorted(n_b.index.tolist()), dtype=float)
    if all_budgets.size == 0:
        _save(fig, ax, Path(output_path))
        return

    candidates = np.geomspace(all_budgets[0], all_budgets[-1], n_violin_checkpoints)
    checkpoints = np.array(
        sorted({int(min(all_budgets, key=lambda b, c=c: abs(b - c))) for c in candidates}),
        dtype=float,
    )

    # Per-method log10 offsets within each budget group so violins don't overlap.
    # 4 methods → offsets at -0.195, -0.065, +0.065, +0.195 log10.
    n_methods = len(APPROXIMATORS_TO_PLOT)
    step_log = 0.13
    half_span = step_log * (n_methods - 1) / 2
    method_offsets_log = np.linspace(-half_span, half_span, n_methods)

    # Violin half-width: 0.06 log10 units expressed in data coords at each center.
    HALF_WIDTH_LOG = 0.06

    legend_handles: list = []
    legend_labels: list = []

    for m_idx, method in enumerate(APPROXIMATORS_TO_PLOT):
        style = STYLE.get(method, {"color": "#888888", "marker": "o", "linestyle": "-", "linewidth": 1.0, "markersize": 4.0, "zorder": 2})
        offset_log = method_offsets_log[m_idx]

        method_ratios = ratio_df[ratio_df["approximator"] == method]
        medians_x: list[float] = []
        medians_y: list[float] = []

        for chk in checkpoints:
            x_center = chk * (10 ** offset_log)
            half_width = x_center * (10 ** HALF_WIDTH_LOG - 1)

            vals = method_ratios[method_ratios["budget"] == chk]["ratio"].to_numpy()
            median = _draw_violin(
                ax, x_center, vals,
                color=style["color"], half_width=half_width,
                y_min=Y_MIN, y_max=Y_MAX,
                zorder=style["zorder"], alpha=0.50,
            )
            if np.isfinite(median):
                medians_x.append(x_center)
                medians_y.append(median)

        # Thin line connecting medians across checkpoints.
        if len(medians_x) > 1:
            ax.plot(
                medians_x, medians_y,
                color=style["color"], linestyle=style["linestyle"],
                linewidth=0.8, alpha=0.6, zorder=style["zorder"],
            )

        # Proxy handle for legend (colored line + marker).
        h = ax.plot([], [], color=style["color"], linestyle=style["linestyle"],
                    linewidth=1.5, marker="o", markersize=4)[0]
        legend_handles.append(h)
        legend_labels.append(method)

    # N_b annotations use the checkpoint positions (center of the 4-method group).
    n_b_checkpoints = n_b[n_b.index.isin(checkpoints.astype(int))]
    _annotate_n_b(ax, n_b_checkpoints)

    if x_log_scale:
        ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(bottom=Y_MIN, top=Y_MAX)
    ax.yaxis.set_major_locator(FixedLocator([1, 2, 5, 10]))
    ax.set_yticklabels(["1×", "2×", "5×", "10×"])

    # x-ticks at the checkpoint values.
    ax.xaxis.set_major_locator(FixedLocator(checkpoints.tolist()))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax.xaxis.set_minor_formatter(NullFormatter())

    ax.margins(axis_margin)
    ax.set_xlabel("Budget", fontsize=10, fontweight=W_SEMIBOLD, labelpad=xlabel_pad)
    ax.set_ylabel("MSE ratio (vs. best)", fontsize=10, fontweight=W_SEMIBOLD, labelpad=ylabel_pad)
    if title is not None:
        fig.suptitle(title, fontsize=13, fontweight=W_REGULAR)
    _place_legend(ax, legend_handles, legend_labels)
    _save(fig, ax, Path(output_path))


def _compute_winners(df: pd.DataFrame) -> pd.DataFrame:
    """Return DataFrame [game, budget, winner, n_players].

    Winner = method with lowest mean MSE at each (game, budget) among whatever
    methods are present. KernelSHAPIQ has already been removed from rows where
    its budget is insufficient, so it simply won't compete there.
    """
    df = df[df["approximator"].isin(APPROXIMATORS_TO_PLOT)].copy()
    if df.empty:
        return pd.DataFrame(columns=["game", "budget", "winner", "n_players"])

    mean_mse = df.groupby(["game", "budget", "approximator"], as_index=False)["MSE"].mean()

    idx = mean_mse.groupby(["game", "budget"])["MSE"].idxmin()
    winners = mean_mse.loc[idx, ["game", "budget", "approximator"]].rename(
        columns={"approximator": "winner"}
    ).copy()

    n_players_map = df.groupby("game")["n_players"].first()
    winners["n_players"] = winners["game"].map(n_players_map)
    return winners


def plot_winner_map(
    *,
    index: str,
    order: int,
    game_type: str,
    input_path: str | None,
    output_path: str,
    x_log_scale: bool = True,
    title: str | None = None,
    spacing: dict | None = None,
) -> None:
    """Winner map: x=budget, y=dataset (sorted by n_players), dot=winning method color."""
    xlabel_pad, ylabel_pad, _, _, _ = _shared_params(spacing)

    df = _load_results(index=index, order=order, game_type=game_type, input_path=input_path)
    winners = _compute_winners(df)

    if winners.empty:
        print("No data for winner map.")
        return

    # Sort datasets by n_players ascending.
    games_ordered = (
        winners[["game", "n_players"]].drop_duplicates()
        .sort_values("n_players")["game"].tolist()
    )
    n_games = len(games_ordered)
    game_to_y = {g: i for i, g in enumerate(games_ordered)}

    method_color = {m: STYLE[m]["color"] for m in APPROXIMATORS_TO_PLOT if m in STYLE}

    fig_height = max(3.5, n_games * 0.32 + 1.2)
    fig, ax = plt.subplots(1, 1, figsize=(10, fig_height))

    # Marker size: tile the y-axis rows.
    row_height_pts = (fig_height * 72) / (n_games + 2)
    marker_s = (row_height_pts * 0.72) ** 2

    x_vals = winners["budget"].to_numpy(dtype=float)
    y_vals = winners["game"].map(game_to_y).to_numpy(dtype=float)
    colors = [method_color.get(w, "#888888") for w in winners["winner"]]

    ax.scatter(x_vals, y_vals, c=colors, s=marker_s, marker="s",
               linewidths=0, zorder=3)

    # Subtle horizontal dividers between datasets.
    for i in range(n_games):
        ax.axhline(i - 0.5, color="#dddddd", linewidth=0.4, zorder=1)

    # y-axis: dataset names with n_players annotation.
    n_players_map = winners.set_index("game")["n_players"].to_dict()
    ylabels = [
        f"{g}  (d={n_players_map.get(g, '?')})" for g in games_ordered
    ]
    ax.set_yticks(range(n_games))
    ax.set_yticklabels(ylabels, fontsize=7)
    ax.set_ylim(-0.7, n_games - 0.3)

    # x-axis.
    if x_log_scale:
        ax.set_xscale("log")
    all_budgets = sorted(winners["budget"].unique().tolist())
    tick_candidates = np.unique(
        np.round(np.geomspace(all_budgets[0], all_budgets[-1], 7)).astype(int)
    )
    ticks = sorted({int(min(all_budgets, key=lambda b, t=t: abs(b - t))) for t in tick_candidates})
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.grid(True, which="major", axis="x", linestyle="--", linewidth=0.5, alpha=0.35, zorder=0)

    ax.set_xlabel("Budget", fontsize=10, fontweight=W_SEMIBOLD, labelpad=xlabel_pad)
    ax.set_ylabel("Dataset", fontsize=10, fontweight=W_SEMIBOLD, labelpad=ylabel_pad)
    if title is not None:
        fig.suptitle(title, fontsize=13, fontweight=W_REGULAR)

    # Legend: one square per method.
    legend_handles = [
        ax.scatter([], [], c=STYLE[m]["color"], s=60, marker="s", linewidths=0, label=m)
        for m in APPROXIMATORS_TO_PLOT if m in STYLE
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper left", ncol=1, frameon=False,
        fontsize=8, handlelength=1.0, handletextpad=0.4, labelspacing=0.5,
        bbox_to_anchor=(1.01, 1.0), bbox_transform=ax.transAxes,
    )
    #ax.set_title("Winning method per dataset and budget", fontsize=13, fontweight=W_REGULAR)
    _save(fig, ax, Path(output_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot SOTA comparison across datasets (win-rate, performance ratio, or winner map)."
    )
    parser.add_argument("--index",      type=str, default="SII")
    parser.add_argument("--order",      type=int, default=2)
    parser.add_argument(
        "--game-type", type=str, default="interventional",
        help="interventional | exhaustive | tabpfn | merged  (merged combines all three)",
    )
    parser.add_argument("--input-path", type=str, default=None)
    parser.add_argument("--output-path",type=str, default=None)
    parser.add_argument("--no-x-log-scale", action="store_true")
    parser.add_argument("--title",      type=str, default=None)
    parser.add_argument(
        "--plot-type", choices=["winrate", "ratio", "winnermap", "all"], default="winnermap",
        help="winrate | ratio | winnermap | all",
    )

    args = parser.parse_args()
    x_log = not args.no_x_log_scale

    def _out(suffix: str) -> str:
        if args.output_path and args.plot_type not in ("all",):
            return args.output_path
        base = f"plots/main/main_paper_plot_sota_performance_{args.index}_{args.order}_{args.game_type}"
        return f"{base}_{suffix}.pdf"

    common = dict(
        index=args.index, order=args.order, game_type=args.game_type,
        input_path=args.input_path, x_log_scale=x_log, title=args.title, spacing=SPACING,
    )

    if args.plot_type in ("winrate", "all"):
        plot_win_rate_vs_budget(**common, output_path=_out("winrate"))
    if args.plot_type in ("ratio", "all"):
        plot_performance_ratio_vs_budget(**common, output_path=_out("ratio"))
    if args.plot_type in ("winnermap", "all"):
        plot_winner_map(**common, output_path=_out("winnermap"))
