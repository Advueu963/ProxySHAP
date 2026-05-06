"""Plot aggregated MSE vs number of players for all main-paper approximators.

For each (approximator, dataset, n_players) triple the script averages MSE
across all budget steps, yielding a single summary value per instance.  Those
per-instance values are then aggregated across datasets and explained instances
to produce one curve per approximator.

Example:
    uv run python plot_aggregated_mse_vs_n_players.py \
        --index SII --order 2 --game-type interventional
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).parent / "special_plot_scripts"))

import matplotlib.pyplot as plt
from matplotlib import transforms
from matplotlib.ticker import NullFormatter, FixedLocator
import numpy as np
import pandas as pd

from scipy.special import binom
from shapiq_benchmark.plot import STYLE_DICT
from _plot_style import apply_tick_style, setup_fonts, W_REGULAR, W_SEMIBOLD

setup_fonts()

APPROXIMATOR_RENAMING = {
    # Internal benchmark names
    "RegressionMSRIQ-NoAdjustment": "ProxySHAP (XGBoost) [our]",
    "RegressionMSRIQ": "ProxySHAP (XGBoost, MSR) [our]",
    "Linear-NoAdjustment": "ProxySHAP (Linear) [our]",
    "Linear-RECAP": "ProxySHAP (Linear, MSR) [our]",
    "RegressionMSRIQ-XGB-PreDef-NoAdjustment": "ProxySHAP+ (XGBoost) [our]",
    "RegressionMSRIQ-XGB-PreDef": "ProxySHAP+ (XGBoost, MSR) [our]",
    "RegressionMSRIQ-XGB-CV-BO": "ProxySHAP* (XGBoost, MSR) [our]",
    "RegressionMSRIQ-XGB-CV-BO-NoAdjustment": "ProxySHAP* (XGBoost) [our]",
    "ProxySpex": "ProxySPEX",
    # Display names without [our] suffix (used in some CSVs)
    "ProxySHAP (XGBoost)": "ProxySHAP (XGBoost) [our]",
    "ProxySHAP (XGBoost, MSR)": "ProxySHAP (XGBoost, MSR) [our]",
    "ProxySHAP (Linear)": "ProxySHAP (Linear) [our]",
    "ProxySHAP (Linear, MSR)": "ProxySHAP (Linear, MSR) [our]",
    "ProxySHAP+ (XGBoost)": "ProxySHAP+ (XGBoost) [our]",
    "ProxySHAP+ (XGBoost, MSR)": "ProxySHAP+ (XGBoost, MSR) [our]",
    "ProxySHAP* (XGBoost)": "ProxySHAP* (XGBoost) [our]",
    "ProxySHAP* (XGBoost, MSR)": "ProxySHAP* (XGBoost, MSR) [our]",
}

APPROXIMATORS_TO_PLOT = [
    "KernelSHAPIQ",
    "ProxySPEX (XGBoost)",
    #"ProxySPEX",
    "ProxySPEX (NoRefinement)",
    "ProxySHAP (XGBoost, MSR) [our]",
    "ProxySHAP (XGBoost) [our]",
    "ProxySHAP* (XGBoost, MSR) [our]",
    "ProxySHAP* (XGBoost) [our]",
    #"ProxySHAP (Linear, MSR) [our]",
    #"ProxySHAP (Linear) [our]",
]

LINEAR_METHODS = [
    "ProxySHAP (Linear, MSR) [our]",
    "ProxySHAP (Linear) [our]",
]

# Datasets excluded from the aggregated plot due to numerically degenerate MSE values
# (orders-of-magnitude higher than all other datasets — see debug_bad_datasets.py).
DATASETS_TO_EXCLUDE = [
    "TabArenaMiamiHousingLocalXAI",
    "TabArenaSeismicBumpsLocalXAI",
]


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
        zorder=zorder + 1,
    )[0]


def _configure_n_player_axis(axis, *, x_values: np.ndarray, x_log_scale: bool) -> None:
    """Use fixed x-ticks and annotate low/medium/high n_players sections."""
    finite_x = x_values[np.isfinite(x_values) & (x_values > 0)]
    if finite_x.size == 0:
        return

    x_min = float(np.min(finite_x))
    x_max = float(np.max(finite_x))

    ticks_in_range = [x_min, x_max]
    for anchor in (30.0, 90.0):
        if x_min <= anchor <= x_max:
            ticks_in_range.append(anchor)
    ticks_in_range = sorted({int(round(tick)) for tick in ticks_in_range if tick > 0})
    if not ticks_in_range:
        ticks_in_range = [int(round(float(np.median(finite_x))))]
    axis.xaxis.set_major_locator(FixedLocator(ticks_in_range))
    axis.set_xticklabels([str(int(tick)) for tick in ticks_in_range])
    axis.xaxis.set_minor_formatter(NullFormatter())

    for threshold in (30, 90):
        if x_min < threshold < x_max:
            axis.axvline(threshold, color="#9a9a9a", linestyle="--", linewidth=0.9, zorder=1)

    text_transform = transforms.blended_transform_factory(axis.transData, axis.transAxes)
    sections = [
        ("low", r"$d < 30$", x_min, min(30.0, x_max)),
        ("medium", r"$30 \leq d < 90$", max(30.0, x_min), min(90.0, x_max)),
        ("high", r"$d \geq 90$", max(90.0, x_min), x_max),
    ]
    for label, detail, left, right in sections:
        if right <= left:
            continue
        center = float(np.sqrt(left * right)) if x_log_scale else (left + right) / 2.0
        axis.text(
            center,
            0.97,
            label,
            transform=text_transform,
            ha="center",
            va="top",
            fontsize=10,
            color="#444444",
            fontweight=W_SEMIBOLD,
        )
        axis.text(
            center,
            0.90,
            detail,
            transform=text_transform,
            ha="center",
            va="top",
            fontsize=8,
            color="#444444",
            fontweight=W_SEMIBOLD,
        )


def _load_results(index: str, order: int, game_type: str, input_path: str | None) -> pd.DataFrame:
    if input_path is None:
        path = Path(f"results_benchmark_{index}_{order}_{game_type}.csv")
    else:
        path = Path(input_path)

    if not path.exists():
        raise FileNotFoundError(f"Could not find input CSV: {path}")

    df = pd.read_csv(path, low_memory=False)
    print("AMOUNT OF DATA SETS", df["game"].nunique())

    # Some sub-CSVs were saved with a pandas integer index column (to_csv without
    # index=False). On concatenation those rows are shifted one column right, so
    # game_type holds an integer row-index instead of the expected game-type string.
    # Drop these malformed rows before any further processing.
    shifted_rows = pd.to_numeric(df["game_type"], errors="coerce").notna()
    if shifted_rows.any():
        print(f"Warning: dropping {shifted_rows.sum()} shifted rows (saved with index column).")
        df = df[~shifted_rows]

    df = df.replace({"approximator": APPROXIMATOR_RENAMING})
    df = df[df["n_players"] < 1000]  # leave out the huge datasets
    df = df[~df["game"].isin(DATASETS_TO_EXCLUDE)]
    return df


def _min_budget_for_order(n_players: int, order: int) -> int:
    """Minimum budget for the order-k regression to be fully determined (C(n+1,0)+…+C(n+1,order))."""
    return int(sum(binom(n_players, t) for t in range(order + 1)))
def _apply_min_budget_filter(df: pd.DataFrame, order: int) -> pd.DataFrame:
    """Drop KernelSHAPIQ rows where budget is too small for a determined regression."""
    kernel_mask = df["approximator"] == "KernelSHAPIQ"
    other = df[~kernel_mask]

    kernel = df[kernel_mask]
    valid_kernel_parts = []
    for n, group in kernel.groupby("n_players", sort=False):
        min_b = _min_budget_for_order(int(n), order)
        valid_kernel_parts.append(group[group["budget"] >= min_b])

    if valid_kernel_parts:
        valid_kernel = pd.concat(valid_kernel_parts, ignore_index=True)
    else:
        valid_kernel = kernel.iloc[:0]

    return pd.concat([other, valid_kernel], ignore_index=True)

def _select_per_dataset_nearest_budget_rows(
    df: pd.DataFrame,
    target_budget: int,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Select rows at the nearest budget to target_budget for each dataset (game)."""
    selected_frames: list[pd.DataFrame] = []
    mapping: dict[str, int] = {}

    for game_name, game_df in df.groupby("game"):
        feasible_mask = [
            True if int(n_players) >= 62 else int(budget) < (1 << int(n_players))
            for budget, n_players in zip(game_df["budget"], game_df["n_players"], strict=False)
        ]
        feasible_game_df = game_df.loc[feasible_mask].copy()

        available = np.sort(feasible_game_df["budget"].dropna().astype(int).unique())
        if available.size == 0:
            continue
        nearest = int(min(available, key=lambda b: abs(int(b) - int(target_budget))))
        mapping[str(game_name)] = nearest
        selected_frames.append(feasible_game_df[feasible_game_df["budget"] == nearest].copy())

    if not selected_frames:
        return pd.DataFrame(columns=df.columns), mapping

    return pd.concat(selected_frames, ignore_index=True), mapping


def _compute_fixed_budget_mse(df: pd.DataFrame, order: int, target_budget: int) -> pd.DataFrame:
    df = df[df["approximator"].isin(APPROXIMATORS_TO_PLOT)].copy()
    df = _apply_min_budget_filter(df, order)
    selected, _ = _select_per_dataset_nearest_budget_rows(df, target_budget)
    if selected.empty:
        return pd.DataFrame(columns=["approximator", "n_players", "fixed_budget_mse", "sem_fixed"])
    selected = selected.copy()
    selected["MSE"] = selected["MSE"].clip(lower=1e-8)
    instance_cols = ["approximator", "n_players", "game", "game_id", "id_explain", "iteration"]
    per_instance = selected.groupby(instance_cols, as_index=False)["MSE"].mean()
    group = per_instance.groupby(["approximator", "n_players"], as_index=False)
    agg = group["MSE"].agg(["mean", "sem"]).rename(columns={"mean": "fixed_budget_mse", "sem": "sem_fixed"})
    agg["sem_fixed"] = agg["sem_fixed"].fillna(0.0)
    return agg


def _compute_aggregated_mse(df: pd.DataFrame, order: int) -> pd.DataFrame:
    df = df[df["approximator"].isin(APPROXIMATORS_TO_PLOT)].copy()
    df = _apply_min_budget_filter(df, order)
    # Drop KernelSHAPIQ if budget to small to fit the regression
    # df_kernel_shap = df[df["approximator"] == "KernelSHAPIQ"]
    # valid_rows = []
    # for n, group in df_kernel_shap.groupby("n_players", sort=False):
    #     min_b = _min_budget_for_order(int(n), order)
    #     valid_rows.append(group[group["budget"] >= min_b])
    # df.loc[df["approximator"] == "KernelSHAPIQ", :] = pd.concat(valid_rows, ignore_index=True) if valid_rows else df.iloc[:0]

    if df.empty:
        return pd.DataFrame(
            columns=["approximator", "n_players", "mean_agg_mse", "min_agg_mse", "sem", "count"]
        )

    instance_cols = ["approximator", "n_players", "game", "game_id", "id_explain", "iteration"]

    # Mean MSE over budgets per instance (log-AUC proxy for log-spaced budgets).
    mean_per_instance = (
        df.groupby(instance_cols, as_index=False)["MSE"]
        .mean()
        .rename(columns={"MSE": "mean_over_budgets"})
    )
    mean_per_instance["mean_over_budgets"] = mean_per_instance["mean_over_budgets"].clip(lower=1e-8)

    # Min MSE over budgets per instance (best achievable budget in hindsight).
    min_per_instance = (
        df.groupby(instance_cols, as_index=False)["MSE"]
        .min()
        .rename(columns={"MSE": "min_over_budgets"})
    )
    min_per_instance["min_over_budgets"] = min_per_instance["min_over_budgets"].clip(lower=1e-8)

    per_instance = mean_per_instance.merge(min_per_instance, on=instance_cols)

    # Aggregate both metrics across all instances (mean ± SEM over datasets).
    group = per_instance.groupby(["approximator", "n_players"], as_index=False)
    mean_agg = group["mean_over_budgets"].agg(["mean", "sem", "count"]).rename(
        columns={"mean": "mean_agg_mse", "sem": "sem_mean"}
    )
    min_agg = group["min_over_budgets"].agg(["mean", "sem"]).rename(
        columns={"mean": "min_agg_mse", "sem": "sem_min"}
    )
    agg = mean_agg.merge(min_agg, on=["approximator", "n_players"])
    agg["sem_mean"] = agg["sem_mean"].fillna(0.0)
    agg["sem_min"] = agg["sem_min"].fillna(0.0)
    # Keep "sem" pointing at the mean panel's SEM for backward compat with _fill_panel.
    agg["sem"] = agg["sem_mean"]
    return agg


def _print_and_save_summary(summary: pd.DataFrame, output_path: str) -> None:
    """Print aggregated MSE table to terminal and save as CSV for LaTeX use."""
    col_w = 28
    approxs = [a for a in APPROXIMATORS_TO_PLOT if a in summary["approximator"].values]

    metrics = [
        ("mean_agg_mse", "sem_mean", "mean over budgets ± SEM"),
        ("min_agg_mse", "sem_min", "min over budgets ± SEM"),
        ("fixed_budget_mse", "sem_fixed", "fixed budget MSE ± SEM"),
    ]
    for metric, sem_col, label in [
        (m, s, lbl) for m, s, lbl in metrics if m in summary.columns
    ]:
        pivot = summary.pivot(index="n_players", columns="approximator", values=metric)
        pivot_sem = summary.pivot(index="n_players", columns="approximator", values=sem_col)
        print(f"\nAggregated MSE ({label}) per n_players:")
        header = f"{'n_players':>10}" + "".join(f"{a:>{col_w}}" for a in approxs)
        print(header)
        print("-" * len(header))
        for n in sorted(pivot.index):
            row = f"{int(n):>10}"
            for a in approxs:
                val = pivot.loc[n, a] if (n in pivot.index and a in pivot.columns) else float("nan")
                sem = pivot_sem.loc[n, a] if (n in pivot_sem.index and a in pivot_sem.columns) else float("nan")
                cell = f"{val:.3e} ± {sem:.1e}" if np.isfinite(val) else "—"
                row += f"{cell:>{col_w}}"
            print(row)

    csv_out = Path(output_path).with_suffix(".csv")
    summary.to_csv(csv_out, index=False)
    print(f"\nSaved summary CSV to {csv_out}")


def _fill_panel(
    ax,
    summary: pd.DataFrame,
    y_col: str,
    *,
    x_log_scale: bool,
    y_log_scale: bool,
    line_outline_size: float,
    marker_outline_size: float,
    panel_title: str = "",
) -> tuple[list, list, list[float]]:
    """Draw one panel (min or mean) onto ax. Returns (handles, labels, all_x)."""
    legend_handles: list = []
    legend_labels: list = []
    all_x: list[float] = []

    for approx in APPROXIMATORS_TO_PLOT:
        method_df = summary[summary["approximator"] == approx].sort_values("n_players")
        if method_df.empty:
            continue

        style = STYLE_DICT[approx]
        color = style["color"]
        marker = style.get("marker", "o")
        linestyle = style.get("linestyle", "-")

        x_vals = method_df["n_players"].to_numpy(dtype=float)
        y_vals = method_df[y_col].to_numpy(dtype=float)
        all_x.extend(x_vals.tolist())

        line = _plot_with_white_outline(
            ax,
            x_vals,
            y_vals,
            color=color,
            marker=marker,
            linestyle=linestyle,
            linewidth=1.5,
            markersize=3.5,
            line_outline_size=line_outline_size,
            marker_outline_size=marker_outline_size,
            zorder=2,
        )
        line.set_label(approx)

        if y_col == "mean_agg_mse":
            sem_col = "sem_mean"
        elif y_col == "fixed_budget_mse":
            sem_col = "sem_fixed"
        else:
            sem_col = "sem_min"
        sem_vals = method_df[sem_col].to_numpy(dtype=float)
        ax.fill_between(
            x_vals,
            np.clip(y_vals - sem_vals, 1e-12, None),
            y_vals + sem_vals,
            color=color,
            alpha=0.12,
            linewidth=0.0,
        )

        legend_handles.append(line)
        legend_labels.append(approx)

    if x_log_scale:
        ax.set_xscale("log")
    if y_log_scale:
        ax.set_yscale("log")

    ax.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.35)

    return legend_handles, legend_labels, all_x


def plot_aggregated_mse_vs_n_players(
    *,
    index: str,
    order: int,
    game_type: str,
    input_path: str | None,
    output_path: str,
    budget: int = 1000,
    line_outline_size: float = 1.0,
    marker_outline_size: float = 1.0,
    y_log_scale: bool = True,
    x_log_scale: bool = True,
    title: str | None = None,
) -> None:
    df = _load_results(index=index, order=order, game_type=game_type, input_path=input_path)
    summary = _compute_aggregated_mse(df, order=order)
    fixed_summary = _compute_fixed_budget_mse(df, order=order, target_budget=budget)
    summary = summary.merge(fixed_summary, on=["approximator", "n_players"], how="left")

    if summary.empty:
        raise ValueError(
            "No rows left after filtering. Check CSV path, index/order/game_type, and method naming."
        )

    _print_and_save_summary(summary, output_path)

    fig, (ax_min, ax_mean) = plt.subplots(2, 1, figsize=(10, 7), sharey=False)
    fig.subplots_adjust(hspace=0.08)

    _, _, x_min = _fill_panel(
        ax_min, summary, "fixed_budget_mse",
        x_log_scale=x_log_scale, y_log_scale=y_log_scale,
        line_outline_size=line_outline_size, marker_outline_size=marker_outline_size,
        panel_title="",
    )
    handles_mean, labels_mean, x_mean = _fill_panel(
        ax_mean, summary, "mean_agg_mse",
        x_log_scale=x_log_scale, y_log_scale=y_log_scale,
        line_outline_size=line_outline_size, marker_outline_size=marker_outline_size,
        panel_title="",
    )

    all_x = np.array(x_min + x_mean)
    _configure_n_player_axis(ax_min, x_values=all_x, x_log_scale=x_log_scale)
    _configure_n_player_axis(ax_mean, x_values=all_x, x_log_scale=x_log_scale)

    # Hide x-tick labels on the top panel
    ax_min.tick_params(labelbottom=False)

    ax_min.set_ylabel(f"MSE (budget ~ {budget:,})", fontsize=12, fontweight=W_SEMIBOLD)
    ax_mean.set_ylabel("MSE (log-AUC)", fontsize=12, fontweight=W_SEMIBOLD)
    ax_mean.set_xlabel("Number of Players", fontsize=12, fontweight=W_SEMIBOLD)

    if title is None:
        title = f"{index} order {order} ({game_type})"
    fig.suptitle(title, fontsize=13, fontweight=W_REGULAR)
    

    if handles_mean:
        ncol = min(len(handles_mean), 4)
        fig.legend(
            handles_mean,
            labels_mean,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.0),
            ncol=ncol,
            frameon=True,
            fancybox=True,
            framealpha=1.0,
            fontsize=9,
        )

    #fig.tight_layout(rect=(0.03, 0.12, 1.0, 0.93))

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    apply_tick_style(ax_min, ax_mean)
    fig.savefig(output, bbox_inches="tight", pad_inches=0.05, dpi=300)
    plt.close(fig)

    print(f"Saved plot to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot aggregated MSE vs number of players for all main-paper approximators."
    )
    parser.add_argument("--index", type=str, default="SII")
    parser.add_argument("--order", type=int, default=2)
    parser.add_argument("--game-type", type=str, default="interventional")
    parser.add_argument("--input-path", type=str, default=None)
    parser.add_argument("--output-path", type=str, default=None)
    parser.add_argument("--x-log-scale", action="store_true", default=True)
    parser.add_argument("--linear-x", action="store_true")
    parser.add_argument("--y-log-scale", action="store_true", default=True)
    parser.add_argument("--linear-y", action="store_true")
    parser.add_argument("--line-outline-size", type=float, default=1.0)
    parser.add_argument("--marker-outline-size", type=float, default=1.0)
    parser.add_argument("--budget", type=int, default=1000)
    parser.add_argument("--title", type=str, default=None)
    args = parser.parse_args()

    if args.output_path is None:
        args.output_path = (
            f"plots/main/main_paper_proxyshap_aggregated_mse_vs_n_players_"
            f"{args.index}_order{args.order}_{args.game_type}.pdf"
        )

    plot_aggregated_mse_vs_n_players(
        index=args.index,
        order=args.order,
        game_type=args.game_type,
        input_path=args.input_path,
        output_path=args.output_path,
        budget=args.budget,
        line_outline_size=args.line_outline_size,
        marker_outline_size=args.marker_outline_size,
        x_log_scale=not args.linear_x,
        y_log_scale=not args.linear_y,
        title=args.title,
    )
