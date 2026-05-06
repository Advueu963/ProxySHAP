"""Flipped variant of `main_paper_plot_relative_mse_n_players`.

Flips the y-axis so that the *adjusted* method's improvement appears on top
(green) and degradation on the bottom (red). Achieved by inverting the ratio at
the data layer (relative_mse = baseline / method instead of method / baseline)
and swapping the sticker positions, axhspan colors, and y-tick label colors.

Example:
    uv run python special_plot_scripts/main_paper_plot_relative_mse_n_players_flipped.py \
        --index SII --orders 2 3 --game-type interventional --budgets 1000 5000 10000
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import transforms
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator

import main_paper_plot_relative_mse_n_players as orig
from _plot_style import W_REGULAR, W_SEMIBOLD, apply_tick_style


# ── Inverted ratio: > 1 means the adjusted method beats baseline ─────────────
def _compute_relative_mse_flipped(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["approximator"].isin(orig.TARGET_APPROXIMATORS)].copy()

    if df.empty:
        return pd.DataFrame(columns=["n_players", "approximator", "relative_mse"])

    key_cols = ["game", "game_id", "id_explain", "iteration", "budget", "n_players"]

    grouped = (
        df.groupby(key_cols + ["approximator"], as_index=False)["MSE"]
        .mean()
        .rename(columns={"MSE": "mse"})
    )

    baseline_df = grouped[grouped["approximator"] == orig.BASELINE][key_cols + ["mse"]].rename(
        columns={"mse": "baseline_mse"}
    )

    merged = grouped.merge(baseline_df, on=key_cols, how="inner")
    # Need both > 0 since we now divide by method's mse.
    merged = merged[(merged["baseline_mse"] > 0) & (merged["mse"] > 0)]
    merged["relative_mse"] = merged["baseline_mse"] / merged["mse"]  # FLIPPED

    out = merged[merged["approximator"].isin(orig.METHODS)].copy()
    out = out[np.isfinite(out["relative_mse"])]
    out = out[out["relative_mse"] > 0]
    out = out[out["n_players"] >= 17]
    out = out[out["game"].isin(orig.DATASET_ALLOWLIST)]
    return out


# Monkey-patch so `_plot_order_data` uses the flipped ratio when called below.
orig._compute_relative_mse = _compute_relative_mse_flipped


def plot_relative_mse_vs_n_players_flipped(
    *,
    index: str,
    orders: list[int],
    game_type: str,
    input_paths: dict[int, str | None],
    output_path: str,
    budgets: list[int],
    x_log_scale: bool = True,
    title: str | None = None,
    spacing: dict | None = None,
) -> None:
    _sp = spacing or {}
    xlabel_pad: float = _sp.get("xlabel_pad", 4.0)
    ylabel_pad: float = _sp.get("ylabel_pad", 4.0)
    axis_margin: float = _sp.get("axis_margin", 0.05)

    if not orders:
        raise ValueError("At least one order must be provided.")
    if len(budgets) == 0:
        raise ValueError("At least one budget must be provided.")

    n_panels = len(orders)
    fig, axes = plt.subplots(
        1, n_panels,
        figsize=(5 * n_panels, 4.9),
        sharey=True,
    )
    if n_panels == 1:
        axes = [axes]

    _ytick_positions = [1/4, 1/3, 1/2, 1, 2, 3, 4]
    _ytick_labels    = ["4x", "3x", "2x", "1", "2x", "3x", "4x"]
    # FLIPPED palette: bottom = worse (red), top = better (green).
    _ytick_colors    = ["#9b2226"] * 3 + ["#555555"] + ["#2d6a4f"] * 3

    last_budget_labels: list[str] = []
    any_data_overall = False
    panel_x_values: list[np.ndarray] = []
    panel_has_data: list[bool] = []

    for order, ax in zip(orders, axes):
        # FLIPPED axhspans: green above the baseline, red below.
        ax.axhspan(1.0, 100, color="#4CAF50", alpha=0.04, zorder=0)
        ax.axhspan(0, 1.0, color="#E53935", alpha=0.04, zorder=0)
        ax.axhline(1.0, color="#555555", linestyle="--", linewidth=1.0, zorder=1)

        _sticker_tf = transforms.blended_transform_factory(ax.transAxes, ax.transData)
        _sticker_kw = dict(
            transform=_sticker_tf, ha="right", va="bottom",
            fontsize=orig.FONT_SIZE - 3, fontweight=W_SEMIBOLD, clip_on=False, zorder=8
        )
        # FLIPPED sticker positions: "better" on top, "worse" on bottom.
        ax.text(0.98, 4.02, "better", color="#2d6a4f", **_sticker_kw)
        ax.text(0.13, 0.21, "worse",  color="#9b2226", **_sticker_kw)
        # Order 1: drop "no effect" below the line; otherwise keep above.
        _no_effect_kw = {**_sticker_kw, "va": "top"} if order == 1 else _sticker_kw
        ax.text(0.98, 1, "no effect", color="#555555", **_no_effect_kw)

        palette = orig.ORDER_PALETTES.get(order, orig.BUDGET_COLORS)
        has_data, _, labels, all_x = orig._plot_order_data(
            ax,
            index=index,
            order=order,
            game_type=game_type,
            input_path=input_paths.get(order),
            budgets=budgets,
            palette=palette,
        )
        any_data_overall = any_data_overall or has_data
        panel_has_data.append(has_data)
        panel_x_values.append(all_x)
        if labels:
            last_budget_labels = labels

    combined_x = (
        np.concatenate([x for x in panel_x_values if x.size > 0])
        if any(x.size > 0 for x in panel_x_values)
        else np.array([])
    )

    finite_combined_x = combined_x[np.isfinite(combined_x) & (combined_x > 0)]
    if finite_combined_x.size > 0:
        x_min_global = float(np.min(finite_combined_x))
        x_max_global = float(np.max(finite_combined_x))
    else:
        x_min_global = x_max_global = None

    for panel_idx, (order, ax, has_data) in enumerate(
        zip(orders, axes, panel_has_data)
    ):
        ax.set_ylim(0.2, 4.8)
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(FixedLocator(_ytick_positions))
        ax.set_yticklabels(_ytick_labels, fontsize=orig.TICK_FONT_SIZE)
        ax.yaxis.set_minor_locator(plt.NullLocator())
        if x_log_scale:
            ax.set_xscale("log")
        orig._configure_n_player_axis(ax, x_values=combined_x, x_log_scale=x_log_scale)
        ax.grid(True, which="both", axis="y", linestyle="--", linewidth=0.6, alpha=0.35)
        ax.margins(axis_margin)
        if x_min_global is not None and x_max_global is not None:
            if x_log_scale:
                pad = (x_max_global / x_min_global) ** axis_margin
                ax.set_xlim(x_min_global / pad, x_max_global * pad)
            else:
                pad = (x_max_global - x_min_global) * axis_margin
                ax.set_xlim(x_min_global - pad, x_max_global + pad)
        ax.set_xlabel(
            "Number of Players",
            fontsize=orig.FONT_SIZE, fontweight=W_SEMIBOLD, labelpad=xlabel_pad,
        )
        if panel_idx == 0:
            ax.set_ylabel(
                "Adjustment effect on MSE",
                fontsize=orig.FONT_SIZE, fontweight=W_SEMIBOLD, labelpad=ylabel_pad,
            )
        ax.set_title(
            f"Order {order}",
            fontsize=orig.FONT_SIZE + 2,
            fontweight=W_SEMIBOLD,
            pad=18,
        )

        if not has_data:
            ax.text(
                0.5, 0.5, "No matching rows",
                ha="center", va="center",
                transform=ax.transAxes,
                fontsize=10, color="#666666",
            )

    palette_used = orig.ORDER_PALETTES.get(orders[0], orig.BUDGET_COLORS)
    budget_lines = [
        Line2D([], [], color=palette_used[i % len(palette_used)], linestyle="-", linewidth=1.5)
        for i in range(len(budgets))
    ]
    legend_labels = last_budget_labels or [f"{b // 1000}k" for b in budgets]
    legend_handles: list = [orig._SectionHandle(), *budget_lines]
    legend_labels_full = ["Budget:", *legend_labels]

    fig.suptitle(title, fontsize=13, fontweight=W_REGULAR)

    fig.legend(
        legend_handles,
        legend_labels_full,
        loc="lower center",
        ncol=len(legend_handles),
        frameon=False,
        fontsize=orig.FONT_SIZE,
        handlelength=2.0,
        handletextpad=0.3,
        columnspacing=0.6,
        bbox_to_anchor=(0.5, -0.02),
        handler_map={orig._SectionHandle: orig._SectionHandler()},
    )

    fig.tight_layout(rect=(0, 0.04, 1, 1))

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    for ax in axes:
        apply_tick_style(ax)
    fig.canvas.draw()
    for ax in axes:
        for tick_label, color in zip(ax.get_yticklabels(), _ytick_colors):
            tick_label.set_color(color)
    fig.savefig(output, bbox_inches="tight", pad_inches=0.05, dpi=300)
    plt.close(fig)

    print(f"Saved plot to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Flipped relative-MSE plot (improvement on top, worse on bottom).",
    )
    parser.add_argument("--index", type=str, default="SII")
    parser.add_argument("--orders", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--game-type", type=str, default="interventional")
    parser.add_argument("--input-path-order1", type=str, default=None)
    parser.add_argument("--input-path-order2", type=str, default=None)
    parser.add_argument("--input-path-order3", type=str, default=None)
    parser.add_argument("--output-path", type=str, default=None)
    parser.add_argument("--budgets", type=int, nargs="+", default=[1000, 5000, 10000, 20000, 35000])
    parser.add_argument("--title", type=str, default=None)

    args = parser.parse_args()

    if args.output_path is None:
        args.output_path = "plots/main/main_paper_plots_adjustment_benefit_flipped.pdf"

    input_paths: dict[int, str | None] = {
        1: args.input_path_order1,
        2: args.input_path_order2,
        3: args.input_path_order3,
    }

    plot_relative_mse_vs_n_players_flipped(
        index=args.index,
        orders=args.orders,
        game_type=args.game_type,
        input_paths=input_paths,
        output_path=args.output_path,
        budgets=args.budgets,
        x_log_scale=True,
        title=args.title,
        spacing=orig.SPACING,
    )
