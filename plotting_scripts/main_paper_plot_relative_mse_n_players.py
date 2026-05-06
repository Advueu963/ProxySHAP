"""Plot relative MSE over n_players for ProxySHAP XGBoost adjustment variants.

This script compares only three methods:
- ProxySHAP (XGBoost) [our] as baseline
- ProxySHAP (XGBoost, MSR) [our]

For each row context (game, game_id, id_explain, iteration, budget), the script
computes the relative ratio

    relative_mse = MSE(method) / MSE(baseline)

and then aggregates these ratios per n_players for fixed budgets to obtain one

Order 2 results are shown in the left panel (blue palette) and order 3 in the
right panel (red palette).

Example:
    uv run python special_plot_scripts/main_paper_plot_relative_mse_n_players.py \
        --index SII --orders 2 3 --game-type interventional --budgets 1000 5000 10000
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import transforms
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, NullFormatter
import numpy as np
import pandas as pd


from matplotlib.legend_handler import HandlerBase

from _plot_style import apply_tick_style, setup_fonts, W_REGULAR, W_SEMIBOLD, W_BOLD
from shapiq.plot._config import BLUE, RED


class _SectionHandle(Line2D):
    """Invisible legend artist used as a section-title placeholder."""
    def __init__(self) -> None:
        super().__init__([], [], alpha=0, linewidth=0)


class _SectionHandler(HandlerBase):
    def legend_artist(self, _legend, _orig_handle, _fontsize, handlebox):
        handlebox.width = 0
        handlebox.xdescent = 0
        return Line2D([], [], alpha=0)

setup_fonts()

APPROXIMATOR_RENAMING = {
    "RegressionMSRIQ-NoAdjustment": "ProxySHAP (XGBoost) [our]",
    "RegressionMSRIQ": "ProxySHAP (XGBoost, MSR) [our]",
    "ProxySHAP (XGBoost)": "ProxySHAP (XGBoost) [our]",
    "ProxySHAP (XGBoost, MSR)": "ProxySHAP (XGBoost, MSR) [our]",
}

BASELINE = "ProxySHAP (XGBoost) [our]"
METHODS = [
    "ProxySHAP (XGBoost, MSR) [our]",
]
TARGET_APPROXIMATORS = [BASELINE, *METHODS]

STYLE = {
    "ProxySHAP (XGBoost, MSR) [our]": {
        "marker": "o",
        "linestyle": "-",
    },
}
FONT_SIZE = 18
TICK_FONT_SIZE = 18
# ── Budget-colour palettes (light → dark, 7 stops) ────────────────────────
# Violet — original palette
PALETTE_VIOLET = [
    "#d8b4fe",  # lightest
    "#c084fc",
    "#7c3aed",  # vivid violet – mid
    "#5b21b6",
    "#3b0764",
    "#2e0650",
    "#1a0338",  # darkest
]

# Shapiq red (#ff0d57)
PALETTE_RED = [
    "#ffb3c6",  # lightest
    "#ff6690",
    RED.hex,    # shapiq red – mid
    "#cc0a46",
    "#990835",
    "#660524",
    "#330213",  # darkest
]

# Shapiq blue (#1e88e5)
PALETTE_BLUE = [
    "#8ec3f2",  # lightest
    "#78b7ef",
    "#4a9fea",  
    "#3493e7",
    BLUE.hex,   # shapiq blue – mid
    "#1565c0",
    "#0d47a1",
    "#082f6a",
    "#041833",  # darkest
]

# Teal — high contrast against red/blue, clean on white
PALETTE_TEAL = [
    "#b2dfdb",  # lightest
    "#4db6ac",
    "#00897b",  # vivid teal – mid
    "#00695c",
    "#004d40",
    "#003330",
    "#001a18",  # darkest
]

# ── Active palette — swap this variable to change all lines at once ─────────
BUDGET_COLORS = PALETTE_BLUE

# Maps order -> color palette; any other order falls back to BUDGET_COLORS.
ORDER_PALETTES: dict[int, list[str]] = {
    1: BUDGET_COLORS,
    2: BUDGET_COLORS,
    3: BUDGET_COLORS,
}

ORDER_LINESTYLES: dict[int, str] = {
    1: "-",
    2: "-",
    3: "-",
}

# Representative header color per order (mid-palette entry).
ORDER_HEADER_COLORS: dict[int, str] = {
    1: BUDGET_COLORS[2],
    2: BUDGET_COLORS[2],
    3: BUDGET_COLORS[2],
}

SPACING = {
    "xlabel_pad":  0,
    "ylabel_pad":  -0.1,
    "axis_margin": 0.05,
}

# ── Dataset allowlist ─────────────────────────────────────────────────────────
# All datasets that appear in the benchmark CSVs, sorted by n_players.
# Comment out any entry to exclude that dataset from the plot.
DATASET_ALLOWLIST = {
    "AdultCensusLocalXAI",                  # n=14
    "RealEstateLocalXAI",                   # n=15
    "TabArenaMiamiHousingLocalXAI",         # n=15
    "TabArenaSeismicBumpsLocalXAI",         # n=15
    #"ZooLocalXAI",                          # n=16
    "TabArenaOnlineShoppersLocalXAI",       # n=17
    "HepatitisLocalXAI",                    # n=19
    "TabArenaChurnLocalXAI",                # n=19
    #"TabArenaCreditGLocalXAI",              # n=20
    "TabArenaAirlineSatisfactionLocalXAI",  # n=21
    "TabArenaJm1LocalXAI",                  # n=21
    #"ThyroidLocalXAI",                      # n=21
    #"MushroomLocalXAI",                     # n=22
    "TabArenaCreditCardDefaultLocalXAI",    # n=23
    "TabArenaHelocLocalXAI",                # n=23
    "TabArenaCouponRecommendationLocalXAI", # n=24
    "TabArenaMarketingCampaignLocalXAI",    # n=25
    "BreastCancerLocalXAI",                 # n=30
    "TabArenaHazelnutLocalXAI",             # n=30
    "IonosphereLocalXAI",                   # n=33
    #"SoybeanLocalXAI",                      # n=35
    "TabArenaStudentsDropoutLocalXAI",      # n=36
   # "TabArenaAnnealLocalXAI",               # n=38
    "TabArenaQsarBiodegLocalXAI",           # n=41
    "TabArenaDiabetes130usLocalXAI",        # n=47
    "Corrgroups60LocalXAI",                 # n=60
    "IndependentLinear60LocalXAI",          # n=60
    "TabArenaSpliceLocalXAI",               # n=60
    "TabArenaBankruptcyLocalXAI",           # n=64
    #"NHANESILocalXAI",                      # n=79
    "TabArenaSuperconductivityLocalXAI",    # n=81
    "TabArenaCoil2000LocalXAI",             # n=85
    "TabArenaNaticusdroidLocalXAI",         # n=86
    "TabArenaTaiwaneseBankruptcyLocalXAI",  # n=94
    "CommunitiesAndCrimeLocalXAI",          # n=101
    "TabArenaMicLocalXAI",                  # n=111
    "TabArenaApsFailureLocalXAI",           # n=170
    "TabArenaKddcup09LocalXAI",             # n=212
    "TabArenaQsarTid11LocalXAI",            # n=1024
    "TabArenaHivaAgnosticLocalXAI",         # n=1617
    "TabArenaBioresponseLocalXAI",          # n=1776
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


def _load_results(index: str, order: int, game_type: str, input_path: str | None) -> pd.DataFrame:
    if input_path is None:
        path = Path(f"results_benchmark_{index}_{order}_{game_type}.csv")
    else:
        path = Path(input_path)

    if not path.exists():
        raise FileNotFoundError(f"Could not find input CSV: {path}")

    df = pd.read_csv(path)
    df = df.replace({"approximator": APPROXIMATOR_RENAMING})
    return df


def _compute_relative_mse(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["approximator"].isin(TARGET_APPROXIMATORS)].copy()

    if df.empty:
        return pd.DataFrame(columns=["n_players", "approximator", "relative_mse"])

    key_cols = ["game", "game_id", "id_explain", "iteration", "budget", "n_players"]

    # Mean over any duplicate rows within the same context and approximator.
    grouped = (
        df.groupby(key_cols + ["approximator"], as_index=False)["MSE"]
        .mean()
        .rename(columns={"MSE": "mse"})
    )

    baseline_df = grouped[grouped["approximator"] == BASELINE][key_cols + ["mse"]].rename(
        columns={"mse": "baseline_mse"}
    )

    merged = grouped.merge(baseline_df, on=key_cols, how="inner")
    merged = merged[merged["baseline_mse"] > 0]
    merged["relative_mse"] = merged["mse"] / merged["baseline_mse"]

    out = merged[merged["approximator"].isin(METHODS)].copy()
    # Remove trivial zero-error points to avoid ratio=0 artifacts in the panels.
    out = out[out["mse"] > 0]
    out = out[np.isfinite(out["relative_mse"])]
    out = out[out["relative_mse"] > 0]
    # Restrict to datasets with at least 14 players/features.
    out = out[out["n_players"] >= 17]
    # Restrict to TabArena datasets only.
    #out = out[out["game"].str.startswith("TabArena")]
    # Apply dataset allowlist — comment out entries in DATASET_ALLOWLIST to exclude them.
    out = out[out["game"].isin(DATASET_ALLOWLIST)]
    return out


def _aggregate_by_n_players(relative_df: pd.DataFrame) -> pd.DataFrame:
    if relative_df.empty:
        return pd.DataFrame(
            columns=["n_players", "approximator", "mean_relative_mse", "std", "count"]
        )

    # Geometric mean ± geometric std factor across explained instances (pooled over games)
    # for each (n_players, approximator). Each row in relative_df is one (game, id_explain)
    # observation, so we weight per-instance.
    # `std` stores exp(std(log(x))), a multiplicative spread — band = [mean/std, mean*std].
    grp = relative_df.groupby(["n_players", "approximator"])["relative_mse"]
    agg = pd.DataFrame({
        "mean_relative_mse": grp.agg(lambda x: np.exp(np.log(x).mean())),
        "std":               grp.agg(lambda x: np.exp(np.log(x).std(ddof=1)) if len(x) > 1 else 1.0),
        "count":             grp.count(),
    }).reset_index()
    return agg


def _select_per_dataset_nearest_budget_rows(
    relative_df: pd.DataFrame,
    target_budget: int,
    min_budget: int = 0,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Select rows for datasets whose best available budget falls in (min_budget, target_budget]."""
    selected_frames: list[pd.DataFrame] = []
    mapping: dict[str, int] = {}

    for game_name, game_df in relative_df.groupby("game"):
        # Exclude rows where the queried budget reaches or exceeds the coalition space.
        # For n_players >= 62, all budgets in this benchmark are far below 2**n.
        feasible_mask = [
            True if int(n_players) >= 62 else int(budget) < (1 << int(n_players))
            for budget, n_players in zip(game_df["budget"], game_df["n_players"], strict=False)
        ]
        feasible_game_df = game_df.loc[feasible_mask].copy()

        available = np.sort(feasible_game_df["budget"].dropna().astype(int).unique())
        valid = available[(available > min_budget) & (available <= target_budget)]
        if valid.size == 0:
            continue
        nearest = int(valid[-1])  # largest budget in the band
        mapping[str(game_name)] = nearest
        selected_frames.append(feasible_game_df[feasible_game_df["budget"] == nearest].copy())

    if not selected_frames:
        return pd.DataFrame(columns=relative_df.columns), mapping

    return pd.concat(selected_frames, ignore_index=True), mapping


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
    axis.set_xticks(ticks_in_range)
    axis.set_xticklabels([str(int(tick)) for tick in ticks_in_range], fontsize=TICK_FONT_SIZE)
    axis.xaxis.set_minor_formatter(NullFormatter())
    axis.xaxis.set_minor_locator(plt.NullLocator())

    # Vertical separators between low / medium / high sections.
    for threshold in (30, 90):
        if x_min < threshold < x_max:
            axis.axvline(threshold, color="#9a9a9a", linestyle="--", linewidth=0.9, zorder=1)

    # Section labels above the axes.
    text_transform = transforms.blended_transform_factory(axis.transData, axis.transAxes)
    sections = [
        ("low", r"$n < 30$", x_min, min(30.0, x_max)),
        ("medium", r"$30 \leq n < 90$", max(30.0, x_min), min(90.0, x_max)),
        ("high", r"$n \geq 90$", max(90.0, x_min), x_max),
    ]
    for label, detail, left, right in sections:
        if right <= left:
            continue
        center = float(np.sqrt(left * right)) if x_log_scale else (left + right) / 2.0
        axis.text(
            center,
            1.0,
            label, #+ f" ({detail})",
            transform=text_transform,
            ha="center",
            va="bottom",
            fontsize=FONT_SIZE-2,
            color="#444444",
            fontweight=W_SEMIBOLD,
            clip_on=False,
        )
        # axis.text(
        #     center,
        #     1.01,
        #     detail,
        #     transform=text_transform,
        #     ha="center",
        #     va="bottom",
        #     fontsize=8,
        #     color="#444444",
        #     fontweight=W_SEMIBOLD,
        #     clip_on=False,
        # )


def _plot_order_data(
    ax,
    *,
    index: str,
    order: int,
    game_type: str,
    input_path: str | None,
    budgets: list[int],
    palette: list[str],
) -> tuple[bool, list, list, np.ndarray]:
    """Draw one order's lines/fills into *ax*. Returns (has_data, handles, labels, all_x)."""
    df = _load_results(index=index, order=order, game_type=game_type, input_path=input_path)
    relative_df = _compute_relative_mse(df)

    all_x_vals: list[np.ndarray] = []
    has_data = False
    legend_handles: list = []
    legend_labels: list = []

    for b_idx, requested_budget in enumerate(budgets):
        color = palette[b_idx % len(palette)]
        min_budget = budgets[b_idx - 1] if b_idx > 0 else 0
        selected_rows, per_game_budget_map = _select_per_dataset_nearest_budget_rows(
            relative_df,
            requested_budget,
            min_budget=min_budget,
        )
        agg = _aggregate_by_n_players(selected_rows)
        all_x_vals.append(selected_rows["n_players"].to_numpy(dtype=float))

        for method in METHODS:
            method_df = agg[agg["approximator"] == method].sort_values("n_players")
            if method_df.empty:
                continue

            has_data = True
            style = STYLE[method]
            x_vals = method_df["n_players"].to_numpy()
            y_vals = method_df["mean_relative_mse"].to_numpy()
            std_vals = method_df["std"].to_numpy()
            lower = y_vals / std_vals
            upper = y_vals * std_vals

            _plot_with_white_outline(
                ax,
                x_vals,
                y_vals,
                color=color,
                marker=style["marker"],
                linestyle=ORDER_LINESTYLES.get(order, style["linestyle"]),
                linewidth=1,
                markersize=2.3,
                line_outline_size=2,
                marker_outline_size=2,
                zorder=2,
            )
            ax.fill_between(
                x_vals,
                lower,
                upper,
                color=color,
                alpha=0.12,
                linewidth=0.0,
            )

        if per_game_budget_map:
            unique_used = sorted(set(per_game_budget_map.values()))
            print(
                f"[Order {order}] Target budget {requested_budget}: "
                f"matched {len(per_game_budget_map)} datasets, "
                f"used budgets in [{unique_used[0]}, {unique_used[-1]}]"
            )

        #order_linestyle = "-" if order == 2 else "--"
        (h,) = ax.plot([], [], color=color, linestyle="-", linewidth=1.5)
        legend_handles.append(h)
        budget_k = requested_budget // 1000
        legend_labels.append(f"{budget_k}k")

    all_x = np.concatenate(all_x_vals) if all_x_vals else np.array([])
    return has_data, legend_handles, legend_labels, all_x


def plot_relative_mse_vs_n_players(
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
    _ytick_colors    = ["#2d6a4f"] * 3 + ["#555555"] + ["#9b2226"] * 3

    last_budget_labels: list[str] = []
    any_data_overall = False
    panel_x_values: list[np.ndarray] = []
    panel_has_data: list[bool] = []

    for order, ax in zip(orders, axes):
        ax.axhspan(0, 1.0, color="#4CAF50", alpha=0.04, zorder=0)
        ax.axhspan(1.0, 100, color="#E53935", alpha=0.04, zorder=0)
        ax.axhline(1.0, color="#555555", linestyle="--", linewidth=1.0, zorder=1)

        _sticker_tf = transforms.blended_transform_factory(ax.transAxes, ax.transData)
        _sticker_kw = dict(
            transform=_sticker_tf, ha="right", va="bottom",
            fontsize=FONT_SIZE-3, fontweight=W_SEMIBOLD, clip_on=False, zorder=8
        )
        ax.text(0.98, 0.21, "better", color="#2d6a4f", **_sticker_kw)
        ax.text(0.13, 4.02, "worse",  color="#9b2226", **_sticker_kw)
        ax.text(0.98, 1, "no effect", color="#555555", **_sticker_kw)

        palette = ORDER_PALETTES.get(order, BUDGET_COLORS)
        has_data, _, labels, all_x = _plot_order_data(
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
        ax.set_yticklabels(_ytick_labels, fontsize=TICK_FONT_SIZE)
        ax.yaxis.set_minor_locator(plt.NullLocator())
        if x_log_scale:
            ax.set_xscale("log")
        _configure_n_player_axis(ax, x_values=combined_x, x_log_scale=x_log_scale)
        ax.grid(True, which="both", axis="y", linestyle="--", linewidth=0.6, alpha=0.35)
        ax.margins(axis_margin)
        if x_min_global is not None and x_max_global is not None:
            if x_log_scale:
                pad = (x_max_global / x_min_global) ** axis_margin
                ax.set_xlim(x_min_global / pad, x_max_global * pad)
            else:
                pad = (x_max_global - x_min_global) * axis_margin
                ax.set_xlim(x_min_global - pad, x_max_global + pad)
        ax.set_xlabel("Number of Players", fontsize=FONT_SIZE, fontweight=W_SEMIBOLD, labelpad=xlabel_pad)
        if panel_idx == 0:
            ax.set_ylabel(
                "Adjustment effect on MSE",
                fontsize=FONT_SIZE, fontweight=W_SEMIBOLD, labelpad=ylabel_pad,
            )
        ax.set_title(
            f"Order {order}",
            fontsize=FONT_SIZE + 2,
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

    palette_used = ORDER_PALETTES.get(orders[0], BUDGET_COLORS)
    budget_lines = [
        Line2D([], [], color=palette_used[i % len(palette_used)], linestyle="-", linewidth=1.5)
        for i in range(len(budgets))
    ]
    legend_labels = last_budget_labels or [f"{b // 1000}k" for b in budgets]
    legend_handles: list = [_SectionHandle(), *budget_lines]
    legend_labels_full = ["Budget:", *legend_labels]

    fig.suptitle(title, fontsize=13, fontweight=W_REGULAR)

    fig.legend(
        legend_handles,
        legend_labels_full,
        loc="lower center",
        ncol=len(legend_handles),
        frameon=False,
        fontsize=FONT_SIZE,
        handlelength=2.0,
        handletextpad=0.3,
        columnspacing=0.6,
        bbox_to_anchor=(0.5, -0.02),
        handler_map={_SectionHandle: _SectionHandler()},
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
        description="Plot relative MSE over n_players for ProxySHAP XGBoost variants."
    )
    parser.add_argument("--index", type=str, default="SII")
    parser.add_argument("--orders", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--game-type", type=str, default="interventional")
    parser.add_argument("--input-path-order1", type=str, default=None)
    parser.add_argument("--input-path-order2", type=str, default=None)
    parser.add_argument("--input-path-order3", type=str, default=None)
    parser.add_argument("--output-path", type=str, default=None)
    parser.add_argument("--x-log-scale", action="store_true")
    parser.add_argument("--linear-x", action="store_true")
    parser.add_argument("--y-log-scale", action="store_true")
    parser.add_argument("--linear-y", action="store_true")
    parser.add_argument("--budgets", type=int, nargs="+", default=[1000, 5000,10000,20000,35000])
    parser.add_argument("--title", type=str, default=None)

    args = parser.parse_args()

    if args.output_path is None:
        order_str = "_".join(str(o) for o in args.orders)
        args.output_path = (
            f"plots/main/main_paper_plots_adjustment_benefit.pdf"
        )

    input_paths: dict[int, str | None] = {
        1: args.input_path_order1,
        2: args.input_path_order2,
        3: args.input_path_order3,
    }

    plot_relative_mse_vs_n_players(
        index=args.index,
        orders=args.orders,
        game_type=args.game_type,
        input_paths=input_paths,
        output_path=args.output_path,
        budgets=args.budgets,
        x_log_scale=True,
        title=args.title,
        spacing=SPACING,
    )
