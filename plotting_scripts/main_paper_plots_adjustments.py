"""Plot combined main-paper figures with a shared legend and labels.

Use this script by editing the `plot_specs` list near the bottom of the file.
Each entry defines one subplot via `game_name`, `order`, `index`, `game_type`,
`ylim`, and `max_budget`. The order of entries in `plot_specs` is the order in
which the panels are placed in the combined figure.

The figure size is derived automatically from `panel_figsize` unless you pass
an explicit `figsize` override to `plot_main_paper_plots`. You can also adjust
`legend_bottom_margin`, `shared_xlabel_y`, `subplot_wspace`, and
`subplot_hspace` if legend/label placement or subplot spacing needs manual
tuning.

The `shared_xlabel_y` parameter controls the vertical position of the shared
figure x-label. Smaller values move it closer to the subplot row; larger values
move it farther down.

Run the script directly with:

    uv run python special_plot_scripts/main_paper_plots_adjustments.py
"""

from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import patheffects as pe
from matplotlib.lines import Line2D

import shapiq_benchmark.plot as benchmark_plot
from shapiq_benchmark.plot import plot_approximation_quality

from _plot_style import apply_tick_style, setup_fonts, W_REGULAR, W_SEMIBOLD

setup_fonts()


def _build_outlined_legend_handles(handles, labels):
    """Create legend handles that mimic the white-highlighted plot style."""
    outlined_handles = []
    for handle, label in zip(handles, labels, strict=True):
        if not isinstance(handle, Line2D):
            outlined_handles.append(handle)
            continue

        style = benchmark_plot.STYLE_DICT.get(label, {})
        color = style.get("color", handle.get_color())
        marker = style.get("marker", handle.get_marker())
        linestyle = style.get("linestyle", handle.get_linestyle())
        linewidth = handle.get_linewidth()
        markersize = handle.get_markersize()
        legend_markersize = max(3.0, markersize * 0.72)
        has_marker = marker not in {None, "None", "", " "}

        legend_handle = Line2D(
            [],
            [],
            color=color,
            linestyle=linestyle,
            marker=marker,
            linewidth=linewidth,
            markersize=legend_markersize,
            markerfacecolor=color if has_marker else "none",
            markeredgecolor=color if has_marker else "none",
            markeredgewidth=0.0 if not has_marker else 1.1,
        )
        legend_handle.set_path_effects(
            [pe.Stroke(linewidth=linewidth + 1.8, foreground="white"), pe.Normal()]
        )
        outlined_handles.append(legend_handle)

    return outlined_handles


def _prepare_results_df(
    game_name,
    order,
    index,
    game_type,
    approximators_to_plot,
    min_budget=0,
    max_budget=float("inf"),
):
    results_df = pd.read_csv(
        f"icml_submission_data/results_benchmark_{index}_{order}_{game_type}.csv"
    )
    results_df = results_df.replace({"approximator": APPROXIMATOR_RENAMING})
    results_df = results_df[results_df["approximator"].isin(approximators_to_plot)]
    results_df = results_df[results_df["game"] == game_name]
    print("Available methods:", results_df["approximator"].unique())

    n_players = results_df["n_players"].values[0]
    min_b = n_players + 1 if n_players < 1000 else 101
    max_b = min(2**n_players, max_budget) if n_players <= 20 else max_budget
    budget_range = (
        np.ceil(np.logspace(np.log10(min_b), np.log10(max_b), 20))
        .clip(min_b, max_b)
        .astype(int)
    )
    results_df = results_df[
        results_df["budget"].isin(budget_range)
        & (results_df["budget"] >= min_budget)
        & (results_df["budget"] <= max_budget)
    ]

    for method in LINEAR_METHODS:
        method_mask = results_df["approximator"] == method
        valid_budgets = []
        for b in results_df[method_mask]["budget"].unique():
            if b >= sum([math.comb(n_players + 1, i) for i in range(0, order + 1)]):
                valid_budgets.append(b)
        results_df = results_df[
            ~(method_mask & (~results_df["budget"].isin(valid_budgets)))
        ]

    for method in ["KernelSHAPIQ"]:
        method_mask = results_df["approximator"] == method
        valid_budgets = []
        for b in results_df[method_mask]["budget"].unique():
            if b >= math.comb(n_players + 1, order):
                valid_budgets.append(b)
        results_df = results_df[
            ~(method_mask & (~results_df["budget"].isin(valid_budgets)))
        ]

    return results_df


def _plot_panel_on_axis(
    ax,
    results_df,
    game_name,
    order,
    index,
    game_type,
    figsize,
    ylim,
    y_log_scale,
    x_log_scale,
    marker_size,
    linewidth,
    highlight_size,
):
    original_subplots = benchmark_plot.plt.subplots
    original_tight_layout = benchmark_plot.plt.tight_layout

    try:
        benchmark_plot.plt.subplots = lambda *args, **kwargs: (ax.figure, ax)
        benchmark_plot.plt.tight_layout = lambda *args, **kwargs: None
        plot_approximation_quality(
            data=results_df,
            metric="MSE",
            log_scale_y=y_log_scale,
            log_scale_x=x_log_scale,
            figsize=figsize,
            log_scale_min=ylim[0] if y_log_scale else None,
            log_scale_max=ylim[1] if y_log_scale else None,
            legend=True,
            marker_size=marker_size,
            linewidth=linewidth,
            highlight_size=highlight_size,
            confidence_metric="sem",
            plot_labels=False,
        )
    finally:
        benchmark_plot.plt.subplots = original_subplots
        benchmark_plot.plt.tight_layout = original_tight_layout

    legend = ax.get_legend()
    if legend is not None:
        legend.remove()

    ax.set_title(DATA_NAMES[game_name], fontsize=TITLE_FONT_SIZE, fontweight=W_REGULAR)
    ax.text(
        0.03,
        0.04,
        f"order {order}",
        transform=ax.transAxes,
        fontsize=18,
        fontweight=W_SEMIBOLD,
        ha="left",
        va="bottom",
    )


def plot_main_paper_plots(
    plots,
    approximators_to_plot,
    panel_figsize=(4, 3),
    figsize=None,
    legend_bottom_margin=None,
    shared_xlabel_y=None,
    subplot_wspace=0.2,
    subplot_hspace=0.24,
    ylim=(1e-7, 1e2),
    y_log_scale=True,
    x_log_scale=True,
    min_budget=0,
    max_budget=float("inf"),
    marker_size=6,
    linewidth=2,
    highlight_size=2,
    shared_xlabel="Model Evaluations",
    shared_xlabel_fontsize=12,
    axis_ylabel="MSE",
    axis_ylabel_fontsize=12,
    shared_ylabel_x=None,
    shared_ylabel_fontsize=12,
    output_path="plots/main/main_paper_plots_adjustments_combined.pdf",
):
    num_plots = len(plots)
    if num_plots <= 4:
        n_rows, n_cols = 1, num_plots
    else:
        n_cols = 4
        n_rows = math.ceil(num_plots / n_cols)
    if figsize is None:
        panel_width, panel_height = panel_figsize
        width = (panel_width * n_cols) + (0.55 * max(n_cols - 1, 0)) + 0.5
        height = (panel_height * n_rows) + (0.65 if n_rows > 1 else 0.25)
        if n_rows > 1:
            height += 0.7
        figsize = (width, height)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes_flat = np.atleast_1d(axes).reshape(-1)
    legend_handles = None
    legend_labels = None

    for axis, plot_spec in zip(axes_flat, plots, strict=True):
        results_df = _prepare_results_df(
            game_name=plot_spec["game_name"],
            order=plot_spec["order"],
            index=plot_spec["index"],
            game_type=plot_spec["game_type"],
            approximators_to_plot=approximators_to_plot,
            min_budget=min_budget,
            max_budget=plot_spec["max_budget"],
        )
        _plot_panel_on_axis(
            axis,
            results_df,
            game_name=plot_spec["game_name"],
            order=plot_spec["order"],
            index=plot_spec["index"],
            game_type=plot_spec["game_type"],
            figsize=plot_spec.get("figsize", (4, 3)),
            ylim=plot_spec["ylim"],
            y_log_scale=y_log_scale,
            x_log_scale=x_log_scale,
            marker_size=marker_size,
            linewidth=linewidth,
            highlight_size=highlight_size,
        )

        if legend_handles is None:
            handles, labels = axis.get_legend_handles_labels()
            filtered = [
                (handle, label)
                for handle, label in zip(handles, labels, strict=True)
                if label.startswith("ProxySHAP")
                or label.startswith("ProxySHAP+")
                or label.startswith("ProxySHAP*")
            ]

            legend_priority = {
                "ProxySHAP (XGBoost, MSR) [our]": 1,
            }
            filtered.sort(key=lambda item: (legend_priority.get(item[1], 10), item[1]))

            legend_handles = [handle for handle, _ in filtered]
            legend_labels = [label for _, label in filtered]

    for axis_index, axis in enumerate(axes_flat):
        if axis_index >= num_plots:
            axis.set_visible(False)
            continue
        col = axis_index % n_cols
        if shared_ylabel_x is None and col == 0:
            axis.set_ylabel(axis_ylabel, fontsize=axis_ylabel_fontsize)
        else:
            axis.set_ylabel("")
        axis.set_xlabel("")

    legend_ncol = 2
    if legend_bottom_margin is None:
        legend_rows = math.ceil(len(legend_labels) / legend_ncol) if legend_labels else 1
        legend_bottom_margin = 0.045 + (0.03 * legend_rows)
    bottom_margin = 0.12 + legend_bottom_margin + (0.04 if n_rows == 1 else 0.08)

    fig.legend(
        _build_outlined_legend_handles(legend_handles, legend_labels),
        legend_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -legend_bottom_margin),
        ncol=legend_ncol,
        frameon=False,
        fancybox=True,
        framealpha=1,
    )
    if shared_xlabel_y is None:
        shared_xlabel_y = 0.055 if n_rows == 1 else 0.045
    fig.supxlabel(shared_xlabel, fontsize=shared_xlabel_fontsize, y=shared_xlabel_y, fontweight=W_SEMIBOLD)
    if shared_ylabel_x is not None:
        fig.supylabel(axis_ylabel, fontsize=shared_ylabel_fontsize, x=shared_ylabel_x, fontweight=W_SEMIBOLD)
    fig.subplots_adjust(
        bottom=bottom_margin,
        wspace=subplot_wspace,
        hspace=subplot_hspace,
    )
    apply_tick_style(*axes_flat[:num_plots])
    fig.savefig(output_path)


TITLE_FONT_SIZE = 18
BASE_GENERAL_PARAMS = {
    "panel_figsize": (4, 3),
    "figsize": None,
    "legend_bottom_margin": None,
    "shared_xlabel_y": None,
    "subplot_wspace": 0.13,
    "subplot_hspace": 0.24,
    "ylim": (1e-7, 1e2),
    "y_log_scale": True,
    "x_log_scale": True,
    "marker_size": 4,
    "linewidth": 2,
    "highlight_size": 2,
    "min_budget": 14,
    "max_budget": 35_000,
    "shared_xlabel": "Model Evaluations",
    "shared_xlabel_fontsize": 12,
    "axis_ylabel": "MSE",
    "axis_ylabel_fontsize": 12,
    "shared_ylabel_x": None,
    "shared_ylabel_fontsize": 12,
}
DATA_NAMES = {
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
    "SentimentAnalysisLocalXAI": "DistilBERT ($d=14$)",
    "SOUM": "soum",
    "SOUM10k": "soum10k",
    "SOUM100k": "soum100k",
    "MicroresponseLocalXAI": "Microresponse ($d=1300$)",
    "LeukemiaLocalXAI": "Leukemia ($d=7129$)",
    "BioresponseLocalXAI": "Bioresponse ($d=1776$)",
    "AmazonLocalXAI": "Amazon ($d=10000$)",
}
APPROXIMATORS_TO_PLOT = [
    # "SHAPIQ",
    # "SVARMIQ",
    # "PermutationSamplingSV",
    # "PermutationSamplingSII",
    # "KernelSHAPIQ",
    # "ProxySpex",
    # Our Methods
    "ProxySHAP (XGBoost, MSR) [our]",
    #"ProxySHAP (XGBoost) [our]",
    ## DEFAULT METHODS ##
    # "ProxySHAP+ (XGBoost, MSR) [our]",
    # "ProxySHAP+ (XGBoost) [our]",
    ## HPO METHODS ##
    # "ProxySHAP* (XGBoost, MSR) [our]",
    # "ProxySHAP* (XGBoost) [our]",
    ## LINEAR METHODS ##
    # "ProxySHAP (Linear, MSR) [our]",
    # "ProxySHAP (Linear) [our]",
]

LINEAR_METHODS = [
    "ProxySHAP (Linear, MSR) [our]",
    "ProxySHAP (Linear) [our]",
]
APPROXIMATOR_RENAMING = {
    # "PermutationSamplingSV": "Permutation Sampling (SV)",
    # "PermutationSamplingSII": "Permutation Sampling (SII)",
    "RegressionMSRIQ-NoAdjustment": "ProxySHAP (XGBoost) [our]",
    "RegressionMSRIQ": "ProxySHAP (XGBoost, MSR) [our]",
    ## LINEAR METHODS ##
    "Linear-NoAdjustment": "ProxySHAP (Linear) [our]",
    "Linear-RECAP": "ProxySHAP (Linear, MSR) [our]",
    ## DEFAULT METHODS ##
    "RegressionMSRIQ-XGB-PreDef-NoAdjustment": "ProxySHAP+ (XGBoost) [our]",
    "RegressionMSRIQ-XGB-PreDef": "ProxySHAP+ (XGBoost, MSR) [our]",
    ## HPO METHODS ##
    "RegressionMSRIQ-XGB-CV-BO": "ProxySHAP* (XGBoost, MSR) [our]",
    "RegressionMSRIQ-XGB-CV-BO-NoAdjustment": "ProxySHAP* (XGBoost) [our]",
}

if __name__ == "__main__":
    # If you render a single row of panels, `subplot_hspace` has no visible effect.
    # Use `subplot_wspace` to reduce horizontal spacing between panels.
    general_params = {
        **BASE_GENERAL_PARAMS,
    }
    plot_specs = [
        {
            "game_name": "CommunitiesAndCrimeLocalXAI",
            "order": 1,
            "index": "SV",
            "game_type": "interventional",
            "ylim": (10**(-1), 10 ** (2.4)),
            "max_budget": 35_000,
        },
        {
            "game_name": "CommunitiesAndCrimeLocalXAI",
            "order": 2,
            "index": "SII",
            "game_type": "interventional",
            "ylim": (10**(-0.7), 10 ** (1.6)),
            "max_budget": 35_000,
        },
        {
            "game_name": "CommunitiesAndCrimeLocalXAI",
            "order": 3,
            "index": "SII",
            "game_type": "interventional",
            "ylim": (1e-1, 10 ** (1.4)),
            "max_budget": 35_000,
        },
    ]
    plot_main_paper_plots(
        plots=plot_specs,
        approximators_to_plot=APPROXIMATORS_TO_PLOT,
        **general_params,
    )
