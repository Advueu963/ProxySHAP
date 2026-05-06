"""This file plots the figures in the style of the main paper."""

import math
import os

import numpy as np
from matplotlib import patheffects as pe
from matplotlib.lines import Line2D
from matplotlib import pyplot as plt
import pandas as pd

import shapiq_benchmark.plot as benchmark_plot
from shapiq_benchmark.plot import (
    plot_approximation_quality_vstime,
)

from _plot_style import apply_tick_style, setup_fonts, W_REGULAR, W_SEMIBOLD

setup_fonts()

STYLE_DICT: dict[str, dict[str, str]] = {
    # permutation sampling
    "PermutationSamplingSII": {
        "color": "#252525",
        "marker": None,
    },
    "PermutationSamplingSTII": {
        "color": "#252525",
        "marker": "o",
    },
    "PermutationSamplingSV": {
        "color": "#252525",
        "marker": "o",
    },
    # KernelSHAP-IQ
    "KernelSHAP": {"color": "#ff6f00", "marker": None, "linestyle": (0, (5, 1))},
    "KernelSHAPIQ": {"color": "#ff6f00", "marker": "o"},
    # inconsistent KernelSHAP-IQ
    "InconsistentKernelSHAPIQ": {"color": "#ffba08", "marker": "o"},
    "kADDSHAP": {"color": "#ffba08", "marker": "o"},
    # SVARM-based
    "SVARMIQ": {"color": "#707070", "marker": "o"},
    "SVARM": {"color": "#00b4d8", "marker": "o"},
    # shapiq
    "SHAPIQ": {"color": "#959595", "marker": "o"},
    "UnbiasedKernelSHAP": {"color": "#ef27a6", "marker": "o"},
    # misc SV
    "OwenSamplingSV": {"color": "#7DCE82", "marker": "o"},
    "StratifiedSamplingSV": {"color": "#4B7B4E", "marker": "o"},
    # Regression MSR
    "ProxySpex": {"color": "#ef27a6", "marker": "o", "linestyle": "solid"},
    "ProxySHAP (Linear, MSR) [our]": {
        "color": "#15B01A",
        "marker": "^",
        "linestyle": "solid",
    },
    "ProxySHAP (Linear) [our]": {
        "color": "#15B01A",
        "marker": "s",
        "linestyle": "solid",
    },
    "ProxySHAP+ (XGBoost, MSR) [our]": {
        "color": "#1e25e5",
        "marker": "^",
        "linestyle": "solid",
    },
    "ProxySHAP+ (XGBoost) [our]": {
        "color": "#1e25e5",
        "marker": "s",
        "linestyle": "solid",
    },
    "ProxySHAP* (XGBoost, MSR) [our]": {
        "color": "#06C2AC",
        "marker": "o",
        "linestyle": "solid",
    },
    "ProxySHAP* (XGBoost) [our]": {
        "color": "#06C2AC",
        "marker": "s",
        "linestyle": "solid",
    },
    "ProxySHAP (XGBoost) [our]": {
        "color": "#1e88e5",
        "marker": "s",
        "linestyle": "solid",
    },
    "ProxySHAP (XGBoost, MSR) [our]": {
        "color": "#1e88e5",
        "marker": "^",
        "linestyle": "solid",
    },
    # "ProxySHAP (LGBM FixedHPO Det. MSR) [our]": {"color": "#fd3232", "marker": "o"},
    # "KernelSHAP": {"color": "#d62728", "marker": "o"},
    # "LeverageSHAP": {"color": "#009688", "marker": "o"},
}

APPROXIMATOR_TO_ZORDER = {
    "ProxySHAP* (XGBoost, MSR) [our]": 7,
    "ProxySHAP* (XGBoost) [our]": 7,
    "ProxySHAP (XGBoost, MSR) [our]": 7,
    "ProxySHAP (XGBoost) [our]": 7,
    "KernelSHAPIQ": 5,
    "ProxySpex": 5,
}
APPROXIMATOR_TO_ALPHA = {
    "ProxySHAP* (XGBoost, MSR) [our]": 1.0,
    "ProxySHAP* (XGBoost) [our]": 1.0,
    "ProxySHAP (XGBoost, MSR) [our]": 1.0,
    "ProxySHAP (XGBoost) [our]": 1.0,
    "KernelSHAPIQ": 1,
    "ProxySpex": 1,
}


def _load_and_prepare_results_df(
    *,
    game_name: str,
    order: int,
    index: str,
    game_type: str,
    approximators_to_plot: list[str],
    max_budget: float,
    time_step: float | None = None,
) -> pd.DataFrame:
    if time_step is None:
        time_step = float(globals().get("TIME", 0.0))

    results_df = pd.read_csv(
        f"icml_submission_data/results_benchmark_{index}_{order}_{game_type}_with_hpo.csv"
    )
    df2_proxyspex_runtime = pd.read_csv(
        f"icml_submission_data/results_benchmark_{index}_{order}_{game_type}_test.csv"
    )
    df2_proxyspex_runtime = df2_proxyspex_runtime[df2_proxyspex_runtime["approximator"] == "ProxySPEX (XGBoost)"]
    results_df = pd.concat([results_df, df2_proxyspex_runtime], ignore_index=True)
    results_df = results_df.replace({"approximator": APPROXIMATOR_RENAMING})
    msr_mask = results_df["approximator"] == "ProxySHAP* (XGBoost, MSR) [our]"
    runtime_map = results_df[msrb_mask].groupby(["game_id", "budget"])["total_runtime"].mean()
    results_df.loc[msr_mask, "total_runtime"] = (
        results_df[msr_mask].set_index(["game_id", "budget"]).index.map(runtime_map).values
    )
    results_df = results_df[results_df["approximator"].isin(approximators_to_plot)]
    results_df = results_df[(results_df["game"] == game_name)]

    results_df["total_runtime"] = (
        results_df["total_runtime"]
        - results_df["evaluations"]
        + time_step * results_df["used_budget"]
    ).round(1)
    results_df.sort_values(by=["total_runtime"], inplace=True)

    for (game_id, approximator, budget), group_data in results_df.groupby(
        ["game_id", "approximator", "budget"]
    ):
        mean_time = group_data["total_runtime"].mean()
        results_df.loc[
            (results_df["game_id"] == game_id)
            & (results_df["approximator"] == approximator)
            & (results_df["budget"] == budget),
            "total_runtime",
        ] = mean_time

    if results_df.empty:
        return results_df

    n_players = int(results_df["n_players"].values[0])
    min_b = n_players + 1 if n_players < 1000 else 101
    max_b = min(2**n_players, max_budget) if n_players <= 20 else max_budget
    budget_range = (
        np.ceil(np.logspace(np.log10(min_b), np.log10(max_b), 20))
        .clip(min_b, max_b)
        .astype(int)
    )
    results_df = results_df[results_df["budget"].isin(budget_range)]

    # Filter out those budgets for Linear and KernelSHAPIQ that are invalid
    for method in LINEAR_METHODS:
        method_mask = results_df["approximator"] == method
        valid_budgets: list[int] = []
        for b in results_df[method_mask]["budget"].unique():
            if b >= sum([math.comb(n_players + 1, i) for i in range(0, order + 1)]):
                valid_budgets.append(b)
        results_df = results_df[
            ~(method_mask & (~results_df["budget"].isin(valid_budgets)))
        ]
    for method in ["KernelSHAPIQ"]:
        method_mask = results_df["approximator"] == method
        valid_budgets: list[int] = []
        for b in results_df[method_mask]["budget"].unique():
            if b >= math.comb(n_players + 1, order):
                valid_budgets.append(b)
        results_df = results_df[
            ~(method_mask & (~results_df["budget"].isin(valid_budgets)))
        ]

    return results_df


def _build_runtime_legend_handles(
    handles: list,
    labels: list[str],
    *,
    legend_scale: float = 1.0,
) -> tuple[list, list[str]]:
    """Create legend handles with the same outlined look as the main plots."""

    outlined_handles: list = []
    outlined_labels: list[str] = []
    seen_labels: set[str] = set()

    for handle, label in zip(handles, labels, strict=True):
        if label in seen_labels:
            continue
        seen_labels.add(label)

        if not isinstance(handle, Line2D):
            outlined_handles.append(handle)
            outlined_labels.append(label)
            continue

        style = benchmark_plot.STYLE_DICT.get(label, {})
        color = style.get("color", handle.get_color())
        marker = style.get("marker", handle.get_marker())
        linestyle = style.get("linestyle", handle.get_linestyle())
        linewidth = handle.get_linewidth() * legend_scale
        markersize = handle.get_markersize()
        legend_markersize = max(3.0, markersize * 0.72 * legend_scale)
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
            markeredgewidth=0.0 if not has_marker else 1.1 * legend_scale,
        )
        legend_handle.set_path_effects(
            [
                pe.Stroke(linewidth=linewidth + (1.8 * legend_scale), foreground="white"),
                pe.Normal(),
            ]
        )
        outlined_handles.append(legend_handle)
        outlined_labels.append(label)

    return outlined_handles, outlined_labels


def _order_runtime_legend_two_rows(
    handles: list,
    labels: list[str],
) -> tuple[list, list[str]]:
    """Order legend entries for a fixed 2x5 layout (column-wise fill)."""

    # Matplotlib fills multi-column legends column-wise.
    desired_pairs: list[tuple[str, str]] = [
        ("PermutationSamplingSII", "SHAPIQ"),
        ("KernelSHAPIQ", "ProxySPEX (XGBoost)"),
        ("ProxySHAP (XGBoost) [our]", "ProxySHAP (XGBoost, MSR) [our]"),
        ("ProxySHAP (Linear) [our]", "ProxySHAP (Linear, MSR) [our]"),
        ("ProxySHAP* (XGBoost) [our]", "ProxySHAP* (XGBoost, MSR) [our]"),
    ]
    display_name_map: dict[str, str] = {
        "PermutationSamplingSII": "PermutationSamplingSII",
        "SHAPIQ": "SHAPIQ",
        "KernelSHAPIQ": "KernelSHAPIQ",
        "ProxySPEX (XGBoost)": "ProxySPEX (XGBoost)",
        "ProxySHAP (XGBoost) [our]": "ProxySHAP (XGBoost) [our]",
        "ProxySHAP (XGBoost, MSR) [our]": "ProxySHAP (XGBoost, MSR) [our]",
        "ProxySHAP (Linear) [our]": "ProxySHAP (Linear) [our]",
        "ProxySHAP (Linear, MSR) [our]": "ProxySHAP (Linear, MSR) [our]",
        "ProxySHAP* (XGBoost) [our]": "ProxySHAP (XGBoost+HPO) [our]",
        "ProxySHAP* (XGBoost, MSR) [our]": "ProxySHAP (XGBoost+HPO, MSR) [our]",
    }

    by_label = {label: handle for handle, label in zip(handles, labels, strict=True)}
    ordered_handles: list = []
    ordered_labels: list[str] = []

    for top_label, bottom_label in desired_pairs:
        if top_label in by_label:
            ordered_handles.append(by_label[top_label])
            ordered_labels.append(display_name_map.get(top_label, top_label))
        if bottom_label in by_label:
            ordered_handles.append(by_label[bottom_label])
            ordered_labels.append(display_name_map.get(bottom_label, bottom_label))

    return ordered_handles, ordered_labels


def _plot_runtime_panel_on_axis(
    ax,
    *,
    results_df: pd.DataFrame,
    game_name: str,
    time_step: float,
    order: int,
    ylim: tuple[float, float],
    y_log_scale: bool,
    x_log_scale: bool,
    figsize: tuple[float, float],
    marker_size: float,
    linewidth: float,
    highlight_size: float,
    show_caption: bool = True,
    show_x_ticks: bool = True,
) -> tuple[list, list[str]]:
    """Render one runtime subplot into an existing axis."""

    original_subplots = benchmark_plot.plt.subplots
    original_tight_layout = benchmark_plot.plt.tight_layout

    try:
        benchmark_plot.plt.subplots = lambda *args, **kwargs: (ax.figure, ax)
        benchmark_plot.plt.tight_layout = lambda *args, **kwargs: None
        plot_approximation_quality_vstime(
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
        )
    finally:
        benchmark_plot.plt.subplots = original_subplots
        benchmark_plot.plt.tight_layout = original_tight_layout

    legend = ax.get_legend()
    if legend is not None:
        legend.remove()

    runtime_min = results_df[results_df["approximator"].str.contains("ProxySHAP")][
        "total_runtime"
    ].min()
    runtime_max = results_df[results_df["approximator"].str.contains("ProxySHAP")][
        "total_runtime"
    ].max()
    if np.isfinite(runtime_min) and np.isfinite(runtime_max):
        ax.set_xlim(runtime_min * 0.8, runtime_max * 1.2)

    if show_caption:
        ax.set_title(DATA_NAMES[game_name], fontsize=TITLE_FONT_SIZE, fontweight=W_REGULAR)
    else:
        ax.set_title("")

    if not show_x_ticks:
        ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)

    ax.text(
        0.03,
        0.04,
        f"order {order}\nt = {time_step * 1000:g} ms",
        transform=ax.transAxes,
        fontsize=14,
        fontweight=W_SEMIBOLD,
        ha="left",
        va="bottom",
    )

    handles, labels = ax.get_legend_handles_labels()
    return handles, labels


def plot_combined_runtime_plots(
    *,
    plot_specs: list[dict],
    approximators_to_plot: list[str],
    output_path: str,
    legend_output_path: str | None = None,
    plots_per_row: int = 2,
    panel_figsize: tuple[float, float] = (4, 3),
    figsize: tuple[float, float] | None = None,
    legend_bottom_margin: float | None = None,
    bottom_margin: float | None = None,
    shared_ylabel_x: float | None = None,
    shared_xlabel_y: float | None = None,
    subplot_wspace: float = 0.2,
    subplot_hspace: float = 0.24,
    shared_ylabel_fontsize: int = 12,
    shared_xlabel_fontsize: int = 12,
    tick_fontsize: int = 10,
    legend_scale: float = 1.0,
    show_captions_only_first_row: bool = True,
    caption_row_period: int | None = None,
    x_ticks_only_last_row: bool = True,
    y_log_scale: bool = True,
    x_log_scale: bool = True,
    marker_size: float = 6,
    linewidth: float = 2,
    highlight_size: float = 2,
    shared_xlabel: str = "Total Runtime (s)",
    shared_ylabel: str = "Relative MSE (ProxySHAP)",
) -> None:
    """Build one combined runtime figure from multiple panel specifications."""

    if plots_per_row < 1:
        raise ValueError("plots_per_row must be at least 1")
    if len(plot_specs) == 0:
        raise ValueError("plot_specs must not be empty")

    num_plots = len(plot_specs)
    n_cols = min(num_plots, plots_per_row)
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

    last_row_start_index = max(0, (n_rows - 1) * n_cols)

    legend_handles = None
    legend_labels = None

    for axis_index, (axis, plot_spec) in enumerate(
        zip(axes_flat[:num_plots], plot_specs, strict=True)
    ):
        results_df = _load_and_prepare_results_df(
            game_name=plot_spec["game_name"],
            order=plot_spec["order"],
            index=plot_spec["index"],
            game_type=plot_spec["game_type"],
            approximators_to_plot=approximators_to_plot,
            max_budget=plot_spec.get("max_budget", float("inf")),
            time_step=plot_spec["time_step"],
        )
        if caption_row_period is not None:
            show_caption = (axis_index // n_cols) % caption_row_period == 0
        else:
            show_caption = not show_captions_only_first_row or axis_index < n_cols
        show_x_ticks = not x_ticks_only_last_row or axis_index >= last_row_start_index
        handles, labels = _plot_runtime_panel_on_axis(
            axis,
            results_df=results_df,
            game_name=plot_spec["game_name"],
            time_step=plot_spec["time_step"],
            order=plot_spec["order"],
            ylim=plot_spec["ylim"],
            y_log_scale=y_log_scale,
            x_log_scale=x_log_scale,
            figsize=plot_spec.get("figsize", panel_figsize),
            marker_size=marker_size,
            linewidth=linewidth,
            highlight_size=highlight_size,
            show_caption=show_caption,
            show_x_ticks=show_x_ticks,
        )

        if legend_handles is None:
            legend_handles, legend_labels = _build_runtime_legend_handles(handles, labels)

        axis.set_xlabel("")
        axis.set_ylabel("")

    for axis_index, axis in enumerate(axes_flat):
        if axis_index >= num_plots:
            axis.set_visible(False)

    if legend_handles is None or legend_labels is None:
        raise ValueError("No legend entries were produced for the combined figure")

    legend_ncol = 5
    if bottom_margin is None:
        bottom_margin = 0.08
    if shared_xlabel_y is None:
        shared_xlabel_y = 0.04 if n_rows > 1 else 0.055
    if shared_ylabel_x is None:
        shared_ylabel_x = 0.015 if n_cols == 1 else 0.008

    fig.supylabel(shared_ylabel, fontsize=shared_ylabel_fontsize, x=shared_ylabel_x, fontweight=W_SEMIBOLD)
    fig.supxlabel(shared_xlabel, fontsize=shared_xlabel_fontsize, y=shared_xlabel_y, fontweight=W_SEMIBOLD)
    fig.subplots_adjust(bottom=bottom_margin, wspace=subplot_wspace, hspace=subplot_hspace)

    if legend_output_path is not None:
        ordered_handles, ordered_labels = _order_runtime_legend_two_rows(
            legend_handles,
            legend_labels,
        )
        outlined_handles, outlined_labels = _build_runtime_legend_handles(
            ordered_handles,
            ordered_labels,
            legend_scale=legend_scale,
        )
        fig_leg = plt.figure(figsize=(12.0, 2.6))
        legend = fig_leg.legend(
            outlined_handles,
            outlined_labels,
            loc="center",
            ncol=legend_ncol,
            frameon=True,
            fancybox=True,
            framealpha=1,
            edgecolor="none",
        )
        fig_leg.canvas.draw()
        legend_bbox = legend.get_window_extent(fig_leg.canvas.get_renderer())
        legend_bbox_inches = legend_bbox.transformed(fig_leg.dpi_scale_trans.inverted())
        fig_leg.savefig(legend_output_path, bbox_inches=legend_bbox_inches, pad_inches=0)
        plt.close(fig_leg)
    else:
        ordered_handles, ordered_labels = _order_runtime_legend_two_rows(
            legend_handles,
            legend_labels,
        )
        outlined_handles, outlined_labels = _build_runtime_legend_handles(
            ordered_handles,
            ordered_labels,
            legend_scale=legend_scale,
        )
        legend = fig.legend(
            outlined_handles,
            outlined_labels,
            loc="lower center",
            ncol=legend_ncol,
            frameon=True,
            fancybox=True,
            framealpha=1,
            edgecolor="none",
        )
    apply_tick_style(*axes_flat[:num_plots])
    for axis in axes_flat[:num_plots]:
        axis.tick_params(axis="both", which="major", labelsize=tick_fontsize)
    fig.savefig(output_path)


def plot_grouped_scatter_relative_mse(
    *,
    configs: list[dict] | None = None,
    game_names: list[str] | None = None,
    order: int | None = None,
    index: str | None = None,
    game_type: str | None = None,
    approximators_to_plot: list[str],
    max_budget: float = float("inf"),
    metric: str = "MSE",
    figsize: tuple[float, float] = (6.0, 2.6),
    marker_size: float = 8.0,
    y_log_scale: bool = True,
    y_lim: tuple[float, float] | None = (10**-1, 10**6),
    output_path: str = "plots/main/grouped_scatter_relative_mse.pdf",
    legend_output_path: (
        str | None
    ) = "plots/main/grouped_scatter_relative_mse_legend.pdf",
    legend_ncol: int = 1,
) -> None:
    """Grouped scatter plot where each dataset contributes repeated x-axis triplets.

    X positions are categorical-like: for each dataset we plot the same time steps
    (e.g. 100, 1000, 10000) again, so the x-axis looks like:
    100, 1000, 10000, 100, 1000, 10000, ...
    """

    if configs is None:
        if game_names is None or len(game_names) == 0:
            raise ValueError("Provide either configs or a non-empty game_names")
        if order is None or index is None or game_type is None:
            raise ValueError(
                "order/index/game_type must be provided when using game_names"
            )
        configs = [
            {
                "game_name": g,
                "order": order,
                "index": index,
                "game_type": game_type,
                "max_budget": max_budget,
                "time_steps": [10, 100, 1_000],
            }
            for g in game_names
        ]
    if len(configs) == 0:
        raise ValueError("configs must not be empty")

    fig, ax = plt.subplots(figsize=figsize)

    x_positions: list[int] = []
    x_labels: list[str] = []
    group_centers: list[float] = []

    seen_labels: set[str] = set()
    points_per_group = len(configs[0].get("time_steps"))

    # Reference line: baseline performance
    ax.axhline(1.0, color="#444444", linestyle="--", linewidth=1.0, zorder=0)

    for dataset_idx, cfg in enumerate(configs):
        game_name = cfg["game_name"]
        results_df = _load_and_prepare_results_df(
            game_name=game_name,
            order=cfg["order"],
            index=cfg["index"],
            game_type=cfg["game_type"],
            approximators_to_plot=approximators_to_plot,
            max_budget=cfg.get("max_budget", max_budget),
        )
        if results_df.empty:
            # Still reserve x-axis slots so the layout matches the requested pattern.
            for j, t in enumerate(cfg.get("time_steps")):
                x = dataset_idx * points_per_group + j
                x_positions.append(x)
                x_labels.append(str(int(t)))
            group_centers.append(
                dataset_idx * points_per_group + (points_per_group - 1) / 2
            )
            continue

        for time_idx, time_step in enumerate(cfg.get("time_steps")):
            x = dataset_idx * points_per_group + time_idx
            x_positions.append(x)
            x_labels.append(str(int(time_step)))
            df_time = results_df[
                (results_df["total_runtime"] >= time_step * 0.8)
                & (results_df["total_runtime"] <= time_step * 1.2)
            ]
            if df_time.empty:
                continue

            baseline_val = df_time[df_time["approximator"] == baseline_approximator][
                metric
            ].mean()
            if not np.isfinite(baseline_val) or baseline_val <= 0:
                continue

            for approximator, group_data in df_time.groupby("approximator"):
                # print(
                #     "Approximator,",
                #     approximator,
                #     "budgets:",
                #     group_data["budget"].unique(),
                # )
                # if approximator == baseline_approximator:
                #     # Baseline is always exactly 1.0 by construction.
                #     continue
                mean_val = group_data[metric].mean()
                rel = mean_val / baseline_val

                marker = STYLE_DICT.get(approximator, {}).get("marker") or "o"
                color = STYLE_DICT.get(approximator, {}).get("color", "black")
                z = APPROXIMATOR_TO_ZORDER.get(approximator, 1)

                outline_s = marker_size
                main_s = marker_size**2
                # Highlight outline
                ax.scatter(
                    x,
                    rel,
                    s=outline_s,
                    c="white",
                    marker=marker,
                    zorder=z + 1,
                    alpha=APPROXIMATOR_TO_ALPHA.get(approximator, 1.0),
                )
                # Main marker
                label = approximator if approximator not in seen_labels else None
                if label is not None:
                    seen_labels.add(approximator)
                ax.scatter(
                    x,
                    rel,
                    s=main_s,
                    label=label,
                    c=color,
                    marker=marker,
                    zorder=z,
                    alpha=APPROXIMATOR_TO_ALPHA.get(approximator, 1.0),
                )

        ## Make background for each second dataset slightly shaded
        if dataset_idx % 2 == 1:
            ax.axvspan(
                (dataset_idx - 0.12) * points_per_group,
                (dataset_idx + 0.88) * points_per_group,
                color="#f7f7f7",
                zorder=-1,
            )
        group_centers.append(
            dataset_idx * points_per_group + (points_per_group - 1) / 2
        )
    # Make the width of axis tighter
    ax.set_xlim(-0.5, len(configs) * points_per_group - 0.5)
    # x-axis: repeated time-step labels
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, rotation=0)
    ax.set_xlabel("Estimation Time (s)", fontsize=12)

    # Put dataset names centered over each group
    for dataset_idx, cfg in enumerate(configs):
        game_name = cfg["game_name"]
        ax.text(
            group_centers[dataset_idx],
            10 ** (-0.8),
            DATA_NAMES.get(game_name, game_name)
            .replace(" ", "\n")
            .replace("(", "")
            .replace(")", ""),
            fontsize=14,
            fontweight=W_SEMIBOLD,
            ha="center",
            va="bottom",
        )
    # Visual separators between datasets
    for dataset_idx in range(1, len(configs)):
        ax.axvline(dataset_idx * points_per_group - 0.5, color="#dddddd", linewidth=1)

    if y_log_scale:
        ax.set_yscale("log")
    ax.set_ylabel("Relative MSE (ProxySHAP)", fontsize=12)
    if y_lim is not None:
        ax.set_ylim(y_lim)

    # Save legend separately (paper-style)
    if legend_output_path is not None:
        handles, labels = ax.get_legend_handles_labels()
        if len(handles) > 0:
            fig_leg, ax_leg = plt.subplots(figsize=(3.2, 2.6))
            ax_leg.axis("off")
            fig_leg.legend(
                handles,
                labels,
                loc="center",
                ncol=max(1, legend_ncol),
                frameon=True,
                fancybox=True,
                framealpha=1,
            )
            fig_leg.savefig(legend_output_path, bbox_inches="tight", pad_inches=0.1)
            plt.close(fig_leg)
    # Pad figure bottom and left
    # fig.subplots_adjust(left=0.15, bottom=0.2)
    fig.tight_layout()
    fig.savefig(output_path)


def plot_main_paper_plots(
    game_name,
    order,
    index,
    game_type,
    approximators_to_plot,
    figsize,
    ylim,
    y_log_scale,
    x_log_scale,
    min_budget=0,
    max_budget=float("inf"),
    marker_size=6,
    linewidth=2,
    highlight_size=2,
    save_path=None,
):
    os.makedirs(save_path, exist_ok=True)
    results_df = _load_and_prepare_results_df(
        game_name=game_name,
        order=order,
        index=index,
        game_type=game_type,
        approximators_to_plot=approximators_to_plot,
        max_budget=max_budget,
    )
    n_players = int(results_df["n_players"].values[0])
    min_b = n_players + 1 if n_players < 1000 else 101
    max_b = min(2**n_players, max_budget) if n_players <= 20 else max_budget
    budget_range = (
        np.ceil(np.logspace(np.log10(min_b), np.log10(max_b), 20))
        .clip(min_b, max_b)
        .astype(int)
    )
    # results_df = results_df[results_df["budget"].isin(budget_range)]

    ## Save Legend ##
    fig, ax = plot_approximation_quality_vstime(
        data=results_df,
        metric="MSE",
        log_scale_y=y_log_scale,
        log_scale_x=x_log_scale,
        legend=True,
    )
    ax.axis("off")
    # ax.axis("off")
    # # Get handles and labels
    # handles, labels = ax.get_legend_handles_labels()
    # # Replace old labels with new ones
    # labels = [APPROXIMATOR_RENAMING.get(l, l) for l in labels]
    # # Update legend
    # ax.legend(
    #     handles,
    #     labels,
    #     bbox_to_anchor=(1, 0.5),
    # )
    # Save the legend separately
    fig.savefig(save_path + "legend.pdf", bbox_inches="tight", pad_inches=0.1)
    # fig_legend.show()

    # Plot approximation quality for standard
    # Plot approximation quality for standard
    ## Adjust the limits ##
    # Plot approximation quality for standard
    ## Adjust the limits ##
    # Set min value to the min_value of second highest budget
    second_highest_budget = budget_range[-2]
    temp_df = results_df[results_df["budget"] == second_highest_budget]
    min_value_in_results = (
        temp_df[temp_df["approximator"].str.startswith("ProxySHAP")]["MSE"].min() * 1.2
    )
    max_value_in_results = results_df["MSE"].max()
    new_ylim_max = max_value_in_results * 0.4
    new_ylim_min = min_value_in_results * 1.5
    ylim = (new_ylim_min, new_ylim_max)
    ## Plot MSE ##
    metric = "MSE"
    fig, ax = plot_approximation_quality_vstime(
        data=results_df,
        metric=metric,
        log_scale_y=y_log_scale,
        log_scale_x=x_log_scale,
        figsize=figsize,
        log_scale_min=ylim[0] if y_log_scale else None,
        log_scale_max=ylim[1] if y_log_scale else None,
        legend=False,
        marker_size=marker_size,
        linewidth=linewidth,
        highlight_size=highlight_size,
        confidence_metric="sem",
    )
    # Set xlim based on our budget range
    runtime_our_min = results_df[results_df["approximator"].str.contains("ProxySHAP")][
        "total_runtime"
    ].min()
    runtime_our_max = results_df[results_df["approximator"].str.contains("ProxySHAP")][
        "total_runtime"
    ].max()
    ax.set_xlim(runtime_our_min * 0.8, runtime_our_max * 1.2)
    ax.set_title(DATA_NAMES[game_name], fontsize=TITLE_FONT_SIZE, fontweight=W_REGULAR)
    fig.savefig(
        save_path
        + f"{game_name}_{index}_{order}_{game_type}_approx_qualtiy_{metric}_vs_time.pdf"
    )


TITLE_FONT_SIZE = 18
DATA_NAMES = {
    "BreastCancerLocalXAI": "Cancer ($n=30$)",
    "CommunitiesAndCrimeLocalXAI": "Crime ($n=101$)",
    "Corrgroups60LocalXAI": "CG60 ($n=60$)",
    "ForestFiresLocalXAI": "Forest ($n=13$)",
    "IndependentLinear60LocalXAI": "IL60 ($n=60$)",
    "NHANESILocalXAI": "NHANES ($n=79$)",
    "RealEstateLocalXAI": "Estate ($n=15$)",
    "wine_quality": "Wine ($n=11$)",
    "AdultCensusLocalXAI": "Adult ($n=14$)",
    "CaliforniaHousingLocalXAI": "Housing ($n=8$)",
    "BikeSharingLocalXAI": "Bike ($n=12$)",
    "ViT4by4Patches": "ViT16 ($n=16$)",
    "ViT3by3Patches": "ViT9 ($n=9$)",
    "ResNet18w14Superpixel": "ResNet18 ($n=14$)",
    "SentimentAnalysisLocalXAI": "DistilBERT ($n=14$)",
    "SOUM": "soum",
    "SOUM10k": "soum10k",
    "SOUM100k": "soum100k",
    "MicroresponseLocalXAI": "Microresponse ($n=1300$)",
    "LeukemiaLocalXAI": "Leukemia ($n=7129$)",
    "BioresponseLocalXAI": "Bioresponse ($n=1776$)",
    "AmazonLocalXAI": "Amazon ($n=10000$)",
}
APPROXIMATORS_TO_PLOT = [
    "SHAPIQ",
    "SVARMIQ",
    "PermutationSamplingSV",
    "PermutationSamplingSII",
    "KernelSHAPIQ",
    "ProxySPEX (XGBoost)",
    # Our Methods
    "ProxySHAP (Linear, MSR) [our]",
    "ProxySHAP (Linear) [our]",
    ## XGBOOST METHODS ##
    "ProxySHAP (XGBoost, MSR) [our]",
    "ProxySHAP (XGBoost) [our]",
    ## DEFAULT ##
    # "ProxySHAP+ (XGBoost Det. MSR) [our]",
    # "ProxySHAP+ (XGBoost Prob. MSR) [our]",
    # "ProxySHAP+ (XGBoost) [our]",
    ## HPO ##
    "ProxySHAP* (XGBoost, MSR) [our]",
    "ProxySHAP* (XGBoost) [our]",
]
LINEAR_METHODS = [
    "ProxySHAP (Linear Det. MSR) [our]",
    "ProxySHAP (Linear Prob. MSR) [our]",
    "ProxySHAP (Linear) [our]",
]
APPROXIMATOR_RENAMING = {
    # "PermutationSamplingSV": "Permutation Sampling (SV)",
    # "PermutationSamplingSII": "Permutation Sampling (SII)",
    "RegressionMSRIQ-NoAdjustment": "ProxySHAP (XGBoost) [our]",
    "RegressionMSRIQ": "ProxySHAP (XGBoost, MSR) [our]",
    #"ProxySHAP (XGBoost)": "ProxySHAP (XGBoost) [our]",
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
GAME_TYPE_ABBREVIATIONS = {
    "exhaustive": "EXH",
    "interventional": "INT",
    "exhaustive_tabpfn": "TABPFN",
    "pathdependent": "PD",
}
GAMES_EXHAUSTIVE = [
    # "ResNet18w14Superpixel",
    # "ViT4by4Patches",
    # "SentimentAnalysisLocalXAI",
    # "ViT3by3Patches",
]
GAMES_INTERVENTIONAL = [
    "CommunitiesAndCrimeLocalXAI",
    # "NHANESILocalXAI",
    "Corrgroups60LocalXAI",
    # "IndependentLinear60LocalXAI",
    # "BreastCancerLocalXAI",
    # "RealEstateLocalXAI",
    # "AdultCensusLocalXAI",
    # "BikeSharingLocalXAI",
]
GAMES_TABPFN = [
    # "AdultCensusLocalXAI",
    # "RealEstateLocalXAI",
    # "BikeSharingLocalXAI",
    # "CaliforniaHousingLocalXAI",
    # "ForestFiresLocalXAI",
]
GAMES_PATHDEPENDENT = [
    # "CommunitiesAndCrimeLocalXAI",
    # "Corrgroups60LocalXAI",
    # "IndependentLinear60LocalXAI",
    # "RealEstateLocalXAI",
    # "AdultCensusLocalXAI",
    # "BikeSharingLocalXAI",
]
ITERATIONS = {
    # "SV": 1,
    "SII@2": 2,
    "SII@3": 3,
    # "BV": 1,
    # "BII@2": 2,
    # "BII@3": 3,
}
if __name__ == "__main__":
    MAKE_GROUPED_SCATTER = False
    general_params = {
        "figsize": (4, 4),  # (5.5, 3.5),
        "y_log_scale": True,
        "x_log_scale": True,
        "marker_size": 3,
        "linewidth": 2,
        "highlight_size": 2,
        "min_budget": 14,
        "max_budget": 35_000,
        "ylim": (1e-6, 1e2),
    }
    Y_LIM_MIN = 1e-8

    if MAKE_GROUPED_SCATTER:
        TIME = 0.001
        plot_grouped_scatter_relative_mse(
            configs=[
                {
                    "game_name": "ViT4by4Patches",
                    "order": 2,
                    "index": "SII",
                    "game_type": "exhaustive",
                    "max_budget": 66666,
                    "time_steps": [1, 10, 30, 60],
                },
                {
                    "game_name": "IndependentLinear60LocalXAI",
                    "order": 2,
                    "index": "SII",
                    "game_type": "interventional",
                    "max_budget": general_params["max_budget"],
                    "time_steps": [1, 10, 30, 60],
                },
                {
                    "game_name": "NHANESILocalXAI",
                    "order": 2,
                    "index": "SII",
                    "game_type": "interventional",
                    "max_budget": general_params["max_budget"],
                    "time_steps": [1, 10, 30, 60],
                },
                {
                    "game_name": "CommunitiesAndCrimeLocalXAI",
                    "order": 2,
                    "index": "SII",
                    "game_type": "interventional",
                    "max_budget": general_params["max_budget"],
                    "time_steps": [1, 10, 30, 60],
                },
                # {
                #     "game_name": "ResNet18w14Superpixel",
                #     "order": 2,
                #     "index": "SII",
                #     "game_type": "exhaustive",
                #     "max_budget": 35_000,
                #     "time_steps": [10, 100, 1_000],
                # },
            ],
            approximators_to_plot=APPROXIMATORS_TO_PLOT,
            marker_size=9.0,
            figsize=(5.6, 5),
            output_path=f"plots/main/grouped_scatter_SII2_{TIME}.pdf",
            legend_output_path=f"plots/main/grouped_scatter_SII2_legend_{TIME}.pdf",
            legend_ncol=1,
            y_log_scale=True,
            y_lim=(10 ** (-1), 10 ** (1.2)),
        )

        TIME = 0.01
        plot_grouped_scatter_relative_mse(
            configs=[
                {
                    "game_name": "ViT4by4Patches",
                    "order": 2,
                    "index": "SII",
                    "game_type": "exhaustive",
                    "max_budget": 66666,
                    "time_steps": [10, 30, 60, 120],
                },
                {
                    "game_name": "IndependentLinear60LocalXAI",
                    "order": 2,
                    "index": "SII",
                    "game_type": "interventional",
                    "max_budget": general_params["max_budget"],
                    "time_steps": [10, 30, 60, 120],
                },
                {
                    "game_name": "NHANESILocalXAI",
                    "order": 2,
                    "index": "SII",
                    "game_type": "interventional",
                    "max_budget": general_params["max_budget"],
                    "time_steps": [10, 30, 60, 120],
                },
                {
                    "game_name": "CommunitiesAndCrimeLocalXAI",
                    "order": 2,
                    "index": "SII",
                    "game_type": "interventional",
                    "max_budget": general_params["max_budget"],
                    "time_steps": [10, 30, 60, 120],
                },
                # {
                #     "game_name": "ResNet18w14Superpixel",
                #     "order": 2,
                #     "index": "SII",
                #     "game_type": "exhaustive",
                #     "max_budget": 35_000,
                #     "time_steps": [10, 100, 1_000],
                # },
            ],
            approximators_to_plot=APPROXIMATORS_TO_PLOT,
            marker_size=9.0,
            figsize=(5, 3),
            output_path=f"plots/main/grouped_scatter_SII2_{TIME}.pdf",
            legend_output_path=f"plots/main/grouped_scatter_SII2_legend_{TIME}.pdf",
            legend_ncol=1,
            y_log_scale=True,
            y_lim=(10 ** (-0.8), 10 ** (2)),
        )

        TIME = 0.1
        plot_grouped_scatter_relative_mse(
            configs=[
                {
                    "game_name": "ViT4by4Patches",
                    "order": 2,
                    "index": "SII",
                    "game_type": "exhaustive",
                    "max_budget": 66666,
                    "time_steps": [60, 120, 300, 600],
                },
                {
                    "game_name": "IndependentLinear60LocalXAI",
                    "order": 2,
                    "index": "SII",
                    "game_type": "interventional",
                    "max_budget": general_params["max_budget"],
                    "time_steps": [60, 120, 300, 600],
                },
                {
                    "game_name": "NHANESILocalXAI",
                    "order": 2,
                    "index": "SII",
                    "game_type": "interventional",
                    "max_budget": general_params["max_budget"],
                    "time_steps": [60, 120, 300, 600],
                },
                {
                    "game_name": "CommunitiesAndCrimeLocalXAI",
                    "order": 2,
                    "index": "SII",
                    "game_type": "interventional",
                    "max_budget": general_params["max_budget"],
                    "time_steps": [60, 120, 300, 600],
                },
                # {
                #     "game_name": "ResNet18w14Superpixel",
                #     "order": 2,
                #     "index": "SII",
                #     "game_type": "exhaustive",
                #     "max_budget": 35_000,
                #     "time_steps": [10, 100, 1_000],
                # },
            ],
            approximators_to_plot=APPROXIMATORS_TO_PLOT,
            marker_size=9.0,
            figsize=(5.6, 5),
            output_path=f"plots/main/grouped_scatter_SII2_{TIME}.pdf",
            legend_output_path=f"plots/main/grouped_scatter_SII2_legend_{TIME}.pdf",
            legend_ncol=1,
            y_log_scale=True,
            y_lim=(10 ** (-1), 10 ** (1.2)),
        )
    orders_by_index: dict[str, list[int]] = {}
    for index_with_order, order in ITERATIONS.items():
        base_index = index_with_order.split("@")[0]
        orders_by_index.setdefault(base_index, []).append(order)

    for INDEX, orders in orders_by_index.items():
        orders = sorted(orders)
        print(f"Processing index {INDEX} with orders {orders}...")

        # Easy y-limit control:
        # 1) Set per-dataset defaults in `runtime_y_limits_default`.
        # 2) Override any single subplot via (order, time_key, game_name).
        #    time_key must match one of: "0", "0.0001", "0.01", "0.1".
        runtime_y_limits_default: dict[str, tuple[float, float]] = {
            "CommunitiesAndCrimeLocalXAI": (10 ** (-1.2), 10**2),
            "IndependentLinear60LocalXAI": (10 ** (-6), 10 ** (-2)),
            "NHANESILocalXAI": (10 ** (-8), 10**(8)),
        }
        runtime_y_limits_overrides: dict[tuple[int, str, str], tuple[float, float]] = {
            # Order 2
            (2, "0.001", "CommunitiesAndCrimeLocalXAI"): (10 ** (-0.8), 10**(1.5)),
            (2, "0.01", "CommunitiesAndCrimeLocalXAI"): (10 ** (-0.8), 10**(1.5)),
            (2, "0.1", "CommunitiesAndCrimeLocalXAI"): (10 ** (-0.8), 10**(1.5)),
            (2, "0.001", "NHANESILocalXAI"): (10 ** (-5), 10**(-1)),
            (2, "0.01", "NHANESILocalXAI"): (10 ** (-5), 10**(-1)),
            (2, "0.1", "NHANESILocalXAI"): (10 ** (-5), 10**(-1)),
            (2, "0.001", "IndependentLinear60LocalXAI"): (10 ** (-6.5), 10 ** (-2)),
            (2, "0.01", "IndependentLinear60LocalXAI"): (10 ** (-6.5), 10 ** (-2)),
            (2, "0.1", "IndependentLinear60LocalXAI"): (10 ** (-6.5), 10 ** (-2)),
            # Order 3
            (3, "0.001", "CommunitiesAndCrimeLocalXAI"): (10 ** (-1.2), 10**(0.01)),
            (3, "0.01", "CommunitiesAndCrimeLocalXAI"): (10 ** (-1.2), 10**(0.01)),
            (3, "0.1", "CommunitiesAndCrimeLocalXAI"): (10 ** (-1.2), 10**(0.01)),
            (3, "0.001", "NHANESILocalXAI"): (10 ** (-4.5), 10**(-2.6)),
            (3, "0.01", "NHANESILocalXAI"): (10 ** (-4.5), 10**(-2.6)),
            (3, "0.1", "NHANESILocalXAI"): (10 ** (-4.5), 10**(-2.6)),
            (3, "0.001", "IndependentLinear60LocalXAI"): (10 ** (-6), 10 ** (-3.2)),
            (3, "0.01", "IndependentLinear60LocalXAI"): (10 ** (-6), 10 ** (-3.2)),
            (3, "0.1", "IndependentLinear60LocalXAI"): (10 ** (-6), 10 ** (-3.2)),
        }

        def _time_key(time_step: float) -> str:
            return f"{time_step:g}"

        def _get_runtime_ylim(game_name: str, order: int, time_step: float) -> tuple[float, float]:
            return runtime_y_limits_overrides.get(
                (order, _time_key(time_step), game_name),
                runtime_y_limits_default[game_name],
            )

        base_panel_specs = [
            {
                "game_name": "IndependentLinear60LocalXAI",
                "game_type": "interventional",
                "max_budget": general_params["max_budget"],
            },
            {
                "game_name": "NHANESILocalXAI",
                "game_type": "interventional",
                "max_budget": general_params["max_budget"],
            }
            # {
            #     "game_name": "CommunitiesAndCrimeLocalXAI",
            #     "game_type": "interventional",
            #     "max_budget": general_params["max_budget"],
            # },
        ]
        time_steps = [0.001, 0.01, 0.1]

        runtime_plot_specs: list[dict] = []
        for panel_spec in base_panel_specs:
            for order in orders:
                for time_step in time_steps:
                    runtime_plot_specs.append(
                        {
                            "game_name": panel_spec["game_name"],
                            "order": order,
                            "index": INDEX,
                            "game_type": panel_spec["game_type"],
                            "time_step": time_step,
                            "ylim": _get_runtime_ylim(
                                panel_spec["game_name"],
                                order,
                                time_step,
                            ),
                            "max_budget": panel_spec["max_budget"],
                        },
                    )

        runtime_output_dir = f"plots/appendix/runtime/combined/{INDEX}/orders-{'-'.join(str(order) for order in orders)}"
        os.makedirs(runtime_output_dir, exist_ok=True)
        plot_combined_runtime_plots(
            plot_specs=runtime_plot_specs,
            approximators_to_plot=APPROXIMATORS_TO_PLOT,
            output_path=f"{runtime_output_dir}/runtime_combined.pdf",
            plots_per_row=len(time_steps),
            panel_figsize=general_params["figsize"],
            y_log_scale=general_params["y_log_scale"],
            x_log_scale=general_params["x_log_scale"],
            marker_size=general_params["marker_size"],
            linewidth=general_params["linewidth"],
            highlight_size=general_params["highlight_size"],
            show_captions_only_first_row=True,
            caption_row_period=len(orders),
            x_ticks_only_last_row=False,
            shared_ylabel_x=0.08,
            shared_xlabel_y=0.04,
            subplot_wspace=0.18,
            subplot_hspace=0.2,
            shared_ylabel_fontsize=14,
            shared_xlabel_fontsize=14,
            shared_ylabel="Relative MSE ± SEM",
            shared_xlabel="Runtime (s)",
            tick_fontsize=14,
            legend_scale=1.4
        )
