"""Plot relative extraction runtime from benchmark CSV results."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent / "special_plot_scripts"))

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib.ticker as ticker
import matplotlib.transforms as mtransforms
import numpy as np
from matplotlib.lines import Line2D

from shapiq_benchmark.plot import LIGHT_GRAY
from _plot_style import apply_tick_style, setup_fonts, W_REGULAR, W_SEMIBOLD


DATASET_DISPLAY_NAMES = {
    "AdultCensusLocalXAI": "Adult ($n=14$)",
    "BikeSharingLocalXAI": "Bike ($n=12$)",
    "BreastCancerLocalXAI": "Cancer ($n=30$)",
    "CaliforniaHousingLocalXAI": "Housing ($n=8$)",
    "CommunitiesAndCrimeLocalXAI": "Crime ($n=101$)",
    "Corrgroups60LocalXAI": "CG60 ($n=60$)",
    "ForestFiresLocalXAI": "Forest ($n=13$)",
    "IndependentLinear60LocalXAI": "IL60 ($n=60$)",
    "NHANESILocalXAI": "NHANES ($n=79$)",
    "RealEstateLocalXAI": "Estate ($n=15$)",
}

DATASET_PLAYER_COUNTS = {
    "AdultCensusLocalXAI": 14,
    "BikeSharingLocalXAI": 12,
    "BreastCancerLocalXAI": 30,
    "CaliforniaHousingLocalXAI": 8,
    "CommunitiesAndCrimeLocalXAI": 101,
    "Corrgroups60LocalXAI": 60,
    "ForestFiresLocalXAI": 13,
    "IndependentLinear60LocalXAI": 60,
    "NHANESILocalXAI": 79,
    "RealEstateLocalXAI": 15,
}


# Colorblind-safe, publication-grade palette — one entry per dataset (10 total).
DATASET_PALETTE = [
    "#7FAFD4",  # softened blue
    "#E29A73",  # softened vermilion
    "#86C9B1",  # softened bluish green
    "#D5A5C6",  # softened reddish purple
    "#9CCCEC",  # softened sky blue
    "#E7BF72",  # softened orange
    "#EDE48A",  # softened yellow
    "#7A7A7A",  # softened gray
    "#C05050",  # muted red
    "#5B8A72",  # muted teal
]


def _style_ytick_labels(ax: plt.Axes) -> None:
    """Custom y-tick labels: Nx format showing the factor, colored green/red by direction."""
    def _fmt(val, _pos):
        if val <= 0:
            return ""
        factor = val if val >= 1 else 1.0 / val
        return f"{int(round(factor))}x"

    ax.yaxis.set_major_formatter(ticker.FuncFormatter(_fmt))
    ax.figure.canvas.draw()
    for tick in ax.yaxis.get_major_ticks():
        pos = tick.get_loc()
        if pos > 1.0:
            tick.label1.set_color("#2d6a4f")
        elif pos < 1.0:
            tick.label1.set_color("#9b2226")


def load_summary_results(path: Path) -> dict[tuple[str, str, int], dict[int, float]]:
    """Load summary CSV and organize by (dataset, index, order) over depth as speedup."""
    grouped: dict[tuple[str, str, int], dict[int, float]] = defaultdict(dict)

    with path.open("r", newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            dataset = row["dataset"]
            index = row["index"]
            order = int(row["order"])
            depth = int(row["depth"])
            relative_runtime = float(row["relative_runtime"])
            if relative_runtime <= 0 or not np.isfinite(relative_runtime):
                continue
            speedup = 1.0 / relative_runtime
            grouped[(dataset, index, order)][depth] = speedup

    return grouped


def plot_from_summary_csv(
    summary_csv: Path = Path("results/extraction_times_summary.csv"),
    output_dir: Path = Path("results/"),
) -> None:
    """For each unique index in the CSV, create a 1x3 figure (order 1, 2, 3)."""
    if not summary_csv.exists():
        raise FileNotFoundError(
            f"Summary CSV not found at {summary_csv}. Run compare_extraction_times.py first."
        )

    grouped = load_summary_results(summary_csv)
    datasets = sorted({dataset for dataset, _, _ in grouped.keys()})
    indices = ["BII", "FBII", "SII", "FSII"]  # Explicit order for indices
    #indices = sorted({index for _, index, _ in grouped.keys()})

    plt.style.use(["science", "no-latex"])
    setup_fonts()

    general_params = {
        "panel_figsize": (2, 2),
        "marker_size": 4,
        "linewidth": 1,
        "legend_fontsize": 6,
        "label_fontsize": 12,
        "x_label": "Max Depth",
        "y_label": "Speedup",
        "x_label_y": 0.05,
        "y_label_x": -0.18,
        "order_fontsize": 16,
        "combined_hspace": 0.12,
    }

    dataset_colors = {
        name: DATASET_PALETTE[i % len(DATASET_PALETTE)] for i, name in enumerate(datasets)
    }

    output_dir.mkdir(parents=True, exist_ok=True)

    n_panels = 3
    panel_width, panel_height = general_params["panel_figsize"]
    fig_size = (panel_width * n_panels, panel_height)

    for index in indices:
        fig, axes = plt.subplots(1, n_panels, figsize=fig_size, sharey=True)

        legend_handles: list = []
        legend_labels: list = []

        for axis_idx, order in enumerate([1, 2, 3]):
            ax = axes[axis_idx]

            for dataset in datasets:
                series = grouped.get((dataset, index, order), {})
                if not series:
                    continue

                depths = sorted(series.keys())
                values = [series[d] for d in depths]

                valid_pairs = [
                    (d, v) for d, v in zip(depths, values) if not np.isnan(v) and np.isfinite(v)
                ]
                if not valid_pairs:
                    continue

                valid_depths = [d for d, _ in valid_pairs]
                valid_values = [v for _, v in valid_pairs]

                ax.plot(
                    valid_depths,
                    valid_values,
                    color=dataset_colors[dataset],
                    linewidth=general_params["linewidth"],
                    marker="o",
                    markersize=general_params["marker_size"],
                    markeredgecolor="white",
                    markeredgewidth=0.9,
                    label=DATASET_DISPLAY_NAMES.get(dataset, dataset),
                    zorder=4,
                    path_effects=[
                        pe.Stroke(linewidth=general_params["linewidth"] + 1.4, foreground="white"),
                        pe.Normal(),
                    ],
                )

            ax.axhline(1.0, color="#444444", linestyle="dashed", linewidth=0.8, zorder=3, alpha=0.6)
            ax.axhspan(1.0, 10**4, color="#4CAF50", alpha=0.04, linewidth=0, zorder=1)
            ax.axhspan(0, 1.0, color="#E53935", alpha=0.04, linewidth=0, zorder=1)

            _faster_trans = mtransforms.blended_transform_factory(ax.transAxes, ax.transData)
            ax.text(
                0.98, 1.5, "faster", transform=_faster_trans,
                fontsize=7, fontweight=W_SEMIBOLD, color="#2d6a4f",
                ha="right", va="bottom", alpha=0.85,
            )
            ax.text(
                0.98, 0.03, "slower", transform=ax.transAxes,
                fontsize=7, fontweight=W_SEMIBOLD, color="#9b2226",
                ha="right", va="bottom", alpha=0.85,
            )

            ax.text(
                0.03,
                0.97,
                f"order {order}",
                transform=ax.transAxes,
                fontsize=general_params["order_fontsize"],
                fontweight=W_SEMIBOLD,
                ha="left",
                va="top",
                color="#222222",
            )

            ax.set_xlabel("")
            ax.grid(axis="both", color=LIGHT_GRAY, linestyle="dashed", alpha=0.5)
            ax.set_yscale("log")
            ax.minorticks_off()
            ax.set_ylim(1e-1, 1e4)
            ax.set_xticks(list(range(order, 9)))

            if axis_idx == 0:
                handles, labels = ax.get_legend_handles_labels()
                legend_handles = handles
                legend_labels = labels

        fig.supxlabel(
            general_params["x_label"],
            fontsize=general_params["label_fontsize"],
            y=general_params["x_label_y"],
            fontweight=W_SEMIBOLD,
        )
        axes[0].set_ylabel(
            general_params["y_label"],
            fontsize=general_params["label_fontsize"],
            fontweight=W_SEMIBOLD,
        )
        axes[0].yaxis.set_label_coords(general_params["y_label_x"], 0.5)

        baseline_handle = Line2D([0], [0], color="#444444", linestyle="dashed", linewidth=0.8, alpha=0.6)

        if legend_handles:
            legend_entries = list(zip(legend_handles, legend_labels))
            legend_entries.sort(
                key=lambda entry: (
                    DATASET_PLAYER_COUNTS.get(
                        next(
                            (
                                dataset_name
                                for dataset_name, display_name in DATASET_DISPLAY_NAMES.items()
                                if display_name == entry[1]
                            ),
                            "",
                        ),
                        0,
                    ),
                    entry[1],
                )
            )
            legend_handles = [baseline_handle] + [handle for handle, _ in legend_entries]
            legend_labels = ["Fourier baseline"] + [label for _, label in legend_entries]

        fig.subplots_adjust(left=0.065, right=0.995, top=0.865, bottom=0.22, wspace=0.02)
        fig.canvas.draw()
        # Index caption centred over all three panels.
        all_left = axes[0].get_position().x0
        all_right = axes[2].get_position().x1
        caption_y = 0.885
        fig.text(
            (all_left + all_right) / 2,
            caption_y,
            index,
            ha="center",
            va="bottom",
            fontsize=14,
            fontweight=W_SEMIBOLD,
        )

        apply_tick_style(*[ax for ax in fig.axes if ax.get_visible()])
        fig.canvas.draw()
        _style_ytick_labels(axes[0])

        output_pdf = output_dir / f"runtime_fourier_vs_interventional_{index}.pdf"
        plt.savefig(output_pdf, dpi=300, bbox_inches="tight")
        print(f"Saved plot: {output_pdf}")

        legend_output_pdf = output_dir / f"runtime_fourier_vs_interventional_{index}_legend.pdf"
        _ds_h, _ds_l = legend_handles[1:], legend_labels[1:]
        _nrow = -(-len(_ds_h) // 3)
        _empty = Line2D([], [], alpha=0)
        _leg_h = [_empty] + [legend_handles[0]] + [_empty] + _ds_h
        _leg_l = [""] + [legend_labels[0]] + [""] + _ds_l
        legend_fig = plt.figure(figsize=(fig_size[0], 0.35 * _nrow))
        legend_fig.legend(
            _leg_h, _leg_l,
            loc="center", ncol=4, frameon=False,
            fontsize=general_params["legend_fontsize"], handlelength=2.2, columnspacing=0.9,
        )
        legend_fig.savefig(legend_output_pdf, dpi=300, bbox_inches="tight")
        print(f"Saved legend: {legend_output_pdf}")
        plt.close(legend_fig)
        plt.close(fig)

    _plot_combined(grouped, datasets, indices, dataset_colors, general_params, output_dir)


def _plot_combined(
    grouped: dict,
    datasets: list[str],
    indices: list[str],
    dataset_colors: dict[str, str],
    params: dict,
    output_dir: Path,
) -> None:
    """4×3 combined figure: rows=indices, cols=orders, shared axes, tight spacing."""
    n_rows = len(indices)
    n_cols = 3
    panel_w, panel_h = params["panel_figsize"]
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(panel_w * n_cols, panel_h * n_rows),
        sharey="row",
        sharex="col",
    )

    legend_handles: list = []
    legend_labels: list = []

    for row_idx, index in enumerate(indices):
        for col_idx, order in enumerate([1, 2, 3]):
            ax = axes[row_idx][col_idx]

            for dataset in datasets:
                series = grouped.get((dataset, index, order), {})
                if not series:
                    continue
                depths = sorted(series.keys())
                values = [series[d] for d in depths]
                valid_pairs = [
                    (d, v) for d, v in zip(depths, values) if not np.isnan(v) and np.isfinite(v)
                ]
                if not valid_pairs:
                    continue
                valid_depths = [d for d, _ in valid_pairs]
                valid_values = [v for _, v in valid_pairs]

                line, = ax.plot(
                    valid_depths,
                    valid_values,
                    color=dataset_colors[dataset],
                    linewidth=params["linewidth"],
                    marker="o",
                    markersize=params["marker_size"],
                    markeredgecolor="white",
                    markeredgewidth=0.9,
                    label=DATASET_DISPLAY_NAMES.get(dataset, dataset),
                    zorder=4,
                    path_effects=[
                        pe.Stroke(
                            linewidth=params["linewidth"] + 1.4, foreground="white"
                        ),
                        pe.Normal(),
                    ],
                )
                if row_idx == 0 and col_idx == 0:
                    legend_handles.append(line)
                    legend_labels.append(DATASET_DISPLAY_NAMES.get(dataset, dataset))

            ax.axhline(1.0, color="#444444", linestyle="dashed", linewidth=0.8, zorder=3, alpha=0.6)
            ax.axhspan(1.0, 1e4, color="#4CAF50", alpha=0.04, linewidth=0, zorder=1)
            ax.axhspan(0, 1.0, color="#E53935", alpha=0.04, linewidth=0, zorder=1)
            _faster_trans = mtransforms.blended_transform_factory(ax.transAxes, ax.transData)
            ax.text(
                0.98, 1.5, "faster", transform=_faster_trans,
                fontsize=7, fontweight=W_SEMIBOLD, color="#2d6a4f",
                ha="right", va="bottom", alpha=0.85,
            )
            ax.text(
                0.98, 0.03, "slower", transform=ax.transAxes,
                fontsize=7, fontweight=W_SEMIBOLD, color="#9b2226",
                ha="right", va="bottom", alpha=0.85,
            )
            ax.grid(axis="both", color=LIGHT_GRAY, linestyle="dashed", alpha=0.5)
            ax.set_yscale("log")
            ax.minorticks_off()
            ax.set_ylim(1e-1, 1e4)
            ax.set_xticks(list(range(order, 9)))

            # Order label on top row only
            if row_idx == 0:
                ax.set_title(
                    f"order {order}",
                    fontsize=params["order_fontsize"],
                    fontweight=W_SEMIBOLD,
                    pad=3,
                )

            ax.text(
                0.03, 0.97, index,
                transform=ax.transAxes,
                fontsize=params["order_fontsize"],
                fontweight=W_SEMIBOLD,
                ha="left", va="top",
                color="#222222",
            )

    fig.supxlabel(
        params["x_label"],
        fontsize=params["label_fontsize"],
        y=0.01,
        fontweight=W_SEMIBOLD,
    )

    fig.subplots_adjust(hspace=params["combined_hspace"], wspace=0.04, left=0.12, right=0.995, top=0.93, bottom=0.09)
    fig.canvas.draw()
    fig.text(
        0.02, 0.5, params["y_label"],
        fontsize=params["label_fontsize"],
        fontweight=W_SEMIBOLD,
        ha="center", va="center",
        rotation="vertical",
    )

    baseline_handle = Line2D([0], [0], color="#444444", linestyle="dashed", linewidth=0.8, alpha=0.6)
    if legend_handles:
        legend_entries = list(zip(legend_handles, legend_labels))
        legend_entries.sort(
            key=lambda entry: (
                DATASET_PLAYER_COUNTS.get(
                    next(
                        (
                            dataset_name
                            for dataset_name, display_name in DATASET_DISPLAY_NAMES.items()
                            if display_name == entry[1]
                        ),
                        "",
                    ),
                    0,
                ),
                entry[1],
            )
        )
        final_handles = [baseline_handle] + [h for h, _ in legend_entries]
        final_labels = ["Fourier baseline"] + [lbl for _, lbl in legend_entries]
    else:
        final_handles = [baseline_handle]
        final_labels = ["Fourier baseline"]

    apply_tick_style(*[ax for ax in fig.axes if ax.get_visible()])
    fig.canvas.draw()
    for row_idx in range(n_rows):
        _style_ytick_labels(axes[row_idx][0])

    output_pdf = output_dir / "runtime_fourier_vs_interventional_combined.pdf"
    fig.savefig(output_pdf, dpi=300, bbox_inches="tight")
    print(f"Saved combined plot: {output_pdf}")

    _ds_h, _ds_l = final_handles[1:], final_labels[1:]
    _nrow = -(-len(_ds_h) // 3)
    _empty = Line2D([], [], alpha=0)
    _leg_h = [_empty] + [final_handles[0]] + [_empty] + _ds_h
    _leg_l = [""] + [final_labels[0]] + [""] + _ds_l
    legend_fig = plt.figure(figsize=(panel_w * n_cols, 0.35 * _nrow))
    legend_fig.legend(
        _leg_h, _leg_l,
        loc="center", ncol=4, frameon=False,
        fontsize=params["legend_fontsize"], handlelength=2.2, columnspacing=0.9,
    )
    legend_output_pdf = output_dir / "runtime_fourier_vs_interventional_combined_legend.pdf"
    legend_fig.savefig(legend_output_pdf, dpi=300, bbox_inches="tight")
    print(f"Saved combined legend: {legend_output_pdf}")
    plt.close(legend_fig)
    plt.close(fig)


if __name__ == "__main__":
    plot_from_summary_csv()
