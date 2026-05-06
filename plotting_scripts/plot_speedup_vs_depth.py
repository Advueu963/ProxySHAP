"""Plot average speedup (Fourier vs Interventional) per (index, order) over tree depth."""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator

from shapiq_benchmark.plot import LIGHT_GRAY
from shapiq.plot._config import BLUE, RED
from _plot_style import W_SEMIBOLD, W_THIN, apply_tick_style, setup_fonts

# ---------------------------------------------------------------------------
# CONTROL: edit this list to choose which (index, order) combos are plotted
# ---------------------------------------------------------------------------
SHOW_COMBINATIONS: list[tuple[str, int]] = [
    ("SII", 1),
    ("SII", 2),
    ("SII", 3),
    ("BII", 1),
    ("BII", 2),
    ("BII", 3),
    ("FBII", 1),
    ("FBII", 2),
    ("FBII", 3),
    ("FSII", 1),
    ("FSII", 2),
    ("FSII", 3),
]

# SII uses shapiq blue, BII uses shapiq red; Fourier variants are lighter tints
INDEX_COLORS: dict[str, str] = {
    "SII": BLUE.hex,  # shapiq blue  #1e88e5
    "FSII": "#90CAF9",  # light blue tint
    "BII": RED.hex,  # shapiq red   #ff0d57
    "FBII": "#FF8FA3",  # light red tint
}

# SII/BII drawn on top of their Fourier counterparts
INDEX_ZORDER: dict[str, int] = {
    "FSII": 3,
    "FBII": 3,
    "SII": 5,
    "BII": 5,
}

# One line style per order
ORDER_LINESTYLES: dict[int, str] = {
    1: "solid",
    2: "dashed",
    3: "dotted",
}

# Label padding in points — decrease to pull labels closer to the axes
XLABEL_LABELPAD: float = -1.2
YLABEL_LABELPAD: float = 0

# Font sizes (points)
AXIS_LABEL_FONTSIZE: float = 14  # "Tree Depth" / "Speedup"
TICK_LABEL_FONTSIZE: float = 12  # x- and y-tick numbers
REGION_LABEL_FONTSIZE: float = 14  # "faster" / "slower" inline labels
LEGEND_FONTSIZE: float = 12

# Axis label text
XAXIS_LABEL: str = "Tree Depth"
YAXIS_LABEL: str = "Runtime Improvement over Fourier"
MARKERSIZE: float = 6
LINEWIDTH: float = 2

def load_speedup_data(
    path: Path,
) -> dict[tuple[str, int, int], list[float]]:
    """Return speedup values grouped by (index, order, depth) across datasets."""
    grouped: dict[tuple[str, int, int], list[float]] = defaultdict(list)

    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            relative_runtime = float(row["relative_runtime"])
            if relative_runtime <= 0 or not np.isfinite(relative_runtime):
                continue
            speedup = 1.0 / relative_runtime
            key = (row["index"], int(row["order"]), int(row["depth"]))
            grouped[key].append(speedup)

    return grouped


def plot_speedup_vs_depth(
    summary_csv: Path = Path("results/extraction_times_summary.csv"),
    output_path: Path = Path("results/main_interventional_speedup.pdf"),
    combinations: list[tuple[str, int]] = SHOW_COMBINATIONS,
    log_scale: bool = True,
) -> None:
    if not summary_csv.exists():
        raise FileNotFoundError(f"Summary CSV not found: {summary_csv}")

    grouped = load_speedup_data(summary_csv)

    # Aggregate mean ± std in log space (multiplicative spread), then exponentiate.
    # Raw speedup spans orders of magnitude across datasets, so log-space stats are
    # the correct summary for a log-scale y-axis.
    stats: dict[tuple[str, int], dict[int, tuple[float, float, float]]] = {}
    all_depths: set[int] = set()
    for index, order in combinations:
        key_prefix = (index, order)
        depth_stats: dict[int, tuple[float, float, float]] = {}
        for depth in range(1, 9):
            values = grouped.get((index, order, depth), [])
            if not values:
                continue
            log_arr = np.log(np.array(values))
            log_mean = log_arr.mean()
            log_std = log_arr.std()
            depth_stats[depth] = (
                float(np.exp(log_mean)),  # geometric mean
                float(np.exp(log_mean - log_std)),  # lower band
                float(np.exp(log_mean + log_std)),  # upper band
            )
            all_depths.add(depth)
        if depth_stats:
            stats[key_prefix] = depth_stats

    plt.style.use(["science", "no-latex"])
    setup_fonts()

    fig, ax = plt.subplots(figsize=(5, 4))

    for index, order in combinations:
        key = (index, order)
        if key not in stats:
            continue
        depth_stats = stats[key]
        depths = sorted(depth_stats)
        means = np.array([depth_stats[d][0] for d in depths])
        lowers = np.array([depth_stats[d][1] for d in depths])
        uppers = np.array([depth_stats[d][2] for d in depths])

        color = INDEX_COLORS.get(index, "#888888")
        ls = ORDER_LINESTYLES.get(order, "solid")
        zo = INDEX_ZORDER.get(index, 4)

        ax.fill_between(
            depths,
            lowers,
            uppers,
            color=color,
            alpha=0.13,
            linewidth=0,
            zorder=zo - 1,
        )
        # Line with thick white outline
        ax.plot(
            depths,
            means,
            color=color,
            linestyle=ls,
            linewidth=LINEWIDTH,
            marker="none",
            zorder=zo,
            path_effects=[pe.Stroke(linewidth=LINEWIDTH+3, foreground="white"), pe.Normal()],
        )
        # Markers with thin white outline on top
        ax.plot(
            depths,
            means,
            color=color,
            linestyle="none",
            marker="o",
            markersize=MARKERSIZE,
            markeredgecolor="white",
            markeredgewidth=1,
            zorder=zo + 1,
        )
    # Set y-limits to show 20% slowdown up to 10x speedup (adjust as needed)
    ax.set_ylim(bottom=10 ** (-1), top=10 ** (5.2))
    # Shaded regions + baseline line
    ax.axhline(
        1.0, color="#444444", linestyle="dashed", linewidth=0.8, zorder=3, alpha=0.6
    )
    ax.axhspan(1.0, 1e9, color="#4CAF50", alpha=0.04, linewidth=0, zorder=1)
    ax.axhspan(0, 1.0, color="#E53935", alpha=0.04, linewidth=0, zorder=1)

    # Inline region labels (drawn after yscale is set so transform is correct)
    ax.text(
        0.98,
        0.97,
        "faster",
        transform=ax.transAxes,
        fontsize=REGION_LABEL_FONTSIZE,
        fontweight=W_SEMIBOLD,
        color="#2d6a4f",
        ha="right",
        va="top",
        alpha=0.85,
    )
    ax.text(
        0.98,
        0.03,
        "slower",
        transform=ax.transAxes,
        fontsize=REGION_LABEL_FONTSIZE,
        fontweight=W_SEMIBOLD,
        color="#9b2226",
        ha="right",
        va="bottom",
        alpha=0.85,
    )

    ax.set_xlabel(
        XAXIS_LABEL,
        fontsize=AXIS_LABEL_FONTSIZE,
        fontweight=W_SEMIBOLD,
        labelpad=XLABEL_LABELPAD,
    )
    ax.set_ylabel(
        YAXIS_LABEL,
        fontsize=AXIS_LABEL_FONTSIZE,
        fontweight=W_SEMIBOLD,
        labelpad=YLABEL_LABELPAD,
    )
    ax.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE)
    ax.grid(axis="both", color=LIGHT_GRAY, linestyle="dashed", alpha=0.5)
    if log_scale:
        ax.set_yscale("log")

    # Symmetric "x"-style y-tick labels (mirrors main_paper_plot_relative_mse_n_players).
    # Above 1 = faster (green), 1 = baseline (gray), below 1 = slower (red).
    _ytick_positions = [1e-1, 1.0, 1e1, 1e2, 1e3, 1e4, 1e5]
    _ytick_labels = ["10x", "1", "10x", r"$10^2$x", r"$10^3$x", r"$10^4$x", r"$10^5$x"]
    _ytick_colors = [
        "#9b2226",
        "#555555",
        "#2d6a4f",
        "#2d6a4f",
        "#2d6a4f",
        "#2d6a4f",
        "#2d6a4f",
    ]
    ax.yaxis.set_major_locator(FixedLocator(_ytick_positions))
    ax.set_yticklabels(_ytick_labels)
    ax.yaxis.set_minor_locator(plt.NullLocator())

    sorted_depths = sorted(all_depths)
    ax.set_xticks(sorted_depths)
    ax.xaxis.set_minor_locator(plt.NullLocator())

    # Legend: one entry per index (color) + one per order (linestyle) + baseline
    legend_handles: list = []
    legend_labels: list = []

    index_order = ["BII", "FBII", "SII", "FSII"]
    shown_indices = [i for i in index_order if any((i, o) in stats for o in [1, 2, 3])]
    for index in shown_indices:
        legend_handles.append(
            Line2D([0], [0], color=INDEX_COLORS.get(index, "#888"), linewidth=2)
        )
        legend_labels.append(index)

    shown_orders = dict.fromkeys(ord_ for _, ord_ in combinations)
    for order in shown_orders:
        legend_handles.append(
            Line2D(
                [0], [0], color="#555", linestyle=ORDER_LINESTYLES[order], linewidth=1.3
            )
        )
        legend_labels.append(f"order {order}")

    ax.legend(
        legend_handles,
        legend_labels,
        fontsize=LEGEND_FONTSIZE,
        frameon=False,
        ncol=2,
        loc="upper left",
        handlelength=2.0,
        columnspacing=0.7,
        labelspacing=0.35,
    )

    apply_tick_style(ax)
    # fig.tight_layout(pad=0.4)

    fig.canvas.draw()
    for tick_label, color in zip(ax.get_yticklabels(), _ytick_colors):
        tick_label.set_color(color)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close(fig)


if __name__ == "__main__":
    plot_speedup_vs_depth()
