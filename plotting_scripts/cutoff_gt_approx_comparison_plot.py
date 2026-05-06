from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from shapiq.interaction_values import InteractionValues

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "special_plot_scripts"))
from _plot_style import (  # noqa: E402
    W_REGULAR,
    W_SEMIBOLD,
    apply_tick_style,
    setup_fonts,
)
from computation_of_approximation_metrics_local import (  # noqa: E402
    iter_results_from_individual_files,
    iter_results_from_shards,
    iv_from_result_dict,
)

CONFIG_PATH = REPO_ROOT / "shapiq-benchmark/benchmarks/tabarena/cutoff_experiment.json"
GROUND_TRUTH_DIR = REPO_ROOT / "ground_truth/interventional"
APPROX_DIR = REPO_ROOT / "approximations/interventional"
SHARD_DIR = APPROX_DIR / "_shards"

CONFIG_APPROX = 37
RANDOM_STATE = 40
TARGET_BUDGETS = [1000, 5000, 10000]
MAX_BUDGET = 35_000
TINY = 1e-6

GAMES_TO_PLAYERS = {
    "TabArenaHelocLocalXAI": 23,
    "Corrgroups60LocalXAI": 60,
    "TabArenaCoil2000LocalXAI": 85,
}
# Soft palette — same family as plot_extraction_time_comparison.
DATASET_COLORS = {
    "TabArenaHelocLocalXAI": "#7FAFD4",   # softened blue
    "Corrgroups60LocalXAI": "#E29A73",    # softened vermilion
    "TabArenaCoil2000LocalXAI": "#86C9B1",  # softened bluish green
}
# Display labels — sourced from DATA_NAMES in special_plot_scripts/appendix_paper_plots.py.
DATASET_DISPLAY = {
    "TabArenaHelocLocalXAI": "Heloc ($n=23$)",
    "Corrgroups60LocalXAI": "CG60 ($n=60$)",
    "TabArenaCoil2000LocalXAI": "Coil2000 ($n=85$)",
}

# Top-level columns (ProxySPEX is special — replaced by a 2x2 mini-grid).
COLUMNS: list[dict] = [
    {"title": "ProxySHAP (XGBoost)", "approximator": "ProxySHAP (XGBoost)"},
    {"title": "ProxySPEX (XGBoost)", "approximator": None},  # 2x2 mini-grid below
    {"title": "SHAPIQ", "approximator": "SHAPIQ"},
    {"title": "PermutationSamplingSII", "approximator": "PermutationSamplingSII"},
    {"title": "KernelSHAPIQ", "approximator": "KernelSHAPIQ"},
]
# 2x2 placement of the four cutoffs inside the ProxySPEX cell.
PROXYSPEX_GRID = [
    [("0.9", "ProxySPEX90 (XGBoost)"),  ("0.95", "ProxySPEX (XGBoost)")],
    [("0.99", "ProxySPEX99 (XGBoost)"), ("0.999", "ProxySPEX999 (XGBoost)")],
]

# Method-level colors for the MSE row. Pink ramp for ProxySPEX cutoffs (light=0.9 -> dark=0.999),
# canonical colors for the others (sourced from shapiq_benchmark.plot.STYLE_DICT).
METHOD_STYLE: dict[str, dict] = {
    "ProxySHAP (XGBoost)":      {"color": "#61abec", "label": "ProxySHAP (XGBoost)",       "linestyle": "-"},
    "ProxySHAP (XGBoost, MSR)": {"color": "#1e88e5", "label": "ProxySHAP (XGBoost, MSR)", "linestyle": "-"},
    "ProxySPEX90 (XGBoost)":   {"color": "#f8a8db", "label": "ProxySPEX 0.9",         "linestyle": "-"},
    "ProxySPEX (XGBoost)":     {"color": "#ef27a6", "label": "ProxySPEX 0.95",        "linestyle": "-"},
    "ProxySPEX99 (XGBoost)":   {"color": "#a71b74", "label": "ProxySPEX 0.99",        "linestyle": "-"},
    "ProxySPEX999 (XGBoost)":  {"color": "#5d0f43", "label": "ProxySPEX 0.999",       "linestyle": "-"},
    "SHAPIQ":                  {"color": "#959595", "label": "SHAPIQ",                "linestyle": "-"},
    "PermutationSamplingSII":  {"color": "#252525", "label": "PermutationSamplingSII","linestyle": "-"},
    "KernelSHAPIQ":            {"color": "#ff6f00", "label": "KernelSHAPIQ",          "linestyle": "-"},
}
# Toggle to also plot the MSR-adjusted ProxySHAP curve in the MSE row.
INCLUDE_PROXYSHAP_MSR = True

# --- Font sizes (tweak these to scale ticks/labels) ---------------------------
TICK_FONTSIZE_MAIN = 12       # tick labels on main scatter axes
TICK_FONTSIZE_MSE = 12        # tick labels on MSE-vs-budget axes
TICK_FONTSIZE_MINI = 12       # tick labels on ProxySPEX 2x2 mini-axes
COL_TITLE_FONTSIZE = 14      # column titles ("ProxySHAP (XGBoost)" etc.)
ROW_LABEL_FONTSIZE = 14       # row y-labels ("Budget ≈ ...")
MSE_AXIS_LABEL_FONTSIZE = 14  # x/y axis labels on MSE row
MINI_CUTOFF_FONTSIZE = 12     # cutoff annotation inside ProxySPEX mini cells
DATASET_LEGEND_FONTSIZE = 14
METHOD_LEGEND_FONTSIZE = 14
SUPLABEL_FONTSIZE = 14       # fig.supxlabel / fig.supylabel

# Order in which methods are plotted / appear in the legend.
METHOD_ORDER: list[str] = [
    "ProxySHAP (XGBoost)",
    *(["ProxySHAP (XGBoost, MSR)"] if INCLUDE_PROXYSHAP_MSR else []),
    "ProxySPEX90 (XGBoost)",
    "ProxySPEX (XGBoost)",
    "ProxySPEX99 (XGBoost)",
    "ProxySPEX999 (XGBoost)",
    "SHAPIQ",
    "PermutationSamplingSII",
    "KernelSHAPIQ",
]


def get_budget_targets(n_players: int, targets: list[int], max_budget: int) -> list[int]:
    min_budget = n_players + 1
    grid = (
        np.ceil(np.logspace(np.log10(min_budget), np.log10(max_budget), 20))
        .clip(min_budget, max_budget)
        .astype(int)
    )
    return [int(grid[np.argmin(np.abs(grid - t))]) for t in targets]


def scale_values(values: np.ndarray, tiny: float = TINY) -> np.ndarray | None:
    filtered = values[np.abs(values) > tiny]
    if filtered.size == 0:
        return None
    lo, hi = float(filtered.min()), float(filtered.max())
    if hi - lo <= 0:
        return None
    return 2 * ((values - lo) / (hi - lo)) - 1


def load_approx_for_game(
    game: str, approximators: set[str], index: str, order: int
) -> dict[tuple[str, int, int], InteractionValues]:
    out: dict[tuple[str, int, int], InteractionValues] = {}
    game_id = f"{game}_3"
    for run, aname, budget, result_dict, _ in iter_results_from_shards(
        SHARD_DIR, game_id, CONFIG_APPROX, index, order
    ):
        if aname in approximators:
            try:
                out[(aname, run, budget)] = iv_from_result_dict(result_dict)
            except Exception as exc:
                print(f"  [WARN] shard parse failed {game} {aname} run={run} b={budget}: {exc}")
    for run, aname, budget, file_path, _ in iter_results_from_individual_files(
        APPROX_DIR, game_id, CONFIG_APPROX, index, order
    ):
        key = (aname, run, budget)
        if aname in approximators and key not in out:
            try:
                out[key] = InteractionValues.from_json_file(Path(file_path))
            except Exception as exc:
                print(f"  [WARN] individual load failed {file_path}: {exc}")
    return out


def load_ground_truth(
    game: str, n_runs: int, index: str, order: int
) -> dict[int, InteractionValues]:
    out: dict[int, InteractionValues] = {}
    for run in range(n_runs):
        path = (
            GROUND_TRUTH_DIR
            / f"{game}_3_{RANDOM_STATE}_{run}_{index}_{order}_exact_values.json"
        )
        if not path.exists():
            continue
        try:
            out[run] = InteractionValues.from_json_file(path)
        except Exception as exc:
            print(f"  [WARN] GT load failed {path.name}: {exc}")
    return out


def min_budget_for_approximator(aname: str, n_players: int, order: int) -> int:
    """Smallest budget at which a given approximator's regression problem is well-defined."""
    if aname == "KernelSHAPIQ":
        return sum(math.comb(n_players, k) for k in range(order + 1))
    return 0


def compute_mse_curves(
    aname: str,
    games: list[str],
    gt_data: dict,
    approx_data: dict,
    order: int,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Return {game: (budgets_sorted, mean_mse_per_budget)} for the given approximator."""
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for game in games:
        min_b = min_budget_for_approximator(aname, GAMES_TO_PLAYERS[game], order)
        per_budget: dict[int, list[float]] = {}
        for (a, run, budget), ap_iv in approx_data[game].items():
            if a != aname:
                continue
            if budget < min_b:
                continue
            gt_iv = gt_data[game].get(run)
            if gt_iv is None:
                continue
            ap_vals = np.array([ap_iv[inter] for inter in gt_iv.interactions])
            mse = float(np.sum((ap_vals - gt_iv.values) ** 2)) / np.sum(gt_iv.values ** 2)
            per_budget.setdefault(budget, []).append(mse)
        if not per_budget:
            continue
        budgets_sorted = np.array(sorted(per_budget.keys()))
        means = np.array([np.mean(per_budget[b]) for b in budgets_sorted])
        out[game] = (budgets_sorted, means)
    return out


def style_mse_axis(ax: plt.Axes, *, mini: bool = False) -> None:
    ax.set_xscale("log")
    ax.set_yscale("log")
    if mini:
        ax.tick_params(axis="both", which="both", labelsize=TICK_FONTSIZE_MINI, length=2)
    ax.grid(alpha=0.2, linewidth=0.5, which="both")


def gather_xy(
    aname: str,
    games: list[str],
    gt_data: dict,
    approx_data: dict,
    budgets_by_game: dict,
    row: int,
    n_runs: int,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Return {game: (gt_concat, approx_concat)} for the given approximator + row."""
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for game in games:
        budget = budgets_by_game[game][row]
        gt_chunks: list[np.ndarray] = []
        ap_chunks: list[np.ndarray] = []
        for run in range(n_runs):
            gt_iv = gt_data[game].get(run)
            ap_iv = approx_data[game].get((aname, run, budget))
            if gt_iv is None or ap_iv is None:
                continue
            ap_vals = np.array([ap_iv[inter] for inter in gt_iv.interactions])
            gt_scaled = scale_values(gt_iv.values)
            ap_scaled = scale_values(ap_vals)
            if gt_scaled is None or ap_scaled is None:
                continue
            gt_chunks.append(gt_scaled)
            ap_chunks.append(ap_scaled)
        if gt_chunks:
            out[game] = (np.concatenate(gt_chunks), np.concatenate(ap_chunks))
    return out


def style_scatter_axis(ax: plt.Axes, *, mini: bool = False) -> None:
    ax.plot([-1, 1], [-1, 1], color="#444444", linestyle="--", linewidth=0.8, zorder=10)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    if mini:
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(0.6)
            spine.set_color("#888888")
    else:
        ax.set_xticks([-1, 0, 1])
        ax.set_yticks([-1, 0, 1])
    ax.grid(alpha=0.2, linewidth=0.5)


def main(show_mse: bool = False) -> None:
    setup_fonts()
    cfg = json.loads(CONFIG_PATH.read_text())
    games = list(cfg.keys())
    sample = next(iter(cfg.values()))
    index = sample["index"]
    order = sample["order"]
    n_runs = sample["n_games"]

    approximators_needed: set[str] = {
        col["approximator"] for col in COLUMNS if col["approximator"] is not None
    }
    for row in PROXYSPEX_GRID:
        for _label, aname in row:
            approximators_needed.add(aname)
    approximators_needed.update(METHOD_ORDER)

    budgets_by_game = {
        g: get_budget_targets(GAMES_TO_PLAYERS[g], TARGET_BUDGETS, MAX_BUDGET) for g in games
    }
    print("Per-game budgets:", budgets_by_game)

    print("Loading ground truth...")
    gt_data = {g: load_ground_truth(g, n_runs, index, order) for g in games}
    print("Loading approximations...")
    approx_data = {g: load_approx_for_game(g, approximators_needed, index, order) for g in games}
    for g in games:
        print(f"  {g}: {len(approx_data[g])} approx entries, {len(gt_data[g])} GT runs")

    n_rows = len(TARGET_BUDGETS)
    n_cols = len(COLUMNS)
    # Optional extra row at the bottom for MSE-vs-budget curves.
    extra_rows = 1 if show_mse else 0
    fig_height = 3.1 * n_rows + (2.6 if show_mse else 0)
    fig = plt.figure(figsize=(3.0 * n_cols, fig_height))
    height_ratios = [3.1] * n_rows + ([2.6] if show_mse else [])
    outer = fig.add_gridspec(
        n_rows + extra_rows, n_cols, wspace=0.25, hspace=0.30, height_ratios=height_ratios,
    )

    main_axes: list[plt.Axes] = []
    mini_axes: list[plt.Axes] = []
    mse_axes: list[plt.Axes] = []

    def scatter_one_cell(ax: plt.Axes, aname: str, row: int, *, mini: bool, point_size: float):
        xy = gather_xy(aname, games, gt_data, approx_data, budgets_by_game, row, n_runs)
        for game, (xs, ys) in xy.items():
            ax.scatter(
                xs, ys, alpha=0.45, color=DATASET_COLORS[game],
                marker="o", s=point_size, linewidths=0, zorder=3,
            )
        style_scatter_axis(ax, mini=mini)

    for row, target_budget in enumerate(TARGET_BUDGETS):
        for col, col_spec in enumerate(COLUMNS):
            if col_spec["approximator"] is None:
                # ProxySPEX cell: 2x2 mini-grid sharing a transparent host axis for the title.
                inner = outer[row, col].subgridspec(2, 2, wspace=0.08, hspace=0.18)
                for r_mini in range(2):
                    for c_mini in range(2):
                        cutoff_label, aname = PROXYSPEX_GRID[r_mini][c_mini]
                        ax_m = fig.add_subplot(inner[r_mini, c_mini])
                        scatter_one_cell(ax_m, aname, row, mini=True, point_size=4)
                        ax_m.text(
                            0.04, 0.94, cutoff_label,
                            transform=ax_m.transAxes, ha="left", va="top",
                            fontsize=MINI_CUTOFF_FONTSIZE, fontweight=W_SEMIBOLD, color="#333333",
                            bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=1.0),
                        )
                        mini_axes.append(ax_m)
                if row == 0:
                    # Group title for ProxySPEX, centered above the 2x2 cell.
                    bbox = outer[row, col].get_position(fig)
                    fig.text(
                        (bbox.x0 + bbox.x1) / 2, bbox.y1 + 0.012,
                        "ProxySPEX (XGBoost)", ha="center", va="bottom",
                        fontsize=COL_TITLE_FONTSIZE, fontweight=W_REGULAR,
                    )
            else:
                ax = fig.add_subplot(outer[row, col])
                scatter_one_cell(ax, col_spec["approximator"], row, mini=False, point_size=10)
                if row == 0:
                    ax.set_title(col_spec["title"], fontsize=COL_TITLE_FONTSIZE, fontweight=W_REGULAR)
                if col == 0:
                    ax.set_ylabel(
                        f"Budget ≈ {target_budget}", fontsize=ROW_LABEL_FONTSIZE, fontweight=W_SEMIBOLD,
                    )
                main_axes.append(ax)

    if show_mse:
        # Bottom row: one MSE-vs-budget panel per dataset, with all methods overlaid.
        mse_row = n_rows
        n_datasets = len(games)
        bottom_inner = outer[mse_row, :].subgridspec(1, n_datasets, wspace=0.20)

        mse_curves_by_method: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {
            m: compute_mse_curves(m, games, gt_data, approx_data, order)
            for m in METHOD_ORDER
        }

        for ds_idx, game in enumerate(games):
            ax = fig.add_subplot(bottom_inner[0, ds_idx])
            for method in METHOD_ORDER:
                curves = mse_curves_by_method[method]
                if game not in curves:
                    continue
                xs, ys = curves[game]
                style = METHOD_STYLE[method]
                ax.plot(
                    xs, ys,
                    color=style["color"],
                    linestyle=style["linestyle"],
                    linewidth=1.6,
                    marker="o", markersize=3.5, markeredgewidth=0,
                    alpha=0.9, label=style["label"] if ds_idx == 0 else None,
                )
            style_mse_axis(ax, mini=False)
            ax.set_title(DATASET_DISPLAY[game], fontsize=COL_TITLE_FONTSIZE, fontweight=W_REGULAR)
            ax.set_xlabel("Budget", fontsize=MSE_AXIS_LABEL_FONTSIZE, fontweight=W_REGULAR)
            if ds_idx == 0:
                ax.set_ylabel("Relative MSE", fontsize=MSE_AXIS_LABEL_FONTSIZE, fontweight=W_SEMIBOLD)
            mse_axes.append(ax)

    dataset_handles = [
        Line2D(
            [0], [0], marker="o", color="w",
            markerfacecolor=DATASET_COLORS[g], markeredgecolor=DATASET_COLORS[g],
            markersize=7, label=DATASET_DISPLAY.get(g, g), linestyle="",
        )
        for g in games
    ]
    fig.legend(
        handles=dataset_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=len(games),
        frameon=False,
        fontsize=DATASET_LEGEND_FONTSIZE,
    )

    if show_mse:
        method_handles = [
            Line2D(
                [0], [0],
                color=METHOD_STYLE[m]["color"],
                linestyle=METHOD_STYLE[m]["linestyle"],
                marker="o", markersize=5, markeredgewidth=0,
                label=METHOD_STYLE[m]["label"], linewidth=1.6,
            )
            for m in METHOD_ORDER
        ]
        fig.legend(
            handles=method_handles,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.06),
            ncol=4,
            frameon=False,
            fontsize=METHOD_LEGEND_FONTSIZE,
        )

    fig.supxlabel(
        "Ground truth (normalized)", fontsize=SUPLABEL_FONTSIZE, y=0.04, fontweight=W_SEMIBOLD,
    )
    fig.supylabel(
        "Approximations (normalized)", fontsize=SUPLABEL_FONTSIZE, x=0.08, fontweight=W_SEMIBOLD,
    )

    apply_tick_style(*main_axes, tick_label_fontsize=TICK_FONTSIZE_MAIN)
    apply_tick_style(*mse_axes, tick_label_fontsize=TICK_FONTSIZE_MSE)

    out_path = REPO_ROOT / "plots" / "cutoff_gt_approx_comparison.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--show-mse",
        action="store_true",
        help="Include the bottom row of MSE-vs-budget curves.",
    )
    args = parser.parse_args()
    main(show_mse=args.show_mse)
