from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.legend_handler import HandlerLine2D
from matplotlib.lines import Line2D


class _WhiteBorderHandler(HandlerLine2D):
    def create_artists(
        self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans
    ):
        ymid = height / 2
        xmid = xdescent + width / 2
        color = orig_handle.get_color()
        line = Line2D(
            [xdescent, xdescent + width],
            [ymid, ymid],
            color=color,
            linewidth=2,
            marker="None",
            transform=trans,
        )
        white_dot = Line2D(
            [xmid],
            [ymid],
            color="white",
            linewidth=0,
            marker="o",
            markersize=7,
            transform=trans,
        )
        colored_dot = Line2D(
            [xmid],
            [ymid],
            color=color,
            linewidth=0,
            marker="o",
            markersize=5,
            transform=trans,
        )
        return [line, white_dot, colored_dot]


try:
    from special_plot_scripts._plot_style import (
        apply_tick_style,
        setup_fonts,
        W_REGULAR,
        W_SEMIBOLD,
    )
except ModuleNotFoundError:
    from _plot_style import apply_tick_style, setup_fonts, W_REGULAR, W_SEMIBOLD

setup_fonts()

# PANEL_FIGSIZE = (3.5, 4)
PANEL_FIGSIZE = (2.8, 3.2)
TITLE_FONT_SIZE = 11
LABEL_FONT_SIZE = 10
TICK_LABEL_FONT_SIZE = 12
LEGEND_FONT_SIZE = 9

# --- Layout knobs (tweak freely) ---------------------------------------------
# x position of the y-axis label (figure coords; smaller = further left)
YLABEL_X = 0.02
# y position of the x-axis label (figure coords; more negative = further down)
XLABEL_Y = -0.03
# Legend anchor in figure coords: (x_center, y); y < 0 places it below the figure
LEGEND_BBOX = (0.5, 0)
# Legend anchor reference point on the legend box itself
LEGEND_LOC = "upper center"
# Number of legend columns; None => one row (ncol = number of entries)
LEGEND_NCOL: int | None = None
# Spacing between subplots (and outer margins)
SUBPLOTS_BOTTOM = 0.1
SUBPLOTS_LEFT = 0.1
SUBPLOTS_WSPACE = 0.0
# -----------------------------------------------------------------------------

METHOD_RENAMING = {
    "regression": "FIxLIP+Baseline",
    "proxyshap": "ProxySHAP (XGBoost Det. MSR) [our]",
    "proxyshap-noadjustment": "FIxLIP+ProxySHAP (XGBoost) [our]",
    "proxyspex": "FIxLIP+ProxySPEX",
}
METHOD_TO_HEX_COLOR = {
    "FIxLIP+Baseline": "#000000",
    "FIxLIP+ProxySHAP (XGBoost) [our]": "#1e88e5",
    "FIxLIP+ProxySPEX": "#ef27a6",
}
METHOD_TO_ZORDER = {
    "FIxLIP+Baseline": 3,
    "FIxLIP+ProxySHAP (XGBoost) [our]": 7,
    "FIxLIP+ProxySPEX": 2,
}


def _plot_panel(ax: plt.Axes, df: pd.DataFrame, title: str) -> tuple[list, list]:
    """Plot one FIxLIP panel and return (handles, labels) for the shared legend."""
    order_2_data = df[df["order"] == 2]

    for method, method_data in order_2_data.groupby("method"):
        sorted_data = method_data.sort_values(by="budget")
        budget_grouped = (
            sorted_data.groupby("budget").agg({"area": "mean"}).reset_index()
        )
        color = METHOD_TO_HEX_COLOR[method]
        zorder = METHOD_TO_ZORDER[method]
        ax.plot(
            budget_grouped["budget"],
            budget_grouped["area"],
            marker="o",
            color="white",
            zorder=zorder,
            linewidth=4,
            markersize=7,
        )
        ax.plot(
            budget_grouped["budget"],
            budget_grouped["area"],
            marker="o",
            label=method,
            color=color,
            zorder=zorder,
            markersize=5,
            linewidth=2,
        )

    ax.set_title(f"FIxLIP: {title}", fontsize=TITLE_FONT_SIZE, fontweight=W_REGULAR)
    ax.set_xscale("log")
    ax.set_ylim(0, 1.08)
    ax.grid(linestyle="dotted")

    handles, labels = ax.get_legend_handles_labels()
    return handles, labels


if __name__ == "__main__":
    vit16_df = pd.read_csv("icml_submission_data/fixlip_aid_vit16_final.csv")
    vit32_df = pd.read_csv("icml_submission_data/fixlip_aid_vit32_final.csv")

    for df in (vit16_df, vit32_df):
        df["method"] = df["method"].map(METHOD_RENAMING)

    # Remove the Det. MSR variant from both datasets
    for df in (vit16_df, vit32_df):
        df.drop(
            df[df["method"] == "ProxySHAP (XGBoost Det. MSR) [our]"].index, inplace=True
        )

    fig_width = PANEL_FIGSIZE[0] * 2 + 0.6
    fig_height = PANEL_FIGSIZE[1]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(fig_width, fig_height))

    handles1, labels1 = _plot_panel(ax1, vit16_df, "ViT16")
    handles2, labels2 = _plot_panel(ax2, vit32_df, "ViT32")

    # Build a deduplicated legend from both panels
    by_label: dict = {}
    for handle, label in zip(handles1, labels1):
        by_label[label] = handle
    for handle, label in zip(handles2, labels2):
        if label not in by_label:
            by_label[label] = handle

    ax1.set_ylabel("")
    ax2.set_ylabel("")
    ax2.tick_params(labelleft=False, width=0)  # Hide y-axis labels on the right panel
    fig.supylabel(
        "Area under the Insertion-Deletion Curve",
        fontsize=LABEL_FONT_SIZE,
        fontweight=W_SEMIBOLD,
        x=YLABEL_X,
    )
    fig.supxlabel(
        "CLIP Calls",
        fontsize=LABEL_FONT_SIZE,
        fontweight=W_SEMIBOLD,
        y=XLABEL_Y,
    )

    fig.legend(
        list(by_label.values()),
        list(by_label.keys()),
        loc=LEGEND_LOC,
        bbox_to_anchor=LEGEND_BBOX,
        ncol=LEGEND_NCOL if LEGEND_NCOL is not None else len(by_label),
        frameon=True,
        fancybox=False,
        framealpha=0,
        edgecolor="#cccccc",
        fontsize=LEGEND_FONT_SIZE,
        handlelength=1.8,
        handletextpad=0.45,
        columnspacing=1.2,
        handler_map={Line2D: _WhiteBorderHandler()},
    )

    fig.subplots_adjust(
        bottom=SUBPLOTS_BOTTOM, wspace=SUBPLOTS_WSPACE, left=SUBPLOTS_LEFT
    )

    apply_tick_style(ax1, ax2, tick_label_fontsize=TICK_LABEL_FONT_SIZE)
    fig.patch.set_facecolor("none")
    fig.patch.set_alpha(0)
    for ax in (ax1, ax2):
        ax.patch.set_facecolor("none")
        ax.patch.set_alpha(0)
    fig.savefig(
        "plots/clip/main_fixlip.pdf",
        bbox_inches="tight",
        dpi=300,
        transparent=True,
        facecolor="none",
        edgecolor="none",
    )
    print("Saved to plots/clip/main_fixlip.pdf")
