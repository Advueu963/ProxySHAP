"""Shared plotting helpers for the combined main-paper figure scripts.

The workflow is intentionally split into three layers:

1. `prepare_results_df(...)` loads a benchmark CSV, renames approximators,
    filters the selected game, and removes invalid budgets for specific method
    families.
2. `plot_panel_on_axis(...)` renders one subplot into an existing Matplotlib
    axis while reusing `plot_approximation_quality(...)` from shapiq-benchmark.
3. `plot_combined_main_paper_plots(...)` assembles multiple panels into a
    single figure, extracts one shared legend from the first panel, and places
    shared x/y labels and spacing controls at the figure level.

The concrete plot scripts only need to define `plot_specs`, pass the method
lists and renaming map, and optionally override spacing parameters such as
`bottom_margin`, `legend_bottom_margin`, or `shared_ylabel_x`.

Recommended script pattern:
- keep one `BASE_GENERAL_PARAMS` dictionary with all function-level defaults,
- override only a small `general_params` subset in each script,
- keep plot-specific limits like `max_budget` or `ylim` in `plot_specs`.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib import patheffects as pe
from matplotlib.transforms import Bbox

import shapiq_benchmark.plot as benchmark_plot
from shapiq_benchmark.plot import plot_approximation_quality

from _plot_style import apply_tick_style, setup_fonts, W_REGULAR, W_SEMIBOLD

setup_fonts()


def _read_csv_robust(path: Path) -> pd.DataFrame:
    """Read CSV with fallback to robust row normalization for malformed rows.
    
    Some CSVs (e.g., *_local_big_data.csv) have extra blank columns in certain rows.
    This function first tries standard pandas reading, then falls back to Python engine
    with row normalization if a ParserError occurs.
    """
    try:
        return pd.read_csv(path)
    except pd.errors.ParserError:
        print(f"Warning: parsing {Path(path).name} with robust row normalization due to malformed rows.")

    with Path(path).open(newline="") as file_handle:
        reader = csv.reader(file_handle)
        header = next(reader)
        rows = []
        normalized_rows = 0
        for row in reader:
            if len(row) > len(header):
                row = row[: len(header)]
                normalized_rows += 1
            elif len(row) < len(header):
                row = row + [""] * (len(header) - len(row))
                normalized_rows += 1
            rows.append(row)
    if normalized_rows:
        print(f"Warning: normalized {normalized_rows} malformed rows in {Path(path).name}.")
    df = pd.DataFrame(rows, columns=header)
    # Coerce numeric columns
    string_columns = {"game_type", "game", "model", "game_id", "approximator"}
    for column in df.columns:
        if column not in string_columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def _read_csv_robust(path: Path) -> pd.DataFrame:
    """Read CSV with fallback to robust row normalization for malformed rows.
    
    Some CSVs (e.g., *_local_big_data.csv) have extra blank columns in certain rows.
    This function first tries standard pandas reading, then falls back to Python engine
    with row normalization if a ParserError occurs.
    """
    try:
        return pd.read_csv(path)
    except pd.errors.ParserError:
        print(f"Warning: parsing {Path(path).name} with robust row normalization due to malformed rows.")

    with Path(path).open(newline="") as file_handle:
        reader = csv.reader(file_handle)
        header = next(reader)
        rows = []
        normalized_rows = 0
        for row in reader:
            if len(row) > len(header):
                row = row[: len(header)]
                normalized_rows += 1
            elif len(row) < len(header):
                row = row + [""] * (len(header) - len(row))
                normalized_rows += 1
            rows.append(row)
    if normalized_rows:
        print(f"Warning: normalized {normalized_rows} malformed rows in {Path(path).name}.")
    df = pd.DataFrame(rows, columns=header)
    # Coerce numeric columns
    string_columns = {"game_type", "game", "model", "game_id", "approximator"}
    for column in df.columns:
        if column not in string_columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def prepare_results_df(
    *,
    game_name,
    order,
    index,
    game_type,
    approximators_to_plot,
    approximator_renaming,
    linear_methods,
    min_budget=0,
    max_budget=float("inf"),
    data_path_template="icml_submission_data/results_benchmark_{index}_{order}_{game_type}.csv",
):
    """Load and filter one benchmark table for a single subplot.

    The returned DataFrame is already reduced to the selected game, the chosen
    approximators, and a log-spaced budget subset. The method-specific budget
    filtering mirrors the older single-plot scripts: linear methods are only
    kept once their coalition space is large enough, and `KernelSHAPIQ` is
    clipped to budgets that can support the requested order.

    Args:
        game_name: Benchmark game identifier, for example `ViT4by4Patches`.
        order: Interaction order used both for loading the CSV and for budget
            filtering.
        index: Benchmark index, such as `SII` or `SV`.
        game_type: CSV suffix describing the benchmark family.
        approximators_to_plot: Final approximator labels that should remain in
            the DataFrame.
        approximator_renaming: Mapping from raw benchmark method names to the
            labels used in the figure legend.
        linear_methods: Labels that require the stricter coalition-space budget
            check.
        min_budget: Lower budget bound applied before plotting.
        max_budget: Upper budget bound applied before plotting.
        data_path_template: Path template or list of templates tried in order
            until one resolves to an existing file.
        data_path_template: Path template or list of templates tried in order
            until one resolves to an existing file.

    Returns:
        A filtered `pandas.DataFrame` ready for plotting.
    """
    templates = (
        [data_path_template] if isinstance(data_path_template, str) else data_path_template
    )
    frames = []
    missing = []
    for template in templates:
        path = template.format(index=index, order=order, game_type=game_type)
        try:
            frames.append(_read_csv_robust(path))
        except FileNotFoundError:
            missing.append(path)
    if not frames:
        raise FileNotFoundError(
            f"Could not find benchmark CSV for index={index}, order={order}, "
            f"game_type={game_type}. Tried: {missing}"
        )
    # Apply the renaming to each frame BEFORE the override step so the override
    # matches on canonical names. Some CSVs contain both the raw name (e.g.
    # `ProxySHAP (XGBoost)`) and the already-renamed name (`ProxySHAP (XGBoost)
    # [our]`) as separate duplicate rows; without renaming first, an override
    # keyed on raw names only catches half of them and the surviving copies get
    # averaged with the new data, masking the actual change.
    frames = [f.replace({"approximator": approximator_renaming}) for f in frames]
    # Earlier templates take precedence at the `(game, approximator, budget)`
    # level. The intended ordering is: most-authoritative first (e.g. a re-run
    # CSV), then the production CSV, then any legacy fallback. For each later
    # frame, drop rows whose key already appears in an earlier frame so the
    # earlier reading is preserved and the later frame only contributes
    # coverage that doesn't exist higher up.
    override_keys = ["game", "approximator", "budget"]
    for later_idx in range(1, len(frames)):
        for earlier_idx in range(later_idx):
            earlier = frames[earlier_idx]
            if earlier.empty:
                continue
            already_covered = set(
                map(tuple, earlier[override_keys].drop_duplicates().to_numpy())
            )
            later = frames[later_idx]
            if later.empty:
                continue
            keys = list(zip(*(later[k] for k in override_keys)))
            mask = np.array([key not in already_covered for key in keys])
            frames[later_idx] = later[mask]
    results_df = pd.concat(frames, ignore_index=True)
    results_df = results_df[results_df["approximator"].isin(approximators_to_plot)]
    results_df = results_df[results_df["game"] == game_name]
    print("Available methods:", results_df["approximator"].unique())

    if results_df.empty:
        raise ValueError(
            f"No results found for game={game_name}, index={index}, order={order}, "
            f"game_type={game_type}. Available approximators: {results_df['approximator'].unique()}"
        )

    n_players = results_df["n_players"].values[0]
    min_b = n_players + 1
    min_b = n_players + 1
    max_b = min(2**n_players, max_budget) if n_players <= 20 else max_budget
    budget_range = (
        np.ceil(np.logspace(np.log10(min_b), np.log10(max_b), 20))
        .clip(min_b, max_b)
        .astype(int)
    )
    full_budget = 2**n_players
    full_budget = 2**n_players
    results_df = results_df[
        results_df["budget"].isin(budget_range)
        & (results_df["budget"] >= min_budget)
        & (results_df["budget"] <= max_budget)
        & (results_df["budget"] != full_budget)
        & (results_df["budget"] != full_budget)
    ]

    for method in linear_methods:
        method_mask = results_df["approximator"] == method
        valid_budgets = []
        for budget in results_df[method_mask]["budget"].unique():
            if budget >= sum([math.comb(n_players + 1, i) for i in range(0, order + 1)]):
                valid_budgets.append(budget)
        results_df = results_df[
            ~(method_mask & (~results_df["budget"].isin(valid_budgets)))
        ]

    for method in ["KernelSHAPIQ"]:
        method_mask = results_df["approximator"] == method
        valid_budgets = []
        for budget in results_df[method_mask]["budget"].unique():
            if budget >= math.comb(n_players + 1, order):
                valid_budgets.append(budget)
        results_df = results_df[
            ~(method_mask & (~results_df["budget"].isin(valid_budgets)))
        ]

    return results_df


def _auto_ylim_from_results_df(results_df, *, lower_factor=0.85, upper_factor=1.15):
    """Infer a y-range from all methods in a filtered results table."""
    """Infer a y-range from all methods in a filtered results table."""

    all_values = results_df["RelativeMSE"]
    lower = max(float(all_values.min()) * lower_factor, 1e-8)
    upper = float(all_values.max()) * upper_factor
    all_values = results_df["RelativeMSE"]
    lower = max(float(all_values.min()) * lower_factor, 1e-8)
    upper = float(all_values.max()) * upper_factor
    if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
        lower = 1e-8
        upper = 1.0
        lower = 1e-8
        upper = 1.0
    return lower, upper


def _ensure_min_runtime_xticks(ax, results_df, *, min_ticks=3):
    """Ensure runtime plots show at least `min_ticks` x-axis ticks."""
    if "total_runtime" not in results_df.columns:
        return
    runtime_values = results_df["total_runtime"].dropna().astype(float)
    if runtime_values.empty:
        return

    existing_ticks = [t for t in ax.get_xticks() if np.isfinite(t)]
    if len(existing_ticks) >= min_ticks:
        return

    x_scale = ax.get_xscale()
    low = float(runtime_values.min())
    high = float(runtime_values.max())
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return

    if x_scale == "log":
        positive_values = runtime_values[runtime_values > 0]
        if positive_values.empty:
            return
        low = float(positive_values.min())
        high = float(positive_values.max())
        if high <= low:
            return
        ticks = np.geomspace(low, high, min_ticks)
    else:
        ticks = np.linspace(low, high, min_ticks)

    ax.set_xticks(ticks)


def plot_panel_on_axis(
    ax,
    *,
    results_df,
    game_name,
    order,
    game_type,
    ylim,
    y_log_scale,
    x_log_scale,
    figsize,
    marker_size,
    linewidth,
    highlight_size,
    data_names,
    title_font_size,
    corner_label_mode="game_type",
    show_corner_label=True,
    corner_label_position="left",
    show_caption=True,
    show_x_ticks=True,
    style_dict=None,
    panel_grid_style=None,
    x_axis_mode="budget",
    runtime_smoothing_window=None,
    runtime_monotone=None,
    runtime_round_decimals=None,
    runtime_bin_count=None,
    runtime_binning=None,
):
    """Render one subplot into an existing axis.

    `plot_approximation_quality(...)` normally creates its own figure. This
    helper temporarily patches `matplotlib.pyplot.subplots` inside the
    shapiq-benchmark plotting module so the function draws into the caller's
    axis instead. That lets the combined figure keep a single shared layout
    while still reusing the upstream plotting code unchanged.

    After rendering, the temporary legend generated by the helper is removed,
    the panel title is stamped onto the axis when requested, the game-type tag
    is stamped onto the axis, and the legend handles/labels are returned so the
    caller can build a shared figure legend.
    """
    original_subplots = benchmark_plot.plt.subplots
    original_tight_layout = benchmark_plot.plt.tight_layout

    plot_kwargs = dict(
        data=results_df,
        metric="RelativeMSE",
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
    if x_axis_mode == "budget":
        plot_func = plot_approximation_quality
        plot_kwargs["plot_labels"] = False
        if style_dict is not None:
            plot_kwargs["style_dict"] = style_dict
    elif x_axis_mode == "runtime":
        plot_func = benchmark_plot.plot_approximation_quality_vstime
        if runtime_smoothing_window is not None:
            plot_kwargs["smoothing_window"] = int(runtime_smoothing_window)
        if runtime_monotone is not None:
            plot_kwargs["runtime_monotone"] = bool(runtime_monotone)
        if runtime_round_decimals is not None:
            plot_kwargs["runtime_round_decimals"] = int(runtime_round_decimals)
        if runtime_bin_count is not None:
            plot_kwargs["runtime_bin_count"] = int(runtime_bin_count)
        if runtime_binning is not None:
            plot_kwargs["runtime_binning"] = runtime_binning
    else:
        raise ValueError(f"Unknown x_axis_mode: {x_axis_mode}")

    try:
        benchmark_plot.plt.subplots = lambda *args, **kwargs: (ax.figure, ax)
        benchmark_plot.plt.tight_layout = lambda *args, **kwargs: None
        if style_dict is not None and x_axis_mode == "runtime":
            original_style = benchmark_plot.STYLE_DICT
            benchmark_plot.STYLE_DICT = style_dict
        else:
            original_style = None
        plot_func(**plot_kwargs)
    finally:
        if original_style is not None:
            benchmark_plot.STYLE_DICT = original_style
        benchmark_plot.plt.subplots = original_subplots
        benchmark_plot.plt.tight_layout = original_tight_layout

    legend = ax.get_legend()
    if legend is not None:
        legend.remove()

    if x_axis_mode == "runtime":
        _ensure_min_runtime_xticks(ax, results_df, min_ticks=3)

    if panel_grid_style is not None:
        ax.grid(True, **panel_grid_style)

    if not show_x_ticks:
        ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)

    game_type_abbreviations = {
        "exhaustive": "EXT",
        "interventional": "INT",
        "exhaustive_tabpfn": "TABPFN",
        "pathdependent": "PD",
    }

    if show_caption:
        ax.set_title(data_names[game_name], fontsize=title_font_size, fontweight=W_REGULAR)

    if corner_label_mode == "order":
        corner_label = f"order {order}"
    else:
        corner_label = game_type_abbreviations.get(game_type, game_type.upper())

    if show_corner_label:
        label_x = 0.97 if corner_label_position == "right" else 0.03
        label_ha = "right" if corner_label_position == "right" else "left"
        ax.text(
            label_x,
            0.04,
            corner_label,
            transform=ax.transAxes,
            fontsize=18,
            fontweight=W_SEMIBOLD,
            ha=label_ha,
            va="bottom",
        )
    if show_corner_label:
        label_x = 0.97 if corner_label_position == "right" else 0.03
        label_ha = "right" if corner_label_position == "right" else "left"
        ax.text(
            label_x,
            0.04,
            corner_label,
            transform=ax.transAxes,
            fontsize=18,
            fontweight=W_SEMIBOLD,
            ha=label_ha,
            va="bottom",
        )

    handles, labels = ax.get_legend_handles_labels()
    return handles, labels


def _is_legend_header_label(label):
    """Return `True` for legend entries that are structural headers.

    shapiq-benchmark includes entries like `Order all` and `Method` in the
    automatically generated legend. Those are useful for a standalone panel,
    but they are noise in the shared figure legend, so we strip them here.
    """
    normalized_label = label.replace("$\\bf{", "").replace("}$", "").strip().lower()
    if normalized_label in {"method", "order", "order all"}:
        return True
    if "order" in normalized_label and "all" in normalized_label:
        return True
    return False


def _clean_legend_entries(handles, labels):
    """Drop duplicate labels and structural header rows from a legend list."""
    filtered_handles = []
    filtered_labels = []
    seen_labels = set()
    for handle, label in zip(handles, labels, strict=True):
        if _is_legend_header_label(label):
            continue
        if label in seen_labels:
            continue
        seen_labels.add(label)
        filtered_handles.append(handle)
        filtered_labels.append(label)
    return filtered_handles, filtered_labels


def _estimate_legend_row_width(labels, fontsize=9):
    """Estimate how wide one legend row will be in figure-relative units."""
    char_width = 0.06 * (fontsize / 9)
    handle_width = 0.7
    spacing_width = 0.35
    total_width = 0.0
    for index, label in enumerate(labels):
        total_width += handle_width + (len(label) * char_width)
        if index < len(labels) - 1:
            total_width += spacing_width
    return total_width


def _choose_legend_ncol(fig, labels):
    """Pick the largest legend column count that still fits on one row.

    The function tries to keep the legend to a single line whenever possible.
    If the available figure width is too small, it progressively falls back to
    fewer columns until the estimated row width fits.
    """
    if not labels:
        return 1

    figure_width = fig.get_size_inches()[0]
    max_allowed_width = figure_width * 0.94

    for ncol in range(len(labels), 0, -1):
        per_row = math.ceil(len(labels) / ncol)
        longest_row = 0.0
        for row_index in range(per_row):
            row_labels = labels[row_index * ncol : (row_index + 1) * ncol]
            longest_row = max(longest_row, _estimate_legend_row_width(row_labels))
        if longest_row <= max_allowed_width:
            return ncol

    return 1


def _build_outlined_legend_handles(handles, labels, legend_scale=1.0, show_markers=True):
    """Create stroked legend handles that mirror plot line styles and markers."""
def _build_outlined_legend_handles(handles, labels, legend_scale=1.0, show_markers=True):
    """Create stroked legend handles that mirror plot line styles and markers."""
    outlined_handles = []
    for handle, label in zip(handles, labels, strict=True):
        if not isinstance(handle, Line2D):
            outlined_handles.append(handle)
            continue

        style = benchmark_plot.STYLE_DICT.get(label, {})
        color = style.get("color", handle.get_color())
        linestyle = style.get("linestyle", handle.get_linestyle())
        linewidth = handle.get_linewidth() * legend_scale

        style_marker = style.get("marker")
        raw_marker = style_marker if style_marker is not None else handle.get_marker()
        marker = raw_marker if (show_markers and raw_marker not in (None, "None", "none", "")) else "none"
        markersize = handle.get_markersize() * legend_scale if marker != "none" else 0

        style_marker = style.get("marker")
        raw_marker = style_marker if style_marker is not None else handle.get_marker()
        marker = raw_marker if (show_markers and raw_marker not in (None, "None", "none", "")) else "none"
        markersize = handle.get_markersize() * legend_scale if marker != "none" else 0

        legend_handle = Line2D(
            [],
            [],
            color=color,
            linestyle=linestyle,
            marker=marker,
            linewidth=linewidth,
            markersize=markersize,
            markerfacecolor=color if marker != "none" else "none",
            markeredgecolor=color if marker != "none" else "none",
            markeredgewidth=0.5 * legend_scale if marker != "none" else 0.0,
        )
        legend_handle.set_path_effects(
            [
                pe.Stroke(linewidth=linewidth + (1.8 * legend_scale), foreground="white"),
                pe.Normal(),
            ]
        )
        outlined_handles.append(legend_handle)

    return outlined_handles


def _order_legend_for_two_rows(handles, labels):
    """Return legend entries in a fixed order suited for a 2x3 legend grid.

    Desired columns (top, bottom):

    Matplotlib lays out legend entries column-wise for multi-column legends,
    so for 3 columns we return:
    [top1, bottom1, top2, bottom2, top3, bottom3].
    """
    desired_columns = [
        ("ProxySHAP (XGBoost) [our]", "ProxySHAP (XGBoost, MSR) [our]"),
        ("ProxySHAP* (XGBoost) [our]", "ProxySHAP* (XGBoost, MSR) [our]"),
        ("ProxySHAP (Linear) [our]", "ProxySHAP (Linear, MSR) [our]"),
        ("ProxySHAP (XGBoost) [our]", "ProxySHAP (XGBoost, MSR) [our]"),
        ("ProxySHAP* (XGBoost) [our]", "ProxySHAP* (XGBoost, MSR) [our]"),
        ("ProxySHAP (Linear) [our]", "ProxySHAP (Linear, MSR) [our]"),
    ]

    by_label = {label: handle for handle, label in zip(handles, labels, strict=True)}
    ordered_entries = []
    used = set()

    for top_label, bottom_label in desired_columns:
        if top_label in by_label:
            ordered_entries.append((by_label[top_label], top_label))
            used.add(top_label)
        if bottom_label in by_label:
            ordered_entries.append((by_label[bottom_label], bottom_label))
            used.add(bottom_label)

    # Keep all non-target entries at the end while preserving their original order.
    extras = [
        (handle, label)
        for handle, label in zip(handles, labels, strict=True)
        if label not in used
    ]

    ordered_entries = ordered_entries + extras
    ordered_handles = [handle for handle, _ in ordered_entries]
    ordered_labels = [label for _, label in ordered_entries]
    return ordered_handles, ordered_labels


_DEFERRED_LEGEND_LABELS = ("PermutationSamplingSII",)


def _draw_stretched_perm_sampling_sii(
    fig,
    legend,
    fontsize,
    *,
    x_left=None,
    x_right=None,
    y=None,
    x_left_offset=0.0,
    x_right_offset=0.0,
    y_offset=0.0,
):
    """Render PermutationSamplingSII below SHAPIQ–SVARMIQ.

    Positioning parameters (all in figure-relative coords 0..1):
        x_left         absolute left x of the text. None → auto-detect from SHAPIQ.
        x_right        absolute right x of the text. None → auto-detect from SVARMIQ.
        y              absolute baseline y. None → auto-detect from legend row 1.
        x_left_offset  added to auto-detected x_left (only when x_left is None).
        x_right_offset added to auto-detected x_right (only when x_right is None).
        y_offset       added to auto-detected y (only when y is None).

    The text artist is excluded from tight-bbox layout (set_in_layout(False))
    so it cannot grow the saved figure beyond the normal axes/legend bounds.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    label_to_text = {t.get_text(): t for t in legend.get_texts()}

    if x_left is None:
        if "SHAPIQ" not in label_to_text:
            return
        bb = label_to_text["SHAPIQ"].get_window_extent(renderer)
        x_left = inv.transform((bb.x0, 0))[0] + x_left_offset
    if x_right is None:
        if "SVARMIQ" not in label_to_text:
            return
        bb = label_to_text["SVARMIQ"].get_window_extent(renderer)
        x_right = inv.transform((bb.x1, 0))[0] + x_right_offset
    if y is None:
        if "SHAPIQ" not in label_to_text:
            return
        shapiq_bb = label_to_text["SHAPIQ"].get_window_extent(renderer)
        row1_y_px = None
        for text in legend.get_texts():
            bb = text.get_window_extent(renderer)
            if bb.y1 < shapiq_bb.y0 - 1:
                row_center_px = (bb.y0 + bb.y1) / 2
                row1_y_px = row_center_px if row1_y_px is None else max(row1_y_px, row_center_px)

        # Fallback: if no lower legend row can be inferred (for example due to
        # placeholder label handling), place the text roughly one line below SHAPIQ.
        if row1_y_px is None:
            row1_y_px = shapiq_bb.y0 - (0.55 * shapiq_bb.height)

        y = inv.transform((0, row1_y_px))[1] + y_offset

    # Keep baseline inside the legend's visible vertical extent.
    legend_bb = legend.get_window_extent(renderer)
    legend_y0 = inv.transform((0, legend_bb.y0))[1]
    legend_y1 = inv.transform((0, legend_bb.y1))[1]
    if legend_y1 > legend_y0:
        pad = 0.06 * (legend_y1 - legend_y0)
        y = min(max(y, legend_y0 + pad), legend_y1 - pad)
    if x_right <= x_left:
        return

    x_center = (x_left + x_right) / 2
    color = benchmark_plot.STYLE_DICT.get("PermutationSamplingSII", {}).get("color", "#252525")
    text_artist = fig.text(
        x_center,
        y,
        "PermutationSamplingSII",
        ha="center",
        va="center",
        fontsize=fontsize,
        color=color,
        zorder=10,
        transform=fig.transFigure,
    )
    # Keep the manual text out of tight-bbox calculations to avoid growing
    # the exported figure with large empty margins.
    text_artist.set_in_layout(False)
    # Figure-level custom label should not be clipped by any axes clip box.
    text_artist.set_clip_on(False)
    print(
        "[perm_sampling] manual text drawn "
        f"x_left={x_left:.4f} x_right={x_right:.4f} y={y:.4f}"
    )
    return text_artist


def _draw_manual_perm_sampling_text(
    fig,
    *,
    text,
    x,
    y,
    fontsize,
    color="#252525",
    draw_handle=True,
):
    """Draw a freely positionable figure-level text block for PermutationSamplingSII.

    Coordinates are in figure-relative space (0..1).
    """
    handle_artist = None
    if draw_handle:
        # Draw a compact legend-like handle (line + marker) directly before text.
        scale = max(0.6, fontsize / 9.0)
        handle_len = 0.027
        handle_gap = 0.008
        x2 = x - handle_gap
        x1 = x2 - handle_len
        if x1 < 0:
            # Keep the handle inside the figure when the text sits at the edge.
            x1 = 0.0
            x2 = x1 + handle_len
        xm = (x1 + x2) / 2
        line_width = 3
        marker_size = 9
        marker_edge_width =0
        handle_artist = Line2D(
            [x1, xm, x2],
            [y, y, y],
            transform=fig.transFigure,
            color=color,
            linewidth=line_width,
            marker="o",
            markersize=marker_size,
            markerfacecolor=color,
            markeredgecolor="white",
            markeredgewidth=marker_edge_width,
            markevery=[1],
            zorder=10,
        )
        handle_artist.set_path_effects(
            [
                pe.Stroke(linewidth=line_width + (1.8 * scale), foreground="white"),
                pe.Normal(),
            ]
        )
        handle_artist.set_in_layout(False)
        handle_artist.set_clip_on(False)
        fig.add_artist(handle_artist)

    text_artist = fig.text(
        x,
        y,
        text,
        ha="left",
        va="center",
        fontsize=fontsize+0.5,
        color=color,
        zorder=10,
        transform=fig.transFigure,
    )
    text_artist.set_in_layout(False)
    text_artist.set_clip_on(False)
    if handle_artist is not None:
        return [handle_artist, text_artist]
    return [text_artist]


def _order_legend_main_paper(handles, labels, marker_handles, marker_labels):
    """Return legend entries for the main-paper 2-row × N-column layout.

    Column layout (top row / bottom row per column):
      1. order marker entries  (circle = order 2, square = order 3)
      2. ProxySHAP (XGBoost)  / ProxySHAP (XGBoost, MSR)
      3. ProxySHAP (Linear)   / ProxySHAP (Linear, MSR)
      4. KernelSHAPIQ         / ProxySPEX (XGBoost)
      5. SHAPIQ               / (placeholder)
      6. SVARMIQ              / (placeholder)

    Matplotlib fills columns top→bottom, so the flat list must follow
    column-major order: [col0_top, col0_bot, col1_top, col1_bot, ...].
    The two bottom placeholders keep the bottom row aligned; PermutationSamplingSII
    is rendered manually after the legend draw so it can be stretched to span
    exactly from SHAPIQ.text.left to SVARMIQ.text.right.
    """
    desired_columns = [
        ("ProxySHAP (XGBoost) [our]", "ProxySHAP (XGBoost, MSR) [our]"),
        ("ProxySHAP (Linear) [our]", "ProxySHAP (Linear, MSR) [our]"),
        ("KernelSHAPIQ", "ProxySPEX (XGBoost)"),
        ("SHAPIQ", None),
        ("SVARMIQ", None),
    ]

    by_label = {label: handle for handle, label in zip(handles, labels, strict=True)}
    ordered_entries = []
    used = set()

    # First column: order marker entries (already pre-built, pass through as-is)
    for mh, ml in zip(marker_handles, marker_labels):
        ordered_entries.append((mh, ml))

    # Remaining columns
    for top_label, bottom_label in desired_columns:
        if top_label in by_label:
            ordered_entries.append((by_label[top_label], top_label))
            used.add(top_label)
            if bottom_label is None:
                placeholder = Line2D([], [], color="none", linestyle="none", marker="none")
                ordered_entries.append((placeholder, " "))
                continue
        if bottom_label is not None and bottom_label in by_label:
            ordered_entries.append((by_label[bottom_label], bottom_label))
            used.add(bottom_label)

    # Append any unlisted methods at the end.
    for handle, label in zip(handles, labels, strict=True):
        if label not in used:
            ordered_entries.append((handle, label))

    return (
        [h for h, _ in ordered_entries],
        [l for _, l in ordered_entries],
    )


def _order_legend_for_three_rows(handles, labels):
    """Return legend entries in a fixed order suited for a 3x5 legend grid."""

    has_permutation = (
        "PermutationSamplingSV" in labels or "PermutationSamplingSII" in labels
    )

    if not has_permutation:
        # BII-style layout: 3 rows x 4 columns with a fixed last column.
        desired_labels = [
            "ProxySHAP (XGBoost) [our]",
            "ProxySHAP (XGBoost, MSR) [our]",
            "ProxySHAP (Linear) [our]",
            "ProxySHAP (Linear, MSR) [our]",
            "ProxySHAP* (XGBoost) [our]",
            "ProxySHAP* (XGBoost, MSR) [our]",
            "ProxySpex",
            "SHAPIQ",
            "SVARMIQ",
        ]
        by_label = {label: handle for handle, label in zip(handles, labels, strict=True)}
        ordered_entries = []
        used = set()
        for label in desired_labels:
            if label in by_label:
                ordered_entries.append((by_label[label], label))
                used.add(label)
        extras = [
            (handle, label)
            for handle, label in zip(handles, labels, strict=True)
            if label not in used
        ]
        # Keep any extras before the fixed final column.
        final_column = [entry for entry in ordered_entries if entry[1] in {"ProxySpex", "SHAPIQ", "SVARMIQ"}]
        prefix = [entry for entry in ordered_entries if entry[1] not in {"ProxySpex", "SHAPIQ", "SVARMIQ"}]
        ordered_entries = prefix + extras + final_column
        ordered_handles = [handle for handle, _ in ordered_entries]
        ordered_labels = [label for _, label in ordered_entries]
        return ordered_handles, ordered_labels

    desired_rows = [
        [
            "ProxySHAP (XGBoost) [our]",
            "ProxySHAP (XGBoost, MSR) [our]",
            "ProxySHAP (Linear) [our]",
        ],
        [
            "ProxySHAP (Linear, MSR) [our]",
            "ProxySHAP* (XGBoost) [our]",
            "ProxySHAP* (XGBoost, MSR) [our]",
            "KernelSHAPIQ",
        ],
        [
            "ProxySpex",
            "PermutationSamplingSV",
            "PermutationSamplingSII",
            "SHAPIQ",
            "SVARMIQ",
        ],
    ]

    by_label = {label: handle for handle, label in zip(handles, labels, strict=True)}
    ordered_entries = []
    used = set()

    for row in desired_rows:
        for label in row:
            if label in by_label:
                ordered_entries.append((by_label[label], label))
                used.add(label)

    extras = [
        (handle, label)
        for handle, label in zip(handles, labels, strict=True)
        if label not in used
    ]

    ordered_entries = ordered_entries + extras
    ordered_handles = [handle for handle, _ in ordered_entries]
    ordered_labels = [label for _, label in ordered_entries]
    return ordered_handles, ordered_labels


def _build_order_marker_entries(plot_orders, legend_scale=1.0):
    """Create pre-styled marker legend handles encoding interaction order.

    Returns a pair (handles, labels) where each handle is a gray Line2D with
    the marker shape used for that order: circle for order 2, square for order 3.
    """
    marker_for_order = {2: "o", 3: "s"}
    color = "#555555"
    handles = []
    labels = []
    for order in sorted(plot_orders):
        marker = marker_for_order.get(order, "o")
        lw = 1.0 * legend_scale
        ms = 5 * legend_scale
        handle = Line2D(
            [],
            [],
            color=color,
            linestyle="-",
            marker=marker,
            linewidth=lw,
            markersize=ms,
            markerfacecolor=color,
            markeredgecolor=color,
            markeredgewidth=0.5 * legend_scale,
        )
        handle.set_path_effects(
            [
                pe.Stroke(linewidth=lw + (1.8 * legend_scale), foreground="white"),
                pe.Normal(),
            ]
        )
        handles.append(handle)
        labels.append(f"order {order}")
    return handles, labels


def _build_order_marker_entries(plot_orders, legend_scale=1.0):
    """Create pre-styled marker legend handles encoding interaction order.

    Returns a pair (handles, labels) where each handle is a gray Line2D with
    the marker shape used for that order: circle for order 2, square for order 3.
    """
    marker_for_order = {2: "o", 3: "s"}
    color = "#555555"
    handles = []
    labels = []
    for order in sorted(plot_orders):
        marker = marker_for_order.get(order, "o")
        lw = 1.0 * legend_scale
        ms = 5 * legend_scale
        handle = Line2D(
            [],
            [],
            color=color,
            linestyle="-",
            marker=marker,
            linewidth=lw,
            markersize=ms,
            markerfacecolor=color,
            markeredgecolor=color,
            markeredgewidth=0.5 * legend_scale,
        )
        handle.set_path_effects(
            [
                pe.Stroke(linewidth=lw + (1.8 * legend_scale), foreground="white"),
                pe.Normal(),
            ]
        )
        handles.append(handle)
        labels.append(f"order {order}")
    return handles, labels


def _scale_legend_value(value, scale, *, minimum=None):
    """Scale a legend layout value while optionally enforcing a minimum."""
    scaled_value = value * scale
    if minimum is not None:
        return max(minimum, scaled_value)
    return scaled_value


def _apply_custom_legend_order(handles, labels, desired_order):
    """Reorder legend entries to match a caller-supplied label sequence.

    Labels listed in `desired_order` come first in that exact order; any
    labels not listed are appended afterwards in their original order. Labels
    in `desired_order` that are not present in `labels` are silently skipped.
    """
    if not desired_order:
        return handles, labels
    by_label = {label: handle for handle, label in zip(handles, labels, strict=True)}
    ordered_handles = []
    ordered_labels = []
    used = set()
    for label in desired_order:
        if label in by_label and label not in used:
            ordered_handles.append(by_label[label])
            ordered_labels.append(label)
            used.add(label)
    for handle, label in zip(handles, labels, strict=True):
        if label not in used:
            ordered_handles.append(handle)
            ordered_labels.append(label)
    return ordered_handles, ordered_labels


def _apply_custom_legend_order(handles, labels, desired_order):
    """Reorder legend entries to match a caller-supplied label sequence.

    Labels listed in `desired_order` come first in that exact order; any
    labels not listed are appended afterwards in their original order. Labels
    in `desired_order` that are not present in `labels` are silently skipped.
    """
    if not desired_order:
        return handles, labels
    by_label = {label: handle for handle, label in zip(handles, labels, strict=True)}
    ordered_handles = []
    ordered_labels = []
    used = set()
    for label in desired_order:
        if label in by_label and label not in used:
            ordered_handles.append(by_label[label])
            ordered_labels.append(label)
            used.add(label)
    for handle, label in zip(handles, labels, strict=True):
        if label not in used:
            ordered_handles.append(handle)
            ordered_labels.append(label)
    return ordered_handles, ordered_labels


def plot_combined_main_paper_plots(
    *,
    plot_specs,
    approximators_to_plot,
    approximator_renaming,
    linear_methods,
    data_names,
    title_font_size,
    panel_figsize=(4, 3),
    figsize=None,
    legend_bottom_margin=None,
    bottom_margin=None,
    shared_ylabel_x=None,
    shared_xlabel_y=None,
    subplot_wspace=0.2,
    subplot_hspace=0.24,
    subplot_left_margin=None,
    plots_per_row=4,
    legend_scale=1.0,
    use_two_row_legend=False,
    show_captions_only_first_row=False,
    x_ticks_only_last_row=False,
    shared_ylabel_fontsize=12,
    shared_xlabel_fontsize=12,
    tick_label_fontsize=None,
    ylim=(1e-7, 1e2),
    y_log_scale=True,
    x_log_scale=True,
    min_budget=0,
    max_budget=float("inf"),
    marker_size=6,
    linewidth=2,
    highlight_size=2,
    output_path="plots/main/main_paper_plots_combined.pdf",
    legend_output_path=None,
    legend_ncol_override=None,
    auto_ylim=False,
    use_three_row_legend=False,
    corner_label_mode="game_type",
    data_path_template="icml_submission_data/results_benchmark_{index}_{order}_{game_type}.csv",
    style_dict=None,
    panel_grid_style=None,
    shared_xlabel="Model Evaluations",
    legend_show_markers=False,
    order_label_in_corner=False,
    column_headers=None,
    column_group_headers=None,
    row_title_on_right=False,
    legend_order=None,
    perm_sampling_x_left=None,
    perm_sampling_x_right=None,
    perm_sampling_y=None,
    perm_sampling_x_left_offset=0.0,
    perm_sampling_x_right_offset=0.0,
    perm_sampling_y_offset=0.0,
    perm_sampling_enabled=True,
    perm_sampling_manual_only=False,
    perm_sampling_text_x=None,
    perm_sampling_text_y=None,
    perm_sampling_text="PermutationSamplingSII",
    bottom_row_x_axis=None,
    runtime_smoothing_window=None,
    show_row_order_labels=True,
    runtime_round_decimals=None,
    runtime_bin_count=None,
    runtime_binning=None,
    runtime_monotone=None,
    runtime_adjust_evaluations=True,
    runtime_eval_time=None,
    show_legend=True,
):
    """Build one combined main-paper figure from multiple plot specifications.

    The function creates a grid of axes, renders one benchmark panel per entry
    in `plot_specs`, and then removes all per-axis legends. The first rendered
    panel provides the shared legend entries, which are filtered, deduplicated,
    and then placed below the entire figure.

    Layout behavior:
    - up to `plots_per_row` panels are arranged in a single row,
    - more panels are wrapped into rows with at most `plots_per_row` columns,
    - `panel_figsize` controls the size of a single subplot and the full figure
      is derived from that base size unless `figsize` is overridden,
    - `bottom_margin`, `legend_bottom_margin`, `shared_xlabel_y`, and
      `shared_ylabel_x` can be set globally or overridden per figure battery.

        Frequently tuned parameters (typically exposed via script-level
        `BASE_GENERAL_PARAMS`):
        - whitespace and placement: `bottom_margin`, `legend_bottom_margin`,
            `shared_xlabel_y`, `shared_ylabel_x`
        - shared label text sizing: `shared_xlabel_fontsize`,
            `shared_ylabel_fontsize`, `shared_xlabel`
        - geometry: `panel_figsize`, `figsize`
        - rendering style: `style_dict`, `panel_grid_style`
        - filtering and axes behavior: `min_budget`, `max_budget`,
            `y_log_scale`, `x_log_scale`

        Each `plot_spec` dictionary may contain:
        - `game_name`, `order`, `index`, `game_type`, `ylim` for the actual panel,
        - optional `max_budget` to override the global budget limit,
        - optional `figsize` for that one panel,
        - optional `bottom_margin` to override the figure bottom padding,
        - optional `x_axis_mode` set to "budget" or "runtime" to override
            the row-based x-axis choice.

    Args:
        plot_specs: Ordered list of panel configurations.
        approximators_to_plot: Final labels to keep in the benchmark data.
        approximator_renaming: Raw-to-display label mapping.
        linear_methods: Method labels that require the stricter budget filter.
        data_names: Human-readable game title mapping.
        title_font_size: Font size for each subplot title.
        panel_figsize: Base size of one panel before grid expansion.
        figsize: Optional explicit override for the full figure size.
        legend_bottom_margin: Space between the figure content and the legend.
        bottom_margin: Space reserved below the subplot grid before legend and
            shared x-label are placed.
        shared_ylabel_x: Horizontal position of the figure-level y-label.
        shared_xlabel_y: Vertical position of the figure-level x-label.
        subplot_wspace: Horizontal spacing between subplot columns.
        subplot_hspace: Vertical spacing between subplot rows.
        plots_per_row: Maximum number of panels placed in each row.
        legend_scale: Scaling factor for legend text, handles, and spacing.
        use_two_row_legend: If true, force a two-row legend with fixed method
            ordering tailored to the main-paper method family layout.
        show_captions_only_first_row: If true, only the first row of panels
            receives a title/order caption.
        x_ticks_only_last_row: If true, only the last row keeps x-axis ticks
            and tick labels.
        shared_ylabel_fontsize: Font size for the figure-level y-label.
        shared_xlabel_fontsize: Font size for the figure-level x-label.
        tick_label_fontsize: Optional font size applied to all x- and y-tick
            labels in the rendered panels.
        tick_label_fontsize: Optional font size applied to all x- and y-tick
            labels in the rendered panels.
        ylim: Legacy parameter kept for API compatibility.
        y_log_scale: Whether the y-axis should use log scaling.
        x_log_scale: Whether the x-axis should use log scaling.
        min_budget: Global minimum budget applied to every panel.
        max_budget: Global maximum budget used unless a panel overrides it.
        marker_size: Passed through to the upstream plotting helper.
        linewidth: Passed through to the upstream plotting helper.
        highlight_size: Passed through to the upstream plotting helper.
        output_path: File path of the final combined figure.
        legend_output_path: Optional file path for a separate legend export.
        data_path_template: CSV path template used by `prepare_results_df(...)`.
        style_dict: Optional plotting style dictionary for the upstream helper.
        panel_grid_style: Optional grid keyword arguments applied to each panel.
        shared_xlabel: Figure-level x-label text.
        legend_order: Optional iterable of display labels defining the legend
            order. Listed labels appear first in the given sequence; any
            remaining labels are appended in their original order. Labels not
            present in the figure are silently skipped. Overrides the layouts
            applied by `use_two_row_legend` / `use_three_row_legend`.
        bottom_row_x_axis: If set to "runtime", plots the last row against
            total runtime instead of model evaluations.
        runtime_smoothing_window: Optional rolling-mean window (number of points)
            applied to runtime plots.
        runtime_round_decimals: Optional rounding applied to runtime values before
            binning.
        runtime_bin_count: Number of bins used for runtime aggregation.
        runtime_binning: Binning strategy for runtime curves: "log" or "linear".
        runtime_monotone: If true, enforce a non-increasing metric curve over runtime.
        runtime_adjust_evaluations: Whether to subtract game evaluation time from
            total runtime when plotting runtime on the x-axis.
        runtime_eval_time: Optional per-evaluation time to add back using
            `used_budget`. When provided, total runtime becomes
            `total_runtime - evaluations + runtime_eval_time * used_budget`.
        show_row_order_labels: Whether to add row-level order labels on the
            right side when `corner_label_mode` is "order".

    Returns:
        None. The figure is written directly to `output_path`.
    """
    num_plots = len(plot_specs)
    if plots_per_row < 1:
        raise ValueError("plots_per_row must be at least 1")
    if legend_scale <= 0:
        raise ValueError("legend_scale must be greater than 0")

    if bottom_row_x_axis is None:
        bottom_row_x_axis = "budget"
    if bottom_row_x_axis not in {"budget", "runtime"}:
        raise ValueError("bottom_row_x_axis must be one of: budget, runtime")

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
    legend_handles = None
    legend_labels = None
    last_row_start_index = max(0, (n_rows - 1) * n_cols)
    spec_bottom_margin = next(
        (plot_spec.get("bottom_margin") for plot_spec in plot_specs if plot_spec.get("bottom_margin") is not None),
        None,
    )
    # plot_approximation_quality mutates the global benchmark_plot.STYLE_DICT when
    # style_dict is passed (``global STYLE_DICT = style_dict``).  Save a snapshot
    # before the loop so that (a) order-2 style is never overwritten by a prior
    # order-3 render and (b) we can restore the original before building the legend.
    import copy as _copy
    _original_style_dict = _copy.deepcopy(benchmark_plot.STYLE_DICT)

    axis_x_axis_modes: list[str] = []
    axis_show_x_ticks: list[bool] = []

    for axis_index, (axis, plot_spec) in enumerate(zip(axes_flat[:num_plots], plot_specs)):
        spec_max_budget = plot_spec.get("max_budget", max_budget)
        show_caption = (not row_title_on_right) and (
            not show_captions_only_first_row or axis_index < n_cols
        )
        show_caption = (not row_title_on_right) and (
            not show_captions_only_first_row or axis_index < n_cols
        )
        show_x_ticks = not x_ticks_only_last_row or axis_index >= last_row_start_index
        axis_show_x_ticks.append(show_x_ticks)

        is_last_in_row = (axis_index + 1) % n_cols == 0 or axis_index == num_plots - 1
        show_panel_corner_label = corner_label_mode != "order" or order_label_in_corner
        corner_label_pos = (
            "left"
            if order_label_in_corner or corner_label_mode != "order"
            else ("right" if is_last_in_row else "left")
        )

        panel_style_dict = style_dict
        if corner_label_mode == "order":
            if plot_spec.get("order") == 3:
                order3_style = {
                    label: {**props, "marker": "s"}
                    for label, props in _original_style_dict.items()
                }
                if style_dict is not None:
                    for lbl, props in style_dict.items():
                        order3_style[lbl] = {**order3_style.get(lbl, {}), **props}
                panel_style_dict = order3_style
            else:
                # Order 2: always pass an explicit copy of the original style so that
                # a prior order-3 render cannot pollute the global STYLE_DICT used here.
                order2_style = _copy.deepcopy(_original_style_dict)
                # Give circle markers to baselines that have no marker defined.
                for lbl, props in order2_style.items():
                    if props.get("marker") is None:
                        order2_style[lbl] = {**props, "marker": "o"}
                if style_dict is not None:
                    for lbl, props in style_dict.items():
                        order2_style[lbl] = {**order2_style.get(lbl, {}), **props}
                panel_style_dict = order2_style

        row_index = axis_index // n_cols
        x_axis_mode = plot_spec.get("x_axis_mode")
        if x_axis_mode is None:
            x_axis_mode = bottom_row_x_axis if row_index == (n_rows - 1) else "budget"
        if x_axis_mode not in {"budget", "runtime"}:
            raise ValueError(f"Unknown x_axis_mode: {x_axis_mode}")
        axis_x_axis_modes.append(x_axis_mode)

        spec_exclude = set(plot_spec.get("exclude_approximators", ()))
        spec_approximators = (
            [a for a in approximators_to_plot if a not in spec_exclude]
            if spec_exclude
            else approximators_to_plot
        )
        try:
            results_df = prepare_results_df(
                game_name=plot_spec["game_name"],
                order=plot_spec["order"],
                index=plot_spec["index"],
                game_type=plot_spec["game_type"],
                approximators_to_plot=spec_approximators,
                approximator_renaming=approximator_renaming,
                linear_methods=linear_methods,
                min_budget=min_budget,
                max_budget=spec_max_budget,
                data_path_template=data_path_template,
            )
        except ValueError as e:
            print(f"Skipping panel {axis_index}: {e}")
            continue
        if x_axis_mode == "runtime" and "total_runtime" in results_df.columns:
            runtime_df = results_df.copy()
            if runtime_adjust_evaluations and "evaluations" in runtime_df.columns:
                runtime_df["total_runtime"] = (
                    runtime_df["total_runtime"]
                )
            if runtime_eval_time is not None and "used_budget" in runtime_df.columns:
                runtime_df["total_runtime"] = (
                    runtime_df["total_runtime"] + float(runtime_eval_time) * runtime_df["used_budget"]
                )
            group_keys = [
                k for k in ["game_id", "approximator", "budget"] if k in runtime_df.columns
            ]
            if group_keys:
                runtime_df["total_runtime"] = runtime_df.groupby(group_keys)[
                    "total_runtime"
                ].transform("mean")
            if runtime_round_decimals is not None:
                runtime_df["total_runtime"] = runtime_df["total_runtime"].round(
                    int(runtime_round_decimals)
                )
            results_df = runtime_df
        panel_ylim = plot_spec.get("ylim")
        if auto_ylim or panel_ylim is None:
            panel_ylim = _auto_ylim_from_results_df(results_df)
        handles, labels = plot_panel_on_axis(
            axis,
            results_df=results_df,
            game_name=plot_spec["game_name"],
            order=plot_spec["order"],
            game_type=plot_spec["game_type"],
            ylim=panel_ylim,
            y_log_scale=y_log_scale,
            x_log_scale=x_log_scale,
            figsize=plot_spec.get("figsize", panel_figsize),
            marker_size=marker_size,
            linewidth=linewidth,
            highlight_size=highlight_size,
            data_names=data_names,
            title_font_size=title_font_size,
            corner_label_mode=corner_label_mode,
            show_corner_label=show_panel_corner_label,
            corner_label_position=corner_label_pos,
            show_caption=show_caption,
            show_x_ticks=show_x_ticks,
            style_dict=panel_style_dict,
            panel_grid_style=panel_grid_style,
            x_axis_mode=x_axis_mode,
            runtime_smoothing_window=runtime_smoothing_window,
            runtime_monotone=runtime_monotone,
            runtime_round_decimals=runtime_round_decimals,
            runtime_bin_count=runtime_bin_count,
            runtime_binning=runtime_binning,
        )

        xlim_override = plot_spec.get("xlim")
        if x_axis_mode == "runtime" and plot_spec.get("xlim_runtime") is not None:
            xlim_override = plot_spec.get("xlim_runtime")
        if x_axis_mode == "budget" and plot_spec.get("xlim_budget") is not None:
            xlim_override = plot_spec.get("xlim_budget")
        if xlim_override is not None:
            axis.set_xlim(xlim_override)

        panel_handles, panel_labels = _clean_legend_entries(handles, labels)
        if legend_handles is None:
            legend_handles, legend_labels = panel_handles, panel_labels
        else:
            for handle, label in zip(panel_handles, panel_labels):
                if label not in legend_labels:
                    legend_handles.append(handle)
                    legend_labels.append(label)

    for axis_index, axis in enumerate(axes_flat):
        if axis_index >= num_plots:
            axis.set_visible(False)
            continue
        axis.set_ylabel("")
        axis.set_xlabel("")

    x_axis_mode_set = set(axis_x_axis_modes[:num_plots])
    mixed_x_axes = len(x_axis_mode_set) > 1
    if x_axis_mode_set == {"runtime"} and shared_xlabel == "Model Evaluations":
        if runtime_eval_time == 0.001:
            shared_xlabel = "Total Runtime (s; 1ms per Model call)"
        if runtime_eval_time == 0.01:
            shared_xlabel = "Total Runtime (s; 10ms per Model call)"
        if runtime_eval_time == 0.1:
            shared_xlabel = "Total Runtime (s; 100ms per Model call)"
        if runtime_eval_time == 1.0:
            shared_xlabel = "Total Runtime (s; 1s per Model call)"
        if runtime_eval_time == 0:
            shared_xlabel = "Total Runtime (s; without Model call time)"
        raise ValueError(
            "Mixed x-axis modes detected but shared_xlabel is still 'Model Evaluations'. "            "Please set shared_xlabel to None or a custom label that applies to all panels."
        )
    if mixed_x_axes and shared_xlabel == "Model Evaluations":
        shared_xlabel = None
    if mixed_x_axes and shared_xlabel is None:
        for axis_index, axis in enumerate(axes_flat[:num_plots]):
            if not axis_show_x_ticks[axis_index]:
                continue
            axis_mode = axis_x_axis_modes[axis_index]
            axis_xlabel = "Total Runtime (s)" if axis_mode == "runtime" else "Model Evaluations"
            axis.set_xlabel(axis_xlabel, fontsize=shared_xlabel_fontsize, fontweight=W_SEMIBOLD)
    extra_row_xlabels: list[tuple[int, str]] = []
    if mixed_x_axes and shared_xlabel is not None:
        # A custom figure-level `shared_xlabel` (e.g. "Total Runtime (s; 10ms
        # per Model call)") is centered below the bottom row only. For each
        # non-runtime row, queue a row-level "Model Evaluations" caption that
        # we'll place after `subplots_adjust` so the axis positions are final.
        for row_idx in range(n_rows):
            row_axes_modes = axis_x_axis_modes[row_idx * n_cols : (row_idx + 1) * n_cols]
            if row_axes_modes and all(mode == "budget" for mode in row_axes_modes):
                extra_row_xlabels.append((row_idx, "Model Evaluations"))

    # Snapshot labels before reordering — needed to detect deferred labels (e.g.
    # PermutationSamplingSII) that are dropped from the legend grid for manual placement.
    original_legend_labels = list(legend_labels) if legend_labels is not None else []

    # Build marker-style entries for order encoding (circle = order 2, square = order 3).
    extra_marker_handles: list = []
    extra_marker_labels: list = []
    if corner_label_mode == "order":
        plot_orders = sorted({spec["order"] for spec in plot_specs})
        if len(plot_orders) > 1:
            extra_marker_handles, extra_marker_labels = _build_order_marker_entries(
                plot_orders, legend_scale=legend_scale
            )

    if use_three_row_legend:
        legend_handles, legend_labels = _order_legend_for_three_rows(legend_handles, legend_labels)
        has_permutation = (
            "PermutationSamplingSV" in legend_labels
            or "PermutationSamplingSII" in legend_labels
        )
        legend_ncol = 5 if has_permutation else 4
    elif use_two_row_legend and extra_marker_handles:
        # Main-paper layout: markers column first, then method pairs, ncol=4.
        legend_handles, legend_labels = _order_legend_main_paper(
            legend_handles, legend_labels, extra_marker_handles, extra_marker_labels
        )
        extra_marker_handles = []
        extra_marker_labels = []
        legend_ncol = 4
    elif use_two_row_legend and extra_marker_handles:
        # Main-paper layout: markers column first, then method pairs, ncol=4.
        legend_handles, legend_labels = _order_legend_main_paper(
            legend_handles, legend_labels, extra_marker_handles, extra_marker_labels
        )
        extra_marker_handles = []
        extra_marker_labels = []
        legend_ncol = 4
    elif use_two_row_legend:
        legend_handles, legend_labels = _order_legend_for_two_rows(legend_handles, legend_labels)
        legend_ncol = 3
    else:
        legend_ncol = _choose_legend_ncol(fig, legend_labels + extra_marker_labels)

    if legend_order:
        legend_handles, legend_labels = _apply_custom_legend_order(
            legend_handles, legend_labels, legend_order
        )

    if perm_sampling_manual_only:
        filtered = [
            (h, l)
            for h, l in zip(legend_handles, legend_labels, strict=True)
            if l != "PermutationSamplingSII"
        ]
        legend_handles = [h for h, _ in filtered]
        legend_labels = [l for _, l in filtered]
        legend_ncol = _choose_legend_ncol(fig, legend_labels + extra_marker_labels)

    if legend_order:
        legend_handles, legend_labels = _apply_custom_legend_order(
            legend_handles, legend_labels, legend_order
        )

    if perm_sampling_manual_only:
        filtered = [
            (h, l)
            for h, l in zip(legend_handles, legend_labels, strict=True)
            if l != "PermutationSamplingSII"
        ]
        legend_handles = [h for h, _ in filtered]
        legend_labels = [l for _, l in filtered]

    if legend_ncol_override is not None:
        legend_ncol = max(1, int(legend_ncol_override))

    # Add row-level order labels on the right side of each row of panels.
    # Skipped when column_headers are used — the column headers carry the order info instead.
    if (
        show_row_order_labels
        and corner_label_mode == "order"
        and not order_label_in_corner
        and column_headers is None
    ):
        for row_idx in range(n_rows):
            spec_idx = row_idx * n_cols
            if spec_idx >= num_plots:
                continue
            row_order = plot_specs[spec_idx]["order"]
            last_ax_idx = min((row_idx + 1) * n_cols - 1, num_plots - 1)
            last_ax = axes_flat[last_ax_idx]
            last_ax.annotate(
                f"order {row_order}",
                xy=(1.04, 0.5),
                xycoords="axes fraction",
                fontsize=16,
                fontweight=W_SEMIBOLD,
                ha="left",
                va="center",
                rotation=90,
                annotation_clip=False,
            )

    all_legend_labels = legend_labels + extra_marker_labels
    legend_rows = math.ceil(len(all_legend_labels) / legend_ncol) if all_legend_labels else 1
    if legend_bottom_margin is None:
        legend_bottom_margin = 0.016 + (0.008 * max(0, legend_rows - 1))
    if bottom_margin is None:
        bottom_margin = spec_bottom_margin
    if bottom_margin is None:
        bottom_margin = 0.078 + (0.024 * max(0, legend_rows - 1))

    # Restore original style so _build_outlined_legend_handles reads the canonical
    # marker shapes (circles), not the mutated square-only style from the last order-3 panel.
    benchmark_plot.STYLE_DICT = _original_style_dict

    legend_display_handles = list(
        _build_outlined_legend_handles(
            legend_handles, legend_labels, legend_scale=legend_scale, show_markers=legend_show_markers
        )
    ) + extra_marker_handles
    # Restore original style so _build_outlined_legend_handles reads the canonical
    # marker shapes (circles), not the mutated square-only style from the last order-3 panel.
    benchmark_plot.STYLE_DICT = _original_style_dict

    legend_display_handles = list(
        _build_outlined_legend_handles(
            legend_handles, legend_labels, legend_scale=legend_scale, show_markers=legend_show_markers
        )
    ) + extra_marker_handles
    legend_fontsize = _scale_legend_value(9, legend_scale, minimum=5)
    legend_handlelength = _scale_legend_value(1.5, legend_scale, minimum=0.8)
    legend_handletextpad = _scale_legend_value(0.45, legend_scale, minimum=0.2)
    legend_columnspacing = _scale_legend_value(0.9, legend_scale, minimum=0.4)
    legend_borderaxespad = _scale_legend_value(0.0, legend_scale)
    legend_labelspacing = _scale_legend_value(0.35, legend_scale, minimum=0.15)

    shared_legend = None
    if legend_output_path is None and show_legend:
        shared_legend = fig.legend(
            legend_display_handles,
            all_legend_labels,
            all_legend_labels,
            loc="lower center",
            bbox_to_anchor=(0.5, -legend_bottom_margin),
            ncol=legend_ncol,
            frameon=False,
            fancybox=False,
            framealpha=0,
            fontsize=legend_fontsize,
            handlelength=legend_handlelength,
            handletextpad=legend_handletextpad,
            columnspacing=legend_columnspacing,
            borderaxespad=legend_borderaxespad,
            labelspacing=legend_labelspacing,
        )
    if shared_xlabel_y is None:
        shared_xlabel_y = 0.055 if n_rows == 1 else 0.045
    if shared_ylabel_x is None:
        shared_ylabel_x = 0.015 if n_cols == 1 else 0.008
    fig.supylabel(
        "Relative MSE ± SEM",
        fontsize=shared_ylabel_fontsize,
        x=shared_ylabel_x,
        fontweight=W_SEMIBOLD,
    )
    if shared_xlabel:
        fig.supxlabel(
            shared_xlabel,
            fontsize=shared_xlabel_fontsize,
            y=shared_xlabel_y,
            fontweight=W_SEMIBOLD,
        )
    adjust_kwargs = dict(bottom=bottom_margin, wspace=subplot_wspace, hspace=subplot_hspace)
    if subplot_left_margin is not None:
        adjust_kwargs["left"] = subplot_left_margin
    if corner_label_mode == "order" and n_rows > 1 and column_headers is None:
        adjust_kwargs.setdefault("right", 0.93)
    fig.subplots_adjust(**adjust_kwargs)

    for row_idx, row_label in extra_row_xlabels:
        row_axes = [
            axes_flat[i]
            for i in range(row_idx * n_cols, min((row_idx + 1) * n_cols, num_plots))
        ]
        if not row_axes:
            continue
        positions = [a.get_position() for a in row_axes]
        x_center = 0.5 * (
            min(p.x0 for p in positions) + max(p.x1 for p in positions)
        )
        # Sit the caption just below the row's bottom edge, leaving enough
        # gap so it doesn't collide with the next row's axis ticks.
        row_bottom = min(p.y0 for p in positions)
        next_row_top = max(
            (a.get_position().y1 for a in axes_flat[(row_idx + 1) * n_cols : (row_idx + 2) * n_cols]),
            default=row_bottom - 0.06,
        )
        y_caption = 0.5 * (row_bottom + next_row_top)
        fig.text(
            x_center,
            y_caption,
            row_label,
            fontsize=shared_xlabel_fontsize,
            fontweight=W_SEMIBOLD,
            ha="center",
            va="center",
        )

    if (
        shared_legend is not None
        and perm_sampling_enabled
        and "PermutationSamplingSII" in original_legend_labels
        and (
            perm_sampling_manual_only
            or "PermutationSamplingSII" not in all_legend_labels
        )
    ):
        if perm_sampling_manual_only and perm_sampling_text_x is not None and perm_sampling_text_y is not None:
            color = benchmark_plot.STYLE_DICT.get("PermutationSamplingSII", {}).get("color", "#252525")
            _draw_manual_perm_sampling_text(
                fig,
                text=perm_sampling_text,
                x=float(perm_sampling_text_x),
                y=float(perm_sampling_text_y),
                fontsize=legend_fontsize,
                color=color,
            )
            print(
                "[perm_sampling] manual text-block drawn "
                f"x={perm_sampling_text_x} y={perm_sampling_text_y} text={perm_sampling_text}"
            )
       
    if legend_output_path is not None:
        legend_fig_height = 3.2 if legend_ncol >= 5 else 2.6
        fig_leg = plt.figure(figsize=(12.0, legend_fig_height))
        legend = fig_leg.legend(
            legend_display_handles,
            all_legend_labels,
            all_legend_labels,
            loc="center",
            ncol=legend_ncol,
            frameon=False,
            fancybox=False,
            framealpha=0,
            fontsize=legend_fontsize,
            handlelength=legend_handlelength,
            handletextpad=legend_handletextpad,
            columnspacing=legend_columnspacing,
            borderaxespad=legend_borderaxespad,
            labelspacing=legend_labelspacing,
        )
        extra_legend_artists = []
        if (
            perm_sampling_enabled
            and "PermutationSamplingSII" in original_legend_labels
            and (
                perm_sampling_manual_only
                or "PermutationSamplingSII" not in all_legend_labels
            )
        ):
            if (
                perm_sampling_manual_only
                and perm_sampling_text_x is not None
                and perm_sampling_text_y is not None
            ):
                legend_x = float(perm_sampling_text_x)
                legend_y = float(perm_sampling_text_y)
                if not (0.0 <= legend_x <= 1.0 and 0.0 <= legend_y <= 1.0):
                    legend_x = 0.5
                    legend_y = 0.25
                    print(
                        "[perm_sampling] legend-only manual text-block moved into view; "
                        "adjust perm_sampling_text_x/perm_sampling_text_y to fine-tune"
                    )
                color = benchmark_plot.STYLE_DICT.get("PermutationSamplingSII", {}).get(
                    "color", "#252525"
                )
                extra_legend_artists = _draw_manual_perm_sampling_text(
                    fig_leg,
                    text=perm_sampling_text,
                    x=legend_x,
                    y=legend_y,
                    fontsize=legend_fontsize,
                    color=color,
                )
                print(
                    "[perm_sampling] legend-only manual text-block drawn "
                    f"x={legend_x} y={legend_y} text={perm_sampling_text}"
                )
            else:
                stretched = _draw_stretched_perm_sampling_sii(
                    fig_leg,
                    legend,
                    legend_fontsize,
                    x_left=perm_sampling_x_left,
                    x_right=perm_sampling_x_right,
                    y=perm_sampling_y,
                    x_left_offset=perm_sampling_x_left_offset,
                    x_right_offset=perm_sampling_x_right_offset,
                    y_offset=perm_sampling_y_offset,
                )
                if stretched is not None:
                    extra_legend_artists = [stretched]
        fig_leg.canvas.draw()
        legend_bbox = legend.get_window_extent(fig_leg.canvas.get_renderer())
        if extra_legend_artists:
            extra_bbox = None
            renderer = fig_leg.canvas.get_renderer()
            for artist in extra_legend_artists:
                bb = artist.get_window_extent(renderer)
                extra_bbox = bb if extra_bbox is None else Bbox.union([extra_bbox, bb])
            legend_bbox = Bbox.union([legend_bbox, extra_bbox])
        legend_bbox_inches = legend_bbox.transformed(fig_leg.dpi_scale_trans.inverted())
        fig_leg.savefig(legend_output_path, bbox_inches=legend_bbox_inches, pad_inches=0)
        plt.close(fig_leg)

    if row_title_on_right:
        for row_idx in range(n_rows):
            spec_idx = row_idx * n_cols
            if spec_idx >= num_plots:
                continue
            game_name = plot_specs[spec_idx]["game_name"]
            row_label = data_names.get(game_name, game_name)
            last_ax_idx = min((row_idx + 1) * n_cols - 1, num_plots - 1)
            last_ax = axes_flat[last_ax_idx]
            last_ax.annotate(
                row_label,
                xy=(1.04, 0.5),
                xycoords="axes fraction",
                fontsize=title_font_size,
                fontweight=W_REGULAR,
                ha="left",
                va="center",
                rotation=90,
                annotation_clip=False,
            )

    if column_headers is not None:
        for col_idx, header in enumerate(column_headers[:n_cols]):
            top_ax = axes_flat[col_idx]
            top_ax.annotate(
                header,
                xy=(0.5, 1.0),
                xycoords="axes fraction",
                xytext=(0, 14),
                textcoords="offset points",
                fontsize=title_font_size + 2,
                fontweight=W_SEMIBOLD,
                ha="center",
                va="bottom",
                annotation_clip=False,
            )

    if column_group_headers is not None:
        # Each entry: (label, start_col_inclusive, end_col_inclusive). Placed
        # above column_headers so a two-tier header is rendered correctly.
        group_y_offset_pts = 32 if column_headers is not None else 14
        fig_height_in = fig.get_size_inches()[1]
        y_offset_fig = (group_y_offset_pts / 72.0) / fig_height_in
        for label, start_col, end_col in column_group_headers:
            start_col = max(0, min(start_col, n_cols - 1))
            end_col = max(0, min(end_col, n_cols - 1))
            if end_col < start_col:
                continue
            left_pos = axes_flat[start_col].get_position()
            right_pos = axes_flat[end_col].get_position()
            x_center = 0.5 * (left_pos.x0 + right_pos.x1)
            y_top = max(left_pos.y1, right_pos.y1)
            fig.text(
                x_center,
                y_top + y_offset_fig,
                label,
                fontsize=title_font_size + 4,
                fontweight=W_SEMIBOLD,
                ha="center",
                va="bottom",
            )

    apply_tick_style(*axes_flat[:num_plots], tick_label_fontsize=tick_label_fontsize)
    fig.savefig(output_path, bbox_inches="tight")
