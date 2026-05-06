"""Combined runtime plots for the main-paper scaling setting.

Mirrors `main_paper_plots_new_runtime.py`: uses the shared
`plot_combined_main_paper_plots` harness so the panels look and behave
identically. The differences live in `plot_specs`, the approximator set, and
the per-script time-step sweep.
"""

from __future__ import annotations

import os

try:
    from special_plot_scripts._combined_main_paper_plotting import (
        plot_combined_main_paper_plots,
    )
except ModuleNotFoundError:
    from _combined_main_paper_plotting import plot_combined_main_paper_plots

try:
    from special_plot_scripts._plot_style import setup_fonts
except ModuleNotFoundError:
    from _plot_style import setup_fonts

setup_fonts()

TITLE_FONT_SIZE = 11

BASE_GENERAL_PARAMS = {
    "panel_figsize": (3, 2),
    "figsize": None,
    "legend_bottom_margin": 0.1,
    "bottom_margin": 0.1,
    "legend_scale": 1.4,
    "use_two_row_legend": False,
    "plots_per_row": 3,
    "show_captions_only_first_row": False,
    "x_ticks_only_last_row": False,
    "subplot_wspace": 0.15,
    "subplot_hspace": 0.28,
    "shared_ylabel_x": None,
    "shared_xlabel_y": 0,
    "y_log_scale": True,
    "x_log_scale": True,
    "ylim": (1e-7, 1e2),
    "marker_size": 3,
    "linewidth": 1.5,
    "highlight_size": 2,
    "min_budget": 14,
    "max_budget": 35_000,
    "data_path_template": [
        "results_benchmark_{index}_{order}_{game_type}_big_games.csv",
        "results_benchmark_{index}_{order}_{game_type}_local.csv",
        "results_benchmark_{index}_{order}_{game_type}.csv",
        "icml_submission_data/results_benchmark_{index}_{order}_{game_type}.csv",
        "icml_submission_data/results_benchmark_{index}_{order}_{game_type}_with_hpo.csv",
        "icml_submission_data/results_benchmark_{index}_{order}_{game_type}_time.csv",
        "icml_submission_data/results_benchmark_{index}_{order}_{game_type}_time_bioresponse.csv",
        
    ],
    "style_dict": None,
    "panel_grid_style": None,
    "shared_ylabel_fontsize": 14,
    "shared_xlabel_fontsize": 14,
    "tick_label_fontsize": 12,
    "corner_label_mode": "order",
    "legend_show_markers": True,
    "legend_ncol_override": 4,
}

DATA_NAMES = {
    "TabArenaBioresponseLocalXAI": "Bioresponse ($d=1776$)",
    "TabArenaHivaAgnosticLocalXAI": "HIVAgnostic ($d=1617$)",
    "TabArenaQsarTid11LocalXAI": "Qsar ($d=1024$)",
    "NHANESILocalXAI": "NHANESI ($d=79$)",
    "CommunitiesAndCrimeLocalXAI": "Crime ($d=101$)",
}

APPROXIMATORS_TO_PLOT = [
    "ProxySPEX",
    "ProxySPEX (XGBoost)",
    "ProxySHAP (XGBoost) [our]",
]

LINEAR_METHODS = [
    "ProxySHAP (Linear, MSR) [our]",
    "ProxySHAP (Linear) [our]",
]

APPROXIMATOR_RENAMING = {
    "RegressionMSRIQ-NoAdjustment": "ProxySHAP (XGBoost) [our]",
    "RegressionMSRIQ-XGB-PreDef-NoAdjustment": "ProxySHAP+ (XGBoost) [our]",
    "ProxySPEXXGBoost": "ProxySPEX (XGBoost)",
    "ProxySPEXXGBoostNoTruncationNoRefinement": "ProxySPEX (XGBoost)",
    "ProxySHAP (XGBoost)": "ProxySHAP (XGBoost) [our]",
    "ProxySHAP+ (XGBoost)": "ProxySHAP+ (XGBoost) [our]",
}


def _runtime_specs() -> list[dict]:
    return [
        {
            "game_name": "TabArenaQsarTid11LocalXAI",
            "order": 2,
            "index": "SII",
            "game_type": "interventional",
            "ylim": (10 ** (-7.5), 10 ** (-4)),
            "x_axis_mode": "runtime",
        },
        {
            "game_name": "TabArenaBioresponseLocalXAI",
            "order": 2,
            "index": "SII",
            "game_type": "interventional",
            "ylim": (10 ** (-7), 10 ** (-5.9)),
            "x_axis_mode": "runtime",
        },
    ]


if __name__ == "__main__":
    runtime_output_dir = "plots/main/runtime"
    os.makedirs(runtime_output_dir, exist_ok=True)

    for time_step in (0.001, 0.01):
        general_params = {
            **BASE_GENERAL_PARAMS,
            "shared_ylabel_x": 0.08,
            "runtime_eval_time": time_step,
        }

        if time_step == 0.001:
            xlabel = "Total Runtime (s; 1ms per Model call)"
        elif time_step == 0.01:
            xlabel = "Total Runtime (s; 10ms per Model call)"
        elif time_step == 0.1:
            xlabel = "Total Runtime (s; 100ms per Model call)"
        elif time_step == 1.0:
            xlabel = "Total Runtime (s; 1s per Model call)"
        elif time_step == 0:
            xlabel = "Total Runtime (s; without Model call time)"
        else:
            xlabel = "Total Runtime (s)"

        plot_combined_main_paper_plots(
            plot_specs=_runtime_specs(),
            approximators_to_plot=APPROXIMATORS_TO_PLOT,
            approximator_renaming=APPROXIMATOR_RENAMING,
            linear_methods=LINEAR_METHODS,
            data_names=DATA_NAMES,
            title_font_size=TITLE_FONT_SIZE,
            shared_xlabel=xlabel,
            output_path=f"{runtime_output_dir}/runtime_combined_{time_step}.pdf",
            legend_output_path=f"{runtime_output_dir}/runtime_combined_legend_{time_step}.pdf",
            **general_params,
        )
