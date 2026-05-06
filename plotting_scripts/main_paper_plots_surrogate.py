"""Combined main-paper surrogate plot script."""

from __future__ import annotations

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

TITLE_FONT_SIZE = 18
# Quick tuning guide for combined plots:
# - Figure/grid: panel_figsize, figsize
# - Whitespace: bottom_margin, legend_bottom_margin, subplot_wspace, subplot_hspace
# - Legend: legend_scale
# - Captions: show_captions_only_first_row
# - Axes: x_ticks_only_last_row
# - Shared labels: shared_ylabel_x, shared_xlabel_y, shared_ylabel_fontsize, shared_xlabel_fontsize
# - Axes/data: y_log_scale, x_log_scale, min_budget, max_budget
# - Style: style_dict, panel_grid_style, shared_xlabel
BASE_GENERAL_PARAMS = {
    "panel_figsize": (4, 3),
    "figsize": None,
    "legend_bottom_margin": None,
    "bottom_margin": None,
    "legend_scale": 1.0,
    "plots_per_row": 4,
    "show_captions_only_first_row": True,
    "x_ticks_only_last_row": True,
    "subplot_wspace": 0.2,
    "subplot_hspace": 0.01,
    "shared_ylabel_x": 0.08,
    "shared_xlabel_y": 0.01,
    "y_log_scale": True,
    "x_log_scale": True,
    "ylim": (1e-7, 1e2),
    "marker_size": 4,
    "linewidth": 2,
    "highlight_size": 2,
    "min_budget": 14,
    "max_budget": 35_000,
    "data_path_template": "icml_submission_data/results_benchmark_{index}_{order}_{game_type}.csv",
    "style_dict": None,
    "panel_grid_style": None,
    "shared_xlabel": "Model Evaluations",
    "shared_ylabel_fontsize": 18,
    "shared_xlabel_fontsize": 18,
    "corner_label_mode": "order",
    "legend_ncol_override": 2,
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
}
APPROXIMATORS_TO_PLOT = [
    "KernelSHAPIQ",
    "ProxySpex",
    "ProxySHAP (XGBoost, MSR) [our]",
    "ProxySHAP (XGBoost) [our]",
]
SURROGATE_APPROXIMATORS_TO_PLOT = [
    "ProxySHAP (Linear, MSR) [our]",
    "ProxySHAP (Linear) [our]",
    "ProxySHAP (XGBoost, MSR) [our]",
    "ProxySHAP (XGBoost) [our]",
]
LINEAR_METHODS = [
    "ProxySHAP (Linear, MSR) [our]",
    "ProxySHAP (Linear) [our]",
]
APPROXIMATOR_RENAMING = {
    "RegressionMSRIQ-NoAdjustment": "ProxySHAP (XGBoost) [our]",
    "RegressionMSRIQ": "ProxySHAP (XGBoost, MSR) [our]",
    "Linear-NoAdjustment": "ProxySHAP (Linear) [our]",
    "Linear-RECAP": "ProxySHAP (Linear, MSR) [our]",
    "RegressionMSRIQ-XGB-PreDef-NoAdjustment": "ProxySHAP+ (XGBoost) [our]",
    "RegressionMSRIQ-XGB-PreDef": "ProxySHAP+ (XGBoost, MSR) [our]",
    "RegressionMSRIQ-XGB-CV-BO": "ProxySHAP* (XGBoost, MSR) [our]",
    "RegressionMSRIQ-XGB-CV-BO-NoAdjustment": "ProxySHAP* (XGBoost) [our]",
}



if __name__ == "__main__":
    general_params = {
        **BASE_GENERAL_PARAMS,
        "panel_figsize": (3.5, 3.5),
        "bottom_margin": 0.09,
        "plots_per_row": 4,
    }
    plot_specs = [
        ### ORDER 2 ###
        {
            "game_name": "ResNet18w14Superpixel",
            "order": 2,
            "index": "SII",
            "game_type": "exhaustive",
            "ylim": (10 ** (-8), 10 ** (-2.3)),
        },
        {
            "game_name": "ViT4by4Patches",
            "order": 2,
            "index": "SII",
            "game_type": "exhaustive",
            "ylim": (10 ** (-8), 10 ** (-2.3)),
            "max_budget": 70_000,
        },
        {
            "game_name": "IndependentLinear60LocalXAI",
            "order": 2,
            "index": "SII",
            "game_type": "interventional",
            "ylim": (1e-6, 10 ** (-2.5)),
        },
        {
            "game_name": "CommunitiesAndCrimeLocalXAI",
            "order": 2,
            "index": "SII",
            "game_type": "interventional",
            "ylim": (10**(-0.5), 10**(1.8)),
        },
        ### ORDER 3 ###
        {
            "game_name": "ResNet18w14Superpixel",
            "order": 2,
            "index": "SII",
            "game_type": "exhaustive",
            "ylim": (10 ** (-6.7), 10 ** (-2.3)),
        },
        {
            "game_name": "ViT4by4Patches",
            "order": 3,
            "index": "SII",
            "game_type": "exhaustive",
            "ylim": (10 ** (-8), 10 ** (-2.7)),
            "max_budget": 70_000,
        },
        {
            "game_name": "IndependentLinear60LocalXAI",
            "order": 3,
            "index": "SII",
            "game_type": "interventional",
            "ylim": (10**(-5.5), 10 ** (-3.5)),
        },
        {
            "game_name": "CommunitiesAndCrimeLocalXAI",
            "order": 3,
            "index": "SII",
            "game_type": "interventional",
            "ylim": (10**(-1), 10**(2.01)),
        },
        
    ]

    plot_combined_main_paper_plots(
        plot_specs=plot_specs,
        approximators_to_plot=SURROGATE_APPROXIMATORS_TO_PLOT,
        approximator_renaming=APPROXIMATOR_RENAMING,
        linear_methods=LINEAR_METHODS,
        data_names=DATA_NAMES,
        title_font_size=TITLE_FONT_SIZE,
        output_path="plots/main/main_paper_plots_surrogate_combined.pdf",
        legend_output_path="plots/main/main_paper_plots_surrogate_combined_legend.pdf",
        **general_params,
    )
