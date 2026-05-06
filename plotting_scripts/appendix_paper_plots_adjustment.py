"""Combined main-paper rebuttal plot script."""

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
BASE_GENERAL_PARAMS = {
    "panel_figsize": (4, 3),
    "legend_scale": 1.0,
    "use_two_row_legend": True,
    "plots_per_row": 4,
    "show_captions_only_first_row": True,
    "x_ticks_only_last_row": True,
    "y_log_scale": True,
    "x_log_scale": True,
    "marker_size": 4,
    "linewidth": 2,
    "highlight_size": 2,
    "min_budget": 14,
    "max_budget": 35_000,
    "shared_ylabel_fontsize": 14,
    "shared_xlabel_fontsize": 14,
    "subplot_wspace": 0.18,
    "subplot_hspace": 0.01,
    "shared_ylabel_x": 0.08,
    "shared_xlabel_y": 0.01,
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
    "ProxySPEXXGBoost": "ProxySPEX (XGBoost, default)",
    "ProxySPEXXGBoostNoTruncationNoRefinement": "ProxySPEX (XGBoost, no trunc., no refin.)",
}


if __name__ == "__main__":
    plot_specs = [
        {
            "game_name": "ResNet18w14Superpixel",
            "order": 2,
            "index": "SII",
            "game_type": "exhaustive",
            "ylim": (10 ** (-8), 10 ** (-2.3)),
        },
        {
            "game_name": "SentimentAnalysisLocalXAI",
            "order": 2,
            "index": "SII",
            "game_type": "exhaustive",
            "ylim": (10 ** (-8), 10 ** (-1.7)),
        },
        {
            "game_name": "RealEstateLocalXAI",
            "order": 2,
            "index": "SII",
            "game_type": "interventional",
            "ylim": (10 ** (-8), 10 ** (-1.6)),
        },
        {
            "game_name": "ViT4by4Patches",
            "order": 2,
            "index": "SII",
            "game_type": "exhaustive",
            "ylim": (10 ** (-8), 10 ** (-2.4)),
            "max_budget": 70_000,
        },
        # {
        #     "game_name": "BreastCancerLocalXAI",
        #     "order": 2,
        #     "index": "SII",
        #     "game_type": "interventional",
        #     "ylim": (10 ** (-5.6), 10 ** (-1.3)),
        # },
        {
            "game_name": "ResNet18w14Superpixel",
            "order": 3,
            "index": "SII",
            "game_type": "exhaustive",
            "ylim": (10 ** (-8), 10 ** (-2.7)),
        },
        {
            "game_name": "SentimentAnalysisLocalXAI",
            "order": 3,
            "index": "SII",
            "game_type": "exhaustive",
            "ylim": (10 ** (-8), 10 ** (-1.7)),
        },
        {
            "game_name": "RealEstateLocalXAI",
            "order": 3,
            "index": "SII",
            "game_type": "interventional",
            "ylim": (10 ** (-8), 10 ** (-1.6)),
        },
        {
            "game_name": "ViT4by4Patches",
            "order": 3,
            "index": "SII",
            "game_type": "exhaustive",
            "ylim": (10 ** (-8), 10 ** (-2.8)),
            "max_budget": 70_000,
        },
        # {
        #     "game_name": "BreastCancerLocalXAI",
        #     "order": 3,
        #     "index": "SII",
        #     "game_type": "interventional",
        #     "ylim": (10 ** (-4.8), 10 ** (-2.3)),
        # },
    ]

    plot_combined_main_paper_plots(
        plot_specs=plot_specs,
        approximators_to_plot=[
            "ProxySHAP (XGBoost, MSR) [our]",
            "ProxySHAP (XGBoost) [our]",
            "ProxySHAP* (XGBoost, MSR) [our]",
            "ProxySHAP* (XGBoost) [our]",
            "ProxySHAP (Linear, MSR) [our]",
            "ProxySHAP (Linear) [our]",
        ],
        approximator_renaming=APPROXIMATOR_RENAMING,
        linear_methods=LINEAR_METHODS,
        data_names=DATA_NAMES,
        title_font_size=TITLE_FONT_SIZE,
        output_path="plots/appendix/appendix_paper_plots_adjustment.pdf",
        legend_output_path="plots/appendix/appendix_paper_plots_adjustment_legend.pdf",
        **BASE_GENERAL_PARAMS,
    )
