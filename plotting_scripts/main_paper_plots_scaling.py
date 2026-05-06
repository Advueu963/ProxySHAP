"""Combined main-paper plot script.

Edit the `plot_specs` list in the `__main__` block to control which panels are
rendered. Each entry becomes one subplot in the combined figure. The
legend is shared across all panels and placed below the figure.
"""

from __future__ import annotations

import argparse

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
LEGEND_FONT_SIZE = 12
# Per-evaluation model time used when the bottom row plots runtime. Mirrors the
# constant of the same name in `main_paper_plots_new_runtime.py` so both
# scripts compute the runtime axis identically.
TIME = 0
# Quick tuning guide for combined plots:
# - Figure/grid: panel_figsize, figsize
# - Whitespace: bottom_margin, legend_bottom_margin, subplot_wspace, subplot_hspace
# - Legend: legend_scale
# - Optional 2-row legend layout: use_two_row_legend
# - Captions: show_captions_only_first_row
# - Axes: x_ticks_only_last_row
# - Shared labels: shared_ylabel_x, shared_xlabel_y, shared_ylabel_fontsize, shared_xlabel_fontsize
# - Axes/data: y_log_scale, x_log_scale, min_budget, max_budget
# - Style: style_dict, panel_grid_style, shared_xlabel
BASE_GENERAL_PARAMS = {
    "panel_figsize": (3, 2),
    "figsize": None,
    "legend_bottom_margin": 0.16,
    "bottom_margin": 0.1,
    "legend_scale": 1.4,
    "use_two_row_legend": True,
    "plots_per_row": 2,
    "show_captions_only_first_row": False,
    "x_ticks_only_last_row": False,
    "subplot_wspace": 0.15,
    "subplot_hspace": 0.45,
    "shared_ylabel_x": None,
    "shared_xlabel_y": 0,
    "legend_ncol_override": 2,
    "y_log_scale": True,
    "x_log_scale": True,
    "ylim": (1e-7, 1e2),
    "marker_size": 3,
    "linewidth": 1.5,
    "highlight_size": 2,
    "tick_label_fontsize": 12,
    "max_budget": 35_000,
    "data_path_template": [
        # `_big_games` first so its (game, approximator, budget) rows take
        # precedence over the older runtime measurements in `_local`/main.
        "results_benchmark_{index}_{order}_{game_type}_big_games.csv",
        "results_benchmark_{index}_{order}_{game_type}_local.csv",
        "results_benchmark_{index}_{order}_{game_type}_large.csv",
        "results_benchmark_{index}_{order}_{game_type}_local_big_data.csv",
        # "results_benchmark_{index}_{order}_{game_type}.csv",
        "icml_submission_data/results_benchmark_{index}_{order}_{game_type}.csv",
        "icml_submission_data/results_benchmark_{index}_{order}_{game_type}_test.csv",
    ],
    "style_dict": None,
    "panel_grid_style": None,
    "shared_xlabel": "Model Evaluations",
    "shared_ylabel_fontsize": 14,
    "shared_xlabel_fontsize": 14,
    "corner_label_mode": "order",
    "legend_show_markers": True,
    "order_label_in_corner": False,
    "legend_order" : [
        "ProxySHAP (XGBoost+Default) [our]",
        "ProxySHAP (XGBoost+HPO-Informed) [our]",
        "ProxySHAP (XGBoost+HPO) [our]",
        #"ProxySPEX (XGBoost+Default)",
        "ProxySPEX (XGBoost+HPO-Informed)",
    ]
}
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
    "LeukemiaLocalXAI": "Leukemia ($n=1776$)",
    "BioresponseLocalXAI": "Bioresponse ($n=1776$)",
    "NurseryLocalXAI": "Nursery ($n=8$)",
    "ZooLocalXAI": "Zoo ($n=16$)",
    "ThyroidLocalXAI": "Thyroid ($n=21$)",
    "HepatitisLocalXAI": "Hepatitis ($n=19$)",
    "IonosphereLocalXAI": "Ionosphere ($n=33$)",
    "MushroomLocalXAI": "Mushroom ($n=22$)",
    "SoybeanLocalXAI": "Soybean ($n=35$)",
    # TabArena datasets
    "TabArenaAirfoilSelfNoiseLocalXAI": "Airfoil ($n=5$)",
    "TabArenaAirlineSatisfactionLocalXAI": "Airline ($n=21$)",
    "TabArenaAmazonEmployeeAccessLocalXAI": "Amazon ($n=9$)",
    "TabArenaAnnealLocalXAI": "Anneal ($n=38$)",
    "TabArenaApsFailureLocalXAI": "ApsFailure ($n=170$)",
    "TabArenaBankCustomerChurnLocalXAI": "BankChurn ($n=10$)",
    "TabArenaBankMarketingLocalXAI": "BankMkt ($n=13$)",
    "TabArenaBankruptcyLocalXAI": "Bankruptcy ($n=64$)",
    "TabArenaBloodTransfusionLocalXAI": "Blood ($n=4$)",
    "TabArenaChurnLocalXAI": "Churn ($n=19$)",
    "TabArenaCoil2000LocalXAI": "Coil2000 ($n=85$)",
    "TabArenaConcreteStrengthLocalXAI": "Concrete ($n=8$)",
    "TabArenaCouponRecommendationLocalXAI": "CouponRec ($n=24$)",
    "TabArenaCreditCardDefaultLocalXAI": "CreditCard ($n=23$)",
    "TabArenaCreditGLocalXAI": "CreditG ($n=20$)",
    "TabArenaDiabetes130usLocalXAI": "Diabetes130 ($n=47$)",
    "TabArenaDiabetesLocalXAI": "Diabetes ($n=8$)",
    "TabArenaDiamondsLocalXAI": "Diamonds ($n=9$)",
    "TabArenaEcommerceShippingLocalXAI": "Ecommerce ($n=10$)",
    "TabArenaFiat500LocalXAI": "Fiat500 ($n=7$)",
    "TabArenaFitnessClubLocalXAI": "Fitness ($n=6$)",
    "TabArenaFoodDeliveryLocalXAI": "FoodDel ($n=9$)",
    "TabArenaGiveMeCreditLocalXAI": "GiveCredit ($n=10$)",
    "TabArenaGoodCustomerLocalXAI": "GoodCustomer ($n=13$)",
    "TabArenaHazelnutLocalXAI": "Hazelnut ($n=30$)",
    "TabArenaHealthInsuranceLocalXAI": "HealthIns ($n=6$)",
    "TabArenaHelocLocalXAI": "Heloc ($n=23$)",
    "TabArenaHousesLocalXAI": "Houses ($n=8$)",
    "TabArenaJm1LocalXAI": "Jm1 ($n=21$)",
    "TabArenaKddcup09LocalXAI": "Kddcup09 ($n=212$)",
    "TabArenaMarketingCampaignLocalXAI": "MarketingCamp ($n=25$)",
    "TabArenaMaternalHealthLocalXAI": "MaternalHealth ($n=6$)",
    "TabArenaMiamiHousingLocalXAI": "MiamiHousing ($n=15$)",
    "TabArenaMicLocalXAI": "MIC ($n=111$)",
    "TabArenaNaticusdroidLocalXAI": "NaticusDroid ($n=86$)",
    "TabArenaOnlineShoppersLocalXAI": "OnlineShoppers ($n=17$)",
    "TabArenaProteinLocalXAI": "Protein ($n=9$)",
    "TabArenaQsarBiodegLocalXAI": "QsarBiodeg ($n=41$)",
    "TabArenaQsarFishToxicityLocalXAI": "QsarFishTox ($n=6$)",
    "TabArenaQsarTid11LocalXAI": "QsarTid11 ($n=1024$)",
    "TabArenaSeismicBumpsLocalXAI": "SeismicBumps ($n=15$)",
    "TabArenaHivaAgnosticLocalXAI": "HivaAgnostic ($n=1617$)",
    "TabArenaSpliceLocalXAI": "Splice ($n=60$)",
    "TabArenaStudentsDropoutLocalXAI": "StudentsDropout ($n=36$)",
    "TabArenaSuperconductivityLocalXAI": "Superconductivity ($n=81$)",
    "TabArenaTaiwaneseBankruptcyLocalXAI": "TWBankruptcy ($n=94$)",
    "TabArenaBioresponseLocalXAI": "Bioresponse ($n=1776$)",
    "TabArenaWebsitePhishingLocalXAI": "Phishing ($n=9$)",
    "TabArenaSdss17LocalXAI": "Sdss17 ($n=11$)",
    "TabArenaWineQualityLocalXAI": "WineQuality ($n=12$)",
    "TabArenaHrAnalyticsLocalXAI": "HRAnalytics ($n=9$)",
}
APPROXIMATORS_TO_PLOT = [
    # "KernelSHAPIQ",
    "ProxySPEX (XGBoost+Default)",
    "ProxySHAP (XGBoost+Default) [our]",
    "ProxySHAP (XGBoost+HPO-Informed) [our]",
    "ProxySPEX (XGBoost+HPO-Informed)",
    "ProxySHAP (XGBoost+HPO) [our]",
]
LINEAR_METHODS = [
    "ProxySHAP (Linear, MSR) [our]",
    "ProxySHAP (Linear) [our]",
]
APPROXIMATOR_RENAMING = {
    "ProxySHAP (XGBoost)": "ProxySHAP (XGBoost+Default) [our]",
    "ProxySHAP (XGBoost, MSR)": "ProxySHAP (XGBoost, MSR) [our]",
    "ProxySHAP+ (XGBoost)": "ProxySHAP (XGBoost+HPO-Informed) [our]",
    "ProxySHAP* (XGBoost)": "ProxySHAP (XGBoost+HPO) [our]",
    "ProxySPEX (XGBoost)": "ProxySPEX (XGBoost+Default)",
    "RegressionMSRIQ-NoAdjustment": "ProxySHAP (XGBoost+Default) [our]",
    "ProxySPEX+ (XGBoost)": "ProxySPEX (XGBoost+HPO-Informed)",
    # "RegressionMSRIQ": "ProxySHAP (XGBoost, MSR) [our]",
    # "Linear-NoAdjustment": "ProxySHAP (Linear) [our]",
    # "Linear-RECAP": "ProxySHAP (Linear, MSR) [our]",
    "RegressionMSRIQ-XGB-PreDef-NoAdjustment": "ProxySHAP (XGBoost+HPO-Informed) [our]",
    # "RegressionMSRIQ-XGB-PreDef": "ProxySHAP+ (XGBoost, MSR) [our]",
    # "RegressionMSRIQ-XGB-CV-BO": "ProxySHAP* (XGBoost, MSR) [our]",
    "RegressionMSRIQ-XGB-CV-BO-NoAdjustment": "ProxySHAP (XGBoost+HPO) [our]",
    # half-renamed names in the newer root-level CSV format
    "ProxySHAP (Linear, MSR)": "ProxySHAP (Linear, MSR) [our]",
    "ProxySHAP (Linear)": "ProxySHAP (Linear) [our]",
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot scaling panels for the main paper.")
    parser.add_argument(
        "--bottom-row-x-axis",
        choices=["budget", "runtime"],
        default="budget",
        help="Use model evaluations (budget) or total runtime on the bottom row x-axis.",
    )
    args = parser.parse_args()

    bottom_row_x_axis = None if args.bottom_row_x_axis == "budget" else args.bottom_row_x_axis
    general_params = {
        **BASE_GENERAL_PARAMS,
        "legend_scale": LEGEND_FONT_SIZE / 9,
        "bottom_row_x_axis": bottom_row_x_axis,
        # Match `main_paper_plots_new_runtime.py`: no smoothing/binning, just
        # `runtime_eval_time` to add per-call model time onto `total_runtime`.
        "runtime_eval_time": TIME if bottom_row_x_axis == "runtime" else None,
        "show_row_order_labels": False,
    }
    if bottom_row_x_axis == "runtime":
        if TIME == 0.001:
            shared_xlabel = "Total Runtime (s; 1ms per Model call)"
        elif TIME == 0.01:
            shared_xlabel = "Total Runtime (s; 10ms per Model call)"
        elif TIME == 0.1:
            shared_xlabel = "Total Runtime (s; 100ms per Model call)"
        elif TIME == 1.0:
            shared_xlabel = "Total Runtime (s; 1s per Model call)"
        elif TIME == 0:
            shared_xlabel = "Total Runtime (s; without Model call time)"
        else:
            shared_xlabel = "Total Runtime (s)"
        general_params["shared_xlabel"] = shared_xlabel
    plot_specs = [
        {
            "game_name": "Corrgroups60LocalXAI",
            "order": 2,
            "index": "SII",
            "game_type": "interventional",
            "ylim": (10 ** (-3), 10 ** (0.5)),
        },
        {
            "game_name": "CommunitiesAndCrimeLocalXAI",
            "order": 2,
            "index": "SII",
            "game_type": "interventional",
            "ylim": (10 ** (-3), 10 ** (0)),
        },
        {
            "game_name": "TabArenaQsarTid11LocalXAI",
            "order": 2,
            "index": "SII",
            "game_type": "interventional",
            "ylim": (10 ** (-3.2), 10 ** (0)),
             "xlim_runtime": (1, 90),   # or "xlim" if you want it for both modes
        },
        {
            "game_name": "TabArenaBioresponseLocalXAI",
            "order": 2,
            "index": "SII",
            "game_type": "interventional",
            "ylim": (10 ** (-1.2), 10 ** (0)),
             "xlim_runtime": (2.5, 100),   # or "xlim" if you want it for both modes
        },
    ]

    plot_combined_main_paper_plots(
        plot_specs=plot_specs,
        approximators_to_plot=APPROXIMATORS_TO_PLOT,
        approximator_renaming=APPROXIMATOR_RENAMING,
        linear_methods=LINEAR_METHODS,
        data_names=DATA_NAMES,
        title_font_size=TITLE_FONT_SIZE,
        output_path="plots/main/main_paper_plots_scaling.pdf",
        **general_params,
    )
