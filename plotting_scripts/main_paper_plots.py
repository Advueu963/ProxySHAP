"""Combined main-paper plot script.

Edit the `plot_specs` list in the `__main__` block to control which panels are
rendered. Each entry becomes one subplot in the combined figure. The
legend is shared across all panels and placed below the figure.
"""

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

TITLE_FONT_SIZE = 11
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
    "legend_bottom_margin": 0.1,
    "bottom_margin": 0.1,
    "legend_scale": 1.4,
    "use_two_row_legend": False,
    "plots_per_row": 4,
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
        "results_benchmark_{index}_{order}_{game_type}.csv",
        "results_benchmark_{index}_{order}_{game_type}_local.csv",
        "icml_submission_data/results_benchmark_{index}_{order}_{game_type}.csv",
        "results_benchmark_{index}_{order}_{game_type}_sii2_permutation.csv",
        "results_benchmark_{index}_{order}_{game_type}_sii3_permutation.csv",
    ],
    "style_dict": None,
    "panel_grid_style": None,
    "shared_xlabel": "Model Evaluations",
    "shared_ylabel_fontsize": 14,
    "shared_xlabel_fontsize": 14,
    "tick_label_fontsize": 12,
    "corner_label_mode": "order",
    "legend_show_markers": True,
    "legend_ncol_override": 6,
}
DATA_NAMES = {
    "BreastCancerLocalXAI": "Cancer ($n=30$)",
    "CommunitiesAndCrimeLocalXAI": "Crime ($n=101$)",
    "Corrgroups60LocalXAI": "CG60 ($n=60$)",
    "ForestFiresLocalXAI": "Forest ($n=13$)",
    "IndependentLinear60LocalXAI": "IL60 ($n=60$)",
    "NHANESILocalXAI": "NHANES ($n=79$)",
    "RealEstateLocalXAI": "TabPFN on Estate ($n=15$)",
    "wine_quality": "Wine ($n=11$)",
    "AdultCensusLocalXAI": "Adult ($n=14$)",
    "CaliforniaHousingLocalXAI": "Housing ($n=8$)",
    "BikeSharingLocalXAI": "Bike ($n=12$)",
    "ViT4by4Patches": "ViT16 on ImageNet ($n=16$)",
    "ViT3by3Patches": "ViT9 ($n=9$)",
    "ResNet18w14Superpixel": "ResNet18 ($n=14$)",
    "SentimentAnalysisLocalXAI": "DistilBERT on IMDB ($n=14$)",
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
    "TabArenaCoil2000LocalXAI": "LGBM on Coil2000 ($n=85$)",
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
    "TabArenaKddcup09LocalXAI": "LGBM on Kddcup09 ($n=212$)",
    "TabArenaMarketingCampaignLocalXAI": "LGBM on MarketingCamp ($n=25$)",
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
    "TabArenaSpliceLocalXAI": "LGBM on Splice ($n=60$)",
    "TabArenaStudentsDropoutLocalXAI": "StudentsDropout ($n=36$)",
    "TabArenaSuperconductivityLocalXAI": "Superconductivity ($n=81$)",
    "TabArenaTaiwaneseBankruptcyLocalXAI": "TWBankruptcy ($n=94$)",
    "TabArenaBioresponseLocalXAI": "Bioresponse ($n=1776$)",
    "TabArenaWebsitePhishingLocalXAI": "Phishing ($n=9$)",
    "TabArenaSdss17LocalXAI": "Sdss17 ($n=11$)",
    "TabArenaWineQualityLocalXAI": "WineQuality ($n=12$)",
    "TabArenaHrAnalyticsLocalXAI": "HRAnalytics ($n=9$)",
    "Benzene_n25": "Benzene ($n=25$)",
    "Mutagenicity_n25": "Mutagenicity ($n=25$)",
    "Mutagenicity_n30": "Mutagenicity ($n=30$)",
    "Mutagenicity_n35": "GNN on Mutagenicity ($n=35$)",
}
# Make selected model names bold inside DATA_NAMES values (for plot labels).
# We wrap the model token with mathtext bold so only the model name is bold,
# e.g. "LGBM on Coil2000 ($n=85$)" -> "$\\mathbf{LGBM}$ on Coil2000 ($n=85$)".
_BOLD_MODELS = ["LGBM", "GNN", "TabPFN", "ViT16", "ViT9", "ResNet18", "DistilBERT"]
for _k, _v in list(DATA_NAMES.items()):
    new_v = _v
    for _m in _BOLD_MODELS:
        new_v = new_v.replace(f"{_m} on ", f"$\\mathbf{{{_m}}}$ on ")
    DATA_NAMES[_k] = new_v
APPROXIMATORS_TO_PLOT = [
    "KernelSHAPIQ",
    "ProxySPEX (XGBoost)",
    "ProxySHAP (XGBoost, MSR) [our]",
    "ProxySHAP (XGBoost) [our]",
    # "ProxySHAP* (XGBoost, MSR) [our]",
    # "ProxySHAP* (XGBoost) [our]",
    "ProxySHAP (Linear, MSR) [our]",
    "ProxySHAP (Linear) [our]",
    "SHAPIQ",
    "PermutationSamplingSII",
    "SVARMIQ",
    # "ProxySPEX"
]
LINEAR_METHODS = [
    "ProxySHAP (Linear, MSR) [our]",
    "ProxySHAP (Linear) [our]",
]
APPROXIMATOR_RENAMING = {
    "ProxySHAP (XGBoost)": "ProxySHAP (XGBoost) [our]",
    "ProxySHAP (XGBoost, MSR)": "ProxySHAP (XGBoost, MSR) [our]",
    "ProxySHAP (Linear)": "ProxySHAP (Linear) [our]",
    "ProxySHAP (Linear, MSR)": "ProxySHAP (Linear, MSR) [our]",
    "ProxySPEX (XGBoost)": "ProxySPEX (XGBoost)",
    # OLD NAMES #
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
        "shared_ylabel_x": 0.08,
        "use_two_row_legend": True,
        "perm_sampling_manual_only": True,
        "perm_sampling_text_x": 0.8 - 0.002,
        "perm_sampling_text_y": -0.07,
    }

    plot_specs = [
        {
            "game_name": "RealEstateLocalXAI",
            "order": 2,
            "index": "SII",
            "game_type": "tabpfn",
            "ylim": (10 ** (-4.1), 10 ** (0.2)),
        },
        {
            "game_name": "ViT4by4Patches",
            "order": 2,
            "index": "SII",
            "game_type": "exhaustive",
            "ylim": (10 ** (-3.5), 10 ** (0)),
        },
        {
            "game_name": "Mutagenicity_n35",
            "order": 2,
            "index": "SII",
            "game_type": "graph",
            "ylim": (10 ** (-3.2), 10 ** (0.5)),
        },
        {
            "game_name": "TabArenaKddcup09LocalXAI",
            "order": 2,
            "index": "SII",
            "game_type": "interventional",
            "ylim": (10 ** (-4), 10 ** (-0.5)),
        },
        ## Order 3 ##
        {
            "game_name": "SentimentAnalysisLocalXAI",
            "order": 3,
            "index": "SII",
            "game_type": "exhaustive",
            "ylim": (10 ** (-3.2), 10 ** (0.3)),
        },
        {
            "game_name": "TabArenaMarketingCampaignLocalXAI",
            "order": 3,
            "index": "SII",
            "game_type": "interventional",
            "ylim": (10 ** (-4.7), 10 ** (0)),
        },
        {
            "game_name": "TabArenaSpliceLocalXAI",
            "order": 3,
            "index": "SII",
            "game_type": "interventional",
            "ylim": (10 ** (-2.5), 10 ** (0)),
        },
        {
            "game_name": "TabArenaCoil2000LocalXAI",
            "order": 3,
            "index": "SII",
            "game_type": "interventional",
            "ylim": (10 ** (-2), 10 ** (0)),
        },
    ]

    plot_combined_main_paper_plots(
        plot_specs=plot_specs,
        approximators_to_plot=APPROXIMATORS_TO_PLOT,
        approximator_renaming=APPROXIMATOR_RENAMING,
        linear_methods=LINEAR_METHODS,
        data_names=DATA_NAMES,
        title_font_size=TITLE_FONT_SIZE,
        output_path="plots/main/main_paper_plots_combined.pdf",
        **general_params,
    )
