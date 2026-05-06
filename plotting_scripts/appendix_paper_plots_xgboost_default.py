"""Combined appendix plots for the xgboost-default comparison battery."""

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

TITLE_FONT_SIZE = 18
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
    "BioresponseLocalXAI": "Bioresponse ($d=1776$)",
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
    # "SHAPIQ",
    # "SVARMIQ",
    # "PermutationSamplingSV",
    # "PermutationSamplingSII",
    #"KernelSHAPIQ",
    # "ProxySpex (XGBoost)",
    # Our Methods [LINEAR]
    # "ProxySHAP (Linear) [our]",
    # "ProxySHAP (Linear, MSR) [our]",
    # Our Methods [XGBOOST]
    "ProxySHAP (XGBoost) [our]",
    "ProxySHAP (XGBoost, MSR) [our]",
    # Our Methods [XGBOOST]
    "ProxySHAP (XGBoost+HPO-Informed) [our]",
    "ProxySHAP (XGBoost+HPO-Informed, MSR) [our]",
    # OUR HPO METHODS
    "ProxySHAP (XGBoost+HPO) [our]",
    "ProxySHAP (XGBoost+HPO, MSR) [our]",
]
LINEAR_METHODS = [
    "ProxySHAP (Linear, MSR) [our]",
    "ProxySHAP (Linear) [our]",
]
# Order in which legend entries should appear. Labels listed here come first
# in the given sequence; any remaining labels are appended in their original
# order. Set to None (or []) to keep the auto-generated order.
LEGEND_ORDER = [
    "ProxySHAP (XGBoost) [our]",
    "ProxySHAP (XGBoost, MSR) [our]",
    "ProxySHAP (XGBoost+HPO-Informed) [our]",
    "ProxySHAP (XGBoost+HPO-Informed, MSR) [our]",
    "ProxySHAP (XGBoost+HPO) [our]",
    "ProxySHAP (XGBoost+HPO, MSR) [our]",
]
APPROXIMATOR_RENAMING = {
    # "PermutationSamplingSV": "Permutation Sampling (SV)",
    # "PermutationSamplingSII": "Permutation Sampling (SII)",
    "RegressionMSRIQ": "ProxySHAP (XGBoost, MSR) [our]",
    "RegressionMSRIQ-NoAdjustment": "ProxySHAP (XGBoost) [our]",
    ## LINEAR METHODS ##
    "Linear-NoAdjustment": "ProxySHAP (Linear) [our]",
    "Linear-RECAP": "ProxySHAP (Linear, MSR) [our]",
    ## DEFAULT METHODS ##
    "RegressionMSRIQ-XGB-PreDef-NoAdjustment": "ProxySHAP (XGBoost+HPO-Informed) [our]",
    "RegressionMSRIQ-XGB-PreDef": "ProxySHAP (XGBoost+HPO-Informed, MSR) [our]",
    ## HPO METHODS ##
    "RegressionMSRIQ-XGB-CV-BO": "ProxySHAP (XGBoost+HPO, MSR) [our]",
    "RegressionMSRIQ-XGB-CV-BO-NoAdjustment": "ProxySHAP (XGBoost+HPO) [our]",
    ## TABARENA GAMES (already use friendly names, missing [our] suffix) ##
    "ProxySHAP (XGBoost)": "ProxySHAP (XGBoost) [our]",
    "ProxySHAP (XGBoost, MSR)": "ProxySHAP (XGBoost, MSR) [our]",
    "ProxySHAP+ (XGBoost)": "ProxySHAP (XGBoost+HPO-Informed) [our]",
    "ProxySHAP+ (XGBoost, MSR)": "ProxySHAP (XGBoost+HPO-Informed, MSR) [our]",
    "ProxySHAP* (XGBoost)": "ProxySHAP (XGBoost+HPO) [our]",
    "ProxySHAP* (XGBoost, MSR)": "ProxySHAP (XGBoost+HPO, MSR) [our]",
}
GAME_TYPE_ABBREVIATIONS = {
    "exhaustive": "EXH",
    "interventional": "INT",
    "exhaustive_tabpfn": "TABPFN",
    "pathdependent": "PD",
}
ITERATIONS = {
    # "SV": 1,
    "SII@2": 2,
    # "SII@3": 3,
    # "BV": 1,
    # "BII@2": 2,
    # "BII@3": 3,
}


if __name__ == "__main__":
    general_params = {}
    for INDEX, ORDER in ITERATIONS.items():
        if INDEX.endswith(f"@{ORDER}"):
            INDEX = INDEX.split("@")[0]
        print(f"Processing index {INDEX} with order {ORDER}...")

        benchmark_index = INDEX
        output_dir = (
            "plots/appendix/xgboost/"
            f"{'SII' if benchmark_index == 'SV' else 'BII' if benchmark_index == 'BV' else benchmark_index}/"
            f"order{ORDER}/"
        )
        os.makedirs(output_dir, exist_ok=True)

        plot_specs = [
            {
                "game_name": "BreastCancerLocalXAI",
                "order": ORDER,
                "index": benchmark_index,
                "game_type": "interventional",
                "ylim": (10 ** (-5), 10 ** (0)),
            },
            {
                "game_name": "Corrgroups60LocalXAI",
                "order": ORDER,
                "index": benchmark_index,
                "game_type": "interventional",
                "ylim": (10 ** (-3.5), 10 ** (1.3)),
            },
            {
                "game_name": "IndependentLinear60LocalXAI",
                "order": ORDER,
                "index": benchmark_index,
                "game_type": "interventional",
                "ylim": (10 ** (-3.5), 10 ** (0)),
            },
            {
                "game_name": "NHANESILocalXAI",
                "order": ORDER,
                "index": benchmark_index,
                "game_type": "interventional",
                "ylim": (10 ** (-3.5), 10 ** (0)),
            },
            {
                "game_name": "CommunitiesAndCrimeLocalXAI",
                "order": ORDER,
                "index": benchmark_index,
                "game_type": "interventional",
                "ylim": (10 ** (-3), 10 ** (0)),
            },
            {
                "game_name": "TabArenaQsarTid11LocalXAI",
                "order": 2,
                "index": "SII",
                "game_type": "interventional",
                "ylim": (10 ** (-3.3), 10 ** (-2.3)),
            },
            {
                "game_name": "TabArenaHivaAgnosticLocalXAI",
                "order": 2,
                "index": "SII",
                "game_type": "interventional",
                "ylim": (10 ** (-2), 10 ** (-0.9)),
            },
            {
                "game_name": "TabArenaBioresponseLocalXAI",
                "order": 2,
                "index": "SII",
                "game_type": "interventional",
                "ylim": (10 ** (-1.15), 10 ** (-0.45)),
            },
        ]
        plot_combined_main_paper_plots(
            plot_specs=plot_specs,
            approximators_to_plot=APPROXIMATORS_TO_PLOT,
            approximator_renaming=APPROXIMATOR_RENAMING,
            linear_methods=LINEAR_METHODS,
            data_names=DATA_NAMES,
            title_font_size=TITLE_FONT_SIZE,
            panel_figsize=(4.0, 3.0),
            plots_per_row=4,
            show_captions_only_first_row=False,
            x_ticks_only_last_row=False,
            subplot_wspace=0.18,
            subplot_hspace=0.25,
            shared_ylabel_x=0.07,
            shared_xlabel_y=0,
            legend_bottom_margin=0.1,
            shared_ylabel_fontsize=14,
            shared_xlabel_fontsize=14,
            legend_scale=1.4,
            use_three_row_legend=False,
            legend_ncol_override=3,
            legend_order=LEGEND_ORDER,
            auto_ylim=False,
            tick_label_fontsize=12,
            data_path_template=[
                #"results_benchmark_{index}_{order}_{game_type}.csv",
                "results_benchmark_{index}_{order}_{game_type}_local_big_data.csv",
                "icml_submission_data/results_benchmark_{index}_{order}_{game_type}.csv",
            ],
            output_path=f"{output_dir}appendix_paper_plots_xgboost_default_combined.pdf",
            y_log_scale=True,
            x_log_scale=True,
            marker_size=3,
            linewidth=1,
            highlight_size=2,
            min_budget=0,
            max_budget=35_000,
            corner_label_mode="order",
            legend_show_markers=True,
            **general_params,
        )
