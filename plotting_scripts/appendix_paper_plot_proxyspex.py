"""This file plots the figures in the style of the main paper."""

import math
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from shapiq.interaction_values import InteractionValues
from shapiq_benchmark.plot import plot_approximation_quality

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


@lru_cache(maxsize=None)
def _get_n_players(game_name: str, order: int, index: str, game_type: str) -> int:
    """Load the number of players for one plot specification."""

    results_df = pd.read_csv(
        f"icml_submission_data/results_benchmark_{index}_{order}_{game_type}.csv"
    )
    n_players = results_df.loc[results_df["game"] == game_name, "n_players"]
    if n_players.empty:
        raise ValueError(
            f"Could not determine n_players for game={game_name}, index={index}, order={order}, game_type={game_type}"
        )
    return int(n_players.iloc[0])

def plot_main_paper_plots(
    game_name,
    order,
    index,
    game_type,
    approximators_to_plot,
    figsize,
    ylim,
    y_log_scale,
    x_log_scale,
    min_budget=0,
    max_budget=float("inf"),
    marker_size=6,
    linewidth=2,
    highlight_size=2,
    save_path=None,
):
    os.makedirs(save_path, exist_ok=True)
    ## Read data and preprare ##
    results_df = pd.read_csv(f"results_benchmark_{index}_{order}_{game_type}.csv")
    # Change the naming of the approximators for better visualization
    results_df = results_df.replace({"approximator": APPROXIMATOR_RENAMING})
    results_df = results_df[results_df["approximator"].isin(approximators_to_plot)]
    results_df = results_df[(results_df["game"] == game_name)]

    # Only pick the budgets with the correct values
    n_players = results_df["n_players"].values[0]
    min_b = n_players + 1
    max_b = (
        min(2**n_players, max_budget)
        if n_players <= 20
        else (
            35_000
            if max_budget == float("inf") and game_name != "ViT4by4Patches"
            else 66_666
        )
    )
    budget_range = (
        np.ceil(np.logspace(np.log10(min_b), np.log10(max_b), 20))
        .clip(min_b, max_b)
        .astype(int)
    )
    results_df = results_df[
        results_df["budget"].isin(budget_range)
        & (results_df["budget"] >= min_budget)
        & (results_df["budget"] <= max_budget)
    ]
    ## Filter out those budgets for Linear and KernelSHAPIQ that are invalid ##
    for method in LINEAR_METHODS:
        method_mask = results_df["approximator"] == method
        valid_budgets = []
        for b in results_df[method_mask]["budget"].unique():
            if b >= sum([math.comb(n_players + 1, i) for i in range(0, order + 1)]):
                valid_budgets.append(b)
        results_df = results_df[
            ~(method_mask & (~results_df["budget"].isin(valid_budgets)))
        ]
    for method in ["KernelSHAPIQ"]:
        method_mask = results_df["approximator"] == method
        valid_budgets = []
        for b in results_df[method_mask]["budget"].unique():
            if b >= math.comb(n_players + 1, order):
                valid_budgets.append(b)
        results_df = results_df[
            ~(method_mask & (~results_df["budget"].isin(valid_budgets)))
        ]
    results_df["RelativeMSE"] = results_df["RelativeMSE"].clip(lower=1e-8)

    ## Save Legend ##
    fig, ax = plot_approximation_quality(
        data=results_df,
        metric="RelativeMSE",
        log_scale_y=y_log_scale,
        log_scale_x=x_log_scale,
        legend=True,
    )
    ax.axis("off")
    # ax.axis("off")
    # Get handles and labels
    handles, labels = ax.get_legend_handles_labels()
    fig, ax = plt.subplots()
    ax.axis("off")
    leg = ax.legend(
        handles,
        labels,
        bbox_to_anchor=(1.05, 1),
        frameon=True,
        fancybox=True,
        framealpha=1,
    )
    leg.get_frame().set_linewidth(1.0)
    leg.get_frame().set_facecolor("none")
    # # Replace old labels with new ones
    # labels = [APPROXIMATOR_RENAMING.get(l, l) for l in labels]
    # # Update legend
    # ax.legend(
    #     handles,
    #     labels,
    #     bbox_to_anchor=(1, 0.5),
    # )
    # Save the legend separately
    fig.savefig(f"{save_path}legend.pdf", bbox_inches="tight", pad_inches=0.1)
    # fig_legend.show()

    # Plot approximation quality for standard
    ## Adjust the limits ##
    # Set min value to the min_value of second highest budget
    second_highest_budget = budget_range[-2]
    temp_df = results_df[results_df["budget"] == second_highest_budget]
    min_value_in_results = (
        temp_df[temp_df["approximator"].str.startswith("ProxySHAP")][
            "RelativeMSE"
        ].min()
        * 1.2
    )
    max_value_in_results = results_df["RelativeMSE"].max()
    new_ylim_max = max_value_in_results * 1.1
    new_ylim_min = min_value_in_results
    ylim = (new_ylim_min, new_ylim_max)
    ## Plot RelativeMSE ##
    metric = "RelativeMSE"
    fig, ax = plot_approximation_quality(
        data=results_df,
        metric=metric,
        log_scale_y=y_log_scale,
        log_scale_x=x_log_scale,
        figsize=figsize,
        log_scale_min=ylim[0] if y_log_scale else None,
        log_scale_max=ylim[1] if y_log_scale else None,
        legend=False,
        marker_size=marker_size,
        linewidth=linewidth,
        highlight_size=highlight_size,
        confidence_metric="sem",
    )
    if index == "SII" or index == "BII":
        ax.set_title(
            GAME_TYPE_ABBREVIATIONS.get(game_type, game_type)
            + "-"
            + DATA_NAMES[game_name]
            + "-"
            + index
            + f"-{str(order)}",
            fontsize=TITLE_FONT_SIZE,
        )
    else:
        ax.set_title(
            GAME_TYPE_ABBREVIATIONS.get(game_type, game_type)
            + "-"
            + DATA_NAMES[game_name]
            + "-"
            + index,
            fontsize=TITLE_FONT_SIZE,
        )
    fig.tight_layout()
    fig.savefig(
        f"{save_path}{game_name}_{index}_{order}_{game_type}_approx_quality_{metric}.pdf"
    )


def plot_gt_vs_approx_by_budget(
    games_to_players: dict[str, int],
    approximators: list[str],
    index: str,
    order: int,
    game_type: str,
    ground_truth_dir: str,
    approx_dir: str,
    n_runs: int,
    target_budgets: list[int],
    config_id_ground_truth: int = 40,
    config_id_approx: int = 37,
    max_budget: int = 35_000,
    tiny_threshold: float = 1e-6,
    output_prefix: str = "gt_approx_comparison_budget_",
):
    def _budget_for_game(n_players: int, target: int) -> int:
        min_budget = n_players + 1
        budget_range = (
            np.ceil(np.logspace(np.log10(min_budget), np.log10(max_budget), 20))
            .clip(min_budget, max_budget)
            .astype(int)
        )
        return int(budget_range[np.argmin(np.abs(budget_range - target))])

    def _normalize_for_plot(values: np.ndarray) -> tuple[float, float] | None:
        filtered = values[np.abs(values) > tiny_threshold]
        if filtered.size == 0:
            return None
        return float(filtered.min()), float(filtered.max())

    budgets_per_game = {
        target: {
            game: _budget_for_game(n_players, target)
            for game, n_players in games_to_players.items()
        }
        for target in target_budgets
    }

    colors = list(plt.cm.tab10.colors)
    game_colors = {
        game: colors[i % len(colors)] for i, game in enumerate(games_to_players.keys())
    }

    data: dict[tuple[str, str, int, int], tuple[np.ndarray, np.ndarray]] = {}
    for game, _ in games_to_players.items():
        for approximator in approximators:
            for target in target_budgets:
                budget = budgets_per_game[target][game]
                for run in range(n_runs):
                    gt_path = Path(
                        f"{ground_truth_dir}/{game_type}/{game}_3_{config_id_ground_truth}_{run}_{index}_{order}_exact_values.json"
                    )
                    approx_path = Path(
                        f"{approx_dir}/{game_type}/{game}_3_{config_id_approx}_{run}_{approximator}_{budget}_{index}_{order}.json"
                    )
                    try:
                        gt_values = InteractionValues.from_json_file(gt_path)
                        approx_values = InteractionValues.from_json_file(approx_path)
                    except Exception:
                        continue

                    approx_vals = np.array(
                        [approx_values[inter] for inter in gt_values.interactions]
                    )
                    gt_vals = gt_values.values

                    approx_bounds = _normalize_for_plot(approx_vals)
                    gt_bounds = _normalize_for_plot(gt_vals)
                    if approx_bounds is None or gt_bounds is None:
                        continue

                    approx_min, approx_max = approx_bounds
                    gt_min, gt_max = gt_bounds
                    if approx_max - approx_min <= 0 or gt_max - gt_min <= 0:
                        continue

                    approx_scaled = (
                        2 * ((approx_vals - approx_min) / (approx_max - approx_min)) - 1
                    )
                    gt_scaled = 2 * ((gt_vals - gt_min) / (gt_max - gt_min)) - 1
                    data[(game, approximator, budget, run)] = (
                        gt_scaled,
                        approx_scaled,
                    )

    for target in target_budgets:
        n_cols = len(approximators)
        fig, axes = plt.subplots(figsize=(4 * n_cols, 5), nrows=1, ncols=n_cols)
        if n_cols == 1:
            axes = np.array([axes])

        for col_idx, approximator in enumerate(approximators):
            ax = axes[col_idx]
            for game in games_to_players.keys():
                budget = budgets_per_game[target][game]
                gt_vals_all_runs = []
                approx_vals_all_runs = []
                for run in range(n_runs):
                    key = (game, approximator, budget, run)
                    if key in data:
                        gt_vals, approx_vals = data[key]
                        gt_vals_all_runs.append(gt_vals)
                        approx_vals_all_runs.append(approx_vals)

                if gt_vals_all_runs and approx_vals_all_runs:
                    gt_vals_concat = np.concatenate(gt_vals_all_runs)
                    approx_vals_concat = np.concatenate(approx_vals_all_runs)
                    ax.scatter(
                        gt_vals_concat,
                        approx_vals_concat,
                        alpha=0.5,
                        label=game,
                        color=game_colors[game],
                        s=20,
                    )

            ax.plot(
                [-1, 1],
                [-1, 1],
                color="red",
                linestyle="--",
                label="Perfect Approximation",
                linewidth=2,
            )
            ax.set_xlabel("Ground Truth Values")
            ax.set_ylabel("Approximated Values")
            ax.set_title(approximator)
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
            ax.grid(alpha=0.3)
            if col_idx == 0:
                ax.legend()

        fig.suptitle(f"Budget {target}", fontsize=16, y=1.02)
        fig.tight_layout()
        fig.savefig(f"{output_prefix}{target}.png", bbox_inches="tight", dpi=150)


TITLE_FONT_SIZE = 24
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
    "ProxySPEX (XGBoost) [cutoff 0.95]",
    "ProxySPEX (XGBoost) [cutoff 0.96]",
    "ProxySPEX (XGBoost) [cutoff 0.97]",
    "ProxySPEX (XGBoost) [cutoff 0.98]",
    "ProxySPEX (XGBoost) [cutoff 0.99]",
    "ProxySPEX (XGBoost) [no truncation, no refinement]",
    # Our Methods [XGBOOST]
    "ProxySHAP (XGBoost) [our]",
    "ProxySHAP (XGBoost, MSR) [our]",
]
LINEAR_METHODS = [
    "ProxySHAP (Linear, MSR) [our]",
    "ProxySHAP (Linear) [our]",
]
APPROXIMATOR_RENAMING = {
    "RegressionMSRIQ": "ProxySHAP (XGBoost, MSR) [our]",
    "RegressionMSRIQ-NoAdjustment": "ProxySHAP (XGBoost) [our]",
    ## LINEAR METHODS ##
    "Linear-NoAdjustment": "ProxySHAP (Linear) [our]",
    "Linear-RECAP": "ProxySHAP (Linear, MSR) [our]",
    "RegressionMSRIQ-XGB-CV-BO": "ProxySHAP* (XGBoost, MSR) [our]",
    "RegressionMSRIQ-XGB-CV-BO-NoAdjustment": "ProxySHAP* (XGBoost) [our]",
    "ProxySHAP (XGBoost)": "ProxySHAP (XGBoost) [our]",
    "ProxySHAP (XGBoost, MSR)": "ProxySHAP (XGBoost, MSR) [our]",
    "ProxySHAP (Linear)": "ProxySHAP (Linear) [our]",
    "ProxySHAP (Linear, MSR)": "ProxySHAP (Linear, MSR) [our]",
    # ProxySPEX CuttOffs
    "ProxySPEX (XGBoost)" : "ProxySPEX (XGBoost) [cutoff 0.95]",
    "ProxySPEX96 (XGBoost)": "ProxySPEX (XGBoost) [cutoff 0.96]",
    "ProxySPEX97 (XGBoost)": "ProxySPEX (XGBoost) [cutoff 0.97]",
    "ProxySPEX98 (XGBoost)": "ProxySPEX (XGBoost) [cutoff 0.98]",
    "ProxySPEX99 (XGBoost)": "ProxySPEX (XGBoost) [cutoff 0.99]",
    "ProxySPEX (XGBoost, NoTruncation, NoRefinement)": "ProxySPEX (XGBoost) [no truncation, no refinement]",
}

if __name__ == "__main__":
    # Interleave row-by-row so column 0 = order 2 and column 1 = order 3.
    plot_specs_a4 = [
        {
            "game_name": "TabArenaCoil2000LocalXAI",
            "order": 2,
            "index": "SII",
            "game_type": "interventional",
        },
    ]
    data_path_template = [
        "results_benchmark_{index}_{order}_{game_type}_coil_exp.csv",
        "results_benchmark_{index}_{order}_{game_type}.csv",
        # "icml_submission_data/results_benchmark_{index}_{order}_{game_type}.csv"
    ]
    output_dir_a4 = "plots/appendix/"
    os.makedirs(output_dir_a4, exist_ok=True)
    plot_combined_main_paper_plots(
        plot_specs=plot_specs_a4,
        approximators_to_plot=APPROXIMATORS_TO_PLOT,
        approximator_renaming=APPROXIMATOR_RENAMING,
        linear_methods=LINEAR_METHODS,
        data_names=DATA_NAMES,
        title_font_size=9,
        panel_figsize=(8, 4),
        plots_per_row=2,
        max_budget=35_000,  # needed for high-d games (d>20) to avoid inf in logspace
        show_captions_only_first_row=False,
        x_ticks_only_last_row=False,
        subplot_wspace=0.15,
        subplot_hspace=0.15,
        shared_ylabel_x=0.06,
        shared_xlabel_y=0,
        shared_ylabel_fontsize=9,
        shared_xlabel_fontsize=9,
        bottom_margin=0.1,
        legend_bottom_margin=0.2,
        use_three_row_legend=True,
        row_title_on_right=True,
        legend_ncol_override=4,
        linewidth=2,
        auto_ylim=True,
        data_path_template=data_path_template,
        output_path="plots/appendix/proxyspex_truncation_comparison.pdf",
        corner_label_mode="order",
        order_label_in_corner=True,
        y_log_scale=True,
        x_log_scale=True,
        marker_size=4,
        highlight_size=3,
    )
