from __future__ import annotations

import math
import re
import sys
from pathlib import Path as _Path

sys.path.insert(0, str(_Path(__file__).parent / "special_plot_scripts"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from shapiq_benchmark.plot import (
    plot_approximation_quality,
    plot_approximation_quality_vstime,
)
import argparse

from _plot_style import setup_fonts, W_REGULAR

setup_fonts()

parser = argparse.ArgumentParser(
    description="Run benchmark approximations on explanation games."
)
parser.add_argument(
    "--config_approximators",
    type=int,
    default=37,
    help="Configuration ID for approximators: 40 (PAIRING=False, REPLACEMENT=True), "
    "39 (PAIRING=False, REPLACEMENT=False), "
    "38 (PAIRING=True, REPLACEMENT=True), "
    "37 (PAIRING=True, REPLACEMENT=False). Default is 37.",
)
parser.add_argument(
    "--game_type",
    type=str,
    help="Filter by game type.",
)
parser.add_argument(
    "--index",
    type=str,
    default=None,
    help="Index to compute (default: SII). Options: SV, SII, BV, BII",
)
parser.add_argument(
    "--order", type=int, default=None, help="Order of interaction index (default: 1)."
)
parser.add_argument(
    "--n_budget_steps",
    type=int,
    default=20,
    help="Number of budget steps for approximations. Default is 20.",
)
parser.add_argument(
    "--max_budget",
    type=int,
    default=35000,
    help="Maximum budget for approximations. Default is 35000.",
)
args = parser.parse_args()

MAX_BUDGET = args.max_budget
N_BUDGET_STEPS = args.n_budget_steps
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
    # SOUM
    "SOUM30LocalXAI": "SOUM ($n=30$)",
    "SOUM50LocalXAI": "SOUM ($n=50$)",
    "SOUM100LocalXAI": "SOUM ($n=100$)",
    # GRAPH
    "GIN_Mutagenicity_2_data1002": "Mutagenicity1002 ($n=33$)",
    "GIN_Mutagenicity_2_data1060": "Mutagenicity1060 ($n=36$)",
    "GIN_Mutagenicity_2_data1377": "Mutagenicity1377 ($n=35$)",
    "GIN_Mutagenicity_2_data1417": "Mutagenicity1417 ($n=39$)",
    "GIN_Mutagenicity_2_data1459": "Mutagenicity1459 ($n=32$)",
    "GIN_Mutagenicity_2_data150": "Mutagenicity150 ($n=35$)",
    "GIN_Mutagenicity_2_data1564": "Mutagenicity1564 ($n=31$)",
    "GIN_Mutagenicity_2_data1644": "Mutagenicity1644 ($n=30$)",
    "GIN_Mutagenicity_2_data1758": "Mutagenicity1758 ($n=38$)",
    "GIN_Mutagenicity_2_data1939": "Mutagenicity1939 ($n=30$)",
    "GIN_Mutagenicity_2_data2095": "Mutagenicity2095 ($n=35$)",
    "GIN_Mutagenicity_2_data2430": "Mutagenicity2430 ($n=32$)",
    "GIN_Mutagenicity_2_data2441": "Mutagenicity2441 ($n=40$)",
    "GIN_Mutagenicity_2_data2493": "Mutagenicity2493 ($n=29$)",
    "GIN_Mutagenicity_2_data2795": "Mutagenicity2795 ($n=33$)",
    "GIN_Mutagenicity_2_data2818": "Mutagenicity2818 ($n=33$)",
    "GIN_Mutagenicity_2_data282": "Mutagenicity282 ($n=31$)",
    "GIN_Mutagenicity_2_data3035": "Mutagenicity3035 ($n=31$)",
    "GIN_Mutagenicity_2_data3122": "Mutagenicity3122 ($n=33$)",
    "GIN_Mutagenicity_2_data3129": "Mutagenicity3129 ($n=32$)",
    "GIN_Mutagenicity_2_data3249": "Mutagenicity3249 ($n=30$)",
    "GIN_Mutagenicity_2_data3325": "Mutagenicity3325 ($n=33$)",
    "GIN_Mutagenicity_2_data3338": "Mutagenicity3338 ($n=38$)",
    "GIN_Mutagenicity_2_data337": "Mutagenicity337 ($n=30$)",
    "GIN_Mutagenicity_2_data3435": "Mutagenicity3435 ($n=36$)",
    "GIN_Mutagenicity_2_data3470": "Mutagenicity3470 ($n=42$)",
    "GIN_Mutagenicity_2_data3496": "Mutagenicity3496 ($n=33$)",
    "GIN_Mutagenicity_2_data3603": "Mutagenicity3603 ($n=31$)",
    "GIN_Mutagenicity_2_data370": "Mutagenicity370 ($n=33$)",
    "GIN_Mutagenicity_2_data3854": "Mutagenicity3854 ($n=32$)",
    "GIN_Mutagenicity_2_data3937": "Mutagenicity3937 ($n=39$)",
    "GIN_Mutagenicity_2_data4097": "Mutagenicity4097 ($n=30$)",
    "GIN_Mutagenicity_2_data464": "Mutagenicity464 ($n=31$)",
    "GIN_Mutagenicity_2_data487": "Mutagenicity487 ($n=33$)",
    "GIN_Mutagenicity_2_data973": "Mutagenicity973 ($n=35$)",
    # New graph dataset format (Mutagenicity with fixed n_players)
    "Mutagenicity_n30": "Mutagenicity ($n=30$)",
    "Mutagenicity_n40": "Mutagenicity ($n=40$)",
    "Mutagenicity_n20": "Mutagenicity ($n=20$)",
    "Mutagenicity_n25": "Mutagenicity ($n=25$)",
}

APPROXIMATOR_RENAMING = {
    "ProxySHAP (XGBoost)": "ProxySHAP (XGBoost) [our]",
    "ProxySHAP (XGBoost, MSR)": "ProxySHAP (XGBoost, MSR) [our]",
    "ProxySHAP (Linear)": "ProxySHAP (Linear) [our]",
    "ProxySHAP (Linear, MSR)": "ProxySHAP (Linear, MSR) [our]",
    "ProxySPEX (XGBoost)": "ProxySPEX (XGBoost)",
    ## Test Methods ##
    "ConsistentTree": "ProxySHAP-Special [our]",
    "ConsistentTree-Value": "ProxySHAP-Special (Value) [our]",
    "ConsistentTree-Marginal": "ProxySHAP-Special (Marginal) [our]",
    "ConsistentTree-Value-KernelSHAP": "ProxySHAP-Special (Value, KernelSHAP) [our]",
    "ConsistentTree-Simple-InverseBinom": "ProxySHAP-Special (Simple, InverseBinom) [our]",
    "ConsistentTree-Simple-KernelSHAP": "ProxySHAP-Special (Simple, KernelSHAP) [our]",
    "ConsistentTree-Two-Stage": "ProxySHAP-Special (Two-Stage) [our]",
    "ConsistentTreeWeighting": "ConsistentTreeWeighting [our]",
    "ConsistentTreeNoWeighting": "ConsistentTreeNoWeighting [our]",
    "ConsistentTreeEfficiency": "ConsistentTreeEfficiency [our]",
    "ConsistentTreeNoEfficiency": "ConsistentTreeNoEfficiency [our]",
    "NeuralSHAP": "NeuralSHAP [our]",
    "ConsistentLinearNoWeighting": "ConsistentLinearNoWeighting [our]",
    "ConsistentLinearWeighting": "ConsistentLinearWeighting [our]",
    "ConsistentTreeWeightingDepth6": "ConsistentTreeWeightingDepth6 [our]",
    "ConsistentTreeNoWeightingDepth6": "ConsistentTreeNoWeightingDepth6 [our]",
    "ConsistentTreeWeightingDepthN": "ConsistentTreeWeightingDepthN [our]",
    "ConsistentTreeNoWeightingDepthN": "ConsistentTreeNoWeightingDepthN [our]",
    "TreeKernelSHAP": "TreeKernelSHAP [our]",
}

TITLE_FONT_SIZE = 24


# Function to extract p and q
def parse_approximator(s):
    if pd.isna(s):
        return np.nan, np.nan
    match = re.search(r"PolySHAP-(\d+)ADD(?:-(\d+)%)*", s)
    if not match:
        return np.nan, np.nan
    if match:
        p = int(match.group(1))
        q = int(match.group(2)) if match.group(2) else 100
        return p, q
    return None, None


def compute_value(row):
    p = row["p"]
    q = row["q"]
    n = int(row["n_players"])

    # treat NaN (or any missing) as "not a PolySHAP" -> return 0
    if pd.isna(p) or pd.isna(q):
        return 0.0

    # now safe to cast p,q to int
    p = int(p)
    q = int(q)

    # guard if p > n: combinations for k>n are zero, so sum_{i=1}^{p-1} reduces to i=1..n
    if p > n:
        return float(sum(math.comb(n, i) for i in range(1, n + 1)))

    total = sum(math.comb(n, i) for i in range(1, p))
    total += math.comb(n, p) * (q / 100.0)
    return float(total)


LINEAR_METHODS = [
    "ProxySHAP (Linear, MSR) [our]",
    "ProxySHAP (Linear) [our]",
]


def extract_base_game_id(game_id: str) -> str:
    """Extract base game_id by removing id_explain suffix for graph datasets.

    For new graph datasets: "Mutagenicity_n30_0" -> "Mutagenicity_n30"
    For old datasets: "GIN_Mutagenicity_2_data1002" -> "GIN_Mutagenicity_2_data1002" (unchanged)
    """
    parts = game_id.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0]
    return game_id


def plot_runtime(plot_df, INDEX, ORDER, GAME_TYPES, GAME_IDS):
    """Plot MSE vs overhead runtime (total_runtime - evaluations) for each game."""
    COST_PER_EVAL = 0.01  # assumed seconds per budget evaluation

    df = plot_df.copy()

    def round_sig(x, sig=1):
        if x <= 0:
            return 0.0
        magnitude = int(np.floor(np.log10(x)))
        return round(x, -magnitude + (sig - 1))

    df["overhead_time"] = df["total_runtime"] - df["evaluations"]
    df = df.dropna(subset=["overhead_time"])
    df["overhead_time"] = df["overhead_time"].apply(round_sig)
    df["overhead_time"] = (df["budget"] * COST_PER_EVAL + df["overhead_time"]).round(0)

    runtime_params = {
        "figsize": (6, 4),
        "log_scale_y": True,
        "log_scale_min": 1e-8,
        "log_scale_max": 1e2,
        "marker_size": 4.5,
        "linewidth": 2,
        "highlight_size": 1,
        "time_column": "overhead_time",
    }

    for game_type in GAME_TYPES:
        df_gt = df[df["game_type"] == game_type]
        for game_id in GAME_IDS:
            df_game = df_gt[df_gt["game_id"] == game_id]
            if df_game.empty:
                continue
            dataset = df_game["game"].unique()[0]
            fig, ax = plot_approximation_quality_vstime(
                data=df_game,
                metric="MSE",
                legend=False,
                **runtime_params,
            )
            ax.set_xlabel(r"Est. Runtime (s)  [$b \cdot 0.01 + t_{\mathrm{overhead}}$]")
            ax.set_title(DATA_NAMES.get(dataset, dataset), fontsize=TITLE_FONT_SIZE)
            fig.tight_layout()
            fig.savefig(
                f"plots/{game_type}/{game_id}_runtime_{INDEX}_{ORDER}_standard.pdf"
            )
            plt.close(fig)


if __name__ == "__main__":
    if args.index is None and args.order is None:
        iterations = {
            #"SV": 1,
            "SII@2": 2,
            "SII@3": 3,
            #"BV": 1,
            "BII@2": 2,
            "BII@3": 3,
        }
        print("No index or order specified, processing all combinations.")
    elif args.index is not None and args.order is None:
        iterations = {
            args.index: 1,
            args.index: 2,
            args.index: 3,
        }
        print("No order specified, processing all orders for index", args.index)
    elif args.index is None and args.order is not None:
        iterations = {
            "SV": args.order,
            "SII": args.order,
            "BV": args.order,
            "BII": args.order,
        }
        print("No index specified, processing all indices for order", args.order)
    else:
        iterations = {args.index: args.order}
        print("Processing specified index and order:", args.index, args.order)
    print("Iterations to process:", iterations)
    print(args.order, args.index)

    for INDEX, ORDER in iterations.items():
        if INDEX.endswith(f"@{ORDER}"):
            INDEX = INDEX.split("@")[0]
        print(f"Processing index {INDEX} with order {ORDER}...")
        # Load the results from the CSV file
        if args.game_type is not None:
            results_df = pd.read_csv(
                f"results_benchmark_{INDEX}_{ORDER}_{args.game_type}.csv"
            )
        else:
            results_df = pd.read_csv(f"results_benchmark_{INDEX}_{ORDER}.csv")
        results_df = results_df.sort_values(by="n_players")

        # Normalize game_ids for graph datasets: remove id_explain suffix so plotting function
        # can compute SEM/error bands across multiple explanation points
        results_df["game_id"] = results_df["game_id"].apply(extract_base_game_id)
        results_df = results_df.sort_values(by="n_players")

        GAME_IDS = results_df["game_id"].unique()
        GAME_TYPES = results_df["game_type"].unique()

        info = results_df[["game_id", "n_players"]].drop_duplicates()

        results_df[["p", "q"]] = results_df["approximator"].apply(
            lambda x: pd.Series(parse_approximator(x))
        )
        results_df["minimum_budget_to_plot"] = results_df.apply(compute_value, axis=1)
        results_df = results_df[
            results_df["used_budget"] >= results_df["minimum_budget_to_plot"]
        ]
        # Change the naming of the approximators for better visualization
        results_df = results_df.replace({"approximator": APPROXIMATOR_RENAMING})

        # Create and save a legend for the plots
        fig, ax = plot_approximation_quality(
            data=results_df,
            metric="MSE",
            log_scale_y=True,
            # log_scale_x=False,
            legend=True,
        )

        ax.axis("off")
        # Get handles and labels
        handles, labels = ax.get_legend_handles_labels()
        # Replace old labels with new ones
        labels = [APPROXIMATOR_RENAMING.get(l, l) for l in labels]
        # Update legend
        ax.legend(
            handles,
            labels,
            bbox_to_anchor=(1, 0.5),
        )
        # Save the legend separately
        fig.savefig("plots/legend.pdf", bbox_inches="tight", pad_inches=0.1)
        # fig_legend.show()

        # Plot approximation quality for standard
        print("Plotting standard approximations...")
        print(
            "Available approximators in the data:", results_df["approximator"].unique()
        )
        plot_df = results_df[
            # Baselines
            (results_df["approximator"] == "SHAPIQ")
            | (results_df["approximator"] == "SVARMIQ")
            # | (results_df["approximator"] == "PermutationSamplingSV")
            | (results_df["approximator"] == "PermutationSamplingSII")
            | (results_df["approximator"] == "KernelSHAPIQ")
            | (results_df["approximator"] == "ProxySPEX (XGBoost)")
            # Proposed methods
            | (results_df["approximator"] == "ProxySHAP (XGBoost) [our]")
            | (results_df["approximator"] == "ProxySHAP (XGBoost, MSR) [our]")
            | (results_df["approximator"] == "ProxySHAP (Linear) [our]")
            | (results_df["approximator"] == "ProxySHAP (Linear, MSR) [our]")
        ]
        general_params = {
            "figsize": (6, 4),  # (5.5, 3.5),
            "log_scale_y": True,
            "log_scale_x": True,
            "marker_size": 4.5,
            "linewidth": 2,
            "highlight_size": 1,
            # The ylims
            "log_scale_min": 1e-8,
            "log_scale_max": 1e2,
        }
        # plot_df = plot_df[plot_df["id_config_approximator"] == 37]
        print(plot_df["approximator"].unique())
        for game_type in GAME_TYPES:
            plot_df_game_type = plot_df[results_df["game_type"] == game_type]
            for game_id in GAME_IDS:
                plot_df_game_id = plot_df_game_type[
                    plot_df_game_type["game_id"] == game_id
                ]
                if len(plot_df_game_id) > 0:
                    # Only pick the budgets with the correct values
                    n_players = plot_df_game_id["n_players"].values[0]
                    min_b = n_players + 1
                    max_b = (
                        min(2**n_players, MAX_BUDGET) if n_players <= 20 else MAX_BUDGET
                    )
                    budget_range = (
                        np.ceil(
                            np.logspace(
                                np.log10(min_b), np.log10(max_b), N_BUDGET_STEPS
                            )
                        )
                        .clip(min_b, max_b)
                        .astype(int)
                    )
                    plot_df_game_id = plot_df_game_id[
                        plot_df_game_id["budget"].isin(budget_range)
                    ]

                    ## Filter out those budgets for Linear and KernelSHAPIQ that are invalid ##
                    n_players = plot_df_game_id["n_players"].values[0]
                    for method in LINEAR_METHODS:
                        method_mask = plot_df_game_id["approximator"] == method
                        valid_budgets = []
                        for b in plot_df_game_id[method_mask]["budget"].unique():
                            if b >= sum(
                                [
                                    math.comb(n_players + 1, i)
                                    for i in range(0, ORDER + 1)
                                ]
                            ):
                                valid_budgets.append(b)
                        plot_df_game_id = plot_df_game_id[
                            ~(
                                method_mask
                                & (~plot_df_game_id["budget"].isin(valid_budgets))
                            )
                        ]
                    for method in ["KernelSHAPIQ"]:
                        method_mask = plot_df_game_id["approximator"] == method
                        valid_budgets = []
                        for b in plot_df_game_id[method_mask]["budget"].unique():
                            if b >= math.comb(n_players + 1, ORDER):
                                valid_budgets.append(b)
                        plot_df_game_id = plot_df_game_id[
                            ~(
                                method_mask
                                & (~plot_df_game_id["budget"].isin(valid_budgets))
                            )
                        ]

                    # print("Plotting", game_type, game_id)
                    metric = "MSE"
                    dataset = plot_df_game_id["game"].unique()[0]
                    fig, ax = plot_approximation_quality(
                        data=plot_df_game_id,
                        metric=metric,
                        legend=False,
                        **general_params,
                    )
                    ax.set_title(
                        DATA_NAMES.get(dataset, dataset), fontsize=TITLE_FONT_SIZE
                    )
                    fig.tight_layout()
                    # print(
                    #     "Saving figure...",
                    #     f"plots/{game_type}/{game_id}_{metric}_{INDEX}_{ORDER}_standard.png",
                    # )
                    fig.savefig(
                        f"plots/{game_type}/mse/{game_id}_{metric}_{INDEX}_{ORDER}_standard.pdf"
                    )
                    plt.close(fig)

                    metric = "RelativeMSE"
                    dataset = plot_df_game_id["game"].unique()[0]
                    fig, ax = plot_approximation_quality(
                        data=plot_df_game_id,
                        metric=metric,
                        legend=False,
                        **general_params,
                    )
                    ax.set_title(
                        DATA_NAMES.get(dataset, dataset), fontsize=TITLE_FONT_SIZE
                    )
                    fig.tight_layout()
                    # print(
                    #     "Saving figure...",
                    #     f"plots/{game_type}/{game_id}_{metric}_{INDEX}_{ORDER}_standard.png",
                    # )
                    fig.savefig(
                        f"plots/{game_type}/relative_mse/{game_id}_{metric}_{INDEX}_{ORDER}_standard.pdf"
                    )
                    plt.close(fig)
                    
                    metric = "Precision@5"
                    dataset = plot_df_game_id["game"].unique()[0]
                    fig, ax = plot_approximation_quality(
                        data=plot_df_game_id,
                        metric=metric,
                        # log_scale_x=False,
                        legend=False,
                        **{**general_params, "log_scale_y": False},
                    )
                    ax.set_title(
                        DATA_NAMES.get(dataset, dataset), fontsize=TITLE_FONT_SIZE
                    )
                    fig.tight_layout()
                    fig.savefig(
                        f"plots/{game_type}/precision/{game_id}_{metric}_{INDEX}_{ORDER}_standard.pdf"
                    )
                    plt.close(fig)

                    metric = "Precision@10"
                    dataset = plot_df_game_id["game"].unique()[0]
                    fig, ax = plot_approximation_quality(
                        data=plot_df_game_id,
                        metric=metric,
                        # log_scale_x=False,
                        legend=False,
                        **{**general_params, "log_scale_y": False},
                    )
                    ax.set_title(
                        DATA_NAMES.get(dataset, dataset), fontsize=TITLE_FONT_SIZE
                    )
                    fig.tight_layout()
                    fig.savefig(
                        f"plots/{game_type}/precision/{game_id}_{metric}_{INDEX}_{ORDER}_standard.pdf"
                    )
                    plt.close(fig)

                    metric = "SpearmanCorrelation@10"
                    dataset = plot_df_game_id["game"].unique()[0]
                    fig, ax = plot_approximation_quality(
                        data=plot_df_game_id,
                        metric=metric,
                        # log_scale_x=False,
                        legend=False,
                        **{**general_params, "log_scale_y": False},
                    )
                    ax.set_title(
                        DATA_NAMES.get(dataset, dataset), fontsize=TITLE_FONT_SIZE
                    )
                    fig.tight_layout()
                    fig.savefig(
                        f"plots/{game_type}/spearman/{game_id}_{metric}_{INDEX}_{ORDER}_standard.pdf"
                    )
                    plt.close(fig)

                    metric = "SpearmanCorrelation"
                    dataset = plot_df_game_id["game"].unique()[0]
                    fig, ax = plot_approximation_quality(
                        data=plot_df_game_id,
                        metric=metric,
                        # log_scale_x=False,
                        legend=False,
                        **{**general_params, "log_scale_y": False},
                    )
                    ax.set_title(
                        DATA_NAMES.get(dataset, dataset), fontsize=TITLE_FONT_SIZE
                    )
                    fig.tight_layout()
                    fig.savefig(
                        f"plots/{game_type}/spearman/{game_id}_{metric}_{INDEX}_{ORDER}_standard.pdf"
                    )
                    plt.close(fig)

                    plot_df_game_id = plot_df_game_id.sort_values(by="budget")
                    metric = "KendallTau"
                    dataset = plot_df_game_id["game"].unique()[0]
                    fig, ax = plot_approximation_quality(
                        data=plot_df_game_id,
                        metric=metric,
                        # log_scale_x=False,
                        legend=False,
                        **{**general_params, "log_scale_y": False},
                    )
                    ax.set_title(
                        DATA_NAMES.get(dataset, dataset), fontsize=TITLE_FONT_SIZE
                    )
                    fig.tight_layout()
                    fig.savefig(
                        f"plots/{game_type}/kendall/{game_id}_{metric}_{INDEX}_{ORDER}_standard.pdf"
                    )
                    plt.close(fig)

                    metric = "KendallTau@10"
                    dataset = plot_df_game_id["game"].unique()[0]
                    fig, ax = plot_approximation_quality(
                        data=plot_df_game_id,
                        metric=metric,
                        # log_scale_x=False,
                        legend=False,
                        **{**general_params, "log_scale_y": False},
                    )
                    ax.set_title(
                        DATA_NAMES.get(dataset, dataset), fontsize=TITLE_FONT_SIZE
                    )
                    fig.tight_layout()
                    fig.savefig(
                        f"plots/{game_type}/kendall/{game_id}_{metric}_{INDEX}_{ORDER}_standard.pdf"
                    )
                    plt.close(fig)

                    if INDEX == "FBII":
                        metric = "Faithfulness"
                        dataset = plot_df_game_id["game"].unique()[0]
                        fig, ax = plot_approximation_quality(
                            data=plot_df_game_id,
                            metric=metric,
                            log_scale_y=False,
                            # log_scale_x=False,
                            legend=False,
                            **general_params,
                        )
                        ax.set_title(
                            DATA_NAMES.get(dataset, dataset), fontsize=TITLE_FONT_SIZE
                        )
                        fig.tight_layout()
                        fig.savefig(
                            f"plots/{game_type}/faithfulness/{game_id}_{metric}_{INDEX}_{ORDER}_standard.pdf"
                        )
                        plt.close(fig)

        # print("Plotting runtime plots...")
        # plot_runtime(plot_df, INDEX, ORDER, GAME_TYPES, GAME_IDS)
