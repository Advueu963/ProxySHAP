"""Winner map: order-2 and order-3 side by side for a given index and game type.

Each panel shows x=budget, y=dataset (sorted by n_players ascending), colored
square = method with the lowest mean MSE at that (dataset, budget) pair.

Use --game-type merged to combine interventional, exhaustive, and tabpfn results;
dataset names are suffixed with (INT), (EXT), (TABPFN).

Example:
    uv run python special_plot_scripts/main_paper_plot_winnermap.py \\
        --index SII --game-type interventional

    uv run python special_plot_scripts/main_paper_plot_winnermap.py \\
        --index SII --game-type merged
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.ticker import FixedLocator, NullFormatter
import numpy as np
import pandas as pd

from _plot_style import apply_tick_style, setup_fonts, W_REGULAR, W_SEMIBOLD

setup_fonts()

APPROXIMATOR_RENAMING: dict[str, str] = {
    "RegressionMSRIQ-NoAdjustment": "ProxySHAP (XGBoost) $\\mathbf{[our]}$",
    "ProxySHAP (XGBoost)": "ProxySHAP (XGBoost) $\\mathbf{[our]}$",
    "ProxySHAP (XGBoost, MSR)": "ProxySHAP (XGBoost, MSR) $\\mathbf{[our]}$",
    "RegressionMSRIQ": "ProxySHAP (XGBoost, MSR) $\\mathbf{[our]}$",
    "ProxySHAP (Linear)": "ProxySHAP (Linear) $\\mathbf{[our]}$",
    "ProxySHAP (Linear, MSR)": "ProxySHAP (Linear, MSR) $\\mathbf{[our]}$",
    "ProxySPEX (XGBoost)": "ProxySPEX (XGBoost)",
    "ProxySPEX (XGBoost, NoRefinement)": "ProxySPEX (XGBoost, NoRef.)",
    "ProxySHAP+ (XGBoost)": "ProxySHAP (XGBoost+HPO-Informed) $\\mathbf{[our]}$",
    "ProxySPEX+ (XGBoost)": "ProxySPEX (XGBoost+HPO-Informed)",
    "ProxySPEX (XGBoost, NoTruncation, NoRefinement)": "ProxySPEX (XGBoost, NoTrunc.)",
}

APPROXIMATORS_TO_PLOT: list[str] = [
    "ProxySHAP (XGBoost) $\\mathbf{[our]}$",
    "ProxySHAP (XGBoost, MSR) $\\mathbf{[our]}$",
    "ProxySHAP (Linear) $\\mathbf{[our]}$",
    "ProxySHAP (Linear, MSR) $\\mathbf{[our]}$",
    "ProxySHAP (XGBoost+HPO-Informed) $\\mathbf{[our]}$",
    "KernelSHAPIQ",
    "ProxySPEX (XGBoost)",
    "ProxySPEX (XGBoost+HPO-Informed)",
    "PermutationSamplingSII",
    "SHAPIQ",
    "SVARMIQ",
]

STYLE: dict[str, dict] = {
    "ProxySHAP (XGBoost) $\\mathbf{[our]}$": {"color": "#61abec"},
    "ProxySHAP (XGBoost, MSR) $\\mathbf{[our]}$": {"color": "#1e88e5"},
    "ProxySHAP (XGBoost+HPO-Informed) $\\mathbf{[our]}$": {"color": "#1e25e5"},
    "ProxySPEX (XGBoost+HPO-Informed)": {"color": "#a91ee5"},
    "ProxySHAP (Linear) $\\mathbf{[our]}$": {"color": "#5bc75e"},
    "ProxySHAP (Linear, MSR) $\\mathbf{[our]}$": {"color": "#15B01A"},
    "KernelSHAPIQ": {"color": "#ff6f00"},
    "ProxySPEX (XGBoost)": {"color": "#ef27a6"},
    "PermutationSamplingSII": {"color": "#252525"},
    "SHAPIQ": {"color": "#959595"},
    "SVARMIQ": {"color": "#707070"},
}

LEGEND_NCOL = 6
LEGEND_METHOD_ORDER = [
    "ProxySHAP (XGBoost) $\\mathbf{[our]}$",
    "ProxySHAP (XGBoost, MSR) $\\mathbf{[our]}$",
    "ProxySHAP (Linear) $\\mathbf{[our]}$",
    "ProxySHAP (Linear, MSR) $\\mathbf{[our]}$",
    "KernelSHAPIQ",
    "ProxySPEX (XGBoost)",
    "ProxySPEX (XGBoost+HPO-Informed)",
    "ProxySHAP (XGBoost+HPO-Informed) $\\mathbf{[our]}$",
    "PermutationSamplingSII",
    "SHAPIQ",
    "SVARMIQ",
]

MERGE_GAME_TYPES: dict[str, str] = {
    "interventional": "(INT)",
    "exhaustive": "(EXT)",
    "tabpfn": "(TABPFN)",
    "graph": "(GRAPH)",
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

_MERGE_SUFFIXES: tuple[str, ...] = tuple(MERGE_GAME_TYPES.values())

# File-name suffixes appended after `results_benchmark_{index}_{order}_{game_type}`
# (before `.csv`). All matching CSVs are concatenated. Add new variants here.
CSV_FILE_SUFFIXES: tuple[str, ...] = (
    "",
    "_local",
    "_local_big_data",
    "_bii3_missing",
    "_bii2_large",

)


def _format_game_label(game: str, n_players: object) -> str:
    """Map a (possibly suffixed) game name to its display label.

    Falls back to ``"{game}  (d={n_players})"`` when the game is not in
    ``DATA_NAMES``. Any merge suffix (e.g. " (INT)") is preserved at the end.
    """
    base, suffix = game, ""
    for s in _MERGE_SUFFIXES:
        marker = f" {s}"
        if base.endswith(marker):
            base = base[: -len(marker)]
            suffix = f" {s}"
            break
    if base in DATA_NAMES:
        return f"{DATA_NAMES[base]}{suffix}"
    return f"{game}  (d={n_players})"


def _read_csv_robust(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except pd.errors.ParserError:
        print(
            f"Warning: parsing {path.name} with robust row normalization due to malformed rows."
        )

    with path.open(newline="") as file_handle:
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
        print(f"Warning: normalized {normalized_rows} malformed rows in {path.name}.")
    df = pd.DataFrame(rows, columns=header)
    string_columns = {"game_type", "game", "model", "game_id", "approximator"}
    for column in df.columns:
        if column not in string_columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def _load_single_csv(
    *, index: str, order: int, game_type: str, suffix: str | None
) -> pd.DataFrame:
    base = f"results_benchmark_{index}_{order}_{game_type}"
    candidate_paths = [Path(f"{base}{s}.csv") for s in CSV_FILE_SUFFIXES]
    found = []
    for path in candidate_paths:
        if not path.exists():
            continue
        found.append((path, _read_csv_robust(path)))
    if not found:
        raise FileNotFoundError(
            f"Could not find any input CSV: tried {[str(p) for p in candidate_paths]}"
        )
    missing = [str(p) for p in candidate_paths if not p.exists()]
    if missing:
        print(f"Warning: only found {[str(p) for p, _ in found]}, missing {missing}")
    df = (
        pd.concat([d for _, d in found], ignore_index=True)
        if len(found) > 1
        else found[0][1]
    )
    df = df[df["n_players"] >= 12].copy()

    if game_type == "interventional":
        df = df[df["n_players"] >= 17].copy()
    df["approximator"] = df["approximator"].str.replace(
        r"\s*\[our\]\s*$", "", regex=True
    )
    df = df.replace({"approximator": APPROXIMATOR_RENAMING})
    min_budget_mask = df["approximator"].isin(
        [
            "KernelSHAPIQ",
            "ProxySHAP (Linear) $\\mathbf{[our]}$",
            "ProxySHAP (Linear, MSR) $\\mathbf{[our]}$",
        ]
    )
    min_budget = df["n_players"].apply(lambda n: math.comb(int(n) + 1, order))
    df = df[~(min_budget_mask & (df["budget"] < min_budget))].copy()

    hpo_informed = [
        "ProxySHAP (XGBoost+HPO-Informed) $\\mathbf{[our]}$",
        "ProxySPEX (XGBoost+HPO-Informed)",
    ]
    df = df[~(df["approximator"].isin(hpo_informed) & (df["n_players"] < 1000))].copy()

    if suffix is not None:
        df["game"] = df["game"].astype(str) + f" {suffix}"

    return df


def _load_results(*, index: str, order: int, game_type: str) -> pd.DataFrame:
    if game_type == "merged":
        frames: list[pd.DataFrame] = []
        for gt, suffix in MERGE_GAME_TYPES.items():
            try:
                frames.append(
                    _load_single_csv(
                        index=index, order=order, game_type=gt, suffix=suffix
                    )
                )
            except FileNotFoundError as exc:
                print(f"Warning: skipping {gt} — {exc}")
        if not frames:
            raise FileNotFoundError(
                "No CSV files found for any of the merged game types."
            )
        return pd.concat(frames, ignore_index=True)
    return _load_single_csv(index=index, order=order, game_type=game_type, suffix=None)


def _compute_winners(df: pd.DataFrame) -> pd.DataFrame:
    """Return DataFrame [game, budget, winner, n_players]."""
    df = df[df["approximator"].isin(APPROXIMATORS_TO_PLOT)].copy()
    if df.empty:
        return pd.DataFrame(columns=["game", "budget", "winner", "n_players"])

    mean_mse = df.groupby(["game", "budget", "approximator"], as_index=False)[
        "MSE"
    ].mean()
    idx = mean_mse.groupby(["game", "budget"])["MSE"].idxmin()
    winners = (
        mean_mse.loc[idx, ["game", "budget", "approximator"]]
        .rename(columns={"approximator": "winner"})
        .copy()
    )
    n_players_map = df.groupby("game")["n_players"].first()
    winners["n_players"] = winners["game"].map(n_players_map)
    return winners


def _draw_winner_panel(
    ax,
    winners: pd.DataFrame,
    *,
    games_ordered: list[str],
    x_log_scale: bool,
    xlabel_pad: float,
    ylabel_pad: float,
    xlabel_size: float,
    ylabel_size: float,
    tick_size: float,
    show_ylabel: bool,
    subtitle: str | None,
) -> None:
    """Render one winner-map panel onto ax."""
    n_games = len(games_ordered)
    game_to_y = {g: i for i, g in enumerate(games_ordered)}
    method_color = {m: STYLE[m]["color"] for m in APPROXIMATORS_TO_PLOT if m in STYLE}

    rect_height = 0.72
    x_min = float("inf")
    x_max = 0.0
    for game, group in winners.groupby("game"):
        if game not in game_to_y:
            continue
        y = game_to_y[game]
        group = group.sort_values("budget").reset_index(drop=True)
        budgets = group["budget"].astype(float).tolist()
        winner_names = group["winner"].tolist()
        if not budgets:
            continue
        x_min = min(x_min, budgets[0])
        for i, (b, w) in enumerate(zip(budgets, winner_names)):
            if i + 1 < len(budgets):
                x_end = budgets[i + 1]
            elif len(budgets) >= 2 and budgets[-2] > 0:
                x_end = b * (budgets[-1] / budgets[-2])
            else:
                x_end = b * 1.5
            x_max = max(x_max, x_end)
            # Small multiplicative overlap so anti-aliasing doesn't leave hairline gaps
            # between adjacent rectangles on the log axis.
            x_end_drawn = x_end * 1.01
            color = method_color.get(w, "#888888")
            ax.add_patch(
                Rectangle(
                    (b, y - rect_height / 2),
                    x_end_drawn - b,
                    rect_height,
                    facecolor=color,
                    edgecolor=color,
                    linewidth=0.5,
                    antialiased=False,
                    zorder=3,
                )
            )

    for i in range(n_games):
        ax.axhline(i - 0.5, color="#dddddd", linewidth=0.4, zorder=1)

    n_players_map = winners.set_index("game")["n_players"].to_dict()
    ylabels = [_format_game_label(g, n_players_map.get(g, "?")) for g in games_ordered]
    ax.set_yticks(range(n_games))
    if show_ylabel:
        ax.set_yticklabels(ylabels, fontsize=tick_size)
        ax.set_ylabel(
            "Dataset", fontsize=ylabel_size, fontweight=W_SEMIBOLD, labelpad=ylabel_pad
        )
    else:
        ax.set_yticklabels([])
    ax.set_ylim(-0.7, n_games - 0.3)

    if x_log_scale:
        ax.set_xscale("log")
    if math.isfinite(x_min) and x_max > x_min:
        ax.set_xlim(x_min, x_max)
    all_budgets = sorted(winners["budget"].unique().tolist())
    tick_candidates = np.unique(
        np.round(np.geomspace(all_budgets[0], all_budgets[-1], 7)).astype(int)
    )
    ticks = sorted(
        {int(min(all_budgets, key=lambda b, t=t: abs(b - t))) for t in tick_candidates}
    )
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.grid(
        True,
        which="major",
        axis="x",
        linestyle="--",
        linewidth=0.5,
        alpha=0.35,
        zorder=0,
    )

    ax.set_xlabel(
        "Budget", fontsize=xlabel_size, fontweight=W_SEMIBOLD, labelpad=xlabel_pad
    )

    if subtitle is not None:
        ax.set_title(subtitle, fontsize=10, fontweight=W_REGULAR, pad=4)


def plot_dual_winner_map(
    *,
    index: str,
    game_type: str,
    output_path: str,
    x_log_scale: bool = True,
    title: str | None = None,
    xlabel_size: float = 14,
    ylabel_size: float = 14,
    tick_size: float = 7,
    legend_size: float = 14,
    legend_pad: float = 0.02,
) -> None:
    """Side-by-side winner maps for order 2 (left) and order 3 (right)."""
    df2 = _load_results(index=index, order=2, game_type=game_type)
    df3 = _load_results(index=index, order=3, game_type=game_type)

    winners2 = _compute_winners(df2)
    winners3 = _compute_winners(df3)

    # Build a shared y-axis: union of all datasets, sorted by n_players.
    # Lower n_players appear at the bottom and higher n_players at the top.
    all_games_df = pd.concat(
        [
            winners2[["game", "n_players"]].drop_duplicates(),
            winners3[["game", "n_players"]].drop_duplicates(),
        ]
    ).drop_duplicates(subset="game")
    games_ordered = all_games_df.sort_values(
        ["n_players", "game"],
        ascending=[True, True],
        kind="stable",
    )["game"].tolist()
    n_games = len(games_ordered)

    fig_height = max(4.0, n_games * 0.32 + 1.5)
    fig, axes = plt.subplots(1, 2, figsize=(16, fig_height), sharey=False)

    _draw_winner_panel(
        axes[0],
        winners2,
        games_ordered=games_ordered,
        x_log_scale=x_log_scale,
        xlabel_pad=-1,
        ylabel_pad=-1,
        xlabel_size=xlabel_size,
        ylabel_size=ylabel_size,
        tick_size=tick_size,
        show_ylabel=True,
        subtitle=f"Order 2  ({index})",
    )
    _draw_winner_panel(
        axes[1],
        winners3,
        games_ordered=games_ordered,
        x_log_scale=x_log_scale,
        xlabel_pad=-1,
        ylabel_pad=-1,
        xlabel_size=xlabel_size,
        ylabel_size=ylabel_size,
        tick_size=tick_size,
        show_ylabel=False,
        subtitle=f"Order 3  ({index})",
    )

    # Shared legend at the bottom, single row: XGBoost variants, Linear variants, baselines.
    method_color = {m: STYLE[m]["color"] for m in APPROXIMATORS_TO_PLOT if m in STYLE}
    ordered_methods = LEGEND_METHOD_ORDER
    legend_handles = [
        axes[0].scatter(
            [], [], c=method_color[m], s=60, marker="s", linewidths=0, label=m
        )
        for m in ordered_methods
        if m in method_color
    ]
    legend_offset = -abs(legend_pad)
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=min(LEGEND_NCOL, len(legend_handles)) if legend_handles else 1,
        frameon=False,
        fontsize=legend_size,
        handlelength=1.0,
        handletextpad=0.4,
        labelspacing=0.5,
        columnspacing=1.2,
        bbox_to_anchor=(0.5, legend_offset),
    )

    if title is not None:
        fig.suptitle(title, fontsize=13, fontweight=W_REGULAR)

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.08)
    for ax in axes:
        apply_tick_style(ax, tick_label_fontsize=tick_size)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", pad_inches=0.05, dpi=300)
    plt.close(fig)
    print(f"Saved plot to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Side-by-side winner maps for order 2 and order 3."
    )
    parser.add_argument("--index", type=str, default="SII")
    parser.add_argument(
        "--game-type",
        type=str,
        default="interventional",
        help="interventional | exhaustive | tabpfn | merged  (merged combines all three)",
    )
    parser.add_argument("--output-path", type=str, default=None)
    parser.add_argument("--no-x-log-scale", action="store_true")
    parser.add_argument("--title", type=str, default=None)
    parser.add_argument("--xlabel-size", type=float, default=25)
    parser.add_argument("--ylabel-size", type=float, default=25)
    parser.add_argument("--tick-size", type=float, default=16)
    parser.add_argument("--ytick-size", dest="tick_size", type=float, default=14)
    parser.add_argument("--legend-size", type=float, default=15)
    parser.add_argument("--legend-pad", type=float, default=0.1)

    args = parser.parse_args()
    x_log = not args.no_x_log_scale

    out = args.output_path or (
        f"plots/main/winnermap_{args.index}_{args.game_type}_order2_order3.pdf"
    )

    plot_dual_winner_map(
        index=args.index,
        game_type=args.game_type,
        output_path=out,
        x_log_scale=x_log,
        title=args.title,
        xlabel_size=args.xlabel_size,
        ylabel_size=args.ylabel_size,
        tick_size=args.tick_size,
        legend_size=args.legend_size,
        legend_pad=args.legend_pad,
    )
