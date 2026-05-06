from __future__ import annotations

# pyright: reportGeneralTypeIssues=false, reportUnknownMemberType=false, reportMissingImports=false

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _build_configuration_label(row: pd.Series) -> str:
    parts = [row["value_model"], row["sampling_weight"], row["residual_approximator"]]
    extras: list[str] = []
    if not pd.isna(row.get("max_depth")):
        extras.append(f"depth={int(row['max_depth']) if not pd.isna(row['max_depth']) else 'None'}")
    if not pd.isna(row.get("n_estimators")):
        extras.append(f"estimators={int(row['n_estimators'])}")
    label = " / ".join(parts)
    if extras:
        label += " (" + ", ".join(extras) + ")"
    return label


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot MSE distribution per configuration and dataset.")
    parser.add_argument("csv", type=Path, help="Path to the CSV file produced by find_optimal_model.py")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("plots/find_optimal_model"),
        help="Directory where plots and rankings are written.",
    )
    parser.add_argument(
        "--log-scale",
        action="store_true",
        help="Plot the MSE axis on a logarithmic scale.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    df = pd.read_csv(args.csv)
    if "game_identifier" not in df.columns:
        raise ValueError("Expected column 'game_identifier' in CSV")

    df["configuration"] = df.apply(_build_configuration_label, axis=1)

    summary = (
        df.groupby(["game_identifier", "value_model", "sampling_weight", "residual_approximator", "configuration"], dropna=False)["MSE"]
        .agg(["count", "median", "mean", "std"])
        .reset_index()
    )
    best_overall_idx = summary.groupby("game_identifier")["median"].idxmin()
    best_configs = summary.loc[best_overall_idx].copy()
    best_per_model = (
        df.loc[df.groupby(["game_identifier", "value_model"]) ["MSE"].idxmin()]  # type: ignore[index]
        .reset_index(drop=True)
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_dir / "configuration_summary.csv", index=False)
    best_configs.to_csv(args.output_dir / "best_configurations.csv", index=False)
    best_per_model.to_csv(args.output_dir / "best_configurations_per_model.csv", index=False)

    plt.rcParams.update({"axes.grid": True, "grid.alpha": 0.3})

    for dataset, ds_df in df.groupby("game_identifier"):
        # sort value models to keep dt, rf, xgb, lightgbm order when present
        value_models = [model for model in ["dt", "rf", "xgb", "lightgbm"] if model in ds_df["value_model"].unique()]
        if not value_models:
            continue

        positions: list[float] = []
        data_series: list[np.ndarray] = []
        best_points: list[tuple[float, float]] = []
        best_labels: list[str] = []

        for idx, model in enumerate(value_models, start=1):
            model_df = ds_df[ds_df["value_model"] == model]
            values = model_df["MSE"].dropna().to_numpy()
            if values.size == 0:
                continue
            positions.append(float(idx))
            data_series.append(values)
            best_row_model = model_df.sort_values("MSE", kind="mergesort").iloc[0]
            best_points.append((float(idx), float(best_row_model["MSE"])))
            best_labels.append(str(best_row_model["configuration"]))

        if not data_series:
            continue

        fig, ax = plt.subplots(figsize=(10, 6))
        box = ax.boxplot(
            data_series,
            positions=positions,
            widths=0.6,
            patch_artist=True,
            medianprops={"color": "#2f4b7c", "linewidth": 1.5},
            boxprops={"linewidth": 1.2, "facecolor": "#cbd5f5"},
            whiskerprops={"linewidth": 1.2},
            capprops={"linewidth": 1.2},
        )

        # Highlight minimum configuration per value model
        ax.scatter(
            [x for x, _ in best_points],
            [y for _, y in best_points],
            marker="*",
            color="#d62728",
            s=120,
            zorder=5,
            label="Best configuration",
        )

        # Jittered scatter of all configurations for context
        for (x_coord, values) in zip(positions, data_series):
            jitter = (np.random.rand(values.size) - 0.5) * 0.15
            ax.scatter(
                np.full(values.shape, x_coord, dtype=float) + jitter,
                values,
                color="#1f77b4",
                alpha=0.5,
                s=18,
            )

        for patch, model in zip(box["boxes"], value_models):
            patch.set_facecolor("#cbd5f5")
        annotation_offsets = [15, -18, 30, -33, 45, -48]
        for idx, ((x_coord, y_coord), label) in enumerate(zip(best_points, best_labels)):
            offset = annotation_offsets[idx % len(annotation_offsets)]
            vertical_alignment = "bottom" if offset > 0 else "top"
            ax.annotate(
                label,
                xy=(x_coord, y_coord),
                xytext=(0, offset),
                textcoords="offset points",
                ha="center",
                va=vertical_alignment,
                fontsize=9,
                rotation=0,
                bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "#cccccc", "alpha": 0.9},
            )

        if args.log_scale:
            ax.set_yscale("log")
        ax.set_xlabel("Value model")
        ax.set_ylabel("MSE")
        ax.set_title(f"{dataset} — configuration spread by value model")
        ax.set_xticks(positions)
        ax.set_xticklabels([model.upper() for model in value_models])
        ax.legend(loc="upper right")
        fig.tight_layout()
        fig.savefig(args.output_dir / f"{dataset}_mse_boxplot.png", dpi=300)
        plt.close(fig)


if __name__ == "__main__":
    main()
