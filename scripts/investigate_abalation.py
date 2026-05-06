from __future__ import annotations

import glob
from pathlib import Path

import pandas as pd


LIGHT_GRAY = "#d3d3d3"


def _bh_fdr(p_values: list[float]) -> list[float]:
    """Benjamini–Hochberg FDR correction (returns q-values).

    Expects p-values in [0, 1]. NaNs are treated as 1.0.
    """

    import math

    m = len(p_values)
    indexed = []
    for i, p in enumerate(p_values):
        if p is None or (isinstance(p, float) and math.isnan(p)):
            p = 1.0
        p = max(0.0, min(1.0, float(p)))
        indexed.append((i, p))
    indexed.sort(key=lambda t: t[1])

    q = [1.0] * m
    prev = 1.0
    for rank, (orig_i, p) in enumerate(reversed(indexed), start=1):
        # reversed => rank from m..1
        r = m - rank + 1
        val = (p * m) / max(1, r)
        prev = min(prev, val)
        q[orig_i] = min(1.0, prev)
    return q


def _format_pval(p: float, *, digits: int = 4, eps: float = 2.2e-16) -> str:
    """Format p-values similar to R (e.g., '<2.2e-16', '0.0312', '1.2e-06')."""

    import math

    if p is None or (isinstance(p, float) and math.isnan(p)):
        return "NA"
    p = float(p)
    if p < eps:
        return f"<{eps:.1e}"
    # Use fixed for moderate p, scientific for tiny p.
    if p < 10 ** (-(digits)):
        return f"{p:.{digits}e}"
    return f"{p:.{digits}g}"


def _r_signif_code(p: float) -> str:
    """R-like significance codes."""

    import math

    if p is None or (isinstance(p, float) and math.isnan(p)):
        return ""
    p = float(p)
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    if p < 0.1:
        return "."
    return ""


def _print_significance_summary(
    table: pd.DataFrame,
    *,
    alpha_p: float = 0.05,
    alpha_q: float = 0.05,
    max_rows: int | None = None,
) -> None:
    if table is None or len(table) == 0:
        print("No significance results to report.")
        return

    required = {
        "value_model",
        "knob",
        "test",
        "k_groups",
        "n_total",
        "statistic",
        "p_value",
        "q_value_fdr_bh",
    }
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"Significance table is missing columns: {sorted(missing)}")

    t = table.copy()
    t["p_fmt"] = t["p_value"].map(lambda x: _format_pval(x))
    t["q_fmt"] = t["q_value_fdr_bh"].map(lambda x: _format_pval(x))
    t["p_sig"] = t["p_value"].map(_r_signif_code)
    t["q_sig"] = t["q_value_fdr_bh"].map(_r_signif_code)
    t["reject_p"] = t["p_value"] <= alpha_p
    t["reject_q"] = t["q_value_fdr_bh"] <= alpha_q

    # Sort by strongest evidence (q then p), like you'd typically scan in R.
    t = t.sort_values(["q_value_fdr_bh", "p_value", "value_model", "knob"]).reset_index(
        drop=True
    )
    if max_rows is not None:
        t = t.head(max_rows)

    header = (
        f"Significance summary (alpha_p={alpha_p:g}, alpha_q={alpha_q:g}, BH-FDR on q):\n"
        "  Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 '' 1\n"
    )
    print(header)

    # Pretty fixed-width columns.
    cols = [
        ("value_model", 10),
        ("knob", 20),
        ("test", 12),
        ("k_groups", 7),
        ("n_total", 7),
        ("p_fmt", 12),
        ("p_sig", 4),
        ("q_fmt", 12),
        ("q_sig", 4),
        ("reject_p", 9),
        ("reject_q", 9),
        ("effect_size", 12),
    ]

    def _pad(s: str, w: int) -> str:
        s = str(s)
        if len(s) > w:
            return s[: w - 1] + "…"
        return s.ljust(w)

    print(
        " ".join(
            [
                _pad("value_model", 10),
                _pad("knob", 20),
                _pad("test", 12),
                _pad("k", 7),
                _pad("n", 7),
                _pad("p", 12),
                _pad("", 4),
                _pad("q", 12),
                _pad("", 4),
                _pad("reject_p", 9),
                _pad("reject_q", 9),
                _pad("effect", 12),
            ]
        )
    )
    print("-" * 115)

    for _, row in t.iterrows():
        effect = row.get("effect_size")
        effect_str = "NA" if pd.isna(effect) else f"{float(effect):.4g}"
        print(
            " ".join(
                [
                    _pad(row["value_model"], 10),
                    _pad(row["knob"], 20),
                    _pad(row["test"], 12),
                    _pad(int(row["k_groups"]), 7),
                    _pad(int(row["n_total"]), 7),
                    _pad(row["p_fmt"], 12),
                    _pad(row["p_sig"], 4),
                    _pad(row["q_fmt"], 12),
                    _pad(row["q_sig"], 4),
                    _pad("REJECT" if bool(row["reject_p"]) else "keep", 9),
                    _pad("REJECT" if bool(row["reject_q"]) else "keep", 9),
                    _pad(effect_str, 12),
                ]
            )
        )


def _prepare_mse_for_stats(
    series: pd.Series,
    *,
    winsor_q: tuple[float, float] | None = (0.01, 0.99),
    log10: bool = True,
) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").dropna().copy()
    s = s[s > 0]
    if winsor_q is not None:
        q_low, q_high = winsor_q
        lo = float(s.quantile(q_low))
        hi = float(s.quantile(q_high))
        s = s.clip(lower=lo, upper=hi)
    if log10:
        import numpy as np

        s = pd.Series(np.log10(s.to_numpy()), index=s.index)
    return s


def _significance_table_by_value_model(
    df: pd.DataFrame,
    *,
    knobs: list[str],
    value_model_col: str = "value_model",
    metric_col: str = "MSE",
    winsor_q: tuple[float, float] | None = (0.01, 0.99),
    log10: bool = True,
    min_group_n: int = 5,
    apply_bh_fdr: bool = True,
) -> pd.DataFrame:
    """Compute omnibus tests per (value_model, knob).

    Test choice:
    - 2 groups: Mann–Whitney U (nonparametric)
    - >=3 groups: Kruskal–Wallis H (nonparametric)

    Note: These tests assume independent samples. If you have a paired/blocking
    structure (same game instance evaluated under all knob values), a blocked
    test (e.g., Friedman) or a mixed-effects model can be more appropriate.
    """

    try:
        from scipy import stats  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "scipy is required for significance tests. Install with: pip install scipy"
        ) from e

    if value_model_col not in df.columns:
        raise ValueError(f"Missing '{value_model_col}' column")
    if metric_col not in df.columns:
        raise ValueError(f"Missing '{metric_col}' column")

    results: list[dict] = []

    for vm in sorted(df[value_model_col].dropna().astype(str).unique().tolist()):
        df_vm = df[df[value_model_col].astype(str) == vm]
        for knob in knobs:
            if knob not in df_vm.columns:
                continue

            sub = df_vm[[knob, metric_col]].copy()
            sub[knob] = (
                sub[knob].where(~sub[knob].isna(), other="<missing>").astype(str)
            )

            # Build groups.
            groups = []
            group_names = []
            for level, grp in sub.groupby(knob, sort=True):
                y = _prepare_mse_for_stats(
                    grp[metric_col], winsor_q=winsor_q, log10=log10
                )
                if len(y) >= min_group_n:
                    groups.append(y.to_numpy())
                    group_names.append(str(level))

            k = len(groups)
            n_total = int(sum(len(g) for g in groups))
            if k < 2:
                continue

            test_name = ""
            stat = float("nan")
            p = float("nan")
            effect = float("nan")

            if k == 2:
                test_name = "mannwhitneyu"
                # Use asymptotic method for speed; exact can be slow for big n.
                u_res = stats.mannwhitneyu(
                    groups[0], groups[1], alternative="two-sided"
                )
                stat = float(u_res.statistic)
                p = float(u_res.pvalue)
                # Rank-biserial correlation effect size in [-1, 1].
                n1, n2 = len(groups[0]), len(groups[1])
                u = stat
                effect = 1.0 - (2.0 * u) / (n1 * n2)
            else:
                test_name = "kruskal"
                h_res = stats.kruskal(*groups)
                stat = float(h_res.statistic)
                p = float(h_res.pvalue)
                # Epsilon-squared effect size (approx): (H - k + 1) / (n - k)
                denom = max(1.0, (n_total - k))
                effect = max(0.0, (stat - k + 1.0) / denom)

            results.append(
                {
                    "value_model": vm,
                    "knob": knob,
                    "test": test_name,
                    "k_groups": k,
                    "n_total": n_total,
                    "group_levels": "|".join(group_names),
                    "statistic": stat,
                    "p_value": p,
                    "effect_size": effect,
                    "metric": f"{'log10(' if log10 else ''}{metric_col}{')' if log10 else ''}",
                    "winsor_q": (
                        None if winsor_q is None else f"{winsor_q[0]}-{winsor_q[1]}"
                    ),
                    "min_group_n": min_group_n,
                }
            )

    out = pd.DataFrame(results)
    if len(out) == 0:
        return out
    if apply_bh_fdr:
        out["q_value_fdr_bh"] = _bh_fdr(out["p_value"].tolist())
    return out.sort_values(["value_model", "knob"]).reset_index(drop=True)


def _slugify(s: str) -> str:
    return "".join(ch if (ch.isalnum() or ch in "-_.") else "_" for ch in str(s))


def _safe_import_seaborn():
    try:
        import seaborn as sns  # type: ignore

        return sns
    except Exception:
        return None


def _plot_mse_boxplot_by_knob_with_value_model_hue(
    df: pd.DataFrame,
    *,
    column: str,
    output_dir: Path,
    title_prefix: str = "",
) -> None:
    """One figure per knob, with one axis per approximator.

    Each axis shows grouped boxplots with hue=value_model.
    """

    if column not in df.columns or "value_model" not in df.columns:
        return
    if "approximator" not in df.columns:
        raise ValueError("Expected column 'approximator' in the input data.")
    if "MSE" not in df.columns:
        raise ValueError("Expected column 'MSE' in the input data.")

    sns = _safe_import_seaborn()
    if sns is None:
        return

    import matplotlib.pyplot as plt
    from matplotlib.ticker import LogLocator

    plot_df = df[["approximator", "value_model", column, "MSE"]].copy()

    # Log-scale needs positive values.
    plot_df = plot_df[pd.to_numeric(plot_df["MSE"], errors="coerce") > 0].copy()

    # Cut off extreme outliers (keep 1%..99% quantiles).
    q_low, q_high = 0.01, 0.99
    low = float(plot_df["MSE"].quantile(q_low))
    high = float(plot_df["MSE"].quantile(q_high))
    plot_df = plot_df[(plot_df["MSE"] >= low) & (plot_df["MSE"] <= high)].copy()

    plot_df["value_model"] = plot_df["value_model"].where(
        ~plot_df["value_model"].isna(), other="<missing>"
    )
    plot_df["value_model"] = plot_df["value_model"].astype(str)

    plot_df["approximator"] = plot_df["approximator"].where(
        ~plot_df["approximator"].isna(), other="<missing>"
    )
    plot_df["approximator"] = plot_df["approximator"].astype(str)

    plot_df[column] = plot_df[column].where(~plot_df[column].isna(), other="<missing>")
    plot_df[column] = plot_df[column].astype(str)

    # Consistent x-category ordering.
    order = (
        plot_df.groupby(column)["MSE"]
        .median()
        .sort_values(ascending=True)
        .index.tolist()
    )

    # Stable hue ordering.
    hue_order = sorted(plot_df["value_model"].unique().tolist())

    # Softer colors for different value_models.
    hue_palette = dict(
        zip(hue_order, sns.color_palette("pastel", n_colors=len(hue_order)))
    )

    # Stable approximator ordering.
    approx_order = sorted(plot_df["approximator"].unique().tolist())

    # Some approximators do not use all knob values. For each subplot, we
    # restrict the x-axis categories to only those present for that approximator.
    order_by_approx: dict[str, list[str]] = {}
    for approx in approx_order:
        present = set(plot_df.loc[plot_df["approximator"] == approx, column].tolist())
        order_by_approx[approx] = [lvl for lvl in order if lvl in present]

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"boxplot_MSE_by_{column}__faceted_by_approximator.png"

    # Match shapiq_benchmark.plot.plot_approximation_quality styling:
    # no horizontal y-gridlines, only dashed x-gridlines.
    sns.set_theme(style="ticks", context="talk")

    max_order_len = max((len(v) for v in order_by_approx.values()), default=len(order))
    per_ax_w = min(14, max(9.5, 0.72 * max_order_len))
    fig_w = min(36, max(12, per_ax_w * len(approx_order)))
    fig_h = 7.2
    fig, axes = plt.subplots(
        1,
        len(approx_order),
        figsize=(fig_w, fig_h),
        sharey=True,
        squeeze=False,
    )
    axes = axes[0]

    legend_handles = None
    legend_labels = None

    for i, approx in enumerate(approx_order):
        ax = axes[i]
        sub_df = plot_df[plot_df["approximator"] == approx]
        sub_order = order_by_approx.get(approx, [])
        if len(sub_df) == 0 or len(sub_order) == 0:
            ax.set_title(approx)
            ax.set_axis_off()
            ax.text(
                0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes
            )
            continue
        sns.boxplot(
            data=sub_df,
            x=column,
            y="MSE",
            hue="value_model",
            order=sub_order,
            hue_order=hue_order,
            palette=hue_palette,
            showfliers=False,
            linewidth=1.1,
            width=0.75,
            dodge=True,
            ax=ax,
        )

        ax.set_yscale("log")
        ax.yaxis.set_major_locator(LogLocator(base=10))
        ax.grid(False)
        ax.grid(axis="x", color=LIGHT_GRAY, linestyle="dashed")
        ax.tick_params(axis="y", which="both", direction="out", length=6, width=1)
        ax.tick_params(axis="x", rotation=35, labelsize=10)

        ax.set_title(approx)
        ax.set_xlabel(column)
        ax.set_ylabel("MSE (log scale)" if i == 0 else "")

        handles, labels = ax.get_legend_handles_labels()
        if legend_handles is None:
            legend_handles, legend_labels = handles, labels
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()

        sns.despine(ax=ax, left=False, bottom=False)

    if legend_handles is not None and legend_labels is not None:
        fig.legend(
            legend_handles,
            legend_labels,
            title="value_model",
            bbox_to_anchor=(1.01, 1.0),
            loc="upper left",
            borderaxespad=0.0,
        )

    fig.suptitle(f"{title_prefix}MSE by {column} (hue=value_model)".strip())
    fig.tight_layout(rect=(0, 0, 0.88, 0.95))
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    files = sorted(glob.glob("abalation/find_optimal_model_results.shard*-of-*.csv"))
    if not files:
        # Allow running the plotting step after you've already merged.
        merged_path = Path("find_optimal_model_results_merged.csv")
        if not merged_path.exists():
            raise FileNotFoundError(
                "No shard files found in 'abalation/' and no merged CSV present at "
                f"{merged_path}."
            )
        df = pd.read_csv(merged_path)
        print("loaded merged CSV ->", len(df), "rows")
    else:
        df = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
        df.to_csv("find_optimal_model_results_merged.csv", index=False)
        print("merged", len(files), "files ->", len(df), "rows")

    # Basic sanity: MSE must be numeric.
    df["MSE"] = pd.to_numeric(df["MSE"], errors="coerce")
    df = df.dropna(subset=["MSE"]).copy()

    if "approximator" not in df.columns:
        raise ValueError("Expected column 'approximator' in the merged results CSV.")

    out_dir = Path("plots/abalation/mse_boxplots/")
    # Boxplots: one figure per knob, faceted by approximator.
    per_model_cols = [
        "sampling_weight",
        "residual_approximator",
        "max_depth",
        "n_estimators",
    ]
    for col in per_model_cols:
        _plot_mse_boxplot_by_knob_with_value_model_hue(
            df, column=col, output_dir=out_dir
        )

    # Significance table: for each value_model × knob, test if all knob levels
    # share the same MSE distribution.
    stats_knobs = [
        "sampling_weight",
        "residual_approximator",
        "max_depth",
        "n_estimators",
    ]
    stats_out_dir = Path("plots/abalation/mse_stats/by_approximator")
    stats_out_dir.mkdir(parents=True, exist_ok=True)

    for approx in sorted(df["approximator"].dropna().astype(str).unique().tolist()):
        df_a = df[df["approximator"].astype(str) == approx].copy()
        table = _significance_table_by_value_model(df_a, knobs=stats_knobs)
        table.insert(0, "approximator", approx)
        table_path = (
            stats_out_dir / f"significance__approximator_{_slugify(approx)}.csv"
        )
        table.to_csv(table_path, index=False)
        print(f"wrote significance table to {table_path} ({len(table)} rows)")

        print(f"\n=== Approximator: {approx} ===")
        _print_significance_summary(table, alpha_p=0.05, alpha_q=0.05)

    print(f"wrote boxplots to {out_dir}")
