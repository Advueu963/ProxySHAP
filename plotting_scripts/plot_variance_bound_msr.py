"""Plot empirical SHAPIQ MSR variance vs. theoretical bound from Theorem `var_msr`."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

try:
    from special_plot_scripts._plot_style import (
        apply_tick_style,
        setup_fonts,
        W_REGULAR,
        W_SEMIBOLD,
    )
except ModuleNotFoundError:
    from _plot_style import apply_tick_style, setup_fonts, W_REGULAR, W_SEMIBOLD

setup_fonts()

PANEL_FIGSIZE = (5, 4)
TITLE_FONT_SIZE = 11
LABEL_FONT_SIZE = 10
TICK_LABEL_FONT_SIZE = 12
LEGEND_FONT_SIZE = 8
SUPTITLE_FONT_SIZE = 12


def bound_shape(size: int, n: int, budgets: np.ndarray, v_inf: float) -> np.ndarray:
    """Asymptotic bound shape (without the unknown constant): ||v||_inf^2 * f(n, |S|) / |T|."""
    if size == 1:
        prefactor = np.log(n)
    else:
        prefactor = float(n ** (size - 1))
    return (v_inf**2) * prefactor / budgets.astype(float)


def plot(cache: Path, out_path: Path) -> None:
    data = np.load(cache, allow_pickle=False)
    phi = data["phi"]
    sizes = data["sizes"]
    budgets = data["budgets"]
    n = int(data["n"])
    v_inf = float(data["v_inf"])
    max_order = int(data["max_order"])
    print(
        f"[load] phi {phi.shape} | n={n} | v_inf={v_inf:.4f} | budgets={budgets.tolist()}"
    )

    var = phi.var(axis=0, ddof=1)

    fig_width = PANEL_FIGSIZE[0] * max_order
    fig_height = PANEL_FIGSIZE[1]
    fig, axes = plt.subplots(
        2,
        max_order,
        figsize=(fig_width, 2 * fig_height),
        squeeze=False,
    )

    axes_main = axes[0]
    axes_ratio = axes[1]

    colors = plt.get_cmap("tab10")

    for k in range(1, max_order + 1):
        ax = axes_main[k - 1]
        rax = axes_ratio[k - 1]
        mask = sizes == k
        if not mask.any():
            axes_main[k - 1].set_visible(False)
            axes_ratio[k - 1].set_visible(False)
            continue
        var_k = var[:, mask]
        mean_var = var_k.mean(axis=1)
        max_var = var_k.max(axis=1)
        min_var = var_k.min(axis=1)

        shape = bound_shape(k, n, budgets, v_inf)
        eps = 1e-300
        ratio_mean = mean_var / np.maximum(shape, eps)
        ratio_max = max_var / np.maximum(shape, eps)
        ratio_min = min_var / np.maximum(shape, eps)
        c_fit = np.exp(np.mean(np.log(max_var) - np.log(shape)))
        bound = c_fit * shape

        color = colors(k - 1)
        ax.loglog(
            budgets,
            mean_var,
            "o-",
            color=color,
            linewidth=2,
            markersize=5,
            label=f"empirical Var (mean over |S|={k})",
        )
        ax.loglog(
            budgets,
            max_var,
            "v--",
            color=color,
            alpha=0.4,
            linewidth=2,
            markersize=5,
            label=f"empirical Var (max over |S|={k})",
        )
        ax.loglog(
            budgets,
            min_var,
            "^--",
            color=color,
            alpha=0.4,
            linewidth=2,
            markersize=5,
            label=f"empirical Var (min over |S|={k})",
        )

        if k == 1:
            label = rf"bound $\propto \|v\|_\infty^2 \log n / |T|$"
        else:
            label = rf"bound $\propto \|v\|_\infty^2 \, n^{{{k - 1}}} / |T|$"
        ax.loglog(budgets, bound, "k-", lw=1.5, label=label)

        ax.set_xlabel("budget |T|", fontsize=LABEL_FONT_SIZE, fontweight=W_SEMIBOLD)
        ax.set_ylabel("Var", fontsize=LABEL_FONT_SIZE, fontweight=W_SEMIBOLD)
        ax.set_title(
            f"|S| = {k}   (fitted C = {c_fit:.3g})",
            fontsize=TITLE_FONT_SIZE,
            fontweight=W_REGULAR,
        )
        ax.grid(True, which="both", linestyle="dotted", alpha=0.5)
        ax.legend(
            fontsize=LEGEND_FONT_SIZE, loc="upper right", frameon=True, framealpha=0.9
        )

        rax.semilogx(
            budgets,
            ratio_mean,
            "o-",
            color=color,
            linewidth=2,
            markersize=5,
            label="mean / bound",
        )

        rax.semilogx(
            budgets,
            ratio_max,
            "v--",
            color=color,
            alpha=0.4,
            linewidth=2,
            markersize=5,
            label="max / bound",
        )

        rax.semilogx(
            budgets,
            ratio_min,
            "^--",
            color=color,
            alpha=0.4,
            linewidth=2,
            markersize=5,
            label="min / bound",
        )

        rax.axhline(
            c_fit, color="black", linestyle="-", linewidth=1.5, label="fitted C"
        )

        rax.set_xlabel("budget |T|", fontsize=LABEL_FONT_SIZE, fontweight=W_SEMIBOLD)
        rax.set_ylabel("Var / bound", fontsize=LABEL_FONT_SIZE, fontweight=W_SEMIBOLD)
        rax.grid(True, which="both", linestyle="dotted", alpha=0.5)
        rax.legend(
            fontsize=LEGEND_FONT_SIZE, loc="upper right", frameon=True, framealpha=0.9
        )

    # fig.suptitle(

    #     fontsize=SUPTITLE_FONT_SIZE,
    #     fontweight=W_SEMIBOLD,
    # )
    fig.tight_layout()

    apply_tick_style(*list(axes_main), tick_label_fontsize=TICK_LABEL_FONT_SIZE)
    apply_tick_style(*list(axes_ratio), tick_label_fontsize=TICK_LABEL_FONT_SIZE)
    
    fig.patch.set_facecolor("none")
    fig.patch.set_alpha(0)
    for ax in list(axes_main) + list(axes_ratio):
        ax.patch.set_facecolor("none")
        ax.patch.set_alpha(0)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        out_path,
        bbox_inches="tight",
        dpi=300,
        transparent=True,
        facecolor="none",
        edgecolor="none",
    )
    print(f"[done] wrote {out_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--cache",
        type=Path,
        default=Path(__file__).parents[1]
        / "experiments"
        / "cache"
        / "variance_bound_msr_independent60.npz",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).parents[1]
        / "plots"
        / "variance_bound_msr"
        / "variance_bound_mse.pdf",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    plot(cache=args.cache, out_path=args.out)
