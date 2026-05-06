"""Shared Fira Sans font styling for all project plotting scripts."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
from matplotlib import font_manager

FONT = "Fira Sans"
W_THIN = 300        # Ticks
W_REGULAR = "normal"   # Titles
W_SEMIBOLD = "semibold"  # Axis labels & stickers
W_BOLD = "bold"      # Legend

_FONT_DIR = Path.home() / "Library" / "Fonts"
_FIRA_SANS_FILES = [
    "FiraSans-Thin.ttf",
    "FiraSans-ThinItalic.ttf",
    "FiraSans-ExtraLight.ttf",
    "FiraSans-ExtraLightItalic.ttf",
    "FiraSans-Light.ttf",
    "FiraSans-LightItalic.ttf",
    "FiraSans-Regular.ttf",
    "FiraSans-Italic.ttf",
    "FiraSans-Medium.ttf",
    "FiraSans-MediumItalic.ttf",
    "FiraSans-SemiBold.ttf",
    "FiraSans-SemiBoldItalic.ttf",
    "FiraSans-Bold.ttf",
    "FiraSans-BoldItalic.ttf",
    "FiraSans-ExtraBold.ttf",
    "FiraSans-ExtraBoldItalic.ttf",
    "FiraSans-Black.ttf",
    "FiraSans-BlackItalic.ttf",
]

for _fname in _FIRA_SANS_FILES:
    _fpath = _FONT_DIR / _fname
    if _fpath.exists():
        font_manager.fontManager.addfont(str(_fpath))


def setup_fonts() -> None:
    """Set Fira Sans as the global matplotlib font family."""
    mpl.rcParams["font.family"] = FONT


def apply_tick_style(*axes, tick_label_fontsize: float | None = None) -> None:
    """Apply Fira Sans Thin to all tick labels.

    Args:
        *axes: Matplotlib axes whose tick labels should be styled.
        tick_label_fontsize: Optional font size applied to both x- and y-tick labels.

    Call just before fig.savefig().
    """
    for ax in axes:
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontfamily(FONT)
            label.set_fontweight(W_THIN)
            if tick_label_fontsize is not None:
                label.set_fontsize(tick_label_fontsize)
    