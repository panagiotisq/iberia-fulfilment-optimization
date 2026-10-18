"""Shared matplotlib style for report figures (static, light background, print-friendly).

Categorical hues are assigned to *entities* in a fixed order (never cycled by rank), so a
method keeps its colour across every figure of the report.
"""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3df"

CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
NEUTRAL = "#9a9890"

# fixed entity -> colour mapping used across all figures
METHOD_COLOR = {
    "oracle": CAT[2], "sp_oracle": CAT[2],
    "lgbm_cqr": CAT[0], "sp_lgbm_cop": CAT[0], "lgbm": CAT[0],
    "sp_lgbm_indep": CAT[6],
    "snaive_cqr": CAT[1], "sp_snaive": CAT[1], "snaive": CAT[1],
    "chronos_small_cqr": CAT[4], "chronos_base_cqr": CAT[4], "chronos_small": CAT[4], "chronos_base": CAT[4],
    "det_point": CAT[3], "det_p90": CAT[7],
    "greedy": CAT[1], "bid_static": CAT[3], "bid_resolve": CAT[0], "lp_tracking": CAT[2], "hindsight": NEUTRAL,
}

LABEL = {
    "det_point": "Deterministic (median)", "det_p90": "Deterministic (P90)",
    "sp_snaive": "SP + seasonal naive", "sp_lgbm_indep": "SP + LightGBM (indep.)",
    "sp_lgbm_cop": "SP + LightGBM (copula)", "sp_oracle": "SP + true distribution",
    "greedy": "Greedy", "bid_static": "Bid price (static)", "bid_resolve": "Bid price (re-solve)",
    "lp_tracking": "Fluid LP tracking", "hindsight": "Hindsight LP",
    "snaive": "Seasonal naive", "snaive_cqr": "Seasonal naive + CP", "lgbm": "LightGBM-Q",
    "lgbm_cqr": "LightGBM-Q + CP", "oracle": "Oracle", "chronos_small": "Chronos-Bolt-S",
    "chronos_small_cqr": "Chronos-Bolt-S + CP", "chronos_base": "Chronos-Bolt-B",
    "chronos_base_cqr": "Chronos-Bolt-B + CP",
}


def setup():
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": "DejaVu Sans", "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
        "axes.edgecolor": INK2, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": INK2, "ytick.color": INK2, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
        "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "axes.axisbelow": True,
        "lines.linewidth": 1.6, "lines.markersize": 4.5, "legend.frameon": False, "legend.fontsize": 7.5,
        "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight", "pdf.fonttype": 42,
    })


def save(fig, name):
    from .config import FIGURES
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{name}.pdf")
    fig.savefig(FIGURES / f"{name}.png", dpi=160)
    plt.close(fig)
