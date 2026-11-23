"""Common style of every figure of the PQ-PKS paper, following the graphics guidelines of the IEEE Author Center.

Rules (https://conferences.ieeeauthorcenter.ieee.org/write-your-paper/improve-your-graphics/):
  * width 3.5 in (one column) or 7.16 in (two columns), height at most 8.8 in;
  * Times New Roman font (fallback: Liberation Serif, same metrics; STIX for mathematics), text about 8-9 pt at print size;
  * readable in black and white: each system has its own colour, marker, line style and hatch; Okabe-Ito palette
    (safe for red-green colour blindness);
  * no title inside the figure (the content is described in the caption); panel labels (a), (b) in the upper-left corner;
  * vector PDF with embedded fonts (TrueType, fonttype 42) for submission; PNG at 600 dpi for preview.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ONE_COL, TWO_COL = 3.5, 7.16           # inch

# Okabe-Ito. Fixed order by system: PQ-PKS always dark blue, REF-U vermillion, BL5 green, BL5-L orange.
COL = {"PQ-PKS": "#0072B2", "REF-U": "#D55E00", "BL5": "#009E73", "BL5-L": "#E69F00"}
MRK = {"PQ-PKS": "o", "REF-U": "s", "BL5": "^", "BL5-L": "D"}
LS = {"PQ-PKS": "-", "REF-U": "--", "BL5": "-.", "BL5-L": ":"}
HATCH = {"PQ-PKS": "", "REF-U": "////", "BL5": "....", "BL5-L": "xxxx"}
INK = "#000000"
INK2 = "#4d4d4d"
GRID = "#d9d9d9"          # used only for shaded bands (equivalence region, G3 margin); no grid lines are drawn
SURF = "#ffffff"

RC = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Liberation Serif", "Nimbus Roman", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.labelsize": 8.5,
    "axes.titlesize": 8.5,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7.5,
    "axes.linewidth": 0.6,
    "axes.edgecolor": INK,
    "axes.labelcolor": INK,
    "axes.facecolor": SURF,
    "figure.facecolor": SURF,
    "savefig.facecolor": SURF,
    "axes.grid": False,
    "axes.spines.top": True,
    "axes.spines.right": True,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "xtick.minor.size": 1.5,
    "ytick.minor.size": 1.5,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.color": INK,
    "ytick.color": INK,
    "lines.linewidth": 1.1,
    "lines.markersize": 3.6,
    "errorbar.capsize": 2,
    "legend.frameon": True,
    "legend.framealpha": 1.0,
    "legend.edgecolor": "#000000",
    "legend.fancybox": False,
    "legend.borderpad": 0.3,
    "legend.handlelength": 2.2,
    "patch.linewidth": 0.5,
    "hatch.linewidth": 0.5,
    "hatch.color": INK,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "savefig.dpi": 600,
}
plt.rcParams.update(RC)


def style(ax):
    """Kept for compatibility with older code: rcParams already set the style; this function only ensures a four-sided frame without grid."""
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.6)
        s.set_color(INK)
    ax.grid(False)
    ax.tick_params(which="both", direction="in", top=True, right=True)


def panel(ax, label: str, x: float = -0.02, y: float = 1.02):
    """Panel labels (a), (b), ... in the upper-left corner, outside the data area."""
    ax.text(x, y, label, transform=ax.transAxes, ha="right", va="bottom", fontsize=9)


def save(fig, out: Path, name: str):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out / f"{name}.png", dpi=600, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
