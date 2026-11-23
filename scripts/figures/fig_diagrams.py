"""Diagrams of Figs. 1, 3 and 4 of the paper, drawn with matplotlib in IEEE style.

    python scripts/figures/fig_diagrams.py --out results/figures

Reads no data. Coordinates are in inches on a canvas of exactly the figure size, so the text keeps its size in print.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon

from ieee_style import COL, INK, SURF, TWO_COL, plt, save

TINT = "#e8f1f8"      # light background of the PQ-PKS blocks
GREY = "#f2f2f2"      # light background of the warden area / the layers
FS = 7.5


def canvas(w, h):
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.axis("off")
    return fig, ax


def box(ax, cx, cy, w, h, text, fc=SURF, ec=INK, lw=0.6, fs=FS, ls="-", rounded=True, z=3):
    style = "round,pad=0,rounding_size=0.04" if rounded else "square,pad=0"
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h, boxstyle=style, fc=fc, ec=ec, lw=lw, ls=ls, zorder=z))
    if text:
        ax.text(cx, cy, text, ha="center", va="center", fontsize=fs, zorder=z + 1, linespacing=1.15)


def region(ax, x0, y0, x1, y1, title=None, fc=GREY, ls="-", title_y=None):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle="square,pad=0", fc=fc, ec=INK, lw=0.5, ls=ls, zorder=1))
    if title:
        ax.text((x0 + x1) / 2, title_y if title_y is not None else y1 - 0.13, title, ha="center", va="center", fontsize=FS, zorder=4)


def diamond(ax, cx, cy, w, h, text, fs=FS):
    ax.add_patch(Polygon([(cx - w / 2, cy), (cx, cy + h / 2), (cx + w / 2, cy), (cx, cy - h / 2)], closed=True,
                         fc=SURF, ec=INK, lw=0.6, zorder=3))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs, zorder=4, linespacing=1.1)


def arrow(ax, pts, label=None, lpos=None, ha="left", va="bottom", fs=7):
    """Polyline arrow through the points pts; the arrow head is at the last point."""
    for (x0, y0), (x1, y1) in zip(pts[:-2], pts[1:-1]):
        ax.plot([x0, x1], [y0, y1], color=INK, lw=0.6, zorder=2, solid_capstyle="butt")
    ax.add_patch(FancyArrowPatch(pts[-2], pts[-1], arrowstyle="-|>", mutation_scale=7, lw=0.6, color=INK, zorder=2,
                                 shrinkA=0, shrinkB=0))
    if label:
        x, y = lpos if lpos else ((pts[-2][0] + pts[-1][0]) / 2, (pts[-2][1] + pts[-1][1]) / 2)
        ax.text(x, y, label, ha=ha, va=va, fontsize=fs, zorder=5, style="italic")


# ------------------------------------------------------------------------------------------------ Fig. 1
def fig1(out):
    W, H = TWO_COL, 1.95
    fig, ax = canvas(W, H)
    region(ax, 4.98, 0.08, 7.12, 1.87, None, fc=GREY, ls=(0, (4, 2)))
    ax.text(6.05, 1.76, r"warden (knows $ek_R$ and all public parameters)", ha="center", va="center", fontsize=7)
    rows = [(1.28, "Crypto-then-embed\n(e.g., BL1, BL5)", "IND-CCA encryption,\nraw serialization", "embedding",
             "structured string:\nLLR or curve test\ndetects it", SURF),
            (0.48, "PQ-PKS", "ML-KEM-768 + Kemeleon,\none-time AEAD", "density-matched\ntwo-stage STC",
             "pseudorandom string,\nlike a cover\nsyndrome", TINT)]
    for y, lab, enc, emb, res, fc in rows:
        ax.text(0.06, y, lab, ha="left", va="center", fontsize=FS, linespacing=1.15)
        box(ax, 1.42, y, 0.3, 0.3, r"$m$")
        box(ax, 2.33, y, 1.32, 0.5, enc, fc=fc)
        box(ax, 3.62, y, 0.98, 0.5, emb, fc=fc)
        box(ax, 4.55, y, 0.62, 0.5, "stego\nimage")
        box(ax, 5.55, y, 0.82, 0.5, "public\nextractor " + r"$\mathrm{Ext}_0$")
        box(ax, 6.6, y, 0.92, 0.58, res, fc=SURF, fs=7)
        for x0, x1 in ((1.57, 1.67), (2.99, 3.13), (4.11, 4.24), (4.86, 5.14), (5.96, 6.14)):
            arrow(ax, [(x0, y), (x1, y)])
    save(fig, out, "fig1_overview")


# ------------------------------------------------------------------------------------------------ Fig. 3
def fig3(out):
    W, H = TWO_COL, 2.6
    fig, ax = canvas(W, H)
    top, bh, step = 2.17, 0.34, 0.45
    cols = [
        (0.06, 2.36, "Layer 1: cryptographic (sender)", [
            "HPKE.SetupBaseS($ek_R$), KEM = ML-KEM-768",
            r"Kemeleon encoding: $\hat c$",
            r"HKDF key schedule: Export $\to k^{\mathrm{ae}}_i,k^{\pi}_i,k^{H}_i$",
            "ML-DSA-65 signature (optional)",
            r"one-time AEAD: ChaCha20-Poly1305 under $k^{\mathrm{ae}}_i$"], TINT),
        (2.62, 4.5, "Layer 2: steganographic (sender)", [
            "HILL costs",
            r"public partition: $A_0(B)$, $A_1(B)$",
            r"STC stage 1: $\hat c\to A_0$, public $\mathbf{H}_0$",
            r"STC stage 2: body $\to\pi_K(A_1)$, keyed $\mathbf{H}_K$",
            r"stego image $Y$"], TINT),
        (4.76, 7.1, "Layer 3: receiver", [
            r"public extraction of $\hat c'$ for each bucket",
            "Kemeleon decoding, ML-KEM Decaps",
            "key schedule, keyed extraction",
            "AEAD decryption",
            r"policy ($\sigma$, $t_0$), then commit: $m$ or $\perp$"], SURF),
    ]
    for x0, x1, title, items, fc in cols:
        region(ax, x0, 0.12, x1, H - 0.06, title, fc=GREY, title_y=H - 0.2)
        cx, w = (x0 + x1) / 2, (x1 - x0) - 0.16
        for k, t in enumerate(items):
            y = top - k * step
            box(ax, cx, y, w, bh, t, fc=fc, fs=7)
            if k:
                arrow(ax, [(cx, y + step - bh / 2), (cx, y + bh / 2)])
    yb = top - 4 * step
    arrow(ax, [(2.36 - 0.08, top - 2 * step), (2.62 + 0.08, top - 2 * step)])
    ax.text(2.49, top - 2 * step + 0.06, r"$\hat c$, body", ha="center", va="bottom", fontsize=6.5)
    arrow(ax, [(4.5 - 0.08, yb), (4.63, yb), (4.63, top), (4.76 + 0.08, top)])
    ax.text(4.63, (yb + top) / 2, "lossless channel", ha="center", va="center", fontsize=6.5, style="italic",
            rotation=90, bbox=dict(fc=SURF, ec="none", pad=0.6), zorder=6)
    save(fig, out, "fig3_architecture")


# ------------------------------------------------------------------------------------------------ Fig. 4
def fig4(out):
    W, H = TWO_COL, 4.25
    fig, ax = canvas(W, H)
    ax.text(0.06, H - 0.12, "(a) sender", ha="left", va="center", fontsize=8.5)
    ax.text(3.78, H - 0.12, "(b) receiver", ha="left", va="center", fontsize=8.5)
    steps = [
        r"HPKE.SetupBaseS($ek_R$): enc (1088 B), context",
        r"Kemeleon.EncodeCt(enc): $\hat c$ (1536 B)",
        r"$sid=\mathrm{SHA3}$-$256(pid\,\|\,\hat c)$; Export $\to k^{\mathrm{ae}}_i,k^{\pi}_i,k^{H}_i$",
        r"frame $pt$, pad to bucket $B$; sign transcript if required",
        r"$body=$ ChaCha20-Poly1305($k^{\mathrm{ae}}_i$, nonce $0^{96}$, $ad_i$, $pt$)",
        r"HILL costs; $A_0(B)$ public, $A_1(B)$ keyed (layout A8)",
        r"STC: $\hat c\to A_0$ with $\mathbf{H}_0$; body $\to\pi_K(A_1)$ with $\mathbf{H}_K$",
        r"stego image $Y$ (on STC failure: new cover, same body)",
    ]
    cx, w, bh, top, step = 1.78, 3.38, 0.32, 3.82, 0.48
    for k, t in enumerate(steps):
        y = top - k * step
        box(ax, cx, y, w, bh, t, fc=TINT if k in (1, 6) else SURF, fs=7)
        if k:
            arrow(ax, [(cx, y + step - bh / 2), (cx, y + bh / 2)])
    ax.plot([3.62, 3.62], [0.1, H - 0.05], color=INK, lw=0.4, ls=(0, (3, 3)))

    m, r = 5.0, 6.55               # main column, right column
    y1, y2, y3, y4, y5, y6 = 3.82, 3.2, 2.58, 1.95, 1.3, 0.55
    box(ax, m, y1, 2.1, 0.36, r"for each $B\in\mathcal{B}_1$: $\hat c'_B=\mathbf{H}_0\cdot\mathrm{LSB}(y[A_0(B)])$, $sid'_B$", fs=6.8)
    diamond(ax, m, y2, 1.55, 0.46, r"$sid'_B$ already seen?", fs=6.8)
    box(ax, m, y3, 2.1, 0.36, "decode, Decaps (implicit rejection),\n" + r"Export($sid'_B$, 1)", fs=6.8)
    diamond(ax, m, y4, 1.55, 0.46, "AEAD opens\nfor some $B$?", fs=6.8)
    diamond(ax, m, y5, 1.55, 0.46, "canonical frame,\nfresh $t_0$, valid $\\sigma$?", fs=6.5)
    box(ax, m, y6, 1.5, 0.34, r"commit session, output $m$", fc=TINT, fs=6.8)
    box(ax, r, y2, 0.55, 0.3, r"$\perp$", fs=8)
    box(ax, r, y4, 1.02, 0.5, "open sessions:\nunseen $j$ in window,\nall buckets", fs=6.5)
    box(ax, 3.98, y5, 0.3, 0.3, r"$\perp$", fs=8)
    arrow(ax, [(m, y1 - 0.18), (m, y2 + 0.23)])
    arrow(ax, [(m + 0.775, y2), (r - 0.275, y2)], "yes", (m + 0.85, y2 + 0.03))
    arrow(ax, [(m, y2 - 0.23), (m, y3 + 0.18)], "no", (m + 0.05, y2 - 0.33))
    arrow(ax, [(m, y3 - 0.18), (m, y4 + 0.23)])
    arrow(ax, [(m + 0.775, y4), (r - 0.51, y4)], "no", (m + 0.85, y4 + 0.03))
    arrow(ax, [(m, y4 - 0.23), (m, y5 + 0.23)], "yes", (m + 0.05, y4 - 0.36))
    arrow(ax, [(m, y5 - 0.23), (m, y6 + 0.17)], "yes", (m + 0.05, y5 - 0.36))
    arrow(ax, [(m - 0.775, y5), (3.98 + 0.15, y5)], "no", (4.2, y5 + 0.04), ha="left")
    arrow(ax, [(r, y4 + 0.25), (r, y2 - 0.15)], "none", (r + 0.05, (y2 + y4) / 2), va="center")
    arrow(ax, [(r, y4 - 0.25), (r, y6), (m + 0.75, y6)], "found", (r + 0.05, (y4 + y6) / 2 - 0.1), va="center")
    save(fig, out, "fig4_flows")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[2] / "results" / "figures"))
    a = ap.parse_args()
    out = Path(a.out)
    fig1(out)
    fig3(out)
    fig4(out)
    print("diagrams in", out)


if __name__ == "__main__":
    main()
