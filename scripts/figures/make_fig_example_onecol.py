"""One-column version of the figure "One cover, end to end" (fig5_one_cover), redrawn from the stored data file.

    python scripts/figures/make_fig_example_onecol.py [--data results/figures/fig5_one_cover_data.npz] [--out results/figures]

No new embedding and no run result: only fig5_one_cover_data.npz written by make_fig_example.py is read, so every number and
every pixel equals the two-column version. Only the layout differs: (a)-(c) on one row, (d)-(f) on a second row, (g) horizontal
below, as wide as both image rows (x axis Lambda, y axis number of strings).
Output: fig5_one_cover_1col.pdf/.png.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from ieee_style import COL, INK, INK2, ONE_COL, panel, plt, save, style  # noqa: E402

SYS3 = ("BL5", "PQ-PKS", "REF-U")
CROP = (384, 0, 128)          # as in make_fig_example.py
FS = 6.6                      # font size of the panel labels (three panels per row)


def draw(d, out: Path):
    x = d["cover"]
    fig = plt.figure(figsize=(ONE_COL, 4.35))
    gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 1.02], wspace=0.06, hspace=0.62,
                          left=0.085, right=0.985, top=0.975, bottom=0.085)
    axs = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(3)]
    axg = fig.add_subplot(gs[2, :])
    for ax in axs:
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_linewidth(0.5)
    # (a) cover
    axs[0].imshow(x, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    panel(axs[0], "(a)", x=0.0, y=1.01)
    axs[0].set_xlabel(f"cover\n(BOSSbase {int(d['cover_id'])})", fontsize=FS, labelpad=2)
    # (b) HILL cost
    rho = np.asarray(d["rho"], dtype=np.float64)
    fin = np.isfinite(rho)
    lr = np.log10(np.clip(rho, 1e-3, None))
    lr[~fin] = np.nanmax(lr[fin])
    lo, hi = np.percentile(lr[fin], [1, 97])
    axs[1].imshow(np.clip(lr, lo, hi), cmap="gray_r", interpolation="nearest")
    panel(axs[1], "(b)", x=0.0, y=1.01)
    axs[1].set_xlabel("HILL cost $\\rho$\n($\\log_{10}$; dark = costly)", fontsize=FS, labelpad=2)
    # (c) PQ-PKS changes by stage in the crop
    yp = d["PQ-PKS_stego"]
    ch = yp != x
    a0 = d["PQ-PKS_a0"].astype(bool)
    r0_, c0_, L = CROP
    sl = (slice(r0_, r0_ + L), slice(c0_, c0_ + L))
    rgb = np.ones((L, L, 3))
    rgb[(ch & ~a0)[sl]] = (0.0, 0.447, 0.698)
    rgb[(ch & a0)[sl]] = (0.835, 0.369, 0.0)
    axs[2].imshow(rgb, interpolation="nearest")
    panel(axs[2], "(c)", x=0.0, y=1.01)
    axs[2].set_xlabel(f"PQ-PKS changes ({L}×{L});\n$A_0$ orange, $A_1$ blue", fontsize=FS, labelpad=2)
    axs[0].add_patch(plt.Rectangle((c0_, r0_), L, L, fill=False, ec="#E69F00", lw=0.8))
    # (d)-(f) change maps
    for k, name in enumerate(SYS3):
        chm = d[f"{name}_stego"] != x
        axs[3 + k].imshow(~chm, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        panel(axs[3 + k], f"({'def'[k]})", x=0.0, y=1.01)
        axs[3 + k].set_xlabel(f"{name}\n{chm.mean() * 100:.2f}% changed", fontsize=FS, labelpad=2)
    # (g) LLR of the public string, horizontal
    ax = axg
    style(ax)
    raw, uni = d["llr_raw"], d["llr_uni"]
    bins = np.linspace(min(raw.min(), uni.min()), max(raw.max(), uni.max()), 40)
    ax.hist(uni, bins=bins, histtype="stepfilled", color=COL["REF-U"], alpha=0.22, lw=0)
    ax.hist(uni, bins=bins, histtype="step", color=COL["REF-U"], lw=0.8, ls="--")
    ax.hist(raw, bins=bins, histtype="stepfilled", color=COL["BL5"], alpha=0.22, lw=0, hatch="....")
    ax.hist(raw, bins=bins, histtype="step", color=COL["BL5"], lw=0.8)
    ax.axvline(0, color=INK, lw=0.6, ls=":")
    ymax = ax.get_ylim()[1]
    ax.set_ylim(0, ymax * 1.92)
    # (label, Lambda, colour, marker, relative height, label above/below): staggered so that the labels do not overlap
    marks = [("cover syndrome", float(d["PQ-PKS_llr_cover"]), INK2, "x", 1.22, "below"),
             ("BL5", float(d["BL5_llr"]), COL["BL5"], "^", 1.36, "above"),
             ("PQ-PKS", float(d["PQ-PKS_llr"]), COL["PQ-PKS"], "o", 1.36, "above"),
             ("REF-U", float(d["REF-U_llr"]), COL["REF-U"], "s", 1.50, "above")]
    for lab, v, c, m, h, pos in marks:
        ax.plot([v], [ymax * h], marker=m, color=c, ms=4.5, mec=INK, mew=0.4, ls="none", zorder=5)
        dy, va = (4, "bottom") if pos == "above" else (-4, "top")
        ax.annotate(lab, (v, ymax * h), xytext=(0, dy), textcoords="offset points", fontsize=FS, va=va, ha="center")
    ax.text(0.02, 0.62, "uniform\nstrings", transform=ax.transAxes, fontsize=FS, va="top", ha="left", color=COL["REF-U"])
    ax.text(0.98, 0.62, "raw ML-KEM-768\nciphertexts", transform=ax.transAxes, fontsize=FS, va="top", ha="right", color=COL["BL5"])
    ax.set_xlabel(r"LLR statistic $\Lambda$ of the extracted public string", labelpad=2)
    ax.set_ylabel("count", labelpad=2)
    panel(ax, "(g)", x=0.0, y=1.01)
    save(fig, out, "fig5_one_cover_1col")


def main():
    ap = argparse.ArgumentParser()
    root = HERE.parents[1]
    ap.add_argument("--data", default=str(root / "results" / "figures" / "fig5_one_cover_data.npz"))
    ap.add_argument("--out", default=str(root / "results" / "figures"))
    a = ap.parse_args()
    z = np.load(a.data)
    d = {k: z[k] for k in z.files}
    draw(d, Path(a.out))
    for name in SYS3:
        print(f"{name}: changed {(d[f'{name}_stego'] != d['cover']).mean() * 100:.2f}%, "
              f"LLR(stego) {float(d[f'{name}_llr']):+.2f}, LLR(cover) {float(d[f'{name}_llr_cover']):+.2f}")
    print(Path(a.out) / "fig5_one_cover_1col.pdf")


if __name__ == "__main__":
    main()
