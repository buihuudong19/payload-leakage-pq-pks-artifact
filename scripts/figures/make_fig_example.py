"""Fig. 5 of the paper (Section 7.6, "One cover, end to end"): one BOSSbase cover through the whole pipeline at alpha = 0.4.

    python scripts/figures/make_fig_example.py [--cover 8811] [--alpha 0.4] [--out results/figures] [--data <npz>]

Panels: (a) cover; (b) HILL cost log10 rho; (c) changed positions of PQ-PKS coloured by stage (public A0 / keyed A1, layout A8);
(d)-(f) changed positions of BL5, PQ-PKS, REF-U; (g) LLR statistic Lambda (Proposition 2) of the public string extracted from the
cover and from the three stego images, over the distribution of Lambda for 2000 uniform strings and 2000 raw ML-KEM-768
ciphertexts (generated as for Fig. 2). Frozen profile (DEFAULT_PP, layout A8, h = 10), the same embedding/extraction code as
screen_run.py; no run result is read. Keys, messages and ciphertexts are drawn at random at run time; every array used for the
plot is stored in <out>/fig5_one_cover_data.npz so that exactly the same figure can be redrawn (--data) without embedding again.
Cover 8811 (pilot BOSSbase, training part): sat_frac 0.000431 (median of the 1000 pilot images: 0.000509); 32 % of the 8x8 blocks
have variance < 4 (sky, water) and 32 % have variance > 100 (sand, rigging); chosen by two rules fixed in advance (paper, Section 7.6).
"""
from __future__ import annotations

import argparse
import secrets
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "figures"))
sys.path.insert(0, str(ROOT / "scripts" / "experiments"))
from ieee_style import COL, INK, INK2, TWO_COL, panel, plt, save, style  # noqa: E402

SYS3 = ("BL5", "PQ-PKS", "REF-U")
CROP = (384, 0, 128)                 # (row, column, side) of the crop for panel (c): sand, lower-left corner


def embed_all(cover_id: int, alpha: float):
    from pqpks import data, systems
    from pqpks.analysis import dpub
    from pqpks.crypto import hpke
    from pqpks.experiment.profiles import message_len_for_bucket
    from pqpks.params import AEAD_TAG_BYTES, DEFAULT_PP
    from pqpks.stego.two_stage import TwoStageSTC
    from screen_run import KEM_BYTES, make_system, profile_of

    x = data.load_bossbase(cover_id)
    n = x.size
    emb0 = TwoStageSTC(DEFAULT_PP)
    rho, plus_ok, minus_ok = emb0.cost_map(x)
    out = {"cover": x, "rho": np.asarray(rho).reshape(x.shape), "alpha": alpha, "cover_id": cover_id}
    for name in SYS3:
        kb = KEM_BYTES[name]
        pp = profile_of(alpha, n, kb)
        bb = pp.buckets[0] + AEAD_TAG_BYTES
        obj = make_system(name, pp)
        if name == "REF-U":
            y = obj.stego(x, bb, start=True, cost_map=(rho, plus_ok, minus_ok))
            emb = obj.embedder
        else:
            s, dk, ek = obj
            sess = s.start_session(ek, length=1)
            msg = secrets.token_bytes(message_len_for_bucket(pp.buckets[0], start=True))
            y = s.send(sess, msg, x, cost_map=(rho, plus_ok, minus_ok))
            emb = s.embedder
            if name == "PQ-PKS":
                assert s.receive_msg(dk, y, systems.ReceiverState()) == msg
        a0 = emb.a0_mask(x.shape, kb, bb)
        out[f"{name}_stego"] = y
        out[f"{name}_a0"] = a0
        out[f"{name}_ext0"] = np.frombuffer(emb.extract_kem(y, kb, bb), dtype=np.uint8)
        out[f"{name}_ext0_cover"] = np.frombuffer(emb.extract_kem(x, kb, bb), dtype=np.uint8)
        out[f"{name}_rate"] = 8 * (kb + bb) / n
    # background of panel (g): generated as for Fig. 2 (make_figures.fig2_prop1)
    kem = hpke.MlKem768()
    _, pk = kem.generate()
    out["llr_raw"] = np.array([dpub.mlkem_raw_llr(hpke.setup_base_s(kem, pk, b"fig5")[0]) for _ in range(2000)])
    rng = np.random.default_rng(5)
    out["llr_uni"] = np.array([dpub.mlkem_raw_llr(rng.integers(0, 256, 1088, dtype=np.uint8).tobytes()) for _ in range(2000)])
    for name in SYS3:
        out[f"{name}_llr"] = dpub.mlkem_raw_llr(bytes(out[f"{name}_ext0"]))
        out[f"{name}_llr_cover"] = dpub.mlkem_raw_llr(bytes(out[f"{name}_ext0_cover"]))
    return out


def _dilate(m: np.ndarray, r: int = 1) -> np.ndarray:
    """Enlarge each changed position to a (2r+1)^2 square so that it is visible at print size (512 px over ~1.6 in)."""
    out = m.copy()
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            out |= np.roll(np.roll(m, dy, 0), dx, 1)
    return out


def draw(d, out: Path):
    x = d["cover"]
    fig = plt.figure(figsize=(TWO_COL, 4.55))
    gs = fig.add_gridspec(2, 4, wspace=0.08, hspace=0.36, left=0.01, right=0.99, top=0.97, bottom=0.11)
    axs = [fig.add_subplot(gs[0, k]) for k in range(3)] + [fig.add_subplot(gs[1, k]) for k in range(3)]
    axg = fig.add_subplot(gs[:, 3])
    for ax in axs:
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_linewidth(0.5)
    # (a) cover
    axs[0].imshow(x, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    panel(axs[0], "(a)", x=0.0, y=1.01)
    axs[0].set_xlabel(f"cover (BOSSbase {int(d['cover_id'])})", fontsize=7.5, labelpad=2)
    # (b) HILL cost
    rho = np.asarray(d["rho"], dtype=np.float64)
    fin = np.isfinite(rho)
    lr = np.log10(np.clip(rho, 1e-3, None))
    lr[~fin] = np.nanmax(lr[fin])
    lo, hi = np.percentile(lr[fin], [1, 97])          # clip the tails: flat saturated areas have rho ~ 1e4-1e5 (practically wet)
    axs[1].imshow(np.clip(lr, lo, hi), cmap="gray_r", interpolation="nearest")
    panel(axs[1], "(b)", x=0.0, y=1.01)
    axs[1].set_xlabel(r"HILL cost $\rho$ (dark = costly, $\log_{10}$ scale)", fontsize=7.5, labelpad=2)
    # (c) changes of PQ-PKS by stage, on a 128x128 crop (sand, lower-left corner) so that single points are visible
    yp = d["PQ-PKS_stego"]
    ch = yp != x
    a0 = d["PQ-PKS_a0"].astype(bool)
    r0_, c0_, L = CROP
    sl = (slice(r0_, r0_ + L), slice(c0_, c0_ + L))
    rgb = np.ones((L, L, 3))
    rgb[(ch & ~a0)[sl]] = (0.0, 0.447, 0.698)      # A1 (keyed): blue
    rgb[(ch & a0)[sl]] = (0.835, 0.369, 0.0)       # A0 (public): vermillion
    axs[2].imshow(rgb, interpolation="nearest")
    panel(axs[2], "(c)", x=0.0, y=1.01)
    n0, n1 = int((ch & a0).sum()), int((ch & ~a0).sum())
    r0, r1 = 100 * n0 / a0.sum(), 100 * n1 / (~a0).sum()
    axs[2].set_xlabel(f"PQ-PKS changes in the boxed {L}×{L} crop:\n$A_0$ public (orange), $A_1$ keyed (blue)", fontsize=7.5, labelpad=2)
    print(f"(c) change rate in A0 {r0:.2f}% ({n0}/{int(a0.sum())}), in A1 {r1:.2f}% ({n1}/{int((~a0).sum())})")
    axs[0].add_patch(plt.Rectangle((c0_, r0_), L, L, fill=False, ec="#E69F00", lw=0.8))
    # (d)-(f) change maps
    for k, name in enumerate(SYS3):
        chm = d[f"{name}_stego"] != x
        axs[3 + k].imshow(~chm, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        panel(axs[3 + k], f"({'def'[k]})", x=0.0, y=1.01)
        axs[3 + k].set_xlabel(f"{name}: {chm.mean() * 100:.2f}% of pixels changed", fontsize=7.5, labelpad=2)
    # (g) LLR of the public string
    ax = axg
    style(ax)
    raw, uni = d["llr_raw"], d["llr_uni"]
    bins = np.linspace(min(raw.min(), uni.min()), max(raw.max(), uni.max()), 40)
    ax.hist(uni, bins=bins, orientation="horizontal", histtype="stepfilled", color=COL["REF-U"], alpha=0.22, lw=0)
    ax.hist(uni, bins=bins, orientation="horizontal", histtype="step", color=COL["REF-U"], lw=0.8, ls="--")
    ax.hist(raw, bins=bins, orientation="horizontal", histtype="stepfilled", color=COL["BL5"], alpha=0.22, lw=0, hatch="....")
    ax.hist(raw, bins=bins, orientation="horizontal", histtype="step", color=COL["BL5"], lw=0.8)
    ax.axhline(0, color=INK, lw=0.6, ls=":")
    xmax = ax.get_xlim()[1]
    ax.set_xlim(0, xmax * 1.9)
    marks = [("cover syndrome", float(d["PQ-PKS_llr_cover"]), INK2, "x"), ("BL5", float(d["BL5_llr"]), COL["BL5"], "^"),
             ("PQ-PKS", float(d["PQ-PKS_llr"]), COL["PQ-PKS"], "o"), ("REF-U", float(d["REF-U_llr"]), COL["REF-U"], "s")]
    xs = xmax * 1.25
    for lab, v, c, m in marks:
        ax.plot([xs], [v], marker=m, color=c, ms=4.5, mec=INK, mew=0.4, ls="none", zorder=5)
        ax.annotate(lab, (xs, v), xytext=(5, 0), textcoords="offset points", fontsize=7, va="center", ha="left")
    ax.text(0.03, 0.97, "uniform\nstrings", transform=ax.transAxes, fontsize=7, va="top", ha="left", color=COL["REF-U"])
    ax.text(0.03, 0.03, "raw ML-KEM-768\nciphertexts", transform=ax.transAxes, fontsize=7, va="bottom", ha="left", color=COL["BL5"])
    ax.set_ylabel(r"LLR statistic $\Lambda$ of the extracted public string", labelpad=2)
    ax.set_xlabel("count (2000 strings each)")
    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()
    panel(ax, "(g)", x=0.0, y=1.01)
    save(fig, out, "fig5_one_cover")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cover", type=int, default=8811)
    ap.add_argument("--alpha", type=float, default=0.4)
    ap.add_argument("--out", default=str(ROOT / "results" / "figures"))
    ap.add_argument("--data", default=None, help="redraw from a stored npz file (no new embedding)")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.data:
        z = np.load(a.data)
        d = {k: z[k] for k in z.files}
    else:
        d = embed_all(a.cover, a.alpha)
        np.savez_compressed(out / "fig5_one_cover_data.npz", **{k: np.asarray(v) for k, v in d.items()})
    draw(d, out)
    for name in SYS3:
        print(f"{name}: rate {float(d[f'{name}_rate']):.4f}, changed {(d[f'{name}_stego'] != d['cover']).mean() * 100:.3f}%, "
              f"Λ(stego) {float(d[f'{name}_llr']):+.2f}, Λ(cover syndrome) {float(d[f'{name}_llr_cover']):+.2f}")
    print(out / "fig5_one_cover.pdf")


if __name__ == "__main__":
    main()
