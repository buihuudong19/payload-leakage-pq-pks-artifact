"""Fig. 5 and Table 9 for the confirmatory E1 (RQ1.3, H1, gate G3) from results/e1_confirm/ (steps C2, C3 of run_all).

Reads existing results only; no statistic of gate G3 is recomputed (the numbers come from e1_summary.json). Only the p-values per
bit position for Fig. 5a are recomputed from the raw syndromes with exactly the formula of e1_confirm.py (normal approximation to
the binomial, on the 17 979 test images) and stored in e1_pos_pvals_<cell>.npy; the number of Holm rejections must equal
e1_summary.json (the script stops otherwise).

    python scripts/experiments/make_e1_confirm.py [--e1 results/e1_confirm] [--out results/figures_confirm]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_figures import COL, GRID, INK, INK2, SURF, TWO_COL, panel, plt, save, style  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from pqpks.analysis import stats  # noqa: E402

BUCKETS = (256, 512, 1024, 2048, 4096, 8192)
VARIANTS = (1536, 1252)
FIG_CELL = "e1_k1536_b4096"            # cell of Fig. 5a, as in the screening version


def positional_pvals(raw: Path, split: dict, cell: str) -> np.ndarray:
    Z = [np.load(f) for f in sorted(raw.glob("chunk_*.npz"))]
    keys = np.concatenate([z["keys"] for z in Z])
    S = np.concatenate([z[cell] for z in Z])
    folds = split["e1"]["folds"]
    test = np.array([folds.get(str(k), -1) for k in keys]) >= 0
    bits = np.unpackbits(S[test], axis=1).astype(np.float64)
    n = bits.shape[0]
    z = (bits.sum(0) - n / 2) / np.sqrt(n / 4)
    return 2 * norm.sf(np.abs(z))


def fig5(E1: dict, pv: np.ndarray, out: Path):
    """IEEE style (scripts/figures/ieee_style.py): no title inside the figure; image counts and cells are given in the caption."""
    fig, axes = plt.subplots(1, 2, figsize=(TWO_COL, 2.3), gridspec_kw={"width_ratios": [1.5, 1], "wspace": 0.3})
    ax = axes[0]
    style(ax)
    m = pv.size
    thr = -np.log10(0.05 / m)
    ax.scatter(np.arange(m), -np.log10(np.maximum(pv, 1e-300)), s=0.6, color=INK, lw=0, rasterized=True)
    ax.axhline(thr, color=COL["REF-U"], lw=0.9, ls="--")
    ax.text(m * 0.985, thr - 0.12, "Bonferroni/Holm, 5%", ha="right", va="top", fontsize=7.5)
    ax.set_xlim(0, m)
    ax.set_ylim(0, max(thr + 0.6, float(np.nanmax(-np.log10(np.maximum(pv, 1e-300)))) + 0.3))
    ax.set_xlabel(f"bit position of the public syndrome ($B={FIG_CELL.split('_b')[1]}$)")
    ax.set_ylabel(r"$-\log_{10} p$ (binomial test)")
    panel(ax, "(a)")
    ax = axes[1]
    style(ax)
    xs = np.arange(len(BUCKETS))
    ax.axhspan(-0.02, 0.02, color=GRID, lw=0, zorder=0)
    ax.axhline(0, color=INK, lw=0.5)
    series = [("LLR, main", 1536, "llr", -0.24, "PQ-PKS", True), ("learned, main", 1536, "learned", -0.08, "PQ-PKS", False),
              ("LLR, reduced", 1252, "llr", 0.08, "REF-U", True), ("learned, reduced", 1252, "learned", 0.24, "REF-U", False)]
    for lab, kb, key, off, name, filled in series:
        v, lo, hi = [], [], []
        for b in BUCKETS:
            r = E1["cells"][f"e1_k{kb}_b{b}"][key]
            d = r["diff"] if key == "llr" else r["adv"]
            v.append(d); lo.append(d - r["ci95"][0]); hi.append(r["ci95"][1] - d)
        ax.errorbar(xs + off, v, yerr=[lo, hi], fmt="o" if key == "llr" else "s", ms=3.2, lw=0.8, capsize=1.5,
                    color=COL[name], mfc=COL[name] if filled else SURF, label=lab)
    ax.set_xticks(xs, [str(b) for b in BUCKETS])
    ax.set_xlabel("bucket $B$")
    ax.set_ylabel("advantage against uniform")
    ax.legend(loc="upper left", ncol=2, fontsize=6.8, handlelength=1.0, columnspacing=0.8)
    lo_, hi_ = ax.get_ylim()
    ax.set_ylim(lo_, hi_ + 0.35 * (hi_ - lo_))
    panel(ax, "(b)")
    save(fig, out, "fig5_e1_confirm")


def table9(E1: dict, out: Path) -> str:
    L = ["| Variant | Bucket | LLR diff [95 % CI] | upper \\|·\\| | learned adv. [95 % CI] | upper \\|·\\| | Holm rejections | byte χ² p | G3 cell |",
         "|---|---|---|---|---|---|---|---|---|"]
    for kb in VARIANTS:
        for b in BUCKETS:
            r = E1["cells"][f"e1_k{kb}_b{b}"]
            ll, le = r["llr"], r["learned"]
            strata_ok = all(v["upper_abs"] <= v["margin"] for k, v in r.items() if k.startswith("llr_") and isinstance(v, dict) and "margin" in v)
            ok = ll["upper_abs"] <= 0.02 and le["upper_abs"] <= 0.02 and r["positions"]["holm_rejections"] == 0 and strata_ok
            L.append(f"| {'main' if kb == 1536 else 'reduced'} ({kb} B) | {b} | {ll['diff']:+.4f} [{ll['ci95'][0]:+.4f}, {ll['ci95'][1]:+.4f}] | "
                     f"{ll['upper_abs']:.4f} | {le['adv']:+.4f} [{le['ci95'][0]:+.4f}, {le['ci95'][1]:+.4f}] | {le['upper_abs']:.4f} | "
                     f"{r['positions']['holm_rejections']} | {r['byte_chi2_p']:.3f} | {'pass' if ok else '**fail**'} |")
    rb = E1["cells"]["e1_k1252_b256"]["registered_bit"]
    L += ["", f"n = {E1['n_test']:,} test covers (5-fold, split_v1 {E1['split_sha256'][:8]}). "
          f"G3, all cells: {'pass' if E1['G3_all_cells'] else 'fail'}; main variant: {'pass' if E1['G3_main_variant'] else 'fail'}. "
          f"Registered test, bit {rb['bit']} of the reduced variant at B = 256: frequency {rb['freq']:.4f} "
          f"[{rb['ci95'][0]:.4f}, {rb['ci95'][1]:.4f}], p = {rb['p_two_sided']:.2f} ({'confirmed' if rb['confirmed'] else 'not confirmed'})."]
    t = "\n".join(L) + "\n"
    (out / "table9_e1_confirm.md").write_text(t, encoding="utf-8")
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--e1", default=str(ROOT / "results" / "e1_confirm"))
    ap.add_argument("--split", default=str(ROOT / "configs" / "splits" / "split_v1.json"))
    ap.add_argument("--out", default=str(ROOT / "results" / "figures_confirm"))
    a = ap.parse_args()
    e1, out = Path(a.e1), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    E1 = json.loads((e1 / "e1_summary.json").read_text(encoding="utf-8"))
    split = json.loads(Path(a.split).read_text(encoding="utf-8"))
    pf = e1 / f"e1_pos_pvals_{FIG_CELL}.npy"
    if pf.exists():
        pv = np.load(pf)
    else:
        pv = positional_pvals(e1 / "raw", split, FIG_CELL)
        np.save(pf, pv)
    rej = int(stats.holm(pv).sum())
    want = E1["cells"][FIG_CELL]["positions"]["holm_rejections"]
    if rej != want or pv.size != E1["cells"][FIG_CELL]["positions"]["n_positions"]:
        raise SystemExit(f"recomputed p-values do not match e1_summary.json: Holm {rej} vs {want}")
    fig5(E1, pv, out)
    print(table9(E1, out))
    print("wrote", out)


if __name__ == "__main__":
    main()
