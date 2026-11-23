"""Figures and tables of the confirmatory E2/E3 step (paper, Sections 7.6 and 7.8; Figs. 6-8, Tables 10-11, alpha* table) from results/confirm_v1/.

Reads results only (summary.json of analyze_confirm.py, decisions.json of apply_rules.py, and, when SRNet is available,
results/srnet/confirm_srnet_summary.json and decisions_srnet.json). No statistic is recomputed.

    python scripts/experiments/make_figures_confirm.py [--confirm results/confirm_v1] [--out results/figures_confirm]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_figures import (COL, GRID, HATCH, INK, INK2, LS, MRK, ONE_COL, SURF, TWO_COL, WARD, panel,  # noqa: E402
                          plt, save, style)

ROOT = Path(__file__).resolve().parents[2]
SYS = ("PQ-PKS", "REF-U", "BL5", "BL5-L")
C_M = {1: 0.899, 2: 0.980, 3: 0.996, 4: 0.999}      # as analyze_confirm.C_M


def rows_at(S, name):
    """(alpha, row) of one system, sorted by alpha."""
    out = [(float(k.split("/")[0]), r) for k, r in S["E2E3"].items() if k.split("/")[1] == name]
    return sorted(out, key=lambda t: t[0])


def fig6_dpub(S, out, a="0.4"):
    fig, ax = plt.subplots(figsize=(ONE_COL, 2.35))
    style(ax)
    ax.axhspan(0.49, 0.51, color=GRID, lw=0, zorder=0)
    Mmax = 1
    for name in SYS:
        r = S["E2E3"].get(f"{a}/{name}")
        if r is None:
            continue
        Ms = sorted(int(m) for m in r["dpub_acc"])
        if not Ms:                                              # too few test images for every M (code tests only)
            continue
        Mmax = max(Mmax, Ms[-1])
        ys = [r["dpub_acc"][str(m)]["acc"] for m in Ms]
        hollow = name in ("REF-U", "BL5-L")
        ax.plot(Ms, ys, marker=MRK[name], ls=LS[name], color=COL[name], mfc=SURF if hollow else COL[name], label=name)
        for m in Ms:
            ci = r["dpub_acc"][str(m)].get("ci95")
            if ci:
                y = r["dpub_acc"][str(m)]["acc"]
                ax.errorbar([m], [y], yerr=[[y - ci[0]], [ci[1] - y]], color=COL[name], capsize=2, lw=0.8)
        if "H2a" in r and name in ("BL5", "BL5-L"):
            th = {int(m): v["threshold"] for m, v in r["H2a"].items()}
            ax.scatter(list(th), list(th.values()), marker="_", s=80, color=COL[name], lw=1.0, zorder=6)
    ax.scatter(list(C_M), list(C_M.values()), marker="_", s=80, color=INK, lw=1.0, zorder=5, label=r"lower bound $c_M$")
    ax.set_xscale("log", base=2)
    ticks = [2 ** k for k in range(0, int(np.log2(max(Mmax, 2))) + 1)]
    ax.set_xticks(ticks, [str(t) for t in ticks])
    ax.minorticks_off()
    ax.set_xlim(0.85, ticks[-1] * 1.18)
    ax.set_xlabel("independent sessions pooled, $M$")
    ax.set_ylabel(r"accuracy of $D_{\mathrm{pub}}$")
    ax.set_ylim(0.45, 1.03)
    ax.legend(loc="center right", fontsize=7)
    save(fig, out, f"fig6_dpub_vs_M_confirm_a{a.replace('.', '')}")


def _tpr_panel(ax, S, a, legend):
    names = [n for n in SYS if f"{a}/{n}" in S["E2E3"]]
    w = 0.8 / max(len(names), 1)
    for k, n in enumerate(names):
        r = S["E2E3"][f"{a}/{n}"]
        ys = [r[wd]["TPR@5"] for wd in WARD]
        lo = [r[wd]["TPR@5"] - r[wd]["TPR@5_ci95"][0] for wd in WARD]
        hi = [r[wd]["TPR@5_ci95"][1] - r[wd]["TPR@5"] for wd in WARD]
        x = np.arange(len(WARD)) + (k - (len(names) - 1) / 2) * w
        ax.bar(x, ys, w * 0.9, color=COL[n], hatch=HATCH[n], edgecolor=INK, lw=0.4, label=n)
        ax.errorbar(x, ys, yerr=[lo, hi], fmt="none", ecolor=INK, lw=0.6, capsize=1.2)
    ax.axhline(0.05, color=INK, lw=0.7, ls="--")
    ax.set_xticks(np.arange(len(WARD)), WARD)
    ax.tick_params(axis="x", which="both", top=False)
    ax.set_ylim(0, 1.05)
    if legend and names:
        ax.legend(loc="upper left", ncol=2, fontsize=7, handlelength=1.6)
    return names


def fig7_tpr(S, out, a):
    """One figure per alpha (old file names kept) and one figure combining both alpha values (used in the paper)."""
    if not any(f"{a}/{n}" in S["E2E3"] for n in SYS):
        return
    fig, ax = plt.subplots(figsize=(ONE_COL, 2.3))
    style(ax)
    _tpr_panel(ax, S, a, legend=True)
    ax.set_ylabel("TPR at 5% FPR")
    save(fig, out, f"fig7_tpr_wardens_confirm_a{a.replace('.', '')}")


def fig7_tpr_pair(S, out, rates=("0.4", "0.2")):
    rates = [a for a in rates if any(f"{a}/{n}" in S["E2E3"] for n in SYS)]
    if len(rates) < 2:
        return
    fig, axes = plt.subplots(1, 2, figsize=(TWO_COL, 2.25), sharey=True, gridspec_kw={"wspace": 0.08})
    for k, (ax, a) in enumerate(zip(axes, rates)):
        style(ax)
        _tpr_panel(ax, S, a, legend=(k == 0))
        panel(ax, "(a)" if k == 0 else "(b)")
    axes[0].set_ylabel("TPR at 5% FPR")
    save(fig, out, "fig7_tpr_wardens_confirm")


def fig8_pe(S, out, SR=None):
    """SR: results/srnet/confirm_srnet_summary.json (G12), if present: SRNet points at 0.2/0.4 in (a) and SRNet TOST rows in (b)."""
    A = S["alpha_star"]
    fig, axes = plt.subplots(1, 2, figsize=(TWO_COL, 2.45), gridspec_kw={"width_ratios": [1.2, 1], "wspace": 0.5})
    ax = axes[0]
    style(ax)
    ci = A.get("alpha_star_i_ci95")
    if ci:
        ax.axvspan(ci[0], ci[1], color=GRID, lw=0, zorder=0)
    for n in SYS:
        rr = rows_at(S, n)
        if not rr:
            continue
        xs = [a for a, _ in rr]
        ys = [r["W-img"]["P_E"] for _, r in rr]
        hollow = n in ("REF-U", "BL5-L")
        ax.plot(xs, ys, marker=MRK[n], ls=LS[n], color=COL[n], mfc=SURF if hollow else COL[n], label=n)
        if n in ("PQ-PKS", "REF-U"):
            ax.fill_between(xs, [r["W-img"]["P_E_ci95"][0] for _, r in rr], [r["W-img"]["P_E_ci95"][1] for _, r in rr],
                            color=COL[n], alpha=0.12, lw=0)
    if SR:
        for n, fc in (("REF-U", SURF), ("PQ-PKS", COL["PQ-PKS"])):
            pts = sorted((float(al), row[n]) for al, row in SR.get("alphas", {}).items() if isinstance(row.get(n), dict))
            if pts:
                ax.errorbar([a for a, _ in pts], [r["P_E"] for _, r in pts],
                            yerr=[[r["P_E"] - r["P_E_ci95"][0] for _, r in pts], [r["P_E_ci95"][1] - r["P_E"] for _, r in pts]],
                            fmt="D", ms=3.6, color=COL[n], mfc=fc, mec=COL[n], ecolor=COL[n], elinewidth=0.7, capsize=1.5, ls="none",
                            label=f"{n}, SRNet", zorder=5)
    ax.axhline(0.45, color=INK, lw=0.7, ls="--")
    sens = A.get("sensitivity_G4_offset")
    if sens:
        ax.axhline(sens["threshold"], color=INK, lw=0.6, ls=":")
        if sens.get("alpha_star"):
            ax.axvline(sens["alpha_star"], color=INK, lw=0.6, ls=":")
    if A.get("alpha_star"):
        ax.axvline(A["alpha_star"], color=INK, lw=0.7, ls="-")
        ax.text(A["alpha_star"], 1.01, r"$\alpha^\ast$", transform=ax.get_xaxis_transform(), fontsize=8.5, ha="center", va="bottom")
    ax.set_xscale("log")
    from matplotlib.ticker import NullLocator
    ax.xaxis.set_minor_locator(NullLocator())
    grid = sorted({float(k.split("/")[0]) for k in S["E2E3"]})
    ax.set_xticks(grid, [f"{g:g}" for g in grid], fontsize=7, rotation=45)
    ax.set_xlabel(r"embedding rate $\alpha$ (bpp, log scale)")
    ax.set_ylabel(r"$P_E$ of W-img")
    ax.legend(loc="lower left", fontsize=7)
    panel(ax, "(a)", y=1.06)
    ax = axes[1]
    style(ax)
    rows = []
    for k, t in S["TOST"].items():
        a, comp, wd = k.split("/")
        if comp == "PQ-PKS-vs-REF-U":
            rows.append((float(a), wd, t))
    rows.sort(key=lambda t: (t[0], WARD.index(t[1]) if t[1] in WARD else 9))
    if SR:
        for al, row in sorted(SR.get("alphas", {}).items(), key=lambda kv: float(kv[0])):
            tt = (row.get("PQ-PKS_vs_REF-U") or {}).get("TOST")
            if tt:
                rows.append((float(al), "SRNet", tt))
        rows.sort(key=lambda t: (t[0], WARD.index(t[1]) if t[1] in WARD else 9))
    ax.axvspan(-0.02, 0.02, color=GRID, lw=0, zorder=0)
    ax.axvline(0, color=INK, lw=0.5)
    vstyle = {"equivalent": ("o", INK, INK), "inconclusive": ("o", INK, SURF), "different": ("x", COL["REF-U"], COL["REF-U"])}
    for k, (a, wd, t) in enumerate(rows):
        y = len(rows) - 1 - k
        mk, c, fc = vstyle[t["verdict"]]
        ax.plot(t["ci90"], [y, y], color=c, lw=0.9)
        ax.plot([t["diff"]], [y], marker=mk, ms=3.4, color=c, mfc=fc, ls="none")
    ax.set_yticks(range(len(rows)), [rf"$\alpha$ = {a:g}, {wd}" for a, wd, _ in rows][::-1], fontsize=6.5)
    ax.tick_params(axis="y", which="both", right=False)
    ax.set_xlabel(r"$P_E$(PQ-PKS) $-$ $P_E$(REF-U), 90% CI")
    from matplotlib.lines import Line2D
    hs = [Line2D([], [], marker=vstyle[v][0], color=vstyle[v][1], mfc=vstyle[v][2], ls="none", label=v)
          for v in ("equivalent", "inconclusive", "different")]
    ax.legend(handles=hs, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, fontsize=6.8, handletextpad=0.3,
              columnspacing=0.8)
    panel(ax, "(b)", y=1.06)
    save(fig, out, "fig8_pe_alpha_tost_confirm")


def fig9_bits_confirm(E5, L, S, out):
    """Confirmatory Fig. 9: (a) useful bits per image by N at the confirmatory alpha* (e5_summary_confirm.json, merged ladder, sigma
    placed by the A5b decision: FN-DSA-512 in image 2; ML-DSA does not fit, so there is no curve); (b) N*_sigma of ML-DSA-65 by alpha
    (e5_ladder.json, independent of alpha*) with the 95 % CI band of the confirmatory alpha* and the alpha_min values."""
    fig, axes = plt.subplots(1, 2, figsize=(TWO_COL, 2.75), gridspec_kw={"wspace": 0.34})
    ax = axes[0]
    style(ax)
    Ns = [1, 2, 4, 8, 16, 32]
    C = E5["curves"]
    series = [("Kemeleon main|merged|none", "unsigned, merged ladder", COL["PQ-PKS"], "o", "-", COL["PQ-PKS"]),
              ("Kemeleon main|merged|FN-DSA-512", "FN-DSA-512 in image 2, merged ladder", COL["PQ-PKS"], "^", "-", SURF),
              ("Kemeleon main|default|none", "unsigned, single ladder", COL["PQ-PKS"], "o", "--", SURF),
              ("Kemeleon R|merged|none", "unsigned, merged ladder, variant R", COL["REF-U"], "s", "-", COL["REF-U"]),
              ("Kemeleon R|merged|FN-DSA-512", "FN-DSA-512 in image 2, variant R", COL["REF-U"], "^", "-", SURF)]
    for key, lab, c, mk, ls, fc in series:
        cur = C.get(key, {}).get("bits_per_image", {})
        xs = [n for n in Ns if cur.get(str(n)) is not None]
        if xs:
            ax.plot(xs, [cur[str(n)] / 8 / 1024 for n in xs], marker=mk, ls=ls, color=c, mfc=fc, label=lab)
    single = C["Kemeleon main|default|none"]["bits_per_image"]["1"] / 8 / 1024
    ax.axhline(single, color=INK, lw=0.7, ls=":")
    ax.text(30, single + 0.03, "single image", fontsize=7, ha="right", va="bottom")
    ax.set_xscale("log", base=2)
    ax.set_xticks(Ns, [str(n) for n in Ns])
    ax.minorticks_off()
    ax.set_xlabel("session length $N$")
    ax.set_ylabel(rf"useful message per image (KiB), $\alpha^\ast$ = {E5['alpha_star']:.4f}")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=2, fontsize=6.3, handlelength=1.6, columnspacing=0.9)
    panel(ax, "(a)")
    ax = axes[1]
    style(ax)
    ci = (S or {}).get("alpha_star", {}).get("alpha_star_i_ci95") or E5.get("alpha_star_i_ci95")
    if ci:
        ax.axvspan(ci[0], ci[1], color=GRID, lw=0, zorder=0)
    ax.axvline(E5["alpha_star"], color=INK, lw=0.7, ls="-")
    ax.text(E5["alpha_star"], 1.01, r"$\alpha^\ast$", transform=ax.get_xaxis_transform(), fontsize=8.5, ha="center", va="bottom")
    D = L["designs"]
    for d, lab, c, mk, fc in (("D0", r"single ladder, $\sigma$ in image 1", COL["REF-U"], "s", SURF),
                              ("D5", r"merged ladder, $\sigma$ in image 1", COL["PQ-PKS"], "o", COL["PQ-PKS"]),
                              ("D6", r"merged ladder, $\sigma$ in image 2", COL["BL5"], "^", SURF)):
        if d not in D:
            continue
        g = D[d]["sig"]["ML-DSA-65"]["N_star_grid"]
        xs = sorted(float(a) for a in g if g[a] is not None)
        ax.plot(xs, [g[str(a) if str(a) in g else f"{a}"] for a in xs], ls="none", marker=mk, ms=3.2, color=c, mfc=fc, label=lab)
        ax.axvline(D[d]["sig"]["ML-DSA-65"]["alpha_min"], color=c, lw=0.7, ls=":")
    ax.set_yscale("log", base=2)
    ax.set_ylim(1.5, 12)
    ax.set_yticks([2, 4, 8], ["2", "4", "8"])
    ax.set_xlim(0.05, 0.31)
    ax.set_xlabel(r"embedding rate $\alpha$ (bpp)")
    ax.set_ylabel(r"break-even length $N^\ast_\sigma$ (ML-DSA-65)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=1, fontsize=6.3, handlelength=1.6)
    panel(ax, "(b)")
    save(fig, out, "fig9_bits_vs_N_confirm")


def fig10_budget_confirm(E5, out):
    """Confirmatory Fig. 10: the same budget_A8 rows, budget marker at the confirmatory alpha*."""
    from make_figures import fig10_budget
    fig10_budget(E5, out, name="fig10_bit_budget_confirm")


def f3(v):
    return "–" if v is None else f"{v:.3f}"


def tables(S, D, out):
    L = ["| α | System | W-img $P_E$ | W-A0 $P_E$ | W-pay $P_E$ | W-comb $P_E$ | W-img TPR@5% | W-pay TPR@5% | ΔTPR (W-pay − W-img) | $D_{pub}$ acc. M = 1 / 2 / 3 | H2a |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for name in SYS:
        for a, r in rows_at(S, name):
            d = r["dTPR_pay_minus_img"]
            da = r["dpub_acc"]
            h = all(v["holds"] for v in r.get("H2a", {}).values()) if r.get("H2a") else None
            L.append(f"| {a:g} | {name} | " + " | ".join(
                f"{r[w]['P_E']:.3f} [{r[w]['P_E_ci95'][0]:.3f}, {r[w]['P_E_ci95'][1]:.3f}]" for w in WARD)
                + f" | {r['W-img']['TPR@5']:.3f} | {r['W-pay']['TPR@5']:.3f} | {d['dTPR']:+.3f} [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}] | "
                + " / ".join(f3(da.get(str(m), {}).get("acc")) for m in (1, 2, 3)) + f" | {'–' if h is None else ('holds' if h else 'fails')} |")
    (out / "table10_e2e3_confirm.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    T = ["| α | Comparison | Warden | Δ$P_E$ | 90 % CI | Verdict (δ = 0.02) |", "|---|---|---|---|---|---|"]
    for k, t in sorted(S["TOST"].items(), key=lambda kv: float(kv[0].split("/")[0])):
        a, comp, w = k.split("/")
        T.append(f"| {float(a):g} | {comp} | {w} | {t['diff']:+.4f} | [{t['ci90'][0]:+.4f}, {t['ci90'][1]:+.4f}] | {t['verdict']} |")
    (out / "table11_tost_confirm.md").write_text("\n".join(T) + "\n", encoding="utf-8")
    A = S["alpha_star"]
    sens = A.get("sensitivity_G4_offset") or {}
    Q = ["| | Threshold of (i) | $\\alpha^\\ast_i$ | $\\alpha^\\ast_{ii}$ | $\\alpha^\\ast$ | σ placement | Recommended signature |", "|---|---|---|---|---|---|---|"]
    if D:
        Q.append(f"| main (registered) | 0.45 | {f3(A.get('alpha_star_i'))} [{', '.join(f3(x) for x in (A.get('alpha_star_i_ci95') or [None, None]))}] | "
                 f"{f3(A.get('alpha_star_ii'))} | {f3(D['alpha_star'])} | {D['rule_A5b']['sig_placement']} | {D['rule_signature']['recommended']} |")
        s = D.get("sensitivity_G4_offset")
        if s:
            Q.append(f"| sensitivity analysis | {s['threshold']:.4f} | {f3(s.get('alpha_star_i'))} | {f3(A.get('alpha_star_ii'))} | {f3(s['alpha_star'])} | "
                     f"{s['rule_A5b']['sig_placement']} | {s['recommended']} |")
        Q += ["", f"Statement (registered): {D['statement']}"]
        if s:
            Q.append(f"Statement (sensitivity, not a conclusion): {s['statement']}")
    elif sens:
        Q.append(f"| sensitivity analysis | {sens['threshold']:.4f} | {f3(sens.get('alpha_star_i'))} | {f3(A.get('alpha_star_ii'))} | {f3(sens['alpha_star'])} | – | – |")
    (out / "table_alpha_star_confirm.md").write_text("\n".join(Q) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--confirm", default=str(ROOT / "results" / "confirm_v1"))
    ap.add_argument("--out", default=str(ROOT / "results" / "figures_confirm"))
    a = ap.parse_args()
    base, out = Path(a.confirm), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    S = json.loads((base / "summary.json").read_text(encoding="utf-8"))
    dp = base / "decisions.json"
    D = json.loads(dp.read_text(encoding="utf-8")) if dp.exists() else None
    fig6_dpub(S, out, "0.4")
    fig7_tpr(S, out, "0.4")
    fig7_tpr(S, out, "0.2")
    fig7_tpr_pair(S, out)
    srp = ROOT / "results" / "srnet" / "confirm_srnet_summary.json"
    SR = json.loads(srp.read_text(encoding="utf-8")) if srp.exists() else None
    fig8_pe(S, out, SR)
    e5p, lp = ROOT / "results" / "e5_budget" / "e5_summary_confirm.json", ROOT / "results" / "e5_budget" / "e5_ladder.json"
    if e5p.exists():
        E5 = json.loads(e5p.read_text(encoding="utf-8"))
        if lp.exists():
            fig9_bits_confirm(E5, json.loads(lp.read_text(encoding="utf-8")), S, out)
        fig10_budget_confirm(E5, out)
    tables(S, D, out)
    print("wrote", out)


if __name__ == "__main__":
    main()
