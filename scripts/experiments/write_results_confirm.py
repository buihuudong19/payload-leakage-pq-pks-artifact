"""Step R0: write results/RESULTS_TABLES.md from the CONFIRMATORY results (replaces the screening version of write_results_md.py).

The blocks "**Table 9 ...**", "**Table 10 ...**", "**Table 11 ...**", "**Table 12 ...**" and "**Table S1 ...**" are located
by these labels, so no other block may start with "**Table ". Sources:
  Table 9   results/e1_confirm/e1_summary.json (C3) + results/e1bis/e1bis_summary.json (X5)
  Table 10  results/confirm_v1/summary.json (C7) [+ results/srnet/confirm_srnet_summary.json once G12 is done]
  Table 11  TOST from the same file; alpha* table and rules from results/confirm_v1/decisions.json (C8) [+ decisions_srnet.json, G13]
  Table 12  results/e5_budget/e5_summary_confirm.json (C10: E5 at the confirmatory alpha*, sigma by A5b) + results/e6_desktop/e6_summary.json (C9)
  Table S1  results/e4_full/e4_summary.json (E4)
A missing file yields a block reading "no results yet". The earlier screening version is kept in results/RESULTS_TABLES_screening.md (written only if absent).

    python scripts/experiments/write_results_confirm.py [--out results/RESULTS_TABLES.md]
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "experiments"))
RES = ROOT / "results"
WARD = ("W-img", "W-A0", "W-pay", "W-comb")
SYS = ("PQ-PKS", "REF-U", "BL5", "BL5-L")


def jl(p: Path):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def f3(v, d=3):
    return "–" if v is None else f"{v:.{d}f}"


def ci(c, d=3):
    return "–" if not c else f"[{c[0]:.{d}f}, {c[1]:.{d}f}]"


def table9(E1, EB):
    L = ["**Table 9 [confirmatory].** E1 on the 17,979 non-pilot, non-screening BOSSbase + BOWS2 images of split_v1 (5-fold; C3), "
         "and E1-bis on 40,000 ALASKA2 images (X5; one-sided rule adv + 1.645·SD_null ≤ 0.02). "
         "LLR: fixed distinguisher of Proposition 2 on the cover syndrome against uniform strings; learned: linear distinguisher on byte features.", ""]
    if E1:
        L += ["| Set | Variant | Bucket | LLR diff [95 % CI] | upper \\|·\\| | learned adv. [95 % CI] | upper \\|·\\| | Holm rejections / positions | byte χ² p | gate cell |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for k, r in E1["cells"].items():
            _, kb, b = k.split("_")
            ll, le = r["llr"], r["learned"]
            strata_ok = all(v["upper_abs"] <= v["margin"] for kk, v in r.items() if kk.startswith("llr_") and isinstance(v, dict) and "margin" in v)
            ok = ll["upper_abs"] <= 0.02 and le["upper_abs"] <= 0.02 and r["positions"]["holm_rejections"] == 0 and strata_ok
            L.append(f"| E1 | {'main' if kb == 'k1536' else 'reduced'} ({kb[1:]} B) | {b[1:]} | {ll['diff']:+.4f} {ci(ll['ci95'], 4)} | {ll['upper_abs']:.4f} | "
                     f"{le['adv']:+.4f} {ci(le['ci95'], 4)} | {le['upper_abs']:.4f} | {r['positions']['holm_rejections']} / {r['positions']['n_positions']} | "
                     f"{r['byte_chi2_p']:.3f} | {'pass' if ok else '**fail**'} |")
        rb = E1["cells"].get("e1_k1252_b256", {}).get("registered_bit")
        L.append("")
        L.append(f"E1: n = {E1['n_test']:,} test covers; gate G3 all cells: {'pass' if E1['G3_all_cells'] else '**fail**'} "
                 f"(main variant: {'pass' if E1['G3_main_variant'] else 'fail'})"
                 + (f"; registered bit {rb['bit']} of the reduced variant at B = 256: frequency {rb['freq']:.4f} {ci(rb['ci95'], 4)}, "
                    f"p = {rb['p_two_sided']:.2f} ({'confirmed' if rb['confirmed'] else 'not confirmed'})." if rb else "."))
    else:
        L.append("E1 confirmatory: no results yet (results/e1_confirm/e1_summary.json).")
    if EB:
        L += ["", "| Set | Variant | Bucket | LLR diff [95 % CI] | learned adv. | SD_null | one-sided bound | Holm (pooled) | byte χ² p | gate cell |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for k, r in EB["cells"].items():
            _, kb, b = k.split("_")
            ll, le = r["llr"], r["learned"]
            L.append(f"| E1-bis | {'main' if kb == 'k1536' else 'reduced'} ({kb[1:]} B) | {b[1:]} | {ll['diff']:+.4f} {ci(ll['ci95'], 4)} | "
                     f"{le['adv']:+.4f} | {le['sd_null']:.4f} | {le['upper_one_sided']:.4f} | {r['positions']['holm_pooled_rejections']} / {r['positions']['n_positions']} | "
                     f"{r['byte_chi2_p']:.3f} | {'pass' if le['pass'] else '**fail**'} |")
        d2, d3 = EB["D2"], EB["D3_replication"]
        L.append("")
        L.append(f"E1-bis: n = {EB['n']:,} ALASKA2 covers; gate G3′ all cells: {'pass' if EB['G3prime_all_cells'] else '**fail**'}; "
                 f"D2 positions: {d2['rejections']} rejections / {d2['n_positions']:,} (Holm, α = {d2['alpha']}); "
                 f"D3 replication of {d3['cell']}: p = {d3['p']:.3f} ({'replicated' if d3['confirmed'] else 'not replicated'}).")
    return L


def table10_11(S, D, SR, DS):
    L = [f"**Table 10 [confirmatory].** E2/E3 on 20,000 covers (10,000 BOSSbase + 10,000 BOWS2, split_v1; test 5,000 per split, "
         f"{S['reps']} repeated splits; W-img = SRM + FLD ensemble; 95 % two-level bootstrap CIs; C7).", "",
         "| α | System | W-img $P_E$ | W-A0 $P_E$ | W-pay $P_E$ | W-comb $P_E$ | W-img TPR@5% | W-pay TPR@5% | ΔTPR (W-pay − W-img) [95 % CI] | $D_{\\rm pub}$ acc. M = 1 / 2 / 3 / 4 | H2a |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    rows = sorted(S["E2E3"].items(), key=lambda kv: (float(kv[0].split("/")[0]), SYS.index(kv[0].split("/")[1]) if kv[0].split("/")[1] in SYS else 9))
    for k, r in rows:
        a, name = k.split("/")
        d, da = r["dTPR_pay_minus_img"], r["dpub_acc"]
        h = all(v["holds"] for v in r.get("H2a", {}).values()) if r.get("H2a") else None
        L.append(f"| {float(a):g} | {name} | " + " | ".join(f"{r[w]['P_E']:.3f} {ci(r[w]['P_E_ci95'])}" if w in r else "–" for w in WARD)
                 + f" | {r['W-img']['TPR@5']:.3f} | {r['W-pay']['TPR@5']:.3f} | {d['dTPR']:+.3f} [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}] | "
                 + " / ".join(f3(da.get(str(m), {}).get("acc")) for m in (1, 2, 3, 4)) + f" | {'–' if h is None else ('holds' if h else '**fails**')} |")
    ex = S.get("excluded_covers", {})
    n_ex = ex.get("n", 0) if isinstance(ex, dict) else len(ex)      # analyze_confirm records the list of ids
    L.append("")
    L.append(f"Excluded covers: {n_ex}. W-A0: SPAM on the public stage $A_0$. $D_{{\\rm pub}}$ for PQ-PKS and REF-U must lie in [0.49, 0.51]. "
             f"$D_{{\\rm pub}}$ at M = 4 is reported against $c_4$ = 0.999 (Table 7; H2a column covers M = 1–4).")
    if SR:
        # confirm_srnet_summary.json (srnet_evaluate.py): alphas[alpha][system] = {P_E, P_E_ci95, TPR@5, AUC, ...};
        # alphas[alpha]["PQ-PKS_vs_REF-U"] = {TOST, McNemar: {PQ_right_REF_wrong, REF_right_PQ_wrong, p_exact}}
        L += ["", "SRNet (confirmatory, G12): $P_E$ of W-img = SRNet on 512×512 test images, one split; thresholds from validation; "
                  "95 % bootstrap CI over test covers; paired TOST (δ = 0.02, 90 % CI) and exact McNemar test against REF-U.", "",
              "| α | System | SRNet $P_E$ [95 % CI] | TPR@5% | AUC | Δ$P_E$ vs REF-U [90 % CI] | TOST | McNemar (PQ right/REF wrong, REF right/PQ wrong; p) |",
              "|---|---|---|---|---|---|---|---|"]
        for al, row in sorted(SR.get("alphas", {}).items(), key=lambda kv: float(kv[0])):
            cmpr = row.get("PQ-PKS_vs_REF-U", {})
            for name in ("REF-U", "PQ-PKS"):
                v = row.get(name)
                if not isinstance(v, dict):
                    continue
                if name == "PQ-PKS" and cmpr:
                    t, mc = cmpr.get("TOST", {}), cmpr.get("McNemar", {})
                    tail = (f"{t.get('diff', 0):+.4f} [{t['ci90'][0]:+.4f}, {t['ci90'][1]:+.4f}] | {t.get('verdict', '–')} | "
                            f"{mc.get('PQ_right_REF_wrong', '–')} / {mc.get('REF_right_PQ_wrong', '–')}; p = {mc.get('p_exact', float('nan')):.3f}")
                else:
                    tail = "– | – | –"
                L.append(f"| {float(al):g} | {name} | {v['P_E']:.4f} {ci(v.get('P_E_ci95'))} | {v.get('TPR@5', float('nan')):.3f} | "
                         f"{v.get('AUC', float('nan')):.3f} | {tail} |")
        srm = {al: (row.get("cond_i_SRNet") or {}) for al, row in SR.get("alphas", {}).items()}
        L.append("")
        L.append("W-img for α\\* = min(SRM, SRNet) on REF-U: " + "; ".join(
            f"α = {float(al):g}: SRM {c.get('P_E_REF-U_SRM', float('nan')):.4f}, SRNet {c.get('P_E_REF-U', float('nan')):.4f} (stronger: {c.get('stronger_on_REF-U', '–')})"
            for al, c in sorted(srm.items(), key=lambda kv: float(kv[0])) if c) + ".")
    L += ["", "**Table 11 [confirmatory].** Paired TOST on $P_E$ against REF-U (δ = 0.02; 90 % CI of the difference), C7.", "",
          "| α | Comparison | Warden | Δ$P_E$ | 90 % CI | Verdict |", "|---|---|---|---|---|---|"]
    for k, t in sorted(S["TOST"].items(), key=lambda kv: float(kv[0].split("/")[0])):
        a, comp, w = k.split("/")
        L.append(f"| {float(a):g} | {comp} | {w} | {t['diff']:+.4f} | [{t['ci90'][0]:+.4f}, {t['ci90'][1]:+.4f}] | {t['verdict']} |")
    A = S["alpha_star"]
    L += ["", "α\\* (registered rule: largest α with $P_E$(W-img, REF-U) ≥ 0.45 and TPR@5 %(W-comb, PQ-PKS) − TPR@5 %(W-comb, REF-U) ≤ 0.02; "
          "linear interpolation on the grid; 95 % bootstrap CI over covers and splits):", "",
          "| | Threshold of (i) | $\\alpha^\\ast_i$ [95 % CI] | $\\alpha^\\ast_{ii}$ | $\\alpha^\\ast$ | bootstrap below grid | σ placement (A5b) | Recommended signature |",
          "|---|---|---|---|---|---|---|---|"]
    if D:
        L.append(f"| main (registered, SRM) | 0.45 | {f3(A.get('alpha_star_i'), 4)} {ci(A.get('alpha_star_i_ci95'), 4)} | {f3(A.get('alpha_star_ii'), 4)}"
                 f"{' (censored at the grid end)' if A.get('alpha_star_ii_censored') else ''} | {f3(D['alpha_star'], 4)} | "
                 f"{f3(A.get('alpha_star_i_boot_below_grid'), 3)} | {D['rule_A5b']['sig_placement']} | {D['rule_signature']['recommended'] or 'none fits (FN-DSA-512 in image 2: analytic only)'} |")
        s = D.get("sensitivity_G4_offset")
        if s:
            L.append(f"| sensitivity (+G4 offset) | {s['threshold']:.4f} | {f3(s.get('alpha_star_i'), 4)} | {f3(A.get('alpha_star_ii'), 4)} | {f3(s['alpha_star'], 4)} | – | "
                     f"{s['rule_A5b']['sig_placement']} | {s['recommended'] or 'none fits'} |")
        if DS:
            L.append(f"| with SRNet (G13) | 0.45 | {f3(DS.get('alpha_star_i'), 4)} | {f3(DS.get('alpha_star_ii'), 4)} | {f3(DS.get('alpha_star'), 4)} | – | "
                     f"{DS['rule_A5b']['sig_placement']} | {DS['rule_signature']['recommended'] or 'none fits (FN-DSA-512 in image 2: analytic only)'} |")
        L += ["", f"Statement (registered): {D['statement']}"]
        if s:
            L.append(f"Statement (sensitivity, not a conclusion): {s['statement']}")
        mf = D.get("minimal_first_image", {}).get("direct_check")
        if mf:
            L.append(f"Direct check at the minimal first image (α = {mf['alpha']}): $P_E$(W-img, REF-U) = {mf['P_E_W-img_REF-U']:.4f}, "
                     f"TPR difference {mf['TPR_diff_W-comb']:+.4f}, criterion {'holds' if mf['holds'] else 'fails'}.")
        pe = A.get("P_E_W-img_REF-U", {})
        if pe:
            L.append("$P_E$(W-img, REF-U) on the grid: " + "; ".join(f"α = {float(k):g}: {v:.4f}" for k, v in sorted(pe.items(), key=lambda kv: float(kv[0]))) + ".")
    return L


def table12(E5, E6, D):
    L = ["**Table 12 [confirmatory].** RQ3: detectability budget, capacity and cost at the confirmatory α\\*.", ""]
    if E5:
        pl = E5.get("sig_placement", "image1")
        L += [f"α\\* = {E5['alpha_star']:.4f} ({E5['alpha_star_source']}); budget {int(E5['alpha_star'] * E5['n'] / 8)} B per 512×512 image; "
              f"σ placement: {pl} ({'A5b' if pl == 'image2' else 'A5a'}). Useful message per image (bytes) by session length N, merged ladder "
              f"(first-image buckets 256…8192, later-image buckets $\\mathcal B\\cup(\\mathcal B+|\\hat c|)$); '–' = no feasible session.", "",
              "| Mode | Variant | N = 1 | N = 2 | N = 4 | N = 8 | N = 16 | N = 32 | $N^\\ast_\\sigma$ |", "|---|---|---|---|---|---|---|---|---|"]
        for key, cur in E5["curves"].items():
            k, b, sname = key.split("|")
            if b != "merged":
                continue
            bp = cur["bits_per_image"]
            L.append(f"| {sname if sname != 'none' else 'unsigned session'} | {k} | " + " | ".join(f"{bp[str(n)] / 8:.0f}" if bp.get(str(n)) is not None else "–" for n in (1, 2, 4, 8, 16, 32))
                     + (f" | {cur.get('N_star_sigma', '–')} |" if sname != "none" else " | trivial |"))
        feas = E5.get("signed_session_feasibility_merged", {})
        if feas:
            L += ["", "Signed sessions at α\\* (merged ladder): smallest feasible N and useful bytes per image. 'span' (A5c) is exploratory: "
                  "σ spread over the later images, not part of the registered design.", "",
                  "| σ placement | Variant | Signature | $N_{\\min}$ | N = 2 | N = 4 | N = 8 | N = 16 |", "|---|---|---|---|---|---|---|---|"]
            for key, v in feas.items():
                pl_, k, sname = key.split("|")
                bp = v["bits_per_image"]
                L.append(f"| {pl_} | {k} | {sname} | {v['N_min_feasible'] if v['N_min_feasible'] is not None else '–'} | "
                         + " | ".join(f"{bp[str(n)] / 8:.0f}" if bp.get(str(n)) is not None else "–" for n in (2, 4, 8, 16)) + " |")
        sens = E5.get("sensitivity", {})
        if sens:
            L += ["", "Sensitivity to α (single unsigned image, bytes; ML-DSA-65 session feasible?):", "",
                  "| α | single unsigned (B) | signed first image feasible (ML-DSA-65, A5a) |", "|---|---|---|"]
            for a, v in sorted(sens.items(), key=lambda kv: float(kv[0])):
                sb = f"{v['single_unsigned_bits'] / 8:.0f}" if v['single_unsigned_bits'] is not None else "–"
                L.append(f"| {float(a):g} | {sb} | {'yes' if v['signed_first_image_feasible'] else 'no'} |")
    else:
        L.append("E5 at the confirmatory α\\*: no results yet (results/e5_budget/e5_summary_confirm.json; step C10).")
    if E6:
        m = E6.get("machine", {})
        L += ["", f"E6 timings ({m.get('note', '')}; CPU {m.get('cpu', '')}; Python {m.get('python', '')}; C9).", "",
              "| Step | median (ms) | IQR (ms) | reps |", "|---|---|---|---|"]
        for k, v in E6.items():
            if isinstance(v, dict) and "median_ms" in v:
                lab = k.replace("|", "\\|")
                L.append(f"| {lab} | {v['median_ms']:.2f} | {v['iqr_ms']:.2f} | {v['reps']} |")
        share = E6.get("H6 crypto share of Send (unsigned, N=8 key precompute)")
        inv = E6.get("Recv invalid image, S_max sessions open (default profile)", {})
        L.append("")
        L.append(f"Share of cryptography in Send (unsigned, 8 image keys precomputed): {100 * share:.2f} % (H6 threshold 10 %)"
                 + (f"; receiver trials per invalid image: {inv.get('trials_per_image', ['–'])[0]} (bound 2646)." if inv else "."))
    else:
        L.append("E6 (desktop): no results yet (results/e6_desktop/e6_summary.json).")
    return L


def tableS1(E4):
    if not E4:
        return ["**Table S1 [full-scale E4].** no results yet (results/e4_full/e4_summary.json).", ""]
    n, fa, pr = E4["normal"], E4["false_alarm"], E4["profile"]
    L = [f"**Table S1 [full-scale E4].** Functional block on 512×512 images (layout {pr['layout']}, buckets {pr['buckets']}).", "",
         f"- Normal send/receive: {n['images']:,} images, {n['errors']} errors (upper 95 % bound {n['upper95_error']:.2e}), "
         f"{n['inadmissible']} inadmissible, {n['embed_failures']} embedding failures.", "",
         "| Class | Trials | Expected outcome observed | Rate |", "|---|---|---|---|"]
    for k, v in E4["tamper"].items():
        L.append(f"| {k} | {v['trials']:,} | {v['expected_outcome']:,} | {v['rate']:.3f} |")
    L.append("")
    L.append(f"False alarms: {fa['accepted']} accepted among {fa['cover_images']} cover images with {fa['aead_trials']:,} AEAD trials "
             f"(mean {fa['trials_per_image_mean']:.0f} per image, bound {fa['trials_per_image_max_bound']}); upper 95 % bound per trial "
             f"{fa['upper95_per_trial']:.2e}; median receiver time {fa['receiver_time_s_median']:.2f} s per invalid image (512×512).")
    return L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(RES / "RESULTS_TABLES.md"))
    a = ap.parse_args()
    out = Path(a.out)
    keep = out.with_name("RESULTS_TABLES_screening.md")
    if out.exists() and not keep.exists() and "[screening]" in out.read_text(encoding="utf-8"):
        shutil.copy2(out, keep)
    E1, EB = jl(RES / "e1_confirm" / "e1_summary.json"), jl(RES / "e1bis" / "e1bis_summary.json")
    S, D = jl(RES / "confirm_v1" / "summary.json"), jl(RES / "confirm_v1" / "decisions.json")
    SR, DS = jl(RES / "srnet" / "confirm_srnet_summary.json"), jl(RES / "srnet" / "decisions_srnet.json")
    E5, E6, E4 = jl(RES / "e5_budget" / "e5_summary_confirm.json"), jl(RES / "e6_desktop" / "e6_summary.json"), jl(RES / "e4_full" / "e4_summary.json")
    L = ["Confirmatory results (written by scripts/experiments/write_results_confirm.py, step R0; screening version kept in RESULTS_TABLES_screening.md).", ""]
    L += table9(E1, EB) + [""]
    if S:
        L += table10_11(S, D, SR, DS) + [""]
    else:
        L += ["**Table 10 [confirmatory].** no results yet (results/confirm_v1/summary.json).", "", "**Table 11 [confirmatory].** no results yet.", ""]
    L += table12(E5, E6, D) + [""] + tableS1(E4)
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    missing = [n for n, v in (("E1", E1), ("E1-bis", EB), ("C7", S), ("C8", D), ("E5-confirm", E5), ("E6", E6), ("E4", E4), ("SRNet G12", SR), ("SRNet G13", DS)) if v is None]
    print(out, "| missing:", ", ".join(missing) if missing else "none")


if __name__ == "__main__":
    main()
