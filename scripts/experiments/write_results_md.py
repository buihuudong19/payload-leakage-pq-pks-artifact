"""Result tables (Markdown) for Sections 7.6-7.8 of the paper from the summary JSON files, so that no number is copied by
hand. Output: results/RESULTS_TABLES.md (screening version; see write_results_confirm.py for the confirmatory version)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def f(x, d=3):
    return "–" if x is None else f"{x:.{d}f}"


def ci(c, d=3):
    return "–" if c is None else f"[{c[0]:.{d}f}, {c[1]:.{d}f}]"


def main(screen=ROOT / "results" / "archive" / "pre-confirm" / "screen_v1", out=ROOT / "results" / "RESULTS_TABLES.md"):
    S = json.loads((screen / "summary.json").read_text())
    E5 = json.loads((ROOT / "results" / "e5_budget" / "e5_summary.json").read_text())
    p4 = ROOT / "results" / "archive" / "pre-confirm" / "e4_functional" / "e4_summary.json"
    p6 = ROOT / "results" / "archive" / "pre-confirm" / "e6_perf" / "e6_summary.json"
    E4 = json.loads(p4.read_text()) if p4.exists() else None
    E6 = json.loads(p6.read_text()) if p6.exists() else None
    L = [f"Screening set: {S['n_covers']} covers ({S['n_boss']} BOSSbase pilot, {S['n_bows2']} BOWS2).", ""]
    # Table 9
    L += ["**Table 9 [screening].** E1: public cover syndrome against uniform strings (main variant, 1536 B; reduced variant, 1252 B). "
          "The same 20 000 uniform strings are the reference in every cell, so the LLR differences are correlated across cells; "
          "their CIs come from a two-sample bootstrap.", "",
          "| Variant | Bucket | $n$ | LLR (Λ > 0) difference [95 % CI] | upper bound of $|\\cdot|$ | learned distinguisher [95 % CI] | positions: Holm rejections / $\\ell_{\\hat c}$; max $|z|$; sd($z$) | pooled byte χ² $p$ |",
          "|---|---|---|---|---|---|---|---|"]
    for k, r in S["E1"].items():
        _, kb, b = k.split("_")
        lc = r["learned_ci95"]
        L.append(f"| {kb[1:]} B | {b[1:]} | {r['n']} | {r['llr_fixed_diff']:+.4f} {ci(r['llr_fixed_ci95'], 4)} | {r['llr_fixed_upper95_abs']:.4f} | "
                 f"{r['learned_adv']:+.4f} [{lc[0]:+.4f}, {lc[1]:+.4f}] | {r['pos_rejections_holm']} / {r['n_positions']}; "
                 f"{abs(r['pos_max_z']):.2f} (bit {r['pos_argmax']}); {r['pos_z_sd']:.3f} | {r['byte_chi2_p_pooled']:.3f} |")
    # Table 10
    L += ["", "**Table 10 [screening].** E2/E3: detectability by warden class (5-fold cross-validation over covers; 95 % cover-cluster bootstrap CIs).", "",
          "| α | System | $n$ | $P_E$ W-img | $P_E$ W-A0 | $P_E$ W-pay | $P_E$ W-comb | TPR@5 % W-img | TPR@5 % W-pay | ΔTPR (W-pay − W-img) | $D_{\\rm pub}$ acc. $M$ = 1 / 2 / 4 | change rate |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for key, r in S["E2E3"].items():
        a, n = key.split("/")
        d = r["dTPR_pay_minus_img"]
        da = r["dpub_acc"]
        L.append(f"| {a} | {n} | {r['n']} | " + " | ".join(f"{r[w]['P_E']:.3f} {ci(r[w]['P_E_ci95'])}" for w in ("W-img", "W-A0", "W-pay", "W-comb"))
                 + f" | {r['W-img']['TPR@5']:.3f} {ci(r['W-img']['TPR@5_ci95'])} | {r['W-pay']['TPR@5']:.3f} {ci(r['W-pay']['TPR@5_ci95'])} | "
                 f"{d['dTPR']:+.3f} [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}] | "
                 + " / ".join(f(da.get(str(m))) for m in (1, 2, 4)) + f" | {r['change_rate']:.4f} |")
    L += ["", "Realized test FPR at the validation 5 % threshold (W-img / W-A0 / W-pay / W-comb): " +
          "; ".join(f"{k}: " + " / ".join(f"{r[w]['FPR_test']:.3f}" for w in ("W-img", "W-A0", "W-pay", "W-comb"))
                    for k, r in S["E2E3"].items()) + "."]
    L += ["", "ΔTPR by source (stop criterion (b) of the decision round):", ""]
    L += ["| α | System | BOSSbase ΔTPR [95 % CI] | BOWS2 ΔTPR [95 % CI] |", "|---|---|---|---|"]
    for key, r in S["E2E3"].items():
        a, n = key.split("/")
        bs = r.get("dTPR_by_source", {})
        cells = []
        for sname in ("boss", "bows2"):
            d = bs.get(sname)
            cells.append("–" if d is None else f"{d['dTPR']:+.3f} [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}]")
        L.append(f"| {a} | {n} | {cells[0]} | {cells[1]} |")
    L += ["", "Rejection rates of generic tests (monobit, runs, byte χ²) at the 1 % level on the extracted stego strings:", "",
          "| α | System | monobit | runs | byte χ² |", "|---|---|---|---|---|"]
    for key, r in S["E2E3"].items():
        a, n = key.split("/")
        g = r["generic_reject_rate@1%"]
        L.append(f"| {a} | {n} | {g['monobit']:.3f} | {g['runs']:.3f} | {g['byte_chi2']:.3f} |")
    # Table 11
    L += ["", "**Table 11 [screening].** Paired TOST on $P_E$ (δ = 0.02): difference with 90 % and 95 % CIs.", "",
          "| α | Comparison | Warden | Δ$P_E$ | 90 % CI | 95 % CI | Verdict |", "|---|---|---|---|---|---|---|"]
    for k, t in S["TOST"].items():
        a, comp, w = k.split("/")
        L.append(f"| {a} | {comp} | {w} | {t['diff']:+.4f} | [{t['ci90'][0]:+.4f}, {t['ci90'][1]:+.4f}] | [{t['ci95'][0]:+.4f}, {t['ci95'][1]:+.4f}] | {t['verdict']} |")
    L += ["", "Strongest warden chosen on validation, per system and rate: " +
          "; ".join(f"{k}: {r['strongest_on_val']}" for k, r in S["E2E3"].items()) + "."]
    # Table 12
    A = S["alpha_star"]
    L += ["", "**Table 12.** RQ3: detectability budget, capacity, cost.", "",
          f"- $\\alpha^\\ast$ [screening, SPAM + FLD] = {f(A['alpha_star_i'])} (95 % CI {ci(A['ci95'])}); $P_E$(W-img, REF-U) at α = 0.1 / 0.2 / 0.4: "
          + " / ".join(f(v) for v in A["P_E_W-img_REF-U"].values())
          + "; condition (ii), TPR@5 % W-comb PQ-PKS − REF-U: " + " / ".join(f"{v:+.3f}" for v in A["cond_ii_TPR_diff_W-comb"].values()) + "."]
    pf = S["p_fail"]
    L += [f"- STC failures: {pf['failures']} in {pf['embeddings']} embeddings; upper 95 % bound per cover {f(pf['upper95_per_cover'], 4)}.",
          f"- Decoding spot checks of PQ-PKS stego images: {S['decode_checks']['ok']} / {S['decode_checks']['n']}."]
    L += ["", "Useful message per image (bytes) at $\\alpha^\\ast$, main Kemeleon variant, 512×512 covers. $N^\\ast_\\sigma$ is computed against "
          "single-image sending with the same ladder and, in brackets, against the better single-image option over the ladders; "
          "for unsigned sessions the break-even is trivially $N=2$. `default`: one power-of-two ladder for every image (design D0 of the "
          "screening profile); `merged`: the frozen profile, later images may also use $B+1536$ (design D5/D6).", "",
          "| Mode | Buckets | " + " | ".join(f"N = {n}" for n in (1, 2, 4, 8, 16, 32)) + " | $N^\\ast_\\sigma$ |",
          "|---|---|" + "---|" * 7]
    for key, cur in E5["curves"].items():
        k, b, sname = key.split("|")
        if k != "Kemeleon main" or b not in ("default", "merged"):
            continue
        bp = cur["bits_per_image"]
        L.append(f"| {sname if sname != 'none' else 'unsigned'} | {b} | " + " | ".join(f"{bp[str(n)] / 8:.0f} B" if bp[str(n)] is not None else "–" for n in (1, 2, 4, 8, 16, 32))
                 + (f" | {cur.get('N_star_sigma', '–')} ({cur.get('N_star_sigma_vs_best_single', '–')}) |" if sname != "none" else " | trivial |"))
    L.append(f"\nSingle-image unsigned capacity at α* (default buckets): {E5['curves']['Kemeleon main|default|none']['single_unsigned'] / 8:.0f} B; "
             f"merged ladder: {E5['curves']['Kemeleon main|merged|none']['single_unsigned'] / 8:.0f} B (same first-image ladder).")
    pl = ROOT / "results" / "e5_budget" / "e5_ladder.json"
    if pl.exists():
        LD = json.loads(pl.read_text())
        L += ["", "Bucket ladder and signature placement (analytic, 512×512, main variant). $\\alpha_{\\min}$: smallest rate at "
              "which a signed session is feasible; worst $N^\\ast_\\sigma$ over the feasible range up to α = 0.30.", "",
              "| Design | buckets (first / later) | α of a later image with 64 B | receiver trial bound | $\\alpha_{\\min}$ ML-DSA-65 / ML-DSA-44 / FN-DSA-512 | worst $N^\\ast_\\sigma$ (65 / 44 / FN) | $N^\\ast_\\sigma$ at $\\alpha^\\ast$ (65) |",
              "|---|---|---|---|---|---|---|"]
        for k, r in LD["designs"].items():
            sg = r["sig"]
            L.append(f"| {k}: {r['label']} | {r['n_buckets']} / {r['n_next_buckets']} | {r['alpha_later_image_64B']:.4f} | "
                     f"{r['receiver_trials_bound']} | "
                     + " / ".join(f(sg[x]["alpha_min"], 4) for x in ("ML-DSA-65", "ML-DSA-44", "FN-DSA-512")) + " | "
                     + " / ".join(str(sg[x]["N_star_worst_on_feasible_range"]) for x in ("ML-DSA-65", "ML-DSA-44", "FN-DSA-512"))
                     + f" | {sg['ML-DSA-65']['N_star_at_alpha_star']} |")
        L.append(f"\nDecision: {LD['decision']['ladder']}; σ placement now {LD['decision']['sig_placement_default_now']}; rule: {LD['decision']['rule']}.")
    if E6:
        L += ["", "| Step (E6, reduced run, Python reference implementation) | median (ms) | IQR (ms) | reps |", "|---|---|---|---|"]
        for k, v in E6.items():
            if isinstance(v, dict) and "median_ms" in v:
                lab = k.replace("|", "\\|")                       # "|B|" in a label would break the Markdown table
                L.append(f"| {lab} | {v['median_ms']:.2f} | {v['iqr_ms']:.2f} | {v['reps']} |")
        L.append(f"\nShare of cryptography in Send (unsigned, 8 image keys precomputed): {100 * E6['H6 crypto share of Send (unsigned, N=8 key precompute)']:.2f} %; "
                 f"Kemeleon-R mean attempts: {E6['Kemeleon-R attempts mean']:.3f}.")
    if E4:
        n = E4["normal"]
        L += ["", "**Table S1 [scaled E4].**", "",
              f"- Normal send/receive: {n['images']} images, {n['errors']} errors (upper 95 % bound {f(n['upper95_error'], 5)}), "
              f"{n['inadmissible']} inadmissible, {n['embed_failures']} embedding failures.",
              "", "| Class | Trials | Expected outcome observed | Rate |", "|---|---|---|---|"]
        for k, v in E4["tamper"].items():
            L.append(f"| {k} | {v['trials']} | {v['expected_outcome']} | {f(v['rate'])} |")
        fa = E4["false_alarm"]
        L.append(f"\nFalse alarms: {fa['accepted']} accepted among {fa['cover_images']} cover images with {fa['aead_trials']} AEAD trials "
                 f"(mean {fa['trials_per_image_mean']:.0f} per image, bound {fa['trials_per_image_max_bound']}); upper 95 % bound per trial "
                 f"{f(fa['upper95_per_trial'], 7)}; median receiver time {fa['receiver_time_s_median']:.2f} s per image (256×256).")
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main(*(Path(a) for a in sys.argv[1:3]))
