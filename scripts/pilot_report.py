"""Build the tables of the pilot report from results/archive/pre-confirm/pilot_baselines/report.json -> results/archive/pre-confirm/pilot_baselines/TABLES.md.
The hand-written assessment is in results/archive/pre-confirm/pilot_baselines/REPORT.md and quotes its numbers from TABLES.md."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "archive" / "pre-confirm" / "pilot_baselines"
ORDER = ("BL1", "BL2", "BL3", "BL4", "BL4-E", "BL5", "BL5-L", "PQ-PKS", "PQ-PKS-R", "REF-U", "REF-1S")


def f(v, nd=3):
    return "–" if v is None else (f"{v:.{nd}f}".replace(".", ",") if isinstance(v, float) else str(v))


def main():
    r = json.loads((OUT / "report.json").read_text())
    L = ["# Pilot tables (generated from report.json)", ""]
    s = r["setup"]
    L += [f"- Data: {s['dataset']}; W-pay {s['n_pay']} images, W-img {s['n_img']} images; α ∈ {s['alphas']}.",
          f"- Image warden: {s['warden_img']}.", f"- Profile: {s['profile']}.", f"- Run time (seconds, 2 CPU cores): {r.get('elapsed_s')}. The times of section 5 were measured with 2 processes running in parallel.", ""]
    L += ["## 1. Correctness and false alarms", "", "| Check | Pass |", "|---|---|"]
    L += [f"| {k} | {v} |" for k, v in r.get("checks", {}).items()]
    L += ["", "| System | Covers tried | Falsely accepted |", "|---|---|---|"]
    L += [f"| {k} | {v['covers']} | {v['accepted']} |" for k, v in r.get("false_alarm", {}).items()]
    L += ["", "## 2. W-pay: public extraction (P = kem_part, Q = Ext₀(cover)), balanced accuracy", "",
          "| System | LLR Compress₁₀ (95 % CI) | P-256 point | ASCII fraction | Learned (FLD, 5-fold) | Best |", "|---|---|---|---|---|---|"]
    wp = r.get("wpay", {})
    for k in ORDER:
        if k in wp:
            v = wp[k]
            llr = f"{f(v['llr_acc'])} ({f(v['llr_acc_ci95'][0])}–{f(v['llr_acc_ci95'][1])})" if "llr_acc" in v else "–"
            L.append(f"| {k} | {llr} | {f(v['p256_acc'])} | {f(v['ascii_acc'])} | {f(v['learned_acc'])} | **{f(v['best_acc'])}** |")
    v = wp.get("delta_syn_LLR")
    if v:
        L += ["", "δ_syn of the LLR test on cover syndromes (pilot of E1; first 960 B, shared by the 1536 B and 1088 B layouts):", "",
              "| Pr[LLR(Q) > 0] | Pr[LLR(U) > 0] | Difference (95 % CI) | Upper 95 % bound of δ | n |", "|---|---|---|---|---|",
              f"| {f(v['Pr[LLR(Q)>0]'])} | {f(v['Pr[LLR(U)>0]'])} | {f(v['diff'])} ({f(v['diff_ci95'][0])} … {f(v['diff_ci95'][1])}) | {f(v['delta_upper95'])} | {v['n']} |"]
    wi = r.get("wimg", {})
    for a in s["alphas"]:
        L += ["", f"## 3. W-img (SPAM-686 + FLD ensemble), α = {f(a, 1)}", "", "| System | P_E (95 % CI) | AUC | Changed pixels |", "|---|---|---|---|"]
        for k in ORDER:
            v = wi.get(f"{k}:{a}")
            if v:
                L.append(f"| {k} | {f(v['P_E'])} ({f(v['P_E_ci95'][0])}–{f(v['P_E_ci95'][1])}) | {f(v['AUC'])} | {f(v['change_rate'], 4)} |")
        L += ["", "| Paired comparison (a − b) | P_E(a) − P_E(b) | 95 % CI | 90 % CI | bootstrap p | Holm p (14 comparisons) |", "|---|---|---|---|---|---|"]
        for k, v in wi.items():
            if k.startswith("diff:") and k.endswith(f":{a}"):
                L.append(f"| {k[5:-len(str(a)) - 1]} | {f(v['diff'])} | {f(v['ci95'][0])} … {f(v['ci95'][1])} | {f(v['ci90'][0])} … {f(v['ci90'][1])} | {f(v.get('p_boot'))} | {f(v.get('p_holm'))} |")
        t = wi.get(f"tost:PQ-PKS-REF-U:{a}")
        if t:
            L.append(f"\nTOST PQ-PKS ↔ REF-U, δ = 0.02: {'equivalent' if t['equivalent'] else 'equivalence not established'}.")
    L += ["", "## 4. Regime R-a (|m| = 256 B, unsigned, default ladder): α of the session-start image", "",
          "| System | kem_part (B) | Payload (B) | α | α₁ |", "|---|---|---|---|---|"]
    for k, v in r.get("ra_budget", {}).items():
        L.append(f"| {k} | {v['kem_part_B']} | {v['payload_B']} | {f(v['alpha'], 4)} | {f(v.get('alpha1'), 4)} |")
    L += ["", "## 5. Time (median, seconds; send excludes the HILL cost, computed once for every system)", "",
          "| Step | Median (s) |", "|---|---|"]
    L += [f"| {k} | {f(v)} |" for k, v in r.get("timing_median_s", {}).items()]
    if "bl1_payload_bits" in r:
        L += ["", f"BL1: actual payload (median, bits) by α: {r['bl1_payload_bits']}."]
    (OUT / "TABLES.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("ghi", OUT / "TABLES.md")


if __name__ == "__main__":
    main()
