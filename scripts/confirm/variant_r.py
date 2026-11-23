"""Phase E (EXPLORATORY): alpha* and the signature rule for the rejection-sampled Kemeleon variant (A4, |c_hat| = 1252 B).

Why: the confirmatory alpha* of the main variant, 0.0596 [0.0562, 0.0623], lies close to its smallest session-start image 0.0552
(1536 + 256 + 16 B), and under the sensitivity analysis even that image fails. Variant A4 is 284 B shorter: its smallest
session-start image is at 0.04651 (1252 + 256 + 16 B). Step confirm_r (screen_run --plan confirm_r) generates PQ-PKS-R and REF-U-1252
at that alpha on the 20 000 confirmatory covers; this file merges that cell with the grid of the main variant to compute alpha*_R and
applies the decision rules again with |c_hat| = 1252 B.

Assumptions (recorded in the output): for alpha >= 0.0552, P_E(W-img) of REF-U (1536 uniform bytes + uniform body) replaces that of
REF-U-1252 with the same total number of bytes - the two controls differ only in how the uniform string is split between A0 and A1,
with the same embedder and the same total number of bits; condition (ii) at those alpha values is taken from PQ-PKS (main variant)
because PQ-PKS-R was not measured there on the 20 000 covers (S2 on 2000 covers: equivalent).
There is no joint bootstrap for alpha*_R (two separate runs): only the point estimate, the CI of cell 0.04651 and the direct check are reported.

    python scripts/confirm/variant_r.py [--main results/confirm_v1/summary.json] [--r results/confirm_r/summary.json]
                                        [--out results/confirm_r/decisions_r.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "confirm"))
sys.path.insert(0, str(ROOT / "scripts" / "experiments"))

import apply_rules as AR  # noqa: E402
import e5_ladder as EL  # noqa: E402

KEM_R = 1252
MIN_FIRST_IMAGE_R = 0.04651
SYS_R, REF_R = "PQ-PKS-R", "REF-U-1252"


def ladder_for_kem(kem: int, alpha_star: float) -> dict:
    """e5_ladder.json recomputed with |c_hat| = kem (the e5_ladder functions use the global constant KEM; it is set temporarily and restored)."""
    keep = EL.KEM
    EL.KEM = kem
    try:
        out = {"designs": {}}
        for key, d in EL.DESIGNS.items():
            rec = {"label": d["label"], "sig": {}}
            for sname, sb in EL.SIG.items():
                amin = EL.alpha_min(d, sb)
                rec["sig"][sname] = {"alpha_min": amin, "N_star_at_alpha_star": EL.n_star(d, alpha_star, sb)}
            rec["single_unsigned_at_alpha_star_B"] = EL.capacities(d, alpha_star, 0)[0]
            out["designs"][key] = rec
        return out
    finally:
        EL.KEM = keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--main", default=str(ROOT / "results" / "confirm_v1" / "summary.json"))
    ap.add_argument("--r", default=str(ROOT / "results" / "confirm_r" / "summary.json"))
    ap.add_argument("--out", default=str(ROOT / "results" / "confirm_r" / "decisions_r.json"))
    a = ap.parse_args()
    M = json.loads(Path(a.main).read_text(encoding="utf-8"))
    R = json.loads(Path(a.r).read_text(encoding="utf-8"))
    cell = next(k for k in R["E2E3"] if k.endswith("/" + REF_R))
    al_r = cell.split("/")[0]
    ref = R["E2E3"][cell]
    pq = R["E2E3"][f"{al_r}/{SYS_R}"]
    pe = {al_r: ref["W-img"]["P_E"]} | dict(M["alpha_star"]["P_E_W-img_REF-U"])
    cond = {al_r: pq["W-comb"]["TPR@5"] - ref["W-comb"]["TPR@5"]} | dict(M["alpha_star"]["cond_ii_TPR_diff_W-comb"])
    xs = sorted(pe, key=float)
    a_i = AR.interp_crossing([float(k) for k in xs], [pe[k] for k in xs], AR.COND_I)
    a_ii = AR.interp_crossing([float(k) for k in xs], [-cond[k] for k in xs], -AR.COND_II)   # (ii) <= 0.02 <=> -diff >= -0.02
    a_star = None if a_i is None or a_ii is None else min(a_i, a_ii)
    astar = {"alpha_star_i": a_i, "alpha_star_ii": a_ii, "alpha_star": a_star, "alpha_star_i_ci95": None,
             "P_E_W-img_REF-U": pe, "cond_ii_TPR_diff_W-comb": cond,
             "note": "variant R: cell 0.04651 of REF-U-1252/PQ-PKS-R merged with the grid of the main variant (REF-U, PQ-PKS) at alpha >= 0.0552; "
                     "no joint bootstrap CI"}
    ladder = ladder_for_kem(KEM_R, a_star if a_star is not None else MIN_FIRST_IMAGE_R)
    keep = AR.MIN_FIRST_IMAGE
    AR.MIN_FIRST_IMAGE = MIN_FIRST_IMAGE_R
    try:
        dec = AR.decide({"alpha_star": astar}, ladder, None)
    finally:
        AR.MIN_FIRST_IMAGE = keep
    tost = {k: v for k, v in R.get("TOST", {}).items() if k.startswith(al_r)}
    dec |= {
        "variant": "Kemeleon rejection-sampled (A4), |c_hat| = 1252 B", "exploratory": True,
        "kem_bytes": KEM_R, "cell_alpha": float(al_r),
        "cell": {"P_E_W-img_" + REF_R: ref["W-img"]["P_E"], "P_E_W-img_ci95_" + REF_R: ref["W-img"]["P_E_ci95"],
                 "P_E_W-img_" + SYS_R: pq["W-img"]["P_E"], "P_E_W-pay_" + SYS_R: pq["W-pay"]["P_E"],
                 "dpub_acc_M1_" + SYS_R: pq["dpub_acc_M1"]["acc"], "TOST": tost},
        "alpha_star_main": M["alpha_star"].get("alpha_star"),
        "ladder_R": ladder,
        "assumptions": ["REF-U (1536 uniform bytes) replaces REF-U-1252 at alpha >= 0.0552 (same total bits, same embedder)",
                        "condition (ii) at alpha >= 0.0552 is taken from PQ-PKS (main variant)",
                        "exploratory result, added after the confirmatory alpha* was known; decision D-1 is unchanged (the main variant is the default)"],
    }
    dec["inputs"] = {"main": a.main, "r": a.r}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(dec, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"alpha*_R = {a_star} (i: {a_i}, ii: {a_ii}); cell {al_r}: P_E(W-img, {REF_R}) = {ref['W-img']['P_E']:.4f}; {dec['statement']}")


if __name__ == "__main__":
    main()
