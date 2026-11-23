"""Apply the two pre-registered rules to the confirmatory alpha* (paper, Section 7.8). There are no free parameters.

A5b rule (registered 05/11/2026): sigma goes to image 2 if alpha* < alpha_min(ML-DSA-65, sigma in image 1) = 0.1724, otherwise
to image 1.
Signature selection rule (registered 06/11/2026, before any confirmatory E2/E3 data): for the chosen placement of sigma,
recommend the validated scheme with the highest security level such that alpha_min(scheme, placement) <= alpha*, in the order
ML-DSA-65 (NIST level 3), then ML-DSA-44 (level 2). If no scheme fits: no recommendation; reported as a finding; FN-DSA-512 is
discussed analytically only.
  - alpha* = None (a condition already fails at the smallest alpha of the grid) counts as smaller than every threshold.
  - Direct check: at the operating rate of a scheme with sigma in image 2 (the confirmatory grid has alpha = 0.0787 and 0.1099),
    conditions (i) P_E(W-img, REF-U) >= 0.45 and (ii) TPR@5 %(W-comb, PQ-PKS) - TPR@5 %(W-comb, REF-U) <= 0.02 must both hold.
    If the conclusion from the interpolated alpha* differs from the direct check, the scheme with the lower operating rate is
    chosen (conservative) and the disagreement is recorded.
  - The decision uses the point estimate of alpha*; thresholds inside the 95 % CI of alpha*_i are listed for the paper.
alpha_min is read from results/e5_budget/e5_ladder.json (ladder "merged": D5 with sigma in image 1, D6 with sigma in image 2).
SRNet: W-img is the strongest warden, so at the rates with SRNet (0.2 and 0.4) P_E(W-img, REF-U) = min(SRM, SRNet), and alpha*_i
is recomputed with the same interpolation. results/srnet/confirm_srnet_summary.json is read when present; the CI of alpha*_i
remains that of SRM.

    python scripts/confirm/apply_rules.py                      # reads results/confirm_v1/summary.json
    python scripts/confirm/apply_rules.py --summary <file>     # another summary file (e.g. a run with --reps 3)
    python scripts/confirm/apply_rules.py --no-srnet           # SRM only (while SRNet is not finished; recorded in the output)
Output: results/confirm_v1/decisions.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LADDER = ROOT / "results" / "e5_budget" / "e5_ladder.json"
SRNET = ROOT / "results" / "srnet" / "confirm_srnet_summary.json"
G4 = ROOT / "results" / "g4" / "g4_summary.json"
ORDER = ("ML-DSA-65", "ML-DSA-44")             # validated schemes, by decreasing security level
ANALYTIC = "FN-DSA-512"
DESIGN = {"image1": "D5", "image2": "D6"}
COND_I, COND_II = 0.45, 0.02
MIN_FIRST_IMAGE = 0.0552        # smallest session-start image: Kemeleon 1536 B + bucket 256 B + tag 16 B on a 512x512 image


def interp_crossing(xs, ys, level: float) -> float | None:
    """As pqpks.analysis.stats.interp_crossing (equality checked in tests/test_apply_rules.py)."""
    if ys[0] < level:
        return None
    for k in range(len(xs) - 1):
        if ys[k] >= level > ys[k + 1]:
            t = (ys[k] - level) / (ys[k] - ys[k + 1])
            return float(xs[k] + t * (xs[k + 1] - xs[k]))
    return float(xs[-1])


def with_srnet(astar: dict, srnet: dict | None, level: float = COND_I) -> tuple[dict, dict]:
    """Include SRNet in condition (i): P_E(W-img, REF-U) = min(SRM, SRNet) at the rates with SRNet; recompute alpha*_i and alpha*."""
    if srnet is None and level == COND_I:
        return astar, {"included": False}
    pe = dict(astar["P_E_W-img_REF-U"])
    used = {}
    for al, row in (srnet or {}).get("alphas", {}).items():
        k = next((k for k in pe if abs(float(k) - float(al)) < 1e-9), None)
        if k is not None and "cond_i_SRNet" in row:
            used[k] = row["cond_i_SRNet"]["P_E_REF-U"]
            pe[k] = min(pe[k], used[k])
    xs = sorted(pe, key=float)
    a_i = interp_crossing([float(k) for k in xs], [pe[k] for k in xs], level)
    a_ii = astar.get("alpha_star_ii")
    a = None if a_i is None or a_ii is None else min(a_i, a_ii)
    new = dict(astar) | {"P_E_W-img_REF-U": pe, "alpha_star_i": a_i, "alpha_star": a}
    return new, {"included": bool(used), "SRNet_P_E_REF-U": used, "alpha_star_SRM_only": astar.get("alpha_star"),
                 "alpha_star_with_SRNet": a, "changed": a != astar.get("alpha_star"),
                 "note": "the 95 % CI of alpha*_i is that of SRM; SRNet is measured at 0.2 and 0.4 only"}


def alpha_min_table(ladder: dict) -> dict:
    return {place: {s: v["alpha_min"] for s, v in ladder["designs"][d]["sig"].items()} for place, d in DESIGN.items()}


def direct_check(astar: dict, alpha: float, level: float = COND_I) -> dict | None:
    """Conditions (i) and (ii) at one grid point; None if the grid lacks that point."""
    pe = astar.get("P_E_W-img_REF-U", {})
    cond = astar.get("cond_ii_TPR_diff_W-comb", {})
    key = next((k for k in pe if abs(float(k) - alpha) < 1e-9), None)
    if key is None or key not in cond:
        return None
    return {"alpha": float(key), "P_E_W-img_REF-U": pe[key], "cond_i": pe[key] >= level,
            "TPR_diff_W-comb": cond[key], "cond_ii": cond[key] <= COND_II, "holds": pe[key] >= level and cond[key] <= COND_II}


def decide(summary: dict, ladder: dict, srnet: dict | None = None, level: float = COND_I) -> dict:
    astar, srnet_info = with_srnet(summary["alpha_star"], srnet, level)
    a = astar.get("alpha_star")
    a_eff = float("-inf") if a is None else float(a)
    amin = alpha_min_table(ladder)
    a5b_threshold = amin["image1"]["ML-DSA-65"]
    place = "image2" if a_eff < a5b_threshold else "image1"
    by_alpha = next((s for s in ORDER if amin[place][s] <= a_eff), None)
    checks = {s: direct_check(astar, amin["image2"][s], level) for s in ORDER}
    note = []
    final = by_alpha
    if place == "image2":
        by_direct = next((s for s in ORDER if checks[s] is not None and checks[s]["holds"]), None)
        if any(checks[s] is None for s in ORDER):
            note.append("the grid lacks one of the operating rates 0.0787 and 0.1099: only the interpolated alpha* is used")
            by_direct = by_alpha
        if by_direct != by_alpha:
            rank = {s: i for i, s in enumerate(ORDER)} | {None: len(ORDER)}
            final = max((by_alpha, by_direct), key=lambda s: rank[s])      # the scheme with the lower operating rate
            note.append(f"the interpolated alpha* selects {by_alpha}, the direct check selects {by_direct}: {final} is taken (conservative)")
    else:
        by_direct = None
    first = direct_check(astar, MIN_FIRST_IMAGE, level)
    below_min = a_eff < MIN_FIRST_IMAGE
    ci = astar.get("alpha_star_i_ci95")
    thresholds = {f"A5b ({a5b_threshold})": a5b_threshold} | {f"{s}/image2 ({amin['image2'][s]})": amin["image2"][s] for s in ORDER}
    in_ci = [name for name, t in thresholds.items() if ci and ci[0] <= t <= ci[1]]
    return {
        "alpha_star": a, "alpha_star_i_ci95": ci, "alpha_star_i": astar.get("alpha_star_i"), "alpha_star_ii": astar.get("alpha_star_ii"),
        "rule_A5b": {"threshold": a5b_threshold, "sig_placement": place},
        "rule_signature": {"by_alpha_star": by_alpha, "by_direct_check": by_direct, "recommended": final,
                           "direct_checks": checks, "notes": note},
        "alpha_min": amin,
        "srnet": srnet_info,
        "analytic_only": {ANALYTIC: {p: amin[p].get(ANALYTIC) for p in amin}},
        "minimal_first_image": {"alpha": MIN_FIRST_IMAGE, "direct_check": first, "alpha_star_below": below_min},
        "thresholds_inside_ci95_of_alpha_star_i": in_ci,
        "statement": (f"sigma in {'image 2 (A5b)' if place == 'image2' else 'image 1'}; "
                      + (f"recommended scheme {final}" if final else
                         "no validated signature scheme fits a 512x512 image at a safe rate: reported as a finding")
                      + (f"; alpha* below {MIN_FIRST_IMAGE} (smallest session-start image): no PQ-PKS session-start image, "
                         "signed or not, meets the alpha* criterion on a 512x512 image" if below_min else "")
                      + (f"; thresholds inside the 95 % CI of alpha*: {', '.join(in_ci)}" if in_ci else "")),
    }


def g4_offset(g4: dict) -> float:
    """Mean deviation P_E(ours) - P_E(published) over the G4 cells with reference values."""
    d = [r["P_E_mean"] - r["reference"]["P_E"] for r in g4["results"] if r.get("reference")]
    if not d:
        raise ValueError("g4_summary.json has no cell with reference values")
    return sum(d) / len(d)


def sensitivity(summary: dict, ladder: dict, srnet: dict | None, g4_path: Path) -> dict:
    """Pre-registered sensitivity analysis: condition (i) uses the threshold 0.45 + the measured G4 deviation. It does not replace
    the main conclusion (still 0.45); it only reports how alpha* and the rules would change if the systematic weakness of our SRM were compensated."""
    off = g4_offset(json.loads(Path(g4_path).read_text(encoding="utf-8")))
    d = decide(summary, ladder, srnet, COND_I + off)
    return {"offset": off, "threshold": COND_I + off, "alpha_star": d["alpha_star"], "alpha_star_i": d["alpha_star_i"],
            "rule_A5b": d["rule_A5b"], "recommended": d["rule_signature"]["recommended"],
            "minimal_first_image": d["minimal_first_image"], "statement": d["statement"],
            "note": "sensitivity analysis, not the main conclusion"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default=str(ROOT / "results" / "confirm_v1" / "summary.json"))
    ap.add_argument("--out", default=str(ROOT / "results" / "confirm_v1" / "decisions.json"))
    ap.add_argument("--no-srnet", action="store_true")
    a = ap.parse_args()
    srnet = None if a.no_srnet or not SRNET.exists() else json.loads(SRNET.read_text(encoding="utf-8"))
    res = decide(json.loads(Path(a.summary).read_text(encoding="utf-8")), json.loads(LADDER.read_text(encoding="utf-8")), srnet)
    res["inputs"] = {"summary": a.summary, "ladder": str(LADDER), "srnet": None if srnet is None else str(SRNET)}
    if srnet is None:
        print("SRNet not included: this conclusion rests on SRM only", flush=True)
    summ, ladder = json.loads(Path(a.summary).read_text(encoding="utf-8")), json.loads(LADDER.read_text(encoding="utf-8"))
    sens = sensitivity(summ, ladder, srnet, G4) if G4.exists() else None
    if sens is not None:
        res["sensitivity_G4_offset"] = sens
        print("sensitivity (threshold %.4f): %s" % (sens["threshold"], sens["statement"]), flush=True)
    Path(a.out).write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    print(res["statement"])


if __name__ == "__main__":
    main()
