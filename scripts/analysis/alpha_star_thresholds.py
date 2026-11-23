"""Sensitivity of the detectability budget alpha* to the threshold of condition (i).

Post-hoc analysis added on 23 Nov 2026 in answer to a review comment. It is not part of the registered protocol and
changes no registered decision: the conclusions of the paper use the threshold 0.45; the registered sensitivity
threshold 0.45 + mean G4 offset (about 0.4584) is reported by scripts/confirm/apply_rules.py.

The script reads the confirmatory summary and recomputes alpha*_i for several thresholds from the measured curve
P_E(W-img, REF-U) with the same interpolation as the registered analysis (pqpks.analysis.stats.interp_crossing).
W-img takes the smaller P_E of SRM and SRNet at the rates where SRNet was run (rule of the registered protocol),
when an SRNet decisions file is given. No model is trained and no data are resampled.

Run:
    python scripts/analysis/alpha_star_thresholds.py results/confirm_v1/summary.json \
        --srnet results/srnet/decisions_srnet.json --out results/confirm_v1/alpha_star_thresholds.json
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

N_PIX = 512 * 512
THRESHOLDS = (0.35, 0.40, 0.42, 0.44, 0.45)


def interp_crossing(xs, ys, level: float) -> float | None:
    """Largest x on the grid (linear interpolation) with y >= level, for y decreasing in x; None if y[0] < level.

    Same rule as pqpks.analysis.stats.interp_crossing (checked in tests/test_alpha_star_thresholds.py).
    """
    if ys[0] < level:
        return None
    for k in range(len(xs) - 1):
        if ys[k] >= level > ys[k + 1]:
            t = (ys[k] - level) / (ys[k] - ys[k + 1])
            return float(xs[k] + t * (xs[k + 1] - xs[k]))
    return float(xs[-1])


def curve(summary: dict, srnet: dict | None) -> tuple[list[float], list[float], dict]:
    pe = {float(k): float(v) for k, v in summary["alpha_star"]["P_E_W-img_REF-U"].items()}
    used = {}
    if srnet is not None:
        for k, v in srnet.get("srnet", {}).get("SRNet_P_E_REF-U", {}).items():
            a = float(k)
            if a in pe and float(v) < pe[a]:
                used[k] = float(v)
                pe[a] = float(v)
    xs = sorted(pe)
    return xs, [pe[a] for a in xs], used


def analyse(summary: dict, srnet: dict | None, decisions: dict | None = None) -> dict:
    xs, ys, used = curve(summary, srnet)
    thresholds = list(THRESHOLDS)
    sens = None
    if decisions is not None and decisions.get("sensitivity_G4_offset", {}).get("threshold") is not None:
        sens = float(decisions["sensitivity_G4_offset"]["threshold"])
        thresholds.append(sens)
    rows = []
    for th in thresholds:
        a = interp_crossing(xs, ys, th)
        rows.append({"threshold": round(th, 6), "alpha_star_i": a,
                     "budget_bytes": None if a is None else math.floor(a * N_PIX / 8),
                     "first_image_0.0552_meets": ys[0] >= th,
                     "role": "registered sensitivity (E-35)" if sens is not None and th == sens
                     else ("registered" if th == 0.45 else "post hoc")})
    cell = summary["E2E3"]["0.0552/REF-U"]["W-img"]
    return {
        "note": "post-hoc threshold sensitivity of alpha*_i (point estimates); the registered conclusion uses 0.45",
        "grid": xs, "P_E_W-img_REF-U": ys, "SRNet_used": used,
        "minimal_first_image": {"alpha": xs[0], "P_E": cell["P_E"], "P_E_ci95": cell["P_E_ci95"]},
        "alpha_star_i_registered": summary["alpha_star"]["alpha_star_i"],
        "alpha_star_i_ci95_registered": summary["alpha_star"]["alpha_star_i_ci95"],
        "bootstrap_share_without_alpha_star_on_grid": summary["alpha_star"].get("alpha_star_i_boot_below_grid"),
        "rows": rows,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("summary", type=Path)
    ap.add_argument("--srnet", type=Path, help="decisions_srnet.json (adds SRNet to W-img at 0.2 and 0.4)")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args(argv)
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    srnet = json.loads(args.srnet.read_text(encoding="utf-8")) if args.srnet else None
    res = analyse(summary, srnet, srnet)
    res["inputs"] = {"summary": args.summary.name, "srnet": args.srnet.name if args.srnet else None}
    for r in res["rows"]:
        a = "none on the grid" if r["alpha_star_i"] is None else f"{r['alpha_star_i']:.4f} bpp ({r['budget_bytes']} B)"
        print(f"threshold {r['threshold']:.4f}: alpha*_i = {a}  [{r['role']}]")
    m = res["minimal_first_image"]
    print(f"smallest first image alpha={m['alpha']}: P_E = {m['P_E']:.5f}, 95% CI {m['P_E_ci95']}")
    print(f"bootstrap replicates without alpha* on the grid: {res['bootstrap_share_without_alpha_star_on_grid']}")
    if args.out:
        args.out.write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
        print(f"written {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
