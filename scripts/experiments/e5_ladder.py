"""Design stage A, steps 2-4: choice of the bucket ladder and of the signature placement with the analytic E5.

Compared designs (512x512 images, main Kemeleon variant |c_hat| = 1536 B, the same number of buckets for session-start and later images):
  D0  common powers-of-2 ladder, sigma in image 1            (earlier profile)
  D1  shifted powers-of-2 ladder, sigma in image 1           (later images use B + 1536)
  D2  shifted powers-of-2 ladder, sigma in image 2 (A5b)
  D3  shifted sqrt(2) ladder (11 buckets), sigma in image 1
  D4  shifted sqrt(2) ladder (11 buckets), sigma in image 2 (A5b)
  D5  merged powers-of-2 ladder: later images use B U (B + 1536) (11 buckets), sigma in image 1
  D6  merged powers-of-2 ladder, sigma in image 2 (A5b)
Signatures: ML-DSA-65 (3309 B, default), ML-DSA-44 (2420 B), FN-DSA-512 (666 B, analytic only, FIPS 206 not yet published).

Criteria (recorded before computing):
  (i)  alpha_min(sigma): the smallest alpha at which a session signed with ML-DSA-65 is feasible on a 512x512 image (smaller is better);
  (ii) N*_sigma <= 32 at every alpha on the grid [alpha_min(sigma), 0.30] (step 0.005);
  (iii) |B| <= 12. Among the designs meeting (ii)-(iii), choose the smallest alpha_min; on a tie, fewer buckets.
Finding during implementation (additional, criteria unchanged): the shifted ladder forces a short message in a later image to be
padded to 256 + 1536 B, i.e. the minimum embedding rate of a later image rises from 0.008 to 0.055; the merged ladder keeps the
small buckets, at the cost of 11 instead of 6 bucket trials at the receiver for later images. Both quantities are reported next
to the criteria.
Note: below alpha ~ 0.102 no 512x512 image can hold an ML-DSA-65 sigma (3309 B + frame header + tag) with any ladder; (ii) is
therefore evaluated on the feasible range only, and alpha_min is reported as a result.
N*_sigma: the smallest N at which a signed session carries on average at least as much as a single unsigned image WITH THE SAME LADDER.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
N_PIX = 512 * 512
TAG, H_START, H_NEXT, KEM = 16, 14, 6, 1536
SIG = {"ML-DSA-65": 3309, "ML-DSA-44": 2420, "FN-DSA-512": 666}


def pow2():
    return [256, 512, 1024, 2048, 4096, 8192]


def sqrt2():
    return sorted({int(round(256 * 2 ** (k / 2))) for k in range(11)})     # 256 … 8192, 11 bucket


DESIGNS = {
    "D0": {"ladder": "same", "start": pow2(), "sig": "image1", "label": "same ladder, powers of two, σ in image 1 (old)"},
    "D1": {"ladder": "shifted", "start": pow2(), "sig": "image1", "label": "shifted, powers of two, σ in image 1"},
    "D2": {"ladder": "shifted", "start": pow2(), "sig": "image2", "label": "shifted, powers of two, σ in image 2 (A5b)"},
    "D3": {"ladder": "shifted", "start": sqrt2(), "sig": "image1", "label": "shifted, √2 spacing, σ in image 1"},
    "D4": {"ladder": "shifted", "start": sqrt2(), "sig": "image2", "label": "shifted, √2 spacing, σ in image 2 (A5b)"},
    "D5": {"ladder": "merged", "start": pow2(), "sig": "image1", "label": "merged, powers of two, σ in image 1"},
    "D6": {"ladder": "merged", "start": pow2(), "sig": "image2", "label": "merged, powers of two, σ in image 2 (A5b)"},
}
W, S_MAX, P_MAX = 8, 16, 16


def ladders(d):
    s = d["start"]
    if d["ladder"] == "same":
        return s, list(s)
    sh = [b + KEM for b in s]
    return s, (sh if d["ladder"] == "shifted" else sorted(set(s) | set(sh)))


def best(bks, limit, overhead):
    """Largest message (bytes) in the largest bucket <= limit, after subtracting the overhead; None if there is none."""
    ok = [b for b in bks if b <= limit and b - overhead >= 0]
    return max(ok) - overhead if ok else None


def capacities(d, alpha, sig_bytes):
    budget = math.floor(alpha * N_PIX / 8)
    S, Nx = ladders(d)
    lim1, limn = budget - TAG - KEM, budget - TAG
    single = best(S, lim1, H_START)
    nxt = best(Nx, limn, H_NEXT)
    if d["sig"] == "image1":
        m1, m2 = best(S, lim1, H_START + sig_bytes), nxt
    else:
        m1, m2 = single, best(Nx, limn, H_NEXT + sig_bytes)
    return single, m1, m2, nxt


def per_image(d, alpha, sig_bytes, N):
    single, m1, m2, nxt = capacities(d, alpha, sig_bytes)
    if N == 1:
        return m1 if d["sig"] == "image1" else None
    if None in (m1, m2, nxt):
        return None
    return (m1 + m2 + (N - 2) * nxt) / N


def n_star(d, alpha, sig_bytes, nmax=4096):
    single = capacities(d, alpha, sig_bytes)[0]
    if single is None:
        return None
    for N in range(1, nmax + 1):
        v = per_image(d, alpha, sig_bytes, N)
        if v is not None and v >= single:
            return N
    return None


def alpha_min(d, sig_bytes):
    for k in range(200, 4001):
        a = k / 10000
        if any(per_image(d, a, sig_bytes, N) is not None for N in (1, 2)):
            return a
    return None


def main(alpha_star=None):
    summ = json.loads((ROOT / "results" / "archive" / "pre-confirm" / "screen_v1" / "summary.json").read_text())
    alpha_star = alpha_star or summ["alpha_star"]["alpha_star_i"]
    out = {"alpha_star_screening": alpha_star, "designs": {}, "criterion": "see module docstring"}
    for key, d in DESIGNS.items():
        S, Nx = ladders(d)
        short_next = min(b for b in Nx if b >= H_NEXT + 64)               # 64-byte message in a later image
        rec = {"label": d["label"], "n_buckets": len(d["start"]), "n_next_buckets": len(Nx), "start_buckets": S,
               "next_buckets": Nx, "sig": {},
               "alpha_later_image_64B": 8 * (short_next + TAG) / N_PIX,
               "receiver_trials_bound": len(S) + len(Nx) * ((2 * W - 1) * S_MAX + (P_MAX if d["sig"] == "image2" else 0))}
        for sname, sb in SIG.items():
            amin = alpha_min(d, sb)
            grid = [round(amin + 0.005 * k, 4) for k in range(int((0.30 - amin) / 0.005) + 1)] if amin else []
            ns = {str(a): n_star(d, a, sb) for a in grid}
            worst = None if not ns or any(v is None for v in ns.values()) else max(ns.values())
            rec["sig"][sname] = {"alpha_min": amin, "N_star_worst_on_feasible_range": worst,
                                 "N_star_at_alpha_star": n_star(d, alpha_star, sb), "N_star_grid": ns}
        rec["single_unsigned_B"] = {f"{a:.2f}": capacities(d, a, 0)[0] for a in [k / 100 for k in range(8, 41)]}
        rec["single_unsigned_at_alpha_star_B"] = capacities(d, alpha_star, 0)[0]
        rec["per_image_at_alpha_star_ML-DSA-65"] = {N: per_image(d, alpha_star, SIG["ML-DSA-65"], N) for N in (1, 2, 4, 8, 16, 32)}
        rec["per_image_at_alpha_star_unsigned_session"] = {
            N: (None if N == 1 else (capacities(d, alpha_star, 0)[0] + (N - 1) * capacities(d, alpha_star, 0)[3]) / N)
            for N in (1, 2, 4, 8, 16, 32)}
        r65 = rec["sig"]["ML-DSA-65"]
        rec["meets_ii_iii"] = bool(r65["N_star_worst_on_feasible_range"] is not None
                                   and r65["N_star_worst_on_feasible_range"] <= 32
                                   and max(len(d["start"]), rec["n_next_buckets"]) <= 12)
        out["designs"][key] = rec
    ok = [k for k, r in out["designs"].items() if r["meets_ii_iii"]]
    out["meets_criterion"] = ok
    out["best_by_alpha_min"] = (min(ok, key=lambda k: (out["designs"][k]["sig"]["ML-DSA-65"]["alpha_min"],
                                                       out["designs"][k]["n_buckets"])) if ok else None)
    # Decision: shifted powers-of-2 ladder (D1/D2 share the ladder, 6 buckets). The placement of sigma follows the pre-recorded rule:
    # A5b (D2) becomes the default only if the confirmatory alpha* is below alpha_min of D1, because A5b has a cost the criteria do
    # not measure: m1 is held until image 2 and the receiver needs an extra table of pending sessions.
    a1 = out["designs"]["D1"]["sig"]["ML-DSA-65"]["alpha_min"]
    out["decision"] = {"ladder": "merged, powers of two (D5/D6): same capacity as shifted at full messages, keeps "
                                 "short later messages cheap; receiver bound 6 + 11*(2W-1)*S_max = 2646 trials",
                       "sig_placement_default_now": "image1",
                       "rule": f"sig_placement = image2 (A5b) if confirmatory alpha* < {a1} else image1",
                       "alpha_min_D1": a1, "alpha_min_D2": out["designs"]["D2"]["sig"]["ML-DSA-65"]["alpha_min"]}
    p = ROOT / "results" / "e5_budget" / "e5_ladder.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1))
    for k, r in out["designs"].items():
        s = r["sig"]
        print(k, r["label"], "| |B| =", r["n_buckets"], "| α_min 65/44/FN:",
              s["ML-DSA-65"]["alpha_min"], s["ML-DSA-44"]["alpha_min"], s["FN-DSA-512"]["alpha_min"],
              "| worst N* 65/44/FN:", s["ML-DSA-65"]["N_star_worst_on_feasible_range"],
              s["ML-DSA-44"]["N_star_worst_on_feasible_range"], s["FN-DSA-512"]["N_star_worst_on_feasible_range"],
              "| N*(α*) 65:", s["ML-DSA-65"]["N_star_at_alpha_star"],
              "| α later 64B:", round(r["alpha_later_image_64B"], 4), "| trials:", r["receiver_trials_bound"],
              "| meets:", r["meets_ii_iii"])
    print("meets:", ok, "| best by α_min:", out["best_by_alpha_min"], "| decision:", out["decision"])


if __name__ == "__main__":
    main()
