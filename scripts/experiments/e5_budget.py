"""E5 steps 1, 3, 4 (analytic; RQ3.1, H5): bit budget with layout A8, useful bits per image by N at alpha*,
and the break-even point of a signed session.

alpha* comes from the confirmatory results (results/confirm_v1/summary.json) or from --alpha-star. The signature placement comes
from decisions.json (A5b rule) if present, or from --sig-place.
512x512 images; the message is filled up to l_stego,i <= alpha* n (paper, Section 5.5).
- Single image: every image is a session-start image (carries c_hat), unsigned.
- Session, sigma in image 1 (A5a): image 1 carries c_hat and sigma, images 2..N only the body.
- Session, sigma in image 2 (A5b): image 1 carries c_hat, image 2 carries sigma, images 3..N only the body (a signed session needs N >= 2).
- Session, sigma spread over the later images (A5c, EXPLORATORY, not part of the registered design): sigma is split into pieces that
  fill the body of images 2, 3, ...; useful bits = m_1 + (N-1) c - |sigma| when (N-1) c >= |sigma|.
Signatures: ML-DSA-65 (3309 B), ML-DSA-44 (2420 B), FN-DSA-512 (666 B, analytic only; FIPS 206 not yet published).
- N*_sigma: the smallest N at which a signed session gives average useful bits per image >= a single unsigned image.
  (Comparing an unsigned session with a single image is uninformative: a later image always carries more than a session-start
  image, so the break-even point is N = 2 whenever a session-start image is feasible.)
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

N_PIX = 512 * 512
TAG = 16
H_START, H_NEXT = 14, 6
SIG = {"none": 0, "ML-DSA-44": 2420, "ML-DSA-65": 3309, "FN-DSA-512": 666}
PLACES = ("image1", "image2", "span")
KEM = {"Kemeleon main": 1536, "Kemeleon R": 1252}


def buckets_default():
    return [256, 512, 1024, 2048, 4096, 8192]


def buckets_fine(lo=256, hi=16384, ratio=1.25):
    out, b = [], lo
    while b <= hi:
        out.append(int(b))
        b = math.ceil(b * ratio)
    return out


def buckets_merged(kem=1536):
    """Merged ladder of the frozen profile: session-start images use the powers-of-2 ladder, later images use B and B + |c_hat|.
    Returns (first-image ladder, later-image ladder); the functions below accept either one shared list or such a pair."""
    base = buckets_default()
    return base, sorted(set(base) | {b + kem for b in base})


def _split(bks):
    return bks if isinstance(bks, tuple) else (bks, bks)


LADDERS = {"default": buckets_default(), "merged": buckets_merged(), "fine1.25": buckets_fine()}


def max_msg(budget_bytes, start, kem, sig, bks):
    """Largest |m| (bytes) such that the image fits the budget; None if infeasible. `sig` is the number of signature bytes placed in
    this image (the first image under A5a, the second under A5b)."""
    best = None
    for B in _split(bks)[0 if start else 1]:
        total = B + TAG + (kem if start else 0)
        if total > budget_bytes:
            continue
        m = B - (H_START if start else H_NEXT) - sig
        if m >= 0:
            best = m if best is None else max(best, m)
    return best


def bits_per_image(alpha, N, kem, sig, bks, single=False, place="image1"):
    """Average useful bits per image of a session of N images (None if infeasible). place: image1 (A5a), image2 (A5b),
    span (A5c, exploratory)."""
    budget = math.floor(alpha * N_PIX / 8)
    if single:
        m = max_msg(budget, True, kem, sig, bks)
        return None if m is None else 8 * m
    if sig == 0 or place == "image1":
        m1 = max_msg(budget, True, kem, sig, bks)
        if m1 is None:
            return None
        mn = max_msg(budget, False, kem, 0, bks) or 0
        return 8 * (m1 + (N - 1) * mn) / N
    m1 = max_msg(budget, True, kem, 0, bks)
    mn = max_msg(budget, False, kem, 0, bks)
    if m1 is None or mn is None or N < 2:
        return None
    if place == "image2":
        m2 = max_msg(budget, False, kem, sig, bks)
        if m2 is None:
            return None
        return 8 * (m1 + m2 + (N - 2) * mn) / N
    if place == "span":
        if (N - 1) * mn < sig:
            return None
        return 8 * (m1 + (N - 1) * mn - sig) / N
    raise ValueError(place)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha-star", type=float, default=None)
    ap.add_argument("--summary", default=str(ROOT / "results" / "confirm_v1" / "summary.json"),
                    help="summary.json of analyze_confirm (confirmatory) or of the screening round (results/archive/.../screen_v1)")
    ap.add_argument("--decisions", default=str(ROOT / "results" / "confirm_v1" / "decisions.json"),
                    help="decisions.json of apply_rules (placement of sigma by the A5b rule); ignored if absent")
    ap.add_argument("--sig-place", choices=PLACES, default=None, help="force the signature placement (default: from decisions.json, otherwise image1)")
    ap.add_argument("--out", default=str(ROOT / "results" / "e5_budget"))
    ap.add_argument("--name", default="e5_summary.json")
    args = ap.parse_args()
    a_star = args.alpha_star
    src = "argument"
    s = None
    dp = Path(args.decisions)
    if a_star is None and dp.exists() and json.loads(dp.read_text(encoding="utf-8")).get("alpha_star") is not None:
        # alpha* = min(alpha*_i, alpha*_ii) from apply_rules; when the decisions include SRNet (decisions_srnet.json) this is the
        # final alpha*. Earlier versions took alpha_star_i of the summary (equal when (ii) does not bind).
        d = json.loads(dp.read_text(encoding="utf-8"))
        a_star = d["alpha_star"]
        src = ("confirmatory, after merging SRNet (W-img = min(SRM, SRNet) on REF-U, P_E ≥ 0.45)"
               if d.get("srnet", {}).get("included") else "confirmatory (SRM + FLD ensemble, 20 000 covers, W-img on REF-U, P_E ≥ 0.45)")
        if Path(args.summary).exists():                   # the 95 % CI of alpha*_i (SRM) and the sensitivity grid still come from the summary
            s = json.loads(Path(args.summary).read_text())
    if a_star is None:
        s = json.loads(Path(args.summary).read_text())
        a_star = s["alpha_star"]["alpha_star_i"]
        src = ("confirmatory (SRM + FLD ensemble, 20 000 covers, W-img on REF-U, P_E ≥ 0.45)" if s.get("split_mode") == "confirm"
               else "screening (SPAM-686 + FLD, W-img on REF-U, P_E ≥ 0.45)")
    place = args.sig_place
    if place is None and dp.exists():
        place = json.loads(dp.read_text(encoding="utf-8"))["rule_A5b"]["sig_placement"]
    place = place or "image1"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    res = {"alpha_star": a_star, "alpha_star_source": src, "n": N_PIX, "sig_placement": place,
           "alpha_star_i_ci95": (s or {}).get("alpha_star", {}).get("alpha_star_i_ci95") if s else None}
    # A8 budget table (paper, Table 5): equal densities in the two stages
    rows = []
    for m, sig in [(64, "none"), (256, "none"), (256, "ML-DSA-65"), (256, "ML-DSA-44"), (1024, "none"), (4096, "none")]:
        need = H_START + m + SIG[sig]
        B = next((b for b in buckets_default() if need <= b), None)
        tot = 1536 + B + TAG
        n0 = -(-12288 * N_PIX // (8 * tot))
        rows.append({"msg": m, "sig": sig, "bucket": B, "first_bytes": tot, "alpha": 8 * tot / N_PIX,
                     "alpha0": 12288 / n0, "alpha1": 8 * (B + TAG) / (N_PIX - n0), "pad": B - need})
    res["budget_A8"] = rows
    Ns = [1, 2, 4, 8, 16, 32]
    curves = {}
    by_place = {}
    for pl in PLACES:
        cur = {}
        for kname, kem in KEM.items():
            for bname, bks in LADDERS.items():
                single = bits_per_image(a_star, 1, kem, 0, bks, single=True)
                for sname, sig in SIG.items():
                    key = f"{kname}|{bname}|{sname}"
                    cur[key] = {"bits_per_image": {N: bits_per_image(a_star, N, kem, sig, bks, place=pl) for N in Ns},
                                "single_unsigned": single}
                    if sname != "none" and single is not None:
                        nstar = next((N for N in range(1, 4097)
                                      if (v := bits_per_image(a_star, N, kem, sig, bks, place=pl)) is not None and v >= single), None)
                        cur[key]["N_star_sigma"] = nstar
        by_place[pl] = cur
    curves = by_place[place]                       # the old key "<kem>|<ladder>|<sigma>" is kept: tables/figures read the decided placement
    res["curves_by_placement"] = by_place
    res["exploratory"] = {"span": "A5c: sigma spread over the later images - exploratory, not part of the registered design"}
    # Also compare with the BEST single-image option over both bucket ladders (the fine ladder is not always better: at some alpha
    # a power-of-2 bucket fits the budget of the session-start image exactly)
    for kname, kem in KEM.items():
        best = max(v for v in (bits_per_image(a_star, 1, kem, 0, b, single=True) for b in LADDERS.values())
                   if v is not None)
        for bname, bks in LADDERS.items():
            for sname, sig in SIG.items():
                if sname == "none":
                    continue
                curves[f"{kname}|{bname}|{sname}"]["N_star_sigma_vs_best_single"] = next(
                    (N for N in range(1, 4097) if (v := bits_per_image(a_star, N, kem, sig, bks, place=place)) is not None and v >= best), None)
                curves[f"{kname}|{bname}|{sname}"]["best_single_unsigned"] = best
    res["curves"] = curves
    # alpha threshold at which a session-start image with ML-DSA-65 is feasible with the default buckets (main variant)
    need = 1536 + TAG + next(b for b in buckets_default() if b >= H_START + SIG["ML-DSA-65"])
    res["alpha_min_signed_first_image_default"] = 8 * need / N_PIX
    need_f = 1536 + TAG + next(b for b in buckets_fine() if b >= H_START + SIG["ML-DSA-65"])
    res["alpha_min_signed_first_image_fine"] = 8 * need_f / N_PIX
    # sensitivity to alpha* (fixed grid and both ends of the CI of the screening alpha*, if available)
    sens = {}
    grid = [0.05, 0.0552, 0.06, 0.0787, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4]
    if s is not None:
        ci = s["alpha_star"].get("ci95") or s["alpha_star"].get("alpha_star_i_ci95")
        if ci:
            grid += [round(x, 4) for x in ci]
        grid.append(round(a_star, 4))
    for a in sorted(grid):
        single = bits_per_image(a, 1, 1536, 0, buckets_default(), single=True)
        n65 = None
        if single is not None:
            n65 = next((N for N in range(1, 4097) if (v := bits_per_image(a, N, 1536, 3309, buckets_default())) is not None
                        and v >= single), None)
        n65f = next((N for N in range(1, 4097) if (v := bits_per_image(a, N, 1536, 3309, buckets_fine())) is not None
                     and v >= max(single or 0, bits_per_image(a, 1, 1536, 0, buckets_fine(), single=True) or 0)), None)
        sens[str(a)] = {"single_unsigned_bits": single, "N_star_ML-DSA-65": n65, "N_star_ML-DSA-65_fine_vs_best": n65f,
                        "signed_first_image_feasible": max_msg(math.floor(a * N_PIX / 8), True, 1536, 3309,
                                                               buckets_default()) is not None}
    res["sensitivity"] = sens
    # fine alpha grid: N*_sigma (ML-DSA-65, main variant) against the best single image over both ladders
    def nstar(a, bks, ref):
        return next((N for N in range(1, 4097) if (v := bits_per_image(a, N, 1536, 3309, bks)) is not None and v >= ref), None)
    grid2 = {}
    for k in range(10, 41):
        a = k / 100
        sd = bits_per_image(a, 1, 1536, 0, buckets_default(), single=True)
        sf = bits_per_image(a, 1, 1536, 0, buckets_fine(), single=True)
        best = max(v for v in (sd, sf) if v is not None)
        grid2[f"{a:.2f}"] = {"single_default_B": sd and sd // 8, "single_fine_B": sf and sf // 8,
                             "N_star_default": nstar(a, buckets_default(), best), "N_star_fine": nstar(a, buckets_fine(), best)}
    res["sensitivity_grid_vs_best_single"] = grid2
    # budget at alpha* by sigma placement and scheme (table for Section 7.10): smallest N at which a signed session is feasible, bits/image at N = 2, 4, 8
    feas = {}
    for pl in PLACES:
        for kname, kem in KEM.items():
            for sname, sig in SIG.items():
                if sname == "none":
                    continue
                nmin = next((N for N in range(1, 65) if bits_per_image(a_star, N, kem, sig, buckets_merged(), place=pl) is not None), None)
                feas[f"{pl}|{kname}|{sname}"] = {"N_min_feasible": nmin,
                                                  "bits_per_image": {N: bits_per_image(a_star, N, kem, sig, buckets_merged(), place=pl) for N in (2, 4, 8, 16)}}
    res["signed_session_feasibility_merged"] = feas
    (out / args.name).write_text(json.dumps(res, indent=1))
    print(json.dumps({"alpha_star": a_star, "sensitivity": sens}, indent=1))


if __name__ == "__main__":
    main()
