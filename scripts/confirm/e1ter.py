"""E1-ter: calibrated two-sided test of the learned distinguisher D4 on fresh ALASKA2 covers
(registered after review, 24 Nov 2026; registrations/E1TER_PLAN.md and registrations/e1ter_plan.json).

Why. The registered gate G3 of E1 failed for D4 on BOSSbase/BOWS2, and two post hoc analyses showed that neither the registered
interval (fixed models, too narrow) nor the model-refitting bootstrap (too wide) calibrates the 0.02 rule. E1-bis (G3') used a
null-calibrated but one-sided bound. E1-ter fixes, before any of its images is processed, a two-sided rule whose null
distribution includes the retraining of the five models, and states its type-I error and power. It changes no registered
decision of E1 or E1-bis.

Data. Every ALASKA2 image not used before (data/external/alaska2/e1ter_ids.txt, 38 533 images; make_alaska2_e1ter_subset.py),
512x512 grayscale PGM written by verify_alaska2.py; 5 folds assigned at random per image (seed 20261124); same layout A8, two
Kemeleon variants and six buckets as E1, so twelve cells.

D4 (unchanged from E1): seq_features, standardization on the training part, logistic regression C = 0.05, 5-fold cross-validation;
signed advantage adv = 2 acc - 1 on all images.

Null distribution (design step, before any image): R data sets per syndrome length L in which "cover" and reference are both
uniform strings, n images, random folds; each data set retrains the five models (e1bis_power.one, seed tag 40). SD_null(L) is
the s.d. of adv over the R data sets; the draws also give the two-sided null p-value of an observed adv.

Registered rule.
  * Equivalence (per cell, two one-sided tests at 0.05): the cell passes when |adv| + 1.645 SD_null(L) <= 0.02.
    The conjunction over the twelve cells is an intersection-union test, so its level is 0.05 without a multiplicity correction.
  * Detection: two-sided null p-value p = (1 + #{|adv_null| >= |adv|}) / (R + 1), Holm over the twelve cells at 0.05.
  * Verdict: "equivalent within 0.02" when all twelve cells pass; "distinguisher detected" when a Holm-adjusted p <= 0.05;
    otherwise "inconclusive". Type-I error of the equivalence claim at |adv| = 0.02: 0.05 per cell under the normal
    approximation with the s.d. at the margin equal to SD_null; power at adv = 0 is reported by the design step.

    python scripts/data/make_alaska2_e1ter_subset.py          # list (its SHA-256 is registered)
    powershell -File scripts/data/download_alaska2.ps1 -List data/external/alaska2/e1ter_ids.txt
    python scripts/data/verify_alaska2.py --list data/external/alaska2/e1ter_ids.txt --budget 1e9
    python scripts/confirm/e1ter.py split
    python scripts/confirm/e1ter.py design --R 1000 --workers 7    # null draws, before extraction
    python scripts/confirm/e1ter.py extract --workers 4
    python scripts/confirm/e1ter.py analyze
  --work <folder> points to the folder that holds data/ and results/ when they are not under this repository.
Output: <work>/results/e1ter/e1ter_summary.json; <work>/results/e1ter_design/sd_null.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from common import ROOT, check_frozen, one_thread_per_worker, workers_default  # noqa: E402
import e1bis  # noqa: E402  (job(): syndrome extraction identical to E1-bis)

SEED = 20261124
DELTA, Z1, ALPHA = 0.02, 1.6448536269514722, 0.05
VARIANTS = (1536, 1252)
TAG = 40
CHUNK = 500
PLAN = ROOT / "registrations" / "e1ter_plan.json"


def paths(work: Path) -> dict:
    return {"ids": ROOT / "data" / "external" / "alaska2" / "e1ter_ids.txt",
            "pgm": work / "data" / "external" / "alaska2" / "pgm",
            "split": work / "results" / "e1ter" / "e1ter_split.json",
            "design": work / "results" / "e1ter_design",
            "out": work / "results" / "e1ter"}


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def check_plan(P: dict) -> dict:
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    if sha(P["ids"]) != plan["ids_sha256"]:
        raise SystemExit(f"{P['ids']} differs from the registered list ({plan['ids_sha256']})")
    return plan


# ----------------------------------------------------------------------------- rule (pure functions, tested)
def cell_pass(adv: float, sd_null: float, delta: float = DELTA, z: float = Z1) -> bool:
    """Two one-sided tests: both |adv| bounds inside the margin."""
    return bool(abs(adv) + z * sd_null <= delta)


def null_p(adv: float, draws: np.ndarray) -> float:
    return float((1 + np.sum(np.abs(draws) >= abs(adv))) / (draws.size + 1))


def holm(p: np.ndarray) -> np.ndarray:
    o = np.argsort(p)
    m = p.size
    adj = np.empty(m)
    run = 0.0
    for r, i in enumerate(o):
        run = max(run, (m - r) * p[i])
        adj[i] = min(1.0, run)
    return adj


def verdict(passes: list[bool], p_holm: np.ndarray, alpha: float = ALPHA) -> str:
    if np.any(p_holm <= alpha):
        return "distinguisher detected"
    return "equivalent within 0.02" if all(passes) else "inconclusive"


def power_normal(sd: float, delta: float = DELTA, z: float = Z1) -> float:
    """P(|adv| + z sd <= delta) when adv ~ N(0, sd^2)."""
    from scipy.stats import norm
    c = delta / sd - z
    return float(max(0.0, 2 * norm.cdf(c) - 1))


# ----------------------------------------------------------------------------- steps
def split(a, P):
    check_plan(P)
    rows = [ln.split() for ln in P["ids"].read_text().splitlines() if ln.strip()]
    ids = [r[0] for r in rows]
    fold = np.random.default_rng(SEED).permutation(len(ids)) % 5
    doc = {"note": "E1-ter: 5 random folds per image, seed 20261124; every image has a fold",
           "ids_file": "data/external/alaska2/e1ter_ids.txt", "ids_sha256": sha(P["ids"]), "seed": SEED,
           "fold": {k: int(f) for k, f in zip(ids, fold)}, "camera": {r[0]: (r[1] if len(r) > 1 else "") for r in rows}}
    P["split"].parent.mkdir(parents=True, exist_ok=True)
    P["split"].write_bytes(json.dumps(doc, indent=0).encode("utf-8"))
    print(f"{len(ids)} images; folds {np.bincount(fold).tolist()}; SHA-256 {sha(P['split'])}")


def design(a, P):
    """Null draws with retraining (before any image is processed): SD_null, null p-values, power."""
    import e1bis_power as ep
    plan = check_plan(P)
    if (P["out"] / "raw").exists() and any((P["out"] / "raw").glob("chunk_*.npz")) and not a.force:
        raise SystemExit("syndromes already extracted: the design step must precede extraction (use --force only for tests)")
    n = a.n or plan["n"]
    jobs = [(n, L, r, a.pool, TAG) for L in VARIANTS for r in range(a.R)]
    draws: dict[int, list[float]] = {L: [] for L in VARIANTS}
    t0 = time.time()
    with ProcessPoolExecutor(a.workers) as ex:
        for k, (_, L, _, v) in enumerate(ex.map(ep.one, jobs, chunksize=10), 1):
            draws[L].append(v)
            if k % 50 == 0:
                print(f"{k}/{len(jobs)} null data sets after {time.time() - t0:.0f}s", flush=True)
    res = {"label": "E1-ter design step (registered 24 Nov 2026), computed before any E1-ter image was processed",
           "n": n, "R": a.R, "pool": a.pool, "seed_tag": TAG, "delta": DELTA, "z": Z1, "sd_null": {}, "power": {}}
    for L in VARIANTS:
        v = np.array(draws[L])
        sd = float(v.std(ddof=1))
        res["sd_null"][str(L)] = sd
        res["power"][str(L)] = {"p_cell_pass_null_empirical": float(np.mean(np.abs(v) + Z1 * sd <= DELTA)),
                                "p_cell_pass_null_normal": power_normal(sd), "mean_adv_null": float(v.mean())}
    res["p_all12_pass_null_normal_independent"] = float(np.prod([res["power"][str(L)]["p_cell_pass_null_normal"] ** 6 for L in VARIANTS]))
    res["type1_per_cell_at_margin_normal"] = 0.05
    res["seconds"] = round(time.time() - t0)
    P["design"].mkdir(parents=True, exist_ok=True)
    (P["design"] / "sd_null.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    np.savez_compressed(P["design"] / "null_draws.npz", **{f"L{L}": np.array(v) for L, v in draws.items()})
    print(json.dumps({k: res[k] for k in ("n", "R", "sd_null", "power", "p_all12_pass_null_normal_independent")}, indent=1))


def extract(a, P):
    check_plan(P)
    if not a.no_frozen_check:
        check_frozen()
    if not (P["design"] / "sd_null.json").exists() and not a.force:
        raise SystemExit("run the design step first (null draws before any image is processed)")
    sp = json.loads(P["split"].read_text(encoding="utf-8"))
    keys = sorted(sp["fold"])
    (P["out"] / "raw").mkdir(parents=True, exist_ok=True)
    chunks = [(k, keys[k * CHUNK:(k + 1) * CHUNK], str(P["pgm"]), str(P["out"])) for k in range((len(keys) + CHUNK - 1) // CHUNK)]
    t0 = time.time()
    one_thread_per_worker()
    with ProcessPoolExecutor(a.workers) as ex:
        for i, k in enumerate(ex.map(e1bis.job, chunks), 1):
            print(f"chunk {k} - {i}/{len(chunks)} after {time.time() - t0:.0f}s", flush=True)


def d4_advantage(S: np.ndarray, Uraw: np.ndarray, fold: np.ndarray) -> float:
    from screen_analyze import seq_features      # noqa: E402
    from sklearn.linear_model import LogisticRegression
    n = len(S)
    X = np.vstack([seq_features(S[i:i + 2000]) for i in range(0, n, 2000)])
    Ux = np.vstack([seq_features(Uraw[i:i + 2000]) for i in range(0, n, 2000)])
    cc, cu = np.zeros(n, bool), np.zeros(n, bool)
    for f in range(5):
        trn, te = fold != f, fold == f
        Xtr = np.vstack([X[trn], Ux[trn]])
        ytr = np.r_[np.zeros(trn.sum()), np.ones(trn.sum())]
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
        clf = LogisticRegression(C=0.05, max_iter=3000).fit((Xtr - mu) / sd, ytr)
        cc[te] = clf.predict((X[te] - mu) / sd) == 0
        cu[te] = clf.predict((Ux[te] - mu) / sd) == 1
    return float(2 * np.r_[cc, cu].mean() - 1)


def analyze(a, P):
    from screen_analyze import llr_bytes      # noqa: E402
    from pqpks.analysis import stats          # noqa: E402
    check_plan(P)
    sp = json.loads(P["split"].read_text(encoding="utf-8"))
    des = json.loads((P["design"] / "sd_null.json").read_text(encoding="utf-8"))
    nd = np.load(P["design"] / "null_draws.npz")
    Z = [np.load(f) for f in sorted((P["out"] / "raw").glob("chunk_*.npz"))]
    keys = np.concatenate([z["keys"] for z in Z])
    fold = np.array([sp["fold"][str(k)] for k in keys])
    n = len(keys)
    if n != len(sp["fold"]):
        raise SystemExit(f"{n} images extracted, {len(sp['fold'])} expected")
    out = {"label": "E1-ter (registered after review, 24 Nov 2026); E1 and E1-bis are unchanged",
           "plan_sha256": sha(PLAN), "split_sha256": sha(P["split"]), "design_sha256": sha(P["design"] / "sd_null.json"),
           "n": int(n), "rule": "cell passes when |adv| + 1.645 SD_null(L) <= 0.02; detection by Holm over twelve null p-values at 0.05",
           "cells": {}}
    cells = [c for c in Z[0].files if c.startswith("e1_")]
    for c in cells:
        S = np.concatenate([z[c] for z in Z])
        L = S.shape[1]
        rng = np.random.default_rng(zlib.crc32(("e1ter:" + c).encode()))
        Uraw = rng.integers(0, 256, size=S.shape, dtype=np.uint8)
        adv = d4_advantage(S, Uraw, fold)
        sd = float(des["sd_null"][str(L)])
        q = np.array([llr_bytes(s) > 0 for s in S], float)          # D1, descriptive
        U = rng.integers(0, 256, size=(50000, L), dtype=np.uint8)
        u = np.array([llr_bytes(x) > 0 for x in U], float)
        lo, hi = stats.ci(stats.two_sample_bootstrap(lambda i1, i2: q[i1].mean() - u[i2].mean(), q.size, u.size, B=2000, seed=11))
        out["cells"][c] = {"length": int(L), "adv": adv, "sd_null": sd, "lower": adv - Z1 * sd, "upper": adv + Z1 * sd,
                           "pass": cell_pass(adv, sd), "p_null": null_p(adv, nd[f"L{L}"]),
                           "llr_descriptive": {"diff": float(q.mean() - u.mean()), "ci95": [lo, hi]}}
        r = out["cells"][c]
        print(f"{c}: adv {adv:+.4f}, bounds [{r['lower']:+.4f}, {r['upper']:+.4f}], pass {r['pass']}, null p {r['p_null']:.3f}", flush=True)
    p = np.array([out["cells"][c]["p_null"] for c in cells])
    ph = holm(p)
    for c, v in zip(cells, ph):
        out["cells"][c]["p_holm"] = float(v)
    out["cells_passing"] = int(sum(out["cells"][c]["pass"] for c in cells))
    out["verdict"] = verdict([out["cells"][c]["pass"] for c in cells], ph)
    out["max_abs_bound"] = float(max(abs(out["cells"][c]["adv"]) + Z1 * out["cells"][c]["sd_null"] for c in cells))
    P["out"].mkdir(parents=True, exist_ok=True)
    (P["out"] / "e1ter_summary.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("E1-ter:", out["verdict"], "| cells passing:", out["cells_passing"], "/ 12 | largest |adv| + 1.645 SD_null:", round(out["max_abs_bound"], 4))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["split", "design", "extract", "analyze"])
    ap.add_argument("--work", default=str(ROOT), help="folder with data/ and results/ (default: this repository)")
    ap.add_argument("--workers", type=int, default=workers_default())
    ap.add_argument("--R", type=int, default=1000)
    ap.add_argument("--pool", type=int, default=400_000)
    ap.add_argument("--n", type=int, help="design step: override n (code tests only)")
    ap.add_argument("--force", action="store_true", help="code tests only")
    ap.add_argument("--no-frozen-check", action="store_true", help="code tests only")
    a = ap.parse_args()
    P = paths(Path(a.work))
    {"split": split, "design": design, "extract": extract, "analyze": analyze}[a.step](a, P)


if __name__ == "__main__":
    main()
