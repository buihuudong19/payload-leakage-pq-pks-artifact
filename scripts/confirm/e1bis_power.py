"""Power calculation for E1-bis (design step). Uses uniform strings only, no image data.

Purpose: choose the sample size n of E1-bis BEFORE any data exist. For each candidate n and each syndrome length L (1536 B, 1252 B):
  * simulate R times (drawn from a pool of uniform-string features, as in the post hoc analysis of E1) the learned distinguisher D4
    without signal: "cover" and U are both uniform strings of length L, n samples each, with exactly the D4 procedure of
    e1_confirm.py (seq_features, standardization on the training part, logistic C = 0.05, 5 folds); folds are assigned at random
    per image because E1-bis has no training-only images;
  * estimate SD_null(n, L) of the signed advantage 2 acc - 1;
  * under the registered D4 rule (a cell passes when adv + 1.645 SD_null <= 0.02), compute the probability that a cell passes
    under the null from the iterations, and the probability that all 12 cells pass (6 cells per variant; cells treated as independent).
Output: results/e1bis_design/power.json.

    python scripts/confirm/e1bis_power.py --n 20000 30000 35000 40000 --R 300 --workers 2
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "experiments"))
from screen_analyze import seq_features  # noqa: E402

DELTA, Z1 = 0.02, 1.6448536269514722
OUT = ROOT / "results" / "e1bis_design"


_POOL: dict[int, np.ndarray] = {}


def pool(L: int, size: int, tag: int = 38) -> np.ndarray:
    """Pool of uniform-string features of length L, fixed seed. Each iteration draws 2n vectors without replacement."""
    if (L, tag) not in _POOL:
        prng = np.random.default_rng([20261107, tag, L])
        _POOL[(L, tag)] = np.vstack([seq_features(prng.integers(0, 256, size=(2000, L), dtype=np.uint8)) for _ in range(size // 2000)])
    return _POOL[(L, tag)]


def one(args):
    n, L, r, size, tag = args
    P = pool(L, size, tag)
    rng = np.random.default_rng([20261107, tag, n, L, r])
    perm = rng.permutation(len(P))[:2 * n]
    X, Ux = P[perm[:n]], P[perm[n:]]
    fold = rng.permutation(n) % 5
    cc = np.zeros(n, bool); cu = np.zeros(n, bool)
    for f in range(5):
        trn, te = fold != f, fold == f
        Xtr = np.vstack([X[trn], Ux[trn]])
        ytr = np.r_[np.zeros(trn.sum()), np.ones(trn.sum())]
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
        clf = LogisticRegression(C=0.05, max_iter=3000).fit((Xtr - mu) / sd, ytr)
        cc[te] = clf.predict((X[te] - mu) / sd) == 0
        cu[te] = clf.predict((Ux[te] - mu) / sd) == 1
    return n, L, r, float(2 * np.r_[cc, cu].mean() - 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, nargs="+", default=[20000, 30000, 35000, 40000])
    ap.add_argument("--L", type=int, nargs="+", default=[1536, 1252])
    ap.add_argument("--R", type=int, default=300)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--pool", type=int, default=300_000)
    ap.add_argument("--stage", choices=["power", "sdnull"], default="power",
                    help="power: design step 0 (choose n); sdnull: step 3 (SD_null at the chosen n, own seed, writes sd_null.json)")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    tag = 38 if a.stage == "power" else 39
    jobs = [(n, L, r, a.pool, tag) for L in a.L for n in a.n for r in range(a.R)]
    adv: dict[tuple[int, int], list[float]] = {}
    t0 = time.time()
    with ProcessPoolExecutor(a.workers) as ex:
        for k, (n, L, r, v) in enumerate(ex.map(one, jobs, chunksize=25), 1):
            adv.setdefault((n, L), []).append(v)
            if k % 50 == 0:
                print(f"{k}/{len(jobs)} after {time.time() - t0:.0f}s", flush=True)
    res = {"stage": "design", "delta": DELTA, "R": a.R, "pool": a.pool, "rule": "a cell passes when adv + 1.645 SD_null(n) <= delta", "rows": []}
    for n in a.n:
        row = {"n": n, "by_L": {}}
        p_all = 1.0
        for L in a.L:
            v = np.array(adv[(n, L)])
            sd = float(v.std(ddof=1))
            p_cell = float(np.mean(v + Z1 * sd <= DELTA))
            row["by_L"][str(L)] = {"sd_null": sd, "sd_se": sd / np.sqrt(2 * (len(v) - 1)), "mean": float(v.mean()),
                                   "p_cell_pass_null": p_cell, "p_cell_pass_normal": float(_phi(DELTA / sd - Z1))}
            p_all *= _phi(DELTA / sd - Z1) ** 6
        row["p_all12_pass_null_normal"] = float(p_all)
        res["rows"].append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    res["seconds"] = round(time.time() - t0)
    name = "power" if a.stage == "power" else "sd_null"
    if a.stage == "sdnull":
        res["stage"] = "step 3 (registered 07/11/2026): SD_null used in the D4 upper bound of E1-bis"
        res["seed_tag"] = tag
        res["sd_null"] = {str(L): adv_sd for L, adv_sd in ((L, float(np.std(adv[(a.n[0], L)], ddof=1))) for L in a.L)}
        res["n"] = a.n[0]
    (OUT / f"{name}.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    np.savez_compressed(OUT / f"{name}_draws.npz", **{f"n{n}_L{L}": np.array(v) for (n, L), v in adv.items()})


def _phi(x):
    from scipy.stats import norm
    return norm.cdf(x)


if __name__ == "__main__":
    main()
