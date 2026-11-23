"""POST HOC analysis of the learned distinguisher (D4) of the confirmatory E1 (procedure fixed on 06/11/2026 before computing).

Does not change the registered conclusion (G3 failed). It answers whether the three D4 values above the 0.02 threshold are
consistent with the noise of the measurement procedure itself in the absence of any signal. Procedure:
  1. recompute D4 on the real syndromes with exactly the code of e1_confirm.py (same seed, same draw order, same folds) and check
     that the advantages equal e1_summary.json;
  2. null: replace the cover syndromes by uniform strings of the same length, keeping n, folds, features, model and bootstrap;
     R iterations per variant, drawn from a pool of uniform-string features;
  3. report the null SD, two-sided null p-values (Holm over the 12 cells), the actual coverage of the registered 95 % CI, the
     probability that a null cell "fails", the expected number of failing cells, and per-fold CIs.

    python scripts/confirm/e1_posthoc_null.py [--e1 results/e1_confirm] [--R 1000] [--pool 400000]
Output: results/e1_confirm/e1_posthoc_null.json (+ e1_posthoc_null_draws.npz).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import zlib
from pathlib import Path

import numpy as np
from scipy import stats as st
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "experiments"))
from pqpks.analysis import stats  # noqa: E402
from screen_analyze import seq_features  # noqa: E402

B_BOOT, MARGIN = 1000, 0.02


def feats(S: np.ndarray, step: int = 2000) -> np.ndarray:
    """seq_features by row blocks (features are computed per row, so the result equals a single pass)."""
    return np.vstack([seq_features(S[i:i + step]) for i in range(0, len(S), step)])


def cv_correct(X, Ux, fold):
    """Exactly as e1_confirm.analyze: logistic C = 0.05, standardization on the training part, 5 folds; fold -1 is training only."""
    n = len(X)
    correct = np.zeros(2 * n, bool)
    for f in range(5):
        trn, te = fold != f, fold == f
        Xtr = np.vstack([X[trn], Ux[trn]])
        ytr = np.r_[np.zeros(trn.sum()), np.ones(trn.sum())]
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
        clf = LogisticRegression(C=0.05, max_iter=3000).fit((Xtr - mu) / sd, ytr)
        correct[:n][te] = clf.predict((X[te] - mu) / sd) == 0
        correct[n:][te] = clf.predict((Ux[te] - mu) / sd) == 1
    return correct[:n], correct[n:]


def summarize(cc, cu, fold_test, rng=None, seed=None):
    """Advantage, registered 95 % CI (cluster bootstrap over test images), advantage per fold."""
    m = cc.size
    adv = 2 * np.r_[cc, cu].mean() - 1
    if seed is not None:                                          # exactly the registered function (code check)
        lo, hi = stats.ci(stats.cluster_bootstrap(lambda ii: 2 * np.r_[cc[ii], cu[ii]].mean() - 1, m, B=B_BOOT, seed=seed))
    else:                                                         # same distribution, vectorized for speed
        idx = rng.integers(0, m, (B_BOOT, m))
        bs = (cc[idx].sum(1) + cu[idx].sum(1)) / m - 1
        lo, hi = np.quantile(bs, [0.025, 0.975])
    per = np.array([2 * np.r_[cc[fold_test == f], cu[fold_test == f]].mean() - 1 for f in range(5)])
    half = st.t.ppf(0.975, 4) * per.std(ddof=1) / np.sqrt(5)
    return {"adv": float(adv), "ci95": [float(lo), float(hi)], "upper_abs": float(max(abs(lo), abs(hi))),
            "per_fold": per.tolist(), "fold_ci95": [float(per.mean() - half), float(per.mean() + half)]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--e1", default=str(ROOT / "results" / "e1_confirm"))
    ap.add_argument("--split", default=str(ROOT / "configs" / "splits" / "split_v1.json"))
    ap.add_argument("--R", type=int, default=1000)
    ap.add_argument("--pool", type=int, default=400_000)
    a = ap.parse_args()
    e1 = Path(a.e1)
    E1 = json.loads((e1 / "e1_summary.json").read_text(encoding="utf-8"))
    split = json.loads(Path(a.split).read_text(encoding="utf-8"))
    Z = [np.load(f) for f in sorted((e1 / "raw").glob("chunk_*.npz"))]
    keys = np.concatenate([z["keys"] for z in Z])
    folds = split["e1"]["folds"]
    fold = np.array([folds.get(str(k), -1) for k in keys])
    test = fold >= 0
    fold_test = fold[test]
    cells = [c for c in Z[0].files if c.startswith("e1_")]
    res = {"label": "post hoc", "R": a.R, "pool": a.pool, "B": B_BOOT, "n_test": int(test.sum()),
           "observed": {}, "null": {}}
    t0 = time.time()
    # 1. code check on the real data
    for c in cells:
        S = np.concatenate([z[c] for z in Z])
        rng = np.random.default_rng(zlib.crc32(c.encode()))
        rng.integers(0, 256, size=(50000, S.shape[1]), dtype=np.uint8)     # U of D1: drawn to keep the draw order
        X, Ux = feats(S), feats(rng.integers(0, 256, size=S.shape, dtype=np.uint8))
        cc, cu = cv_correct(X, Ux, fold)
        r = summarize(cc[test], cu[test], fold_test, seed=13)
        reg = E1["cells"][c]["learned"]
        if abs(r["adv"] - reg["adv"]) > 1e-12 or max(abs(r["ci95"][0] - reg["ci95"][0]), abs(r["ci95"][1] - reg["ci95"][1])) > 1e-12:
            raise SystemExit(f"{c}: recomputation differs from e1_summary.json ({r['adv']} vs {reg['adv']})")
        res["observed"][c] = r | {"length": int(S.shape[1])}
        print(f"[{time.time() - t0:6.0f}s] {c}: advantage {r['adv']:+.4f} equals the registered value; per-fold CI "
              f"[{r['fold_ci95'][0]:+.4f}, {r['fold_ci95'][1]:+.4f}]", flush=True)
    # 2. null by length
    n = len(keys)
    draws = {}
    for L in sorted({v["length"] for v in res["observed"].values()}):
        prng = np.random.default_rng([20261106, L])
        P = np.vstack([seq_features(prng.integers(0, 256, size=(2000, L), dtype=np.uint8)) for _ in range(a.pool // 2000)])
        print(f"[{time.time() - t0:6.0f}s] uniform-string feature pool L = {L}: {len(P)} vectors", flush=True)
        advs, ups, excl0, fold_excl0 = [], [], [], []
        for r_ in range(a.R):
            rr = np.random.default_rng([20261107, L, r_])
            perm = rr.permutation(len(P))[:2 * n]
            cc, cu = cv_correct(P[perm[:n]], P[perm[n:]], fold)
            s = summarize(cc[test], cu[test], fold_test, rng=rr)
            advs.append(s["adv"]); ups.append(s["upper_abs"])
            excl0.append(s["ci95"][0] > 0 or s["ci95"][1] < 0)
            fold_excl0.append(s["fold_ci95"][0] > 0 or s["fold_ci95"][1] < 0)
            if (r_ + 1) % 100 == 0:
                print(f"[{time.time() - t0:6.0f}s] L = {L}: {r_ + 1}/{a.R} iterations", flush=True)
        advs, ups = np.array(advs), np.array(ups)
        draws[f"adv_L{L}"], draws[f"upper_L{L}"] = advs, ups
        res["null"][str(L)] = {"sd_adv": float(advs.std(ddof=1)), "mean_adv": float(advs.mean()),
                               "q": {q: float(np.quantile(advs, float(q))) for q in ("0.005", "0.025", "0.975", "0.995")},
                               "ci95_coverage_excl0": float(np.mean(excl0)), "fold_ci95_excl0": float(np.mean(fold_excl0)),
                               "p_fail_cell": float(np.mean(ups > MARGIN))}
    # 3. null p-values and Holm
    ps = {}
    for c, r in res["observed"].items():
        advs = draws[f"adv_L{r['length']}"]
        ps[c] = (1 + np.sum(np.abs(advs) >= abs(r["adv"]))) / (len(advs) + 1)
    rej = stats.holm(np.array(list(ps.values())))
    holm_p = dict(zip(ps, _holm_adjusted(np.array(list(ps.values())))))
    for (c, p), rj in zip(ps.items(), rej):
        res["observed"][c] |= {"p_null": float(p), "p_holm": float(holm_p[c]), "holm_reject": bool(rj),
                               "registered_fail": res["observed"][c]["upper_abs"] > MARGIN}
    res["expected_failing_cells_null"] = float(sum(res["null"][str(r["length"])]["p_fail_cell"] for r in res["observed"].values()))
    res["observed_failing_cells"] = int(sum(r["registered_fail"] for r in res["observed"].values()))
    exp_ok = res["expected_failing_cells_null"] >= 0.5
    res["verdict_rule"] = "registered post hoc rule, item 4"
    res["verdict"] = ("consistent with the noise of the measurement procedure" if not any(rej) and exp_ok else
                      "a cell has Holm p <= 0.05: evidence of a real distinguisher" if any(rej) else
                      "no cell has Holm p <= 0.05, but the expected number of failing cells under the null is < 0.5")
    res["seconds"] = round(time.time() - t0)
    (e1 / "e1_posthoc_null.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    np.savez_compressed(e1 / "e1_posthoc_null_draws.npz", **draws)
    print(json.dumps({k: v for k, v in res.items() if k != "observed"}, indent=1, ensure_ascii=False))
    for c, r in res["observed"].items():
        print(f"{c}: advantage {r['adv']:+.4f}, null p {r['p_null']:.3f}, Holm p {r['p_holm']:.3f}")


def _holm_adjusted(p: np.ndarray) -> np.ndarray:
    o = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    run = 0.0
    for k, i in enumerate(o):
        run = max(run, min(1.0, (m - k) * p[i]))
        adj[i] = run
    return adj


if __name__ == "__main__":
    main()
