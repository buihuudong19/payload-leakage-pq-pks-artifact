"""POST HOC (23 Nov 2026): model-refitting bootstrap for the learned distinguisher D4 of the confirmatory E1.

Does not change the registered outcome (gate G3 failed). The registered 95 % CI of D4 resamples the test images but keeps the
five fitted models fixed, and under the uniform null it excludes zero in 11.4-11.6 % of the replicates instead of 5 %
(e1_posthoc_null.py). This script computes an interval that also refits the models:

  0. code check: recompute D4 on the real syndromes with exactly the code and draws of e1_confirm.py and stop if the advantage
     differs from results/e1_confirm/e1_summary.json;
  1. refit bootstrap, per cell: resample the images with replacement within each fold of split_v1 (and within the training-only
     images), keep each image's fold and its paired uniform string, refit the five logistic models, recompute the advantage on
     the resampled test images; B replicates, percentile 95 % CI, upper bound of |advantage|;
  2. (skipped with --null-R 0) coverage of this interval under the uniform null: R null data sets drawn from the same uniform-string
     feature pool as e1_posthoc_null.py, each with its own refit bootstrap of --null-B replicates; reports how often the interval
     excludes zero (nominal 5 %) and how often its upper bound exceeds 0.02.

    python scripts/confirm/e1_posthoc_refit.py --e1 <...>/results/e1_confirm --quick          # timing only, writes nothing
    python scripts/confirm/e1_posthoc_refit.py --e1 <...>/results/e1_confirm --workers 8                 # B 2000, null R 200 x 400
    (run_all: step PH2 of the manual lane W12, scripts/orchestration/pipeline.ps1)
Output: <e1>/e1_posthoc_refit.json and e1_posthoc_refit.log (the console output). Reads only; changes no existing file.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):     # one thread per worker process
    os.environ.setdefault(_v, "1")

import argparse
import hashlib
import json
import sys
import time
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "experiments"))
from pqpks.analysis import stats  # noqa: E402
from screen_analyze import seq_features  # noqa: E402

MARGIN = 0.02
SEED = 20261123
G = {}                                         # per-process data: keys, fold, path of the chunks


def feats(S: np.ndarray, step: int = 2000) -> np.ndarray:
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


def advantage(X, Ux, fold):
    cc, cu = cv_correct(X, Ux, fold)
    t = fold >= 0
    return 2 * np.r_[cc[t], cu[t]].mean() - 1


def strata(fold):
    return [np.flatnonzero(fold == g) for g in np.unique(fold)]


def refit_boot(X, Ux, fold, B, rng):
    """Bootstrap within each fold (and within the training-only images); every replicate refits the five models."""
    groups = strata(fold)
    out = np.empty(B)
    for b in range(B):
        idx = np.concatenate([rng.choice(g, g.size, replace=True) for g in groups])
        out[b] = advantage(X[idx], Ux[idx], fold[idx])
    return out


def init(e1: str, split: str):
    e1 = Path(e1)
    files = sorted((e1 / "raw").glob("chunk_*.npz"))
    keys = np.concatenate([np.load(f)["keys"] for f in files])
    folds = json.loads(Path(split).read_text(encoding="utf-8"))["e1"]["folds"]
    G.update(files=files, keys=keys, fold=np.array([folds.get(str(k), -1) for k in keys]))


def cell_features(c):
    S = np.concatenate([np.load(f)[c] for f in G["files"]])
    rng = np.random.default_rng(zlib.crc32(c.encode()))
    rng.integers(0, 256, size=(50000, S.shape[1]), dtype=np.uint8)       # U of D1 in e1_confirm: drawn to keep the draw order
    return S.shape[1], feats(S), feats(rng.integers(0, 256, size=S.shape, dtype=np.uint8))


def job_cell(args):
    c, B, reg = args
    t0 = time.time()
    L, X, Ux = cell_features(c)
    fold = G["fold"]
    adv = advantage(X, Ux, fold)
    if reg is not None and abs(adv - reg) > 1e-12:
        raise SystemExit(f"{c}: recomputed advantage {adv!r} differs from e1_summary.json {reg!r}; stop")
    boot = refit_boot(X, Ux, fold, B, np.random.default_rng([SEED, zlib.crc32(c.encode())]))
    lo, hi = stats.ci(boot)
    return c, {"length": int(L), "adv": float(adv), "refit_ci95": [lo, hi], "refit_upper_abs": float(max(abs(lo), abs(hi))),
               "refit_sd": float(boot.std(ddof=1)), "excludes_zero": bool(lo > 0 or hi < 0),
               "refit_fail_0.02": bool(max(abs(lo), abs(hi)) > MARGIN), "seconds": round(time.time() - t0)}


def job_null(args):
    L, r, Bn, pool_n = args
    pool = G.setdefault(("pool", L), None)
    if pool is None:                                                     # same pool as e1_posthoc_null.py
        prng = np.random.default_rng([20261106, L])
        pool = np.vstack([seq_features(prng.integers(0, 256, size=(2000, L), dtype=np.uint8)) for _ in range(pool_n // 2000)])
        G[("pool", L)] = pool
    n = len(G["keys"])
    rr = np.random.default_rng([SEED, 1, L, r])
    perm = rr.permutation(len(pool))[:2 * n]
    X, Ux, fold = pool[perm[:n]], pool[perm[n:]], G["fold"]
    adv = advantage(X, Ux, fold)
    lo, hi = stats.ci(refit_boot(X, Ux, fold, Bn, rr))
    return L, float(adv), bool(lo > 0 or hi < 0), bool(max(abs(lo), abs(hi)) > MARGIN)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--e1", default=str(ROOT / "results" / "e1_confirm"))
    ap.add_argument("--split", default=str(ROOT / "configs" / "splits" / "split_v1.json"))
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--null-R", type=int, default=200)
    ap.add_argument("--null-B", type=int, default=400)
    ap.add_argument("--pool", type=int, default=400_000)
    ap.add_argument("--quick", action="store_true", help="time 5 refit replicates of one cell, write nothing")
    a = ap.parse_args()
    e1 = Path(a.e1)
    log_lines = []

    def say(s):
        print(s, flush=True)
        log_lines.append(s)

    E1 = json.loads((e1 / "e1_summary.json").read_text(encoding="utf-8"))
    split_sha = hashlib.sha256(Path(a.split).read_bytes()).hexdigest()
    if E1.get("split_sha256") and E1["split_sha256"] != split_sha:
        raise SystemExit(f"split file {a.split} (SHA-256 {split_sha}) is not the split of e1_summary.json ({E1['split_sha256']})")
    init(str(e1), a.split)
    cells = sorted(E1["cells"])
    say(f"e1 = {e1}\nsplit = {a.split} (SHA-256 {split_sha})\n{len(G['files'])} chunk files, {len(G['keys'])} images, "
        f"{int((G['fold'] >= 0).sum())} test images, {len(cells)} cells")
    if a.quick:
        t0 = time.time()
        L, X, Ux = cell_features(cells[0])
        t1 = time.time()
        adv = advantage(X, Ux, G["fold"])
        ok = abs(adv - E1["cells"][cells[0]]["learned"]["adv"]) <= 1e-12
        t2 = time.time()
        refit_boot(X, Ux, G["fold"], 5, np.random.default_rng(1))
        per = (time.time() - t2) / 5
        say(f"code check on {cells[0]}: advantage {adv:+.6f}, equal to e1_summary.json: {ok}")
        say(f"features {t1 - t0:.0f} s per cell; one refit replicate {per:.2f} s")
        w = max(1, min(a.workers, len(cells)))
        main_h = (len(cells) * (t1 - t0 + a.B * per)) / w / 3600
        null_h = (2 * 200 * 400 * per) / max(1, a.workers) / 3600
        say(f"estimate with {a.workers} workers: step 1 (B = {a.B}) about {main_h:.1f} h; "
            f"step 2 with --null-R 200 --null-B 400 about {null_h:.1f} h more")
        return
    t0 = time.time()
    res = {"label": "post hoc (23 Nov 2026); the registered outcome of G3 (failed) is unchanged", "B": a.B,
           "split_sha256": split_sha, "n_images": int(len(G["keys"])), "n_test": int((G["fold"] >= 0).sum()),
           "method": "stratified bootstrap within folds; every replicate refits the five logistic models; percentile 95 % CI",
           "cells": {}}
    with ProcessPoolExecutor(a.workers, initializer=init, initargs=(str(e1), a.split)) as ex:
        jobs = [(c, a.B, E1["cells"][c]["learned"]["adv"]) for c in cells]
        for c, r in ex.map(job_cell, jobs):
            r["registered_ci95"] = E1["cells"][c]["learned"]["ci95"]
            r["registered_upper_abs"] = E1["cells"][c]["learned"]["upper_abs"]
            res["cells"][c] = r
            say(f"[{time.time() - t0:7.0f}s] {c}: adv {r['adv']:+.4f}; registered CI [{r['registered_ci95'][0]:+.4f}, "
                f"{r['registered_ci95'][1]:+.4f}]; refit CI [{r['refit_ci95'][0]:+.4f}, {r['refit_ci95'][1]:+.4f}], "
                f"upper {r['refit_upper_abs']:.4f}, SD {r['refit_sd']:.4f}")
    cs = res["cells"].values()
    res["summary"] = {"cells_refit_upper_gt_0.02": int(sum(r["refit_fail_0.02"] for r in cs)),
                      "cells_refit_ci_excludes_zero": int(sum(r["excludes_zero"] for r in cs)),
                      "max_refit_upper_abs": float(max(r["refit_upper_abs"] for r in cs)),
                      "registered_failing_cells": int(sum(r["registered_upper_abs"] > MARGIN for r in cs))}
    if a.null_R > 0:
        lengths = sorted({r["length"] for r in cs})
        null = {str(L): {"adv": [], "excl0": [], "fail": []} for L in lengths}
        with ProcessPoolExecutor(a.workers, initializer=init, initargs=(str(e1), a.split)) as ex:
            jobs = [(L, r, a.null_B, a.pool) for L in lengths for r in range(a.null_R)]
            for k, (L, adv, ex0, fl) in enumerate(ex.map(job_null, jobs), 1):
                d = null[str(L)]
                d["adv"].append(adv); d["excl0"].append(ex0); d["fail"].append(fl)
                if k % 20 == 0:
                    say(f"[{time.time() - t0:7.0f}s] null: {k}/{len(jobs)} data sets")
        res["null"] = {"R": a.null_R, "B": a.null_B, "pool": a.pool}
        for L, d in null.items():
            res["null"][L] = {"sd_adv": float(np.std(d["adv"], ddof=1)), "refit_ci_excludes_zero": float(np.mean(d["excl0"])),
                              "refit_upper_gt_0.02": float(np.mean(d["fail"]))}
            say(f"null L = {L}: SD {res['null'][L]['sd_adv']:.4f}; refit CI excludes 0 in "
                f"{100 * res['null'][L]['refit_ci_excludes_zero']:.1f} % (nominal 5 %); upper > 0.02 in "
                f"{100 * res['null'][L]['refit_upper_gt_0.02']:.1f} %")
    res["seconds"] = round(time.time() - t0)
    say(json.dumps(res["summary"], indent=1))
    (e1 / "e1_posthoc_refit.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    (e1 / "e1_posthoc_refit.log").write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    say(f"written {e1 / 'e1_posthoc_refit.json'} and e1_posthoc_refit.log")


if __name__ == "__main__":
    main()
