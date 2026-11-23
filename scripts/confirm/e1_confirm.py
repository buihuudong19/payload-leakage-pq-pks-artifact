"""Confirmatory E1 (RQ1.3, H1, gate G3; paper, Table 7). Runs on a machine with all 20 000 images.

Data: the public syndrome Ext0 of the cover at A0(B) for the main Kemeleon variant (1536 B) and the rejection-sampled variant
(1252 B), 6 buckets, layout A8 (frozen profile). Test images: the 17 979 images in `e1.folds` of split_v1.json (18 000 minus 21
images in the same near-duplicate group as a pilot/screening image); pilot and screening images are only in the training part of every fold.

Distinguishers (each cell variant x bucket):
  D1  LLR of Algorithm 6 with the fixed threshold Lambda > 0: difference Pr[Lambda>0 | cover] - Pr[Lambda>0 | U], two-sample bootstrap;
      U is 50 000 uniform strings per cell;
  D2  bias per bit position (binomial, Holm within the cell) and a PRE-REGISTERED TEST: bit 3197 of the rejection-sampled variant
      at B = 256, two-sided, level 0.05, no correction (a single hypothesis);
  D3  byte-wise chi-square, pooled;
  D4  learner (logistic regression, low-dimensional string features), cross-validated over the 5 folds of the split;
  strata: source, saturation > 5 %; pooling M = 4, 16.
G3 passes when every cell has an upper 95 % CI bound of |advantage| <= 0.02 for D1 and D4, overall and per stratum (stratum
margin 2.8 sqrt(0.5/n_stratum)), and D2 rejects no position after Holm.

    python scripts/confirm/e1_confirm.py extract --workers 8      # about 20 000 images x 12 syndromes
    python scripts/confirm/e1_confirm.py analyze
Output: results/e1_confirm/e1_summary.json
"""
from __future__ import annotations

import argparse
import json
import zlib
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from common import ROOT, check_frozen, load_split, one_thread_per_worker, workers_default  # noqa: E402
from pqpks import data  # noqa: E402
from pqpks.analysis import stats  # noqa: E402
from pqpks.params import AEAD_TAG_BYTES, DEFAULT_PP  # noqa: E402
from pqpks.stego.two_stage import TwoStageSTC  # noqa: E402

OUT = ROOT / "results" / "e1_confirm"
VARIANTS = (1536, 1252)
REGISTERED = {"cell": "e1_k1252_b256", "bit": 3197}
CHUNK = 500


def keys_all(split):
    return sorted(split["e1"]["folds"]) + sorted(split["e1"]["train_only"])


def job(args):
    k, keys = args
    p = OUT / "raw" / f"chunk_{k:03d}.npz"
    if p.exists():
        return k
    emb = TwoStageSTC(DEFAULT_PP)
    rec = {f"e1_k{kb}_b{b}": [] for kb in VARIANTS for b in DEFAULT_PP.buckets}
    sat = []
    for key in keys:
        src, i = key.split("_")
        x = data.load_cover(src, int(i))
        sat.append(float(((x == 0) | (x == 255)).mean()))
        for kb in VARIANTS:
            for b in DEFAULT_PP.buckets:
                rec[f"e1_k{kb}_b{b}"].append(np.frombuffer(emb.extract_kem(x, kb, b + AEAD_TAG_BYTES), np.uint8))
    np.savez_compressed(p, keys=np.array(keys), sat=np.array(sat), **{c: np.stack(v) for c, v in rec.items()})
    return k


def extract(a):
    split = load_split()
    check_frozen()
    (OUT / "raw").mkdir(parents=True, exist_ok=True)
    ks = keys_all(split)
    chunks = [(k, ks[k * CHUNK:(k + 1) * CHUNK]) for k in range((len(ks) + CHUNK - 1) // CHUNK)]
    t0 = time.time()
    one_thread_per_worker()
    with ProcessPoolExecutor(a.workers) as ex:
        for n, k in enumerate(ex.map(job, chunks), 1):
            print(f"chunk {k} — {n}/{len(chunks)} after {time.time() - t0:.0f}s", flush=True)


def analyze(a):
    from screen_analyze import llr_bytes, seq_features  # noqa: E402
    from sklearn.linear_model import LogisticRegression
    from scipy.stats import binomtest, chisquare, norm
    split = load_split()
    files = sorted((OUT / "raw").glob("chunk_*.npz"))
    Z = [np.load(f) for f in files]
    keys = np.concatenate([z["keys"] for z in Z])
    sat = np.concatenate([z["sat"] for z in Z])
    folds = split["e1"]["folds"]
    fold = np.array([folds.get(str(k), -1) for k in keys])       # -1: training only
    test = fold >= 0
    src = np.array([str(k).split("_")[0] for k in keys])
    out = {"split_sha256": split["_sha256"], "n_test": int(test.sum()), "n_train_only": int((~test).sum()), "cells": {}}
    for c in Z[0].files:
        if not c.startswith("e1_"):
            continue
        S = np.concatenate([z[c] for z in Z])
        rng = np.random.default_rng(zlib.crc32(c.encode()))
        r = {}
        # D1: LLR, fixed threshold
        q = np.array([llr_bytes(s) > 0 for s in S[test]], float)
        U = rng.integers(0, 256, size=(50000, S.shape[1]), dtype=np.uint8)
        u = np.array([llr_bytes(x) > 0 for x in U], float)
        d = q.mean() - u.mean()
        boot = stats.two_sample_bootstrap(lambda i1, i2: q[i1].mean() - u[i2].mean(), q.size, u.size, B=2000, seed=11)
        lo, hi = stats.ci(boot)
        r["llr"] = {"diff": float(d), "ci95": [lo, hi], "upper_abs": float(max(abs(lo), abs(hi)))}
        for name, m in (("boss", src[test] == "boss"), ("bows2", src[test] == "bows2"), ("sat>5%", sat[test] > 0.05)):
            nm = int(m.sum())
            if nm >= 50:
                bb = stats.two_sample_bootstrap(lambda i1, i2, qq=q[m]: qq[i1].mean() - u[i2].mean(), nm, u.size, B=1000, seed=12)
                l2, h2 = stats.ci(bb)
                r[f"llr_{name}"] = {"n": nm, "diff": float(q[m].mean() - u.mean()), "upper_abs": float(max(abs(l2), abs(h2))),
                                    "margin": 2.8 * float(np.sqrt(0.5 / nm))}
        # D2: positions
        bits = np.unpackbits(S[test], axis=1).astype(np.float64)
        n = bits.shape[0]
        k1 = bits.sum(0)
        z = (k1 - n / 2) / np.sqrt(n / 4)
        pv = 2 * norm.sf(np.abs(z))
        rej = stats.holm(pv)
        r["positions"] = {"n_positions": int(k1.size), "holm_rejections": int(rej.sum()), "max_abs_z": float(np.abs(z).max()),
                          "argmax": int(np.abs(z).argmax()), "z_sd": float(z.std())}
        if c == REGISTERED["cell"]:
            j = REGISTERED["bit"]
            bt = binomtest(int(k1[j]), n, 0.5)
            ci = bt.proportion_ci(0.95)
            r["registered_bit"] = {"bit": j, "freq": float(k1[j] / n), "ci95": [ci.low, ci.high], "p_two_sided": float(bt.pvalue),
                                   "confirmed": bool(bt.pvalue < 0.05), "advantage_one_bit": float(abs(k1[j] / n - 0.5))}
        # D3
        r["byte_chi2_p"] = float(chisquare(np.bincount(S[test].reshape(-1), minlength=256)).pvalue)
        # D4: learner, cross-validated over the folds of the split
        X = seq_features(S)
        Ux = seq_features(rng.integers(0, 256, size=S.shape, dtype=np.uint8))
        correct = np.zeros(2 * len(S), bool)
        for f in range(5):
            trn = fold != f
            te = fold == f
            Xtr = np.vstack([X[trn], Ux[trn]])
            ytr = np.r_[np.zeros(trn.sum()), np.ones(trn.sum())]
            mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
            clf = LogisticRegression(C=0.05, max_iter=3000).fit((Xtr - mu) / sd, ytr)
            correct[:len(S)][te] = clf.predict((X[te] - mu) / sd) == 0
            correct[len(S):][te] = clf.predict((Ux[te] - mu) / sd) == 1
        cc, cu = correct[:len(S)][test], correct[len(S):][test]
        adv = 2 * np.r_[cc, cu].mean() - 1
        bl = stats.cluster_bootstrap(lambda ii: 2 * np.r_[cc[ii], cu[ii]].mean() - 1, cc.size, B=1000, seed=13)
        l3, h3 = stats.ci(bl)
        r["learned"] = {"adv": float(adv), "ci95": [l3, h3], "upper_abs": float(max(abs(l3), abs(h3)))}
        # pooling over M
        qs = np.array([llr_bytes(s) for s in S[test]])
        us = np.array([llr_bytes(x) for x in U])
        for M in (4, 16):
            g = len(qs) // M
            r[f"llr_M{M}"] = float(abs((qs[: g * M].reshape(g, M).sum(1) > 0).mean() - (us[: (len(us) // M) * M].reshape(-1, M).sum(1) > 0).mean()))
        out["cells"][c] = r
        print(c, "llr", round(r["llr"]["upper_abs"], 4), "learned", round(r["learned"]["upper_abs"], 4),
              "holm", r["positions"]["holm_rejections"], flush=True)
    ok = all(r["llr"]["upper_abs"] <= 0.02 and r["learned"]["upper_abs"] <= 0.02 and r["positions"]["holm_rejections"] == 0
             and all(v["upper_abs"] <= v["margin"] for k, v in r.items() if k.startswith("llr_") and isinstance(v, dict) and "margin" in v)
             for r in out["cells"].values())
    main_ok = all(r["llr"]["upper_abs"] <= 0.02 and r["learned"]["upper_abs"] <= 0.02 and r["positions"]["holm_rejections"] == 0
                  for c, r in out["cells"].items() if "_k1536_" in c)
    out["G3_all_cells"] = bool(ok)
    out["G3_main_variant"] = bool(main_ok)
    (OUT / "e1_summary.json").write_text(json.dumps(out, indent=1))
    print("G3 (all cells):", ok, "| G3 (main variant):", main_ok, "| registered bit:", out["cells"][REGISTERED["cell"]].get("registered_bit"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["extract", "analyze"])
    ap.add_argument("--workers", type=int, default=workers_default())
    a = ap.parse_args()
    extract(a) if a.step == "extract" else analyze(a)


if __name__ == "__main__":
    main()
