"""E1-bis: confirmatory replication of H1 on fresh ALASKA2 images with gate G3' (registered 07/11/2026).

The rules were fixed before any data existed; this script only carries them out. E1 (e1_confirm.py) is unchanged except
where the registration states otherwise:
  * data: 40 000 ALASKA2 images (data/external/alaska2/e1bis_ids.txt), 512x512 PGM written by verify_alaska2.py;
    5 folds assigned at random per image, seed 20261107, every image has a fold (no training-only images);
  * D1 (LLR, fixed threshold): as E1 (upper 95 % CI bound of |difference| <= 0.02; saturated stratum > 5 % with margin
    2.8 sqrt(0.5/n)); per camera model, descriptive only;
  * D2 (bit positions): Holm over all positions of the 12 cells together, FWER 0.025;
  * D3 (byte-wise chi-square): reported for the 12 cells; the registered replication test is the cell of the rejection-sampled
    variant at B = 4096, two-sided, level 0.025;
  * D4 (learner): as E1 (seq_features, standardization on the training part, logistic C = 0.05, 5 folds); SIGNED advantage
    adv = 2 acc - 1; one-sided upper bound U = adv + 1.645 SD_null(L), SD_null read from results/e1bis_design/sd_null.json
    (estimated before any image, n = 40 000, R = 1000 per L); a cell passes when U <= 0.02. Two-sided CIs as in E1 are also reported.
  * G3' passes when D1 passes in every cell and the saturated stratum, D2 rejects no position, D4 passes in every cell and
    the D3 replication test does not confirm.

    python scripts/confirm/e1bis.py split
    python scripts/confirm/e1bis.py extract --workers 4
    python scripts/confirm/e1bis.py analyze
Output: results/e1bis/e1bis_summary.json
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
from pqpks.analysis import stats  # noqa: E402
from pqpks.params import AEAD_TAG_BYTES, DEFAULT_PP  # noqa: E402
from pqpks.stego.two_stage import TwoStageSTC  # noqa: E402

VARIANTS = (1536, 1252)
SEED = 20261107
DELTA, Z1 = 0.02, 1.6448536269514722
ALPHA_D2, ALPHA_D3 = 0.025, 0.025
REPL_CELL = "e1_k1252_b4096"                   # cell of the chi-square replication test (E1: p = 0.004; Holm 0.048)
CHUNK = 500
P = {"ids": ROOT / "data" / "external" / "alaska2" / "e1bis_ids.txt",
     "pgm": ROOT / "data" / "external" / "alaska2" / "pgm",
     "split": ROOT / "configs" / "splits" / "e1bis_v1.json",
     "sdnull": ROOT / "results" / "e1bis_design" / "sd_null.json",
     "out": ROOT / "results" / "e1bis"}


def load_pgm(d: Path, key: str) -> np.ndarray:
    f = d / f"{key}.pgm"
    if not f.exists():                         # test run on BOSSbase: file names without leading zeros
        f = d / f"{int(key)}.pgm"
    from PIL import Image
    x = np.asarray(Image.open(f), dtype=np.uint8)
    assert x.shape == (512, 512), (f, x.shape)
    return x


def split(a):
    rows = [ln.split() for ln in P["ids"].read_text().splitlines() if ln.strip()]
    ids = [r[0] for r in rows]
    rng = np.random.default_rng(SEED)
    fold = rng.permutation(len(ids)) % 5
    # The note is kept verbatim (in Vietnamese) because the SHA-256 of e1bis_v1.json is registered.
    # English: "E1-bis: 5 random folds per image, seed 20261107; every image has a fold".
    doc = {"note": "E1-bis (E-38): 5 fold ngẫu nhiên theo ảnh, seed 20261107; mọi ảnh đều có fold",
           "ids_file": (P["ids"].relative_to(ROOT).as_posix() if P["ids"].is_relative_to(ROOT) else str(P["ids"])),
           "ids_sha256": hashlib.sha256(P["ids"].read_bytes()).hexdigest(),
           "seed": SEED, "fold": {k: int(f) for k, f in zip(ids, fold)},
           "camera": {r[0]: (r[1] if len(r) > 1 else "") for r in rows}}
    P["split"].parent.mkdir(parents=True, exist_ok=True)
    P["split"].write_bytes(json.dumps(doc, indent=0).encode("utf-8"))     # bytes, no newline conversion: the same SHA on every platform
    print(f"{len(ids)} images; folds {np.bincount(fold).tolist()}; SHA-256 {hashlib.sha256(P['split'].read_bytes()).hexdigest()}")


def job(args):
    k, keys, pgm, out = args
    p = Path(out) / "raw" / f"chunk_{k:03d}.npz"
    if p.exists():
        return k
    emb = TwoStageSTC(DEFAULT_PP)
    rec = {f"e1_k{kb}_b{b}": [] for kb in VARIANTS for b in DEFAULT_PP.buckets}
    sat = []
    for key in keys:
        x = load_pgm(Path(pgm), key)
        sat.append(float(((x == 0) | (x == 255)).mean()))
        for kb in VARIANTS:
            for b in DEFAULT_PP.buckets:
                rec[f"e1_k{kb}_b{b}"].append(np.frombuffer(emb.extract_kem(x, kb, b + AEAD_TAG_BYTES), np.uint8))
    tmp = p.with_name(f".{p.name}")
    np.savez_compressed(tmp, keys=np.array(keys), sat=np.array(sat), **{c: np.stack(v) for c, v in rec.items()})
    tmp.replace(p)
    return k


def extract(a):
    if not a.no_frozen_check:
        check_frozen()
    sp = json.loads(P["split"].read_text(encoding="utf-8"))
    keys = sorted(sp["fold"])
    (P["out"] / "raw").mkdir(parents=True, exist_ok=True)
    chunks = [(k, keys[k * CHUNK:(k + 1) * CHUNK], str(P["pgm"]), str(P["out"])) for k in range((len(keys) + CHUNK - 1) // CHUNK)]
    t0 = time.time()
    one_thread_per_worker()
    with ProcessPoolExecutor(a.workers) as ex:
        for n, k in enumerate(ex.map(job, chunks), 1):
            print(f"chunk {k} — {n}/{len(chunks)} after {time.time() - t0:.0f}s", flush=True)


def holm_adjusted(p: np.ndarray) -> np.ndarray:
    o = np.argsort(p)
    m = p.size
    adj = np.empty(m)
    run = 0.0
    for r, i in enumerate(o):
        run = max(run, (m - r) * p[i])
        adj[i] = min(1.0, run)
    return adj


def analyze(a):
    from screen_analyze import llr_bytes, seq_features      # noqa: E402  (scripts/experiments is put on sys.path by common)
    from sklearn.linear_model import LogisticRegression
    from scipy.stats import chisquare, norm
    sp = json.loads(P["split"].read_text(encoding="utf-8"))
    sdn = json.loads(P["sdnull"].read_text(encoding="utf-8"))
    sd_null = {int(k): float(v) for k, v in sdn["sd_null"].items()}
    Z = [np.load(f) for f in sorted((P["out"] / "raw").glob("chunk_*.npz"))]
    keys = np.concatenate([z["keys"] for z in Z])
    sat = np.concatenate([z["sat"] for z in Z])
    fold = np.array([sp["fold"][str(k)] for k in keys])
    cam = np.array([sp["camera"].get(str(k), "") for k in keys])
    n = len(keys)
    out = {"split_sha256": hashlib.sha256(P["split"].read_bytes()).hexdigest(),
           "sd_null_file_sha256": hashlib.sha256(P["sdnull"].read_bytes()).hexdigest(), "sd_null": sd_null,
           "n": int(n), "n_expected": len(sp["fold"]), "cells": {}}
    pos_p = {}
    for c in Z[0].files:
        if not c.startswith("e1_"):
            continue
        S = np.concatenate([z[c] for z in Z])
        L = S.shape[1]
        rng = np.random.default_rng(zlib.crc32(("e1bis:" + c).encode()))
        r = {"length": int(L)}
        # D1: as E1
        q = np.array([llr_bytes(s) > 0 for s in S], float)
        U = rng.integers(0, 256, size=(50000, L), dtype=np.uint8)
        u = np.array([llr_bytes(x) > 0 for x in U], float)
        boot = stats.two_sample_bootstrap(lambda i1, i2: q[i1].mean() - u[i2].mean(), q.size, u.size, B=2000, seed=11)
        lo, hi = stats.ci(boot)
        r["llr"] = {"diff": float(q.mean() - u.mean()), "ci95": [lo, hi], "upper_abs": float(max(abs(lo), abs(hi)))}
        m = sat > 0.05
        nm = int(m.sum())
        if nm >= 50:
            bb = stats.two_sample_bootstrap(lambda i1, i2, qq=q[m]: qq[i1].mean() - u[i2].mean(), nm, u.size, B=1000, seed=12)
            l2, h2 = stats.ci(bb)
            r["llr_sat>5%"] = {"n": nm, "diff": float(q[m].mean() - u.mean()), "upper_abs": float(max(abs(l2), abs(h2))),
                               "margin": 2.8 * float(np.sqrt(0.5 / nm))}
        r["llr_by_camera"] = {k: {"n": int((cam == k).sum()), "diff": float(q[cam == k].mean() - u.mean())}
                              for k in sorted(set(cam)) if (cam == k).sum() >= 200}      # descriptive, not part of the gate
        # D2: p per position; Holm over the 12 cells below
        bits = np.unpackbits(S, axis=1).astype(np.float64)
        z = (bits.sum(0) - n / 2) / np.sqrt(n / 4)
        pos_p[c] = 2 * norm.sf(np.abs(z))
        r["positions"] = {"n_positions": int(z.size), "max_abs_z": float(np.abs(z).max()), "argmax": int(np.abs(z).argmax()),
                          "z_sd": float(z.std())}
        # D3
        r["byte_chi2_p"] = float(chisquare(np.bincount(S.reshape(-1), minlength=256)).pvalue)
        # D4
        X = np.vstack([seq_features(S[i:i + 2000]) for i in range(0, n, 2000)])
        Uraw = rng.integers(0, 256, size=S.shape, dtype=np.uint8)
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
        adv = float(2 * np.r_[cc, cu].mean() - 1)
        upper = adv + Z1 * sd_null[L]
        bl = stats.cluster_bootstrap(lambda ii: 2 * np.r_[cc[ii], cu[ii]].mean() - 1, n, B=1000, seed=13)
        l3, h3 = stats.ci(bl)
        per = np.array([2 * np.r_[cc[fold == f], cu[fold == f]].mean() - 1 for f in range(5)])
        r["learned"] = {"adv": adv, "sd_null": sd_null[L], "upper_one_sided": float(upper), "pass": bool(upper <= DELTA),
                        "ci95_two_sided_E1_style": [l3, h3], "per_fold": per.tolist()}
        qs = np.array([llr_bytes(s) for s in S])
        us = np.array([llr_bytes(x) for x in U])
        for M in (4, 16):
            g = len(qs) // M
            r[f"llr_M{M}"] = float(abs((qs[: g * M].reshape(g, M).sum(1) > 0).mean() - (us[: (len(us) // M) * M].reshape(-1, M).sum(1) > 0).mean()))
        out["cells"][c] = r
        print(c, "llr", round(r["llr"]["upper_abs"], 4), "D4 adv", round(adv, 4), "U", round(upper, 4), "chi2 p",
              round(r["byte_chi2_p"], 4), flush=True)
    # D2: Holm over every position of the 12 cells, FWER 0.025
    cells = list(pos_p)
    allp = np.concatenate([pos_p[c] for c in cells])
    rej = stats.holm(allp, alpha=ALPHA_D2)
    k0 = 0
    for c in cells:
        k1 = k0 + pos_p[c].size
        out["cells"][c]["positions"]["holm_pooled_rejections"] = int(rej[k0:k1].sum())
        k0 = k1
    out["D2"] = {"n_positions": int(allp.size), "alpha": ALPHA_D2, "rejections": int(rej.sum()), "min_p": float(allp.min())}
    # D3: registered replication test
    p3 = out["cells"][REPL_CELL]["byte_chi2_p"]
    out["D3_replication"] = {"cell": REPL_CELL, "p": p3, "alpha": ALPHA_D3, "confirmed": bool(p3 < ALPHA_D3)}
    out["D3_all_holm"] = dict(zip(cells, holm_adjusted(np.array([out["cells"][c]["byte_chi2_p"] for c in cells])).tolist()))

    def cell_ok(c):
        r = out["cells"][c]
        d1 = r["llr"]["upper_abs"] <= DELTA and all(v["upper_abs"] <= v["margin"] for k, v in r.items()
                                                     if k == "llr_sat>5%")
        return d1 and r["learned"]["pass"] and r["positions"]["holm_pooled_rejections"] == 0 and \
            not (c == REPL_CELL and out["D3_replication"]["confirmed"])
    out["G3prime_all_cells"] = bool(all(cell_ok(c) for c in cells))
    out["G3prime_main_variant"] = bool(all(cell_ok(c) for c in cells if "_k1536_" in c))
    out["failing_cells"] = [c for c in cells if not cell_ok(c)]
    P["out"].mkdir(parents=True, exist_ok=True)
    (P["out"] / "e1bis_summary.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print("G3' (12 cells):", out["G3prime_all_cells"], "| main variant:", out["G3prime_main_variant"], "| failing cells:", out["failing_cells"],
          "| D2 rejections:", out["D2"]["rejections"], "| D3 replication:", out["D3_replication"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["split", "extract", "analyze"])
    ap.add_argument("--workers", type=int, default=workers_default())
    ap.add_argument("--ids", help="test runs only: another image list")
    ap.add_argument("--pgm", help="test runs only: another PGM directory")
    ap.add_argument("--split-file", help="test runs only")
    ap.add_argument("--sdnull", help="test runs only")
    ap.add_argument("--out", help="test runs only")
    ap.add_argument("--no-frozen-check", action="store_true", help="test runs only")
    a = ap.parse_args()
    for k, v in (("ids", a.ids), ("pgm", a.pgm), ("split", a.split_file), ("sdnull", a.sdnull), ("out", a.out)):
        if v:
            P[k] = Path(v)
    {"split": split, "extract": extract, "analyze": analyze}[a.step](a)


if __name__ == "__main__":
    main()
