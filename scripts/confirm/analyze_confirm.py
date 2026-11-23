"""Confirmatory E2/E3 (RQ1.1, RQ1.2, RQ2.1, alpha*; paper, Table 7 and Section 7.4). Runs on a machine with all images, after
    python scripts/experiments/screen_run.py --plan confirm --srm --workers 8 --out results/confirm_v1/raw

Step 1 - collect the features into memmaps (once):   python scripts/confirm/analyze_confirm.py consolidate
Step 2 - analysis:                                    python scripts/confirm/analyze_confirm.py analyze --reps 10

Design (paper, Sections 7.2 and 7.4): R splits (10 by default, `srm_repeated_test_sets` of split_v1.json): test = 5000
non-pilot BOSSbase images; validation = 1000 of the remaining non-pilot BOSSbase images; train = the rest of BOSSbase
(including the 1000 pilot images) + 10 000 BOWS2 images. Wardens: W-img = SRM + FLD ensemble (sealwatch, tie-breaking scores);
W-A0 = SPAM split into A0 and the rest of the image (1372 dimensions; no masked SRM); W-pay = the LLR of Algorithm 6;
W-comb = logistic regression on (W-img, W-pay) fitted on validation. The 5 % FPR threshold and the P_E threshold are chosen on
validation and evaluated once on test. CI: two-level bootstrap (resample the splits, then the covers inside each split), so the
CI includes the variation between splits. TOST on P_E, delta = 0.02. alpha*: condition (i) P_E(W-img, REF-U) >= 0.45 and
(ii) the TPR@5 % difference of W-comb (PQ-PKS - REF-U) <= 0.02.
Output: results/confirm_v1/summary.json (same main structure as results/archive/pre-confirm/screen_v1/summary.json).
Every bootstrap uses 10^4 resamples (earlier versions used 10^3 for the AUC, the pooled D_pub, delta_syn and the CI of
alpha*_i); H2a includes M = 4 (c_4 = 0.999). The "bootstrap" record of the summary stores the B, B_AUC and c_M used. The final
aggregation step (PF1, lane W11) reruns `analyze --reps 10` from the checkpoints (_R10) without retraining; promote_r10.py
then makes the 10-split version summary.json/decisions.json.

Phase D uses the same code with extra arguments:
  --base results/e7 --split e7 --cluster camera --reps 3      E7: 3 splits from configs/splits/e7_v1.json (test 2500,
                                                              validation 1000, stratified by camera model); CI by camera-model cluster bootstrap
  --base results/ablate --split cv5 --tost-ref REF-U,PQ-PKS   S2: 5 folds by cover (stratified by source), each fold is one
                                                              "split"; TOST of every variant against REF-U and against PQ-PKS
  --base results/ra --split cv5                               S5: as S2; the cells are "L1024", "L4096" (fixed message), no alpha*
  --base results/jpeg --split cv5 --wimg dctr|gfr             S3: W-img = DCTR or GFR on the decompressed image; no W-A0;
                                                              the cells are "Q2048", "Q3072" (total payload in bytes)
"""
from __future__ import annotations

import argparse
import os
import json
import time
from pathlib import Path

import numpy as np

from common import ROOT, SRM_DIM, Ensemble, load_split  # noqa: E402
from pqpks.analysis import ensemble as spam_ens, stats  # noqa: E402

BASE = ROOT / "results" / "confirm_v1"
FEAT = BASE / "feat"
CLUSTERS = None            # cluster bootstrap (E7: camera model): CLUSTERS[r] = cluster label of each test index in split r
WIMG = "srm"               # features of W-img: srm (default), dctr or gfr (S3, JPEG)
FPR, DELTA, B = 0.05, 0.02, 10000                      # B = 10^4 resamples (paper, Section 7.4)
B_AUC = 10000              # AUC, pooled D_pub by M, delta_syn and CI of alpha*_i (10^3 in earlier versions)
KEM_BYTES_BY_NAME = {"BL5": 1088, "BL5-L": 1536, "PQ-PKS": 1536, "REF-U": 1536}
M_POOL = (1, 2, 3, 4, 8, 16, 32, 64)                    # number of independent sessions pooled for D_pub (Fig. 6)
C_M = {1: 0.899, 2: 0.980, 3: 0.996, 4: 0.999}          # Bayes bound c_M = 1 - 0.5 * 0.2024^M of Prop. 2 for BL5 (Table 7, H2a);
                                                        # M = 4 matches Table 7 (M = 1, 2, 4) and Prop. 2


def _wanted(k: str) -> bool:
    return (k.endswith("_srm") or k == "srm_cover" or k.endswith("_spam") or k.endswith("spamA0")
            or k.endswith("_spamA0_cover") or k.endswith("_ext0") or k.endswith("_ext0_cover") or k.endswith("_change")
            or k == "spam_cover" or k.endswith("_dctr") or k == "dctr_cover" or k.endswith("_gfr") or k == "gfr_cover"
            or k.endswith("_bpnzac"))


def consolidate(a):
    """Collect the features into memmaps. When an (alpha, system) pair could not be embedded in a cover (no STC solution), its
    row is filled with NaN (features) or 0 (Ext0) and marked in avail.npy; `analyze` removes such covers from every analysis."""
    files = sorted((BASE / "raw").glob("*.npz"))
    meta = json.loads((BASE / "run_meta.json").read_text())
    combos = [f"a{_akey(al)}_{n}" for al, names in meta["systems_at"].items() for n in names]
    tmpl = next((f for f in files if int(np.load(f)["stc_fail"]) == 0), files[0])     # a record with every pair serves as template
    z0 = np.load(tmpl)
    keys = [k for k in z0.files if _wanted(k)]
    FEAT.mkdir(parents=True, exist_ok=True)
    ids = []
    mm = {}
    for k in keys:
        arr = np.asarray(z0[k])
        dt = np.float32 if arr.dtype.kind == "f" else arr.dtype
        mm[k] = np.lib.format.open_memmap(FEAT / f"{k}.npy", mode="w+", dtype=dt, shape=(len(files),) + arr.shape)
    avail = np.ones((len(files), len(combos)), dtype=bool)
    for r, f in enumerate(files):
        z = np.load(f)
        ids.append(f"{z['src']}_{int(z['id'])}")
        failed = set(str(v) for v in z["failed_combos"]) if "failed_combos" in z.files else set()
        for j, c in enumerate(combos):
            if c in failed or not any(f"{c}_{suf}" in z.files for suf in ("srm", "spam", "dctr", "gfr")):
                avail[r, j] = False
        for k in keys:
            if k in z.files:
                mm[k][r] = z[k]
            else:
                mm[k][r] = np.nan if mm[k].dtype.kind == "f" else 0
        if r % 1000 == 0:
            print(r, "/", len(files), flush=True)
    for m in mm.values():
        m.flush()
    (FEAT / "ids.json").write_text(json.dumps(ids))
    np.save(FEAT / "avail.npy", avail)
    (FEAT / "avail.json").write_text(json.dumps({"combos": combos, "n_covers": len(files),
                                                 "n_covers_missing_any": int((~avail.all(1)).sum()),
                                                 "missing_by_combo": {c: int((~avail[:, j]).sum()) for j, c in enumerate(combos)}}))
    print("keys:", keys)
    print("covers missing at least one (alpha, system) pair:", int((~avail.all(1)).sum()), flush=True)


def _akey(al):
    """Cell label as written by screen_run/jpeg_run: alpha (number), "L1024" (regime R-a) or "Q2048" (JPEG)."""
    try:
        return float(al)
    except (TypeError, ValueError):
        return al


def F(name):
    return np.load(FEAT / f"{name}.npy", mmap_mode="r")


def ckey(a_, name, suffix):
    """Feature key of the COVER for cell (a_, system): per system when the system has its own A0 (BL5, BL3, BL4, PQ-PKS-R, A8p),
    otherwise the common key of the default layout."""
    k = f"a{a_}_{name}_{suffix}"
    return k if (FEAT / f"{k}.npy").exists() else f"a{a_}_{suffix}"


_P256_P = 2 ** 256 - 2 ** 224 + 2 ** 192 + 2 ** 96 - 1
_P256_B = 0x5AC635D8AA3A93E7B3EBBD55769886BC651D06B0CC53B0F63BCE3C3E27D2604B


def p256_point_valid(s) -> bool:
    """D_pub of BL3 (paper, Table 4): is the 65-byte string a valid uncompressed P-256 point (0x04 || x || y, y^2 = x^3 - 3x + b)?
    A uniform string (x, y uniformly random) satisfies the curve equation with probability ~ 2^-256 (Table 4: advantage
    >= 1 - 2^-256 per point), so acc ~ 1 >= 0.99 (Table 7)."""
    b = bytes(np.asarray(s, dtype=np.uint8).tobytes())
    if len(b) != 65 or b[0] != 4:
        return False
    x, y = int.from_bytes(b[1:33], "big"), int.from_bytes(b[33:65], "big")
    return x < _P256_P and y < _P256_P and (y * y - (x * x * x - 3 * x + _P256_B)) % _P256_P == 0


def llr_vec(name):
    from screen_analyze import llr_bytes  # noqa: E402
    S = F(name)
    if S.shape[1] == 65:                       # BL3 (uncompressed P-256 point): score +-1 from the curve check
        return np.array([1.0 if p256_point_valid(s) else -1.0 for s in S])
    if S.shape[1] < 960:                       # kem_part shorter than an ML-KEM ciphertext (BL4 in regime R-a): D_pub does not apply
        return np.zeros(S.shape[0])
    return np.array([llr_bytes(np.asarray(s)) for s in S])


def rep_sets(split, ids, r, excluded=frozenset()):
    """Split r. `excluded`: indices of covers missing an (alpha, system) pair or without a result file; they are removed from
    train, validation and test of every system, so every paired comparison stays on the same images."""
    pos = {k: n for n, k in enumerate(ids) if n not in excluded}
    test = [pos[f"boss_{i}"] for i in split["srm_repeated_test_sets"][r] if f"boss_{i}" in pos]
    tset = set(test)
    pilot = {pos[f"boss_{i}"] for i in split["boss"]["pilot"] if f"boss_{i}" in pos}
    rest_boss = [pos[f"boss_{i}"] for i in split["boss"]["test"] + split["boss"]["val"] + split["boss"]["train"]
                 if f"boss_{i}" in pos and pos[f"boss_{i}"] not in tset and pos[f"boss_{i}"] not in pilot]
    rng = np.random.default_rng(500 + r)
    rest_boss = list(rng.permutation(rest_boss))
    nv = min(1000, len(rest_boss) // 2)                     # 1000 in the confirmatory step; smaller only for code tests
    val = rest_boss[:nv]
    train = rest_boss[nv:] + sorted(pilot) + [pos[f"bows2_{i}"] for i in split["bows2"]["train"] if f"bows2_{i}" in pos]
    return np.array(train, dtype=np.int64), np.array(val, dtype=np.int64), np.array(test, dtype=np.int64)


def rep_sets_e7(split, ids, r, excluded=frozenset()):
    """E7: split r of configs/splits/e7_v1.json (test 2500, validation 1000 stratified by camera model; train = the rest)."""
    pos = {k: n for n, k in enumerate(ids) if n not in excluded}
    test = [pos[k] for k in split["test_sets"][r] if k in pos]
    val = [pos[k] for k in split["val_sets"][r] if k in pos]
    used = set(test) | set(val)
    train = [n for k, n in pos.items() if n not in used]
    return np.array(train, dtype=np.int64), np.array(val, dtype=np.int64), np.array(test, dtype=np.int64)


def rep_sets_cv5(ids, r, excluded=frozenset(), k=5, seed=2026):
    """S2/S5: 5 folds by cover, stratified by source (as screen_analyze.make_folds); split r = fold r as test,
    validation = 20 % of the rest selected by a hash of the cover identifier (the same set for every system), train = the rest."""
    import hashlib
    keep = [n for n in range(len(ids)) if n not in excluded]
    srcs = np.array([ids[n].split("_")[0] for n in keep])
    rng = np.random.default_rng(seed)
    fold = np.empty(len(keep), int)
    for sname in np.unique(srcs):
        ii = rng.permutation(np.nonzero(srcs == sname)[0])
        fold[ii] = np.arange(len(ii)) % k
    keep = np.array(keep)
    te = keep[fold == r]
    rest = keep[fold != r]
    rest = rest[np.argsort([hashlib.sha256(f"PQPKS-screen-val/{r}/{ids[i]}".encode()).digest() for i in rest], kind="stable")]
    nv = max(10, len(rest) // 5) if len(rest) >= 50 else max(1, len(rest) // 2)     # the second branch only for code tests on very small sets
    return np.array(rest[nv:], dtype=np.int64), np.array(rest[:nv], dtype=np.int64), np.array(te, dtype=np.int64)


def _cell_path(al, name, R=None):
    """Checkpoint file. When the number of splits R differs from the version stored under the old name, the suffix _R<R> is
    used so that the results of an earlier run are not overwritten (e.g. C7 with 3 splits and with 10 splits)."""
    base = BASE / "cells" / f"a{al}_{name}.npz"
    if R is None:
        return base
    if base.exists():
        try:
            if int(np.load(base)["_reps"]) != R:
                return BASE / "cells" / f"a{al}_{name}_R{R}.npz"
        except Exception:
            pass
    return base


def save_cell(al, name, per, tag):
    """Checkpoint per (cell, system): the warden scores of every split are written to disk as soon as training ends, so that
    a rerun after an interruption does not retrain. The algorithms are deterministic (Ensemble seed 0, SPAM rng 17, same
    splits), so results restored from a checkpoint equal those of an uninterrupted run."""
    p = _cell_path(al, name, len(per))
    p.parent.mkdir(parents=True, exist_ok=True)
    arrs = {"_tag": np.array(tag), "_wardens": np.array(sorted(per[0].keys())), "_reps": np.array(len(per))}
    for r, d in enumerate(per):
        for w, f in d.items():
            for k, v in f.items():
                arrs[f"{r}|{w}|{k}"] = np.asarray(v)
    tmp = p.with_suffix(".tmp.npz")
    np.savez(tmp, **arrs)
    os.replace(tmp, p)


def load_cell(al, name, tag, R):
    p = _cell_path(al, name, R)
    if not p.exists():
        return None
    z = np.load(p)
    if str(z["_tag"]) != tag or int(z["_reps"]) != R:
        return None
    per = []
    for r in range(R):
        d = {}
        for w in z["_wardens"]:
            d[str(w)] = {k: (z[f"{r}|{w}|{k}"] if k in ("c", "s") else float(z[f"{r}|{w}|{k}"]))
                         for k in ("c", "s", "t5", "t1", "tpe", "val_pe")}
        per.append(d)
    return per


def _boot_idx(rng, r, n):
    """Bootstrap indices of split r: by cover, or by cluster (CLUSTERS, E7: camera model) when given."""
    if CLUSTERS is None:
        return rng.integers(0, n, n)
    lab = CLUSTERS[r]
    groups = [np.nonzero(lab == g)[0] for g in np.unique(lab)]
    pick = rng.integers(0, len(groups), len(groups))
    return np.concatenate([groups[g] for g in pick])


def wardens(a_, name, sets, cache):
    tr, va, te = sets
    key = f"a{a_}_{name}"
    out = {}
    Xc, Xs = F(f"{WIMG}_cover"), F(f"{key}_{WIMG}")
    ens = Ensemble(seed=0).fit(np.asarray(Xc[tr]), np.asarray(Xs[tr]))
    out["W-img"] = {p: (ens.score(np.asarray(Xc[ix])), ens.score(np.asarray(Xs[ix]))) for p, ix in (("va", va), ("te", te))}
    if (FEAT / f"{key}_spamA0.npy").exists():                    # no W-A0 in the DCT domain (S3)
        a0c = ckey(a_, name, "spamA0_cover")
        Gc, Gs = F(a0c), F(key + "_spamA0")
        lr = spam_ens.fit(np.asarray(Gc[tr]), np.asarray(Gs[tr]), L=51, d_sub=300, rng=np.random.default_rng(17))
        out["W-A0"] = {p: (spam_ens.score(lr, np.asarray(Gc[ix])), spam_ens.score(lr, np.asarray(Gs[ix]))) for p, ix in (("va", va), ("te", te))}
    e0c = ckey(a_, name, "ext0_cover")
    if e0c not in cache:
        cache[e0c] = llr_vec(e0c)
    if key + "_ext0" not in cache:
        cache[key + "_ext0"] = llr_vec(key + "_ext0")
    Pc, Ps = cache[e0c], cache[key + "_ext0"]
    out["W-pay"] = {p: (Pc[ix], Ps[ix]) for p, ix in (("va", va), ("te", te))}
    from sklearn.linear_model import LogisticRegression
    Xv = np.vstack([np.c_[out["W-img"]["va"][0], Pc[va]], np.c_[out["W-img"]["va"][1], Ps[va]]])
    mu, sd = Xv.mean(0), Xv.std(0) + 1e-12
    lrc = LogisticRegression(max_iter=1000).fit((Xv - mu) / sd, np.r_[np.zeros(len(va)), np.ones(len(va))])
    out["W-comb"] = {p: (lrc.decision_function((np.c_[out["W-img"][p][0], Pc[ix]] - mu) / sd),
                         lrc.decision_function((np.c_[out["W-img"][p][1], Ps[ix]] - mu) / sd)) for p, ix in (("va", va), ("te", te))}
    res = {}
    for w, d in out.items():
        vc, vs = d["va"]
        tc, ts = d["te"]
        res[w] = {"c": tc, "s": ts, "t5": stats.threshold_at_fpr(vc, FPR), "t1": stats.threshold_at_fpr(vc, 0.01),
                  "tpe": stats.threshold_min_pe(vc, vs),
                  "val_pe": stats.pe_min(vc, vs)}
    return res


def two_level(stat, reps, n_boot=B, seed=7):
    """stat(r, idx) on split r with cover indices idx; returns (mean estimate, two-level bootstrap sample)."""
    R = len(reps)
    est = float(np.mean([stat(r, np.arange(reps[r])) for r in range(R)]))
    rng = np.random.default_rng(seed)
    bs = np.empty(n_boot)
    for b in range(n_boot):
        rr = rng.integers(0, R, R)
        bs[b] = np.mean([stat(r, _boot_idx(rng, r, reps[r])) for r in rr])
    return est, bs


def pooled_acc(c, s, M, perm_c, perm_s):
    """Accuracy of D_pub (pooled Lambda > 0) when the session-start images of M independent sessions are pooled, for one grouping."""
    g = min(len(c), len(s)) // M
    cc, ss = c[perm_c[: g * M]].reshape(g, M).sum(1), s[perm_s[: g * M]].reshape(g, M).sum(1)
    return 0.5 * ((cc <= 0).mean() + (ss > 0).mean())


def dpub_rows(per, n_te, seed=60):
    """H2a, Fig. 6: accuracy of D_pub by M (mean over the splits and 20 groupings); two-level 95 % CI for M = 1, 2, 3
    (resample covers, then pool in the resampled order, so the groups are random as well)."""
    rng = np.random.default_rng(seed)
    out = {}
    for M in M_POOL:
        if min(n_te) // M < 10:
            continue
        accs = []
        for r, p in enumerate(per):
            c, s_ = p["W-pay"]["c"], p["W-pay"]["s"]
            for _ in range(20):
                accs.append(pooled_acc(c, s_, M, rng.permutation(len(c)), rng.permutation(len(s_))))
        row = {"acc": float(np.mean(accs))}
        if M in C_M:
            _, bs = two_level(lambda r, ii, M=M: pooled_acc(per[r]["W-pay"]["c"][ii], per[r]["W-pay"]["s"][ii], M,
                                                             np.arange(len(ii)), np.arange(len(ii))),
                              n_te, n_boot=B_AUC, seed=seed + M)
            row["ci95"] = stats.ci(bs)
        out[str(M)] = row
    return out


def delta_syn(cover_llr, kem_bytes, seed=70):
    """delta^LLR_syn,M (Table 7, H2a): upper 95 % CI bound of |Pr[pooled Lambda > 0 | cover syndrome] - Pr[pooled Lambda > 0 | uniform string]|,
    M = 1, 2, 3. Uniform strings: 50 000 strings of the kem_part length, generated deterministically."""
    from screen_analyze import llr_bytes  # noqa: E402
    rng = np.random.default_rng(seed)
    u = np.array([llr_bytes(x) for x in rng.integers(0, 256, size=(50000, kem_bytes), dtype=np.uint8)])
    out = {}
    for M in C_M:
        def adv(i1, i2, M=M):
            q = cover_llr[i1]
            uu = u[i2]
            gq, gu = len(q) // M, len(uu) // M
            return (q[: gq * M].reshape(gq, M).sum(1) > 0).mean() - (uu[: gu * M].reshape(gu, M).sum(1) > 0).mean()
        bs = stats.two_sample_bootstrap(adv, cover_llr.size, u.size, B=B_AUC, seed=seed + M)
        lo, hi = stats.ci(bs)
        out[str(M)] = {"adv": float(adv(np.arange(cover_llr.size), np.arange(u.size))), "upper_abs": float(max(abs(lo), abs(hi)))}
    return out


def analyze(a):
    global CLUSTERS
    ids = json.loads((FEAT / "ids.json").read_text())
    meta = json.loads((BASE / "run_meta.json").read_text())
    alphas = [_akey(al) for al in meta["alphas"]]
    systems_at = {_akey(k): v for k, v in meta["systems_at"].items()}
    kem_bytes = {**KEM_BYTES_BY_NAME, **{k: int(v) for k, v in meta.get("kem_bytes", {}).items()}}
    R = a.reps
    av_p = FEAT / "avail.npy"
    avail = np.load(av_p) if av_p.exists() else np.ones((len(ids), 1), dtype=bool)
    excluded = frozenset(int(i) for i in np.nonzero(~avail.all(1))[0])
    if a.split == "confirm":
        split = load_split()
        sets = [rep_sets(split, ids, r, excluded) for r in range(R)]
        split_sha = split["_sha256"]
    elif a.split == "e7":
        from common import sha256_file
        sp_path = ROOT / "configs" / "splits" / "e7_v1.json"
        split = json.loads(sp_path.read_text(encoding="utf-8"))
        R = min(R, len(split["test_sets"]))
        sets = [rep_sets_e7(split, ids, r, excluded) for r in range(R)]
        split_sha = sha256_file(sp_path)
    else:                                                  # cv5: 5 fold theo cover
        R = min(R, 5)
        sets = [rep_sets_cv5(ids, r, excluded) for r in range(R)]
        split_sha = None
    if a.cluster == "camera":                              # E7: CI by camera-model cluster (Table 7, H7)
        cam = split.get("camera", {})
        CLUSTERS = [np.array([cam.get(ids[i], "") for i in s_[2]]) for s_ in sets]
    cache, W = {}, {}
    tag = f"{split_sha}|{a.split}|{R}|{WIMG}|{len(excluded)}|{len(ids)}|pc1"   # pc1: D_pub of BL3 from the point check
    for al in alphas:
        for name in systems_at[al]:
            t0 = time.time()
            per = load_cell(al, name, tag, R)
            if per is not None:
                W[(al, name)] = per
                print("loaded", al, name, f"{R} reps from checkpoint", flush=True)
                continue
            W[(al, name)] = [wardens(al, name, sets[r], cache) for r in range(R)]
            save_cell(al, name, W[(al, name)], tag)
            print("trained", al, name, f"{R} reps in {time.time() - t0:.0f}s", flush=True)
    n_te = [len(s[2]) for s in sets]
    summ = {"split_sha256": split_sha, "split_mode": a.split, "cluster": a.cluster, "reps": R, "plan": meta.get("plan"), "wimg": WIMG,
            "bootstrap": {"B": B, "B_AUC": B_AUC, "c_M": {str(k): v for k, v in C_M.items()},
                          "note": "B: P_E, TPR, dTPR, TOST; B_AUC: AUC, pooled D_pub by M, delta_syn, CI of alpha*_i"},
            "alpha_by_cell": meta.get("alpha_by_cell"), "E2E3": {}, "TOST": {},
            "excluded_covers": {"n": len(excluded), "ids": sorted(ids[i] for i in excluded),
                                "rule": "covers missing at least one (alpha, system) pair because STC had no solution; removed for every system"}}
    for (al, name), per in W.items():
        row = {}
        for w in [w for w in ("W-img", "W-A0", "W-pay", "W-comb") if w in per[0]]:
            pe, pb = two_level(lambda r, ii: stats.pe_at(per[r][w]["c"][ii], per[r][w]["s"][ii], per[r][w]["tpe"]), n_te)
            tp, tb = two_level(lambda r, ii: float((per[r][w]["s"][ii] > per[r][w]["t5"]).mean()), n_te, seed=8)
            t1, t1b = two_level(lambda r, ii: float((per[r][w]["s"][ii] > per[r][w]["t1"]).mean()), n_te, seed=12)
            au, aub = two_level(lambda r, ii: stats.auc(per[r][w]["c"][ii], per[r][w]["s"][ii]), n_te, n_boot=B_AUC, seed=14)
            fpr5 = float(np.mean([(p[w]["c"] > p[w]["t5"]).mean() for p in per]))
            row[w] = {"P_E": pe, "P_E_ci95": stats.ci(pb), "TPR@5": tp, "TPR@5_ci95": stats.ci(tb),
                      "TPR@1": t1, "TPR@1_ci95": stats.ci(t1b), "AUC": au, "AUC_ci95": stats.ci(aub),
                      "realized_FPR@5_test": fpr5, "val_P_E": float(np.mean([p[w]["val_pe"] for p in per]))}
        d, db = two_level(lambda r, ii: float((per[r]["W-pay"]["s"][ii] > per[r]["W-pay"]["t5"]).mean()
                                               - (per[r]["W-img"]["s"][ii] > per[r]["W-img"]["t5"]).mean()), n_te, seed=9)
        row["dTPR_pay_minus_img"] = {"dTPR": d, "ci95": stats.ci(db)}
        acc, ab = two_level(lambda r, ii: 0.5 * ((per[r]["W-pay"]["c"][ii] <= 0).mean() + (per[r]["W-pay"]["s"][ii] > 0).mean()), n_te, seed=10)
        row["dpub_acc_M1"] = {"acc": acc, "ci95": stats.ci(ab)}
        row["dpub_acc"] = dpub_rows(per, n_te)
        if name in ("BL5", "BL5-L"):                         # H2a: acc_M >= c_M - 0.5 delta_syn,M, lower 95 % CI bound
            e0c = ckey(al, name, "ext0_cover")
            if e0c not in cache:                             # pair restored from a checkpoint: the cover LLR is not computed yet
                cache[e0c] = llr_vec(e0c)
            ds = delta_syn(cache[e0c][sets[0][2]], kem_bytes[name])
            row["delta_syn"] = ds
            row["H2a"] = {str(M): {"threshold": C_M[M] - 0.5 * ds[str(M)]["upper_abs"],
                                   "lower95": row["dpub_acc"][str(M)]["ci95"][0],
                                   "holds": bool(row["dpub_acc"][str(M)]["ci95"][0] >= C_M[M] - 0.5 * ds[str(M)]["upper_abs"])}
                          for M in C_M if str(M) in row["dpub_acc"]}
        else:                                                # H2a: PQ-PKS, REF-U at chance level, 95 % CI inside [0.49, 0.51]
            lo_, hi_ = row["dpub_acc_M1"]["ci95"]
            row["H2a"] = {"1": {"band": [0.49, 0.51], "ci95": [lo_, hi_], "holds": bool(0.49 <= lo_ and hi_ <= 0.51)}}
        row["strongest_on_val"] = min([w for w in ("W-img", "W-A0", "W-comb") if w in row], key=lambda w: row[w]["val_P_E"])
        summ["E2E3"][f"{al}/{name}"] = row
        print(al, name, {w: round(row[w]["P_E"], 3) for w in ("W-img", "W-A0", "W-pay", "W-comb") if w in row}, flush=True)
    for refname in a.tost_ref:
        for al in alphas:
            if (al, refname) not in W:
                continue
            ref = W[(al, refname)]
            for name in systems_at[al]:
                if name == refname:
                    continue
                cur = W[(al, name)]
                for w in [w for w in ("W-img", "W-A0", "W-comb") if w in cur[0]]:
                    dd, bb = two_level(lambda r, ii: stats.pe_at(cur[r][w]["c"][ii], cur[r][w]["s"][ii], cur[r][w]["tpe"])
                                       - stats.pe_at(ref[r][w]["c"][ii], ref[r][w]["s"][ii], ref[r][w]["tpe"]), n_te, seed=31)
                    summ["TOST"][f"{al}/{name}-vs-{refname}/{w}"] = stats.tost_from_boot(dd, bb, DELTA).as_dict()
    if any(isinstance(al, str) for al in alphas) or any((al, "REF-U") not in W or (al, "PQ-PKS") not in W for al in alphas):
        summ["alpha_star"] = None                          # regime R-a or PQ-PKS/REF-U missing: no alpha*
        (BASE / (a.out or "summary.json")).write_text(json.dumps(summ, indent=1, default=float))
        print("alpha_star: not applicable to plan", meta.get("plan"))
        return
    # alpha* (paper, Section 7.4): the largest alpha on the grid (linear interpolation) at which both (i) P_E(W-img, REF-U) >= 0.45
    # and (ii) TPR@5 %(W-comb, PQ-PKS) - TPR@5 %(W-comb, REF-U) <= 0.02 hold. Each condition gives a crossing; alpha* is the smaller one.
    pe_ref = [summ["E2E3"][f"{al}/REF-U"]["W-img"]["P_E"] for al in alphas]
    cond = {str(al): summ["E2E3"][f"{al}/PQ-PKS"]["W-comb"]["TPR@5"] - summ["E2E3"][f"{al}/REF-U"]["W-comb"]["TPR@5"] for al in alphas}
    a_i = stats.interp_crossing(alphas, pe_ref, 0.45)
    a_ii = stats.interp_crossing(alphas, [-cond[str(al)] for al in alphas], -0.02)
    a_star = None if a_i is None or a_ii is None else min(a_i, a_ii)
    # CI of alpha*_i: two-level bootstrap shared by every alpha (same splits, same cover indices), B_AUC resamples
    rng = np.random.default_rng(41)
    bs = []
    for _ in range(B_AUC):
        rr = rng.integers(0, R, R)
        idx = [_boot_idx(rng, r, n_te[r]) for r in rr]
        curve = [float(np.mean([stats.pe_at(W[(al, "REF-U")][r]["W-img"]["c"][ii], W[(al, "REF-U")][r]["W-img"]["s"][ii],
                                            W[(al, "REF-U")][r]["W-img"]["tpe"]) for r, ii in zip(rr, idx)])) for al in alphas]
        v = stats.interp_crossing(alphas, curve, 0.45)
        bs.append(np.nan if v is None else v)
    bs = np.array(bs)
    summ["alpha_star"] = {"alpha_star_i": a_i, "alpha_star_i_ci95": stats.ci(bs[~np.isnan(bs)]) if np.isfinite(bs).any() else None,
                          "alpha_star_i_boot_below_grid": float(np.isnan(bs).mean()),
                          "alpha_star_i_censored": bool(pe_ref[-1] >= 0.45),          # True: (i) holds on the whole grid, alpha* >= grid maximum
                          "alpha_star_ii": a_ii, "alpha_star_ii_censored": bool(cond[str(alphas[-1])] <= 0.02),
                          "P_E_W-img_REF-U": dict(zip(map(str, alphas), pe_ref)),
                          "cond_ii_TPR_diff_W-comb": cond,
                          "alpha_star": a_star,
                          "note": "alpha_star = min(alpha_star_i, alpha_star_ii); None if a condition already fails at the smallest alpha of the grid"}
    # Sensitivity analysis: threshold (i) plus the mean deviation of G4 from [CZZYF16]; the main conclusion still uses 0.45
    g4p = ROOT / "results" / "g4" / "g4_summary.json"
    if g4p.exists():
        g4 = json.loads(g4p.read_text(encoding="utf-8"))
        diffs = [r["P_E_mean"] - r["reference"]["P_E"] for r in g4["results"] if r.get("reference")]
        off = float(np.mean(diffs))
        a_i_s = stats.interp_crossing(alphas, pe_ref, 0.45 + off)
        summ["alpha_star"]["sensitivity_G4_offset"] = {
            "offset": off, "threshold": 0.45 + off, "alpha_star_i": a_i_s,
            "alpha_star": None if a_i_s is None or a_ii is None else min(a_i_s, a_ii),
            "note": "sensitivity analysis, not the main conclusion"}
    (BASE / (a.out or "summary.json")).write_text(json.dumps(summ, indent=1, default=float))
    print(json.dumps(summ["alpha_star"], indent=1, default=float))


def main():
    global BASE, FEAT
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["consolidate", "analyze"])
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--base", default=None, help="result directory (default results/confirm_v1); phase D: results/e7, results/ablate, results/ra")
    ap.add_argument("--split", choices=["confirm", "e7", "cv5"], default="confirm")
    ap.add_argument("--cluster", choices=["none", "camera"], default="none")
    ap.add_argument("--tost-ref", default="REF-U", help="reference systems of the TOST, comma separated (S2: REF-U,PQ-PKS)")
    ap.add_argument("--wimg", default=None, help="W-img features: srm (default), dctr, gfr (S3); default from run_meta.wimg_feature")
    ap.add_argument("--out", default=None, help="name of the summary file (default summary.json; S3 with GFR: summary_gfr.json)")
    a = ap.parse_args()
    a.tost_ref = [t.strip() for t in a.tost_ref.split(",") if t.strip()]
    global WIMG
    if a.base:
        BASE = (ROOT / a.base) if not Path(a.base).is_absolute() else Path(a.base)
        FEAT = BASE / "feat"
    if a.step == "analyze":
        meta = json.loads((BASE / "run_meta.json").read_text())
        WIMG = a.wimg or meta.get("wimg_feature", "srm")
    consolidate(a) if a.step == "consolidate" else analyze(a)


if __name__ == "__main__":
    main()
