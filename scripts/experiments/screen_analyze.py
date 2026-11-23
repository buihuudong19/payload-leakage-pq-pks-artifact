"""Analysis of the screening round (reduced E1, E2, E3; alpha*; p_fail). Input: results/archive/pre-confirm/screen_v1/raw/*.npz.

Design (paper, Sections 7.1-7.4; Table 7):
- 5-fold cross-validation by cover, stratified by source; in each training part 20 % of the covers form the validation set
  (5 % FPR threshold, fitting W-comb, choice of the strongest warden), selected by a hash of the cover identifier so that every
  system uses the same split; every cover is in the test set exactly once;
- the FLD ensemble score is the tie-breaking vote fraction, so the 5 % FPR threshold can be set exactly;
- W-img: SPAM-686 + FLD ensemble; W-A0: SPAM computed separately on A0 and on the rest of the image (1372 dimensions);
  W-pay: the LLR of Algorithm 6 on the publicly extracted string (no training); W-comb: logistic regression on
  (W-img score, Lambda of W-pay) fitted on validation;
- P_E on the pooled test scores; TPR@5%FPR with the threshold from the validation covers; CIs by cluster (cover) bootstrap;
- TOST on P_E with delta = 0.02; dTPR(W-pay - W-img) for RQ1.2(ii).
These are screening results: the variation due to retraining (>= 10 splits) is not included in the CIs.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from sklearn.linear_model import LogisticRegression  # noqa: E402

from pqpks.analysis import dpub, ensemble, stats  # noqa: E402
from pqpks.params import DEFAULT_PP  # noqa: E402

ALPHAS = (0.1, 0.2, 0.4)
SYSTEMS_AT = {0.1: ("PQ-PKS", "REF-U"), 0.2: ("BL5", "BL5-L", "PQ-PKS", "REF-U"),
              0.4: ("BL5", "BL5-L", "PQ-PKS", "REF-U")}
FPR = 0.05
DELTA = 0.02
B_BOOT = 2000


# ------------------------------------------------------------------ loading the data
def load(raw: Path):
    recs = []
    for f in sorted(raw.glob("*.npz")):
        z = np.load(f, allow_pickle=False)
        recs.append({k: z[k] for k in z.files})
    return recs


def llr_bytes(b: np.ndarray) -> float:
    return dpub.mlkem_raw_llr(bytes(b[:960]))


# ------------------------------------------------------------------ E1
def e1(recs, rng):
    out = {}
    nU = 20000
    U = rng.integers(0, 256, size=(nU, 1536), dtype=np.uint8)
    llr_u = np.array([llr_bytes(u) for u in U])
    srcs = np.array([str(r["src"]) for r in recs])
    sat = np.array([float(r["sat_frac"]) for r in recs])
    for kb in (1536, 1252):
        for b in DEFAULT_PP.buckets:
            key = f"e1_k{kb}_b{b}"
            have = [r for r in recs if key in r]
            if len(have) < 20:
                continue
            S = np.stack([r[key] for r in have])
            src = np.array([str(r["src"]) for r in have])
            satk = np.array([float(r["sat_frac"]) for r in have])
            llr_q = np.array([llr_bytes(s) for s in S])
            res = {"n": len(have)}
            # D1: LLR with the fixed threshold Lambda > 0 (the test of the paper, Prop. 3)
            iq, iu = (llr_q > 0).astype(float), (llr_u > 0).astype(float)
            def adv_fixed(i1, i2):                        # signed difference; delta = |difference|
                return iq[i1].mean() - iu[i2].mean()
            a = adv_fixed(np.arange(len(have)), np.arange(nU))
            # two independent samples: both the covers and the uniform strings are resampled (p_U is not treated as a constant)
            boot = stats.two_sample_bootstrap(adv_fixed, len(have), nU, B=B_BOOT, seed=11)
            lo, hi = stats.ci(boot)
            res["llr_fixed_diff"] = float(a)
            res["llr_fixed_ci95"] = (lo, hi)
            res["llr_fixed_upper95_abs"] = float(max(abs(lo), abs(hi)))   # compared with the 0.02 threshold of G3
            # by stratum
            for name, m in (("boss", src == "boss"), ("bows2", src == "bows2"), ("sat>5%", satk > 0.05)):
                if m.sum() >= 20:
                    res[f"llr_fixed_adv_{name}"] = abs((llr_q[m] > 0).mean() - (llr_u > 0).mean())
                    res[f"n_{name}"] = int(m.sum())
            # D2: bias per bit position (binomial), Holm
            bits = np.unpackbits(S, axis=1).astype(np.float64)
            k1 = bits.sum(0)
            n = bits.shape[0]
            from scipy.stats import binomtest
            if kb == 1536 and b == 4096:
                pv = np.array([binomtest(int(k), n, 0.5).pvalue for k in k1])
                res["pos_pvals_head"] = None
                np.save(OUT / "e1_pos_pvals_k1536_b4096.npy", pv)
            else:
                z = (k1 - n / 2) / np.sqrt(n / 4)
                from scipy.stats import norm
                pv = 2 * norm.sf(np.abs(z))
            rej = stats.holm(pv)
            res["pos_rejections_holm"] = int(rej.sum())
            res["pos_max_abs_bias"] = float(np.abs(k1 / n - 0.5).max())
            zpos = (k1 - n / 2) / np.sqrt(n / 4)
            res["pos_argmax"] = int(np.abs(zpos).argmax())
            res["pos_max_z"] = float(zpos[res["pos_argmax"]])
            res["pos_z_sd"] = float(zpos.std())                       # calibration: ~ 1 under H0
            res["pos_holm_positions"] = [int(i) for i in np.nonzero(rej)[0]]
            res["n_positions"] = int(k1.size)
            # D3: byte-wise chi-square, pooled
            cnt = np.bincount(S.reshape(-1), minlength=256)
            from scipy.stats import chisquare
            res["byte_chi2_p_pooled"] = float(chisquare(cnt).pvalue)
            # D4: learned distinguisher (logistic) on string features, 5-fold cross-validation
            res["learned_adv"], res["learned_ci95"] = learned_distinguisher(S, rng)
            # pooling over M
            for M in (4, 16):
                g = len(have) // M
                if g >= 10:
                    qs = llr_q[: g * M].reshape(g, M).sum(1)
                    us = llr_u[: (nU // M) * M].reshape(-1, M).sum(1)
                    res[f"llr_fixed_adv_M{M}"] = abs((qs > 0).mean() - (us > 0).mean())
            out[key] = res
    return out


def seq_features(S: np.ndarray) -> np.ndarray:
    """Low-dimensional features (to avoid overfitting at the screening sample size): fraction of 1 bits, autocorrelation at lags
    1-32, byte-wise chi-square statistic, number of runs, autocorrelation over 10-bit blocks (structure of Compress_10)."""
    bits = np.unpackbits(S, axis=1).astype(np.float32) * 2 - 1
    ac = np.stack([(bits[:, :-l] * bits[:, l:]).mean(1) for l in range(1, 33)], 1)
    mean = bits.mean(1, keepdims=True)
    cnt = np.stack([np.bincount(S[i], minlength=256) for i in range(S.shape[0])]).astype(np.float32)
    e = S.shape[1] / 256
    chi = (((cnt - e) ** 2) / e).sum(1, keepdims=True) / 255
    runs = (bits[:, 1:] != bits[:, :-1]).mean(1, keepdims=True)
    return np.hstack([mean, ac, chi, runs])


def learned_distinguisher(S, rng, folds=5):
    U = rng.integers(0, 256, size=S.shape, dtype=np.uint8)
    X = np.vstack([seq_features(S), seq_features(U)])
    y = np.r_[np.zeros(len(S)), np.ones(len(U))]
    idx = rng.permutation(len(S))
    fold = np.empty(len(S), int)
    fold[idx] = np.arange(len(S)) % folds
    fold = np.r_[fold, fold]
    correct = np.zeros(len(y), bool)
    for f in range(folds):
        tr, te = fold != f, fold == f
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9            # standardization on the training part
        clf = LogisticRegression(C=0.05, max_iter=3000).fit((X[tr] - mu) / sd, y[tr])
        correct[te] = clf.predict((X[te] - mu) / sd) == y[te]
    n = len(S)
    def adv(ii):
        return 2 * np.r_[correct[:n][ii], correct[n:][ii]].mean() - 1
    a = adv(np.arange(n))
    boot = stats.cluster_bootstrap(adv, n, B=1000, seed=13)
    lo, hi = stats.ci(boot)
    return float(a), (lo, hi, float(max(abs(lo), abs(hi))))


# ------------------------------------------------------------------ E2/E3
def _vkey(cover_id: str, fold: int) -> bytes:
    import hashlib
    return hashlib.sha256(f"PQPKS-screen-val/{fold}/{cover_id}".encode()).digest()


def make_folds(srcs, k=5, seed=2026):
    rng = np.random.default_rng(seed)
    fold = np.empty(len(srcs), int)
    for s in np.unique(srcs):
        ii = np.nonzero(srcs == s)[0]
        ii = rng.permutation(ii)
        fold[ii] = np.arange(len(ii)) % k
    return fold


def wardens_for(recs, a, name, folds, rng):
    """Pooled test scores (each cover exactly once) of the four wardens for (system, alpha)."""
    key = f"a{a}_{name}"
    ok = np.array([key + "_spam" in r for r in recs])
    R = [r for r, o in zip(recs, ok) if o]
    fo = folds[ok]
    n = len(R)
    Fc = np.stack([r["spam_cover"] for r in R]).astype(np.float64)
    Fs = np.stack([r[key + "_spam"] for r in R]).astype(np.float64)
    a0c_key = f"a{a}_{name}_spamA0_cover" if name == "BL5" else f"a{a}_spamA0_cover"
    Gc = np.stack([r[a0c_key] for r in R]).astype(np.float64)
    Gs = np.stack([r[key + "_spamA0"] for r in R]).astype(np.float64)
    e0c_key = f"a{a}_{name}_ext0_cover" if name == "BL5" else f"a{a}_ext0_cover"
    Pc = np.array([llr_bytes(r[e0c_key]) for r in R])
    Ps = np.array([llr_bytes(r[key + "_ext0"]) for r in R])
    gen_stego = np.array([[p < 0.01 for p in dpub.generic_uniformity(bytes(r[key + "_ext0"])).values()] for r in R])
    out = {w: {"c": np.zeros(n), "s": np.zeros(n), "thr": np.zeros(n), "tpe": np.zeros(n)}
           for w in ("W-img", "W-A0", "W-pay", "W-comb")}
    val_pe = {w: [] for w in out}
    ids = [f"{r['src']}_{int(r['id'])}" for r in R]
    for f in range(folds.max() + 1):
        te = np.nonzero(fo == f)[0]
        trall = np.nonzero(fo != f)[0]
        # validation is selected by cover identifier and fold (not by a shared rng), so every system at the same alpha uses the SAME
        # train/validation/test sets: paired comparisons (TOST, dTPR) do not include the variation of the split
        trall = trall[np.argsort([_vkey(ids[i], f) for i in trall], kind="stable")]
        nv = max(10, len(trall) // 5)
        va, tr = trall[:nv], trall[nv:]
        scores = {}
        for w, (C, S, L, d) in {"W-img": (Fc, Fs, 51, 200), "W-A0": (Gc, Gs, 51, 300)}.items():
            lr = ensemble.fit(C[tr], S[tr], L=L, d_sub=d, rng=np.random.default_rng(f + 17))
            scores[w] = {part: (ensemble.score(lr, C[ix]), ensemble.score(lr, S[ix])) for part, ix in (("va", va), ("te", te))}
        scores["W-pay"] = {part: (Pc[ix], Ps[ix]) for part, ix in (("va", va), ("te", te))}
        # W-comb: logistic regression on (W-img, W-pay), fitted on validation
        Xv = np.vstack([np.c_[scores["W-img"]["va"][0], Pc[va]], np.c_[scores["W-img"]["va"][1], Ps[va]]])
        yv = np.r_[np.zeros(len(va)), np.ones(len(va))]
        mu, sd = Xv.mean(0), Xv.std(0) + 1e-12                 # the two features have different scales: standardization on validation
        lrc = LogisticRegression(max_iter=1000).fit((Xv - mu) / sd, yv)
        comb = {}
        for part, ix in (("va", va), ("te", te)):
            sc = lrc.decision_function((np.c_[scores["W-img"][part][0], Pc[ix]] - mu) / sd)
            ss = lrc.decision_function((np.c_[scores["W-img"][part][1], Ps[ix]] - mu) / sd)
            comb[part] = (sc, ss)
        scores["W-comb"] = comb
        for w in out:
            vc, vs = scores[w]["va"]
            t = stats.threshold_at_fpr(vc, FPR)
            tc, ts = scores[w]["te"]
            out[w]["c"][te], out[w]["s"][te], out[w]["thr"][te] = tc, ts, t
            out[w]["tpe"][te] = stats.threshold_min_pe(vc, vs)      # P_E threshold chosen on validation
            val_pe[w].append(stats.pe_min(vc, vs))
    for w in out:
        out[w]["val_pe"] = float(np.mean(val_pe[w]))
    return out, ids, gen_stego, np.array([float(r[key + "_change"]) for r in R])


def summarize_warden(o):
    """Main P_E: 1/2 (FPR + FNR) at the threshold chosen on validation (no optimistic bias); P_E_min_test for reference only."""
    c, s, t, tp = o["c"], o["s"], o["thr"], o["tpe"]
    pe = stats.pe_at(c, s, tp)
    tpr = float((s > t).mean())
    fpr = float((c > t).mean())
    def pe_b(ii):
        return stats.pe_at(c[ii], s[ii], tp[ii])
    def tpr_b(ii):
        return float((s[ii] > t[ii]).mean())
    pb = stats.cluster_bootstrap(pe_b, len(c), B=B_BOOT, seed=21)
    tb = stats.cluster_bootstrap(tpr_b, len(c), B=B_BOOT, seed=22)
    return {"P_E": pe, "P_E_ci95": stats.ci(pb), "P_E_min_test": stats.pe_min(c, s), "TPR@5": tpr, "TPR@5_ci95": stats.ci(tb), "FPR_test": fpr,
            "AUC": stats.auc(c, s), "val_P_E": o["val_pe"]}


def paired_pe_tost(oa, ob, ids_a, ids_b):
    common = sorted(set(ids_a) & set(ids_b))
    ia = np.array([ids_a.index(i) for i in common])
    ib = np.array([ids_b.index(i) for i in common])
    ca, sa, ta, cb, sb, tb = oa["c"][ia], oa["s"][ia], oa["tpe"][ia], ob["c"][ib], ob["s"][ib], ob["tpe"][ib]
    d = stats.pe_at(ca, sa, ta) - stats.pe_at(cb, sb, tb)
    def f(ii):
        return stats.pe_at(ca[ii], sa[ii], ta[ii]) - stats.pe_at(cb[ii], sb[ii], tb[ii])
    boot = stats.cluster_bootstrap(f, len(common), B=B_BOOT, seed=31)
    return stats.tost_from_boot(d, boot, DELTA).as_dict() | {"n": len(common)}


def delta_tpr(o_pay, o_img, mask=None):
    m = slice(None) if mask is None else mask
    s1, t1, s2, t2 = o_pay["s"][m], o_pay["thr"][m], o_img["s"][m], o_img["thr"][m]
    d = float((s1 > t1).mean() - (s2 > t2).mean())
    def f(ii):
        return (s1[ii] > t1[ii]).mean() - (s2[ii] > t2[ii]).mean()
    boot = stats.cluster_bootstrap(f, len(s1), B=B_BOOT, seed=41)
    lo, hi = stats.ci(boot)
    return {"dTPR": d, "ci95": (lo, hi), "positive": lo > 0}


def dpub_accuracy(o_pay, M_list=(1, 2, 4, 8), rng=None):
    """Accuracy of D_pub (LLR, threshold Lambda > 0) when the session-start images of M independent sessions are pooled (P-Q comparison)."""
    c, s = o_pay["c"], o_pay["s"]
    res = {}
    for M in M_list:
        g = len(c) // M
        if g < 10:
            continue
        accs = []
        for rep in range(20):
            pc = rng.permutation(len(c))[: g * M].reshape(g, M)
            ps = rng.permutation(len(s))[: g * M].reshape(g, M)
            accs.append(0.5 * ((c[pc].sum(1) <= 0).mean() + (s[ps].sum(1) > 0).mean()))
        res[M] = float(np.mean(accs))
    # 95 % CI for M = 1 (bootstrap by cover; cover and stego of the same image stay together), Table 7, RQ1.1
    def acc1(ii):
        return 0.5 * ((c[ii] <= 0).mean() + (s[ii] > 0).mean())
    res["1_ci95"] = stats.ci(stats.cluster_bootstrap(acc1, len(c), B=B_BOOT, seed=61))
    return res


def main():
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(ROOT / "results" / "archive" / "pre-confirm" / "screen_v1" / "raw"))
    ap.add_argument("--out", default=str(ROOT / "results" / "archive" / "pre-confirm" / "screen_v1"))
    args = ap.parse_args()
    OUT = Path(args.out)
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261105)
    recs = load(Path(args.raw))
    srcs = np.array([str(r["src"]) for r in recs])
    folds = make_folds(srcs)
    summary = {"n_covers": len(recs), "n_boss": int((srcs == "boss").sum()), "n_bows2": int((srcs == "bows2").sum()),
               "screening": True}
    summary["E1"] = e1(recs, rng)
    E2 = {}
    raw_scores = {}
    for a in ALPHAS:
        for name in SYSTEMS_AT[a]:
            o, ids, gen, chg = wardens_for(recs, a, name, folds, rng)
            raw_scores[(a, name)] = (o, ids)
            row = {w: summarize_warden(o[w]) for w in o}
            strongest = min(("W-img", "W-A0", "W-comb"), key=lambda w: row[w]["val_P_E"])
            row["strongest_on_val"] = strongest
            row["dpub_acc"] = dpub_accuracy(o["W-pay"], rng=rng)
            row["generic_reject_rate@1%"] = dict(zip(("monobit", "runs", "byte_chi2"), gen.mean(0).tolist()))
            row["change_rate"] = float(chg.mean())
            row["n"] = len(ids)
            row["dTPR_pay_minus_img"] = delta_tpr(o["W-pay"], o["W-img"])
            src_of = np.array([i.split("_")[0] for i in ids])
            row["dTPR_by_source"] = {sname: delta_tpr(o["W-pay"], o["W-img"], src_of == sname)
                                     for sname in ("boss", "bows2") if (src_of == sname).sum() >= 20}
            E2[f"{a}/{name}"] = row
            print(a, name, {w: round(row[w]["P_E"], 3) for w in ("W-img", "W-A0", "W-pay", "W-comb")}, flush=True)
    summary["E2E3"] = E2
    # TOST: PQ-PKS vs REF-U (RQ2.1) for each warden; payload vs REF-U with W-img (H2b)
    tost = {}
    for a in ALPHAS:
        ref = raw_scores[(a, "REF-U")]
        for name in SYSTEMS_AT[a]:
            if name == "REF-U":
                continue
            cur = raw_scores[(a, name)]
            for w in ("W-img", "W-A0", "W-comb"):
                tost[f"{a}/{name}-vs-REF-U/{w}"] = paired_pe_tost(cur[0][w], ref[0][w], cur[1], ref[1])
    summary["TOST"] = tost
    # alpha*: (i) P_E(W-img, REF-U) >= 0.45 ; (ii) TPR W-comb PQ-PKS - REF-U <= 0.02
    pe_ref = [E2[f"{a}/REF-U"]["W-img"]["P_E"] for a in ALPHAS]
    a_i = stats.interp_crossing(ALPHAS, pe_ref, 0.45)
    cond_ii = {str(a): E2[f"{a}/PQ-PKS"]["W-comb"]["TPR@5"] - E2[f"{a}/REF-U"]["W-comb"]["TPR@5"] for a in ALPHAS}
    def astar_boot(ii, cache={}):
        vals = []
        for a in ALPHAS:
            o = raw_scores[(a, "REF-U")][0]["W-img"]
            jj = ii % len(o["c"])
            vals.append(stats.pe_at(o["c"][jj], o["s"][jj], o["tpe"][jj]))
        x = stats.interp_crossing(ALPHAS, vals, 0.45)
        return np.nan if x is None else x
    nref = len(raw_scores[(0.1, "REF-U")][1])
    ab = stats.cluster_bootstrap(astar_boot, nref, B=500, seed=51)
    ab = ab[~np.isnan(ab)]
    summary["alpha_star"] = {"alpha_star_i": a_i, "ci95": stats.ci(ab) if ab.size else None,
                             "P_E_W-img_REF-U": dict(zip(map(str, ALPHAS), pe_ref)),
                             "cond_ii_TPR_diff_W-comb": cond_ii}
    fails = sum(int(r["stc_fail"]) for r in recs)
    n_emb = sum(sum(1 for k in r if k.endswith("_spam") and k != "spam_cover") for r in recs)
    checks = np.concatenate([r["checks"] for r in recs]) if recs else np.array([])
    summary["p_fail"] = {"failures": fails, "embeddings": n_emb + fails,
                         "upper95_per_cover": stats.rule_of_three(len(recs)) if fails == 0 else None}
    summary["decode_checks"] = {"n": int(checks.size), "ok": int(checks.sum())}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1, default=lambda x: x if not isinstance(x, np.generic) else x.item()))
    np.savez_compressed(OUT / "scores.npz", **{f"{a}|{n}|{w}|{p}": raw_scores[(a, n)][0][w][p]
                                               for (a, n) in raw_scores for w in raw_scores[(a, n)][0]
                                               for p in ("c", "s", "thr")})
    print(json.dumps({k: summary[k] for k in ("n_covers", "alpha_star", "p_fail", "decode_checks")}, indent=1, default=str))


if __name__ == "__main__":
    main()
