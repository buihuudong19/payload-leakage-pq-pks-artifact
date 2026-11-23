"""FLD ensemble classifier (Kodovsky, Fridrich, Holub, IEEE TIFS 2012), compact implementation.

L FLD base learners on random subspaces of dimension d_sub; each base learner sets the threshold that minimizes P_E on the
training set; the final decision is a majority vote. Score = fraction of "stego" votes plus a tie-breaking term: the vote
fraction takes only L + 1 values, so a threshold at a fixed FPR falls inside a group of equal scores and the realized FPR
on the test set drifts far from the nominal level; the extra term makes the score continuous without changing the order
by number of votes.
Reimplemented from the original paper. Calibration against reference detection errors is gate G4 (scripts/confirm/g4_calibrate.py).
Evaluation uses cross-validation by IMAGE PAIR: the cover and the stego image of the same picture are always in the same fold.
The threshold of each FLD is set on the training folds; P_E is reported on the test fold (no threshold is tuned on test data).
"""
from __future__ import annotations

import numpy as np


def _fld(xc, xs, reg=1e-6):
    mc, ms = xc.mean(0), xs.mean(0)
    sw = np.cov(xc, rowvar=False) + np.cov(xs, rowvar=False)
    sw += reg * np.trace(sw) / sw.shape[0] * np.eye(sw.shape[0])
    w = np.linalg.solve(sw, ms - mc)
    pc, ps = xc @ w, xs @ w
    scale = float(np.concatenate([pc, ps]).std()) + 1e-12      # scale of the margin, used for tie breaking
    cand = np.sort(np.concatenate([pc, ps]))
    # threshold that minimizes P_E = (FPR + FNR)/2 on the training set
    best_t, best = cand[0], 1.0
    for t in cand[:: max(1, cand.size // 400)]:
        pe = 0.5 * ((pc > t).mean() + (ps <= t).mean())
        if pe < best:
            best, best_t = pe, t
    return w, best_t, scale


def fit(xc, xs, L=51, d_sub=200, rng=None):
    rng = rng or np.random.default_rng(0)
    d = xc.shape[1]
    learners = []
    for _ in range(L):
        sub = rng.choice(d, size=min(d_sub, d), replace=False)
        w, t, sc = _fld(xc[:, sub], xs[:, sub])
        learners.append((sub, w, t, sc))
    return learners


def score(learners, x):
    """(number of votes + 0.49*tanh(normalized mean margin)) / L.

    The extra term lies in (-0.49, 0.49), so the score intervals for k and k + 1 votes do not overlap: the order by number
    of votes is preserved and the majority decision (score > 0.5, L odd) is the same as in the original method. Among
    images with the same number of votes, a larger mean margin ranks closer to stego."""
    votes = np.zeros(x.shape[0])
    margin = np.zeros(x.shape[0])
    for sub, w, t, sc in learners:
        p = x[:, sub] @ w
        votes += p > t
        margin += (p - t) / sc
    L = len(learners)
    return (votes + 0.49 * np.tanh(margin / L)) / L


def _auc(sc, ss):
    allv = np.concatenate([sc, ss])
    ranks = allv.argsort().argsort().astype(np.float64) + 1
    # handle tied ranks
    _, inv, counts = np.unique(allv, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=ranks)
    ranks = (sums / counts)[inv]
    rs = ranks[sc.size:].sum()
    return (rs - ss.size * (ss.size + 1) / 2) / (sc.size * ss.size)


def cross_validate(fc, fs, folds=5, seed=0, L=51, d_sub=200):
    """fc, fs: cover/stego features; row i of both arrays is the same picture. Returns P_E, AUC and the per-image
    decisions on the test fold (for bootstrap and for paired comparisons between two systems)."""
    n = fc.shape[0]
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    dec_c = np.zeros(n, dtype=bool)
    dec_s = np.zeros(n, dtype=bool)
    sc_c = np.zeros(n)
    sc_s = np.zeros(n)
    for k in range(folds):
        te = order[k::folds]
        tr = np.setdiff1d(order, te)
        mu, sd = fc[tr].mean(0), fc[tr].std(0) + 1e-9
        model = fit((fc[tr] - mu) / sd, (fs[tr] - mu) / sd, L, d_sub, np.random.default_rng(seed + k))
        sc_c[te] = score(model, (fc[te] - mu) / sd)
        sc_s[te] = score(model, (fs[te] - mu) / sd)
    dec_c, dec_s = sc_c > 0.5, sc_s > 0.5
    pe = 0.5 * (dec_c.mean() + (~dec_s).mean())
    return {"P_E": float(pe), "AUC": float(_auc(sc_c, sc_s)), "fpr": float(dec_c.mean()),
            "tpr": float(dec_s.mean()), "dec_cover": dec_c, "dec_stego": dec_s,
            "score_cover": sc_c, "score_stego": sc_s}


def bootstrap_auc(res: dict, B=2000, seed=3):
    """95 % CI of the AUC by bootstrap over images (cover/stego pairs kept together), same convention as bootstrap_pe.
    As in bootstrap_pe, the classifier is kept fixed; the variation due to fold assignment is measured by repeated_cv."""
    rng = np.random.default_rng(seed)
    sc, ss = res["score_cover"], res["score_stego"]
    n = sc.size
    vals = np.empty(B)
    for b in range(B):
        i = rng.integers(0, n, n)
        vals[b] = _auc(sc[i], ss[i])
    return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def repeated_cv(fc, fs, R=10, folds=5, L=51, d_sub=200, seed0=100):
    """Repeat cross-validation R times with different seeds (different folds and subspaces) on the SAME data.
    Returns the mean and standard deviation of P_E and AUC over repetitions: the training component of the variance,
    which the bootstrap CI (fixed classifier) does not contain. The registered protocol uses 10 splits for SRM."""
    pe, auc, runs = [], [], []
    for r in range(R):
        res = cross_validate(fc, fs, folds=folds, seed=seed0 + 1000 * r, L=L, d_sub=d_sub)
        pe.append(res["P_E"])
        auc.append(res["AUC"])
        runs.append(res)
    pe, auc = np.array(pe), np.array(auc)
    return {"P_E_mean": float(pe.mean()), "P_E_sd": float(pe.std(ddof=1)), "AUC_mean": float(auc.mean()),
            "AUC_sd": float(auc.std(ddof=1)), "P_E_runs": pe.tolist(), "runs": runs}


def paired_pe_terms(res_a: dict, res_b: dict) -> np.ndarray:
    """Contribution of each image to P_E(a) - P_E(b): g_i = 1/2[(c_a - c_b) + (not s_a - not s_b)], so the difference is mean(g)."""
    ca, sa, cb, sb = (res_a["dec_cover"], res_a["dec_stego"], res_b["dec_cover"], res_b["dec_stego"])
    return 0.5 * ((ca.astype(float) - cb) + ((~sa).astype(float) - (~sb)))


def tost_power(sd_term: float, n: int, delta: float, alpha=0.05, true_diff=0.0, sd_train=0.0) -> float:
    """Normal-approximation power of the paired TOST at margin delta: SE^2 = sd_term^2/n + sd_train^2.
    sd_term: standard deviation of g_i (paired_pe_terms); sd_train: standard deviation of the difference due to training
    (repeated_cv), assumed already scaled to sample size n."""
    from statistics import NormalDist
    z = NormalDist().inv_cdf(1 - alpha)
    se = (sd_term ** 2 / n + sd_train ** 2) ** 0.5
    nd = NormalDist()
    p = nd.cdf((delta - true_diff) / se - z) - nd.cdf((-delta - true_diff) / se + z)
    return float(max(0.0, p))


def tost_pairs_needed(sd_term: float, delta: float, alpha=0.05, power=0.8, true_diff=0.0, sd_train=0.0) -> int:
    """Smallest number of image pairs for the paired TOST to reach power `power` (normal approximation).
    With true_diff = 0 and sd_train = 0: n >= sd^2(z_{1-alpha} + z_{1-beta/2})^2/delta^2; in general the smallest n is found
    with tost_power. `sd_train` is the training component and is treated as NOT decreasing with n; when it is already too
    large compared with delta, no sample size is sufficient and the function raises an error."""
    if tost_power(sd_term, 10 ** 8, delta, alpha, true_diff, sd_train) < power:
        raise ValueError("the target power cannot be reached with this delta at any sample size")
    n = 2
    while tost_power(sd_term, n, delta, alpha, true_diff, sd_train) < power:
        n = int(n * 1.05) + 1
        if n > 10 ** 8:
            raise ValueError("the target power cannot be reached with this delta")
    lo, hi = max(2, int(n / 1.05) - 1), n
    while lo < hi:                                     # narrow down to the smallest n
        mid = (lo + hi) // 2
        if tost_power(sd_term, mid, delta, alpha, true_diff, sd_train) >= power:
            hi = mid
        else:
            lo = mid + 1
    return lo


def bootstrap_pe(dec_c, dec_s, B=2000, seed=1):
    """95 % CI of P_E by bootstrap over IMAGES (image indices resampled, cover/stego pairs kept)."""
    rng = np.random.default_rng(seed)
    n = dec_c.size
    vals = np.empty(B)
    for b in range(B):
        i = rng.integers(0, n, n)
        vals[b] = 0.5 * (dec_c[i].mean() + (~dec_s[i]).mean())
    return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def paired_diff_ci(res_a: dict, res_b: dict, B=2000, seed=2):
    """CI of P_E(a) - P_E(b) when both systems use the SAME covers (paired bootstrap over images; each system has its own
    classifier and therefore its own decisions on the covers). TOST at margin delta: equivalent if the 90 % CI lies
    entirely inside (-delta, delta). p_boot: two-sided p-value from the bootstrap distribution (no multiplicity correction).
    Note: the classifiers are kept fixed during resampling, so the CI does not include the variation due to fold
    assignment and subspace selection; the CI is therefore slightly narrow."""
    rng = np.random.default_rng(seed)
    ca, sa, cb, sb = res_a["dec_cover"], res_a["dec_stego"], res_b["dec_cover"], res_b["dec_stego"]
    n = ca.size
    d = np.empty(B)
    for b in range(B):
        i = rng.integers(0, n, n)
        d[b] = 0.5 * (ca[i].mean() + (~sa[i]).mean()) - 0.5 * (cb[i].mean() + (~sb[i]).mean())
    p_boot = float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())))
    return {"diff": float(res_a["P_E"] - res_b["P_E"]), "p_boot": p_boot,
            "ci95": (float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))),
            "ci90": (float(np.quantile(d, 0.05)), float(np.quantile(d, 0.95)))}


def holm(pvals: dict) -> dict:
    """Holm-Bonferroni adjustment for a family of tests {name: p}. Returns {name: adjusted p}."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    out, running = {}, 0.0
    for k, (name, p) in enumerate(items):
        running = max(running, min(1.0, (m - k) * p))
        out[name] = running
    return out
