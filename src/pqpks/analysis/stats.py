"""Statistics shared by E1-E5 (paper Section 7.1).

Conventions (single source for the paper):
- P_E is the primary metric of the equivalence tests; TOST with margin delta = 0.02 in P_E: equivalent when the 90 % CI
  of the difference lies in (-delta, delta).
- TPR at a fixed FPR is a secondary metric; the threshold is set on the validation covers, never on the test set.
- DeltaTPR@5%FPR = TPR(W-pay) - TPR(W-img) on the same stego images (RQ1.2(ii)); CI by cluster bootstrap (cover).
- Cluster bootstrap: the resampling unit is the cover; all images generated from one cover move together.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import stats as _st


def pe_min(score_cover: np.ndarray, score_stego: np.ndarray) -> float:
    """P_E = min_t 1/2(FPR(t) + FNR(t)) with the rule 'stego if score > t' (larger scores favour stego)."""
    sc = np.asarray(score_cover, float)
    ss = np.asarray(score_stego, float)
    allv = np.concatenate([sc, ss])
    lab = np.concatenate([np.zeros(sc.size), np.ones(ss.size)])
    order = np.argsort(-allv, kind="mergesort")
    v, y = allv[order], lab[order]
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    # consider only thresholds between distinct values
    last = np.r_[v[1:] != v[:-1], True]
    tpr = np.r_[0.0, tp[last] / ss.size]
    fpr = np.r_[0.0, fp[last] / sc.size]
    pe = 0.5 * (fpr + 1 - tpr)
    return float(min(pe.min(), 0.5))


def threshold_min_pe(score_cover: np.ndarray, score_stego: np.ndarray) -> float:
    """Threshold t that minimizes 1/2(FPR + FNR) on a given set (used on validation, then applied to test)."""
    sc = np.asarray(score_cover, float)
    ss = np.asarray(score_stego, float)
    cand = np.unique(np.concatenate([sc, ss]))
    cand = np.r_[cand[0] - 1.0, cand]
    best_t, best = cand[0], 1.0
    for t in cand:
        pe = 0.5 * ((sc > t).mean() + (ss <= t).mean())
        if pe < best:
            best, best_t = pe, t
    return float(best_t)


def pe_at(score_cover, score_stego, t) -> float:
    """1/2(FPR + FNR) at a fixed threshold t (t may be an array with one value per sample)."""
    return float(0.5 * ((np.asarray(score_cover) > t).mean() + (np.asarray(score_stego) <= t).mean()))


def auc(score_cover, score_stego) -> float:
    return float(_st.mannwhitneyu(score_stego, score_cover, alternative="two-sided").statistic
                 / (len(score_cover) * len(score_stego)))


def threshold_at_fpr(score_cover_val: np.ndarray, fpr: float) -> float:
    """Threshold t such that the fraction of validation covers with score > t does not exceed fpr."""
    s = np.sort(np.asarray(score_cover_val, float))
    k = int(math.floor((1 - fpr) * s.size))
    k = min(max(k, 0), s.size - 1)
    return float(s[k])


def clopper_pearson(k: int, n: int, level: float = 0.95) -> tuple[float, float]:
    a = 1 - level
    lo = 0.0 if k == 0 else _st.beta.ppf(a / 2, k, n - k + 1)
    hi = 1.0 if k == n else _st.beta.ppf(1 - a / 2, k + 1, n - k)
    return float(lo), float(hi)


def rule_of_three(n: int) -> float:
    """95 % upper bound on a probability when 0 events are observed in n trials (about 3/n)."""
    return float(1 - 0.05 ** (1 / n)) if n > 0 else 1.0


def cluster_bootstrap(stat_fn, n_clusters: int, B: int = 2000, seed: int = 7) -> np.ndarray:
    """Bootstrap distribution of stat_fn(idx), where idx are cluster indices resampled with replacement."""
    rng = np.random.default_rng(seed)
    out = np.empty(B)
    for b in range(B):
        out[b] = stat_fn(rng.integers(0, n_clusters, n_clusters))
    return out


def two_sample_bootstrap(stat_fn, n1: int, n2: int, B: int = 2000, seed: int = 7) -> np.ndarray:
    """Bootstrap of stat_fn(i1, i2) for two independent samples (for example cover syndromes and uniform strings): resample
    with replacement WITHIN each sample, so that the CI includes the error of the reference sample."""
    rng = np.random.default_rng(seed)
    out = np.empty(B)
    for b in range(B):
        out[b] = stat_fn(rng.integers(0, n1, n1), rng.integers(0, n2, n2))
    return out


def ci(boot: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    a = (1 - level) / 2
    return float(np.quantile(boot, a)), float(np.quantile(boot, 1 - a))


@dataclass
class TostResult:
    diff: float
    ci90: tuple[float, float]
    ci95: tuple[float, float]
    delta: float

    @property
    def verdict(self) -> str:
        """The three branches of RQ2.1."""
        if self.delta > 0 and -self.delta < self.ci90[0] and self.ci90[1] < self.delta:
            return "equivalent"
        if self.ci95[0] > 0 or self.ci95[1] < 0:
            return "different"
        return "inconclusive"

    def as_dict(self):
        return {"diff": self.diff, "ci90": self.ci90, "ci95": self.ci95, "delta": self.delta, "verdict": self.verdict}


def tost_from_boot(diff: float, boot: np.ndarray, delta: float = 0.02) -> TostResult:
    return TostResult(float(diff), ci(boot, 0.90), ci(boot, 0.95), delta)


def holm(pvals: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Boolean array: which hypotheses are rejected under Holm-Bonferroni."""
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    m = p.size
    rej = np.zeros(m, bool)
    for k, i in enumerate(order):
        if p[i] <= alpha / (m - k):
            rej[i] = True
        else:
            break
    return rej


def interp_crossing(xs, ys, level: float) -> float | None:
    """Largest x on the grid (linear interpolation) with y >= level, for y decreasing in x. None if y < level everywhere."""
    xs = np.asarray(xs, float)
    ys = np.asarray(ys, float)
    if ys[0] < level:
        return None
    for k in range(len(xs) - 1):
        if ys[k] >= level > ys[k + 1]:
            t = (ys[k] - level) / (ys[k] - ys[k + 1])
            return float(xs[k] + t * (xs[k + 1] - xs[k]))
    return float(xs[-1])                                   # y >= level on the whole grid: alpha* >= grid maximum
