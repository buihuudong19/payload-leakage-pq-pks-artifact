"""Statistics of the experiment harness: tie-breaking FLD ensemble scores, thresholds at a fixed FPR,
the two-sample bootstrap and the P_E/TOST functions used in Sections 7.4-7.6 of the paper."""
import numpy as np

from pqpks.analysis import ensemble, stats


def _toy(n=400, d=40, shift=0.25, seed=0):
    rng = np.random.default_rng(seed)
    xc = rng.normal(size=(n, d))
    xs = rng.normal(size=(n, d)) + shift
    return xc, xs


def test_ensemble_majority_decision_is_unchanged_by_tie_breaking():
    xc, xs = _toy()
    L = 31
    lr = ensemble.fit(xc[:300], xs[:300], L=L, d_sub=10, rng=np.random.default_rng(1))
    x = np.vstack([xc[300:], xs[300:]])
    votes = np.zeros(x.shape[0])
    for sub, w, t, _ in lr:
        votes += x[:, sub] @ w > t
    sc = ensemble.score(lr, x)
    assert np.array_equal(sc > 0.5, votes > L / 2)          # majority decision as in the original ensemble
    assert np.array_equal(np.rint(sc * L), votes)            # the order by number of votes is preserved
    assert np.unique(sc).size > L + 1                        # continuous scores (not only L + 1 levels)


def test_threshold_at_fpr_hits_nominal_rate_on_continuous_scores():
    rng = np.random.default_rng(3)
    val = rng.normal(size=4000)
    test = rng.normal(size=200_000)
    t = stats.threshold_at_fpr(val, 0.05)
    assert (val > t).mean() <= 0.05
    assert abs((test > t).mean() - 0.05) < 0.01


def test_two_sample_bootstrap_includes_reference_sampling_error():
    rng = np.random.default_rng(5)
    a = rng.random(500) < 0.5
    b = rng.random(500) < 0.5
    def diff(i1, i2):
        return a[i1].mean() - b[i2].mean()
    def diff_fixed_ref(i1):
        return a[i1].mean() - b.mean()
    two = stats.two_sample_bootstrap(diff, a.size, b.size, B=2000, seed=1)
    one = stats.cluster_bootstrap(diff_fixed_ref, a.size, B=2000, seed=1)
    # two samples of equal size: the bootstrap standard deviation grows by about sqrt(2)
    assert 1.25 < two.std() / one.std() < 1.6


def test_pe_at_validation_threshold_and_tost_verdicts():
    rng = np.random.default_rng(7)
    c, s = rng.normal(size=3000), rng.normal(size=3000) + 0.5
    t = stats.threshold_min_pe(c[:1000], s[:1000])
    pe = stats.pe_at(c[1000:], s[1000:], t)
    assert abs(pe - stats.pe_min(c[1000:], s[1000:])) < 0.03
    assert stats.tost_from_boot(0.0, rng.normal(0, 0.004, 4000), 0.02).verdict == "equivalent"
    assert stats.tost_from_boot(0.05, rng.normal(0.05, 0.004, 4000), 0.02).verdict == "different"
    assert stats.tost_from_boot(0.01, rng.normal(0.01, 0.02, 4000), 0.02).verdict == "inconclusive"
