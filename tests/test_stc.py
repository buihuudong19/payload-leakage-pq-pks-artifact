"""STC tests: correctness (H y = m) and embedding efficiency relative to the rate-distortion bound."""
import math

import numpy as np
import pytest
from scipy.optimize import brentq

from pqpks.stego import partition, stc


def _h2(p):
    return 0.0 if p in (0, 1) else -p * math.log2(p) - (1 - p) * math.log2(1 - p)


@pytest.mark.parametrize("h", [7, 10])
@pytest.mark.parametrize("n,m", [(4000, 1000), (6000, 600), (5003, 1201)])
def test_stc_roundtrip(h, n, m):
    rng = np.random.default_rng(n + m + h)
    x = rng.integers(0, 2, n)
    rho = rng.random(n) + 0.01
    msg = rng.integers(0, 2, m)
    y, cost = stc.embed(x, rho, msg, b"seed", h)
    assert np.array_equal(stc.extract(y, m, b"seed", h), msg)
    changed = y != x
    assert abs(rho[changed].sum() - cost) < 1e-6 * max(1, cost)


def test_stc_efficiency_vs_bound():
    """Equal costs: efficiency e = m / number of changes; upper bound e* = alpha / H2^-1(alpha).
    The threshold is 80 % of the bound for the current public H_hat table (measured 0.81-0.93 depending on w). The
    optimized tables of [FJF11] reach about 90-95 %."""
    rng = np.random.default_rng(7)
    n = 20000
    for alpha in (0.5, 0.25, 0.1):
        m = int(alpha * n)
        x = rng.integers(0, 2, n)
        msg = rng.integers(0, 2, m)
        y, _ = stc.embed(x, np.ones(n), msg, b"eff", 10)
        changes = int((y != x).sum())
        e = m / changes
        a_eff = m / n                                   # all n elements are used; effective rate = m/n
        p = brentq(lambda q: _h2(q) - a_eff, 1e-9, 0.5)
        bound = a_eff / p
        assert e >= 0.80 * bound, (alpha, e, bound)


def test_fisher_yates_is_permutation_and_deterministic():
    p1 = partition.fisher_yates(10000, b"k")
    p2 = partition.fisher_yates(10000, b"k")
    assert np.array_equal(p1, p2)
    assert np.array_equal(np.sort(p1), np.arange(10000))
    assert not np.array_equal(p1, partition.fisher_yates(10000, b"k2"))


@pytest.mark.parametrize("n,m", [(5003, 1201), (180224, 92568), (81920, 12288), (262144, 2176)])
def test_block_widths_cover_all_elements(n, m):
    """sum w_i = n and w_i in {floor(n/m), ceil(n/m)}; for alpha = 0.4 in a session-start image (n1 = 180 224,
    m = 92 568) an earlier version used w = 1 and only 51 % of the region."""
    w = stc.block_widths(n, m)
    assert w.size == m and int(w.sum()) == n
    assert set(np.unique(w)) <= {n // m, -(-n // m)}


def test_full_coverage_roundtrip_and_rate():
    rng = np.random.default_rng(19)
    n, m = 9001, 4321                                   # rate ~ 0.48: an earlier version used w = 2 and 96 % of the region
    x = rng.integers(0, 2, n)
    msg = rng.integers(0, 2, m)
    y, _ = stc.embed(x, rng.random(n) + 0.01, msg, b"full", 10)
    assert np.array_equal(stc.extract(y, m, b"full", 10), msg)
    # the last element also belongs to the parity-check matrix: changing it changes the syndrome
    y2 = y.copy()
    y2[-1] ^= 1
    assert not np.array_equal(stc.extract(y2, m, b"full", 10), msg)


def test_table_covers_w_up_to_128():
    t = stc._table()
    assert t["h"] == 10 and all(str(w) in t["table"] for w in range(1, 129))
