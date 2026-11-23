"""Tests of the receiver bounds and of the masked SPAM features:
- a replayed session-start image returns bottom at once, without Decaps and without trying branch (b) (Algorithm 4);
  the number of AEAD trials of the receiver never exceeds |B|(1 + (2W - 1)S_max) (paper, Proposition 4), even under a
  flood of invalid images;
- `receive_msg`: every reason for rejection maps to None (A5);
- masked SPAM for warden W-A0; CI of the AUC; repeated cross-validation; sample size and power of the paired TOST.
"""
from dataclasses import replace
from statistics import NormalDist

import numpy as np
import pytest

from pqpks import data, systems
from pqpks.analysis import ensemble, spam
from pqpks.params import KEMELEON_CT_BYTES, DEFAULT_PP
from pqpks.systems import ReceiverPolicy, ReceiverState


@pytest.fixture(scope="module")
def pool():
    return data.cover_pool()


class Clock:
    def __init__(self, t=1_800_000_000.0):
        self.t = t

    def __call__(self):
        return self.t


# ------------------------------------------------------------------ receiver
@pytest.fixture(scope="module")
def flooded(pool):
    """S_max = 3 open sessions, W = 3, N = 8; each session has received images 1 and 4, so the window has moved."""
    pp = replace(DEFAULT_PP, max_open_sessions=3, window=3, session_len=8)
    clock = Clock()
    sysm = systems.make("PQ-PKS", pp, clock=clock)
    dk, ek = sysm.keygen()
    st = ReceiverState(clock=clock)
    starts = []
    covers = list(pool.values())
    for s in range(pp.max_open_sessions):
        sess = sysm.start_session(ek, length=8)
        y1 = sysm.send(sess, b"start %d" % s, covers[s])
        assert sysm.receive(dk, y1, st).status == "ok"
        starts.append(y1)
        ys = [sysm.send(sess, b"m%d" % i, covers[(s + i) % len(covers)]) for i in (2, 3, 4)]
        assert sysm.receive(dk, ys[2], st).msg == b"m4"            # image 4 received: last = 4, seen = {1, 4}
    return sysm, dk, ek, st, pp, starts


def _bound(pp, kem_bytes=KEMELEON_CT_BYTES):
    """|B_1| + (2W - 1) S_max |B_2| (separate ladder for later images; no A5b, so P_max is not added)."""
    return len(pp.buckets_for(True, kem_bytes)) + (2 * pp.window - 1) * pp.max_open_sessions * len(pp.buckets_for(False, kem_bytes))


def test_receiver_trials_bounded_under_flood(flooded, pool):
    sysm, dk, _, st, pp, _ = flooded
    for x in list(pool.values())[:4]:                           # flood of invalid images: real covers
        r = sysm.receive(dk, x, st)
        assert r.status == "bottom"
        assert 0 < r.trials <= _bound(pp)
    # per session: unseen indices in [max(2, 4 - 3 + 1), 4 + 3] = [2, 7] \\ {4} -> 5 = 2W - 1 indices
    assert r.trials == _bound(pp)


def test_replayed_start_image_is_rejected_without_decaps_or_branch_b(flooded, monkeypatch):
    sysm, dk, _, st, pp, starts = flooded
    calls = []
    orig = sysm.crypto.accept
    monkeypatch.setattr(sysm.crypto, "accept", lambda *a: calls.append(1) or orig(*a))
    r = sysm.receive(dk, starts[0], st)
    assert r.status == "replay" and r.msg is None
    assert calls == []                                          # no second Decaps
    assert r.trials == 0                                        # no AEAD trial


def test_receive_msg_collapses_every_rejection_to_none(pool):
    clock = Clock()
    sysm = systems.make("PQ-PKS", clock=clock)
    dk, ek = sysm.keygen()
    sk_s, vk_s = sysm.sig_keygen()
    st = ReceiverState(clock=clock)
    sess = sysm.start_session(ek, sk_s=sk_s, length=2)
    y = sysm.send(sess, b"hello", pool["T1"])
    pol = ReceiverPolicy(require_auth=True, vk_s=vk_s)
    assert sysm.receive_msg(dk, y, st, pol, ek) == b"hello"
    assert sysm.receive_msg(dk, y, st, pol, ek) is None                       # replay
    assert sysm.receive_msg(dk, pool["T2"], st, pol, ek) is None     # ⊥
    _, vk_other = sysm.sig_keygen()
    sess2 = sysm.start_session(ek, sk_s=sk_s, length=1)
    y2 = sysm.send(sess2, b"x", pool["T4"])
    bad = ReceiverPolicy(require_auth=True, vk_s=vk_other)
    assert sysm.receive(dk, y2, ReceiverState(clock=clock), bad, ek).status == "auth_fail"
    assert sysm.receive_msg(dk, y2, ReceiverState(clock=clock), bad, ek) is None   # bad signature
    sess3 = sysm.start_session(ek, length=1)
    y3 = sysm.send(sess3, b"old", pool["T5"])
    late = ReceiverState(clock=Clock(clock.t + sysm.pp.sid_ttl_s + 10))
    assert sysm.receive(dk, y3, late).status == "stale"
    assert sysm.receive_msg(dk, y3, late) is None                              # t0 expired


# ------------------------------------------------------------------ masked SPAM (W-A0)
def _spam_reference(img):
    """Verbatim copy of the earlier unmasked SPAM-686, used to check that the masked rewrite gives identical features."""
    T, K = 3, 7

    def mk(d, p):
        d1, d2, d3 = p
        idx = (d1 + T) * K * K + (d2 + T) * K + (d3 + T)
        cnt = np.bincount(idx.ravel(), minlength=K ** 3).astype(np.float64).reshape(K * K, K)
        nrm = cnt.sum(axis=1, keepdims=True)
        return np.divide(cnt, nrm, out=np.zeros_like(cnt), where=nrm > 0).ravel()
    x = np.asarray(img, dtype=np.int32)
    h = np.clip(x[:, :-1] - x[:, 1:], -T, T)
    v = np.clip(x[:-1, :] - x[1:, :], -T, T)
    rd = np.clip(x[:-1, :-1] - x[1:, 1:], -T, T)
    ru = np.clip(x[1:, :-1] - x[:-1, 1:], -T, T)
    hl, vu, lu, ld = -h[:, ::-1], -v[::-1], -rd[::-1, ::-1], -ru[::-1, ::-1]
    f = {"R": mk(h, (h[:, :-2], h[:, 1:-1], h[:, 2:])), "L": mk(hl, (hl[:, :-2], hl[:, 1:-1], hl[:, 2:])),
         "D": mk(v, (v[:-2], v[1:-1], v[2:])), "U": mk(vu, (vu[:-2], vu[1:-1], vu[2:])),
         "RD": mk(rd, (rd[:-2, :-2], rd[1:-1, 1:-1], rd[2:, 2:])), "LU": mk(lu, (lu[:-2, :-2], lu[1:-1, 1:-1], lu[2:, 2:])),
         "RU": mk(ru, (ru[2:, :-2], ru[1:-1, 1:-1], ru[:-2, 2:])), "LD": mk(ld, (ld[2:, :-2], ld[1:-1, 1:-1], ld[:-2, 2:]))}
    return np.concatenate([(f["R"] + f["L"] + f["D"] + f["U"]) / 4, (f["RD"] + f["LU"] + f["RU"] + f["LD"]) / 4])


def test_spam_rewrite_matches_previous_version(pool):
    for k in ("T1", "T7"):
        assert np.allclose(spam.spam686(pool[k]), _spam_reference(pool[k]), atol=1e-6)


def test_spam_full_mask_equals_unmasked_and_a0_features(pool):
    x = pool["T2"]
    assert np.allclose(spam.spam686(x, np.ones(x.shape, bool)), spam.spam686(x), atol=1e-6)
    sysm = systems.make("PQ-PKS")
    m = sysm.embedder.a0_mask(x.shape, sysm.crypto.kem_part_bytes, 4096 + 16)   # A8: A0 of bucket 4096
    f = spam.spam_a0(x, m)
    assert f.shape == (1372,)
    assert not np.allclose(f[:686], f[686:])                    # the two regions count over different sets of triples


def test_spam_mask_selects_triples_by_anchor():
    """Image whose columns 0..W-1 increase: every horizontal difference is -1. A one-pixel anchor mask selects exactly one
    triple per horizontal direction, so the (-1, -1) row of the transition matrices for -> and <- has probability 1 at D3 = -1."""
    x = np.tile(np.arange(16), (16, 1)).astype(np.uint8)
    m = np.zeros_like(x, bool)
    m[5, 7] = True
    f = spam.spam686(x, m)
    row = (-1 + 3) * 7 + (-1 + 3)
    # F1 averages 4 directions; -> contributes 1/4 at D3 = -1; <- (sign reversed) contributes 1/4 at D3 = +1
    assert f[row * 7 + (-1 + 3)] == pytest.approx(0.25)


# ------------------------------------------------------------------ statistics
def test_bootstrap_auc_and_repeated_cv():
    rng = np.random.default_rng(5)
    a = rng.normal(size=(300, 30))
    b = rng.normal(size=(300, 30)) + 0.25
    res = ensemble.cross_validate(a, b, L=11, d_sub=10)
    lo, hi = ensemble.bootstrap_auc(res, B=300)
    assert lo < res["AUC"] < hi and hi - lo < 0.2
    rep = ensemble.repeated_cv(a, b, R=4, L=11, d_sub=10)
    assert len(rep["P_E_runs"]) == 4 and rep["P_E_sd"] > 0
    assert abs(rep["P_E_mean"] - res["P_E"]) < 0.08


def test_paired_terms_reproduce_pe_difference():
    rng = np.random.default_rng(6)
    a, b, c = (rng.normal(size=(200, 20)) for _ in range(3))
    ra = ensemble.cross_validate(a, b + 0.3, L=7, d_sub=10)
    rb = ensemble.cross_validate(a, c + 0.2, L=7, d_sub=10)
    g = ensemble.paired_pe_terms(ra, rb)
    assert g.mean() == pytest.approx(ra["P_E"] - rb["P_E"])


def test_tost_sample_size_matches_closed_form():
    sd, delta = 0.3, 0.02
    z = NormalDist().inv_cdf
    closed = (sd * (z(0.95) + z(0.9)) / delta) ** 2               # true difference 0, power 0.8: z_{1-beta/2} = z_{0.9}
    n = ensemble.tost_pairs_needed(sd, delta)
    assert abs(n - closed) / closed < 0.01
    assert ensemble.tost_power(sd, n, delta) >= 0.8 > ensemble.tost_power(sd, n - 50, delta)
