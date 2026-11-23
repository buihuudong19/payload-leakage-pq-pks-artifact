"""Tests of the additional baselines and experiment tools: BL4 (adaptation of [Zha25]), REF-U / REF-1S, the
single-bucket profile parameterized by alpha, ablation A1 V1-V4, SPAM-686, the FLD ensemble and the numba
Fisher-Yates shuffle."""
import hashlib
import math
import secrets

import numpy as np
import pytest

from pqpks import data, systems
from pqpks.analysis import ensemble, spam
from pqpks.crypto import zha25
from pqpks.experiment.profiles import bucket_for_alpha, message_len_for_bucket, profile_for_alpha
from pqpks.experiment.references import ReferenceSystem
from pqpks.params import AEAD_TAG_BYTES, KEMELEON_CT_BYTES
from pqpks.stego import partition

N = 512 * 512


@pytest.fixture(scope="module")
def pool():
    return data.cover_pool()


def test_bl4_third_party_file_is_verbatim():
    if not zha25._FILE.is_file():
        pytest.skip(f"{zha25._FILE} not found (run scripts/setup/fetch_third_party.py)")
    assert hashlib.sha256(zha25._FILE.read_bytes()).hexdigest() == zha25._SHA256


def test_bl4_point_encoding_roundtrip_and_length():
    c = systems.crypto_layer("BL4")
    from pqpks.systems.base import profile_id_for
    from pqpks.params import DEFAULT_PP
    pid = profile_id_for(c, DEFAULT_PP)
    sk, pk = c.keygen()
    for _ in range(20):
        kp, ctx, sid = c.start(pk, pid)
        assert len(kp) == zha25.KEM_PART_BYTES == 72
        ctx2, sid2 = c.accept(sk, kp, pid)
        assert sid2 == sid and ctx2.export(b"t", 32) == ctx.export(b"t", 32)
    # any 72-byte string decodes to a valid point: the receiver does not reveal a "point decoding failure"
    assert all(c.accept(sk, secrets.token_bytes(72), pid) is not None for _ in range(10))


def test_bl4_end_to_end(pool):
    sysm = systems.make("BL4")
    dk, ek = sysm.keygen()
    y = sysm.send(sysm.start_session(ek, length=1), b"BL4 [Zha25]", pool["T1"])
    r = sysm.receive(dk, y, systems.ReceiverState())
    assert (r.status, r.msg) == ("ok", b"BL4 [Zha25]")
    assert sysm.receive(dk, pool["T1"], systems.ReceiverState()).status == "bottom"


@pytest.mark.parametrize("alpha", [0.1, 0.2, 0.4])
@pytest.mark.parametrize("name", ["PQ-PKS", "BL5"])
def test_profile_for_alpha_hits_exact_payload_and_roundtrips(alpha, name, pool):
    kb = systems.crypto_layer(name).kem_part_bytes
    pp = profile_for_alpha(alpha, N, kb)
    B = pp.buckets[0]
    assert kb + B + AEAD_TAG_BYTES == math.floor(alpha * N / 8)
    sysm = systems.make(name, pp)
    dk, ek = sysm.keygen()
    msg = secrets.token_bytes(message_len_for_bucket(B))
    y = sysm.send(sysm.start_session(ek, length=1), msg, pool["T2"])
    assert sysm.receive(dk, y, systems.ReceiverState()).msg == msg


def test_profile_for_alpha_rejects_too_small_alpha():
    with pytest.raises(ValueError):
        bucket_for_alpha(0.03, N, KEMELEON_CT_BYTES)


@pytest.mark.parametrize("ref", ["REF-U", "REF-1S"])
def test_reference_systems_embed_same_total_payload(ref, pool):
    x = pool["T1"]
    body = profile_for_alpha(0.2, N, KEMELEON_CT_BYTES).buckets[0] + AEAD_TAG_BYTES
    y = ReferenceSystem(ref).stego(x, body)
    d = y.astype(int) - x
    assert np.abs(d).max() == 1 and 0.01 < (d != 0).mean() < 0.2


@pytest.mark.parametrize("perm,matrix", [(False, False), (True, False), (False, True), (True, True)])
def test_ablation_a1_variants_roundtrip(perm, matrix, pool):
    sysm = systems.StcSystem(systems.crypto_layer("PQ-PKS"), keyed_perm=perm, keyed_matrix=matrix)
    dk, ek = sysm.keygen()
    y = sysm.send(sysm.start_session(ek, length=1), b"A1", pool["T4"])
    assert sysm.receive(dk, y, systems.ReceiverState()).msg == b"A1"
    assert sysm.embedder.stage2_keyed == (perm and matrix)


def test_spam686_shape_and_normalization(pool):
    f = spam.spam686(pool["T1"])
    assert f.shape == (686,)
    rows = f[:343].reshape(49, 7).sum(1)
    assert np.all((np.abs(rows - 1) < 1e-5) | (rows == 0))
    assert np.array_equal(f, spam.spam686(pool["T1"]))


def test_ensemble_separates_and_does_not_hallucinate():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(200, 50))
    r_same = ensemble.cross_validate(a, rng.normal(size=(200, 50)), L=11, d_sub=20)
    assert 0.4 < r_same["P_E"] <= 0.6
    r_diff = ensemble.cross_validate(a, rng.normal(size=(200, 50)) + 0.5, L=11, d_sub=20)
    assert r_diff["P_E"] < 0.1 and r_diff["AUC"] > 0.95


def _fy_reference(n, seed):
    perm = np.arange(n, dtype=np.int64)
    rnd = partition._Shake(seed)
    draws = rnd.u64(2 * n)
    di = 0
    for i in range(n - 1, 0, -1):
        bound = i + 1
        limit = (1 << 64) - ((1 << 64) % bound)
        while True:
            v = int(draws[di])
            di += 1
            if v < limit:
                break
        j = v % bound
        perm[i], perm[j] = perm[j], perm[i]
    return perm


@pytest.mark.parametrize("n", [2, 7, 1000, 65536])
def test_numba_fisher_yates_matches_python_reference(n):
    assert np.array_equal(partition.fisher_yates(n, b"seed-%d" % n), _fy_reference(n, b"seed-%d" % n))


def test_holm_and_paired_bootstrap():
    adj = ensemble.holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adj == pytest.approx({"a": 0.03, "c": 0.06, "b": 0.06})
    rng = np.random.default_rng(3)
    dc = rng.random(300) < 0.3
    res = {"P_E": 0.4, "dec_cover": dc, "dec_stego": rng.random(300) < 0.5}
    d = ensemble.paired_diff_ci(res, res)                     # a system compared with itself: the difference is exactly 0
    assert d["diff"] == 0 and d["ci95"] == (0.0, 0.0) and d["p_boot"] == 1.0
