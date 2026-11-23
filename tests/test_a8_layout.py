"""Layout A8 (the default) and the public admissibility predicate Adm (paper, Section 5.3; Adm in Definition 1).

A8: alpha_0(B) = l_start(B)/n, so the densities in A0 and A1 are equal; the receiver tries |B| candidate sets A0(B).
A8' ("A8p"): fixed alpha_0 = 0.15, kept as an ablation.
Adm: a message that fits no bucket, or an image with too few pixels for l_start(B), is rejected BEFORE the index is spent.
"""
from dataclasses import replace

import numpy as np
import pytest

from pqpks import data, systems
from pqpks.params import AEAD_TAG_BYTES, DEFAULT_PP, KEMELEON_CT_BITS
from pqpks.systems.base import profile_id_for


@pytest.fixture(scope="module")
def pool():
    return data.cover_pool()


def test_default_layout_is_a8():
    assert DEFAULT_PP.layout == "A8"
    with pytest.raises(ValueError):
        replace(DEFAULT_PP, layout="A9")


@pytest.mark.parametrize("bucket", DEFAULT_PP.buckets)
def test_a8_densities_match_per_bucket(bucket):
    n = 512 * 512
    body_bits = 8 * (bucket + AEAD_TAG_BYTES)
    n0 = DEFAULT_PP.n0(n, KEMELEON_CT_BITS, body_bits)
    a0_density = KEMELEON_CT_BITS / n0
    a1_density = body_bits / (n - n0)
    target = (KEMELEON_CT_BITS + body_bits) / n
    assert a0_density <= target + 1e-12 <= a1_density + 1e-12
    assert abs(a0_density - a1_density) < 2e-4                 # differs only by rounding n0 up by 1 pixel


def test_a8_partition_differs_by_bucket_and_is_public():
    emb = systems.make("PQ-PKS").embedder
    m1 = emb.a0_mask((512, 512), 1536, 256 + 16)
    m2 = emb.a0_mask((512, 512), 1536, 4096 + 16)
    assert m1.sum() > m2.sum()                                 # larger bucket: smaller A0
    # A0 of a small bucket is a prefix of the same public pi_0, so it contains A0 of a larger bucket
    assert np.all(m1[m2])


@pytest.mark.parametrize("msg_len", [10, 700, 3000])
def test_a8_receiver_finds_start_image_in_any_bucket(pool, msg_len):
    sysm = systems.make("PQ-PKS")
    dk, ek = sysm.keygen()
    sess = sysm.start_session(ek, length=2)
    msg = bytes(range(256)) * (msg_len // 256) + bytes(msg_len % 256)
    y1 = sysm.send(sess, msg, pool["T1"])
    y2 = sysm.send(sess, b"next", pool["T2"])
    st = systems.ReceiverState()
    r1 = sysm.receive(dk, y1, st)
    assert r1.status == "ok" and r1.msg == msg
    assert r1.trials <= len(sysm.pp.buckets)                   # session-start branch: at most |B| AEAD openings
    r2 = sysm.receive(dk, y2, st)
    assert r2.status == "ok" and r2.msg == b"next" and r2.index == 2


def test_a8p_ablation_still_round_trips(pool):
    pp = replace(DEFAULT_PP, layout="A8p")
    sysm = systems.make("PQ-PKS", pp)
    assert sysm.pid != systems.make("PQ-PKS").pid              # the layout is part of profile_id
    dk, ek = sysm.keygen()
    sess = sysm.start_session(ek, length=1)
    y = sysm.send(sess, b"A8p", pool["T1"])
    assert sysm.receive(dk, y, systems.ReceiverState()).msg == b"A8p"


def test_profile_id_binds_layout():
    c = systems.crypto_layer("PQ-PKS")
    assert profile_id_for(c, DEFAULT_PP) != profile_id_for(c, replace(DEFAULT_PP, layout="A8p"))


def test_adm_too_long_message_does_not_consume_index(pool):
    sysm = systems.make("PQ-PKS")
    _, ek = sysm.keygen()
    sess = sysm.start_session(ek, length=2)
    with pytest.raises(systems.Inadmissible):
        sysm.send(sess, bytes(9000), pool["T1"])     # exceeds the largest bucket, 8192
    assert sess.next_index == 1                                # index not spent


def test_adm_image_too_small_does_not_consume_index():
    sysm = systems.make("PQ-PKS")
    _, ek = sysm.keygen()
    sess = sysm.start_session(ek, length=1)
    small = np.full((96, 96), 128, np.uint8)                   # 9216 pixel < ℓ_ĉ = 12 288 bit
    with pytest.raises(systems.Inadmissible):
        sysm.send(sess, b"x", small)
    assert sess.next_index == 1


def test_adm_is_public_predicate():
    """Adm depends only on |m|, pp, n and op: the same public inputs give the same result for every session and key."""
    sysm = systems.make("PQ-PKS")
    out = []
    for _ in range(2):
        _, ek = sysm.keygen()
        sess = sysm.start_session(ek, length=1)
        try:
            sysm.send(sess, bytes(9000), np.zeros((512, 512), np.uint8))
            out.append(True)
        except systems.Inadmissible:
            out.append(False)
    assert out == [False, False]
