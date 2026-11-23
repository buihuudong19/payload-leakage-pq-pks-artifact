"""E4 tests by the class of the change applied to the extracted bits.

The expected outcome follows from the extraction itself, not from the number of modified pixels:
  (a) a change that alters the extracted bits of stage 1 or stage 2 -> bottom;
  (b) a change that keeps the extracted bits (+-2 keeps the LSB) -> the same m;
  (c) a change at pixels outside the positions used by STC -> the same m. STC uses all of A0 and A1, so for a
      session-start image this class is empty; the test checks that;
  (d) destruction (splicing two stego images) -> bottom;
  property test: for random modifications the outcome matches exactly whether the extracted bits changed.
In addition, Kemeleon is many-to-one, so a modified c_hat may decode to the same c; bottom still follows because sid changes.
"""
import secrets

import numpy as np
import pytest

from pqpks import data, systems
from pqpks.crypto import framing, kemeleon
from pqpks.params import AEAD_TAG_BYTES, FLAG_SESSION_START, KEMELEON_B, KEMELEON_T, KEMELEON_VEC_BYTES, MLKEM_N, MLKEM_Q
from pqpks.systems.base import image_keys, open_image, seal_image

MSG = b"E4: tamper test message"


@pytest.fixture(scope="module")
def setup():
    pool = data.cover_pool()
    sysm = systems.make("PQ-PKS")
    dk, ek = sysm.keygen()
    sess = sysm.start_session(ek, length=1)
    keys = sess.keys[1]                                   # send() deletes the key from the session after use
    stego = sysm.send(sess, MSG, pool["T1"])
    emb = sysm.embedder
    kb = sysm.crypto.kem_part_bytes
    m0 = 8 * kb
    body_len = len(framing.pack(framing.Frame(MSG, FLAG_SESSION_START, None, 0), sysm.pp.buckets)) + AEAD_TAG_BYTES
    a0, _ = emb._sets(stego.shape, m0, 8 * body_len)      # A8: A0 depends on the bucket
    used0 = a0                                            # STC uses the whole region
    idx1 = emb.stage2_index(stego.shape, kb, keys.perm, body_len)
    used1 = idx1
    return dict(pool=pool, sysm=sysm, dk=dk, ek=ek, sess=sess, keys=keys, stego=stego, kb=kb,
                used0=used0, used1=used1, body_len=body_len, idx1=idx1)


def _extractions(s, img):
    emb = s["sysm"].embedder
    e0 = emb.extract_kem(img, s["kb"], s["body_len"])
    e1 = emb.extract_body(img, s["idx1"], s["body_len"], s["keys"].stc)
    return e0, e1


def _modify(img, positions, delta):
    out = img.astype(np.int16).reshape(-1).copy()
    out[positions] += delta
    assert out.min() >= 0 and out.max() <= 255
    return out.astype(np.uint8).reshape(img.shape)


def _pick(rng, pool_idx, k, flat, lo, hi):
    ok = pool_idx[(flat[pool_idx] >= lo) & (flat[pool_idx] <= hi)]
    return rng.choice(ok, size=k, replace=False)


def _recv(s, img):
    return s["sysm"].receive(s["dk"], img, systems.ReceiverState())


def test_baseline_ok(setup):
    assert _recv(setup, setup["stego"]).msg == MSG


@pytest.mark.parametrize("k", [1, 4, 16])
@pytest.mark.parametrize("stage", [0, 1])
def test_a_change_of_extracted_bits_gives_bottom(setup, k, stage):
    rng = np.random.default_rng(1000 * k + stage)
    flat = setup["stego"].reshape(-1)
    pos = _pick(rng, setup["used0"] if stage == 0 else setup["used1"], k, flat, 1, 254)
    img = _modify(setup["stego"], pos, rng.choice([-1, 1], size=k))
    before, after = _extractions(setup, setup["stego"]), _extractions(setup, img)
    assert after[stage] != before[stage]                  # condition of class (a)
    assert _recv(setup, img).status == "bottom"


@pytest.mark.parametrize("stage", [0, 1])
def test_b_lsb_preserving_change_keeps_message(setup, stage):
    rng = np.random.default_rng(7 + stage)
    flat = setup["stego"].reshape(-1)
    pos = _pick(rng, setup["used0"] if stage == 0 else setup["used1"], 16, flat, 2, 253)
    img = _modify(setup["stego"], pos, rng.choice([-2, 2], size=16))
    assert _extractions(setup, img) == _extractions(setup, setup["stego"])
    assert _recv(setup, img).msg == MSG


def test_c_no_unused_positions_after_e19(setup):
    """STC covers all of A0 and A1, so every pixel of a session-start image lies in the used region and class (c) is
    empty. An earlier version left the remainder after n' = m floor(n/m) unused."""
    n = setup["stego"].size
    outside = np.setdiff1d(np.arange(n), np.union1d(setup["used0"], setup["used1"]))
    assert outside.size == 0


def test_d_splicing_two_stego_images_gives_bottom(setup):
    s = setup
    sess2 = s["sysm"].start_session(s["ek"], length=1)
    other = s["sysm"].send(sess2, MSG, s["pool"]["T2"])
    h = s["stego"].shape[0] // 2
    spliced = np.vstack([s["stego"][:h], other[h:]])
    assert _recv(s, spliced).status == "bottom"


def test_property_random_tampering_matches_extraction(setup):
    """Random modification (+-1 or +-2 at k arbitrary positions): bottom if and only if the extracted bits (stage 1 or 2) change."""
    rng = np.random.default_rng(2026)
    flat = setup["stego"].reshape(-1)
    base = _extractions(setup, setup["stego"])
    counts = {"changed": 0, "unchanged": 0}
    for t in range(24):
        k = int(rng.choice([1, 2, 4, 16]))
        delta = int(rng.choice([1, 2]))
        pos = _pick(rng, np.arange(flat.size), k, flat, 2, 253)
        img = _modify(setup["stego"], pos, rng.choice([-delta, delta], size=k))
        changed = _extractions(setup, img) != base
        res = _recv(setup, img)
        if changed:
            counts["changed"] += 1
            assert res.status == "bottom", t
        else:
            counts["unchanged"] += 1
            assert res.msg == MSG, t
    assert counts["changed"] > 0 and counts["unchanged"] > 0


def test_kemeleon_many_to_one_still_rejected_via_sid(setup):
    """c_hat' != c_hat but KDec(c_hat') = KDec(c_hat): the same shared key, but sid' = H(c_hat') != sid, so the Export keys and
    the aad change and the AEAD returns bottom (paper, Theorem 2; layout A8)."""
    s = setup
    kp = s["sess"].kem_part
    x = int.from_bytes(kp[:KEMELEON_VEC_BYTES], "big")
    qn = MLKEM_Q ** MLKEM_N
    x2 = x + qn if x + qn < (1 << (KEMELEON_B + KEMELEON_T)) else x - qn
    assert x2 >= 0
    kp2 = x2.to_bytes(KEMELEON_VEC_BYTES, "big") + kp[KEMELEON_VEC_BYTES:]
    assert kp2 != kp and kemeleon.decode_ct(kp2) == kemeleon.decode_ct(kp)
    crypto = s["sysm"].crypto
    pid = s["sysm"].pid
    ctx2, sid2 = crypto.accept(s["dk"], kp2, pid)
    assert sid2 != s["sess"].sid
    pt = framing.pack(framing.Frame(MSG, FLAG_SESSION_START, None, 0), s["sysm"].pp.buckets)
    body = seal_image(s["keys"], pid, s["sess"].sid, 1, pt)
    assert open_image(image_keys(ctx2, pid, sid2, 1), pid, sid2, 1, body) is None
    # Same shared key: with the original sid the context ctx2 opens, which shows that bottom comes from sid and not from K.
    assert open_image(image_keys(ctx2, pid, s["sess"].sid, 1), pid, s["sess"].sid, 1, body) == pt
