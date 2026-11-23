"""End-to-end functional tests of every system on synthetic test covers (a reduced E4), and the E4-P protocol
regression block (receiver policy, replay, ordering, freshness and one-time keys).

Covers: the eight synthetic test covers of pqpks.data.cover_pool.
Messages: text and a binary test message (pqpks.data.test_messages)."""
import threading
from dataclasses import replace

import numpy as np
import pytest

from pqpks import data, systems
from pqpks.crypto import framing, sig
from pqpks.params import (DEFAULT_PP, FLAG_SESSION_START, KEMELEON_CT_BYTES, KEMELEON_R_CT_BYTES,
                          MLDSA44_SIG_BYTES, MLDSA65_SIG_BYTES)
from pqpks.systems import ReceiverPolicy, ReceiverState
from pqpks.systems.base import open_image, seal_image, session_id
from pqpks.systems.stc_system import EmbedFailure

TEXT = "Test message for the PQ-PKS reference implementation (rev. 2026)".encode()


@pytest.fixture(scope="module")
def pool():
    return data.cover_pool()


class Clock:
    def __init__(self, t=1_800_000_000.0):
        self.t = t

    def __call__(self):
        return self.t


@pytest.mark.parametrize("name", ["PQ-PKS", "BL3", "BL4-E", "BL5"])
def test_stc_systems_single_image(name, pool):
    sysm = systems.make(name)
    dk, ek = sysm.keygen()
    sess = sysm.start_session(ek, length=1)
    cover = pool["T1"]
    stego = sysm.send(sess, TEXT, cover)
    assert stego.shape == cover.shape and np.abs(stego.astype(int) - cover).max() <= 1
    res = sysm.receive(dk, stego, ReceiverState())
    assert res.status == "ok" and res.msg == TEXT and res.flags & FLAG_SESSION_START
    assert res.authenticated is None                          # anonymous mode: no sender attributed
    assert sysm.receive(dk, cover, ReceiverState()).status == "bottom"
    dk2, _ = sysm.keygen()
    assert sysm.receive(dk2, stego, ReceiverState()).status == "bottom"


def test_pqpks_session_with_signature_and_image_message(pool):
    """Session of 4 images; sigma (ML-DSA-65) in image 1; image 3 carries a binary message; out-of-order reception; replays rejected."""
    sysm = systems.make("PQ-PKS")
    dk, ek = sysm.keygen()
    sk_s, vk_s = sysm.sig_keygen()
    sess = sysm.start_session(ek, sk_s=sk_s, length=4)
    assert len(sess.signature) == MLDSA65_SIG_BYTES
    blob = data.test_messages()[0]
    covers = [pool["T1"], pool["T2"], pool["T3"], pool["T4"]]
    msgs = [b"image 1: session start, signed", b"image 2", blob, b"image 4: end of session"]
    stegos = [sysm.send(sess, m, c) for m, c in zip(msgs, covers)]
    assert sess.exhausted and not sess.keys                   # every image key was deleted after use
    st, pol = ReceiverState(), ReceiverPolicy(require_auth=True, vk_s=vk_s)
    r1 = sysm.receive(dk, stegos[0], st, pol, ek)
    assert (r1.status, r1.msg, r1.index, r1.authenticated) == ("ok", msgs[0], 1, True)
    r3 = sysm.receive(dk, stegos[2], st, pol, ek)             # arrives before image 2
    assert (r3.status, r3.msg, r3.index, r3.authenticated) == ("ok", blob, 3, True)
    assert sysm.receive(dk, stegos[1], st, pol, ek).msg == msgs[1]
    assert sysm.receive(dk, stegos[2], st, pol, ek).status == "bottom"     # replay -> bottom
    assert sysm.receive(dk, stegos[0], st, pol, ek).status == "replay"     # sid already seen (internal label)
    assert sysm.receive(dk, stegos[3], st, pol, ek).msg == msgs[3]


def test_bl2_mpvd(pool):
    s = systems.make("BL2")
    dk, ek = s.keygen()
    stego = s.send_once(ek, TEXT, pool["T1"])
    assert s.receive_once(dk, stego).msg == TEXT


def test_bl1_reference(pool):
    s = systems.make("BL1")
    sk, pk = s.keygen()
    text = "Test message for the PQ-PKS reference implementation (rev. 2026)"
    stego = s.send_once(pk, text, pool["T2"])
    assert s.receive_once(sk, stego).msg.decode("latin-1") == text


def test_pqpks_last_image_first_does_not_drop_late_images(pool):
    """The last image of the session (REKEY flag) arrives first: late images are still accepted; replay -> bottom."""
    sysm = systems.make("PQ-PKS")
    dk, ek = sysm.keygen()
    sess = sysm.start_session(ek, length=3)
    covers = [pool["T1"], pool["T2"], pool["T6"]]
    stegos = [sysm.send(sess, b"m%d" % i, c) for i, c in enumerate(covers, start=1)]
    st = ReceiverState()
    assert sysm.receive(dk, stegos[0], st).msg == b"m1"
    assert sysm.receive(dk, stegos[2], st).msg == b"m3"
    assert sysm.receive(dk, stegos[1], st).msg == b"m2"
    assert sysm.receive(dk, stegos[2], st).status == "bottom"


@pytest.mark.parametrize("name,size", [("BL5-L", KEMELEON_CT_BYTES), ("PQ-PKS-R", KEMELEON_R_CT_BYTES)])
def test_new_systems_roundtrip(name, size, pool):
    sysm = systems.make(name)
    dk, ek = sysm.keygen()
    sess = sysm.start_session(ek, length=2)
    assert len(sess.kem_part) == size and sess.attempts >= 1
    s1 = sysm.send(sess, b"one", pool["T1"])
    s2 = sysm.send(sess, b"two", pool["T2"])
    st = ReceiverState()
    assert sysm.receive(dk, s2, st).status == "bottom"      # image 2 before the session exists: not accepted yet
    assert sysm.receive(dk, s1, st).msg == b"one"
    assert sysm.receive(dk, s2, st).msg == b"two"


def test_per_image_aead_keys_are_distinct_and_bound_to_index_and_profile(pool):
    """One-time AEAD key per image; aad binds (pid, sid, i)."""
    sysm = systems.make("PQ-PKS")
    _, ek = sysm.keygen()
    sess = sysm.start_session(ek, length=3)
    keys = [sess.keys[i] for i in (1, 2, 3)]
    assert len({k.aead for k in keys}) == 3 and len({k.perm for k in keys}) == 3 and len({k.stc for k in keys}) == 3
    pid, sid = sysm.pid, sess.sid
    body = seal_image(keys[1], pid, sid, 2, b"pt")
    assert open_image(keys[1], pid, sid, 2, body) == b"pt"
    assert open_image(keys[2], pid, sid, 3, body) is None
    assert open_image(keys[1], pid, sid, 3, body) is None              # same key, wrong aad
    assert open_image(keys[1], b"\x00" * 32, sid, 2, body) is None     # sai profile


def test_opening_image_replay_after_eviction_is_detected(pool):
    """S_max = 1; session A is evicted when session B opens; a replay of the start image of A is still blocked."""
    pp = replace(DEFAULT_PP, max_open_sessions=1)
    sysm = systems.make("PQ-PKS", pp)
    dk, ek = sysm.keygen()
    a1 = sysm.send(sysm.start_session(ek, length=2), b"A1", pool["T1"])
    b1 = sysm.send(sysm.start_session(ek, length=2), b"B1", pool["T2"])
    st = ReceiverState()
    assert sysm.receive(dk, a1, st).msg == b"A1"
    assert sysm.receive(dk, b1, st).msg == b"B1"
    assert len(st.sessions) == 1
    assert sysm.receive(dk, a1, st).status == "replay"


# ===========================================================================
# E4-P protocol regression block
# ===========================================================================
@pytest.fixture(scope="module")
def signed(pool):
    sysm = systems.make("PQ-PKS")
    dk, ek = sysm.keygen()
    sk_s, vk_s = sysm.sig_keygen()
    unsigned = sysm.send(sysm.start_session(ek, length=1), b"anonymous", pool["T1"])
    good = sysm.send(sysm.start_session(ek, sk_s=sk_s, length=1), b"signed", pool["T2"])
    return dict(sysm=sysm, dk=dk, ek=ek, sk_s=sk_s, vk_s=vk_s, unsigned=unsigned, good=good)


def test_e4p_1_require_auth_rejects_unsigned_and_keeps_no_state(signed):
    s = signed
    st = ReceiverState()
    pol = ReceiverPolicy(require_auth=True, vk_s=s["vk_s"])
    assert s["sysm"].receive(s["dk"], s["unsigned"], st, pol, s["ek"]).status == "auth_fail"
    assert not st.sessions and not st.seen_sids              # no commit on bottom
    assert s["sysm"].receive(s["dk"], s["good"], st, pol, s["ek"]).status == "ok"


def test_e4p_2_3_wrong_signer_or_wrong_recipient_rejected_without_state(signed):
    s = signed
    _, vk_other = s["sysm"].sig_keygen()
    st = ReceiverState()
    r = s["sysm"].receive(s["dk"], s["good"], st, ReceiverPolicy(require_auth=True, vk_s=vk_other), s["ek"])
    assert r.status == "auth_fail" and not st.sessions and not st.seen_sids
    _, ek_other = s["sysm"].keygen()                         # transcript verified with another ek_R
    r = s["sysm"].receive(s["dk"], s["good"], ReceiverState(), ReceiverPolicy(True, s["vk_s"]), ek_other)
    assert r.status == "auth_fail"
    # anonymous mode with vk_S: valid sigma -> authenticated label; no sigma -> no sender attributed
    r = s["sysm"].receive(s["dk"], s["good"], ReceiverState(), ReceiverPolicy(vk_s=s["vk_s"]), s["ek"])
    assert (r.status, r.authenticated) == ("ok", True)
    r = s["sysm"].receive(s["dk"], s["unsigned"], ReceiverState(), ReceiverPolicy(vk_s=s["vk_s"]), s["ek"])
    assert (r.status, r.authenticated) == ("ok", None)
    with pytest.raises(ValueError):
        ReceiverPolicy(require_auth=True)                    # mandatory authentication requires vk_S


def test_e4p_2_signature_bound_to_profile_sid_and_t0(signed):
    s = signed
    t = sig.transcript(s["sysm"].pid, b"\x01" * 32, s["ek"], 5)
    sg = sig.sign(s["sk_s"], t)
    assert sig.verify(s["vk_s"], t, sg)
    assert not sig.verify(s["vk_s"], sig.transcript(b"\x02" * 32, b"\x01" * 32, s["ek"], 5), sg)
    assert not sig.verify(s["vk_s"], sig.transcript(s["sysm"].pid, b"\x03" * 32, s["ek"], 5), sg)
    assert not sig.verify(s["vk_s"], sig.transcript(s["sysm"].pid, b"\x01" * 32, s["ek"], 6), sg)


def test_e4p_4_canonical_frame_parsing():
    pp = DEFAULT_PP
    pt = framing.pack(framing.Frame(b"abc", FLAG_SESSION_START, None, 7), pp.buckets)
    fr = framing.unpack(pt, pp.sig_bytes)
    assert (fr.msg, fr.t0, fr.sig) == (b"abc", 7, None)
    assert framing.unpack(pt[:-1] + b"\x01", pp.sig_bytes) is None                       # pad ≠ 0
    assert framing.unpack(pt[:1] + bytes([pt[1] | 0x80]) + pt[2:], pp.sig_bytes) is None  # unknown flag
    nxt = framing.pack(framing.Frame(b"abc"), pp.buckets)
    assert framing.unpack(nxt[:1] + bytes([nxt[1] | 0x01]) + nxt[2:], pp.sig_bytes) is None  # sigma outside image 1
    assert framing.unpack(nxt[:2] + (10 ** 6).to_bytes(4, "big") + nxt[6:], pp.sig_bytes) is None
    with pytest.raises(ValueError):
        framing.pack(framing.Frame(b"x", 0, b"s" * 10), pp.buckets)                       # sigma only in image 1
    with pytest.raises(ValueError):
        framing.pack(framing.Frame(b"x", FLAG_SESSION_START), pp.buckets)                 # t0 missing


def test_e4p_5_concurrent_senders_never_share_an_index(pool):
    sysm = systems.make("PQ-PKS")
    dk, ek = sysm.keygen()
    reserved = []
    sess = sysm.start_session(ek, length=4, on_reserve=lambda sid, i: reserved.append(i))
    covers = [pool["T1"], pool["T2"], pool["T3"], pool["T4"]]
    out = [None] * 4

    def worker(k):
        out[k] = sysm.send(sess, b"luong %d" % k, covers[k])

    th = [threading.Thread(target=worker, args=(k,)) for k in range(4)]
    for t in th:
        t.start()
    for t in th:
        t.join()
    assert sorted(reserved) == [1, 2, 3, 4]
    st = ReceiverState()
    first = [k for k in range(4) if sysm.receive(dk, out[k], ReceiverState()).status == "ok"]
    assert len(first) == 1                                   # exactly one image carries index 1
    got = {sysm.receive(dk, out[k], st).msg for k in first + [k for k in range(4) if k not in first]}
    assert got == {b"luong %d" % k for k in range(4)}


def test_e4p_6_cover_retry_encrypts_exactly_once(pool, monkeypatch):
    """STC fails on the first cover and succeeds on the second: AEAD.Enc runs exactly once and the body is unchanged."""
    from pqpks.systems import stc_system
    sysm = systems.make("PQ-PKS")
    dk, ek = sysm.keygen()
    sess = sysm.start_session(ek, length=2)
    calls, bodies = [], []
    real_seal, real_embed = stc_system.seal_image, sysm.embedder.embed

    def counting_seal(*a):
        calls.append(a[3])
        return real_seal(*a)

    def flaky_embed(cover, kem, body, kp, kh, cost_map=None):
        bodies.append(body)
        if len(bodies) == 1:
            raise ValueError("STC capacity exceeded (simulated)")
        return real_embed(cover, kem, body, kp, kh, cost_map)

    monkeypatch.setattr(stc_system, "seal_image", counting_seal)
    monkeypatch.setattr(sysm.embedder, "embed", flaky_embed)
    stego = sysm.send(sess, b"retry", [pool["T1"], pool["T2"]])
    assert calls == [1] and len(bodies) == 2 and bodies[0] == bodies[1]
    assert sysm.receive(dk, stego, ReceiverState()).msg == b"retry"

    def always_fail(*a):
        raise ValueError("failure")

    monkeypatch.setattr(sysm.embedder, "embed", always_fail)
    with pytest.raises(EmbedFailure):
        sysm.send(sess, b"lost", [pool["T3"]])
    assert sess.next_index == 3 and calls == [1, 2] and sess.exhausted      # index 2 is spent


def test_e4p_7_8_t0_window_ttl_and_restart(pool):
    """A session-start image is accepted only when t0 in [now - TTL, now + Delta_skew]; a replay after TTL is still
    blocked; after a restart that lost seen_sids, images with t0 <= restart time + Delta_skew are rejected."""
    pp = replace(DEFAULT_PP, sid_ttl_s=600.0, max_skew_s=60.0)
    s_clock, r_clock = Clock(), Clock()
    sysm = systems.make("PQ-PKS", pp, clock=s_clock)
    dk, ek = sysm.keygen()
    img = sysm.send(sysm.start_session(ek, length=1), b"t0", pool["T1"])
    st = ReceiverState(clock=r_clock)
    assert sysm.receive(dk, img, st).msg == b"t0"
    r_clock.t += 300
    assert sysm.receive(dk, img, st).status == "replay"       # within TTL: blocked by seen_sids
    st.sessions.clear()
    r_clock.t += 2000                                          # beyond TTL + Delta_skew: seen_sids is purged
    assert sysm.receive(dk, img, st).status == "stale"        # still blocked by t0
    assert not st.seen_sids
    s_clock.t = r_clock.t + 120                                # t0 further in the future than Delta_skew
    fut = sysm.send(sysm.start_session(ek, length=1), b"future", pool["T2"])
    assert sysm.receive(dk, fut, ReceiverState(clock=r_clock)).status == "stale"
    s_clock.t = r_clock.t                                      # restart
    before = sysm.send(sysm.start_session(ek, length=1), b"before", pool["T3"])
    st2 = ReceiverState.after_restart(pp, clock=r_clock)
    assert sysm.receive(dk, before, st2).status == "stale"
    r_clock.t += 61
    s_clock.t = r_clock.t
    after = sysm.send(sysm.start_session(ek, length=1), b"late", pool["T4"])
    assert sysm.receive(dk, after, st2).msg == b"late"


def test_e4p_10_profile_mismatch_gives_bottom(pool):
    """An image of the PQ-PKS profile given to a receiver configured for PQ-PKS-R (other kem_part, other pid) -> bottom."""
    a, b = systems.make("PQ-PKS"), systems.make("PQ-PKS-R")
    assert a.pid != b.pid
    dk, ek = a.keygen()
    img = a.send(a.start_session(ek, length=1), b"x", pool["T1"])
    assert b.receive(dk, img, ReceiverState()).status == "bottom"
    assert session_id(a.pid, b"\x00" * 10) != session_id(b.pid, b"\x00" * 10)


def test_e17_default_signature_is_mldsa65_and_44_is_ablation(pool):
    assert DEFAULT_PP.sig_alg == "ML-DSA-65" and DEFAULT_PP.sig_bytes == MLDSA65_SIG_BYTES
    pp44 = replace(DEFAULT_PP, sig_alg="ML-DSA-44")
    sysm = systems.make("PQ-PKS", pp44)
    dk, ek = sysm.keygen()
    sk_s, vk_s = sysm.sig_keygen()
    sess = sysm.start_session(ek, sk_s=sk_s, length=1)
    assert len(sess.signature) == MLDSA44_SIG_BYTES
    img = sysm.send(sess, b"44", pool["T1"])
    r = sysm.receive(dk, img, ReceiverState(), ReceiverPolicy(True, vk_s), ek)
    assert (r.status, r.authenticated) == ("ok", True)
    with pytest.raises(ValueError):                           # ML-DSA-65 key in an ML-DSA-44 profile
        sysm.start_session(ek, sk_s=sig.keygen("ML-DSA-65")[0], length=1)
