"""Shifted bucket ladder and A5b (sigma in image 2); paper, Sections 5.4 and 7.8.

Ladder of later images: "same" (B, earlier profile), "shifted" (B + |c_hat|), "merged" (B U (B + |c_hat|), default). With
"shifted" and "merged" the room of the c_hat that is not repeated is used for the message; "merged" also keeps the small
buckets for short messages, at the cost of more bucket trials at the receiver for later images.
A5b: sigma is placed in image 2. When the receiver requires authentication, image 1 only creates a pending session; m1 is
released only when image 2 carries a valid sigma. Pending sessions use at most P_max slots, separate from S_max.
"""
from dataclasses import replace

import numpy as np
import pytest

from pqpks import data, systems
from pqpks.crypto import framing
from pqpks.params import AEAD_TAG_BYTES, DEFAULT_PP, FLAG_SESSION_START, KEMELEON_CT_BYTES
from pqpks.systems import ReceiverPolicy, ReceiverState
from pqpks.systems.base import profile_id_for


@pytest.fixture(scope="module")
def covers():
    ids = data.pilot_ids()[:12]
    return [data.load_bossbase(i) for i in ids]


def test_shifted_ladder_definition_and_profile_binding():
    assert DEFAULT_PP.ladder == "merged"
    kb = KEMELEON_CT_BYTES
    assert DEFAULT_PP.buckets_for(True, kb) == DEFAULT_PP.buckets
    shifted = tuple(b + kb for b in DEFAULT_PP.buckets)
    assert replace(DEFAULT_PP, ladder="shifted").buckets_for(False, kb) == shifted
    assert DEFAULT_PP.buckets_for(False, kb) == tuple(sorted(set(DEFAULT_PP.buckets) | set(shifted)))
    assert len(DEFAULT_PP.buckets_for(False, kb)) == 11                  # 2048 = 512 + 1536 coincides
    same = replace(DEFAULT_PP, ladder="same")
    assert same.buckets_for(False, KEMELEON_CT_BYTES) == DEFAULT_PP.buckets
    s1, s2 = systems.make("PQ-PKS"), systems.make("PQ-PKS", same)
    assert profile_id_for(s1.crypto, s1.pp) != profile_id_for(s2.crypto, s2.pp)
    with pytest.raises(ValueError):
        replace(DEFAULT_PP, ladder="pairs")
    with pytest.raises(ValueError):
        replace(DEFAULT_PP, sig_placement="image3")


def test_later_image_uses_space_of_c_hat(covers):
    """With the shifted ladder a later image can carry a message longer than the largest bucket of the common ladder (4096 - 6 B)."""
    m = bytes(range(256)) * 20                     # 5120 B > 4090 B
    s = systems.make("PQ-PKS")
    dk, ek = s.keygen()
    sess = s.start_session(ek, length=2)
    y1 = s.send(sess, b"first", covers[0])
    y2 = s.send(sess, m, covers[1])
    st = ReceiverState()
    assert s.receive(dk, y1, st).msg == b"first"
    r = s.receive(dk, y2, st)
    assert r.status == "ok" and r.msg == m and r.index == 2
    pp = s.pp
    n1, n2 = len(pp.buckets_for(True, KEMELEON_CT_BYTES)), len(pp.buckets_for(False, KEMELEON_CT_BYTES))
    bound = n1 + n2 * ((2 * pp.window - 1) * pp.max_open_sessions + pp.max_pending_sessions)
    assert r.trials <= bound
    # same message: the shifted ladder pads to 4096 + 1536 = 5632 B, the common ladder must go up to 8192 B (45 % more embedding)
    kb = KEMELEON_CT_BYTES
    pt_merged = framing.pack(framing.Frame(m), DEFAULT_PP.buckets_for(False, kb))
    pt_same = framing.pack(framing.Frame(m), replace(DEFAULT_PP, ladder="same").buckets_for(False, kb))
    assert len(pt_merged) == 4096 + kb and len(pt_same) == 8192
    # short message in a later image: "merged" keeps the 256 B bucket, "shifted" must pad to 256 + 1536 B
    short = framing.Frame(b"hi")
    assert len(framing.pack(short, DEFAULT_PP.buckets_for(False, kb))) == 256
    assert len(framing.pack(short, replace(DEFAULT_PP, ladder="shifted").buckets_for(False, kb))) == 256 + kb


def test_shifted_later_lengths_equal_first_image_lengths():
    """With "shifted", the total embedded length of a later image is in the same set {|c_hat| + B + 16} as for a session-start image."""
    kb = KEMELEON_CT_BYTES
    pp = replace(DEFAULT_PP, ladder="shifted")
    first = {kb + b + AEAD_TAG_BYTES for b in pp.buckets_for(True, kb)}
    later = {b + AEAD_TAG_BYTES for b in pp.buckets_for(False, kb)}
    assert first == later


@pytest.fixture(scope="module")
def a5b():
    pp = replace(DEFAULT_PP, sig_placement="image2")
    s = systems.make("PQ-PKS", pp)
    dk, ek = s.keygen()
    sk, vk = s.sig_keygen()
    return {"s": s, "dk": dk, "ek": ek, "sk": sk, "vk": vk}


def _session(a, covers, k, n=3):
    s = a["s"]
    sess = s.start_session(a["ek"], a["sk"], length=n)
    ys = [s.send(sess, f"m{i}".encode(), covers[k + i]) for i in range(n)]
    return ys


def test_a5b_signature_in_image2_releases_held_message(a5b, covers):
    s = a5b["s"]
    y1, y2, y3 = _session(a5b, covers, 0)
    pol, st = ReceiverPolicy(require_auth=True, vk_s=a5b["vk"]), ReceiverState()
    r1 = s.receive(a5b["dk"], y1, st, pol, a5b["ek"])
    assert r1.status == "pending" and r1.msg is None and s.receive_msg(a5b["dk"], y1, ReceiverState(), pol, a5b["ek"]) is None
    assert len(st.pending) == 1 and len(st.sessions) == 0
    r2 = s.receive(a5b["dk"], y2, st, pol, a5b["ek"])
    assert r2.status == "ok" and r2.msg == b"m1" and r2.released == ((1, b"m0"),) and r2.authenticated is True
    assert len(st.pending) == 0 and len(st.sessions) == 1
    assert s.receive(a5b["dk"], y3, st, pol, a5b["ek"]).msg == b"m2"
    assert s.receive(a5b["dk"], y1, st, pol, a5b["ek"]).status == "replay"


def test_a5b_wrong_signer_keeps_session_pending_and_releases_nothing(a5b, covers):
    s = a5b["s"]
    y1, y2, _ = _session(a5b, covers, 3)
    _, vk_other = s.sig_keygen()
    pol, st = ReceiverPolicy(require_auth=True, vk_s=vk_other), ReceiverState()
    assert s.receive(a5b["dk"], y1, st, pol, a5b["ek"]).status == "pending"
    r2 = s.receive(a5b["dk"], y2, st, pol, a5b["ek"])
    assert r2.status == "auth_fail" and r2.msg is None and r2.released == ()
    assert len(st.pending) == 1 and len(st.sessions) == 0


def test_a5b_later_image_before_image2_is_not_accepted(a5b, covers):
    s = a5b["s"]
    y1, y2, y3 = _session(a5b, covers, 6)
    pol, st = ReceiverPolicy(require_auth=True, vk_s=a5b["vk"]), ReceiverState()
    s.receive(a5b["dk"], y1, st, pol, a5b["ek"])
    assert s.receive(a5b["dk"], y3, st, pol, a5b["ek"]).status == "bottom"   # a pending session tries index 2 only
    assert s.receive(a5b["dk"], y2, st, pol, a5b["ek"]).released == ((1, b"m0"),)
    assert s.receive(a5b["dk"], y3, st, pol, a5b["ek"]).msg == b"m2"         # after authentication: normal reception


def test_a5b_pending_table_is_bounded(a5b, covers):
    s = a5b["s"]
    pp = replace(s.pp, max_pending_sessions=2)
    s2 = systems.StcSystem(s.crypto, pp)
    pol, st = ReceiverPolicy(require_auth=True, vk_s=a5b["vk"]), ReceiverState()
    firsts = []
    for k in range(3):
        sess = s2.start_session(a5b["ek"], a5b["sk"], length=2)
        firsts.append(s2.send(sess, b"x", covers[9 + k]))
    for y in firsts:
        assert s2.receive(a5b["dk"], y, st, pol, a5b["ek"]).status == "pending"
    assert len(st.pending) == 2                   # FIFO: the oldest pending session is evicted


def test_a5b_requires_two_images_and_canonical_frames(a5b, covers):
    s = a5b["s"]
    with pytest.raises(ValueError):
        s.start_session(a5b["ek"], a5b["sk"], length=1)
    pt = framing.pack(framing.Frame(b"x", 0, b"s" * s.pp.sig_bytes), s.pp.buckets, allow_sig_next=True)
    assert framing.unpack(pt, s.pp.sig_bytes) is None                     # sigma outside image 1 without A5b
    assert framing.unpack(pt, s.pp.sig_bytes, allow_sig_next=True).sig == b"s" * s.pp.sig_bytes


def test_a5b_anonymous_receiver_accepts_image1_and_marks_authentication_later(a5b, covers):
    s = a5b["s"]
    y1, y2, _ = _session(a5b, covers, 0)
    st = ReceiverState()
    r1 = s.receive(a5b["dk"], y1, st, ReceiverPolicy(require_auth=False, vk_s=a5b["vk"]), a5b["ek"])
    assert r1.status == "ok" and r1.msg == b"m0" and r1.authenticated is None
    r2 = s.receive(a5b["dk"], y2, st, ReceiverPolicy(require_auth=False, vk_s=a5b["vk"]), a5b["ek"])
    assert r2.msg == b"m1" and r2.authenticated is True
