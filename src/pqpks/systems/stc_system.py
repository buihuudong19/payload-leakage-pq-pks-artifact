"""Systems with the two-stage HILL-STC embedder and session mode: PQ-PKS, PQ-PKS-R, BL3, BL4-E, BL5, BL5-L.

PQ-PKS supports long sessions with N >= 1 images. The baselines are compared at N = 1 (one image carries both
kem_part and body); they share this code and therefore also run with N > 1.

Design points:
- The keys of image i (one-time AEAD key, permutation, STC matrix) are obtained with HPKE Export under CE(pid, sid, i);
  the HPKE `info`, sid and every aad bind profile_id.
- Invariant I-1: the sender reserves the index under a lock before encryption and encrypts EXACTLY ONCE for (sid, i).
  On an embedding failure it changes the cover and keeps the body; after R_cover failures the index is spent and
  nothing is re-encrypted. The sender precomputes the 3N image keys when the session opens, then discards the HPKE
  context; each key is deleted after use.
- The receiver follows a `ReceiverPolicy`; state is written only after the AEAD, frame, t0 and Verify checks.
- Indices already seen are not searched again; a replayed image yields bottom.
- t0 is carried in pt_1 and in the transcript; R accepts when t0 in [now - TTL, now + Delta_skew] and t0 > restart_floor.
- The set of seen sids is kept for at least TTL + Delta_skew.
- A session-start image whose sid was already seen returns bottom at once, without Decaps and without trying branch (b)
  (Algorithm 4 of the paper). Trying branch (b) would cost about 11 s per replayed image with the default profile; the
  price of the early exit is the collision term eps_col in Theorem 2. `ReceiveResult.trials` counts AEAD openings;
  `receive_msg` returns only m or None and does not distinguish the reason (A5).

Limitations of the reference implementation: Python cannot guarantee that memory is erased ("delete" means dropping
the reference), and no state is written to disk. The `on_reserve` hook of SenderSession lets an application persist
the index before the image is released (PersistBeforeExternalSend in Algorithm 2).
"""
from __future__ import annotations

import threading
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from ..crypto import framing, sig
from ..params import AEAD_TAG_BYTES, DEFAULT_PP, FLAG_REKEY, FLAG_SESSION_START, PublicParams
from ..stego.two_stage import TwoStageSTC
from .base import (HpkeCrypto, ImageKeys, ReceiveResult, image_keys, open_image, profile_id_for, seal_image,
                   session_id)


class EmbedFailure(RuntimeError):
    """STC failed on every cover tried; the index of the image is spent (invariant I-1)."""


class Inadmissible(ValueError):
    """The public admissibility predicate Adm fails (paper, Definition 1, step 3): no bucket can hold m, or the image
    has too few pixels for l_start(B). Raised BEFORE the index is spent, and depends only on public quantities."""


@dataclass
class SenderSession:
    kem_part: bytes
    sid: bytes
    keys: dict                            # i -> ImageKeys (precomputed; deleted after use)
    t0: int
    signature: bytes | None
    length: int
    attempts: int = 1                     # number of SetupBaseS calls (> 1 possible with rejection-sampled Kemeleon)
    next_index: int = 1
    on_reserve: Callable[[bytes, int], None] | None = None
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @property
    def exhausted(self) -> bool:
        return self.next_index > self.length


@dataclass(frozen=True)
class ReceiverPolicy:
    """Receiver policy (paper, Definition 6). Never read from the payload."""

    require_auth: bool = False
    vk_s: bytes | None = None             # trusted signature verification key of the sender (authenticated out of band)

    def __post_init__(self):
        if self.require_auth and self.vk_s is None:
            raise ValueError("require_auth = True needs a trusted vk_S")


@dataclass
class _RxSession:
    ctx: object
    last: int = 0
    seen: set = field(default_factory=set)
    authenticated: bool | None = None
    max_index: int | None = None          # set on a REKEY flag: no larger index remains
    t0: int | None = None                 # t0 of the session-start image (needed to verify sigma in image 2, A5b)
    held: bytes | None = None             # A5b: m1 held back until a valid sigma arrives in image 2


@dataclass
class ReceiverState:
    sessions: dict = field(default_factory=dict)      # sid -> _RxSession (at most S_max open sessions)
    pending: dict = field(default_factory=dict)       # A5b: sid -> _RxSession waiting for sigma in image 2 (at most P_max, FIFO)
    seen_sids: dict = field(default_factory=dict)     # sid -> session start time; kept >= TTL + Delta_skew
    clock: Callable[[], float] = time.time
    restart_floor: float | None = None                # set on a restart that lost seen_sids

    def prune(self, keep_s: float) -> None:
        now = self.clock()
        for s in [s for s, t in self.seen_sids.items() if now - t > keep_s]:
            del self.seen_sids[s]

    @classmethod
    def after_restart(cls, pp: PublicParams = DEFAULT_PP, clock: Callable[[], float] = time.time):
        """Fresh state after seen_sids was lost: accept only session-start images with t0 > now + Delta_skew
        (the same rule as the Kerberos replay cache, RFC 4120 Section 3.2.3; Algorithm 4)."""
        return cls(clock=clock, restart_floor=clock() + pp.max_skew_s)


class StcSystem:
    def __init__(self, crypto: HpkeCrypto, pp: PublicParams = DEFAULT_PP, cost: str = "hill",
                 stage2_keyed: bool = True, clock: Callable[[], float] = time.time,
                 keyed_perm: bool | None = None, keyed_matrix: bool | None = None):
        """keyed_perm / keyed_matrix: ablation A1 V1-V4 (both keyed by default, i.e. V4 = PQ-PKS)."""
        self.crypto = crypto
        self.pp = pp
        self.embedder = TwoStageSTC(pp, cost, stage2_keyed, keyed_perm, keyed_matrix)
        self.name = crypto.name
        self.pid = profile_id_for(crypto, pp)
        self.clock = clock                # sender clock (t0)

    # ------------------------------------------------------------ keys
    def keygen(self):
        return self.crypto.keygen()

    def sig_keygen(self):
        return sig.keygen(self.pp.sig_alg)

    # ------------------------------------------------------------ sender
    def start_session(self, ek_r: bytes, sk_s=None, length: int | None = None,
                      on_reserve: Callable[[bytes, int], None] | None = None) -> SenderSession:
        length = length or self.pp.session_len
        kem_part, ctx, sid = self.crypto.start(ek_r, self.pid)
        keys = {i: image_keys(ctx, self.pid, sid, i) for i in range(1, length + 1)}
        del ctx                           # drop the HPKE context (exporter secret) once the keys are precomputed
        t0 = int(self.clock())
        signature = None
        if sk_s is not None and self.pp.sig_placement == "image2" and length < 2:
            raise ValueError("A5b (sigma in image 2) needs a session of at least 2 images")
        if sk_s is not None:
            signature = sig.sign(sk_s, sig.transcript(self.pid, sid, ek_r, t0))
            if len(signature) != self.pp.sig_bytes:
                raise ValueError(f"signing key does not match the profile ({self.pp.sig_alg})")
        return SenderSession(kem_part, sid, keys, t0, signature, length, self.crypto.last_attempts,
                             on_reserve=on_reserve)

    def send(self, sess: SenderSession, msg: bytes, cover, cost_map=None) -> np.ndarray:
        """Send one image. `cover` is one image or a sequence of candidate images (at most pp.max_cover_tries are tried).
        `cost_map` (optional, single cover only): precomputed costs, so that an experiment can reuse them across systems.

        Raises EmbedFailure if every cover fails: the index is spent and nothing is re-encrypted under it."""
        with sess.lock:                   # index reservation: two threads never obtain the same i
            if sess.exhausted:
                raise RuntimeError("the session has used all N images: start a new session (rekey)")
            i = sess.next_index
            start = i == 1
            flags = (FLAG_SESSION_START if start else 0) | (FLAG_REKEY if i == sess.length else 0)
            a5b = self.pp.sig_placement == "image2"
            sig_here = sess.signature if (start and not a5b) or (a5b and i == 2) else None
            frame = framing.Frame(msg, flags, sig_here, sess.t0 if start else None)
            try:                                          # message too long: Adm fails, index not yet spent
                pt = framing.pack(frame, self.pp.buckets_for(start, len(sess.kem_part)), allow_sig_next=a5b and i == 2)
            except ValueError as e:
                raise Inadmissible(str(e)) from e
            shape = cover.shape if isinstance(cover, np.ndarray) else None
            if shape is not None:
                need = 8 * (len(pt) + AEAD_TAG_BYTES) + (8 * len(sess.kem_part) if start else 0)
                if need > shape[0] * shape[1]:
                    raise Inadmissible(f"image {shape} cannot hold {need} bits")
            sess.next_index += 1
            keys: ImageKeys = sess.keys.pop(i)
            if sess.on_reserve is not None:
                sess.on_reserve(sess.sid, i)          # persist "i used" BEFORE the image is released
        body = seal_image(keys, self.pid, sess.sid, i, pt)        # exactly once for (sid, i)
        covers = [cover] if isinstance(cover, np.ndarray) else list(_take(cover, self.pp.max_cover_tries))
        last_err = None
        for x in covers[: self.pp.max_cover_tries]:
            try:
                cm = cost_map if (cost_map is not None and len(covers) == 1) else None
                return self.embedder.embed(x, sess.kem_part if start else None, body, keys.perm, keys.stc, cm)
            except ValueError as e:                               # change the cover, KEEP the body
                last_err = e
        raise EmbedFailure(f"embedding failed on {len(covers)} covers; index {i} is spent") from last_err

    # ------------------------------------------------------------ receiver
    def _try_open(self, stego, ctx, sid, j, kem_bytes, tally, buckets=None):
        """Open image j under ctx. For a session-start image with layout A8, `buckets` = [B] (A1 depends on B)."""
        keys = image_keys(ctx, self.pid, sid, j)
        if buckets is None:                   # bucket ladder of the image type
            buckets = self.pp.buckets_for(kem_bytes is not None, self.crypto.kem_part_bytes)
        per_bucket = kem_bytes is not None and self.pp.layout == "A8"
        idx1 = None if per_bucket else self.embedder.stage2_index(stego.shape, kem_bytes, keys.perm)
        for b in buckets:
            if per_bucket:
                try:
                    idx1 = self.embedder.stage2_index(stego.shape, kem_bytes, keys.perm, b + AEAD_TAG_BYTES)
                except ValueError:            # the image cannot hold l_start(B)
                    continue
            body = self.embedder.extract_body(stego, idx1, b + AEAD_TAG_BYTES, keys.stc)
            if body is None:
                continue
            tally[0] += 1
            pt = open_image(keys, self.pid, sid, j, body)
            if pt is not None:
                return pt
        return None

    def _t0_ok(self, t0: int, now: float, state: ReceiverState) -> bool:
        if not (now - self.pp.sid_ttl_s <= t0 <= now + self.pp.max_skew_s):
            return False
        return state.restart_floor is None or t0 > state.restart_floor

    def receive(self, dk_r, stego: np.ndarray, state: ReceiverState, policy: ReceiverPolicy | None = None,
                ek_r: bytes | None = None) -> ReceiveResult:
        """`ek_r` is the receiver's own public key, needed to verify the transcript when a signature is present."""
        policy = policy or ReceiverPolicy()
        state.prune(self.pp.sid_ttl_s + self.pp.max_skew_s)
        now = state.clock()
        kb = self.crypto.kem_part_bytes
        tally = [0]
        # (a) session-start image: public extraction of kem_part from A0. With A8 each bucket B has its own A0(B), so there
        #     are |B| candidates (paper, Algorithm 4); with A8' there is a single A0 and every bucket is tried.
        cands = []                                         # (kem_part, buckets to try)
        if self.pp.layout == "A8":
            for b in self.pp.buckets:
                try:
                    cands.append((self.embedder.extract_kem(stego, kb, b + AEAD_TAG_BYTES), (b,)))
                except ValueError:
                    pass
        else:
            try:
                cands.append((self.embedder.extract_kem(stego, kb), None))
            except ValueError:
                pass
        for kem_part, _ in cands:
            sid0 = session_id(self.pid, kem_part)
            if sid0 in state.sessions or sid0 in state.pending or sid0 in state.seen_sids:
                # replayed session-start image: bottom at once, no Decaps, branch (b) not tried (Algorithm 4).
                # Internal label only; externally the result is still bottom (A5).
                return ReceiveResult(None, "replay", sid0, 1, trials=0)
        for kem_part, bset in cands:
            res = self._receive_start(dk_r, stego, state, policy, ek_r, kem_part, kb, now, tally, bset)
            if res is not None:
                res.trials = tally[0]
                return res
        # (b) later image: try only the UNSEEN indices in [max(2, last - W + 1), min(last + W, max_index)].
        #     A5b: a session waiting for sigma tries index 2 only (the image carrying sigma); nothing is returned before sigma is valid.
        a5b = self.pp.sig_placement == "image2"
        cand_sessions = [(sid, rx, False) for sid, rx in state.sessions.items()] + \
                        [(sid, rx, True) for sid, rx in state.pending.items()]
        for sid, rx, is_pending in cand_sessions:
            if is_pending:
                js = [2] if 2 not in rx.seen else []
            else:
                lo = max(2, rx.last - self.pp.window + 1)
                hi = rx.last + self.pp.window if rx.max_index is None else min(rx.last + self.pp.window, rx.max_index)
                js = range(lo, hi + 1)
            for j in js:
                if j in rx.seen:
                    continue              # replays are not searched
                pt = self._try_open(stego, rx.ctx, sid, j, None, tally)
                if pt is None:
                    continue
                fr = framing.unpack(pt, self.pp.sig_bytes, allow_sig_next=a5b and j == 2)
                if fr is None or fr.flags & FLAG_SESSION_START:
                    continue
                if a5b and j == 2 and fr.sig is not None and policy.vk_s is not None:
                    if ek_r is None:
                        raise ValueError("the receiver's ek_R is needed to verify the transcript")
                    if not sig.verify(policy.vk_s, sig.transcript(self.pid, sid, ek_r, rx.t0), fr.sig, self.pp.sig_alg):
                        return ReceiveResult(None, "auth_fail", sid, j, False, fr.flags, tally[0])   # state unchanged
                    authenticated = True
                else:
                    authenticated = rx.authenticated
                if is_pending and authenticated is not True:
                    return ReceiveResult(None, "auth_fail", sid, j, False, fr.flags, tally[0])
                released = ()
                if is_pending:                                    # COMMIT the pending session: it becomes an open session
                    del state.pending[sid]
                    if len(state.sessions) >= self.pp.max_open_sessions:
                        state.sessions.pop(next(iter(state.sessions)))
                    state.sessions[sid] = rx
                    released = ((1, rx.held),)
                    rx.held = None
                rx.authenticated = authenticated
                rx.seen.add(j)                                    # COMMIT
                rx.last = max(rx.last, j)
                if fr.flags & FLAG_REKEY:
                    rx.max_index = j      # unseen images with index < j (late arrivals) are still accepted
                return ReceiveResult(fr.msg, "ok", sid, j, rx.authenticated, fr.flags, tally[0], released)
        return ReceiveResult(None, "bottom", trials=tally[0])

    def receive_msg(self, dk_r, stego: np.ndarray, state: ReceiverState, policy: ReceiverPolicy | None = None,
                    ek_r: bytes | None = None) -> bytes | None:
        """Application interface: returns m or None. Every reason for rejection (bottom, replay, expiry, bad signature)
        maps to None, so that no distinguishable error code leaves the receiver (assumption A5).
        `receive` keeps the detailed labels for diagnostics and tests only."""
        res = self.receive(dk_r, stego, state, policy, ek_r)
        return res.msg if res.status == "ok" else None

    def _receive_start(self, dk_r, stego, state, policy, ek_r, kem_part, kb, now, tally, buckets=None):
        acc = self.crypto.accept(dk_r, kem_part, self.pid)
        if acc is None:
            return None
        ctx, sid = acc
        pt = self._try_open(stego, ctx, sid, 1, kb, tally, buckets)
        if pt is None:
            return None
        fr = framing.unpack(pt, self.pp.sig_bytes)
        a5b = self.pp.sig_placement == "image2"
        if fr is None or not fr.flags & FLAG_SESSION_START or (a5b and fr.sig is not None):
            return ReceiveResult(None, "bottom")                 # A5b: sigma in image 1 is not canonical
        if not self._t0_ok(fr.t0, now, state):
            return ReceiveResult(None, "stale", sid, 1)
        if a5b and policy.require_auth:
            # A5b with mandatory authentication: image 1 only creates a PENDING session (m1 is not returned) until image 2
            # carries a valid sigma. Pending sessions use at most P_max slots (FIFO), separate from the S_max authenticated sessions.
            if len(state.pending) >= self.pp.max_pending_sessions:
                state.pending.pop(next(iter(state.pending)))
            state.pending[sid] = _RxSession(ctx, last=1, seen={1}, authenticated=False,
                                            max_index=1 if fr.flags & FLAG_REKEY else None, t0=fr.t0, held=fr.msg)
            state.seen_sids[sid] = now
            return ReceiveResult(None, "pending", sid, 1, False, fr.flags)
        authenticated = None
        if fr.sig is not None and policy.vk_s is not None:
            if ek_r is None:
                raise ValueError("the receiver's ek_R is needed to verify the transcript")
            authenticated = sig.verify(policy.vk_s, sig.transcript(self.pid, sid, ek_r, fr.t0), fr.sig,
                                       self.pp.sig_alg)
            if not authenticated:
                return ReceiveResult(None, "auth_fail", sid, 1, False, fr.flags)
        if policy.require_auth and authenticated is not True:
            return ReceiveResult(None, "auth_fail", sid, 1, False, fr.flags)
        # COMMIT: only after every check
        if len(state.sessions) >= self.pp.max_open_sessions:
            state.sessions.pop(next(iter(state.sessions)))
        rx = _RxSession(ctx, last=1, seen={1}, authenticated=authenticated,
                        max_index=1 if fr.flags & FLAG_REKEY else None, t0=fr.t0)
        state.sessions[sid] = rx
        state.seen_sids[sid] = now
        return ReceiveResult(fr.msg, "ok", sid, 1, authenticated, fr.flags)


def _take(it: Iterable, k: int):
    for n, x in enumerate(it):
        if n >= k:
            break
        yield x
