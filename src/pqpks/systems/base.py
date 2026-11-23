"""Common interface of all systems.

All systems share HPKE (SetupBaseS/R and Export only), a one-time AEAD key per image and the same payload framing;
every label and every aad binds profile_id. The systems differ only in
  (i)  the KEM and the encoding of enc into kem_part (raw / Kemeleon / Elligator 2);
  (ii) the embedder (two-stage HILL-STC or sequential MPVD).
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Callable

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305

from ..crypto import hpke
from ..crypto.profile import ce, profile_id
from ..params import HPKE_AEAD_CHACHA20POLY1305, HPKE_KDF_HKDF_SHA256


def profile_id_for(crypto: "HpkeCrypto", pp) -> bytes:
    """profile_id of a system (canonical encoding CE): version, system name, HPKE suite, kem_part encoding (through the
    system name and the length), KDF/AEAD, then the public fields of pp."""
    return profile_id(crypto.name, crypto.kem.kem_id, crypto.kem_part_bytes, HPKE_KDF_HKDF_SHA256,
                      HPKE_AEAD_CHACHA20POLY1305, *pp.profile_fields())


def session_id(pid: bytes, kem_part: bytes) -> bytes:
    """sid = SHA3-256("PQPKS/sid" ‖ pid ‖ ĉ)."""
    return hashlib.sha3_256(b"PQPKS/sid" + pid + kem_part).digest()


def aad(pid: bytes, sid: bytes, i: int) -> bytes:
    return ce("PQPKS/image", pid, sid, i)


@dataclass(frozen=True)
class ImageKeys:
    """Keys of image i, all obtained with HPKE Export under CE(pid, sid, i)."""

    aead: bytes      # k_ae,i: ONE-TIME AEAD key
    perm: bytes      # k_π,i
    stc: bytes       # k_H,i


def image_keys(ctx: hpke.Context, pid: bytes, sid: bytes, i: int) -> ImageKeys:
    info_i = ce(pid, sid, i)
    return ImageKeys(aead=ctx.export(b"PQPKS/aead" + info_i, 32),
                     perm=ctx.export(b"PQPKS/perm" + info_i, 32),
                     stc=ctx.export(b"PQPKS/stc" + info_i, 32))


_ZERO_NONCE = bytes(12)


def seal_image(keys: ImageKeys, pid: bytes, sid: bytes, i: int, pt: bytes) -> bytes:
    """body_i = ChaCha20-Poly1305.Enc(k_ae,i, nonce = 0^96, ad_i, pt_i), ad_i = CE("PQPKS/image", pid, sid, i).

    A fixed all-zero nonce is safe because each k_ae,i encrypts exactly one pt (invariant I-1).
    HPKE Seal with a sequence number is not used, because RFC 9180 (Sections 5.2 and 9.1) forbids out-of-order opening.
    """
    return ChaCha20Poly1305(keys.aead).encrypt(_ZERO_NONCE, pt, aad(pid, sid, i))


def open_image(keys: ImageKeys, pid: bytes, sid: bytes, i: int, body: bytes) -> bytes | None:
    try:
        return ChaCha20Poly1305(keys.aead).decrypt(_ZERO_NONCE, body, aad(pid, sid, i))
    except InvalidTag:
        return None


class KemEncodingExhausted(RuntimeError):
    """The rejection-sampled Kemeleon variant was rejected T = max_attempts times in a row (the bottom output of `start` in Definition 1 of the paper)."""


@dataclass
class HpkeCrypto:
    """Cryptographic layer: HPKE base mode (info = profile_id) + encoder enc <-> kem_part.

    `encode` may return None (rejection-sampled Kemeleon variant, draft-02 Section 5.1). The sender then runs SetupBaseS
    again with fresh randomness (a new encapsulation), offline; the number of attempts is stored in `last_attempts`.
    """

    name: str
    kem: hpke.Kem
    encode: Callable[[bytes], bytes | None] = field(default=lambda b: b)
    decode: Callable[[bytes], bytes] = field(default=lambda b: b)
    kem_part_bytes: int = 0
    max_attempts: int = 64
    last_attempts: int = 0

    def keygen(self):
        return self.kem.generate()           # (sk, pk_bytes)

    def start(self, pk: bytes, pid: bytes):
        for attempt in range(1, self.max_attempts + 1):
            enc, ctx = hpke.setup_base_s(self.kem, pk, pid)
            kem_part = self.encode(enc)
            if kem_part is not None:
                break
        else:
            # T attempts exhausted (rejection-sampled Kemeleon): probability (1 - p_acc)^T, about 2^-134.7 for T = 64. In the
            # security game (paper, Definition 1) this is the bottom output in branch b = 0, already covered by the lemma.
            raise KemEncodingExhausted(f"{self.name}: kem_part encoding rejected {self.max_attempts} times in a row")
        self.last_attempts = attempt
        assert len(kem_part) == self.kem_part_bytes
        return kem_part, ctx, session_id(pid, kem_part)

    def accept(self, sk, kem_part: bytes, pid: bytes):
        try:
            enc = self.decode(kem_part)
        except Exception:
            return None
        ctx = hpke.setup_base_r(self.kem, enc, sk, pid)
        return None if ctx is None else (ctx, session_id(pid, kem_part))


@dataclass
class ReceiveResult:
    msg: bytes | None
    status: str                       # "ok" | "bottom" | "replay" | "auth_fail" | "stale" | "pending" (internal use only, A5)
    sid: bytes | None = None
    index: int | None = None
    authenticated: bool | None = None  # True: valid sigma under a trusted vk_S; None: anonymous session (no sender attributed)
    flags: int = 0
    trials: int = 0                   # AEAD openings tried for this image (<= |B|(1 + (2W - 1)S_max + P_max))
    released: tuple = ()              # A5b: ((1, m1),) when image 2 carries a valid sigma and releases the pending message of image 1
