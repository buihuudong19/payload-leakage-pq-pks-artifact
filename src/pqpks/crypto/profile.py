"""Canonical encoding CE and profile_id.

CE(f_1, ..., f_k) = concatenation of tag(1 B) || len(2 B, big-endian) || value, in a fixed field order.
- bytes  -> tag 0x01, value verbatim;
- int    -> tag 0x02, 8-byte unsigned big-endian;
- str    → tag 0x03, UTF-8;
- float  -> tag 0x04, repr string (used only for alpha0 in the profile).
Every field carries its length, so CE is injective on field sequences of equal length and types.

profile_id = SHA3-256("PQPKS/profile" || CE(profile)): binds the version, the KEM, the kem_part encoding, the KDF,
the AEAD, the signature, h, alpha0, B, W and N_max. The receiver must know the profile before extracting A0.
"""
from __future__ import annotations

import hashlib

PROFILE_LABEL = b"PQPKS/profile"


def _field(v) -> bytes:
    if isinstance(v, bool):
        raise TypeError("CE: use int 0/1 instead of bool")
    if isinstance(v, (bytes, bytearray)):
        tag, val = 0x01, bytes(v)
    elif isinstance(v, int):
        if not 0 <= v < 1 << 64:
            raise ValueError("CE: integer outside [0, 2^64)")
        tag, val = 0x02, v.to_bytes(8, "big")
    elif isinstance(v, str):
        tag, val = 0x03, v.encode()
    elif isinstance(v, float):
        tag, val = 0x04, repr(v).encode()
    else:
        raise TypeError(f"CE: unsupported type {type(v)!r}")
    if len(val) >= 1 << 16:
        raise ValueError("CE: field too long")
    return bytes([tag]) + len(val).to_bytes(2, "big") + val


def ce(*fields) -> bytes:
    """Canonical encoding of a field sequence (tag || length || value)."""
    return b"".join(_field(f) for f in fields)


def profile_id(*fields) -> bytes:
    return hashlib.sha3_256(PROFILE_LABEL + ce(*fields)).digest()
