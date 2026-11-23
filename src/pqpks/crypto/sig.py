"""ML-DSA (FIPS 204) through the `cryptography` package (OpenSSL), context string "PQPKS-v1".

ML-DSA-65 by default (Category 3, the level of ML-KEM-768); ML-DSA-44 in ablation A5.
The signature covers the session transcript:
    transcript = CE("PQPKS/auth", pid, sid, SHA3-256(ek_R), u64(t₀))
The sender key identifier is not needed in the transcript: ML-DSA hashes the public key into the signed message
(tr = H(pk), FIPS 204).
"""
from __future__ import annotations

import hashlib

from cryptography.hazmat.primitives.asymmetric import mldsa

from ..params import CONTEXT
from .profile import ce

_ALGS = {
    "ML-DSA-44": (mldsa.MLDSA44PrivateKey, mldsa.MLDSA44PublicKey),
    "ML-DSA-65": (mldsa.MLDSA65PrivateKey, mldsa.MLDSA65PublicKey),
}


def keygen(alg: str = "ML-DSA-65"):
    sk = _ALGS[alg][0].generate()
    return sk, sk.public_key().public_bytes_raw()


def transcript(pid: bytes, sid: bytes, ek_r: bytes, t0: int) -> bytes:
    return ce("PQPKS/auth", pid, sid, hashlib.sha3_256(ek_r).digest(), t0)


def sign(sk, msg: bytes) -> bytes:
    return sk.sign(msg, context=CONTEXT)


def verify(vk: bytes, msg: bytes, signature: bytes, alg: str = "ML-DSA-65") -> bool:
    try:
        _ALGS[alg][1].from_public_bytes(vk).verify(signature, msg, context=CONTEXT)
        return True
    except Exception:
        return False
