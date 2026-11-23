"""Pseudorandom point encoding of [Zha25] on P-256, used by baseline BL4.

Loads `third_party/zha25_public_key_discop/PRNEncryption_P256.py` VERBATIM (commit a03633c, SHA-256 checked) without
changing any line. The file calls argparse at module level, so sys.argv is temporarily set to [file name] while loading.
The file is not redistributed with this repository; scripts/setup/fetch_third_party.py downloads it.

Only the "Point Hiding" and "Bias Eliminating" parts of the original work are used:
    point P -> (u, v) with forward_map(u) + forward_map(v) = P   (admissible encoding, SWU)
    u, v in F_p -> u + k*p, v + k'*p < 2^(256+32)                 (LT = 32, 36 B each)
kem_part = u' || v' (72 B). The shared key follows HPKE DHKEM(P-256) as in BL3, NOT the `encrypt` method of the
original file (docs/baselines.md explains why).
"""
from __future__ import annotations

import hashlib
import importlib.util
import secrets
import sys
from functools import lru_cache
from pathlib import Path

_DIR = Path(__file__).resolve().parents[3] / "third_party" / "zha25_public_key_discop"
_FILE = _DIR / "PRNEncryption_P256.py"
_SHA256 = "eb7d15a3eec534943f0a99b3734f09ab44625b1a41e24c9110134037c5952ed9"

LT = 32                     # slack of the bias-elimination step, the default value of the original work
FIELD_BYTES = (256 + LT) // 8   # 36
KEM_PART_BYTES = 2 * FIELD_BYTES   # 72


@lru_cache(maxsize=1)
def _mod():
    if not _FILE.is_file():
        from pqpks.data import MissingResource
        raise MissingResource(f"{_FILE} not found (run scripts/setup/fetch_third_party.py)")
    data = _FILE.read_bytes()
    if hashlib.sha256(data).hexdigest() != _SHA256:
        raise RuntimeError(f"{_FILE} does not match the recorded SHA-256 (third_party/.../SHA256SUMS)")
    spec = importlib.util.spec_from_file_location("zha25_prn_p256", _FILE)
    mod = importlib.util.module_from_spec(spec)
    argv = sys.argv
    sys.argv = [str(_FILE)]
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.argv = argv
    return mod


def _bias_eliminate(x: int) -> int:
    m = _mod()
    return m.PRNEncryption_P256.encode_bytes(None, x, m.P_256.p, LT, 256)


def encode_point(enc: bytes) -> bytes:
    """enc: uncompressed P-256 point (0x04 || X || Y, 65 B, DHKEM Encaps output) -> kem_part of 72 B."""
    if len(enc) != 65 or enc[0] != 4:
        raise ValueError("enc must be an uncompressed 65-byte P-256 point")
    m = _mod()
    P = (int.from_bytes(enc[1:33], "big"), int.from_bytes(enc[33:], "big"), 1)
    u, v = m.encode(P, secrets.token_bytes(32))
    return (_bias_eliminate(u.val).to_bytes(FIELD_BYTES, "big")
            + _bias_eliminate(v.val).to_bytes(FIELD_BYTES, "big"))


def decode_point(kem_part: bytes) -> bytes:
    """kem_part of 72 B -> uncompressed 65-byte P-256 point (a public map that anyone can compute)."""
    if len(kem_part) != KEM_PART_BYTES:
        raise ValueError("a BL4 kem_part must be 72 bytes long")
    m = _mod()
    p = m.P_256.p
    u = int.from_bytes(kem_part[:FIELD_BYTES], "big") % p
    v = int.from_bytes(kem_part[FIELD_BYTES:], "big") % p
    x, y = m.decode(m.fe(u), m.fe(v))
    return b"\x04" + x.val.to_bytes(32, "big") + y.val.to_bytes(32, "big")
