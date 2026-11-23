"""Public distinguishers D_pub.

Applied to the publicly extracted string (kem_part) or to the payload. Used by the sanity matrix
(scripts/sanity_matrix.py) and by E2.
"""
from __future__ import annotations

import math
from collections import Counter

import numpy as np
from scipy import special, stats

from ..crypto import fips203
from ..params import MLKEM_Q as Q

# ----------------------------------------------------------------------------
# Distinguishers specific to a serialization
# ----------------------------------------------------------------------------
_COMP10 = Counter(fips203.compress(np.arange(Q), 10).tolist())
_LLR10 = np.array([math.log((_COMP10[j] / Q) / (1 / 1024)) for j in range(1024)])


def mlkem_raw_llr(blob: bytes) -> float:
    """Lambda = sum_t log(P_10(u_t) / 2^-10) over the first 768 10-bit coefficients of a raw ML-KEM-768 ciphertext.
    Lambda > 0 favours 'raw ML-KEM ciphertext' (paper, Propositions 2-3)."""
    vals = fips203.byte_decode(blob[:960], 10, 768)
    return float(_LLR10[vals].sum())


_ASCII_SET = set(b"0123456789 \n")


def ascii_digit_fraction(blob: bytes) -> float:
    """Fraction of bytes in {digits, space, newline}: about 12/256 for uniform data, about 1 for BL1."""
    return sum(b in _ASCII_SET for b in blob) / max(1, len(blob))


# P-256 curve parameters (FIPS 186-5)
_P256_P = 0xFFFFFFFF00000001000000000000000000000000FFFFFFFFFFFFFFFFFFFFFFFF
_P256_B = 0x5AC635D8AA3A93E7B3EBBD55769886BC651D06B0CC53B0F63BCE3C3E27D2604B


def p256_uncompressed_on_curve(blob: bytes) -> bool:
    """True if the first 65 bytes are a valid uncompressed P-256 point (0x04 || x || y, y^2 = x^3 - 3x + b)."""
    if len(blob) < 65 or blob[0] != 0x04:
        return False
    x = int.from_bytes(blob[1:33], "big")
    y = int.from_bytes(blob[33:65], "big")
    if x >= _P256_P or y >= _P256_P:
        return False
    return (y * y - (x * x * x - 3 * x + _P256_B)) % _P256_P == 0


# ----------------------------------------------------------------------------
# Generic tests (basic tests of NIST SP 800-22)
# ----------------------------------------------------------------------------
def _bits(blob: bytes) -> np.ndarray:
    return np.unpackbits(np.frombuffer(blob, dtype=np.uint8))


def monobit_p(blob: bytes) -> float:
    b = _bits(blob).astype(np.int64)
    s = abs(int((2 * b - 1).sum())) / math.sqrt(b.size)
    return float(special.erfc(s / math.sqrt(2)))


def runs_p(blob: bytes) -> float:
    b = _bits(blob).astype(np.int64)
    n = b.size
    pi = b.mean()
    if abs(pi - 0.5) >= 2 / math.sqrt(n):
        return 0.0
    v = 1 + int((b[1:] != b[:-1]).sum())
    return float(special.erfc(abs(v - 2 * n * pi * (1 - pi)) / (2 * math.sqrt(2 * n) * pi * (1 - pi))))


def byte_chi2_p(blob: bytes) -> float:
    counts = np.bincount(np.frombuffer(blob, dtype=np.uint8), minlength=256)
    return float(stats.chisquare(counts).pvalue)


def generic_uniformity(blob: bytes) -> dict[str, float]:
    return {"monobit_p": monobit_p(blob), "runs_p": runs_p(blob), "byte_chi2_p": byte_chi2_p(blob)}
