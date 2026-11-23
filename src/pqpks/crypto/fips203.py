"""FIPS 203 (ML-KEM) byte encoding and coefficient compression needed by Kemeleon.

Only the parts needed to split and reassemble ciphertexts and encapsulation keys are implemented:
ByteEncode_d / ByteDecode_d (FIPS 203, Alg. 5-6) and Compress_d / Decompress_d (Section 4.2.1).
The KEM itself (KeyGen/Encaps/Decaps) comes from the `cryptography` package (OpenSSL).
"""
from __future__ import annotations

import numpy as np

from ..params import MLKEM_N, MLKEM_Q

Q = MLKEM_Q
N = MLKEM_N


def compress(x: np.ndarray, d: int) -> np.ndarray:
    """Compress_d(x) = round((2^d / q) * x) mod 2^d, rounding half up as in FIPS 203."""
    x = np.asarray(x, dtype=np.int64)
    return (((x << d) + Q // 2) // Q) % (1 << d)


def decompress(y: np.ndarray, d: int) -> np.ndarray:
    """Decompress_d(y) = round((q / 2^d) * y)."""
    y = np.asarray(y, dtype=np.int64)
    return (Q * y + (1 << (d - 1))) >> d


def byte_encode(f: np.ndarray, d: int) -> bytes:
    """ByteEncode_d: d bits per coefficient, little-endian bit order (FIPS 203, Alg. 5)."""
    f = np.asarray(f, dtype=np.int64).reshape(-1)
    bits = ((f[:, None] >> np.arange(d)) & 1).astype(np.uint8).reshape(-1)
    return np.packbits(bits, bitorder="little").tobytes()


def byte_decode(b: bytes, d: int, count: int) -> np.ndarray:
    """ByteDecode_d (FIPS 203, Alg. 6). For d = 12 the result is reduced modulo q."""
    bits = np.unpackbits(np.frombuffer(b, dtype=np.uint8), bitorder="little")[: count * d]
    vals = (bits.reshape(count, d).astype(np.int64) << np.arange(d)).sum(axis=1)
    return vals % Q if d == 12 else vals


def split_ciphertext(ct: bytes, k: int, du: int, dv: int) -> tuple[np.ndarray, np.ndarray]:
    """Split c = (c1, c2) into compressed coefficients: c1 has k*N coefficients of du bits, c2 has N of dv bits."""
    n1 = 32 * du * k
    c1 = byte_decode(ct[:n1], du, k * N)
    c2 = byte_decode(ct[n1:], dv, N)
    return c1, c2


def join_ciphertext(c1: np.ndarray, c2: np.ndarray, du: int, dv: int) -> bytes:
    return byte_encode(c1, du) + byte_encode(c2, dv)


def split_ek(ek: bytes, k: int) -> tuple[np.ndarray, bytes]:
    """ek = ByteEncode_12(t_hat) || rho. Returns (t_hat with k*N coefficients, rho)."""
    t = byte_decode(ek[: 384 * k], 12, k * N)
    return t, ek[384 * k:]


def join_ek(t: np.ndarray, rho: bytes) -> bytes:
    return byte_encode(t, 12) + rho
