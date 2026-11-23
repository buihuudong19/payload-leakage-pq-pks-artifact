"""Fisher-Yates pseudorandom permutation (PRP) driven by a SHAKE256 stream.

A0: the public pixel set (public seed s0 that depends on the image size); carries c_hat in stage 1.
A1: the remaining pixels, permuted under k_pi (stage 2). For image i >= 2 of a session, A1 is the whole image.
"""
from __future__ import annotations

import hashlib
import math
from functools import lru_cache

import numpy as np
from numba import njit


class _Shake:
    def __init__(self, seed: bytes):
        self._x = hashlib.shake_256(seed)
        self._buf, self._pos, self._len = b"", 0, 0

    def u64(self, count: int) -> np.ndarray:
        need = 8 * count
        self._len += need
        data = self._x.digest(self._len)[self._len - need:]
        return np.frombuffer(data, dtype="<u8")


@njit(cache=True)
def _fy_core(n, draws):
    """Fisher-Yates with rejection sampling on 64-bit words. Returns (permutation, number of draws used); when the draws
    run out, returns a negative count so that the Python caller supplies more (rejection probability < 2^-40 per draw for n <= 2^24)."""
    perm = np.arange(n, dtype=np.int64)
    di = 0
    two64 = 18446744073709551616.0
    for i in range(n - 1, 0, -1):
        bound = np.uint64(i + 1)
        rem = (np.uint64(0) - bound) % bound          # (2^64 mod bound)
        while True:
            if di >= draws.size:
                return perm, -1
            v = draws[di]
            di += 1
            if v < np.uint64(0) - rem or rem == np.uint64(0):
                break
        j = np.int64(v % bound)
        t = perm[i]
        perm[i] = perm[j]
        perm[j] = t
    return perm, di


def fisher_yates(n: int, seed: bytes) -> np.ndarray:
    """Permutation of [0, n) determined by the seed. Each index j is drawn uniformly without bias (rejection).

    limit = 2^64 - (2^64 mod bound); v is accepted when v < limit. The core runs under numba (same algorithm and same
    SHAKE256 stream as the earlier pure-Python version, hence the same permutation)."""
    rnd = _Shake(seed)
    draws = rnd.u64(2 * n).astype(np.uint64)
    while True:
        perm, used = _fy_core(n, draws)
        if used >= 0:
            return perm
        draws = np.concatenate([draws, rnd.u64(n).astype(np.uint64)])


def public_seed(height: int, width: int, context: bytes) -> bytes:
    return hashlib.sha3_256(context + b"/A0" + height.to_bytes(4, "big") + width.to_bytes(4, "big")).digest()


def stage_sizes(n: int, kem_bits: int, alpha0: float) -> int:
    """n0 = ceil(l_c / alpha0), with alpha0 a PUBLIC parameter."""
    n0 = math.ceil(kem_bits / alpha0)
    if n0 >= n:
        raise ValueError(f"image too small for stage 1: need n0 = {n0} < n = {n}")
    return n0


@lru_cache(maxsize=8)
def public_permutation(height: int, width: int, context: bytes) -> np.ndarray:
    """Public pi_0 on [n) (seed s0 depends only on the image size and the context)."""
    pi0 = fisher_yates(height * width, public_seed(height, width, context))
    pi0.flags.writeable = False
    return pi0


def partition(height: int, width: int, context: bytes, kem_bits: int, alpha0: float):
    """(A0, A1) with a fixed alpha0 (layout A8'). Kept for compatibility; layout A8 uses `partition_n0`."""
    return partition_n0(height, width, context, stage_sizes(height * width, kem_bits, alpha0))


@lru_cache(maxsize=64)
def partition_n0(height: int, width: int, context: bytes, n0: int):
    """Returns (A0, A1) as flat (row-major) indices, A0 = pi_0[0:n0]. The partition is public and deterministic, so it is
    cached (read-only arrays); under A8 every bucket has its own n0."""
    pi0 = public_permutation(height, width, context)
    a0, a1 = pi0[:n0].copy(), pi0[n0:].copy()
    a0.flags.writeable = False
    a1.flags.writeable = False
    return a0, a1
