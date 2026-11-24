"""Receiver with cached key-dependent work (post hoc, 24 Nov 2026; finding F06 of the regression audit).

Profiling of the reference receiver (src/pqpks/systems/stc_system.py, unchanged) on an invalid 512x512 image with S_max = 16 open
sessions shows where the time goes: the STC syndromes of stage 2 (about 60 %), the keyed stage-2 permutation, i.e. SHAKE output and
Fisher-Yates over A1 (about 30 %), and gathering the LSBs (about 6 %); the 1414 ChaCha20-Poly1305 openings take well under 1 %.

The stage-2 permutation depends only on the image key (session, index, bucket), not on the image, so it can be computed once and
reused for every later image; the LSB plane and the gathered bits depend only on the image and the permutation. CachedEmbedder
wraps the reference embedder and caches exactly these three objects, and computes the stage-2 syndrome with a shift-register
form of the same linear map (syndrome_shift), which gives bit-identical output. Everything else, including the order of trials, the AEAD
and every decision of the receiver, is the inherited reference code, so the output (message, status, number of trials) is
identical by construction; tests/test_receiver_fast.py checks it on valid and invalid images.

    from receiver_fast import fast_system
    s = fast_system("PQ-PKS")          # same object as systems.make("PQ-PKS"), with the cached embedder
"""
from __future__ import annotations

import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from numba import njit  # noqa: E402

from pqpks import systems  # noqa: E402
from pqpks.stego import stc  # noqa: E402
from pqpks.stego.two_stage import _bytes  # noqa: E402


@njit(cache=True)
def syndrome_shift(y, cols, widths, M):
    """Same syndrome as stc._syndrome, computed with an h-bit shift register: while block i is read, bit r of the register holds the
    partial parity of row i + r; after the block, bit 0 is s_i. Contributions to rows >= M are shifted out and never output,
    which is what the row mask of stc._syndrome does. One XOR per set bit instead of a loop over the h bits of the column."""
    s = np.zeros(M, dtype=np.uint8)
    state = np.int64(0)
    idx = 0
    for i in range(M):
        for j in range(widths[i]):
            if y[idx]:
                state ^= cols[j]
            idx += 1
        s[i] = state & 1
        state >>= 1
    return s


_COLS: dict = {}


def syndrome(bits: np.ndarray, m_len: int, seed: bytes, h: int) -> np.ndarray:
    """Drop-in for stc.extract (same checks, same widths and submatrix); the submatrix is cached by (seed, h, width)."""
    n = bits.size
    if m_len < 1 or m_len > n:
        raise ValueError("requested length exceeds the number of elements")
    widths = stc.block_widths(n, m_len)
    w = stc.width_for(n, m_len)
    key = (seed, h, w)
    cols = _COLS.get(key)
    if cols is None:
        cols = stc.submatrix(seed, h, w) & ((1 << h) - 1)     # the reference masks every column to h bits
        if len(_COLS) > 4096:
            _COLS.clear()
        _COLS[key] = cols
    return syndrome_shift(bits, cols, widths, m_len)


class CachedEmbedder:
    """Delegates to a TwoStageSTC; caches stage-2 permutations (LRU, by key), and the LSB plane and gathered bits of the image
    currently being received (by object identity)."""

    def __init__(self, base, max_perms: int = 2048):
        self._b = base
        self._perms: OrderedDict = OrderedDict()
        self._max = max_perms
        self._img = None
        self._lsb = None
        self._gather: dict = {}
        self.hits = self.misses = 0

    def __getattr__(self, name):
        return getattr(self._b, name)

    def stage2_index(self, shape, kem_bytes, k_pi, body_bytes=None):
        key = (tuple(shape), kem_bytes, bytes(k_pi), body_bytes)
        v = self._perms.get(key)
        if v is not None:
            self._perms.move_to_end(key)
            self.hits += 1
            return v
        self.misses += 1
        v = self._b.stage2_index(shape, kem_bytes, k_pi, body_bytes)
        v.setflags(write=False)
        self._perms[key] = v
        if len(self._perms) > self._max:
            self._perms.popitem(last=False)
        return v

    def _bits_for(self, stego: np.ndarray, idx1: np.ndarray) -> np.ndarray:
        if stego is not self._img:
            self._img, self._gather = stego, {}
            self._lsb = (np.asarray(stego, dtype=np.uint8).reshape(-1) & 1).astype(np.uint8)
        k = id(idx1)
        g = self._gather.get(k)
        if g is None or g[0] is not idx1:
            g = (idx1, self._lsb[idx1])
            self._gather[k] = g
        return g[1]

    def extract_body(self, stego, idx1, body_bytes, k_h):
        if idx1.size < 8 * body_bytes:
            return None
        y = np.asarray(stego, dtype=np.uint8)
        bits = self._bits_for(stego, idx1)
        return _bytes(syndrome(bits, 8 * body_bytes, self._b._stage2_seed(y.shape, k_h), self._b.pp.stc_h))


def fast_system(name: str = "PQ-PKS", *args, **kw):
    s = systems.make(name, *args, **kw)
    s.embedder = CachedEmbedder(s.embedder)
    return s
