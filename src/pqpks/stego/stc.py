"""Syndrome-Trellis Codes (Filler, Judas, Fridrich, IEEE TIFS 2011), binary version.

The sender solves  min_y sum rho_i [y_i != x_i]  s.t.  H * y = m (mod 2)  with the Viterbi algorithm on the trellis.
The receiver only computes the syndrome H * y and does not need the costs rho.

Structure of H: the submatrix H_hat (h x w_max) is repeated along the diagonal, each block shifted down by one row;
the columns of the last h - 1 blocks are truncated to the first m rows. A column of H_hat is stored as an h-bit
integer whose bit 0 corresponds to the row of the current message bit.

Non-integer width: when n/m is not an integer, block i has width
w_i = floor((i+1)n/m) - floor(i*n/m) in {floor(n/m), ceil(n/m)} and uses the first w_i columns of H_hat (selected with
w_max = ceil(n/m)). The widths sum to n, so every cover element is used and the actual rate is exactly m/n.

The submatrix H_hat comes from a PUBLIC TABLE (stc_table_h10.json, built deterministically by
scripts/build_stc_table.py: the top 8 of 48 candidates for each w = 1..128); the seed only selects the table index
(SubmatrixSelect). The table replaces the optimized H_hat table of [FJF11]; its efficiency is checked against the
rate-distortion bound in tests/test_stc.py.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from numba import njit

INF = 1e300


def random_submatrix(seed: bytes, h: int, w: int) -> np.ndarray:
    """Deterministic generation of w random columns of H_hat (h bits each) from a seed; the first and last bits are always 1."""
    stream = hashlib.shake_256(b"PQPKS-v1/STC-submatrix" + seed).digest(4 * w)
    cols = np.frombuffer(stream, dtype="<u4").astype(np.int64) & ((1 << h) - 1)
    cols |= 1 | (1 << (h - 1))
    return cols


_TABLE = None


def _table():
    global _TABLE
    if _TABLE is None:
        p = Path(__file__).with_name("stc_table_h10.json")
        _TABLE = json.loads(p.read_text()) if p.exists() else {"table": {}, "top_k": 1}
    return _TABLE


def submatrix(seed: bytes, h: int, w: int) -> np.ndarray:
    """SubmatrixSelect(seed): pick H_hat from the public table (top-K candidates for w); the seed decides the index.
    When (h, w) is outside the table, a random H_hat generated from the seed is used."""
    t = _table()
    if h == t.get("h") and str(w) in t["table"]:
        cands = t["table"][str(w)]
        idx = int.from_bytes(hashlib.sha3_256(b"PQPKS-v1/STC-select" + seed).digest()[:4], "big") % len(cands)
        return np.array(cands[idx], dtype=np.int64)
    return random_submatrix(seed, h, w)


@njit(cache=True)
def _viterbi(x, rho, m, cols, h, widths):
    M = m.shape[0]
    S = 1 << h
    words = max(1, S // 64)
    n_used = 0
    for i in range(M):
        n_used += widths[i]
    path = np.zeros((n_used, words), dtype=np.uint64)
    wght = np.full(S, INF)
    wght[0] = 0.0
    neww = np.empty(S)
    idx = 0
    for i in range(M):
        rows_left = M - i
        mask = (1 << h) - 1 if rows_left >= h else (1 << rows_left) - 1
        for j in range(widths[i]):
            c = cols[j] & mask
            xb = x[idx]
            r = rho[idx]
            c0 = r if xb == 1 else 0.0          # choose y = 0
            c1 = 0.0 if xb == 1 else r          # choose y = 1
            for k in range(S):
                w0 = wght[k] + c0
                w1 = wght[k ^ c] + c1
                if w1 < w0:
                    neww[k] = w1
                    path[idx, k >> 6] |= np.uint64(1) << np.uint64(k & 63)
                else:
                    neww[k] = w0
            for k in range(S):
                wght[k] = neww[k]
            idx += 1
        # Pruning: bit 0 of the state must equal m[i]; shift down by one row.
        half = S >> 1
        for k in range(half):
            neww[k] = wght[(k << 1) | m[i]]
        for k in range(half, S):
            neww[k] = INF
        for k in range(S):
            wght[k] = neww[k]
    cost = wght[0]
    y = np.zeros(n_used, dtype=np.uint8)
    state = 0
    idx = n_used - 1
    for i in range(M - 1, -1, -1):
        state = (state << 1) | m[i]
        rows_left = M - i
        mask = (1 << h) - 1 if rows_left >= h else (1 << rows_left) - 1
        for j in range(widths[i] - 1, -1, -1):
            bit = (path[idx, state >> 6] >> np.uint64(state & 63)) & np.uint64(1)
            y[idx] = np.uint8(bit)
            if bit:
                state ^= cols[j] & mask
            idx -= 1
    return y, cost


@njit(cache=True)
def _syndrome(y, cols, h, widths, M):
    s = np.zeros(M, dtype=np.uint8)
    idx = 0
    for i in range(M):
        rows_left = M - i
        mask = (1 << h) - 1 if rows_left >= h else (1 << rows_left) - 1
        for j in range(widths[i]):
            if y[idx]:
                c = cols[j] & mask
                r = 0
                while c:
                    if c & 1:
                        s[i + r] ^= 1
                    c >>= 1
                    r += 1
            idx += 1
    return s


def block_widths(n: int, m_bits: int) -> np.ndarray:
    """w_i = floor((i+1)n/m) - floor(i*n/m), i = 0..m-1: each w_i is in {floor(n/m), ceil(n/m)} and sum w_i = n."""
    if m_bits < 1 or m_bits > n:
        raise ValueError("payload longer than the number of cover elements")
    i = np.arange(m_bits + 1, dtype=np.int64)
    return np.diff((i * n) // m_bits).astype(np.int64)


def width_for(n: int, m_bits: int) -> int:
    """w_max = ceil(n / m): width of H_hat (the key of the public table)."""
    return -(-n // m_bits)


def embed_cols(x_bits: np.ndarray, rho: np.ndarray, msg_bits: np.ndarray, cols: np.ndarray, h: int,
               widths: np.ndarray | None = None):
    """Viterbi with a given submatrix. No widths are passed: every block has width cols.size."""
    if widths is None:
        widths = np.full(msg_bits.size, cols.size, dtype=np.int64)
    return _viterbi(x_bits.astype(np.uint8), rho.astype(np.float64), msg_bits.astype(np.int64), cols, h,
                    widths.astype(np.int64))


def embed(x_bits: np.ndarray, rho: np.ndarray, msg_bits: np.ndarray, seed: bytes, h: int):
    """Embed msg_bits into the cover bit string x_bits (length n), using all n elements.
    Returns (y_bits, cost) or raises ValueError."""
    n, m_len = x_bits.size, msg_bits.size
    widths = block_widths(n, m_len)
    cols = submatrix(seed, h, width_for(n, m_len))
    rho_ = np.where(np.isfinite(rho), rho, 1e12).astype(np.float64)
    y, cost = embed_cols(x_bits, rho_, msg_bits, cols, h, widths)
    if cost >= 1e12:
        raise ValueError("STC found no finite-cost solution (too many wet elements)")
    return y.astype(np.uint8), cost


def extract(y_bits: np.ndarray, m_len: int, seed: bytes, h: int) -> np.ndarray:
    n = y_bits.size
    if m_len < 1 or m_len > n:
        raise ValueError("requested length exceeds the number of elements")
    widths = block_widths(n, m_len)
    cols = submatrix(seed, h, width_for(n, m_len))
    return _syndrome(y_bits.astype(np.uint8), cols, h, widths, m_len)
