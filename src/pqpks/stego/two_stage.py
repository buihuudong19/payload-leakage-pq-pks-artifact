"""Two-stage HILL-STC embedder, shared by BL3, BL4, BL5, PQ-PKS and REF-U.

Stage 1 (first image of a session only): kem_part -> LSB(y[A0]), matrix H0 selected by a PUBLIC seed.
Stage 2: body -> LSB(y[pi_K(A1)]), permutation pi_K under k_pi, matrix H_K selected under k_H.
Images without stage 1 (image i >= 2 of a session): A1 is the whole image.
+/-1 changes: pixel 0 -> +1; pixel 255 -> -1; otherwise the direction is random (CSPRNG).
"""
from __future__ import annotations

import hashlib
import secrets

import numpy as np

from ..params import CONTEXT, DEFAULT_PP, PublicParams
from . import costs, partition, stc


def _h0_seed(height: int, width: int) -> bytes:
    return hashlib.sha3_256(CONTEXT + b"/H0" + height.to_bytes(4, "big") + width.to_bytes(4, "big")).digest()


def _bits(data: bytes) -> np.ndarray:
    return np.unpackbits(np.frombuffer(data, dtype=np.uint8))


def _bytes(bits: np.ndarray) -> bytes:
    return np.packbits(bits.astype(np.uint8)).tobytes()


class TwoStageSTC:
    def __init__(self, pp: PublicParams = DEFAULT_PP, cost: str = "hill", stage2_keyed: bool = True,
                 keyed_perm: bool | None = None, keyed_matrix: bool | None = None):
        """Ablation A1: V1 (permutation, matrix) = (public, public); V2 = (keyed, public);
        V3 = (public, keyed); V4 = (keyed, keyed) is PQ-PKS. `stage2_keyed` sets both at once (backward compatible)."""
        self.pp = pp
        self.cost = cost
        self.keyed_perm = stage2_keyed if keyed_perm is None else keyed_perm
        self.keyed_matrix = stage2_keyed if keyed_matrix is None else keyed_matrix
        self.stage2_keyed = self.keyed_perm and self.keyed_matrix

    # ------------------------------------------------------------------ indices
    def _sets(self, shape, kem_bits: int | None, body_bits: int | None = None):
        """(A0, A1). Under layout A8, A0 depends on the bucket through body_bits = 8(B + 16)."""
        h, w = shape
        if kem_bits is None:
            return None, np.arange(h * w, dtype=np.int64)
        if self.pp.layout == "A8" and body_bits is None:
            raise ValueError("layout A8: the body length (bucket) is needed to determine A0(B)")
        return partition.partition_n0(h, w, CONTEXT, self.pp.n0(h * w, kem_bits, body_bits or 0))

    def a0_mask(self, shape, kem_bytes: int, body_bytes: int | None = None) -> np.ndarray:
        """Boolean mask of A0 (warden W-A0 knows A0, by Kerckhoffs' principle)."""
        a0, _ = self._sets(shape, 8 * kem_bytes, None if body_bytes is None else 8 * body_bytes)
        m = np.zeros(shape[0] * shape[1], dtype=bool)
        m[a0] = True
        return m.reshape(shape)

    def _stage2_index(self, a1: np.ndarray, k_pi: bytes) -> np.ndarray:
        if not self.keyed_perm:
            return a1
        return a1[partition.fisher_yates(a1.size, k_pi)]

    def _stage2_seed(self, shape, k_h: bytes) -> bytes:
        return k_h if self.keyed_matrix else _h0_seed(*shape)

    # ------------------------------------------------------------------ embedding
    def cost_map(self, cover: np.ndarray):
        """(rho, +1 allowed, -1 allowed) of a cover; computed once and reused by several systems in an experiment."""
        return tuple(a.reshape(-1) for a in costs.compute(np.asarray(cover, dtype=np.uint8), self.cost))

    def embed(self, cover: np.ndarray, kem_part: bytes | None, body: bytes, k_pi: bytes, k_h: bytes,
              cost_map=None) -> np.ndarray:
        x = np.asarray(cover, dtype=np.uint8)
        flat = x.reshape(-1).astype(np.int16)
        lsb = (flat & 1).astype(np.uint8)
        rho, plus_ok, minus_ok = cost_map if cost_map is not None else self.cost_map(x)
        new_lsb = lsb.copy()
        kem_bits = None if kem_part is None else 8 * len(kem_part)
        a0, a1 = self._sets(x.shape, kem_bits, 8 * len(body))
        if kem_part is not None:
            y0, _ = stc.embed(lsb[a0], rho[a0], _bits(kem_part), _h0_seed(*x.shape), self.pp.stc_h)
            new_lsb[a0] = y0
        idx1 = self._stage2_index(a1, k_pi)
        y1, _ = stc.embed(lsb[idx1], rho[idx1], _bits(body), self._stage2_seed(x.shape, k_h), self.pp.stc_h)
        new_lsb[idx1] = y1
        changed = np.nonzero(new_lsb != lsb)[0]
        for p in changed:                               # +/-1 on the pixels whose low bit must change
            if plus_ok[p] and minus_ok[p]:
                flat[p] += 1 if secrets.randbits(1) else -1
            elif plus_ok[p]:
                flat[p] += 1
            elif minus_ok[p]:
                flat[p] -= 1
            else:
                raise ValueError("STC selected a pixel that may not be changed")
        return flat.astype(np.uint8).reshape(x.shape)

    # ------------------------------------------------------------------ extraction
    def extract_kem(self, stego: np.ndarray, kem_bytes: int, body_bytes: int | None = None) -> bytes:
        """Ext0: public extraction of kem_part from A0 (under A8: the A0 of the bucket whose body has body_bytes)."""
        y = np.asarray(stego, dtype=np.uint8)
        a0, _ = self._sets(y.shape, 8 * kem_bytes, None if body_bytes is None else 8 * body_bytes)
        lsb = (y.reshape(-1) & 1).astype(np.uint8)
        return _bytes(stc.extract(lsb[a0], 8 * kem_bytes, _h0_seed(*y.shape), self.pp.stc_h))

    def stage2_index(self, shape, kem_bytes: int | None, k_pi: bytes, body_bytes: int | None = None) -> np.ndarray:
        _, a1 = self._sets(shape, None if kem_bytes is None else 8 * kem_bytes,
                           None if body_bytes is None else 8 * body_bytes)
        return self._stage2_index(a1, k_pi)

    def extract_body(self, stego: np.ndarray, idx1: np.ndarray, body_bytes: int, k_h: bytes) -> bytes | None:
        y = np.asarray(stego, dtype=np.uint8)
        lsb = (y.reshape(-1) & 1).astype(np.uint8)
        if idx1.size < 8 * body_bytes:
            return None
        return _bytes(stc.extract(lsb[idx1], 8 * body_bytes, self._stage2_seed(y.shape, k_h), self.pp.stc_h))
