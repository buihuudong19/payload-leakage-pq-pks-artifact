"""Two reference systems (not baselines):

REF-U  : uniform payload (CSPRNG) with the SAME length and the SAME two-stage A0/A1 layout as the compared system
         (PQ-PKS by default, |kem_part| = 1536 B). It is the TOST counterpart of PQ-PKS: only the payload distribution differs.
REF-1S : uniform payload of the same TOTAL length, embedded in ONE stage over the whole image (permutation and matrix under
         a random key). Used only to measure the cost of the two-stage layout (alpha0 != alpha1), never in a TOST.
Both systems only produce stego images; there is no receiver.
"""
from __future__ import annotations

import secrets

import numpy as np

from ..params import DEFAULT_PP, PublicParams
from ..stego.two_stage import TwoStageSTC


class ReferenceSystem:
    def __init__(self, name: str, kem_bytes: int = 1536, pp: PublicParams = DEFAULT_PP, cost: str = "hill"):
        if name not in ("REF-U", "REF-1S"):
            raise KeyError(name)
        self.name = name
        self.kem_bytes = kem_bytes
        self.pp = pp
        self.embedder = TwoStageSTC(pp, cost)

    def stego(self, cover: np.ndarray, body_bytes: int, start: bool = True, cost_map=None) -> np.ndarray:
        """body_bytes = |pt| + 16 (the body length of the compared system)."""
        k_pi, k_h = secrets.token_bytes(32), secrets.token_bytes(32)
        if self.name == "REF-U":
            kem = secrets.token_bytes(self.kem_bytes) if start else None
            return self.embedder.embed(cover, kem, secrets.token_bytes(body_bytes), k_pi, k_h, cost_map)
        total = (self.kem_bytes if start else 0) + body_bytes
        return self.embedder.embed(cover, None, secrets.token_bytes(total), k_pi, k_h, cost_map)
