"""Single-bucket profile for a given rate alpha, used by regime R-b.

With the default B, a 512 x 512 image reaches at most alpha = 0.297 (first image) and 0.2505 (later image).
Regime R-b therefore uses a separate profile for each alpha, with B_alpha = {B_alpha}:
    B_alpha = floor(alpha*n/8) - [first image]*|kem_part| - 16   (bytes of pt; 16 B is the AEAD tag).
The total payload of the image (kem_part + body) is then exactly floor(alpha*n/8) bytes. This is still the real protocol,
because B is a parameter of the profile (the pid changes accordingly).
"""
from __future__ import annotations

import math
from dataclasses import replace

from ..params import AEAD_TAG_BYTES, DEFAULT_PP, FRAME_HEADER_BYTES, START_HEADER_BYTES, PublicParams


def bucket_for_alpha(alpha: float, n: int, kem_bytes: int, start: bool = True) -> int:
    total = math.floor(alpha * n / 8)
    b = total - (kem_bytes if start else 0) - AEAD_TAG_BYTES
    if b < (START_HEADER_BYTES if start else FRAME_HEADER_BYTES):
        raise ValueError(f"alpha = {alpha} is too small for a kem_part of {kem_bytes} B in an image of {n} pixels")
    return b


def profile_for_alpha(alpha: float, n: int, kem_bytes: int, start: bool = True,
                      pp: PublicParams = DEFAULT_PP) -> PublicParams:
    return replace(pp, buckets=(bucket_for_alpha(alpha, n, kem_bytes, start),))


def message_len_for_bucket(bucket: int, start: bool = True, sig_bytes: int = 0) -> int:
    """Message length that fills the bucket exactly (no padding left)."""
    return bucket - (START_HEADER_BYTES if start else FRAME_HEADER_BYTES) - sig_bytes
