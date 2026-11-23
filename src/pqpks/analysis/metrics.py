"""Image distortion metrics (secondary metrics only; they are NOT security metrics)."""
from __future__ import annotations

import math

import numpy as np


def distortion(cover: np.ndarray, stego: np.ndarray) -> dict:
    c = cover.astype(np.int64)
    s = stego.astype(np.int64)
    d = s - c
    mse = float((d ** 2).mean())
    return {"changed": int((d != 0).sum()), "change_rate": float((d != 0).mean()),
            "max_abs_diff": int(np.abs(d).max()),
            "psnr": float("inf") if mse == 0 else 10 * math.log10(255 ** 2 / mse)}
