"""Spatial-domain distortion costs (HILL [LWHN14], S-UNIWARD [HFD14]) through the conseal package.

Returns the binary cost rho (for STC on the LSB plane) and the flags that allow +1 / -1.
Pixel value 0 may only be changed by +1 and pixel value 255 only by -1.
"""
from __future__ import annotations

import numpy as np


def compute(x: np.ndarray, kind: str = "hill") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import conseal
    x = np.asarray(x, dtype=np.uint8)
    if kind == "hill":
        rho_p, rho_m = conseal.hill.compute_cost_adjusted(x)
    elif kind == "suniward":
        rho_p, rho_m = conseal.suniward.compute_cost_adjusted(x)
    else:
        raise ValueError(kind)
    plus_ok = np.isfinite(rho_p) & (x < 255)
    minus_ok = np.isfinite(rho_m) & (x > 0)
    rho = np.minimum(np.where(plus_ok, rho_p, np.inf), np.where(minus_ok, rho_m, np.inf))
    return rho.astype(np.float64), plus_ok, minus_ok
