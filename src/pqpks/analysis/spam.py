"""SPAM-686 features (Pevny, Bas, Fridrich, IEEE TIFS 2010): second-order Markov model of neighbouring pixel differences,
threshold T = 3, 8 directions; F1 averages the 4 horizontal/vertical directions and F2 the 4 diagonal ones, 2 x 343 = 686 dimensions.

Reimplemented from the original paper. Used by warden W-img in the pilot and screening rounds (CPU). SPAM is weaker than
SRM (34 671 dimensions) and SRNet, so the pilot P_E values are only loose lower bounds on detectability and do not replace E2/E3.
"""
from __future__ import annotations

import numpy as np

T = 3
K = 2 * T + 1


def _markov2(d: np.ndarray, axis_pairs, sel=None) -> np.ndarray:
    """Second-order transition matrix Pr(D3 | D1, D2) over consecutive triples in one direction.
    `sel` (optional, same shape as the triples): count only the triples with sel = True."""
    d1, d2, d3 = axis_pairs
    idx = (d1 + T) * K * K + (d2 + T) * K + (d3 + T)
    idx = idx.ravel() if sel is None else idx[sel]
    cnt = np.bincount(idx, minlength=K ** 3).astype(np.float64).reshape(K * K, K)
    norm = cnt.sum(axis=1, keepdims=True)
    return np.divide(cnt, norm, out=np.zeros_like(cnt), where=norm > 0).ravel()


def _triples(a, kind):
    """Three consecutive slices (D1, D2, D3) of the difference array `a` in direction `kind`; also used for coordinate arrays."""
    if kind == "h":
        return a[:, :-2], a[:, 1:-1], a[:, 2:]
    if kind == "v":
        return a[:-2], a[1:-1], a[2:]
    if kind == "d":
        return a[:-2, :-2], a[1:-1, 1:-1], a[2:, 2:]
    return a[2:, :-2], a[1:-1, 1:-1], a[:-2, 2:]          # "a": anti-diagonal


def spam686(img: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """SPAM-686. `mask` (optional, boolean of the image size): count only the triples (D1, D2, D3) whose D2 anchor pixel
    is in the mask. The anchor pixel of a difference D[i, j] is the first pixel in the definition x[i, j] - x[neighbour],
    before the direction is reversed. Without a mask the features equal the original SPAM-686. Warden W-A0 uses
    concat(spam686(y, A0), spam686(y, not A0))."""
    x = np.asarray(img, dtype=np.int32)
    H, W = x.shape
    pos = np.arange(H * W).reshape(H, W)
    flat = None if mask is None else np.asarray(mask, dtype=bool).reshape(-1)
    D = {   # differences in the 4 base directions and anchor pixel coordinates
        "R": (np.clip(x[:, :-1] - x[:, 1:], -T, T), pos[:, :-1], "h"),      # → : x[i,j] − x[i,j+1]
        "D": (np.clip(x[:-1, :] - x[1:, :], -T, T), pos[:-1, :], "v"),      # ↓
        "RD": (np.clip(x[:-1, :-1] - x[1:, 1:], -T, T), pos[:-1, :-1], "d"),  # ↘
        "RU": (np.clip(x[1:, :-1] - x[:-1, 1:], -T, T), pos[1:, :-1], "a"),   # ↗ : x[i+1,j] − x[i,j+1]
    }
    flips = {"h": (slice(None), slice(None, None, -1)), "v": (slice(None, None, -1), slice(None)),
             "d": (slice(None, None, -1), slice(None, None, -1)), "a": (slice(None, None, -1), slice(None, None, -1))}
    feats = {}
    for name, (d, p, kind) in D.items():
        for rev in (False, True):                      # reversed direction: negate and reverse the order
            dd, pp = (-d[flips[kind]], p[flips[kind]]) if rev else (d, p)
            trip = _triples(dd, kind)
            sel = None if flat is None else flat[_triples(pp, kind)[1]]
            feats[(name, rev)] = _markov2(dd, trip, sel)
    f1 = sum(feats[(k, r)] for k in ("R", "D") for r in (False, True)) / 4
    f2 = sum(feats[(k, r)] for k in ("RD", "RU") for r in (False, True)) / 4
    return np.concatenate([f1, f2]).astype(np.float32)


def spam_a0(img: np.ndarray, a0_mask: np.ndarray) -> np.ndarray:
    """Features of warden W-A0 (knows the public A0): SPAM computed separately on A0 and on the rest, 1372 dimensions."""
    m = np.asarray(a0_mask, dtype=bool)
    return np.concatenate([spam686(img, m), spam686(img, ~m)])
