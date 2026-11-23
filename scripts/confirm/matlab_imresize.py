"""MATLAB imresize (default: bicubic with antialiasing) for uint8 grayscale images, reimplemented with numpy.

Used to downsample BOSSbase and BOWS2 to 256x256 for SRNet G4, following the [BCF19] protocol (MATLAB "imresize with default
setting"). The algorithm follows MATLAB's imresize.m:
  - cubic kernel with a = -0.5; when downsampling (scale < 1) the kernel is stretched by 1/scale (antialiasing), width 4/scale;
  - coordinate u = x/scale + 0.5 (1 - 1/scale), x = 1..n_out; weights normalized to sum 1;
  - indices outside the image are mirrored symmetrically ([1..n, n..1]);
  - the dimension with the smaller scale goes first; with equal scales dimension 1 (rows) goes first;
  - for uint8 input MATLAB (imresizemex) rounds and clips to uint8 after EACH dimension, not only at the end.
Checked pixel by pixel against the Python version of fatheral/matlab_imresize (MIT), which its author reports as pixel-identical
to MATLAB R2015b for uint8 images; see tests/test_matlab_imresize.py.
"""
from __future__ import annotations

import math

import numpy as np


def _cubic(x: np.ndarray) -> np.ndarray:
    ax = np.abs(x)
    ax2, ax3 = ax * ax, ax * ax * ax
    return ((1.5 * ax3 - 2.5 * ax2 + 1) * (ax <= 1)
            + (-0.5 * ax3 + 2.5 * ax2 - 4 * ax + 2) * ((ax > 1) & (ax <= 2)))


def weight_matrix(n_in: int, n_out: int) -> np.ndarray:
    """(n_out, n_in) matrix of one dimension: row k holds the weights of output sample k over the input samples."""
    scale = n_out / n_in
    if scale < 1:
        width = 4.0 / scale
        h = lambda t: scale * _cubic(scale * t)          # noqa: E731
    else:
        width = 4.0
        h = _cubic
    x = np.arange(1, n_out + 1, dtype=np.float64)
    u = x / scale + 0.5 * (1 - 1 / scale)
    left = np.floor(u - width / 2)
    P = int(math.ceil(width)) + 2
    idx = left[:, None] + np.arange(P)[None, :]          # indices 1..n (may fall outside the image)
    w = h(u[:, None] - idx)
    w = w / w.sum(axis=1, keepdims=True)
    aux = np.r_[np.arange(n_in), np.arange(n_in - 1, -1, -1)]
    idx0 = aux[np.mod(idx.astype(np.int64) - 1, aux.size)]
    W = np.zeros((n_out, n_in))
    np.add.at(W, (np.repeat(np.arange(n_out), P), idx0.ravel()), w.ravel())
    return W


def _to_uint8(a: np.ndarray) -> np.ndarray:
    return np.floor(np.clip(a, 0, 255) + 0.5).astype(np.uint8)      # MATLAB uint8(): round half away from zero


def imresize_uint8(x: np.ndarray, out_shape: tuple[int, int], _cache: dict = {}) -> np.ndarray:  # noqa: B006
    """MATLAB imresize(x, out_shape) for a two-dimensional uint8 grayscale image."""
    x = np.asarray(x)
    if x.dtype != np.uint8 or x.ndim != 2:
        raise ValueError("only two-dimensional uint8 grayscale images are accepted")
    (h, w), (ho, wo) = x.shape, out_shape
    key = (h, w, ho, wo)
    if key not in _cache:
        _cache[key] = (weight_matrix(h, ho), weight_matrix(w, wo))
    Wr, Wc = _cache[key]
    order = (0, 1) if ho / h <= wo / w else (1, 0)
    y = x
    for d in order:
        y = _to_uint8(Wr @ y.astype(np.float64)) if d == 0 else _to_uint8(y.astype(np.float64) @ Wc.T)
    return y
