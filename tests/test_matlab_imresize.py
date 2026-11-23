"""MATLAB-style imresize (scripts/confirm/matlab_imresize.py) matches a direct transcription of imresize.m.

The reference below is written with loops and follows each step of imresize.m / imresizemex (each output sample is
computed separately, taps are summed in order, and the result is rounded to uint8 after each dimension). The version in
scripts uses matrix products; the two must agree pixel by pixel.
"""
import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("matlab_imresize", ROOT / "scripts" / "confirm" / "matlab_imresize.py")
mi = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mi)


def _cubic(t):
    a = abs(t)
    if a <= 1:
        return 1.5 * a ** 3 - 2.5 * a ** 2 + 1
    if a <= 2:
        return -0.5 * a ** 3 + 2.5 * a ** 2 - 4 * a + 2
    return 0.0


def _contrib(n_in, n_out):
    scale = n_out / n_in
    width = 4.0 / scale if scale < 1 else 4.0
    rows = []
    for x in range(1, n_out + 1):
        u = x / scale + 0.5 * (1 - 1 / scale)
        left = math.floor(u - width / 2)
        taps = []
        for j in range(math.ceil(width) + 2):
            k = left + j                                  # index 1..n, may fall outside the image
            w = scale * _cubic(scale * (u - k)) if scale < 1 else _cubic(u - k)
            m = (k - 1) % (2 * n_in)                      # reflection [1..n, n..1]
            taps.append((m if m < n_in else 2 * n_in - 1 - m, w))
        s = sum(w for _, w in taps)
        rows.append([(i, w / s) for i, w in taps])
    return rows


def _round_u8(v):
    return int(math.floor(min(max(v, 0.0), 255.0) + 0.5))


def _ref(x, out):
    h, w = x.shape
    ho, wo = out
    y = x.astype(np.int64)
    dims = [0, 1] if ho / h <= wo / w else [1, 0]
    for d in dims:
        if d == 0:
            c = _contrib(y.shape[0], ho)
            z = np.zeros((ho, y.shape[1]), np.int64)
            for r, taps in enumerate(c):
                for col in range(y.shape[1]):
                    z[r, col] = _round_u8(sum(wt * y[i, col] for i, wt in taps))
        else:
            c = _contrib(y.shape[1], wo)
            z = np.zeros((y.shape[0], wo), np.int64)
            for r in range(y.shape[0]):
                for q, taps in enumerate(c):
                    z[r, q] = _round_u8(sum(wt * y[r, i] for i, wt in taps))
        y = z
    return y.astype(np.uint8)


@pytest.mark.parametrize("shape, out", [((64, 64), (32, 32)), ((37, 53), (17, 29)), ((40, 30), (25, 12)),
                                        ((24, 24), (45, 31)), ((50, 80), (50, 40))])
def test_matches_loop_reference(shape, out):
    rng = np.random.default_rng(sum(shape) + sum(out))
    x = rng.integers(0, 256, shape, dtype=np.uint8)
    assert np.array_equal(mi.imresize_uint8(x, out), _ref(x, out))


def test_smooth_image_and_constant():
    yy, xx = np.mgrid[0:64, 0:64]
    x = (127 + 100 * np.sin(xx / 7.0) * np.cos(yy / 5.0)).astype(np.uint8)
    assert np.array_equal(mi.imresize_uint8(x, (32, 32)), _ref(x, (32, 32)))
    c = np.full((512, 512), 173, np.uint8)
    assert np.array_equal(mi.imresize_uint8(c, (256, 256)), np.full((256, 256), 173, np.uint8))


def test_half_scale_weights():
    W = mi.weight_matrix(512, 256)
    assert W.shape == (256, 512)
    assert np.allclose(W.sum(axis=1), 1.0)
    # an output sample in the middle of the image uses exactly 8 input samples (width 4/scale = 8), symmetric around u = 2x - 0.5
    nz = np.nonzero(W[100])[0]
    assert nz.size == 8 and np.allclose(W[100, nz], W[100, nz][::-1])


def test_rejects_non_uint8():
    with pytest.raises(ValueError):
        mi.imresize_uint8(np.zeros((8, 8), np.float32), (4, 4))
