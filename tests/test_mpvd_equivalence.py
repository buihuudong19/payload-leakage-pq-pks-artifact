"""Bit-exact check of the single-channel MPVD port against the BL1 reference implementation.

Method: run the reference Embed.mainEmbed on an RGB image (a crop of a synthetic test image) with a payload small enough to use
only the R channel; run the port on the R channel with the same bit string (the "len " header followed by the data, as in
mainEmbed). The R channel is compared pixel by pixel. All 4 size cases of mainEmbed are covered. The images are the
synthetic test images of pqpks.data; the comparison runs only where the BL1 reference code is available.
"""
import random

import numpy as np
import pytest

from conftest import load_bl1_reference_modules
from pqpks import data
from pqpks.stego import mpvd

SIZES = [(510, 510), (512, 512), (511, 511), (510, 512), (512, 510), (511, 512), (512, 511),
         (508, 508), (509, 509), (508, 509), (509, 508), (500, 500), (505, 507), (507, 505), (506, 509)]


@pytest.mark.parametrize("hw", SIZES)
@pytest.mark.parametrize("seed", [1, 2])
def test_port_matches_original(hw, seed):
    Embed, TextToBin = load_bl1_reference_modules()
    h, w = hw
    rgb = data.synthetic_rgb(seed).crop((0, 0, w, h))
    rng = random.Random(h * 1000 + w)
    text = "".join(chr(rng.randrange(32, 127)) for _ in range(rng.choice([300, 3000, 12000])))
    orig = rgb.copy()
    Embed.mainEmbed(orig, text)
    assert Embed.ReturnComplete() == 1
    r_orig = np.asarray(orig)[:, :, 0]
    bits = TextToBin.Text(str(len(text)) + " " + text)
    r_port = mpvd.embed(np.asarray(rgb)[:, :, 0], bits)
    # The payload must fit entirely in the R channel for the comparison to be meaningful.
    assert np.array_equal(np.asarray(orig)[:, :, 1], np.asarray(rgb)[:, :, 1])
    assert np.array_equal(r_port, r_orig)
    assert mpvd.extract(r_port, len(bits)) == bits


def test_port_roundtrip_random_bits():
    x = data.cover_pool()["T1"]
    for n in (100, 5000, 200000):
        bits = "".join(random.choice("01") for _ in range(n))
        assert mpvd.extract(mpvd.embed(x, bits), n) == bits
