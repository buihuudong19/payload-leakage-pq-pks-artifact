"""Diagnosis of the bit position rejected by Holm in the screening E1 (paper, Section 7.6, Table 9): rejection-sampled variant
(1252 B), bucket 256, bit 3197 of the public syndrome Ext0(cover).
Writes results/archive/pre-confirm/screen_v1/e1_flag_k1252_b256.json:
Ghi ra results/archive/pre-confirm/screen_v1/e1_flag_k1252_b256.json:
  - bit frequency by source (pilot BOSSbase, screening BOWS2);
  - the number of A0 pixels contributing to that bit (the corresponding row of H0) and the bit frequency by the number of
    saturated pixels (0 or 255) among them, to test the "saturated pixels" hypothesis;
  - control: another row (5000) of the same configuration.
Uses no image outside the 2000 screening images. This position is pre-registered as a SINGLE separate test for the confirmatory E1.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from pqpks import data  # noqa: E402
from pqpks.params import AEAD_TAG_BYTES, DEFAULT_PP  # noqa: E402
from pqpks.stego import stc  # noqa: E402
from pqpks.stego.two_stage import TwoStageSTC, _h0_seed  # noqa: E402

KB, BUCKET, ROW, CONTROL = 1252, 256, 3197, 5000


def row_pixels(i, a0, widths, cols, h):
    start = np.r_[0, np.cumsum(widths)]
    return np.array([a0[start[k] + j] for k in range(max(0, i - h + 1), i + 1)
                     for j in range(widths[k]) if (cols[j] >> (i - k)) & 1])


def main(raw=ROOT / "results" / "archive" / "pre-confirm" / "screen_v1" / "raw"):
    emb = TwoStageSTC(DEFAULT_PP)
    shape, h, m = (512, 512), DEFAULT_PP.stc_h, 8 * KB
    a0, _ = emb._sets(shape, m, 8 * (BUCKET + AEAD_TAG_BYTES))
    widths = stc.block_widths(a0.size, m)
    cols = stc.submatrix(_h0_seed(*shape), h, stc.width_for(a0.size, m))
    P = row_pixels(ROW, a0, widths, cols, h)
    Q = row_pixels(CONTROL, a0, widths, cols, h)
    bits, ctrl, sat, src = [], [], [], []
    for f in sorted(Path(raw).glob("*.npz")):
        z = np.load(f)
        x = data.load_cover(str(z["src"]), int(z["id"])).reshape(-1)
        v = x[P]
        bits.append(int((v & 1).sum() & 1))
        ctrl.append(int((x[Q] & 1).sum() & 1))
        sat.append(int(((v == 0) | (v == 255)).sum()))
        src.append(str(z["src"]))
        # cross-check with the stored syndromes
        assert bits[-1] == int(np.unpackbits(z[f"e1_k{KB}_b{BUCKET}"])[ROW])
    bits, sat, src = np.array(bits), np.array(sat), np.array(src)
    n = bits.size
    res = {"variant_bytes": KB, "bucket": BUCKET, "row": ROW, "pixels_in_row": int(P.size), "n": n,
           "freq": float(bits.mean()), "z": float((bits.sum() - n / 2) / np.sqrt(n / 4)),
           "freq_by_source": {s: float(bits[src == s].mean()) for s in np.unique(src)},
           "n_by_source": {s: int((src == s).sum()) for s in np.unique(src)},
           "control_row": CONTROL, "control_freq": float(np.mean(ctrl)),
           "by_saturated_pixels": {}}
    for lo, hi in ((0, 0), (1, 5), (6, 50), (51, P.size)):
        mm = (sat >= lo) & (sat <= hi)
        res["by_saturated_pixels"][f"{lo}-{hi}"] = {"n": int(mm.sum()), "freq": float(bits[mm].mean()) if mm.any() else None}
    out = ROOT / "results" / "archive" / "pre-confirm" / "screen_v1" / f"e1_flag_k{KB}_b{BUCKET}.json"
    out.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
