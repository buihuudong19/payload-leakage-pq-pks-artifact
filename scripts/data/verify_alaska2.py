"""Check the downloaded ALASKA2 subset and convert it to PGM.

For each index in data/external/alaska2/subset_ids.txt:
  * read data/external/raw/alaska2_tiff512/<id>.tif with PIL; it must be an 8-bit 512x512 grayscale image;
  * write data/external/alaska2/pgm/<id>.pgm (P5, pixel values unchanged);
  * record the SHA-256 of the original TIFF file in data/external/alaska2/MANIFEST_tiff.sha256.
Can be rerun (converted images are skipped); --budget limits the time of one run (seconds).
"""
from __future__ import annotations

import argparse
import hashlib
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
LIST = ROOT / "data" / "external" / "alaska2" / "subset_ids.txt"
RAW = ROOT / "data" / "external" / "raw" / "alaska2_tiff512"
PGM = ROOT / "data" / "external" / "alaska2" / "pgm"
MAN = ROOT / "data" / "external" / "alaska2" / "MANIFEST_tiff.sha256"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=float, default=150.0)
    ap.add_argument("--list", default=str(LIST), help="image list (default: E7; E1-bis: data/external/alaska2/e1bis_ids.txt)")
    a = ap.parse_args()
    t0 = time.time()
    PGM.mkdir(parents=True, exist_ok=True)
    ids = [ln.split()[0] for ln in Path(a.list).read_text().splitlines() if ln.strip()]
    done = {ln.split()[1][:5] for ln in MAN.read_text().splitlines()} if MAN.exists() else set()
    missing, bad, new = [], [], 0
    with MAN.open("a") as man:
        for i in ids:
            if i in done:
                continue
            f = RAW / f"{i}.tif"
            if not f.exists():
                missing.append(i)
                continue
            raw = f.read_bytes()
            im = Image.open(f)
            arr = np.asarray(im)
            if im.mode != "L" or arr.shape != (512, 512) or arr.dtype != np.uint8:
                bad.append((i, im.mode, arr.shape))
                continue
            (PGM / f"{i}.pgm").write_bytes(b"P5\n512 512\n255\n" + arr.tobytes())
            man.write(f"{hashlib.sha256(raw).hexdigest()}  {i}.tif\n")
            new += 1
            if time.time() - t0 > a.budget:
                break
    total = sum(1 for i in ids if i in done) + new
    print(f"checked {total}/{len(ids)}; new {new}; missing {len(missing)}; wrong format {len(bad)} {bad[:3]}")
    if missing or bad or total < len(ids):
        raise SystemExit(1)                 # incomplete: run_all marks the step failed; download the rest and rerun


if __name__ == "__main__":
    main()
