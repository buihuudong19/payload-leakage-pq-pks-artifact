"""Select the ALASKA2 images for E1-bis: 40 000 images.

The set contains the 10 000 E7 images (data/external/alaska2/subset_ids.txt, unchanged) and 30 000 additional images. The
additional images are stratified by camera model, proportional to the number of REMAINING images (not in E7) of each camera,
with largest-remainder rounding, drawn at random without replacement with seed 20261107. The groups *_FLATS and *_TEXTURES are
excluded as in E7. make_alaska2_subset.py (E7) is not modified.

Output: data/external/alaska2/e1bis_ids.txt (one line per image: 5-digit index, camera name; sorted by index); prints its SHA-256.
    python scripts/data/make_alaska2_e1bis_subset.py
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DIR = ROOT / "data" / "external" / "alaska2"
SRC, E7, OUT = DIR / "ALASKAv2Base_by_DSC.txt", DIR / "subset_ids.txt", DIR / "e1bis_ids.txt"
TOTAL, SEED = 40_000, 20261107


def groups() -> list[tuple[int, int, str]]:
    g = []
    for line in SRC.read_text().splitlines()[1:]:
        m = re.match(r"\s*\d+ :: (\d+) --> (\d+)\s+\./(\S+)", line)
        if m and not re.search(r"_(FLATS|TEXTURES)$", m.group(3)):
            g.append((int(m.group(1)), int(m.group(2)), m.group(3)))
    return g


def main():
    G = groups()
    e7 = {}
    for line in E7.read_text().splitlines():
        if line.strip():
            i, cam = line.split()
            e7[int(i)] = cam
    rest = []
    for a, b, cam in G:
        rest.append((cam, np.array([i for i in range(a, b + 1) if i not in e7], dtype=np.int64)))
    extra = TOTAL - len(e7)
    sizes = np.array([len(r) for _, r in rest])
    quota = sizes / sizes.sum() * extra
    take = np.floor(quota).astype(int)
    for k in np.argsort(-(quota - take), kind="stable")[: extra - take.sum()]:
        take[k] += 1
    assert take.sum() == extra and np.all(take <= sizes)
    rng = np.random.default_rng(SEED)
    rows = dict(e7)
    for (cam, pool), k in zip(rest, take):
        for i in rng.choice(pool, size=k, replace=False):
            rows[int(i)] = cam
    assert len(rows) == TOTAL
    txt = "".join(f"{i:05d} {rows[i]}\n" for i in sorted(rows))
    OUT.write_bytes(txt.encode("ascii"))          # bytes: the same SHA on Windows and Linux
    print(f"{len(rows)} images ({len(e7)} from E7 + {extra} additional) from {len(G)} cameras; {sizes.sum() + len(e7)} eligible images")
    print("SHA-256", hashlib.sha256(txt.encode()).hexdigest(), "→", OUT)


if __name__ == "__main__":
    main()
