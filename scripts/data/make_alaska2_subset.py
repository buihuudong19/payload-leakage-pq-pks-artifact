"""Select the ALASKA2 subset for E7: 10 000 images, sampled stratified by camera model.

ALASKA2 images are numbered consecutively BY CAMERA MODEL (ALASKAv2Base_by_DSC.txt), so the first 10 000 images would cover only
a few cameras. Here each camera contributes a number of images proportional to its size (largest-remainder rounding), drawn at
random with a fixed seed for reproducibility.
The groups *_FLATS and *_TEXTURES are excluded: they are calibration images (flat fields, textures), not natural images.
The index range is taken from the column "first --> last" (the count column at the start of a line sometimes disagrees with the range).

Output: data/external/alaska2/subset_ids.txt (one line per image: 5-digit index, camera name).
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "data" / "external" / "alaska2" / "ALASKAv2Base_by_DSC.txt"
OUT = ROOT / "data" / "external" / "alaska2" / "subset_ids.txt"
TOTAL, SEED = 10_000, 20261029


def main():
    groups = []
    for line in SRC.read_text().splitlines()[1:]:
        m = re.match(r"\s*\d+ :: (\d+) --> (\d+)\s+\./(\S+)", line)
        if m and not re.search(r"_(FLATS|TEXTURES)$", m.group(3)):
            groups.append((int(m.group(1)), int(m.group(2)), m.group(3)))
    sizes = np.array([b - a + 1 for a, b, _ in groups])
    quota = sizes / sizes.sum() * TOTAL
    take = np.floor(quota).astype(int)
    for i in np.argsort(-(quota - take))[: TOTAL - take.sum()]:
        take[i] += 1
    rng = np.random.default_rng(SEED)
    rows = []
    for (a, b, cam), k in zip(groups, take):
        for idx in sorted(rng.choice(np.arange(a, b + 1), size=k, replace=False)):
            rows.append(f"{idx:05d} {cam}")
    OUT.write_text("\n".join(rows) + "\n")
    print(f"{len(rows)} images from {len(groups)} camera groups ({sizes.sum()} eligible images in total) -> {OUT}")


if __name__ == "__main__":
    main()
