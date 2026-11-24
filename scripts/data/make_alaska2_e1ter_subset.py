"""Select the ALASKA2 images for E1-ter (registered after review, 24 Nov 2026; registrations/E1TER_PLAN.md).

E1-ter uses every ALASKA2 image that no earlier step has used: all indices of ALASKAv2Base_by_DSC.txt except the groups
*_FLATS and *_TEXTURES (excluded as in E7 and E1-bis) and except the 40 000 images of E1-bis (data/external/alaska2/e1bis_ids.txt,
which contains the 10 000 images of E7). No sampling is involved, so the list is determined by these two files.

Output: data/external/alaska2/e1ter_ids.txt (one line per image: 5-digit index, camera name; sorted by index); prints its SHA-256.
    python scripts/data/make_alaska2_e1ter_subset.py
"""
from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DIR = ROOT / "data" / "external" / "alaska2"


def groups(src: Path) -> list[tuple[int, int, str]]:
    g = []
    for line in src.read_text().splitlines()[1:]:
        m = re.match(r"\s*\d+ :: (\d+) --> (\d+)\s+\./(\S+)", line)
        if m and not re.search(r"_(FLATS|TEXTURES)$", m.group(3)):
            g.append((int(m.group(1)), int(m.group(2)), m.group(3)))
    return g


def select(src: Path, used: list[Path]) -> list[tuple[int, str]]:
    taken = set()
    for u in used:
        taken |= {int(ln.split()[0]) for ln in u.read_text().splitlines() if ln.strip()}
    return [(i, cam) for a, b, cam in groups(src) for i in range(a, b + 1) if i not in taken]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(DIR / "ALASKAv2Base_by_DSC.txt"))
    ap.add_argument("--used", nargs="+", default=[str(DIR / "e1bis_ids.txt"), str(DIR / "subset_ids.txt")])
    ap.add_argument("--out", default=str(DIR / "e1ter_ids.txt"))
    a = ap.parse_args()
    rows = select(Path(a.src), [Path(u) for u in a.used])
    out = Path(a.out)
    out.write_bytes("".join(f"{i:05d} {cam}\n" for i, cam in rows).encode("ascii"))
    cams = {}
    for _, c in rows:
        cams[c] = cams.get(c, 0) + 1
    print(f"{len(rows)} images, {len(cams)} camera groups; SHA-256 {hashlib.sha256(out.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
