"""Check a local copy of the BL1 reference modules against the SHA-256 published in baselines/bl1_reference.SHA256SUMS.

The BL1 reference implementation is not redistributed (docs/baselines.md). Whoever holds a copy can check that it is the
code used in the pilot rounds:
    python scripts/setup/check_bl1_reference.py [folder]      # default: baselines/bl1_reference or $PQPKS_BL1_REFERENCE_DIR
"""
from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SUMS = ROOT / "baselines" / "bl1_reference.SHA256SUMS"


def check(folder: Path) -> list[str]:
    bad = []
    for line in SUMS.read_text().splitlines():
        if not line.strip():
            continue
        h, name = line.split()
        f = folder / name
        if not f.exists():
            bad.append(f"{name}: missing")
        elif hashlib.sha256(f.read_bytes()).hexdigest() != h:
            bad.append(f"{name}: SHA-256 differs")
    return bad


def main():
    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(os.environ.get("PQPKS_BL1_REFERENCE_DIR", ROOT / "baselines" / "bl1_reference"))
    bad = check(folder)
    print("BL1 reference modules match the published SHA-256" if not bad else "\n".join(bad))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
