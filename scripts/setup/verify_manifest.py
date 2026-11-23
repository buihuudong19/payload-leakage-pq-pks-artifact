"""Check a folder of images against a SHA-256 manifest (lines "<sha256>  <file name>").

Run:  python scripts/setup/verify_manifest.py data/external/bossbase
      python scripts/setup/verify_manifest.py <folder> --manifest <file>

By default the manifest is <folder>/MANIFEST.sha256. Prints the number of matching, differing and missing files and
exits with status 1 if any file differs or is missing.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", type=Path)
    ap.add_argument("--manifest", type=Path, default=None)
    a = ap.parse_args()
    manifest = a.manifest or a.folder / "MANIFEST.sha256"
    ok, bad, missing = 0, [], []
    for line in manifest.read_text(encoding="ascii").splitlines():
        if not line.strip():
            continue
        digest, name = line.split(maxsplit=1)
        f = a.folder / name.strip().lstrip("*")
        if not f.is_file():
            missing.append(name)
        elif hashlib.sha256(f.read_bytes()).hexdigest() == digest.lower():
            ok += 1
        else:
            bad.append(name)
    print(f"{manifest}: {ok} match, {len(bad)} differ, {len(missing)} missing")
    for name in (bad + missing)[:20]:
        print("  ", name)
    sys.exit(1 if bad or missing else 0)


if __name__ == "__main__":
    main()
