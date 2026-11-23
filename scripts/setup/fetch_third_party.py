"""Download the third-party module used by baseline BL4 and check its SHA-256.

The module of [Zha25] (Public-key Discop) is not redistributed with this repository because its source repository has
no license file. This script downloads the exact file used in the article from the public repository, at a fixed commit,
and refuses to keep it if its SHA-256 differs from third_party/zha25_public_key_discop/SHA256SUMS.

Run:  python scripts/setup/fetch_third_party.py
"""
from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
COMMIT = "a03633c6fbd343852798a872a9b52e98f8984f8f"
URL = f"https://raw.githubusercontent.com/XinZhang1999/Public-key-Discop/{COMMIT}/src/PRNEncryption_P256.py"
DEST = ROOT / "third_party" / "zha25_public_key_discop" / "PRNEncryption_P256.py"
SUMS = DEST.with_name("SHA256SUMS")


def expected_sha256() -> str:
    for line in SUMS.read_text(encoding="ascii").splitlines():
        digest, name = line.split()
        if name == DEST.name:
            return digest
    raise SystemExit(f"{DEST.name} is not listed in {SUMS}")


def main() -> None:
    want = expected_sha256()
    if DEST.is_file() and hashlib.sha256(DEST.read_bytes()).hexdigest() == want:
        print(f"{DEST.relative_to(ROOT)} already present, SHA-256 matches")
        return
    with urllib.request.urlopen(URL, timeout=60) as r:
        data = r.read()
    got = hashlib.sha256(data).hexdigest()
    if got != want:
        sys.exit(f"SHA-256 mismatch for {URL}: {got} (expected {want}); file not written")
    DEST.write_bytes(data)
    print(f"wrote {DEST.relative_to(ROOT)} ({len(data)} bytes, SHA-256 {got})")


if __name__ == "__main__":
    main()
