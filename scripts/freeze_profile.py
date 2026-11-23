"""Freeze the profile before the confirmatory step.

Writes configs/profile_frozen_v1.json: every default public parameter pp, the profile_id (hex) of each system, decision D-1,
the bucket ladder, the signature placement rule, library versions, and the SHA-256 of every source file (src/, scripts/,
tests/, configs/ except this file itself). The confirmatory scripts read this file and refuse to run if the source has
changed (checked with --check).

The released repository keeps the original configs/profile_frozen_v1.json byte for byte. Files of src/ and configs/ whose
comments, docstrings or messages were edited for the release are listed with both hashes in
provenance/release_equivalence_v1.json (see docs/provenance.md); --check accepts a file whose hash matches either entry.

Run:   python scripts/freeze_profile.py --check    # check that the current source matches the frozen profile
       python scripts/freeze_profile.py --write    # write a new frozen profile (overwrites configs/profile_frozen_v1.json)
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pqpks import systems  # noqa: E402
from pqpks.params import DEFAULT_PP  # noqa: E402

OUT = ROOT / "configs" / "profile_frozen_v1.json"
SYSTEMS = ("PQ-PKS", "PQ-PKS-R", "BL3", "BL4-E", "BL5", "BL5-L")
EQUIV = ROOT / "provenance" / "release_equivalence_v1.json"


def _generated(f: Path) -> bool:
    """Files written by tools, not source: bytecode and the `*.egg-info` metadata that `pip/uv install -e` writes into src/."""
    return f.suffix == ".pyc" or any(p == "__pycache__" or p.endswith(".egg-info") for p in f.parts)


def accepted_hashes(path: str, frozen_sha: str | None) -> set:
    """Hashes accepted for `path`: the frozen hash, plus the hash of the released file when the release equivalence map
    links it to that frozen hash."""
    ok = {frozen_sha}
    if frozen_sha is not None and EQUIV.exists():
        e = json.loads(EQUIV.read_text(encoding="utf-8"))["files"].get(path)
        if e and e["frozen_sha256"] == frozen_sha:
            ok.add(e["release_sha256"])
    return ok


def source_hashes() -> dict[str, str]:
    out = {}
    for d in ("src", "scripts", "tests", "configs"):
        for f in sorted((ROOT / d).rglob("*")):
            if f.is_file() and f != OUT and not _generated(f.relative_to(ROOT)):
                out[f.relative_to(ROOT).as_posix()] = hashlib.sha256(f.read_bytes()).hexdigest()
    return out


def build() -> dict:
    import numba
    import numpy as np
    pp = dataclasses.asdict(DEFAULT_PP)
    pp["context"] = DEFAULT_PP.context.decode()
    pids = {}
    for name in SYSTEMS:
        s = systems.make(name)
        pids[name] = s.pid.hex()
    # The three PQ-PKS configurations that the A5b rule and the signature selection rule can select; their pid is recorded before the confirmatory alpha* is known
    candidates = {}
    for label, alg, place in (("ML-DSA-65/image1 (default)", "ML-DSA-65", "image1"),
                              ("ML-DSA-65/image2 (A5b)", "ML-DSA-65", "image2"),
                              ("ML-DSA-44/image2 (A5b)", "ML-DSA-44", "image2")):
        cpp = dataclasses.replace(DEFAULT_PP, sig_alg=alg, sig_placement=place)
        candidates[label] = {"sig_alg": alg, "sig_placement": place, "pid": systems.make("PQ-PKS", cpp).pid.hex()}
    return {
        "version": "profile_frozen_v1",
        "date": "2026-11-05",
        "decisions": {
            "D-1": "Kemeleon main variant (1536 B) is the default; the rejection-sampled variant (1252 B) is ablation A4",
            "ladder": "merged: first images use B in {256,...,8192}; later images use B and B + |c_hat| (11 buckets, E-28)",
            "sig_placement": "image1 now; switch to image2 (A5b) if the confirmatory alpha* < 0.1724 "
                             "(alpha_min of a signed first image under this ladder, results/e5_budget/e5_ladder.json)",
            "FN-DSA-512": "analytic only (E5); not implemented",
            "signature_rule": "registered 2026-11-06, before any confirmatory E2/E3 data (errata E-30): after the A5b rule, "
                              "recommend the vetted scheme of highest security level whose alpha_min (merged ladder, chosen "
                              "sigma placement; results/e5_budget/e5_ladder.json) is <= the confirmatory alpha*: ML-DSA-65 "
                              "(0.1724 image 1; 0.1099 image 2), else ML-DSA-44 (0.0787 image 2), else none (report as a "
                              "finding; FN-DSA-512, 0.0552 image 2, analytic only). E5 reports all three schemes. Applied by "
                              "scripts/confirm/apply_rules.py",
        },
        "candidate_profiles_PQ-PKS": candidates,
        "public_params": pp,
        "profile_ids": pids,
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "numba": numba.__version__},
        "source_sha256": source_hashes(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    if not (a.check or a.write):
        ap.error("use --check, or --write to overwrite configs/profile_frozen_v1.json")
    if a.check:
        frozen = json.loads(OUT.read_text(encoding="utf-8"))
        cur = source_hashes()
        diff = sorted(k for k in set(cur) | set(frozen["source_sha256"])
                      if cur.get(k) not in accepted_hashes(k, frozen["source_sha256"].get(k))
                      and (k.startswith("src/") or (k.startswith("configs/") and not k.startswith("configs/splits/"))))
        print("profile_ids match:", all(systems.make(n).pid.hex() == frozen["profile_ids"][n] for n in SYSTEMS))
        print("files of src/ or configs/ that differ from the frozen profile:", diff or "none")
        sys.exit(1 if diff else 0)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(build(), indent=1, ensure_ascii=False), encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
