"""Cross-check matrix (screening before E2) run before the large experiments.

For each system: generate many kem_part strings (the part a warden can extract publicly), then apply:
  * specific distinguishers: LLR for raw ML-KEM, P-256 point check, fraction of ASCII bytes;
  * generic tests on all data concatenated: monobit, runs, byte-wise chi-square (alpha = 0.01).
In addition, for the STC systems: public extraction H0 LSB(y[A0]) on real stego images and on covers (the 8 sample covers)
to confirm that public extraction returns kem_part.
Expectation: D_pub at N = 1 ~ 1 (BL1, BL3), >= 0.89 (BL2, BL5), ~ 0.5 (BL4-E, PQ-PKS, REF-U).
"""
from __future__ import annotations

import json
import secrets
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pqpks import data, systems  # noqa: E402
from pqpks.analysis import dpub  # noqa: E402
from pqpks.params import DEFAULT_PP  # noqa: E402
from pqpks.systems.base import profile_id_for  # noqa: E402

OUT = ROOT / "results" / "archive" / "pre-confirm" / "sanity_matrix"
N_SAMPLES = {"BL1": 30, "BL2": 400, "BL3": 400, "BL4-E": 400, "BL5": 400, "BL5-L": 400, "PQ-PKS": 300,
             "PQ-PKS-R": 300, "REF-U": 400}
EXPECT = {"BL1": ("≈1", "fail"), "BL2": ("≥0.89", "pass"), "BL3": ("≈1", "fail"),
          "BL4-E": ("≈0.5", "pass"), "BL5": ("≥0.89", "pass"), "BL5-L": ("≥0.89", "pass"), "PQ-PKS-R": ("≈0.5", "pass"), "PQ-PKS": ("≈0.5", "pass"), "REF-U": ("≈0.5", "pass")}


def kem_parts(name: str, n: int) -> list[bytes]:
    if name == "REF-U":
        return [secrets.token_bytes(1536) for _ in range(n)]
    if name == "BL1":
        s = systems.make("BL1")
        _, pk = s.keygen()
        rng = np.random.default_rng(1)
        return [s.payload(pk, "".join(chr(int(c)) for c in rng.integers(32, 127, 200))).encode("latin-1")
                for _ in range(n)]
    crypto = systems.crypto_layer("BL5" if name == "BL2" else name)
    _, ek = crypto.keygen()
    pid = profile_id_for(crypto, DEFAULT_PP)
    return [crypto.start(ek, pid)[0] for _ in range(n)]


def distinguisher_accuracy(name: str, parts: list[bytes]) -> tuple[str, float]:
    """Accuracy of the most suitable specific distinguisher at N = 1 (control: uniform strings of the same length)."""
    unif = [secrets.token_bytes(len(p)) for p in parts]
    if name == "BL1":
        f = lambda b: dpub.ascii_digit_fraction(b) > 0.5          # noqa: E731
        test = "fraction of bytes in {digits, space, \\n} > 0.5"
    elif name == "BL3":
        f = dpub.p256_uncompressed_on_curve
        test = "the first 65 bytes are a valid P-256 point"
    elif len(parts[0]) >= 960:
        f = lambda b: dpub.mlkem_raw_llr(b) > 0                    # noqa: E731
        test = "LLR Compress₁₀ > 0"
    else:
        f = lambda b: dpub.ascii_digit_fraction(b) > 0.5 or dpub.p256_uncompressed_on_curve(b)  # noqa: E731
        test = "union of the tests above"
    acc = (sum(map(f, parts)) + sum(not f(u) for u in unif)) / (2 * len(parts))
    # For kem_part >= 960 B the LLR is applied to every system for a direct comparison
    return test, acc


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in ["BL1", "BL2", "BL3", "BL4-E", "BL5", "BL5-L", "PQ-PKS", "PQ-PKS-R", "REF-U"]:
        parts = kem_parts(name, N_SAMPLES[name])
        test, acc = distinguisher_accuracy(name, parts)
        llr_acc = None
        if len(parts[0]) >= 960:
            unif = [secrets.token_bytes(len(p)) for p in parts]
            llr_acc = (sum(dpub.mlkem_raw_llr(p) > 0 for p in parts) + sum(dpub.mlkem_raw_llr(u) <= 0 for u in unif)) / (2 * len(parts))
        blob = b"".join(parts)
        gen = dpub.generic_uniformity(blob)
        passed = all(v > 0.01 for v in gen.values())
        rows.append({"system": name, "kem_part_bytes": len(parts[0]), "samples": len(parts), "test": test,
                     "dpub_acc_N1": round(acc, 4), "llr_acc_N1": None if llr_acc is None else round(llr_acc, 4),
                     "generic": {k: float(f"{v:.3g}") for k, v in gen.items()}, "generic_pass": passed,
                     "expected_dpub": EXPECT[name][0], "expected_generic": EXPECT[name][1]})
        print(rows[-1], flush=True)

    # Public extraction on real images for the STC systems
    pool = data.cover_pool()
    extraction = []
    for name in ["BL3", "BL4-E", "BL5", "BL5-L", "PQ-PKS", "PQ-PKS-R"]:
        s = systems.make(name)
        dk, ek = s.keygen()
        ok = 0
        for cname, cover in pool.items():
            sess = s.start_session(ek, length=1)
            stego = s.send(sess, b"sanity", cover)
            body_len = s.pp.buckets[0] + 16              # b"sanity" fits the smallest bucket (A8: A0 depends on the bucket)
            ok += s.embedder.extract_kem(stego, s.crypto.kem_part_bytes, body_len) == sess.kem_part
        extraction.append({"system": name, "covers": len(pool), "public_extract_equals_kem_part": ok})
        print(extraction[-1], flush=True)

    (OUT / "report.json").write_text(json.dumps({"rows": rows, "extraction": extraction}, indent=2, ensure_ascii=False))
    md = ["# Cross-check matrix (screening before E2)", "",
          "| System | |kem_part| (B) | Samples | Specific distinguisher | D_pub N=1 | Expected | LLR N=1 | monobit p | runs p | byte χ² p | Generic tests | Expected |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        g = r["generic"]
        md.append(f"| {r['system']} | {r['kem_part_bytes']} | {r['samples']} | {r['test']} | {r['dpub_acc_N1']} | {r['expected_dpub']} | "
                  f"{r['llr_acc_N1'] if r['llr_acc_N1'] is not None else '–'} | {g['monobit_p']} | {g['runs_p']} | {g['byte_chi2_p']} | "
                  f"{'pass' if r['generic_pass'] else 'fail'} | {r['expected_generic']} |")
    md += ["", "## Public extraction on real stego images (8 sample covers)", "", "| System | Covers | Public extraction = kem_part |", "|---|---|---|"]
    for e in extraction:
        md.append(f"| {e['system']} | {e['covers']} | {e['public_extract_equals_kem_part']}/{e['covers']} |")
    md += ["", "_BL1: kem_part is the whole decimal ASCII string embedded by the BL1 reference implementation (ciphertext + header)._",
           "_The generic tests here are a screening (3 tests); the full NIST SP 800-22 belongs to E1, E2._"]
    (OUT / "REPORT.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
