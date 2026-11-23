"""Figures for Proposition 1: raw ML-KEM-768 ciphertexts are not pseudorandom.

(a) Analytic: the distribution P_10 of Compress_10 on U(Z_q); KL, Bhattacharyya coefficient BC, and the bounds
    1 - BC^768 <= TV <= sqrt(1 - BC^1536) for one ciphertext (the 768 coefficients of u).
(b) Empirical on REAL ciphertexts (OpenSSL ML-KEM-768): accuracy of the LLR test at N = 1, 2, 4, against uniform strings and
    against ciphertexts encoded with Kemeleon.
"""
from __future__ import annotations

import json
import math
import secrets
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from cryptography.hazmat.primitives.asymmetric import mlkem

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pqpks.analysis.dpub import mlkem_raw_llr  # noqa: E402
from pqpks.crypto import fips203, kemeleon  # noqa: E402

OUT = ROOT / "results" / "archive" / "pre-confirm" / "prop1"
Q = 3329


def analytic(d: int, k: int) -> dict:
    M = 1 << d
    c = Counter(fips203.compress(np.arange(Q), d).tolist())
    p = np.array([c[j] / Q for j in range(M)])
    kl = float((p * np.log(p * M)).sum())
    bc = float(np.sqrt(p / M).sum())
    tv_lo = 1 - bc ** k
    tv_hi = math.sqrt(1 - bc ** (2 * k))
    return {"d": d, "coeffs": k, "preimage_sizes": dict(Counter(c.values())), "KL_per_coef_nats": kl,
            "BC_per_coef": bc, "KL_total_nats": k * kl, "TV_lower": tv_lo, "TV_upper": tv_hi,
            "bayes_acc_range": [(1 + tv_lo) / 2, (1 + tv_hi) / 2]}


def main(n=1000):
    OUT.mkdir(parents=True, exist_ok=True)
    res = {"analytic_u": analytic(10, 768), "analytic_v": analytic(4, 256)}
    pk = mlkem.MLKEM768PrivateKey.generate().public_key()
    raw = [pk.encapsulate()[1] for _ in range(n)]
    llr_raw = np.array([mlkem_raw_llr(c) for c in raw])
    llr_unif = np.array([mlkem_raw_llr(secrets.token_bytes(1088)) for _ in range(n)])
    llr_kem = np.array([mlkem_raw_llr(kemeleon.encode_ct(c)) for c in raw[:300]])
    emp = {}
    for N in (1, 2, 4):
        a = llr_raw[: n - n % N].reshape(-1, N).sum(1)
        u = llr_unif[: n - n % N].reshape(-1, N).sum(1)
        emp[f"N={N}"] = float(((a > 0).mean() + (u <= 0).mean()) / 2)
    res["empirical_raw_vs_uniform"] = emp
    res["kemeleon_llr_positive_rate"] = float((llr_kem > 0).mean())
    res["uniform_llr_positive_rate"] = float((llr_unif > 0).mean())
    (OUT / "report.json").write_text(json.dumps(res, indent=2))
    u = res["analytic_u"]
    md = ["# Proposition 1: raw ML-KEM-768 ciphertexts are not pseudorandom", "",
          f"- Compress₁₀ on Z₃₃₂₉: preimage sizes {u['preimage_sizes']}",
          f"- KL = {u['KL_per_coef_nats']:.4e} nats/coefficient; total over 768 coefficients = {u['KL_total_nats']:.3f} nats",
          f"- BC = {u['BC_per_coef']:.8f}; 1 − BC^768 = {u['TV_lower']:.4f} ≤ TV ≤ {u['TV_upper']:.4f}",
          f"- ⇒ Bayes accuracy at N = 1 lies in [{u['bayes_acc_range'][0]:.4f}, {u['bayes_acc_range'][1]:.4f}]",
          f"- Part v (d_v = 4): TV ≤ {res['analytic_v']['TV_upper']:.4f} (practically indistinguishable)", "",
          "| Experiment (real ciphertexts, OpenSSL) | LLR accuracy |", "|---|---|"]
    for k, v in emp.items():
        md.append(f"| raw ML-KEM vs uniform strings, {k} | {v:.4f} |")
    md += ["", f"- Fraction LLR > 0: Kemeleon-encoded ciphertexts {res['kemeleon_llr_positive_rate']:.3f}; uniform strings {res['uniform_llr_positive_rate']:.3f}"]
    (OUT / "REPORT.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
