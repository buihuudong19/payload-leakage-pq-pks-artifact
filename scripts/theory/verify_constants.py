"""Recompute every constant used in Sections 3-6 of the PQ-PKS paper (supplementary material). Reads no data, uses no randomness.

    python scripts/theory/verify_constants.py

Exact integer/rational arithmetic where possible; the TV of Proposition 2 is computed exactly via the binomial sufficient statistic.
"""
from __future__ import annotations

from collections import Counter
from fractions import Fraction as F
from math import exp, floor, lgamma, log, log2, sqrt

q = 3329


def compress(x: int, d: int) -> int:
    """FIPS 203: Compress_d(x) = round((2^d / q) x) mod 2^d, rounding half up."""
    return floor(F(2 ** d, q) * x + F(1, 2)) % (2 ** d)


def cells(d: int) -> list[int]:
    c = [0] * (2 ** d)
    for x in range(q):
        c[compress(x, d)] += 1
    return c


def binom_tv(n: int, p: float, u: float) -> float:
    lp = lambda k, a: lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1) + k * log(a) + (n - k) * log(1 - a)
    return 0.5 * sum(abs(exp(lp(k, p)) - exp(lp(k, u))) for k in range(n + 1))


def main() -> None:
    print("== Section 3.2: number of preimages of Compress_d ==")
    for d in (1, 4, 5, 10, 11):
        c = cells(d)
        print(f"d={d:2d}: {sorted(Counter(c).items())}, cell 0 has {c[0]}")

    print("\n== Proposition 2 ==")
    c10 = cells(10)
    bc = sum(sqrt(x / q / 1024) for x in c10)
    kl = sum(x / q * log(x / q * 1024) for x in c10)
    p4, u4 = 1028 / q, 257 / 1024
    print(f"BC10 = {bc:.8f}; 1 - BC^768 = {1 - bc ** 768:.6f}; KL = {kl:.6e} nat/coefficient, {768 * kl:.4f} nat/ciphertext")
    for M in (1, 2, 4, 8):
        tv = binom_tv(768 * M, p4, u4)
        print(f"M={M}: exact TV = {tv:.6f}, Bayes accuracy = {(1 + tv) / 2:.6f}")
    print(f"Hellinger interval for the accuracy (M=1): [{(1 + 1 - bc ** 768) / 2:.4f}, {(1 + sqrt(1 - bc ** 1536)) / 2:.4f}]")
    c4 = cells(4)
    tv4 = binom_tv(256, max(c4) / q, 1 / 16)
    print(f"v part (d_v = 4, 256 coefficients): TV = {tv4:.4f}, accuracy = {(1 + tv4) / 2:.4f}")

    print("\n== Lemma 2M: epsilon_VE (exact integers) ==")
    Q, L = q ** 256, 3072
    M, R0 = divmod(2 ** L, Q)
    eps = F(R0 * (Q - R0), Q * 2 ** L)
    le = log2(eps.numerator) - log2(eps.denominator)
    print(f"log2 eps_VE = {le:.4f}; log2 4 eps_VE = {le + 2:.4f}; log2(2^L/Q) = {L - 256 * log2(q):.3f}")

    print("\n== Lemma 2R ==")
    lg = 768 * log2(q)
    b = 8986
    pacc = 2 ** (b - lg) * (1 - 1 / q) ** 256
    print(f"log2 q^768 = {lg:.4f}; p_acc = {pacc:.6f}; 1/p_acc = {1 / pacc:.6f}; log2 (1-p_acc)^64 = {64 * log2(1 - pacc):.2f}")

    print("\n== Section 6.4: min-entropy of the decoding candidate from a uniform syndrome ==")
    print(f"H_inf >= 768 log2(q/4) + 256 log2(q/209) = {768 * log2(q / 4) + 256 * log2(q / 209):.2f} bit")

    print("\n== Lemma S-A.7: the terms of [Xag22, Theorem 4.1] for ML-KEM-768 ==")
    hmax = 768 * log2(q / 4) + 256 * log2(q / 209)          # -log2 of the largest probability of S_$compr (exact, without subtracting 1)
    delta = -164.8
    qq = 64
    print(f"-log2 max Pr[S_$compr] = {hmax:.2f}; log2 Disj <= 256 - {hmax:.2f} = {256 - hmax:.2f}")
    print(f"log2 4δ = {log2(4) + delta:.2f}; log2 4(q_J+q_D)2^-128 (q=2^{qq}) = {log2(4) + qq + 1 - 128:.2f}; "
          f"log2 16(q_G+q_D+2)^2 δ = {log2(16) + 2 * log2(2 ** qq + 2 ** qq + 2) + delta:.2f}; "
          f"log2 16(2q_G+2)^2 δ = {log2(16) + 2 * log2(2 ** (qq + 1) + 2) + delta:.2f}")

    print("\n== Section 5.4: number of trials ==")
    print(f"N_T (two ladders) = {6 + 15 * 16 * 11}; (one ladder) = {6 + 15 * 16 * 6}; log2(2646 * 2^-128) = {log2(2646) - 128:.2f}")

    print("\n== Table 5 (n = 262 144, main Kemeleon variant) ==")
    n, lc = 262144, 1536
    B1 = [256, 512, 1024, 2048, 4096, 8192]
    B2 = sorted(set(B1) | {x + lc for x in B1})
    print("ladder of later images:", B2, len(B2))
    for m, sig in [(64, 0), (256, 0), (256, 3309), (256, 2420), (1024, 0), (4096, 0)]:
        need = 14 + m + sig
        B = min(x for x in B1 if x >= need)
        tot = lc + B + 16
        need2 = 6 + m
        B_2 = min(x for x in B2 if x >= need2)
        print(f"|m|={m:5d} signature={sig:4d}: first image {tot} B, alpha {tot * 8 / n:.4f}, padding {B - need} B; later image {B_2 + 16} B, alpha {(B_2 + 16) * 8 / n:.4f}")
    print(f"rejection-sampled variant, smallest first image: alpha = {(1252 + 528) * 8 / n:.4f}; smallest 256x256 image: alpha = {1808 * 8 / 65536:.4f}")


if __name__ == "__main__":
    main()
