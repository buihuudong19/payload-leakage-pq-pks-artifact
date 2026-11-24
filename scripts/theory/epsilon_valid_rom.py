"""Bound on epsilon_valid of the correctness theorem without assumption A-MIN, with G modelled as a random oracle
(post hoc, 24 Nov 2026; finding F04 of the regression audit). Exact arithmetic, no data.

A wrong candidate c' = (u', v') of the new-session branch is accepted by ML-KEM.Decaps only if re-encryption reproduces it:
Enc(ek, m''; coins(G(m'' || H(ek)))) = c' with m'' = Dec(dk, c'). If m'' is the honest message m*, re-encryption gives the
honest ciphertext c* != c' and the check fails. Otherwise the stego image (hence c') does not depend on G(m'' || H(ek)), so with
G and the coin expansion modelled as random oracles the coins are fresh: e1 has 768 independent CBD_2 coefficients added
after the matrix product, and u' must equal Compress_10(w + e1) coefficientwise for the w fixed by the other coins. Each
Compress_10 cell is a set of 3 or 4 consecutive residues mod q, so each coefficient matches with probability at most
p = max over cells and shifts of the CBD_2 mass of the cell; the candidate passes with probability at most p^768, and a union
over the at most |B_1| wrong candidates gives epsilon_valid <= |B_1| p^768.

    python scripts/theory/epsilon_valid_rom.py
"""
from __future__ import annotations

import math
from fractions import Fraction

Q, D, K, N, ETA2, B1 = 3329, 10, 3, 256, 2, 6


def compress(x: int, d: int = D, q: int = Q) -> int:
    return ((x << d) + q // 2) // q % (1 << d)        # round(2^d x / q) mod 2^d (FIPS 203, Compress_d)


def cells() -> dict[int, list[int]]:
    c: dict[int, list[int]] = {}
    for x in range(Q):
        c.setdefault(compress(x), []).append(x)
    return c


def cbd(eta: int) -> dict[int, Fraction]:
    pmf: dict[int, Fraction] = {}
    for a in range(1 << eta):
        for b in range(1 << eta):
            v = bin(a).count("1") - bin(b).count("1")
            pmf[v] = pmf.get(v, Fraction(0)) + Fraction(1, 1 << (2 * eta))
    return pmf


def max_cell_mass() -> tuple[Fraction, int]:
    pmf = cbd(ETA2)
    best, size = Fraction(0), 0
    for cell in cells().values():
        s = set(cell)
        for w in range(Q):                     # shift by the fixed part w of the coefficient
            m = sum((p for e, p in pmf.items() if (w + e) % Q in s), Fraction(0))
            if m > best:
                best, size = m, len(cell)
    return best, size


def main():
    c = cells()
    sizes = sorted({len(v) for v in c.values()})
    counts = {s: sum(1 for v in c.values() if len(v) == s) for s in sizes}
    consecutive = all(sorted(v) == list(range(min(v), min(v) + len(v))) or len(set((x - v[0]) % Q for x in v)) == len(v)
                      for v in c.values())
    p, size = max_cell_mass()
    gamma = -K * N * math.log2(p)
    eps = math.log2(B1) - gamma
    print(f"Compress_10 cells: sizes {sizes} with counts {counts}; all cells consecutive residues: {consecutive}")
    print(f"largest CBD_2 mass of a cell (size {size}): p = {p} = {float(p):.6f}")
    print(f"per-candidate bound p^768 = 2^{-gamma:.2f}; epsilon_valid <= |B_1| p^768 = 2^{eps:.2f} with |B_1| = {B1}")


if __name__ == "__main__":
    main()
