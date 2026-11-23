"""Single-channel port of the LSB-MPVD embedder [MPS19] from the BL1 reference implementation (Embed.py, Extract.py).

The port keeps EVERY logical branch of the reference code: the Range table, opvd, mpvd, scruStep, the zero-bit padding at
the end, the traversal of 3x3 blocks and then 2x2 blocks over the four image-size cases (mainEmbed), and the stopping
condition. It differs only in that (i) it runs on one 2-D channel (grey-scale image) instead of the R channel of an RGB
image and (ii) it uses no global variables and no tkinter. Bit-exactness with the reference code is checked in
tests/test_mpvd_equivalence.py.
"""
from __future__ import annotations

import numpy as np


def _range(pvd: int) -> int:
    if pvd < 16:
        return 4
    if 16 <= pvd < 32:
        return 4
    if 32 <= pvd <= 255:
        return 5
    return 0


def _opvd(old: int, new: int) -> int:
    d = old - new
    if d > 8 and 0 <= new + 16 <= 255:
        return new + 16
    if d < -8 and 0 <= new - 16 <= 255:
        return new - 16
    return new


def _mpvd(old: int, new: int, nbit: int) -> int:
    defl = old - new
    if defl > 2 ** (nbit - 1) and 0 <= new + 2 ** nbit <= 255:
        g3 = new + 2 ** nbit
    elif defl < -(2 ** (nbit - 1)) and 0 <= new - 2 ** nbit <= 255:
        g3 = new - 2 ** nbit
    else:
        g3 = new
    if abs(old - new) < abs(old - g3) and 0 <= new <= 255:
        return new
    return g3


def _scru(main: int, old: int, new: int) -> int:
    if abs(main - old) >= 32 and abs(main - new) <= 31:
        gf = new - 32 if main > new else new + 32
        if gf < 0:
            gf += 96
        elif gf > 255:
            gf -= 96
        return gf
    if abs(main - old) <= 31 and abs(main - new) >= 32:
        return new + 32 if main > new else new - 32
    return new


def layout(hi: int, wi: int) -> list[tuple]:
    """List of regions in the exact order of mainEmbed (first channel).
    ('B3', h_end, w_end) or ('B2', his, hie_excl, wis, wie_excl)."""
    def b3(hi3e, wi3e):
        return ("B3", hi3e + 1, wi3e + 1)

    def b2(hs, he, ws, we):
        return ("B2", hs, he + 1, ws, we + 1)

    if hi % 3 == 0 and wi % 3 == 0:
        return [("B3", hi, wi)]
    if hi % 3 != 0 and wi % 3 == 0:
        wi3e = wi - 1
        hi3e = hi - 3 if hi % 3 == 2 else hi - 5
        wi2e = wi - 2 if wi % 2 == 1 else wi - 1
        return [b3(hi3e, wi3e), b2(hi3e + 1, hi - 1, 0, wi2e)]
    if hi % 3 == 0 and wi % 3 != 0:
        hi3e = hi - 1
        wi3e = wi - 3 if wi % 3 == 2 else wi - 5
        hi2e = hi - 2 if hi % 2 == 1 else hi - 1
        return [b3(hi3e, wi3e), b2(0, hi2e, wi3e + 1, wi - 1)]
    widu, hidu = wi % 3, hi % 3
    if widu == 2 and hidu == 2:
        hi3e, wi3e = hi - 3, wi - 3
        hi2e_d = hi - 1 if hi % 2 == 0 else hi - 2
        wi2e_n = wi3e if wi % 2 == 0 else wi3e - 1
        return [b3(hi3e, wi3e), b2(0, hi2e_d, wi3e + 1, wi - 1), b2(hi3e + 1, hi - 1, 0, wi2e_n)]
    if widu == 1 and hidu == 1:
        wi3e = wi - 2 if (wi % 2 == 0 and hi % 2 == 1) else wi - 5
        hi3e = hi - 5 if wi % 2 == 0 else hi - 2
        if wi % 2 == 0 and hi % 2 == 1:
            reg = b2(hi - 4, hi - 1, 0, wi - 1)
        else:
            hi2e = hi - 2 if (wi % 2 == 1 and hi % 2 == 1) else hi - 1
            reg = b2(0, hi2e, wi - 4, wi - 1)
        if wi % 2 == 0 and hi % 2 == 0:
            return [b3(hi3e, wi3e), reg, b2(hi - 4, hi - 1, 0, wi - 5)]
        return [b3(hi3e, wi3e), reg]
    # widu, hidu ∈ {(2,1), (1,2)}
    if hidu == 2 and hi % 2 == 1:
        wi3e = wi - 2
    elif hidu == 2 and hi % 2 == 0:
        wi3e = wi - 5
    else:
        wi3e = wi - 3
    if widu == 2 and wi % 2 == 1:
        hi3e = hi - 2
    elif widu == 2 and wi % 2 == 0:
        hi3e = hi - 5
    else:
        hi3e = hi - 3
    if (widu == 2 and wi % 2 == 1) or (hidu == 2 and hi % 2 == 1):
        if widu == 2 and wi % 2 == 1:
            hi2e = hi - 1 if hi % 2 == 0 else hi - 2
            return [b3(hi3e, wi3e), b2(0, hi2e, wi - 2, wi - 1)]
        wi2e = wi - 1 if wi % 2 == 0 else wi - 2
        return [b3(hi3e, wi3e), b2(hi - 2, hi - 1, 0, wi2e)]
    if widu == 2 and wi % 2 == 0:
        hi2e_d = hi - 1 if hi % 2 == 0 else hi - 2
        return [b3(hi3e, wi3e), b2(0, hi2e_d, wi - 2, wi - 1), b2(hi - 4, hi - 1, 0, wi - 3)]
    wi2e_n = wi - 5 if wi % 2 == 0 else wi - 6
    return [b3(hi3e, wi3e), b2(0, hi - 1, wi - 4, wi - 1), b2(hi - 2, hi - 1, 0, wi2e_n)]


def _blocks(hi: int, wi: int):
    """Yields (center(y, x), neighbors[(y, x)...]) in the traversal order of the reference code."""
    for reg in layout(hi, wi):
        if reg[0] == "B3":
            _, he, we = reg
            for i in range(0, he, 3):
                for j in range(0, we, 3):
                    nb = [(k, l) for k in range(i, i + 3) if k < he for l in range(j, j + 3)
                          if l < we and not (k == i + 1 and l == j + 1)]
                    # The reference code breaks when k >= high or l >= wide; with increasing loops this is a filter.
                    yield (i + 1, j + 1), nb
        else:
            _, hs, he, ws, we = reg
            for i in range(hs, he, 2):
                for j in range(ws, we, 2):
                    nb = [(k, l) for k in range(i, i + 2) if k < he for l in range(j, j + 2)
                          if l < we and not (k == i and l == j)]
                    yield (i, j), nb


class _Stream:
    def __init__(self, bits: str):
        self.bits = bits
        self.complete = len(bits) == 0

    def take(self, pixel: int, nbit: int) -> int:
        if nbit <= len(self.bits):
            newbit, self.bits = self.bits[:nbit], self.bits[nbit:]
        else:
            newbit = self.bits + "0" * (nbit - len(self.bits))
            self.bits = ""
        if len(self.bits) == 0:
            self.complete = True
        pb = format(pixel, "08b")
        return int(pb[: len(pb) - len(newbit)] + newbit, 2)


def embed(cover: np.ndarray, bits: str) -> np.ndarray:
    """Embed a bit string ('0101...') with MPVD, sequentially from the top-left corner. Raises ValueError when space runs out."""
    a = np.asarray(cover, dtype=np.int64).copy()
    hi, wi = a.shape
    st = _Stream(bits)
    for (cy, cx), nbs in _blocks(hi, wi):
        if st.complete:
            break
        old_c = int(a[cy, cx])
        new_c = _opvd(old_c, st.take(old_c, 4))
        a[cy, cx] = new_c
        for (k, l) in nbs:
            if st.complete:
                break
            g = int(a[k, l])
            n = _range(abs(new_c - g))
            g1 = st.take(g, n)
            a[k, l] = _scru(new_c, g, _mpvd(g, g1, n))
    if not st.complete:
        raise ValueError("MPVD: insufficient capacity for the payload (one channel)")
    return a.astype(np.uint8)


def extract(stego: np.ndarray, nbits: int) -> str:
    a = np.asarray(stego, dtype=np.int64)
    hi, wi = a.shape
    out = []
    total = 0
    for (cy, cx), nbs in _blocks(hi, wi):
        c = int(a[cy, cx])
        s = format(c, "08b")[4:]
        out.append(s)
        total += 4
        for (k, l) in nbs:
            g = int(a[k, l])
            n = _range(abs(c - g))
            out.append(format(g, "08b")[8 - n:])
            total += n
        if total >= nbits:
            break
    return "".join(out)[:nbits]


def capacity_bits(cover: np.ndarray) -> int:
    """Approximate capacity, as CheckCapacity of the reference code (uses the values before embedding)."""
    a = np.asarray(cover, dtype=np.int64)
    hi, wi = a.shape
    tot = 0
    for (cy, cx), nbs in _blocks(hi, wi):
        tot += 4 + sum(_range(abs(int(a[cy, cx]) - int(a[k, l]))) for k, l in nbs)
    return tot
