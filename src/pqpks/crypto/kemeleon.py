"""Kemeleon (draft-irtf-cfrg-kemeleon-02) for ML-KEM-768: main variant (Section 4) and rejection-sampled variant (Section 5.1).

Maps ML-KEM encapsulation keys and ciphertexts to byte strings that are close to uniform.

Main variant: when the input coefficients are uniform on Z_q, the statistical distance of EACH vector (384 B) from a
uniform string is exactly R0(Q - R0)/(Q*2^L), with Q = q^256, L = 3072, R0 = 2^L mod Q, i.e. 2^-78.88 (function
`vector_encode_tv`). A ciphertext has 4 vectors, hence <= 2^-76.88; an encapsulation key (3 vectors) <= 2^-77.3.
Uniformity of the coefficients before compression is a computational step (MLWE / SPR of ML-KEM, [GSV24]).
Rejection-sampled variant: conditioned on acceptance, the output is EXACTLY uniform when the input is uniform.

Implementation notes:
  * The VectorEncode integer is serialized big-endian on 384 bytes. The draft has no test vectors yet, so the
    byte order could not be cross-checked. Uniformity and invertibility do not depend on this choice.
  * All randomness (SamplePreimage, the multiple m*q^n) comes from the operating-system CSPRNG (`secrets`) and is
    never published (a security requirement of the draft).
"""
from __future__ import annotations

import secrets

import numpy as np

from ..params import (KEMELEON_B, KEMELEON_R_CT_BYTES, KEMELEON_R_INT_BITS, KEMELEON_R_INT_BYTES,
                      KEMELEON_T, KEMELEON_VEC_BYTES, MLKEM768_DU, MLKEM768_DV, MLKEM768_K, MLKEM_N, MLKEM_Q)
from . import fips203

Q, N, K = MLKEM_Q, MLKEM_N, MLKEM768_K
_QN = Q ** N
_BOUND = 1 << (KEMELEON_B + KEMELEON_T)


def _rand_int(lo: int, hi: int) -> int:
    """Uniform integer in [lo, hi] (both ends included), from the CSPRNG."""
    return lo + secrets.randbelow(hi - lo + 1)


# ---------------------------------------------------------------------------
# VectorEncode / VectorDecode
# ---------------------------------------------------------------------------
def vector_encode_tv(q: int = Q, n: int = N, bits: int = KEMELEON_B + KEMELEON_T):
    """EXACT statistical distance (Fraction) between VectorEncode(a), a uniform on Z_q^n, and a uniform `bits`-bit string.

    Write 2^L = M*Q + R0 (Q = q^n). Every s in [0, 2^L) has exactly one pair (r = s mod Q, m = s div Q). Residues r < R0
    have M + 1 values of m and residues r >= R0 have M, so Pr[s] = 1/(Q(M+1)) or 1/(QM). Summing the deviations gives
    TV = R0(Q - R0)/(Q*2^L) <= Q/2^(L+2). Checked exhaustively on small parameters in tests/test_kemeleon.py."""
    from fractions import Fraction
    qn = q ** n
    r0 = (1 << bits) % qn
    return Fraction(r0 * (qn - r0), qn * (1 << bits))


def vector_encode(a: np.ndarray) -> bytes:
    """r = sum q^(i-1)*a[i]; m <-$ [0, floor((2^(b+t) - 1 - r)/q^n)]; returns r + m*q^n (384 bytes).

    Draft-02 Section 4.1 bounds m by floor((2^(b+t) - r)/q^n). When r = 2^(b+t) (mod q^n) and m takes its largest
    value, the result equals 2^(b+t) and does not fit in b + t bits (probability about 2^-3072). This implementation
    uses floor((2^(b+t) - 1 - r)/q^n): every output lies in [0, 2^(b+t)), and the distribution differs from the draft
    only on an event of probability at most 2^-3000."""
    a = [int(v) % Q for v in np.asarray(a).reshape(-1)]
    assert len(a) == N
    r = 0
    for coef in reversed(a):           # Horner: a[1] is the least significant digit
        r = r * Q + coef
    m = _rand_int(0, (_BOUND - 1 - r) // _QN)
    out = r + m * _QN
    assert out < _BOUND
    return out.to_bytes(KEMELEON_VEC_BYTES, "big")


def vector_decode(blob: bytes) -> np.ndarray:
    assert len(blob) == KEMELEON_VEC_BYTES
    r = int.from_bytes(blob, "big") % _QN
    a = np.empty(N, dtype=np.int64)
    for i in range(N):
        r, a[i] = divmod(r, Q)
    return a


# ---------------------------------------------------------------------------
# SamplePreimage
# ---------------------------------------------------------------------------
def sample_preimage(d: int, u: int, c: int) -> int:
    """Uniformly random preimage under Compress_d of the value c, around u = Decompress_d(c)."""
    if d == 10:
        if int(fips203.compress(np.array([(u + 2) % Q]), d)[0]) == c:
            rnd = _rand_int(-1, 2)
        elif int(fips203.compress(np.array([(u - 2) % Q]), d)[0]) == c:
            rnd = _rand_int(-2, 1)
        else:
            rnd = _rand_int(-1, 1)
    elif d == 11:
        if int(fips203.compress(np.array([(u + 1) % Q]), d)[0]) == c:
            rnd = _rand_int(0, 1)
        elif int(fips203.compress(np.array([(u - 1) % Q]), d)[0]) == c:
            rnd = _rand_int(-1, 0)
        else:
            rnd = 0
    elif d == 5:
        rnd = _rand_int(-52, 52) if u == 0 else (_rand_int(-51, 52) if u <= 1560 else _rand_int(-52, 51))
    elif d == 4:
        rnd = _rand_int(-104, 104) if u == 0 else (_rand_int(-103, 104) if u <= 1456 else _rand_int(-104, 103))
    else:
        raise ValueError(f"SamplePreimage does not support d = {d}")
    return (u + rnd) % Q


def _preimage_vector(c: np.ndarray, d: int) -> np.ndarray:
    u = fips203.decompress(c, d)
    return np.array([sample_preimage(d, int(ui), int(ci)) for ui, ci in zip(u, c)], dtype=np.int64)


# ---------------------------------------------------------------------------
# Encapsulation keys and ciphertexts
# ---------------------------------------------------------------------------
def encode_ek(ek: bytes) -> bytes:
    t, rho = fips203.split_ek(ek, K)
    return b"".join(vector_encode(t[i * N:(i + 1) * N]) for i in range(K)) + rho


def decode_ek(blob: bytes) -> bytes:
    vecs = [vector_decode(blob[i * KEMELEON_VEC_BYTES:(i + 1) * KEMELEON_VEC_BYTES]) for i in range(K)]
    rho = blob[K * KEMELEON_VEC_BYTES:]
    return fips203.join_ek(np.concatenate(vecs), rho)


def encode_ct(ct: bytes) -> bytes:
    """EncodeCtxt: decompress c1 and c2, sample preimages, then VectorEncode each vector (k + 1 blocks)."""
    c1, c2 = fips203.split_ciphertext(ct, K, MLKEM768_DU, MLKEM768_DV)
    u = _preimage_vector(c1, MLKEM768_DU)
    v = _preimage_vector(c2, MLKEM768_DV)
    blocks = [vector_encode(u[i * N:(i + 1) * N]) for i in range(K)] + [vector_encode(v)]
    return b"".join(blocks)


def decode_ct(blob: bytes) -> bytes:
    """DecodeCtxt: VectorDecode each block and compress again; recovers the original ciphertext exactly."""
    vecs = [vector_decode(blob[i * KEMELEON_VEC_BYTES:(i + 1) * KEMELEON_VEC_BYTES]) for i in range(K + 1)]
    c1 = fips203.compress(np.concatenate(vecs[:K]), MLKEM768_DU)
    c2 = fips203.compress(vecs[K], MLKEM768_DV)
    return fips203.join_ciphertext(c1, c2, MLKEM768_DU, MLKEM768_DV)


# ---------------------------------------------------------------------------
# Rejection-sampled variant (draft-02 Section 5.1: EncodeCtxtR / DecodeCtxtR)
# Ablation A4; the default of PQ-PKS remains the main variant.
# ---------------------------------------------------------------------------
_R_UNUSED = 8 - (KEMELEON_R_INT_BITS % 8)          # x = 8 - (b mod 8) = 6 unused bits in the first byte
_R_RAND_MASK = (0xFF << (8 - _R_UNUSED)) & 0xFF     # 0xFC
_R_CLEAR_MASK = 0xFF >> _R_UNUSED                   # 0x03
_C2_REJECT_DEN = -(-Q // (1 << MLKEM768_DV))        # ceil(q / 2^dv) = 209
_C1_BYTES = 32 * MLKEM768_DU * K                    # 960


def vector_encode_r(a: np.ndarray) -> bytes | None:
    """VectorEncodeR: merge the k*n coefficients into one integer r; reject if msb(r) = 1; randomize the unused bits.

    r < q^(k*n) < 2^(b+1) with b = floor(log2 q^(k*n)) = 8986, and msb(r) is bit b. On acceptance r < 2^b, which fills
    the b low bits of 1124 bytes (big-endian); the 6 top bits of the first byte are filled at random.
    """
    a = [int(v) % Q for v in np.asarray(a).reshape(-1)]
    assert len(a) == K * N
    r = 0
    for coef in reversed(a):
        r = r * Q + coef
    if r >> KEMELEON_R_INT_BITS:                    # msb(r) = 1 → err
        return None
    rb = bytearray(r.to_bytes(KEMELEON_R_INT_BYTES, "big"))
    rb[0] |= secrets.randbits(8) & _R_RAND_MASK      # IntegerRandomizeUnused
    return bytes(rb)


def vector_decode_r(blob: bytes) -> np.ndarray:
    assert len(blob) == KEMELEON_R_INT_BYTES
    rb = bytearray(blob)
    rb[0] &= _R_CLEAR_MASK                          # IntegerClearUnused
    r = int.from_bytes(rb, "big")
    a = np.empty(K * N, dtype=np.int64)
    for i in range(K * N):
        r, a[i] = divmod(r, Q)
    return a


def encode_ct_r(ct: bytes) -> bytes | None:
    """EncodeCtxtR. Returns None on rejection; the caller encapsulates again with fresh randomness (offline).

    Besides msb(r), every coefficient c_2[i] = 0 causes a rejection with probability 1/ceil(q/2^dv) = 1/209. Cell 0 of
    Compress_4 has 209 preimages and the other 15 cells have 208, so after this step c_2 is, conditionally, uniform on
    {0..15}. Each c_2[i] equals 0 with probability 209/q, so the per-coefficient rejection probability is exactly 1/q.
    Theoretical acceptance rate: 2^(b - log2 q^768) * (1 - 1/q)^256 = 0.7676 (draft: 0.77). This is the byte-aligned
    form of the original Kemeleon of [GSV24] (draft-02 Section 5.1); ciphertext uniformity for the construction
    "encapsulate, then encode, repeat until success" is a theorem of [GSV24] (ePrint 2024/1086, Section 2.4; game
    without K). With K, the paper's lemma (a hybrid over attempts, same ek) gives the bound
    ((1 - (1 - p_acc)^T)/p_acc)*Adv^spr + (1 - p_acc)^T <= 1.303*Adv^spr + 2^-134.7 with T = 64.
    """
    c1, c2 = fips203.split_ciphertext(ct, K, MLKEM768_DU, MLKEM768_DV)
    u = _preimage_vector(c1, MLKEM768_DU)
    r = vector_encode_r(u)
    if r is None:
        return None
    for ci in c2:
        if int(ci) == 0 and secrets.randbelow(_C2_REJECT_DEN) == 0:
            return None
    out = r + ct[_C1_BYTES:]                        # c_2 stays in its ByteEncode_4 form (128 bytes)
    assert len(out) == KEMELEON_R_CT_BYTES
    return out


def decode_ct_r(blob: bytes) -> bytes:
    """DecodeCtxtR: u = VectorDecodeR(r); c_1 = Compress_du(u); c_2 unchanged."""
    assert len(blob) == KEMELEON_R_CT_BYTES
    u = vector_decode_r(blob[:KEMELEON_R_INT_BYTES])
    c1 = fips203.compress(u, MLKEM768_DU)
    return fips203.byte_encode(c1, MLKEM768_DU) + blob[KEMELEON_R_INT_BYTES:]
