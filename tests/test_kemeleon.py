"""Tests of the FIPS 203 codec and of Kemeleon."""
import random
import secrets

import numpy as np
import pytest
from cryptography.hazmat.primitives.asymmetric import mlkem
from scipy import stats

from pqpks.crypto import fips203, kemeleon
from pqpks.params import KEMELEON_CT_BYTES, KEMELEON_EK_BYTES, MLKEM_Q as Q


@pytest.mark.parametrize("d", [4, 5, 10, 11])
def test_sample_preimage_exhaustive(d):
    """For EVERY c: the set of values SamplePreimage can return equals the preimage {x : Compress_d(x) = c}."""
    comp = fips203.compress(np.arange(Q), d)
    for c in range(1 << d):
        u = int(fips203.decompress(np.array([c]), d)[0])
        expected = set(np.nonzero(comp == c)[0].tolist())
        # Enumerate exactly the branches of the specification to obtain the full range.
        if d == 10:
            if int(fips203.compress(np.array([(u + 2) % Q]), d)[0]) == c:
                rng = range(-1, 3)
            elif int(fips203.compress(np.array([(u - 2) % Q]), d)[0]) == c:
                rng = range(-2, 2)
            else:
                rng = range(-1, 2)
        elif d == 11:
            if int(fips203.compress(np.array([(u + 1) % Q]), d)[0]) == c:
                rng = range(0, 2)
            elif int(fips203.compress(np.array([(u - 1) % Q]), d)[0]) == c:
                rng = range(-1, 1)
            else:
                rng = range(0, 1)
        elif d == 5:
            rng = range(-52, 53) if u == 0 else (range(-51, 53) if u <= 1560 else range(-52, 52))
        else:
            rng = range(-104, 105) if u == 0 else (range(-103, 105) if u <= 1456 else range(-104, 104))
        got = {(u + r) % Q for r in rng}
        assert got == expected, f"d={d}, c={c}: {sorted(got)[:5]} vs {sorted(expected)[:5]}"
        # The actual function always returns an element of the preimage.
        for _ in range(3):
            assert kemeleon.sample_preimage(d, u, c) in expected


def test_byte_codec_roundtrip():
    for d in (4, 10, 12):
        f = np.array([random.randrange(Q if d == 12 else 1 << d) for _ in range(256)])
        assert np.array_equal(fips203.byte_decode(fips203.byte_encode(f, d), d, 256), f)


def test_vector_encode_decode():
    for _ in range(50):
        a = np.array([random.randrange(Q) for _ in range(256)])
        blob = kemeleon.vector_encode(a)
        assert len(blob) == 384
        assert np.array_equal(kemeleon.vector_decode(blob), a)


def test_ct_roundtrip_and_decaps():
    """decode_ct(encode_ct(c)) = c byte for byte, and Decaps still yields the same key."""
    sk = mlkem.MLKEM768PrivateKey.generate()
    pk = sk.public_key()
    for _ in range(30):
        K, ct = pk.encapsulate()
        enc = kemeleon.encode_ct(ct)
        assert len(enc) == KEMELEON_CT_BYTES == 1536
        assert enc != kemeleon.encode_ct(ct)          # the encoding is randomized
        back = kemeleon.decode_ct(enc)
        assert back == ct
        assert sk.decapsulate(back) == K


def test_ek_roundtrip():
    sk = mlkem.MLKEM768PrivateKey.generate()
    ek = sk.public_key().public_bytes_raw()
    enc = kemeleon.encode_ek(ek)
    assert len(enc) == KEMELEON_EK_BYTES == 1184
    assert kemeleon.decode_ek(enc) == ek


def test_ct_encoding_looks_uniform():
    """Byte-wise chi-square test on 200 encoded ciphertexts: uniformity is not rejected at level 0.001.
    (The full NIST SP 800-22 battery on >= 10^8 bits is in scripts/sanity_matrix.py.)"""
    pk = mlkem.MLKEM768PrivateKey.generate().public_key()
    data = b"".join(kemeleon.encode_ct(pk.encapsulate()[1]) for _ in range(200))
    counts = np.bincount(np.frombuffer(data, dtype=np.uint8), minlength=256)
    p = stats.chisquare(counts).pvalue
    assert p > 1e-3, p


def test_raw_ct_is_not_uniform_llr():
    """Control for Proposition 1: the LLR on the u part of raw ML-KEM ciphertexts leans clearly towards 'non-uniform'."""
    from pqpks.analysis.dpub import mlkem_raw_llr
    pk = mlkem.MLKEM768PrivateKey.generate().public_key()
    raw = [mlkem_raw_llr(pk.encapsulate()[1]) for _ in range(100)]
    unif = [mlkem_raw_llr(secrets.token_bytes(1088)) for _ in range(100)]
    acc = (sum(x > 0 for x in raw) + sum(x <= 0 for x in unif)) / 200
    assert acc > 0.85, acc


# ---------------------------------------------------------------------------
# Rejection-sampled variant (draft-02 Section 5.1, EncodeCtxtR), ablation A4
# ---------------------------------------------------------------------------
from pqpks.params import KEMELEON_R_CT_BYTES, KEMELEON_R_INT_BITS, KEMELEON_R_INT_BYTES  # noqa: E402


def test_compact_constants_match_draft():
    """b = floor(log2 q^768) = 8986; r takes 1124 B; the rejection-sampled ciphertext has 1124 + 128 = 1252 B (draft, Table 2)."""
    assert (Q ** 768).bit_length() - 1 == KEMELEON_R_INT_BITS == 8986
    assert KEMELEON_R_INT_BYTES == 1124 and KEMELEON_R_CT_BYTES == 1252


def test_vector_encode_r_roundtrip_and_rejection():
    acc = 0
    tops = set()
    for _ in range(300):
        a = np.array([random.randrange(Q) for _ in range(768)])
        blob = kemeleon.vector_encode_r(a)
        r = 0
        for v in reversed(a.tolist()):
            r = r * Q + int(v)
        if r >> KEMELEON_R_INT_BITS:
            assert blob is None                       # msb(r) = 1 → err
            continue
        acc += 1
        assert len(blob) == KEMELEON_R_INT_BYTES
        assert np.array_equal(kemeleon.vector_decode_r(blob), a)
        tops.add(blob[0] >> 2)                        # 6 randomized bits
    # P(msb = 0) = 2^8986 / q^768 ~ 0.829; 300 samples: SE ~ 0.022
    assert 0.74 < acc / 300 < 0.92
    assert len(tops) > 40                             # the 6 high bits take many values


def test_ct_r_roundtrip_decaps_and_rate():
    """decode_ct_r(encode_ct_r(c)) = c when accepted; acceptance rate ~ 0.768 (draft: 0.77)."""
    sk = mlkem.MLKEM768PrivateKey.generate()
    pk = sk.public_key()
    acc, nibbles = 0, []
    n = 400
    for _ in range(n):
        K, ct = pk.encapsulate()
        enc = kemeleon.encode_ct_r(ct)
        if enc is None:
            continue
        acc += 1
        assert len(enc) == KEMELEON_R_CT_BYTES
        back = kemeleon.decode_ct_r(enc)
        assert back == ct and sk.decapsulate(back) == K
        c2 = enc[KEMELEON_R_INT_BYTES:]
        nibbles += [b & 0x0F for b in c2] + [b >> 4 for b in c2]
    rate = acc / n
    assert 0.69 < rate < 0.84, rate                   # 0.768 +- 3.7 SE
    # Screening: after rejection, c_2 is conditionally uniform on {0..15}
    counts = np.bincount(nibbles, minlength=16)
    assert stats.chisquare(counts).pvalue > 1e-4


def test_compact_c2_rejection_makes_zero_symbol_uniform():
    """Analytic check of the distribution: Compress_4 on Z_q gives 209 preimages for cell 0 and 208 for the other 15 cells;
    rejecting cell 0 with probability 1/209 gives all 16 cells weight 208."""
    counts = np.bincount(fips203.compress(np.arange(Q), 4), minlength=16)
    assert counts[0] == 209 and set(counts[1:].tolist()) == {208}
    assert counts[0] * (1 - 1 / 209) == 208


# ---------------------------------------------------------------- statistical distance of VectorEncode
@pytest.mark.parametrize("q,n,bits", [(5, 3, 8), (7, 2, 7), (13, 2, 9), (3329, 1, 14)])
def test_vector_encode_tv_formula_exhaustive(q, n, bits):
    """TV(VectorEncode(U_q^n), U_{2^bits}) = R0(Q - R0)/(Q 2^bits): exhaustive output distribution for small parameters,
    with the same rule for drawing m as the implementation (upper bound floor((2^bits - 1 - r)/Q))."""
    from fractions import Fraction
    Qn, top = q ** n, 1 << bits
    pr = {}
    for r in range(Qn):
        hi = (top - 1 - r) // Qn
        for m in range(hi + 1):
            s = r + m * Qn
            assert s < top
            pr[s] = Fraction(1, Qn * (hi + 1))
    assert sum(pr.values()) == 1
    tv = sum(abs(pr.get(s, 0) - Fraction(1, top)) for s in range(top)) / 2
    assert tv == kemeleon.vector_encode_tv(q, n, bits)


def test_vector_encode_tv_value_for_mlkem768():
    import math
    tv = kemeleon.vector_encode_tv()
    log2 = math.log2(tv.numerator) - math.log2(tv.denominator)
    assert -78.9 < log2 < -78.8                    # each vector has 384 B; a ciphertext of 4 vectors <= 2^-76.88


def test_vector_encode_output_always_fits(monkeypatch):
    """Edge case of draft-02 Section 4.1: r = 2^L mod q^n and the largest m. With the implementation bound the output stays < 2^L."""
    r0 = kemeleon._BOUND % kemeleon._QN
    digits, r = [], r0
    for _ in range(256):
        r, d = divmod(r, Q)
        digits.append(d)
    a = np.array(digits)
    monkeypatch.setattr(kemeleon, "_rand_int", lambda lo, hi: hi)
    blob = kemeleon.vector_encode(a)
    assert len(blob) == 384 and np.array_equal(kemeleon.vector_decode(blob), a)
