"""HPKE tests: interoperability with pyhpke (an independent RFC 9180 implementation) for DHKEM P-256 and X25519,
which checks the key schedule and the AEAD; round-trip tests for ML-KEM-768 and the Elligator variant."""
import pyhpke
from cryptography.hazmat.primitives.asymmetric import ec, x25519
from pyhpke import AEADId, CipherSuite, KDFId, KEMId, KEMKey

from pqpks.analysis.dpub import p256_uncompressed_on_curve
from pqpks.crypto import hpke

INFO = b"PQPKS-v1"


def _interop(our_kem, pyhpke_kem, gen):
    sk = gen()
    suite = CipherSuite.new(pyhpke_kem, KDFId.HKDF_SHA256, AEADId.CHACHA20_POLY1305)
    pk_bytes = our_kem._ser(sk.public_key())
    # Sender: this implementation. Receiver: pyhpke.
    enc, ctx = hpke.setup_base_s(our_kem, pk_bytes, INFO)
    ct0 = ctx.seal(0, b"aad0", b"hello 0")
    ct1 = ctx.seal(1, b"aad1", b"hello 1")
    exp = ctx.export(b"exp-ctx", 32)
    rkey = KEMKey.from_pyca_cryptography_key(sk)
    recv = suite.create_recipient_context(enc, rkey, info=INFO)
    assert recv.open(ct0, aad=b"aad0") == b"hello 0"
    assert recv.open(ct1, aad=b"aad1") == b"hello 1"
    assert recv.export(b"exp-ctx", 32) == exp
    # Reverse direction: pyhpke sends, this implementation receives.
    enc2, sender = suite.create_sender_context(KEMKey.from_pyca_cryptography_key(sk.public_key()), info=INFO)
    c = sender.seal(b"from pyhpke", aad=b"x")
    rctx = hpke.setup_base_r(our_kem, enc2, sk, INFO)
    assert rctx.open(0, b"x", c) == b"from pyhpke"


def test_interop_p256():
    _interop(hpke.DhkemP256(), KEMId.DHKEM_P256_HKDF_SHA256, lambda: ec.generate_private_key(ec.SECP256R1()))


def test_interop_x25519():
    _interop(hpke.DhkemX25519(), KEMId.DHKEM_X25519_HKDF_SHA256, x25519.X25519PrivateKey.generate)


def test_p256_enc_is_curve_point():
    kem = hpke.DhkemP256()
    _, pk = kem.generate()
    enc, _ = hpke.setup_base_s(kem, pk, INFO)
    assert len(enc) == 65 and p256_uncompressed_on_curve(enc)


def test_mlkem768_roundtrip_and_export():
    kem = hpke.MlKem768()
    sk, pk = kem.generate()
    assert len(pk) == 1184
    enc, ctx = hpke.setup_base_s(kem, pk, INFO)
    assert len(enc) == 1088
    r = hpke.setup_base_r(kem, enc, sk, INFO)
    for seq in (1, 7, 3):                      # opened in arbitrary order
        assert r.open(seq, b"ad", ctx.seal(seq, b"ad", b"m%d" % seq)) == b"m%d" % seq
    assert r.export(b"e", 32) == ctx.export(b"e", 32)
    assert r.open(2, b"ad", ctx.seal(1, b"ad", b"x")) is None     # sai seq → ⊥


def test_mlkem_implicit_rejection_gives_wrong_context():
    kem = hpke.MlKem768()
    sk, pk = kem.generate()
    enc, ctx = hpke.setup_base_s(kem, pk, INFO)
    bad = bytes([enc[0] ^ 1]) + enc[1:]
    r = hpke.setup_base_r(kem, bad, sk, INFO)
    assert r is not None and r.open(1, b"", ctx.seal(1, b"", b"m")) is None


def test_elligator_variant():
    kem = hpke.DhkemX25519Elligator()
    sk, pk = kem.generate()
    for _ in range(20):
        enc, ctx = hpke.setup_base_s(kem, pk, INFO)
        assert len(enc) == 32
        r = hpke.setup_base_r(kem, enc, sk, INFO)
        assert r.open(1, b"", ctx.seal(1, b"", b"ok")) == b"ok"
