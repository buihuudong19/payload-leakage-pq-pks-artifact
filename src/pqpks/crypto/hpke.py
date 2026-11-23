"""HPKE base mode (RFC 9180), shared by every compared system.

Fixed suite:
    KDF  = HKDF-SHA256 (0x0001), two-stage key schedule of RFC 9180
    AEAD = ChaCha20-Poly1305 (0x0003)
KEM:
    0x0041 ML-KEM-768        (draft-ietf-hpke-pq-05; Encap = ML-KEM.Encaps, enc = 1088-byte ciphertext)
    0x0010 DHKEM(P-256, HKDF-SHA256)   (RFC 9180; enc = 65-byte uncompressed point)
    0x0020 DHKEM(X25519, HKDF-SHA256)  (RFC 9180; enc = 32 B). The Elligator 2 variant used by BL4-E changes only
                                        the serialization of enc; see class DhkemX25519Elligator.

The systems differ only in the KEM and in the encoding of enc; the key schedule and the AEAD are identical.
"""
from __future__ import annotations

from dataclasses import dataclass

from cryptography.hazmat.primitives import hashes, hmac, serialization
from cryptography.hazmat.primitives.asymmetric import ec, mlkem, x25519
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305

from ..params import (HPKE_AEAD_CHACHA20POLY1305, HPKE_KDF_HKDF_SHA256, HPKE_KEM_DHKEM_P256,
                      HPKE_KEM_DHKEM_X25519, HPKE_KEM_MLKEM768)

MODE_BASE = 0x00
NH = 32          # HKDF-SHA256
NK, NN, NT = 32, 12, 16   # ChaCha20-Poly1305


def i2osp(x: int, n: int) -> bytes:
    return x.to_bytes(n, "big")


# --------------------------------------------------------------------------
# Labelled HKDF-SHA256 (RFC 9180, Section 4)
# --------------------------------------------------------------------------
def _hmac(key: bytes, data: bytes) -> bytes:
    h = hmac.HMAC(key if key else b"\x00" * NH, hashes.SHA256())
    h.update(data)
    return h.finalize()


def hkdf_extract(salt: bytes, ikm: bytes) -> bytes:
    return _hmac(salt, ikm)


def hkdf_expand(prk: bytes, info: bytes, length: int) -> bytes:
    out, t, i = b"", b"", 1
    while len(out) < length:
        t = _hmac(prk, t + info + bytes([i]))
        out += t
        i += 1
    return out[:length]


def labeled_extract(suite_id: bytes, salt: bytes, label: bytes, ikm: bytes) -> bytes:
    return hkdf_extract(salt, b"HPKE-v1" + suite_id + label + ikm)


def labeled_expand(suite_id: bytes, prk: bytes, label: bytes, info: bytes, length: int) -> bytes:
    return hkdf_expand(prk, i2osp(length, 2) + b"HPKE-v1" + suite_id + label + info, length)


# --------------------------------------------------------------------------
# KEMs
# --------------------------------------------------------------------------
class Kem:
    kem_id: int

    def generate(self) -> tuple[object, bytes]:
        """Returns (secret key, serialized public key)."""
        raise NotImplementedError

    def encap(self, pk: bytes) -> tuple[bytes, bytes]:
        """Returns (shared_secret, enc)."""
        raise NotImplementedError

    def decap(self, enc: bytes, sk: object) -> bytes:
        raise NotImplementedError


class MlKem768(Kem):
    """ML-KEM-768 as in draft-ietf-hpke-pq-05: the shared_secret is the ML-KEM shared key itself."""

    kem_id = HPKE_KEM_MLKEM768
    n_enc = 1088

    def generate(self):
        sk = mlkem.MLKEM768PrivateKey.generate()
        return sk, sk.public_key().public_bytes_raw()

    def encap(self, pk: bytes):
        ss, ct = mlkem.MLKEM768PublicKey.from_public_bytes(pk).encapsulate()
        return ss, ct

    def decap(self, enc: bytes, sk):
        return sk.decapsulate(enc)          # implicit rejection: always returns a key


class _Dhkem(Kem):
    def _suite(self) -> bytes:
        return b"KEM" + i2osp(self.kem_id, 2)

    def _extract_and_expand(self, dh: bytes, kem_context: bytes) -> bytes:
        prk = labeled_extract(self._suite(), b"", b"eae_prk", dh)
        return labeled_expand(self._suite(), prk, b"shared_secret", kem_context, 32)


class DhkemP256(_Dhkem):
    kem_id = HPKE_KEM_DHKEM_P256
    n_enc = 65

    @staticmethod
    def _ser(pub) -> bytes:
        return pub.public_bytes(serialization.Encoding.X962, serialization.PublicFormat.UncompressedPoint)

    def generate(self):
        sk = ec.generate_private_key(ec.SECP256R1())
        return sk, self._ser(sk.public_key())

    def encap(self, pk: bytes):
        pkr = ec.EllipticCurvePublicKey.from_encoded_point(ec.SECP256R1(), pk)
        ske = ec.generate_private_key(ec.SECP256R1())
        dh = ske.exchange(ec.ECDH(), pkr)
        enc = self._ser(ske.public_key())
        return self._extract_and_expand(dh, enc + pk), enc

    def decap(self, enc: bytes, sk):
        pke = ec.EllipticCurvePublicKey.from_encoded_point(ec.SECP256R1(), enc)
        dh = sk.exchange(ec.ECDH(), pke)
        return self._extract_and_expand(dh, enc + self._ser(sk.public_key()))


class DhkemX25519(_Dhkem):
    kem_id = HPKE_KEM_DHKEM_X25519
    n_enc = 32

    @staticmethod
    def _ser(pub) -> bytes:
        return pub.public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)

    def generate(self):
        sk = x25519.X25519PrivateKey.generate()
        return sk, self._ser(sk.public_key())

    def encap(self, pk: bytes):
        ske = x25519.X25519PrivateKey.generate()
        dh = ske.exchange(x25519.X25519PublicKey.from_public_bytes(pk))
        enc = self._ser(ske.public_key())
        return self._extract_and_expand(dh, enc + pk), enc

    def decap(self, enc: bytes, sk):
        dh = sk.exchange(x25519.X25519PublicKey.from_public_bytes(enc))
        return self._extract_and_expand(dh, enc + self._ser(sk.public_key()))


class DhkemX25519Elligator(DhkemX25519):
    """BL4-E: DHKEM(X25519) where enc is the Elligator 2 representative (32 B, close to uniform) of the ephemeral key.

    The ephemeral key is generated by Monocypher's crypto_elligator_key_pair (a "dirty" key, so that representatives
    cover every point). The low-order component vanishes because the receiver's secret key is clamped, so the DH value
    equals that of the clean key. kem_context uses the true public key pkE = elligator_map(enc), which keeps the DHKEM
    semantics of RFC 9180.
    """

    def encap(self, pk: bytes):
        import monocypher
        hidden, ske_bytes = monocypher.elligator_key_pair()
        ske = x25519.X25519PrivateKey.from_private_bytes(ske_bytes)
        pke = monocypher.elligator_map(hidden)
        dh = ske.exchange(x25519.X25519PublicKey.from_public_bytes(pk))
        return self._extract_and_expand(dh, pke + pk), hidden

    def decap(self, enc: bytes, sk):
        import monocypher
        pke = monocypher.elligator_map(enc)
        dh = sk.exchange(x25519.X25519PublicKey.from_public_bytes(pke))
        return self._extract_and_expand(dh, pke + self._ser(sk.public_key()))


# --------------------------------------------------------------------------
# Key schedule and encryption context
# --------------------------------------------------------------------------
@dataclass
class Context:
    suite_id: bytes
    key: bytes
    base_nonce: bytes
    exporter_secret: bytes

    def _nonce(self, seq: int) -> bytes:
        s = i2osp(seq, NN)
        return bytes(a ^ b for a, b in zip(self.base_nonce, s))

    def seal(self, seq: int, aad: bytes, pt: bytes) -> bytes:
        """Seal with an explicit seq (nonce = base_nonce XOR seq), as in RFC 9180 Section 5.2.

        Used ONLY for the interoperability test against another RFC 9180 implementation (tests/test_hpke.py). PQ-PKS and
        the baselines do NOT use Seal/Open: RFC 9180 does not allow out-of-order opening (Sections 5.2 and 9.1); the
        systems use one-time AEAD keys obtained from export() for each image (systems/base.py)."""
        return ChaCha20Poly1305(self.key).encrypt(self._nonce(seq), pt, aad)

    def open(self, seq: int, aad: bytes, ct: bytes) -> bytes | None:
        try:
            return ChaCha20Poly1305(self.key).decrypt(self._nonce(seq), ct, aad)
        except Exception:
            return None

    def export(self, exporter_context: bytes, length: int) -> bytes:
        return labeled_expand(self.suite_id, self.exporter_secret, b"sec", exporter_context, length)


def _suite_id(kem: Kem) -> bytes:
    return b"HPKE" + i2osp(kem.kem_id, 2) + i2osp(HPKE_KDF_HKDF_SHA256, 2) + i2osp(HPKE_AEAD_CHACHA20POLY1305, 2)


def key_schedule(kem: Kem, shared_secret: bytes, info: bytes) -> Context:
    sid = _suite_id(kem)
    psk_id_hash = labeled_extract(sid, b"", b"psk_id_hash", b"")
    info_hash = labeled_extract(sid, b"", b"info_hash", info)
    ctx = bytes([MODE_BASE]) + psk_id_hash + info_hash
    secret = labeled_extract(sid, shared_secret, b"secret", b"")
    return Context(
        suite_id=sid,
        key=labeled_expand(sid, secret, b"key", ctx, NK),
        base_nonce=labeled_expand(sid, secret, b"base_nonce", ctx, NN),
        exporter_secret=labeled_expand(sid, secret, b"exp", ctx, NH),
    )


def setup_base_s(kem: Kem, pk_r: bytes, info: bytes) -> tuple[bytes, Context]:
    ss, enc = kem.encap(pk_r)
    return enc, key_schedule(kem, ss, info)


def setup_base_r(kem: Kem, enc: bytes, sk_r, info: bytes) -> Context | None:
    try:
        ss = kem.decap(enc, sk_r)
    except Exception:
        return None
    return key_schedule(kem, ss, info)
