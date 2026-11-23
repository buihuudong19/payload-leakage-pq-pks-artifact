"""Factory of the compared systems.

| Name    | Cryptographic layer (HPKE base, KDF HKDF-SHA256, AEAD ChaCha20-Poly1305) | kem_part  | Embedder      |
|---------|------------------------------------------------------------------|-----------------|---------------|
| BL1     | EC-ElGamal P-256 + [RS16] (reference implementation)             | decimal ASCII   | sequential MPVD |
| BL2     | ML-KEM-768                                                       | raw 1088 B      | sequential MPVD |
| BL3     | DHKEM(P-256)                                                     | 65-byte point   | two-stage HILL-STC |
| BL4     | DHKEM(P-256) + point encoding of [Zha25] (SWU admissible encoding, LT = 32) | 72 B, near uniform | two-stage HILL-STC |
| BL4-E   | DHKEM(X25519) + Elligator 2 (cross-check for BL4)                | 32 B, near uniform | two-stage HILL-STC |
| BL5     | ML-KEM-768                                                       | raw 1088 B      | two-stage HILL-STC |
| BL5-L   | ML-KEM-768, length-matched control (uniform padding up to |c_hat|) | 1088 + 448 B   | two-stage HILL-STC |
| PQ-PKS  | ML-KEM-768 + Kemeleon main variant (default)                     | 1536 B, near uniform | two-stage HILL-STC |
| PQ-PKS-R| ML-KEM-768 + Kemeleon rejection-sampled variant (ablation A4; re-encapsulates on rejection) | 1252 B | two-stage HILL-STC |
"""
from __future__ import annotations

import secrets

from ..crypto import hpke, kemeleon, zha25
from ..params import DEFAULT_PP, KEMELEON_CT_BYTES, KEMELEON_R_CT_BYTES, MLKEM768_CT_BYTES, PublicParams
from .base import HpkeCrypto
from .mpvd_systems import Bl1System, Bl2System
from .base import KemEncodingExhausted
from .stc_system import EmbedFailure, Inadmissible, ReceiverPolicy, ReceiverState, SenderSession, StcSystem


def crypto_layer(name: str) -> HpkeCrypto:
    if name == "PQ-PKS":
        return HpkeCrypto("PQ-PKS", hpke.MlKem768(), kemeleon.encode_ct, kemeleon.decode_ct, KEMELEON_CT_BYTES)
    if name == "PQ-PKS-R":
        return HpkeCrypto("PQ-PKS-R", hpke.MlKem768(), kemeleon.encode_ct_r, kemeleon.decode_ct_r, KEMELEON_R_CT_BYTES)
    if name == "BL5":
        return HpkeCrypto("BL5", hpke.MlKem768(), kem_part_bytes=MLKEM768_CT_BYTES)
    if name == "BL5-L":
        # Control: same length as the c_hat of PQ-PKS (default) and same A0/A1; only the distribution of the first 1088 B differs.
        pad = KEMELEON_CT_BYTES - MLKEM768_CT_BYTES
        return HpkeCrypto("BL5-L", hpke.MlKem768(), lambda enc: enc + secrets.token_bytes(pad),
                          lambda part: part[:MLKEM768_CT_BYTES], KEMELEON_CT_BYTES)
    if name == "BL3":
        return HpkeCrypto("BL3", hpke.DhkemP256(), kem_part_bytes=65)
    if name == "BL4":
        # Adaptation of [Zha25]: shared key from HPKE DHKEM(P-256) as in BL3; only the serialization of enc changes.
        return HpkeCrypto("BL4", hpke.DhkemP256(), zha25.encode_point, zha25.decode_point, zha25.KEM_PART_BYTES)
    if name == "BL4-E":
        return HpkeCrypto("BL4-E", hpke.DhkemX25519Elligator(), kem_part_bytes=32)
    raise KeyError(name)


def make(name: str, pp: PublicParams = DEFAULT_PP, **kw):
    if name == "BL1":
        return Bl1System()
    if name == "BL2":
        return Bl2System(pp)
    return StcSystem(crypto_layer(name), pp, **kw)


__all__ = ["make", "crypto_layer", "StcSystem", "SenderSession", "ReceiverState", "ReceiverPolicy", "EmbedFailure",
           "Inadmissible", "KemEncodingExhausted",
           "Bl1System", "Bl2System"]
