"""Public parameters (pp) of PQ-PKS.

Every value in this module is PUBLIC and fixed by the protocol version ("PQPKS-v1").
"""
from __future__ import annotations

from dataclasses import dataclass, field

CONTEXT = b"PQPKS-v1"

# --- ML-KEM-768 (FIPS 203) -------------------------------------------------
MLKEM_Q = 3329
MLKEM_N = 256
MLKEM768_K = 3
MLKEM768_DU = 10
MLKEM768_DV = 4
MLKEM768_EK_BYTES = 1184
MLKEM768_CT_BYTES = 1088

# --- Kemeleon (draft-irtf-cfrg-kemeleon-02, main variant) ----------------
KEMELEON_T = 76            # statistical distance parameter (2^-76)
KEMELEON_B = 2996          # ceil(n * log2 q) with n = 256, q = 3329
KEMELEON_VEC_BYTES = (KEMELEON_B + KEMELEON_T) // 8   # 384 bytes per vector
KEMELEON_EK_BYTES = MLKEM768_K * KEMELEON_VEC_BYTES + 32          # 1184
KEMELEON_CT_BYTES = (MLKEM768_K + 1) * KEMELEON_VEC_BYTES         # 1536
KEMELEON_CT_BITS = 8 * KEMELEON_CT_BYTES                          # 12 288

# --- Kemeleon, rejection-sampled variant (draft-02 Section 5.1, EncodeCtxtR; ablation A4) --------
KEMELEON_R_INT_BITS = 8986          # b = floor(log2 q^(k*n)) with k = 3: bit length of r when msb(r) = 0
KEMELEON_R_INT_BYTES = (KEMELEON_R_INT_BITS + 1 + 7) // 8          # 1124 bytes (the 6 top bits are random)
KEMELEON_R_CT_BYTES = KEMELEON_R_INT_BYTES + MLKEM_N * MLKEM768_DV // 8   # 1124 + 128 = 1252

# --- HPKE (RFC 9180; ML-KEM theo draft-ietf-hpke-pq-05) --------------------
HPKE_KEM_MLKEM768 = 0x0041
HPKE_KEM_DHKEM_P256 = 0x0010
HPKE_KEM_DHKEM_X25519 = 0x0020
HPKE_KDF_HKDF_SHA256 = 0x0001
HPKE_AEAD_CHACHA20POLY1305 = 0x0003
AEAD_TAG_BYTES = 16

# --- ML-DSA (FIPS 204): ML-DSA-65 by default, ML-DSA-44 in ablation A5 ----
MLDSA44_SIG_BYTES = 2420
MLDSA44_VK_BYTES = 1312
MLDSA65_SIG_BYTES = 3309
MLDSA65_VK_BYTES = 1952
SIG_BYTES = {"ML-DSA-44": MLDSA44_SIG_BYTES, "ML-DSA-65": MLDSA65_SIG_BYTES}

# --- Plaintext framing ------------------------------------------------------
FRAME_VERSION = 1
FLAG_HAS_SIG = 0x01
FLAG_SESSION_START = 0x02
FLAG_REKEY = 0x04
KNOWN_FLAGS = FLAG_HAS_SIG | FLAG_SESSION_START | FLAG_REKEY      # every other flag bit must be 0
FRAME_HEADER_BYTES = 1 + 1 + 4          # ver | flags | msg_len
T0_BYTES = 8                            # timestamp t0 (seconds, u64), first image of a session only
START_HEADER_BYTES = FRAME_HEADER_BYTES + T0_BYTES


@dataclass(frozen=True)
class PublicParams:
    """Public parameter set pp."""

    stc_h: int = 10                                   # STC constraint height
    alpha0: float = 0.15                              # fixed alpha0, used only by layout A8' (ablation)
    layout: str = "A8"                                # "A8": alpha0(B) = l_start(B)/n (default); "A8p": fixed alpha0
    buckets: tuple[int, ...] = (256, 512, 1024, 2048, 4096, 8192)   # |pt| in bytes (ladder of first images)
    ladder: str = "merged"                            # ladder of later images: "same" = B; "shifted" = B + |c_hat|;
                                                      # "merged" = B U (B + |c_hat|) (default)
    sig_placement: str = "image1"                     # sigma in image 1 (default) or in image 2 ("image2", ablation A5b)
    window: int = 8                                   # window W of image indices tried by the receiver
    max_open_sessions: int = 16                       # S_max
    max_pending_sessions: int = 16                    # P_max: sessions waiting for the signature in image 2 (A5b, require_auth); receiver parameter
    session_len: int = 8                              # default N (selected again from E5)
    sid_ttl_s: float = 30 * 24 * 3600                 # TTL: acceptance window of t0 and minimum lifetime of a seen sid
    max_skew_s: float = 300.0                         # Delta_skew: maximum clock skew between S and R (RFC 4120 uses 5 minutes as an example)
    sig_alg: str = "ML-DSA-65"                        # signature scheme of the profile
    max_cover_tries: int = 8                          # R_cover: maximum number of covers tried for one body
    context: bytes = field(default=CONTEXT)

    @property
    def sig_bytes(self) -> int:
        return SIG_BYTES[self.sig_alg]

    def profile_fields(self) -> tuple:
        """Public fields of pp that enter profile_id (canonical encoding CE). TTL, Delta_skew and S_max are receiver
        parameters and R_cover is a sender parameter; none of them belongs to the profile."""
        from .crypto.profile import ce
        return (self.context.decode(), self.sig_alg, self.stc_h, self.layout, self.alpha0, ce(*self.buckets),
                self.window, self.session_len, self.ladder, self.sig_placement)

    def __post_init__(self):
        if self.layout not in ("A8", "A8p"):
            raise ValueError(f"invalid layout: {self.layout!r} (expected 'A8' or 'A8p')")
        if self.ladder not in ("same", "shifted", "merged"):
            raise ValueError(f"invalid bucket ladder: {self.ladder!r} (expected 'same', 'shifted' or 'merged')")
        if self.sig_placement not in ("image1", "image2"):
            raise ValueError(f"invalid signature placement: {self.sig_placement!r} (expected 'image1' or 'image2')")

    def buckets_for(self, start: bool, kem_bytes: int) -> tuple[int, ...]:
        """Bucket ladder of an image type. First images always use `buckets`.
        - "same": later images use the same ladder; the space left by c_hat usually becomes padding.
        - "shifted": later images use B + |c_hat|; the c_hat space is used and the number of buckets is unchanged, but a
          short message in a later image is still padded to at least 256 + |c_hat| bytes.
        - "merged" (default): B U (B + |c_hat|); keeps both advantages, at the price of more buckets tried by the receiver
          for later images (11 instead of 6 with the default B and |c_hat| = 1536 bytes)."""
        if start or self.ladder == "same":
            return self.buckets
        shifted = tuple(b + kem_bytes for b in self.buckets)
        if self.ladder == "shifted":
            return shifted
        return tuple(sorted(set(self.buckets) | set(shifted)))

    def n0(self, n: int, kem_bits: int, body_bits: int) -> int:
        """Number of pixels of A0 for a first image of n pixels, a kem_part of kem_bits bits and a body of body_bits bits.

        A8: alpha0(B) = l_start(B)/n with l_start(B) = l_c + 8(B + 16), hence n0 = ceil(l_c*n/l_start), and the change
        densities in A0 and A1 are equal up to rounding (at most one pixel).
        A8': n0 = ceil(l_c/alpha0) with a fixed alpha0. Computed with integers only."""
        if self.layout == "A8":
            total = kem_bits + body_bits
            if total > n:
                raise ValueError(f"an image of {n} pixels cannot hold l_start = {total} bits")
            return -(-kem_bits * n // total)
        import math
        n0 = math.ceil(kem_bits / self.alpha0)
        if n0 >= n:
            raise ValueError(f"image too small for stage 1: need n0 = {n0} < n = {n}")
        return n0


DEFAULT_PP = PublicParams()
