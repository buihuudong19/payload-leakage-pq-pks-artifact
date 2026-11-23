"""DCT domain for S3 (JPEG QF75, hypothesis H8). src/ is not modified (frozen profile profile_frozen_v1).

Cover: grayscale PGM compressed to JPEG QF75 with libjpeg (PIL, standard quantization tables, no chroma subsampling). The embedder
works on the set U = the non-zero AC coefficients of the cover (blocks in row order, row order inside a block), seen as a one-row
"image" (1, |U|) of coefficient values; everything else of the two-stage A0/A1 layout (public permutation by the size (1, |U|),
STC with the public H_hat table, keyed stage 2) is as in the spatial domain (src/pqpks/stego/two_stage.py), except:
  - costs: J-UNIWARD (conseal) for +1 and -1 on each coefficient, wet outside U;
  - no shrinkage to 0: a coefficient +1 may not take -1 and a coefficient -1 may not take +1, so U of the stego = U of the cover and
    the receiver recovers U from the stego image alone (Kerckhoffs: U is public);
  - the carrier bit is the parity of the coefficient (c & 1, negative coefficients included).
Payload: the total number of bytes Q is fixed for every system of a cell (Q2048, Q3072), bucket of a system = Q - |kem_part| - 16.
The actual rate of an image = 8Q/|U| bpnzAC, recorded per image. Images with |U| < 8Q/0.5 (bpnzAC > 0.5) are excluded from that
cell (recorded in failed_combos). Warden W-img: DCTR [HF15] (8000 dimensions) and GFR (17 000 dimensions), computed on the
decompressed image; W-pay: LLR on the publicly extracted kem_part. W-A0 (SPAM split by A0) does not exist in the DCT domain.
"""
from __future__ import annotations

import io
import secrets
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from pqpks.params import CONTEXT, DEFAULT_PP, PublicParams  # noqa: E402
from pqpks.stego import partition, stc  # noqa: E402
from pqpks.stego.two_stage import _bits, _bytes, _h0_seed  # noqa: E402

QF = 75                               # default (S3); phase E also runs QF = 95 via --qf
MAX_BPNZAC = 0.5                      # an image qualifies for cell Q only when 8Q/|U| <= 0.5


def dctr_q(qf: int) -> float:
    """DCTR quantization step by QF (sealwatch.dctr.extract_from_file, Eq. 10)."""
    return max(8 * (2 - qf / 50), 0.2)


DCTR_Q = dctr_q(QF)


class JpegCover:
    """JPEG cover: quantized DCT coefficients Y (nb_v, nb_h, 8, 8), table qt, decompressed image, and the set U."""

    def __init__(self, Y: np.ndarray, qt: np.ndarray, spatial: np.ndarray, qf: int = QF):
        self.Y = np.asarray(Y, dtype=np.int16)
        self.qt = qt
        self.spatial = spatial
        self.qf = qf
        flat = self.Y.reshape(-1)
        ac = np.ones(self.Y.shape, dtype=bool)
        ac[:, :, 0, 0] = False
        self.U = np.nonzero((flat != 0) & ac.reshape(-1))[0]        # flat indices of the non-zero AC coefficients
        self.n_u = int(self.U.size)

    @classmethod
    def from_gray(cls, x: np.ndarray, qf: int = QF) -> "JpegCover":
        """Compress at QF (default 75) with libjpeg through PIL, then read the DCT with jpeglib (same bytes, hence the same coefficients)."""
        import jpeglib
        from PIL import Image
        buf = io.BytesIO()
        Image.fromarray(np.asarray(x, dtype=np.uint8)).save(buf, format="JPEG", quality=int(qf), subsampling=0)
        return cls.from_bytes(buf.getvalue(), qf=qf)

    @classmethod
    def from_bytes(cls, data: bytes, qf: int = QF) -> "JpegCover":
        import jpeglib
        import tempfile, os
        fd, p = tempfile.mkstemp(suffix=".jpg")
        os.close(fd)
        try:
            Path(p).write_bytes(data)
            d = jpeglib.read_dct(p)
            s = jpeglib.read_spatial(p, jpeglib.JCS_GRAYSCALE).spatial
        finally:
            os.unlink(p)
        s = s[..., 0] if s.ndim == 3 else s
        return cls(d.Y[..., :, :] if d.Y.ndim == 4 else d.Y, d.qt, np.asarray(s, dtype=np.uint8), qf=qf)

    def values(self) -> np.ndarray:
        """(1, |U|) representation passed to the embedder / system (StcSystem checks the capacity with shape[0] * shape[1])."""
        return self.Y.reshape(-1)[self.U].reshape(1, -1).copy()

    def with_values(self, vals: np.ndarray) -> "JpegCover":
        """Stego: write the new values into U and decompress again (through libjpeg) to compute features. U of the stego = U of the cover."""
        import jpeglib
        import tempfile, os
        Y = self.Y.copy().reshape(-1)
        Y[self.U] = np.asarray(vals, dtype=np.int16).reshape(-1)
        Y = Y.reshape(self.Y.shape)
        fd, p = tempfile.mkstemp(suffix=".jpg")
        os.close(fd)
        try:
            jpeglib.from_dct(Y=Y, qt=self.qt).write_dct(p)
            s = jpeglib.read_spatial(p, jpeglib.JCS_GRAYSCALE).spatial
        finally:
            os.unlink(p)
        s = s[..., 0] if s.ndim == 3 else s
        out = JpegCover(Y, self.qt, np.asarray(s, dtype=np.uint8), qf=self.qf)
        assert np.array_equal(out.U, self.U), "the set U changed after embedding (violates the no-shrinkage rule)"
        return out

    def juniward_cost(self):
        """(rho, +1 allowed, -1 allowed) on U from J-UNIWARD; no shrinkage to 0; wet = inf."""
        import conseal
        rp, rm = conseal.juniward.compute_cost_adjusted(x0=self.spatial.astype(np.float64), y0=self.Y, qt=self.qt[0])
        v = self.Y.reshape(-1)[self.U].astype(np.int64)
        rp, rm = rp.reshape(-1)[self.U], rm.reshape(-1)[self.U]
        plus_ok = np.isfinite(rp) & (v != -1)
        minus_ok = np.isfinite(rm) & (v != 1)
        rho = np.minimum(np.where(plus_ok, rp, np.inf), np.where(minus_ok, rm, np.inf))
        return rho.astype(np.float64), plus_ok, minus_ok

    def bpnzac(self, total_bytes: int) -> float:
        return 8 * total_bytes / self.n_u

    def dctr(self) -> np.ndarray:
        import sealwatch as sw
        f = sw.dctr.extract(x=self.spatial.astype(np.float64) - 128, q=dctr_q(self.qf))
        return np.concatenate([np.ravel(a) for a in f.values()]).astype(np.float32)

    def gfr(self) -> np.ndarray:
        import sealwatch as sw
        f = sw.gfr.extract(x=self.spatial.astype(np.float64) - 128)
        return np.concatenate([np.ravel(a) for a in f.values()]).astype(np.float32)


class JpegTwoStageSTC:
    """Two-stage embedder on (1, |U|) coefficient values; the same interface as TwoStageSTC (embed, extract_kem, a0_mask,
    stage2_index, extract_body, cost_map), so that it can replace the `embedder` attribute of StcSystem."""

    def __init__(self, pp: PublicParams = DEFAULT_PP, keyed_perm: bool = True, keyed_matrix: bool = True):
        self.pp = pp
        self.cost = "juniward"
        self.keyed_perm, self.keyed_matrix = keyed_perm, keyed_matrix
        self.stage2_keyed = keyed_perm and keyed_matrix
        self._cover: JpegCover | None = None       # current cover (to compute costs when none are supplied)

    def _sets(self, shape, kem_bits, body_bits=None):
        h, w = shape
        if kem_bits is None:
            return None, np.arange(h * w, dtype=np.int64)
        if self.pp.layout == "A8" and body_bits is None:
            raise ValueError("layout A8: the body length (bucket) is needed to determine A0(B)")
        return partition.partition_n0(h, w, CONTEXT, self.pp.n0(h * w, kem_bits, body_bits or 0))

    def a0_mask(self, shape, kem_bytes: int, body_bytes: int | None = None) -> np.ndarray:
        a0, _ = self._sets(shape, 8 * kem_bytes, None if body_bytes is None else 8 * body_bytes)
        m = np.zeros(shape[0] * shape[1], dtype=bool)
        m[a0] = True
        return m.reshape(shape)

    def _stage2_index(self, a1, k_pi):
        if not self.keyed_perm:
            return a1
        return a1[partition.fisher_yates(a1.size, k_pi)]

    def _stage2_seed(self, shape, k_h):
        return k_h if self.keyed_matrix else _h0_seed(*shape)

    def cost_map(self, cover):
        if isinstance(cover, JpegCover):
            return cover.juniward_cost()
        if self._cover is None:
            raise ValueError("JpegTwoStageSTC.cost_map: a JpegCover is needed (or supply cost_map)")
        return self._cover.juniward_cost()

    def embed(self, cover, kem_part, body, k_pi, k_h, cost_map=None):
        x = np.asarray(cover, dtype=np.int16)
        flat = x.reshape(-1).astype(np.int32)
        lsb = (flat & 1).astype(np.uint8)
        rho, plus_ok, minus_ok = cost_map if cost_map is not None else self.cost_map(cover)
        new_lsb = lsb.copy()
        kem_bits = None if kem_part is None else 8 * len(kem_part)
        a0, a1 = self._sets(x.shape, kem_bits, 8 * len(body))
        if kem_part is not None:
            y0, _ = stc.embed(lsb[a0], rho[a0], _bits(kem_part), _h0_seed(*x.shape), self.pp.stc_h)
            new_lsb[a0] = y0
        idx1 = self._stage2_index(a1, k_pi)
        y1, _ = stc.embed(lsb[idx1], rho[idx1], _bits(body), self._stage2_seed(x.shape, k_h), self.pp.stc_h)
        new_lsb[idx1] = y1
        for p in np.nonzero(new_lsb != lsb)[0]:
            if plus_ok[p] and minus_ok[p]:
                flat[p] += 1 if secrets.randbits(1) else -1
            elif plus_ok[p]:
                flat[p] += 1
            elif minus_ok[p]:
                flat[p] -= 1
            else:
                raise ValueError("STC chose to modify a coefficient that may not be modified")
        return flat.astype(np.int16).reshape(x.shape)

    def extract_kem(self, stego, kem_bytes: int, body_bytes: int | None = None) -> bytes:
        y = np.asarray(stego, dtype=np.int16)
        a0, _ = self._sets(y.shape, 8 * kem_bytes, None if body_bytes is None else 8 * body_bytes)
        lsb = (y.reshape(-1).astype(np.int32) & 1).astype(np.uint8)
        return _bytes(stc.extract(lsb[a0], 8 * kem_bytes, _h0_seed(*y.shape), self.pp.stc_h))

    def stage2_index(self, shape, kem_bytes, k_pi, body_bytes=None):
        _, a1 = self._sets(shape, None if kem_bytes is None else 8 * kem_bytes,
                           None if body_bytes is None else 8 * body_bytes)
        return self._stage2_index(a1, k_pi)

    def extract_body(self, stego, idx1, body_bytes: int, k_h: bytes) -> bytes | None:
        y = np.asarray(stego, dtype=np.int16)
        lsb = (y.reshape(-1).astype(np.int32) & 1).astype(np.uint8)
        if idx1.size < 8 * body_bytes:
            return None
        return _bytes(stc.extract(lsb[idx1], 8 * body_bytes, self._stage2_seed(y.shape, k_h), self.pp.stc_h))
