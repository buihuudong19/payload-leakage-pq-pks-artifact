"""The two systems with the sequential MPVD embedder: BL1 and BL2 (compared at N = 1).

BL1: EC-ElGamal P-256 + the [RS16] mapping + decimal ASCII + a "len " header + MPVD.
    Runs the BL1 reference implementation unchanged, except that its RNG is replaced by a CSPRNG. The reference code is
    not redistributed: place it in baselines/bl1_reference/ or set PQPKS_BL1_REFERENCE_DIR (see baselines/README.md).
    The message must be text (a limitation of the original system).
BL2: HPKE-ML-KEM-768, kem_part = raw enc of 1088 B, AEAD body under a one-time key k_ae,1 obtained with HPKE Export
    (bound to profile_id); sequential MPVD embedding without a header (the length is in the framing). The frame is the
    one of a PQ-PKS first image (with t0), so that the bit budgets are comparable.
"""
from __future__ import annotations

import contextlib
import os
import secrets
import sys
import tempfile
import time
import types
from pathlib import Path

import numpy as np

from ..crypto import framing, hpke
from ..params import AEAD_TAG_BYTES, DEFAULT_PP, FLAG_SESSION_START, PublicParams
from ..stego import mpvd
from .base import HpkeCrypto, ReceiveResult, image_keys, open_image, profile_id_for, seal_image

_BL1_REF = Path(os.environ.get("PQPKS_BL1_REFERENCE_DIR") or Path(__file__).resolve().parents[3] / "baselines" / "bl1_reference")


def _bits(b: bytes) -> str:
    return "".join(format(x, "08b") for x in b)


def _from_bits(s: str) -> bytes:
    return bytes(int(s[i:i + 8], 2) for i in range(0, len(s) - len(s) % 8, 8))


# ======================================================================== BL2
class Bl2System:
    name = "BL2"

    def __init__(self, pp: PublicParams = DEFAULT_PP):
        self.pp = pp
        self.crypto = HpkeCrypto("BL2", hpke.MlKem768(), kem_part_bytes=1088)
        self.pid = profile_id_for(self.crypto, pp)

    def keygen(self):
        return self.crypto.keygen()

    def payload(self, ek_r: bytes, msg: bytes) -> tuple[bytes, bytes]:
        kem_part, ctx, sid = self.crypto.start(ek_r, self.pid)
        pt = framing.pack(framing.Frame(msg, FLAG_SESSION_START, None, int(time.time())), self.pp.buckets)
        return kem_part, seal_image(image_keys(ctx, self.pid, sid, 1), self.pid, sid, 1, pt)

    def send_once(self, ek_r: bytes, msg: bytes, cover: np.ndarray) -> np.ndarray:
        kem_part, body = self.payload(ek_r, msg)
        return mpvd.embed(cover, _bits(kem_part + body))

    def receive_once(self, dk_r, stego: np.ndarray) -> ReceiveResult:
        need = 8 * (1088 + max(self.pp.buckets) + AEAD_TAG_BYTES)
        stream = _from_bits(mpvd.extract(stego, need))
        acc = self.crypto.accept(dk_r, stream[:1088], self.pid)
        if acc is None:
            return ReceiveResult(None, "bottom")
        ctx, sid = acc
        keys = image_keys(ctx, self.pid, sid, 1)
        for b in self.pp.buckets:
            pt = open_image(keys, self.pid, sid, 1, stream[1088:1088 + b + AEAD_TAG_BYTES])
            if pt is not None:
                fr = framing.unpack(pt, self.pp.sig_bytes)
                return ReceiveResult(fr.msg if fr else None, "ok" if fr else "bottom", sid, 1)
        return ReceiveResult(None, "bottom")


# ======================================================================== BL1
@contextlib.contextmanager
def _in_tempdir():
    """The reference code writes a file (Send.txt) into the working directory: run it in a temporary directory."""
    old = os.getcwd()
    with tempfile.TemporaryDirectory() as d:
        os.chdir(d)
        try:
            yield
        finally:
            os.chdir(old)


def _load_bl1_reference():
    if not (_BL1_REF / "ECCurve.py").is_file():
        from ..data import MissingResource
        raise MissingResource(f"BL1 reference code not found in {_BL1_REF} (see baselines/README.md)")
    tk = types.ModuleType("tkinter")
    mb = types.ModuleType("tkinter.messagebox")
    mb.showerror = mb.showinfo = lambda *a, **k: None
    tk.messagebox, tk.filedialog = mb, types.ModuleType("tkinter.filedialog")
    sys.modules.setdefault("tkinter", tk)
    sys.modules.setdefault("tkinter.messagebox", mb)
    sys.modules.setdefault("tkinter.filedialog", tk.filedialog)
    if str(_BL1_REF) not in sys.path:
        sys.path.append(str(_BL1_REF))
    import ECCurve
    import ECdecrypt
    import ECencrypt
    ECCurve.random = secrets.SystemRandom()       # the only permitted change: RNG -> CSPRNG
    return ECCurve, ECencrypt, ECdecrypt


class Bl1System:
    name = "BL1"

    def __init__(self):
        self.ECCurve, self.ECencrypt, self.ECdecrypt = _load_bl1_reference()

    def keygen(self):
        sk = self.ECCurve.Egal.genPrivate()
        return sk, self.ECCurve.Egal.genPublic(sk)          # (sk, public key point)

    def payload(self, pk, text: str) -> str:
        """Text string that is embedded, exactly as in the original application: str(len(s)) + ' ' + s, s = decimal ciphertext."""
        with _in_tempdir():
            pm = self.ECencrypt.MappingMsg(text)
            self.ECencrypt.Mahoa(pk, pm)
            s = self.ECencrypt.ReturnString()
        return str(len(s)) + " " + s

    def send_once(self, pk, text: str, cover: np.ndarray) -> np.ndarray:
        return mpvd.embed(cover, "".join(format(ord(c), "08b") for c in self.payload(pk, text)))

    def receive_once(self, sk, stego: np.ndarray) -> ReceiveResult:
        # Parse the "len " header as the original extraction code does: digits followed by a space.
        head = mpvd.extract(stego, 8 * 12)
        chars = "".join(chr(int(head[i:i + 8], 2)) for i in range(0, len(head), 8))
        digits = chars.split(" ")[0]
        if not digits.isdigit():
            return ReceiveResult(None, "bottom")
        total = 8 * (len(digits) + 1 + int(digits))
        bits = mpvd.extract(stego, total)
        s = "".join(chr(int(bits[i:i + 8], 2)) for i in range(0, len(bits), 8))[len(digits) + 1:]
        try:
            pts = self.ECdecrypt.DecryptMsg(self.ECdecrypt.ModifMsg(s.split()), sk)
            text = self.ECdecrypt.PointtoMsg(pts)
        except Exception:
            return ReceiveResult(None, "bottom")
        return ReceiveResult(text.encode("latin-1"), "ok")
