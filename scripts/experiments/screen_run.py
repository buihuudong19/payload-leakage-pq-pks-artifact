"""Screening round and confirmatory runs: generation of stego images and features for E1/E2/E3.

Data: `data.screening_covers()` = the 1000 BOSSbase images of the layout pilot + 1000 BOWS2 images (both inside the
training part of the split; the test set is NOT touched). The screening results are not confirmatory results.

For every cover x and every alpha in ALPHAS (regime R-b, single-bucket profile, layout A8):
  - real systems (StcSystem, real keys and sessions, random message filling the bucket): BL5, BL5-L, PQ-PKS;
  - REF-U: uniform payload with the same length and layout as PQ-PKS.
Recorded for every image (cover and stego):
  - SPAM-686 of the whole image (W-img), SPAM-686 on A0 and on the rest of the image (W-A0; 1372 dimensions);
  - the publicly extracted string Ext0 (stored in full for the LLR, the joint test and E1);
  - the fraction of changed pixels; the number of STC failures (p_fail); a decoding check of PQ-PKS on part of the images.
For E1, in addition: Ext0 of the cover at A0(B) for every bucket of the default ladder (layout A8, several buckets).

Run:  PQPKS_BOSSBASE_DIR=... PQPKS_BOWS2_DIR=... python scripts/experiments/screen_run.py --workers 2
The run can be resumed: covers that already have a result file are skipped.

Phase D adds three plans with the same result format:
  --plan e7      E7 (H7, Table S4): 10 000 ALASKA2 images (subset_ids.txt), alpha in {0.2, 0.4}, four systems, SRM;
  --plan ablate  S2 (ablations A1, A3, A4, A8'): 2000 screening covers, alpha in {0.2, 0.4}, PQ-PKS, REF-U and the variants
                 V1/V2/V3 (A1), SUNI and REF-U-SUNI (A3, S-UNIWARD), PQ-PKS-R (A4, 1252 B), A8p (A8', alpha_0 = 0.15);
  --plan ra      S5 (regime R-a): the same message of L in {1024, 4096} bytes for BL3, BL4, BL5, BL5-L, PQ-PKS, REF-U; each
                 system has its own alpha = 8(|kem_part| + B + 16)/n, recorded in run_meta.json; the cell labels are "L1024",
                 "L4096" instead of alpha.
  --plan ablate2 A7: hybrid ML-KEM-768 + X25519 with two separate encodings (Kemeleon 1536 B || Elligator 2 32 B = 1568 B), shared
                 key SHA3-256 of the two secrets (no combiner proof is claimed), length-matched control REF-U-1568; A9: STC
                 h = 8, 12 (H8, H12) with a separate public H_hat table per h (scripts/experiments/stc_tables/, same construction).
S3 (JPEG QF75) is in scripts/experiments/jpeg_run.py.
"""
from __future__ import annotations

import argparse
import json
import os
import secrets
import sys
import time
import traceback
from multiprocessing import Process
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from dataclasses import replace  # noqa: E402

from pqpks import data, systems  # noqa: E402
from pqpks.analysis import spam  # noqa: E402
from pqpks.crypto import hpke, kemeleon, zha25  # noqa: E402
from pqpks.stego import stc as stc_mod  # noqa: E402
from pqpks.systems.base import HpkeCrypto  # noqa: E402
from pqpks.experiment.profiles import message_len_for_bucket, profile_for_alpha  # noqa: E402
from pqpks.experiment.references import ReferenceSystem  # noqa: E402
from pqpks.params import (AEAD_TAG_BYTES, DEFAULT_PP, KEMELEON_CT_BYTES, KEMELEON_R_CT_BYTES, MLKEM768_CT_BYTES,  # noqa: E402
                          START_HEADER_BYTES)
from pqpks.stego.two_stage import TwoStageSTC  # noqa: E402
from confirm.common import warnings_to_stdout  # noqa: E402,F401  (importing the module suffices: warnings go to stdout, also in child processes)

ALPHAS = (0.1, 0.2, 0.4)
SYSTEMS = ("BL5", "BL5-L", "PQ-PKS", "REF-U")
SYSTEMS_AT = {0.1: ("PQ-PKS", "REF-U"), 0.2: SYSTEMS, 0.4: SYSTEMS}   # alpha = 0.1 is needed only for the P_E(alpha) curve and alpha*
# Confirmatory step: alpha = 0.3 is added for the P_E(alpha) curve of PQ-PKS and REF-U, together with the two operating rates of
# a signed session with sigma in image 2 (A5b): 0.0787 (ML-DSA-44) and 0.1099 (ML-DSA-65), so that the signature selection rule
# can be checked directly at its thresholds; and 0.0552, the smallest rate of a PQ-PKS session-start image (Kemeleon part
# 1536 B + bucket 256 B + tag 16 B), which is also the operating rate of FN-DSA-512 with sigma in image 2. The grid was frozen on
# 06/11/2026, before any confirmatory stego image was generated.
MIN_FIRST_IMAGE, OP_ML_DSA_44, OP_ML_DSA_65 = 0.0552, 0.0787, 0.1099
ABLATE_SYSTEMS = ("PQ-PKS", "REF-U", "V1", "V2", "V3", "SUNI", "REF-U-SUNI", "PQ-PKS-R", "A8p")
ABLATE2_SYSTEMS = ("PQ-PKS", "REF-U", "A7", "REF-U-1568", "H8", "H12")
A7_KEM_BYTES = KEMELEON_CT_BYTES + 32
RA_SYSTEMS = ("BL3", "BL4", "BL5", "BL5-L", "PQ-PKS", "REF-U")
# Phase E (exploratory, after the confirmatory alpha* = 0.0596 was found close to the floor 0.0552): the smallest session-start
# image of the rejection-sampled Kemeleon variant (A4, 1252 B): 1252 + 256 + 16 = 1524 B -> alpha = 0.04651 (cell label 0.04651:
# budget 1524 B, bucket exactly 256). Length-matched control: REF-U-1252 (1252 uniform bytes + uniform body). 20 000 confirmatory covers, SRM.
MIN_FIRST_IMAGE_R = 0.04651
CONFIRM_R_SYSTEMS = ("PQ-PKS-R", "REF-U-1252")
RA_MSG_LENS = (1024, 4096)                                    # message bytes of regime R-a (cell labels "L1024", "L4096")
PLANS = {"screen": (ALPHAS, SYSTEMS_AT),
         "confirm": ((MIN_FIRST_IMAGE, OP_ML_DSA_44, 0.1, OP_ML_DSA_65, 0.2, 0.3, 0.4),
                     {MIN_FIRST_IMAGE: ("PQ-PKS", "REF-U"), OP_ML_DSA_44: ("PQ-PKS", "REF-U"), 0.1: ("PQ-PKS", "REF-U"),
                      OP_ML_DSA_65: ("PQ-PKS", "REF-U"),
                      0.2: SYSTEMS, 0.3: ("PQ-PKS", "REF-U"), 0.4: SYSTEMS}),
         "e7": ((0.2, 0.4), {0.2: SYSTEMS, 0.4: SYSTEMS}),
         "ablate": ((0.2, 0.4), {0.2: ABLATE_SYSTEMS, 0.4: ABLATE_SYSTEMS}),
         "ra": (tuple(f"L{L}" for L in RA_MSG_LENS), {f"L{L}": RA_SYSTEMS for L in RA_MSG_LENS}),
         "ablate2": ((0.2, 0.4), {0.2: ABLATE2_SYSTEMS, 0.4: ABLATE2_SYSTEMS}),
         "confirm_r": ((MIN_FIRST_IMAGE_R,), {MIN_FIRST_IMAGE_R: CONFIRM_R_SYSTEMS})}
PLAN_COVERS = {"screen": "screen", "confirm": "confirm", "e7": "alaska2-e7", "ablate": "screen", "ra": "screen", "ablate2": "screen",
               "confirm_r": "confirm"}
KEM_BYTES = {"BL3": 65, "BL4": zha25.KEM_PART_BYTES, "BL5": MLKEM768_CT_BYTES, "BL5-L": KEMELEON_CT_BYTES,
             "PQ-PKS": KEMELEON_CT_BYTES, "REF-U": KEMELEON_CT_BYTES,
             "V1": KEMELEON_CT_BYTES, "V2": KEMELEON_CT_BYTES, "V3": KEMELEON_CT_BYTES, "SUNI": KEMELEON_CT_BYTES,
             "REF-U-SUNI": KEMELEON_CT_BYTES, "PQ-PKS-R": KEMELEON_R_CT_BYTES, "A8p": KEMELEON_CT_BYTES,
             "A7": A7_KEM_BYTES, "REF-U-1568": A7_KEM_BYTES, "H8": KEMELEON_CT_BYTES, "H12": KEMELEON_CT_BYTES,
             "REF-U-1252": KEMELEON_R_CT_BYTES}
COST_OF = {"SUNI": "suniward", "REF-U-SUNI": "suniward"}      # systems using S-UNIWARD (A3); all others use HILL
OWN_MASK = {"A8p"}                                            # layout other than the default A8: own A0 despite the same |kem_part|
STC_H_OF = {"H8": 8, "H12": 12}                               # A9: STC height other than the default (own H_hat table, see install_stc_tables)


class HybridKem(hpke.Kem):
    """A7: ML-KEM-768 || X25519 (Elligator 2) with two separate encodings. kem_id uses a private range (0xFFA7); shared key
    = SHA3-256("PQPKS-A7" || ss_mlkem || ss_x25519 || ct_mlkem || enc_x25519). No combiner proof is claimed (paper, Section 7.3
    and the supplementary material)."""

    kem_id = 0xFFA7
    n_enc = MLKEM768_CT_BYTES + 32

    def __init__(self):
        self.k1, self.k2 = hpke.MlKem768(), hpke.DhkemX25519Elligator()

    def generate(self):
        sk1, pk1 = self.k1.generate()
        sk2, pk2 = self.k2.generate()
        return (sk1, sk2), pk1 + pk2

    def _combine(self, ss1, ss2, enc):
        import hashlib
        return hashlib.sha3_256(b"PQPKS-A7" + ss1 + ss2 + enc).digest()

    def encap(self, pk: bytes):
        pk1, pk2 = pk[:-32], pk[-32:]
        ss1, c1 = self.k1.encap(pk1)
        ss2, c2 = self.k2.encap(pk2)
        enc = c1 + c2
        return self._combine(ss1, ss2, enc), enc

    def decap(self, enc: bytes, sk):
        sk1, sk2 = sk
        c1, c2 = enc[:MLKEM768_CT_BYTES], enc[MLKEM768_CT_BYTES:]
        return self._combine(self.k1.decap(c1, sk1), self.k2.decap(c2, sk2), enc)


def a7_crypto() -> HpkeCrypto:
    """kem_part = Kemeleon(ct_mlkem) || enc_x25519 (Elligator 2, already near uniform): 1536 + 32 = 1568 B."""
    def enc(e: bytes):
        k = kemeleon.encode_ct(e[:MLKEM768_CT_BYTES])
        return None if k is None else k + e[MLKEM768_CT_BYTES:]
    def dec(part: bytes):
        return kemeleon.decode_ct(part[:KEMELEON_CT_BYTES]) + part[KEMELEON_CT_BYTES:]
    return HpkeCrypto("A7", HybridKem(), enc, dec, A7_KEM_BYTES)


_STC_TABLES = {}


def install_stc_tables(root: Path = ROOT) -> None:
    """A9: for STC h != 10, use a public H_hat table built with the same procedure (scripts/build_stc_table.py --h 8|12) instead
    of a seeded random matrix (the fallback branch of stc.submatrix). The function of the stc module is replaced at run time,
    src/ is not modified; sender and receiver in the same process use the same table. h = 10 keeps the table of the profile."""
    import hashlib
    if _STC_TABLES:
        return
    for h in (8, 12):
        f = root / "scripts" / "experiments" / "stc_tables" / f"stc_table_h{h}.json"
        if f.exists():
            _STC_TABLES[h] = json.loads(f.read_text())
    orig = stc_mod.submatrix

    def submatrix(seed: bytes, h: int, w: int):
        t = _STC_TABLES.get(h)
        if t and str(w) in t["table"]:
            cands = t["table"][str(w)]
            idx = int.from_bytes(hashlib.sha3_256(b"PQPKS-v1/STC-select" + seed).digest()[:4], "big") % len(cands)
            return np.array(cands[idx], dtype=np.int64)
        return orig(seed, h, w)

    stc_mod.submatrix = submatrix
CHECK_EVERY = 25          # check PQ-PKS decoding on every 25th cover (correctness; E4 performs the full check)


# ---- ALASKA2 (E7): the loader lives here, not in src/pqpks/data.py, so that the frozen profile profile_frozen_v1 is unchanged (check_frozen)
def alaska2_pgm_dir() -> Path:
    """512x512 grayscale PGM written by scripts/data/verify_alaska2.py from the ALASKA2 TIFF files (file name: 5-digit index)."""
    env = os.environ.get("PQPKS_ALASKA2_DIR")
    p = Path(env) if env else ROOT / "data" / "external" / "alaska2" / "pgm"
    if not p.is_dir():
        raise FileNotFoundError(f"ALASKA2 (PGM) not found: {p} (set PQPKS_ALASKA2_DIR)")
    return p


def load_alaska2(image_id: int) -> np.ndarray:
    from PIL import Image
    x = np.asarray(Image.open(alaska2_pgm_dir() / f"{int(image_id):05d}.pgm"), dtype=np.uint8)
    if x.shape != (512, 512):
        raise ValueError(f"ALASKA2 {image_id}: size {x.shape}")
    return x


def alaska2_e7_rows(path: Path | None = None) -> list[tuple[int, str]]:
    """E7 subset (scripts/data/make_alaska2_subset.py): 10 000 images (index, camera model), stratified by camera model,
    seed 20261029; a subset of the 40 000 E1-bis images, so already on disk."""
    path = path or ROOT / "data" / "external" / "alaska2" / "subset_ids.txt"
    rows = [ln.split() for ln in Path(path).read_text().splitlines() if ln.strip()]
    return [(int(r[0]), r[1] if len(r) > 1 else "") for r in rows]


def load_cover(src: str, cid: int) -> np.ndarray:
    return load_alaska2(cid) if src == "alaska" else data.load_cover(src, cid)


def is_msglen(a) -> bool:
    return isinstance(a, str) and a.startswith("L")


def bucket_of(a, n: int, kem_bytes: int) -> int:
    """|pt| (bytes) of cell a: from alpha (regime R-b) or from the fixed message length L (regime R-a: the same bucket for every system)."""
    if is_msglen(a):
        return int(a[1:]) + START_HEADER_BYTES
    return profile_for_alpha(a, n, kem_bytes).buckets[0]


def profile_of(a, n: int, kem_bytes: int):
    return replace(DEFAULT_PP, buckets=(bucket_of(a, n, kem_bytes),))


def make_system(name: str, pp):
    """System or variant (table at the top of the file). Returns a ReferenceSystem, or (system, dk, ek)."""
    cost = COST_OF.get(name, "hill")
    if name in ("REF-U", "REF-U-SUNI"):
        return ReferenceSystem("REF-U", KEMELEON_CT_BYTES, pp=pp, cost=cost)
    if name == "V1":
        s = systems.make("PQ-PKS", pp, keyed_perm=False, keyed_matrix=False)
    elif name == "V2":
        s = systems.make("PQ-PKS", pp, keyed_perm=True, keyed_matrix=False)
    elif name == "V3":
        s = systems.make("PQ-PKS", pp, keyed_perm=False, keyed_matrix=True)
    elif name == "SUNI":
        s = systems.make("PQ-PKS", pp, cost="suniward")
    elif name == "A8p":
        s = systems.make("PQ-PKS", replace(pp, layout="A8p", alpha0=0.15))
    elif name == "REF-U-1568":
        return ReferenceSystem("REF-U", A7_KEM_BYTES, pp=pp)
    elif name == "REF-U-1252":
        return ReferenceSystem("REF-U", KEMELEON_R_CT_BYTES, pp=pp)
    elif name == "A7":
        s = systems.StcSystem(a7_crypto(), pp)
    elif name in STC_H_OF:
        if STC_H_OF[name] not in _STC_TABLES:
            raise RuntimeError(f"{name}: no H_hat table for h = {STC_H_OF[name]} (scripts/experiments/stc_tables/); run build_stc_table.py --h")
        s = systems.make("PQ-PKS", replace(pp, stc_h=STC_H_OF[name]))
    else:
        s = systems.make(name, pp)
    dk, ek = s.keygen()
    return (s, dk, ek)


def build(n: int, plan: str = "screen"):
    """Systems per (system, cell). Single-bucket profile by alpha (R-b) or by message length L (R-a), layout A8 (default)."""
    alphas, systems_at = PLANS[plan]
    if plan == "ablate2":
        install_stc_tables()
    return {(name, a): make_system(name, profile_of(a, n, KEM_BYTES[name])) for a in alphas for name in systems_at[a]}


def alpha_of(a, n: int, kem_bytes: int) -> float:
    """Actual embedding rate of a system in cell a (R-a: differs between systems)."""
    return 8 * (kem_bytes + bucket_of(a, n, kem_bytes) + AEAD_TAG_BYTES) / n


def ext0(embedder: TwoStageSTC, img, kem_bytes, body_bytes):
    return np.frombuffer(embedder.extract_kem(img, kem_bytes, body_bytes), dtype=np.uint8)


def one_cover(src: str, cid: int, sysd: dict, out_dir: Path, plan: str = "screen", srm: bool = False,
              keep_img: frozenset = frozenset()):
    x = load_cover(src, cid)
    n = x.size
    emb_default = TwoStageSTC(DEFAULT_PP)
    cm = emb_default.cost_map(x)                     # HILL computed once for every system (same pp.stc_h)
    alphas, systems_at = PLANS[plan]
    rec = {"src": src, "id": cid, "spam_cover": spam.spam686(x).astype(np.float32)}
    if srm:
        from confirm.common import srm_vec        # SRM, 34 671 dimensions (sealwatch), confirmatory step only
        rec["srm_cover"] = srm_vec(x)
    # E1: Ext0(cover) at A0(B) of the default ladder (A8, several buckets) for the main and the rejection-sampled variant
    for kb in (KEMELEON_CT_BYTES, 1252):
        for b in DEFAULT_PP.buckets:
            try:
                rec[f"e1_k{kb}_b{b}"] = ext0(emb_default, x, kb, b + AEAD_TAG_BYTES)
            except ValueError:
                pass
    rec["sat_frac"] = float(((x == 0) | (x == 255)).mean())
    rec["local_var"] = float(np.var(x.astype(np.float32) - np.roll(x, 1, 1).astype(np.float32)))
    fails, checks, failed = 0, [], []
    cm_by_cost = {"hill": cm}
    for a in alphas:
        # The default layout (1536 B) applies only when alpha suffices for the smallest session-start image of the main variant;
        # below that (phase E, confirm_r: 0.04651) every system of the cell has its own A0 and its own cover key (else branch below).
        default_ok = is_msglen(a) or 8 * (KEMELEON_CT_BYTES + DEFAULT_PP.buckets[0] + AEAD_TAG_BYTES) <= float(a) * n
        if default_ok:
            bucket = bucket_of(a, n, KEMELEON_CT_BYTES)
            body_bytes = bucket + AEAD_TAG_BYTES
            mask = emb_default.a0_mask(x.shape, KEMELEON_CT_BYTES, body_bytes)   # the same A0 for BL5-L, PQ-PKS, REF-U
            rec[f"a{a}_spamA0_cover"] = spam.spam_a0(x, mask).astype(np.float32)
            rec[f"a{a}_ext0_cover"] = ext0(emb_default, x, KEMELEON_CT_BYTES, body_bytes)
        else:
            if any(KEM_BYTES[nm] == KEMELEON_CT_BYTES and nm not in OWN_MASK for nm in systems_at[a]):
                raise ValueError(f"cell {a}: the system uses the default 1536 B layout but alpha is below the smallest session-start image")
            mask = None
        for name in systems_at[a]:
            kb = KEM_BYTES[name]
            cost = COST_OF.get(name, "hill")
            if cost not in cm_by_cost:                   # S-UNIWARD (A3): costs computed once for every system that uses it
                cm_by_cost[cost] = TwoStageSTC(DEFAULT_PP, cost=cost).cost_map(x)
            cmap = cm_by_cost[cost]
            if name in ("REF-U", "REF-U-SUNI", "REF-U-1568", "REF-U-1252"):
                ref = sysd[(name, a)]
                # body sized by the bucket of the control itself (profile_of(a, n, kb)): equal to body_bytes when kb = 1536. Before
                # 10/11/2026, REF-U-1568 (D12) used body_bytes of the 1536 B layout, i.e. it carried 32 B more than A7 (recorded, not rerun).
                bb = ref.pp.buckets[0] + AEAD_TAG_BYTES
                try:                                     # STC without a solution: same as EmbedFailure of a real system
                    y = ref.stego(x, bb, start=True, cost_map=cmap)
                except ValueError:
                    fails += 1
                    failed.append(f"a{a}_{name}")
                    continue
                emb = ref.embedder
            else:
                s, dk, ek = sysd[(name, a)]
                bb = s.pp.buckets[0] + AEAD_TAG_BYTES
                sess = s.start_session(ek, length=1)
                msg = secrets.token_bytes(message_len_for_bucket(s.pp.buckets[0], start=True))
                try:
                    y = s.send(sess, msg, x, cost_map=cmap)
                except systems.EmbedFailure:
                    fails += 1
                    failed.append(f"a{a}_{name}")
                    continue
                emb = s.embedder
                if name in ("PQ-PKS", "V1", "V2", "V3", "SUNI", "PQ-PKS-R", "A8p", "A7", "H8", "H12") and (cid % CHECK_EVERY == 0):
                    checks.append(s.receive_msg(dk, y, systems.ReceiverState()) == msg)
            key = f"a{a}_{name}"
            rec[key + "_spam"] = spam.spam686(y).astype(np.float32)
            if srm:
                rec[key + "_srm"] = srm_vec(y)
            if kb == KEMELEON_CT_BYTES and name not in OWN_MASK:
                m = mask
            else:
                m = emb.a0_mask(x.shape, kb, bb)
                rec[f"a{a}_{name}_spamA0_cover"] = spam.spam_a0(x, m).astype(np.float32)
                rec[f"a{a}_{name}_ext0_cover"] = ext0(emb, x, kb, bb)
            rec[key + "_spamA0"] = spam.spam_a0(y, m).astype(np.float32)
            rec[key + "_ext0"] = ext0(emb, y, kb, bb)
            rec[key + "_change"] = float((y != x).mean())
            if (a, name) in keep_img:                    # stego image kept for SRNet (confirmatory step, optional)
                rec[key + "_img"] = y
    rec["stc_fail"] = fails
    rec["failed_combos"] = np.array(failed, dtype="U32")      # (alpha, system) pairs that could not be embedded in this cover
    rec["checks"] = np.array(checks, dtype=bool)
    tmp = out_dir / f".{src}_{cid}.npz"
    np.savez_compressed(tmp, **{k: (np.asarray(v) if not isinstance(v, str) else np.array(v)) for k, v in rec.items()})
    os.replace(tmp, out_dir / f"{src}_{cid}.npz")


def worker(wi: int, nw: int, covers, out_dir: Path, limit: int | None, plan: str = "screen", srm: bool = False,
           keep_img: frozenset = frozenset()):
    sysd = build(512 * 512, plan)
    todo = [c for k, c in enumerate(covers) if k % nw == wi]
    if limit:
        todo = todo[:limit]
    t0 = time.time()
    for k, (src, cid) in enumerate(todo):
        if (out_dir / f"{src}_{cid}.npz").exists():
            continue
        try:
            one_cover(src, cid, sysd, out_dir, plan, srm, keep_img)
        except Exception:                               # record the error, do not stop the batch
            (out_dir / f"{src}_{cid}.err").write_text(traceback.format_exc())
        if k % 20 == 0:
            print(f"[w{wi}] {k + 1}/{len(todo)} {time.time() - t0:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", default=str(ROOT / "results" / "archive" / "pre-confirm" / "screen_v1" / "raw"))
    ap.add_argument("--limit", type=int, default=None, help="maximum number of covers per worker (quick test)")
    ap.add_argument("--plan", choices=list(PLANS), default="screen", help="screen: screening round; confirm: confirmatory step")
    ap.add_argument("--srm", action="store_true", help="also store SRM (34 671 dimensions) for the cover and every stego (confirmatory step)")
    ap.add_argument("--save-stego", default="", metavar="SPEC",
                    help="keep stego images (for SRNet, maxSRMd2, masked SRM): all, or alpha:system pairs, e.g. \"0.2:PQ-PKS,REF-U;0.4:PQ-PKS,REF-U\"")
    ap.add_argument("--covers", choices=["screen", "confirm", "alaska2-e7"], default=None,
                    help="cover set; default from the plan (screen/ablate/ra: the 2000 screening images; confirm: split_v1; e7: ALASKA2)")
    args = ap.parse_args()
    atag = (lambda a: a if is_msglen(a) else float(a))
    if args.save_stego.strip().lower() == "all":         # every (cell, system) pair of the plan
        keep = frozenset((atag(a), n) for a, names in PLANS[args.plan][1].items() for n in names)
    else:
        keep = frozenset((atag(a), n) for part in filter(None, args.save_stego.split(";"))
                         for a, names in [part.split(":")] for n in names.split(","))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cov_kind = args.covers or PLAN_COVERS[args.plan]
    sp = None
    if args.plan != "screen":
        sys.path.insert(0, str(ROOT / "scripts"))
        from confirm.common import check_frozen, load_split
        check_frozen()
    if cov_kind == "confirm":
        sp = load_split()
        covers = [("boss", i) for i in sp["boss"]["test"] + sp["boss"]["val"] + sp["boss"]["train"]] + \
                 [("bows2", i) for i in sp["bows2"]["train"]]
    elif cov_kind == "alaska2-e7":
        covers = [("alaska", i) for i, _ in alaska2_e7_rows()]
    else:
        covers = data.screening_covers()
    boss = [c for c in covers if c[0] == "boss"]
    bows = [c for c in covers if c[0] == "bows2"]
    other = [c for c in covers if c[0] not in ("boss", "bows2")]
    covers = [c for pair in zip(boss, bows) for c in pair]        # interleave the two sources: an early stop stays balanced
    k = min(len(boss), len(bows))
    covers += boss[k:] + bows[k:] + other                          # remainder (when the two sources differ in size)
    alphas, systems_at = PLANS[args.plan]
    notes = {"screen": "screening round: pilot BOSSbase images + 1000 BOWS2 images; test set not touched",
             "confirm": "confirmatory step: 10 000 BOSSbase + 10 000 BOWS2 images according to split_v1.json",
             "e7": "E7: 10 000 ALASKA2 images (subset_ids.txt), external source; the warden is trained and tested on this source",
             "ablate": "S2: ablations A1 (V1-V3), A3 (SUNI, REF-U-SUNI), A4 (PQ-PKS-R), A8' (A8p) on the 2000 screening covers",
             "ra": "S5: regime R-a, the same message of L bytes for every system; the alpha of each system is in alpha_by_cell",
             "ablate2": "S2 part 2: A7 hybrid ML-KEM-768 + X25519 (1568 B, control REF-U-1568), A9 STC h = 8, 12 (own H_hat tables)",
             "confirm_r": "Phase E (exploratory): smallest session-start image of the rejection-sampled variant (PQ-PKS-R, 1252 B) and the control "
                          "REF-U-1252 at alpha = 0.04651 on the 20 000 confirmatory covers; added after the confirmatory alpha* was known, not part of the registered grid"}
    meta = {"covers": len(covers), "alphas": alphas, "systems_at": {str(k): v for k, v in systems_at.items()},
            "plan": args.plan, "covers_kind": cov_kind, "srm": args.srm, "save_stego": sorted(f"{a}:{n}" for a, n in keep),
            "layout": DEFAULT_PP.layout, "note": notes[args.plan],
            "kem_bytes": {n: KEM_BYTES[n] for a in alphas for n in systems_at[a]},
            "alpha_by_cell": {str(a): {n: alpha_of(a, 512 * 512, KEM_BYTES[n]) for n in systems_at[a]} for a in alphas}}
    if sp is not None:
        meta["split_sha256"] = sp["_sha256"]
    (out.parent / "run_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "RAYON_NUM_THREADS"):   # one thread per child process
        os.environ.setdefault(k, "1")
    ps = [Process(target=worker, args=(i, args.workers, covers, out, args.limit, args.plan, args.srm, keep))
          for i in range(args.workers)]
    for p in ps:
        p.start()
    for p in ps:
        p.join()


if __name__ == "__main__":
    main()
