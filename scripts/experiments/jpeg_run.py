"""S3 (JPEG QF75, H8): stego generation in the DCT domain and DCTR/GFR features for the 2000 screening covers.

Cells: Q in {2048, 3072} bytes of total payload (labels "Q2048", "Q3072"); systems BL5, BL5-L, PQ-PKS, REF-U; bucket of a system = Q - |kem_part| - 16.
Images with 8Q/|U| > 0.5 bpnzAC are excluded from cell Q (failed_combos). Result files have the same format as screen_run (one .npz per cover),
feature keys: dctr_cover, gfr_cover, a<Q>_<system>_dctr, _gfr, _ext0, _ext0_cover, _change, _bpnzac.
Run:  python scripts/experiments/jpeg_run.py --workers 6 --out results\\jpeg\\raw
Phase E: QF95 (|U| about 2-3 times larger, so bpnzAC is lower and P_E is not at the floor):
       python scripts/experiments/jpeg_run.py --qf 95 --workers 6 --out results\\jpeg95\\raw
"""
from __future__ import annotations

import argparse
import json
import os
import secrets
import sys
import time
import traceback
from dataclasses import replace
from multiprocessing import Process
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "experiments"))

from pqpks import systems  # noqa: E402
from pqpks.experiment.profiles import message_len_for_bucket  # noqa: E402
from pqpks.experiment.references import ReferenceSystem  # noqa: E402
from pqpks.params import AEAD_TAG_BYTES, DEFAULT_PP, KEMELEON_CT_BYTES  # noqa: E402
from confirm.common import warnings_to_stdout  # noqa: E402,F401
from jpeg_domain import MAX_BPNZAC, JpegCover, JpegTwoStageSTC  # noqa: E402
from screen_run import KEM_BYTES, SYSTEMS, load_cover  # noqa: E402

Q_TOTALS = (2048, 3072)
CELLS = tuple(f"Q{q}" for q in Q_TOTALS)
CHECK_EVERY = 25


def build():
    """Systems per (system, cell): single-bucket profile B = Q - |kem_part| - 16, DCT-domain embedder."""
    out = {}
    for q in Q_TOTALS:
        for name in SYSTEMS:
            pp = replace(DEFAULT_PP, buckets=(q - KEM_BYTES[name] - AEAD_TAG_BYTES,))
            if name == "REF-U":
                ref = ReferenceSystem("REF-U", KEMELEON_CT_BYTES, pp=pp)
                ref.embedder = JpegTwoStageSTC(pp)
                out[(name, f"Q{q}")] = ref
            else:
                s = systems.make(name, pp)
                s.embedder = JpegTwoStageSTC(pp)
                dk, ek = s.keygen()
                out[(name, f"Q{q}")] = (s, dk, ek)
    return out


def one_cover(src, cid, sysd, out_dir: Path, qf: int = 75):
    x = load_cover(src, cid)
    jc = JpegCover.from_gray(x, qf=qf)
    cm = jc.juniward_cost()
    xv = jc.values()
    rec = {"src": src, "id": cid, "n_u": jc.n_u, "dctr_cover": jc.dctr(), "gfr_cover": jc.gfr(),
           "sat_frac": float(((x == 0) | (x == 255)).mean())}
    fails, checks, failed = 0, [], []
    for q in Q_TOTALS:
        cell = f"Q{q}"
        if jc.bpnzac(q) > MAX_BPNZAC:                    # the image has too few non-zero AC coefficients for this cell
            for name in SYSTEMS:
                failed.append(f"a{cell}_{name}")
            continue
        for name in SYSTEMS:
            kb = KEM_BYTES[name]
            if name == "REF-U":
                ref = sysd[(name, cell)]
                bb = ref.pp.buckets[0] + AEAD_TAG_BYTES
                try:
                    y = ref.stego(xv, bb, start=True, cost_map=cm)
                except ValueError:
                    fails += 1
                    failed.append(f"a{cell}_{name}")
                    continue
                emb = ref.embedder
            else:
                s, dk, ek = sysd[(name, cell)]
                bb = s.pp.buckets[0] + AEAD_TAG_BYTES
                sess = s.start_session(ek, length=1)
                msg = secrets.token_bytes(message_len_for_bucket(s.pp.buckets[0], start=True))
                try:
                    y = s.send(sess, msg, xv, cost_map=cm)
                except systems.EmbedFailure:
                    fails += 1
                    failed.append(f"a{cell}_{name}")
                    continue
                emb = s.embedder
                if name == "PQ-PKS" and (cid % CHECK_EVERY == 0):
                    checks.append(s.receive_msg(dk, y, systems.ReceiverState()) == msg)
            js = jc.with_values(y)
            key = f"a{cell}_{name}"
            rec[key + "_dctr"] = js.dctr()
            rec[key + "_gfr"] = js.gfr()
            rec[key + "_ext0_cover"] = np.frombuffer(emb.extract_kem(xv, kb, bb), dtype=np.uint8)
            rec[key + "_ext0"] = np.frombuffer(emb.extract_kem(y, kb, bb), dtype=np.uint8)
            rec[key + "_change"] = float((y != xv).mean())
            rec[key + "_bpnzac"] = float(jc.bpnzac(q))
    rec["stc_fail"] = fails
    rec["failed_combos"] = np.array(failed, dtype="U32")
    rec["checks"] = np.array(checks, dtype=bool)
    tmp = out_dir / f".{src}_{cid}.npz"
    np.savez_compressed(tmp, **{k: (np.asarray(v) if not isinstance(v, str) else np.array(v)) for k, v in rec.items()})
    os.replace(tmp, out_dir / f"{src}_{cid}.npz")


def worker(wi, nw, covers, out_dir: Path, limit, qf=75):
    sysd = build()
    todo = [c for k, c in enumerate(covers) if k % nw == wi]
    if limit:
        todo = todo[:limit]
    t0 = time.time()
    for k, (src, cid) in enumerate(todo):
        if (out_dir / f"{src}_{cid}.npz").exists():
            continue
        try:
            one_cover(src, cid, sysd, out_dir, qf)
        except Exception:
            (out_dir / f"{src}_{cid}.err").write_text(traceback.format_exc())
        if k % 20 == 0:
            print(f"[w{wi}] {k + 1}/{len(todo)} {time.time() - t0:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", default=str(ROOT / "results" / "jpeg" / "raw"))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--qf", type=int, default=75, help="JPEG quality factor of the covers (75 = registered S3; 95 = phase E)")
    args = ap.parse_args()
    from confirm.common import check_frozen
    from pqpks import data
    check_frozen()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    covers = data.screening_covers()
    boss = [c for c in covers if c[0] == "boss"]
    bows = [c for c in covers if c[0] == "bows2"]
    covers = [c for pair in zip(boss, bows) for c in pair]
    k = min(len(boss), len(bows))
    covers += boss[k:] + bows[k:]
    meta = {"covers": len(covers), "alphas": list(CELLS), "systems_at": {c: list(SYSTEMS) for c in CELLS}, "plan": "jpeg", "qf": args.qf,
            "covers_kind": "screen", "srm": False, "wimg_feature": "dctr", "wimg_alternatives": ["gfr"], "layout": DEFAULT_PP.layout,
            "kem_bytes": {n: KEM_BYTES[n] for n in SYSTEMS},
            "note": f"S3: JPEG QF{args.qf}, DCT domain (non-zero AC coefficients), J-UNIWARD, fixed payload of Q bytes per cell; "
                    "actual bpnzAC of each image under key _bpnzac; images with bpnzAC > 0.5 are excluded from the cell",
            "q_totals": list(Q_TOTALS), "max_bpnzac": MAX_BPNZAC}
    (out.parent / "run_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "RAYON_NUM_THREADS"):
        os.environ.setdefault(k, "1")
    ps = [Process(target=worker, args=(i, args.workers, covers, out, args.limit, args.qf)) for i in range(args.workers)]
    for p in ps:
        p.start()
    for p in ps:
        p.join()


if __name__ == "__main__":
    main()
