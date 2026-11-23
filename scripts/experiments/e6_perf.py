"""E6 (H6, RQ3.2): time of each step of Send/Recv on the CPU (Python + numba; not an optimized implementation).
Reports median and IQR. Each cryptographic operation is repeated >= 1000 times; image operations --img-reps times.

Run: python scripts/experiments/e6_perf.py --img-reps 20
"""
from __future__ import annotations

import argparse
import json
import platform
import secrets
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from pqpks import data, systems  # noqa: E402
from pqpks.crypto import hpke, kemeleon, sig  # noqa: E402
from pqpks.experiment.profiles import message_len_for_bucket, profile_for_alpha  # noqa: E402
from pqpks.systems.base import image_keys, seal_image  # noqa: E402
from pqpks.params import DEFAULT_PP  # noqa: E402


def timeit(fn, reps):
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t)
    q1, med, q3 = np.percentile(ts, [25, 50, 75])
    return {"median_ms": 1e3 * med, "iqr_ms": 1e3 * (q3 - q1), "reps": reps}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crypto-reps", type=int, default=1000)
    ap.add_argument("--img-reps", type=int, default=20)
    ap.add_argument("--out", default=str(ROOT / "results" / "archive" / "pre-confirm" / "e6_perf"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    R, I = args.crypto_reps, args.img_reps
    import os
    import socket
    # The machine note is derived from the actual environment.
    res = {"machine": {"python": platform.python_version(), "cpu": platform.processor() or platform.machine(),
                       "logical_cpus": os.cpu_count(), "host": socket.gethostname(), "platform": platform.platform(),
                       "note": ("desktop (Windows); " if os.name == "nt" else "non-Windows machine; ")
                               + "reference implementation in Python (Kemeleon in big integers); one process, "
                               + "other steps may share the CPU (see run_status at the time of the run)"}}
    prev = out / "e6_summary.json"
    if prev.exists():                                   # keep the previous measurement (same directory), never overwrite it
        import time as _t
        prev.rename(out / f"e6_summary_{_t.strftime('%Y-%m-%d_%H%M', _t.localtime(prev.stat().st_mtime))}.json")
    kem = hpke.MlKem768()
    sk, pk = kem.generate()
    res["ML-KEM-768 KeyGen"] = timeit(kem.generate, R)
    pid = systems.make("PQ-PKS").pid
    res["HPKE SetupBaseS (Encaps)"] = timeit(lambda: hpke.setup_base_s(kem, pk, pid), R)
    enc, ctx = hpke.setup_base_s(kem, pk, pid)
    res["HPKE SetupBaseR (Decaps)"] = timeit(lambda: hpke.setup_base_r(kem, enc, sk, pid), R)
    res["Kemeleon EncodeCt (main)"] = timeit(lambda: kemeleon.encode_ct(enc), R)
    ce = kemeleon.encode_ct(enc)
    res["Kemeleon DecodeCt (main)"] = timeit(lambda: kemeleon.decode_ct(ce), R)
    att = []
    def enc_r():
        k = 0
        while True:
            k += 1
            e2, _ = hpke.setup_base_s(kem, pk, pid)
            if kemeleon.encode_ct_r(e2) is not None:
                att.append(k)
                return
    res["Kemeleon EncodeCtxtR incl. re-encapsulation"] = timeit(enc_r, R)
    res["Kemeleon-R attempts mean"] = float(np.mean(att))
    sid = secrets.token_bytes(32)
    res["Export x3 (one image)"] = timeit(lambda: image_keys(ctx, pid, sid, 1), R)
    keys = image_keys(ctx, pid, sid, 1)
    pt = secrets.token_bytes(4096)
    res["ChaCha20-Poly1305 Enc 4 KB"] = timeit(lambda: seal_image(keys, pid, sid, 1, pt), R)
    ssk, svk = sig.keygen("ML-DSA-65")
    tr = secrets.token_bytes(200)
    res["ML-DSA-65 Sign"] = timeit(lambda: sig.sign(ssk, tr), R)
    sg = sig.sign(ssk, tr)
    res["ML-DSA-65 Verify"] = timeit(lambda: sig.verify(svk, tr, sg, "ML-DSA-65"), R)
    # image operations on 512x512, alpha = 0.2 (single-bucket profile) and the default profile
    S = DEFAULT_PP.max_open_sessions
    allc = [data.load_bossbase(i) for i in data.pilot_ids()[: I + S]]
    covers, openers = allc[:I], allc[I:]          # the images that open S_max sessions differ from those used as invalid images
    pp = profile_for_alpha(0.2, 512 * 512, 1536)
    s = systems.make("PQ-PKS", pp)
    dk, ek = s.keygen()
    emb = s.embedder
    cms = []
    t_cost, t_send, t_recv = [], [], []
    for x in covers:
        t = time.perf_counter(); cm = emb.cost_map(x); t_cost.append(time.perf_counter() - t)
        sess = s.start_session(ek, length=1)
        m = secrets.token_bytes(message_len_for_bucket(pp.buckets[0]))
        t = time.perf_counter(); y = s.send(sess, m, x, cost_map=cm); t_send.append(time.perf_counter() - t)
        t = time.perf_counter(); r = s.receive(dk, y, systems.ReceiverState()); t_recv.append(time.perf_counter() - t)
        assert r.msg == m
    def stat(ts):
        q1, med, q3 = np.percentile(ts, [25, 50, 75])
        return {"median_ms": 1e3 * med, "iqr_ms": 1e3 * (q3 - q1), "reps": len(ts)}
    res["HILL costs 512x512"] = stat(t_cost)
    res["Send per image (frame, AEAD, 2 STC stages; session start timed separately) alpha=0.2"] = stat(t_send)
    res["Recv valid first image alpha=0.2 (single bucket)"] = stat(t_recv)
    crypto_ms = (res["HPKE SetupBaseS (Encaps)"]["median_ms"] + res["Kemeleon EncodeCt (main)"]["median_ms"]
                 + 8 * res["Export x3 (one image)"]["median_ms"] + res["ChaCha20-Poly1305 Enc 4 KB"]["median_ms"])
    total_ms = crypto_ms + res["HILL costs 512x512"]["median_ms"] + res["Send per image (frame, AEAD, 2 STC stages; session start timed separately) alpha=0.2"]["median_ms"]
    res["H6 crypto share of Send (unsigned, N=8 key precompute)"] = crypto_ms / total_ms
    # receiver with the default profile (A8, |B| = 6): valid session-start image and invalid image with S_max open sessions
    sd = systems.make("PQ-PKS")
    dkd, ekd = sd.keygen()
    t_ok = []
    for x in covers:
        sess = sd.start_session(ekd, length=1)
        y = sd.send(sess, b"perf", x)
        t = time.perf_counter(); assert sd.receive(dkd, y, systems.ReceiverState()).msg == b"perf"
        t_ok.append(time.perf_counter() - t)
    res["Recv valid first image default profile (A8, |B| = 6)"] = stat(t_ok)
    st = systems.ReceiverState()
    for x in openers:
        ss = sd.start_session(ekd, length=DEFAULT_PP.session_len)
        sd.receive(dkd, sd.send(ss, b"o", x), st)
    t_bad, trials = [], []
    for x in covers:
        t = time.perf_counter(); r = sd.receive(dkd, x, st); t_bad.append(time.perf_counter() - t)
        trials.append(r.trials)
    res["Recv invalid image, S_max sessions open (default profile)"] = stat(t_bad) | {"trials_per_image": sorted(set(trials))}
    (out / "e6_summary.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
