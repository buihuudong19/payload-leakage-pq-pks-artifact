"""Functional E4 (H4, RQ2.2): correctness, image modification, replay and false alarms on the real systems with the default
profile (layout A8, B = {256, ..., 8192}).

Covers (--size 256): 256x256 crops of the pilot BOSSbase images (4 crops per image), so that the run fits a small CPU-only
machine. At 256x256 bucket 8192 does not satisfy Adm for a session-start image; this is also a test of Adm.
The reduced run uses fewer trials than the H4 targets (10^5 and 10^6); the upper 95 % bound by the rule of three is reported for
the actual number of trials. The full-scale run (--size 512) uses --normal 100000 --cover-images ...

Run: python scripts/experiments/e4_functional.py --normal 3000 --tamper 300 --cover-images 300
"""
from __future__ import annotations

import argparse
import json
import secrets
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from pqpks import data, systems  # noqa: E402
from pqpks.analysis import stats  # noqa: E402
from pqpks.crypto import framing  # noqa: E402
from pqpks.params import (AEAD_TAG_BYTES, DEFAULT_PP, FLAG_SESSION_START, FRAME_HEADER_BYTES,  # noqa: E402
                          START_HEADER_BYTES)


def crops(ids):
    """256x256 crops, cycled indefinitely; each cycle shifts the crop origin by 64 pixels so that no crop repeats exactly."""
    k = 0
    while True:
        off = (64 * k) % 256
        for i in ids:
            x = data.load_bossbase(i)
            for r in (0, 256 - off if off else 256):
                for c in (off, 256):
                    if r + 256 <= 512 and c + 256 <= 512:
                        yield x[r:r + 256, c:c + 256].copy()
        k += 1


def full_images(ids):
    """Whole 512x512 images, cycled indefinitely (--size 512)."""
    while True:
        for src, i in ids:
            yield data.load_cover(src, i)


def msg_cap(pp, size: int, start: bool, kem_bytes: int) -> int:
    """Largest message length still accepted by Adm on a size x size image: the largest bucket with l_start <= n, minus the header."""
    n = size * size
    hdr = START_HEADER_BYTES if start else FRAME_HEADER_BYTES
    ok = [b for b in pp.buckets_for(start, kem_bytes) if 8 * ((kem_bytes if start else 0) + b + AEAD_TAG_BYTES) <= n]
    return max(ok) - hdr


def rand_msg(rng, start, caps):
    cap = caps[start]                                 # 256x256: 4096 - header; 512x512: 8192 - 14 or 9728 - 6
    L = int(rng.choice([16, 64, 200, 700, 1500, 3000, cap]))
    return secrets.token_bytes(min(L, cap))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--normal", type=int, default=3000)
    ap.add_argument("--tamper", type=int, default=300)
    ap.add_argument("--cover-images", type=int, default=300)
    ap.add_argument("--out", default=str(ROOT / "results" / "archive" / "pre-confirm" / "e4_functional"))
    ap.add_argument("--size", type=int, choices=(256, 512), default=256,
                    help="256: 256x256 crops of the pilot images (reduced run); 512: whole 512x512 images outside the test set (confirmatory step)")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(4)
    sysm = systems.make("PQ-PKS")
    dk, ek = sysm.keygen()
    if args.size == 256:
        pool = crops(data.pilot_ids())
        image_desc = "256x256 crops of BOSSbase pilot"
    else:
        sys.path.insert(0, str(ROOT / "scripts"))
        from confirm.common import load_split
        sp = load_split()
        ids = [("boss", i) for i in sp["boss"]["train"] + sp["boss"]["val"]] + [("bows2", i) for i in sp["bows2"]["train"]]
        ids = [ids[j] for j in np.random.default_rng(5).permutation(len(ids))]
        pool = full_images(ids)
        image_desc = f"512x512 BOSSbase train/val and BOWS2 images of split_v1 ({len(ids)} images, test set untouched)"
    kb = sysm.crypto.kem_part_bytes
    if args.size == 256:                              # keep the message distribution of the reported reduced run
        caps, bin_max = {True: 4096 - 14, False: 4096 - 6}, 4000
    else:
        caps = {True: msg_cap(sysm.pp, args.size, True, kb), False: msg_cap(sysm.pp, args.size, False, kb)}
        bin_max = caps[True]
    bin_msgs = list(data.test_messages())   # two binary messages of 919 and 2420 bytes
    res = {"profile": {"layout": sysm.pp.layout, "buckets": sysm.pp.buckets, "image": image_desc, "size": args.size,
                       "message_caps": {"first": caps[True], "later": caps[False]}}}
    t0 = time.time()

    # ---------------- block 1: normal send/receive (sessions of length 1-4, out-of-order reception within the window)
    n_img = n_err = n_inadm = n_fail = 0
    k_runs = 0
    while n_img < args.normal:
        N = int(rng.integers(1, 5))
        sess = sysm.start_session(ek, length=N)
        st = systems.ReceiverState()
        imgs, msgs = [], []
        for i in range(N):
            m = bin_msgs[k_runs % 2] if (k_runs % 10 == 0 and i == 0 and len(bin_msgs[k_runs % 2]) < bin_max) else rand_msg(rng, i == 0, caps)
            x = next(pool)
            try:
                imgs.append(sysm.send(sess, m, x))
                msgs.append(m)
            except systems.Inadmissible:
                n_inadm += 1
                break
            except systems.EmbedFailure:
                n_fail += 1
                break
        k_runs += 1
        if not imgs:
            continue
        order = [0] + list(rng.permutation(range(1, len(imgs))))   # session-start image first, the rest shuffled
        for j in order:
            r = sysm.receive(dk, imgs[j], st)
            n_img += 1
            n_err += not (r.status == "ok" and r.msg == msgs[j])
    res["normal"] = {"images": n_img, "errors": n_err, "inadmissible": n_inadm, "embed_failures": n_fail,
                     "upper95_error": stats.rule_of_three(n_img) if n_err == 0 else None}
    print("normal", res["normal"], f"{time.time() - t0:.0f}s", flush=True)

    # ---------------- block 2: image modification and replay
    tam = {k: [0, 0] for k in ("a_A0_k1", "a_A0_k4", "a_A0_k16", "a1_A1_k4", "b_pm2_k16", "d_splice", "replay_start",
                               "replay_next", "wrong_dk", "inadmissible_long")}
    dk2, _ = sysm.keygen()
    for t in range(args.tamper):
        sess = sysm.start_session(ek, length=2)
        m1 = rand_msg(rng, True, caps)
        y1 = sysm.send(sess, m1, next(pool))
        y2 = sysm.send(sess, b"second", next(pool))
        bl = len(framing.pack(framing.Frame(m1, FLAG_SESSION_START, None, 0), sysm.pp.buckets)) + AEAD_TAG_BYTES
        emb = sysm.embedder
        a0, a1 = emb._sets(y1.shape, 8 * sysm.crypto.kem_part_bytes, 8 * bl)
        flat = y1.reshape(-1)

        def modified(pos, delta):
            z = flat.astype(np.int16).copy()
            z[pos] = np.clip(z[pos] + delta, 0, 255)
            return z.astype(np.uint8).reshape(y1.shape)

        def bottom(img):
            return sysm.receive(dk, img, systems.ReceiverState()).status != "ok"

        for k in (1, 4, 16):
            pos = rng.choice(a0[(flat[a0] > 0) & (flat[a0] < 255)], k, replace=False)
            img = modified(pos, rng.choice([-1, 1], k))
            if emb.extract_kem(img, sysm.crypto.kem_part_bytes, bl) != emb.extract_kem(y1, sysm.crypto.kem_part_bytes, bl):
                tam[f"a_A0_k{k}"][0] += 1
                tam[f"a_A0_k{k}"][1] += bottom(img)
        pos = rng.choice(a1[(flat[a1] > 0) & (flat[a1] < 255)], 4, replace=False)
        img = modified(pos, rng.choice([-1, 1], 4))
        tam["a1_A1_k4"][0] += 1
        tam["a1_A1_k4"][1] += bottom(img)                         # A1 is fully used; changing an LSB changes the body (bottom expected)
        pos = rng.choice(np.nonzero((flat > 1) & (flat < 254))[0], 16, replace=False)
        img = modified(pos, rng.choice([-2, 2], 16))
        tam["b_pm2_k16"][0] += 1
        tam["b_pm2_k16"][1] += sysm.receive(dk, img, systems.ReceiverState()).msg == m1   # expected: the same m
        other = sysm.send(sysm.start_session(ek, length=1), m1, next(pool))
        half = y1.shape[0] // 2
        img = np.vstack([y1[:half], other[half:]])
        tam["d_splice"][0] += 1
        tam["d_splice"][1] += bottom(img)
        st = systems.ReceiverState()
        sysm.receive(dk, y1, st)
        tam["replay_start"][0] += 1
        tam["replay_start"][1] += sysm.receive(dk, y1, st).status != "ok"
        sysm.receive(dk, y2, st)
        tam["replay_next"][0] += 1
        tam["replay_next"][1] += sysm.receive(dk, y2, st).status != "ok"
        tam["wrong_dk"][0] += 1
        tam["wrong_dk"][1] += sysm.receive(dk2, y1, systems.ReceiverState()).status != "ok"
        s3 = sysm.start_session(ek, length=1)
        tam["inadmissible_long"][0] += 1
        try:
            sysm.send(s3, bytes(caps[True] + 1), next(pool))          # longer than any bucket Adm accepts
        except systems.Inadmissible:
            tam["inadmissible_long"][1] += s3.next_index == 1
    res["tamper"] = {k: {"trials": v[0], "expected_outcome": v[1], "rate": (v[1] / v[0]) if v[0] else None}
                     for k, v in tam.items()}
    print("tamper", {k: v["rate"] for k, v in res["tamper"].items()}, f"{time.time() - t0:.0f}s", flush=True)

    # ---------------- block 3: false alarms with S_max open sessions and a full window
    pp = DEFAULT_PP
    st = systems.ReceiverState()
    for _ in range(pp.max_open_sessions):
        s = sysm.start_session(ek, length=pp.session_len)
        sysm.receive(dk, sysm.send(s, b"open", next(pool)), st)
    acc = trials = 0
    t_cov = []
    for _ in range(args.cover_images):
        tt = time.time()
        r = sysm.receive(dk, next(pool), st)
        t_cov.append(time.time() - tt)
        acc += r.status == "ok"
        trials += r.trials
    kb = sysm.crypto.kem_part_bytes            # |B_1| + (2W - 1) S_max |B_2|
    bound = len(pp.buckets_for(True, kb)) + (2 * pp.window - 1) * pp.max_open_sessions * len(pp.buckets_for(False, kb))
    res["false_alarm"] = {"cover_images": args.cover_images, "accepted": acc, "aead_trials": trials,
                          "trials_per_image_max_bound": bound, "trials_per_image_mean": trials / args.cover_images,
                          "upper95_per_trial": stats.rule_of_three(trials) if acc == 0 else None,
                          "receiver_time_s_median": float(np.median(t_cov))}
    print("false_alarm", res["false_alarm"], f"{time.time() - t0:.0f}s", flush=True)
    (out / "e4_summary.json").write_text(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
