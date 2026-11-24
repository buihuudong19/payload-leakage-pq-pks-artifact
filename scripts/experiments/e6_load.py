"""E6-load (post hoc, 24 Nov 2026; finding F06): receiver cost of invalid images under concurrent load.

Each worker process holds its own receiver with S_max = 16 open sessions (default profile, layout A8), opened with its own covers,
and then receives invalid images (covers that carry no stego). After one warm-up image (excluded), all workers start together
(barrier) and every receive is timed. For each concurrency level c (number of worker processes) the script reports the latency
percentiles p50/p95/p99 over all timed images and the throughput (images per second over the whole machine, from the common
start to the last finish). Two receivers are measured:
  reference  systems.make("PQ-PKS"), the implementation of the paper (unchanged);
  cached     receiver_fast.fast_system("PQ-PKS"): same decisions and outputs, with cached stage-2 permutations and a
             shift-register syndrome (tests/test_receiver_fast.py checks the equality of outputs).

    python scripts/experiments/e6_load.py --levels 1 4 16 64 --images 8 --reps 3 --mode reference cached
    python scripts/experiments/e6_load.py --synthetic --levels 1 2 --images 2 --reps 1      # code test without cover images
Output: results/e6_load/e6_load_summary.json (+ every latency in e6_load_latencies.json).
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "e6_load"


def synthetic_cover(rng: np.random.Generator) -> np.ndarray:
    x = rng.normal(0, 30, (512, 512)).cumsum(0).cumsum(1)
    x = (x - x.min()) / (np.ptp(x) + 1e-9) * 200 + 25 + rng.normal(0, 3, (512, 512))
    return np.clip(x, 0, 255).astype(np.uint8)


def worker(wid, mode, n_img, synthetic, offset, barrier, q):
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ[v] = "1"
    sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts" / "experiments")]
    from pqpks import data, systems
    from pqpks.params import DEFAULT_PP
    from receiver_fast import fast_system
    S = DEFAULT_PP.max_open_sessions
    need = S + n_img + 1
    if synthetic:
        rng = np.random.default_rng([20261124, wid])
        covers = [synthetic_cover(rng) for _ in range(need)]
    else:
        ids = data.pilot_ids()
        start = (offset + wid * need) % max(1, len(ids) - need)
        covers = [data.load_bossbase(i) for i in ids[start:start + need]]
    s = fast_system("PQ-PKS") if mode == "cached" else systems.make("PQ-PKS")
    dk, ek = s.keygen()
    st = systems.ReceiverState()
    for x in covers[:S]:
        sess = s.start_session(ek, length=DEFAULT_PP.session_len)
        s.receive(dk, s.send(sess, b"o", x), st)
    bad = covers[S:]
    r = s.receive(dk, bad[0], st)                          # warm-up (numba compilation, caches), not timed
    barrier.wait()
    t_start = time.perf_counter()
    lat, trials = [], {r.trials}
    for x in bad[1:]:
        t = time.perf_counter()
        r = s.receive(dk, x, st)
        lat.append(time.perf_counter() - t)
        trials.add(r.trials)
        assert r.msg is None
    q.put({"wid": wid, "lat": lat, "t_start": t_start, "t_end": time.perf_counter(), "trials": sorted(trials)})


def run_level(mode, c, n_img, synthetic, rep):
    ctx = mp.get_context("spawn")
    barrier, q = ctx.Barrier(c), ctx.Queue()
    ps = [ctx.Process(target=worker, args=(w, mode, n_img, synthetic, 97 * rep, barrier, q)) for w in range(c)]
    for p in ps:
        p.start()
    res = [q.get() for _ in ps]
    for p in ps:
        p.join()
    lat = np.concatenate([r["lat"] for r in res])
    wall = max(r["t_end"] for r in res) - min(r["t_start"] for r in res)
    return lat, wall, sorted({t for r in res for t in r["trials"]})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", type=int, nargs="+", default=[1, 4, 16, 64])
    ap.add_argument("--images", type=int, default=8, help="timed invalid images per worker")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--mode", nargs="+", choices=["reference", "cached"], default=["reference", "cached"])
    ap.add_argument("--synthetic", action="store_true", help="code test: synthetic covers instead of BOSSbase pilot covers")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    res = {"label": "post hoc (24 Nov 2026), finding F06; E6 of the paper is unchanged", "cpu_count": os.cpu_count(),
           "platform": platform.platform(), "processor": platform.processor(), "python": sys.version.split()[0],
           "S_max": 16, "images_per_worker": a.images, "reps": a.reps, "synthetic": a.synthetic, "rows": []}
    raw = {}
    t0 = time.time()
    for mode in a.mode:
        for c in a.levels:
            lats, walls, tr = [], [], set()
            for rep in range(a.reps):
                lat, wall, trials = run_level(mode, c, a.images, a.synthetic, rep)
                lats.append(lat); walls.append(wall); tr |= set(trials)
                print(f"[{time.time() - t0:6.0f}s] {mode} c={c} rep {rep}: median {np.median(lat):.2f} s, "
                      f"throughput {lat.size / wall:.2f} img/s", flush=True)
            L = np.concatenate(lats)
            thr = [l.size / w for l, w in zip(lats, walls)]
            row = {"mode": mode, "concurrency": c, "n_images": int(L.size),
                   "p50_s": float(np.percentile(L, 50)), "p95_s": float(np.percentile(L, 95)), "p99_s": float(np.percentile(L, 99)),
                   "mean_s": float(L.mean()), "throughput_img_per_s": float(np.mean(thr)),
                   "throughput_min_max": [float(min(thr)), float(max(thr))], "trials_per_image": sorted(tr)}
            res["rows"].append(row)
            raw[f"{mode}_c{c}"] = L.tolist()
    res["seconds"] = round(time.time() - t0)
    (out / "e6_load_summary.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    (out / "e6_load_latencies.json").write_text(json.dumps(raw), encoding="utf-8")
    for r in res["rows"]:
        print(f"{r['mode']:9s} c={r['concurrency']:3d}: p50 {r['p50_s']:.2f} s, p95 {r['p95_s']:.2f} s, p99 {r['p99_s']:.2f} s, "
              f"{r['throughput_img_per_s']:.2f} img/s")


if __name__ == "__main__":
    main()
