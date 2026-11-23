"""Prepare the SRNet data: uint8 arrays on disk that the training script in .venv-gpu reads directly.

Two datasets, both from split_v1 (train 4000 BOSSbase + 10 000 BOWS2, validation 1000, test 5000 BOSSbase; the sizes of the
[BCF19] protocol):

  g4       SRNet G4. BOSSbase and BOWS2 images downsampled to 256x256 with MATLAB imresize (matlab_imresize.py), then embedded
           with HILL and S-UNIWARD at 0.4 and 0.2 bpp by the conseal simulator (the same `g4_calibrate.stego` as SRM G4).
           Embedding seed: image id for BOSSbase, 100 000 + image id for BOWS2. No PQ-PKS code is used.
  confirm  Confirmatory SRNet. Original 512x512 covers and PQ-PKS, REF-U stego images at alpha = 0.4 and 0.2, taken from the files
           results/confirm_v1/raw/*.npz that step 5a writes with --save-stego (key a<alpha>_<system>_img).

Output: results/srnet/data_<name>/<set>.npy (N x S x S, uint8), index.json (image order, train/val/test indices, which
images have a stego for which set). Resumable after an interruption (each finished chunk has a marker file in done/).

    python scripts/confirm/srnet_prepare.py g4 --workers 7
    python scripts/confirm/srnet_prepare.py confirm --workers 7
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from common import ROOT, check_frozen, load_split, one_thread_per_worker, sha256_file, workers_default  # noqa: E402
from pqpks import data  # noqa: E402

CFG = json.loads((ROOT / "configs" / "srnet_runs_v2.json").read_text(encoding="utf-8"))
CHUNK = 200
BOWS2_SEED_OFFSET = 100_000


def items_from_split(sp: dict) -> list[tuple[str, int, str]]:
    """Fixed image order in every array: BOSSbase train, val, test, then BOWS2 train."""
    out = [("boss", int(i), part) for part in ("train", "val", "test") for i in sp["boss"][part]]
    return out + [("bows2", int(i), "train") for i in sp["bows2"]["train"]]


def parse_set(name: str) -> tuple[str, float]:
    """'hill_0.4' → ('hill', 0.4); 'a0.4_PQ-PKS' → ('PQ-PKS', 0.4)."""
    if name.startswith("a"):
        a, sysname = name[1:].split("_", 1)
        return sysname, float(a)
    m, a = name.rsplit("_", 1)
    return m, float(a)


def _open(dirp, name, shape, mode):
    return np.lib.format.open_memmap(dirp / f"{name}.npy", mode=mode, dtype=np.uint8, shape=shape)


def job_g4(args):
    dirp, k, chunk, n, size, sets = args
    from g4_calibrate import stego                         # the same simulator as SRM G4
    from matlab_imresize import imresize_uint8
    mm = {s: _open(dirp, s, (n, size, size), "r+") for s in sets}
    for pos, src, i in chunk:
        x = imresize_uint8(data.load_cover(src, i), (size, size))
        mm["cover"][pos] = x
        for s in sets[1:]:
            m, a = parse_set(s)
            mm[s][pos] = stego(m, x, a, seed=i if src == "boss" else BOWS2_SEED_OFFSET + i)
    for m in mm.values():
        m.flush()
    (dirp / "done" / f"chunk_{k:03d}").write_text("ok")
    return k


def job_confirm(args):
    dirp, k, chunk, n, size, sets = args
    raw = ROOT / "results" / "confirm_v1" / "raw"
    mm = {s: _open(dirp, s, (n, size, size), "r+") for s in sets}
    have = {s: [] for s in sets[1:]}
    for pos, src, i in chunk:
        mm["cover"][pos] = data.load_cover(src, i)
        f = raw / f"{src}_{i}.npz"
        z = np.load(f) if f.exists() else None
        for s in sets[1:]:
            key = f"{s}_img"
            if z is not None and key in z.files:
                mm[s][pos] = z[key]
                have[s].append(pos)
    for m in mm.values():
        m.flush()
    (dirp / "done" / f"chunk_{k:03d}.json").write_text(json.dumps(have))
    return k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=["g4", "confirm"])
    ap.add_argument("--workers", type=int, default=workers_default())
    ap.add_argument("--limit", type=int, default=None, help="code tests only: number of leading images of each train/val/test part")
    a = ap.parse_args()
    frozen = check_frozen()
    sp = load_split()
    ds = CFG["datasets"][a.dataset]
    dirp = ROOT / ds["dir"]
    (dirp / "done").mkdir(parents=True, exist_ok=True)
    items = items_from_split(sp)
    if a.limit:
        cnt: dict = {}
        keep = []
        for it in items:
            key = (it[0], it[2])
            if cnt.get(key, 0) < a.limit:
                keep.append(it)
                cnt[key] = cnt.get(key, 0) + 1
        items = keep
    n, size, sets = len(items), ds["size"], ds["sets"]
    for s in sets:
        if not (dirp / f"{s}.npy").exists():
            _open(dirp, s, (n, size, size), "w+").flush()
    index = {"dataset": a.dataset, "size": size, "sets": sets, "split_sha256": sp["_sha256"], "frozen_sha256": frozen,
             "ids": [f"{s}_{i}" for s, i, _ in items],
             "train": [p for p, it in enumerate(items) if it[2] == "train"],
             "val": [p for p, it in enumerate(items) if it[2] == "val"],
             "test": [p for p, it in enumerate(items) if it[2] == "test"]}
    idx_p = dirp / "index.json"
    if idx_p.exists():
        old = json.loads(idx_p.read_text(encoding="utf-8"))
        if old["ids"] != index["ids"]:
            raise SystemExit(f"{idx_p} exists with another image order (earlier run with another --limit?): delete {dirp} and rerun")
    chunks = [(k, [(p, items[p][0], items[p][1]) for p in range(k * CHUNK, min(n, (k + 1) * CHUNK))])
              for k in range((n + CHUNK - 1) // CHUNK)]
    done_name = (lambda k: f"chunk_{k:03d}") if a.dataset == "g4" else (lambda k: f"chunk_{k:03d}.json")
    todo = [c for c in chunks if not (dirp / "done" / done_name(c[0])).exists()]
    print(f"{a.dataset}: {n} images, {len(sets)} sets, {len(chunks)} chunks, {len(todo)} chunks left", flush=True)
    t0 = time.time()
    one_thread_per_worker()
    fn = job_g4 if a.dataset == "g4" else job_confirm
    with ProcessPoolExecutor(a.workers) as ex:
        for m, k in enumerate(ex.map(fn, [(dirp, k, c, n, size, sets) for k, c in todo]), 1):
            print(f"chunk {k} done - {m}/{len(todo)} after {time.time() - t0:.0f}s", flush=True)
    if a.dataset == "confirm":
        have = {s: [] for s in sets[1:]}
        for k, _ in chunks:
            for s, v in json.loads((dirp / "done" / done_name(k)).read_text()).items():
                have[s] += v
        index["available"] = {s: sorted(v) for s, v in have.items()}
        index["missing"] = {s: n - len(v) for s, v in have.items()}
        print("images without a stego, by set:", index["missing"], flush=True)
    index["array_sha256"] = {s: sha256_file(dirp / f"{s}.npy") for s in sets}
    idx_p.write_text(json.dumps(index), encoding="utf-8")
    print(f"ghi {idx_p}", flush=True)


if __name__ == "__main__":
    main()
