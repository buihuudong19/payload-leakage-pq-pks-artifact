"""Create and freeze the split file of the confirmatory step (paper, Section 7.2). Run ONCE on a machine with all
10 000 BOSSbase and 10 000 BOWS2 images; nothing is written into data/.

Rules:
- Known byte-identical pairs (paper, Section 7.2) and near-duplicate pairs are grouped; all images of a group fall into the same
  part. Near-duplicate pair: 64-bit dHash distance <= 6 bits AND correlation coefficient of the 32x32 thumbnails >= 0.97 (the
  second condition excludes flat images whose dHash coincide by chance, which would otherwise chain into large groups).
- BOSSbase: the 1000 pilot images are in train only. The other 9000 images: test 5000, validation 1000, train 3000 (seed 20261106).
- BOWS2 (10 000 images numbered 0-10000, 3661 missing): all in the train part of E2/E3. The 1000 screening images (seed 20261105)
  are only in the training part of E1.
- E1: 5-fold cross-validation over the images that are neither pilot nor screening images nor in a near-duplicate group with them
  (split_v1: 17 979 images), stratified by source, groups kept together.
- SRM: 10 random splits of the 9000 non-pilot BOSSbase images into train 4000 / test 5000 (pilot images always in train).
Writes configs/splits/split_v1.json and split_v1.sha256. Refuses to overwrite an existing file (use --force deliberately).

Run:  python scripts/confirm/make_split.py --workers 8
"""
from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from common import ROOT, SPLIT, one_thread_per_worker  # noqa: E402
from pqpks import data  # noqa: E402

KNOWN_PAIRS = [("boss", 5341, 5536), ("boss", 5342, 5537), ("boss", 5343, 5538), ("boss", 5344, 5539),
               ("bows2", 6532, 6533), ("bows2", 7175, 7263), ("bows2", 7177, 7265), ("bows2", 7577, 7602),
               ("bows2", 7808, 7856)]
SEED_BOSS, N_TEST, N_VAL = 20261106, 5000, 1000


def dhash(item):
    src, i = item
    from PIL import Image
    x = data.load_cover(src, i)
    im = Image.fromarray(x).resize((9, 8), Image.BILINEAR)
    a = np.asarray(im, dtype=np.int16)
    bits = (a[:, 1:] > a[:, :-1]).ravel()
    thumb = np.asarray(Image.fromarray(x).resize((32, 32), Image.BILINEAR), dtype=np.float32).ravel()
    return src, i, int("".join("1" if b else "0" for b in bits), 2), thumb


def groups(items, hashes, thumbs, max_ham=6, min_corr=0.97):
    parent = {it: it for it in items}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for src, a, b in KNOWN_PAIRS:
        if (src, a) in parent and (src, b) in parent:
            union((src, a), (src, b))
    keys = list(hashes)
    H = np.array([hashes[k] for k in keys], dtype=np.uint64)
    near = []
    for idx in range(len(keys)):                     # O(n^2) in blocks; 20 000 images: a few minutes
        x = np.bitwise_xor(H[idx + 1:], H[idx])
        d = np.unpackbits(x.view(np.uint8).reshape(-1, 8), axis=1).sum(1)
        for j in np.nonzero(d <= max_ham)[0]:
            ka, kb = keys[idx], keys[idx + 1 + j]
            c = float(np.corrcoef(thumbs[ka], thumbs[kb])[0, 1])
            if c >= min_corr:
                union(ka, kb)
                near.append((ka, kb, int(d[j]), round(c, 4)))
    g = {}
    for it in items:
        g.setdefault(find(it), []).append(it)
    return list(g.values()), near


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--smoke", help="code tests only: use the available images and write to this path instead of split_v1.json")
    a = ap.parse_args()
    out_path = Path(a.smoke) if a.smoke else SPLIT
    if out_path.exists() and not a.force:
        raise SystemExit(f"{out_path} exists (frozen); not overwritten")
    # Image ids are taken from the directory, not assumed to be 1..10000: BOWS2-Original is numbered 0-10000 with 3661 missing (data/README.md).
    boss = sorted(("boss", int(p.stem)) for p in data.bossbase_dir().glob("*.pgm"))
    bows = sorted(("bows2", int(p.stem)) for p in data.bows2_dir().glob("*.pgm"))
    if not a.smoke:
        ids_boss, ids_bows = {i for _, i in boss}, {i for _, i in bows}
        if ids_boss != set(range(1, 10001)):
            raise SystemExit(f"BOSSbase must contain exactly 1.pgm ... 10000.pgm; found {len(ids_boss)} images")
        if ids_bows != (set(range(0, 10001)) - {3661}):
            raise SystemExit(f"BOWS2 must contain exactly 0.pgm ... 10000.pgm except 3661.pgm; found {len(ids_bows)} images")
    one_thread_per_worker()
    with ProcessPoolExecutor(a.workers) as ex:
        res = list(ex.map(dhash, boss + bows, chunksize=64))
    hs = {(s, i): h for s, i, h, _ in res}
    th = {(s, i): t for s, i, _, t in res}
    G, near = groups(boss + bows, hs, th)
    gid = {it: k for k, grp in enumerate(G) for it in grp}
    pilot = {("boss", i) for i in data.pilot_ids()} & set(boss)
    screen = {("bows2", i) for i in data.screen_bows2_ids()} & set(bows)
    # groups containing a pilot (or screening) image, or spanning both sources: train only (never test/validation/E1)
    tainted = {gid[it] for it in pilot | screen} | {k for k, grp in enumerate(G) if len({s for s, _ in grp}) > 1}
    rng = np.random.default_rng(SEED_BOSS)
    boss_groups = sorted({gid[it] for it in boss if gid[it] not in tainted})
    rng.shuffle(boss_groups)
    test, val, train_b = [], [], []
    for g in boss_groups:
        members = [it for it in G[g] if it[0] == "boss"]
        if len(test) + len(members) <= N_TEST:
            test += members
        elif len(val) + len(members) <= N_VAL:
            val += members
        else:
            train_b += members
    train_b += [it for it in boss if gid[it] in tainted and it[0] == "boss"]
    # E1: about 18 000 images (split_v1: 17 979), 5 folds, stratified by source, groups kept; pilot + screening (and their groups) training only
    e1_pool = [it for it in boss + bows if gid[it] not in tainted]
    e1_set = set(e1_pool)
    e1_fold = {}
    for src in ("boss", "bows2"):
        gs = sorted({gid[it] for it in e1_pool if it[0] == src})
        np.random.default_rng(SEED_BOSS + (0 if src == "boss" else 1)).shuffle(gs)
        for k, g in enumerate(gs):
            for it in G[g]:
                if it[0] == src and it in e1_set:
                    e1_fold[it] = k % 5
    # SRM: 10 splits of the 9000 non-pilot BOSSbase images (groups kept) into train 4000 / test 5000
    srm_splits = []
    nonpilot_groups = sorted({gid[it] for it in boss if gid[it] not in tainted})
    for r in range(10):
        gs = list(nonpilot_groups)
        np.random.default_rng(SEED_BOSS + 100 + r).shuffle(gs)
        te = []
        for g in gs:
            m = [it[1] for it in G[g] if it[0] == "boss"]
            if len(te) + len(m) <= N_TEST:
                te += m
        srm_splits.append(sorted(te))
    out = {
        "version": "split_v1", "seed": SEED_BOSS,
        "boss": {"test": sorted(i for _, i in test), "val": sorted(i for _, i in val), "train": sorted(i for _, i in train_b),
                 "pilot": sorted(i for _, i in pilot)},
        "bows2": {"train": sorted(i for _, i in bows), "screening": sorted(i for _, i in screen)},
        "e1": {"folds": {f"{s}_{i}": f for (s, i), f in sorted(e1_fold.items())},
               "train_only": sorted(f"{s}_{i}" for (s, i) in boss + bows if gid[(s, i)] in tainted)},
        "srm_repeated_test_sets": srm_splits,
        "near_duplicates": [[f"{a[0]}_{a[1]}", f"{b[0]}_{b[1]}", d, c] for a, b, d, c in near],
        "groups_with_more_than_one_image": [[f"{s}_{i}" for s, i in grp] for grp in G if len(grp) > 1],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=0), encoding="utf-8")
    h = hashlib.sha256(out_path.read_bytes()).hexdigest()
    out_path.with_suffix(".sha256").write_text(f"{h}  {out_path.name}\n", encoding="utf-8")
    print(f"test {len(out['boss']['test'])}, val {len(out['boss']['val'])}, train BOSS {len(out['boss']['train'])}, "
          f"E1 fold images {len(out['e1']['folds'])}, near-duplicate pairs {len(near)}, "
          f"largest group {max(len(g) for g in G)}")
    print("SHA-256:", h)


if __name__ == "__main__":
    main()
