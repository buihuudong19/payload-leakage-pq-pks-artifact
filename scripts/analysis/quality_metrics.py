"""Quality of the stego images: PSNR and SSIM on 1000 images (paper, Section 8.2).

No stego image is regenerated. The confirmatory step ran with --save-stego for every cell of PQ-PKS and REF-U (and BL5, BL5-L at
alpha = 0.2, 0.4), so the stego images used here are exactly those analysed by the wardens:
  results/confirm_v1/raw/<source>_<id>.npz, key a<alpha>_<system>_img  (run_meta.json: "save_stego").
Covers are read with pqpks.data.load_cover (read-only). Nothing in src/ is changed; nothing is written into results/confirm_v1/.

Sample: N images (default 1000) drawn from the 20 000 covers of split_v1, half BOSSbase and half BOWS2, fixed seed (default
20261122); the image list is written to the output. An image without an .npz file or without the key of a cell is skipped for
that cell and counted as missing (never replaced by another image).

Metrics for each pair (cover x, stego y), 8-bit grayscale:
  PSNR = 10 log10(255^2 / MSE) (dB); MSE = 0 is recorded as +inf and counted separately (not averaged).
  SSIM after Wang et al. (2004): Gaussian window sigma = 1.5 (11 x 11, truncate 3.5), K1 = 0.01, K2 = 0.03, L = 255,
       uncorrected sample covariance, averaged over the image without a 5-pixel border. Equal to
       skimage.metrics.structural_similarity(gaussian_weights=True, sigma=1.5, use_sample_covariance=False,
       data_range=255), but needs only scipy.
  Fraction of changed pixels and max |y - x| (+-1 check).
Summary per cell (alpha, system): mean, standard deviation, minimum, median, 95 % bootstrap CI over images (10^4 resamples, fixed
seed); paired difference PQ-PKS - REF-U on the same images, with 95 % CI. Split by source (BOSSbase, BOWS2).

Output (directory --out, default results/quality/):
  quality_per_image.csv   one row per (image, alpha, system)
  quality_summary.json    summary, paired differences, image list, SHA-256 of the split and of the frozen profile
  quality_summary.md      quick-reading table

    .venv-confirm\\Scripts\\python.exe scripts\\analysis\\quality_metrics.py --n 1000 --workers 7
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "confirm"))
sys.path.insert(0, str(ROOT / "src"))

RAW = ROOT / "results" / "confirm_v1" / "raw"
META = ROOT / "results" / "confirm_v1" / "run_meta.json"
N_BOOT = 10_000
SSIM_SIGMA, SSIM_TRUNC, SSIM_PAD = 1.5, 3.5, 5
K1, K2, L = 0.01, 0.03, 255.0


# ---------------------------------------------------------------------------------------------------- metrics
def psnr(x: np.ndarray, y: np.ndarray) -> float:
    mse = float(np.mean((x.astype(np.float64) - y.astype(np.float64)) ** 2))
    return math.inf if mse == 0.0 else 10.0 * math.log10(L * L / mse)


def ssim(x: np.ndarray, y: np.ndarray) -> float:
    from scipy.ndimage import gaussian_filter

    def g(a):
        return gaussian_filter(a, sigma=SSIM_SIGMA, truncate=SSIM_TRUNC, mode="reflect")

    x = x.astype(np.float64)
    y = y.astype(np.float64)
    ux, uy = g(x), g(y)
    vx = g(x * x) - ux * ux
    vy = g(y * y) - uy * uy
    vxy = g(x * y) - ux * uy
    c1, c2 = (K1 * L) ** 2, (K2 * L) ** 2
    s = ((2 * ux * uy + c1) * (2 * vxy + c2)) / ((ux * ux + uy * uy + c1) * (vx + vy + c2))
    p = SSIM_PAD
    return float(s[p:-p, p:-p].mean())


# ---------------------------------------------------------------------------------------------------- image sample
def sample_covers(sp: dict, n: int, seed: int) -> list[tuple[str, int]]:
    boss = sorted(int(i) for part in ("train", "val", "test") for i in sp["boss"][part])
    bows = sorted(int(i) for i in sp["bows2"]["train"])
    rng = np.random.default_rng(seed)
    nb = n // 2
    pick_b = sorted(rng.choice(boss, size=nb, replace=False).tolist())
    pick_w = sorted(rng.choice(bows, size=n - nb, replace=False).tolist())
    return [("boss", i) for i in pick_b] + [("bows2", i) for i in pick_w]


def cells_from_meta(meta: dict) -> list[tuple[str, str]]:
    """Cells (alpha as a string as in the .npz keys, system) with stored stego images."""
    out = []
    for spec in meta["save_stego"]:
        a, name = spec.split(":")
        out.append((str(float(a)), name))
    order = {"PQ-PKS": 0, "REF-U": 1, "BL5-L": 2, "BL5": 3}
    return sorted(set(out), key=lambda t: (float(t[0]), order.get(t[1], 9), t[1]))


# ---------------------------------------------------------------------------------------------------- work
def job(args):
    src, cid, cells = args
    from pqpks import data
    f = RAW / f"{src}_{cid}.npz"
    if not f.exists():
        return src, cid, None
    x = data.load_cover(src, cid)
    rows = []
    with np.load(f) as z:
        keys = set(z.files)
        for a, name in cells:
            k = f"a{a}_{name}_img"
            if k not in keys:
                continue
            y = np.asarray(z[k], dtype=np.uint8)
            if y.shape != x.shape:
                raise ValueError(f"{f.name} {k}: size {y.shape} differs from the cover {x.shape}")
            d = y.astype(np.int16) - x.astype(np.int16)
            rows.append({"src": src, "id": cid, "alpha": a, "system": name,
                         "psnr_db": psnr(x, y), "ssim": ssim(x, y),
                         "change_rate": float((d != 0).mean()), "max_abs_diff": int(np.abs(d).max())})
    return src, cid, rows


# ---------------------------------------------------------------------------------------------------- summary
def boot_ci(v: np.ndarray, rng: np.random.Generator) -> list[float]:
    if v.size < 2:
        return [float("nan"), float("nan")]
    idx = rng.integers(0, v.size, size=(N_BOOT, v.size))
    m = v[idx].mean(axis=1)
    return [float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))]


def describe(v: np.ndarray, rng) -> dict:
    fin = v[np.isfinite(v)]
    return {"n": int(v.size), "n_inf": int(v.size - fin.size),
            "mean": float(fin.mean()) if fin.size else float("nan"),
            "sd": float(fin.std(ddof=1)) if fin.size > 1 else float("nan"),
            "min": float(fin.min()) if fin.size else float("nan"),
            "median": float(np.median(fin)) if fin.size else float("nan"),
            "ci95_mean": boot_ci(fin, rng)}


def summarise(rows: list[dict], cells, seed: int) -> dict:
    rng = np.random.default_rng(seed + 1)
    by = {}
    for r in rows:
        by.setdefault((r["alpha"], r["system"]), {})[(r["src"], r["id"])] = r
    out = {"cells": {}, "paired_PQPKS_minus_REFU": {}}
    for a, name in cells:
        d = by.get((a, name), {})
        cell = {}
        for scope in ("all", "boss", "bows2"):
            rr = [r for (s, _), r in d.items() if scope == "all" or s == scope]
            cell[scope] = {m: describe(np.array([r[m] for r in rr], dtype=float), rng)
                           for m in ("psnr_db", "ssim", "change_rate")}
            cell[scope]["max_abs_diff"] = int(max((r["max_abs_diff"] for r in rr), default=0))
        out["cells"][f"{a}/{name}"] = cell
    alphas = sorted({a for a, _ in cells}, key=float)
    for a in alphas:
        p, u = by.get((a, "PQ-PKS"), {}), by.get((a, "REF-U"), {})
        common = sorted(set(p) & set(u))
        if not common:
            continue
        res = {"n_pairs": len(common)}
        for m in ("psnr_db", "ssim", "change_rate"):
            dv = np.array([p[k][m] - u[k][m] for k in common], dtype=float)
            dv = dv[np.isfinite(dv)]
            res[m] = {"mean_diff": float(dv.mean()) if dv.size else float("nan"), "ci95": boot_ci(dv, rng)}
        out["paired_PQPKS_minus_REFU"][a] = res
    return out


def write_md(path: Path, summ: dict, info: dict) -> None:
    lines = [f"# PSNR / SSIM of the stego images ({info['n_images_requested']} images, seed {info['seed']})", "",
             f"Stego images: {info['raw_dir']} (the images analysed by the wardens). Images without an .npz file: {info['n_missing_npz']}.", "",
             "| α (bpp) | System | n | PSNR (dB), mean [95 % CI] | PSNR min | SSIM, mean [95 % CI] | SSIM min | Change rate | max |y−x| |",
             "|---|---|---|---|---|---|---|---|---|"]
    for key, c in summ["cells"].items():
        a, name = key.split("/", 1)
        P, S, C = c["all"]["psnr_db"], c["all"]["ssim"], c["all"]["change_rate"]
        lines.append(f"| {a} | {name} | {P['n']} | {P['mean']:.3f} [{P['ci95_mean'][0]:.3f}, {P['ci95_mean'][1]:.3f}] | "
                     f"{P['min']:.3f} | {S['mean']:.6f} [{S['ci95_mean'][0]:.6f}, {S['ci95_mean'][1]:.6f}] | {S['min']:.6f} | "
                     f"{C['mean']:.5f} | {c['all']['max_abs_diff']} |")
    lines += ["", "## Paired difference PQ-PKS − REF-U (same images)", "",
              "| α (bpp) | n pairs | ΔPSNR (dB) [95 % CI] | ΔSSIM [95 % CI] | Δ change rate [95 % CI] |", "|---|---|---|---|---|"]
    for a, r in summ["paired_PQPKS_minus_REFU"].items():
        lines.append(f"| {a} | {r['n_pairs']} | {r['psnr_db']['mean_diff']:+.4f} [{r['psnr_db']['ci95'][0]:+.4f}, {r['psnr_db']['ci95'][1]:+.4f}] | "
                     f"{r['ssim']['mean_diff']:+.2e} [{r['ssim']['ci95'][0]:+.2e}, {r['ssim']['ci95'][1]:+.2e}] | "
                     f"{r['change_rate']['mean_diff']:+.2e} [{r['change_rate']['ci95'][0]:+.2e}, {r['change_rate']['ci95'][1]:+.2e}] |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000, help="number of images (half BOSSbase, half BOWS2)")
    ap.add_argument("--seed", type=int, default=20261122)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--out", default=str(ROOT / "results" / "quality"))
    ap.add_argument("--no-frozen-check", action="store_true", help="code tests only, outside the measurement machine")
    a = ap.parse_args()

    from common import load_split, one_thread_per_worker, sha256_file, workers_default  # scripts/confirm/common.py
    frozen = None
    if not a.no_frozen_check:
        from common import check_frozen
        frozen = check_frozen()                      # refuse to run if src/ or configs/ differ from the frozen profile
    sp = load_split()
    meta = json.loads(META.read_text(encoding="utf-8"))
    if meta.get("split_sha256") and meta["split_sha256"] != sp["_sha256"]:
        raise SystemExit("split_v1.json differs from the split file used by the confirmatory step (run_meta.json)")
    cells = cells_from_meta(meta)
    covers = sample_covers(sp, a.n, a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"{len(covers)} images, {len(cells)} cells: {', '.join(f'{x}/{y}' for x, y in cells)}", flush=True)

    one_thread_per_worker()
    rows, missing = [], []
    t0 = time.time()
    with ProcessPoolExecutor(a.workers or workers_default()) as ex:
        for k, (src, cid, rr) in enumerate(ex.map(job, [(s, i, cells) for s, i in covers], chunksize=4), 1):
            if rr is None:
                missing.append(f"{src}_{cid}")
            else:
                rows.extend(rr)
            if k % 50 == 0 or k == len(covers):
                print(f"{k}/{len(covers)} images after {time.time() - t0:.0f}s", flush=True)

    with (out / "quality_per_image.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["src", "id", "alpha", "system", "psnr_db", "ssim", "change_rate", "max_abs_diff"])
        w.writeheader()
        w.writerows(rows)
    summ = summarise(rows, cells, a.seed)
    info = {"n_images_requested": len(covers), "seed": a.seed, "n_missing_npz": len(missing), "missing_npz": missing,
            "raw_dir": str(RAW.relative_to(ROOT)), "split_sha256": sp["_sha256"], "frozen_sha256": frozen,
            "ssim": {"window": "gaussian", "sigma": SSIM_SIGMA, "truncate": SSIM_TRUNC, "K1": K1, "K2": K2, "L": L,
                     "border_cropped": SSIM_PAD, "covariance": "population"},
            "n_boot": N_BOOT, "covers": [f"{s}_{i}" for s, i in covers]}
    summ = {"info": info, **summ}
    (out / "quality_summary.json").write_text(json.dumps(summ, indent=1, ensure_ascii=False), encoding="utf-8")
    write_md(out / "quality_summary.md", summ, info)
    print(f"ghi {out / 'quality_summary.json'}, {out / 'quality_summary.md'}, {out / 'quality_per_image.csv'}", flush=True)
    print((out / "quality_summary.md").read_text(encoding="utf-8"), flush=True)


if __name__ == "__main__":
    main()
