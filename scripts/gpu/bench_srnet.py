"""Benchmark of SRNet training speed on the GPU of the machine.

Purpose: speed only, NOT accuracy. The "stego" images here are covers plus random +-1 noise, not images of PQ-PKS or of any
baseline.

Each configuration (image size, batch, fp32 or AMP) is measured twice:
  (A) GPU only: one fixed batch kept on the GPU -> pure compute speed;
  (B) end to end: PGM images read from disk through a DataLoader with rotation/flip augmentation -> actual training speed.
If (B) is much slower than (A), the bottleneck is the CPU/disk, not the GPU.

During the measurement a helper thread calls nvidia-smi every 2 seconds to record temperature, clocks, power and VRAM.
Output: results/gpu_bench/<timestamp>/report.json and REPORT.md.

Run (after setup_gpu.ps1), from the repository root:
    .venv-gpu\\Scripts\\python.exe scripts\\gpu\\bench_srnet.py
    .venv-gpu\\Scripts\\python.exe scripts\\gpu\\bench_srnet.py --quick      # about 3 minutes
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from pqpks.steganalysis.srnet import SRNet, param_count  # noqa: E402

SMI_FIELDS = "temperature.gpu,utilization.gpu,clocks.sm,power.draw,memory.used"


# ----------------------------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------------------------
class PairDataset(torch.utils.data.Dataset):
    """Returns (image pair 2x1xSxS, labels [0, 1]). Central SxS crop; stego = cover +- 1 at fraction `rate`."""

    def __init__(self, files: list[str], size: int, rate: float = 0.2, seed: int = 0):
        self.files, self.size, self.rate, self.seed = files, size, rate, seed

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        rng = np.random.default_rng(self.seed * 1_000_003 + i)
        x = np.asarray(Image.open(self.files[i]).convert("L"), dtype=np.int16)
        h, w = x.shape
        s = self.size
        x = x[(h - s) // 2:(h - s) // 2 + s, (w - s) // 2:(w - s) // 2 + s]
        mask = rng.random(x.shape) < self.rate
        y = np.clip(x + mask * rng.choice((-1, 1), x.shape), 0, 255)
        k, flip = int(rng.integers(4)), bool(rng.integers(2))   # synchronized augmentation for the pair
        pair = np.stack([x, y])
        pair = np.rot90(pair, k, axes=(1, 2))
        if flip:
            pair = pair[:, :, ::-1]
        return torch.from_numpy(np.ascontiguousarray(pair, dtype=np.float32)).unsqueeze(1), torch.tensor([0, 1])


def collate(batch):
    xs, ys = zip(*batch)
    return torch.cat(xs), torch.cat(ys)


# ----------------------------------------------------------------------------------------------
# GPU monitoring
# ----------------------------------------------------------------------------------------------
class SmiMonitor(threading.Thread):
    def __init__(self, period: float = 2.0):
        super().__init__(daemon=True)
        self.period, self.samples, self._halt = period, [], threading.Event()

    def run(self):
        while not self._halt.is_set():
            try:
                out = subprocess.run(["nvidia-smi", f"--query-gpu={SMI_FIELDS}", "--format=csv,noheader,nounits"],
                                     capture_output=True, text=True, timeout=5).stdout.strip().splitlines()[0]
                self.samples.append([float(v) if v.strip() not in ("[N/A]", "N/A") else float("nan") for v in out.split(",")])
            except Exception:  # noqa: BLE001  (skipped if nvidia-smi is unavailable)
                pass
            self._halt.wait(self.period)

    def stop(self) -> dict:
        self._halt.set()
        self.join(timeout=5)
        if not self.samples:
            return {}
        a = np.array(self.samples)
        keys = SMI_FIELDS.split(",")
        return {k: {"mean": round(float(np.nanmean(a[:, j])), 1), "max": round(float(np.nanmax(a[:, j])), 1)}
                for j, k in enumerate(keys)} | {"n_samples": len(a)}


def smi_static() -> dict:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,power.limit,clocks.max.sm",
                              "--format=csv,noheader"], capture_output=True, text=True, timeout=10).stdout.strip()
        return dict(zip(["name", "driver", "memory_total", "power_limit", "max_sm_clock"], [v.strip() for v in out.split(",")]))
    except Exception:  # noqa: BLE001
        return {}


# ----------------------------------------------------------------------------------------------
# Measurement
# ----------------------------------------------------------------------------------------------
def train_steps(model, opt, scaler, batches, iters: int, warmup: int, amp: bool, dev) -> float:
    """Run warmup + iters steps; returns the seconds taken by the `iters` measured steps."""
    loss_fn = nn.CrossEntropyLoss()
    it = iter(batches)
    t0 = None
    for step in range(warmup + iters):
        if step == warmup:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
        x, y = next(it)
        x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            loss = loss_fn(model(x), y)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
    torch.cuda.synchronize()
    return time.perf_counter() - t0


def run_config(files, size, batch, amp, iters, warmup, workers, dev) -> dict:
    torch.manual_seed(0)
    model = SRNet().to(dev).to(memory_format=torch.channels_last)
    opt = torch.optim.Adamax(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    pairs = batch // 2
    ds = PairDataset(files, size)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    res = {"size": size, "batch": batch, "amp": amp}
    try:
        # (A) GPU only
        fixed = collate([ds[i] for i in range(pairs)])
        fixed = (fixed[0].to(dev).contiguous(memory_format=torch.channels_last), fixed[1].to(dev))
        sec = train_steps(model, opt, scaler, _repeat(fixed), iters, warmup, amp, dev)
        res["gpu_only_img_s"] = round(iters * batch / sec, 1)
        res["gpu_only_ms_iter"] = round(1000 * sec / iters, 2)
        # (B) end to end
        dl = torch.utils.data.DataLoader(ds, batch_size=pairs, shuffle=True, num_workers=workers,
                                         pin_memory=True, persistent_workers=workers > 0, collate_fn=collate,
                                         drop_last=True)
        sec = train_steps(model, opt, scaler, _cycle(dl), iters, warmup, amp, dev)
        res["end2end_img_s"] = round(iters * batch / sec, 1)
        res["end2end_ms_iter"] = round(1000 * sec / iters, 2)
        res["peak_vram_alloc_mb"] = round(torch.cuda.max_memory_allocated() / 2**20)
        res["peak_vram_reserved_mb"] = round(torch.cuda.max_memory_reserved() / 2**20)
        res["status"] = "ok"
    except torch.cuda.OutOfMemoryError:
        res["status"] = "OOM"
    finally:
        del model, opt
        torch.cuda.empty_cache()
    return res


def _repeat(b):
    while True:
        yield b


def _cycle(dl):
    while True:
        yield from dl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(ROOT / "data" / "external" / "bossbase"))
    ap.add_argument("--n-images", type=int, default=2000, help="number of cover images used for the measurement (read from disk)")
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--quick", action="store_true", help="measure only 256/b32 and 512/b8, 100 iterations each")
    a = ap.parse_args()

    if not torch.cuda.is_available():
        sys.exit("CUDA not available: check the NVIDIA driver and the PyTorch build (must be +cu126).")
    dev = torch.device("cuda")
    torch.backends.cudnn.benchmark = True

    files = sorted((str(p) for p in Path(a.data_dir).glob("*.pgm")), key=lambda s: int(Path(s).stem))[: a.n_images]
    if len(files) < 64:
        sys.exit(f"Not enough images in {a.data_dir} (>= 64 .pgm files needed).")

    if a.quick:
        configs = [(256, 32, False), (256, 32, True), (512, 8, False), (512, 8, True)]
        a.iters = min(a.iters, 100)
    else:
        configs = [(256, 32, False), (256, 32, True),
                   (512, 32, False), (512, 32, True), (512, 16, False), (512, 16, True), (512, 8, False), (512, 8, True)]

    out_dir = ROOT / "results" / "gpu_bench" / dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    env = {"python": sys.version.split()[0], "torch": torch.__version__, "cuda": torch.version.cuda,
           "cudnn": torch.backends.cudnn.version(), "gpu": torch.cuda.get_device_name(0),
           "os": platform.platform(), "cpu": platform.processor(), "cpu_count": os.cpu_count(),
           "smi": smi_static(), "srnet_params": param_count(SRNet()),
           "data_dir": a.data_dir, "n_images": len(files), "iters": a.iters, "warmup": a.warmup, "workers": a.workers}
    print(json.dumps(env, indent=2, ensure_ascii=False), flush=True)

    rows = []
    for size, batch, amp in configs:
        mon = SmiMonitor()
        mon.start()
        t = time.time()
        r = run_config(files, size, batch, amp, a.iters, a.warmup, a.workers, dev)
        r["smi"] = mon.stop()
        r["wall_s"] = round(time.time() - t, 1)
        rows.append(r)
        print(json.dumps(r, ensure_ascii=False), flush=True)
        (out_dir / "report.json").write_text(json.dumps({"env": env, "rows": rows}, indent=2, ensure_ascii=False), encoding="utf-8")

    # Training time forecast: normalized to "iterations of batch 32" (16 pairs) for comparison with the original SRNet schedule
    md = ["# SRNet speed on the GPU", "",
          f"- GPU: {env['gpu']} (driver {env['smi'].get('driver', '?')}), PyTorch {env['torch']}, CUDA {env['cuda']}, cuDNN {env['cudnn']}",
          f"- Python {env['python']}, {env['os']}, {env['cpu_count']} CPU threads; SRNet {env['srnet_params']:,} parameters",
          f"- {a.iters} measured iterations after {a.warmup} warm-up iterations; DataLoader {a.workers} workers; images from `{a.data_dir}`",
          "- The stego images here are random +-1 noise: the figures measure speed only.", "",
          "| Image | Batch | AMP | Status | GPU only (img/s) | End to end (img/s) | Peak VRAM (MB) | Max temperature (°C) | Mean SM clock (MHz) | Mean power (W) | Hours / 100k it b32 |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if r["status"] != "ok":
            md.append(f"| {r['size']} | {r['batch']} | {'yes' if r['amp'] else 'no'} | {r['status']} | – | – | – | – | – | – | – |")
            continue
        s = r["smi"]
        h100k = 100_000 * 32 / r["end2end_img_s"] / 3600
        md.append(f"| {r['size']} | {r['batch']} | {'yes' if r['amp'] else 'no'} | ok | {r['gpu_only_img_s']} | {r['end2end_img_s']} | "
                  f"{r['peak_vram_reserved_mb']} | {s.get('temperature.gpu', {}).get('max', '?')} | {s.get('clocks.sm', {}).get('mean', '?')} | "
                  f"{s.get('power.draw', {}).get('mean', '?')} | {h100k:.1f} |")
    md += ["", "_Hours / 100k it b32 = time for 100 000 iterations of 32 images (16 pairs) at the end-to-end speed._",
           "_If the end-to-end speed is much lower than GPU only, the bottleneck is image loading/CPU: increase --workers or preload the images into RAM._"]
    (out_dir / "REPORT.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    print(f"\nWritten: {out_dir}")


if __name__ == "__main__":
    main()
