"""Decide whether two SRNet processes run at the same time on one GPU (step G2b of run_all).

Reads the latest benchmark report results/gpu_bench/<timestamp>/report.json (bench_srnet.py --quick), row 256x256 images,
batch 32, AMP (the training configuration). Two processes only help when one process leaves the GPU idle, so the rule is:
  parallel when the mean GPU utilization is < 85 % AND twice the peak VRAM still fits the card memory (1 GB reserved).
Writes results/gpu_bench/decision.json: parallel, its (end-to-end iterations/s), reason. Runs with any Python.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BENCH = ROOT / "results" / "gpu_bench"
UTIL_MAX, VRAM_SPARE_MB = 85.0, 1024


def decide(report: dict) -> dict:
    row = next((r for r in report["rows"] if r.get("size") == 256 and r.get("batch") == 32 and r.get("amp")
                and r.get("status") == "ok"), None)
    if row is None:
        return {"parallel": False, "its": None, "reason": "no valid 256/b32/AMP row in the report: run sequentially"}
    its = 1000.0 / row["end2end_ms_iter"]
    util = row.get("smi", {}).get("utilization.gpu", {}).get("mean")
    total = report["env"].get("smi", {}).get("memory_total", "")
    try:
        total_mb = float(str(total).split()[0])
    except (ValueError, IndexError):
        total_mb = None
    need = 2 * row["peak_vram_reserved_mb"] + VRAM_SPARE_MB
    fits = total_mb is not None and need <= total_mb
    par = util is not None and util < UTIL_MAX and fits
    why = (f"mean GPU utilization {util} % (threshold {UTIL_MAX:g} %), two processes need about {need} MB, the card has "
           f"{total_mb} MB: " + ("run two processes" if par else "run one process at a time"))
    return {"parallel": bool(par), "its": round(its, 2), "util_mean": util, "vram_needed_mb": need,
            "vram_total_mb": total_mb, "reason": why}


def main():
    reps = sorted(BENCH.glob("*/report.json"))
    if not reps:
        sys.exit("results/gpu_bench/*/report.json not found: run bench_srnet.py --quick first")
    res = decide(json.loads(reps[-1].read_text(encoding="utf-8"))) | {"report": str(reps[-1].relative_to(ROOT))}
    (BENCH / "decision.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    print(res["reason"], f"| speed {res['its']} it/s", flush=True)


if __name__ == "__main__":
    main()
