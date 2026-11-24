"""Clean-environment check of the released artifacts (post hoc, 24 Nov 2026; question Q7 of the regression audit).

Run inside a fresh clone of a release tag, in a fresh virtual environment (scripts/verify/clean_env_check.ps1 prepares both).
The script
  1. checks the SHA-256 of the core archive of the dataset record against its published .sha256 file and unpacks it;
  2. copies its results/ tree into this clone, keeping a copy of every archived output that will be regenerated;
  3. REGENERATES from the archived inputs, with the released code:
       - results/RESULTS_TABLES.md          (write_results_confirm.py, step R0/PF6; Tables 9-12 and S1)
       - results/phase_d/PHASE_D_TABLES_en.md (phase_d.py tables, step PF5; supplementary tables of phases D and E)
       - results/confirm_v1/alpha_star_thresholds.json (alpha_star_thresholds.py, step PH3; Table S13)
       - the aggregates of results/quality/quality_summary.json from quality_per_image.csv (Table S12)
       - optionally (--syndromes): the code check of e1_posthoc_refit.py --quick, which recomputes D4 of E1 on one cell from
         the raw syndromes and stops unless it equals e1_summary.json (Table S9; the full refit takes about 4 h and is not rerun);
     and compares each regenerated output with the archived one (byte equality for text, numerical equality for JSON);
  4. writes clean_env_check.json: environment, commit, archive hashes, and for every output "rerun, identical",
     "rerun, differs" (with the first differences) or "not rerun" (with the reason).
Nothing is downloaded by this script and nothing outside the clone is written.

    python scripts/verify/clean_env_check.py --core-zip <path>/pq-pks-raw-data-v1-core.zip [--syndromes <dir with chunk_*.npz>]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results"
NAME = "pq-pks-raw-data-v1"


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def run(args: list[str], log: list) -> tuple[int, str]:
    t = time.time()
    p = subprocess.run([sys.executable, *args], cwd=ROOT, capture_output=True, text=True)
    out = (p.stdout + p.stderr)[-4000:]
    log.append({"cmd": " ".join(args), "returncode": p.returncode, "seconds": round(time.time() - t, 1), "tail": out[-1500:]})
    return p.returncode, out


def json_diff(a, b, path="", tol=1e-12, out=None, limit=10):
    """First differences between two JSON values (numbers compared with a relative tolerance)."""
    out = [] if out is None else out
    if len(out) >= limit:
        return out
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                out.append(f"{path}/{k}: present in only one file")
            else:
                json_diff(a[k], b[k], f"{path}/{k}", tol, out, limit)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append(f"{path}: lengths {len(a)} and {len(b)}")
        for i, (x, y) in enumerate(zip(a, b)):
            json_diff(x, y, f"{path}[{i}]", tol, out, limit)
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        if not (math.isclose(a, b, rel_tol=tol, abs_tol=tol) or (math.isnan(a) and math.isnan(b))):
            out.append(f"{path}: {a!r} != {b!r}")
    elif a != b:
        out.append(f"{path}: {str(a)[:60]!r} != {str(b)[:60]!r}")
    return out


def text_diff(a: Path, b: Path, limit=5) -> list[str]:
    la, lb = a.read_text(encoding="utf-8").splitlines(), b.read_text(encoding="utf-8").splitlines()
    d = [f"line {i + 1}: {x[:90]!r} | {y[:90]!r}" for i, (x, y) in enumerate(zip(la, lb)) if x != y]
    if len(la) != len(lb):
        d.append(f"{len(la)} and {len(lb)} lines")
    return d[:limit]


def compare_text(name, archived: Path, regenerated: Path, rc: int, out_tail: str) -> dict:
    if rc != 0 or not regenerated.exists():
        return {"output": name, "status": "rerun failed", "detail": out_tail[-800:]}
    ba, bb = archived.read_bytes(), regenerated.read_bytes()
    same = ba == bb
    crlf, lf = bytes([13, 10]), bytes([10])
    if not same and ba.replace(crlf, lf) == bb.replace(crlf, lf):
        return {"output": name, "status": "rerun, identical", "note": "identical text; only the line endings (CRLF/LF) differ",
                "sha256_archived": sha(archived), "sha256_regenerated": sha(regenerated)}
    return {"output": name, "status": "rerun, identical" if same else "rerun, differs",
            "sha256_archived": sha(archived), "sha256_regenerated": sha(regenerated),
            "first_differences": [] if same else text_diff(archived, regenerated)}


def quality_check(qdir: Path) -> dict:
    sys.path[:0] = [str(ROOT / "scripts" / "analysis"), str(ROOT / "scripts" / "confirm"), str(ROOT / "src")]
    import quality_metrics as qm
    arch = json.loads((qdir / "quality_summary.json").read_text(encoding="utf-8"))
    rows = []
    with (qdir / "quality_per_image.csv").open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            rows.append({"src": r["src"], "id": int(r["id"]), "alpha": r["alpha"], "system": r["system"],
                         "psnr_db": float(r["psnr_db"]), "ssim": float(r["ssim"]), "change_rate": float(r["change_rate"]),
                         "max_abs_diff": int(r["max_abs_diff"])})
    cells = qm.cells_from_meta(json.loads((RES / "confirm_v1" / "run_meta.json").read_text(encoding="utf-8")))   # same order as the run
    summ = qm.summarise(rows, cells, arch["info"]["seed"])
    d = json_diff({k: arch[k] for k in ("cells", "paired_PQPKS_minus_REFU")},
                  {k: summ[k] for k in ("cells", "paired_PQPKS_minus_REFU")}, tol=1e-9)
    return {"output": "results/quality/quality_summary.json (aggregates, from quality_per_image.csv)",
            "status": "rerun, identical" if not d else "rerun, differs", "first_differences": d,
            "note": "the per-image PSNR/SSIM need the stego images, which are not archived; only the aggregation is rerun"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--core-zip", required=True)
    ap.add_argument("--sha256-file", help="published .sha256 of the core archive (default: <core-zip>.sha256 if present)")
    ap.add_argument("--syndromes", help="folder where the e1-syndromes archive was unpacked (contains pq-pks-raw-data-v1/results/e1_confirm/raw)")
    ap.add_argument("--out", default=str(ROOT / "clean_env_check.json"))
    a = ap.parse_args()
    t0 = time.time()
    log, checks = [], []
    z = Path(a.core_zip)
    shaf = Path(a.sha256_file) if a.sha256_file else Path(str(z) + ".sha256")
    zsha = sha(z)
    published = shaf.read_text().split()[0] if shaf.exists() else None
    if published is not None and published != zsha:
        raise SystemExit(f"SHA-256 of {z.name} is {zsha}, the published value is {published}: stop")
    git = subprocess.run(["git", "describe", "--tags", "--always", "--dirty"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True).stdout.splitlines()
    # 1-2: unpack and copy results/
    work = ROOT / "_clean_env_archive"
    if work.exists():
        shutil.rmtree(work)
    with zipfile.ZipFile(z) as zf:
        zf.extractall(work)
    src = work / NAME / "results"
    shutil.copytree(src, RES, dirs_exist_ok=True)
    keep = ROOT / "_clean_env_archived_outputs"
    if keep.exists():
        shutil.rmtree(keep)
    targets = ["RESULTS_TABLES.md", "phase_d/PHASE_D_TABLES_en.md", "confirm_v1/alpha_star_thresholds.json"]
    for t in targets:
        (keep / t).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(RES / t, keep / t)
    # 3a: RESULTS_TABLES.md
    regen = ROOT / "_clean_env_regenerated"
    regen.mkdir(exist_ok=True)
    rc, tail = run(["scripts/experiments/write_results_confirm.py", "--out", str(regen / "RESULTS_TABLES.md")], log)
    checks.append(compare_text("results/RESULTS_TABLES.md", keep / "RESULTS_TABLES.md", regen / "RESULTS_TABLES.md", rc, tail))
    # 3b: PHASE_D_TABLES_en.md (written in place by phase_d.py)
    rc, tail = run(["scripts/confirm/phase_d.py", "tables"], log)
    shutil.copy2(RES / "phase_d" / "PHASE_D_TABLES_en.md", regen / "PHASE_D_TABLES_en.md") if (RES / "phase_d" / "PHASE_D_TABLES_en.md").exists() else None
    checks.append(compare_text("results/phase_d/PHASE_D_TABLES_en.md", keep / "phase_d" / "PHASE_D_TABLES_en.md",
                               regen / "PHASE_D_TABLES_en.md", rc, tail))
    # 3c: alpha* thresholds (Table S13)
    rc, tail = run(["scripts/analysis/alpha_star_thresholds.py", "results/confirm_v1/summary.json", "--srnet",
                    "results/srnet/decisions_srnet.json", "--out", str(regen / "alpha_star_thresholds.json")], log)
    if rc == 0:
        d = json_diff(json.loads((keep / "confirm_v1" / "alpha_star_thresholds.json").read_text(encoding="utf-8")),
                      json.loads((regen / "alpha_star_thresholds.json").read_text(encoding="utf-8")))
        d = [x for x in d if "sha256" not in x and "generated" not in x]
        checks.append({"output": "results/confirm_v1/alpha_star_thresholds.json (Table S13)",
                       "status": "rerun, identical" if not d else "rerun, differs", "first_differences": d})
    else:
        checks.append({"output": "results/confirm_v1/alpha_star_thresholds.json (Table S13)", "status": "rerun failed", "detail": tail[-800:]})
    # 3d: quality aggregates (Table S12)
    try:
        checks.append(quality_check(RES / "quality"))
    except Exception as e:                       # noqa: BLE001
        checks.append({"output": "results/quality/quality_summary.json", "status": "rerun failed", "detail": repr(e)})
    # 3e: D4 of E1 from the raw syndromes (Table S9 code check)
    if a.syndromes:
        raw = Path(a.syndromes) / NAME / "results" / "e1_confirm" / "raw"
        if not raw.exists():
            raw = Path(a.syndromes)
        shutil.copytree(raw, RES / "e1_confirm" / "raw", dirs_exist_ok=True)
        rc, tail = run(["scripts/confirm/e1_posthoc_refit.py", "--e1", "results/e1_confirm", "--quick"], log)
        ok = rc == 0 and "equal to e1_summary.json: True" in tail
        checks.append({"output": "Table S9 (D4 of E1): code check of e1_posthoc_refit.py --quick", "status":
                       "rerun, identical" if ok else ("rerun, differs" if rc == 0 else "rerun failed"), "detail": tail[-600:]})
    else:
        checks.append({"output": "Table S9 (refit of D4)", "status": "not rerun",
                       "detail": "needs the e1-syndromes archive (--syndromes) for the code check, and about 4 h for the full refit"})
    res = {"label": "clean-environment check (post hoc, Q7)", "seconds": round(time.time() - t0), "git": git,
           "python": sys.version, "platform": platform.platform(), "pip_freeze": freeze,
           "core_zip": {"file": z.name, "sha256": zsha, "published_sha256": published}, "checks": checks, "commands": log}
    Path(a.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    for c in checks:
        print(f"{c['status']:18s} {c['output']}")
        for x in c.get("first_differences", [])[:3]:
            print("      ", x)
    print("written", a.out)


if __name__ == "__main__":
    main()
