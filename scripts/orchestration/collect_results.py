"""Collect the summary result files into results/manuscript_inputs/ (step R1 of run_all).

Only summary files (JSON, MD) are copied, no large data. Also writes params.md (collect_params.py): every experimental parameter
with its source. INDEX.md records for each file the step that produces it, the section of the paper that uses it, its size,
SHA-256 and modification time. Missing files are listed as "missing" so that unfinished steps are visible at once.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "manuscript_inputs"
ITEMS = [  # (path inside results/, step, use in the paper)
    ("g4/g4_summary.json", "3f", "Section 7.4, SRM G4; G4 deviation for the sensitivity analysis"),
    ("e1_confirm/e1_summary.json", "C3", "§7.5 E1, Fig. 5, Table 9"),
    ("e1bis/e1bis_summary.json", "X5", "Section 7.9, E1-bis, gate G3 of E1-bis"),
    ("e1bis_design/sd_null.json", "X5", "Section 7.9, E1-bis: SD_null of D4"),
    ("confirm_v1/summary.json", "C7", "§7.6 E2/E3, α*, H2a, Table 7, 10, 11, Fig. 6–9"),
    ("confirm_v1/decisions.json", "C8", "Section 7.8, decision rules (SRM only) and sensitivity analysis"),
    ("confirm_v1/feat/avail.json", "C6", "Section 7.6, number of excluded covers"),
    ("srnet/g4_srnet_summary.json", "G7", "Section 7.4, SRNet G4"),
    ("srnet/confirm_srnet_summary.json", "G12", "Section 7.6, confirmatory SRNet"),
    ("srnet/decisions_srnet.json", "G13", "Section 7.8, decision rules with SRNet included"),
    ("e4_full/e4_summary.json", "E4", "§7.7, Table S1"),
    ("e6_desktop/e6_summary.json", "C9", "§7.9, Table 12"),
    ("gpu_bench/decision.json", "G2", "Appendix: run configuration"),
]


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = ["# Collected results", "", f"Collected at {time.strftime('%Y-%m-%d %H:%M:%S')} by scripts/orchestration/collect_results.py.",
            "", "| File | Step | Used in | Size | SHA-256 (first 16 characters) | Modified |", "|---|---|---|---|---|---|"]
    index = []
    for rel, step, use in ITEMS:
        src = ROOT / "results" / rel
        if not src.exists():
            rows.append(f"| `{rel}` | {step} | {use} | missing | - | - |")
            index.append({"file": rel, "step": step, "present": False})
            continue
        dst = OUT / rel.replace("/", "__")
        shutil.copy2(src, dst)
        h = sha256(src)
        t = time.strftime("%Y-%m-%d %H:%M", time.localtime(src.stat().st_mtime))
        rows.append(f"| `{rel}` | {step} | {use} | {src.stat().st_size:,} B | `{h[:16]}` | {t} |")
        index.append({"file": rel, "step": step, "present": True, "sha256": h, "copied_to": dst.name})
    (OUT / "INDEX.md").write_text("\n".join(rows) + "\n", encoding="utf-8")
    (OUT / "index.json").write_text(json.dumps(index, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"{sum(i['present'] for i in index)}/{len(index)} files, see {OUT / 'INDEX.md'}", flush=True)
    # Experimental parameters: regenerate params.md once every measurement is finished
    import subprocess
    import sys
    subprocess.run([sys.executable, str(Path(__file__).with_name("collect_params.py")), "--out", str(OUT / "params.md")], check=False)


if __name__ == "__main__":
    main()
