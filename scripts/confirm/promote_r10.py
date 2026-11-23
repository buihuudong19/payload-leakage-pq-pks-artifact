"""Step PF2: make the 10-split analysis the main version of results/confirm_v1.

Why: C7 (3 splits) writes summary.json/decisions.json, and every later step (srnet_evaluate confirm, apply_rules, e5_budget,
write_results_confirm, collect_results, make_figures_confirm) reads these two names. Table 7 registers >= 10 splits; PE1/PF1
write summary_r10.json. Instead of changing every reader, this step:
  1. keeps the 3-split version as summary_r3.json / decisions_r3.json (renamed only when the current file has reps = 3 and the
     new name does not exist; nothing is ever deleted);
  2. copies summary_r10.json (reps = 10, with the "bootstrap" record) to summary.json;
  3. does NOT copy decisions: PF3 (apply_rules --no-srnet) recomputes decisions.json from the new summary.json, then G13 adds SRNet.
Refuses if summary_r10.json is missing, reps != 10, or the "bootstrap" record is missing (i.e. an older PE1 version not processed by PF1).

    python scripts/confirm/promote_r10.py [--base results/confirm_v1] [--reps 10]
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=str(ROOT / "results" / "confirm_v1"))
    ap.add_argument("--reps", type=int, default=10)
    a = ap.parse_args()
    base = Path(a.base) if Path(a.base).is_absolute() else ROOT / a.base
    new = base / f"summary_r{a.reps}.json"
    if not new.exists():
        raise SystemExit(f"{new} missing: PF1 has not run")
    S = json.loads(new.read_text(encoding="utf-8"))
    if int(S.get("reps", 0)) != a.reps:
        raise SystemExit(f"{new}: reps = {S.get('reps')} != {a.reps}")
    if "bootstrap" not in S:
        raise SystemExit(f"{new} has no 'bootstrap' record (older PE1 version): rerun PF1")
    cur = base / "summary.json"
    if cur.exists():
        C = json.loads(cur.read_text(encoding="utf-8"))
        r = int(C.get("reps", 0))
        if r != a.reps:
            keep = base / f"summary_r{r}.json"
            if not keep.exists():
                shutil.copy2(cur, keep)
                print(f"keeping the {r}-split version: {keep.name}")
            dec = base / "decisions.json"
            keep_d = base / f"decisions_r{r}.json"
            if dec.exists() and not keep_d.exists():
                shutil.copy2(dec, keep_d)
                print(f"keeping the {r}-split version: {keep_d.name}")
        elif C.get("bootstrap") == S.get("bootstrap") and C.get("E2E3") == S.get("E2E3"):
            print("summary.json is already this version; nothing to do")
            return
    shutil.copy2(new, cur)
    print(f"{cur.name} <- {new.name} (reps = {a.reps}, B = {S['bootstrap']['B']}, B_AUC = {S['bootstrap']['B_AUC']})")


if __name__ == "__main__":
    main()
