"""Gate G4 for SRM on fresh seeds (registered after review, 24 Nov 2026; registrations/G4_NEW_SEEDS_PLAN.md).

The SRM reference values of G4 were replaced on 6 Nov 2026 after two seeds of HILL at 0.3 bpp had been seen (Table S6, D1 of
the supplementary material). This script repeats the G4 evaluation with seeds that have never been used, under a rule fixed
before it is run, and compares the result with both the adopted and the replaced references. It changes no registered decision
and does not overwrite results/g4/.

Procedure (identical to g4_calibrate.py evaluate, except the seeds): the SRM features already extracted by g4_calibrate.py
features (results/g4/feat, 10 000 BOSSbase images; stego images from the conseal simulator with seed = image index) are read;
for each seed s in the plan a random 5000/5000 split (numpy default_rng(1000 + s)) and an FLD ensemble (Ensemble(seed=s)) are
drawn; P_E = 1/2 (FPR + FNR) on the test half.

Registered rule (plan): seeds 100-109; a cell passes when |mean P_E over the ten seeds - adopted reference| <= tolerance of
configs/g4_reference.json (0.02); the fresh-seed G4 passes when all four cells pass. Reported as well: per-seed P_E, the
difference from the replaced references, the seed s.d., and the comparison with the three original seeds.

    python scripts/confirm/g4_new_seeds.py --plan registrations/g4_new_seeds_plan.json
    python scripts/confirm/g4_new_seeds.py --plan ... --seeds 100 101 --smoke     # code test on few seeds; writes to a temp dir
Output: results/g4_new_seeds/g4_new_seeds_summary.json (+ per-seed scores).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from common import ROOT, Ensemble  # noqa: E402

G4 = ROOT / "results" / "g4"
REF = ROOT / "configs" / "g4_reference.json"
OUT = ROOT / "results" / "g4_new_seeds"
CHUNK = 250


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load(feat: Path, name: str, nchunks: int) -> np.ndarray:
    return np.vstack([np.load(feat / f"{name}_{k:03d}.npy") for k in range(nchunks)])


def pe(sc: np.ndarray, ss: np.ndarray) -> float:
    """P_E = 1/2 (FPR + FNR) of the ensemble majority decision (score > 0.5 means stego), as in g4_calibrate.py."""
    return float(0.5 * ((sc > 0.5).mean() + (ss <= 0.5).mean()))


def match(rows, m, al):
    return next((x for x in rows if x["method"] == m and abs(x["alpha"] - al) < 1e-9 and x.get("features", "SRM") == "SRM"), None)


def cell_verdict(pes: list[float], ref: dict | None, tol: float) -> bool | None:
    return None if ref is None else bool(abs(float(np.mean(pes)) - ref["P_E"]) <= tol)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", default=str(ROOT / "registrations" / "g4_new_seeds_plan.json"))
    ap.add_argument("--seeds", type=int, nargs="+", help="override the plan (code tests only)")
    ap.add_argument("--g4", default=str(G4), help="folder with ids.json and feat/ of g4_calibrate.py")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--smoke", action="store_true", help="code test: results are written but flagged as a smoke run")
    a = ap.parse_args()
    plan_p = Path(a.plan)
    plan = json.loads(plan_p.read_text(encoding="utf-8"))
    seeds = a.seeds or plan["seeds"]
    used = set(plan.get("seeds_already_used", [0, 1, 2]))
    if used & set(seeds) and not a.smoke:
        raise SystemExit(f"seeds {sorted(used & set(seeds))} were used by the original G4; the plan requires fresh seeds")
    g4, out = Path(a.g4), Path(a.out)
    ids = json.loads((g4 / "ids.json").read_text())
    nch = (len(ids) + CHUNK - 1) // CHUNK
    ref = json.loads(REF.read_text(encoding="utf-8"))
    tol = float(plan.get("tolerance", ref["tolerance"]))
    orig = json.loads((g4 / "g4_summary.json").read_text()) if (g4 / "g4_summary.json").exists() else None
    out.mkdir(parents=True, exist_ok=True)
    (out / "scores").mkdir(exist_ok=True)
    res = {"label": "registered after review (24 Nov 2026); the registered G4 decision of 7 Nov 2026 is unchanged",
           "smoke": bool(a.smoke), "plan_file": plan_p.name, "plan_sha256": sha(plan_p), "reference_file_sha256": sha(REF),
           "n_images": len(ids), "seeds": seeds, "tolerance": tol, "rule": plan["rule"], "results": []}
    t0 = time.time()
    C = load(g4 / "feat", "cover", nch)
    for m in plan["methods"]:
        for al in plan["alphas"]:
            S = load(g4 / "feat", f"{m}_{al:g}", nch)
            pes = []
            for s in seeds:
                rng = np.random.default_rng(1000 + s)
                perm = rng.permutation(len(ids))
                tr, te = perm[: len(ids) // 2], perm[len(ids) // 2:]
                ens = Ensemble(seed=s).fit(C[tr], S[tr])
                sc, ss = ens.score(C[te]), ens.score(S[te])
                pes.append(pe(sc, ss))
                np.savez_compressed(out / "scores" / f"{m}_{al:g}_seed{s}.npz", test_ids=np.asarray(ids)[te], cover=sc, stego=ss)
                print(f"[{time.time() - t0:6.0f}s] {m} {al:g} seed {s}: P_E {pes[-1]:.4f}", flush=True)
            r, old = match(ref["references"], m, al), match(ref.get("superseded", []), m, al)
            o = None if orig is None else next((x for x in orig["results"] if x["method"] == m and abs(x["alpha"] - al) < 1e-9), None)
            res["results"].append({
                "method": m, "alpha": al, "P_E_per_seed": pes, "P_E_mean": float(np.mean(pes)),
                "P_E_sd": float(np.std(pes, ddof=1)) if len(pes) > 1 else None,
                "reference": r, "diff_vs_reference": None if r is None else float(np.mean(pes) - r["P_E"]),
                "pass": cell_verdict(pes, r, tol),
                "superseded_reference": old, "diff_vs_superseded": None if old is None else float(np.mean(pes) - old["P_E"]),
                "pass_vs_superseded": cell_verdict(pes, old, tol),
                "original_seeds_P_E_mean": None if o is None else o["P_E_mean"]})
    rows = res["results"]
    res["G4_fresh_seeds_pass"] = all(x["pass"] for x in rows if x["pass"] is not None) and any(x["pass"] is not None for x in rows)
    res["mean_offset_vs_reference"] = float(np.mean([x["diff_vs_reference"] for x in rows if x["diff_vs_reference"] is not None]))
    res["seed_sd_max"] = max((x["P_E_sd"] or 0) for x in rows)
    res["seconds"] = round(time.time() - t0)
    (out / "g4_new_seeds_summary.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    for x in rows:
        print(f"{x['method']} {x['alpha']:g}: mean {x['P_E_mean']:.4f} (sd {x['P_E_sd']}), adopted ref "
              f"{x['reference'] and x['reference']['P_E']} -> {x['diff_vs_reference']:+.4f} pass {x['pass']}; replaced ref "
              f"{x['superseded_reference'] and x['superseded_reference']['P_E']} -> {x['diff_vs_superseded']}")
    print("G4 on fresh seeds:", res["G4_fresh_seeds_pass"], "| mean offset", round(res["mean_offset_vs_reference"], 4))


if __name__ == "__main__":
    main()
