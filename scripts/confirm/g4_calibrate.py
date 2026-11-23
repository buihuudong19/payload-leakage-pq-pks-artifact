"""Gate G4 (paper, Section 7.4): reproduce the published P_E of HILL and S-UNIWARD with SRM + FLD ensemble, and measure the
variation of P_E between training runs (>= 3 seeds). Runs on a machine with all 10 000 BOSSbase images.

Embedding: the optimal embedding simulator (payload-limited sender) of the conseal library, as in the published papers.
Features: SRM, 34 671 dimensions (sealwatch). Classifier: FLD ensemble (sealwatch, automatic L and d_sub).
Evaluation: each seed draws a random 5000/5000 split (the cover and the stego of an image are always on the same side),
P_E = 1/2 (FPR + FNR) of the majority decision on the test part. Compared with configs/g4_reference.json: pass if
|mean P_E - published| <= 0.02.

Step 1 (features, resumable after an interruption):
    python scripts/confirm/g4_calibrate.py features --workers 8
Step 2 (training and verdict):
    python scripts/confirm/g4_calibrate.py evaluate --seeds 3
Output: results/g4/g4_summary.json. Disk: about 1.4 GB per feature set (cover, each method x alpha pair).
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from common import ROOT, SRM_DIM, Ensemble, one_thread_per_worker, srm_vec, workers_default  # noqa: E402
from pqpks import data  # noqa: E402

OUT = ROOT / "results" / "g4"
REF = ROOT / "configs" / "g4_reference.json"
CHUNK = 250


def stego(method: str, x: np.ndarray, alpha: float, seed: int) -> np.ndarray:
    """Reference simulator of conseal, unchanged. Warnings of the lambda search (if any) are printed to stdout with the image id;
    g4_audit_simulator.py measures the payload deviation of those images."""
    import warnings
    import conseal as cl
    sim = {"hill": cl.hill.simulate_single_channel, "suniward": cl.suniward.simulate_single_channel}[method]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        y = sim(x, alpha, seed=seed)
    for w in caught:
        print(f"[conseal] {method} alpha={alpha:g} image {seed}: {w.message}", flush=True)
    return y


def job(args):
    chunk_ids, tasks, k = args
    out = {}
    for name, method, alpha in tasks:
        p = OUT / "feat" / f"{name}_{k:03d}.npy"
        if p.exists():
            continue
        F = np.empty((len(chunk_ids), SRM_DIM), np.float32)
        for r, i in enumerate(chunk_ids):
            x = data.load_bossbase(i)
            F[r] = srm_vec(x if method is None else stego(method, x, alpha, seed=i))
        np.save(p, F)
        out[name] = len(chunk_ids)
    return k, out


def task_list(methods, alphas):
    t = [("cover", None, None)]
    for m in methods:
        for a in alphas:
            t.append((f"{m}_{a:g}", m, a))
    return t


def features(a):
    ids = list(range(1, a.n + 1))
    if a.smoke:                                     # code test only: the first n images present in the BOSSbase directory
        ids = sorted(int(p.stem) for p in data.bossbase_dir().glob("*.pgm"))[: a.n]
    (OUT / "feat").mkdir(parents=True, exist_ok=True)
    (OUT / "ids.json").write_text(json.dumps(ids))
    tasks = task_list(a.methods, a.alphas)
    chunks = [(ids[k * CHUNK:(k + 1) * CHUNK], tasks, k) for k in range((len(ids) + CHUNK - 1) // CHUNK)]
    t0 = time.time()
    one_thread_per_worker()
    with ProcessPoolExecutor(a.workers) as ex:
        for n, (k, done) in enumerate(ex.map(job, chunks), 1):
            print(f"chunk {k} {done} — {n}/{len(chunks)} after {time.time() - t0:.0f}s", flush=True)


def load(name, nchunks):
    return np.vstack([np.load(OUT / "feat" / f"{name}_{k:03d}.npy") for k in range(nchunks)])


def evaluate(a):
    ids = json.loads((OUT / "ids.json").read_text())
    nch = (len(ids) + CHUNK - 1) // CHUNK
    ref = json.loads(REF.read_text(encoding="utf-8"))
    C = load("cover", nch)
    res = {"n_images": len(ids), "seeds": a.seeds, "results": [], "tolerance": ref["tolerance"],
           "reference_primary": ref.get("primary"), "reference_file_sha256": __import__("hashlib").sha256(REF.read_bytes()).hexdigest()}
    (OUT / "scores").mkdir(exist_ok=True)
    audit_p = OUT / "simulator_audit.json"          # g4_audit_simulator.py: images on which the lambda search of conseal did not converge
    audit = json.loads(audit_p.read_text()) if audit_p.exists() else None
    res["simulator_audit"] = None if audit is None else audit["summary"]
    for m in a.methods:
        for al in a.alphas:
            S = load(f"{m}_{al:g}", nch)
            pes, pemins, pes_ex = [], [], []
            flagged = set() if audit is None else {r["id"] for r in audit["flagged"]
                                                   if r["method"] == m and abs(r["alpha"] - al) < 1e-9}
            for seed in range(a.seeds):
                rng = np.random.default_rng(1000 + seed)
                perm = rng.permutation(len(ids))
                tr, te = perm[: len(ids) // 2], perm[len(ids) // 2:]
                ens = Ensemble(seed=seed).fit(C[tr], S[tr])
                sc, ss = ens.score(C[te]), ens.score(S[te])
                pes.append(0.5 * ((sc > 0.5).mean() + (ss <= 0.5).mean()))
                keep = np.array([ids[j] not in flagged for j in te])     # sensitivity: flagged images removed from the test part
                pes_ex.append(0.5 * ((sc[keep] > 0.5).mean() + (ss[keep] <= 0.5).mean()))
                np.savez_compressed(OUT / "scores" / f"{m}_{al:g}_seed{seed}.npz", test_ids=np.asarray(ids)[te], cover=sc, stego=ss)
                from pqpks.analysis import stats
                pemins.append(stats.pe_min(sc, ss))
                print(m, al, "seed", seed, "P_E", round(pes[-1], 4), flush=True)
            def match(rows):
                return next((x for x in rows if x["method"] == m and abs(x["alpha"] - al) < 1e-9
                             and x.get("features", "SRM") == "SRM"), None)
            r = match(ref["references"])
            old = match(ref.get("superseded", []))
            row = {"method": m, "alpha": al, "P_E_mean": float(np.mean(pes)), "P_E_sd": float(np.std(pes, ddof=1)) if len(pes) > 1 else None,
                   "P_E_per_seed": [float(v) for v in pes], "P_E_min_per_seed": [float(v) for v in pemins],
                   "reference": r, "pass": None if r is None else bool(abs(np.mean(pes) - r["P_E"]) <= ref["tolerance"]),
                   "superseded_reference": old,
                   "diff_vs_superseded": None if old is None else float(np.mean(pes) - old["P_E"]),
                   "n_flagged_by_audit": None if audit is None else len(flagged),
                   "P_E_mean_excluding_flagged_test_images": None if audit is None else float(np.mean(pes_ex))}
            res["results"].append(row)
    res["G4_pass"] = all(r["pass"] for r in res["results"] if r["pass"] is not None) and any(r["pass"] is not None for r in res["results"])
    res["seed_sd_max"] = max((r["P_E_sd"] or 0) for r in res["results"])
    res["delta_check"] = "δ = 0.02 is adequate" if res["seed_sd_max"] <= 0.01 else "seed s.d. > 0.01: revisit δ before E3 (plan §5.4)"
    (OUT / "g4_summary.json").write_text(json.dumps(res, indent=1))
    (OUT / f"g4_summary_{time.strftime('%Y%m%d-%H%M%S')}.json").write_text(json.dumps(res, indent=1))   # every run is kept
    print(json.dumps({k: v for k, v in res.items() if k != "results"}, indent=1))
    for r in res["results"]:
        print(r["method"], r["alpha"], "mean", round(r["P_E_mean"], 4), "sd", r["P_E_sd"], "ref", r["reference"] and r["reference"]["P_E"], "pass", r["pass"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["features", "evaluate"])
    ap.add_argument("--methods", nargs="+", default=["hill", "suniward"])
    ap.add_argument("--alphas", nargs="+", type=float, default=[0.3, 0.1])     # 0.1: the region of alpha*
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--workers", type=int, default=workers_default())
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--smoke", action="store_true", help="code test only, on a few available images; delete results/g4 afterwards")
    a = ap.parse_args()
    features(a) if a.step == "features" else evaluate(a)


if __name__ == "__main__":
    main()
