"""Audit of the G4 embedding simulator: images on which the lambda search of conseal stops before reaching the payload tolerance.

The optimal embedding simulator (payload-limited sender) of conseal searches lambda by doubling and then bisection, at most 30
rounds, and stops when the actual payload is within 1/1000 of the requested payload. If the 30 rounds run out first, conseal only
warns "optimization might not have converged" and embeds with the last lambda, so the actual payload of that image may deviate
from alpha n. This script repeats exactly the lambda computation of conseal (same cost function, same default parameters, no
sampling, no SRM) for every G4 image, and records which images are flagged and by how much their actual payload deviates.

No G4 feature is changed: G4 still uses the reference simulator. `g4_calibrate.py evaluate` reads this output (if present)
to also report P_E with the flagged images removed from the test part (sensitivity analysis).

    python scripts/confirm/g4_audit_simulator.py --workers 7
Output: results/g4/simulator_audit.json
"""
from __future__ import annotations

import argparse
import inspect
import json
import time
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from common import ROOT, one_thread_per_worker, workers_default  # noqa: E402
from pqpks import data  # noqa: E402

OUT = ROOT / "results" / "g4"
TOL = 1e-3                     # relative tolerance of conseal: |actual payload / requested payload - 1| <= 1/1000


def _rhos(method: str, x: np.ndarray):
    import conseal as cl
    from conseal.hill import _costmap as hill_cost
    from conseal.suniward import _costmap as suniward_cost
    if method == "hill":
        d = inspect.signature(cl.hill.simulate_single_channel).parameters
        return hill_cost.compute_cost_adjusted(x0=x, wet_cost=d["wet_cost"].default)
    if method == "suniward":
        d = inspect.signature(cl.suniward.simulate_single_channel).parameters
        return suniward_cost.compute_cost_adjusted(x0=x, sigma=d["sigma"].default, wet_cost=d["wet_cost"].default)
    raise KeyError(method)


def job(args):
    ids, methods, alphas = args
    from conseal.simulate import _optim, _ternary
    objective = _optim.get_objective(q=3)
    rows = []
    for i in ids:
        x = data.load_bossbase(i)
        n = x.size
        for m in methods:
            rhos = _rhos(m, x)
            for a in alphas:
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    _, lbda = _ternary.probability(rhos=rhos, alpha=a, n=n)
                _, H = objective(lbda=lbda, rhos=rhos)
                rel = float(H) / round(a * n) - 1.0
                warned = any("converged" in str(v.message) or "unbounded" in str(v.message) for v in w)
                rows.append({"id": int(i), "method": m, "alpha": a, "warned": warned, "rel_payload_err": rel,
                             "flagged": warned or abs(rel) > TOL})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", default=["hill", "suniward"])
    ap.add_argument("--alphas", nargs="+", type=float, default=[0.3, 0.1])
    ap.add_argument("--workers", type=int, default=workers_default())
    a = ap.parse_args()
    ids = json.loads((OUT / "ids.json").read_text())
    chunks = [ids[k:k + 100] for k in range(0, len(ids), 100)]
    t0 = time.time()
    one_thread_per_worker()
    rows = []
    with ProcessPoolExecutor(a.workers) as ex:
        for n, r in enumerate(ex.map(job, [(c, a.methods, a.alphas) for c in chunks]), 1):
            rows += r
            if n % 10 == 0:
                print(f"{n}/{len(chunks)} after {time.time() - t0:.0f}s", flush=True)
    out = {"tolerance": TOL, "n_images": len(ids), "summary": {}, "flagged": [r for r in rows if r["flagged"]]}
    for m in a.methods:
        for al in a.alphas:
            rr = [r for r in rows if r["method"] == m and r["alpha"] == al]
            e = np.array([r["rel_payload_err"] for r in rr])
            out["summary"][f"{m}_{al:g}"] = {
                "n": len(rr), "n_warned": sum(r["warned"] for r in rr), "n_flagged": sum(r["flagged"] for r in rr),
                "max_abs_rel_payload_err": float(np.abs(e).max()), "mean_abs_rel_payload_err": float(np.abs(e).mean()),
                "worst": sorted(({"id": r["id"], "rel_payload_err": r["rel_payload_err"]} for r in rr),
                                key=lambda v: -abs(v["rel_payload_err"]))[:10]}
    (OUT / "simulator_audit.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out["summary"], indent=1))


if __name__ == "__main__":
    main()
