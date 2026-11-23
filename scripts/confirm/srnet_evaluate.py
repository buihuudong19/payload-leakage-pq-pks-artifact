"""SRNet G4 verdict and confirmatory SRNet analysis. Runs in .venv-confirm (needs scipy).

Reads results/srnet/<run>/scores.npz written by scripts/gpu/srnet_train.py.

  g4       SRNet G4: HILL 0.4 bpp only (256x256 images), compared with [BCF19] Table I (0.1414,
           configs/g4_srnet_reference.json). P_E as defined in the original paper: minimum over thresholds of 1/2 (P_FA + P_MD) on test.
           Registered order (configs/srnet_runs_v2.json, g4_protocol): the short branch g4_hill_0.4_short (200k + 50k iterations)
           first; it passes and is selected if it deviates by at most 0.02. Otherwise g4_hill_0.4 is run to the end of the 500k
           schedule and decides. Output: results/srnet/g4_srnet_summary.json (G4_SRNet_pass, selected_run, next).
  confirm  PQ-PKS and REF-U at alpha = 0.4 and 0.2 (512x512 images, one split). As in the confirmatory SRM analysis: the P_E
           threshold and the 1 %, 5 % FPR thresholds are chosen on validation and applied to test; 95 % bootstrap CI over test
           images (B = 2000). PQ-PKS against REF-U: paired P_E difference (same covers), TOST delta = 0.02 on the 90 % CI, exact McNemar
           test on the correct/incorrect decisions of the two SRNets. P_E(SRNet, REF-U) at 0.2 and 0.4 is recorded for condition (i)
           of alpha*; apply_rules.py combines it with SRM (W-img = the strongest warden). Output: results/srnet/confirm_srnet_summary.json.

    python scripts/confirm/srnet_evaluate.py g4
    python scripts/confirm/srnet_evaluate.py confirm
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
from scipy import stats as st

from common import ROOT  # noqa: E402
from pqpks.analysis import stats  # noqa: E402

CFG = json.loads((ROOT / "configs" / "srnet_runs_v2.json").read_text(encoding="utf-8"))
REF = json.loads((ROOT / "configs" / "g4_srnet_reference.json").read_text(encoding="utf-8"))
OUT = ROOT / "results" / "srnet"
B_BOOT, DELTA, COND_I = 2000, 0.02, 0.45


def load(run: str, root=None):
    d = (root or OUT) / run
    if not (d / "done.json").exists():
        return None, None
    z = np.load(d / "scores.npz")
    return {k: z[k] for k in z.files}, json.loads((d / "done.json").read_text(encoding="utf-8"))


def thresholds(z) -> dict:
    return {"pe": stats.threshold_min_pe(z["val_cover"], z["val_stego"]),
            "fpr5": stats.threshold_at_fpr(z["val_cover"], 0.05), "fpr1": stats.threshold_at_fpr(z["val_cover"], 0.01)}


def metrics(z, seed: int = 7) -> dict:
    tc, ts = z["test_cover"], z["test_stego"]
    th = thresholds(z)
    n = tc.size
    fns = {"P_E_min_test": lambda ii: stats.pe_min(tc[ii], ts[ii]),
           "P_E": lambda ii: stats.pe_at(tc[ii], ts[ii], th["pe"]),
           "TPR@5": lambda ii: float((ts[ii] > th["fpr5"]).mean()),
           "TPR@1": lambda ii: float((ts[ii] > th["fpr1"]).mean()),
           "AUC": lambda ii: stats.auc(tc[ii], ts[ii])}
    allidx = np.arange(n)
    out = {}
    for k, f in fns.items():
        out[k] = f(allidx)
        out[k + "_ci95"] = stats.ci(stats.cluster_bootstrap(f, n, B=B_BOOT, seed=seed))
    out["accuracy_at_0"] = 1 - stats.pe_at(tc, ts, 0.0)
    out["realized_FPR@5_test"] = float((tc > th["fpr5"]).mean())
    out["val_P_E_min"] = stats.pe_min(z["val_cover"], z["val_stego"])
    out["thresholds_from_val"] = th
    out["n_test_pairs"] = int(n)
    return out


def g4_row(run: str, ref: float, tol: float) -> dict | None:
    z, info = load(run)
    if z is None:
        return None
    row = metrics(z)
    row |= {"reference_P_E": ref, "diff": row["P_E_min_test"] - ref,
            "pass": bool(abs(row["P_E_min_test"] - ref) <= tol) and not info["smoke"],
            "best_iter": info["best_iter"], "best_val_acc": info["best_val_acc"], "iterations": info["iterations"],
            "smoke": info["smoke"], "device": info["device"]}
    return row


def g4_decide(rows: dict, order: list[str]) -> dict:
    """Registered rule applied to the available rows: the first run in the registered order that has a result decides; a later
    run is considered only if the earlier one failed. Returns G4_SRNet_pass, selected_run, next ('fallback_full', 'wait' or None)."""
    short, full = order
    if rows.get(short) is None:
        return {"G4_SRNet_pass": False, "selected_run": None, "next": "wait"}
    if rows[short]["pass"]:
        return {"G4_SRNet_pass": True, "selected_run": short, "next": None}
    if rows.get(full) is None:
        return {"G4_SRNet_pass": False, "selected_run": None, "next": "fallback_full"}
    ok = rows[full]["pass"]
    return {"G4_SRNet_pass": ok, "selected_run": full if ok else None, "next": None}


def g4(a):
    prot = CFG["g4_protocol"]
    ref = next(r["P_E"] for r in REF["references"] if r["method"] == "hill" and abs(r["alpha"] - 0.4) < 1e-9)
    res = {"reference": REF["source"], "tolerance": REF["tolerance"], "scope": prot["scope"], "rule": prot["rule"],
           "runs": {}}
    rows = {}
    for run in prot["order"]:
        rows[run] = g4_row(run, ref, REF["tolerance"])
        res["runs"][run] = rows[run] or {"status": "not finished"}
        if rows[run]:
            r = rows[run]
            print(f"{run}: P_E (min, test) {r['P_E_min_test']:.4f} [{r['P_E_min_test_ci95'][0]:.4f}; "
                  f"{r['P_E_min_test_ci95'][1]:.4f}], published {ref:.4f}, deviation {r['diff']:+.4f}, pass {r['pass']}", flush=True)
    res |= g4_decide(rows, prot["order"])
    OUT.mkdir(parents=True, exist_ok=True)
    js = json.dumps(res, indent=1, ensure_ascii=False, default=float)
    (OUT / "g4_srnet_summary.json").write_text(js, encoding="utf-8")
    (OUT / f"g4_srnet_summary_{time.strftime('%Y%m%d-%H%M%S')}.json").write_text(js, encoding="utf-8")
    print("G4_SRNet_pass:", res["G4_SRNet_pass"], "| selected:", res["selected_run"], "| next:", res["next"], flush=True)


def paired(zp, zr) -> dict:
    """PQ-PKS against REF-U on the same test images: P_E difference (thresholds from validation), TOST, McNemar."""
    if not np.array_equal(zp["test_ids"], zr["test_ids"]):
        raise SystemExit("the two runs do not use the same test images: check pair_with in configs/srnet_runs_v2.json")
    tp, tr = thresholds(zp)["pe"], thresholds(zr)["pe"]
    n = zp["test_cover"].size

    def d_pe(ii):
        return (stats.pe_at(zp["test_cover"][ii], zp["test_stego"][ii], tp)
                - stats.pe_at(zr["test_cover"][ii], zr["test_stego"][ii], tr))

    def d_pemin(ii):
        return (stats.pe_min(zp["test_cover"][ii], zp["test_stego"][ii])
                - stats.pe_min(zr["test_cover"][ii], zr["test_stego"][ii]))

    boot = stats.cluster_bootstrap(d_pe, n, B=B_BOOT, seed=31)
    tost = stats.tost_from_boot(d_pe(np.arange(n)), boot, DELTA).as_dict()
    ok_p = np.r_[zp["test_cover"] <= tp, zp["test_stego"] > tp]
    ok_r = np.r_[zr["test_cover"] <= tr, zr["test_stego"] > tr]
    b, c = int((ok_p & ~ok_r).sum()), int((~ok_p & ok_r).sum())
    p = float(st.binomtest(b, b + c, 0.5).pvalue) if b + c > 0 else 1.0
    dm = d_pemin(np.arange(n))
    return {"diff_P_E": tost["diff"], "TOST": tost,
            "diff_P_E_min_test": dm, "diff_P_E_min_test_ci95": stats.ci(stats.cluster_bootstrap(d_pemin, n, B=B_BOOT, seed=32)),
            "McNemar": {"PQ_right_REF_wrong": b, "REF_right_PQ_wrong": c, "p_exact": p}}


def confirm(a):
    res = {"design": "SRNet W-img, PQ-PKS and REF-U, alpha = 0.4 and 0.2, one split of split_v1", "alphas": {}}
    summ_p = ROOT / "results" / "confirm_v1" / "summary.json"
    srm = json.loads(summ_p.read_text(encoding="utf-8")) if summ_p.exists() else None
    for al in (0.4, 0.2):
        zp, ip = load(f"confirm_PQ-PKS_{al}")
        zr, ir = load(f"confirm_REF-U_{al}")
        if zp is None or zr is None:
            res["alphas"][str(al)] = {"status": "not finished"}
            continue
        mp, mr = metrics(zp), metrics(zr)
        row = {"PQ-PKS": mp | {"best_iter": ip["best_iter"]}, "REF-U": mr | {"best_iter": ir["best_iter"]},
               "PQ-PKS_vs_REF-U": paired(zp, zr),
               "cond_i_SRNet": {"P_E_REF-U": mr["P_E"], "holds": mr["P_E"] >= COND_I}}
        if srm is not None:
            pe_srm = srm["alpha_star"]["P_E_W-img_REF-U"].get(str(al))
            row["cond_i_SRNet"]["P_E_REF-U_SRM"] = pe_srm
            row["cond_i_SRNet"]["stronger_on_REF-U"] = (None if pe_srm is None else
                                                        ("SRNet" if mr["P_E"] < pe_srm else "SRM"))
        res["alphas"][str(al)] = row
        t = row["PQ-PKS_vs_REF-U"]["TOST"]
        print(f"alpha = {al}: P_E PQ-PKS {mp['P_E']:.4f}, REF-U {mr['P_E']:.4f}, difference {t['diff']:+.4f} "
              f"CI90 [{t['ci90'][0]:+.4f}; {t['ci90'][1]:+.4f}] → {t['verdict']}", flush=True)
    res["alpha_star_SRM"] = None if srm is None else srm["alpha_star"].get("alpha_star")
    OUT.mkdir(parents=True, exist_ok=True)
    js = json.dumps(res, indent=1, ensure_ascii=False, default=float)
    (OUT / "confirm_srnet_summary.json").write_text(js, encoding="utf-8")
    (OUT / f"confirm_srnet_summary_{time.strftime('%Y%m%d-%H%M%S')}.json").write_text(js, encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["g4", "confirm"])
    ap.add_argument("--smoke", action="store_true", help="code test only: reads and writes results/srnet_smoke/")
    a = ap.parse_args()
    if a.smoke:
        global OUT
        OUT = ROOT / "results" / "srnet_smoke"
    {"g4": g4, "confirm": confirm}[a.step](a)


if __name__ == "__main__":
    main()
