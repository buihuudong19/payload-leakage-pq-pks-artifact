"""Supplementary pilot (design review, 30/10/2026). Runs on a CPU with the same 500 BOSSbase images as the first pilot.
NOT a substitute for E2/E3/E6: the image warden is still SPAM and the sample is small.

Four parts:
  wa0      : warden that knows the public A0 (W-A0) against the image-only warden (W-img), on the SAME stego images. W-A0 uses
             SPAM computed separately on A0 and on the rest of the image (1372 dimensions). Systems: PQ-PKS (kem_part 1536 B,
             A0 = 81 920 pixels), BL4 (72 B, A0 = 3 840 pixels), REF-U (same layout as PQ-PKS), REF-1S (single stage; the A0 mask
             of PQ-PKS as control). alpha in {0.2, 0.4}, regime R-b.
             Question: does a warden that knows A0 have an advantage over an image-only warden?
  power    : from the stored features of the first pilot: 95 % CI of the AUC; 10 repeated cross-validations (standard deviation
             due to the fold split and the subspace choice); power of the TOST PQ-PKS <-> REF-U at delta = 0.02 and the number
             of image pairs needed for power 0.8.
  dos      : receiver cost for invalid images with S_max = 16 open sessions, W = 8, |B| = 6 (default configuration): number of
             AEAD openings and time per image; compared with the bound |B|(1 + (2W - 1)S_max) = 1446 (Prop. 2). Also measures a
             valid session-start image and a replayed session-start image.
  a8       : layout A8: alpha_0 matched to the density, i.e. alpha_0 = alpha in the single-bucket profile, for PQ-PKS and REF-U,
             alpha in {0.2, 0.4}; measures W-img and W-A0 (A0 mask from the new alpha_0), compared with the earlier fixed layout
             alpha_0 = 0.15 of part wa0 (same covers).
  kemeleon : the analytic constants of Lemmas 2M / 2R (statistical distance of VectorEncode, p_acc).

Run:  PQPKS_BOSSBASE_DIR=<.pgm directory> python scripts/pilot_review_v3.py --phase all --jobs 2
       [--pilot-cache <.npz directory of the first pilot>] [--cache <directory for the wa0 features>]
Output: results/archive/pre-confirm/pilot_review_v3/report.json (REPORT.md written by hand, numbers quoted from report.json).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import secrets
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pqpks import data, systems  # noqa: E402
from pqpks.analysis import ensemble, spam  # noqa: E402
from pqpks.crypto import kemeleon  # noqa: E402
from pqpks.experiment.profiles import message_len_for_bucket, profile_for_alpha  # noqa: E402
from pqpks.experiment.references import ReferenceSystem  # noqa: E402
from pqpks.params import AEAD_TAG_BYTES, DEFAULT_PP, KEMELEON_CT_BYTES  # noqa: E402
from pqpks.stego.two_stage import TwoStageSTC  # noqa: E402

OUT = ROOT / "results" / "archive" / "pre-confirm" / "pilot_review_v3"
N_PIX = 512 * 512
ALPHAS = (0.2, 0.4)
STC = ("PQ-PKS", "BL4")
REFS = ("REF-U", "REF-1S")
SYSTEMS = STC + REFS
_STATE: dict = {}


def _mask(kem_bytes: int, shape=(512, 512)) -> np.ndarray:
    a0, _ = TwoStageSTC(DEFAULT_PP)._sets(shape, 8 * kem_bytes)
    m = np.zeros(shape[0] * shape[1], bool)
    m[a0] = True
    return m.reshape(shape)


def _setup():
    st = {"stc": {}, "keys": {}}
    for name in STC:
        kb = systems.crypto_layer(name).kem_part_bytes
        for a in ALPHAS:
            st["stc"][(name, a)] = systems.make(name, profile_for_alpha(a, N_PIX, kb))
        st["keys"][name] = st["stc"][(name, ALPHAS[0])].keygen()
    st["ref"] = {r: ReferenceSystem(r, KEMELEON_CT_BYTES) for r in REFS}
    st["emb"] = TwoStageSTC(DEFAULT_PP)
    st["mask"] = {"PQ-PKS": _mask(KEMELEON_CT_BYTES), "BL4": _mask(systems.crypto_layer("BL4").kem_part_bytes)}
    st["mask"]["REF-U"] = st["mask"]["REF-1S"] = st["mask"]["PQ-PKS"]
    return st


def _work(image_id: int):
    st = _STATE
    x = data.load_bossbase(image_id)
    cm = st["emb"].cost_map(x)
    feat = {"cover:img": spam.spam686(x)}
    for key in ("PQ-PKS", "BL4"):
        feat[f"cover:a0:{key}"] = spam.spam_a0(x, st["mask"][key])
    for a in ALPHAS:
        for name in STC:
            sysm = st["stc"][(name, a)]
            _, ek = st["keys"][name]
            msg = secrets.token_bytes(message_len_for_bucket(sysm.pp.buckets[0], start=True))
            y = sysm.send(sysm.start_session(ek, length=1), msg, x, cost_map=cm)
            feat[f"{name}:{a}:img"] = spam.spam686(y)
            feat[f"{name}:{a}:a0"] = spam.spam_a0(y, st["mask"][name])
        body = profile_for_alpha(a, N_PIX, KEMELEON_CT_BYTES).buckets[0] + AEAD_TAG_BYTES
        for r in REFS:
            y = st["ref"][r].stego(x, body, start=True, cost_map=cm)
            feat[f"{r}:{a}:img"] = spam.spam686(y)
            feat[f"{r}:{a}:a0"] = spam.spam_a0(y, st["mask"][r])
    return image_id, feat


def _r(v, nd=4):
    return round(float(v), nd)


def _summ(res: dict) -> dict:
    lo, hi = ensemble.bootstrap_pe(res["dec_cover"], res["dec_stego"])
    alo, ahi = ensemble.bootstrap_auc(res)
    return {"P_E": _r(res["P_E"]), "P_E_ci95": (_r(lo), _r(hi)), "AUC": _r(res["AUC"]), "AUC_ci95": (_r(alo), _r(ahi))}


def run_wa0(ids, cache: Path, jobs: int) -> dict:
    global _STATE
    _STATE.update(_setup())
    cache.mkdir(parents=True, exist_ok=True)
    todo = [i for i in ids if not (cache / f"{i}.npz").exists()]
    t0 = time.time()
    with Pool(jobs) as pool:
        for done, (i, feat) in enumerate(pool.imap_unordered(_work, todo), 1):
            np.savez_compressed(cache / f"{i}.npz", **feat)
            if done % 25 == 0:
                print(f"wa0 {done}/{len(todo)} after {time.time() - t0:.0f} s", flush=True)
    Z = [np.load(cache / f"{i}.npz") for i in ids]
    F = {k: np.array([z[k] for z in Z]) for k in Z[0].files}
    out, raw = {}, {}
    for a in ALPHAS:
        for name in SYSTEMS:
            mk = "PQ-PKS" if name in ("REF-U", "REF-1S") else name
            r_img = ensemble.cross_validate(F["cover:img"], F[f"{name}:{a}:img"])
            r_a0 = ensemble.cross_validate(F[f"cover:a0:{mk}"], F[f"{name}:{a}:a0"])
            raw[(name, a, "img")], raw[(name, a, "a0")] = r_img, r_a0
            d = ensemble.paired_diff_ci(r_a0, r_img)
            out[f"{name}:{a}"] = {"W-img": _summ(r_img), "W-A0": _summ(r_a0),
                                  "W-A0 − W-img": {"diff": _r(d["diff"]), "ci95": tuple(map(_r, d["ci95"])),
                                                   "p_boot": _r(d["p_boot"])}}
        for x_, o in (("PQ-PKS", "REF-U"), ("PQ-PKS", "BL4"), ("REF-U", "REF-1S")):
            d = ensemble.paired_diff_ci(raw[(x_, a, "a0")], raw[(o, a, "a0")])
            out[f"W-A0 diff:{x_}-{o}:{a}"] = {"diff": _r(d["diff"]), "ci95": tuple(map(_r, d["ci95"])),
                                              "ci90": tuple(map(_r, d["ci90"])), "p_boot": _r(d["p_boot"])}
    fam = {k: v["W-A0 − W-img"]["p_boot"] for k, v in out.items() if "W-A0 − W-img" in v}
    fam |= {k: v["p_boot"] for k, v in out.items() if k.startswith("W-A0 diff:")}
    for k, p in ensemble.holm(fam).items():
        (out[k]["W-A0 − W-img"] if "W-A0 − W-img" in out.get(k, {}) else out[k])["p_holm"] = _r(p)
    out["_n"] = len(ids)
    return out


def _setup_a8():
    from dataclasses import replace
    st = {"stc": {}, "ref": {}, "mask": {}}
    kb = KEMELEON_CT_BYTES
    for a in ALPHAS:
        st["stc"][a] = systems.make("PQ-PKS", replace(profile_for_alpha(a, N_PIX, kb), alpha0=a))
        st["ref"][a] = ReferenceSystem("REF-U", kb, pp=replace(DEFAULT_PP, alpha0=a))
        a0, _ = st["ref"][a].embedder._sets((512, 512), 8 * kb)
        m = np.zeros(N_PIX, bool)
        m[a0] = True
        st["mask"][a] = m.reshape(512, 512)
    st["keys"] = st["stc"][ALPHAS[0]].keygen()
    st["emb"] = TwoStageSTC(DEFAULT_PP)
    return st


def _work_a8(image_id: int):
    st = _STATE
    x = data.load_bossbase(image_id)
    cm = st["emb"].cost_map(x)
    feat = {}
    for a in ALPHAS:
        feat[f"cover:a0:{a}"] = spam.spam_a0(x, st["mask"][a])
        sysm = st["stc"][a]
        msg = secrets.token_bytes(message_len_for_bucket(sysm.pp.buckets[0], start=True))
        y = sysm.send(sysm.start_session(st["keys"][1], length=1), msg, x, cost_map=cm)
        feat[f"PQ-PKS-A8:{a}:img"] = spam.spam686(y)
        feat[f"PQ-PKS-A8:{a}:a0"] = spam.spam_a0(y, st["mask"][a])
        body = sysm.pp.buckets[0] + AEAD_TAG_BYTES
        y = st["ref"][a].stego(x, body, start=True, cost_map=cm)
        feat[f"REF-U-A8:{a}:img"] = spam.spam686(y)
        feat[f"REF-U-A8:{a}:a0"] = spam.spam_a0(y, st["mask"][a])
    return image_id, feat


def run_a8(ids, cache: Path, base_cache: Path, jobs: int) -> dict:
    global _STATE
    _STATE.update(_setup_a8())
    cache.mkdir(parents=True, exist_ok=True)
    todo = [i for i in ids if not (cache / f"{i}.npz").exists()]
    t0 = time.time()
    with Pool(jobs) as pool:
        for done, (i, feat) in enumerate(pool.imap_unordered(_work_a8, todo), 1):
            np.savez_compressed(cache / f"{i}.npz", **feat)
            if done % 25 == 0:
                print(f"a8 {done}/{len(todo)} after {time.time() - t0:.0f} s", flush=True)
    Z = [np.load(cache / f"{i}.npz") for i in ids]
    B = [np.load(base_cache / f"{i}.npz") for i in ids]
    F = {k: np.array([z[k] for z in Z]) for k in Z[0].files}
    G = {k: np.array([z[k] for z in B]) for k in B[0].files}
    out = {}
    for a in ALPHAS:
        for name in ("PQ-PKS", "REF-U"):
            r_img = ensemble.cross_validate(G["cover:img"], F[f"{name}-A8:{a}:img"])
            r_a0 = ensemble.cross_validate(F[f"cover:a0:{a}"], F[f"{name}-A8:{a}:a0"])
            b_img = ensemble.cross_validate(G["cover:img"], G[f"{name}:{a}:img"])
            b_a0 = ensemble.cross_validate(G["cover:a0:PQ-PKS"], G[f"{name}:{a}:a0"])
            row = {"A8 W-img": _summ(r_img), "A8 W-A0": _summ(r_a0)}
            for lab, ra, rb in (("A8 − default, W-img", r_img, b_img), ("A8 − default, W-A0", r_a0, b_a0),
                                ("A8: W-A0 − W-img", r_a0, r_img)):
                d = ensemble.paired_diff_ci(ra, rb)
                row[lab] = {"diff": _r(d["diff"]), "ci95": tuple(map(_r, d["ci95"])), "p_boot": _r(d["p_boot"])}
            out[f"{name}:{a}"] = row
    fam = {(k, lab): v[lab]["p_boot"] for k, v in out.items() for lab in v if " − " in lab or "−" in lab}
    flat = {f"{k}|{lab}": p for (k, lab), p in fam.items()}
    for key, p in ensemble.holm(flat).items():
        k, lab = key.split("|")
        out[k][lab]["p_holm"] = _r(p)
    out["_n"] = len(ids)
    out["alpha0"] = {str(a): a for a in ALPHAS}
    return out


def run_power(ids, pilot_cache: Path, R: int) -> dict:
    Z = [np.load(pilot_cache / f"{i}.npz") for i in ids]
    fc = np.array([z["f__cover"] for z in Z])
    out = {"auc_ci": {}, "repeated_cv": {}, "tost": {}}
    names = ("PQ-PKS", "REF-U", "REF-1S", "BL5-L", "BL4", "BL3", "BL5", "BL1", "BL2")
    for a in ALPHAS:
        for name in names:
            fs = np.array([z[f"f__{name}:{a}"] for z in Z])
            out["auc_ci"][f"{name}:{a}"] = _summ(ensemble.cross_validate(fc, fs))
        reps = {}
        for name in ("PQ-PKS", "REF-U"):
            fs = np.array([z[f"f__{name}:{a}"] for z in Z])
            reps[name] = ensemble.repeated_cv(fc, fs, R=R)
            out["repeated_cv"][f"{name}:{a}"] = {k: (_r(v) if isinstance(v, float) else [_r(t) for t in v])
                                                 for k, v in reps[name].items() if k != "runs"}
        # difference PQ-PKS - REF-U per repetition (same seed -> same fold split)
        diffs = np.array([ra["P_E"] - rb["P_E"] for ra, rb in zip(reps["PQ-PKS"]["runs"], reps["REF-U"]["runs"])])
        g_sd = np.mean([ensemble.paired_pe_terms(ra, rb).std(ddof=1)
                        for ra, rb in zip(reps["PQ-PKS"]["runs"], reps["REF-U"]["runs"])])
        n = len(ids)
        sd_train = float(diffs.std(ddof=1))
        need = ensemble.tost_pairs_needed(float(g_sd), 0.02)
        # Three assumptions about the variation due to training (sd_train measured at n = 500), all with a true difference of 0:
        #  (A) it decreases as 1/sqrt(n) like the sampling part (optimistic, unverified);
        #  (B) it does not decrease with n and the statistic is ONE cross-validation: no sample size suffices if sd_train > delta/(z_0.95 + z_0.90);
        #  (C) it does not decrease with n and the statistic is the mean of R cross-validations: the component is sd_train/sqrt(R).
        need_a = ensemble.tost_pairs_needed(float(math.hypot(g_sd, sd_train * math.sqrt(n))), 0.02)
        try:
            need_b = ensemble.tost_pairs_needed(float(g_sd), 0.02, sd_train=sd_train)
        except ValueError:
            need_b = None
        sd_c = sd_train / math.sqrt(R)
        need_c = ensemble.tost_pairs_needed(float(g_sd), 0.02, sd_train=sd_c)
        obs = float(diffs.mean())
        need_obs = ensemble.tost_pairs_needed(float(g_sd), 0.02, true_diff=obs, sd_train=sd_c)
        out["tost"][f"PQ-PKS−REF-U:{a}"] = {
            "n_pairs": n, "diff_mean_over_runs": _r(diffs.mean()), "diff_sd_over_runs": _r(sd_train),
            "sd_per_image_term": _r(g_sd), "power_now_sampling_only": _r(ensemble.tost_power(float(g_sd), n, 0.02)),
            "power_now_single_cv_B": _r(ensemble.tost_power(float(g_sd), n, 0.02, sd_train=sd_train)),
            "power_now_mean_of_R_C": _r(ensemble.tost_power(float(g_sd), n, 0.02, sd_train=sd_c)),
            "pairs_for_power_0.8_sampling_only": need, "pairs_A_train_shrinks": need_a,
            "pairs_B_single_cv": need_b, "pairs_C_mean_of_R": need_c,
            "pairs_C_if_true_diff_equals_observed": need_obs, "R": R}
    return out


def run_dos(n_flood: int) -> dict:
    pp = DEFAULT_PP
    sysm = systems.make("PQ-PKS", pp)
    dk, ek = sysm.keygen()
    st = systems.ReceiverState()
    ids = data.pilot_ids()
    covers = [data.load_bossbase(i) for i in ids[-(2 * pp.max_open_sessions + n_flood + 1):]]
    t_ok, starts = [], []
    # S_max sessions, N = 3W; in each session R receives image 1 and image W + 1 (images 2..W are treated as lost), so last = W + 1
    # and the window [2, 2W + 1] has exactly 2W - 1 unseen indices: the worst case of the bound of Prop. 2.
    for s in range(pp.max_open_sessions):
        sess = sysm.start_session(ek, length=3 * pp.window)
        y1 = sysm.send(sess, b"start", covers[2 * s])
        t = time.perf_counter()
        r = sysm.receive(dk, y1, st)
        t_ok.append(time.perf_counter() - t)
        assert r.status == "ok"
        starts.append(y1)
        for i in range(2, pp.window + 2):
            y = sysm.send(sess, b"x", covers[2 * s + 1])
            if i == pp.window + 1:
                assert sysm.receive(dk, y, st).status == "ok"
    flood = covers[2 * pp.max_open_sessions:]
    tf, trials = [], []
    for x in flood[:n_flood]:
        t = time.perf_counter()
        r = sysm.receive(dk, x, st)
        tf.append(time.perf_counter() - t)
        trials.append(r.trials)
        assert r.status == "bottom"
    t = time.perf_counter()
    r = sysm.receive(dk, starts[0], st)
    t_replay = time.perf_counter() - t
    # Cost of one (sid, j) pair: image keys, keyed permutation, syndrome extraction for every bucket
    from pqpks.systems.base import image_keys
    sid, rx = next(iter(st.sessions.items()))
    x = flood[0]
    tk, tp, te = [], [], []
    for j in range(2, 2 + 10):
        t = time.perf_counter()
        k = image_keys(rx.ctx, sysm.pid, sid, j)
        tk.append(time.perf_counter() - t)
        t = time.perf_counter()
        idx = sysm.embedder.stage2_index(x.shape, None, k.perm)
        tp.append(time.perf_counter() - t)
        t = time.perf_counter()
        for b in pp.buckets:
            sysm.embedder.extract_body(x, idx, b + AEAD_TAG_BYTES, k.stc)
        te.append(time.perf_counter() - t)
    bound = len(pp.buckets) * (1 + (2 * pp.window - 1) * pp.max_open_sessions)
    return {"S_max": pp.max_open_sessions, "W": pp.window, "|B|": len(pp.buckets), "bound_trials": bound,
            "flood_images": n_flood, "trials_per_invalid_image": sorted(set(trials)),
            "time_invalid_image_s": {"median": _r(np.median(tf), 3), "max": _r(np.max(tf), 3)},
            "time_valid_start_image_s_median": _r(np.median(t_ok), 3),
            "replayed_start": {"status": r.status, "trials": r.trials, "time_s": _r(t_replay, 3)},
            "per_candidate_s_median": {"image_keys": _r(np.median(tk), 5), "keyed_permutation": _r(np.median(tp), 4),
                                       "extract_all_buckets": _r(np.median(te), 4)},
            "cpu": os.cpu_count()}


def run_kemeleon() -> dict:
    tv = kemeleon.vector_encode_tv()
    l1 = math.log2(tv.numerator) - math.log2(tv.denominator)
    q = 3329
    p_acc = 2 ** (8986 - 768 * math.log2(q)) * (1 - 1 / q) ** 256
    return {"VectorEncode_TV_log2_per_vector": _r(l1, 3), "ct_main_TV_log2_upper": _r(l1 + 2, 3),
            "ek_TV_log2_upper": _r(l1 + math.log2(3), 3), "p_acc_reduced": _r(p_acc, 5),
            "abort_log2_T64": _r(64 * math.log2(1 - p_acc), 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["all", "wa0", "a8", "power", "dos", "kemeleon"], default="all")
    ap.add_argument("--a8-cache", type=Path, default=OUT / "cache_a8")
    ap.add_argument("--n-img", type=int, default=500)
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--cache", type=Path, default=OUT / "cache")
    ap.add_argument("--pilot-cache", type=Path, default=ROOT / "results" / "archive" / "pre-confirm" / "pilot_baselines" / "cache")
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--n-flood", type=int, default=20)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rp = OUT / "report.json"
    rep = json.loads(rp.read_text()) if rp.exists() else {}
    ids = data.pilot_ids()[: args.n_img]
    t0 = time.time()
    rep["setup"] = {"dataset": "BOSSbase 1.01, the first 500 images of data/external/_pilot/pilot_bossbase_ids.txt",
                    "warden": "SPAM-686 (W-img) / SPAM on A0 and not-A0, 1372 dimensions (W-A0); FLD ensemble L=51, d_sub=200, 5-fold by image pair",
                    "alphas": ALPHAS, "profile": "R-b: single-bucket profile by alpha"}
    for ph in (["kemeleon", "dos", "power", "wa0", "a8"] if args.phase == "all" else [args.phase]):
        t = time.time()
        if ph == "kemeleon":
            rep["kemeleon"] = run_kemeleon()
        elif ph == "dos":
            rep["dos"] = run_dos(args.n_flood)
        elif ph == "power":
            rep["power"] = run_power(ids, args.pilot_cache, args.repeats)
        elif ph == "a8":
            rep["a8"] = run_a8(ids, args.a8_cache, args.cache, args.jobs)
        else:
            rep["wa0"] = run_wa0(ids, args.cache, args.jobs)
        rep.setdefault("elapsed_s", {})[ph] = round(time.time() - t)
        # read again before writing, so that two parts running in parallel do not overwrite each other
        cur = json.loads(rp.read_text()) if rp.exists() else {}
        cur[ph] = rep[ph]
        cur["setup"] = rep["setup"]
        cur.setdefault("elapsed_s", {})[ph] = rep["elapsed_s"][ph]
        rep = cur
        rp.write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=str))
        print(f"{ph} done after {time.time() - t0:.0f} s", flush=True)
    print("wrote", rp)


if __name__ == "__main__":
    main()
