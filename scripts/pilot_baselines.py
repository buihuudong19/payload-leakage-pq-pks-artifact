"""Pilot comparison of PQ-PKS with the baselines on BOSSbase; runs on a CPU-only machine.

NOT a substitute for E2/E3: the image warden is SPAM-686 + FLD ensemble (weaker than SRM/SRNet), the sample is small, and
there is no frozen split. Purpose: check that every system runs correctly on real images, check the harness (single-bucket
profile by alpha), and obtain preliminary figures to tune the experimental design.

Three parts:
  W-pay : public-extraction warden (D_pub). P = kem_part of the system (equal to Ext0 of the stego image, checked on real
          images), Q = Ext0(cover) with the layout of the system. Measures the LLR (Prop. 1A/1B), the P-256 point check, the
          fraction of ASCII bytes, and a learned distinguisher (FLD ensemble on byte/bit features, 5-fold cross-validation).
          Also delta_syn^LLR = |Pr[LLR(Q) > 0] - Pr[LLR(U) > 0]| (pilot of E1 for the LLR test).
  W-img : SPAM-686 + FLD ensemble, 5-fold cross-validation by image pair, at alpha in {0.2, 0.4} (regime R-b).
  Function: correctness, false alarms on covers, send/receive time, bit budget in regime R-a (|m| = 256 B).

Run:  PQPKS_BOSSBASE_DIR=<.pgm directory> python scripts/pilot_baselines.py [--n-img 500] [--n-pay 1000] [--jobs 2]
Output: results/archive/pre-confirm/pilot_baselines/{REPORT.md, report.json}; features stored in --cache (default results/archive/pre-confirm/pilot_baselines/cache).
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
from pqpks.analysis import dpub, ensemble, spam  # noqa: E402
from pqpks.experiment.profiles import message_len_for_bucket, profile_for_alpha  # noqa: E402
from pqpks.experiment.references import ReferenceSystem  # noqa: E402
from pqpks.params import AEAD_TAG_BYTES, DEFAULT_PP, KEMELEON_CT_BYTES, START_HEADER_BYTES  # noqa: E402
from pqpks.stego import mpvd  # noqa: E402
from pqpks.stego.two_stage import TwoStageSTC  # noqa: E402

OUT = ROOT / "results" / "archive" / "pre-confirm" / "pilot_baselines"
N_PIX = 512 * 512
ALPHAS = (0.2, 0.4)
STC_SYSTEMS = ("PQ-PKS", "PQ-PKS-R", "BL5", "BL5-L", "BL3", "BL4", "BL4-E")
ALL_IMG = STC_SYSTEMS + ("REF-U", "REF-1S", "BL2", "BL1")
N_CHECK = 20                                   # number of leading images used for the correctness check (receive)

# --------------------------------------------------------------------------- initialization (before fork)
_STATE: dict = {}


def _setup():
    st = {"stc": {}, "keys": {}}
    for name in STC_SYSTEMS:
        kb = systems.crypto_layer(name).kem_part_bytes
        for a in ALPHAS:
            st["stc"][(name, a)] = systems.make(name, profile_for_alpha(a, N_PIX, kb))
        st["keys"][name] = st["stc"][(name, ALPHAS[0])].keygen()
    for a in ALPHAS:
        st["bl2", a] = systems.Bl2System(profile_for_alpha(a, N_PIX, 1088))
    st["keys"]["BL2"] = st["bl2", ALPHAS[0]].keygen()
    st["bl1"] = systems.make("BL1")
    st["keys"]["BL1"] = st["bl1"].keygen()
    _STATE.update(st)
    st["bl1_len"] = {a: _bl1_len_for_bits(int(a * N_PIX), st["keys"]["BL1"][1]) for a in ALPHAS}
    st["ref"] = {r: ReferenceSystem(r, KEMELEON_CT_BYTES) for r in ("REF-U", "REF-1S")}
    st["emb"] = TwoStageSTC(DEFAULT_PP)
    return st


_ALPHABET = np.frombuffer(b"abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,", dtype=np.uint8)


def _rand_text(n: int, rng) -> str:
    return bytes(rng.choice(_ALPHABET, n)).decode()


def _bl1_len_for_bits(target_bits: int, pk) -> int:
    """BL1 (decimal ASCII, about 11-30x expansion): text length giving a payload close to (<=) target_bits, searched once."""
    b1 = _STATE["bl1"]
    rng = np.random.default_rng(1)
    lo, hi, best = 1, 4000, 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if 8 * len(b1.payload(pk, _rand_text(mid, rng))) <= target_bits:
            best, lo = mid, mid + 1
        else:
            hi = mid - 1
    return best


# --------------------------------------------------------------------------- W-img: one image
def _work_image(args):
    k, image_id = args
    st = _STATE
    x = data.load_bossbase(image_id)
    out = {"id": image_id, "feat": {"cover": spam.spam686(x)}, "time": {}, "check": {}, "alpha_real": {}}
    t = time.perf_counter()
    cm = st["emb"].cost_map(x)
    out["time"]["hill"] = time.perf_counter() - t
    for a in ALPHAS:
        for name in STC_SYSTEMS:
            sysm = st["stc"][(name, a)]
            dk, ek = st["keys"][name]
            B = sysm.pp.buckets[0]
            msg = secrets.token_bytes(message_len_for_bucket(B, start=True))
            t = time.perf_counter()
            sess = sysm.start_session(ek, length=1)
            y = sysm.send(sess, msg, x, cost_map=cm)
            out["time"][f"send:{name}:{a}"] = time.perf_counter() - t
            # Ext0(stego) must equal kem_part: this justifies using kem_part instead of extracting from the image in W-pay
            out["check"][f"ext0:{name}:{a}"] = sysm.embedder.extract_kem(y, len(sess.kem_part)) == sess.kem_part
            if k < N_CHECK:
                t = time.perf_counter()
                r = sysm.receive(dk, y, systems.ReceiverState())
                out["time"][f"recv:{name}:{a}"] = time.perf_counter() - t
                out["check"][f"recv:{name}:{a}"] = r.status == "ok" and r.msg == msg
            out["feat"][f"{name}:{a}"] = spam.spam686(y)
            out["alpha_real"][f"{name}:{a}"] = int((y != x).sum())
        body = profile_for_alpha(a, N_PIX, KEMELEON_CT_BYTES).buckets[0] + AEAD_TAG_BYTES
        for r in ("REF-U", "REF-1S"):
            y = st["ref"][r].stego(x, body, start=True, cost_map=cm)
            out["feat"][f"{r}:{a}"] = spam.spam686(y)
            out["alpha_real"][f"{r}:{a}"] = int((y != x).sum())
        # BL2: raw ML-KEM + sequential MPVD
        b2 = st["bl2", a]
        dk, ek = st["keys"]["BL2"]
        msg = secrets.token_bytes(message_len_for_bucket(b2.pp.buckets[0], start=True))
        t = time.perf_counter()
        y = b2.send_once(ek, msg, x)
        out["time"][f"send:BL2:{a}"] = time.perf_counter() - t
        if k < N_CHECK:
            out["check"][f"recv:BL2:{a}"] = b2.receive_once(dk, y).msg == msg
        out["feat"][f"BL2:{a}"] = spam.spam686(y)
        out["alpha_real"][f"BL2:{a}"] = int((y != x).sum())
        # BL1: BL1 reference implementation, text length tuned so that the payload is ~ alpha n bits
        sk, pk = st["keys"]["BL1"]
        text = _rand_text(st["bl1_len"][a], np.random.default_rng(image_id))
        bits = 8 * len(st["bl1"].payload(pk, text))
        t = time.perf_counter()
        y = st["bl1"].send_once(pk, text, x)
        out["time"][f"send:BL1:{a}"] = time.perf_counter() - t
        if k < N_CHECK:
            r = st["bl1"].receive_once(sk, y)
            out["check"][f"recv:BL1:{a}"] = r.msg is not None and r.msg.decode("latin-1") == text
        out["feat"][f"BL1:{a}"] = spam.spam686(y)
        out["alpha_real"][f"BL1:{a}"] = int((y != x).sum())
        out.setdefault("bl1_bits", {})[a] = bits
    return out


# --------------------------------------------------------------------------- W-pay
def _pay_features(blobs: list[bytes]) -> np.ndarray:
    """Common features of the learned distinguisher: byte frequencies (256) + fraction of 1 bits by position in the byte (8)
    + fraction of 1 bits in 16 equal segments of the string (16)."""
    rows = []
    for b in blobs:
        a = np.frombuffer(b, dtype=np.uint8)
        bits = np.unpackbits(a).reshape(-1, 8)
        seg = np.array_split(bits.ravel(), 16)
        rows.append(np.concatenate([np.bincount(a, minlength=256) / a.size, bits.mean(0),
                                    [s.mean() for s in seg]]))
    return np.array(rows)


def _learned_acc(P: list[bytes], Q: list[bytes], seed=0) -> float:
    """Learned distinguisher: FLD ensemble on byte/bit features, 5-fold cross-validation (P and Q paired by index).
    Accuracy = 1 - P_E on the test fold."""
    r = ensemble.cross_validate(_pay_features(Q), _pay_features(P), L=31, d_sub=100, seed=seed)
    return 1.0 - r["P_E"]


def _wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return round(c - h, 4), round(c + h, 4)


def run_wpay(ids: list[int]) -> dict:
    emb = TwoStageSTC(DEFAULT_PP)
    covers = [data.load_bossbase(i) for i in ids]
    res = {}
    layouts = {}
    for name in STC_SYSTEMS:
        c = systems.crypto_layer(name)
        kb = c.kem_part_bytes
        if kb not in layouts:
            layouts[kb] = [emb.extract_kem(x, kb) for x in covers]
        from pqpks.systems.base import profile_id_for
        pid = profile_id_for(c, DEFAULT_PP)
        _, ek = c.keygen()
        P = [c.start(ek, pid)[0] for _ in ids]
        Q = layouts[kb]
        res[name] = _pay_scores(name, P, Q)
    # REF-U: uniform payload of 1536 B
    res["REF-U"] = _pay_scores("REF-U", [secrets.token_bytes(KEMELEON_CT_BYTES) for _ in ids], layouts[KEMELEON_CT_BYTES])
    # BL2: sequential MPVD, Ext = the first 1088 B extracted by MPVD
    b2 = systems.make("BL2")
    _, ek = b2.keygen()
    P = [b2.payload(ek, b"m")[0] for _ in ids]
    Q = [_bits_to_bytes(mpvd.extract(x, 8 * 1088)) for x in covers]
    res["BL2"] = _pay_scores("BL2", P, Q)
    # BL1: ASCII string embedded by the BL1 reference code, compared with a string of the same length extracted from the cover
    b1 = systems.make("BL1")
    _, pk = b1.keygen()
    P = [b1.payload(pk, "pilot test message: %d" % j).encode("latin-1")[:256] for j in range(min(len(ids), 300))]   # same length as the original plaintexts
    Q = [_bits_to_bytes(mpvd.extract(x, 8 * 256)) for x in covers[: len(P)]]
    res["BL1"] = _pay_scores("BL1", P, Q)
    # delta_syn of the LLR test on cover syndromes (pilot of E1): Q (1536 B and 1088 B) against uniform strings
    # A single measurement: the LLR reads the first 960 B, and the first 960 syndrome bytes of the 1536 B and 1088 B layouts
    # coincide (same prefix of pi_0, same block width for the first 7680 rows), so only the 1536 B layout is reported.
    U = [secrets.token_bytes(1088) for _ in ids]
    n = len(ids)
    pu = float(np.mean([dpub.mlkem_raw_llr(u) > 0 for u in U]))
    pq = float(np.mean([dpub.mlkem_raw_llr(q) > 0 for q in layouts[KEMELEON_CT_BYTES]]))
    se = math.sqrt(pq * (1 - pq) / n + pu * (1 - pu) / n)
    res["delta_syn_LLR"] = {"Pr[LLR(Q)>0]": pq, "Pr[LLR(U)>0]": pu, "diff": pq - pu, "n": n,
                            "diff_ci95": (pq - pu - 1.96 * se, pq - pu + 1.96 * se),
                            "delta_upper95": abs(pq - pu) + 1.96 * se}
    return res


def _bits_to_bytes(bits: str) -> bytes:
    return bytes(int(bits[i:i + 8], 2) for i in range(0, len(bits) - len(bits) % 8, 8))


def _pay_scores(name, P, Q):
    n = min(len(P), len(Q))
    P, Q = P[:n], Q[:n]
    out = {"n": n}
    if all(len(p) >= 960 for p in P):
        lp = np.array([dpub.mlkem_raw_llr(p) > 0 for p in P])
        lq = np.array([dpub.mlkem_raw_llr(q) > 0 for q in Q])
        k = int(lp.sum() + (~lq).sum())
        out["llr_acc"] = round(k / (2 * n), 4)
        out["llr_acc_ci95"] = _wilson(k, 2 * n)
    pp = np.array([dpub.p256_uncompressed_on_curve(p) for p in P])
    pq = np.array([dpub.p256_uncompressed_on_curve(q) for q in Q])
    out["p256_acc"] = round((pp.sum() + (~pq).sum()) / (2 * n), 4)
    ap = np.array([dpub.ascii_digit_fraction(p) for p in P])
    aq = np.array([dpub.ascii_digit_fraction(q) for q in Q])
    out["ascii_acc"] = round(((ap > 0.5).sum() + (aq <= 0.5).sum()) / (2 * n), 4)
    out["learned_acc"] = round(_learned_acc(P, Q), 4)
    out["best_acc"] = max(v for kk, v in out.items() if kk.endswith("_acc"))
    return out


# --------------------------------------------------------------------------- auxiliary functions
def run_false_alarm(ids: list[int]) -> dict:
    res = {}
    for name in STC_SYSTEMS:
        sysm = systems.make(name)
        dk, _ = sysm.keygen()
        bad = sum(sysm.receive(dk, data.load_bossbase(i), systems.ReceiverState()).status != "bottom" for i in ids)
        res[name] = {"covers": len(ids), "accepted": int(bad)}
    return res


def ra_budget() -> dict:
    """Regime R-a: |m| = 256 B, unsigned, default ladder; actual alpha of the session-start image."""
    res = {}
    for name in STC_SYSTEMS:
        kb = systems.crypto_layer(name).kem_part_bytes
        b = next(b for b in DEFAULT_PP.buckets if START_HEADER_BYTES + 256 <= b)
        total = kb + b + AEAD_TAG_BYTES
        n0 = math.ceil(8 * kb / DEFAULT_PP.alpha0)
        res[name] = {"kem_part_B": kb, "bucket_B": b, "payload_B": total, "alpha": round(8 * total / N_PIX, 4),
                     "alpha1": round(8 * (b + AEAD_TAG_BYTES) / (N_PIX - n0), 4)}
    res["BL2"] = {"kem_part_B": 1088, "payload_B": 1088 + 512 + 16, "alpha": round(8 * (1088 + 528) / N_PIX, 4)}
    return res


def wimg_eval(rows: list[dict]) -> dict:
    fc = np.array([r["feat"]["cover"] for r in rows])
    res, raw = {}, {}
    for a in ALPHAS:
        for name in ALL_IMG:
            key = f"{name}:{a}"
            fs = np.array([r["feat"][key] for r in rows])
            r = ensemble.cross_validate(fc, fs)
            lo, hi = ensemble.bootstrap_pe(r["dec_cover"], r["dec_stego"])
            changes = np.array([row["alpha_real"][key] for row in rows])
            res[key] = {"P_E": round(r["P_E"], 4), "P_E_ci95": (round(lo, 4), round(hi, 4)),
                        "AUC": round(r["AUC"], 4), "change_rate": round(float(changes.mean() / N_PIX), 5)}
            raw[key] = r
        pairs = [("PQ-PKS", o) for o in ("REF-U", "BL5-L", "REF-1S", "BL5", "BL4", "BL3")] + [("REF-U", "REF-1S")]
        for x_, o in pairs:
            d = ensemble.paired_diff_ci(raw[f"{x_}:{a}"], raw[f"{o}:{a}"])
            res[f"diff:{x_}-{o}:{a}"] = {k: (tuple(round(v, 4) for v in val) if isinstance(val, tuple)
                                            else round(val, 4)) for k, val in d.items()}
        d = res[f"diff:PQ-PKS-REF-U:{a}"]
        res[f"tost:PQ-PKS-REF-U:{a}"] = {"delta": 0.02, "equivalent": bool(-0.02 < d["ci90"][0] and d["ci90"][1] < 0.02)}
    # Holm correction over the whole family of paired comparisons (every pair x every alpha)
    fam = {k: v["p_boot"] for k, v in res.items() if k.startswith("diff:")}
    for k, p in ensemble.holm(fam).items():
        res[k]["p_holm"] = round(p, 4)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-img", type=int, default=500)
    ap.add_argument("--n-pay", type=int, default=1000)
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--cache", type=Path, default=OUT / "cache")
    ap.add_argument("--phase", choices=["all", "wpay", "wimg", "func"], default="all")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    args.cache.mkdir(parents=True, exist_ok=True)
    ids = data.pilot_ids()
    rep_path = OUT / "report.json"
    report = json.loads(rep_path.read_text()) if rep_path.exists() else {}
    report["setup"] = {"dataset": "BOSSbase 1.01, 512x512 images, list data/external/_pilot/pilot_bossbase_ids.txt",
                       "n_img": args.n_img, "n_pay": args.n_pay, "alphas": ALPHAS, "warden_img": "SPAM-686 + FLD ensemble (L=51, d_sub=200), 5-fold by image pair",
                       "profile": "R-b: single-bucket profile by alpha; default ladder for R-a"}
    t0 = time.time()
    if args.phase in ("all", "wpay"):
        report["wpay"] = run_wpay(ids[: args.n_pay])
        print(f"W-pay done after {time.time() - t0:.0f} s", flush=True)
    if args.phase in ("all", "func"):
        report["false_alarm"] = run_false_alarm(ids[-100:])
        report["ra_budget"] = ra_budget()
        print(f"function checks done after {time.time() - t0:.0f} s", flush=True)
    if args.phase in ("all", "wimg"):
        global _STATE
        _STATE.update(_setup())
        todo = [(k, i) for k, i in enumerate(ids[: args.n_img]) if not (args.cache / f"{i}.npz").exists()]
        with Pool(args.jobs) as pool:
            for done, out in enumerate(pool.imap_unordered(_work_image, todo), 1):
                np.savez_compressed(args.cache / f"{out['id']}.npz",
                                    **{f"f__{k}": v for k, v in out["feat"].items()},
                                    meta=json.dumps({kk: out[kk] for kk in ("time", "check", "alpha_real", "id")} |
                                                    {"bl1_bits": {str(a): b for a, b in out.get("bl1_bits", {}).items()}}))
                if done % 10 == 0:
                    print(f"W-img {done}/{len(todo)} after {time.time() - t0:.0f} s", flush=True)
        rows = []
        for i in ids[: args.n_img]:
            z = np.load(args.cache / f"{i}.npz")
            meta = json.loads(str(z["meta"]))
            rows.append({"feat": {k[3:]: z[k] for k in z.files if k.startswith("f__")}} | meta)
        report["wimg"] = wimg_eval(rows)
        checks = {}
        for r in rows:
            for k, v in r["check"].items():
                checks.setdefault(k, []).append(bool(v))
        report["checks"] = {k: f"{sum(v)}/{len(v)}" for k, v in sorted(checks.items())}
        times = {}
        for r in rows:
            for k, v in r["time"].items():
                times.setdefault(k, []).append(v)
        report["timing_median_s"] = {k: round(float(np.median(v)), 3) for k, v in sorted(times.items())}
        report["bl1_payload_bits"] = {a: int(np.median([r["bl1_bits"][str(a)] for r in rows])) for a in ALPHAS}
    report.setdefault("elapsed_s", {})
    if not isinstance(report["elapsed_s"], dict):
        report["elapsed_s"] = {"all_phases_first_run": report["elapsed_s"]}
    report["elapsed_s"][args.phase] = round(time.time() - t0)
    rep_path.write_text(json.dumps(report, ensure_ascii=False, indent=1, default=str))
    print("ghi", rep_path)


if __name__ == "__main__":
    main()
