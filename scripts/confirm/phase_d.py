"""Phase D: E7 (external source), S2 (ablations), S5 (regime R-a).

Usage:
  python scripts/confirm/phase_d.py split-e7      E7 split file configs/splits/e7_v1.json (3 splits; test 2500, validation
                                                  1000, stratified by camera model; seed 20261108). Deterministic: its
                                                  SHA-256 is registered in pipeline.ps1 before any stego image is generated.
  python scripts/confirm/phase_d.py tables        supplementary tables S2-S7 (Markdown) from results/{e7,ablate,ablate2,ra,jpeg,jpeg95,confirm_r}/summary*.json
                                                  -> results/phase_d/PHASE_D_TABLES_en.md
Stego generation and features: scripts/experiments/screen_run.py --plan {e7,ablate,ra}; analysis: scripts/confirm/analyze_confirm.py
(see the header of both files).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
SEED_E7 = 20261108
R_E7, N_TEST, N_VAL = 3, 2500, 1000
SUBSET = ROOT / "data" / "external" / "alaska2" / "subset_ids.txt"
SPLIT_E7 = ROOT / "configs" / "splits" / "e7_v1.json"


def _quota(sizes: np.ndarray, total: int) -> np.ndarray:
    """Allocate `total` in proportion to sizes, rounding by largest remainder (as make_alaska2_subset.py)."""
    q = sizes / sizes.sum() * total
    take = np.floor(q).astype(int)
    for i in np.argsort(-(q - take))[: total - take.sum()]:
        take[i] += 1
    return take


def split_e7(_a):
    rows = [ln.split() for ln in SUBSET.read_text().splitlines() if ln.strip()]
    ids = [f"alaska_{int(r[0])}" for r in rows]                # same "<src>_<id>" form as ids.json of consolidate
    cam = {k: (r[1] if len(r) > 1 else "") for k, r in zip(ids, rows)}
    cams = sorted(set(cam.values()))
    groups = {c: [k for k in ids if cam[k] == c] for c in cams}
    sizes = np.array([len(groups[c]) for c in cams])
    t_q, v_q = _quota(sizes, N_TEST), _quota(sizes, N_VAL)
    test_sets, val_sets = [], []
    for r in range(R_E7):
        rng = np.random.default_rng(SEED_E7 + r)
        te, va = [], []
        for c, tq, vq in zip(cams, t_q, v_q):
            g = list(rng.permutation(groups[c]))
            te += g[:tq]
            va += g[tq:tq + vq]
        test_sets.append(sorted(te))
        val_sets.append(sorted(va))
    # The note is kept verbatim (in Vietnamese) because the SHA-256 of e7_v1.json is registered. English: "E7: 3 splits;
    # test 2500 and validation 1000 stratified by camera model (largest remainder), seed 20261108 + r; train = the remaining
    # 6500 images. The warden is trained and tested on ALASKA2 itself (H7: external source, not mixed with BOSSbase/BOWS2)."
    doc = {"note": "E7 (E-45): 3 lần chia; test 2500 và validation 1000 phân tầng theo máy ảnh (phần dư lớn nhất), seed 20261108 + r; "
                   "train = 6500 ảnh còn lại. Warden học và kiểm trên chính ALASKA2 (H7: nguồn ngoài, không trộn với BOSSbase/BOWS2).",
           "subset_sha256": hashlib.sha256(SUBSET.read_bytes()).hexdigest(), "seed": SEED_E7,
           "n": len(ids), "cameras": len(cams), "test_sets": test_sets, "val_sets": val_sets, "camera": cam}
    SPLIT_E7.parent.mkdir(parents=True, exist_ok=True)
    SPLIT_E7.write_bytes(json.dumps(doc, indent=0).encode("utf-8"))     # bytes: the same SHA on every platform
    print(f"{len(ids)} images, {len(cams)} camera models; per split test {len(test_sets[0])}, val {len(val_sets[0])}; "
          f"SHA-256 {hashlib.sha256(SPLIT_E7.read_bytes()).hexdigest()}")


def _fmt(x, d=3):
    return "–" if x is None else f"{x:.{d}f}"


def _row_pe(summ, cell, name):
    r = summ["E2E3"].get(f"{cell}/{name}")
    if not r:
        return None
    wi, wp, wc = r["W-img"], r["W-pay"], r["W-comb"]
    d = r["dTPR_pay_minus_img"]
    no_dpub = name.startswith("BL4")          # no registered payload distinguisher for BL4, BL4-E (Table 4: "-"); W-pay score = 0
    pay = "n/a" if no_dpub else _fmt(wp["P_E"])
    tpr_pay = "n/a" if no_dpub else _fmt(wp["TPR@5"])
    dtpr = "n/a" if no_dpub else f"{d['dTPR']:+.3f} [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}]"
    dpub = "n/a" if no_dpub else _fmt(r["dpub_acc_M1"]["acc"])
    return (f"{_fmt(wi['P_E'])} [{_fmt(wi['P_E_ci95'][0])}, {_fmt(wi['P_E_ci95'][1])}] | {_fmt(r['W-A0']['P_E']) if 'W-A0' in r else '–'} | "
            f"{pay} | {_fmt(wc['P_E'])} | {_fmt(wi['TPR@5'])} | {tpr_pay} | {dtpr} | {dpub}")


# The English tables are located by their "**Table Sk" labels. The table of the rejection-sampled variant is S7, because S6 of
# the supplementary material is the qualitative comparison with [Zha24, Zha25].
_TXT = {
    "en": dict(
        head=["Supplementary tables S2-S7 (written by scripts/confirm/phase_d.py tables, step PF5).", "",
              "Column $D_{\\rm pub}$: LLR test of Proposition 2 for strings of at least 960 B; BL3: P-256 point check (Table 4); BL4 and BL4-E have no registered payload distinguisher, so W-pay, ΔTPR and $D_{\\rm pub}$ are 'n/a' and W-comb reduces to W-img. 'True α' is the realized embedding rate of the cell. Equivalence: paired TOST, δ = 0.02, 90 % CI of the difference.", ""],
        missing="no results yet", reps="repeated splits; TOST δ = 0.02.",
        cols="| Cell | System | true α | $P_E$ W-img [95 % CI] | $P_E$ W-A0 | $P_E$ W-pay | $P_E$ W-comb | TPR@5 % W-img | TPR@5 % W-pay | ΔTPR (W-pay − W-img) [95 % CI] | $D_{\\rm pub}$ acc. $M$ = 1 |",
        tost="| Cell | Comparison | Warden | Δ$P_E$ | 90 % CI | Verdict |", fname="PHASE_D_TABLES_en.md",
        titles=["Table S4 [E7: external source, 10,000 ALASKA2 covers, SRM + FLD ensemble; camera-model cluster bootstrap]",
                "Table S2 [ablations A1 (stage-2 keying V1–V3), A3 (S-UNIWARD: SUNI, REF-U-SUNI), A4 (Kemeleon R: PQ-PKS-R), A8′ (fixed α₀ = 0.15: A8p); 2000 screening covers, SRM, 5 folds]",
                "Table S2 (continued) [A7 (hybrid ML-KEM-768 + X25519; REF-U-1568 is its uniform reference), A9 (STC height h = 8: H8, h = 12: H12); 2000 screening covers, SRM, 5 folds]",
                "Table S5 [regime R-a: same message of L bytes, each system at its own rate; SRM, 5 folds]",
                "Table S3 [JPEG QF75, DCT domain, J-UNIWARD on the 2000 screening covers; W-img = DCTR, 5 folds; cells are the payload buckets Q2048 and Q3072 (bytes); the rate column is not defined per cell]",
                "Table S3 (continued) [JPEG QF75; W-img = GFR]",
                "Table S3 (continued) [JPEG QF95; W-img = DCTR, 5 folds]",
                "Table S3 (continued) [JPEG QF95; W-img = GFR]",
                "Table S7 [exploratory: PQ-PKS-R (rejection-sampled Kemeleon, 1252 B) against REF-U-1252 at α = 0.04651, its minimal first image; 20,000 confirmatory covers, SRM]"]),
}
_SOURCES = (("e7", "summary.json"), ("ablate", "summary.json"), ("ablate2", "summary.json"), ("ra", "summary.json"),
            ("jpeg", "summary.json"), ("jpeg", "summary_gfr.json"), ("jpeg95", "summary.json"), ("jpeg95", "summary_gfr.json"),
            ("confirm_r", "summary.json"))


def _tables_lang(lang):
    T = _TXT[lang]
    lines = list(T["head"])
    for (key, fname), title in zip(_SOURCES, T["titles"]):
        p = ROOT / "results" / key / fname
        if not p.exists():
            lines += [f"**{title}.** {T['missing']} ({p.relative_to(ROOT)})", ""]
            continue
        summ = json.loads(p.read_text(encoding="utf-8"))
        lines += [f"**{title}.** {summ.get('reps')} {T['reps']}", "", T["cols"], "|---|---|---|---|---|---|---|---|---|---|---|"]
        abc = summ.get("alpha_by_cell") or {}
        for k in summ["E2E3"]:
            cell, name = k.split("/", 1)
            al = abc.get(cell, {}).get(name)
            lines.append(f"| {cell} | {name} | {_fmt(al, 4)} | {_row_pe(summ, cell, name)} |")
        lines += ["", T["tost"], "|---|---|---|---|---|---|"]
        for k, t in summ["TOST"].items():
            cell, cmp_, w = k.split("/")
            lines.append(f"| {cell} | {cmp_} | {w} | {t['diff']:+.4f} | [{t['ci90'][0]:+.4f}, {t['ci90'][1]:+.4f}] | {t['verdict']} |")
        lines.append("")
    return lines


def tables(_a):
    out = ROOT / "results" / "phase_d"
    out.mkdir(parents=True, exist_ok=True)
    for lang in ("en",):
        f = out / _TXT[lang]["fname"]
        f.write_text("\n".join(_tables_lang(lang)), encoding="utf-8")
        print(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["split-e7", "tables"])
    a = ap.parse_args()
    {"split-e7": split_e7, "tables": tables}[a.step](a)


if __name__ == "__main__":
    main()
