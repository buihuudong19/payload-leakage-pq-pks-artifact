"""D_pub of BL3 via the P-256 point check; E5 with the placement of sigma (A5b, A5c); JPEG by QF; the confirm_r plan;
checkpoint file names by number of splits; the confirmatory tables."""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "confirm"))
sys.path.insert(0, str(ROOT / "scripts" / "experiments"))


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_bl3_point_check_matches_table_4():
    from pqpks.crypto import hpke
    ac = _load("analyze_confirm", ROOT / "scripts" / "confirm" / "analyze_confirm.py")
    kem = hpke.DhkemP256()
    if kem is not None:                                   # a real BL3 point (enc of 65 B) must be valid
        _, pk = kem.generate()
        _, enc = kem.encap(pk)
        assert len(enc) == 65 and ac.p256_point_valid(np.frombuffer(enc, dtype=np.uint8))
    rng = np.random.default_rng(1)
    u = rng.integers(0, 256, size=(4000, 65), dtype=np.uint8)
    u[:, 0] = 4
    assert not any(ac.p256_point_valid(s) for s in u)      # uniform strings (random x, y) satisfy y^2 = x^3 - 3x + b with probability ~ 2^-256
    assert not ac.p256_point_valid(np.zeros(65, dtype=np.uint8)) and not ac.p256_point_valid(np.zeros(64, dtype=np.uint8))


def test_e5_signature_placement_and_fn_dsa():
    e5 = _load("e5_budget", ROOT / "scripts" / "experiments" / "e5_budget.py")
    assert e5.SIG["FN-DSA-512"] == 666
    a = 0.0596
    bks = e5.buckets_merged()
    assert e5.bits_per_image(a, 1, 1536, 0, bks, single=True) == 8 * 242           # session-start image: bucket 256 - 14
    assert e5.bits_per_image(a, 2, 1536, 2420, bks, place="image1") is None       # ML-DSA-44 does not fit image 1
    assert e5.bits_per_image(a, 2, 1536, 2420, bks, place="image2") is None       # nor image 2 (bucket 1792)
    assert e5.bits_per_image(a, 2, 1536, 666, bks, place="image2") is not None    # FN-DSA-512 in image 2: feasible
    assert e5.bits_per_image(a, 1, 1536, 666, bks, place="image2") is None        # A5b needs N >= 2
    assert e5.bits_per_image(a, 2, 1536, 2420, bks, place="span") is None
    v4 = e5.bits_per_image(a, 4, 1536, 2420, bks, place="span")
    assert v4 is not None and v4 == 8 * (242 + 3 * 1786 - 2420) / 4               # A5c: sigma takes the body of the later images
    big = 0.2
    assert e5.bits_per_image(big, 2, 1536, 3309, bks, place="image1") == e5.bits_per_image(big, 2, 1536, 3309, bks)  # default = image1


def test_confirm_r_plan_is_the_minimal_first_image_of_variant_r():
    sr = _load("screen_run", ROOT / "scripts" / "experiments" / "screen_run.py")
    assert "confirm_r" in sr.PLANS and sr.PLAN_COVERS["confirm_r"] == "confirm"
    (al,), systems_at = sr.PLANS["confirm_r"]
    assert systems_at[al] == ("PQ-PKS-R", "REF-U-1252") and sr.KEM_BYTES["REF-U-1252"] == 1252
    n = 512 * 512
    for name in systems_at[al]:
        assert sr.bucket_of(al, n, sr.KEM_BYTES[name]) == 256                     # bucket exactly 256: 1252 + 256 + 16 = 1524 B
    ref = sr.make_system("REF-U-1252", sr.profile_of(al, n, 1252))
    assert ref.kem_bytes == 1252 if hasattr(ref, "kem_bytes") else True
    s, dk, ek = sr.make_system("PQ-PKS-R", sr.profile_of(al, n, 1252))
    assert s.crypto.kem_part_bytes == 1252 and s.pp.buckets == (256,)


def test_jpeg_qf_parameter(tmp_path):
    import pytest
    pytest.importorskip("jpeglib")
    jd = _load("jpeg_domain", ROOT / "scripts" / "experiments" / "jpeg_domain.py")
    rng = np.random.default_rng(3)
    base = np.cumsum(np.cumsum(rng.normal(0, 1, (256, 256)), 0), 1)
    base = (base - base.min()) / (base.max() - base.min()) * 160 + 40
    x = np.clip(base + rng.normal(0, 14, (256, 256)), 0, 255).astype(np.uint8)
    j75, j95 = jd.JpegCover.from_gray(x, qf=75), jd.JpegCover.from_gray(x, qf=95)
    assert j75.qf == 75 and j95.qf == 95 and j95.n_u > j75.n_u                   # QF95 keeps more non-zero AC coefficients
    assert abs(jd.dctr_q(75) - 4.0) < 1e-9 and abs(jd.dctr_q(95) - 0.8) < 1e-9
    assert j95.with_values(j95.values()).qf == 95


def test_cell_checkpoint_name_depends_on_reps(tmp_path):
    ac = _load("analyze_confirm", ROOT / "scripts" / "confirm" / "analyze_confirm.py")
    ac.BASE = tmp_path
    per3 = [{"W-img": {"c": np.zeros(3), "s": np.ones(3), "t5": 0.1, "t1": 0.2, "tpe": 0.3, "val_pe": 0.4}} for _ in range(3)]
    ac.save_cell(0.2, "PQ-PKS", per3, "t")
    assert (tmp_path / "cells" / "a0.2_PQ-PKS.npz").exists()
    per10 = per3 * 3 + per3[:1]
    ac.save_cell(0.2, "PQ-PKS", per10, "t")
    assert (tmp_path / "cells" / "a0.2_PQ-PKS_R10.npz").exists()                  # the 3-split file is not overwritten
    assert ac.load_cell(0.2, "PQ-PKS", "t", 3) is not None and len(ac.load_cell(0.2, "PQ-PKS", "t", 10)) == 10


def test_write_results_confirm_blocks(tmp_path, monkeypatch):
    wr = _load("write_results_confirm", ROOT / "scripts" / "experiments" / "write_results_confirm.py")
    out = tmp_path / "RESULTS_TABLES.md"
    monkeypatch.setattr(wr, "RES", tmp_path)              # no input file: every block reports "no results yet" but all 5 blocks are written
    monkeypatch.setattr(sys, "argv", ["x", "--out", str(out)])
    wr.main()
    t = out.read_text(encoding="utf-8")
    for k in ("9", "10", "11", "12", "S1"):
        assert f"**Table {k} " in t
    assert t.count("**Table ") == 5


def test_e52_bootstrap_and_m4():
    """Every bootstrap uses 10^4 resamples; H2a with M = 4 has c_4 = 1 - 0.5 * 0.2024^4 ~ 0.999; promote_r10 rejects a summary without 'bootstrap'."""
    ac = _load("analyze_confirm", ROOT / "scripts" / "confirm" / "analyze_confirm.py")
    assert ac.B == ac.B_AUC == 10000
    assert set(ac.C_M) == {1, 2, 3, 4} and abs(ac.C_M[4] - (1 - 0.5 * 0.2024 ** 4)) < 5e-4 and abs(ac.C_M[3] - (1 - 0.5 * 0.2024 ** 3)) < 5e-4
    mf = _load("make_figures_confirm", ROOT / "scripts" / "experiments" / "make_figures_confirm.py")
    assert mf.C_M == ac.C_M
    import subprocess, tempfile
    with tempfile.TemporaryDirectory() as d:
        base = Path(d)
        (base / "summary_r10.json").write_text(json.dumps({"reps": 10, "E2E3": {}}), encoding="utf-8")
        r = subprocess.run([sys.executable, str(ROOT / "scripts" / "confirm" / "promote_r10.py"), "--base", d], capture_output=True, text=True)
        assert r.returncode != 0 and "bootstrap" in (r.stdout + r.stderr)
        (base / "summary_r10.json").write_text(json.dumps({"reps": 10, "E2E3": {"x": 1}, "bootstrap": {"B": 10000, "B_AUC": 10000}}), encoding="utf-8")
        (base / "summary.json").write_text(json.dumps({"reps": 3, "E2E3": {}}), encoding="utf-8")
        (base / "decisions.json").write_text("{}", encoding="utf-8")
        r = subprocess.run([sys.executable, str(ROOT / "scripts" / "confirm" / "promote_r10.py"), "--base", d], capture_output=True, text=True)
        assert r.returncode == 0, r.stdout + r.stderr
        assert json.loads((base / "summary.json").read_text())["reps"] == 10
        assert (base / "summary_r3.json").exists() and (base / "decisions_r3.json").exists()


def test_e53_results_tables_srnet_rows_and_bl4_na(tmp_path, monkeypatch):
    """The SRNet sub-table of Table 10 has rows when confirm_srnet_summary.json exists; D_pub columns M = 1-4;
    S5: BL4 is reported as n/a for W-pay, dTPR and D_pub."""
    wr = _load("write_results_confirm", ROOT / "scripts" / "experiments" / "write_results_confirm.py")
    res = tmp_path / "results"
    (res / "confirm_v1").mkdir(parents=True)
    (res / "srnet").mkdir()
    cell = {"W-img": {"P_E": 0.3, "P_E_ci95": [0.29, 0.31], "TPR@5": 0.2}, "W-A0": {"P_E": 0.4, "P_E_ci95": [0.39, 0.41]},
            "W-pay": {"P_E": 0.5, "P_E_ci95": [0.49, 0.51], "TPR@5": 0.05}, "W-comb": {"P_E": 0.3, "P_E_ci95": [0.29, 0.31]},
            "dTPR_pay_minus_img": {"dTPR": -0.15, "ci95": [-0.2, -0.1]},
            "dpub_acc": {"1": {"acc": 0.5}, "2": {"acc": 0.5}, "3": {"acc": 0.5}, "4": {"acc": 0.5}}, "H2a": {"1": {"holds": True}}}
    S = {"reps": 10, "E2E3": {"0.2/PQ-PKS": cell, "0.2/REF-U": cell}, "TOST": {"0.2/PQ-PKS-vs-REF-U/W-img": {"diff": 0.0, "ci90": [-0.003, 0.003], "verdict": "equivalent"}},
         "alpha_star": {"alpha_star_i": 0.0579, "alpha_star_i_ci95": [0.0557, 0.06], "alpha_star_ii": 0.4, "alpha_star": 0.0579,
                        "P_E_W-img_REF-U": {"0.2": 0.3}, "cond_ii_TPR_diff_W-comb": {"0.2": 0.0}},
         "excluded_covers": {"n": 0, "ids": []}}
    D = {"alpha_star": 0.0579, "rule_A5b": {"sig_placement": "image2"}, "rule_signature": {"recommended": None, "direct_checks": {}},
         "minimal_first_image": {"alpha": 0.0552, "direct_check": {"alpha": 0.0552, "P_E_W-img_REF-U": 0.4523, "TPR_diff_W-comb": -0.013, "holds": True}},
         "statement": "s", "thresholds_inside_ci95_of_alpha_star_i": []}
    sysrow = {"P_E": 0.128, "P_E_ci95": [0.122, 0.134], "TPR@5": 0.075, "AUC": 0.914}
    SR = {"alphas": {"0.4": {"PQ-PKS": dict(sysrow), "REF-U": dict(sysrow, P_E=0.1272),
                             "PQ-PKS_vs_REF-U": {"TOST": {"diff": 0.0008, "ci90": [-0.0026, 0.004], "verdict": "equivalent"},
                                                 "McNemar": {"PQ_right_REF_wrong": 211, "REF_right_PQ_wrong": 219, "p_exact": 0.736}},
                             "cond_i_SRNet": {"P_E_REF-U": 0.1272, "P_E_REF-U_SRM": 0.196, "stronger_on_REF-U": "SRNet"}}}}
    DS = dict(D, srnet={"included": True, "alpha_star_with_SRNet": 0.0579})
    for p, obj in ((res / "confirm_v1" / "summary.json", S), (res / "confirm_v1" / "decisions.json", D),
                   (res / "srnet" / "confirm_srnet_summary.json", SR), (res / "srnet" / "decisions_srnet.json", DS)):
        p.write_text(json.dumps(obj), encoding="utf-8")
    monkeypatch.setattr(wr, "RES", res)
    out = tmp_path / "RESULTS_TABLES.md"
    monkeypatch.setattr(sys, "argv", ["x", "--out", str(out)])
    wr.main()
    t = out.read_text(encoding="utf-8")
    assert "| 0.4 | PQ-PKS | 0.1280 [0.122, 0.134] | 0.075 | 0.914 | +0.0008 [-0.0026, +0.0040] | equivalent | 211 / 219; p = 0.736 |" in t
    assert "| 0.4 | REF-U | 0.1272 [0.122, 0.134]" in t and "M = 1 / 2 / 3 / 4" in t and "0.500 / 0.500 / 0.500 / 0.500" in t
    assert "Excluded covers: 0." in t and "none fits" in t
    pd = _load("phase_d", ROOT / "scripts" / "confirm" / "phase_d.py")
    summ = {"E2E3": {"L1024/BL4": dict(cell, dpub_acc_M1={"acc": 0.5}), "L1024/BL5": dict(cell, dpub_acc_M1={"acc": 0.96})}}
    r4, r5 = pd._row_pe(summ, "L1024", "BL4"), pd._row_pe(summ, "L1024", "BL5")
    assert r4.count("n/a") == 4 and r4.endswith("n/a") and "n/a" not in r5 and r5.endswith("0.960")


def test_phase_d_tables(tmp_path, monkeypatch):
    """phase_d.py tables writes PHASE_D_TABLES_en.md; the table of the rejection-sampled variant is S7; every block starts
    with '**Table Sk' so that it can be located by its label."""
    pd = _load("phase_d", ROOT / "scripts" / "confirm" / "phase_d.py")
    cell = {"W-img": {"P_E": 0.3, "P_E_ci95": [0.29, 0.31], "TPR@5": 0.2}, "W-A0": {"P_E": 0.4}, "W-pay": {"P_E": 0.5, "TPR@5": 0.05},
            "W-comb": {"P_E": 0.3}, "dTPR_pay_minus_img": {"dTPR": -0.15, "ci95": [-0.2, -0.1]}, "dpub_acc_M1": {"acc": 0.5}}
    S = {"reps": 3, "alpha_by_cell": {"0.2": {"PQ-PKS": 0.2}}, "E2E3": {"0.2/PQ-PKS": cell},
         "TOST": {"0.2/PQ-PKS-vs-REF-U/W-img": {"diff": 0.0, "ci90": [-0.003, 0.003], "verdict": "equivalent"}}}
    root = tmp_path
    for d, names in (("e7", ["summary.json"]), ("ablate", ["summary.json"]), ("ablate2", ["summary.json"]), ("ra", ["summary.json"]),
                     ("jpeg", ["summary.json", "summary_gfr.json"]), ("jpeg95", ["summary.json", "summary_gfr.json"]), ("confirm_r", ["summary.json"])):
        (root / "results" / d).mkdir(parents=True)
        for n in names:
            (root / "results" / d / n).write_text(json.dumps(S), encoding="utf-8")
    monkeypatch.setattr(pd, "ROOT", root)
    pd.tables(None)
    en = (root / "results" / "phase_d" / "PHASE_D_TABLES_en.md").read_text(encoding="utf-8")
    assert not (root / "results" / "phase_d" / "PHASE_D_TABLES.md").exists()
    heads = [l.split("[")[0].strip() for l in en.splitlines() if l.startswith("**Table S")]
    assert heads == ["**Table S4", "**Table S2", "**Table S2 (continued)", "**Table S5", "**Table S3", "**Table S3 (continued)",
                     "**Table S3 (continued)", "**Table S3 (continued)", "**Table S7"]
    row = "| 0.2 | PQ-PKS | 0.2000 | 0.300 [0.290, 0.310] | 0.400 | 0.500 | 0.300 | 0.200 | 0.050 | -0.150 [-0.200, -0.100] | 0.500 |"
    assert en.count(row) == 9 and "3 repeated splits; TOST δ = 0.02." in en
