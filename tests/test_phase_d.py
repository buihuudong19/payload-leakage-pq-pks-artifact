"""E7/ablation/R-a plans of screen_run, the deterministic E7 split file, the cv5/e7 splits and the cluster bootstrap."""
import hashlib
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

E7_SPLIT_SHA = "cbc62449b63eb82ac129adc70f7387d8d76225dea122e453cdff97cfeb4f0188"      # registered in pipeline.ps1 ($E7SplitSha256)


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_plans_and_variants():
    sr = _load("screen_run", ROOT / "scripts" / "experiments" / "screen_run.py")
    assert set(sr.PLANS) == {"screen", "confirm", "e7", "ablate", "ra", "ablate2", "confirm_r"}
    assert sr.PLANS["e7"][0] == (0.2, 0.4) and sr.PLANS["e7"][1][0.4] == sr.SYSTEMS
    assert sr.PLANS["ra"][0] == ("L1024", "L4096") and sr.is_msglen("L1024") and not sr.is_msglen(0.2)
    n = 512 * 512
    assert sr.bucket_of("L1024", n, 65) == sr.bucket_of("L1024", n, 1536) == 1024 + 14     # R-a: the same bucket for every system
    assert sr.bucket_of(0.2, n, 1536) == sr.profile_for_alpha(0.2, n, 1536).buckets[0]
    assert abs(sr.alpha_of("L1024", n, 1536) - 8 * (1536 + 1038 + 16) / n) < 1e-12
    for name, (kp, km) in {"V1": (False, False), "V2": (True, False), "V3": (False, True), "PQ-PKS": (True, True)}.items():
        s, dk, ek = sr.make_system(name, sr.profile_of(0.2, n, 1536))
        assert (s.embedder.keyed_perm, s.embedder.keyed_matrix) == (kp, km), name
    s, _, _ = sr.make_system("SUNI", sr.profile_of(0.2, n, 1536))
    assert s.embedder.cost == "suniward"
    s, _, _ = sr.make_system("A8p", sr.profile_of(0.2, n, 1536))
    assert s.pp.layout == "A8p" and s.pp.alpha0 == 0.15
    s, _, _ = sr.make_system("PQ-PKS-R", sr.profile_of(0.2, n, 1252))
    assert s.crypto.kem_part_bytes == 1252 == sr.KEM_BYTES["PQ-PKS-R"]
    assert sr.make_system("REF-U-SUNI", sr.profile_of(0.2, n, 1536)).embedder.cost == "suniward"
    assert sr.KEM_BYTES["BL3"] == 65 and sr.KEM_BYTES["BL4"] == 72
    assert sr.PLAN_COVERS["e7"] == "alaska2-e7"


def test_e7_split_is_the_registered_one(tmp_path):
    pd = _load("phase_d", ROOT / "scripts" / "confirm" / "phase_d.py")
    pd.SPLIT_E7 = tmp_path / "e7_v1.json"
    pd.split_e7(None)
    assert hashlib.sha256(pd.SPLIT_E7.read_bytes()).hexdigest() == E7_SPLIT_SHA
    d = json.loads(pd.SPLIT_E7.read_text(encoding="utf-8"))
    assert d["n"] == 10000 and d["cameras"] == 47 and len(d["test_sets"]) == 3
    for te, va in zip(d["test_sets"], d["val_sets"]):
        assert len(te) == 2500 and len(va) == 1000 and not set(te) & set(va)
        cams = {d["camera"][k] for k in te}
        assert len(cams) >= 40                                   # stratification: most camera models appear in the test set
    assert (ROOT / "configs" / "splits" / "e7_v1.json").read_bytes() == pd.SPLIT_E7.read_bytes()


def test_rep_sets_cv5_and_e7_and_cluster_boot():
    ac = _load("analyze_confirm", ROOT / "scripts" / "confirm" / "analyze_confirm.py")
    ids = [f"boss_{i}" for i in range(100)] + [f"bows2_{i}" for i in range(100)]
    seen = []
    for r in range(5):
        tr, va, te = ac.rep_sets_cv5(ids, r, frozenset({3}))
        assert not (set(tr) & set(va)) and not (set(tr) & set(te)) and not (set(va) & set(te))
        assert 3 not in set(tr) | set(va) | set(te) and len(va) == max(10, (len(tr) + len(va)) // 5)
        seen += list(te)
    assert sorted(seen) == [i for i in range(200) if i != 3]          # each cover is tested exactly once
    split = {"test_sets": [["boss_0", "boss_1", "boss_9"]], "val_sets": [["boss_2"]], "camera": {f"boss_{i}": f"c{i % 2}" for i in range(10)}}
    tr, va, te = ac.rep_sets_e7(split, ids[:10], 0, frozenset({9}))
    assert list(te) == [0, 1] and list(va) == [2] and set(tr) == set(range(3, 9))
    ac.CLUSTERS = [np.array(["a", "a", "b", "b", "c"])]
    rng = np.random.default_rng(0)
    idx = ac._boot_idx(rng, 0, 5)
    assert set(idx) <= set(range(5)) and len(idx) in (2, 3, 4, 5, 6)   # 3 clusters resampled with replacement, each as a block
    for g in ("a", "b"):
        members = set(np.nonzero(ac.CLUSTERS[0] == g)[0])
        assert not (set(idx) & members) or members <= set(idx)
    ac.CLUSTERS = None
    assert ac._akey("L1024") == "L1024" and ac._akey("0.2") == 0.2


def test_a7_hybrid_kem_and_stc_tables(tmp_path):
    sr = _load("screen_run", ROOT / "scripts" / "experiments" / "screen_run.py")
    k = sr.HybridKem()
    sk, pk = k.generate()
    assert len(pk) == 1184 + 32
    ss, enc = k.encap(pk)
    assert len(enc) == 1088 + 32 and k.decap(enc, sk) == ss and len(ss) == 32
    c = sr.a7_crypto()
    part = c.encode(enc)
    assert len(part) == 1568 == sr.A7_KEM_BYTES and c.decode(part) == enc
    assert sr.PLANS["ablate2"][1][0.2] == sr.ABLATE2_SYSTEMS and sr.KEM_BYTES["REF-U-1568"] == 1568
    # A9: H_hat table for h = 8 (built), looked up by h; h = 10 is still the table of the profile
    from pqpks.stego import stc
    sr.install_stc_tables()
    if 8 in sr._STC_TABLES:
        m8 = stc.submatrix(b"seed", 8, 5)
        assert m8.shape == (5,) and all(int(v) < 2 ** 8 for v in m8)
        assert np.array_equal(stc.submatrix(b"seed", 10, 5), stc.submatrix(b"seed", 10, 5)) and all(int(v) < 2 ** 10 for v in stc.submatrix(b"seed", 10, 5))
    s, dk, ek = sr.make_system("A7", sr.profile_of(0.2, 512 * 512, sr.A7_KEM_BYTES))
    assert s.crypto.kem_part_bytes == 1568


def test_jpeg_domain_roundtrip():
    import pytest
    pytest.importorskip("jpeglib")
    jd = _load("jpeg_domain", ROOT / "scripts" / "experiments" / "jpeg_domain.py")
    rng = np.random.default_rng(3)
    base = np.cumsum(np.cumsum(rng.normal(0, 1, (512, 512)), 0), 1)
    base = (base - base.min()) / (base.max() - base.min()) * 160 + 40
    x = np.clip(base + rng.normal(0, 14, (512, 512)), 0, 255).astype(np.uint8)
    jc = jd.JpegCover.from_gray(x)
    assert jc.Y.shape == (64, 64, 8, 8) and jc.n_u > 10000
    emb = jd.JpegTwoStageSTC(jd.DEFAULT_PP)
    cm = jc.juniward_cost()
    v = jc.values()
    kem, body = rng.integers(0, 256, 1536, dtype=np.uint8).tobytes(), rng.integers(0, 256, 300, dtype=np.uint8).tobytes()
    y = emb.embed(v, kem, body, b"k" * 32, b"h" * 32, cm)
    assert y.shape == v.shape and (y != v).any() and not ((v == 1) & (y == 0)).any() and not ((v == -1) & (y == 0)).any()
    assert np.abs(y - v).max() == 1 and not ((v != 0) & (y == 0)).any()
    js = jc.with_values(y)                                   # U unchanged (asserted inside)
    assert emb.extract_kem(js.values(), 1536, 300) == kem
    idx1 = emb.stage2_index(v.shape, 1536, b"k" * 32, 300)
    assert emb.extract_body(js.values(), idx1, 300, b"h" * 32) == body
    assert jc.dctr().size == 8000


def test_cell_checkpoint_roundtrip(tmp_path):
    ac = _load("analyze_confirm", ROOT / "scripts" / "confirm" / "analyze_confirm.py")
    ac.BASE = tmp_path
    rng = np.random.default_rng(0)
    per = [{w: {"c": rng.normal(size=7), "s": rng.normal(size=7), "t5": 0.1, "t1": 0.2, "tpe": 0.3, "val_pe": 0.4}
            for w in ("W-img", "W-pay")} for _ in range(2)]
    ac.save_cell(0.2, "PQ-PKS", per, "tag1")
    back = ac.load_cell(0.2, "PQ-PKS", "tag1", 2)
    assert back is not None and len(back) == 2 and set(back[0]) == {"W-img", "W-pay"}
    for r in range(2):
        for w in ("W-img", "W-pay"):
            assert np.array_equal(back[r][w]["c"], per[r][w]["c"]) and np.array_equal(back[r][w]["s"], per[r][w]["s"])
            assert (back[r][w]["t5"], back[r][w]["t1"], back[r][w]["tpe"], back[r][w]["val_pe"]) == (0.1, 0.2, 0.3, 0.4)
    assert ac.load_cell(0.2, "PQ-PKS", "tag2", 2) is None and ac.load_cell(0.2, "PQ-PKS", "tag1", 3) is None
    assert ac.load_cell(0.4, "PQ-PKS", "tag1", 2) is None
