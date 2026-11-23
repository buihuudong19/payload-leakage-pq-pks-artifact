"""analyze_confirm.py: feature collection when a cover lacks one (alpha, system) pair, which removes that cover from every
split, and the accuracy of D_pub when M sessions are pooled (H2a, Fig. 6)."""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "confirm"))
sys.path.insert(0, str(ROOT / "scripts" / "experiments"))
pytest.importorskip("sealwatch")


@pytest.fixture(scope="module")
def ac():
    spec = importlib.util.spec_from_file_location("analyze_confirm", ROOT / "scripts" / "confirm" / "analyze_confirm.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write_raw(raw: Path, n=6, fail_at=2):
    rng = np.random.default_rng(0)
    raw.mkdir(parents=True)
    for i in range(n):
        rec = {"src": np.array("boss"), "id": np.array(i), "srm_cover": rng.normal(size=4).astype(np.float32),
               "a0.2_ext0_cover": rng.integers(0, 256, 8, dtype=np.uint8)}
        failed = []
        for name in ("PQ-PKS", "REF-U"):
            if i == fail_at and name == "REF-U":
                failed.append(f"a0.2_{name}")
                continue
            rec[f"a0.2_{name}_srm"] = rng.normal(size=4).astype(np.float32)
            rec[f"a0.2_{name}_ext0"] = rng.integers(0, 256, 8, dtype=np.uint8)
        rec["stc_fail"] = np.array(len(failed))
        rec["failed_combos"] = np.array(failed, dtype="U32")
        np.savez_compressed(raw / f"boss_{i}.npz", **rec)
    (raw.parent / "run_meta.json").write_text(json.dumps({"systems_at": {"0.2": ["PQ-PKS", "REF-U"]}}))


def test_consolidate_marks_missing_and_rep_sets_exclude(ac, tmp_path, monkeypatch):
    base = tmp_path / "confirm_v1"
    _write_raw(base / "raw")
    monkeypatch.setattr(ac, "BASE", base)
    monkeypatch.setattr(ac, "FEAT", base / "feat")
    ac.consolidate(None)
    avail = np.load(base / "feat" / "avail.npy")
    info = json.loads((base / "feat" / "avail.json").read_text())
    assert info["n_covers_missing_any"] == 1 and info["missing_by_combo"]["a0.2_REF-U"] == 1
    ids = json.loads((base / "feat" / "ids.json").read_text())
    bad = ids.index("boss_2")
    assert not avail[bad].all() and avail[[k for k in range(len(ids)) if k != bad]].all()
    assert np.isnan(np.load(base / "feat" / "a0.2_REF-U_srm.npy")[bad]).all()
    split = {"srm_repeated_test_sets": [[0, 1, 2]], "boss": {"pilot": [], "test": [0, 1, 2], "val": [3, 4], "train": [5]},
             "bows2": {"train": []}}
    tr, va, te = ac.rep_sets(split, ids, 0, frozenset({bad}))
    allidx = set(tr) | set(va) | set(te)
    assert bad not in allidx and len(te) == 2 and te.dtype == np.int64
    ac.rep_sets(split, ids[:-1], 0)                         # cover without a file (.err): no KeyError


def test_dpub_pooling_and_delta_syn(ac):
    rng = np.random.default_rng(1)
    n = 600
    per = [{"W-pay": {"c": rng.normal(0, 1, n), "s": rng.normal(0.6, 1, n)}} for _ in range(2)]
    rows = ac.dpub_rows(per, [n, n])
    accs = [rows[str(M)]["acc"] for M in (1, 2, 4, 8)]
    assert all(b > a for a, b in zip(accs, accs[1:]))           # pooling more sessions makes distinguishing easier
    for M in (1, 2, 3):
        lo, hi = rows[str(M)]["ci95"]
        assert lo <= rows[str(M)]["acc"] <= hi + 0.02
    assert "64" not in rows                                       # 600 // 64 < 10 groups: dropped
    u = np.random.default_rng(2).integers(0, 256, (2000, 1088), dtype=np.uint8)
    from screen_analyze import llr_bytes
    cover_llr = np.array([llr_bytes(x) for x in u])               # uniform "cover syndrome": advantage ~ 0
    ds = ac.delta_syn(cover_llr, 1088)
    assert all(abs(ds[m]["adv"]) < 0.05 and ds[m]["upper_abs"] < 0.08 for m in ("1", "2", "3"))
