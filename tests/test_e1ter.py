"""E1-ter (scripts/confirm/e1ter.py, scripts/data/make_alaska2_e1ter_subset.py): image selection and the registered rule."""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "confirm"))
sys.path.insert(0, str(ROOT / "scripts" / "experiments"))
sys.path.insert(0, str(ROOT / "src"))


def load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_rule_two_sided_and_verdict():
    m = load("e1ter", "scripts/confirm/e1ter.py")
    assert m.cell_pass(0.0, 0.005) and m.cell_pass(-0.01, 0.005)
    assert not m.cell_pass(0.013, 0.005) and not m.cell_pass(-0.013, 0.005)      # 0.013 + 0.0082 > 0.02 on both sides
    draws = np.r_[np.linspace(-0.01, 0.01, 99)]
    assert m.null_p(0.0, draws) == 1.0 and m.null_p(0.02, draws) == 1 / 100
    assert np.allclose(m.holm(np.array([0.01, 0.04, 0.03])), [0.03, 0.06, 0.06])
    assert m.verdict([True] * 12, np.ones(12)) == "equivalent within 0.02"
    assert m.verdict([True] * 11 + [False], np.ones(12)) == "inconclusive"
    assert m.verdict([True] * 12, np.r_[np.ones(11), 0.01]) == "distinguisher detected"
    assert 0.9 < m.power_normal(0.0045) < 1 and m.power_normal(0.02) == 0.0


def test_selection_excludes_used_and_calibration(tmp_path):
    s = load("sel", "scripts/data/make_alaska2_e1ter_subset.py")
    src = tmp_path / "dsc.txt"
    src.write_text("#hdr\n 0003 :: 00001 --> 00003     ./CamA\n 0002 :: 00004 --> 00005     ./CamA_FLATS\n 0002 :: 00006 --> 00007     ./CamB\n")
    used = tmp_path / "used.txt"
    used.write_text("00002 CamA\n00007 CamB\n")
    assert s.select(src, [used]) == [(1, "CamA"), (3, "CamA"), (6, "CamB")]


def test_registered_list_matches_plan():
    import hashlib
    plan = json.loads((ROOT / "registrations" / "e1ter_plan.json").read_text(encoding="utf-8"))
    ids = ROOT / plan["ids_file"]
    assert hashlib.sha256(ids.read_bytes()).hexdigest() == plan["ids_sha256"]
    rows = ids.read_text().splitlines()
    assert len(rows) == plan["n"] == 38533
    e1bis = {ln.split()[0] for ln in (ROOT / "data/external/alaska2/e1bis_ids.txt").read_text().splitlines() if ln.strip()}
    assert not {r.split()[0] for r in rows} & e1bis
