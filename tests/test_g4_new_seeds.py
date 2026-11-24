"""Fresh-seed G4 (scripts/confirm/g4_new_seeds.py): the plan excludes the seeds of the registered G4, and the verdict follows the
registered tolerance."""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "confirm"))
spec = importlib.util.spec_from_file_location("g4_new_seeds", ROOT / "scripts" / "confirm" / "g4_new_seeds.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
PLAN = json.loads((ROOT / "registrations" / "g4_new_seeds_plan.json").read_text(encoding="utf-8"))


def test_plan_uses_fresh_seeds_and_registered_cells():
    assert not set(PLAN["seeds"]) & set(PLAN["seeds_already_used"])
    assert len(PLAN["seeds"]) == 10 and PLAN["tolerance"] == 0.02
    assert PLAN["methods"] == ["hill", "suniward"] and PLAN["alphas"] == [0.3, 0.1]


def test_verdict_and_pe():
    ref = {"P_E": 0.30}
    assert m.cell_verdict([0.31, 0.32], ref, 0.02) is True
    assert m.cell_verdict([0.33, 0.33], ref, 0.02) is False
    assert m.cell_verdict([0.3], None, 0.02) is None
    sc, ss = np.array([0.1, 0.9, 0.2, 0.3]), np.array([0.8, 0.7, 0.4, 0.6])
    assert m.pe(sc, ss) == 0.5 * (0.25 + 0.25)
