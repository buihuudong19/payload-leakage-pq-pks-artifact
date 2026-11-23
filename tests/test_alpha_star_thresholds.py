"""Threshold sensitivity of alpha* (post-hoc analysis): same interpolation as the registered rule, SRNet folded in."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from pqpks.analysis import stats

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("ast_thr", ROOT / "scripts" / "analysis" / "alpha_star_thresholds.py")
thr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(thr)

GRID = (0.0552, 0.0787, 0.1, 0.1099, 0.2, 0.3, 0.4)
PE = (0.45227, 0.43257, 0.40922, 0.40328, 0.32531, 0.2538, 0.19596)   # REF-U, SRM, confirmatory round


def summary(pe=PE):
    return {"alpha_star": {"P_E_W-img_REF-U": {str(a): v for a, v in zip(GRID, pe)},
                           "alpha_star_i": stats.interp_crossing(GRID, pe, 0.45), "alpha_star_i_ci95": [0.0557, 0.0600],
                           "alpha_star_i_boot_below_grid": 0.0172},
            "E2E3": {"0.0552/REF-U": {"W-img": {"P_E": pe[0], "P_E_ci95": [pe[0] - 0.002, pe[0] + 0.002]}}}}


def test_same_interpolation_as_registered_rule():
    rng = np.random.default_rng(3)
    for _ in range(200):
        ys = np.sort(rng.uniform(0.1, 0.5, len(GRID)))[::-1]
        for lvl in (0.3, 0.4, 0.45):
            assert thr.interp_crossing(GRID, list(ys), lvl) == stats.interp_crossing(GRID, ys, lvl)


def test_registered_threshold_reproduces_alpha_star():
    res = thr.analyse(summary(), None)
    row = next(r for r in res["rows"] if r["threshold"] == 0.45)
    assert row["alpha_star_i"] == pytest.approx(0.05791, abs=1e-5)
    assert row["budget_bytes"] == 1897
    assert row["role"] == "registered"


def test_sensitivity_threshold_leaves_no_first_image():
    dec = {"sensitivity_G4_offset": {"threshold": 0.45838333}}
    res = thr.analyse(summary(), None, dec)
    row = res["rows"][-1]
    assert row["alpha_star_i"] is None and not row["first_image_0.0552_meets"]
    assert row["role"].startswith("registered sensitivity")


def test_srnet_lowers_the_curve_only_where_it_was_run():
    srnet = {"srnet": {"SRNet_P_E_REF-U": {"0.2": 0.2367, "0.4": 0.1272}}}
    xs, ys, used = thr.curve(summary(), srnet)
    assert used == {"0.2": 0.2367, "0.4": 0.1272}
    assert ys[:4] == list(PE[:4]) and ys[5] == PE[5]
    lo = thr.analyse(summary(), srnet)["rows"][1]          # threshold 0.40: interpolation reaches the 0.2 cell
    hi = thr.analyse(summary(), None)["rows"][1]
    assert lo["alpha_star_i"] < hi["alpha_star_i"]
