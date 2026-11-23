"""The A5b rule and the signature selection rule give the registered decisions for assumed values of alpha*."""
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("apply_rules", ROOT / "scripts" / "confirm" / "apply_rules.py")
rules = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rules)
LADDER = json.loads((ROOT / "results" / "e5_budget" / "e5_ladder.json").read_text(encoding="utf-8"))
GRID = (0.0552, 0.0787, 0.1, 0.1099, 0.2, 0.3, 0.4)


def summary(alpha_star, pe=None, diff=None, ci=None):
    pe = pe or {g: 0.47 for g in GRID}
    diff = diff or {g: 0.0 for g in GRID}
    return {"alpha_star": {"alpha_star": alpha_star, "alpha_star_i": alpha_star, "alpha_star_ii": None,
                           "alpha_star_i_ci95": ci, "P_E_W-img_REF-U": {str(k): v for k, v in pe.items()},
                           "cond_ii_TPR_diff_W-comb": {str(k): v for k, v in diff.items()}}}


def test_alpha_min_values_from_ladder():
    t = rules.alpha_min_table(LADDER)
    assert t["image1"]["ML-DSA-65"] == pytest.approx(0.1724)
    assert t["image2"]["ML-DSA-65"] == pytest.approx(0.1099)
    assert t["image2"]["ML-DSA-44"] == pytest.approx(0.0787)
    assert t["image2"]["FN-DSA-512"] == pytest.approx(0.0552)


@pytest.mark.parametrize("a, place, rec", [
    (0.20, "image1", "ML-DSA-65"),
    (0.1724, "image1", "ML-DSA-65"),
    (0.15, "image2", "ML-DSA-65"),
    (0.1099, "image2", "ML-DSA-65"),
    (0.09, "image2", "ML-DSA-44"),
    (0.0787, "image2", "ML-DSA-44"),
    (0.06, "image2", None),
    (None, "image2", None),
])
def test_registered_tiers(a, place, rec):
    pe = {g: (0.47 if a is not None and g <= a else 0.40) for g in GRID}
    d = rules.decide(summary(a, pe=pe), LADDER)
    assert d["rule_A5b"]["sig_placement"] == place
    assert d["rule_signature"]["recommended"] == rec


def test_direct_check_overrides_to_the_lower_rate():
    # the interpolated alpha* = 0.12 selects ML-DSA-65, but at 0.1099 condition (ii) fails: ML-DSA-44 must be chosen
    pe = {g: 0.47 for g in GRID}
    diff = {g: (0.03 if g == 0.1099 else 0.0) for g in GRID}
    d = rules.decide(summary(0.12, pe=pe, diff=diff), LADDER)
    assert d["rule_signature"]["by_alpha_star"] == "ML-DSA-65"
    assert d["rule_signature"]["by_direct_check"] == "ML-DSA-44"
    assert d["rule_signature"]["recommended"] == "ML-DSA-44"
    assert d["rule_signature"]["notes"]


@pytest.mark.parametrize("a, below", [(0.06, False), (0.05, True), (None, True)])
def test_minimal_first_image_flag(a, below):
    pe = {g: (0.47 if a is not None and g <= a else 0.40) for g in GRID}
    d = rules.decide(summary(a, pe=pe), LADDER)
    assert d["minimal_first_image"]["alpha_star_below"] is below
    assert d["minimal_first_image"]["direct_check"]["holds"] is (not below)
    assert ("smallest session-start image" in d["statement"]) is below


def test_thresholds_inside_ci_are_listed():
    d = rules.decide(summary(0.10, ci=[0.07, 0.12]), LADDER)
    names = " ".join(d["thresholds_inside_ci95_of_alpha_star_i"])
    assert "ML-DSA-65/image2" in names and "ML-DSA-44/image2" in names and "A5b" not in names


def _curve(cross):
    """P_E(W-img, REF-U) decreases linearly and crosses 0.45 at `cross`."""
    return {g: 0.5 - (0.05 / cross) * g for g in GRID}


@pytest.mark.parametrize("cross, srnet_pe02, changes", [
    (0.15, 0.30, True),      # alpha*(SRM) between 0.1099 and 0.2: the lower P_E(SRNet) at 0.2 pulls alpha* down
    (0.08, 0.30, False),     # alpha*(SRM) < 0.1099: the crossing does not use 0.2, SRNet does not change alpha*
    (0.15, 0.49, False),     # SRNet weaker than SRM at 0.2: the SRM P_E is kept
])
def test_srnet_merged_as_strongest_warden(cross, srnet_pe02, changes):
    pe = _curve(cross)
    xs = sorted(GRID)
    a_srm = rules.interp_crossing(xs, [pe[g] for g in xs], 0.45)
    s = summary(a_srm, pe=pe)
    s["alpha_star"]["alpha_star_ii"] = 0.4
    srnet = {"alphas": {"0.2": {"cond_i_SRNet": {"P_E_REF-U": srnet_pe02}},
                        "0.4": {"cond_i_SRNet": {"P_E_REF-U": 0.0}}}}
    d = rules.decide(s, LADDER, srnet)
    assert d["srnet"]["included"] and d["srnet"]["changed"] is changes
    if changes:
        assert d["alpha_star"] < a_srm
    else:
        assert d["alpha_star"] == pytest.approx(a_srm)
    assert rules.decide(s, LADDER)["srnet"] == {"included": False}


def test_interp_crossing_matches_stats():
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from pqpks.analysis import stats
    for ys in ([0.5, 0.47, 0.44, 0.3], [0.44, 0.4], [0.5, 0.49, 0.48], [0.46, 0.45, 0.449]):
        xs = [0.05, 0.1, 0.2, 0.3][:len(ys)]
        assert rules.interp_crossing(xs, ys, 0.45) == stats.interp_crossing(xs, ys, 0.45)


def test_sensitivity_threshold_with_g4_offset(tmp_path):
    """Threshold (i) = 0.45 + the mean G4 deviation. The P_E curve crosses 0.45 at 0.15 but crosses 0.4584 at a smaller alpha,
    so the sensitivity alpha* must be smaller than the main alpha* and the rule may change level; the main conclusion (threshold 0.45) is unchanged."""
    pe = {0.0552: 0.49, 0.0787: 0.48, 0.1: 0.47, 0.1099: 0.455, 0.2: 0.44, 0.3: 0.40, 0.4: 0.35}
    a_main = rules.interp_crossing(list(pe), list(pe.values()), 0.45)              # ~ 0.14 (between 0.1099 and 0.2)
    s = summary(a_main, pe=pe)
    s["alpha_star"]["alpha_star_ii"] = 0.4
    main = rules.decide(s, LADDER)
    g4 = {"results": [{"P_E_mean": 0.3072, "reference": {"P_E": 0.2996}}, {"P_E_mean": 0.4416, "reference": {"P_E": 0.4340}},
                      {"P_E_mean": 0.2664, "reference": {"P_E": 0.2574}}, {"P_E_mean": 0.4115, "reference": {"P_E": 0.4021}},
                      {"P_E_mean": 0.5, "reference": None}]}
    p = tmp_path / "g4_summary.json"
    p.write_text(json.dumps(g4))
    off = rules.g4_offset(g4)
    assert off == pytest.approx((0.0076 + 0.0076 + 0.0090 + 0.0094) / 4)
    sens = rules.sensitivity(s, LADDER, None, p)
    assert sens["threshold"] == pytest.approx(0.45 + off)
    assert sens["alpha_star"] < main["alpha_star"]
    assert sens["alpha_star"] == pytest.approx(rules.interp_crossing(list(pe), list(pe.values()), 0.45 + off))
    # 0.1099 has P_E 0.455 < 0.4584: the direct check at the sensitivity threshold makes ML-DSA-65 fail
    assert sens["recommended"] == "ML-DSA-44" and main["rule_signature"]["recommended"] == "ML-DSA-65"
    with pytest.raises(ValueError):
        rules.g4_offset({"results": [{"P_E_mean": 0.5, "reference": None}]})


def test_level_default_keeps_main_decision():
    pe = {g: (0.47 if g <= 0.15 else 0.40) for g in GRID}
    a = rules.decide(summary(0.15, pe=pe), LADDER)
    b = rules.decide(summary(0.15, pe=pe), LADDER, None, rules.COND_I)
    assert a == b and a["srnet"] == {"included": False}
