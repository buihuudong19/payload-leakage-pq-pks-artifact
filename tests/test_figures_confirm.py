"""make_figures_confirm.py runs on a real summary.json and decisions.json of the confirmatory pipeline (reduced run: 50 covers,
2 splits, produced with screen_run --plan confirm + analyze_confirm + apply_rules). The numbers have no meaning."""
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
pytest.importorskip("matplotlib")


def test_confirm_figures_and_tables(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("mfc", ROOT / "scripts" / "experiments" / "make_figures_confirm.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(sys, "argv", ["x", "--confirm", str(ROOT / "tests" / "fixtures" / "confirm_smoke"), "--out", str(tmp_path)])
    mod.main()
    names = {p.name for p in tmp_path.iterdir()}
    for f in ("fig6_dpub_vs_M_confirm_a04.png", "fig7_tpr_wardens_confirm_a04.png", "fig8_pe_alpha_tost_confirm.png",
              "table10_e2e3_confirm.md", "table11_tost_confirm.md", "table_alpha_star_confirm.md"):
        assert f in names
    t = (tmp_path / "table_alpha_star_confirm.md").read_text(encoding="utf-8")
    assert "main (registered)" in t and "sensitivity analysis" in t
