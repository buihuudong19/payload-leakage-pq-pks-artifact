"""run_all (scripts/orchestration): the step table is consistent with the code and the configuration; GPU process rule."""
import importlib.util
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ORCH = ROOT / "scripts" / "orchestration"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ORCH / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


STEPS = _load("plan_table").parse((ORCH / "pipeline.ps1").read_text(encoding="utf-8-sig"))


def test_steps_are_consistent():
    ids = [s["id"] for s in STEPS]
    assert len(ids) == len(set(ids)) == 78          # lanes W1-W11
    assert {s["lane"] for s in STEPS} == {"W1", "W2", "W3", "W4", "W5", "W6", "W7", "W8", "W9", "W10", "W11"}
    pos = {s["id"]: k for k, s in enumerate(STEPS)}
    for s in STEPS:
        assert (ROOT / s["args"][0].replace("\\", "/")).exists(), s["args"][0]   # the called script exists
        for d in s["deps"]:
            if d.startswith("file:"):                                   # file dependency (3e, X5): not a step
                continue
            assert d.lstrip("~").split(":")[-1] in pos, (s["id"], d)    # the dependency names an existing step
    cfg = json.loads((ROOT / "configs" / "srnet_runs_v2.json").read_text(encoding="utf-8"))
    runs = [s["args"][s["args"].index("--run") + 1] for s in STEPS if "--run" in s["args"]]
    assert set(runs) <= set(cfg["runs"]) and set(cfg["runs"]) <= set(runs)  # every registered SRNet run has a step
    by = {s["id"]: s for s in STEPS}
    assert "3f" in by["C7"]["deps"]                                  # C7 reads g4_summary.json of 3f (lane W5)
    assert by["3f"]["check"] and by["2"]["check"] and by["G7a"]["check"] and by["G7b"]["check"]
    assert by["G3c"]["when"] and by["G7b"]["when"]
    assert "--size" in by["E4"]["args"] and by["E4"]["args"][by["E4"]["args"].index("--size") + 1] == "512"
    assert "--no-srnet" in by["C8"]["args"] and "decisions_srnet.json" in " ".join(by["G13"]["args"])
    assert set(by["C9"]["deps"]) == {"~E4", "~X5", "~C7", "~C8"}        # C9 does not wait for G13
    assert by["D1"]["deps"] == [] and by["D1"]["check"] and set(by["D4"]["deps"]) == {"~C7", "~C9"}   # stego generation runs alongside C7; the analysis waits for C9
    assert [s["id"] for s in STEPS if s["lane"] == "W8"] == [f"D{k}" for k in range(1, 12)]
    assert "--plan" in by["D2"]["args"] and by["D2"]["args"][by["D2"]["args"].index("--plan") + 1] == "e7"
    assert "--cluster" in by["D4"]["args"] and "camera" in by["D4"]["args"]
    assert by["D12"]["deps"] == ["~D9"] and by["D14"]["deps"] == ["~C9"] and by["D12"]["check"] and [s["id"] for s in STEPS if s["lane"] == "W9"] == [f"D{k}" for k in range(12, 20)]
    assert "--wimg" in by["D17"]["args"] and "gfr" in by["D18"]["args"] and "summary_gfr.json" in by["D18"]["args"]
    # C10 (E5 at the confirmatory alpha*) and R0 (confirmatory table) precede R1 in W1; lane W10 is manual and not opened by run_all
    w1 = [s["id"] for s in STEPS if s["lane"] == "W1"]
    assert w1.index("C10") < w1.index("R0") < w1.index("R1") and by["C10"]["deps"] == ["~C8"] and "~C9" in by["R0"]["deps"]
    text = (ORCH / "pipeline.ps1").read_text(encoding="utf-8-sig")
    assert "W10" not in text.split("$ManualLanes")[0].split("$LaneTitle = ")[1]      # W10 is not in $LaneTitle (run_all)
    assert [s["id"] for s in STEPS if s["lane"] == "W10"] == [f"PE{k}" for k in range(1, 15)]
    assert "summary_r10.json" in by["PE1"]["args"] and "10" in by["PE1"]["args"] and by["PE2"]["deps"] == ["PE1"]
    assert by["PE5"]["args"][by["PE5"]["args"].index("--plan") + 1] == "confirm_r" and "REF-U-1252" in by["PE7"]["args"]
    assert "--qf" in by["PE9"]["args"] and by["PE9"]["args"][by["PE9"]["args"].index("--qf") + 1] == "95"
    assert (ROOT / "run_all.ps1").read_text(encoding="utf-8-sig").count("$LaneTitle.Keys") == 1
    # Manual lane W11: PF1 (10 splits, bootstrap 10^4, M = 4) waits for PE1; PF2 promote; PF3 rules; PF4 E5 waits for G13;
    # PF5 tables S2-S6; PF6 RESULTS_TABLES; PF7 collection; PF8 figures. PF1-PF3 must not wait for any step of W2 (they finish before G12/G13).
    assert "W11" not in text.split("$ManualLanes")[0].split("$LaneTitle = ")[1]
    assert [s["id"] for s in STEPS if s["lane"] == "W11"] == [f"PF{k}" for k in range(1, 9)]
    assert by["PF1"]["deps"] == ["PE1"] and "summary_r10.json" in by["PF1"]["args"] and "10" in by["PF1"]["args"]
    assert by["PF2"]["deps"] == ["PF1", "~PE2"] and by["PF3"]["deps"] == ["PF2"] and "--no-srnet" in by["PF3"]["args"]
    assert set(by["PF4"]["deps"]) == {"PF3", "G13"} and "decisions_srnet.json" in " ".join(by["PF4"]["args"])
    assert all(not d.lstrip("~").startswith("G") for k in ("PF1", "PF2", "PF3") for d in by[k]["deps"])
    assert by["PF6"]["deps"][0] == "PF5" and by["PF7"]["deps"][0] == "PF6" and by["PF8"]["deps"] == ["PF7"]
    assert "W11" in (ORCH / "run_lane.ps1").read_text(encoding="utf-8-sig")


def test_split_hash_is_the_registered_one():
    text = (ORCH / "pipeline.ps1").read_text(encoding="utf-8-sig")
    assert re.search(r"\$SplitSha256 = '8d38c3aab7a773758ecb755f208accdc39b5b2db87da21eab1db5554f1001b57'", text)


def test_gpu_decide_rule():
    gd = _load("gpu_decide")
    env = {"smi": {"memory_total": "8192 MiB"}}
    row = {"size": 256, "batch": 32, "amp": True, "status": "ok", "end2end_ms_iter": 200.0, "peak_vram_reserved_mb": 2500,
           "smi": {"utilization.gpu": {"mean": 70.0}}}
    d = gd.decide({"env": env, "rows": [row]})
    assert d["parallel"] and d["its"] == 5.0
    assert not gd.decide({"env": env, "rows": [row | {"smi": {"utilization.gpu": {"mean": 97.0}}}]})["parallel"]
    assert not gd.decide({"env": env, "rows": [row | {"peak_vram_reserved_mb": 4000}]})["parallel"]
    assert not gd.decide({"env": env, "rows": [row | {"status": "OOM"}]})["parallel"]


def test_ps1_files_have_bom_for_windows_powershell():
    # Windows PowerShell 5.1 reads a .ps1 without BOM in the ANSI code page, which corrupts non-ASCII characters
    for p in [ROOT / "run_all.ps1", *ORCH.glob("*.ps1")]:
        assert p.read_bytes().startswith(b"\xef\xbb\xbf"), p.name
