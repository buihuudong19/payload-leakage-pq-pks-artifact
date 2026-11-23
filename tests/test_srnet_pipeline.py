"""The parts of the SRNet pipeline that run without a GPU: data order, paired batches, learning-rate schedule, evaluation."""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "confirm"))
CFG = json.loads((ROOT / "configs" / "srnet_runs_v2.json").read_text(encoding="utf-8"))
REF = json.loads((ROOT / "configs" / "g4_srnet_reference.json").read_text(encoding="utf-8"))


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_registered_runs_are_consistent():
    runs = CFG["runs"]
    for n in CFG["sequences"]["confirm"]:
        r = runs[n]
        assert r["stego"] in CFG["datasets"][r["data"]]["sets"]
        assert r["select_last"] <= sum(p[0] for p in r["phases"])
        seq = CFG["sequences"]["confirm"]
        assert r["init"] == "@g4_selected" or seq.index(r["init"]) < seq.index(n)
    full, short = runs["g4_hill_0.4"], runs["g4_hill_0.4_short"]
    assert sum(p[0] for p in full["phases"]) == 500_000 and full["phases"][0] == [400000, 0.001]   # [BCF19]: 400k + 100k
    # short branch = the first 200k iterations of the full run (same lr), then 50k iterations at lr 1e-4
    assert short["init_state"] == {"run": "g4_hill_0.4", "iter": full["branch_at"]} == {"run": "g4_hill_0.4", "iter": 200000}
    assert short["phases"] == [[200000, 0.001], [50000, 0.0001]] and short["select_last"] == 50000
    assert CFG["g4_protocol"]["order"] == ["g4_hill_0.4_short", "g4_hill_0.4"]
    assert CFG["datasets"]["g4"]["sets"] == ["cover", "hill_0.4"]
    for n in ("confirm_PQ-PKS_0.4", "confirm_REF-U_0.4", "confirm_PQ-PKS_0.2", "confirm_REF-U_0.2"):
        other = n.replace("PQ-PKS", "X").replace("REF-U", "PQ-PKS").replace("X", "REF-U")
        a, b = runs[n], runs[other]                         # PQ-PKS and REF-U: same schedule, same cell, paired with each other
        assert a["phases"] == b["phases"] and a["train_crop"] == b["train_crop"] == 256
        assert a["pair_with"] == b["stego"] and b["pair_with"] == a["stego"]
    assert any(r["method"] == "hill" and r["alpha"] == 0.4 for r in REF["references"])


def test_g4_protocol_order():
    ev = _load("srnet_evaluate", "scripts/confirm/srnet_evaluate.py")
    order = CFG["g4_protocol"]["order"]
    ok, bad = {"pass": True}, {"pass": False}
    assert ev.g4_decide({}, order)["next"] == "wait"
    assert ev.g4_decide({order[0]: ok}, order) == {"G4_SRNet_pass": True, "selected_run": order[0], "next": None}
    assert ev.g4_decide({order[0]: bad}, order)["next"] == "fallback_full"
    assert ev.g4_decide({order[0]: bad, order[1]: ok}, order)["selected_run"] == order[1]
    assert ev.g4_decide({order[0]: bad, order[1]: bad}, order)["G4_SRNet_pass"] is False
    # if the short branch passes, the full run (if any) is not considered again
    assert ev.g4_decide({order[0]: ok, order[1]: bad}, order)["selected_run"] == order[0]


def test_prepare_order_and_set_names():
    prep = _load("srnet_prepare", "scripts/confirm/srnet_prepare.py")
    sp = {"boss": {"train": [3, 1], "val": [5], "test": [9, 7]}, "bows2": {"train": [0, 2]}}
    items = prep.items_from_split(sp)
    assert [(s, i) for s, i, _ in items] == [("boss", 3), ("boss", 1), ("boss", 5), ("boss", 9), ("boss", 7),
                                             ("bows2", 0), ("bows2", 2)]
    assert prep.parse_set("hill_0.4") == ("hill", 0.4)
    assert prep.parse_set("a0.2_PQ-PKS") == ("PQ-PKS", 0.2)
    assert prep.parse_set("a0.4_REF-U") == ("REF-U", 0.4)


@pytest.fixture(scope="module")
def train_mod():
    pytest.importorskip("torch")
    return _load("srnet_train", "scripts/gpu/srnet_train.py")


class _Data:
    def __init__(self, n=40, size=32, seed=0):
        rng = np.random.default_rng(seed)
        self.cover = rng.integers(0, 256, (n, size, size), dtype=np.uint8)
        self.stego = self.cover ^ np.uint8(1)               # flip the LSB of every pixel: for aligned pairs x_stego ^ x_cover == 1
        self.train = np.arange(0, 30)


@pytest.mark.parametrize("crop", [None, 16])
def test_batches_are_aligned_and_reproducible(train_mod, crop):
    d = _Data()
    b1, b2 = train_mod.Batches(d, 4, 1, crop), train_mod.Batches(d, 4, 1, crop)
    for t in (0, 3, 9, 7):                                  # arbitrary call order (as when resuming after an interruption)
        x1, y1 = b1.get(t)
        x2, y2 = b2.get(t)
        assert np.array_equal(x1, x2) and np.array_equal(y1, y2)
        assert np.all((x1[0::2] ^ x1[1::2]) == 1) and list(y1[:4]) == [0, 1, 0, 1]
        assert x1.shape[-1] == (crop or 32)
    # one epoch uses each training image at most once
    seen = []
    for t in range(b1.per_epoch):
        b1.get(t)
        ep, j = divmod(t, b1.per_epoch)
        seen += list(d.train[b1._perm[1][j * 4:(j + 1) * 4]])
    assert len(seen) == len(set(seen)) == b1.per_epoch * 4


def test_schedule_and_weight_decay_groups(train_mod):
    import torch
    phases = [(400, 1e-3), (100, 1e-4)]
    assert train_mod.lr_at(phases, 0) == 1e-3 and train_mod.lr_at(phases, 399) == 1e-3
    assert train_mod.lr_at(phases, 400) == 1e-4 and train_mod.lr_at(phases, 499) == 1e-4
    from pqpks.steganalysis.srnet import SRNet
    m = SRNet()
    g = train_mod.param_groups(m, 2e-4)
    convs = [x for x in m.modules() if isinstance(x, torch.nn.Conv2d)]
    assert g[0]["weight_decay"] == 2e-4 and len(g[0]["params"]) == len(convs)
    assert g[1]["weight_decay"] == 0.0
    assert sum(p.numel() for gr in g for p in gr["params"]) == sum(p.numel() for p in m.parameters())


def test_evaluate_metrics_and_paired_tests():
    ev = _load("srnet_evaluate", "scripts/confirm/srnet_evaluate.py")
    rng = np.random.default_rng(3)
    n = 400

    def z(shift):
        return {"val_cover": rng.normal(0, 1, n), "val_stego": rng.normal(shift, 1, n), "test_ids": np.arange(n),
                "test_cover": rng.normal(0, 1, n), "test_stego": rng.normal(shift, 1, n)}

    far = z(8.0)
    m = ev.metrics(far)
    assert m["P_E_min_test"] < 0.01 and m["AUC"] > 0.99 and m["TPR@5"] > 0.99
    near = z(0.0)
    assert ev.metrics(near)["P_E_min_test"] > 0.4
    same = ev.paired(far, {k: v.copy() for k, v in far.items()})
    assert same["diff_P_E"] == 0.0 and same["TOST"]["verdict"] == "equivalent"
    assert same["McNemar"]["PQ_right_REF_wrong"] == same["McNemar"]["REF_right_PQ_wrong"] == 0
    other = dict(far)
    other["test_ids"] = np.arange(1, n + 1)
    with pytest.raises(SystemExit):
        ev.paired(far, other)
