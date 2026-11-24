"""Cached receiver (scripts/experiments/receiver_fast.py): bit-identical syndromes and identical receiver outputs."""
import copy
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts" / "experiments")]

from pqpks import systems  # noqa: E402
from pqpks.stego import stc  # noqa: E402
import receiver_fast as rf  # noqa: E402
from e6_load import synthetic_cover  # noqa: E402


def test_shift_register_syndrome_equals_reference():
    rng = np.random.default_rng(0)
    for _ in range(200):
        n = int(rng.integers(20, 4000))
        m = int(rng.integers(1, n))
        seed = rng.bytes(32)
        y = rng.integers(0, 2, n).astype(np.uint8)
        assert np.array_equal(stc.extract(y, m, seed, 10), rf.syndrome(y, m, seed, 10))


def test_cached_receiver_gives_identical_outputs():
    rng = np.random.default_rng(1)
    ref, fast = systems.make("PQ-PKS"), rf.fast_system("PQ-PKS")
    dk, ek = ref.keygen()
    st = systems.ReceiverState()
    sessions = []
    for _ in range(2):
        s = ref.start_session(ek, length=4)
        sessions.append(s)
        assert ref.receive(dk, ref.send(s, b"o", synthetic_cover(rng)), st).msg == b"o"
    stf = copy.deepcopy(st)
    images = [synthetic_cover(rng), synthetic_cover(rng), ref.send(sessions[1], b"later image", synthetic_cover(rng))]
    for x in images:
        a, b = ref.receive(dk, x, st), fast.receive(dk, x, stf)
        assert (a.msg, a.status, a.trials) == (b.msg, b.status, b.trials)
    assert a.msg == b"later image"
    assert fast.embedder.hits > 0
