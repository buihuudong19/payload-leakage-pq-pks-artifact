"""Build the public H_hat table for STC (h = 10, w = 1..128) by deterministic search.

For each w: try N_CAND candidates generated from the seed "cand-<w>-<j>", measure the embedding efficiency (uniform costs,
fixed random payload), keep the TOP_K best candidates in decreasing order.
The table is a PUBLIC parameter (part of pp); the key k_H only selects an index within TOP_K.
Output: src/pqpks/stego/stc_table_h10.json (committed; can be rebuilt deterministically).
"""
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pqpks.stego import stc  # noqa: E402

H, W_MAX, N_CAND, TOP_K, N_EVAL, M_MIN = 10, 128, 48, 8, 12000, 150   # W_MAX raised from 64 to 128


def h2(p):
    return -p * math.log2(p) - (1 - p) * math.log2(1 - p)


def main(h: int = H, out: Path | None = None, w_max: int = W_MAX):
    """h = 10: the public table of the profile (src/pqpks/stego/stc_table_h10.json). h = 8, 12 (ablation A9): the same procedure,
    the same seeds, the same candidates "cand-<w>-<j>"; written to scripts/experiments/stc_tables/ (src/ and the frozen profile untouched)."""
    rng = np.random.default_rng(20261028)
    table, report = {}, {}
    t0 = time.time()
    for w in range(1, w_max + 1):
        m = max(N_EVAL // w, M_MIN)          # w <= 64: unchanged from the earlier table (N_EVAL // w >= 187)
        x = rng.integers(0, 2, m * w)
        msg = rng.integers(0, 2, m)
        bound = 2.0 if w == 1 else (1 / w) / brentq(lambda q: h2(q) - 1 / w, 1e-12, 0.5)
        scored = []
        for j in range(N_CAND):
            cols = stc.random_submatrix(f"cand-{w}-{j}".encode(), h, w)
            y, _ = stc.embed_cols(x, np.ones(m * w), msg, cols, h)
            ch = max(1, int((y != x).sum()))
            scored.append(((m / ch) / bound, cols.tolist()))
        scored.sort(key=lambda t: -t[0])
        table[w] = [c for _, c in scored[:TOP_K]]
        report[w] = [round(s, 4) for s, _ in scored[:TOP_K]]
        print(f"w={w:2d} best={scored[0][0]:.3f} top{TOP_K}_min={scored[TOP_K-1][0]:.3f}  t={time.time()-t0:.0f}s", flush=True)
    if out is None:
        out = Path(__file__).resolve().parents[1] / "src/pqpks/stego/stc_table_h10.json" if h == H else \
            Path(__file__).resolve().parent / "experiments" / "stc_tables" / f"stc_table_h{h}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"h": h, "n_cand": N_CAND, "top_k": TOP_K, "n_eval": N_EVAL,
                               "table": table, "efficiency_ratio": report}))
    print("written", out)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--h", type=int, default=H)
    ap.add_argument("--out", default=None)
    ap.add_argument("--w-max", type=int, default=W_MAX)
    a = ap.parse_args()
    main(a.h, Path(a.out) if a.out else None, a.w_max)
