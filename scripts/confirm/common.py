"""Shared code of the confirmatory scripts.

- SRM features (34 671 dimensions) and the FLD ensemble come from the sealwatch library (Univ. Innsbruck, MPL-2.0; SRM has a
  Rust core, and the ensemble reimplements the Binghamton MATLAB version with automatic search of L and d_sub). Gate G4 checks
  the reproduction of published figures before use in E2/E3.
- Ensemble score: the number of votes plus a tie-breaking term, so that a threshold can be set at a fixed FPR; the majority
  decision is unchanged.
- Frozen split file: configs/splits/split_v1.json (created with make_split.py on a machine with all images).
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def warnings_to_stdout(max_repeats: int = 5) -> None:
    """Print Python warnings (module `warnings`) to stdout as one line prefixed with "[warning]", each identical warning at most
    `max_repeats` times. Real errors (exceptions, tracebacks) still go to stderr, so in PowerShell red text only means an error.
    Called when this module is imported, so it applies to the main process and to every child process (Windows re-imports the script)."""
    import warnings
    seen: dict = {}

    def show(message, category, filename, lineno, file=None, line=None):
        key = (category.__name__, str(message))
        seen[key] = seen.get(key, 0) + 1
        if seen[key] <= max_repeats:
            more = " (further occurrences not printed)" if seen[key] == max_repeats else ""
            print(f"[warning] {category.__name__}: {message} ({Path(filename).name}:{lineno}){more}", flush=True)

    warnings.showwarning = show


warnings_to_stdout()
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "experiments"))

SPLIT = ROOT / "configs" / "splits" / "split_v1.json"
FROZEN = ROOT / "configs" / "profile_frozen_v1.json"
SRM_DIM = 34671


def sha256_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load_split(path: Path | None = None) -> dict:
    path = Path(os.environ.get("PQPKS_SPLIT", SPLIT)) if path is None else path      # PQPKS_SPLIT: code tests only
    if not Path(path).exists():
        raise FileNotFoundError(f"split file {path} not found: run scripts/confirm/make_split.py first")
    s = json.loads(Path(path).read_text(encoding="utf-8"))
    s["_sha256"] = sha256_file(path)
    return s


def check_frozen() -> str:
    """Refuse to run if src/ or configs/ differ from the frozen profile (scripts/freeze_profile.py --check). A released file
    whose hash differs from the frozen one is accepted when provenance/release_equivalence_v1.json links the two hashes."""
    from freeze_profile import accepted_hashes, source_hashes  # noqa: E402  (scripts/ is in sys.path)
    frozen = json.loads(FROZEN.read_text(encoding="utf-8"))
    cur = source_hashes()
    diff = [k for k in set(cur) | set(frozen["source_sha256"])
            if (k.startswith("src/") or (k.startswith("configs/") and "splits/" not in k))
            and cur.get(k) not in accepted_hashes(k, frozen["source_sha256"].get(k))]
    if diff:
        raise SystemExit(f"source differs from the frozen profile profile_frozen_v1: {sorted(diff)[:5]} ...")
    return sha256_file(FROZEN)


sys.path.insert(0, str(ROOT / "scripts"))


def srm_vec(x: np.ndarray) -> np.ndarray:
    import sealwatch as sw
    f = sw.srm.extract(np.ascontiguousarray(x, dtype=np.uint8))
    v = np.concatenate([np.ravel(a) for a in f.values()]).astype(np.float32)
    assert v.size == SRM_DIM, v.size
    return v


class Ensemble:
    """FLD ensemble of sealwatch (automatic L, d_sub) with continuous tie-breaking scores."""

    def __init__(self, seed: int = 0):
        self.seed = seed
        self.clf = None
        self.scales = None

    def fit(self, Xc: np.ndarray, Xs: np.ndarray):
        from sealwatch.ensemble_classifier import FldEnsembleTrainer
        fixed = os.environ.get("PQPKS_ENS_FIXED")          # "L,d_sub": code tests on very small sets only, never for measurements
        kw = {} if not fixed else dict(zip(("L", "d_sub"), map(int, fixed.split(","))))
        tr = FldEnsembleTrainer(Xc, Xs, seed=self.seed, verbose=0, **kw)
        self.clf, _ = tr.train()
        P = self._proj(np.vstack([Xc, Xs]))
        self.scales = P.std(axis=1, keepdims=True) + 1e-12
        return self

    def _proj(self, X):
        return np.stack([bl.predict(X) for bl in self.clf.base_learners])     # [L, n], sign = vote

    def score(self, X: np.ndarray) -> np.ndarray:
        P = self._proj(X)
        L = P.shape[0]
        votes = (P > 0).sum(0)
        margin = (P / self.scales).mean(0)
        return (votes + 0.49 * np.tanh(margin)) / L      # odd L: > 0.5 <=> majority; even L: ties broken by the margin


def one_thread_per_worker() -> None:
    """Call before creating W child processes: each child uses a single BLAS/OpenMP/Rayon thread, which avoids W x (number of
    cores) threads competing for the CPU. Children inherit the environment; variables already set are kept."""
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "RAYON_NUM_THREADS"):
        os.environ.setdefault(k, "1")


def workers_default() -> int:
    return max(1, (os.cpu_count() or 2) - 1)
