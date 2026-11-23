"""Post-hoc model-refit bootstrap of E1 D4 (scripts/confirm/e1_posthoc_refit.py): resampling keeps the folds, and the
advantage is the cross-validated statistic of e1_confirm.py."""
import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("e1_posthoc_refit", ROOT / "scripts" / "confirm" / "e1_posthoc_refit.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_strata_keep_fold_sizes():
    fold = np.r_[np.arange(100) % 5, -np.ones(7, int)]
    groups = m.strata(fold)
    assert sorted(g.size for g in groups) == [7, 20, 20, 20, 20, 20]
    rng = np.random.default_rng(0)
    idx = np.concatenate([rng.choice(g, g.size, replace=True) for g in groups])
    assert np.array_equal(np.bincount(fold[idx] + 1), np.bincount(fold + 1))


def test_refit_bootstrap_on_uniform_features_is_centred():
    rng = np.random.default_rng(1)
    n = 600
    X, U = rng.normal(size=(n, 6)), rng.normal(size=(n, 6))
    fold = np.r_[np.arange(n - 50) % 5, -np.ones(50, int)]
    adv = m.advantage(X, U, fold)
    boot = m.refit_boot(X, U, fold, 20, np.random.default_rng(2))
    assert abs(adv) < 0.15 and boot.shape == (20,) and np.all(np.abs(boot) <= 1)
    cc, cu = m.cv_correct(X, U, fold)
    t = fold >= 0
    assert adv == 2 * np.r_[cc[t], cu[t]].mean() - 1
