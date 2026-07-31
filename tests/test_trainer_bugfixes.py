"""Regression tests for latent trainer bugs (read-before-assignment).

These guard specific crash paths that were previously reachable with ordinary
configs. Kept small (low N) so they run in the default (quick) tier.
"""
import jax
jax.config.update("jax_enable_x64", True)

import numpy as np

from hifi_anova.data.synthetic import generate_ishigami
from hifi_anova.data.preprocessing import preprocess_data
from hifi_anova.training.trainer import HiFiANOVATrainer
from hifi_anova.analysis.sobol import compute_sobol_indices


def test_third_order_default_triple_selection_does_not_crash():
    """K3>0 with the DEFAULT triple_selection ('all_active') must fit.

    That default routes through the threshold-fallback branch that builds a
    temporary first-order model to score variables. It referenced ``_bs`` — a
    name bound only by a *later* local ``import basis_size as _bs`` in fit() — so
    Python treated ``_bs`` as a function-local and raised UnboundLocalError
    before that import ran. (A config that set triple_selection='all' or a
    principled method sidestepped the branch, hiding the bug.) The module-level
    ``basis_size`` is now used instead.
    """
    X, y, _ = generate_ishigami(n_samples=600, noise_std=0.1, seed=0)
    data = preprocess_data(X, y, seed=0)
    cfg = {
        "K1": 6, "K2": 3, "K3": 2, "strategy": "variance",
        "lambda_order1": 1e-3, "lambda_order2": 1e-2, "lambda_order3": 1e-1,
        "stages": ["A", "B"],
        # NOTE: no 'triple_selection' -> default 'all_active' (the crash path).
    }
    model, results = HiFiANOVATrainer(cfg).fit(
        data["x_train"], data["y_train"], data["x_val"], data["y_val"])

    # Sanity: it produced a usable model with well-formed Sobol indices.
    sob = compute_sobol_indices(model, data["x_test"])
    first = sob["mean_sobol"]["first_order"]
    assert set(first) == {0, 1, 2}
    assert all(np.isfinite(v) for v in first.values())
    # Ishigami mean first-order: x1~0.31, x2~0.44, x3~0 (loose bounds).
    assert first[1] > first[2]
