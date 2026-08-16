"""Weighted (GLS) branch of ridge_analytics — Phase 3.

Pure-linear-algebra checks: the unit-weight limit reproduces the unweighted path,
and the weighted quantities satisfy their defining identities (weighted normal
equations, hat-diagonal df, whitened residual scale). See
``hifi_anova/analysis/automl.py`` and ``FittedDesignRecord_brief.md`` §2.5(A).
"""

import numpy as np
import pytest

from hifi_anova.analysis.automl import ridge_analytics
from hifi_anova.training.ridge import leverage_diag

pytestmark = pytest.mark.smoke


def _problem(N=80, F=12, seed=0):
    rng = np.random.RandomState(seed)
    Phi = rng.randn(N, F)
    beta = rng.randn(F)
    y = Phi @ beta + 0.3 * rng.randn(N)
    reg = 0.05 * np.ones(F)
    return Phi, y, reg


def test_unit_weights_reduce_to_unweighted():
    Phi, y, reg = _problem()
    a0 = ridge_analytics(Phi, y, reg)
    a1 = ridge_analytics(Phi, y, reg, weights=np.ones(Phi.shape[0]))
    for k in ('sigma_hat', 'df', 'df_residual', 'loo_cv', 'rss', 'gcv',
              'tr_H2', 'aic', 'bic'):
        assert a1[k] == pytest.approx(a0[k], rel=1e-10, abs=1e-12), k
    np.testing.assert_allclose(a1['w'], a0['w'], rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(a1['leverages'], a0['leverages'],
                               rtol=1e-10, atol=1e-12)


def test_calibration_flag():
    Phi, y, reg = _problem()
    assert ridge_analytics(Phi, y, reg)['noise_scale_is_calibration'] is False
    w = 1.0 / (0.5 + np.abs(y))
    assert ridge_analytics(Phi, y, reg,
                           weights=w)['noise_scale_is_calibration'] is True


def test_weighted_solution_is_gls():
    Phi, y, reg = _problem(seed=2)
    W = 1.0 / (0.2 + np.linspace(0.1, 2.0, Phi.shape[0]))
    a = ridge_analytics(Phi, y, reg, weights=W)
    # Direct GLS solve: (Phi^T W Phi + R)^{-1} Phi^T W y.
    A = Phi.T @ (W[:, None] * Phi) + np.diag(reg)
    w_direct = np.linalg.solve(A, Phi.T @ (W * y))
    np.testing.assert_allclose(a['w'], w_direct, rtol=1e-10, atol=1e-12)


def test_weighted_df_equals_leverage_sum():
    Phi, y, reg = _problem(seed=3)
    W = 1.0 / (0.2 + np.linspace(0.1, 3.0, Phi.shape[0]))
    a = ridge_analytics(Phi, y, reg, weights=W)
    # df = tr S = sum of weighted hat diagonals (leverage_diag), pre-clip.
    lev = leverage_diag(Phi, reg, W)
    assert a['df'] == pytest.approx(float(np.sum(lev)), rel=1e-9, abs=1e-10)


def test_weighted_sigma_is_whitened_scale():
    Phi, y, reg = _problem(seed=4)
    W = 1.0 / (0.2 + np.linspace(0.1, 3.0, Phi.shape[0]))
    a = ridge_analytics(Phi, y, reg, weights=W)
    rss_w = float(np.sum(W * a['residuals'] ** 2))
    assert a['rss'] == pytest.approx(rss_w, rel=1e-12)
    assert a['sigma2_hat'] == pytest.approx(rss_w / a['df_residual'], rel=1e-12)
    assert a['sigma_hat'] == pytest.approx(np.sqrt(a['sigma2_hat']), rel=1e-12)


def test_calibrated_weights_give_unit_scale():
    """When W = 1/sigma_n^2 with the TRUE sigma_n, sigma_hat_w ≈ 1."""
    rng = np.random.RandomState(7)
    N, F = 4000, 6
    Phi = rng.randn(N, F)
    beta = rng.randn(F)
    sigma = 0.5 + np.abs(Phi[:, 0])              # true input-dependent noise
    y = Phi @ beta + sigma * rng.randn(N)
    reg = 1e-6 * np.ones(F)                      # near-OLS: little shrinkage bias
    a = ridge_analytics(Phi, y, reg, weights=1.0 / sigma ** 2)
    assert a['sigma_hat'] == pytest.approx(1.0, abs=0.08)
