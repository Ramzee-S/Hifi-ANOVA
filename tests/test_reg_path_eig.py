"""Equivalence tests for the eigendecomposition fast path of compute_reg_path.

The fast path (solver='eig') must reproduce the per-lambda solve (solver='solve')
to floating-point round-off for every recorded quantity, and it must fall back
gracefully for penalty shapes with zeros (e.g. the 'curvature' strategy leaves
the k=0 term unpenalized).
"""
import jax
jax.config.update("jax_enable_x64", True)

import numpy as np
import pytest

from hifi_anova.data.synthetic import generate_ishigami
from hifi_anova.data.preprocessing import preprocess_data
from hifi_anova.core.features import (
    build_first_order_features, build_second_order_features)
from hifi_anova.core.pairs import PairManager
from hifi_anova.analysis.reg_path import compute_reg_path
from hifi_anova.training.hyperopt import (
    ridge_solve_with_diagnostics, RidgePathEigSolver)


# ---------------------------------------------------------------------------
# Unit level: the solver vs. the reference solve on random designs
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("N,F", [(300, 90), (90, 300), (150, 150)])
def test_eig_solver_matches_reference(N, F):
    """RidgePathEigSolver reproduces ridge_solve_with_diagnostics(lam*shape)."""
    rng = np.random.RandomState(N + F)
    Phi = rng.randn(N, F)
    y = rng.randn(N)
    shape = np.abs(rng.rand(F)) + 0.05  # strictly positive
    solver = RidgePathEigSolver(Phi, y, shape)
    for lam in [1e-5, 1e-2, 1.0, 100.0]:
        ref = ridge_solve_with_diagnostics(Phi, y, lam * shape)
        got = solver.diagnostics(lam)
        # w, df, evidence, rss/mse are well-conditioned; compare tightly.
        assert np.allclose(got['w'], ref['w'], rtol=1e-6, atol=1e-8)
        assert got['df'] == pytest.approx(ref['df'], rel=1e-5)
        assert got['rss'] == pytest.approx(ref['rss'], rel=1e-5, abs=1e-8)
        assert got['log_evidence'] == pytest.approx(
            ref['log_evidence'], rel=1e-5, abs=1e-6)


def test_eig_solver_rejects_zero_shape():
    rng = np.random.RandomState(0)
    Phi = rng.randn(50, 20)
    y = rng.randn(50)
    shape = np.ones(20)
    shape[3] = 0.0  # an unpenalized coordinate
    with pytest.raises(ValueError):
        RidgePathEigSolver(Phi, y, shape)


# ---------------------------------------------------------------------------
# Integration: full compute_reg_path, eig vs solve
# ---------------------------------------------------------------------------

def _ishigami_design(n=700, K1=8, K2=4, seed=0, basis='fourier'):
    X, y, _ = generate_ishigami(n_samples=n, noise_std=0.1, seed=seed)
    d = preprocess_data(X, y, seed=seed)
    xtr = d['x_train']
    ytr = np.asarray(d['y_train']) - float(np.mean(d['y_train']))
    D = 3
    pm = PairManager(D)
    phi1 = np.asarray(build_first_order_features(
        xtr, K1, include_linear=True, basis_name=basis))
    phi2 = np.asarray(build_second_order_features(
        xtr, K2, pm.pair_indices, include_linear=True, basis_name=basis))
    Phi = np.concatenate([phi1, phi2], axis=1)
    kw = dict(D=D, K1=K1, K2=K2, P=pm.P,
              pair_indices=np.asarray(pm.pair_indices),
              n_lambdas=30, lambda_range=(1e-5, 10.0), basis_name=basis)
    return Phi, ytr, kw


@pytest.mark.parametrize("basis", ["fourier", "legendre"])
def test_reg_path_eig_matches_solve(basis):
    """solver='eig' reproduces solver='solve' across all recorded arrays."""
    Phi, y, kw = _ishigami_design(basis=basis)
    p_solve = compute_reg_path(Phi, y, strategy='variance', solver='solve', **kw)
    p_eig = compute_reg_path(Phi, y, strategy='variance', solver='eig', **kw)

    assert p_solve.solver_used == 'solve'
    assert p_eig.solver_used == 'eig'
    np.testing.assert_allclose(p_eig.lambdas, p_solve.lambdas)

    for field in ['mse_values', 'df_values', 'aic_values', 'bic_values',
                  'evidence_values', 'w_norm', 'var_order1', 'var_order2',
                  'var_total']:
        a = np.asarray(getattr(p_eig, field))
        b = np.asarray(getattr(p_solve, field))
        np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-8,
                                   err_msg=f"{field} ({basis})")

    # GCV is compared where it is well-conditioned (away from df -> N, where
    # gcv = mse/(1-df/N)^2 is ill-defined for BOTH methods). The GCV-optimal
    # lambda must agree.
    N = Phi.shape[0]
    wc = (1.0 - p_solve.df_values / N) > 1e-3
    np.testing.assert_allclose(p_eig.gcv_values[wc], p_solve.gcv_values[wc],
                               rtol=1e-5)
    assert p_eig.lambda_gcv_opt == pytest.approx(p_solve.lambda_gcv_opt, rel=1e-9)

    # First-order Sobol paths.
    D = 3
    for i in range(D):
        np.testing.assert_allclose(p_eig.sobol_paths[i], p_solve.sobol_paths[i],
                                   rtol=1e-6, atol=1e-8)


def test_reg_path_auto_falls_back_for_ill_conditioned():
    """'curvature' weights span ~(2*pi*k)^4, an ill-conditioned whitening:
    auto must keep the exact per-lambda solve. Forcing 'eig' is still valid
    (it just agrees more loosely)."""
    Phi, y, kw = _ishigami_design()
    p_auto = compute_reg_path(Phi, y, strategy='curvature', solver='auto', **kw)
    assert p_auto.solver_used == 'solve'
    # eig is mathematically valid here (strictly positive shape) — forcing it
    # runs and stays a good approximation of the exact solve.
    p_eig = compute_reg_path(Phi, y, strategy='curvature', solver='eig', **kw)
    assert p_eig.solver_used == 'eig'
    np.testing.assert_allclose(p_eig.df_values, p_auto.df_values,
                               rtol=1e-3, atol=1e-6)


def test_reg_path_eig_raises_below_logdet_threshold():
    """When some lambda*reg_shape drops under the evidence log-det threshold,
    the eig evidence would diverge — 'eig' must refuse and 'auto' fall back."""
    Phi, y, kw = _ishigami_design()
    kw = {**kw, 'lambda_range': (1e-11, 10.0)}  # curvature floor ~1e-6 -> 1e-17
    p_auto = compute_reg_path(Phi, y, strategy='curvature', solver='auto', **kw)
    assert p_auto.solver_used == 'solve'
    with pytest.raises(ValueError):
        compute_reg_path(Phi, y, strategy='curvature', solver='eig', **kw)


def test_reg_path_auto_uses_eig_for_variance():
    Phi, y, kw = _ishigami_design()
    p_auto = compute_reg_path(Phi, y, strategy='variance', solver='auto', **kw)
    assert p_auto.solver_used == 'eig'
