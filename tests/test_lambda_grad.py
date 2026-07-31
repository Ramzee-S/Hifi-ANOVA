"""Analytic lambda-gradient tests for the model-selection criteria.

The closed-form criterion gradients (from the eigendecomposition, single lambda;
from one A-factorization, multi lambda) must match finite differences, and the
analytic-gradient optimizers must land on the same optima as the numeric ones.
"""
import jax
jax.config.update("jax_enable_x64", True)

import numpy as np
import pytest

from hifi_anova.data.synthetic import generate_friedman1, generate_ishigami
from hifi_anova.data.preprocessing import preprocess_data
from hifi_anova.core.features import (
    build_first_order_features, build_second_order_features)
from hifi_anova.core.pairs import PairManager
from hifi_anova.training.regularization import build_regularization_vector
from hifi_anova.training.hyperopt import (
    RidgePathEigSolver, optimize_single_lambda, optimize_multi_lambda,
    _criterion_valgrad_multi)
from hifi_anova.training.hyperopt_jax import criterion_valgrad_jax


# ---------------------------------------------------------------------------
# Single-lambda: closed-form gradient vs finite differences
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("method", ["gcv", "aic", "bic", "evidence"])
def test_single_lambda_gradient_matches_fd(method):
    """criterion_and_grad's derivative matches a central finite difference in a
    well-conditioned regime (F < N, moderate lambda)."""
    rng = np.random.RandomState(0)
    N, F = 300, 90
    Phi = rng.randn(N, F)
    y = rng.randn(N)
    shape = np.abs(rng.rand(F)) + 0.05
    solver = RidgePathEigSolver(Phi, y, shape)
    for lam in [0.05, 1.0, 20.0]:
        _, g = solver.criterion_and_grad(lam, method)
        h = lam * 1e-6
        vp, _ = solver.criterion_and_grad(lam + h, method)
        vm, _ = solver.criterion_and_grad(lam - h, method)
        gfd = (vp - vm) / (2 * h)
        assert g == pytest.approx(gfd, rel=1e-4, abs=1e-6)


def _friedman_first_order(n=1500, K1=10, seed=0):
    X, y = generate_friedman1(n_samples=n, noise_std=1.0, n_irrelevant=5, seed=seed)
    d = preprocess_data(X, y, seed=seed)
    xtr = d['x_train']
    ytr = np.asarray(d['y_train']) - float(np.mean(d['y_train']))
    D = 10
    Phi = np.asarray(build_first_order_features(
        xtr, K1, include_linear=True, basis_name='fourier'))
    reg_struct = np.asarray(
        build_regularization_vector(D, K1, 0, 0, 'variance', 1.0, 1.0))
    return Phi, ytr, reg_struct


@pytest.mark.parametrize("method", ["gcv", "aic", "bic", "evidence"])
def test_single_lambda_analytic_matches_numeric(method):
    """optimize_single_lambda(grad='analytic') finds the same lambda as numeric."""
    Phi, y, reg = _friedman_first_order()
    rn = optimize_single_lambda(Phi, y, reg, method=method, grad='numeric')
    ra = optimize_single_lambda(Phi, y, reg, method=method, grad='analytic')
    assert ra['lambda_opt'] == pytest.approx(rn['lambda_opt'], rel=2e-3)
    # Diagnostics at the optimum agree.
    assert ra['df'] == pytest.approx(rn['df'], rel=1e-3)
    assert ra['gcv'] == pytest.approx(rn['gcv'], rel=1e-3)


def test_single_lambda_auto_and_guard():
    Phi, y, reg = _friedman_first_order()
    # 'auto' uses analytic for a well-conditioned positive shape (variance).
    ra = optimize_single_lambda(Phi, y, reg, method='gcv', grad='auto')
    rn = optimize_single_lambda(Phi, y, reg, method='gcv', grad='numeric')
    assert ra['lambda_opt'] == pytest.approx(rn['lambda_opt'], rel=2e-3)
    # 'analytic' refuses a non-positive shape.
    bad = reg.copy()
    bad[2] = 0.0
    with pytest.raises(ValueError):
        optimize_single_lambda(Phi, y, bad, method='gcv', grad='analytic')
    with pytest.raises(ValueError):
        optimize_single_lambda(Phi, y, reg, method='gcv', grad='bogus')


# ---------------------------------------------------------------------------
# Multi-lambda: joint gradient and optimizer
# ---------------------------------------------------------------------------

def _ishigami_two_order(n=1500, K1=8, K2=4, seed=0):
    X, y, _ = generate_ishigami(n_samples=n, noise_std=0.1, seed=seed)
    d = preprocess_data(X, y, seed=seed)
    xtr = d['x_train']
    ytr = (np.asarray(d['y_train']) - float(np.mean(d['y_train']))).astype(np.float64)
    D = 3
    pm = PairManager(D)
    phi1 = np.asarray(build_first_order_features(
        xtr, K1, include_linear=True, basis_name='fourier'), dtype=np.float64)
    phi2 = np.asarray(build_second_order_features(
        xtr, K2, pm.pair_indices, include_linear=True, basis_name='fourier'),
        dtype=np.float64)
    Phi = np.concatenate([phi1, phi2], axis=1).astype(np.float64)
    return Phi, ytr, D, K1, K2, pm.P


@pytest.mark.parametrize("method", ["gcv", "aic", "bic", "evidence"])
def test_multi_lambda_gradient_matches_fd(method):
    """The joint (lambda1, lambda2) gradient matches finite differences (f64)."""
    Phi, y, D, K1, K2, P = _ishigami_two_order()
    C = Phi.T @ Phi
    b = Phi.T @ y
    s1 = np.asarray(build_regularization_vector(
        D, K1, K2, P, 'variance', 1.0, 0.0), dtype=np.float64)
    s2 = np.asarray(build_regularization_vector(
        D, K1, K2, P, 'variance', 0.0, 1.0), dtype=np.float64)
    shapes = [s1, s2]
    sizes = [int((s1 > 0).sum()), int((s2 > 0).sum())]
    t = [np.log10(0.01), np.log10(0.05)]
    _, g = _criterion_valgrad_multi(Phi, C, b, y, shapes, sizes, t, method)
    gfd = np.zeros(2)
    for k in range(2):
        h = 1e-5
        tp, tm = list(t), list(t)
        tp[k] += h
        tm[k] -= h
        gfd[k] = (_criterion_valgrad_multi(Phi, C, b, y, shapes, sizes, tp, method)[0]
                  - _criterion_valgrad_multi(Phi, C, b, y, shapes, sizes, tm, method)[0]) / (2 * h)
    np.testing.assert_allclose(g, gfd, rtol=1e-4, atol=1e-7)


@pytest.mark.parametrize("method", ["gcv", "aic", "bic", "evidence"])
def test_multi_lambda_analytic_matches_numeric(method):
    """optimize_multi_lambda(grad='analytic') finds the same (l1,l2) as numeric."""
    Phi, y, D, K1, K2, P = _ishigami_two_order()
    rn = optimize_multi_lambda(Phi, y, D, K1, K2, P, 'variance', method, grad='numeric')
    ra = optimize_multi_lambda(Phi, y, D, K1, K2, P, 'variance', method, grad='analytic')
    assert ra['lambda_order1'] == pytest.approx(rn['lambda_order1'], rel=5e-3, abs=1e-6)
    assert ra['lambda_order2'] == pytest.approx(rn['lambda_order2'], rel=5e-3, abs=1e-6)


def test_multi_lambda_grad_validation():
    Phi, y, D, K1, K2, P = _ishigami_two_order()
    with pytest.raises(ValueError):
        optimize_multi_lambda(Phi, y, D, K1, K2, P, 'variance', 'gcv', grad='bogus')


# ---------------------------------------------------------------------------
# JAX/autodiff gradients: must match the closed-form analytic gradients (~1e-10)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("method", ["gcv", "aic", "bic", "evidence"])
def test_jax_single_lambda_grad_matches_analytic(method):
    """The autodiff gradient (log10 space) matches the eigen closed-form gradient
    (chain-ruled to log10) to ~1e-10 on the same designs as the analytic tests."""
    rng = np.random.RandomState(0)
    N, F = 300, 90
    Phi = rng.randn(N, F)
    y = rng.randn(N)
    shape = np.abs(rng.rand(F)) + 0.05
    solver = RidgePathEigSolver(Phi, y, shape)
    ln10 = np.log(10.0)
    for lam in [0.05, 1.0, 20.0]:
        v_a, g_a_lam = solver.criterion_and_grad(lam, method)   # d/dlambda
        g_a_log = g_a_lam * lam * ln10                          # -> d/dlog10
        v_j, g_j = criterion_valgrad_jax([np.log10(lam)], Phi, y, [shape], method)
        assert v_j == pytest.approx(v_a, rel=1e-10, abs=1e-10)
        assert g_j[0] == pytest.approx(g_a_log, rel=1e-8, abs=1e-9)


@pytest.mark.parametrize("method", ["gcv", "aic", "bic", "evidence"])
def test_jax_multi_lambda_grad_matches_analytic(method):
    """The autodiff joint gradient matches _criterion_valgrad_multi to ~1e-10
    (both already return the gradient w.r.t. log10 lambda)."""
    Phi, y, D, K1, K2, P = _ishigami_two_order()
    C = Phi.T @ Phi
    b = Phi.T @ y
    s1 = np.asarray(build_regularization_vector(
        D, K1, K2, P, 'variance', 1.0, 0.0), dtype=np.float64)
    s2 = np.asarray(build_regularization_vector(
        D, K1, K2, P, 'variance', 0.0, 1.0), dtype=np.float64)
    shapes = [s1, s2]
    sizes = [int((s1 > 0).sum()), int((s2 > 0).sum())]
    t = [np.log10(0.01), np.log10(0.05)]
    v_a, g_a = _criterion_valgrad_multi(Phi, C, b, y, shapes, sizes, t, method)
    v_j, g_j = criterion_valgrad_jax(t, Phi, y, shapes, method)
    assert v_j == pytest.approx(v_a, rel=1e-10, abs=1e-10)
    np.testing.assert_allclose(g_j, g_a, rtol=1e-8, atol=1e-9)


@pytest.mark.parametrize("method", ["gcv", "aic", "bic", "evidence"])
def test_jax_single_lambda_optimum_matches_analytic(method):
    """optimize_single_lambda(grad='jax') lands on the same lambda as 'analytic'."""
    Phi, y, reg = _friedman_first_order()
    ra = optimize_single_lambda(Phi, y, reg, method=method, grad='analytic')
    rj = optimize_single_lambda(Phi, y, reg, method=method, grad='jax')
    assert rj['lambda_opt'] == pytest.approx(ra['lambda_opt'], rel=2e-3)
    assert rj['df'] == pytest.approx(ra['df'], rel=1e-3)
    assert rj['gcv'] == pytest.approx(ra['gcv'], rel=1e-3)


@pytest.mark.parametrize("method", ["gcv", "aic", "bic", "evidence"])
def test_jax_multi_lambda_optimum_matches_analytic(method):
    """optimize_multi_lambda(grad='jax') lands on the same (l1,l2) as 'analytic'."""
    Phi, y, D, K1, K2, P = _ishigami_two_order()
    ra = optimize_multi_lambda(Phi, y, D, K1, K2, P, 'variance', method, grad='analytic')
    rj = optimize_multi_lambda(Phi, y, D, K1, K2, P, 'variance', method, grad='jax')
    assert rj['lambda_order1'] == pytest.approx(ra['lambda_order1'], rel=5e-3, abs=1e-6)
    assert rj['lambda_order2'] == pytest.approx(ra['lambda_order2'], rel=5e-3, abs=1e-6)


def test_jax_grad_validation_and_zero_shape():
    """'jax' is accepted; a bogus mode still raises; and unlike 'analytic' the JAX
    path tolerates a non-positive shape entry (the evidence log-det masks it)."""
    Phi, y, reg = _friedman_first_order()
    with pytest.raises(ValueError):
        optimize_single_lambda(Phi, y, reg, method='gcv', grad='bogus')
    bad = reg.copy()
    bad[2] = 0.0
    # analytic refuses a zero; jax accepts it and still optimizes.
    with pytest.raises(ValueError):
        optimize_single_lambda(Phi, y, bad, method='gcv', grad='analytic')
    rj = optimize_single_lambda(Phi, y, bad, method='gcv', grad='jax')
    assert np.isfinite(rj['lambda_opt']) and rj['lambda_opt'] > 0
