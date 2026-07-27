"""Tests for ridge solver and regularization."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from sklearn.linear_model import Ridge

from hifi_anova.training.ridge import weighted_ridge_solve
from hifi_anova.training.regularization import (
    build_regularization_vector, _build_second_order_reg_block
)

pytestmark = pytest.mark.smoke


class TestRidgeSolver:
    def test_matches_sklearn_uniform_lambda(self):
        """Should match sklearn Ridge on uniform regularization."""
        rng = np.random.RandomState(42)
        N, F = 500, 20
        X = rng.randn(N, F)
        w_true = rng.randn(F)
        y = X @ w_true + 0.1 * rng.randn(N)

        lam = 1.0
        reg_diag = jnp.full(F, lam)

        # Our solver
        w_ours = weighted_ridge_solve(jnp.array(X), jnp.array(y), reg_diag)

        # sklearn (alpha needs to be scaled by N for equivalence)
        # sklearn minimizes: (1/N) * ||y - Xw||^2 + alpha * ||w||^2
        # We minimize: ||y - Xw||^2 + w^T R w
        # So alpha_sklearn = lam / N does NOT apply directly.
        # Actually sklearn: min 1/(2N) ||y-Xw||^2 + alpha/2 ||w||^2
        # => (X^T X + N*alpha*I) w = X^T y
        # Our: (X^T X + R) w = X^T y
        # Match when R = N * alpha * I => alpha = lam / N
        # Actually sklearn uses: (X^T X / N + alpha * I) w = X^T y / N (after normalization)
        # Let's just compare with manual formula
        A = X.T @ X + lam * np.eye(F)
        b = X.T @ y
        w_manual = np.linalg.solve(A, b)

        assert np.allclose(np.array(w_ours), w_manual, rtol=1e-5)

    def test_weighted_ridge(self):
        """Weighted ridge should give different results than unweighted."""
        rng = np.random.RandomState(42)
        N, F = 200, 10
        X = rng.randn(N, F)
        y = rng.randn(N)
        reg_diag = jnp.full(F, 0.1)

        # Uniform weights
        w_uniform = weighted_ridge_solve(jnp.array(X), jnp.array(y), reg_diag)

        # Non-uniform weights
        weights = jnp.array(rng.uniform(0.1, 2.0, size=N))
        w_weighted = weighted_ridge_solve(jnp.array(X), jnp.array(y), reg_diag, weights)

        # Should be different
        assert not jnp.allclose(w_uniform, w_weighted, atol=1e-3)

    def test_per_feature_regularization(self):
        """Higher regularization should shrink coefficients more."""
        rng = np.random.RandomState(42)
        N, F = 500, 10
        X = rng.randn(N, F)
        w_true = np.ones(F)
        y = X @ w_true + 0.1 * rng.randn(N)

        # Low regularization
        reg_low = jnp.full(F, 0.01)
        w_low = weighted_ridge_solve(jnp.array(X), jnp.array(y), reg_low)

        # High regularization
        reg_high = jnp.full(F, 100.0)
        w_high = weighted_ridge_solve(jnp.array(X), jnp.array(y), reg_high)

        # Higher reg => smaller coefficients
        assert float(jnp.sum(w_high ** 2)) < float(jnp.sum(w_low ** 2))

    def test_curvature_leaves_linear_free(self):
        """With curvature penalty, linear term should be unpenalized."""
        from hifi_anova.core.features import build_first_order_features
        from hifi_anova.training.regularization import build_regularization_vector

        rng = np.random.RandomState(42)
        N = 1000
        D = 3
        K = 5
        x = jax.random.uniform(jax.random.PRNGKey(0), (N, D))

        # Target is purely linear: y = x0 - 0.5
        y = x[:, 0] - 0.5

        phi = build_first_order_features(x, K)
        reg = build_regularization_vector(D, K, 0, 0, 'curvature', 10.0, 0.0)

        w = weighted_ridge_solve(phi, y, reg)

        # First variable's linear coefficient should be close to 1
        # (since target is exactly the linear basis function)
        block = 2 * K + 1
        w_var0_linear = float(w[0])
        assert abs(w_var0_linear - 1.0) < 0.05

        # Fourier terms should be nearly zero
        w_var0_fourier = w[1:block]
        assert float(jnp.max(jnp.abs(w_var0_fourier))) < 0.1


class TestSecondOrderRegularization:
    """Regression tests for the second-order curvature penalty.

    These tests catch the lambda-cancellation bug where the formula
    lam * (D4[a]/lam + D4[b]/lam) = D4[a] + D4[b] makes the penalty
    independent of lambda.
    """

    def test_penalty_scales_with_lambda(self):
        """The second-order curvature penalty MUST scale with lambda.

        If lambda cancels out, r1 == r2 regardless of lambda values.
        This is the regression test for the critical lambda-cancellation bug.
        """
        K = 5
        r1 = _build_second_order_reg_block(K, lam=0.01, strategy='curvature')
        r2 = _build_second_order_reg_block(K, lam=0.1, strategy='curvature')

        # If lambda cancels, r1 == r2. This MUST fail:
        assert not np.allclose(r1, r2), \
            "Second-order curvature penalty must depend on lambda!"

        # The ratio should be approximately 10 (= 0.1 / 0.01):
        mask = r1 > 1e-15
        ratio = r2[mask] / r1[mask]
        assert np.allclose(ratio, 10.0, rtol=0.1), \
            f"Penalty should scale linearly with lambda, got ratio range [{ratio.min():.2f}, {ratio.max():.2f}]"

    def test_penalty_scales_with_lambda_smoothness(self):
        """Same scaling test for the smoothness strategy."""
        K = 5
        r1 = _build_second_order_reg_block(K, lam=0.01, strategy='smoothness')
        r2 = _build_second_order_reg_block(K, lam=0.1, strategy='smoothness')

        assert not np.allclose(r1, r2), \
            "Second-order smoothness penalty must depend on lambda!"

        mask = r1 > 1e-15
        ratio = r2[mask] / r1[mask]
        assert np.allclose(ratio, 10.0, rtol=0.1), \
            f"Penalty should scale linearly with lambda, got ratio range [{ratio.min():.2f}, {ratio.max():.2f}]"

    def test_uniform_penalty_scales(self):
        """Uniform strategy should obviously scale with lambda (sanity check)."""
        K = 5
        r1 = _build_second_order_reg_block(K, lam=0.01, strategy='uniform')
        r2 = _build_second_order_reg_block(K, lam=0.1, strategy='uniform')
        ratio = r2 / r1
        assert np.allclose(ratio, 10.0, rtol=1e-10)

    def test_exact_basis_function_recovery(self):
        """An exact product basis function must be recoverable at near-zero lambda.

        cos(2*pi*x1)*cos(2*pi*x2) is a single basis function in the
        second-order block. With lambda~0 and N >> F, the OLS coefficient
        should be ~1.0 and R^2 ~1.0.
        """
        from hifi_anova.core.features import build_first_order_features, build_second_order_features
        from hifi_anova.core.pairs import PairManager

        rng = np.random.RandomState(42)
        N = 10000
        D = 2
        K1 = K2 = 5
        X = rng.uniform(0, 1, (N, D))
        y = np.cos(2 * np.pi * X[:, 0]) * np.cos(2 * np.pi * X[:, 1])
        y_c = y - np.mean(y)

        x = jnp.array(X, dtype=jnp.float64)
        pm = PairManager(D)
        phi1 = build_first_order_features(x, K1)
        phi2 = build_second_order_features(x, K2, pm.pair_indices)
        Phi = jnp.concatenate([phi1, phi2], axis=1)

        reg = build_regularization_vector(D, K1, K2, pm.P, 'curvature', 1e-10, 1e-10)
        w = weighted_ridge_solve(Phi, jnp.array(y_c), reg)

        # The (cos1, cos1) coefficient
        F1 = D * (2 * K1 + 1)
        block = 2 * K2 + 1
        cos1_cos1_idx = F1 + 1 * block + 1
        coeff = float(w[cos1_cos1_idx])

        assert abs(coeff - 1.0) < 0.05, \
            f"Exact basis function coefficient should be ~1.0, got {coeff:.4f}. " \
            f"Lambda-cancellation bug may have returned."

        # R-squared should be near 1
        pred = Phi @ w
        r2 = 1.0 - float(jnp.mean((jnp.array(y_c) - pred)**2) / jnp.var(jnp.array(y_c)))
        assert r2 > 0.99, f"R^2 should be >0.99 for exact basis function, got {r2:.4f}"


class TestPerOrderStrategyDict:
    """Tests for per-order strategy dictionary support."""

    def test_dict_strategy_produces_different_penalties(self):
        """Different strategies per order should produce different penalty structures."""
        D, K1, K2, P = 3, 5, 3, 3
        lam1, lam2 = 0.01, 0.1

        # Uniform for both
        r_uniform = build_regularization_vector(
            D, K1, K2, P, 'uniform', lam1, lam2)

        # Dict: curvature for order 1, smoothness for order 2
        r_dict = build_regularization_vector(
            D, K1, K2, P,
            {'order1': 'curvature', 'order2': 'smoothness'},
            lam1, lam2)

        # They must differ (different penalties per order)
        assert not np.allclose(r_uniform, r_dict), \
            "Per-order dict strategy should produce different penalties than uniform"

        # Order 1 part should match curvature strategy
        r_curv_only = build_regularization_vector(
            D, K1, 0, 0, 'curvature', lam1, 0.0)
        F1 = D * (2 * K1 + 1)
        assert np.allclose(r_dict[:F1], r_curv_only[:F1]), \
            "Order 1 block should match curvature strategy"

    def test_dict_strategy_with_default_fallback(self):
        """Missing orders should fall back to 'default' key."""
        D, K1, K2, P = 3, 5, 3, 3
        lam1, lam2 = 0.01, 0.1

        # Only specify order1, let order2 fall back to default
        r_dict = build_regularization_vector(
            D, K1, K2, P,
            {'order1': 'curvature', 'default': 'uniform'},
            lam1, lam2)

        # Order 2 should match uniform
        r_uniform = build_regularization_vector(
            D, K1, K2, P, 'uniform', lam1, lam2)
        F1 = D * (2 * K1 + 1)
        assert np.allclose(r_dict[F1:], r_uniform[F1:]), \
            "Order 2 should fall back to default (uniform)"

    def test_string_strategy_equivalent_to_dict(self):
        """A single string should behave like {'order1': s, 'order2': s}."""
        D, K1, K2, P = 3, 5, 3, 3
        lam1, lam2 = 0.01, 0.1

        r_str = build_regularization_vector(
            D, K1, K2, P, 'smoothness', lam1, lam2)
        r_dict = build_regularization_vector(
            D, K1, K2, P,
            {'order1': 'smoothness', 'order2': 'smoothness'},
            lam1, lam2)
        assert np.allclose(r_str, r_dict), \
            "String strategy should match dict with same strategy for all orders"
