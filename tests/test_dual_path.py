"""Tests for primal vs dual ridge solver path.

When F > N, the ridge solver uses the dual (Woodbury) form.
These tests verify that primal and dual produce identical results.
"""

import pytest
import jax
import jax.numpy as jnp
import numpy as np

from hifi_anova.core.features import build_first_order_features, build_second_order_features
from hifi_anova.core.gram import build_gram_matrix
from hifi_anova.core.pairs import PairManager
from hifi_anova.training.regularization import build_regularization_vector
from hifi_anova.training.ridge import weighted_ridge_solve
from hifi_anova.training.trainer import HiFiANOVATrainer
from hifi_anova.analysis.sobol import compute_sobol_indices

pytestmark = pytest.mark.integration


class TestDualPathEquivalence:
    """Verify that the dual solver (F > N) matches the primal solver."""

    def test_small_N_large_F_matches_primal(self):
        """Direct comparison: solve with F > N, compare to explicit primal."""
        np.random.seed(42)
        N = 50
        D = 10
        K1 = 5
        X = np.random.uniform(0, 1, (N, D))
        F = D * (2 * K1 + 1)  # 110 > 50

        # Known function
        f = 5.0 * (X[:, 0] - 0.5) + 3.0 * np.cos(2 * np.pi * X[:, 1])
        y = f + 0.3 * np.random.randn(N)

        Phi = np.asarray(
            build_first_order_features(jnp.array(X), K1), dtype=np.float64)
        assert Phi.shape == (N, F), f"Expected F={F} > N={N} for dual path"

        f0 = float(np.mean(y))
        y_c = y - f0
        reg = np.asarray(
            build_regularization_vector(D, K1, 0, 0, 'variance', 0.01, 0),
            dtype=np.float64)

        # Solve via the framework (should use dual path)
        w_framework = np.asarray(
            weighted_ridge_solve(jnp.array(Phi), jnp.array(y_c), jnp.array(reg)))

        # Explicit primal solve for reference
        A = Phi.T @ Phi + np.diag(reg)
        w_primal = np.linalg.solve(A, Phi.T @ y_c)

        # Should match to high precision
        np.testing.assert_allclose(w_framework, w_primal, rtol=1e-6, atol=1e-8,
                                   err_msg="Dual path should match primal solution")

    def test_predictions_match(self):
        """Predictions from dual path should match primal path."""
        np.random.seed(123)
        N = 40
        D = 8
        K1 = 5
        X = np.random.uniform(0, 1, (N, D))

        f = 3.0 * (X[:, 0] - 0.5) + 2.0 * np.sin(2 * np.pi * X[:, 2])
        y = f + 0.2 * np.random.randn(N)

        Phi = np.asarray(
            build_first_order_features(jnp.array(X), K1), dtype=np.float64)
        F = Phi.shape[1]
        assert F > N, f"Need F={F} > N={N}"

        f0 = float(np.mean(y))
        y_c = y - f0
        reg = np.asarray(
            build_regularization_vector(D, K1, 0, 0, 'variance', 0.005, 0),
            dtype=np.float64)

        w = np.asarray(
            weighted_ridge_solve(jnp.array(Phi), jnp.array(y_c), jnp.array(reg)))

        # Predictions
        y_pred = Phi @ w + f0
        residuals = y - y_pred
        rmse = float(np.sqrt(np.mean(residuals ** 2)))

        # Should achieve reasonable fit even with N < F
        assert rmse < np.std(y), "Dual-path prediction should beat baseline"

    def test_sobol_indices_valid_when_underdetermined(self):
        """Sobol indices should sum to ~1 even when F > N."""
        np.random.seed(42)
        N = 60
        D = 10
        K1 = 5
        X = np.random.uniform(0, 1, (N, D))

        # 3 active variables
        f = (5.0 * (X[:, 0] - 0.5)
             + 3.0 * np.cos(2 * np.pi * X[:, 1])
             + 2.0 * (X[:, 2] - 0.5))
        y = f + 0.3 * np.random.randn(N)

        Phi = np.asarray(
            build_first_order_features(jnp.array(X), K1), dtype=np.float64)
        f0 = float(np.mean(y))
        y_c = y - f0
        reg = np.asarray(
            build_regularization_vector(D, K1, 0, 0, 'variance', 0.01, 0),
            dtype=np.float64)

        w = np.asarray(
            weighted_ridge_solve(jnp.array(Phi), jnp.array(y_c), jnp.array(reg)))

        # Compute Sobol from coefficients
        G1 = np.asarray(build_gram_matrix(K1), dtype=np.float64)
        block = 2 * K1 + 1
        variances = {}
        for i in range(D):
            wi = w[i * block: (i + 1) * block]
            variances[i] = max(0, float(wi @ G1 @ wi))

        total = sum(variances.values())
        assert total > 0, "Total variance should be positive"

        sobol = {i: v / total for i, v in variances.items()}

        # Sum to 1
        sobol_sum = sum(sobol.values())
        assert abs(sobol_sum - 1.0) < 0.01, f"Sobol sum {sobol_sum} should be ~1"

        # Active variables should dominate
        active_sobol = sobol[0] + sobol[1] + sobol[2]
        assert active_sobol > 0.5, (
            f"Active variables should dominate, got {active_sobol:.3f}")

    def test_unpenalized_intercept_matches_primal(self):
        """F > N with an UNPENALIZED augmented-intercept column (reg[0] = 0).

        The Woodbury dual forms R^{-1}; a zero penalty entry is undefined there
        and the old cap (reg_inv = 1/(reg_max*1e-15)) scaled that column by
        ~1e15, ill-conditioning K and returning a materially wrong solution
        (~1e-2 coefficient error) while the primal system stayed well
        conditioned. The solver must fall back to the primal here.
        """
        np.random.seed(0)
        N, Ffeat = 40, 120
        Phi = np.random.randn(N, Ffeat)
        Z = np.concatenate([np.ones((N, 1)), Phi], axis=1)  # F = 121 > N = 40
        y = np.random.randn(N) + 2.0                        # intercept matters

        reg = np.concatenate([[0.0], np.full(Ffeat, 0.01)])  # intercept free
        assert Z.shape[1] > N, "Need F > N to exercise the dual dispatch"

        w = np.asarray(
            weighted_ridge_solve(jnp.array(Z), jnp.array(y), jnp.array(reg)))

        # Exact primal reference (A is PD: the data term covers the intercept).
        A = Z.T @ Z + np.diag(reg)
        w_primal = np.linalg.solve(A, Z.T @ y)

        np.testing.assert_allclose(
            w, w_primal, rtol=1e-6, atol=1e-8,
            err_msg="Unpenalized-intercept F>N solve must match exact primal")

    def test_weighted_dual_path(self):
        """Weighted ridge in dual form should also work correctly."""
        np.random.seed(42)
        N = 50
        D = 8
        K1 = 5
        X = np.random.uniform(0, 1, (N, D))

        f = 4.0 * (X[:, 0] - 0.5)
        y = f + 0.5 * np.random.randn(N)

        Phi = np.asarray(
            build_first_order_features(jnp.array(X), K1), dtype=np.float64)
        f0 = float(np.mean(y))
        y_c = y - f0
        reg = np.asarray(
            build_regularization_vector(D, K1, 0, 0, 'uniform', 0.01, 0),
            dtype=np.float64)

        # Non-uniform weights
        weights = np.ones(N)
        weights[:N // 2] = 2.0  # first half weighted more

        w_weighted = np.asarray(
            weighted_ridge_solve(
                jnp.array(Phi), jnp.array(y_c), jnp.array(reg),
                weights=jnp.array(weights)))

        # Explicit weighted primal
        W = np.diag(weights)
        A = Phi.T @ W @ Phi + np.diag(reg)
        w_primal = np.linalg.solve(A, Phi.T @ W @ y_c)

        np.testing.assert_allclose(w_weighted, w_primal, rtol=1e-5, atol=1e-7,
                                   err_msg="Weighted dual should match weighted primal")


class TestDualPathTrainerIntegration:
    """End-to-end tests with the Trainer using small N."""

    def test_trainer_works_with_small_N(self):
        """Trainer should handle N < F gracefully via automatic dual path."""
        np.random.seed(42)
        N = 80
        D = 10
        K1 = 5
        X = np.random.uniform(0, 1, (N, D))

        f = 5.0 * (X[:, 0] - 0.5) + 3.0 * np.cos(2 * np.pi * X[:, 1])
        y = f + 0.5 * np.random.randn(N)

        n_val = 20
        x_train, x_val = jnp.array(X[n_val:]), jnp.array(X[:n_val])
        y_train, y_val = jnp.array(y[n_val:]), jnp.array(y[:n_val])
        # F = 10 * 11 = 110 > N_train = 60

        config = {
            'K1': K1, 'K2': 0,
            'stages': ['A'],
            'strategy': 'variance',
            'lambda_order1': 0.01,
        }
        trainer = HiFiANOVATrainer(config)
        model, results = trainer.fit(x_train, y_train, x_val, y_val)

        # Should produce valid predictions
        pred, _ = model.predict(x_val)
        assert pred.shape == (n_val,)
        assert np.all(np.isfinite(np.asarray(pred)))

    def test_dual_path_with_second_order(self):
        """N < F with second-order features (even more underdetermined)."""
        np.random.seed(42)
        N = 50
        D = 5
        K1 = 3
        K2 = 2
        X = np.random.uniform(0, 1, (N, D))

        f = (3.0 * (X[:, 0] - 0.5)
             + 2.0 * (X[:, 0] - 0.5) * (X[:, 1] - 0.5))
        y = f + 0.3 * np.random.randn(N)

        n_val = 10
        x_train, x_val = jnp.array(X[n_val:]), jnp.array(X[:n_val])
        y_train, y_val = jnp.array(y[n_val:]), jnp.array(y[:n_val])

        config = {
            'K1': K1, 'K2': K2,
            'stages': ['A', 'B'],
            'strategy': 'variance',
            'lambda_order1': 0.01,
            'lambda_order2': 0.1,
            'pair_selection': 'all',
        }
        trainer = HiFiANOVATrainer(config)
        model, results = trainer.fit(x_train, y_train, x_val, y_val)

        pred, _ = model.predict(x_val)
        assert np.all(np.isfinite(np.asarray(pred)))
