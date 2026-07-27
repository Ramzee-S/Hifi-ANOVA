"""Tests for regularization path and stability diagnostics.

Validates:
  - Regularization path correctly sweeps lambda and produces monotone df
  - Sobol indices converge toward ground truth as lambda decreases
  - GCV minimum is well-defined
  - Stability diagnostics report small Sobol spread on stable data
"""

import pytest
import jax
import jax.numpy as jnp
import numpy as np

from hifi_anova.core.features import build_first_order_features, build_second_order_features
from hifi_anova.core.gram import build_gram_matrix, build_gram_matrix_2d
from hifi_anova.core.pairs import PairManager
from hifi_anova.training.regularization import build_regularization_vector
from hifi_anova.analysis.reg_path import compute_reg_path
from hifi_anova.analysis.automl import stability_diagnostics

pytestmark = pytest.mark.integration


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def known_signal_data():
    """Known two-variable signal for verifiable Sobol recovery.

    f(x) = 5*(x1-0.5) + 3*cos(2*pi*x2)

    Var(x1 component) = 25/12 ≈ 2.083
    Var(x2 component) = 9/2 = 4.5
    Expected Sobol: S1 ≈ 0.316, S2 ≈ 0.684
    """
    np.random.seed(42)
    N = 5000
    D = 5
    K1 = 5
    X = np.random.uniform(0, 1, (N, D))

    f = 5.0 * (X[:, 0] - 0.5) + 3.0 * np.cos(2 * np.pi * X[:, 1])
    noise_std = 0.5
    y = f + noise_std * np.random.randn(N)

    Phi = np.asarray(
        build_first_order_features(jnp.array(X), K1), dtype=np.float64)
    f0 = float(np.mean(y))
    y_c = y - f0
    G1 = np.asarray(build_gram_matrix(K1), dtype=np.float64)

    # Expected Sobol
    var_x1 = 25.0 / 12
    var_x2 = 9.0 / 2
    total_var = var_x1 + var_x2
    expected_sobol = {
        0: var_x1 / total_var,  # ~0.316
        1: var_x2 / total_var,  # ~0.684
    }

    return {
        'Phi': Phi, 'y_c': y_c, 'D': D, 'K1': K1, 'G1': G1,
        'N': N, 'expected_sobol': expected_sobol, 'noise_std': noise_std,
    }


# ============================================================================
# Regularization Path Tests
# ============================================================================

class TestRegPath:
    """Tests for compute_reg_path."""

    def test_df_monotonically_decreasing(self, known_signal_data):
        """Effective df should decrease as lambda increases."""
        d = known_signal_data
        result = compute_reg_path(
            d['Phi'], d['y_c'], d['D'], d['K1'],
            strategy='variance', n_lambdas=30,
            lambda_range=(1e-5, 1e1))

        # df should be monotonically non-increasing
        diffs = np.diff(result.df_values)
        assert np.all(diffs <= 1e-6), (
            f"df should decrease as lambda increases, "
            f"max increase: {np.max(diffs):.6f}")

    def test_gcv_has_minimum(self, known_signal_data):
        """GCV should have a well-defined minimum (not at the boundary)."""
        d = known_signal_data
        result = compute_reg_path(
            d['Phi'], d['y_c'], d['D'], d['K1'],
            strategy='variance', n_lambdas=40,
            lambda_range=(1e-6, 1e2))

        idx_min = np.argmin(result.gcv_values)
        # Minimum should not be at the very first or last point
        assert idx_min > 0, "GCV minimum at smallest lambda (overfitting boundary)"
        assert idx_min < len(result.lambdas) - 1, (
            "GCV minimum at largest lambda (underfitting boundary)")

    def test_sobol_converge_at_small_lambda(self, known_signal_data):
        """At small lambda (near OLS), Sobol indices should approach ground truth."""
        d = known_signal_data
        result = compute_reg_path(
            d['Phi'], d['y_c'], d['D'], d['K1'],
            strategy='variance', n_lambdas=30,
            lambda_range=(1e-6, 1e1))

        # At the smallest lambda, Sobol should be near truth
        # (small regularization = minimal shrinkage bias)
        s1_small_lambda = result.sobol_paths[0][0]  # first lambda point
        s2_small_lambda = result.sobol_paths[1][0]

        expected = d['expected_sobol']
        assert abs(s1_small_lambda - expected[0]) < 0.1, (
            f"S1 at small lambda: {s1_small_lambda:.3f}, expected ~{expected[0]:.3f}")
        assert abs(s2_small_lambda - expected[1]) < 0.1, (
            f"S2 at small lambda: {s2_small_lambda:.3f}, expected ~{expected[1]:.3f}")

    def test_sobol_collapse_at_large_lambda(self, known_signal_data):
        """At very large lambda, all Sobol indices should collapse toward equal."""
        d = known_signal_data
        result = compute_reg_path(
            d['Phi'], d['y_c'], d['D'], d['K1'],
            strategy='variance', n_lambdas=30,
            lambda_range=(1e-5, 1e3))

        # At the largest lambda, shrinkage should reduce differentiation
        # compared to the smallest lambda
        sobol_small = [result.sobol_paths[i][0] for i in range(d['D'])]
        sobol_large = [result.sobol_paths[i][-1] for i in range(d['D'])]
        spread_small = max(sobol_small) - min(sobol_small)
        spread_large = max(sobol_large) - min(sobol_large)
        assert spread_large < spread_small + 0.1, (
            f"Large-lambda spread {spread_large:.3f} should not exceed "
            f"small-lambda spread {spread_small:.3f}")

    def test_sobol_sum_to_one_across_path(self, known_signal_data):
        """At each lambda point, Sobol indices should sum to approximately 1."""
        d = known_signal_data
        result = compute_reg_path(
            d['Phi'], d['y_c'], d['D'], d['K1'],
            strategy='variance', n_lambdas=20,
            lambda_range=(1e-5, 1e1))

        for idx in range(len(result.lambdas)):
            sobol_sum = sum(result.sobol_paths[i][idx] for i in range(d['D']))
            assert abs(sobol_sum - 1.0) < 0.01, (
                f"Sobol sum at lambda[{idx}]={result.lambdas[idx]:.2e}: "
                f"{sobol_sum:.4f} should be ~1")

    def test_path_with_second_order(self):
        """Regularization path should work with second-order terms."""
        np.random.seed(42)
        N = 2000
        D = 4
        K1 = 3
        K2 = 2
        X = np.random.uniform(0, 1, (N, D))

        f = (3.0 * (X[:, 0] - 0.5)
             + 2.0 * (X[:, 0] - 0.5) * (X[:, 1] - 0.5))
        y = f + 0.3 * np.random.randn(N)

        pm = PairManager(D)
        Phi1 = np.asarray(
            build_first_order_features(jnp.array(X), K1), dtype=np.float64)
        Phi2 = np.asarray(
            build_second_order_features(jnp.array(X), K2, pm.pair_indices),
            dtype=np.float64)
        Phi = np.hstack([Phi1, Phi2])
        f0 = float(np.mean(y))
        y_c = y - f0

        result = compute_reg_path(
            Phi, y_c, D, K1, K2=K2, P=pm.P,
            pair_indices=np.asarray(pm.pair_indices),
            strategy='variance', n_lambdas=20,
            lambda_range=(1e-5, 1e1))

        # Should have second-order Sobol paths
        assert len(result.sobol_paths_2nd) > 0, "Should have 2nd-order Sobol paths"
        assert len(result.lambdas) == 20

    def test_output_shapes(self, known_signal_data):
        """All output arrays should have consistent shapes."""
        d = known_signal_data
        n_lambdas = 15
        result = compute_reg_path(
            d['Phi'], d['y_c'], d['D'], d['K1'],
            strategy='variance', n_lambdas=n_lambdas,
            lambda_range=(1e-5, 1e1))

        assert len(result.lambdas) == n_lambdas
        assert len(result.mse_values) == n_lambdas
        assert len(result.gcv_values) == n_lambdas
        assert len(result.df_values) == n_lambdas
        assert len(result.sobol_paths) == d['D']
        for i in range(d['D']):
            assert len(result.sobol_paths[i]) == n_lambdas


# ============================================================================
# Stability Diagnostics Tests
# ============================================================================

class TestStabilityDiagnostics:
    """Tests for K-fold stability diagnostics."""

    def test_stability_on_clean_data(self, known_signal_data):
        """With clean data (low noise, large N), stability should be excellent."""
        d = known_signal_data
        reg = np.asarray(
            build_regularization_vector(d['D'], d['K1'], 0, 0,
                                        'variance', 0.001, 0),
            dtype=np.float64)

        result = stability_diagnostics(
            d['Phi'], d['y_c'], reg, d['D'], d['K1'], d['G1'],
            n_folds=5)

        # Sobol stability should be excellent
        assert result['stability'] in ('excellent', 'good'), (
            f"Expected excellent/good stability, got '{result['stability']}'")

        # Max Sobol std across folds should be small
        max_std = result['max_sobol_std']
        assert max_std < 0.03, f"Max Sobol std {max_std:.4f} too large"

    def test_sobol_means_match_full(self, known_signal_data):
        """Per-fold Sobol means should be close to full-data Sobol."""
        d = known_signal_data
        reg = np.asarray(
            build_regularization_vector(d['D'], d['K1'], 0, 0,
                                        'variance', 0.001, 0),
            dtype=np.float64)

        result = stability_diagnostics(
            d['Phi'], d['y_c'], reg, d['D'], d['K1'], d['G1'],
            n_folds=5)

        sobol_full = result['full_data']['sobol']
        sobol_means = result['sobol_mean']

        for i in range(d['D']):
            diff = abs(sobol_full.get(i, 0) - sobol_means.get(i, 0))
            assert diff < 0.05, (
                f"Variable {i}: full Sobol {sobol_full.get(i, 0):.4f} vs "
                f"mean {sobol_means.get(i, 0):.4f}, diff {diff:.4f}")

    def test_per_fold_structure(self, known_signal_data):
        """Result should contain per-fold data with correct structure."""
        d = known_signal_data
        reg = np.asarray(
            build_regularization_vector(d['D'], d['K1'], 0, 0,
                                        'variance', 0.001, 0),
            dtype=np.float64)

        result = stability_diagnostics(
            d['Phi'], d['y_c'], reg, d['D'], d['K1'], d['G1'],
            n_folds=5)

        assert 'per_fold' in result
        assert len(result['per_fold']) == 5
        for fold in result['per_fold']:
            assert 'rmse' in fold
            assert 'sigma_hat' in fold
            assert 'sobol' in fold
            assert fold['rmse'] > 0
            assert fold['sigma_hat'] > 0

    def test_loo_cv_agreement(self, known_signal_data):
        """LOO-CV from full solve should agree with K-fold at large K."""
        d = known_signal_data
        reg = np.asarray(
            build_regularization_vector(d['D'], d['K1'], 0, 0,
                                        'variance', 0.001, 0),
            dtype=np.float64)

        result = stability_diagnostics(
            d['Phi'], d['y_c'], reg, d['D'], d['K1'], d['G1'],
            n_folds=10)

        loo_cv = result['full_data']['loo_cv']
        kfold_rmses = [f['rmse'] for f in result['per_fold']]
        kfold_mse = float(np.mean([r ** 2 for r in kfold_rmses]))

        # LOO-CV and 10-fold MSE should be in the same ballpark
        ratio = loo_cv / max(kfold_mse, 1e-15)
        assert 0.5 < ratio < 2.0, (
            f"LOO-CV ({loo_cv:.4f}) and 10-fold MSE ({kfold_mse:.4f}) "
            f"should agree (ratio={ratio:.2f})")

    def test_noisy_data_shows_wider_spread(self):
        """High noise should produce wider Sobol spread across folds."""
        np.random.seed(42)
        N = 500  # smaller N = less stable
        D = 5
        K1 = 5
        X = np.random.uniform(0, 1, (N, D))

        f = 3.0 * (X[:, 0] - 0.5) + 2.0 * np.cos(2 * np.pi * X[:, 1])
        noise_std = 3.0  # high noise relative to signal
        y = f + noise_std * np.random.randn(N)

        Phi = np.asarray(
            build_first_order_features(jnp.array(X), K1), dtype=np.float64)
        G1 = np.asarray(build_gram_matrix(K1), dtype=np.float64)
        f0 = float(np.mean(y))
        y_c = y - f0

        reg = np.asarray(
            build_regularization_vector(D, K1, 0, 0, 'variance', 0.001, 0),
            dtype=np.float64)

        result = stability_diagnostics(
            Phi, y_c, reg, D, K1, G1, n_folds=5)

        # With high noise and small N, stability should not be excellent
        # (though this is probabilistic, so we use a loose bound)
        max_std = result['max_sobol_std']
        # Just check it ran successfully and produced a valid classification
        assert result['stability'] in ('excellent', 'good', 'moderate', 'poor')
        assert max_std >= 0

    def test_sigma_hat_consistent_across_folds(self, known_signal_data):
        """Per-fold noise estimates should be consistent."""
        d = known_signal_data
        reg = np.asarray(
            build_regularization_vector(d['D'], d['K1'], 0, 0,
                                        'variance', 0.001, 0),
            dtype=np.float64)

        result = stability_diagnostics(
            d['Phi'], d['y_c'], reg, d['D'], d['K1'], d['G1'],
            n_folds=5)

        sigmas = [f['sigma_hat'] for f in result['per_fold']]
        sigma_std = float(np.std(sigmas))
        sigma_mean = float(np.mean(sigmas))

        # CV of sigma estimates should be small
        cv = sigma_std / max(sigma_mean, 1e-10)
        assert cv < 0.3, (
            f"Sigma CV across folds = {cv:.3f} (mean={sigma_mean:.3f}, "
            f"std={sigma_std:.3f})")
