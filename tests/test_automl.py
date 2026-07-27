"""Tests for analytic AutoML: LOO-CV, sandwich CIs, noise estimation, k-fold."""

import pytest
import jax
import jax.numpy as jnp
import numpy as np

from hifi_anova.core.features import build_first_order_features, build_second_order_features
from hifi_anova.core.gram import build_gram_matrix, build_gram_matrix_2d
from hifi_anova.core.pairs import PairManager
from hifi_anova.training.regularization import build_regularization_vector
from hifi_anova.analysis.automl import (
    ridge_analytics,
    sandwich_covariance,
    sobol_confidence_intervals,
    noise_complexity_curve,
    kfold_cv_analytic,
    sample_size_diagnostics,
)

pytestmark = pytest.mark.smoke


@pytest.fixture
def regression_data():
    """Known signal + known noise for verifiable analytics."""
    np.random.seed(42)
    N = 3000
    D = 5
    K1 = 3
    X = np.random.uniform(0, 1, (N, D))
    noise_std = 0.5

    # Known function: 5*(x1-0.5) + 3*cos(2*pi*x2)
    f = 5.0 * (X[:, 0] - 0.5) + 3.0 * np.cos(2 * np.pi * X[:, 1])
    y = f + noise_std * np.random.randn(N)

    Phi = np.asarray(build_first_order_features(jnp.array(X), K1), dtype=np.float64)
    f0 = float(np.mean(y))
    y_c = y - f0

    reg = np.asarray(build_regularization_vector(D, K1, 0, 0, 'curvature', 0.001, 0),
                      dtype=np.float64)
    G1 = np.asarray(build_gram_matrix(K1), dtype=np.float64)

    return {
        'Phi': Phi, 'y': y_c, 'reg': reg, 'D': D, 'K1': K1, 'G1': G1,
        'noise_std': noise_std, 'N': N,
    }


class TestRidgeAnalytics:
    def test_returns_all_fields(self, regression_data):
        a = ridge_analytics(regression_data['Phi'], regression_data['y'],
                             regression_data['reg'])
        for key in ['w', 'A_inv', 'residuals', 'rss', 'mse', 'df',
                    'sigma2_hat', 'sigma_hat', 'leverages', 'loo_cv',
                    'gcv', 'aic', 'bic', 'ess_per_param']:
            assert key in a, f"Missing key: {key}"

    def test_sigma_hat_near_true(self, regression_data):
        """Noise estimate should be close to true noise std."""
        a = ridge_analytics(regression_data['Phi'], regression_data['y'],
                             regression_data['reg'])
        assert abs(a['sigma_hat'] - regression_data['noise_std']) < 0.1, \
            f"sigma_hat={a['sigma_hat']:.3f}, true={regression_data['noise_std']}"

    def test_leverages_bounded(self, regression_data):
        """Leverages must be in [0, 1)."""
        a = ridge_analytics(regression_data['Phi'], regression_data['y'],
                             regression_data['reg'])
        assert np.all(a['leverages'] >= 0)
        assert np.all(a['leverages'] < 1.0)

    def test_loo_cv_close_to_gcv(self, regression_data):
        """Exact LOO and GCV should be close."""
        a = ridge_analytics(regression_data['Phi'], regression_data['y'],
                             regression_data['reg'])
        ratio = a['loo_cv'] / a['gcv']
        assert 0.8 < ratio < 1.2, f"LOO/GCV ratio = {ratio:.3f}"

    def test_df_positive_and_bounded(self, regression_data):
        a = ridge_analytics(regression_data['Phi'], regression_data['y'],
                             regression_data['reg'])
        assert 0 < a['df'] < regression_data['N']


class TestSandwichCovariance:
    def test_shape(self, regression_data):
        a = ridge_analytics(regression_data['Phi'], regression_data['y'],
                             regression_data['reg'])
        Cov = sandwich_covariance(regression_data['Phi'], a['A_inv'], a['residuals'])
        F = regression_data['Phi'].shape[1]
        assert Cov.shape == (F, F)

    def test_positive_semidefinite(self, regression_data):
        a = ridge_analytics(regression_data['Phi'], regression_data['y'],
                             regression_data['reg'])
        Cov = sandwich_covariance(regression_data['Phi'], a['A_inv'], a['residuals'])
        eigvals = np.linalg.eigvalsh(Cov)
        assert np.all(eigvals >= -1e-10)

    def test_diagonal_positive(self, regression_data):
        a = ridge_analytics(regression_data['Phi'], regression_data['y'],
                             regression_data['reg'])
        Cov = sandwich_covariance(regression_data['Phi'], a['A_inv'], a['residuals'])
        assert np.all(np.diag(Cov) > 0)


class TestSobolConfidenceIntervals:
    def test_active_vars_have_positive_sobol(self, regression_data):
        ci = sobol_confidence_intervals(
            regression_data['Phi'], regression_data['y'], regression_data['reg'],
            regression_data['D'], regression_data['K1'], regression_data['G1'])
        # x1 and x2 should have significant Sobol
        S0, lo0, hi0 = ci['first_order'][0]
        S1, lo1, hi1 = ci['first_order'][1]
        assert S0 > 0.1, f"S(x1) = {S0}"
        assert S1 > 0.1, f"S(x2) = {S1}"

    def test_ci_contains_point_estimate(self, regression_data):
        """Point estimate should be within its own CI."""
        ci = sobol_confidence_intervals(
            regression_data['Phi'], regression_data['y'], regression_data['reg'],
            regression_data['D'], regression_data['K1'], regression_data['G1'])
        for i, (S, lo, hi) in ci['first_order'].items():
            assert lo <= S <= hi, f"x{i+1}: S={S} not in [{lo}, {hi}]"

    def test_inactive_vars_ci_includes_zero(self, regression_data):
        """Irrelevant variables should have CI containing or near zero."""
        ci = sobol_confidence_intervals(
            regression_data['Phi'], regression_data['y'], regression_data['reg'],
            regression_data['D'], regression_data['K1'], regression_data['G1'])
        for i in [2, 3, 4]:  # x3, x4, x5 are irrelevant
            S, lo, hi = ci['first_order'][i]
            assert S < 0.05, f"x{i+1}: S={S} should be near zero"

    def test_ci_width_reasonable(self, regression_data):
        """CI width should be positive and not huge."""
        ci = sobol_confidence_intervals(
            regression_data['Phi'], regression_data['y'], regression_data['reg'],
            regression_data['D'], regression_data['K1'], regression_data['G1'])
        for i, (S, lo, hi) in ci['first_order'].items():
            width = hi - lo
            assert width >= 0
            assert width < 1.0, f"x{i+1}: CI width {width} too large"


class TestNoiseComplexityCurve:
    def test_returns_all_fields(self, regression_data):
        reg_struct = regression_data['reg'] / max(
            np.median(regression_data['reg'][regression_data['reg'] > 1e-15]), 1e-10)
        nc = noise_complexity_curve(regression_data['Phi'], regression_data['y'],
                                     reg_struct, n_lambdas=20)
        for key in ['lambdas', 'sigma2', 'df', 'loo_cv', 'gcv', 'aic', 'bic',
                    'sigma2_min', 'lambda_gcv_opt', 'lambda_bic_opt']:
            assert key in nc, f"Missing key: {key}"

    def test_sigma2_min_near_true(self, regression_data):
        """Minimum of sigma^2(lambda) should estimate true noise."""
        reg_struct = regression_data['reg'] / max(
            np.median(regression_data['reg'][regression_data['reg'] > 1e-15]), 1e-10)
        nc = noise_complexity_curve(regression_data['Phi'], regression_data['y'],
                                     reg_struct, n_lambdas=30)
        true_var = regression_data['noise_std'] ** 2
        assert abs(nc['sigma2_min'] - true_var) < 0.15, \
            f"sigma2_min={nc['sigma2_min']:.3f}, true={true_var:.3f}"


class TestKfoldCVAnalytic:
    def test_returns_cv_stats(self, regression_data):
        result = kfold_cv_analytic(regression_data['Phi'], regression_data['y'],
                                    regression_data['reg'], n_folds=5)
        assert 'cv_mean' in result
        assert 'cv_se' in result
        assert len(result['fold_mses']) == 5

    def test_cv_close_to_loo(self, regression_data):
        """5-fold CV should be close to LOO-CV."""
        a = ridge_analytics(regression_data['Phi'], regression_data['y'],
                             regression_data['reg'])
        kf = kfold_cv_analytic(regression_data['Phi'], regression_data['y'],
                                regression_data['reg'], n_folds=5)
        ratio = kf['cv_mean'] / a['loo_cv']
        assert 0.7 < ratio < 1.5, f"kfold/LOO ratio = {ratio:.3f}"

    def test_more_folds_closer_to_loo(self, regression_data):
        """Both 3-fold and 10-fold CV should be close to LOO-CV."""
        a = ridge_analytics(regression_data['Phi'], regression_data['y'],
                             regression_data['reg'])
        kf3 = kfold_cv_analytic(regression_data['Phi'], regression_data['y'],
                                 regression_data['reg'], n_folds=3)
        kf10 = kfold_cv_analytic(regression_data['Phi'], regression_data['y'],
                                  regression_data['reg'], n_folds=10)
        # Both should be within 20% of LOO
        ratio3 = kf3['cv_mean'] / a['loo_cv']
        ratio10 = kf10['cv_mean'] / a['loo_cv']
        assert 0.7 < ratio3 < 1.5, f"3-fold/LOO ratio = {ratio3:.3f}"
        assert 0.8 < ratio10 < 1.3, f"10-fold/LOO ratio = {ratio10:.3f}"


class TestSampleSizeDiagnostics:
    def test_returns_recommendation(self, regression_data):
        diag = sample_size_diagnostics(
            regression_data['Phi'], regression_data['y'], regression_data['reg'],
            regression_data['D'], regression_data['K1'], regression_data['G1'])
        assert 'recommendation' in diag
        assert 'per_variable' in diag
        assert 'order1' in diag

    def test_ess_per_param_positive(self, regression_data):
        diag = sample_size_diagnostics(
            regression_data['Phi'], regression_data['y'], regression_data['reg'],
            regression_data['D'], regression_data['K1'], regression_data['G1'])
        assert diag['order1']['ess_per_param'] > 10
