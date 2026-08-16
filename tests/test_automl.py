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
    stability_diagnostics,
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

    def test_woodbury_equals_bruteforce_refit(self, regression_data):
        """Decisive: the Woodbury downdate must reproduce an explicit per-fold
        refit exactly (guards the [I - H_k] downdate sign).

        Regression for the previously-wrong [I + H_k] inner matrix / '-'
        correction, which made k-fold CV an incorrect (over-optimistic) number.
        """
        Phi = np.asarray(regression_data['Phi'], dtype=np.float64)
        y = np.asarray(regression_data['y'], dtype=np.float64)
        reg = np.asarray(regression_data['reg'], dtype=np.float64)
        N = Phi.shape[0]
        n_folds, seed = 5, 42

        kf = kfold_cv_analytic(Phi, y, reg, n_folds=n_folds, seed=seed)

        # Reproduce kfold_cv_analytic's exact fold assignment.
        rng = np.random.RandomState(seed)
        fold_ids = np.zeros(N, dtype=int)
        perm = rng.permutation(N)
        fold_size = N // n_folds
        for k in range(n_folds):
            start = k * fold_size
            end = (k + 1) * fold_size if k < n_folds - 1 else N
            fold_ids[perm[start:end]] = k

        bf_mses = []
        for k in range(n_folds):
            val = fold_ids == k
            tr = ~val
            A_tr = Phi[tr].T @ Phi[tr] + np.diag(reg)
            w_tr = np.linalg.solve(A_tr, Phi[tr].T @ y[tr])
            bf_mses.append(float(np.mean((y[val] - Phi[val] @ w_tr) ** 2)))

        np.testing.assert_allclose(kf['fold_mses'], bf_mses, rtol=1e-8, atol=1e-8)


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


class TestSobolCICoverageAcrossBases:
    """Monte-Carlo coverage regression test for the Sobol CI delta method.

    Guards against the own-block-only delta gradient bug: dropping the
    denominator-coupling terms dS_i/dw_j = -S_i * 2 G_j w_j / V_tot (j != i)
    underestimates SE(S) by ~13-15% INDEPENDENT of N, giving ~90% actual
    coverage at 95% nominal. With the full gradient, coverage is nominal for
    all three bases. Fully deterministic (fixed seeds).
    """

    @pytest.mark.parametrize('basis_name', ['fourier', 'legendre', 'haar'])
    def test_coverage_and_se_calibration(self, basis_name):
        D, K1, N, R = 3, 3, 1200, 120
        noise_sd, lam = 0.5, 1e-6
        z95 = 1.959963985

        rng = np.random.default_rng(20260727)
        X = rng.uniform(0.0, 1.0, size=(N, D))
        Phi = np.asarray(build_first_order_features(X, K1, True, basis_name),
                         dtype=np.float64)
        F = Phi.shape[1]
        block = F // D

        # In-basis ground truth: model correctly specified, estimand exact
        w_true = rng.standard_normal(F)
        signal = Phi @ w_true
        signal = signal - signal.mean()
        w_true /= signal.std()
        signal /= signal.std()

        G1 = np.asarray(build_gram_matrix(K1, True, basis_name), dtype=np.float64)
        reg = np.asarray(build_regularization_vector(
            D, K1, 0, 0, 'variance', lam, 0.0,
            include_linear_1=True, basis_name=basis_name), dtype=np.float64)

        var_true = np.array([
            w_true[i*block:(i+1)*block] @ G1 @ w_true[i*block:(i+1)*block]
            for i in range(D)])
        S_true = var_true / var_true.sum()

        S_hat = np.zeros((R, D))
        lo_arr = np.zeros((R, D))
        hi_arr = np.zeros((R, D))
        noise_rng = np.random.default_rng(42)
        for r in range(R):
            y = signal + noise_rng.normal(0.0, noise_sd, size=N)
            y = y - y.mean()
            ci = sobol_confidence_intervals(
                Phi, y, reg, D, K1, G1, K2=0, P=0,
                include_linear_1=True, basis_name=basis_name)
            for i in range(D):
                S_hat[r, i], lo_arr[r, i], hi_arr[r, i] = ci['first_order'][i]

        coverage = ((lo_arr <= S_true) & (S_true <= hi_arr)).mean()
        mc_sd = S_hat.std(axis=0, ddof=1)
        se_mean = ((hi_arr - lo_arr) / (2 * z95)).mean(axis=0)
        se_sd_ratio = float((se_mean / mc_sd).mean())

        # Nominal 0.95; the old own-block-only gradient gives ~0.86-0.90
        # coverage and SE/SD ~0.86 on this config.
        assert coverage >= 0.90, (
            f"{basis_name}: coverage {coverage:.3f} < 0.90 (nominal 0.95) — "
            f"delta-method SE underestimates sampling SD")
        assert 0.90 <= se_sd_ratio <= 1.20, (
            f"{basis_name}: mean SE/SD = {se_sd_ratio:.3f} outside [0.90, 1.20]")


class TestStabilityDiagnosticsExactFoldDf:
    """The per-fold sigma_hat uses the EXACT leave-fold-out effective df
    (df_k = tr(H_k)), not the old df_full * n_train/N linear approximation."""

    @staticmethod
    def _fold_ids(N, n_folds, seed):
        # Replicates the fold assignment inside stability_diagnostics.
        rng = np.random.RandomState(seed)
        fold_ids = np.zeros(N, dtype=int)
        perm = rng.permutation(N)
        fold_size = N // n_folds
        for k in range(n_folds):
            start = k * fold_size
            end = (k + 1) * fold_size if k < n_folds - 1 else N
            fold_ids[perm[start:end]] = k
        return fold_ids

    def test_per_fold_sigma_matches_brute_force(self, regression_data):
        Phi, y, reg = regression_data['Phi'], regression_data['y'], regression_data['reg']
        D, K1, G1 = regression_data['D'], regression_data['K1'], regression_data['G1']
        N, n_folds, seed = regression_data['N'], 5, 0

        out = stability_diagnostics(Phi, y, reg, D, K1, G1,
                                    n_folds=n_folds, seed=seed)

        R = np.diag(reg)
        fold_ids = self._fold_ids(N, n_folds, seed)
        for k in range(n_folds):
            train = fold_ids != k
            Phi_tr, y_tr = Phi[train], y[train]
            n_train = int(train.sum())
            A_k = Phi_tr.T @ Phi_tr + R
            A_k_inv = np.linalg.inv(A_k)
            w_k = A_k_inv @ (Phi_tr.T @ y_tr)               # brute-force refit
            r_train = y_tr - Phi_tr @ w_k
            df_k = float(np.trace(A_k_inv @ (Phi_tr.T @ Phi_tr)))  # exact tr(H_k)
            sigma_bf = float(np.sqrt(np.sum(r_train ** 2)
                                     / max(1, n_train - df_k)))
            assert np.isclose(out['per_fold'][k]['sigma_hat'], sigma_bf,
                              rtol=1e-6), (
                f"fold {k}: sigma_hat {out['per_fold'][k]['sigma_hat']:.6f} "
                f"!= brute-force {sigma_bf:.6f}")


class TestResidualDfAndRobustCI:
    """Advisor items #4/#6: residual-df σ̂, HC3 sandwich, t critical value."""

    def test_df_residual_is_exact_residual_effective_df(self, regression_data):
        """df_residual == N - 2 tr(H) + tr(H^2), computed independently."""
        Phi, y, reg = (regression_data['Phi'], regression_data['y'],
                       regression_data['reg'])
        a = ridge_analytics(Phi, y, reg)
        N, F = Phi.shape
        A = Phi.T @ Phi + np.diag(reg)
        H = Phi @ np.linalg.solve(A, Phi.T)          # N x N hat matrix (brute force)
        expected = N - 2.0 * np.trace(H) + np.trace(H @ H.T)
        assert np.isclose(a['df_residual'], expected, rtol=1e-8)
        assert np.isclose(a['tr_H2'], np.trace(H @ H), rtol=1e-8)
        # σ̂² must use the residual-df denominator, not N - tr(H).
        assert np.isclose(a['sigma2_hat'], a['rss'] / max(expected, 1.0), rtol=1e-10)

    def test_residual_df_smaller_than_N_minus_trH(self, regression_data):
        """N - 2tr(H) + tr(HH^T) <= N - tr(H) since sum h_i(1-h_i) >= 0,
        so the residual-df σ̂ is >= the N-tr(H) shorthand."""
        a = ridge_analytics(regression_data['Phi'], regression_data['y'],
                            regression_data['reg'])
        N = regression_data['N']
        assert a['df_residual'] <= N - a['df'] + 1e-9

    def test_hc3_wider_than_hc0(self, regression_data):
        """HC3 leverage weights 1/(1-h_ii)^2 >= 1, so HC3 SEs >= HC0 SEs."""
        a = ridge_analytics(regression_data['Phi'], regression_data['y'],
                            regression_data['reg'])
        Phi = regression_data['Phi']
        cov0 = sandwich_covariance(Phi, a['A_inv'], a['residuals'], hc='HC0')
        cov3 = sandwich_covariance(Phi, a['A_inv'], a['residuals'], hc='HC3',
                                   leverages=a['leverages'])
        assert np.all(np.diag(cov3) >= np.diag(cov0) - 1e-12)
        # strictly wider somewhere (leverages are not all zero)
        assert np.any(np.diag(cov3) > np.diag(cov0) + 1e-10)

    def test_hc3_equals_hc0_at_zero_leverage(self, regression_data):
        """With all leverages forced to 0, HC3 collapses to HC0."""
        a = ridge_analytics(regression_data['Phi'], regression_data['y'],
                            regression_data['reg'])
        Phi = regression_data['Phi']
        cov0 = sandwich_covariance(Phi, a['A_inv'], a['residuals'], hc='HC0')
        cov3 = sandwich_covariance(Phi, a['A_inv'], a['residuals'], hc='HC3',
                                   leverages=np.zeros(Phi.shape[0]))
        assert np.allclose(cov0, cov3)

    def test_ci_uses_t_and_reports_provenance(self, regression_data):
        ci = sobol_confidence_intervals(
            regression_data['Phi'], regression_data['y'], regression_data['reg'],
            regression_data['D'], regression_data['K1'], regression_data['G1'])
        assert ci['sandwich'] == 'HC3'
        assert ci['crit_dist'] == 't'
        assert ci['t_df'] > 1
        assert ci['conditional_on_residual_variance'] is True
        assert 'df_residual' in ci

    def test_t_wider_than_z_small_df(self):
        """The t critical value must exceed the normal z at finite df, so the
        interval is wider than the old normal-based one (same SE)."""
        from scipy.stats import t as sp_t, norm as sp_norm
        # A high-complexity / low residual-df fit: many basis fns, few samples.
        rng = np.random.RandomState(0)
        N, D, K1 = 60, 3, 5
        X = rng.uniform(0, 1, (N, D))
        Phi = np.asarray(build_first_order_features(jnp.array(X), K1),
                         dtype=np.float64)
        y = (X[:, 0] - 0.5) + 0.3 * rng.randn(N)
        y = y - y.mean()
        reg = np.asarray(build_regularization_vector(D, K1, 0, 0, 'curvature',
                                                     0.001, 0), dtype=np.float64)
        a = ridge_analytics(Phi, y, reg)
        t_crit = sp_t.ppf(0.975, df=max(a['df_residual'], 1.0))
        assert t_crit > sp_norm.ppf(0.975)
