"""Tests for analytic hyperparameter optimization (ridge regression).

Covers:
  - ridge_solve_with_diagnostics: diagnostics at fixed reg
  - optimize_single_lambda: scalar lambda optimization
  - optimize_multi_lambda: two-lambda joint optimization
  - optimize_multi_lambda_extended: n-lambda routing and optimization

All test data is JAX-free where possible; Phi matrices are either pure
Gaussian random or built via the Fourier feature pipeline.
"""

import numpy as np
import pytest

from hifi_anova.training.hyperopt import (
    ridge_solve_with_diagnostics,
    optimize_single_lambda,
    optimize_multi_lambda,
    optimize_multi_lambda_extended,
)
from hifi_anova.training.regularization import build_regularization_vector
from hifi_anova.core.features import basis_size


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_phi_y(N=300, F=20, noise_std=0.3, seed=42):
    """Gaussian Phi and a simple linear signal + noise.

    Returns Phi (N, F), y_centered (N,), w_true (F,).
    y is centered (mean subtracted) as required by the hyperopt functions.
    """
    rng = np.random.RandomState(seed)
    Phi = rng.randn(N, F)
    # sparse true weights so signal-to-noise ratio is clear
    w_true = np.zeros(F)
    w_true[:5] = rng.randn(5)
    y_raw = Phi @ w_true + noise_std * rng.randn(N)
    y = y_raw - y_raw.mean()
    return Phi, y, w_true


def _make_fourier_phi_y(N=300, D=3, K=3, noise_std=0.3, seed=42):
    """Build Phi via the actual Fourier pipeline.

    Returns Phi (N, F) as float64 numpy array, y_centered (N,).
    Avoids JAX in the test body itself.
    """
    import jax.numpy as jnp
    from hifi_anova.core.features import build_first_order_features

    rng = np.random.RandomState(seed)
    X = rng.uniform(0, 1, (N, D))
    x_jax = jnp.array(X, dtype=jnp.float64)

    Phi_jax = build_first_order_features(x_jax, K)
    Phi = np.asarray(Phi_jax, dtype=np.float64)  # (N, D*(2K+1))

    F = Phi.shape[1]
    w_true = np.zeros(F)
    w_true[:3] = [1.5, -0.8, 0.5]
    y_raw = Phi @ w_true + noise_std * rng.randn(N)
    y = y_raw - y_raw.mean()
    return Phi, y, w_true


# ---------------------------------------------------------------------------
# TestRidgeSolveWithDiagnostics
# ---------------------------------------------------------------------------

class TestRidgeSolveWithDiagnostics:
    """Unit tests for ridge_solve_with_diagnostics."""

    def _run(self, Phi, y, reg_diag):
        return ridge_solve_with_diagnostics(
            np.asarray(Phi, dtype=np.float64),
            np.asarray(y, dtype=np.float64),
            np.asarray(reg_diag, dtype=np.float64),
        )

    def test_output_keys(self):
        """All expected keys must be present in the returned dict."""
        Phi, y, _ = _make_phi_y(N=200, F=15)
        reg_diag = np.full(15, 0.1)
        result = self._run(Phi, y, reg_diag)
        expected_keys = {'w', 'rss', 'mse', 'df', 'gcv', 'aic', 'bic',
                         'log_evidence', 'sigma2_ml'}
        assert expected_keys == set(result.keys()), (
            f"Key mismatch. Got: {set(result.keys())}, expected: {expected_keys}"
        )

    def test_ols_limit(self):
        """Near-zero regularization → df ≈ F and w ≈ OLS solution."""
        rng = np.random.RandomState(0)
        N, F = 300, 15
        Phi = rng.randn(N, F)
        y = rng.randn(N)
        y -= y.mean()

        reg_diag = np.full(F, 1e-10)
        result = self._run(Phi, y, reg_diag)

        # Effective df should be close to F (N > F)
        assert abs(result['df'] - F) < 0.5, (
            f"Near-OLS df={result['df']:.3f} should be ≈ F={F}"
        )

        # w should match numpy lstsq
        w_lstsq, _, _, _ = np.linalg.lstsq(Phi, y, rcond=None)
        np.testing.assert_allclose(result['w'], w_lstsq, rtol=1e-4, atol=1e-6,
                                   err_msg="Near-OLS w should match lstsq")

    def test_heavy_regularization(self):
        """Very heavy regularization → w ≈ 0, df ≈ 0, rss ≈ ||y||²."""
        rng = np.random.RandomState(1)
        N, F = 250, 12
        Phi = rng.randn(N, F)
        y = rng.randn(N)
        y -= y.mean()

        reg_diag = np.full(F, 1e8)
        result = self._run(Phi, y, reg_diag)

        # Weights should be near zero
        assert np.max(np.abs(result['w'])) < 1e-3, (
            f"Heavy reg: max |w|={np.max(np.abs(result['w'])):.2e} should be ≈ 0"
        )

        # df should be near zero
        assert result['df'] < 0.1, (
            f"Heavy reg: df={result['df']:.4f} should be ≈ 0"
        )

        # RSS should be close to ||y||² (all residual)
        rss_full = float(np.sum(y**2))
        assert abs(result['rss'] - rss_full) / rss_full < 0.01, (
            f"Heavy reg: rss={result['rss']:.4f} should be ≈ ||y||²={rss_full:.4f}"
        )

    def test_df_bounds(self):
        """0 < df < min(N, F) for any positive regularization."""
        rng = np.random.RandomState(2)
        N, F = 300, 20
        Phi = rng.randn(N, F)
        y = rng.randn(N)
        y -= y.mean()

        for lam in [1e-4, 0.01, 1.0, 100.0]:
            reg_diag = np.full(F, lam)
            result = self._run(Phi, y, reg_diag)
            assert result['df'] > 0, f"df must be positive at lam={lam}"
            assert result['df'] < min(N, F), (
                f"df={result['df']:.3f} must be < min(N,F)={min(N,F)} at lam={lam}"
            )

    def test_gcv_formula(self):
        """GCV = mse / (1 - (df + INTERCEPT_DF)/N)² — verify manually.

        The criterion counts the profiled (centered-out) intercept via
        INTERCEPT_DF; the returned 'df' stays tr(H).
        """
        from hifi_anova.training.hyperopt import INTERCEPT_DF
        rng = np.random.RandomState(3)
        N, F = 280, 18
        Phi = rng.randn(N, F)
        y = rng.randn(N)
        y -= y.mean()
        reg_diag = np.full(F, 0.5)

        result = self._run(Phi, y, reg_diag)

        df_sel = result['df'] + INTERCEPT_DF
        gcv_manual = result['mse'] / max(1e-10, (1.0 - df_sel / N))**2
        np.testing.assert_allclose(
            result['gcv'], gcv_manual, rtol=1e-8,
            err_msg=f"GCV={result['gcv']:.6f} vs manual={gcv_manual:.6f}"
        )

    def test_aic_bic_formula(self):
        """AIC/BIC = N*log(mse) + pen*(df + INTERCEPT_DF) (profiled intercept)."""
        rng = np.random.RandomState(4)
        N, F = 260, 16
        Phi = rng.randn(N, F)
        y = rng.randn(N)
        y -= y.mean()
        reg_diag = np.full(F, 0.2)

        result = self._run(Phi, y, reg_diag)

        from hifi_anova.training.hyperopt import INTERCEPT_DF
        df_sel = result['df'] + INTERCEPT_DF
        aic_manual = N * np.log(max(result['mse'], 1e-15)) + 2.0 * df_sel
        bic_manual = N * np.log(max(result['mse'], 1e-15)) + np.log(N) * df_sel

        np.testing.assert_allclose(result['aic'], aic_manual, rtol=1e-8,
                                   err_msg="AIC formula mismatch")
        np.testing.assert_allclose(result['bic'], bic_manual, rtol=1e-8,
                                   err_msg="BIC formula mismatch")

    def test_evidence_negative(self):
        """log_evidence should be negative for typical regression problems."""
        rng = np.random.RandomState(5)
        N, F = 300, 20
        Phi = rng.randn(N, F)
        y = rng.randn(N)
        y -= y.mean()
        reg_diag = np.full(F, 1.0)

        result = self._run(Phi, y, reg_diag)
        assert result['log_evidence'] < 0, (
            f"log_evidence={result['log_evidence']:.4f} should be negative"
        )

    def test_known_signal_recovery(self):
        """With N >> F and moderate reg, recovered w should be close to w_true."""
        rng = np.random.RandomState(6)
        N, F = 500, 10
        Phi = rng.randn(N, F)
        # Normalize columns for better conditioning
        Phi /= np.linalg.norm(Phi, axis=0, keepdims=True)
        w_true = rng.randn(F)
        noise_std = 0.1
        y = Phi @ w_true + noise_std * rng.randn(N)
        y -= y.mean()

        reg_diag = np.full(F, 0.01)
        result = self._run(Phi, y, reg_diag)

        corr = np.corrcoef(result['w'], w_true)[0, 1]
        assert corr > 0.95, (
            f"Signal recovery: correlation={corr:.4f} should be > 0.95"
        )

    def test_mse_equals_rss_over_n(self):
        """mse = rss / N by definition."""
        rng = np.random.RandomState(7)
        N, F = 220, 14
        Phi = rng.randn(N, F)
        y = rng.randn(N)
        y -= y.mean()
        reg_diag = np.full(F, 0.3)

        result = self._run(Phi, y, reg_diag)
        np.testing.assert_allclose(result['mse'], result['rss'] / N, rtol=1e-10,
                                   err_msg="mse should equal rss/N")

    def test_sigma2_ml_equals_mse(self):
        """sigma2_ml should equal mse (clamped at 1e-15)."""
        Phi, y, _ = _make_phi_y(N=300, F=15, noise_std=0.5)
        reg_diag = np.full(15, 0.1)
        result = self._run(Phi, y, reg_diag)
        expected = max(result['mse'], 1e-15)
        np.testing.assert_allclose(result['sigma2_ml'], expected, rtol=1e-10)


# ---------------------------------------------------------------------------
# TestOptimizeSingleLambda
# ---------------------------------------------------------------------------

class TestOptimizeSingleLambda:
    """Tests for optimize_single_lambda."""

    def _reg_struct(self, F):
        """Uniform regularization structure (ones vector)."""
        return np.ones(F)

    def test_output_keys(self):
        """Result must contain lambda_opt, converged, and all diagnostic keys."""
        Phi, y, _ = _make_phi_y(N=250, F=15)
        reg_struct = self._reg_struct(15)
        result = optimize_single_lambda(Phi, y, reg_struct, method='gcv')
        for key in ['lambda_opt', 'converged', 'w', 'rss', 'mse', 'df',
                    'gcv', 'aic', 'bic', 'log_evidence', 'sigma2_ml']:
            assert key in result, f"Missing key: {key}"

    def test_finds_reasonable_lambda(self):
        """Optimal lambda should not sit at the search boundaries."""
        Phi, y, _ = _make_phi_y(N=300, F=20, noise_std=0.3, seed=10)
        reg_struct = self._reg_struct(20)
        bounds = (1e-6, 1e2)

        result = optimize_single_lambda(Phi, y, reg_struct, method='gcv',
                                        bounds=bounds)
        lam = result['lambda_opt']
        lo, hi = bounds
        # Not stuck at either boundary (allow 10x tolerance from bounds)
        assert lam > lo * 10, (
            f"lambda_opt={lam:.2e} is too close to lower bound {lo}"
        )
        assert lam < hi / 10, (
            f"lambda_opt={lam:.2e} is too close to upper bound {hi}"
        )

    def test_gcv_at_optimum_is_local_min(self):
        """GCV at lambda_opt should be <= GCV at lambda_opt/2 and lambda_opt*2."""
        Phi, y, _ = _make_phi_y(N=300, F=20, noise_std=0.3, seed=11)
        reg_struct = self._reg_struct(20)
        bounds = (1e-5, 1e2)

        result = optimize_single_lambda(Phi, y, reg_struct, method='gcv',
                                        bounds=bounds)
        lam_opt = result['lambda_opt']
        gcv_opt = result['gcv']

        reg_diag = reg_struct  # structure is ones → reg_diag = lam * ones
        for factor in [0.5, 2.0]:
            lam_nearby = lam_opt * factor
            # Clamp to bounds to avoid edge effects
            lam_nearby = np.clip(lam_nearby, bounds[0], bounds[1])
            diag_nearby = ridge_solve_with_diagnostics(
                Phi, y, lam_nearby * reg_struct
            )
            assert gcv_opt <= diag_nearby['gcv'] + 1e-6, (
                f"GCV at optimum ({gcv_opt:.6f}) should be <= GCV at "
                f"lam*{factor} ({diag_nearby['gcv']:.6f})"
            )

    def test_score_consistency(self):
        """Re-evaluate at lambda_opt; GCV should match what optimize returns."""
        Phi, y, _ = _make_phi_y(N=280, F=15, seed=12)
        reg_struct = self._reg_struct(15)
        result = optimize_single_lambda(Phi, y, reg_struct, method='gcv')

        lam_opt = result['lambda_opt']
        reeval = ridge_solve_with_diagnostics(Phi, y, lam_opt * reg_struct)
        np.testing.assert_allclose(
            result['gcv'], reeval['gcv'], rtol=1e-6,
            err_msg="GCV from optimize_single_lambda does not match re-evaluation"
        )

    def test_aic_method(self):
        """AIC method should run without error and return finite lambda_opt."""
        Phi, y, _ = _make_phi_y(N=280, F=15, seed=13)
        reg_struct = self._reg_struct(15)
        result = optimize_single_lambda(Phi, y, reg_struct, method='aic')
        assert np.isfinite(result['lambda_opt']), "lambda_opt must be finite"
        assert result['lambda_opt'] > 0, "lambda_opt must be positive"

    def test_bic_method(self):
        """BIC method should run without error and return finite lambda_opt."""
        Phi, y, _ = _make_phi_y(N=280, F=15, seed=14)
        reg_struct = self._reg_struct(15)
        result = optimize_single_lambda(Phi, y, reg_struct, method='bic')
        assert np.isfinite(result['lambda_opt']), "lambda_opt must be finite"
        assert result['lambda_opt'] > 0, "lambda_opt must be positive"

    def test_fourier_phi(self):
        """Works end-to-end with an actual Fourier feature matrix."""
        Phi, y, _ = _make_fourier_phi_y(N=300, D=3, K=3, seed=20)
        F = Phi.shape[1]
        reg_struct = np.ones(F)
        result = optimize_single_lambda(Phi, y, reg_struct, method='gcv',
                                        bounds=(1e-6, 1e1))
        assert 'lambda_opt' in result
        assert result['lambda_opt'] > 0

    def test_converged_flag(self):
        """Result should always have converged=True (scalar optimize_scalar always converges)."""
        Phi, y, _ = _make_phi_y(N=250, F=12, seed=15)
        reg_struct = self._reg_struct(12)
        result = optimize_single_lambda(Phi, y, reg_struct)
        assert result['converged'] is True


# ---------------------------------------------------------------------------
# TestOptimizeMultiLambda
# ---------------------------------------------------------------------------

class TestOptimizeMultiLambda:
    """Tests for optimize_multi_lambda."""

    def _make_two_order_phi_y(self, N=300, D=3, K1=3, K2=2, seed=42):
        """Build [Phi1 | Phi2] from actual Fourier basis, return numpy arrays."""
        import jax.numpy as jnp
        from hifi_anova.core.features import build_first_order_features, build_second_order_features
        from hifi_anova.core.pairs import PairManager

        rng = np.random.RandomState(seed)
        X = rng.uniform(0, 1, (N, D))
        x_jax = jnp.array(X, dtype=jnp.float64)

        pm = PairManager(D)
        phi1 = np.asarray(build_first_order_features(x_jax, K1), dtype=np.float64)
        phi2 = np.asarray(build_second_order_features(x_jax, K2, pm.pair_indices),
                          dtype=np.float64)
        Phi = np.concatenate([phi1, phi2], axis=1)

        # Signal: pure first-order so second-order should get heavier penalty
        F1 = phi1.shape[1]
        w_true = np.zeros(Phi.shape[1])
        w_true[:3] = [1.0, -0.5, 0.3]
        y_raw = Phi @ w_true + 0.3 * rng.randn(N)
        y = y_raw - y_raw.mean()
        return Phi, y, pm.P

    def test_output_keys(self):
        """Result must include lambda_order1, lambda_order2, converged."""
        Phi, y, P = self._make_two_order_phi_y()
        D, K1, K2 = 3, 3, 2
        result = optimize_multi_lambda(Phi, y, D=D, K1=K1, K2=K2, P=P)
        for key in ['lambda_order1', 'lambda_order2', 'converged']:
            assert key in result, f"Missing key: {key}"

    def test_fallback_when_no_second_order(self):
        """K2=0 → falls back to optimize_single_lambda, returns lambda_opt."""
        Phi, y, _ = _make_phi_y(N=250, F=21)  # 21 = D*(2K+1) with D=3,K=3
        D, K1 = 3, 3
        # Recompute Phi with Fourier pipeline
        Phi_f, y_f, _ = _make_fourier_phi_y(N=250, D=D, K=K1, seed=30)
        result = optimize_multi_lambda(Phi_f, y_f, D=D, K1=K1, K2=0, P=0)
        assert 'lambda_opt' in result, (
            "K2=0 fallback should return lambda_opt (from optimize_single_lambda)"
        )

    def test_both_lambdas_within_bounds(self):
        """Both lambda_order1 and lambda_order2 must be within search bounds."""
        Phi, y, P = self._make_two_order_phi_y(N=300, D=3, K1=3, K2=2, seed=31)
        D, K1, K2 = 3, 3, 2
        bounds = (1e-6, 1e2)
        result = optimize_multi_lambda(Phi, y, D=D, K1=K1, K2=K2, P=P,
                                       bounds=bounds)
        lo, hi = bounds
        assert result['lambda_order1'] >= lo, (
            f"lambda_order1={result['lambda_order1']:.2e} below lower bound {lo}"
        )
        assert result['lambda_order1'] <= hi, (
            f"lambda_order1={result['lambda_order1']:.2e} above upper bound {hi}"
        )
        assert result['lambda_order2'] >= lo, (
            f"lambda_order2={result['lambda_order2']:.2e} below lower bound {lo}"
        )
        assert result['lambda_order2'] <= hi, (
            f"lambda_order2={result['lambda_order2']:.2e} above upper bound {hi}"
        )

    def test_first_order_data_prefers_high_lambda2(self):
        """Pure first-order signal → optimizer should find lambda_order2 >= lambda_order1.

        With no second-order signal, the optimizer should not benefit from a
        small second-order lambda, so lambda_order2 should be at least as large
        as lambda_order1 (heavy penalty on useless second-order terms).
        """
        import jax.numpy as jnp
        from hifi_anova.core.features import build_first_order_features, build_second_order_features
        from hifi_anova.core.pairs import PairManager

        rng = np.random.RandomState(32)
        N, D, K1, K2 = 400, 3, 3, 2
        X = rng.uniform(0, 1, (N, D))
        x_jax = jnp.array(X, dtype=jnp.float64)

        pm = PairManager(D)
        phi1 = np.asarray(build_first_order_features(x_jax, K1), dtype=np.float64)
        phi2 = np.asarray(build_second_order_features(x_jax, K2, pm.pair_indices),
                          dtype=np.float64)
        Phi = np.concatenate([phi1, phi2], axis=1)

        # Signal: strictly first-order only
        y_raw = (X[:, 0] - 0.5) * 2.0 + np.cos(2 * np.pi * X[:, 1]) * 1.5
        y_raw += 0.1 * rng.randn(N)
        y = y_raw - y_raw.mean()

        result = optimize_multi_lambda(Phi, y, D=D, K1=K1, K2=K2, P=pm.P,
                                       method='gcv', bounds=(1e-6, 1e2))
        lam1 = result['lambda_order1']
        lam2 = result['lambda_order2']
        # lambda_order2 should be at least as large as lambda_order1
        assert lam2 >= lam1 * 0.5, (
            f"For first-order signal, lambda_order2={lam2:.2e} should be "
            f">= lambda_order1={lam1:.2e} (within factor 2)"
        )

    def test_diagnostics_present(self):
        """Full diagnostic keys should be in result for two-lambda case."""
        Phi, y, P = self._make_two_order_phi_y(N=300, D=3, K1=3, K2=2, seed=33)
        D, K1, K2 = 3, 3, 2
        result = optimize_multi_lambda(Phi, y, D=D, K1=K1, K2=K2, P=P)
        for key in ['w', 'gcv', 'df', 'rss', 'mse']:
            assert key in result, f"Diagnostic key '{key}' missing from two-lambda result"


# ---------------------------------------------------------------------------
# TestOptimizeMultiLambdaExtended
# ---------------------------------------------------------------------------

class TestOptimizeMultiLambdaExtended:
    """Tests for optimize_multi_lambda_extended."""

    def _make_first_order_phi_y(self, N=300, D=3, K1=3, seed=42):
        """First-order Fourier Phi only."""
        import jax.numpy as jnp
        from hifi_anova.core.features import build_first_order_features

        rng = np.random.RandomState(seed)
        X = rng.uniform(0, 1, (N, D))
        x_jax = jnp.array(X, dtype=jnp.float64)
        Phi = np.asarray(build_first_order_features(x_jax, K1), dtype=np.float64)
        y_raw = np.cos(2 * np.pi * X[:, 0]) + 0.3 * rng.randn(N)
        y = y_raw - y_raw.mean()
        return Phi, y

    def _make_two_order_phi_y(self, N=300, D=3, K1=3, K2=2, seed=42):
        """[Phi1 | Phi2] from Fourier basis."""
        import jax.numpy as jnp
        from hifi_anova.core.features import build_first_order_features, build_second_order_features
        from hifi_anova.core.pairs import PairManager

        rng = np.random.RandomState(seed)
        X = rng.uniform(0, 1, (N, D))
        x_jax = jnp.array(X, dtype=jnp.float64)
        pm = PairManager(D)
        phi1 = np.asarray(build_first_order_features(x_jax, K1), dtype=np.float64)
        phi2 = np.asarray(build_second_order_features(x_jax, K2, pm.pair_indices),
                          dtype=np.float64)
        Phi = np.concatenate([phi1, phi2], axis=1)
        y_raw = np.cos(2 * np.pi * X[:, 0]) + 0.3 * rng.randn(N)
        y = y_raw - y_raw.mean()
        return Phi, y, pm.P

    def test_single_lambda_routing(self):
        """K2=0, K3=0, M_residual=0 → n_lambdas_optimized=1, lambda_order1 present."""
        Phi, y = self._make_first_order_phi_y(N=250, D=3, K1=3, seed=40)
        result = optimize_multi_lambda_extended(
            Phi, y, D=3, K1=3, K2=0, P=0, K3=0, T=0, M_residual=0,
            method='gcv', bounds=(1e-6, 1e2), verbose=False,
        )
        # For n_active=1, falls back to optimize_single_lambda which sets lambda_order1
        assert 'lambda_order1' in result, "lambda_order1 should be present for 1-lambda routing"
        assert np.isfinite(result['lambda_order1']), "lambda_order1 must be finite"

    def test_two_lambda_routing(self):
        """K2>0, P>0, K3=0, M_residual=0 → n_lambdas_optimized=2."""
        Phi, y, P = self._make_two_order_phi_y(N=300, D=3, K1=3, K2=2, seed=41)
        result = optimize_multi_lambda_extended(
            Phi, y, D=3, K1=3, K2=2, P=P, K3=0, T=0, M_residual=0,
            method='gcv', bounds=(1e-6, 1e2), verbose=False,
        )
        # Falls back to optimize_multi_lambda which sets lambda_order1 and lambda_order2
        assert 'lambda_order1' in result, "lambda_order1 should be present for 2-lambda routing"
        assert 'lambda_order2' in result, "lambda_order2 should be present for 2-lambda routing"

    def test_all_lambdas_within_bounds_single(self):
        """Single-lambda route: lambda_order1 within search bounds."""
        Phi, y = self._make_first_order_phi_y(N=250, D=3, K1=3, seed=42)
        bounds = (1e-6, 1e2)
        result = optimize_multi_lambda_extended(
            Phi, y, D=3, K1=3, K2=0, P=0, K3=0, T=0, M_residual=0,
            method='gcv', bounds=bounds, verbose=False,
        )
        lo, hi = bounds
        assert result['lambda_order1'] >= lo, (
            f"lambda_order1={result['lambda_order1']:.2e} below lower bound"
        )
        assert result['lambda_order1'] <= hi, (
            f"lambda_order1={result['lambda_order1']:.2e} above upper bound"
        )

    def test_all_lambdas_within_bounds_two(self):
        """Two-lambda route: both lambdas within search bounds."""
        Phi, y, P = self._make_two_order_phi_y(N=300, D=3, K1=3, K2=2, seed=43)
        bounds = (1e-6, 1e2)
        result = optimize_multi_lambda_extended(
            Phi, y, D=3, K1=3, K2=2, P=P, K3=0, T=0, M_residual=0,
            method='gcv', bounds=bounds, verbose=False,
        )
        lo, hi = bounds
        for key in ['lambda_order1', 'lambda_order2']:
            val = result[key]
            assert val >= lo, f"{key}={val:.2e} below lower bound"
            assert val <= hi, f"{key}={val:.2e} above upper bound"

    def test_general_path_three_lambdas(self):
        """K3>0, T>0 activates three-lambda path (n_active=3).

        Uses D=3 so there is exactly C(3,3)=1 triple.
        K1=K2=K3=1 keeps the feature count small for the 8^3 grid search.
        """
        import jax.numpy as jnp
        from hifi_anova.core.features import (
            build_first_order_features, build_second_order_features,
            build_third_order_features,
        )
        from hifi_anova.core.pairs import PairManager, TripleManager

        rng = np.random.RandomState(44)
        N, D, K1, K2, K3 = 200, 3, 1, 1, 1
        X = rng.uniform(0, 1, (N, D))
        x_jax = jnp.array(X, dtype=jnp.float64)

        pm = PairManager(D)     # C(3,2)=3 pairs
        tm = TripleManager(D)   # C(3,3)=1 triple
        T = tm.T
        assert T > 0, "D=3 must produce at least one triple"

        phi1 = np.asarray(build_first_order_features(x_jax, K1), dtype=np.float64)
        phi2 = np.asarray(build_second_order_features(x_jax, K2, pm.pair_indices),
                          dtype=np.float64)
        phi3 = np.asarray(
            build_third_order_features(x_jax, K3, tm.triple_indices),
            dtype=np.float64
        )
        Phi = np.concatenate([phi1, phi2, phi3], axis=1)
        y_raw = np.cos(2 * np.pi * X[:, 0]) + 0.3 * rng.randn(N)
        y = y_raw - y_raw.mean()

        bounds = (1e-6, 1e2)
        result = optimize_multi_lambda_extended(
            Phi, y, D=D, K1=K1, K2=K2, P=pm.P, K3=K3, T=T, M_residual=0,
            method='gcv', bounds=bounds, verbose=False,
        )
        assert 'n_lambdas_optimized' in result, "n_lambdas_optimized should be in result"
        assert result['n_lambdas_optimized'] == 3, (
            f"Expected 3 active lambdas, got {result['n_lambdas_optimized']}"
        )
        for key in ['lambda_order1', 'lambda_order2', 'lambda_order3']:
            assert key in result, f"{key} should be present for 3-lambda result"
            lo, hi = bounds
            assert result[key] >= lo, f"{key}={result[key]:.2e} below lower bound"
            assert result[key] <= hi, f"{key}={result[key]:.2e} above upper bound"

    def test_m_residual_activates_lambda_residual(self):
        """M_residual>0 adds lambda_residual to the active set."""
        import jax.numpy as jnp
        from hifi_anova.core.features import build_first_order_features

        rng = np.random.RandomState(45)
        N, D, K1 = 250, 3, 3
        X = rng.uniform(0, 1, (N, D))
        x_jax = jnp.array(X, dtype=jnp.float64)

        phi1 = np.asarray(build_first_order_features(x_jax, K1), dtype=np.float64)
        # Append M_residual=5 synthetic residual features
        M_res = 5
        Phi_res = rng.randn(N, M_res) * 0.1
        Phi = np.concatenate([phi1, Phi_res], axis=1)

        y_raw = np.cos(2 * np.pi * X[:, 0]) + 0.3 * rng.randn(N)
        y = y_raw - y_raw.mean()

        bounds = (1e-6, 1e2)
        result = optimize_multi_lambda_extended(
            Phi, y, D=D, K1=K1, K2=0, P=0, K3=0, T=0, M_residual=M_res,
            method='gcv', bounds=bounds, verbose=False,
        )
        assert 'n_lambdas_optimized' in result, "n_lambdas_optimized should be present"
        assert result['n_lambdas_optimized'] == 2, (
            f"Expected 2 active lambdas (order1 + residual), got {result['n_lambdas_optimized']}"
        )
        assert 'lambda_residual' in result, "lambda_residual must be present when M_residual>0"
        lo, hi = bounds
        assert result['lambda_residual'] >= lo
        assert result['lambda_residual'] <= hi

    def test_verbose_false_no_output(self, capsys):
        """verbose=False should not print anything to stdout."""
        Phi, y = self._make_first_order_phi_y(N=200, D=3, K1=2, seed=46)
        optimize_multi_lambda_extended(
            Phi, y, D=3, K1=2, K2=0, P=0, K3=0, T=0, M_residual=0,
            method='gcv', bounds=(1e-6, 1e2), verbose=False,
        )
        captured = capsys.readouterr()
        assert captured.out == "", "verbose=False should produce no stdout output"


# ---------------------------------------------------------------------------
# Integration: build_regularization_vector compatibility
# ---------------------------------------------------------------------------

class TestBuildRegularizationVectorIntegration:
    """Confirm that build_regularization_vector output is compatible with
    ridge_solve_with_diagnostics — the correct vector length is produced."""

    def test_first_order_vector_length(self):
        """reg vector length = D * basis_size(K1)."""
        D, K1 = 4, 3
        reg = np.asarray(
            build_regularization_vector(D, K1, 0, 0, 'uniform', 0.01, 0.0),
            dtype=np.float64
        )
        expected_len = D * basis_size(K1, include_linear=True, basis_name='fourier')
        assert len(reg) == expected_len, (
            f"reg length {len(reg)} != expected {expected_len}"
        )

    def test_two_order_vector_length(self):
        """reg vector length = D*B1 + P*B2² for two-order model."""
        from hifi_anova.core.pairs import PairManager
        D, K1, K2 = 3, 3, 2
        pm = PairManager(D)
        P = pm.P
        reg = np.asarray(
            build_regularization_vector(D, K1, K2, P, 'variance', 0.01, 0.1),
            dtype=np.float64
        )
        B1 = basis_size(K1, include_linear=True, basis_name='fourier')
        B2 = basis_size(K2, include_linear=True, basis_name='fourier')
        expected_len = D * B1 + P * B2**2
        assert len(reg) == expected_len, (
            f"reg length {len(reg)} != expected {expected_len}"
        )

    def test_compatible_with_ridge_solve(self):
        """A valid reg vector from build_regularization_vector works in ridge_solve."""
        import jax.numpy as jnp
        from hifi_anova.core.features import build_first_order_features

        rng = np.random.RandomState(99)
        N, D, K1 = 200, 3, 3
        X = rng.uniform(0, 1, (N, D))
        x_jax = jnp.array(X, dtype=jnp.float64)
        Phi = np.asarray(build_first_order_features(x_jax, K1), dtype=np.float64)
        y = rng.randn(N)
        y -= y.mean()

        reg = np.asarray(
            build_regularization_vector(D, K1, 0, 0, 'variance', 0.01, 0.0),
            dtype=np.float64
        )
        assert reg.shape[0] == Phi.shape[1], (
            f"reg dim {reg.shape[0]} != Phi cols {Phi.shape[1]}"
        )

        result = ridge_solve_with_diagnostics(Phi, y, reg)
        assert result['w'].shape == (Phi.shape[1],)
        assert np.isfinite(result['gcv']), "GCV must be finite"
        assert np.isfinite(result['log_evidence']), "log_evidence must be finite"
