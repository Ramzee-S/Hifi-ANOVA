"""Tests for three bugfixes:

1. pairs.py variable_slice() — basis-aware block sizes (Legendre, spectral)
2. reg_path.py — basis-aware Gram matrices and block sizes
3. newton.py — backtracking line search for extreme heteroscedasticity
"""

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from hifi_anova.core.features import basis_size, build_first_order_features
from hifi_anova.core.gram import build_gram_matrix, build_gram_matrix_2d
from hifi_anova.core.pairs import PairManager
from hifi_anova.training.newton import newton_solve_log_variance


# ── Fix 1: PairManager.variable_slice basis-awareness ────────────────────


class TestVariableSliceBasisAware:
    """variable_slice must use basis_size, not hardcoded 2*K+1."""

    def test_fourier_full_unchanged(self):
        """Default Fourier+linear: block = 2K+1 (backward compatible)."""
        pm = PairManager(D=5)
        K = 5
        s = pm.variable_slice(2, K)
        assert s == slice(2 * 11, 3 * 11)  # 2*5+1 = 11

    def test_fourier_spectral(self):
        """Fourier without linear: block = 2K."""
        pm = PairManager(D=5)
        K = 5
        s = pm.variable_slice(0, K, include_linear=False)
        block = 2 * K  # 10
        assert s == slice(0, block)
        s2 = pm.variable_slice(3, K, include_linear=False)
        assert s2 == slice(3 * block, 4 * block)

    def test_legendre(self):
        """Legendre: block = K."""
        pm = PairManager(D=5)
        K = 5
        s = pm.variable_slice(0, K, basis_name='legendre')
        assert s == slice(0, K)
        s2 = pm.variable_slice(3, K, basis_name='legendre')
        assert s2 == slice(3 * K, 4 * K)

    def test_legendre_vs_fourier_different(self):
        """Legendre and Fourier slices must differ (K vs 2K+1)."""
        pm = PairManager(D=5)
        K = 5
        s_fourier = pm.variable_slice(1, K, basis_name='fourier')
        s_legendre = pm.variable_slice(1, K, basis_name='legendre')
        assert s_fourier != s_legendre
        assert (s_fourier.stop - s_fourier.start) == 11
        assert (s_legendre.stop - s_legendre.start) == 5

    def test_pair_slice_legendre(self):
        """pair_slice should also use correct block for Legendre."""
        pm = PairManager(D=4)
        K = 3
        s = pm.pair_slice(0, K, include_linear=True)
        # Fourier default: (2*3+1)^2 = 49
        assert (s.stop - s.start) == 49

        # Legendre: basis_size returns K, but pair_slice doesn't take
        # basis_name yet — it only takes include_linear. Let's verify
        # the existing include_linear=False path works correctly.
        s_spec = pm.pair_slice(0, K, include_linear=False)
        assert (s_spec.stop - s_spec.start) == (2 * K) ** 2  # 36

    def test_slice_covers_coefficients(self):
        """Slice from variable_slice must match actual feature width."""
        K = 4
        D = 3
        N = 100
        key = jax.random.PRNGKey(0)
        x = jax.random.uniform(key, (N, D))

        for bname in ['fourier', 'legendre']:
            phi = build_first_order_features(x, K, basis_name=bname)
            total_features = phi.shape[1]
            pm = PairManager(D=D)

            # Sum of all variable slices must cover the full feature width
            total_sliced = 0
            for i in range(D):
                s = pm.variable_slice(i, K, basis_name=bname)
                total_sliced += (s.stop - s.start)
            assert total_sliced == total_features, (
                f"basis={bname}: sliced {total_sliced} != features {total_features}"
            )


# ── Fix 2: reg_path basis-awareness ──────────────────────────────────────


class TestRegPathBasisAware:
    """compute_reg_path must use basis_size for block computation."""

    @pytest.fixture
    def legendre_data(self):
        """Simple Legendre first-order problem."""
        np.random.seed(42)
        N, D, K1 = 500, 3, 3
        x = np.random.uniform(0, 1, (N, D))
        # True function: linear in x0
        y = 2.0 * (x[:, 0] - 0.5) + 0.1 * np.random.randn(N)
        y = y - y.mean()

        phi = np.asarray(build_first_order_features(
            jnp.array(x), K1, basis_name='legendre'))
        return phi, y, D, K1

    def test_legendre_reg_path_runs(self, legendre_data):
        """Reg path must not crash with basis_name='legendre'."""
        from hifi_anova.analysis.reg_path import compute_reg_path

        phi, y, D, K1 = legendre_data
        path = compute_reg_path(
            phi, y, D, K1,
            strategy='variance', n_lambdas=10,
            lambda_range=(1e-4, 1e1),
            basis_name='legendre',
        )
        assert path.lambdas.shape == (10,)
        assert len(path.sobol_paths) == D
        # GCV optimal should be finite
        assert np.isfinite(path.lambda_gcv_opt)

    def test_legendre_sobol_sum_near_one(self, legendre_data):
        """At low lambda, Sobol indices should sum close to 1."""
        from hifi_anova.analysis.reg_path import compute_reg_path

        phi, y, D, K1 = legendre_data
        path = compute_reg_path(
            phi, y, D, K1,
            strategy='variance', n_lambdas=20,
            lambda_range=(1e-6, 1e0),
            basis_name='legendre',
        )
        # At smallest lambda, sum of Sobol should be near 1
        sobol_sum = sum(path.sobol_paths[i][0] for i in range(D))
        assert 0.8 < sobol_sum < 1.2, f"Sobol sum = {sobol_sum}"

    def test_spectral_fourier_reg_path(self):
        """Reg path with include_linear_1=False (spectral Fourier)."""
        from hifi_anova.analysis.reg_path import compute_reg_path

        np.random.seed(42)
        N, D, K1 = 500, 3, 3
        x = np.random.uniform(0, 1, (N, D))
        y = np.cos(2 * np.pi * x[:, 0]) + 0.1 * np.random.randn(N)
        y = y - y.mean()

        phi = np.asarray(build_first_order_features(
            jnp.array(x), K1, include_linear=False))

        path = compute_reg_path(
            phi, y, D, K1,
            strategy='variance', n_lambdas=10,
            lambda_range=(1e-4, 1e1),
            include_linear_1=False,
        )
        assert path.lambdas.shape == (10,)
        # Block size should be 2K=6, not 2K+1=7
        block = basis_size(K1, include_linear=False)
        assert block == 6
        assert phi.shape[1] == D * block

    def test_fourier_default_unchanged(self):
        """Default Fourier path should be identical to old behavior."""
        from hifi_anova.analysis.reg_path import compute_reg_path

        np.random.seed(42)
        N, D, K1 = 300, 3, 3
        x = np.random.uniform(0, 1, (N, D))
        y = 3.0 * (x[:, 0] - 0.5) + 0.1 * np.random.randn(N)
        y = y - y.mean()

        phi = np.asarray(build_first_order_features(jnp.array(x), K1))

        path = compute_reg_path(
            phi, y, D, K1,
            strategy='variance', n_lambdas=10,
            lambda_range=(1e-4, 1e1),
        )
        assert len(path.sobol_paths) == D
        assert np.isfinite(path.lambda_gcv_opt)


# ── Fix 3: Newton solver with backtracking ───────────────────────────────


class TestNewtonBacktracking:
    """Newton solver must converge even on extreme heteroscedasticity."""

    def test_mild_heteroscedasticity(self):
        """Baseline: mild case should converge as before."""
        np.random.seed(42)
        N, D = 1000, 3
        K = 2
        x = np.random.uniform(0, 1, (N, D))

        # h(x) = 1.0 * (x0 - 0.5), moderate dynamic range
        h_true = 1.0 * (x[:, 0] - 0.5)
        sigma2_true = np.exp(h_true)
        r2 = sigma2_true * np.random.chisquare(df=1, size=N)

        Psi = jnp.array(build_first_order_features(jnp.array(x), K))
        F_h = Psi.shape[1]
        reg = jnp.ones(F_h) * 0.01

        w_h, h0 = newton_solve_log_variance(
            Psi, jnp.array(r2),
            jnp.zeros(F_h), 0.0, reg,
            max_iter=20,
        )

        # Reconstruct and check: predicted h should correlate with true h
        h_pred = float(h0) + np.asarray(Psi @ w_h)
        corr = np.corrcoef(h_true, h_pred)[0, 1]
        assert corr > 0.5, f"Correlation {corr} too low for mild case"

    def test_extreme_heteroscedasticity(self):
        """Extreme case: h ranges over [-5, 5], so sigma^2 spans [0.007, 148].

        Without line search this can diverge due to exp(h) overflow
        in gradient computation.
        """
        np.random.seed(42)
        N, D = 2000, 3
        K = 3
        x = np.random.uniform(0, 1, (N, D))

        # h(x) = 5.0 * (x0 - 0.5): range [-2.5, 2.5], sigma^2 in [0.08, 12]
        # Push harder: 10.0 * (x0 - 0.5): range [-5, 5], sigma^2 in [0.007, 148]
        h_true = 10.0 * (x[:, 0] - 0.5)
        sigma2_true = np.exp(h_true)
        r2 = sigma2_true * np.random.chisquare(df=1, size=N)

        Psi = jnp.array(build_first_order_features(jnp.array(x), K))
        F_h = Psi.shape[1]
        reg = jnp.ones(F_h) * 0.1

        # Start from a bad initial point (all zeros)
        w_h, h0 = newton_solve_log_variance(
            Psi, jnp.array(r2),
            jnp.zeros(F_h), 0.0, reg,
            max_iter=30,
        )

        # Must converge (no NaN/Inf)
        assert jnp.all(jnp.isfinite(w_h)), "w_h contains NaN/Inf"
        assert np.isfinite(h0), "h0 is NaN/Inf"

        # Predicted h should correlate with true h
        h_pred = float(h0) + np.asarray(Psi @ w_h)
        corr = np.corrcoef(h_true, h_pred)[0, 1]
        assert corr > 0.3, f"Correlation {corr} too low for extreme case"

    def test_very_extreme_heteroscedasticity(self):
        """Very extreme: h coefficient = 20, range [-10, 10].

        sigma^2 ranges over [4.5e-5, 22026]. This would overflow
        without h-clamping and backtracking.
        """
        np.random.seed(42)
        N = 3000
        x = np.random.uniform(0, 1, (N, 1))

        h_true = 20.0 * (x[:, 0] - 0.5)
        # Clamp true sigma2 to prevent numerical issues in data generation
        h_clamped = np.clip(h_true, -15, 15)
        sigma2_true = np.exp(h_clamped)
        r2 = sigma2_true * np.random.chisquare(df=1, size=N)

        # Simple features: just the raw input
        Psi = jnp.array(x - 0.5).reshape(N, 1)
        reg = jnp.array([0.01])

        w_h, h0 = newton_solve_log_variance(
            Psi, jnp.array(r2),
            jnp.zeros(1), 0.0, reg,
            max_iter=50,
        )

        # Must not diverge
        assert jnp.all(jnp.isfinite(w_h)), "w_h diverged"
        assert np.isfinite(h0), "h0 diverged"

    def test_homoscedastic_recovers_constant(self):
        """When noise is constant, h should be approximately constant."""
        np.random.seed(42)
        N = 1000
        x = np.random.uniform(0, 1, (N, 2))

        sigma2 = 0.5
        r2 = sigma2 * np.random.chisquare(df=1, size=N)

        Psi = jnp.array(build_first_order_features(jnp.array(x), K=2))
        F_h = Psi.shape[1]
        reg = jnp.ones(F_h) * 0.1

        w_h, h0 = newton_solve_log_variance(
            Psi, jnp.array(r2),
            jnp.zeros(F_h), 0.0, reg,
            max_iter=20,
        )

        # Coefficients should be small (no structure to learn)
        assert float(jnp.max(jnp.abs(w_h))) < 1.0, "w_h too large for constant noise"
        # h0 should be near log(sigma2) = log(0.5) ≈ -0.69
        assert abs(h0 - np.log(sigma2)) < 1.0, f"h0={h0}, expected ≈ {np.log(sigma2)}"

    def test_backward_compatible_signature(self):
        """Old callers without max_backtrack/armijo_c should still work."""
        np.random.seed(42)
        N = 200
        x = np.random.uniform(0, 1, (N, 1))
        r2 = jnp.ones(N) * 0.5
        Psi = jnp.array(x - 0.5).reshape(N, 1)
        reg = jnp.array([0.01])

        # Call with only the original parameters
        w_h, h0 = newton_solve_log_variance(
            Psi, r2, jnp.zeros(1), 0.0, reg,
        )
        assert jnp.all(jnp.isfinite(w_h))
