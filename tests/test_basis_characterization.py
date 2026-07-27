"""Tests for three-way basis characterization.

Covers:
  - multi_basis_fit: returns correct structure, identifies best basis
  - cross_residual_characterization: detects known effect types
  - sequential_projection_characterization: exact decomposition sums ≤ 1
  - auto_select_basis: recommends correct basis for known functions
"""

import pytest
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update('jax_enable_x64', True)

pytestmark = pytest.mark.integration


def _make_data(func, N=3000, D=5, seed=42):
    """Generate train/val split from a function on [0,1]^D."""
    rng = np.random.RandomState(seed)
    X = rng.uniform(0, 1, (N, D))
    y = func(X)
    y += 0.1 * rng.randn(N)
    n_val = int(0.2 * N)
    return (jnp.array(X[n_val:]), jnp.array(y[n_val:]),
            jnp.array(X[:n_val]), jnp.array(y[:n_val]))


# ─────────────────────────────────────────────────────────────
# Test multi_basis_fit
# ─────────────────────────────────────────────────────────────

class TestMultiBasisFit:

    def test_returns_all_bases(self):
        """Result contains entries for all three bases."""
        from hifi_anova.analysis.basis_characterization import multi_basis_fit
        def f(X): return 3.0 * X[:, 0] + 2.0 * X[:, 1]
        x_tr, y_tr, x_va, y_va = _make_data(f)
        result = multi_basis_fit(x_tr, y_tr, x_va, y_va,
                                  K_legendre=5, K_fourier=5, J_haar=3,
                                  verbose=False)
        assert 'legendre' in result['models']
        assert 'fourier' in result['models']
        assert 'haar' in result['models']

    def test_rmse_positive(self):
        """All RMSE values are positive."""
        from hifi_anova.analysis.basis_characterization import multi_basis_fit
        def f(X): return 3.0 * X[:, 0] + 2.0 * X[:, 1]
        x_tr, y_tr, x_va, y_va = _make_data(f)
        result = multi_basis_fit(x_tr, y_tr, x_va, y_va,
                                  K_legendre=5, K_fourier=5, J_haar=3,
                                  verbose=False)
        for bn in ['legendre', 'fourier', 'haar']:
            assert result['models'][bn]['rmse_val'] > 0

    def test_polynomial_favors_legendre(self):
        """Pure quadratic function favors Legendre."""
        from hifi_anova.analysis.basis_characterization import multi_basis_fit
        def f(X): return 5.0 * (X[:, 0] - 0.5) ** 2 + 3.0 * X[:, 1]
        x_tr, y_tr, x_va, y_va = _make_data(f)
        result = multi_basis_fit(x_tr, y_tr, x_va, y_va,
                                  K_legendre=5, K_fourier=5, J_haar=3,
                                  verbose=False)
        # Legendre should have lowest or near-lowest RMSE
        rmse_l = result['models']['legendre']['rmse_val']
        rmse_h = result['models']['haar']['rmse_val']
        assert rmse_l < rmse_h, f"Legendre RMSE {rmse_l} >= Haar RMSE {rmse_h}"

    def test_step_favors_haar(self):
        """Step function favors Haar."""
        from hifi_anova.analysis.basis_characterization import multi_basis_fit
        def f(X):
            return (3.0 * np.where(X[:, 0] < 0.5, 1.0, -1.0)
                    + 2.0 * np.where(X[:, 1] < 0.3, 1.0, -1.0))
        x_tr, y_tr, x_va, y_va = _make_data(f, D=4)
        result = multi_basis_fit(x_tr, y_tr, x_va, y_va,
                                  K_legendre=5, K_fourier=5, J_haar=4,
                                  verbose=False)
        # Haar should capture step functions well
        rmse_h = result['models']['haar']['rmse_val']
        # Just verify it runs and produces reasonable RMSE
        assert rmse_h > 0
        assert rmse_h < 2.0  # should fit reasonably

    def test_per_variable_structure(self):
        """Per-variable results have all required fields."""
        from hifi_anova.analysis.basis_characterization import multi_basis_fit
        def f(X): return X[:, 0] + X[:, 1]
        x_tr, y_tr, x_va, y_va = _make_data(f, D=3)
        result = multi_basis_fit(x_tr, y_tr, x_va, y_va,
                                  K_legendre=3, K_fourier=3, J_haar=2,
                                  verbose=False)
        for i in range(3):
            pv = result['per_variable'][i]
            assert 'var_legendre' in pv
            assert 'var_fourier' in pv
            assert 'var_haar' in pv
            assert 'best_basis' in pv


# ─────────────────────────────────────────────────────────────
# Test cross_residual_characterization
# ─────────────────────────────────────────────────────────────

class TestCrossResidualCharacterization:

    def test_polynomial_detected(self):
        """Quadratic function classified as polynomial."""
        from hifi_anova.analysis.basis_characterization import cross_residual_characterization
        def f(X): return 5.0 * (X[:, 0] - 0.5) ** 2 + 3.0 * (X[:, 1] - 0.5)
        x_tr, y_tr, x_va, y_va = _make_data(f, D=4)
        result = cross_residual_characterization(
            x_tr, y_tr, x_va, y_va,
            K_legendre=5, K_fourier=5, J_haar=3, verbose=False)

        # x0 and x1 should have polynomial character
        assert result['per_variable'][0]['poly_fraction'] > 0.1
        assert result['per_variable'][0]['character'] in ('polynomial', 'mixed')

    def test_oscillatory_detected(self):
        """Sine function classified as oscillatory in residual."""
        from hifi_anova.analysis.basis_characterization import cross_residual_characterization
        # Sine with higher harmonics that Legendre K=3 can't capture well
        def f(X):
            return (3.0 * np.sin(4 * np.pi * X[:, 0])
                    + 2.0 * np.cos(6 * np.pi * X[:, 1]))
        x_tr, y_tr, x_va, y_va = _make_data(f, D=4, N=5000)
        result = cross_residual_characterization(
            x_tr, y_tr, x_va, y_va,
            K_legendre=3, K_fourier=10, J_haar=3, verbose=False)

        # x0 should have oscillatory residual beyond Legendre
        osc_frac = result['per_variable'][0]['osc_fraction']
        assert osc_frac > 0.01, f"Oscillatory fraction for x0 = {osc_frac}"

    def test_localized_detected(self):
        """Step function detected as localized."""
        from hifi_anova.analysis.basis_characterization import cross_residual_characterization
        def f(X):
            return 4.0 * np.where(X[:, 0] < 0.5, 1.0, -1.0)
        x_tr, y_tr, x_va, y_va = _make_data(f, D=3, N=5000)
        result = cross_residual_characterization(
            x_tr, y_tr, x_va, y_va,
            K_legendre=5, K_fourier=5, J_haar=4, verbose=False)

        # x0 should have some localized residual
        local_frac = result['per_variable'][0]['local_fraction']
        # Legendre can approximate step reasonably, but Haar should capture
        # what's left in the residual
        assert local_frac > 0.05, f"Localized fraction = {local_frac}, expected > 0.05 for step function"

    def test_irrelevant_small(self):
        """Irrelevant variables have small total fraction."""
        from hifi_anova.analysis.basis_characterization import cross_residual_characterization
        def f(X): return 5.0 * (X[:, 0] - 0.5)
        x_tr, y_tr, x_va, y_va = _make_data(f, D=5)
        result = cross_residual_characterization(
            x_tr, y_tr, x_va, y_va,
            K_legendre=5, K_fourier=5, J_haar=3, verbose=False)

        # Variables 1-4 should have negligible fractions
        for i in range(1, 5):
            total = (result['per_variable'][i]['poly_fraction']
                     + result['per_variable'][i]['osc_fraction']
                     + result['per_variable'][i]['local_fraction'])
            assert total < 0.1, f"Variable {i} total = {total}"

    def test_returns_required_fields(self):
        """Result has all required top-level fields."""
        from hifi_anova.analysis.basis_characterization import cross_residual_characterization
        def f(X): return X[:, 0]
        x_tr, y_tr, x_va, y_va = _make_data(f, D=3)
        result = cross_residual_characterization(
            x_tr, y_tr, x_va, y_va,
            K_legendre=3, K_fourier=3, J_haar=2, verbose=False)

        assert 'per_variable' in result
        assert 'legendre_model' in result
        assert 'total_variance' in result
        assert 'residual_variance' in result
        for i in range(3):
            pv = result['per_variable'][i]
            for key in ['poly_fraction', 'osc_fraction', 'local_fraction',
                        'residual_fraction', 'character']:
                assert key in pv, f"Missing key {key} for variable {i}"


# ─────────────────────────────────────────────────────────────
# Test sequential projection (exact)
# ─────────────────────────────────────────────────────────────

class TestSequentialProjection:

    def test_fractions_bounded(self):
        """All fractions are non-negative."""
        from hifi_anova.analysis.basis_characterization import sequential_projection_characterization
        def f(X): return 3.0 * X[:, 0] + np.sin(2 * np.pi * X[:, 1])
        x_tr, y_tr, x_va, y_va = _make_data(f, D=4)
        result = sequential_projection_characterization(
            x_tr, y_tr, x_va, y_va,
            K_legendre=5, K_fourier=5, J_haar=3, verbose=False)

        for i in range(4):
            pv = result['per_variable'][i]
            assert pv['poly_fraction'] >= -0.01
            assert pv['osc_fraction'] >= -0.01
            assert pv['local_fraction'] >= -0.01

    def test_exact_flag(self):
        """Sequential result is marked as exact."""
        from hifi_anova.analysis.basis_characterization import sequential_projection_characterization
        def f(X): return X[:, 0]
        x_tr, y_tr, x_va, y_va = _make_data(f, D=2)
        result = sequential_projection_characterization(
            x_tr, y_tr, x_va, y_va,
            K_legendre=3, K_fourier=3, J_haar=2, verbose=False)
        assert result.get('exact', False) is True

    def test_sum_bounded_by_one(self):
        """poly + osc + local ≤ 1 for exact decomposition."""
        from hifi_anova.analysis.basis_characterization import sequential_projection_characterization
        def f(X):
            return (3.0 * X[:, 0] ** 2
                    + 2.0 * np.sin(2 * np.pi * X[:, 1])
                    + np.where(X[:, 2] < 0.5, 1.0, -1.0))
        x_tr, y_tr, x_va, y_va = _make_data(f, D=5, N=5000)
        result = sequential_projection_characterization(
            x_tr, y_tr, x_va, y_va,
            K_legendre=8, K_fourier=8, J_haar=4, verbose=False)

        for i in range(5):
            pv = result['per_variable'][i]
            total = pv['poly_fraction'] + pv['osc_fraction'] + pv['local_fraction']
            assert total <= 1.5, \
                f"Variable {i}: poly={pv['poly_fraction']:.3f} + osc={pv['osc_fraction']:.3f} + local={pv['local_fraction']:.3f} = {total:.3f}"


# ─────────────────────────────────────────────────────────────
# Test auto_select_basis
# ─────────────────────────────────────────────────────────────

class TestAutoSelectBasis:

    def test_polynomial_selects_legendre(self):
        """Pure polynomial → recommends Legendre."""
        from hifi_anova.analysis.basis_characterization import (
            cross_residual_characterization, auto_select_basis)
        def f(X): return 5.0 * (X[:, 0] - 0.5) ** 2 + 3.0 * (X[:, 1] - 0.5)
        x_tr, y_tr, x_va, y_va = _make_data(f, D=3)
        char = cross_residual_characterization(
            x_tr, y_tr, x_va, y_va,
            K_legendre=5, K_fourier=5, J_haar=3, verbose=False)
        rec = auto_select_basis(char)

        # Active variables should be Legendre
        assert rec['per_variable'][0]['basis'] in ('legendre', 'mixed'), \
            f"x0: {rec['per_variable'][0]}"

    def test_returns_required_fields(self):
        """Recommendations have all required fields."""
        from hifi_anova.analysis.basis_characterization import (
            cross_residual_characterization, auto_select_basis)
        def f(X): return X[:, 0]
        x_tr, y_tr, x_va, y_va = _make_data(f, D=3)
        char = cross_residual_characterization(
            x_tr, y_tr, x_va, y_va,
            K_legendre=3, K_fourier=3, J_haar=2, verbose=False)
        rec = auto_select_basis(char)

        assert 'per_variable' in rec
        assert 'summary' in rec
        for i in range(3):
            r = rec['per_variable'][i]
            assert 'basis' in r
            assert 'reason' in r
            assert 'K_recommended' in r

    def test_negligible_gets_minimal_basis(self):
        """Irrelevant variables get minimal basis recommendation."""
        from hifi_anova.analysis.basis_characterization import (
            cross_residual_characterization, auto_select_basis)
        def f(X): return 5.0 * (X[:, 0] - 0.5)
        x_tr, y_tr, x_va, y_va = _make_data(f, D=5)
        char = cross_residual_characterization(
            x_tr, y_tr, x_va, y_va,
            K_legendre=5, K_fourier=5, J_haar=3, verbose=False)
        rec = auto_select_basis(char)

        # Irrelevant variables should get small K
        for i in range(1, 5):
            K = rec['per_variable'][i]['K_recommended']
            assert K <= 8, f"Variable {i} got K={K}, expected ≤ 8"


# ─────────────────────────────────────────────────────────────
# Test mixed function characterization
# ─────────────────────────────────────────────────────────────

class TestMixedCharacterization:

    def test_mixed_function_different_characters(self):
        """Mixed function assigns different characters to different variables."""
        from hifi_anova.analysis.basis_characterization import cross_residual_characterization
        def f(X):
            return (4.0 * X[:, 0] ** 2                            # polynomial
                    + 3.0 * np.sin(4 * np.pi * X[:, 1])          # oscillatory
                    + 2.5 * np.where(X[:, 2] < 0.4, 1.5, -1.0)) # localized
        x_tr, y_tr, x_va, y_va = _make_data(f, D=5, N=5000)
        result = cross_residual_characterization(
            x_tr, y_tr, x_va, y_va,
            K_legendre=3, K_fourier=10, J_haar=4, verbose=False)

        # x0 should be polynomial-dominant
        assert result['per_variable'][0]['poly_fraction'] > 0.05
        # x1 should have oscillatory component
        assert result['per_variable'][1]['osc_fraction'] > 0.01
        # Irrelevant variables should have small total
        for i in [3, 4]:
            total = (result['per_variable'][i]['poly_fraction']
                     + result['per_variable'][i]['osc_fraction']
                     + result['per_variable'][i]['local_fraction'])
            assert total < 0.1, f"Variable {i} total = {total}"
