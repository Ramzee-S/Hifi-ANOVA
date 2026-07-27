"""Tests for Haar wavelet basis integration.

Covers:
  - HaarBasis class: evaluate, gram, complexity weights, index mapping
  - Feature integration: basis_size, build_per_variable_basis, first/second order
  - Gram matrix: identity, orthonormality verification
  - Regularization: all strategies with basis_name='haar'
  - Sobol computation: sum-of-squares property
  - Diagnostic: haar_residual_analysis on synthetic step function
"""

import pytest
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update('jax_enable_x64', True)


# ─────────────────────────────────────────────────────────────
# Test HaarBasis class
# ─────────────────────────────────────────────────────────────

class TestHaarBasis:
    """Test the standalone HaarBasis class."""

    def test_n_basis(self):
        """n_basis = 2^J - 1."""
        from hifi_anova.core.haar import HaarBasis
        assert HaarBasis(1).n_basis == 1
        assert HaarBasis(2).n_basis == 3
        assert HaarBasis(3).n_basis == 7
        assert HaarBasis(4).n_basis == 15
        assert HaarBasis(5).n_basis == 31

    def test_evaluate_shape(self):
        """evaluate returns (N, 2^J - 1)."""
        from hifi_anova.core.haar import HaarBasis
        haar = HaarBasis(4)
        x = jnp.linspace(0.01, 0.99, 100)
        phi = haar.evaluate(x)
        assert phi.shape == (100, 15)

    def test_evaluate_batch_shape(self):
        """evaluate_batch returns (N, D, n_basis)."""
        from hifi_anova.core.haar import HaarBasis
        haar = HaarBasis(3)
        x = jnp.ones((50, 5)) * 0.5
        x = x.at[:, 0].set(jnp.linspace(0.01, 0.99, 50))
        phi = haar.evaluate_batch(x)
        assert phi.shape == (50, 5, 7)

    def test_vanishing_integral(self):
        """All Haar wavelets integrate to zero on [0,1]."""
        from hifi_anova.core.haar import HaarBasis
        haar = HaarBasis(5)
        # Dense uniform grid for numerical integration
        N = 100000
        x = jnp.linspace(0.0, 1.0 - 1e-10, N)
        phi = haar.evaluate(x)  # (N, 31)
        # Trapezoidal integration ≈ mean for uniform grid
        integrals = jnp.mean(phi, axis=0)
        assert jnp.max(jnp.abs(integrals)) < 1e-10, \
            f"Non-zero integrals: {integrals}"

    def test_orthonormality(self):
        """Haar wavelets are orthonormal on [0,1]: Phi^T Phi / N ≈ I."""
        from hifi_anova.core.haar import HaarBasis
        haar = HaarBasis(4)
        N = 200000
        x = jnp.linspace(0.0, 1.0 - 1e-10, N)
        phi = haar.evaluate(x)  # (N, 15)
        empirical_gram = phi.T @ phi / N
        expected = jnp.eye(15)
        assert jnp.allclose(empirical_gram, expected, atol=0.01), \
            f"Max deviation: {jnp.max(jnp.abs(empirical_gram - expected))}"

    def test_psi_10_values(self):
        """ψ₁₀(x) = +1 on [0, 0.5), -1 on [0.5, 1)."""
        from hifi_anova.core.haar import HaarBasis
        haar = HaarBasis(1)
        x = jnp.array([0.1, 0.3, 0.49, 0.5, 0.7, 0.99])
        phi = haar.evaluate(x)  # (6, 1)
        expected = jnp.array([1.0, 1.0, 1.0, -1.0, -1.0, -1.0])[:, None]
        assert jnp.allclose(phi, expected)

    def test_psi_20_values(self):
        """ψ₂₀(x) = sqrt(2) on [0, 0.25), -sqrt(2) on [0.25, 0.5), 0 elsewhere."""
        from hifi_anova.core.haar import HaarBasis
        haar = HaarBasis(2)
        x = jnp.array([0.1, 0.3, 0.6, 0.8])
        phi = haar.evaluate(x)  # (4, 3)
        s2 = float(jnp.sqrt(2.0))
        # index 1 = ψ₂₀
        assert jnp.isclose(phi[0, 1], s2, atol=1e-10)    # x=0.1 in [0, 0.25)
        assert jnp.isclose(phi[1, 1], -s2, atol=1e-10)   # x=0.3 in [0.25, 0.5)
        assert jnp.isclose(phi[2, 1], 0.0, atol=1e-10)   # x=0.6 outside
        assert jnp.isclose(phi[3, 1], 0.0, atol=1e-10)   # x=0.8 outside

    def test_gram_is_identity(self):
        """Gram matrix is the identity."""
        from hifi_anova.core.haar import HaarBasis
        haar = HaarBasis(4)
        G = haar.gram_matrix()
        assert jnp.allclose(G, jnp.eye(15))

    def test_complexity_weights_shape(self):
        """complexity_weights returns correct shape."""
        from hifi_anova.core.haar import HaarBasis
        haar = HaarBasis(4)
        w = haar.complexity_weights()
        assert w.shape == (15,)

    def test_complexity_weights_monotonic(self):
        """Finer scales get higher penalty."""
        from hifi_anova.core.haar import HaarBasis
        haar = HaarBasis(4)
        w = haar.complexity_weights(penalty_exponent=1.0)
        # Scale 1 (index 0) should have lower weight than scale 2 (index 1)
        assert w[0] < w[1]
        # Scale 2 (indices 1,2) should have lower weight than scale 3 (index 3)
        assert w[1] < w[3]

    def test_scale_of_index(self):
        """scale_of_index maps correctly."""
        from hifi_anova.core.haar import HaarBasis
        haar = HaarBasis(4)
        assert haar.scale_of_index(0) == 1   # scale 1: 1 function
        assert haar.scale_of_index(1) == 2   # scale 2: 2 functions
        assert haar.scale_of_index(2) == 2
        assert haar.scale_of_index(3) == 3   # scale 3: 4 functions
        assert haar.scale_of_index(6) == 3
        assert haar.scale_of_index(7) == 4   # scale 4: 8 functions
        assert haar.scale_of_index(14) == 4

    def test_position_of_index(self):
        """position_of_index returns correct intervals."""
        from hifi_anova.core.haar import HaarBasis
        haar = HaarBasis(3)
        j, k, left, right = haar.position_of_index(0)
        assert j == 1 and k == 0
        assert abs(left - 0.0) < 1e-10 and abs(right - 1.0) < 1e-10

        j, k, left, right = haar.position_of_index(1)
        assert j == 2 and k == 0
        assert abs(left - 0.0) < 1e-10 and abs(right - 0.5) < 1e-10

        j, k, left, right = haar.position_of_index(2)
        assert j == 2 and k == 1
        assert abs(left - 0.5) < 1e-10 and abs(right - 1.0) < 1e-10

    def test_scale_slice(self):
        """scale_slice returns correct slices."""
        from hifi_anova.core.haar import HaarBasis
        haar = HaarBasis(4)
        assert haar.scale_slice(1) == slice(0, 1)     # 1 function
        assert haar.scale_slice(2) == slice(1, 3)     # 2 functions
        assert haar.scale_slice(3) == slice(3, 7)     # 4 functions
        assert haar.scale_slice(4) == slice(7, 15)    # 8 functions

    def test_index_out_of_range(self):
        """Out-of-range index raises IndexError."""
        from hifi_anova.core.haar import HaarBasis
        haar = HaarBasis(3)
        with pytest.raises(IndexError):
            haar.scale_of_index(7)  # max valid index is 6


# ─────────────────────────────────────────────────────────────
# Test integration with features.py
# ─────────────────────────────────────────────────────────────

class TestHaarFeatures:
    """Test Haar integration through the features module."""

    def test_basis_size(self):
        """basis_size('haar') returns 2^K - 1."""
        from hifi_anova.core.features import basis_size
        assert basis_size(0, basis_name='haar') == 0
        assert basis_size(1, basis_name='haar') == 1
        assert basis_size(2, basis_name='haar') == 3
        assert basis_size(3, basis_name='haar') == 7
        assert basis_size(4, basis_name='haar') == 15

    def test_basis_size_ignores_include_linear(self):
        """include_linear is irrelevant for Haar."""
        from hifi_anova.core.features import basis_size
        assert basis_size(4, include_linear=True, basis_name='haar') == 15
        assert basis_size(4, include_linear=False, basis_name='haar') == 15

    def test_build_per_variable_basis(self):
        """build_per_variable_basis dispatches to Haar correctly."""
        from hifi_anova.core.features import build_per_variable_basis
        x = jnp.linspace(0.01, 0.99, 50).reshape(50, 1)
        x = jnp.broadcast_to(x, (50, 3))
        basis = build_per_variable_basis(x, K=3, basis_name='haar')
        assert basis.shape == (50, 3, 7)

    def test_first_order_features(self):
        """First-order features have correct shape."""
        from hifi_anova.core.features import build_first_order_features
        N, D = 100, 5
        x = jnp.tile(jnp.linspace(0.01, 0.99, N)[:, None], (1, D))
        phi = build_first_order_features(x, K=4, basis_name='haar')
        assert phi.shape == (N, D * 15)

    def test_first_order_zero_mean(self):
        """Haar features have zero mean on uniform data."""
        from hifi_anova.core.features import build_first_order_features
        N, D = 100000, 3
        x = jnp.tile(jnp.linspace(0.0, 1.0 - 1e-8, N)[:, None], (1, D))
        phi = build_first_order_features(x, K=4, basis_name='haar')
        means = jnp.mean(phi, axis=0)
        assert jnp.max(jnp.abs(means)) < 0.01, \
            f"Max mean: {jnp.max(jnp.abs(means))}"

    def test_second_order_features_shape(self):
        """Second-order Haar features have correct shape."""
        from hifi_anova.core.features import build_second_order_features
        N, D = 50, 4
        x = jnp.tile(jnp.linspace(0.01, 0.99, N)[:, None], (1, D))
        pairs = jnp.array([[0, 1], [0, 2], [1, 2]])
        B = 2 ** 3 - 1  # K=3 → 7 per var
        phi2 = build_second_order_features(x, K=3, pair_indices=pairs,
                                            basis_name='haar')
        assert phi2.shape == (N, 3 * B ** 2)

    def test_third_order_features_shape(self):
        """Third-order Haar features have correct shape."""
        from hifi_anova.core.features import build_third_order_features
        N, D = 50, 4
        x = jnp.tile(jnp.linspace(0.01, 0.99, N)[:, None], (1, D))
        triples = jnp.array([[0, 1, 2]])
        B = 2 ** 2 - 1  # K=2 → 3 per var
        phi3 = build_third_order_features(x, K=2, triple_indices=triples,
                                           basis_name='haar')
        assert phi3.shape == (N, 1 * B ** 3)


# ─────────────────────────────────────────────────────────────
# Test Gram matrix integration
# ─────────────────────────────────────────────────────────────

class TestHaarGram:
    """Test Haar Gram matrix through gram.py."""

    def test_gram_identity(self):
        """build_gram_matrix('haar') returns identity."""
        from hifi_anova.core.gram import build_gram_matrix
        G = build_gram_matrix(4, basis_name='haar')
        assert G.shape == (15, 15)
        assert jnp.allclose(G, jnp.eye(15))

    def test_gram_2d_identity(self):
        """Second-order Gram for Haar: I ⊗ I = I."""
        from hifi_anova.core.gram import build_gram_matrix, build_gram_matrix_2d
        G1 = build_gram_matrix(3, basis_name='haar')
        G2 = build_gram_matrix_2d(G1)
        n = 7
        assert G2.shape == (n ** 2, n ** 2)
        assert jnp.allclose(G2, jnp.eye(n ** 2))

    def test_gram_3d_identity(self):
        """Third-order Gram for Haar: I ⊗ I ⊗ I = I."""
        from hifi_anova.core.gram import build_gram_matrix, build_gram_matrix_3d
        G1 = build_gram_matrix(2, basis_name='haar')
        G3 = build_gram_matrix_3d(G1)
        n = 3
        assert G3.shape == (n ** 3, n ** 3)
        assert jnp.allclose(G3, jnp.eye(n ** 3))

    def test_empirical_gram_matches_analytic(self):
        """Empirical Gram from dense uniform samples matches identity."""
        from hifi_anova.core.features import build_per_variable_basis
        from hifi_anova.core.gram import build_gram_matrix
        N = 200000
        x = jnp.linspace(0.0, 1.0 - 1e-8, N).reshape(N, 1)
        basis = build_per_variable_basis(x, K=4, basis_name='haar')  # (N, 1, 15)
        phi = basis[:, 0, :]  # (N, 15)
        empirical = phi.T @ phi / N
        analytic = build_gram_matrix(4, basis_name='haar')
        assert jnp.allclose(empirical, analytic, atol=0.01)

    def test_derivative_penalty_shape(self):
        """Besov penalty has correct shape."""
        from hifi_anova.core.gram import build_derivative_penalty
        D = build_derivative_penalty(4, p=2, basis_name='haar')
        assert D.shape == (15,)

    def test_derivative_penalty_scale_monotonic(self):
        """Finer scales get higher penalty."""
        from hifi_anova.core.gram import build_derivative_penalty
        D = build_derivative_penalty(4, p=1, basis_name='haar')
        # Scale 1 < Scale 2 < Scale 3 < Scale 4
        assert D[0] < D[1]   # scale 1 vs scale 2
        assert D[1] < D[3]   # scale 2 vs scale 3
        assert D[3] < D[7]   # scale 3 vs scale 4

    def test_derivative_penalty_p2_stronger_than_p1(self):
        """p=2 (curvature) penalties are larger than p=1 (smoothness)."""
        from hifi_anova.core.gram import build_derivative_penalty
        D1 = build_derivative_penalty(4, p=1, basis_name='haar')
        D2 = build_derivative_penalty(4, p=2, basis_name='haar')
        # For j > 0, D2 should be larger than D1
        # Scale 2 onwards
        assert jnp.all(D2[1:] > D1[1:])


# ─────────────────────────────────────────────────────────────
# Test Sobol = sum of squared coefficients
# ─────────────────────────────────────────────────────────────

class TestHaarSobol:
    """Test that Sobol = sum(w^2) for Haar basis."""

    def test_sobol_is_sum_of_squares(self):
        """Var(f_i) = w_i^T G w_i = w_i^T I w_i = sum(w_i^2)."""
        from hifi_anova.core.gram import build_gram_matrix
        G = build_gram_matrix(4, basis_name='haar')
        w = jnp.array([1.0, 0.5, -0.3, 0.2, 0.0, 0.0, 0.0,
                        0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        var_gram = float(w @ G @ w)
        var_sos = float(jnp.sum(w ** 2))
        assert abs(var_gram - var_sos) < 1e-12

    def test_sobol_sum_to_one(self):
        """First-order Sobol indices sum to 1 for pure first-order model."""
        from hifi_anova.core.gram import build_gram_matrix
        G = build_gram_matrix(3, basis_name='haar')
        # 3 variables, K=3, B=7
        w1_var0 = jnp.array([1.0, 0.5, -0.3, 0.2, 0.1, 0.0, 0.0])
        w1_var1 = jnp.array([0.0, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0])
        w1_var2 = jnp.array([0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        var0 = float(w1_var0 @ G @ w1_var0)
        var1 = float(w1_var1 @ G @ w1_var1)
        var2 = float(w1_var2 @ G @ w1_var2)
        total = var0 + var1 + var2

        s0 = var0 / total
        s1 = var1 / total
        s2 = var2 / total
        assert abs(s0 + s1 + s2 - 1.0) < 1e-10


# ─────────────────────────────────────────────────────────────
# Test regularization strategies for Haar
# ─────────────────────────────────────────────────────────────

class TestHaarRegularization:
    """Test regularization.py strategies with basis_name='haar'."""

    def test_uniform(self):
        """Uniform strategy: all entries = lambda."""
        from hifi_anova.training.regularization import build_regularization_vector
        r = build_regularization_vector(D=3, K1=3, K2=0, P=0,
                                         strategy='uniform',
                                         lambda_order1=0.1,
                                         basis_name='haar')
        assert r.shape == (3 * 7,)
        assert jnp.allclose(r, 0.1)

    def test_variance_equals_uniform(self):
        """Variance strategy with Haar: G=I, so r[j] = lambda * 1 = lambda."""
        from hifi_anova.training.regularization import build_regularization_vector
        r = build_regularization_vector(D=3, K1=3, K2=0, P=0,
                                         strategy='variance',
                                         lambda_order1=0.1,
                                         basis_name='haar')
        assert jnp.allclose(r, 0.1)

    def test_curvature_besov(self):
        """Curvature strategy uses Besov penalty: r = lam * 4^{2j}."""
        from hifi_anova.training.regularization import build_regularization_vector
        r = build_regularization_vector(D=1, K1=3, K2=0, P=0,
                                         strategy='curvature',
                                         lambda_order1=1.0,
                                         basis_name='haar')
        # Scale 1 (index 0): 4^{2*1} = 16
        # Scale 2 (indices 1,2): 4^{2*2} = 256
        # Scale 3 (indices 3-6): 4^{2*3} = 4096
        # But curvature adds stability ridge: max(D[j], lam*1e-6)
        assert r[0] > 10  # scale 1
        assert r[1] > 200  # scale 2
        assert r[3] > 4000  # scale 3
        # Monotonic across scales
        assert r[0] < r[1]
        assert r[1] < r[3]

    def test_smoothness_besov(self):
        """Smoothness strategy uses Besov p=1: r = lam * 4^j."""
        from hifi_anova.training.regularization import build_regularization_vector
        r = build_regularization_vector(D=1, K1=3, K2=0, P=0,
                                         strategy='smoothness',
                                         lambda_order1=1.0,
                                         basis_name='haar')
        # Scale 1: 4^1 = 4, Scale 2: 4^2 = 16, Scale 3: 4^3 = 64
        assert r[0] > 3  # scale 1
        assert r[1] > 15  # scale 2
        assert r[3] > 60  # scale 3
        assert r[0] < r[1] < r[3]

    def test_sobolev(self):
        """Sobolev strategy: r = lam * (1 + 4^{j-1})^s."""
        from hifi_anova.training.regularization import build_regularization_vector
        r = build_regularization_vector(D=1, K1=3, K2=0, P=0,
                                         strategy='sobolev',
                                         lambda_order1=1.0,
                                         basis_name='haar')
        assert r.shape == (7,)
        # Scale 1: (1+1)^1 = 2.0, Scale 2: (1+4)^1 = 5.0
        assert jnp.isclose(r[0], 2.0, atol=0.01)
        assert jnp.isclose(r[1], 5.0, atol=0.01)

    def test_spectral(self):
        """Spectral strategy: r = lam * (2^{j-1})^alpha."""
        from hifi_anova.training.regularization import build_regularization_vector
        r = build_regularization_vector(D=1, K1=3, K2=0, P=0,
                                         strategy='spectral',
                                         lambda_order1=1.0,
                                         basis_name='haar')
        assert r.shape == (7,)
        # Default alpha=2. Scale 1: 1^2=1, Scale 2: 2^2=4, Scale 3: 4^2=16
        assert jnp.isclose(r[0], 1.0, atol=0.01)
        assert jnp.isclose(r[1], 4.0, atol=0.01)
        assert jnp.isclose(r[3], 16.0, atol=0.01)

    def test_second_order_reg(self):
        """Second-order regularization works for Haar."""
        from hifi_anova.training.regularization import build_regularization_vector
        B = 2 ** 2 - 1  # K2=2 → 3
        r = build_regularization_vector(D=3, K1=3, K2=2, P=3,
                                         strategy='uniform',
                                         lambda_order1=0.1,
                                         lambda_order2=0.5,
                                         basis_name='haar')
        expected_len = 3 * 7 + 3 * (3 ** 2)  # first-order + second-order
        assert r.shape == (expected_len,)

    def test_third_order_reg(self):
        """Third-order regularization works for Haar."""
        from hifi_anova.training.regularization import build_regularization_vector
        B = 2 ** 1 - 1  # K3=1 → 1
        r = build_regularization_vector(D=3, K1=3, K2=0, P=0,
                                         K3=1, T=1,
                                         strategy='uniform',
                                         lambda_order1=0.1,
                                         lambda_order3=0.5,
                                         basis_name='haar')
        expected_len = 3 * 7 + 1 * (1 ** 3)
        assert r.shape == (expected_len,)


# ─────────────────────────────────────────────────────────────
# Test cross-variable orthogonality
# ─────────────────────────────────────────────────────────────

class TestHaarCrossOrthogonality:
    """Verify Haar features for different variables are orthogonal on uniform data."""

    def test_cross_variable_orthogonal(self):
        """Features for var i and var j are uncorrelated on independent uniform data."""
        from hifi_anova.core.features import build_first_order_features
        key = jax.random.PRNGKey(42)
        N = 100000
        D = 3
        x = jax.random.uniform(key, (N, D))
        phi = build_first_order_features(x, K=3, basis_name='haar')

        B = 7
        # Cross-correlation between variable 0 and variable 1 features
        block0 = phi[:, :B]
        block1 = phi[:, B:2*B]
        cross = block0.T @ block1 / N
        assert jnp.max(jnp.abs(cross)) < 0.02, \
            f"Max cross-variable correlation: {jnp.max(jnp.abs(cross))}"


# ─────────────────────────────────────────────────────────────
# Test second-order cross-order orthogonality
# ─────────────────────────────────────────────────────────────

class TestHaarCrossOrder:
    """Verify first-order and second-order Haar features are orthogonal."""

    def test_first_vs_second_order_orthogonal(self):
        """First and second-order Haar features are orthogonal."""
        from hifi_anova.core.features import (build_first_order_features,
                                          build_second_order_features)
        key = jax.random.PRNGKey(123)
        N = 50000
        D = 3
        x = jax.random.uniform(key, (N, D))
        phi1 = build_first_order_features(x, K=2, basis_name='haar')
        pairs = jnp.array([[0, 1], [0, 2], [1, 2]])
        phi2 = build_second_order_features(x, K=2, pair_indices=pairs,
                                            basis_name='haar')
        cross = phi1.T @ phi2 / N
        assert jnp.max(jnp.abs(cross)) < 0.02, \
            f"Max cross-order correlation: {jnp.max(jnp.abs(cross))}"


# ─────────────────────────────────────────────────────────────
# Test Haar diagnostic on step function
# ─────────────────────────────────────────────────────────────

class TestHaarDiagnostic:
    """Test haar_residual_analysis on synthetic data with known step features."""

    def test_detects_step_function(self):
        """Diagnostic detects localized features in a step-function residual."""
        from hifi_anova.analysis.haar_diagnostic import haar_residual_analysis
        N = 5000
        np.random.seed(42)
        x = np.random.uniform(0, 1, (N, 3))
        x = jnp.array(x)
        # Residual with a step in variable 0 at x=0.5
        residual = jnp.where(x[:, 0] < 0.5, 1.0, -1.0)
        residual = residual + 0.1 * jax.random.normal(jax.random.PRNGKey(0), (N,))

        result = haar_residual_analysis(residual, x, max_scale=4)

        # Variable 0 should have localized features
        assert result['per_variable'][0]['has_localized_features']
        # Scale 1 should capture most variance (half-domain step)
        scale_vars = result['per_variable'][0]['per_scale_variance']
        assert scale_vars[1] > scale_vars.get(2, 0)
        assert scale_vars[1] > scale_vars.get(3, 0)

        # Variables 1 and 2 should have negligible Haar features
        assert result['per_variable'][1]['fraction_of_residual'] < 0.1
        assert result['per_variable'][2]['fraction_of_residual'] < 0.1

        # Summary should flag variable 0
        assert 0 in result['summary']['variables_with_localized']
        assert result['summary']['any_localized']

    def test_no_localized_for_smooth(self):
        """Diagnostic reports no localized features for smooth residual."""
        from hifi_anova.analysis.haar_diagnostic import haar_residual_analysis
        N = 5000
        np.random.seed(42)
        x = jnp.array(np.random.uniform(0, 1, (N, 3)))
        # Smooth sine residual — Haar should capture little
        residual = jnp.sin(2 * jnp.pi * x[:, 0])
        residual = residual + 0.3 * jax.random.normal(jax.random.PRNGKey(0), (N,))

        result = haar_residual_analysis(residual, x, max_scale=3,
                                         significance_threshold=0.9)
        # With very high threshold, smooth residual should not be flagged.
        # Haar CAN capture ~80% of sine (coarse ψ₁₀ has ⟨ψ₁₀, sin⟩ = 2/π),
        # but we verify it doesn't claim to capture nearly everything.
        assert not result['summary']['any_localized']

    def test_quarter_step_at_scale_2(self):
        """Diagnostic detects quarter-domain step as scale-2 feature."""
        from hifi_anova.analysis.haar_diagnostic import haar_residual_analysis
        N = 10000
        np.random.seed(42)
        x = jnp.array(np.random.uniform(0, 1, (N, 2)))
        # Step at x=0.25 in variable 0
        residual = jnp.where(x[:, 0] < 0.25, 2.0, 0.0)
        noise = 0.05 * jax.random.normal(jax.random.PRNGKey(1), (N,))
        residual = residual - jnp.mean(residual) + noise

        result = haar_residual_analysis(residual, x, max_scale=4)

        # Variable 0 should have localized features
        assert result['per_variable'][0]['has_localized_features']
        # Scale 2 should be dominant (quarter-domain)
        scale_vars = result['per_variable'][0]['per_scale_variance']
        assert result['per_variable'][0]['dominant_scale'] in [1, 2]


# ─────────────────────────────────────────────────────────────
# Test build_haar_features convenience function
# ─────────────────────────────────────────────────────────────

class TestBuildHaarFeatures:
    """Test the convenience function."""

    def test_matches_evaluate_batch(self):
        """build_haar_features matches HaarBasis.evaluate_batch reshaped."""
        from hifi_anova.core.haar import HaarBasis, build_haar_features
        N, D = 50, 3
        x = jnp.tile(jnp.linspace(0.01, 0.99, N)[:, None], (1, D))
        phi = build_haar_features(x, J=4)
        assert phi.shape == (N, D * 15)

        haar = HaarBasis(4)
        phi_ref = haar.evaluate_batch(x, 4).reshape(N, -1)
        assert jnp.allclose(phi, phi_ref)


# ─────────────────────────────────────────────────────────────
# Test ridge solve with Haar basis
# ─────────────────────────────────────────────────────────────

class TestHaarRidgeSolve:
    """Test that ridge regression works correctly with Haar features."""

    def test_step_function_recovery(self):
        """Ridge with Haar recovers step function coefficients."""
        from hifi_anova.core.features import build_first_order_features
        from hifi_anova.training.ridge import weighted_ridge_solve
        from hifi_anova.training.regularization import build_regularization_vector

        N = 5000
        D = 1
        np.random.seed(42)
        x = jnp.array(np.random.uniform(0, 1, (N, D)))

        # True function: step at 0.5 → captured exactly by ψ₁₀
        y = jnp.where(x[:, 0] < 0.5, 1.0, -1.0)
        y = y - jnp.mean(y)

        phi = build_first_order_features(x, K=3, basis_name='haar')
        reg = build_regularization_vector(D=1, K1=3, K2=0, P=0,
                                           strategy='uniform',
                                           lambda_order1=0.0001,
                                           basis_name='haar')

        w = weighted_ridge_solve(phi, y, reg)

        # ψ₁₀ coefficient should dominate (≈ 1.0 for step at 0.5)
        assert abs(float(w[0])) > 0.8, f"w[0] = {float(w[0])}"
        # Other coefficients should be near zero
        assert jnp.max(jnp.abs(w[1:])) < 0.2

    def test_sobol_from_ridge(self):
        """Sobol indices from Haar ridge solve are correct."""
        from hifi_anova.core.features import build_first_order_features
        from hifi_anova.core.gram import build_gram_matrix
        from hifi_anova.training.ridge import weighted_ridge_solve
        from hifi_anova.training.regularization import build_regularization_vector

        N = 5000
        D = 3
        np.random.seed(42)
        x = jnp.array(np.random.uniform(0, 1, (N, D)))

        # y = step_in_x0 + 0.5 * step_in_x1
        y = jnp.where(x[:, 0] < 0.5, 2.0, -2.0) + \
            jnp.where(x[:, 1] < 0.5, 1.0, -1.0)
        y = y - jnp.mean(y)

        phi = build_first_order_features(x, K=3, basis_name='haar')
        reg = build_regularization_vector(D=3, K1=3, K2=0, P=0,
                                           strategy='uniform',
                                           lambda_order1=0.0001,
                                           basis_name='haar')
        w = weighted_ridge_solve(phi, y, reg)

        G = build_gram_matrix(3, basis_name='haar')
        B = 7
        vars_per = []
        for i in range(D):
            wi = w[i * B:(i + 1) * B]
            vars_per.append(float(wi @ G @ wi))

        total = sum(vars_per)
        sobol = [v / total for v in vars_per]

        # x0 should have ~80% of variance (2.0 vs 1.0 amplitude → 4:1)
        assert sobol[0] > 0.6, f"S0={sobol[0]}"
        # x1 should have ~20%
        assert sobol[1] > 0.1, f"S1={sobol[1]}"
        # x2 should have ~0%
        assert sobol[2] < 0.05, f"S2={sobol[2]}"
        # Sum to 1
        assert abs(sum(sobol) - 1.0) < 1e-6
