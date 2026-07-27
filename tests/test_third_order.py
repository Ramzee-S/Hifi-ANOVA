"""Tests for third-order Fourier features, TripleManager, and G3 Gram matrix."""

import pytest
import jax
import jax.numpy as jnp
import numpy as np

from hifi_anova.core.features import (
    build_first_order_features,
    build_second_order_features,
    build_third_order_features,
    build_per_variable_basis,
)
from hifi_anova.core.gram import build_gram_matrix, build_gram_matrix_2d, build_gram_matrix_3d
from hifi_anova.core.pairs import PairManager, TripleManager


# =============================================================================
# Third-order feature construction
# =============================================================================

class TestThirdOrderFeatures:
    """Test build_third_order_features shape, properties, and orthogonality."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.N = 500
        self.D = 5
        self.K = 1
        key = jax.random.PRNGKey(42)
        self.x = jax.random.uniform(key, (self.N, self.D))
        self.triple_mgr = TripleManager(self.D)

    def test_shape(self):
        """Phi3 shape must be (N, T*(2K+1)^3)."""
        phi3 = build_third_order_features(self.x, self.K,
                                           self.triple_mgr.triple_indices)
        T = self.triple_mgr.T
        block = (2 * self.K + 1) ** 3
        assert phi3.shape == (self.N, T * block)

    def test_shape_K2(self):
        """Third-order with K=2."""
        K = 2
        phi3 = build_third_order_features(self.x, K,
                                           self.triple_mgr.triple_indices)
        block = (2 * K + 1) ** 3  # 125
        assert phi3.shape == (self.N, self.triple_mgr.T * block)

    def test_zero_mean(self):
        """All third-order features should have approximately zero mean
        on uniform [0,1] data (Hoeffding vanishing-integral condition)."""
        N_large = 100000
        key = jax.random.PRNGKey(99)
        x_large = jax.random.uniform(key, (N_large, self.D))
        phi3 = build_third_order_features(x_large, self.K,
                                           self.triple_mgr.triple_indices)
        means = jnp.mean(phi3, axis=0)
        max_mean = float(jnp.max(jnp.abs(means)))
        assert max_mean < 0.02, f"max mean = {max_mean}"

    def test_orthogonal_to_first_order(self):
        """Third-order features should be approximately orthogonal
        to first-order features on uniform data."""
        N_large = 50000
        key = jax.random.PRNGKey(77)
        x = jax.random.uniform(key, (N_large, self.D))
        phi1 = build_first_order_features(x, self.K)
        phi3 = build_third_order_features(x, self.K,
                                           self.triple_mgr.triple_indices)
        # Cross-correlation: Phi1^T @ Phi3 / N
        cross = phi1.T @ phi3 / N_large
        max_cross = float(jnp.max(jnp.abs(cross)))
        assert max_cross < 0.02, f"max cross 1-3 = {max_cross}"

    def test_orthogonal_to_second_order(self):
        """Third-order features should be approximately orthogonal
        to second-order features on uniform data."""
        N_large = 50000
        key = jax.random.PRNGKey(88)
        x = jax.random.uniform(key, (N_large, self.D))
        pair_mgr = PairManager(self.D)
        phi2 = build_second_order_features(x, self.K, pair_mgr.pair_indices)
        phi3 = build_third_order_features(x, self.K,
                                           self.triple_mgr.triple_indices)
        cross = phi2.T @ phi3 / N_large
        max_cross = float(jnp.max(jnp.abs(cross)))
        assert max_cross < 0.02, f"max cross 2-3 = {max_cross}"


# =============================================================================
# TripleManager
# =============================================================================

class TestTripleManager:
    """Test TripleManager enumeration and selection modes."""

    def test_all_triples_count(self):
        """C(D,3) triples for 'all' mode."""
        D = 6
        tm = TripleManager(D)
        assert tm.T == 20  # C(6,3) = 20
        assert tm.triple_indices.shape == (20, 3)

    def test_all_triples_D10(self):
        D = 10
        tm = TripleManager(D)
        assert tm.T == 120  # C(10,3) = 120

    def test_all_active_selection(self):
        """Only triples where all three variables are active."""
        D = 8
        active = [0, 1, 2, 3]  # 4 active out of 8
        tm = TripleManager(D, active_variables=active, selection_mode='all_active')
        assert tm.T == 4  # C(4,3) = 4

    def test_two_active_selection(self):
        """Triples where at least two variables are active."""
        D = 5
        active = [0, 1]  # 2 active out of 5
        tm = TripleManager(D, active_variables=active, selection_mode='two_active')
        # Triples with at least 2 of {0,1}: (0,1,x) for x in {2,3,4} = 3
        # Plus (0,1,2), (0,1,3), (0,1,4) — wait, already counted
        # Actually: triples with both 0 and 1 = C(3,1) = 3
        assert tm.T == 3

    def test_one_active_selection(self):
        """Triples where at least one variable is active."""
        D = 5
        active = [0]
        tm = TripleManager(D, active_variables=active, selection_mode='one_active')
        # Triples containing 0: C(4,2) = 6
        assert tm.T == 6

    def test_triple_to_variables(self):
        tm = TripleManager(5)
        i, j, k = tm.triple_to_variables(0)
        assert i == 0 and j == 1 and k == 2

    def test_find_triple_index(self):
        tm = TripleManager(5)
        idx = tm.find_triple_index(0, 1, 2)
        assert idx == 0
        idx2 = tm.find_triple_index(2, 1, 0)  # reversed order
        assert idx2 == 0  # should canonicalize

    def test_find_missing_triple(self):
        D = 5
        active = [0, 1, 2]
        tm = TripleManager(D, active_variables=active, selection_mode='all_active')
        # Only triple is (0,1,2)
        assert tm.find_triple_index(0, 1, 3) == -1

    def test_triple_slice(self):
        K = 1
        tm = TripleManager(5)
        s = tm.triple_slice(0, K)
        block = (2 * K + 1) ** 3  # 27
        assert s == slice(0, 27)
        s1 = tm.triple_slice(1, K)
        assert s1 == slice(27, 54)

    def test_empty_triples(self):
        """With only 2 active variables, all_active gives 0 triples."""
        D = 10
        active = [0, 1]
        tm = TripleManager(D, active_variables=active, selection_mode='all_active')
        assert tm.T == 0
        assert tm.triple_indices.shape == (0, 3)


# =============================================================================
# Third-order Gram matrix
# =============================================================================

class TestGramMatrix3D:
    """Test G3 = kron(G1, kron(G1, G1))."""

    def test_shape(self):
        K = 2
        G1 = build_gram_matrix(K)
        G3 = build_gram_matrix_3d(G1)
        size = (2 * K + 1) ** 3  # 125
        assert G3.shape == (size, size)

    def test_symmetry(self):
        K = 2
        G1 = build_gram_matrix(K)
        G3 = build_gram_matrix_3d(G1)
        assert jnp.allclose(G3, G3.T, atol=1e-14)

    def test_positive_semidefinite(self):
        K = 2
        G1 = build_gram_matrix(K)
        G3 = build_gram_matrix_3d(G1)
        eigenvalues = jnp.linalg.eigvalsh(G3)
        assert jnp.all(eigenvalues >= -1e-10)

    def test_kronecker_structure(self):
        """G3 = kron(G1, G2) where G2 = kron(G1, G1)."""
        K = 1
        G1 = build_gram_matrix(K)
        G2 = build_gram_matrix_2d(G1)
        G3 = build_gram_matrix_3d(G1)
        G3_manual = jnp.kron(G1, G2)
        assert jnp.allclose(G3, G3_manual, atol=1e-14)

    def test_variance_computation(self):
        """w^T G3 w should give correct variance for known triple product."""
        K = 1
        G1 = build_gram_matrix(K)
        G3 = build_gram_matrix_3d(G1)

        # Triple product: sin(2*pi*x1) * sin(2*pi*x2) * sin(2*pi*x3)
        # Each sin is at index 2 in basis [lin, cos1, sin1]
        # Coefficient vector: all zeros except position for (sin1, sin1, sin1)
        block = 2 * K + 1  # 3
        w = jnp.zeros(block ** 3)
        # Index of (sin1, sin1, sin1) = 2*9 + 2*3 + 2 = 26
        idx = 2 * block ** 2 + 2 * block + 2
        coeff = 3.0
        w = w.at[idx].set(coeff)

        # Var(c * sin*sin*sin) = c^2 * (1/2)^3 = 9 * 1/8 = 1.125
        expected_var = coeff ** 2 * 0.5 ** 3
        computed_var = float(w @ G3 @ w)
        assert abs(computed_var - expected_var) < 1e-10, \
            f"Expected {expected_var}, got {computed_var}"

    def test_empirical_gram_match(self):
        """Empirical Gram from large uniform samples should match analytic G3."""
        K = 1
        N_large = 200000
        D = 3
        key = jax.random.PRNGKey(42)
        x = jax.random.uniform(key, (N_large, D))
        triple_indices = jnp.array([[0, 1, 2]])

        phi3 = build_third_order_features(x, K, triple_indices)
        # Empirical Gram: phi3_block^T @ phi3_block / N
        # phi3 is (N, 27) for one triple
        empirical_G3 = phi3.T @ phi3 / N_large

        G1 = build_gram_matrix(K)
        G3 = build_gram_matrix_3d(G1)
        assert jnp.allclose(empirical_G3, G3, atol=0.02), \
            f"Max diff = {float(jnp.max(jnp.abs(empirical_G3 - G3)))}"
