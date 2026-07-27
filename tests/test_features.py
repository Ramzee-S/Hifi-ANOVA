"""Tests for feature construction."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hifi_anova.core.features import (
    build_per_variable_basis, build_first_order_features,
    build_second_order_features
)
from hifi_anova.core.gram import build_gram_matrix
from hifi_anova.core.pairs import PairManager

pytestmark = pytest.mark.smoke


class TestPerVariableBasis:
    def test_shape(self):
        x = jax.random.uniform(jax.random.PRNGKey(0), (100, 5))
        K = 3
        basis = build_per_variable_basis(x, K)
        assert basis.shape == (100, 5, 2 * K + 1)

    def test_zero_mean_columns(self):
        """All basis functions should have zero mean on uniform [0,1]."""
        x = jax.random.uniform(jax.random.PRNGKey(0), (100_000, 3))
        K = 5
        basis = build_per_variable_basis(x, K)
        means = jnp.mean(basis, axis=0)  # (D, 2K+1)
        assert jnp.max(jnp.abs(means)) < 0.01


class TestFirstOrderFeatures:
    def test_shape(self):
        x = jax.random.uniform(jax.random.PRNGKey(0), (100, 5))
        K = 3
        phi = build_first_order_features(x, K)
        assert phi.shape == (100, 5 * (2 * K + 1))

    def test_empirical_gram_matches_analytic(self):
        """Critical test: phi^T phi / N should approximate G block-diagonal."""
        key = jax.random.PRNGKey(42)
        N = 200_000
        D = 3
        K = 5
        x = jax.random.uniform(key, (N, D))
        phi = build_first_order_features(x, K)

        # Empirical Gram for first variable block
        block_size = 2 * K + 1
        phi_var0 = phi[:, :block_size]
        empirical_G = (phi_var0.T @ phi_var0) / N

        # Analytic Gram
        G = build_gram_matrix(K)

        assert jnp.allclose(empirical_G, G, atol=0.015)

    def test_cross_variable_orthogonality(self):
        """Features from different variables should be approximately orthogonal."""
        key = jax.random.PRNGKey(1)
        N = 200_000
        D = 3
        K = 3
        x = jax.random.uniform(key, (N, D))
        phi = build_first_order_features(x, K)

        block = 2 * K + 1
        phi_0 = phi[:, :block]
        phi_1 = phi[:, block:2*block]
        cross = (phi_0.T @ phi_1) / N
        assert jnp.max(jnp.abs(cross)) < 0.01


class TestSecondOrderFeatures:
    def test_shape(self):
        D = 5
        K = 3
        x = jax.random.uniform(jax.random.PRNGKey(0), (100, D))
        pm = PairManager(D)
        phi2 = build_second_order_features(x, K, pm.pair_indices)
        expected_cols = pm.P * (2 * K + 1) ** 2
        assert phi2.shape == (100, expected_cols)

    def test_cross_order_orthogonality(self):
        """First-order and second-order features should be approximately orthogonal."""
        key = jax.random.PRNGKey(2)
        N = 200_000
        D = 4
        K = 3
        x = jax.random.uniform(key, (N, D))

        phi1 = build_first_order_features(x, K)
        pm = PairManager(D)
        phi2 = build_second_order_features(x, K, pm.pair_indices)

        cross = (phi1.T @ phi2) / N
        assert jnp.max(jnp.abs(cross)) < 0.02

    def test_empirical_gram_2d(self):
        """Second-order empirical Gram should match G tensor G."""
        key = jax.random.PRNGKey(3)
        N = 200_000
        D = 3
        K = 2
        x = jax.random.uniform(key, (N, D))

        pm = PairManager(D)
        phi2 = build_second_order_features(x, K, pm.pair_indices)

        # First pair block
        block_size = (2 * K + 1) ** 2
        phi2_pair0 = phi2[:, :block_size]
        empirical_G2 = (phi2_pair0.T @ phi2_pair0) / N

        # Analytic: G kron G
        from hifi_anova.core.gram import build_gram_matrix_2d
        G1 = build_gram_matrix(K)
        G2 = build_gram_matrix_2d(G1)

        assert jnp.allclose(empirical_G2, G2, atol=0.02)
