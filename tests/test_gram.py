"""Tests for Gram matrix and derivative penalty."""

import jax.numpy as jnp
import numpy as np
import pytest

from hifi_anova.core.gram import build_gram_matrix, build_gram_matrix_2d, build_derivative_penalty

pytestmark = pytest.mark.smoke


class TestGramMatrix:
    def test_shape(self):
        for K in [1, 3, 5, 10]:
            G = build_gram_matrix(K)
            assert G.shape == (2 * K + 1, 2 * K + 1)

    def test_symmetric(self):
        G = build_gram_matrix(10)
        assert jnp.allclose(G, G.T, atol=1e-15)

    def test_psd(self):
        G = build_gram_matrix(10)
        eigenvalues = jnp.linalg.eigvalsh(G)
        assert jnp.all(eigenvalues >= -1e-10)

    def test_diagonal_entries(self):
        K = 5
        G = build_gram_matrix(K)
        # G[0,0] = 1/12
        assert jnp.isclose(G[0, 0], 1.0 / 12.0, rtol=1e-10)
        # G[2k-1, 2k-1] = G[2k, 2k] = 1/2
        for k in range(1, K + 1):
            assert jnp.isclose(G[2 * k - 1, 2 * k - 1], 0.5, rtol=1e-10)
            assert jnp.isclose(G[2 * k, 2 * k], 0.5, rtol=1e-10)

    def test_off_diagonal_entries(self):
        K = 5
        G = build_gram_matrix(K)
        # G[0, 2k] = -1/(2*pi*k) for the linear-sin cross terms
        for k in range(1, K + 1):
            expected = -1.0 / (2.0 * np.pi * k)
            assert jnp.isclose(G[0, 2 * k], expected, rtol=1e-10)
            assert jnp.isclose(G[2 * k, 0], expected, rtol=1e-10)

    def test_known_variance_linear_only(self):
        """w = (1, 0, ..., 0) => w^T G w = 1/12."""
        K = 5
        G = build_gram_matrix(K)
        w = jnp.zeros(2 * K + 1)
        w = w.at[0].set(1.0)
        var = w @ G @ w
        assert jnp.isclose(var, 1.0 / 12.0, rtol=1e-10)

    def test_known_variance_cos1_only(self):
        """w = (0, 1, 0, ...) => w^T G w = 1/2."""
        K = 5
        G = build_gram_matrix(K)
        w = jnp.zeros(2 * K + 1)
        w = w.at[1].set(1.0)
        var = w @ G @ w
        assert jnp.isclose(var, 0.5, rtol=1e-10)

    def test_known_variance_cross_term(self):
        """w = (1, 0, 1, 0, ...) (linear + sin1).
        Var = 1/12 + 1/2 + 2*(-1/(2*pi)) = 1/12 + 1/2 - 1/pi."""
        K = 5
        G = build_gram_matrix(K)
        w = jnp.zeros(2 * K + 1)
        w = w.at[0].set(1.0)  # linear
        w = w.at[2].set(1.0)  # sin1
        var = w @ G @ w
        expected = 1.0 / 12.0 + 0.5 + 2.0 * (-1.0 / (2.0 * np.pi))
        assert jnp.isclose(var, expected, rtol=1e-8)


class TestGramMatrix2D:
    def test_shape(self):
        K = 3
        G1 = build_gram_matrix(K)
        G2 = build_gram_matrix_2d(G1)
        size = (2 * K + 1) ** 2
        assert G2.shape == (size, size)

    def test_is_kronecker(self):
        K = 3
        G1 = build_gram_matrix(K)
        G2 = build_gram_matrix_2d(G1)
        # Verify it's the Kronecker product
        expected = jnp.kron(G1, G1)
        assert jnp.allclose(G2, expected, atol=1e-14)


class TestDerivativePenalty:
    def test_shape(self):
        K = 5
        D2 = build_derivative_penalty(K, p=2)
        assert D2.shape == (2 * K + 1,)

    def test_linear_unpenalized_curvature(self):
        """D(2)[0,0] = 0: linear term has zero curvature."""
        K = 10
        D2 = build_derivative_penalty(K, p=2)
        assert D2[0] == 0.0

    def test_linear_penalized_first_derivative(self):
        """D(1)[0,0] = 1: linear term has unit first derivative."""
        K = 10
        D1 = build_derivative_penalty(K, p=1)
        assert D1[0] == 1.0

    def test_frequency_scaling(self):
        """D(2)[2k-1] = D(2)[2k] = (2*pi*k)^4 / 2."""
        K = 5
        D2 = build_derivative_penalty(K, p=2)
        for k in range(1, K + 1):
            expected = (2.0 * np.pi * k) ** 4 / 2.0
            assert jnp.isclose(D2[2 * k - 1], expected, rtol=1e-10)
            assert jnp.isclose(D2[2 * k], expected, rtol=1e-10)
