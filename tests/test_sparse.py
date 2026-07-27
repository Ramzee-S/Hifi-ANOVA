"""Tests for L1 / sparse solvers (Lasso, Group Lasso, Elastic Net, SGL)."""

import pytest
import jax
import jax.numpy as jnp
import numpy as np

from hifi_anova.core.features import build_first_order_features
from hifi_anova.core.gram import build_gram_matrix
from hifi_anova.training.regularization import build_regularization_vector
from hifi_anova.training.sparse import (
    lasso_solve, elastic_net_solve, group_lasso_solve,
    sparse_group_lasso_solve, sparse_solve,
)

pytestmark = pytest.mark.smoke


@pytest.fixture
def sparse_data():
    """3 active + 5 irrelevant variables, moderate noise."""
    np.random.seed(42)
    N = 2000
    D = 8
    X = np.random.uniform(0, 1, (N, D))
    f = 5.0 * (X[:, 0] - 0.5) + 3.0 * np.cos(2 * np.pi * X[:, 1]) + 2.0 * np.sin(2 * np.pi * X[:, 2])
    y = f + 0.5 * np.random.randn(N)

    K1 = 3
    x_jax = jnp.array(X)
    Phi1 = np.asarray(build_first_order_features(x_jax, K1), dtype=np.float64)
    f0 = float(np.mean(y))
    y_c = y - f0
    reg = np.asarray(build_regularization_vector(D, K1, 0, 0, 'variance', 0.001, 0),
                      dtype=np.float64)
    block = 2 * K1 + 1
    group_slices = [slice(i * block, (i + 1) * block) for i in range(D)]
    G1 = np.asarray(build_gram_matrix(K1), dtype=np.float64)
    gram_matrices = [G1] * D

    return {
        'Phi': Phi1, 'y': y_c, 'reg': reg, 'D': D, 'K1': K1,
        'group_slices': group_slices, 'gram_matrices': gram_matrices,
        'active_true': [0, 1, 2], 'block': block,
    }


class TestLasso:
    def test_produces_sparsity(self, sparse_data):
        """Lasso should zero out some coefficients."""
        w = lasso_solve(sparse_data['Phi'], sparse_data['y'],
                         sparse_data['reg'], alpha_l1=0.05)
        n_zero = np.sum(np.abs(w) < 1e-10)
        assert n_zero > 0, "Lasso should produce some zero coefficients"

    def test_stronger_l1_more_sparse(self, sparse_data):
        """Higher alpha should produce more zeros."""
        w_weak = lasso_solve(sparse_data['Phi'], sparse_data['y'],
                              sparse_data['reg'], alpha_l1=0.01)
        w_strong = lasso_solve(sparse_data['Phi'], sparse_data['y'],
                                sparse_data['reg'], alpha_l1=0.5)
        n_zero_weak = np.sum(np.abs(w_weak) < 1e-10)
        n_zero_strong = np.sum(np.abs(w_strong) < 1e-10)
        assert n_zero_strong >= n_zero_weak


class TestElasticNet:
    def test_l1_ratio_0_is_ridge_like(self, sparse_data):
        """l1_ratio=0 should produce no exact zeros (pure ridge)."""
        w = elastic_net_solve(sparse_data['Phi'], sparse_data['y'],
                               sparse_data['reg'], alpha=0.01, l1_ratio=0.0)
        n_zero = np.sum(np.abs(w) < 1e-10)
        # Pure L2 shouldn't produce exact zeros (may have tiny values)
        assert n_zero < sparse_data['Phi'].shape[1] // 2

    def test_l1_ratio_1_is_lasso_like(self, sparse_data):
        """l1_ratio=1 should be sparse like lasso."""
        w = elastic_net_solve(sparse_data['Phi'], sparse_data['y'],
                               sparse_data['reg'], alpha=0.1, l1_ratio=1.0)
        n_zero = np.sum(np.abs(w) < 1e-10)
        assert n_zero > 0


class TestGroupLasso:
    def test_zeros_entire_groups(self, sparse_data):
        """Group Lasso should zero out entire variable groups."""
        w = group_lasso_solve(sparse_data['Phi'], sparse_data['y'],
                               sparse_data['group_slices'], sparse_data['reg'],
                               gamma=0.05)
        block = sparse_data['block']
        for g in range(sparse_data['D']):
            sl = sparse_data['group_slices'][g]
            group_norm = np.linalg.norm(w[sl])
            # Group should be either all zero or all nonzero
            n_zero_in_group = np.sum(np.abs(w[sl]) < 1e-10)
            assert n_zero_in_group == 0 or n_zero_in_group == block, \
                f"Group {g} partially zero: {n_zero_in_group}/{block}"

    def test_keeps_active_variables(self, sparse_data):
        """With small gamma, active variables should survive."""
        w = group_lasso_solve(sparse_data['Phi'], sparse_data['y'],
                               sparse_data['group_slices'], sparse_data['reg'],
                               gamma=0.0005)
        for v in sparse_data['active_true']:
            sl = sparse_data['group_slices'][v]
            assert np.linalg.norm(w[sl]) > 1e-5, \
                f"Active variable {v} was zeroed out"

    def test_gram_weighted(self, sparse_data):
        """Gram-weighted group lasso should also work."""
        w = group_lasso_solve(sparse_data['Phi'], sparse_data['y'],
                               sparse_data['group_slices'], sparse_data['reg'],
                               gamma=0.0005,
                               gram_matrices=sparse_data['gram_matrices'])
        for v in sparse_data['active_true']:
            sl = sparse_data['group_slices'][v]
            assert np.linalg.norm(w[sl]) > 1e-5


class TestSparseGroupLasso:
    def test_group_and_element_sparsity(self, sparse_data):
        """SGL combines group and element sparsity penalties.

        At moderate gamma, SGL should shrink coefficients more aggressively
        than pure group lasso due to the additional L1 penalty.
        """
        w_sgl = sparse_group_lasso_solve(
            sparse_data['Phi'], sparse_data['y'],
            sparse_data['group_slices'], sparse_data['reg'],
            gamma_group=0.01, gamma_l1=0.01)

        w_gl = group_lasso_solve(
            sparse_data['Phi'], sparse_data['y'],
            sparse_data['group_slices'], sparse_data['reg'],
            gamma=0.01)

        # SGL should produce sparser solutions than pure GL
        # (more near-zero elements, or at least as many)
        n_small_sgl = int(np.sum(np.abs(w_sgl) < 1e-6))
        n_small_gl = int(np.sum(np.abs(w_gl) < 1e-6))
        assert n_small_sgl >= n_small_gl, (
            f"SGL ({n_small_sgl} near-zero) should be at least as sparse as "
            f"GL ({n_small_gl} near-zero)")


class TestUnifiedSparse:
    def test_dispatch_lasso(self, sparse_data):
        w, info = sparse_solve(sparse_data['Phi'], sparse_data['y'],
                                sparse_data['reg'], method='lasso', alpha_l1=0.05)
        assert info['method'] == 'lasso'
        assert info['sparsity'] > 0

    def test_dispatch_group_lasso(self, sparse_data):
        w, info = sparse_solve(
            sparse_data['Phi'], sparse_data['y'], sparse_data['reg'],
            method='group_lasso', group_slices=sparse_data['group_slices'],
            gamma=0.02)
        assert info['method'] == 'group_lasso'
        assert 'n_active_groups' in info

    def test_dispatch_elastic_net(self, sparse_data):
        w, info = sparse_solve(sparse_data['Phi'], sparse_data['y'],
                                sparse_data['reg'], method='elastic_net',
                                alpha=0.05, l1_ratio=0.5)
        assert info['method'] == 'elastic_net'
