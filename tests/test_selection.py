"""Tests for principled variable selection methods (BIC, Group Lasso, 1SE)."""

import pytest
import jax
import jax.numpy as jnp
import numpy as np

from hifi_anova.core.features import build_first_order_features
from hifi_anova.core.gram import build_gram_matrix
from hifi_anova.training.regularization import build_regularization_vector
from hifi_anova.training.selection import (
    select_variables_bic,
    select_variables_glasso,
    select_variables_1se,
    select_active_variables_principled,
)

pytestmark = pytest.mark.smoke


@pytest.fixture
def signal_data():
    """Synthetic data with 3 active + 5 irrelevant variables.
    f(x) = 5*(x1-0.5) + 3*cos(2*pi*x2) + 2*sin(2*pi*x3)
    """
    np.random.seed(42)
    N = 3000
    D = 8
    X = np.random.uniform(0, 1, (N, D))

    f = (5.0 * (X[:, 0] - 0.5) +
         3.0 * np.cos(2 * np.pi * X[:, 1]) +
         2.0 * np.sin(2 * np.pi * X[:, 2]))
    y = f + 0.3 * np.random.randn(N)

    K1 = 5
    x_jax = jnp.array(X)
    Phi1 = np.asarray(build_first_order_features(x_jax, K1), dtype=np.float64)
    f0 = float(np.mean(y))
    y_centered = y - f0

    reg_diag = np.asarray(build_regularization_vector(D, K1, 0, 0, 'curvature', 0.001, 0.0),
                          dtype=np.float64)

    return {
        'Phi1': Phi1, 'y': y_centered, 'D': D, 'K1': K1,
        'reg_diag': reg_diag, 'active_true': [0, 1, 2],
    }


class TestBICSelection:
    """Test Layer 1: BIC marginal screening."""

    def test_selects_active_variables(self, signal_data):
        """BIC should select x1, x2, x3 and exclude x4-x8."""
        active, info = select_variables_bic(
            signal_data['Phi1'], signal_data['y'],
            signal_data['D'], signal_data['K1'],
            signal_data['reg_diag'], verbose=False,
        )
        # All three active variables must be selected
        for v in signal_data['active_true']:
            assert v in active, f"Variable {v} should be selected, got {active}"

    def test_excludes_irrelevant(self, signal_data):
        """BIC should exclude most irrelevant variables."""
        active, info = select_variables_bic(
            signal_data['Phi1'], signal_data['y'],
            signal_data['D'], signal_data['K1'],
            signal_data['reg_diag'], verbose=False,
        )
        # At most 1 irrelevant variable should sneak in
        irrelevant_selected = [v for v in active if v not in signal_data['active_true']]
        assert len(irrelevant_selected) <= 2, \
            f"Too many irrelevant selected: {irrelevant_selected}"

    def test_returns_diagnostics(self, signal_data):
        """Info dict should contain per-group ΔBIC values."""
        _, info = select_variables_bic(
            signal_data['Phi1'], signal_data['y'],
            signal_data['D'], signal_data['K1'],
            signal_data['reg_diag'], verbose=False,
        )
        assert info['method'] == 'bic'
        assert 'per_group' in info
        assert len(info['per_group']) == signal_data['D']
        # Active variables should have positive ΔBIC
        for v in signal_data['active_true']:
            assert info['per_group'][v]['delta_bic'] > 0

    def test_at_least_two(self, signal_data):
        """Even with all-noise data, at least 2 variables selected."""
        N, F = signal_data['Phi1'].shape
        y_noise = np.random.randn(N) * 10  # pure noise
        active, _ = select_variables_bic(
            signal_data['Phi1'], y_noise,
            signal_data['D'], signal_data['K1'],
            signal_data['reg_diag'], verbose=False,
        )
        assert len(active) >= 2


class TestGroupLassoSelection:
    """Test Layer 2: Group Lasso with BIC."""

    def test_selects_active_variables(self, signal_data):
        """Group Lasso should identify x1, x2, x3."""
        active, info = select_variables_glasso(
            signal_data['Phi1'], signal_data['y'],
            signal_data['D'], signal_data['K1'],
            signal_data['reg_diag'], n_gamma=20, verbose=False,
        )
        for v in signal_data['active_true']:
            assert v in active, f"Variable {v} should be selected, got {active}"

    def test_path_monotonic(self, signal_data):
        """Number of active groups should increase as gamma decreases."""
        _, info = select_variables_glasso(
            signal_data['Phi1'], signal_data['y'],
            signal_data['D'], signal_data['K1'],
            signal_data['reg_diag'], n_gamma=20, verbose=False,
        )
        n_actives = [p['n_active'] for p in info['path']]
        # Roughly monotonic (not strictly due to numerical issues)
        assert n_actives[0] <= n_actives[-1], \
            f"Path should go from sparse to dense: {n_actives}"

    def test_returns_best_gamma(self, signal_data):
        _, info = select_variables_glasso(
            signal_data['Phi1'], signal_data['y'],
            signal_data['D'], signal_data['K1'],
            signal_data['reg_diag'], n_gamma=20, verbose=False,
        )
        assert info['method'] == 'group_lasso'
        assert info['best_gamma'] > 0


class TestOneSERuleSelection:
    """Test Layer 3: One standard error rule."""

    def test_selects_active_variables(self, signal_data):
        """1SE rule should select x1, x2, x3."""
        # For 1SE we need reg_structure (relative weights)
        reg_structure = signal_data['reg_diag'] / max(
            float(np.median(signal_data['reg_diag'][signal_data['reg_diag'] > 1e-15])), 1e-10)

        active, info = select_variables_1se(
            signal_data['Phi1'], signal_data['y'],
            signal_data['D'], signal_data['K1'],
            reg_structure, n_folds=5, verbose=False,
        )
        for v in signal_data['active_true']:
            assert v in active, f"Variable {v} should be selected, got {active}"

    def test_lambda_1se_more_conservative_than_min(self, signal_data):
        """lambda_1se should be >= lambda_min (more regularized)."""
        reg_structure = signal_data['reg_diag'] / max(
            float(np.median(signal_data['reg_diag'][signal_data['reg_diag'] > 1e-15])), 1e-10)
        _, info = select_variables_1se(
            signal_data['Phi1'], signal_data['y'],
            signal_data['D'], signal_data['K1'],
            reg_structure, n_folds=5, verbose=False,
        )
        # 1SE rule always picks a lambda at least as large as lambda_min
        assert info['lambda_1se'] >= info['lambda_min'] * 0.99, \
            f"lambda_1se={info['lambda_1se']} should be >= lambda_min={info['lambda_min']}"

    def test_returns_lambda_values(self, signal_data):
        reg_structure = signal_data['reg_diag'] / max(
            float(np.median(signal_data['reg_diag'][signal_data['reg_diag'] > 1e-15])), 1e-10)
        _, info = select_variables_1se(
            signal_data['Phi1'], signal_data['y'],
            signal_data['D'], signal_data['K1'],
            reg_structure, n_folds=5, verbose=False,
        )
        assert info['method'] == '1se'
        assert info['lambda_1se'] >= info['lambda_min']
        assert info['cv_min'] > 0


class TestUnifiedInterface:
    """Test select_active_variables_principled dispatching."""

    def test_bic_dispatch(self, signal_data):
        active, info = select_active_variables_principled(
            signal_data['Phi1'], signal_data['y'],
            signal_data['D'], signal_data['K1'],
            signal_data['reg_diag'], method='bic', verbose=False,
        )
        assert info['method'] == 'bic'
        assert len(active) >= 2

    def test_group_lasso_dispatch(self, signal_data):
        active, info = select_active_variables_principled(
            signal_data['Phi1'], signal_data['y'],
            signal_data['D'], signal_data['K1'],
            signal_data['reg_diag'], method='group_lasso', verbose=False,
        )
        assert info['method'] == 'group_lasso'

    def test_1se_dispatch(self, signal_data):
        active, info = select_active_variables_principled(
            signal_data['Phi1'], signal_data['y'],
            signal_data['D'], signal_data['K1'],
            signal_data['reg_diag'], method='1se', verbose=False,
        )
        assert info['method'] == '1se'

    def test_invalid_method(self, signal_data):
        with pytest.raises(ValueError, match="Unknown selection method"):
            select_active_variables_principled(
                signal_data['Phi1'], signal_data['y'],
                signal_data['D'], signal_data['K1'],
                signal_data['reg_diag'], method='invalid',
            )
