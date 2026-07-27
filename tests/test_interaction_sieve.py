"""Tests for interaction discovery sieve.

Validates that the sieve correctly identifies missing interactions
by projecting residuals onto unfitted subspaces.
"""

import pytest
import jax
import jax.numpy as jnp
import numpy as np

from hifi_anova.core.features import build_first_order_features, build_second_order_features
from hifi_anova.core.gram import build_gram_matrix, build_gram_matrix_2d
from hifi_anova.core.pairs import PairManager
from hifi_anova.training.regularization import build_regularization_vector
from hifi_anova.training.ridge import weighted_ridge_solve
from hifi_anova.training.trainer import HiFiANOVATrainer
from hifi_anova.analysis.interaction_discovery import (
    scan_missing_pairs,
    _project_residual_score,
)

pytestmark = pytest.mark.integration


def _fit_model(X, y, config):
    """Helper: split data, fit model, return (model, x_all, y_all)."""
    N = X.shape[0]
    n_val = max(N // 5, 20)
    x_train = jnp.array(X[n_val:])
    y_train = jnp.array(y[n_val:])
    x_val = jnp.array(X[:n_val])
    y_val = jnp.array(y[:n_val])
    trainer = HiFiANOVATrainer(config)
    model, _ = trainer.fit(x_train, y_train, x_val, y_val)
    return model


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def interaction_data():
    """Function with a known second-order interaction."""
    np.random.seed(42)
    N = 5000
    D = 5
    K1 = 5
    X = np.random.uniform(0, 1, (N, D))
    f = (5.0 * (X[:, 0] - 0.5)
         + 3.0 * (X[:, 1] - 0.5)
         + 4.0 * (X[:, 0] - 0.5) * (X[:, 1] - 0.5))
    y = f + 0.5 * np.random.randn(N)
    return X, y, D, K1


@pytest.fixture
def pure_noise_data():
    """Pure first-order function with no interactions."""
    np.random.seed(123)
    N = 3000
    D = 4
    K1 = 5
    X = np.random.uniform(0, 1, (N, D))
    f = 5.0 * (X[:, 0] - 0.5) + 3.0 * np.cos(2 * np.pi * X[:, 1])
    y = f + 0.5 * np.random.randn(N)
    return X, y, D, K1


# ============================================================================
# Tests for _project_residual_score (core helper)
# ============================================================================

class TestProjectResidualScore:

    def test_perfect_signal_in_subspace(self):
        """When the residual IS in the subspace, score should be near 1."""
        np.random.seed(42)
        N = 1000
        X = np.random.uniform(0, 1, (N, 2))
        residual = 4.0 * (X[:, 0] - 0.5) * (X[:, 1] - 0.5)
        residual_var = float(np.var(residual))

        pair_idx = jnp.array([[0, 1]], dtype=jnp.int32)
        Phi_pair = np.asarray(
            build_second_order_features(jnp.array(X), 3, pair_idx),
            dtype=np.float64)

        score, coeffs = _project_residual_score(Phi_pair, residual, residual_var)
        assert score > 0.8, f"Score {score} should be near 1 for signal in subspace"
        assert len(coeffs) == Phi_pair.shape[1]

    def test_pure_noise_gives_low_score(self):
        """When residual is pure noise, score should be near 0 after df correction."""
        np.random.seed(42)
        N = 2000
        X = np.random.uniform(0, 1, (N, 3))
        residual = np.random.randn(N)
        residual_var = float(np.var(residual))

        pair_idx = jnp.array([[0, 1]], dtype=jnp.int32)
        Phi_pair = np.asarray(
            build_second_order_features(jnp.array(X), 3, pair_idx),
            dtype=np.float64)

        score, _ = _project_residual_score(Phi_pair, residual, residual_var)
        assert score < 0.05, f"Score {score} should be near 0 for pure noise"

    def test_empty_subspace(self):
        """Zero-width subspace should return 0."""
        residual = np.random.randn(100)
        Phi_empty = np.zeros((100, 0))
        score, coeffs = _project_residual_score(Phi_empty, residual, 1.0)
        assert score == 0.0
        assert len(coeffs) == 0

    def test_score_nonnegative(self):
        """Score should always be non-negative (clamped)."""
        np.random.seed(99)
        N = 200
        X = np.random.uniform(0, 1, (N, 2))
        residual = np.random.randn(N) * 0.1
        residual_var = float(np.var(residual))

        pair_idx = jnp.array([[0, 1]], dtype=jnp.int32)
        Phi_pair = np.asarray(
            build_second_order_features(jnp.array(X), 3, pair_idx),
            dtype=np.float64)

        score, _ = _project_residual_score(Phi_pair, residual, residual_var)
        assert score >= 0.0


# ============================================================================
# Tests for scan_missing_pairs (full sieve)
# ============================================================================

class TestScanMissingPairs:

    def test_identifies_missing_interaction(self, interaction_data):
        """Sieve should identify pair (0,1) as the top missing pair."""
        X, y, D, K1 = interaction_data
        model = _fit_model(X, y, {
            'K1': K1, 'K2': 0, 'stages': ['A'],
            'strategy': 'variance', 'lambda_order1': 0.001,
        })

        result = scan_missing_pairs(
            model, jnp.array(X), jnp.array(y),
            K2=3, significance_threshold=0.01, verbose=False)

        assert result.n_scanned > 0
        assert result.n_significant > 0
        top_pair, top_score = result.ranked_pairs[0]
        assert top_pair == (0, 1), f"Top pair should be (0,1), got {top_pair}"
        assert top_score > 0.05, f"Top score {top_score} too low"

    def test_no_false_positives_on_additive(self, pure_noise_data):
        """Sieve should find no significant pairs on additive function."""
        X, y, D, K1 = pure_noise_data
        model = _fit_model(X, y, {
            'K1': K1, 'K2': 0, 'stages': ['A'],
            'strategy': 'variance', 'lambda_order1': 0.001,
        })

        result = scan_missing_pairs(
            model, jnp.array(X), jnp.array(y),
            K2=3, significance_threshold=0.01, verbose=False)

        for (i, j), score in result.ranked_pairs:
            assert score < 0.05, (
                f"Pair ({i},{j}) score {score:.4f} too high for additive function")

    def test_already_fitted_pairs_excluded(self, interaction_data):
        """Pair (0,1) should be excluded when already fitted."""
        X, y, D, K1 = interaction_data
        model = _fit_model(X, y, {
            'K1': K1, 'K2': 3, 'stages': ['A', 'B'],
            'strategy': 'variance', 'lambda_order1': 0.001,
            'lambda_order2': 0.01,
        })

        result = scan_missing_pairs(
            model, jnp.array(X), jnp.array(y),
            K2=3, significance_threshold=0.01, verbose=False)

        scanned_pairs = set(result.pair_scores.keys())
        assert (0, 1) not in scanned_pairs, "(0,1) should be excluded when already fitted"

    def test_zero_residual_handled(self):
        """Sieve should handle near-zero residual gracefully."""
        np.random.seed(42)
        N = 300
        D = 3
        X = np.random.uniform(0, 1, (N, D))
        y = 2.0 * (X[:, 0] - 0.5)

        model = _fit_model(X, y, {
            'K1': 3, 'K2': 0, 'stages': ['A'],
            'strategy': 'variance', 'lambda_order1': 1e-8,
        })

        result = scan_missing_pairs(
            model, jnp.array(X), jnp.array(y),
            K2=3, verbose=False)
        assert result.n_scanned >= 0

    def test_score_is_fraction(self, interaction_data):
        """Scores should be fractions in [0, 1]."""
        X, y, D, K1 = interaction_data
        model = _fit_model(X, y, {
            'K1': K1, 'K2': 0, 'stages': ['A'],
            'strategy': 'variance', 'lambda_order1': 0.001,
        })

        result = scan_missing_pairs(
            model, jnp.array(X), jnp.array(y), K2=3, verbose=False)

        for (i, j), score in result.pair_scores.items():
            assert 0 <= score <= 1.0 + 0.01, (
                f"Score for ({i},{j}) = {score} should be in [0, 1]")

    def test_interaction_rank_ordering(self):
        """Strong interaction should rank above weak interaction."""
        np.random.seed(42)
        N = 5000
        D = 4
        X = np.random.uniform(0, 1, (N, D))
        f = (5.0 * (X[:, 0] - 0.5) + 3.0 * (X[:, 1] - 0.5)
             + 2.0 * (X[:, 2] - 0.5) + 1.0 * (X[:, 3] - 0.5)
             + 8.0 * (X[:, 0] - 0.5) * (X[:, 1] - 0.5)
             + 1.0 * (X[:, 2] - 0.5) * (X[:, 3] - 0.5))
        y = f + 0.5 * np.random.randn(N)

        model = _fit_model(X, y, {
            'K1': 5, 'K2': 0, 'stages': ['A'],
            'strategy': 'variance', 'lambda_order1': 0.001,
        })

        result = scan_missing_pairs(
            model, jnp.array(X), jnp.array(y), K2=3, verbose=False)

        score_01 = result.pair_scores.get((0, 1), 0.0)
        score_23 = result.pair_scores.get((2, 3), 0.0)
        assert score_01 > score_23, (
            f"Strong (0,1) score {score_01} should exceed weak (2,3) score {score_23}")
