"""Tests for Tier 1 features: component eval, save/load, prediction intervals, one-call API."""

import pytest
import jax
import jax.numpy as jnp
import numpy as np
import os
import tempfile
import shutil

pytestmark = pytest.mark.integration


@pytest.fixture
def fitted_model():
    """Fit a simple model for testing."""
    from hifi_anova.data.test_functions import T2_2_smooth_additive
    from hifi_anova.data.preprocessing import preprocess_data
    from hifi_anova.training.trainer import HiFiANOVATrainer

    X, y, gt = T2_2_smooth_additive(n_samples=3000, noise_std=0.3)
    data = preprocess_data(X, y, seed=42)

    trainer = HiFiANOVATrainer({
        'K1': 3, 'K2': 2, 'stages': ['A', 'B'],
        'lambda_order1': 0.001, 'lambda_order2': 0.01,
        'strategy': 'curvature',
    })
    model, results = trainer.fit(
        data['x_train'], data['y_train'],
        data['x_val'], data['y_val'])

    return model, data, results


# =============================================================================
# Component Evaluation
# =============================================================================

class TestComponentEvaluation:

    def test_first_order_shape(self, fitted_model):
        from hifi_anova.analysis.component_eval import evaluate_first_order
        model, data, _ = fitted_model
        x_vals = jnp.linspace(0, 1, 50)
        f_vals = evaluate_first_order(model, 0, x_vals)
        assert f_vals.shape == (50,)

    def test_first_order_on_grid(self, fitted_model):
        from hifi_anova.analysis.component_eval import first_order_on_grid
        model, _, _ = fitted_model
        x_grid, f_vals = first_order_on_grid(model, 0, n_points=100)
        assert x_grid.shape == (100,)
        assert f_vals.shape == (100,)
        assert x_grid[0] == 0.0 and x_grid[-1] == 1.0

    def test_all_first_order(self, fitted_model):
        from hifi_anova.analysis.component_eval import evaluate_all_first_order
        model, data, _ = fitted_model
        components = evaluate_all_first_order(model, data['x_test'][:20])
        assert len(components) == model.D
        for i, vals in components.items():
            assert vals.shape == (20,)

    def test_second_order_on_grid(self, fitted_model):
        from hifi_anova.analysis.component_eval import second_order_on_grid
        model, _, _ = fitted_model
        if model.pair_indices is not None and model.pair_indices.shape[0] > 0:
            xi, xj, surface = second_order_on_grid(model, 0, n_points=20)
            assert xi.shape == (20,)
            assert xj.shape == (20,)
            assert surface.shape == (20, 20)

    def test_frequency_decomposition(self, fitted_model):
        from hifi_anova.analysis.component_eval import frequency_decomposition
        model, _, _ = fitted_model
        decomp = frequency_decomposition(model, 0)
        assert isinstance(decomp, dict)
        if decomp:  # non-zero variable
            total = sum(decomp.values())
            assert abs(total - 1.0) < 0.15  # approximate due to cross-terms

    def test_interaction_strength_matrix(self, fitted_model):
        from hifi_anova.analysis.component_eval import interaction_strength_matrix
        model, _, _ = fitted_model
        matrix = interaction_strength_matrix(model)
        assert matrix.shape == (model.D, model.D)
        # Symmetric
        assert np.allclose(matrix, matrix.T, atol=1e-10)
        # Diagonal is first-order Sobol
        assert np.all(matrix.diagonal() >= 0)


# =============================================================================
# Save/Load
# =============================================================================

class TestSaveLoad:

    def test_save_and_load(self, fitted_model):
        from hifi_anova.model.io import save_model, load_model
        model, data, results = fitted_model

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'test_model')
            save_model(model, path,
                       config={'K1': 3, 'K2': 2},
                       transformer=data['transformer'],
                       feature_names=[f'x{i}' for i in range(model.D)])

            # Check files exist
            assert os.path.exists(os.path.join(path, 'model.eqx'))
            assert os.path.exists(os.path.join(path, 'meta.json'))
            assert os.path.exists(os.path.join(path, 'transformer.pkl'))

            # Load
            loaded = load_model(path, like_model=model)
            assert loaded['model'] is not None
            assert loaded['transformer'] is not None
            assert loaded['feature_names'] is not None

    def test_roundtrip_predictions(self, fitted_model):
        """Predictions before and after save/load must match."""
        from hifi_anova.model.io import save_model, load_model
        model, data, _ = fitted_model

        pred_before = np.asarray(model.predict_mean_only(data['x_test'][:10]))

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'rt_model')
            save_model(model, path)
            loaded = load_model(path, like_model=model)

            pred_after = np.asarray(loaded['model'].predict_mean_only(data['x_test'][:10]))
            assert np.allclose(pred_before, pred_after, atol=1e-5)


# =============================================================================
# Prediction Intervals
# =============================================================================

class TestPredictionIntervals:

    def test_intervals_shape(self, fitted_model):
        from hifi_anova.model.predict import predict_intervals
        model, data, _ = fitted_model

        result = predict_intervals(model, data['x_test'][:20])
        assert result['mean'].shape == (20,)
        assert result['lower'].shape == (20,)
        assert result['upper'].shape == (20,)
        assert result['var_total'].shape == (20,)

    def test_intervals_ordered(self, fitted_model):
        """Lower < mean < upper."""
        from hifi_anova.model.predict import predict_intervals
        model, data, _ = fitted_model

        result = predict_intervals(model, data['x_test'][:50])
        assert np.all(result['lower'] <= result['mean'] + 1e-10)
        assert np.all(result['mean'] <= result['upper'] + 1e-10)

    def test_intervals_with_epistemic(self, fitted_model):
        from hifi_anova.model.predict import predict_intervals
        model, data, _ = fitted_model

        Phi = np.asarray(model.build_phi_all(data['x_train']), dtype=np.float64)
        from hifi_anova.training.regularization import build_regularization_vector
        reg = np.asarray(build_regularization_vector(
            model.D, model.K1, model.K2,
            model.pair_indices.shape[0] if model.pair_indices is not None else 0,
            'curvature', 0.001, 0.01), dtype=np.float64)

        result = predict_intervals(model, data['x_test'][:10],
                                    Phi_train=Phi, reg_diag=reg, sigma2_hat=0.1)
        # Epistemic should be non-zero
        assert np.any(result['var_epistemic'] > 0)
        # With epistemic, intervals should be wider
        result_no_ep = predict_intervals(model, data['x_test'][:10])
        widths_with = result['upper'] - result['lower']
        widths_without = result_no_ep['upper'] - result_no_ep['lower']
        assert np.mean(widths_with) >= np.mean(widths_without) - 1e-6

    def test_prediction_summary(self, fitted_model):
        from hifi_anova.model.predict import prediction_summary
        model, data, _ = fitted_model

        summary = prediction_summary(model, data['x_test'][0:1])
        assert 'prediction' in summary
        assert 'interval_95' in summary
        assert 'contributions' in summary
        assert len(summary['contributions']) == model.D


# =============================================================================
# One-Call API
# =============================================================================

class TestOneCallAPI:

    def test_basic_call(self):
        """hifi_anova(X, y) should work with minimal arguments."""
        from hifi_anova.api import hifi_anova
        np.random.seed(42)
        N, D = 1000, 5
        X = np.random.uniform(0, 1, (N, D))
        y = 3 * (X[:, 0] - 0.5) + 2 * np.cos(2 * np.pi * X[:, 1]) + 0.3 * np.random.randn(N)

        result = hifi_anova(X, y, K1=3, K2=0, mode='first', verbose=False)
        assert result.r_squared > 0.5
        assert result.sigma_hat > 0
        assert len(result.sobol_ci) == D

    def test_with_names(self):
        from hifi_anova.api import hifi_anova
        np.random.seed(42)
        N = 800
        X = np.random.uniform(0, 1, (N, 3))
        y = 5 * (X[:, 0] - 0.5) + 0.5 * np.random.randn(N)

        result = hifi_anova(X, y,
                             feature_names=['age', 'income', 'score'],
                             K1=3, K2=0, mode='first', verbose=False)
        assert 'age' in result.sobol_ci
        assert 'income' in result.sobol_ci

    def test_predict_on_new_data(self):
        from hifi_anova.api import hifi_anova
        np.random.seed(42)
        N = 800
        X = np.random.uniform(0, 1, (N, 3))
        y = 3 * X[:, 0] + 0.5 * np.random.randn(N)

        result = hifi_anova(X, y, K1=3, K2=0, mode='first', verbose=False)
        X_new = np.random.uniform(0, 1, (10, 3))
        pred = result.predict(X_new)
        assert pred.shape == (10,)

    def test_predict_intervals(self):
        from hifi_anova.api import hifi_anova
        np.random.seed(42)
        N = 800
        X = np.random.uniform(0, 1, (N, 3))
        y = 3 * X[:, 0] + 0.5 * np.random.randn(N)

        result = hifi_anova(X, y, K1=3, K2=0, mode='first', verbose=False)
        X_new = np.random.uniform(0, 1, (10, 3))
        lo, hi = result.predict_intervals(X_new)
        assert lo.shape == (10,)
        assert hi.shape == (10,)
        assert np.all(lo < hi)

    def test_component_curve(self):
        from hifi_anova.api import hifi_anova
        np.random.seed(42)
        N = 800
        X = np.random.uniform(0, 1, (N, 3))
        y = 5 * (X[:, 0] - 0.5) + 0.3 * np.random.randn(N)

        result = hifi_anova(X, y, K1=3, K2=0, mode='first', verbose=False)
        x_grid, f_vals = result.component_curve(0)
        assert x_grid.shape == (200,)
        assert f_vals.shape == (200,)

    def test_save_load_roundtrip(self):
        from hifi_anova.api import hifi_anova
        np.random.seed(42)
        N = 500
        X = np.random.uniform(0, 1, (N, 3))
        y = 3 * X[:, 0] + 0.3 * np.random.randn(N)

        result = hifi_anova(X, y, K1=3, K2=0, mode='first', verbose=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'api_model')
            result.save(path)
            assert os.path.exists(os.path.join(path, 'model.eqx'))
