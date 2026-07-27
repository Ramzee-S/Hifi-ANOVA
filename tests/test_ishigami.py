"""Tests for the Ishigami benchmark: analytic indices, generator, recovery, plot."""

import jax
jax.config.update("jax_enable_x64", True)

import numpy as np
import pytest
import matplotlib
matplotlib.use("Agg")

from hifi_anova.data.synthetic import generate_ishigami, ishigami_sobol_indices


# ---------------------------------------------------------------------------
# Analytic ground truth (pure numpy — fast, runs in the quick tier)
# ---------------------------------------------------------------------------

def test_ishigami_analytic_indices_default():
    """Classic Ishigami Sobol values for a=7, b=0.1 (Marrel et al.)."""
    gt = ishigami_sobol_indices(a=7.0, b=0.1)
    # First-order
    assert gt['first_order'][0] == pytest.approx(0.3139, abs=1e-3)
    assert gt['first_order'][1] == pytest.approx(0.4424, abs=1e-3)
    assert gt['first_order'][2] == 0.0
    # Total-order
    assert gt['total_order'][0] == pytest.approx(0.5576, abs=1e-3)
    assert gt['total_order'][1] == pytest.approx(0.4424, abs=1e-3)
    assert gt['total_order'][2] == pytest.approx(0.2437, abs=1e-3)


def test_ishigami_x3_is_pure_interaction():
    """x3: zero first-order, positive total-order (acts only via x1-x3)."""
    gt = ishigami_sobol_indices()
    assert gt['first_order'][2] == 0.0
    assert gt['total_order'][2] > 0.2
    # x3's total-order equals its interaction share.
    D13, D = gt['partial_variances']['D13'], gt['total_variance']
    assert gt['total_order'][2] == pytest.approx(D13 / D, rel=1e-9)


def test_ishigami_first_order_sums_leq_total_variance():
    gt = ishigami_sobol_indices()
    first_sum = sum(gt['first_order'].values())
    # First-order indices leave the interaction share unexplained (< 1).
    assert 0.75 < first_sum < 0.80
    total_sum = sum(gt['total_order'].values())
    assert total_sum > 1.0  # total-order over-counts shared interaction


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

def test_generate_ishigami_shapes_and_range():
    X, y, sigma = generate_ishigami(n_samples=500, noise_std=0.0, seed=0)
    assert X.shape == (500, 3)
    assert y.shape == (500,)
    assert sigma.shape == (500,)
    assert X.min() >= -np.pi - 1e-6 and X.max() <= np.pi + 1e-6
    # Noiseless: y matches the analytic form exactly.
    x1, x2, x3 = X[:, 0], X[:, 1], X[:, 2]
    f = np.sin(x1) + 7.0 * np.sin(x2) ** 2 + 0.1 * x3 ** 4 * np.sin(x1)
    assert np.allclose(y, f)
    assert np.allclose(sigma, 0.0)


def test_generate_ishigami_heteroscedastic_ramp():
    """Heteroscedastic noise std ramps monotonically with the driving variable."""
    X, y, sigma = generate_ishigami(
        n_samples=2000, heteroscedastic=True, variance_variable=2,
        sigma_min=0.3, sigma_max=3.0, seed=0)
    order = np.argsort(X[:, 2])
    s_sorted = sigma[order]
    # sigma is an affine, increasing function of x3.
    assert np.all(np.diff(s_sorted) >= -1e-9)
    assert sigma.min() == pytest.approx(0.3, abs=0.05)
    assert sigma.max() == pytest.approx(3.0, abs=0.05)


# ---------------------------------------------------------------------------
# Plot smoke test (no fitting)
# ---------------------------------------------------------------------------

def test_plot_sensitivity_ellipses_both_modes():
    from hifi_anova.analysis.visualization import plot_sensitivity_ellipses
    sobol = {
        'mean_sobol': {
            'first_order': {0: 0.31, 1: 0.44, 2: 0.0},
            'total_order': {0: 0.56, 1: 0.44, 2: 0.24},
            'second_order': {(0, 2): 0.24},
        },
        'variance_sobol': {'first_order': {0: 0.0, 1: 0.0, 2: 1.0}},
    }
    fig_g = plot_sensitivity_ellipses(sobol, mode='glyph')
    assert fig_g is not None
    ci = {0: (0.29, 0.33), 1: (0.42, 0.46), 2: (0.0, 0.01)}
    fig_p = plot_sensitivity_ellipses(sobol, mode='plane', mean_ci=ci,
                                      var_ci={0: (0.0, 0.0), 1: (0.0, 0.0),
                                              2: (0.98, 1.0)}, ci_scale=8.0)
    assert fig_p is not None


# ---------------------------------------------------------------------------
# Recovery (requires fitting)
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_ishigami_first_order_recovery():
    """A first+second-order fit recovers the analytic mean Sobol indices."""
    from hifi_anova.data.preprocessing import preprocess_data
    from hifi_anova.training.trainer import HiFiANOVATrainer
    from hifi_anova.analysis.sobol import compute_sobol_indices

    X, y, _ = generate_ishigami(n_samples=6000, noise_std=0.1, seed=0)
    data = preprocess_data(X, y, seed=0)
    cfg = {'K1': 12, 'K2': 6, 'strategy': 'curvature',
           'lambda_order1': 0.001, 'lambda_order2': 0.01,
           'stages': ['A', 'B'], 'residual_nn': {'enabled': False}}
    model, _ = HiFiANOVATrainer(cfg).fit(
        data['x_train'], data['y_train'], data['x_val'], data['y_val'])
    s = compute_sobol_indices(model, data['x_test'])
    gt = ishigami_sobol_indices()

    mf = s['mean_sobol']['first_order']
    assert mf[0] == pytest.approx(gt['first_order'][0], abs=0.05)
    assert mf[1] == pytest.approx(gt['first_order'][1], abs=0.05)
    assert mf[2] == pytest.approx(0.0, abs=0.03)
    # x3 first-order ~ 0 but total-order clearly positive (the interaction).
    assert s['mean_sobol']['total_order'][2] > 0.15


@pytest.mark.slow
def test_heteroscedastic_ishigami_variance_driver():
    """Driving the noise with x3 makes x3 dominate the variance spectrum."""
    from hifi_anova.data.preprocessing import preprocess_data
    from hifi_anova.training.trainer import HiFiANOVATrainer
    from hifi_anova.analysis.sobol import compute_sobol_indices

    X, y, _ = generate_ishigami(
        n_samples=8000, heteroscedastic=True, variance_variable=2, seed=1)
    data = preprocess_data(X, y, seed=1)
    cfg = {'K1': 12, 'K2': 6, 'Kh': 3, 'strategy': 'curvature',
           'lambda_order1': 0.001, 'lambda_order2': 0.01, 'lambda_h': 0.1,
           'stages': ['A', 'B', 'D'], 'residual_nn': {'enabled': False},
           'max_outer_iter': 8, 'newton_max_iter': 10}
    model, _ = HiFiANOVATrainer(cfg).fit(
        data['x_train'], data['y_train'], data['x_val'], data['y_val'])
    s = compute_sobol_indices(model, data['x_test'])

    assert 'variance_sobol' in s
    vf = s['variance_sobol']['first_order']
    # x3 (the hidden driver) dominates the variance; x1/x2 near zero.
    assert vf[2] > 0.8
    assert vf[0] < 0.1 and vf[1] < 0.1
    # ... while x3 stays first-order-silent in the mean.
    assert s['mean_sobol']['first_order'][2] < 0.05
