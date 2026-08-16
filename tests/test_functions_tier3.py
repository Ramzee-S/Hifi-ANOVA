"""Tier 3: Heteroscedastic Test Functions.

Tests the variance decomposition: dual Sobol spectrum, calibration,
mean-variance separation.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.slow

from hifi_anova.data.test_functions import (
    T3_1_orthogonal_mean_variance, T3_2_shared_variable,
    T3_3_hidden_variable, T3_4_interaction_noise,
    T3_5_signal_noise_confusion,
)
from hifi_anova.data.preprocessing import preprocess_data
from hifi_anova.training.trainer import HiFiANOVATrainer
from hifi_anova.analysis.sobol import compute_sobol_indices
from hifi_anova.analysis.diagnostics import calibration_report


def _fit_heteroscedastic(X, y, D, K1=5, K2=0, Kh=3, lambda1=0.001,
                         lambda2=0.01, lambda_h=0.05, n_samples=None,
                         heteroscedastic_guard=True):
    """Helper to fit a heteroscedastic model.

    ``heteroscedastic_guard=False`` keeps the raw variance fit even when the
    DEC-028 safety guard would revert to a constant variance on held-out NLL.
    Tests that verify the variance *model's* Sobol recovery pass this so they
    exercise the variance fit itself rather than the guard's (deliberately
    conservative) model-selection — which, on moderate heteroscedasticity, can
    revert despite correct structure recovery (a known, advisor-gated Stage-D
    calibration matter; see StageD_calibration_brief.md).
    """
    data = preprocess_data(X, y, seed=42)
    stages = ['A', 'B', 'D'] if K2 > 0 else ['A', 'D']
    config = {
        'K1': K1, 'K2': K2, 'Kh': Kh,
        'strategy': 'curvature',
        'lambda_order1': lambda1,
        'lambda_order2': lambda2,
        'lambda_h': lambda_h,
        'stages': stages,
        'residual_nn': {'enabled': False},
        'heteroscedastic_guard': heteroscedastic_guard,
        'max_outer_iter': 10,
        'alternating_tol': 1e-4,
        'newton_max_iter': 15,
    }
    trainer = HiFiANOVATrainer(config)
    model, results = trainer.fit(
        data['x_train'], data['y_train'],
        data['x_val'], data['y_val']
    )
    sobol = compute_sobol_indices(model, data['x_test'])
    return model, results, sobol, data


class TestT3_1_OrthogonalMeanVariance:
    """T3.1: f = 5*(x1-0.5) + 3*cos(2*pi*x2), sigma^2 = exp(2*(x3-0.5)), D=6.

    Mean and variance have completely separate drivers (the easy case).
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        X, y, self.gt = T3_1_orthogonal_mean_variance(n_samples=15000, seed=42)
        self.model, self.results, self.sobol, self.data = _fit_heteroscedastic(
            X, y, D=6, K1=5, K2=0, Kh=3, lambda_h=0.01
        )

    def test_mean_sobol_x1_large(self):
        """x1 should have significant mean Sobol."""
        s1 = self.sobol['mean_sobol']['first_order'][0]
        assert s1 > 0.15, f"Mean S1={s1:.4f}, should be large"

    def test_mean_sobol_x2_large(self):
        """x2 should have significant mean Sobol."""
        s2 = self.sobol['mean_sobol']['first_order'][1]
        assert s2 > 0.15, f"Mean S2={s2:.4f}, should be large"

    def test_mean_sobol_x3_small(self):
        """x3 drives variance only, mean Sobol should be small."""
        s3 = self.sobol['mean_sobol']['first_order'][2]
        assert s3 < 0.1, f"Mean S3={s3:.4f}, should be small (variance driver)"

    def test_variance_sobol_x3_dominates(self):
        """x3 should dominate the variance Sobol spectrum."""
        assert 'log_variance_sobol' in self.sobol
        vs = self.sobol['log_variance_sobol']['first_order']
        s3_h = vs[2]
        # x3 should be the largest
        assert s3_h > 0.3, f"Variance S3={s3_h:.4f}, should dominate"
        for i in [0, 1, 3, 4, 5]:
            assert s3_h > vs[i], \
                f"x3 ({s3_h:.3f}) should dominate x{i+1} ({vs[i]:.3f})"

    def test_variance_sobol_mean_vars_small(self):
        """x1, x2 (mean drivers) should have small variance Sobol."""
        assert 'log_variance_sobol' in self.sobol
        vs = self.sobol['log_variance_sobol']['first_order']
        assert vs[0] < 0.3, f"Variance S1={vs[0]:.4f}, should be small"
        assert vs[1] < 0.3, f"Variance S2={vs[1]:.4f}, should be small"


class TestT3_2_SharedVariable:
    """T3.2: f = 5*(x1-0.5) + 3*(x1-0.5)*(x2-0.5), sigma^2 = exp(1.5*(x1-0.5)), D=5.

    x1 affects BOTH mean and variance. Tests mean-variance separation.
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        X, y, self.gt = T3_2_shared_variable(n_samples=15000, seed=42)
        # This dataset is genuinely heteroscedastic (sigma^2 = exp(1.5*(x1-0.5)))
        # and the variance model recovers x1 (variance_sobol_x1 ~ 0.98). With the
        # joint-GLS Stage-D mean now the default, the guard KEEPS the variance
        # model on its own (no heteroscedastic_guard=False workaround needed) —
        # this exercises the real default path, and asserting the guard keeps is
        # the stronger test (was the false-revert the joint-GLS flip fixed).
        self.model, self.results, self.sobol, self.data = _fit_heteroscedastic(
            X, y, D=5, K1=5, K2=3, Kh=3, lambda1=0.001, lambda2=0.005,
            lambda_h=0.01
        )
        assert self.model.variance_model is not None, \
            "default guard should keep the variance model on genuine heteroscedastic data"

    def test_mean_sobol_x1_large(self):
        """x1 should have large mean Sobol (main effect + interaction)."""
        s1 = self.sobol['mean_sobol']['first_order'][0]
        st1 = self.sobol['mean_sobol']['total_order'][0]
        assert st1 > 0.5, f"Total order x1={st1:.4f}, should be dominant"

    def test_variance_sobol_x1_large(self):
        """x1 should also dominate the variance Sobol."""
        assert 'log_variance_sobol' in self.sobol
        vs = self.sobol['log_variance_sobol']['first_order']
        s1_h = vs[0]
        assert s1_h > 0.3, f"Variance S1={s1_h:.4f}, x1 drives variance too"

    def test_mean_sobol_x1_and_variance_sobol_x1_both_large(self):
        """x1 should appear in BOTH mean and variance spectra (dual role)."""
        assert 'log_variance_sobol' in self.sobol
        st1_mean = self.sobol['mean_sobol']['total_order'][0]
        s1_var = self.sobol['log_variance_sobol']['first_order'][0]
        assert st1_mean > 0.3 and s1_var > 0.2, \
            f"x1 dual role: mean={st1_mean:.3f}, var={s1_var:.3f}"

    def test_irrelevant_variables_small(self):
        """x3, x4, x5 should be small in both spectra."""
        for i in [2, 3, 4]:
            si = self.sobol['mean_sobol']['first_order'][i]
            assert si < 0.1, f"Mean S{i+1}={si:.4f}"


class TestT3_3_HiddenVariable:
    """T3.3: The Showcase — f = 3*cos(2*pi*x1) + 2*(x2-0.5),
    sigma^2 = exp(x3 + 0.5*sin(2*pi*x4) - 0.75), D=8.

    Standard analysis would say x3, x4 are irrelevant.
    HiFiANOVA reveals they drive 100% of the uncertainty.
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        X, y, self.gt = T3_3_hidden_variable(n_samples=15000, seed=42)
        self.model, self.results, self.sobol, self.data = _fit_heteroscedastic(
            X, y, D=8, K1=5, K2=0, Kh=3, lambda_h=0.01
        )

    def test_mean_sobol_x1_x2_active(self):
        """x1, x2 drive the mean function."""
        s1 = self.sobol['mean_sobol']['first_order'][0]
        s2 = self.sobol['mean_sobol']['first_order'][1]
        assert s1 > 0.2, f"Mean S1={s1:.4f}"
        assert s2 > 0.02, f"Mean S2={s2:.4f}"

    def test_mean_sobol_x3_x4_small(self):
        """x3, x4 should NOT appear in mean Sobol (they drive noise only)."""
        s3 = self.sobol['mean_sobol']['first_order'][2]
        s4 = self.sobol['mean_sobol']['first_order'][3]
        assert s3 < 0.1, f"Mean S3={s3:.4f}, should be ~0"
        assert s4 < 0.1, f"Mean S4={s4:.4f}, should be ~0"

    def test_variance_sobol_x3_large(self):
        """x3 should dominate variance Sobol."""
        assert 'log_variance_sobol' in self.sobol
        vs = self.sobol['log_variance_sobol']['first_order']
        s3_h = vs[2]
        assert s3_h > 0.2, \
            f"Variance S3={s3_h:.4f}, x3 drives uncertainty"

    def test_variance_sobol_x4_present(self):
        """x4 should also appear in variance Sobol (secondary driver)."""
        assert 'log_variance_sobol' in self.sobol
        vs = self.sobol['log_variance_sobol']['first_order']
        s4_h = vs[3]
        # x4 contributes via sin(2*pi*x4) with coeff 0.5
        assert s4_h > 0.05, \
            f"Variance S4={s4_h:.4f}, x4 is secondary variance driver"

    def test_hidden_variables_revealed(self):
        """The 'showcase test': variance analysis reveals hidden importance.

        Standard analysis (mean Sobol) would rank x3 as irrelevant.
        Variance Sobol should rank x3 as the most important uncertainty driver.
        """
        assert 'log_variance_sobol' in self.sobol
        mean_s3 = self.sobol['mean_sobol']['first_order'][2]
        var_s3 = self.sobol['log_variance_sobol']['first_order'][2]
        # x3 is "hidden" in mean but "revealed" in variance
        assert mean_s3 < 0.1, f"Mean says x3 unimportant: S3={mean_s3:.4f}"
        assert var_s3 > mean_s3, \
            f"Variance reveals x3: var_S3={var_s3:.4f} > mean_S3={mean_s3:.4f}"

    def test_irrelevant_x5_x8_small_everywhere(self):
        """x5-x8 should be small in both spectra."""
        assert 'log_variance_sobol' in self.sobol
        for i in range(4, 8):
            si = self.sobol['mean_sobol']['first_order'][i]
            assert si < 0.05, f"Mean S{i+1}={si:.4f}"
            vi = self.sobol['log_variance_sobol']['first_order'][i]
            assert vi < 0.15, f"Variance S{i+1}={vi:.4f}"


class TestT3_4_InteractionNoise:
    """T3.4: f = 4*(x1-0.5) + 2*(x2-0.5),
    sigma^2 = exp(8*(x3-0.5)*(x4-0.5)), D=6.

    Variance has a genuine INTERACTION between x3 and x4.
    This tests whether the variance model can capture second-order
    variance effects and whether variance Sobol correctly identifies
    the interacting variables.
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        X, y, self.gt = T3_4_interaction_noise(n_samples=20000, seed=42)
        data = preprocess_data(X, y, seed=42)
        # Need K2h > 0 to capture variance interaction
        config = {
            'K1': 5, 'K2': 0, 'Kh': 3,
            'K2h': 3,  # second-order variance features
            'strategy': 'curvature',
            'lambda_order1': 0.001,
            'lambda_h': 0.01,
            'stages': ['A', 'D'],
            'residual_nn': {'enabled': False},
            'max_outer_iter': 10,
            'alternating_tol': 1e-4,
            'newton_max_iter': 15,
            'var_pair_selection': 'all',
        }
        trainer = HiFiANOVATrainer(config)
        self.model, self.results = trainer.fit(
            data['x_train'], data['y_train'],
            data['x_val'], data['y_val']
        )
        self.sobol = compute_sobol_indices(self.model, data['x_test'])
        self.data = data

    def test_mean_sobol_x1_dominates(self):
        """x1 should dominate mean Sobol (coefficient 4 vs 2)."""
        s1 = self.sobol['mean_sobol']['first_order'][0]
        assert s1 > 0.5, f"Mean S1={s1:.4f}, should be dominant"

    def test_mean_sobol_x2_present(self):
        """x2 should be present in mean Sobol."""
        s2 = self.sobol['mean_sobol']['first_order'][1]
        assert s2 > 0.05, f"Mean S2={s2:.4f}, should be present"

    def test_mean_sobol_x34_negligible(self):
        """x3, x4 should have negligible mean Sobol."""
        for i in [2, 3]:
            si = self.sobol['mean_sobol']['first_order'][i]
            assert si < 0.1, f"Mean S{i+1}={si:.4f}, should be small"

    def test_variance_sobol_first_order_small(self):
        """All first-order variance Sobol should be small.

        h(x) = 8*(x3-0.5)*(x4-0.5) is a PURE interaction with no
        first-order component. All variance signal is in second order.
        """
        assert 'log_variance_sobol' in self.sobol
        vs = self.sobol['log_variance_sobol']['first_order']
        for i in range(6):
            si = vs[i]
            assert si < 0.05, \
                f"Variance S{i+1}={si:.4f}, should be small (pure interaction)"

    def test_variance_sobol_interaction_x3x4(self):
        """The (x3,x4) interaction should dominate variance Sobol.

        h(x) = 8*(x3-0.5)*(x4-0.5) => variance is entirely in the
        second-order interaction between x3 and x4.
        """
        assert 'log_variance_sobol' in self.sobol
        vs2 = self.sobol['log_variance_sobol'].get('second_order', {})
        s34 = vs2.get((2, 3), 0.0)
        assert s34 > 0.5, \
            f"Variance S(3,4)={s34:.4f}, should dominate (pure interaction)"

    def test_variance_sobol_other_pairs_small(self):
        """Non-(x3,x4) pairs should have negligible variance Sobol."""
        assert 'log_variance_sobol' in self.sobol
        vs2 = self.sobol['log_variance_sobol'].get('second_order', {})
        for pair, val in vs2.items():
            if pair != (2, 3):
                assert val < 0.05, \
                    f"Variance S{pair}={val:.4f}, should be small"


class TestT3_5_SignalNoiseConfusion:
    """T3.5: f = 5*sin(2*pi*x1) + 3*cos(4*pi*x2),
    sigma^2 = exp(2*cos(2*pi*x1)), D=5.

    Mean and variance share x1 at the SAME frequency but different phase.
    Tests whether the model can separate sin vs cos at the same harmonic.
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        X, y, self.gt = T3_5_signal_noise_confusion(n_samples=15000, seed=42)
        self.model, self.results, self.sobol, self.data = _fit_heteroscedastic(
            X, y, D=5, K1=5, K2=0, Kh=3, lambda_h=0.01
        )

    def test_mean_sobol_x1_large(self):
        """x1 should dominate mean Sobol (5*sin is strong)."""
        s1 = self.sobol['mean_sobol']['first_order'][0]
        assert s1 > 0.3, f"Mean S1={s1:.4f}, 5*sin should be strong"

    def test_mean_sobol_x2_present(self):
        """x2 should be present in mean Sobol (3*cos at freq 2)."""
        s2 = self.sobol['mean_sobol']['first_order'][1]
        assert s2 > 0.1, f"Mean S2={s2:.4f}"

    def test_variance_sobol_x1_large(self):
        """x1 should also dominate variance Sobol (cos at same frequency)."""
        assert 'log_variance_sobol' in self.sobol
        vs = self.sobol['log_variance_sobol']['first_order']
        s1_h = vs[0]
        assert s1_h > 0.3, \
            f"Variance S1={s1_h:.4f}, x1 drives both mean and variance"

    def test_variance_sobol_x2_small(self):
        """x2 should NOT appear in variance Sobol."""
        assert 'log_variance_sobol' in self.sobol
        vs = self.sobol['log_variance_sobol']['first_order']
        s2_h = vs[1]
        assert s2_h < 0.3, \
            f"Variance S2={s2_h:.4f}, x2 shouldn't drive variance"

    def test_calibration_reasonable(self):
        """Standardized residuals should be reasonably calibrated."""
        cal = calibration_report(self.model, self.data['x_test'], self.data['y_test'])
        # Mean should be near 0
        assert abs(cal['mean_standardized_residual']) < 0.3, \
            f"Mean(z)={cal['mean_standardized_residual']:.3f}"
        # Variance should be near 1 (allow slack for this stress test
        # since mean and variance share frequency content in x1)
        assert 0.5 < cal['var_standardized_residual'] < 2.0, \
            f"Var(z)={cal['var_standardized_residual']:.3f}"
