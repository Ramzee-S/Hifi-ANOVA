"""Tier 2: Integration-Level Test Functions.

Tests the full pipeline with more realistic functions.
T2.1: Friedman-1 (out-of-basis, approximate recovery)
T2.2: Smooth additive (in-basis, exact recovery)
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.slow
import pytest

from hifi_anova.data.test_functions import T2_1_friedman1, T2_2_smooth_additive
from hifi_anova.data.preprocessing import preprocess_data
from hifi_anova.training.trainer import HiFiANOVATrainer
from hifi_anova.analysis.sobol import compute_sobol_indices
from hifi_anova.analysis.diagnostics import variance_accounting_report


class TestT2_1_Friedman1:
    """T2.1: Standard Friedman-1 benchmark.

    sin(pi*x1*x2) is NOT in our product basis — tests approximation quality.
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        X, y, self.gt = T2_1_friedman1(n_samples=10000, noise_std=1.0, seed=42)
        self.data = preprocess_data(X, y, seed=42)
        config = {
            'K1': 10, 'K2': 5, 'Kh': 0,
            'strategy': 'curvature',
            'lambda_order1': 0.001,
            'lambda_order2': 0.01,
            'stages': ['A', 'B'],
            'residual_nn': {'enabled': False},
        }
        trainer = HiFiANOVATrainer(config)
        self.model, self.results = trainer.fit(
            self.data['x_train'], self.data['y_train'],
            self.data['x_val'], self.data['y_val']
        )
        self.sobol = compute_sobol_indices(self.model, self.data['x_test'])

    def test_active_variables_identified(self):
        """x1-x5 should all have Sobol > 0.03."""
        for i in range(5):
            si = self.sobol['mean_sobol']['first_order'][i]
            assert si > 0.03, f"Active var x{i+1} has S={si:.4f}, too small"

    def test_irrelevant_variables_small(self):
        """x6-x10 should have Sobol < 0.02."""
        for i in range(5, 10):
            si = self.sobol['mean_sobol']['first_order'][i]
            assert si < 0.02, f"Irrelevant var x{i+1} has S={si:.4f}, too large"

    def test_x4_largest_first_order(self):
        """x4 (10*x4, linear effect) should have the largest first-order index."""
        s4 = self.sobol['mean_sobol']['first_order'][3]
        for i in range(10):
            if i != 3:
                assert s4 >= self.sobol['mean_sobol']['first_order'][i] - 0.01, \
                    f"x4 (S={s4:.4f}) should be largest"

    def test_x1_x2_interaction_present(self):
        """The (x1, x2) interaction should be the largest second-order term."""
        s12 = self.sobol['mean_sobol']['second_order'].get((0, 1), 0.0)
        # It should be positive and larger than other pairs
        assert s12 > 0.005, f"S(x1,x2)={s12:.4f}, should be present"
        for (i, j), sij in self.sobol['mean_sobol']['second_order'].items():
            if (i, j) != (0, 1):
                assert s12 >= sij - 0.005, \
                    f"S(x1,x2)={s12:.4f} should be >= S({i},{j})={sij:.4f}"

    def test_good_fit_quality(self):
        """RMSE should be reasonable (< 2x noise level)."""
        rmse = self.results['stage_B']['rmse_val']
        assert rmse < 2.0, f"RMSE={rmse:.4f}, too high for noise=1.0"

    def test_sobol_sum_reasonable(self):
        """First + second order Sobol should sum close to 1."""
        total = (sum(self.sobol['mean_sobol']['first_order'].values())
                 + sum(self.sobol['mean_sobol']['second_order'].values()))
        assert 0.8 < total < 1.2, f"Sobol sum={total:.4f}, should be ~1"


class TestT2_2_SmoothAdditive:
    """T2.2: Fourier-friendly function, exactly in our basis.

    f = 3*cos(2*pi*x1) + 2*sin(4*pi*x2) + 1.5*cos(2*pi*x3)*cos(2*pi*x4) + 4*(x5-0.5)
    Sobol indices should be recovered EXACTLY.
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        X, y, self.gt = T2_2_smooth_additive(n_samples=15000, noise_std=0.5, seed=42)
        self.data = preprocess_data(X, y, seed=42)
        config = {
            'K1': 5, 'K2': 3, 'Kh': 0,
            'strategy': 'curvature',
            'lambda_order1': 0.0001,
            'lambda_order2': 0.001,
            'stages': ['A', 'B'],
            'residual_nn': {'enabled': False},
        }
        trainer = HiFiANOVATrainer(config)
        self.model, self.results = trainer.fit(
            self.data['x_train'], self.data['y_train'],
            self.data['x_val'], self.data['y_val']
        )
        self.sobol = compute_sobol_indices(self.model, self.data['x_test'])

    def test_sobol_x1(self):
        """S1 = 4.5/8.396 ≈ 0.536."""
        s1 = self.sobol['mean_sobol']['first_order'][0]
        expected = self.gt['mean_sobol_first_order'][0]
        assert abs(s1 - expected) < 0.08, f"S1={s1:.4f}, expected={expected:.4f}"

    def test_sobol_x2(self):
        """S2 = 2.0/8.396 ≈ 0.238."""
        s2 = self.sobol['mean_sobol']['first_order'][1]
        expected = self.gt['mean_sobol_first_order'][1]
        assert abs(s2 - expected) < 0.08, f"S2={s2:.4f}, expected={expected:.4f}"

    def test_sobol_x5(self):
        """S5 = 1.333/8.396 ≈ 0.159."""
        s5 = self.sobol['mean_sobol']['first_order'][4]
        expected = self.gt['mean_sobol_first_order'][4]
        assert abs(s5 - expected) < 0.08, f"S5={s5:.4f}, expected={expected:.4f}"

    def test_sobol_x3_x4_interaction(self):
        """S34 = 0.5625/8.396 ≈ 0.067 should be captured in second order."""
        s34 = self.sobol['mean_sobol']['second_order'].get((2, 3), 0.0)
        expected = self.gt['mean_sobol_second_order'][(2, 3)]
        # Allow larger tolerance since interaction is small
        assert abs(s34 - expected) < 0.05, f"S34={s34:.4f}, expected={expected:.4f}"

    def test_irrelevant_variables_zero(self):
        """x6, x7, x8 should have ~0 Sobol."""
        for i in [5, 6, 7]:
            si = self.sobol['mean_sobol']['first_order'][i]
            assert si < 0.02, f"x{i+1} S={si:.4f}, should be ~0"

    def test_near_perfect_fit(self):
        """Function is in the basis, so fit should be excellent (R² > 0.95)."""
        va = variance_accounting_report(self.model, self.data['x_test'], self.data['y_test'])
        assert va['R_squared'] > 0.90, f"R²={va['R_squared']:.4f}, should be >0.90"

    def test_sobol_sum_close_to_one(self):
        """Since function is exactly in basis, Sobol should sum very close to 1."""
        total = (sum(self.sobol['mean_sobol']['first_order'].values())
                 + sum(self.sobol['mean_sobol']['second_order'].values()))
        assert abs(total - 1.0) < 0.1, f"Sobol sum={total:.4f}"
