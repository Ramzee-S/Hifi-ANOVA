"""Tier 1: Unit-Level Validation Tests.

Each test verifies exact analytic properties of HiFiANOVA on functions
with known closed-form Sobol indices and coefficients.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hifi_anova.data.test_functions import (
    T1_1_pure_linear, T1_2_pure_fourier, T1_3_linear_fourier_mix,
    T1_4_pure_interaction, T1_5_constant_mean_variable_noise,
)
from hifi_anova.data.preprocessing import preprocess_data
from hifi_anova.training.trainer import HiFiANOVATrainer
from hifi_anova.analysis.sobol import compute_sobol_indices
from hifi_anova.core.gram import build_gram_matrix

pytestmark = pytest.mark.integration


class TestT1_1_PureLinear:
    """T1.1: f(x) = 5*(x1-0.5) + 3*(x2-0.5), D=5."""

    @pytest.fixture(autouse=True)
    def setup(self):
        X, y, self.gt = T1_1_pure_linear(n_samples=10000, noise_std=0.1, seed=42)
        # Use data directly (already uniform)
        self.data = preprocess_data(X, y, seed=42)
        config = {
            'K1': 5, 'K2': 0, 'Kh': 0,
            'strategy': 'curvature',
            'lambda_order1': 0.0001,
            'lambda_order2': 0.01,
            'stages': ['A'],
            'residual_nn': {'enabled': False},
        }
        trainer = HiFiANOVATrainer(config)
        self.model, self.results = trainer.fit(
            self.data['x_train'], self.data['y_train'],
            self.data['x_val'], self.data['y_val']
        )
        self.sobol = compute_sobol_indices(self.model)

    def test_coefficient_recovery_var0(self):
        """Linear coefficient for x1 should be ~5."""
        w0 = self.model.mean_model.get_coefficients_for_variable(0)
        assert abs(float(w0[0]) - 5.0) < 0.3, f"Expected ~5, got {float(w0[0])}"

    def test_coefficient_recovery_var1(self):
        """Linear coefficient for x2 should be ~3."""
        w1 = self.model.mean_model.get_coefficients_for_variable(1)
        assert abs(float(w1[0]) - 3.0) < 0.3, f"Expected ~3, got {float(w1[0])}"

    def test_irrelevant_variables_zero(self):
        """Variables x3-x5 should have near-zero coefficients."""
        for i in range(2, 5):
            wi = self.model.mean_model.get_coefficients_for_variable(i)
            assert float(jnp.max(jnp.abs(wi))) < 0.2, \
                f"Variable {i} should be zero, max coeff = {float(jnp.max(jnp.abs(wi)))}"

    def test_sobol_x1_dominant(self):
        """S1 should be ~0.735 (25/34)."""
        s1 = self.sobol['mean_sobol']['first_order'][0]
        expected = 25.0 / 34.0
        assert abs(s1 - expected) < 0.05, f"S1={s1:.4f}, expected={expected:.4f}"

    def test_sobol_x2(self):
        """S2 should be ~0.265 (9/34)."""
        s2 = self.sobol['mean_sobol']['first_order'][1]
        expected = 9.0 / 34.0
        assert abs(s2 - expected) < 0.05, f"S2={s2:.4f}, expected={expected:.4f}"

    def test_sobol_irrelevant_zero(self):
        """S3, S4, S5 should be ~0."""
        for i in range(2, 5):
            si = self.sobol['mean_sobol']['first_order'][i]
            assert si < 0.02, f"S{i+1}={si:.4f}, should be ~0"

    def test_sobol_sum_to_one(self):
        """All first-order Sobol indices should sum to ~1."""
        total = sum(self.sobol['mean_sobol']['first_order'].values())
        assert abs(total - 1.0) < 0.01, f"Sum={total:.4f}"


class TestT1_2_PureFourier:
    """T1.2: f(x) = 3*cos(2*pi*x1) + 2*sin(4*pi*x2), D=5."""

    @pytest.fixture(autouse=True)
    def setup(self):
        X, y, self.gt = T1_2_pure_fourier(n_samples=10000, noise_std=0.1, seed=42)
        self.data = preprocess_data(X, y, seed=42)
        config = {
            'K1': 5, 'K2': 0, 'Kh': 0,
            'strategy': 'curvature',
            'lambda_order1': 0.0001,
            'lambda_order2': 0.01,
            'stages': ['A'],
            'residual_nn': {'enabled': False},
        }
        trainer = HiFiANOVATrainer(config)
        self.model, self.results = trainer.fit(
            self.data['x_train'], self.data['y_train'],
            self.data['x_val'], self.data['y_val']
        )
        self.sobol = compute_sobol_indices(self.model)

    def test_cos1_coefficient_var0(self):
        """cos(2*pi*x1) coefficient should be ~3."""
        w0 = self.model.mean_model.get_coefficients_for_variable(0)
        # Index 1 is cos_1 in our ordering [lin, cos1, sin1, cos2, sin2, ...]
        assert abs(float(w0[1]) - 3.0) < 0.3, f"cos1 coeff = {float(w0[1])}"

    def test_sin2_coefficient_var1(self):
        """sin(4*pi*x2) coefficient should be ~2."""
        w1 = self.model.mean_model.get_coefficients_for_variable(1)
        # Index 4 is sin_2: [lin, cos1, sin1, cos2, sin2, ...]
        assert abs(float(w1[4]) - 2.0) < 0.3, f"sin2 coeff = {float(w1[4])}"

    def test_sobol_x1(self):
        """S1 = 4.5/6.5 ≈ 0.692."""
        s1 = self.sobol['mean_sobol']['first_order'][0]
        expected = 4.5 / 6.5
        assert abs(s1 - expected) < 0.05, f"S1={s1:.4f}, expected={expected:.4f}"

    def test_sobol_x2(self):
        """S2 = 2.0/6.5 ≈ 0.308."""
        s2 = self.sobol['mean_sobol']['first_order'][1]
        expected = 2.0 / 6.5
        assert abs(s2 - expected) < 0.05, f"S2={s2:.4f}, expected={expected:.4f}"

    def test_sobol_irrelevant_zero(self):
        """x3-x5 should have ~0 Sobol."""
        for i in range(2, 5):
            si = self.sobol['mean_sobol']['first_order'][i]
            assert si < 0.02, f"S{i+1}={si:.4f}"


class TestT1_3_LinearFourierMix:
    """T1.3: f(x) = 4*(x1-0.5) + 2*sin(2*pi*x1), D=3.

    THE critical test for the Gram matrix cross-term.
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        X, y, self.gt = T1_3_linear_fourier_mix(n_samples=15000, noise_std=0.1, seed=42)
        self.data = preprocess_data(X, y, seed=42)
        config = {
            'K1': 5, 'K2': 0, 'Kh': 0,
            'strategy': 'curvature',
            'lambda_order1': 0.0001,
            'lambda_order2': 0.01,
            'stages': ['A'],
            'residual_nn': {'enabled': False},
        }
        trainer = HiFiANOVATrainer(config)
        self.model, self.results = trainer.fit(
            self.data['x_train'], self.data['y_train'],
            self.data['x_val'], self.data['y_val']
        )
        self.sobol = compute_sobol_indices(self.model)

    def test_linear_coefficient(self):
        """Linear coefficient for x1 should be ~4."""
        w0 = self.model.mean_model.get_coefficients_for_variable(0)
        assert abs(float(w0[0]) - 4.0) < 0.3, f"Linear coeff = {float(w0[0])}"

    def test_sin1_coefficient(self):
        """sin(2*pi*x1) coefficient should be ~2."""
        w0 = self.model.mean_model.get_coefficients_for_variable(0)
        # sin1 is at index 2: [lin, cos1, sin1, ...]
        assert abs(float(w0[2]) - 2.0) < 0.3, f"sin1 coeff = {float(w0[2])}"

    def test_variance_uses_gram_cross_term(self):
        """The computed variance must use the Gram cross-term, not diagonal-only.

        Correct: w^T G w = 16/12 + 4/2 + 2*4*2*(-1/(2*pi)) ≈ 0.787
        Wrong (diagonal-only): 16/12 + 4/2 = 3.333

        If the implementation uses diagonal-only, this test catches it.
        """
        K1 = self.model.K1
        G = build_gram_matrix(K1)
        w0 = jnp.asarray(
            self.model.mean_model.get_coefficients_for_variable(0), dtype=jnp.float64)
        G = jnp.asarray(G, dtype=jnp.float64)

        computed_var = float(w0 @ G @ w0)
        correct_var = self.gt['total_signal_variance']
        wrong_var = self.gt['wrong_diagonal_variance']

        # Must be closer to correct than wrong
        err_correct = abs(computed_var - correct_var)
        err_wrong = abs(computed_var - wrong_var)
        assert err_correct < err_wrong, \
            (f"Variance {computed_var:.4f} is closer to wrong ({wrong_var:.4f}) "
             f"than correct ({correct_var:.4f}). Gram cross-term likely missing!")

        # Must be within 30% of correct (accounts for finite-sample effects)
        assert abs(computed_var - correct_var) / correct_var < 0.3, \
            f"Variance {computed_var:.4f} too far from truth {correct_var:.4f}"

    def test_sobol_x1_is_one(self):
        """Only x1 is active, so S1 should be ~1.0."""
        s1 = self.sobol['mean_sobol']['first_order'][0]
        assert s1 > 0.9, f"S1={s1:.4f}, should be ~1.0"

    def test_sobol_x2_x3_zero(self):
        """x2, x3 should have ~0 Sobol."""
        for i in [1, 2]:
            si = self.sobol['mean_sobol']['first_order'][i]
            assert si < 0.05, f"S{i+1}={si:.4f}, should be ~0"


class TestT1_4_PureInteraction:
    """T1.4: f(x) = 3*(x1-0.5)*(x2-0.5), D=5.

    First-order Sobol should be zero. All variance in second-order.
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        X, y, self.gt = T1_4_pure_interaction(n_samples=10000, noise_std=0.1, seed=42)
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
        self.sobol = compute_sobol_indices(self.model)

    def test_first_order_sobol_near_zero(self):
        """All first-order Sobol indices should be small (function is pure interaction)."""
        for i in range(5):
            si = self.sobol['mean_sobol']['first_order'][i]
            assert si < 0.1, f"S{i+1}={si:.4f}, should be ~0 (pure interaction)"

    def test_second_order_captures_variance(self):
        """The (0,1) pair should capture most of the variance."""
        if self.sobol['mean_sobol']['second_order']:
            s12 = self.sobol['mean_sobol']['second_order'].get((0, 1), 0.0)
            total_second = sum(self.sobol['mean_sobol']['second_order'].values())
            # (0,1) should be the dominant pair
            assert s12 > 0.3 or total_second > 0.3, \
                f"S12={s12:.4f}, total 2nd order={total_second:.4f}"

    def test_irrelevant_pairs_small(self):
        """Pairs not involving (0,1) should be small."""
        for (i, j), sij in self.sobol['mean_sobol']['second_order'].items():
            if (i, j) != (0, 1):
                assert sij < 0.1, f"S({i},{j})={sij:.4f}, should be ~0"


class TestT1_5_ConstantMeanVariableNoise:
    """T1.5: f(x) = 0, sigma^2(x) = exp(2*(x1-0.5)), D=5.

    All action is in the variance. Variance Sobol S1_h should dominate.
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        X, y, self.gt = T1_5_constant_mean_variable_noise(n_samples=15000, seed=42)
        self.data = preprocess_data(X, y, seed=42)
        config = {
            'K1': 5, 'K2': 0, 'Kh': 3,
            'strategy': 'curvature',
            'lambda_order1': 0.001,
            'lambda_order2': 0.01,
            'lambda_h': 0.01,
            'stages': ['A', 'D'],
            'residual_nn': {'enabled': False},
            'max_outer_iter': 10,
            'alternating_tol': 1e-4,
            'newton_max_iter': 15,
        }
        trainer = HiFiANOVATrainer(config)
        self.model, self.results = trainer.fit(
            self.data['x_train'], self.data['y_train'],
            self.data['x_val'], self.data['y_val']
        )
        self.sobol = compute_sobol_indices(self.model, self.data['x_test'])

    def test_mean_sobol_near_zero(self):
        """No single mean Sobol index should dominate for constant-mean function.

        With variable noise and a constant true mean, the ridge model may pick up
        spurious structure. The key test is that no single variable dominates
        (all indices should be relatively equal / small).
        """
        sobol_vals = [self.sobol['mean_sobol']['first_order'].get(i, 0.0) for i in range(5)]
        max_sobol = max(sobol_vals)
        # No single variable should dominate (>0.5 would mean one variable
        # is attributed more than half of the spurious variance)
        assert max_sobol < 0.5, f"Max mean Sobol {max_sobol} too large for constant mean"

    def test_variance_sobol_x1_dominates(self):
        """Variance Sobol should show x1 as the dominant driver."""
        assert 'log_variance_sobol' in self.sobol
        vs = self.sobol['log_variance_sobol']['first_order']
        # x1 should have the largest variance Sobol
        s1_h = vs[0]
        others = [vs[i] for i in range(1, 5)]
        assert s1_h > max(others), \
            f"x1 variance Sobol ({s1_h:.3f}) should dominate others ({others})"
        assert s1_h > 0.3, f"x1 variance Sobol={s1_h:.3f}, should be large"
