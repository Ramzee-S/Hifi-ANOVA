"""Tier 4: Controlled Complexity Functions.

Tests tunable parameters: interaction strength, frequency content, SNR.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.slow

from hifi_anova.data.test_functions import (
    T4_1_tunable_interaction, T4_2_tunable_frequency, T4_3_tunable_snr,
)
from hifi_anova.data.preprocessing import preprocess_data
from hifi_anova.training.trainer import HiFiANOVATrainer
from hifi_anova.analysis.sobol import compute_sobol_indices
from hifi_anova.analysis.diagnostics import variance_accounting_report


class TestT4_1_TunableInteraction:
    """T4.1: Interaction strength controlled by alpha parameter.

    alpha=0: purely additive. alpha=1: full interactions.
    Tests the transition from "order 1 is enough" to "order 2 needed."
    """

    def _fit_model(self, alpha, K2=3):
        X, y, gt = T4_1_tunable_interaction(
            n_samples=8000, alpha=alpha, D=5, noise_std=0.1, seed=42
        )
        data = preprocess_data(X, y, seed=42)
        stages = ['A', 'B'] if K2 > 0 else ['A']
        config = {
            'K1': 5, 'K2': K2, 'Kh': 0,
            'strategy': 'curvature',
            'lambda_order1': 0.0001,
            'lambda_order2': 0.001,
            'stages': stages,
            'residual_nn': {'enabled': False},
        }
        trainer = HiFiANOVATrainer(config)
        model, results = trainer.fit(
            data['x_train'], data['y_train'],
            data['x_val'], data['y_val']
        )
        sobol = compute_sobol_indices(model, data['x_test'])
        return model, results, sobol, gt, data

    def test_alpha_zero_first_order_sufficient(self):
        """At alpha=0 (no interactions), first-order model should suffice."""
        model, results, sobol, gt, data = self._fit_model(alpha=0.0)
        # Second-order contributions should be negligible
        total_second = sum(sobol['mean_sobol']['second_order'].values())
        assert total_second < 0.1, \
            f"At alpha=0, second order should be ~0, got {total_second:.4f}"

    def test_alpha_one_second_order_needed(self):
        """At alpha=1 (full interactions), second-order terms add value."""
        model, results, sobol, gt, data = self._fit_model(alpha=1.0)
        total_second = sum(sobol['mean_sobol']['second_order'].values())
        assert total_second > 0.001, \
            f"At alpha=1, second order should be present, got {total_second:.4f}"

    def test_interaction_increases_with_alpha(self):
        """Second-order Sobol fraction should increase with alpha."""
        _, _, sobol_0, _, _ = self._fit_model(alpha=0.0)
        _, _, sobol_1, _, _ = self._fit_model(alpha=1.0)

        second_0 = sum(sobol_0['mean_sobol']['second_order'].values())
        second_1 = sum(sobol_1['mean_sobol']['second_order'].values())
        assert second_1 > second_0, \
            f"Second-order should increase: alpha=0 ({second_0:.4f}) < alpha=1 ({second_1:.4f})"

    def test_rmse_improves_with_second_order_at_high_alpha(self):
        """At high alpha, adding second-order terms should improve RMSE."""
        # First order only
        _, results_k0, _, _, _ = self._fit_model(alpha=2.0, K2=0)
        # First + second order
        _, results_k3, _, _, _ = self._fit_model(alpha=2.0, K2=3)

        rmse_order1 = results_k0['stage_A']['rmse_val']
        rmse_order2 = results_k3['stage_B']['rmse_val']
        assert rmse_order2 < rmse_order1, \
            f"Second order should help at alpha=2: order1={rmse_order1:.4f}, order2={rmse_order2:.4f}"


class TestT4_2_TunableFrequency:
    """T4.2: True frequency complexity controlled by K_true.

    Tests:
    - K1 < K_true: truncation (model can't represent all harmonics)
    - K1 > K_true: suppression (extra harmonics regularized away)
    """

    def _fit_model(self, K_true, K1):
        X, y, gt = T4_2_tunable_frequency(
            n_samples=8000, K_true=K_true, D=3, noise_std=0.1, seed=42
        )
        data = preprocess_data(X, y, seed=42)
        config = {
            'K1': K1, 'K2': 0, 'Kh': 0,
            'strategy': 'curvature',
            'lambda_order1': 0.0001,
            'stages': ['A'],
            'residual_nn': {'enabled': False},
        }
        trainer = HiFiANOVATrainer(config)
        model, results = trainer.fit(
            data['x_train'], data['y_train'],
            data['x_val'], data['y_val']
        )
        sobol = compute_sobol_indices(model)
        return model, results, sobol, gt, data

    def test_exact_recovery_when_K1_equals_Ktrue(self):
        """When K1 = K_true, should get excellent fit."""
        K_true = 3
        model, results, sobol, gt, data = self._fit_model(K_true=K_true, K1=K_true)
        va = variance_accounting_report(model, data['x_test'], data['y_test'])
        assert va['R_squared'] > 0.95, \
            f"K1=K_true={K_true}: R²={va['R_squared']:.4f}, should be >0.95"

    def test_truncation_when_K1_less_than_Ktrue(self):
        """When K1 < K_true, fit degrades (can't represent high frequencies)."""
        model_full, res_full, _, _, data = self._fit_model(K_true=8, K1=8)
        model_trunc, res_trunc, _, _, _ = self._fit_model(K_true=8, K1=3)

        rmse_full = res_full['stage_A']['rmse_val']
        rmse_trunc = res_trunc['stage_A']['rmse_val']
        assert rmse_trunc > rmse_full, \
            f"Truncation should hurt: K1=3 ({rmse_trunc:.4f}) > K1=8 ({rmse_full:.4f})"

    def test_overspecification_handled_by_regularization(self):
        """When K1 > K_true, extra harmonics should be suppressed by regularization."""
        K_true = 3
        model, results, sobol, gt, data = self._fit_model(K_true=K_true, K1=10)
        va = variance_accounting_report(model, data['x_test'], data['y_test'])
        # Should still get good fit (regularization suppresses unused harmonics)
        assert va['R_squared'] > 0.90, \
            f"K1=10 > K_true=3: R²={va['R_squared']:.4f}, regularization should help"

    def test_sobol_equal_across_variables(self):
        """All active variables have same coefficient structure -> equal Sobol."""
        K_true = 3
        model, results, sobol, gt, data = self._fit_model(K_true=K_true, K1=5)
        # All 3 variables have identical function, so Sobol should be ~1/3 each
        for i in range(3):
            si = sobol['mean_sobol']['first_order'][i]
            assert abs(si - 1.0/3.0) < 0.1, \
                f"S{i+1}={si:.4f}, should be ~0.333"


class TestT4_3_TunableSNR:
    """T4.3: Heteroscedasticity severity controlled by beta.

    beta=0: homoscedastic. beta=2: noise varies by ~7x.
    """

    def _fit_model(self, beta):
        X, y, gt = T4_3_tunable_snr(
            n_samples=12000, beta=beta, noise_variable=2, seed=42
        )
        data = preprocess_data(X, y, seed=42)
        config = {
            'K1': 5, 'K2': 0, 'Kh': 3,
            'strategy': 'curvature',
            'lambda_order1': 0.001,
            'lambda_h': 0.01,
            'stages': ['A', 'D'],
            'residual_nn': {'enabled': False},
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
        return model, results, sobol, gt, data

    def test_beta_zero_homoscedastic(self):
        """At beta=0, variance Sobol should show no structure."""
        model, results, sobol, gt, data = self._fit_model(beta=0.0)
        assert 'variance_sobol' in sobol, "variance_sobol missing from results"
        vs = sobol['variance_sobol']['first_order']
        # No variable should dominate
        max_v = max(vs.values())
        # With beta=0 (homoscedastic), the variance model shouldn't find structure
        # Allow some noise in the estimate
        assert max_v < 0.6, \
            f"At beta=0 (homoscedastic), no var should dominate: max={max_v:.4f}"

    def test_beta_two_strong_heteroscedasticity(self):
        """At beta=2, noise variable (x3) should dominate variance Sobol."""
        model, results, sobol, gt, data = self._fit_model(beta=2.0)
        assert 'variance_sobol' in sobol, "variance_sobol missing from results"
        vs = sobol['variance_sobol']['first_order']
        s3_h = vs[2]  # noise_variable=2 (x3)
        assert s3_h > 0.2, \
            f"At beta=2, x3 should dominate variance: S3_h={s3_h:.4f}"

    def test_heteroscedasticity_increases_with_beta(self):
        """Variance Sobol of noise variable should increase with beta."""
        _, _, sobol_low, _, _ = self._fit_model(beta=0.5)
        _, _, sobol_high, _, _ = self._fit_model(beta=2.0)

        assert 'variance_sobol' in sobol_low, "variance_sobol missing from low-beta results"
        assert 'variance_sobol' in sobol_high, "variance_sobol missing from high-beta results"
        s3_low = sobol_low['variance_sobol']['first_order'][2]
        s3_high = sobol_high['variance_sobol']['first_order'][2]
        assert s3_high > s3_low, \
            f"Var Sobol should increase: beta=0.5 ({s3_low:.4f}) < beta=2 ({s3_high:.4f})"

    def test_mean_sobol_stable_across_beta(self):
        """Mean Sobol should be relatively stable regardless of beta."""
        _, _, sobol_0, _, _ = self._fit_model(beta=0.0)
        _, _, sobol_2, _, _ = self._fit_model(beta=2.0)

        s1_0 = sobol_0['mean_sobol']['first_order'][0]
        s1_2 = sobol_2['mean_sobol']['first_order'][0]
        # Mean Sobol for x1 shouldn't change dramatically
        assert abs(s1_0 - s1_2) < 0.3, \
            f"Mean S1 should be stable: beta=0 ({s1_0:.4f}) vs beta=2 ({s1_2:.4f})"
