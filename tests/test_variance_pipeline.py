"""Tests for the variance pipeline: VarianceModel, Newton solver, redecompose,
pipeline edge cases, and Sobol invariants.
"""

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from hifi_anova.core.features import build_first_order_features, basis_size
from hifi_anova.model.variance_model import VarianceModel
from hifi_anova.training.newton import newton_solve_log_variance
from hifi_anova.training.redecompose import redecompose
from hifi_anova.training.trainer import HiFiANOVATrainer
from hifi_anova.analysis.sobol import compute_sobol_indices

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_variance_model(D, Kh, h0=0.0, w1=None, K2h=0, w2=None):
    """Construct a VarianceModel with explicit or default coefficients."""
    B = basis_size(Kh, True, 'fourier')
    if w1 is None:
        w1 = jnp.zeros(D * B)
    if w2 is None:
        w2 = jnp.array([], dtype=jnp.float32)
    return VarianceModel(
        h0=jnp.array(h0, dtype=jnp.float32),
        w1=jnp.array(w1, dtype=jnp.float32),
        Kh=Kh,
        D=D,
        w2=jnp.array(w2, dtype=jnp.float32),
        K2h=K2h,
    )


def _make_psi1(N, D, Kh, seed=0):
    """Build first-order variance features."""
    rng = np.random.RandomState(seed)
    x = jnp.array(rng.rand(N, D).astype(np.float32))
    return build_first_order_features(x, Kh)


def _fit_simple_model(N=500, D=3, K1=3, seed=42, basis_name='fourier'):
    """Fit a first-order HiFiANOVA on simple linear data. Returns (model, results)."""
    rng = np.random.RandomState(seed)
    x = rng.rand(N, D).astype(np.float32)
    y = (x[:, 0] * 2.0 + x[:, 1] * 3.0 + 0.1 * rng.randn(N)).astype(np.float32)
    # Use a 80/20 split
    n_val = max(100, N // 5)
    x_tr, x_val = x[:-n_val], x[-n_val:]
    y_tr, y_val = y[:-n_val], y[-n_val:]

    trainer = HiFiANOVATrainer({
        'K1': K1, 'K2': 0,
        'stages': ['A'],
        'lambda_order1': 0.001,
        'basis_name': basis_name,
    })
    model, results = trainer.fit(
        jnp.array(x_tr), jnp.array(y_tr),
        jnp.array(x_val), jnp.array(y_val),
    )
    return model, results


# ===========================================================================
# 1. VarianceModel
# ===========================================================================

class TestVarianceModel:

    def test_zero_weights_unit_variance(self):
        """w1=zeros, h0=0 => predict_variance all ones."""
        D, Kh, N = 3, 3, 50
        vm = _make_variance_model(D, Kh, h0=0.0)
        psi1 = _make_psi1(N, D, Kh)
        var_pred = vm.predict_variance(psi1)
        assert var_pred.shape == (N,)
        np.testing.assert_allclose(np.array(var_pred), 1.0, atol=1e-5)

    def test_h0_only(self):
        """w1=zeros, h0=log(4.0) => predict_variance all 4.0."""
        D, Kh, N = 3, 3, 50
        h0 = float(np.log(4.0))
        vm = _make_variance_model(D, Kh, h0=h0)
        psi1 = _make_psi1(N, D, Kh)
        var_pred = vm.predict_variance(psi1)
        np.testing.assert_allclose(np.array(var_pred), 4.0, atol=1e-4)

    def test_always_positive(self):
        """predict_variance is strictly positive for any w1."""
        D, Kh, N = 4, 5, 100
        B = basis_size(Kh, True, 'fourier')
        rng = np.random.RandomState(7)
        w1 = rng.randn(D * B).astype(np.float32)
        vm = _make_variance_model(D, Kh, w1=w1)
        psi1 = _make_psi1(N, D, Kh, seed=7)
        var_pred = vm.predict_variance(psi1)
        assert float(jnp.min(var_pred)) > 0.0

    def test_log_variance_consistency(self):
        """exp(predict_log_variance) == predict_variance."""
        D, Kh, N = 3, 4, 80
        B = basis_size(Kh, True, 'fourier')
        rng = np.random.RandomState(13)
        w1 = rng.randn(D * B).astype(np.float32)
        vm = _make_variance_model(D, Kh, h0=0.5, w1=w1)
        psi1 = _make_psi1(N, D, Kh, seed=13)
        log_var = vm.predict_log_variance(psi1)
        var_direct = vm.predict_variance(psi1)
        np.testing.assert_allclose(
            np.array(jnp.exp(log_var)),
            np.array(var_direct),
            atol=1e-5,
        )

    def test_coefficient_slicing(self):
        """get_coefficients_for_variable partitions w1 into D non-overlapping blocks."""
        D, Kh = 4, 3
        B = basis_size(Kh, True, 'fourier')
        rng = np.random.RandomState(99)
        w1 = rng.randn(D * B).astype(np.float32)
        vm = _make_variance_model(D, Kh, w1=w1)

        for i in range(D):
            coef_i = np.array(vm.get_coefficients_for_variable(i))
            expected = np.array(w1[i * B: (i + 1) * B])
            np.testing.assert_array_equal(coef_i, expected)
            assert coef_i.shape == (B,)

    def test_has_second_order_false(self):
        """K2h=0 => has_second_order is False."""
        D, Kh = 3, 3
        vm = _make_variance_model(D, Kh, K2h=0)
        assert vm.has_second_order is False

    def test_has_second_order_true(self):
        """K2h>0 with nonempty w2 => has_second_order is True."""
        D, Kh, K2h = 3, 3, 2
        B2 = basis_size(K2h, True, 'fourier') ** 2
        # One pair for D=3 would give 3 pairs, but we just need nonempty w2
        n_pairs = 1
        w2 = np.ones(n_pairs * B2, dtype=np.float32)
        vm = _make_variance_model(D, Kh, K2h=K2h, w2=w2)
        assert vm.has_second_order is True


# ===========================================================================
# 2. Newton solver
# ===========================================================================

class TestNewtonSolver:

    @pytest.fixture(autouse=True)
    def setup(self):
        self.N = 500
        self.D = 3
        self.Kh = 3
        B = basis_size(self.Kh, True, 'fourier')
        self.B = B
        rng = np.random.RandomState(42)
        x = rng.rand(self.N, self.D)
        self.psi = np.array(
            build_first_order_features(
                jnp.array(x, dtype=jnp.float32), self.Kh
            )
        )
        self.reg_diag = 0.01 * np.ones(self.D * B)

    def _make_r2(self, sigma2=2.0, seed=99):
        return sigma2 * np.random.RandomState(seed).chisquare(1, size=self.N)

    def test_homoscedastic_recovery(self):
        """Constant r² ~ sigma²*chi²(1) => h0 ≈ log(sigma²), ||w_h|| ≈ 0."""
        sigma2_true = 2.0
        r2 = self._make_r2(sigma2_true, seed=99)
        w_h_init = np.zeros(self.D * self.B)
        h0_init = float(np.log(np.var(r2)))

        w_h, h0 = newton_solve_log_variance(
            self.psi, r2.astype(np.float64),
            w_h_init, h0_init, self.reg_diag,
        )
        # h0 should be close to log(sigma²): within ~0.5 nats with N=500
        assert abs(h0 - np.log(sigma2_true)) < 1.0
        # w_h should be small (homoscedastic => no first-order structure)
        assert float(np.max(np.abs(np.array(w_h)))) < 1.0

    def test_all_ones_residuals(self):
        """r²=1 for all => h0 ≈ 0 (sigma² ≈ 1)."""
        r2 = np.ones(self.N, dtype=np.float64)
        w_h_init = np.zeros(self.D * self.B)
        h0_init = 0.0

        w_h, h0 = newton_solve_log_variance(
            self.psi, r2, w_h_init, h0_init, self.reg_diag,
        )
        # h0 should be 0: sigma²=1 is the MLE for unit residuals
        assert abs(h0) < 0.2

    def test_max_iter_zero(self):
        """max_iter=0 returns initial values unchanged."""
        w_h_init = np.array(0.5 * np.ones(self.D * self.B))
        h0_init = 1.23
        r2 = self._make_r2(seed=10)

        w_h, h0 = newton_solve_log_variance(
            self.psi, r2.astype(np.float64),
            w_h_init, h0_init, self.reg_diag,
            max_iter=0,
        )
        np.testing.assert_array_almost_equal(np.array(w_h), w_h_init, decimal=10)
        assert abs(h0 - h0_init) < 1e-10

    def test_output_shapes(self):
        """w_h shape matches input shape, h0 is a Python float."""
        r2 = self._make_r2(seed=77)
        w_h_init = np.zeros(self.D * self.B)
        h0_init = 0.0

        w_h, h0 = newton_solve_log_variance(
            self.psi, r2.astype(np.float64),
            w_h_init, h0_init, self.reg_diag,
        )
        assert np.array(w_h).shape == (self.D * self.B,)
        assert isinstance(h0, float)

    def test_monotone_objective(self):
        """Objective should not increase across Newton iterations (Armijo property).

        We run one iteration at a time by setting max_iter=1 and re-calling,
        verifying the per-iteration objective doesn't go up.
        """
        r2 = self._make_r2(sigma2_true := 3.0, seed=55)
        sigma2_true = 3.0
        r2 = self._make_r2(sigma2_true, seed=55)
        w_h = np.zeros(self.D * self.B)
        h0 = float(np.log(np.mean(r2)))

        def _objective_val(psi, w_h, h0, r2, reg_diag):
            """Evaluate the augmented Newton objective."""
            N, F = psi.shape
            psi_aug = np.concatenate([np.ones((N, 1)), psi], axis=1)
            reg_aug = np.concatenate([[0.0], reg_diag])
            theta = np.concatenate([[h0], w_h])
            h = psi_aug @ theta
            h_clamped = np.clip(h, -30.0, 30.0)
            sigma2 = np.exp(h_clamped)
            data = np.sum(0.5 * h_clamped + 0.5 * r2 / sigma2)
            reg = 0.5 * np.sum(reg_aug * theta ** 2)
            return data + reg

        prev_obj = _objective_val(self.psi, w_h, h0, r2, self.reg_diag)
        objectives = [prev_obj]
        for _ in range(5):
            w_h, h0 = newton_solve_log_variance(
                self.psi, r2.astype(np.float64),
                np.array(w_h), h0, self.reg_diag, max_iter=1,
            )
            w_h = np.array(w_h)
            obj = _objective_val(self.psi, w_h, h0, r2, self.reg_diag)
            objectives.append(obj)

        for i in range(1, len(objectives)):
            assert objectives[i] <= objectives[i - 1] + 1e-6, (
                f"Objective increased at step {i}: "
                f"{objectives[i - 1]:.6f} -> {objectives[i]:.6f}"
            )


# ===========================================================================
# 3. Redecompose
# ===========================================================================

class TestRedecompose:

    @pytest.fixture(autouse=True)
    def setup(self):
        """Fit a small Fourier-only model once for reuse."""
        self.model, _ = _fit_simple_model(N=500, D=3, K1=3)

    def test_identity_redecompose(self):
        """Fourier-only model: redecompose should leave predictions virtually unchanged.

        Total predictions must agree within float32 tolerance because there is no
        NN residual — the Fourier model simply projects onto itself.
        """
        model2 = redecompose(self.model, reg_diag=None, n_eval=10000, seed=0)

        rng = np.random.RandomState(7)
        x_test = jnp.array(rng.rand(200, 3).astype(np.float32))

        phi1_orig = self.model.build_phi1(x_test)
        phi2_orig = self.model.build_phi2(x_test)
        pred_orig = self.model.mean_model.predict(phi1_orig, phi2_orig)

        phi1_new = model2.build_phi1(x_test)
        phi2_new = model2.build_phi2(x_test)
        pred_new = model2.mean_model.predict(phi1_new, phi2_new)

        # Should be close — small discrepancy from Monte Carlo projection
        np.testing.assert_allclose(
            np.array(pred_orig), np.array(pred_new), atol=0.05
        )

    def test_shape_preserved(self):
        """w1 and w2 lengths are identical before and after redecompose."""
        model2 = redecompose(self.model, reg_diag=None, n_eval=10000, seed=0)

        assert len(model2.mean_model.w1) == len(self.model.mean_model.w1)
        assert len(model2.mean_model.w2) == len(self.model.mean_model.w2)

    @pytest.mark.slow
    def test_sobol_preserved(self):
        """Sobol indices should be approximately preserved for a Fourier-only model."""
        model2 = redecompose(self.model, reg_diag=None, n_eval=50000, seed=42)

        s_orig = compute_sobol_indices(self.model)['mean_sobol']['first_order']
        s_new = compute_sobol_indices(model2)['mean_sobol']['first_order']

        for i in range(self.model.D):
            assert abs(s_orig[i] - s_new[i]) < 0.1, (
                f"Sobol[{i}] changed from {s_orig[i]:.4f} to {s_new[i]:.4f}"
            )


# ===========================================================================
# 4. Pipeline edge cases
# ===========================================================================

class TestPipelineEdgeCases:

    def _val_split(self, x, y, val_frac=0.2):
        n = len(y)
        n_val = max(50, int(n * val_frac))
        return (jnp.array(x[:-n_val].astype(np.float32)),
                jnp.array(y[:-n_val].astype(np.float32)),
                jnp.array(x[-n_val:].astype(np.float32)),
                jnp.array(y[-n_val:].astype(np.float32)))

    def test_d1_pipeline(self):
        """D=1: trainer should not crash and Sobol first_order[0] ≈ 1."""
        rng = np.random.RandomState(0)
        N = 500
        x = rng.rand(N, 1).astype(np.float32)
        y = (2.0 * x[:, 0] + 0.05 * rng.randn(N)).astype(np.float32)
        x_tr, y_tr, x_val, y_val = self._val_split(x, y)

        trainer = HiFiANOVATrainer({
            'K1': 3, 'K2': 0, 'stages': ['A'], 'lambda_order1': 0.001,
        })
        model, _ = trainer.fit(x_tr, y_tr, x_val, y_val)

        sobol = compute_sobol_indices(model)
        s1 = sobol['mean_sobol']['first_order']
        assert abs(s1[0] - 1.0) < 1e-6

    def test_legendre_full_pipeline(self):
        """basis_name='legendre': fit → predict → Sobol first-order sums ≈ 1."""
        rng = np.random.RandomState(1)
        N = 600
        D = 3
        x = rng.rand(N, D).astype(np.float32)
        y = (x[:, 0] * 2.0 + x[:, 1] * 3.0 + 0.1 * rng.randn(N)).astype(np.float32)
        x_tr, y_tr, x_val, y_val = self._val_split(x, y)

        trainer = HiFiANOVATrainer({
            'K1': 5, 'K2': 0, 'stages': ['A'],
            'lambda_order1': 0.01, 'basis_name': 'legendre',
        })
        model, _ = trainer.fit(x_tr, y_tr, x_val, y_val)

        sobol = compute_sobol_indices(model)
        s_first = sobol['mean_sobol']['first_order']
        total = sum(s_first.values())
        assert abs(total - 1.0) < 1e-4

        # Model can produce predictions
        pred = model.predict_mean_only(x_val)
        assert pred.shape == (len(y_val),)
        assert jnp.all(jnp.isfinite(pred))

    def test_haar_full_pipeline(self):
        """basis_name='haar': fit → predict → Sobol first-order sums ≈ 1."""
        rng = np.random.RandomState(2)
        N = 600
        D = 3
        x = rng.rand(N, D).astype(np.float32)
        y = (x[:, 0] * 2.0 + x[:, 1] ** 2 + 0.1 * rng.randn(N)).astype(np.float32)
        x_tr, y_tr, x_val, y_val = self._val_split(x, y)

        trainer = HiFiANOVATrainer({
            'K1': 5, 'K2': 0, 'stages': ['A'],
            'lambda_order1': 0.01, 'basis_name': 'haar',
        })
        model, _ = trainer.fit(x_tr, y_tr, x_val, y_val)

        sobol = compute_sobol_indices(model)
        s_first = sobol['mean_sobol']['first_order']
        total = sum(s_first.values())
        assert abs(total - 1.0) < 1e-4

        pred = model.predict_mean_only(x_val)
        assert pred.shape == (len(y_val),)
        assert jnp.all(jnp.isfinite(pred))

    @pytest.mark.slow
    def test_four_stage_pipeline(self):
        """Stages A+B+D on heteroscedastic data: no crash, NLL is finite."""
        rng = np.random.RandomState(42)
        N = 800
        D = 3
        x = rng.rand(N, D).astype(np.float32)
        f_mean = 2.0 * x[:, 0] + 3.0 * x[:, 1] * x[:, 2]
        sigma = np.exp(0.5 * x[:, 0]).astype(np.float32)
        y = (f_mean + sigma * rng.randn(N)).astype(np.float32)
        x_tr, y_tr, x_val, y_val = self._val_split(x, y)

        trainer = HiFiANOVATrainer({
            'K1': 5, 'K2': 3, 'Kh': 3,
            'stages': ['A', 'B', 'D'],
            'lambda_order1': 0.01,
            'lambda_order2': 0.1,
            'selection_method': 'bic',
        })
        model, results = trainer.fit(
            jnp.array(x_tr), jnp.array(y_tr),
            jnp.array(x_val), jnp.array(y_val),
        )

        assert 'stage_D' in results
        assert np.isfinite(results['stage_D']['nll_val'])
        assert np.isfinite(results['stage_D']['rmse_val'])

        # Variance model is attached
        assert model.variance_model is not None

        # Mean predictions finite
        pred = model.predict_mean_only(x_val)
        assert jnp.all(jnp.isfinite(pred))


# ===========================================================================
# 5. Sobol invariants
# ===========================================================================

class TestSobolInvariants:

    @pytest.fixture(autouse=True)
    def setup(self):
        """Fit a second-order model to use throughout."""
        rng = np.random.RandomState(42)
        N = 700
        D = 4
        x = rng.rand(N, D).astype(np.float32)
        y = (
            2.0 * x[:, 0]
            + 3.0 * x[:, 1]
            + 1.5 * x[:, 0] * x[:, 2]
            + 0.1 * rng.randn(N)
        ).astype(np.float32)
        n_val = 140
        x_tr, y_tr = x[:-n_val], y[:-n_val]
        x_val, y_val = x[-n_val:], y[-n_val:]

        trainer = HiFiANOVATrainer({
            'K1': 4, 'K2': 3,
            'stages': ['A', 'B'],
            'lambda_order1': 0.01,
            'lambda_order2': 0.1,
        })
        self.model, _ = trainer.fit(
            jnp.array(x_tr), jnp.array(y_tr),
            jnp.array(x_val), jnp.array(y_val),
        )
        self.D = D
        self.sobol = compute_sobol_indices(self.model)

    def test_total_order_geq_first_order(self):
        """For all variables, S_total >= S_first (total order includes interaction variance)."""
        s1 = self.sobol['mean_sobol']['first_order']
        st = self.sobol['mean_sobol']['total_order']
        for i in range(self.D):
            assert st[i] >= s1[i] - 1e-9, (
                f"Variable {i}: total_order={st[i]:.6f} < first_order={s1[i]:.6f}"
            )

    def test_mean_sobol_first_order_sums(self):
        """First-order + second-order Sobol indices sum to 1."""
        s1 = self.sobol['mean_sobol']['first_order']
        s2 = self.sobol['mean_sobol']['second_order']
        total = sum(s1.values()) + sum(s2.values())
        assert abs(total - 1.0) < 1e-4, f"Sobol sum = {total:.6f}, expected 1.0"

    def test_variance_sobol_sum_to_one(self):
        """If variance Sobol is computed, its first-order terms should sum to 1.

        This requires a heteroscedastic model with a VarianceModel attached.
        We run a small Stage-D pipeline here to get variance_sobol.
        """
        rng = np.random.RandomState(7)
        N = 600
        D = 3
        x = rng.rand(N, D).astype(np.float32)
        sigma = np.exp(0.4 * x[:, 0]).astype(np.float32)
        y = (x[:, 0] * 2.0 + sigma * rng.randn(N)).astype(np.float32)
        n_val = 120
        x_tr, y_tr = x[:-n_val], y[:-n_val]
        x_val, y_val = x[-n_val:], y[-n_val:]

        trainer = HiFiANOVATrainer({
            'K1': 3, 'K2': 0, 'Kh': 3,
            'stages': ['A', 'D'],
            'lambda_order1': 0.01,
            'lambda_h': 0.1,
        })
        model_hsc, results = trainer.fit(
            jnp.array(x_tr), jnp.array(y_tr),
            jnp.array(x_val), jnp.array(y_val),
        )

        sobol_hsc = compute_sobol_indices(model_hsc)
        assert 'variance_sobol' in sobol_hsc, "variance_sobol key missing"

        s_var = sobol_hsc['variance_sobol']['first_order']
        total_var_sobol = sum(s_var.values())

        # With only first-order variance model and no second-order terms,
        # all variance_sobol['first_order'] should sum to 1.0
        assert abs(total_var_sobol - 1.0) < 1e-4, (
            f"Variance Sobol sum = {total_var_sobol:.6f}, expected 1.0"
        )
