"""Tests for the variance pipeline: VarianceModel, Newton solver, redecompose,
pipeline edge cases, and Sobol invariants.
"""

import warnings

import jax
jax.config.update("jax_enable_x64", True)

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest

from hifi_anova.core.features import build_first_order_features, basis_size
from hifi_anova.model.variance_model import VarianceModel
from hifi_anova.model.linear_residual import ProjectedResidual
from hifi_anova.training.newton import newton_solve_log_variance
from hifi_anova.training.redecompose import redecompose
from hifi_anova.training.regularization import build_regularization_vector
from hifi_anova.training.joint_lambda import optimize_joint_lambda
from hifi_anova.training.trainer import HiFiANOVATrainer
from hifi_anova.analysis.sobol import compute_sobol_indices
from hifi_anova.analysis.qmc import sobol_cube_sample

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


class _LowOrderResidual(eqx.Module):
    """A residual that is exactly Fourier-representable: c * sin(2*pi*x0).

    Its low-order projection P(f_NN) equals itself, so after redecompose the
    orthogonal remainder (I-P)f_NN is ~0 — a controllable stand-in for a
    jointly-trained NN that has absorbed low-order structure (issue #3)."""
    c: float = eqx.field(static=True, default=0.5)

    def __call__(self, x):  # x: (D,) -> scalar
        return self.c * jnp.sin(2.0 * jnp.pi * x[0])


def _attach_residual(model, residual):
    return eqx.tree_at(lambda m: m.residual_net, model, residual,
                       is_leaf=lambda z: z is None)


def _accounting_total(model):
    va = compute_sobol_indices(model)['variance_accounting']
    return float(va['total_model_variance'])


def _true_variance(model, D, n=1 << 15):
    """True Var under the uniform cube measure of the full prediction."""
    X = jnp.asarray(sobol_cube_sample(D, n, 0))
    pred, _ = model.predict(X)
    return float(jnp.var(pred))


# ===========================================================================
# 3b. Redecompose with a residual — the corrected f_total operator (#3)
# ===========================================================================

class TestRedecomposeFoldsResidual:
    """The corrected redecompose projects f_total (folds P(f_NN) into Fourier,
    leaves (I-P)f_NN as the residual). Verifies exact prediction preservation and
    that the resulting decomposition is *coherent* — the drift/corruption test."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.model, _ = _fit_simple_model(N=600, D=3, K1=4)

    def test_no_residual_is_exact_identity(self):
        """With no residual net there is nothing to fold: return unchanged."""
        m2 = redecompose(self.model, reg_diag=None, n_eval=10000, seed=0)
        assert m2 is self.model or np.allclose(
            np.asarray(m2.mean_model.w1), np.asarray(self.model.mean_model.w1))

    def test_prediction_preserved_exactly(self):
        """Folding P(f_NN) into the mean and subtracting it in the residual
        preserves the total prediction to float precision."""
        m = _attach_residual(self.model, _LowOrderResidual(c=0.5))
        m2 = redecompose(m, reg_diag=None, n_eval=40000, seed=0)
        assert isinstance(m2.residual_net, ProjectedResidual)
        assert m2.has_nn_residual
        X = jnp.asarray(sobol_cube_sample(m.D, 1 << 14, 0))
        p_before, _ = m.predict(X)
        p_after, _ = m2.predict(X)
        assert float(jnp.max(jnp.abs(p_before - p_after))) < 1e-4

    def test_decomposition_becomes_coherent(self):
        """The corruption test: BEFORE redecompose the variance accounting
        (sum V_i + residual_var) overcounts the true total because the residual is
        not orthogonal to the Fourier part; AFTER, the residual is orthogonal so
        the accounting equals the true uniform-measure variance."""
        m = _attach_residual(self.model, _LowOrderResidual(c=0.5))
        true_var = _true_variance(m, m.D)

        acc_before = _accounting_total(m)
        m2 = redecompose(m, reg_diag=None, n_eval=40000, seed=0)
        acc_after = _accounting_total(m2)

        # true variance is preserved (prediction preserved) ...
        assert abs(_true_variance(m2, m2.D) - true_var) < 1e-3
        # ... the drifted accounting disagrees with it ...
        assert acc_before > true_var + 0.1
        # ... and the re-decomposed accounting is coherent with it.
        assert abs(acc_after - true_var) < 1e-2

    def test_residual_orthogonal_and_idempotent(self):
        """After redecompose the residual has ~zero low-order content: a fully
        Fourier-representable NN leaves an (almost) empty residual, and a second
        redecompose is a near-no-op."""
        m = _attach_residual(self.model, _LowOrderResidual(c=0.5))
        m2 = redecompose(m, reg_diag=None, n_eval=40000, seed=0)
        # residual variance collapses (sin is fully representable)
        va = compute_sobol_indices(m2)['variance_accounting']
        assert va['residual'] < 1e-3
        # idempotent: folding again moves essentially nothing
        w1_2 = np.asarray(m2.mean_model.w1)
        m3 = redecompose(m2, reg_diag=None, n_eval=40000, seed=0)
        assert np.max(np.abs(np.asarray(m3.mean_model.w1) - w1_2)) < 1e-2

    @pytest.mark.integration
    def test_generic_mlp_residual(self):
        """A real (untrained) eqx.nn.MLP residual: prediction preserved and the
        decomposition made coherent, with a genuinely non-zero high-order
        residual."""
        key = jax.random.PRNGKey(0)
        mlp = eqx.nn.MLP(in_size=3, out_size=1, width_size=16, depth=2, key=key)
        m = _attach_residual(self.model, mlp)
        m2 = redecompose(m, reg_diag=None, n_eval=40000, seed=0)
        X = jnp.asarray(sobol_cube_sample(m.D, 1 << 14, 0))
        p_before, _ = m.predict(X)
        p_after, _ = m2.predict(X)
        assert float(jnp.max(jnp.abs(p_before - p_after))) < 1e-4
        assert abs(_accounting_total(m2) - _true_variance(m2, m2.D)) < 1e-2


# ===========================================================================
# 3c. LAML block-diagonal vs full cross-Hessian sensitivity (OpenTopics #6/LAML)
# ===========================================================================

class TestLAMLCrossSensitivity:
    """The joint-lambda LAML uses a **block-diagonal** joint Hessian by default
    (the cross block ``H_wh = Phi^T (r/sigma^2) Psi`` has expectation zero under
    the model). This measures the approximation error against the exact cross
    block (advisor-deferred sensitivity check): on a strongly heteroscedastic
    problem with an *active* variance model it must be small — identical lambda_h,
    a tiny evidence shift, and a small ``cross_ratio``."""

    def _het_problem(self):
        rng = np.random.RandomState(1)
        N, D, K1, Kh = 1500, 3, 4, 3
        X = rng.rand(N, D)
        mean = 1.0 * X[:, 0]
        log_sig = 1.8 * (X[:, 1] - 0.5) + 1.2 * (X[:, 2] - 0.5)  # input-dependent
        y = mean + np.exp(log_sig) * rng.randn(N)
        Phi = np.asarray(build_first_order_features(jnp.asarray(X), K1),
                         dtype=np.float64)
        Psi = np.asarray(build_first_order_features(jnp.asarray(X), Kh),
                         dtype=np.float64)
        mean_reg = np.asarray(build_regularization_vector(D, K1, 0, 0,
                              'curvature', 1.0), dtype=np.float64)
        var_reg = np.asarray(build_regularization_vector(D, Kh, 0, 0,
                             'curvature', 1.0), dtype=np.float64)
        return Phi, Psi, y, mean_reg, var_reg

    @pytest.mark.integration
    def test_cross_block_is_negligible_active_variance(self):
        Phi, Psi, y, mean_reg, var_reg = self._het_problem()
        res = {}
        for cross in (False, True):
            res[cross] = optimize_joint_lambda(
                Phi, Psi, y, mean_reg, var_reg, criterion='laml',
                laml_cross=cross, lambda_h_bounds=(1e-2, 3.0), n_grid=11, seed=0)
        # variance model is active (not pinned homoscedastic at the bound)
        assert res[False]['df_h'] > 1.0
        # identical lambda_h selection, negligible evidence shift
        assert res[True]['lambda_h'] == pytest.approx(res[False]['lambda_h'], rel=1e-6)
        assert abs(res[True]['laml'] - res[False]['laml']) < 0.1
        # the cross block is small relative to the diagonal blocks everywhere
        crs = [p['cross_ratio'] for p in res[True]['path']
               if p.get('cross_ratio') is not None]
        assert crs and max(crs) < 0.05
        print(f"\n  LAML cross-sensitivity: lambda_h identical "
              f"({res[False]['lambda_h']:.3g}); Δlaml="
              f"{res[True]['laml'] - res[False]['laml']:.4f}; "
              f"max cross_ratio={max(crs):.4g}")


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
            'variable_selection': 'bic',
            # This is a pipeline smoke test (no crash, finite NLL) that means to
            # exercise the variance model. The hetero here is weak (sigma =
            # exp(0.5*x0), ~1.6x ratio) and N is small (800), so the DEC-028
            # guard would revert to a constant variance; disable it so the
            # variance path is actually exercised and attached.
            'heteroscedastic_guard': False,
        })
        model, results = trainer.fit(
            jnp.array(x_tr), jnp.array(y_tr),
            jnp.array(x_val), jnp.array(y_val),
        )

        assert 'stage_D' in results
        assert np.isfinite(results['stage_D']['nll_val'])
        assert np.isfinite(results['stage_D']['rmse_val'])

        # Variance model is attached (guard disabled above, so it is kept).
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
        assert 'log_variance_sobol' in sobol_hsc
        s_var = sobol_hsc['log_variance_sobol']['first_order']
        total_var_sobol = sum(s_var.values())

        # With only first-order variance model and no second-order terms,
        # All log_variance_sobol first-order shares should sum to 1.0.
        assert abs(total_var_sobol - 1.0) < 1e-4, (
            f"Variance Sobol sum = {total_var_sobol:.6f}, expected 1.0"
        )


# ---------------------------------------------------------------------------
# Stage-D robustness guard (heteroscedastic_guard)
# ---------------------------------------------------------------------------

class TestStageDGuard:
    """The variance stage must fail *loudly and safely* when it is ill-posed.

    A "clever user" turns on heteroscedastic fitting on data that has no
    input-dependent noise (or is near-noiseless), or with a mean basis so rich
    that the alternating 1/sigma^2 reweighting blows the mean up. Rather than
    silently returning a corrupted mean, HiFi-ANOVA warns and falls back to a
    constant variance, keeping the good mean.
    """

    @staticmethod
    def _ishigami(X, a=7.0, b=0.1):
        return (np.sin(X[:, 0]) + a * np.sin(X[:, 1]) ** 2
                + b * (X[:, 2] ** 4) * np.sin(X[:, 0]))

    def _fit(self, X, y, cfg):
        from hifi_anova.data.preprocessing import preprocess_data
        data = preprocess_data(X, y, seed=0)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            model, res = HiFiANOVATrainer(cfg).fit(
                data['x_train'], data['y_train'],
                data['x_val'], data['y_val'])
        msgs = [str(x.message) for x in w]
        return model, res, msgs, data

    @pytest.mark.integration
    def test_homoscedastic_data_falls_back_to_constant_variance(self):
        """Homoscedastic noise -> Stage D skipped/reverted, mean preserved."""
        rng = np.random.default_rng(0)
        X = rng.uniform(-np.pi, np.pi, size=(1200, 3))
        y = self._ishigami(X) + rng.normal(0, 0.5, 1200)   # constant-variance noise
        cfg = {'stages': ['A', 'B', 'D'], 'K1': 8, 'K2': 4, 'Kh': 3,
               'strategy': 'variance'}
        model, res, msgs, data = self._fit(X, y, cfg)

        # A Stage-D warning was emitted (skip or degrade), and the model is
        # homoscedastic (constant variance), so the mean is the Stage-B mean.
        assert any('Stage D' in m or 'heteroscedastic=True' in m for m in msgs), msgs
        assert model.variance_model is None
        assert model.constant_log_var is not None
        # constant variance -> identical predictive variance everywhere
        _, var = model.predict(data['x_test'])
        assert float(jnp.std(var)) < 1e-6

    @pytest.mark.integration
    def test_severe_blowup_reverts_mean(self):
        """Rich basis + strategy='variance' can diverge; guard reverts the mean."""
        from hifi_anova.data.synthetic import generate_ishigami
        X, y, _ = generate_ishigami(n_samples=1500, heteroscedastic=True,
                                    variance_variable=2, seed=1)
        cfg = {'stages': ['A', 'B', 'D'], 'K1': 12, 'K2': 6, 'Kh': 3,
               'strategy': 'variance'}
        model, res, msgs, data = self._fit(X, y, cfg)
        # Reverted to a finite, sane mean (mean-only Stage B), not nan garbage.
        pred = np.asarray(model.predict_mean_only(data['x_test']))
        assert np.all(np.isfinite(pred))
        if res.get('stage_D', {}).get('reverted'):
            assert model.variance_model is None
            assert any('Revert' in m for m in msgs), msgs

    @pytest.mark.integration
    def test_legit_heteroscedastic_is_not_reverted(self):
        """Genuine input-dependent noise -> Stage D proceeds, no revert."""
        from hifi_anova.data.synthetic import generate_ishigami
        X, y, _ = generate_ishigami(n_samples=1500, heteroscedastic=True,
                                    variance_variable=2, seed=3)
        cfg = {'stages': ['A', 'B', 'D'], 'K1': 5, 'K2': 3, 'Kh': 3,
               'strategy': 'curvature'}
        model, res, msgs, data = self._fit(X, y, cfg)
        assert model.variance_model is not None          # real variance model kept
        assert not res.get('stage_D', {}).get('reverted', False)
        assert not res.get('stage_D', {}).get('skipped', False)

    @pytest.mark.integration
    def test_guard_optout_forces_raw_fit(self):
        """heteroscedastic_guard=False disables the safety net."""
        rng = np.random.default_rng(0)
        X = rng.uniform(-np.pi, np.pi, size=(1000, 3))
        y = self._ishigami(X) + rng.normal(0, 0.5, 1000)
        cfg = {'stages': ['A', 'B', 'D'], 'K1': 8, 'K2': 4, 'Kh': 3,
               'strategy': 'variance', 'heteroscedastic_guard': False}
        model, res, msgs, data = self._fit(X, y, cfg)
        # No fallback: a real variance model is returned regardless of stability.
        assert model.variance_model is not None
        assert not res.get('stage_D', {}).get('skipped', False)

    @pytest.mark.integration
    def test_residual_plus_heteroscedastic_guard(self):
        """Stage C+D together: the guard operates with a residual net present, and
        a Stage-D revert preserves the (independently-fitted) Stage-C residual."""
        from hifi_anova.data.synthetic import generate_ishigami
        cfg = {'stages': ['A', 'B', 'C', 'D'], 'K1': 6, 'K2': 3, 'Kh': 3,
               'strategy': 'curvature', 'max_outer_iter': 8,
               'residual': {'type': 'rbf', 'n_centers': 80, 'sigma': 0.2}}
        # (a) genuine heteroscedasticity -> variance model kept, residual kept
        Xh, yh, _ = generate_ishigami(1500, heteroscedastic=True,
                                      variance_variable=2, seed=1)
        m, res, _, d = self._fit(Xh, yh, dict(cfg))
        assert m.residual_net is not None                     # Stage C survived
        assert np.all(np.isfinite(np.asarray(m.predict_mean_only(d['x_test']))))
        assert res['stage_D'].get('selected') in ('heteroscedastic', 'homoscedastic')
        # (b) homogeneous noise -> Stage D reverts, but the residual net is kept
        rng = np.random.default_rng(0)
        Xc = rng.uniform(-np.pi, np.pi, size=(1400, 3))
        yc = self._ishigami(Xc) + rng.normal(0, 0.5, 1400)
        m2, res2, _, _ = self._fit(Xc, yc, dict(cfg))
        if res2['stage_D'].get('selected') == 'homoscedastic':
            assert m2.variance_model is None
            assert m2.residual_net is not None                # revert keeps Stage C

    @pytest.mark.integration
    def test_selection_margin_is_overridable(self):
        """A large variance_selection_margin forces the homoscedastic choice.

        The model-selection threshold is a scale-free relative NLL improvement;
        setting it above any achievable improvement makes constant variance win
        even on genuinely heteroscedastic data — proving the knob is honored.
        """
        from hifi_anova.data.synthetic import generate_ishigami
        X, y, _ = generate_ishigami(n_samples=1500, heteroscedastic=True,
                                    variance_variable=2, seed=3)
        base = {'stages': ['A', 'B', 'D'], 'K1': 5, 'K2': 3, 'Kh': 3,
                'strategy': 'curvature'}
        # Default margin -> heteroscedastic kept.
        m_def, r_def, _, _ = self._fit(X, y, dict(base))
        assert m_def.variance_model is not None
        assert r_def['stage_D'].get('selected') == 'heteroscedastic'
        # Impossibly-high margin -> reverts to constant variance.
        m_hi, r_hi, _, _ = self._fit(X, y, dict(base, variance_selection_margin=10.0))
        assert m_hi.variance_model is None
        assert r_hi['stage_D'].get('selected') == 'homoscedastic'


class TestStageDLeverageCorrection:
    """DEC-028: the alternating 1/σ² loop is stabilized at the source.

    Root cause of the old instability: the variance solve consumed raw
    in-sample squared residuals, but E[r_n²] ≈ σ_n²(1 − lev_n) under the
    fitted mean. With a rich mean basis the variance model underestimates
    σ² exactly where the mean fits tightly, the 1/σ² weights blow up, the
    weighted mean interpolates the low-σ region harder, and the loop spirals
    (which DEC-027's guards then catch *after the fact*). DEC-028 feeds the
    Newton solve r²/clip(1−lev, 1e-3, 1) (as joint_lambda._joint_fit always
    did), applies the same correction to the constant-variance baseline of
    the model-selection comparison, and keeps the best outer iterate by
    held-out NLL instead of trusting the train-NLL convergence point.
    """

    # The golden `ishigami_hetero_ABD` configuration: rich mean basis under
    # strategy='variance' on genuinely heteroscedastic data — the scenario
    # that made the raw loop spiral (σ² dynamic range 10² → 10⁷).
    BASE = {'stages': ['A', 'B', 'D'], 'K1': 10, 'K2': 5, 'Kh': 3,
            'strategy': 'variance', 'lambda_order1': 1e-3,
            'lambda_order2': 1e-2, 'lambda_h': 0.1, 'max_outer_iter': 8}

    def _fit(self, cfg):
        from hifi_anova.data.preprocessing import preprocess_data
        from hifi_anova.data.synthetic import generate_ishigami
        X, y, _ = generate_ishigami(n_samples=2500, heteroscedastic=True,
                                    seed=0)
        data = preprocess_data(X, y, seed=0)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            model, res = HiFiANOVATrainer(cfg).fit(
                data['x_train'], data['y_train'],
                data['x_val'], data['y_val'])
        return model, res, [str(x.message) for x in w]

    @pytest.mark.integration
    def test_variance_strategy_rich_basis_is_stable(self):
        """The previously-spiraling scenario now keeps a clean hetero fit."""
        model, res, msgs = self._fit(dict(self.BASE))
        d = res['stage_D']
        assert d['selected'] == 'heteroscedastic'
        assert model.variance_model is not None
        # The loop must not degrade the mean it was given (it used to blow it
        # up until the guard reverted); a small tolerance covers reweighting.
        assert d['rmse_val'] <= 1.05 * res['stage_B']['rmse_val']
        assert d['nll_heteroscedastic'] < d['nll_homoscedastic']

    @pytest.mark.integration
    def test_optout_reproduces_raw_loop(self):
        """Both knobs off -> the raw loop spirals and the guard reverts.

        Locks the counterfactual DEC-028 fixes: if this starts *passing* the
        fit without a revert, the raw loop's behavior changed underneath us.
        """
        model, res, msgs = self._fit(dict(
            self.BASE, leverage_correction=False,
            alternating_early_stop=False))
        d = res['stage_D']
        assert d['selected'] == 'homoscedastic'
        assert model.variance_model is None
        assert any('Stage D' in m for m in msgs), msgs

    @pytest.mark.integration
    def test_early_stop_records_trajectory(self):
        """The held-out NLL trajectory and best iterate are reported."""
        model, res, msgs = self._fit(dict(self.BASE))
        d = res['stage_D']
        traj = d['val_nll_trajectory']
        assert traj is not None and len(traj) == d['n_outer_iterations']
        assert d['best_outer_iteration'] is not None
        best = d['best_outer_iteration'] - 1          # 1-based -> index
        assert traj[best] == pytest.approx(min(traj))
        # the kept iterate's NLL is what the model reports
        assert d['nll_val'] == pytest.approx(traj[best], abs=1e-6)


class TestVarianceResidualProjectionShape:
    """Regression: K2h>0 (or K3h>0) combined with a variance residual.

    The training-time projection orthogonalizes the variance-residual features
    against the FULL variance design [psi1|psi2|psi3], so proj_coeffs has
    F1h+F2h+F3h rows. model.predict and compute_sobol_indices used to rebuild
    only psi1 for the new-data projection -> shape-mismatch crash the first
    time the combination was ever exercised (2026-08-07 golden blind-spot
    audit). This pins the fixed, coherent projection.
    """

    @pytest.mark.integration
    def test_predict_and_sobol_with_K2h_and_variance_residual(self):
        from hifi_anova.data.synthetic import generate_ishigami
        from hifi_anova.data.preprocessing import preprocess_data

        Xh, yh, _ = generate_ishigami(n_samples=1500, heteroscedastic=True,
                                      seed=0)
        data = preprocess_data(Xh, yh, seed=0)
        cfg = {'stages': ['A', 'B', 'D'], 'K1': 6, 'K2': 3, 'Kh': 3,
               'K2h': 2, 'strategy': 'variance', 'lambda_h': 0.1,
               'max_outer_iter': 4,
               'variance_residual': {'type': 'rbf', 'n_centers': 40,
                                     'sigma': 0.3}}
        model, res = HiFiANOVATrainer(cfg).fit(
            data['x_train'], data['y_train'], data['x_val'], data['y_val'])
        # The projection path under test only exists when a variance model with a
        # variance-residual block is actually fitted, so that is a precondition of
        # this regression — assert it rather than a pytest.skip that would silently
        # stop exercising the projection if the guard's decision ever drifted
        # (X5 §7.1/§9). The Ishigami het signal keeps under the DEC-039 guard here;
        # if it ever reverts, strengthen the fixture — do not re-add a skip.
        vm = model.variance_model
        assert vm is not None and getattr(vm, 'has_variance_residual', False), (
            "fixture must fit a Stage-D variance model with a variance residual "
            f"(vm={vm is not None}, has_variance_residual="
            f"{getattr(vm, 'has_variance_residual', False)}); the K2h+residual "
            "projection under test requires it.")

        # predict: used to crash with dot_general shape mismatch
        mean, var = model.predict(data['x_test'])
        assert np.all(np.isfinite(np.asarray(mean)))
        assert np.all(np.isfinite(np.asarray(var)))
        assert np.all(np.asarray(var) > 0)

        # compute_sobol_indices: same crash in the var-residual measure term
        sob = compute_sobol_indices(model, data['x_test'])
        vs = sob['log_variance_sobol']
        assert np.isfinite(vs['residual'])
        acc = vs['variance_accounting']
        assert acc['residual'] >= 0.0
        assert np.isfinite(acc['total'])
