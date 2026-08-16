"""Tests for the QMC uniform-measure layer (advisor item #2)."""

import numpy as np
import pytest

from hifi_anova.analysis.qmc import sobol_cube_sample, qmc_uniform_variance

pytestmark = pytest.mark.smoke


class TestSobolCubeSample:
    def test_shape_is_power_of_two_and_in_cube(self):
        pts = sobol_cube_sample(D=4, n_points=1000, seed=0)
        # 1000 -> rounded up to 2**10 = 1024
        assert pts.shape == (1024, 4)
        assert pts.min() >= 0.0 and pts.max() <= 1.0

    def test_deterministic_given_seed(self):
        a = sobol_cube_sample(3, 1 << 12, seed=7)
        b = sobol_cube_sample(3, 1 << 12, seed=7)
        assert np.array_equal(a, b)

    def test_seed_changes_sample(self):
        a = sobol_cube_sample(3, 1 << 12, seed=1)
        b = sobol_cube_sample(3, 1 << 12, seed=2)
        assert not np.array_equal(a, b)


class TestQMCUniformVariance:
    def test_matches_analytic_variance_linear(self):
        """Var_U[0,1](x_0) = 1/12; QMC should nail it to ~1e-3."""
        v = qmc_uniform_variance(lambda X: X[:, 0], D=3, n_points=1 << 16, seed=0)
        assert abs(v - 1.0 / 12.0) < 1e-3

    def test_matches_analytic_variance_sum(self):
        """Var_U(sum_i x_i) = D/12 for independent uniforms."""
        D = 5
        v = qmc_uniform_variance(lambda X: X.sum(axis=1), D=D,
                                 n_points=1 << 16, seed=0)
        assert abs(v - D / 12.0) < 5e-3

    def test_reuses_supplied_points(self):
        pts = sobol_cube_sample(2, 1 << 14, seed=3)
        v1 = qmc_uniform_variance(lambda X: X[:, 0] * X[:, 1], D=2, points=pts)
        v2 = qmc_uniform_variance(lambda X: X[:, 0] * X[:, 1], D=2,
                                  n_points=1 << 14, seed=3)
        assert np.isclose(v1, v2)

    def test_squeezes_trailing_dim(self):
        """A model residual returning (M, 1) must be handled like (M,)."""
        v = qmc_uniform_variance(lambda X: X[:, :1], D=2, n_points=1 << 12, seed=0)
        assert abs(v - 1.0 / 12.0) < 5e-3


class TestResidualMeasureInSobol:
    """compute_sobol_indices residual measure (advisor item #2)."""

    def _model_with_residual(self):
        import jax
        import jax.numpy as jnp
        from hifi_anova.core.gram import build_gram_matrix
        from hifi_anova.model.linear_residual import RBFResidual
        from hifi_anova.model.mean_model import MeanModel
        from hifi_anova.model.hifi_anova import HiFiANOVA

        K1, D = 3, 3
        G1 = build_gram_matrix(K1)
        rng = np.random.RandomState(0)
        w1 = jnp.array(rng.randn(D * (2 * K1 + 1)) * 0.5, dtype=jnp.float32)
        mean_model = MeanModel(
            f0=jnp.array(0.0), w1=w1,
            w2=jnp.array([], dtype=jnp.float32),
            w3=jnp.array([], dtype=jnp.float32),
            K1=K1, K2=0, K3=0, D=D)
        centers = jnp.asarray(rng.uniform(0, 1, (25, D)), dtype=jnp.float32)
        rbf = RBFResidual(weights=jnp.ones(25) * 0.05,
                          proj_coeffs=jnp.array([]), centers=centers, sigma=0.2)
        model = HiFiANOVA(mean_model=mean_model, residual_net=rbf,
                          K1=K1, K2=0, K3=0, Kh=0, D=D)
        return model, D

    def test_qmc_residual_is_invariant_to_x_data(self):
        """The QMC residual is Var over the uniform cube, so it must NOT depend
        on which x_data is passed — unlike the empirical (data-measure) variant."""
        from hifi_anova.analysis.sobol import compute_sobol_indices
        model, D = self._model_with_residual()
        rng = np.random.RandomState(1)
        # two *different*, non-uniform input clouds
        xa = np.clip(rng.normal(0.3, 0.1, (400, D)), 0, 1).astype(np.float32)
        xb = np.clip(rng.normal(0.7, 0.1, (400, D)), 0, 1).astype(np.float32)

        ra = compute_sobol_indices(model, xa)['variance_accounting']['residual']
        rb = compute_sobol_indices(model, xb)['variance_accounting']['residual']
        assert np.isclose(ra, rb, rtol=1e-9), "QMC residual must not depend on x_data"

        ea = compute_sobol_indices(model, xa, residual_measure='empirical'
                                   )['variance_accounting']['residual']
        eb = compute_sobol_indices(model, xb, residual_measure='empirical'
                                   )['variance_accounting']['residual']
        assert not np.isclose(ea, eb, rtol=1e-3), \
            "empirical residual should differ across different input clouds"

    def test_residual_measure_label(self):
        from hifi_anova.analysis.sobol import compute_sobol_indices
        model, D = self._model_with_residual()
        x = np.random.RandomState(2).uniform(0, 1, (200, D)).astype(np.float32)
        va = compute_sobol_indices(model, x)['variance_accounting']
        assert va['residual_measure'] == 'qmc'
        va_e = compute_sobol_indices(model, x, residual_measure='empirical'
                                     )['variance_accounting']
        assert va_e['residual_measure'] == 'empirical'

    def test_no_residual_reports_none_measure(self):
        from hifi_anova.analysis.sobol import compute_sobol_indices
        import jax.numpy as jnp
        from hifi_anova.core.gram import build_gram_matrix
        from hifi_anova.model.mean_model import MeanModel
        from hifi_anova.model.hifi_anova import HiFiANOVA
        K1, D = 3, 3
        G1 = build_gram_matrix(K1)
        mm = MeanModel(f0=jnp.array(0.0),
                       w1=jnp.ones(D * (2 * K1 + 1), dtype=jnp.float32) * 0.1,
                       w2=jnp.array([], dtype=jnp.float32),
                       w3=jnp.array([], dtype=jnp.float32),
                       K1=K1, K2=0, K3=0, D=D)
        model = HiFiANOVA(mean_model=mm, K1=K1, K2=0, K3=0, Kh=0, D=D)
        va = compute_sobol_indices(model)['variance_accounting']
        assert va['residual_measure'] == 'none'
        assert va['residual'] == 0.0
