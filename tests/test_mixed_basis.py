"""Tests for mixed per-variable basis fitting.

Covers:
  - Feature construction: mixed block sizes, correct shapes
  - Gram matrices: per-variable, mixed G_i ⊗ G_j for pairs
  - Regularization: mixed penalty vector sizes
  - Sobol: sum-to-one with mixed bases
  - Trainer: full pipeline with mixed config
  - MeanModel: mixed get_coefficients_for_variable
"""

import pytest
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update('jax_enable_x64', True)

pytestmark = pytest.mark.integration


# ─────────────────────────────────────────────────────────────
# Test mixed feature construction
# ─────────────────────────────────────────────────────────────

class TestMixedFeatures:

    def test_first_order_shape(self):
        """Mixed features have correct total size."""
        from hifi_anova.core.features import build_mixed_first_order_features, basis_size
        N, D = 100, 3
        x = jnp.tile(jnp.linspace(0.01, 0.99, N)[:, None], (1, D))
        specs = [
            {'basis': 'legendre', 'K': 5},   # 5 features
            {'basis': 'fourier', 'K': 3},     # 6 features (no linear)
            {'basis': 'haar', 'K': 3},        # 7 features
        ]
        phi, info = build_mixed_first_order_features(x, specs)
        assert phi.shape == (N, 5 + 6 + 7)
        assert len(info) == 3
        # Check block info
        assert info[0] == ('legendre', 5, True, 5, 0)
        assert info[1] == ('fourier', 3, False, 6, 5)
        assert info[2] == ('haar', 3, False, 7, 11)

    def test_fourier_has_no_linear(self):
        """In mixed mode, Fourier basis excludes linear term."""
        from hifi_anova.core.features import build_mixed_first_order_features
        N = 100
        x = jnp.tile(jnp.linspace(0.01, 0.99, N)[:, None], (1, 2))
        specs = [
            {'basis': 'fourier', 'K': 2},   # 4 features (cos1,sin1,cos2,sin2)
            {'basis': 'legendre', 'K': 2},   # 2 features (P1,P2)
        ]
        phi, info = build_mixed_first_order_features(x, specs)
        assert phi.shape == (N, 4 + 2)
        # Fourier block: 4 features, no linear
        assert info[0][3] == 4  # block_size
        assert info[0][2] is False  # include_linear

    def test_second_order_mixed_shapes(self):
        """Mixed second-order features have variable block sizes."""
        from hifi_anova.core.features import build_mixed_second_order_features
        N, D = 50, 3
        x = jnp.tile(jnp.linspace(0.01, 0.99, N)[:, None], (1, D))
        specs = [
            {'basis': 'legendre', 'K': 2},   # B=2
            {'basis': 'fourier', 'K': 2},     # B=4
            {'basis': 'haar', 'K': 2},        # B=3
        ]
        pairs = jnp.array([[0, 1], [0, 2], [1, 2]])
        phi2, pair_info = build_mixed_second_order_features(x, pairs, specs)
        # Pair (0,1): 2*4=8, Pair (0,2): 2*3=6, Pair (1,2): 4*3=12
        assert phi2.shape == (N, 8 + 6 + 12)
        assert pair_info[0][4] == 8   # block_size for pair (0,1)
        assert pair_info[1][4] == 6   # block_size for pair (0,2)
        assert pair_info[2][4] == 12  # block_size for pair (1,2)

    def test_zero_mean_mixed(self):
        """Mixed features have zero mean on uniform data."""
        from hifi_anova.core.features import build_mixed_first_order_features
        N = 100000
        D = 3
        key = jax.random.PRNGKey(42)
        x = jax.random.uniform(key, (N, D))
        specs = [
            {'basis': 'legendre', 'K': 3},
            {'basis': 'fourier', 'K': 3},
            {'basis': 'haar', 'K': 3},
        ]
        phi, _ = build_mixed_first_order_features(x, specs)
        means = jnp.mean(phi, axis=0)
        assert jnp.max(jnp.abs(means)) < 0.02


# ─────────────────────────────────────────────────────────────
# Test mixed regularization
# ─────────────────────────────────────────────────────────────

class TestMixedRegularization:

    def test_first_order_only(self):
        """Mixed reg vector has correct size for first-order."""
        from hifi_anova.training.regularization import build_mixed_regularization_vector
        specs = [
            {'basis': 'legendre', 'K': 3},   # 3
            {'basis': 'fourier', 'K': 2},     # 4
            {'basis': 'haar', 'K': 2},        # 3
        ]
        reg = build_mixed_regularization_vector(specs, strategy='uniform',
                                                 lambda_order1=0.1)
        assert reg.shape == (3 + 4 + 3,)
        assert jnp.allclose(reg, 0.1)

    def test_with_pairs(self):
        """Mixed reg vector includes per-pair second-order blocks."""
        from hifi_anova.training.regularization import build_mixed_regularization_vector
        specs = [
            {'basis': 'legendre', 'K': 2},   # B=2
            {'basis': 'fourier', 'K': 2},     # B=4
        ]
        pairs = jnp.array([[0, 1]])
        reg = build_mixed_regularization_vector(
            specs, strategy='uniform', lambda_order1=0.1,
            pair_indices=pairs, lambda_order2=0.5)
        # First-order: 2 + 4 = 6. Second-order: 2*4 = 8. Total: 14
        assert reg.shape == (6 + 8,)
        assert jnp.allclose(reg[:6], 0.1)
        # Second-order: additive penalty r(a,b) = r_i[a] + r_j[b] = 0.5 + 0.5 = 1.0
        assert jnp.all(reg[6:] >= 0.5)  # at least lambda_order2

    def test_variance_strategy_haar_equals_uniform(self):
        """Variance strategy with Haar gives uniform (G=I)."""
        from hifi_anova.training.regularization import build_mixed_regularization_vector
        specs = [{'basis': 'haar', 'K': 3}]
        reg_u = build_mixed_regularization_vector(specs, strategy='uniform',
                                                    lambda_order1=0.1)
        reg_v = build_mixed_regularization_vector(specs, strategy='variance',
                                                    lambda_order1=0.1)
        assert jnp.allclose(reg_u, reg_v)


# ─────────────────────────────────────────────────────────────
# Test MeanModel with mixed basis
# ─────────────────────────────────────────────────────────────

class TestMixedMeanModel:

    def test_get_coefficients(self):
        """get_coefficients_for_variable uses correct offsets in mixed mode."""
        from hifi_anova.model.mean_model import MeanModel
        var_specs = (
            ('legendre', 3, True, 3, 0),
            ('fourier', 2, False, 4, 3),
            ('haar', 2, False, 3, 7),
        )
        w1 = jnp.arange(10, dtype=jnp.float32)
        model = MeanModel(
            f0=jnp.array(0.0), w1=w1,
            w2=jnp.array([]), K1=0, K2=0, D=3,
            basis_name='mixed', var_specs=var_specs,
        )
        assert jnp.array_equal(model.get_coefficients_for_variable(0),
                               jnp.array([0, 1, 2], dtype=jnp.float32))
        assert jnp.array_equal(model.get_coefficients_for_variable(1),
                               jnp.array([3, 4, 5, 6], dtype=jnp.float32))
        assert jnp.array_equal(model.get_coefficients_for_variable(2),
                               jnp.array([7, 8, 9], dtype=jnp.float32))

    def test_get_var_gram(self):
        """get_var_gram returns correct Gram for each variable's basis."""
        from hifi_anova.model.mean_model import MeanModel
        var_specs = (
            ('legendre', 2, True, 2, 0),
            ('haar', 2, False, 3, 2),
        )
        model = MeanModel(
            f0=jnp.array(0.0), w1=jnp.zeros(5),
            w2=jnp.array([]), K1=0, K2=0, D=2,
            basis_name='mixed', var_specs=var_specs,
        )
        G0 = model.get_var_gram(0)
        assert G0.shape == (2, 2)  # Legendre K=2
        assert jnp.isclose(G0[0, 0], 1.0 / 3.0, atol=1e-10)  # 1/(2*1+1)

        G1 = model.get_var_gram(1)
        assert G1.shape == (3, 3)  # Haar K=2 → 3 features
        assert jnp.allclose(G1, jnp.eye(3))  # Haar Gram = Identity


# ─────────────────────────────────────────────────────────────
# Test Sobol with mixed basis
# ─────────────────────────────────────────────────────────────

class TestMixedSobol:

    def test_sobol_sum_to_one(self):
        """First-order Sobol sums to 1 with mixed bases."""
        from hifi_anova.model.mean_model import MeanModel
        from hifi_anova.model.hifi_anova import HiFiANOVA
        from hifi_anova.analysis.sobol import compute_sobol_indices
        from hifi_anova.core.gram import build_gram_matrix

        var_specs = (
            ('legendre', 2, True, 2, 0),   # G = diag(1/3, 1/5)
            ('haar', 2, False, 3, 2),       # G = I_3
        )
        # Coefficients
        w1 = jnp.array([1.0, 0.5,  # legendre var 0
                         0.3, 0.2, 0.1], dtype=jnp.float32)  # haar var 1

        G_leg = build_gram_matrix(2, basis_name='legendre')

        mean_model = MeanModel(
            f0=jnp.array(0.0), w1=w1,
            w2=jnp.array([]), K1=0, K2=0, D=2,
            basis_name='mixed', var_specs=var_specs,
        )
        model = HiFiANOVA(
            mean_model=mean_model,
            K1=0, K2=0, K3=0, Kh=0, D=2,
            G1=np.array(G_leg),
            basis_name='mixed', var_specs=var_specs,
        )

        sobol = compute_sobol_indices(model)
        s = sobol['mean_sobol']['first_order']
        total = sum(s.values())
        assert abs(total - 1.0) < 1e-6, f"Sobol sum = {total}"


# ─────────────────────────────────────────────────────────────
# Test trainer with mixed config
# ─────────────────────────────────────────────────────────────

class TestMixedTrainer:

    def test_first_order_mixed_fit(self):
        """Trainer fits with mixed basis and produces valid model."""
        from hifi_anova.training.trainer import HiFiANOVATrainer

        N = 2000
        np.random.seed(42)
        x = np.random.uniform(0, 1, (N, 4))
        # Polynomial in x0, step in x1
        y = 3.0 * (x[:, 0] - 0.5) ** 2 + 2.0 * np.where(x[:, 1] < 0.5, 1, -1)
        y += 0.1 * np.random.randn(N)
        x, y = jnp.array(x), jnp.array(y)

        n_val = 400
        config = {
            'stages': ['A'],
            'strategy': 'uniform',
            'lambda_order1': 0.001,
            'basis_per_variable': {
                0: {'basis': 'legendre', 'K': 5},
                1: {'basis': 'haar', 'K': 4},
                2: {'basis': 'legendre', 'K': 2},
                3: {'basis': 'legendre', 'K': 2},
            },
        }
        trainer = HiFiANOVATrainer(config)
        model, results = trainer.fit(x[n_val:], y[n_val:], x[:n_val], y[:n_val])

        assert model.is_mixed
        assert results['mixed_basis']
        assert results['stage_A']['rmse_val'] < 1.0

    def test_first_plus_second_order_mixed(self):
        """Trainer fits first + second order with mixed basis."""
        from hifi_anova.training.trainer import HiFiANOVATrainer

        N = 2000
        np.random.seed(42)
        x = np.random.uniform(0, 1, (N, 3))
        # Interaction between polynomial x0 and step x1
        y = (3.0 * (x[:, 0] - 0.5)
             + 2.0 * np.where(x[:, 1] < 0.5, 1, -1)
             + 1.0 * (x[:, 0] - 0.5) * np.where(x[:, 1] < 0.5, 1, -1))
        y += 0.1 * np.random.randn(N)
        x, y = jnp.array(x), jnp.array(y)

        n_val = 400
        config = {
            'stages': ['A', 'B'],
            'K2': 3,
            'strategy': 'uniform',
            'lambda_order1': 0.001,
            'lambda_order2': 0.01,
            'basis_per_variable': {
                0: {'basis': 'legendre', 'K': 3},
                1: {'basis': 'haar', 'K': 3},
                2: {'basis': 'legendre', 'K': 2},
            },
        }
        trainer = HiFiANOVATrainer(config)
        model, results = trainer.fit(x[n_val:], y[n_val:], x[:n_val], y[:n_val])

        assert 'stage_B' in results
        # Second-order should improve fit
        assert results['stage_B']['rmse_val'] <= results['stage_A']['rmse_val'] + 0.1

    def test_sobol_from_mixed_fit(self):
        """Sobol indices from mixed model sum to ~1."""
        from hifi_anova.training.trainer import HiFiANOVATrainer
        from hifi_anova.analysis.sobol import compute_sobol_indices

        N = 3000
        np.random.seed(42)
        x = np.random.uniform(0, 1, (N, 3))
        y = 4.0 * (x[:, 0] - 0.5) + 2.0 * np.where(x[:, 1] < 0.5, 1, -1)
        y += 0.1 * np.random.randn(N)
        x, y = jnp.array(x), jnp.array(y)

        n_val = 500
        config = {
            'stages': ['A'],
            'strategy': 'uniform',
            'lambda_order1': 0.0001,
            'basis_per_variable': {
                0: {'basis': 'legendre', 'K': 3},
                1: {'basis': 'haar', 'K': 4},
                2: {'basis': 'legendre', 'K': 2},
            },
        }
        trainer = HiFiANOVATrainer(config)
        model, results = trainer.fit(x[n_val:], y[n_val:], x[:n_val], y[:n_val])

        sobol = compute_sobol_indices(model)
        s = sobol['mean_sobol']['first_order']
        total = sum(s.values())
        assert abs(total - 1.0) < 0.05, f"Sobol sum = {total}"
        # x0 (linear, amplitude=4) and x1 (step, amplitude=2) should be dominant
        # x0 variance ≈ 4²/12 ≈ 1.33, x1 variance ≈ 2²×1 = 4 → x1 > x0
        assert s[0] > 0.15, f"S0 = {s[0]}"
        assert s[1] > 0.3, f"S1 = {s[1]}"
        assert s[2] < 0.1

    def test_predict_works(self):
        """Model prediction works after mixed fit."""
        from hifi_anova.training.trainer import HiFiANOVATrainer

        N = 1000
        np.random.seed(42)
        x = np.random.uniform(0, 1, (N, 3))
        y = x[:, 0] + 0.1 * np.random.randn(N)
        x, y = jnp.array(x), jnp.array(y)

        config = {
            'stages': ['A'],
            'strategy': 'uniform',
            'lambda_order1': 0.001,
            'basis_per_variable': {
                0: {'basis': 'legendre', 'K': 3},
                1: {'basis': 'fourier', 'K': 2},
                2: {'basis': 'haar', 'K': 2},
            },
        }
        trainer = HiFiANOVATrainer(config)
        model, _ = trainer.fit(x[200:], y[200:], x[:200], y[:200])

        # Predict on new data
        x_new = jnp.array(np.random.uniform(0, 1, (50, 3)))
        mean, var = model.predict(x_new)
        assert mean.shape == (50,)
        assert var.shape == (50,)
        assert jnp.all(jnp.isfinite(mean))
