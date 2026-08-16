"""Tests for linear residual models (RBF, RFF, Nystrom) and feature-level projection."""

import pytest
import jax
import jax.numpy as jnp
import numpy as np

from hifi_anova.core.features import build_first_order_features, build_second_order_features
from hifi_anova.core.gram import build_gram_matrix, build_gram_matrix_2d
from hifi_anova.core.projection import project_features_orthogonal, verify_orthogonality
from hifi_anova.core.pairs import PairManager
from hifi_anova.model.linear_residual import RBFResidual, RFFResidual, NystromResidual
from hifi_anova.training.ridge import weighted_ridge_solve

pytestmark = pytest.mark.integration


# =============================================================================
# Feature-level projection tests
# =============================================================================

class TestFeatureLevelProjection:
    """Test that project_features_orthogonal guarantees exact orthogonality."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.N = 500
        self.D = 5
        self.K1 = 3
        self.K2 = 2
        key = jax.random.PRNGKey(42)
        self.x = jax.random.uniform(key, (self.N, self.D))

        # Build Fourier features
        self.phi1 = build_first_order_features(self.x, self.K1)
        pair_mgr = PairManager(self.D)
        self.phi2 = build_second_order_features(self.x, self.K2, pair_mgr.pair_indices)
        self.Phi = jnp.concatenate([self.phi1, self.phi2], axis=1)

        # Build some RBF features
        key2 = jax.random.PRNGKey(99)
        idx = jax.random.choice(key2, self.N, (50,), replace=False)
        self.centers = self.x[idx]
        sigma = 0.2
        diffs = self.x[:, None, :] - self.centers[None, :, :]
        dists_sq = jnp.sum(diffs ** 2, axis=-1)
        self.Z = jnp.exp(-dists_sq / (2.0 * sigma ** 2))

    def test_exact_orthogonality(self):
        """Phi^T @ Z_proj must be near zero (float64 precision)."""
        Z_proj, C = project_features_orthogonal(self.Z, self.Phi)
        result = verify_orthogonality(Z_proj, self.Phi, atol=1e-5)
        assert result['is_orthogonal'], f"max_cross = {result['max_cross']}"
        assert result['max_cross'] < 1e-5

    def test_projection_shape(self):
        """Z_proj must have same shape as Z."""
        Z_proj, C = project_features_orthogonal(self.Z, self.Phi)
        assert Z_proj.shape == self.Z.shape
        assert C.shape == (self.Phi.shape[1], self.Z.shape[1])

    def test_decoupled_ridge_solve(self):
        """Ridge on [Phi|Z_proj] must give same Fourier w as ridge on Phi alone."""
        Z_proj, C = project_features_orthogonal(self.Z, self.Phi)

        # Target
        key = jax.random.PRNGKey(7)
        y = jax.random.normal(key, (self.N,))

        F = self.Phi.shape[1]
        M = Z_proj.shape[1]
        lam_fourier = 0.01
        lam_res = 1.0

        # Solve Fourier only
        reg_f = jnp.full(F, lam_fourier)
        w_fourier_only = weighted_ridge_solve(self.Phi, y, reg_f)

        # Solve joint [Phi | Z_proj]
        Phi_joint = jnp.concatenate([self.Phi, Z_proj], axis=1)
        reg_joint = jnp.concatenate([jnp.full(F, lam_fourier), jnp.full(M, lam_res)])
        w_joint = weighted_ridge_solve(Phi_joint, y, reg_joint)

        # Fourier part of joint solve should match Fourier-only solve
        w_fourier_from_joint = w_joint[:F]
        assert jnp.allclose(w_fourier_only, w_fourier_from_joint, atol=1e-4), \
            f"max diff = {float(jnp.max(jnp.abs(w_fourier_only - w_fourier_from_joint)))}"

    def test_new_data_projection(self):
        """Projection applied to new data must also be approximately orthogonal."""
        Z_proj, C = project_features_orthogonal(self.Z, self.Phi)

        # New data
        key = jax.random.PRNGKey(123)
        x_new = jax.random.uniform(key, (100, self.D))
        phi1_new = build_first_order_features(x_new, self.K1)
        pair_mgr = PairManager(self.D)
        phi2_new = build_second_order_features(x_new, self.K2, pair_mgr.pair_indices)
        Phi_new = jnp.concatenate([phi1_new, phi2_new], axis=1)

        # RBF features for new data
        diffs = x_new[:, None, :] - self.centers[None, :, :]
        dists_sq = jnp.sum(diffs ** 2, axis=-1)
        Z_new = jnp.exp(-dists_sq / (2.0 * 0.2 ** 2))

        # Apply stored projection (use float64 for precision)
        Phi_new_64 = jnp.asarray(Phi_new, dtype=jnp.float64)
        Z_new_64 = jnp.asarray(Z_new, dtype=jnp.float64)
        Z_new_proj = Z_new_64 - Phi_new_64 @ C

        # Note: new-data orthogonality is approximate because
        # the projection was computed on training data. With enough
        # training data the empirical covariance converges, making
        # new-data cross-correlation small.
        result = verify_orthogonality(Z_new_proj, Phi_new)
        assert result['max_cross'] < 5.0, \
            f"New-data cross too large: {result['max_cross']}"


# =============================================================================
# RBF Residual tests
# =============================================================================

class TestRBFResidual:
    """Test RBFResidual creation, features, and vmap interface."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.N = 300
        self.D = 5
        key = jax.random.PRNGKey(42)
        self.x = jax.random.uniform(key, (self.N, self.D))

    def test_create_random(self):
        """RBFResidual.create with random centers."""
        rbf = RBFResidual.create(self.x, n_centers=50, sigma=0.2,
                                  method='random', key=jax.random.PRNGKey(0))
        assert rbf.centers.shape == (50, self.D)
        assert rbf.sigma == 0.2
        assert rbf.weights.shape == (50,)

    def test_create_kmeans(self):
        """RBFResidual.create with k-means centers."""
        rbf = RBFResidual.create(self.x, n_centers=30, sigma=0.15,
                                  method='kmeans')
        assert rbf.centers.shape == (30, self.D)

    def test_build_features_shape(self):
        """Feature matrix shape must be (N, M)."""
        rbf = RBFResidual.create(self.x, n_centers=40, sigma=0.2,
                                  method='random', key=jax.random.PRNGKey(1))
        Z = rbf.build_features(self.x)
        assert Z.shape == (self.N, 40)

    def test_features_in_01(self):
        """RBF features must be in (0, 1]."""
        rbf = RBFResidual.create(self.x, n_centers=40, sigma=0.2,
                                  method='random', key=jax.random.PRNGKey(2))
        Z = rbf.build_features(self.x)
        assert float(jnp.min(Z)) > 0
        assert float(jnp.max(Z)) <= 1.0 + 1e-10

    def test_vmap_interface(self):
        """jax.vmap(rbf)(x) must produce (N,) output."""
        rbf = RBFResidual.create(self.x, n_centers=40, sigma=0.2,
                                  method='random', key=jax.random.PRNGKey(3))
        # Set some weights to make output nonzero
        rbf = RBFResidual(
            weights=jnp.ones(40) * 0.1,
            proj_coeffs=jnp.array([]),
            centers=rbf.centers,
            sigma=rbf.sigma,
        )
        out = jax.vmap(rbf)(self.x)
        assert out.shape == (self.N,)

    def test_sobol_unchanged_with_residual(self):
        """Adding RBF residual must not change Fourier Sobol indices."""
        from hifi_anova.analysis.sobol import compute_sobol_indices
        from hifi_anova.model.mean_model import MeanModel
        from hifi_anova.model.hifi_anova import HiFiANOVA

        K1 = 3
        D = self.D
        G1 = build_gram_matrix(K1)

        # Create a model with known coefficients
        np.random.seed(42)
        w1 = jnp.array(np.random.randn(D * (2 * K1 + 1)) * 0.5, dtype=jnp.float32)
        mean_model = MeanModel(
            f0=jnp.array(0.0), w1=w1,
            w2=jnp.array([], dtype=jnp.float32),
            w3=jnp.array([], dtype=jnp.float32),
            K1=K1, K2=0, K3=0, D=D,
        )

        # Model without residual
        model_no_res = HiFiANOVA(
            mean_model=mean_model,
            K1=K1, K2=0, K3=0, Kh=0, D=D,
        )
        sobol_no_res = compute_sobol_indices(model_no_res)

        # Model with RBF residual (should not change Fourier Sobol)
        rbf = RBFResidual.create(self.x, n_centers=30, sigma=0.2,
                                  method='random', key=jax.random.PRNGKey(5))
        # Give it some weights but empty proj_coeffs (no projection needed for test)
        rbf_fitted = RBFResidual(
            weights=jnp.ones(30) * 0.01,
            proj_coeffs=jnp.array([]),
            centers=rbf.centers,
            sigma=rbf.sigma,
        )
        model_with_res = HiFiANOVA(
            mean_model=mean_model,
            residual_net=rbf_fitted,
            K1=K1, K2=0, K3=0, Kh=0, D=D,
        )
        sobol_with_res = compute_sobol_indices(model_with_res, self.x)

        # Fourier Sobol indices should be very close
        # (they won't be exactly identical because total_var includes residual_var)
        # But the STRUCTURAL indices (w^T G w) are unchanged
        for i in range(D):
            s_no = sobol_no_res['variance_accounting']['per_variable_first_order'][i]
            s_with = sobol_with_res['variance_accounting']['per_variable_first_order'][i]
            assert abs(s_no - s_with) < 1e-6, \
                f"Var {i}: no_res={s_no}, with_res={s_with}"


# =============================================================================
# RFF Residual tests
# =============================================================================

class TestRFFResidual:
    """Test RFFResidual creation and features."""

    def test_create(self):
        rff = RFFResidual.create(D=5, n_features=100, gamma=3.0,
                                  key=jax.random.PRNGKey(0))
        assert rff.omega.shape == (100, 5)
        assert rff.bias.shape == (100,)

    def test_build_features_shape(self):
        rff = RFFResidual.create(D=5, n_features=100, gamma=3.0,
                                  key=jax.random.PRNGKey(0))
        x = jax.random.uniform(jax.random.PRNGKey(1), (200, 5))
        Z = rff.build_features(x)
        assert Z.shape == (200, 100)

    def test_vmap_interface(self):
        rff = RFFResidual.create(D=5, n_features=100, gamma=3.0,
                                  key=jax.random.PRNGKey(0))
        rff = RFFResidual(
            weights=jnp.ones(100) * 0.01,
            proj_coeffs=jnp.array([]),
            omega=rff.omega,
            bias=rff.bias,
            scale=rff.scale,
        )
        x = jax.random.uniform(jax.random.PRNGKey(1), (50, 5))
        out = jax.vmap(rff)(x)
        assert out.shape == (50,)


# =============================================================================
# Nystrom Residual tests
# =============================================================================

class TestNystromResidual:
    """Test NystromResidual with different kernels."""

    @pytest.fixture(autouse=True)
    def setup(self):
        key = jax.random.PRNGKey(42)
        self.x = jax.random.uniform(key, (200, 4))

    def test_create_rbf_kernel(self):
        nys = NystromResidual.create(self.x, n_inducing=30, lengthscale=0.2,
                                     kernel='rbf')
        assert nys.inducing_points.shape == (30, 4)
        assert nys.kernel_type == 'rbf'

    def test_create_matern52(self):
        nys = NystromResidual.create(self.x, n_inducing=30, lengthscale=0.2,
                                     kernel='matern52')
        Z = nys.build_features(self.x)
        assert Z.shape == (200, 30)

    def test_features_positive(self):
        """Kernel features for RBF must be positive."""
        nys = NystromResidual.create(self.x, n_inducing=30, lengthscale=0.2,
                                     kernel='rbf')
        Z = nys.build_features(self.x)
        assert float(jnp.min(Z)) > 0

    def test_posterior_variance(self):
        """GP posterior variance should be non-negative."""
        nys = NystromResidual.create(self.x, n_inducing=30, lengthscale=0.2,
                                     kernel='rbf')
        var = nys.posterior_variance(self.x[:10], lambda_res=0.1)
        assert var.shape == (10,)
        assert float(jnp.min(var)) >= -1e-6  # allow tiny numerical negativity

    def test_vmap_interface(self):
        nys = NystromResidual.create(self.x, n_inducing=30, lengthscale=0.2,
                                     kernel='rbf')
        nys = NystromResidual(
            weights=jnp.ones(30) * 0.01,
            proj_coeffs=jnp.array([]),
            inducing_points=nys.inducing_points,
            lengthscale=nys.lengthscale,
            kernel_type=nys.kernel_type,
            signal_variance=nys.signal_variance,
        )
        out = jax.vmap(nys)(self.x[:20])
        assert out.shape == (20,)
