"""Linear-in-parameters residual models: RBF, RFF, Nystrom/GP.

All models are eqx.Module subclasses supporting jax.vmap(residual)(x),
making them drop-in replacements for the NN residual in HiFiANOVA.predict().

Key property: features can be projected orthogonal to the Fourier basis
BEFORE fitting, guaranteeing exact decoupling from Fourier terms.
This means Sobol indices are guaranteed clean regardless of what the
residual captures.

Usage:
    # Create unfitted residual (choose centers/frequencies)
    residual = RBFResidual.create(x_train, n_centers=300, sigma=0.2)

    # Build features, project, fit via ridge (done by analytic_residual.py)
    Z = residual.build_features(x_train)
    Z_proj, C = project_features_orthogonal(Z, Phi)
    alpha = ridge_solve(Z_proj, residuals, reg)

    # Create fitted residual
    fitted = residual.with_fit(alpha, C, fourier_config)
"""

import jax
import jax.numpy as jnp
import equinox as eqx
import numpy as np
from typing import Optional, Tuple
from functools import partial

from ..core.features import (
    build_first_order_features,
    build_second_order_features,
    build_third_order_features,
    build_per_variable_basis,
)


# =============================================================================
# Base class
# =============================================================================

class LinearResidualBase(eqx.Module):
    """Base class for linear-in-parameters residual models.

    All subclasses must implement:
      - build_features(x_batch) -> (N, M)
      - _features_single(x_single) -> (M,)
      - n_features: int (static)

    The fitted model stores:
      - weights: (M,) fitted coefficients alpha
      - proj_coeffs: (F, M) projection matrix for new-data orthogonalization
      - Fourier config (K1, K2, K3, pair/triple indices) for building
        Fourier features at prediction time
    """
    # Fitted parameters
    weights: jax.Array                  # (M,) or empty if unfitted
    proj_coeffs: jax.Array              # (F, M) or empty if unfitted

    # Fourier config for prediction-time projection
    K1: int = eqx.field(static=True, default=0)
    K2: int = eqx.field(static=True, default=0)
    K3: int = eqx.field(static=True, default=0)
    D: int = eqx.field(static=True, default=0)
    pair_indices: Optional[jnp.ndarray] = eqx.field(static=True, default=None)
    triple_indices: Optional[jnp.ndarray] = eqx.field(static=True, default=None)
    include_linear_1: bool = eqx.field(static=True, default=True)
    include_linear_2: bool = eqx.field(static=True, default=True)
    include_linear_3: bool = eqx.field(static=True, default=True)
    basis_name: str = eqx.field(static=True, default='fourier')

    # Type flag for pipeline dispatch
    is_linear: bool = eqx.field(static=True, default=True)

    def __call__(self, x_single: jnp.ndarray) -> jnp.ndarray:
        """Forward pass for a single input (D,). Returns scalar.

        Used via jax.vmap(residual)(x_batch) in HiFiANOVA.predict().
        Applies the stored projection to ensure orthogonality to Fourier.
        """
        z = self._features_single(x_single)  # (M,)

        # Build Fourier features for this input and apply projection
        # proj_coeffs is (F, M) when fitted, or empty when unfitted
        if self.proj_coeffs.ndim >= 2 and self.proj_coeffs.shape[0] > 0:
            phi = self._fourier_single(x_single)  # (F,)
            z_proj = z - phi @ self.proj_coeffs    # (M,)
        else:
            z_proj = z

        return jnp.dot(z_proj, self.weights)

    def _fourier_single(self, x_single: jnp.ndarray) -> jnp.ndarray:
        """Build Fourier feature vector for a single input.

        Reconstructs the same Fourier features used during training
        so the projection is consistent.
        """
        x_2d = x_single[None, :]  # (1, D)

        phi1 = build_first_order_features(
            x_2d, self.K1,
            include_linear=self.include_linear_1,
            basis_name=self.basis_name)  # (1, F1)
        parts = [phi1]

        if self.K2 > 0 and self.pair_indices is not None:
            phi2 = build_second_order_features(
                x_2d, self.K2, self.pair_indices,
                include_linear=self.include_linear_2,
                basis_name=self.basis_name)
            parts.append(phi2)

        if self.K3 > 0 and self.triple_indices is not None:
            phi3 = build_third_order_features(
                x_2d, self.K3, self.triple_indices,
                include_linear=self.include_linear_3,
                basis_name=self.basis_name)
            parts.append(phi3)

        return jnp.concatenate(parts, axis=1).squeeze(0)  # (F,)

    def build_features(self, x_batch: jnp.ndarray) -> jnp.ndarray:
        """Build feature matrix for a batch. Shape (N, M).
        Must be implemented by subclasses."""
        raise NotImplementedError

    def _features_single(self, x_single: jnp.ndarray) -> jnp.ndarray:
        """Build feature vector for a single input. Shape (M,).
        Must be implemented by subclasses."""
        raise NotImplementedError


# =============================================================================
# RBF Residual
# =============================================================================

class RBFResidual(LinearResidualBase):
    """Radial Basis Function residual.

    f_res(x) = sum_j alpha_j * exp(-||x - c_j||^2 / (2*sigma^2))

    The width sigma controls smoothness:
      Large sigma (0.3): captures only broad, smooth interactions
      Small sigma (0.05): captures localized patterns (risk of overfitting)

    Centers are FIXED (from training data), not learned.
    This keeps the model linear in alpha.
    """
    centers: jax.Array = eqx.field(default_factory=lambda: jnp.zeros((0, 1)))
    sigma: float = eqx.field(static=True, default=0.2)

    @classmethod
    def create(cls, x_train: jnp.ndarray, n_centers: int = 300,
               sigma: float = 0.2, method: str = 'kmeans',
               key: Optional[jax.Array] = None) -> 'RBFResidual':
        """Create an unfitted RBF residual with centers selected from data.

        Args:
            x_train: (N, D) training inputs
            n_centers: number of RBF centers (M)
            sigma: Gaussian width parameter
            method: 'kmeans' or 'random' center selection
            key: PRNG key for random selection
        """
        N, D = x_train.shape
        n_centers = min(n_centers, N)

        if method == 'kmeans':
            centers = _kmeans_centers(np.array(x_train), n_centers)
        else:
            if key is None:
                key = jax.random.PRNGKey(42)
            idx = jax.random.choice(key, N, (n_centers,), replace=False)
            centers = x_train[idx]

        return cls(
            weights=jnp.zeros(n_centers),
            proj_coeffs=jnp.array([]),
            centers=jnp.array(centers),
            sigma=sigma,
        )

    def build_features(self, x_batch: jnp.ndarray) -> jnp.ndarray:
        """Build RBF feature matrix. Shape (N, M).

        Z[n, j] = exp(-||x_n - c_j||^2 / (2*sigma^2))
        """
        # (N, 1, D) - (1, M, D) -> (N, M, D) -> sum -> (N, M)
        diffs = x_batch[:, None, :] - self.centers[None, :, :]
        dists_sq = jnp.sum(diffs ** 2, axis=-1)
        return jnp.exp(-dists_sq / (2.0 * self.sigma ** 2))

    def _features_single(self, x_single: jnp.ndarray) -> jnp.ndarray:
        """RBF features for a single input. Shape (M,)."""
        diffs = x_single - self.centers  # (M, D)
        dists_sq = jnp.sum(diffs ** 2, axis=-1)  # (M,)
        return jnp.exp(-dists_sq / (2.0 * self.sigma ** 2))


# =============================================================================
# RFF Residual
# =============================================================================

class RFFResidual(LinearResidualBase):
    """Random Fourier Features residual.

    z_j(x) = sqrt(2/M) * cos(omega_j^T x + b_j)

    Approximates an RBF kernel in the spectral domain.
    The frequency scale gamma controls smoothness:
      Small gamma (1-3): smooth functions (approx. sigma = 1/(gamma*sqrt(2)))
      Large gamma (10+): high-frequency (localized patterns)

    omega_j ~ N(0, gamma^2 I), b_j ~ U(0, 2*pi).
    Both are FIXED after initialization — model is linear in alpha.
    """
    omega: jax.Array = eqx.field(default_factory=lambda: jnp.zeros((0, 1)))
    bias: jax.Array = eqx.field(default_factory=lambda: jnp.zeros(0))
    scale: float = eqx.field(static=True, default=1.0)

    @classmethod
    def create(cls, D: int, n_features: int = 1000,
               gamma: float = 3.0,
               key: Optional[jax.Array] = None) -> 'RFFResidual':
        """Create an unfitted RFF residual with random frequencies.

        Args:
            D: input dimension
            n_features: number of random features (M)
            gamma: frequency scale (small = smooth)
            key: PRNG key
        """
        if key is None:
            key = jax.random.PRNGKey(42)

        key1, key2 = jax.random.split(key)
        omega = jax.random.normal(key1, (n_features, D)) * gamma
        bias = jax.random.uniform(key2, (n_features,)) * 2.0 * jnp.pi
        scale = jnp.sqrt(2.0 / n_features)

        return cls(
            weights=jnp.zeros(n_features),
            proj_coeffs=jnp.array([]),
            omega=omega,
            bias=bias,
            scale=float(scale),
        )

    def build_features(self, x_batch: jnp.ndarray) -> jnp.ndarray:
        """Build RFF feature matrix. Shape (N, M).

        Z[n, j] = sqrt(2/M) * cos(omega_j^T x_n + b_j)
        """
        proj = x_batch @ self.omega.T + self.bias  # (N, M)
        return self.scale * jnp.cos(proj)

    def _features_single(self, x_single: jnp.ndarray) -> jnp.ndarray:
        """RFF features for a single input. Shape (M,)."""
        proj = x_single @ self.omega.T + self.bias  # (M,)
        return self.scale * jnp.cos(proj)


# =============================================================================
# Nystrom / GP Residual
# =============================================================================

class NystromResidual(LinearResidualBase):
    """GP Nystrom approximation residual.

    f_res(x) = k(x, X_m) @ alpha

    Same math as RBF network with inducing points, but framed as
    a GP approximation. Provides posterior variance (Bayesian
    uncertainty) as a bonus.

    Supports kernels: 'rbf', 'matern32', 'matern52'.
    """
    inducing_points: jax.Array = eqx.field(default_factory=lambda: jnp.zeros((0, 1)))
    lengthscale: float = eqx.field(static=True, default=0.2)
    kernel_type: str = eqx.field(static=True, default='rbf')
    signal_variance: float = eqx.field(static=True, default=1.0)

    @classmethod
    def create(cls, x_train: jnp.ndarray, n_inducing: int = 300,
               lengthscale: float = 0.2, kernel: str = 'rbf',
               signal_variance: float = 1.0,
               key: Optional[jax.Array] = None) -> 'NystromResidual':
        """Create an unfitted Nystrom residual with inducing points.

        Args:
            x_train: (N, D) training inputs
            n_inducing: number of inducing points (M)
            lengthscale: kernel lengthscale
            kernel: 'rbf', 'matern32', or 'matern52'
            signal_variance: kernel signal variance
            key: PRNG key
        """
        N, D = x_train.shape
        n_inducing = min(n_inducing, N)

        # Use k-means for inducing point placement
        inducing = _kmeans_centers(np.array(x_train), n_inducing)

        return cls(
            weights=jnp.zeros(n_inducing),
            proj_coeffs=jnp.array([]),
            inducing_points=jnp.array(inducing),
            lengthscale=lengthscale,
            kernel_type=kernel,
            signal_variance=signal_variance,
        )

    def _kernel_fn(self, x1: jnp.ndarray, x2: jnp.ndarray) -> jnp.ndarray:
        """Compute kernel matrix between x1 (N1, D) and x2 (N2, D).
        Returns (N1, N2)."""
        # Squared distances
        diffs = x1[:, None, :] - x2[None, :, :]  # (N1, N2, D)
        r_sq = jnp.sum(diffs ** 2, axis=-1)  # (N1, N2)
        r = jnp.sqrt(jnp.maximum(r_sq, 1e-20))
        ls = self.lengthscale
        sv = self.signal_variance

        if self.kernel_type == 'rbf':
            return sv * jnp.exp(-r_sq / (2.0 * ls ** 2))
        elif self.kernel_type == 'matern32':
            scaled_r = jnp.sqrt(3.0) * r / ls
            return sv * (1.0 + scaled_r) * jnp.exp(-scaled_r)
        elif self.kernel_type == 'matern52':
            scaled_r = jnp.sqrt(5.0) * r / ls
            return sv * (1.0 + scaled_r + scaled_r ** 2 / 3.0) * jnp.exp(-scaled_r)
        else:
            raise ValueError(f"Unknown kernel: {self.kernel_type}")

    def _kernel_single(self, x_single: jnp.ndarray) -> jnp.ndarray:
        """Kernel vector between single input and inducing points. Shape (M,)."""
        return self._kernel_fn(x_single[None, :], self.inducing_points).squeeze(0)

    def build_features(self, x_batch: jnp.ndarray) -> jnp.ndarray:
        """Build kernel feature matrix. Shape (N, M).

        Z[n, j] = k(x_n, x_m_j)
        """
        return self._kernel_fn(x_batch, self.inducing_points)

    def _features_single(self, x_single: jnp.ndarray) -> jnp.ndarray:
        """Kernel features for a single input. Shape (M,)."""
        return self._kernel_single(x_single)

    def posterior_variance(self, x_new: jnp.ndarray,
                           lambda_res: float) -> jnp.ndarray:
        """GP posterior variance for uncertainty quantification.

        Var[f_res(x)] = k(x,x) - k(x,X_m) (K_mm + lambda*I)^{-1} k(X_m,x)

        Args:
            x_new: (N_new, D) new inputs
            lambda_res: regularization (noise variance proxy)

        Returns:
            (N_new,) posterior variance at each input
        """
        M = self.inducing_points.shape[0]
        K_xm = self._kernel_fn(x_new, self.inducing_points)  # (N_new, M)

        # K_mm + lambda*I
        K_mm = self._kernel_fn(self.inducing_points, self.inducing_points)
        K_mm_reg = K_mm + lambda_res * jnp.eye(M)

        # k(x,x) diagonal (for RBF = signal_variance)
        K_xx_diag = jnp.full(x_new.shape[0], self.signal_variance)

        # Solve (K_mm + lambda*I)^{-1} K_mx
        # K_xm @ solve(K_mm_reg, K_xm.T) gives N_new x N_new, too large
        # Instead: row-wise dot product
        solved = jnp.linalg.solve(K_mm_reg, K_xm.T)  # (M, N_new)
        reduction = jnp.sum(K_xm.T * solved, axis=0)  # (N_new,)

        return K_xx_diag - reduction


# =============================================================================
# Utility: k-means center selection
# =============================================================================

def _kmeans_centers(X: np.ndarray, n_centers: int,
                    max_iter: int = 100, seed: int = 42) -> np.ndarray:
    """Simple k-means for selecting RBF/Nystrom centers.

    Uses sklearn if available, falls back to a simple implementation.

    Args:
        X: (N, D) training data (numpy)
        n_centers: number of cluster centers
        max_iter: max k-means iterations
        seed: random seed

    Returns:
        (n_centers, D) cluster centers
    """
    try:
        from sklearn.cluster import MiniBatchKMeans
        km = MiniBatchKMeans(
            n_clusters=n_centers, max_iter=max_iter,
            random_state=seed, batch_size=min(1024, len(X)),
            n_init=3,
        )
        km.fit(X)
        return km.cluster_centers_
    except ImportError:
        # Fallback: random subset
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(X), n_centers, replace=False)
        return X[idx]
