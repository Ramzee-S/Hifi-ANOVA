"""VarianceModel: Fourier model for log sigma^2(x).

h(x) = h0 + w1^T psi1(x) [+ w2^T psi2(x)] [+ w3^T psi3(x)] [+ w_res^T z_h(x)]

Mirrors the MeanModel structure for the log-variance:
- First-order Fourier (always present)
- Second-order Fourier (optional, for variance interactions)
- Third-order Fourier (optional, for small D with complex noise)
- Linear residual (optional, RBF/RFF/GP for higher-order variance structure)

Each component has its own regularization (λ_h1, λ_h2, λ_h3, λ_h_res).
Variance Sobol indices are computed per order, paralleling mean Sobol.
"""

import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Optional


# Shared bound on the log-variance h = log(sigma^2). Applied IDENTICALLY at fit
# time (the Newton solver in training/newton.py) and at prediction time
# (predict_variance below) so the fitted objective and the model that emits
# predictions never diverge at extreme h. exp(30) ~ 1e13 already spans an
# enormous dynamic range and is safe from overflow in float64.
LOG_VAR_CLIP = 30.0


class VarianceModel(eqx.Module):
    """Fourier model for log-variance with optional second/third-order and residual.

    h(x) = h0 + psi1 @ w1 [+ psi2 @ w2] [+ psi3 @ w3] [+ z_h_proj @ w_res]
    sigma^2(x) = exp(h(x))
    """
    h0: jax.Array                   # scalar (log of baseline variance)
    w1: jax.Array                   # (D*B_h,) first-order variance coefficients
    Kh: int = eqx.field(static=True)
    D: int = eqx.field(static=True)
    # Second-order variance (optional)
    w2: jax.Array = eqx.field(default_factory=lambda: jnp.array([], dtype=jnp.float32))
    K2h: int = eqx.field(static=True, default=0)
    pair_indices_h: Optional[jnp.ndarray] = eqx.field(static=True, default=None)
    # Third-order variance (optional)
    w3: jax.Array = eqx.field(default_factory=lambda: jnp.array([], dtype=jnp.float32))
    K3h: int = eqx.field(static=True, default=0)
    triple_indices_h: Optional[jnp.ndarray] = eqx.field(static=True, default=None)
    # Optional RBF/RFF residual for higher-order variance
    w_var_residual: Optional[jax.Array] = None
    variance_residual: Optional[eqx.Module] = None
    # Basis configuration (needed for correct block size computation)
    basis_name: str = eqx.field(static=True, default='fourier')
    include_linear_h1: bool = eqx.field(static=True, default=True)
    include_linear_h2: bool = eqx.field(static=True, default=True)
    include_linear_h3: bool = eqx.field(static=True, default=True)

    def predict_log_variance(self, psi1: jnp.ndarray,
                              psi2: Optional[jnp.ndarray] = None,
                              psi3: Optional[jnp.ndarray] = None,
                              z_h_proj: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        """Predict h(x) = log sigma^2(x).

        Args:
            psi1: (N, D*(2Kh+1)) first-order variance features
            psi2: (N, Ph*(2K2h+1)^2) second-order variance features (optional)
            psi3: (N, Th*(2K3h+1)^3) third-order variance features (optional)
            z_h_proj: (N, M_h) projected variance residual features (optional)

        Returns: (N,) log-variance values
        """
        h = self.h0 + psi1 @ self.w1
        if psi2 is not None and self.K2h > 0 and len(self.w2) > 0:
            h = h + psi2 @ self.w2
        if psi3 is not None and self.K3h > 0 and len(self.w3) > 0:
            h = h + psi3 @ self.w3
        if z_h_proj is not None and self.w_var_residual is not None:
            h = h + z_h_proj @ self.w_var_residual
        return h

    def predict_variance(self, psi1: jnp.ndarray,
                          psi2: Optional[jnp.ndarray] = None,
                          psi3: Optional[jnp.ndarray] = None,
                          z_h_proj: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        """Predict sigma^2(x) = exp(h(x)). Returns (N,).

        Log-variance is clipped to [-LOG_VAR_CLIP, LOG_VAR_CLIP], the same bound
        used by the Newton fitting solver, so fit and prediction stay consistent.
        """
        h = self.predict_log_variance(psi1, psi2, psi3, z_h_proj)
        return jnp.exp(jnp.clip(h, -LOG_VAR_CLIP, LOG_VAR_CLIP))

    def _block_h1(self) -> int:
        from ..core.features import basis_size
        return basis_size(self.Kh, self.include_linear_h1, self.basis_name)

    def _block_h2(self) -> int:
        from ..core.features import basis_size
        return basis_size(self.K2h, self.include_linear_h2, self.basis_name) ** 2

    def _block_h3(self) -> int:
        from ..core.features import basis_size
        return basis_size(self.K3h, self.include_linear_h3, self.basis_name) ** 3

    def get_coefficients_for_variable(self, i: int) -> jnp.ndarray:
        """Slice of w1 for variable i. Shape (B_h,)."""
        block = self._block_h1()
        return self.w1[i * block: (i + 1) * block]

    def get_coefficients_for_pair(self, p: int) -> jnp.ndarray:
        """Slice of w2 for variance pair p. Shape (B_h2²,)."""
        block = self._block_h2()
        return self.w2[p * block: (p + 1) * block]

    def get_coefficients_for_triple(self, t: int) -> jnp.ndarray:
        """Slice of w3 for variance triple t. Shape (B_h3³,)."""
        block = self._block_h3()
        return self.w3[t * block: (t + 1) * block]

    @property
    def has_second_order(self) -> bool:
        return self.K2h > 0 and len(self.w2) > 0

    @property
    def has_third_order(self) -> bool:
        return self.K3h > 0 and len(self.w3) > 0

    @property
    def has_variance_residual(self) -> bool:
        return self.w_var_residual is not None and self.variance_residual is not None
