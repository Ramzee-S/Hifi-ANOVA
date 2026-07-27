"""MeanModel: the Fourier (linear) part of the mean function.

f(x) = f0 + w1^T phi1(x) + w2^T phi2(x) [+ w3^T phi3(x)]

This is a LINEAR model. No activations, no hidden layers.
The residual (NN or linear) is NOT part of this class.
"""

import jax
import jax.numpy as jnp
import equinox as eqx

from ..core.features import basis_size


class MeanModel(eqx.Module):
    """First + second + (optional) third order Fourier mean model.

    Forward pass: f0 + phi1 @ w1 + phi2 @ w2 [+ phi3 @ w3]

    The include_linear_2/3 flags control whether the second/third-order
    basis includes the linear term (x-0.5) or is spectral-only (cos/sin).
    """
    f0: jax.Array                   # scalar intercept
    w1: jax.Array                   # (D*(2K1+1),) first-order coefficients
    w2: jax.Array                   # (P*B2^2,) second-order coefficients or empty
    K1: int = eqx.field(static=True)
    K2: int = eqx.field(static=True)   # 0 if no second order
    D: int = eqx.field(static=True)
    w3: jax.Array = eqx.field(default_factory=lambda: jnp.array([], dtype=jnp.float32))
    K3: int = eqx.field(static=True, default=0)  # 0 if no third order
    include_linear_1: bool = eqx.field(static=True, default=True)
    include_linear_2: bool = eqx.field(static=True, default=True)
    include_linear_3: bool = eqx.field(static=True, default=True)
    basis_name: str = eqx.field(static=True, default='fourier')
    # Mixed per-variable basis: tuple of (basis, K, include_linear, block_size, offset)
    # None = uniform basis (backward compatible). Set by trainer in mixed mode.
    var_specs: tuple = eqx.field(static=True, default=None)
    # Second-order mixed block info: tuple of (i, j, Bi, Bj, block_size, offset)
    pair_block_info: tuple = eqx.field(static=True, default=None)

    def predict(self, phi1: jnp.ndarray, phi2: jnp.ndarray = None,
                phi3: jnp.ndarray = None) -> jnp.ndarray:
        """Predict from precomputed features.

        Args:
            phi1: (N, D*(2K1+1)) first-order features
            phi2: (N, P*B2^2) second-order features or None
            phi3: (N, T*B3^3) third-order features or None

        Returns: (N,) predictions
        """
        out = self.f0 + phi1 @ self.w1
        if phi2 is not None and self.K2 > 0 and len(self.w2) > 0:
            out = out + phi2 @ self.w2
        if phi3 is not None and self.K3 > 0 and len(self.w3) > 0:
            out = out + phi3 @ self.w3
        return out

    @property
    def is_mixed(self) -> bool:
        """True if using mixed per-variable basis."""
        return self.var_specs is not None

    def get_coefficients_for_variable(self, i: int) -> jnp.ndarray:
        """Slice of w1 for variable i."""
        if self.var_specs is not None:
            _, _, _, block, offset = self.var_specs[i]
            return self.w1[offset: offset + block]
        block = basis_size(self.K1, self.include_linear_1, self.basis_name)
        return self.w1[i * block: (i + 1) * block]

    def get_var_gram(self, i: int) -> jnp.ndarray:
        """Get the Gram matrix for variable i (mixed or uniform)."""
        from ..core.gram import build_gram_matrix
        if self.var_specs is not None:
            bn, K, il, _, _ = self.var_specs[i]
            return build_gram_matrix(K, include_linear=il, basis_name=bn)
        return build_gram_matrix(self.K1, self.include_linear_1, self.basis_name)

    def get_coefficients_for_pair(self, p: int) -> jnp.ndarray:
        """Slice of w2 for pair p."""
        if self.pair_block_info is not None:
            _, _, _, _, block, offset = self.pair_block_info[p]
            return self.w2[offset: offset + block]
        block = basis_size(self.K2, self.include_linear_2, self.basis_name) ** 2
        return self.w2[p * block: (p + 1) * block]

    def get_pair_gram(self, p: int) -> jnp.ndarray:
        """Get the Gram matrix for pair p (mixed: G_i ⊗ G_j; uniform: G₁ ⊗ G₁)."""
        from ..core.gram import build_gram_matrix, build_gram_matrix_2d
        if self.pair_block_info is not None:
            vi, vj, Bi, Bj, _, _ = self.pair_block_info[p]
            Gi = self.get_var_gram(vi)
            Gj = self.get_var_gram(vj)
            return jnp.kron(Gi, Gj)
        return build_gram_matrix_2d(
            build_gram_matrix(self.K2, self.include_linear_2, self.basis_name))

    def get_coefficients_for_triple(self, t: int) -> jnp.ndarray:
        """Slice of w3 for triple t."""
        block = basis_size(self.K3, self.include_linear_3, self.basis_name) ** 3
        return self.w3[t * block: (t + 1) * block]
