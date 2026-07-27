"""HiFiANOVA: top-level model combining mean + variance + residual.

The residual can be:
  - None: Fourier-only model
  - eqx.nn.MLP: Neural network (SGD pipeline, nonlinear)
  - LinearResidualBase subclass: RBF/RFF/Nystrom (analytic pipeline, linear)
"""

import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Optional, Tuple

from ..core.features import (
    build_first_order_features,
    build_second_order_features,
    build_third_order_features,
    build_mixed_first_order_features,
    build_mixed_second_order_features,
)
from .mean_model import MeanModel
from .variance_model import VarianceModel


class HiFiANOVA(eqx.Module):
    """Top-level model combining all components.

    Configuration determines which components are active:
      - mean_model: always present (orders 1, 2, optionally 3)
      - variance_model: None for homoscedastic, VarianceModel for heteroscedastic
      - residual_net: None, eqx.nn.MLP, or LinearResidualBase subclass
    """
    # Required fields first (no defaults)
    mean_model: MeanModel

    # Shared static config (no defaults)
    K1: int = eqx.field(static=True)
    K2: int = eqx.field(static=True)
    Kh: int = eqx.field(static=True)
    D: int = eqx.field(static=True)
    pair_indices: Optional[jnp.ndarray] = eqx.field(static=True, default=None)
    G1: Optional[jnp.ndarray] = eqx.field(static=True, default=None)
    G2: Optional[jnp.ndarray] = eqx.field(static=True, default=None)

    # Third-order support
    K3: int = eqx.field(static=True, default=0)
    triple_indices: Optional[jnp.ndarray] = eqx.field(static=True, default=None)
    G3: Optional[jnp.ndarray] = eqx.field(static=True, default=None)

    # Basis configuration — mean model
    include_linear_1: bool = eqx.field(static=True, default=True)
    include_linear_2: bool = eqx.field(static=True, default=True)
    include_linear_3: bool = eqx.field(static=True, default=True)
    # Basis configuration — variance model (defaults follow mean model)
    include_linear_h1: bool = eqx.field(static=True, default=True)
    include_linear_h2: bool = eqx.field(static=True, default=True)
    include_linear_h3: bool = eqx.field(static=True, default=True)
    basis_name: str = eqx.field(static=True, default='fourier')

    # Mixed per-variable basis (None = uniform, backward compatible)
    var_specs: tuple = eqx.field(static=True, default=None)
    pair_block_info: tuple = eqx.field(static=True, default=None)

    # Optional fields (with defaults)
    variance_model: Optional[VarianceModel] = None
    residual_net: Optional[eqx.Module] = None  # MLP or LinearResidualBase
    constant_log_var: Optional[jax.Array] = None

    @property
    def has_linear_residual(self) -> bool:
        """True if residual is a linear-in-parameters model (RBF/RFF/Nystrom)."""
        return (self.residual_net is not None and
                hasattr(self.residual_net, 'is_linear') and
                self.residual_net.is_linear)

    @property
    def has_nn_residual(self) -> bool:
        """True if residual is a neural network."""
        return self.residual_net is not None and not self.has_linear_residual

    def predict(self, x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Full prediction for batch x (N, D).

        Returns: (mean (N,), variance (N,))
        """
        phi1 = self.build_phi1(x)
        phi2 = self.build_phi2(x)
        phi3 = self.build_phi3(x)

        # Mean prediction
        mean = self.mean_model.predict(phi1, phi2, phi3)
        if self.residual_net is not None:
            res_out = jax.vmap(self.residual_net)(x)
            if res_out.ndim > 1:
                res_out = res_out.squeeze(-1)
            mean = mean + res_out

        # Variance prediction
        if self.variance_model is not None:
            vm = self.variance_model
            psi1 = self.build_psi1(x)

            # Second-order variance features (if present)
            psi2 = None
            K2h = getattr(vm, 'K2h', 0)
            pair_idx_h = getattr(vm, 'pair_indices_h', None)
            if K2h > 0 and pair_idx_h is not None:
                psi2 = build_second_order_features(x, K2h, pair_idx_h,
                                                    include_linear=self.include_linear_h2,
                                                    basis_name=self.basis_name)

            # Third-order variance features (if present)
            psi3 = None
            K3h = getattr(vm, 'K3h', 0)
            triple_idx_h = getattr(vm, 'triple_indices_h', None)
            if K3h > 0 and triple_idx_h is not None:
                psi3 = build_third_order_features(x, K3h, triple_idx_h,
                                                    include_linear=self.include_linear_h3,
                                                    basis_name=self.basis_name)

            # Variance residual features (if present)
            z_h_proj = None
            if (hasattr(vm, 'variance_residual') and
                    vm.variance_residual is not None):
                var_res = vm.variance_residual
                z_h = var_res.build_features(x)
                if var_res.proj_coeffs.ndim >= 2 and var_res.proj_coeffs.shape[0] > 0:
                    z_h_proj = z_h - psi1 @ var_res.proj_coeffs
                else:
                    z_h_proj = z_h

            variance = vm.predict_variance(psi1, psi2, psi3, z_h_proj)
        elif self.constant_log_var is not None:
            variance = jnp.full(x.shape[0], jnp.exp(self.constant_log_var))
        else:
            variance = jnp.ones(x.shape[0])

        return mean, variance

    def predict_mean_only(self, x: jnp.ndarray) -> jnp.ndarray:
        """Mean prediction only. For RMSE evaluation."""
        mean, _ = self.predict(x)
        return mean

    # --- Feature building helpers ---
    # These encapsulate all basis config (K, include_linear, basis_name)
    # so callers don't need to pass flags manually.

    @property
    def is_mixed(self) -> bool:
        """True if using mixed per-variable basis."""
        return self.var_specs is not None

    def _var_specs_as_dicts(self):
        """Convert var_specs tuple-of-tuples to list-of-dicts for feature builders."""
        return [{'basis': bn, 'K': K} for bn, K, _, _, _ in self.var_specs]

    def build_phi1(self, x: jnp.ndarray) -> jnp.ndarray:
        """Build first-order mean features with correct flags."""
        if self.var_specs is not None:
            phi, _ = build_mixed_first_order_features(x, self._var_specs_as_dicts())
            return phi
        return build_first_order_features(
            x, self.K1, include_linear=self.include_linear_1,
            basis_name=self.basis_name)

    def build_phi2(self, x: jnp.ndarray) -> Optional[jnp.ndarray]:
        """Build second-order mean features, or None if K2=0."""
        if self.var_specs is not None and self.pair_indices is not None:
            phi2, _ = build_mixed_second_order_features(
                x, self.pair_indices, self._var_specs_as_dicts())
            return phi2 if phi2.shape[1] > 0 else None
        if self.K2 > 0 and self.pair_indices is not None:
            return build_second_order_features(
                x, self.K2, self.pair_indices,
                include_linear=self.include_linear_2,
                basis_name=self.basis_name)
        return None

    def build_phi3(self, x: jnp.ndarray) -> Optional[jnp.ndarray]:
        """Build third-order mean features, or None if K3=0."""
        if self.K3 > 0 and self.triple_indices is not None:
            return build_third_order_features(
                x, self.K3, self.triple_indices,
                include_linear=self.include_linear_3,
                basis_name=self.basis_name)
        return None

    def build_phi_all(self, x: jnp.ndarray) -> jnp.ndarray:
        """Build concatenated mean features [phi1 | phi2 | phi3]."""
        parts = [self.build_phi1(x)]
        phi2 = self.build_phi2(x)
        if phi2 is not None:
            parts.append(phi2)
        phi3 = self.build_phi3(x)
        if phi3 is not None:
            parts.append(phi3)
        return jnp.concatenate(parts, axis=1) if len(parts) > 1 else parts[0]

    def build_psi1(self, x: jnp.ndarray) -> jnp.ndarray:
        """Build first-order variance features with correct flags."""
        return build_first_order_features(
            x, self.Kh, include_linear=self.include_linear_h1,
            basis_name=self.basis_name)
