"""HiFiANOVA: top-level model combining mean + variance + residual.

The residual can be:
  - None: Fourier-only model
  - eqx.nn.MLP: Neural network (SGD pipeline, nonlinear)
  - LinearResidualBase subclass: RBF/RFF/Nystrom (analytic pipeline, linear)
"""

import jax
from ..array_backend import xp as jnp  # switchable array backend (numpy exact core)
import equinox as eqx
from typing import Optional, Tuple

from ..core.features import (
    build_first_order_features,
    build_second_order_features,
    build_third_order_features,
    build_mean_phi1,
    build_mean_phi2,
    build_mean_design,
)
from ..core.gram import (build_gram_matrix, build_gram_matrix_2d,
                         build_gram_matrix_3d)
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
    # Index arrays are ordinary (dynamic) integer pytree leaves, not static
    # fields — an array in a static field triggers equinox's "JAX array set as
    # static" warning and, worse, lands the array in the pytree *metadata* where
    # optax rejects it during joint fine-tuning. Being integer leaves they are
    # excluded from the float-parameter optimizer filter (``is_inexact_array``),
    # so they are carried but never trained.
    pair_indices: Optional[jnp.ndarray] = None

    # Third-order support
    K3: int = eqx.field(static=True, default=0)
    triple_indices: Optional[jnp.ndarray] = None

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
    # Per-pair second-order harmonic order (uniform basis family): tuple of P
    # ints aligned with pair_indices; None = one shared K2 (backward
    # compatible). Mirrors MeanModel.pair_k2; K2 holds max(pair_k2).
    pair_k2: tuple = eqx.field(static=True, default=None)

    # Order-selective first-order membership (X11C-S03 / BR-06): the ascending
    # TRUE variable indices whose first-order block was in the SOLVED mean
    # design; None = all D. The model keeps the full uniform ``w1`` (exact
    # zeros in excluded blocks) so ``predict`` / Sobol slicing are unchanged —
    # this field records the fitted design's layout ONLY so the epistemic
    # posterior can rebuild a column-consistent design (see build_phi_all_fit).
    # Mirrors the variance model's ``variance_variables``. Older pickles lack
    # it → readers use ``getattr(model, 'fo_included', None)``.
    fo_included: tuple = eqx.field(static=True, default=None)

    # Effective Stage-D mean-estimator convention (DEC-039/DEC-047 provenance):
    # carried ON the model so a bare ``save_model(model, path)`` (no results dict)
    # still persists the right vintage. Static (not a leaf), defaults to the
    # ordinary unit-weight centered mean; the trainer sets the weighted vintage on
    # a heteroscedastic fit. Older pickled/serialised models lack it → readers use
    # ``getattr(model, 'mean_intercept_mode', None)``.
    # Literal 'unweighted_centered' rather than importing the constant: the
    # constants live in ``training.fitted_design`` and importing it here would
    # create a model↔training import cycle (redecompose imports HiFiANOVA). The
    # trainer sets the weighted vintage via one of those same constants, so this
    # default is the single homoscedastic value; keep it in sync with
    # ``fitted_design.MEAN_INTERCEPT_UNWEIGHTED`` (pinned by a test).
    mean_intercept_mode: str = eqx.field(static=True, default='unweighted_centered')

    # Optional fields (with defaults)
    variance_model: Optional[VarianceModel] = None
    residual_net: Optional[eqx.Module] = None  # MLP or LinearResidualBase
    constant_log_var: Optional[jax.Array] = None

    # Gram matrices are pure functions of the (static) basis config, so they are
    # exposed as cached-free properties rather than stored array fields — storing
    # them was both redundant and a source of the "JAX array set as static"
    # warning. They are None in mixed-basis mode (each variable has its own Gram,
    # accessed via the mean model). Kept as read-only properties so every existing
    # ``model.G1`` / ``model.G2`` / ``model.G3`` read is unchanged.
    @property
    def G1(self) -> Optional[jnp.ndarray]:
        if self.var_specs is not None:
            return None
        return build_gram_matrix(self.K1, self.include_linear_1, self.basis_name)

    @property
    def G2(self) -> Optional[jnp.ndarray]:
        # No single shared pair Gram in mixed or per-pair-K2 mode: use
        # mean_model.get_pair_gram(p) per pair instead.
        if (self.var_specs is not None or self.K2 <= 0
                or getattr(self, 'pair_k2', None) is not None):
            return None
        return build_gram_matrix_2d(
            build_gram_matrix(self.K2, self.include_linear_2, self.basis_name))

    @property
    def G3(self) -> Optional[jnp.ndarray]:
        if self.var_specs is not None or self.K3 <= 0:
            return None
        return build_gram_matrix_3d(
            build_gram_matrix(self.K3, self.include_linear_3, self.basis_name))

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
            from .linear_residual import predict_residual_batch
            mean = mean + predict_residual_batch(self.residual_net, x)

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
                    # proj_coeffs was fitted against the FULL variance design
                    # [psi1|psi2|psi3] (trainer Stage D); psi1 alone mismatches
                    # whenever K2h/K3h > 0.
                    psi_fourier = psi1
                    if psi2 is not None:
                        psi_fourier = jnp.concatenate([psi_fourier, psi2], axis=1)
                    if psi3 is not None:
                        psi_fourier = jnp.concatenate([psi_fourier, psi3], axis=1)
                    z_h_proj = z_h - psi_fourier @ var_res.proj_coeffs
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

    def build_phi1(self, x: jnp.ndarray) -> jnp.ndarray:
        """Build first-order mean features with correct flags.

        Always the FULL layout (all D variables — excluded blocks carry exact
        zero coefficients on a BR-06 fit); the solved-subset layout lives in
        :meth:`build_phi_all_fit`. Delegates to the shared layout builder
        (``core.features.build_mean_phi1``) — the residual projector rebuilds
        through the same code path (BR-11)."""
        return build_mean_phi1(
            x, self.K1, include_linear=self.include_linear_1,
            basis_name=self.basis_name, var_specs=self.var_specs)

    def build_phi2(self, x: jnp.ndarray) -> Optional[jnp.ndarray]:
        """Build second-order mean features, or None if K2=0. Delegates to
        the shared layout builder (BR-11)."""
        return build_mean_phi2(
            x, self.K2, self.pair_indices,
            include_linear=self.include_linear_2,
            basis_name=self.basis_name, var_specs=self.var_specs,
            pair_k2=getattr(self, 'pair_k2', None))

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

    def build_phi_all_fit(self, x: jnp.ndarray) -> jnp.ndarray:
        """Mean design in the FITTED-DESIGN layout (matches ``record.Phi``).

        Identical to :meth:`build_phi_all` except that, for an order-selective
        first-order fit (``fo_included`` set, BR-06), the first-order block
        spans only the included variables — the layout the model was actually
        solved on and the one ``record.Phi`` stores. The epistemic posterior
        needs ``Phi_new`` column-consistent with ``Phi_train`` (=record.Phi);
        the uniform ``build_phi_all`` used for the mean prediction would carry
        the excluded variables' (zero-coefficient) columns and mismatch it.
        ``fo_included is None`` ⇒ byte-identical to ``build_phi_all``.
        """
        fo = getattr(self, 'fo_included', None)
        if fo is None:
            return self.build_phi_all(x)
        return build_mean_design(
            x, K1=self.K1, K2=self.K2, K3=self.K3,
            pair_indices=self.pair_indices,
            triple_indices=self.triple_indices,
            include_linear_1=self.include_linear_1,
            include_linear_2=self.include_linear_2,
            include_linear_3=self.include_linear_3,
            basis_name=self.basis_name, var_specs=self.var_specs,
            pair_k2=getattr(self, 'pair_k2', None), fo_included=fo)

    def build_psi1(self, x: jnp.ndarray) -> jnp.ndarray:
        """Build first-order variance features with correct flags.

        Honors the fitted variance model's variable subset
        (``variance_variables``): only the included variables' blocks are
        built, matching the layout of ``variance_model.w1``.
        """
        vv = (getattr(self.variance_model, 'variance_variables', None)
              if self.variance_model is not None else None)
        if vv is not None:
            from ..core.features import build_first_order_features_subset
            return build_first_order_features_subset(
                x, self.Kh, list(vv), include_linear=self.include_linear_h1,
                basis_name=self.basis_name)
        return build_first_order_features(
            x, self.Kh, include_linear=self.include_linear_h1,
            basis_name=self.basis_name)
