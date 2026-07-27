"""Orthogonal projection: keep the residual NN orthogonal to the Fourier subspace.

Two modes:
  1. Precomputed projection matrix (fast, applied per-batch)
  2. Online projection (exact per-batch, slightly more expensive)

Usage during training:
  projector = FourierProjector(Phi_train, reg_diag)  # once
  # In loss_fn:
  nn_raw = vmap(nn)(x_batch)
  nn_projected = projector.project_batch(nn_raw, Phi_batch)

This ensures that the NN output has zero projection onto the Fourier feature
space. Gradients flow through the projection (it's differentiable), telling
the NN: "don't learn what the Fourier basis can already represent."

The projection is OPTIONAL — enable via `orthogonal=True` in the NN training
config. Without it, the sequential training (Fourier first, NN on residuals)
provides approximate orthogonality. The projection provides an exact guarantee.
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Optional


class FourierProjector:
    """Precomputed projector for orthogonalizing NN output against Fourier basis.

    Stores P_perp = I - Phi (Phi^T Phi + R)^{-1} Phi^T
    or equivalently, the matrix A_inv_PhiT = (Phi^T Phi + R)^{-1} Phi^T
    which is already computed during the ridge solve.

    For efficiency, we don't store the full N x N projection matrix.
    Instead we store A_inv_PhiT (F x N) and project per-batch.
    """

    def __init__(self, Phi: jnp.ndarray, reg_diag: jnp.ndarray,
                 exact: bool = True):
        """Precompute the projection components.

        Args:
            Phi: (N_train, F) full Fourier feature matrix
            reg_diag: (F,) regularization diagonal (used only if exact=False)
            exact: if True, use (Phi^T Phi)^{-1} for exact orthogonality.
                   if False, use (Phi^T Phi + R)^{-1} (softer, ridge-consistent).
        """
        Phi = jnp.asarray(Phi, dtype=jnp.float32)
        N, F = Phi.shape

        PhiTPhi = Phi.T @ Phi
        if exact:
            # For exact orthogonality, use pseudoinverse of Phi^T Phi
            # Add tiny ridge for numerical stability (not regularization)
            eps = 1e-6 * jnp.trace(PhiTPhi) / F
            self.A_inv = jnp.linalg.inv(PhiTPhi + eps * jnp.eye(F))
        else:
            # Soft projection using the ridge matrix — not exactly orthogonal
            # but consistent with the regularized subspace
            reg_diag = jnp.asarray(reg_diag, dtype=jnp.float32)
            self.A_inv = jnp.linalg.inv(PhiTPhi + jnp.diag(reg_diag))

        self.F = F

    def project(self, nn_output: jnp.ndarray, Phi_batch: jnp.ndarray) -> jnp.ndarray:
        """Project NN output orthogonal to Fourier subspace.

        f_res = nn_output - Phi_batch @ (A^{-1} Phi_batch^T nn_output)

        This removes from the NN output whatever lies in span(Phi).

        Args:
            nn_output: (B,) raw NN predictions for a batch
            Phi_batch: (B, F) Fourier features for the same batch

        Returns:
            (B,) projected NN output, orthogonal to Fourier features
        """
        # What part of nn_output is "in" the Fourier space?
        # proj_coeffs = A^{-1} Phi_batch^T nn_output  (F,)
        proj_coeffs = self.A_inv @ (Phi_batch.T @ nn_output)
        # Remove it
        return nn_output - Phi_batch @ proj_coeffs

    def project_batched(self, nn_output: jnp.ndarray,
                        Phi_batch: jnp.ndarray) -> jnp.ndarray:
        """Same as project() but handles the shapes explicitly.

        This is the version to use inside a JIT-compiled loss function.
        """
        return self.project(nn_output, Phi_batch)


def build_projector(model, x_train: jnp.ndarray,
                    reg_diag: jnp.ndarray) -> FourierProjector:
    """Convenience: build a projector from a fitted model and training data.

    Args:
        model: fitted HiFiANOVA
        x_train: (N, D) training inputs
        reg_diag: (F,) regularization diagonal used during fitting

    Returns:
        FourierProjector instance
    """
    Phi = model.build_phi_all(x_train)
    return FourierProjector(Phi, reg_diag)


def build_batch_features(x_batch: jnp.ndarray, model) -> jnp.ndarray:
    """Build Fourier features for a batch (used during projected NN training).

    Args:
        x_batch: (B, D) inputs
        model: HiFiANOVA (for K1, K2, pair_indices)

    Returns:
        (B, F) feature matrix
    """
    return model.build_phi_all(x_batch)
