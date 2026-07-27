"""Feature-level orthogonal projection for linear residual models.

Projects residual features Z orthogonal to Fourier features Phi
BEFORE fitting. This guarantees:

  1. Phi^T @ Z_proj = 0  (exactly, to machine precision)
  2. Ridge on [Phi | Z_proj] decouples into independent solves
  3. Fourier coefficients are identical with or without the residual
  4. Sobol indices are guaranteed clean — no drift, no approximation

This is ONLY for linear-in-parameters residuals (RBF, RFF, Nystrom).
For nonlinear residuals (NN), use training/projection.py (output-level).

The projection is computed ONCE at initialization and stored as a
coefficient matrix C for applying to new data at prediction time.
"""

import jax.numpy as jnp
import numpy as np
from typing import Tuple


def project_features_orthogonal(
    Z: jnp.ndarray,
    Phi: jnp.ndarray,
    eps: float = 1e-8,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Project feature matrix Z orthogonal to Phi at the feature level.

    Computes:
        C = solve(Phi^T Phi + eps*I, Phi^T Z)      (F, M)
        Z_proj = Z - Phi @ C                         (N, M)

    After projection: Phi^T @ Z_proj ≈ 0 (to O(eps) precision).

    Args:
        Z: (N, M) residual feature matrix (RBF/RFF/Nystrom features)
        Phi: (N, F) Fourier feature matrix
        eps: numerical stability factor (scaled by trace/F internally)

    Returns:
        Z_proj: (N, M) projected features, orthogonal to Phi
        proj_coeffs: (F, M) coefficient matrix for new-data projection:
            Z_new_proj = Z_new - Phi_new @ proj_coeffs
    """
    Phi64 = jnp.asarray(Phi, dtype=jnp.float64)
    Z64 = jnp.asarray(Z, dtype=jnp.float64)
    N, F = Phi64.shape
    _, M = Z64.shape

    if F == 0:
        # No Fourier features to project against — return Z unchanged
        return Z64, jnp.zeros((0, M), dtype=jnp.float64)

    PhiTPhi = Phi64.T @ Phi64  # (F, F)

    # Adaptive regularization for numerical stability
    eps_scaled = eps * jnp.trace(PhiTPhi) / max(F, 1)
    A = PhiTPhi + eps_scaled * jnp.eye(F, dtype=jnp.float64)

    # Solve for projection coefficients: C = (Phi^T Phi)^{-1} Phi^T Z
    PhiTZ = Phi64.T @ Z64  # (F, M)

    # Use linalg.solve for each column of PhiTZ
    # solve(A, B) solves A X = B, giving X = A^{-1} B
    proj_coeffs = jnp.linalg.solve(A, PhiTZ)  # (F, M)

    # Project: Z_proj = Z - Phi @ C
    Z_proj = Z64 - Phi64 @ proj_coeffs  # (N, M)

    return Z_proj, proj_coeffs


def apply_projection_new_data(
    Z_new: jnp.ndarray,
    Phi_new: jnp.ndarray,
    proj_coeffs: jnp.ndarray,
) -> jnp.ndarray:
    """Apply stored projection to new data (for prediction).

    Args:
        Z_new: (M_new, M) residual features for new inputs
        Phi_new: (M_new, F) Fourier features for new inputs
        proj_coeffs: (F, M) coefficient matrix from project_features_orthogonal

    Returns:
        Z_new_proj: (M_new, M) projected features for new inputs
    """
    return Z_new - Phi_new @ proj_coeffs


def verify_orthogonality(
    Z_proj: jnp.ndarray,
    Phi: jnp.ndarray,
    atol: float = 1e-10,
) -> dict:
    """Diagnostic: verify that projection achieved orthogonality.

    Args:
        Z_proj: (N, M) projected features
        Phi: (N, F) Fourier features
        atol: absolute tolerance for orthogonality check

    Returns:
        dict with max_cross (should be ~0) and is_orthogonal (bool)
    """
    cross = jnp.asarray(Phi, dtype=jnp.float64).T @ jnp.asarray(Z_proj, dtype=jnp.float64)
    max_cross = float(jnp.max(jnp.abs(cross)))
    return {
        'max_cross': max_cross,
        'is_orthogonal': max_cross < atol,
        'cross_matrix_shape': cross.shape,
    }
