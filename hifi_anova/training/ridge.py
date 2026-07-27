"""Weighted ridge regression with per-feature regularization.

Uses float64 for numerical stability. Selects the most efficient
solve strategy based on problem dimensions:

  F <= N (standard):  Primal form, F×F system.
  F > N:              Dual form via Woodbury identity, N×N system.
                      Mathematically identical — no accuracy loss.

Large problems are solved on CPU via numpy to avoid GPU OOM.
The GPU memory limit is configurable via `gpu_memory_gb`.
"""

import jax
import jax.numpy as jnp
import numpy as np

# Default GPU memory budget for the ridge solve (GB).
# The largest allocation is the Phi matmul: min(N,F) × max(N,F) × 8 bytes.
# Set conservatively; override via set_gpu_memory_limit().
_GPU_MEMORY_GB = 4.0


def set_gpu_memory_limit(gb: float):
    """Set the GPU memory budget for ridge solves (in GB).

    Solves requiring more than this will fall back to CPU numpy.
    Set to 0 to force all solves onto CPU.
    Set to a large value (e.g. 24) if you have a big GPU.
    """
    global _GPU_MEMORY_GB
    _GPU_MEMORY_GB = gb


def weighted_ridge_solve(Phi: jnp.ndarray, y: jnp.ndarray,
                         reg_diag: jnp.ndarray,
                         weights: jnp.ndarray = None) -> jnp.ndarray:
    """Weighted ridge regression with per-feature regularization.

    Solves: min_w  sum_n w_n*(y_n - Phi_n @ w)^2 + w^T diag(reg_diag) w

    Strategy selection:
      - F <= N: primal form (F×F system)
      - F > N:  dual form via Woodbury (N×N system) — same accuracy
      - Estimated memory > gpu budget: CPU numpy fallback

    Args:
        Phi: (N, F) feature matrix
        y: (N,) targets (centered)
        reg_diag: (F,) per-feature regularization weights
        weights: (N,) observation weights (default: uniform)

    Returns:
        w: (F,) coefficient vector
    """
    N, F = Phi.shape

    # Estimate peak memory: Phi itself + matmul intermediate ≈ 2 × N × F × 8 bytes
    estimated_gb = 2 * N * F * 8 / 1e9
    use_cpu = estimated_gb > _GPU_MEMORY_GB

    if use_cpu:
        # Convert to numpy FIRST to avoid GPU materialization of large arrays
        Phi_np = np.asarray(Phi, dtype=np.float64)
        y_np = np.asarray(y, dtype=np.float64)
        reg_np = np.asarray(reg_diag, dtype=np.float64)
        if weights is not None:
            sqrt_w = np.sqrt(np.asarray(weights, dtype=np.float64))
            Phi_np = Phi_np * sqrt_w[:, None]
            y_np = y_np * sqrt_w
        return _solve_numpy(Phi_np, y_np, reg_np, N, F)

    Phi = jnp.asarray(Phi, dtype=jnp.float64)
    y = jnp.asarray(y, dtype=jnp.float64)
    reg_diag = jnp.asarray(reg_diag, dtype=jnp.float64)

    if weights is not None:
        weights = jnp.asarray(weights, dtype=jnp.float64)
        sqrt_w = jnp.sqrt(weights)
        Phi_w = Phi * sqrt_w[:, None]
        y_w = y * sqrt_w
    else:
        Phi_w = Phi
        y_w = y

    if F <= N:
        return _solve_primal_jax(Phi_w, y_w, reg_diag)
    else:
        return _solve_dual_jax(Phi_w, y_w, reg_diag, N)


def _solve_primal_jax(Phi_w, y_w, reg_diag):
    """Primal form: (Phi^T Phi + R) w = Phi^T y.  Size: F×F."""
    A = Phi_w.T @ Phi_w + jnp.diag(reg_diag)
    b = Phi_w.T @ y_w
    return jax.scipy.linalg.solve(A, b, assume_a='pos')


def _solve_dual_jax(Phi_w, y_w, reg_diag, N):
    """Dual form via Woodbury: solves N×N instead of F×F.

    w = R^{-1} Phi^T (Phi R^{-1} Phi^T + I)^{-1} y
    Mathematically identical to primal — no accuracy loss.
    """
    reg_max = jnp.max(reg_diag)
    threshold = jnp.maximum(reg_max * 1e-15, 1e-30)
    reg_inv = jnp.where(reg_diag > threshold, 1.0 / reg_diag, 1.0 / threshold)
    Phi_scaled = Phi_w * reg_inv[None, :]
    K = Phi_scaled @ Phi_w.T + jnp.eye(N)
    alpha = jax.scipy.linalg.solve(K, y_w, assume_a='pos')
    return reg_inv * (Phi_w.T @ alpha)


def _solve_numpy(Phi_w, y_w, reg_diag, N, F):
    """CPU fallback using numpy. Same primal/dual logic."""
    if F <= N:
        A = Phi_w.T @ Phi_w + np.diag(reg_diag)
        b = Phi_w.T @ y_w
        w = np.linalg.solve(A, b)
    else:
        reg_max = np.max(reg_diag)
        threshold = max(reg_max * 1e-15, 1e-30)
        reg_inv = np.where(reg_diag > threshold, 1.0 / reg_diag, 1.0 / threshold)
        Phi_scaled = Phi_w * reg_inv[None, :]
        K = Phi_scaled @ Phi_w.T + np.eye(N)
        alpha = np.linalg.solve(K, y_w)
        w = reg_inv * (Phi_w.T @ alpha)
    return jnp.array(w)
