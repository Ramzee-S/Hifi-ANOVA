"""Weighted ridge regression with per-feature regularization.

Uses float64 for numerical stability. Selects the most efficient
solve strategy based on problem dimensions:

  F <= N (standard):  Primal form, F×F system.
  F > N:              Dual form via Woodbury identity, N×N system.
                      Mathematically identical to the primal when the penalty
                      ``R`` is positive definite (the Woodbury form needs
                      ``R^{-1}``). When ``R`` has a zero/near-zero entry — e.g.
                      the unpenalized augmented intercept — the dual is skipped
                      and the primal (which stays PD, the data term covers the
                      unpenalized direction) is used instead.

Large problems are solved on CPU via numpy to avoid GPU OOM.
The GPU memory limit is configurable via `gpu_memory_gb`.
"""

import warnings

import jax
from ..array_backend import xp as jnp  # switchable array backend (numpy exact core)
import numpy as np

# Default GPU memory budget for the ridge solve (GB).
# The largest allocation is the Phi matmul: min(N,F) × max(N,F) × 8 bytes.
# Set conservatively; override via set_gpu_memory_limit().
_GPU_MEMORY_GB = 4.0

# Warn only once per process about the x64 fallback (the solve itself is
# still float64 via numpy — the warning is informational, not a failure).
_WARNED_X64_FALLBACK = False


def _x64_enabled() -> bool:
    """True iff JAX honors float64 (jax_enable_x64 is on)."""
    return jnp.zeros((), dtype=jnp.float64).dtype == jnp.float64


def set_gpu_memory_limit(gb: float):
    """Set the GPU memory budget for ridge solves (in GB).

    Solves requiring more than this will fall back to CPU numpy.
    Set to 0 to force all solves onto CPU.
    Set to a large value (e.g. 24) if you have a big GPU.
    """
    global _GPU_MEMORY_GB
    _GPU_MEMORY_GB = gb


def kfold_indices(N: int, n_folds: int, seed: int,
                  scheme: str = 'strided', return_perm: bool = False):
    """Held-out (test) index arrays for k-fold CV over a seeded permutation.

    Single source of truth for the fold split shared by the CV loops in
    ``selection`` and ``joint_lambda`` (they hand-rolled the same permutation
    two different ways). Both schemes now **partition all N points** — every
    point is held out in exactly one fold:

      - ``'strided'`` (``joint_lambda._kfold_nll``): fold ``i`` is
        ``perm[i::n_folds]``; fold sizes differ by at most one.
      - ``'contiguous'`` (``selection._kfold_cv_ridge``): contiguous blocks of
        the permutation, **remainder-inclusive** (as in ``sklearn`` ``KFold``):
        the first ``N % n_folds`` folds get one extra point, so the trailing
        ``N % n_folds`` points are held out rather than silently kept in every
        train complement. When ``N % n_folds == 0`` the blocks are exactly
        ``perm[k*fs:(k+1)*fs]`` — byte-identical to the pre-fix split.

    ``n_folds`` must be an integer in ``[2, N]`` (CV needs at least two folds and
    each fold needs at least one point); anything else raises ``ValueError``.

    ``return_perm=True`` additionally returns the underlying permutation, so a
    caller can reconstruct the train fold in *permutation order* rather than the
    sorted order of ``setdiff1d`` — the ridge normal equations are order-invariant
    only up to ULP round-off, and the contiguous caller reconstructs train this
    way to stay byte-identical to its pre-dedup output.
    """
    if scheme not in ('strided', 'contiguous'):
        raise ValueError(
            f"scheme must be 'strided' or 'contiguous'; got {scheme!r}.")
    if not isinstance(n_folds, (int, np.integer)) or isinstance(n_folds, bool):
        raise ValueError(
            f"n_folds must be an integer; got {type(n_folds).__name__} "
            f"{n_folds!r}.")
    n_folds = int(n_folds)
    if n_folds < 2:
        raise ValueError(
            f"n_folds must be >= 2 for k-fold CV (got {n_folds}); a single fold "
            f"leaves nothing held out.")
    if n_folds > N:
        raise ValueError(
            f"n_folds={n_folds} exceeds the number of samples N={N}: each fold "
            f"needs at least one point. Reduce n_folds (or use N samples).")
    rng = np.random.RandomState(seed)
    perm = rng.permutation(N)
    if scheme == 'strided':
        folds = [perm[i::n_folds] for i in range(n_folds)]
    else:
        # Remainder-inclusive contiguous blocks (sklearn KFold): the first
        # ``rem`` folds get ``fs + 1`` points. Every point is held out once.
        fs, rem = divmod(N, n_folds)
        sizes = [fs + 1 if i < rem else fs for i in range(n_folds)]
        bounds = np.concatenate([[0], np.cumsum(sizes)])
        folds = [perm[bounds[k]:bounds[k + 1]] for k in range(n_folds)]
    return (folds, perm) if return_perm else folds


def leverage_diag(Phi: np.ndarray, reg_diag: np.ndarray,
                  weights: np.ndarray) -> np.ndarray:
    """Diagonal of the weighted ridge hat matrix.

    ``S = Phi (Phi^T W Phi + diag(reg))^{-1} Phi^T W``;
    ``lev_n = w_n * phi_n^T A^{-1} phi_n``. ``sum(lev) = tr(S) = df_mean``.

    Used to de-bias in-sample squared residuals before a variance fit:
    under the fitted mean ``E[r_n^2] ≈ sigma_n^2 (1 - lev_n)``, so the
    variance solve should see ``r_n^2 / (1 - lev_n)``.
    """
    Phi = np.asarray(Phi, dtype=np.float64)
    W = np.asarray(weights, dtype=np.float64)
    A = Phi.T @ (W[:, None] * Phi) + np.diag(np.asarray(reg_diag,
                                                        dtype=np.float64))
    X = np.linalg.solve(A, Phi.T)                     # A^{-1} Phi^T  (F, N)
    quad = np.einsum('nf,fn->n', Phi, X)              # phi_n^T A^{-1} phi_n
    return W * quad


# DEC-028 leverage-correction clip floor: bounds 1/(1-lev) when lev -> 1
# (near-interpolation). This constant is the single point of drift risk between
# the trainer Stage-D loop and joint_lambda — keep it here, not inlined.
_LEV_CORRECTION_CLIP_LO = 1e-3


def debias_squared_residuals(r2: np.ndarray, lev: np.ndarray,
                             *, correct: bool = True) -> np.ndarray:
    """DEC-028 leverage de-biasing of in-sample squared residuals.

    Under a fitted mean the in-sample squared residuals are biased low,
    ``E[r_n^2] ≈ sigma_n^2 (1 - lev_n)``, so the log-variance solve should see
    ``r_n^2 / (1 - lev_n)``. The ``(1e-3, 1)`` clip guards ``lev -> 1``
    (near-interpolation) from blowing the correction up. ``correct=False``
    returns ``r^2`` unchanged (float64 numpy).

    Single source of truth for the correction shared by the trainer's Stage-D
    alternating loop (``training/trainer.py::_fit_heteroscedastic``) and
    ``joint_lambda._joint_fit`` — the two must not drift (see the trainer
    comment at the leverage block).
    """
    r2 = np.asarray(r2, dtype=np.float64)
    if not correct:
        return r2
    lev = np.asarray(lev, dtype=np.float64)
    return r2 / np.clip(1.0 - lev, _LEV_CORRECTION_CLIP_LO, 1.0)


# The Woodbury dual forms R^{-1}, so it is only valid (and well conditioned)
# when the penalty is strictly positive definite. A zero/near-zero entry — the
# unpenalized augmented intercept is the canonical case — makes R^{-1} undefined;
# the previous cap (reg_inv = 1/(reg_max*1e-15)) both perturbed the estimator
# and scaled that column by ~1e15, wrecking cond(K) and returning a materially
# wrong solution while the primal system stayed well conditioned. Route such
# problems to the primal solve instead. The ratio floor keeps the dual only when
# cond contribution from R^{-1} (reg_max/reg_min) is safe in float64.
_DUAL_MIN_REG_RATIO = 1e-8


def _dual_is_safe(reg_diag) -> bool:
    """True iff the Woodbury dual may be used: ``R`` is positive definite with a
    float64-safe dynamic range. Otherwise the primal solve must be used."""
    reg = np.asarray(reg_diag, dtype=np.float64)
    if reg.size == 0:
        return True
    reg_max = float(np.max(reg))
    if reg_max <= 0.0:
        return False
    return float(np.min(reg)) > reg_max * _DUAL_MIN_REG_RATIO


def weighted_ridge_solve(Phi: jnp.ndarray, y: jnp.ndarray,
                         reg_diag: jnp.ndarray,
                         weights: jnp.ndarray = None) -> jnp.ndarray:
    """Weighted ridge regression with per-feature regularization.

    Solves: min_w  sum_n w_n*(y_n - Phi_n @ w)^2 + w^T diag(reg_diag) w

    Strategy selection:
      - F <= N: primal form (F×F system)
      - F > N and R positive definite: dual form via Woodbury (N×N) — same
        accuracy
      - F > N but R has a zero/near-zero entry (e.g. unpenalized intercept):
        primal form (the dual's R^{-1} is undefined / ill-conditioned there)
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

    # numpy exact core: the whole solve goes through the (float64, LAPACK)
    # numpy path — no jnp ops, no per-shape XLA compilation.
    from ..array_backend import get_array_backend
    if get_array_backend() == "numpy":
        Phi_np = np.asarray(Phi, dtype=np.float64)
        y_np = np.asarray(y, dtype=np.float64)
        reg_np = np.asarray(reg_diag, dtype=np.float64)
        if weights is not None:
            sqrt_w = np.sqrt(np.asarray(weights, dtype=np.float64))
            Phi_np = Phi_np * sqrt_w[:, None]
            y_np = y_np * sqrt_w
        return _solve_numpy(Phi_np, y_np, reg_np, N, F)

    # Estimate peak memory: Phi itself + matmul intermediate ≈ 2 × N × F × 8 bytes
    estimated_gb = 2 * N * F * 8 / 1e9
    use_cpu = estimated_gb > _GPU_MEMORY_GB

    # This module promises float64 normal equations, but without
    # jax_enable_x64 JAX silently truncates the jnp.float64 casts below to
    # float32 — numerically unsafe for the ill-conditioned Fourier+linear
    # Gram. The one-call API enables x64; direct callers (e.g. a bare
    # HiFiANOVATrainer) may not. Honor the promise via the numpy float64
    # path instead of solving in float32.
    if not use_cpu and not _x64_enabled():
        global _WARNED_X64_FALLBACK
        if not _WARNED_X64_FALLBACK:
            _WARNED_X64_FALLBACK = True
            warnings.warn(
                "jax_enable_x64 is off: ridge normal equations would silently "
                "run in float32. Falling back to the numpy float64 solver. "
                "Enable x64 (jax.config.update('jax_enable_x64', True) or "
                "HIFI_ANOVA_X64=1) to solve on the JAX backend.",
                RuntimeWarning, stacklevel=2)
        use_cpu = True

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

    if F <= N or not _dual_is_safe(reg_diag):
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

    Requires ``R`` positive definite (it forms ``R^{-1}``); callers gate this
    behind ``_dual_is_safe``. Under that gate it is mathematically identical to
    the primal — no accuracy loss.
    """
    reg_max = jnp.max(reg_diag)
    threshold = jnp.maximum(reg_max * 1e-15, 1e-30)
    reg_inv = jnp.where(reg_diag > threshold, 1.0 / reg_diag, 1.0 / threshold)
    Phi_scaled = Phi_w * reg_inv[None, :]
    K = Phi_scaled @ Phi_w.T + jnp.eye(N)
    alpha = jax.scipy.linalg.solve(K, y_w, assume_a='pos')
    return reg_inv * (Phi_w.T @ alpha)


def _solve_numpy(Phi_w, y_w, reg_diag, N, F):
    """CPU fallback using numpy. Same primal/dual logic (and the same
    ``_dual_is_safe`` gate on the Woodbury form)."""
    if F <= N or not _dual_is_safe(reg_diag):
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
