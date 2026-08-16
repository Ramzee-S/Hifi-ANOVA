"""Quasi-Monte-Carlo sampling of the unit cube for measure-coherent estimates.

The analytic Sobol/variance machinery in :mod:`hifi_anova.analysis.sobol` defines
component variances as Gram quadratic forms ``w^T G w`` — the *exact* variance of a
parametric component under the **uniform (independent) input measure on [0,1]^D**.
Any quantity that has to share a denominator with those forms (most importantly the
residual-network variance, but also a re-decomposition target) must be estimated
under the *same* measure to keep the reported fractions coherent.

This module provides a small, deterministic QMC layer for exactly that:

  - :func:`sobol_cube_sample`      — a low-discrepancy Sobol point set on [0,1]^D
  - :func:`qmc_uniform_variance`   — Var_uniform(f) for a callable f: [0,1]^D -> R

Determinism: given ``(D, n_points, seed)`` the point set is fixed, so every derived
estimate is reproducible. A QMC estimate of a variance converges far faster than
plain Monte-Carlo (≈ O(1/n) vs O(1/√n) for smooth integrands), so a single
2^16–2^20 point sweep pins ``Var_uniform`` tightly enough that its own sampling
error is negligible relative to the model's other uncertainties.
"""

import numpy as np
from typing import Callable, Optional

__all__ = ["sobol_cube_sample", "qmc_uniform_variance"]


def _next_pow2_m(n_points: int) -> int:
    """Smallest m with 2**m >= n_points (Sobol nets are balanced at powers of 2)."""
    m = 0
    while (1 << m) < n_points:
        m += 1
    return m


def sobol_cube_sample(
    D: int,
    n_points: int = 1 << 16,
    seed: int = 0,
) -> np.ndarray:
    """Deterministic low-discrepancy sample of the uniform cube [0,1]^D.

    Uses a scrambled Sobol sequence, drawn at the next power of two ≥ ``n_points``
    so the net stays balanced (Sobol' loses its equidistribution guarantees on a
    non-2**m prefix). The scramble is seeded, so the point set is reproducible.

    Args:
        D: input dimension.
        n_points: requested number of points (rounded up to a power of two).
        seed: RNG seed for the Owen scramble.

    Returns:
        (2**m, D) float64 array in [0,1], with 2**m >= n_points.
    """
    from scipy.stats import qmc

    m = _next_pow2_m(max(1, int(n_points)))
    engine = qmc.Sobol(d=D, scramble=True, seed=seed)
    return np.asarray(engine.random_base2(m=m), dtype=np.float64)


def qmc_uniform_variance(
    fn: Callable[[np.ndarray], np.ndarray],
    D: int,
    n_points: int = 1 << 16,
    seed: int = 0,
    points: Optional[np.ndarray] = None,
) -> float:
    """Estimate ``Var_{x~U([0,1]^D)}(fn(x))`` by QMC.

    Args:
        fn: vectorised callable mapping an (M, D) batch of cube points to (M,)
            outputs (extra trailing singleton dims are squeezed).
        D: input dimension.
        n_points: QMC sample size (rounded up to a power of two).
        seed: scramble seed.
        points: optional precomputed cube sample to reuse (skips generation).

    Returns:
        Empirical variance of ``fn`` over the QMC sample (population variance).
    """
    if points is None:
        points = sobol_cube_sample(D, n_points, seed)
    vals = np.asarray(fn(points))
    vals = np.squeeze(vals)
    if vals.ndim > 1:
        vals = vals.reshape(vals.shape[0], -1)[:, 0]
    return float(np.var(vals))
