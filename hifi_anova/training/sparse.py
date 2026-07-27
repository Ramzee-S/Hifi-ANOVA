"""L1, Group Lasso, Elastic Net, and Sparse Group Lasso solvers.

These complement the ridge (L2) solver in ridge.py by producing
SPARSE solutions — coefficients or entire groups driven to exactly zero.

Solvers:
  lasso_solve:              Lasso (L1 on individual coefficients)
  group_lasso_solve:        Group Lasso (L1 on group norms)
  elastic_net_solve:        Elastic Net (L1 + L2)
  sparse_group_lasso_solve: Sparse Group Lasso (group L1 + within-group L1)

All solvers support:
  - Per-feature L2 regularization (same reg_diag as ridge.py)
  - Gram-weighted group norms for Fourier-aware penalties
  - Warm starting from a previous solution

The key property: L1 penalties drive coefficients to EXACTLY zero,
enabling automatic variable selection and model sparsification.
This is complementary to the BIC/1SE selection in selection.py —
here sparsity is part of the optimization objective, not a post-hoc step.

Usage:
    # Elastic Net: sparse coefficients within each variable
    w = elastic_net_solve(Phi, y, reg_l2, alpha_l1=0.1, l1_ratio=0.5)

    # Group Lasso: zero out entire variables/pairs
    w = group_lasso_solve(Phi, y, group_slices, reg_l2, gamma=0.1)

    # Sparse Group Lasso: group selection + sparse harmonics
    w = sparse_group_lasso_solve(Phi, y, group_slices, reg_l2,
                                  gamma_group=0.1, gamma_l1=0.01)
"""

import numpy as np
from typing import List, Optional, Tuple


# =============================================================================
# Proximal operators
# =============================================================================

def _soft_threshold(v: np.ndarray, threshold: float) -> np.ndarray:
    """Soft-thresholding (proximal operator for L1).
    S(v, t) = sign(v) * max(|v| - t, 0)
    """
    return np.sign(v) * np.maximum(np.abs(v) - threshold, 0.0)


def _group_soft_threshold(v: np.ndarray, threshold: float,
                           G: Optional[np.ndarray] = None) -> np.ndarray:
    """Group soft-thresholding (proximal operator for group L1).

    If G is None:  S(v, t) = max(0, 1 - t/||v||_2) * v
    If G provided: S(v, t) = max(0, 1 - t/||v||_G) * v

    The Gram-weighted norm ||v||_G = sqrt(v^T G v) penalizes groups
    proportionally to their variance contribution.
    """
    if G is not None:
        norm = np.sqrt(max(0.0, float(v @ G @ v)))
    else:
        norm = np.linalg.norm(v)

    if norm > threshold:
        return (1.0 - threshold / norm) * v
    else:
        return np.zeros_like(v)


# =============================================================================
# Lasso (L1 on individual coefficients)
# =============================================================================

def lasso_solve(
    Phi: np.ndarray,
    y: np.ndarray,
    reg_l2: np.ndarray,
    alpha_l1: float = 0.1,
    reg_l1: Optional[np.ndarray] = None,
    max_iter: int = 1000,
    tol: float = 1e-6,
) -> np.ndarray:
    """Lasso: L1 regularization on individual coefficients.

    min_w  1/(2N) ||y - Phi w||^2 + w^T diag(reg_l2) w
           + alpha_l1 * sum_j reg_l1[j] * |w_j|

    The per-feature L1 weights reg_l1[j] allow the L1 penalty to be
    frequency-weighted (matching the L2 strategy). If reg_l1 is None,
    uniform L1 is applied (standard Lasso).

    Solved via coordinate descent (Friedman et al. 2010).

    Args:
        Phi: (N, F) feature matrix
        y: (N,) targets
        reg_l2: (F,) per-feature L2 penalty diagonal
        alpha_l1: L1 penalty strength (global scale)
        reg_l1: (F,) per-feature L1 weights (None = uniform = all 1s)
        max_iter: maximum iterations
        tol: convergence tolerance

    Returns:
        w: (F,) sparse coefficient vector
    """
    Phi = np.asarray(Phi, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    reg_l2 = np.asarray(reg_l2, dtype=np.float64)
    N, F = Phi.shape

    if reg_l1 is None:
        l1_weights = np.ones(F, dtype=np.float64)
    else:
        l1_weights = np.asarray(reg_l1, dtype=np.float64)

    # Precompute column norms
    col_norms_sq = np.sum(Phi ** 2, axis=0)  # (F,)

    w = np.zeros(F, dtype=np.float64)
    residual = y.copy()

    for iteration in range(max_iter):
        w_old = w.copy()

        for j in range(F):
            # Partial residual
            residual += Phi[:, j] * w[j]

            # Unconstrained update
            rho_j = Phi[:, j] @ residual / N

            # Coordinate-wise soft thresholding with per-feature L1
            threshold_j = alpha_l1 * l1_weights[j]
            denom = col_norms_sq[j] / N + 2 * reg_l2[j]
            w[j] = _soft_threshold(np.array([rho_j]), threshold_j)[0] / denom

            # Update residual
            residual -= Phi[:, j] * w[j]

        if np.max(np.abs(w - w_old)) < tol:
            break

    return w


# =============================================================================
# Elastic Net (L1 + L2)
# =============================================================================

def elastic_net_solve(
    Phi: np.ndarray,
    y: np.ndarray,
    reg_l2: np.ndarray,
    alpha: float = 0.1,
    l1_ratio: float = 0.5,
    adaptive_l1: bool = True,
    max_iter: int = 1000,
    tol: float = 1e-6,
) -> np.ndarray:
    """Elastic Net: combined L1 and L2 regularization.

    min_w  1/(2N) ||y - Phi w||^2
           + (1-l1_ratio)*alpha * w^T diag(reg_l2_scaled) w
           + l1_ratio * alpha * sum_j reg_l1[j] * |w_j|

    When adaptive_l1=True (default), the per-feature L1 weights match
    the L2 structure: reg_l1[j] = sqrt(reg_l2[j] / median(reg_l2)).
    This ensures L1 and L2 penalties are frequency-coherent — both
    penalize high frequencies more under smoothness/curvature/sobolev.

    When adaptive_l1=False, uniform L1 weights are used (standard elastic net).

    l1_ratio=1 → pure Lasso, l1_ratio=0 → pure ridge.
    l1_ratio=0.5 (default) → balanced sparsity + stability.

    Args:
        Phi: (N, F) feature matrix
        y: (N,) targets
        reg_l2: (F,) L2 structure vector (from any regularization strategy)
        alpha: overall penalty strength
        l1_ratio: fraction of penalty that is L1 (0 to 1)
        adaptive_l1: if True, L1 weights derived from L2 structure
        max_iter: maximum iterations
        tol: convergence tolerance

    Returns:
        w: (F,) sparse coefficient vector
    """
    reg_l2 = np.asarray(reg_l2, dtype=np.float64)
    alpha_l1 = alpha * l1_ratio
    alpha_l2 = alpha * (1.0 - l1_ratio)

    # Scale L2 by alpha_l2 (reg_l2 provides the structure)
    max_reg = np.max(reg_l2[reg_l2 > 0]) if np.any(reg_l2 > 0) else 1.0
    reg_l2_scaled = alpha_l2 * reg_l2 / max_reg

    # Build adaptive L1 weights from L2 structure
    reg_l1 = None
    if adaptive_l1:
        # L1 weight proportional to sqrt of L2 weight
        # sqrt because L1 penalty is on |w| while L2 is on w^2
        nonzero = reg_l2[reg_l2 > 1e-15]
        if len(nonzero) > 0:
            median_reg = float(np.median(nonzero))
            reg_l1 = np.sqrt(np.maximum(reg_l2, 1e-15) / median_reg)
        # else: leave as None → uniform

    return lasso_solve(Phi, y, reg_l2_scaled, alpha_l1, reg_l1, max_iter, tol)


# =============================================================================
# Group Lasso (L1 on group norms)
# =============================================================================

def group_lasso_solve(
    Phi: np.ndarray,
    y: np.ndarray,
    group_slices: List[slice],
    reg_l2: np.ndarray,
    gamma: float = 0.1,
    gram_matrices: Optional[List[Optional[np.ndarray]]] = None,
    max_iter: int = 500,
    tol: float = 1e-5,
    warm_start: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Group Lasso: L1 on group norms for structured sparsity.

    min_w  1/(2N) ||y - Phi w||^2 + w^T diag(reg_l2) w
           + gamma * sum_g ||w_g||_{G_g}

    Drives entire groups to exactly zero. With Gram-weighted norms,
    the penalty reflects actual variance contribution.

    Block coordinate descent: for each group g:
      1. Compute partial residual r_g = y - Phi_{-g} w_{-g}
      2. Unconstrained update v_g = (Phi_g^T Phi_g + N*R_g)^{-1} Phi_g^T r_g
      3. Group soft-threshold: w_g = S(v_g, N*gamma)

    Args:
        Phi: (N, F) feature matrix
        y: (N,) targets
        group_slices: list of slice objects defining groups
        reg_l2: (F,) L2 regularization diagonal
        gamma: group L1 penalty strength
        gram_matrices: list of Gram matrices per group for weighted norms
        max_iter: maximum iterations
        tol: convergence tolerance
        warm_start: (F,) initial coefficients

    Returns:
        w: (F,) coefficient vector with some groups exactly zero
    """
    Phi = np.asarray(Phi, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    reg_l2 = np.asarray(reg_l2, dtype=np.float64)
    N, F = Phi.shape
    n_groups = len(group_slices)

    w = warm_start.copy() if warm_start is not None else np.zeros(F, dtype=np.float64)

    # Precompute per-group inverse matrices
    A_inv_groups = []
    for sl in group_slices:
        Phi_g = Phi[:, sl]
        R_g = np.diag(reg_l2[sl])
        A = Phi_g.T @ Phi_g + N * R_g
        A_inv_groups.append(np.linalg.inv(A))

    for iteration in range(max_iter):
        w_old = w.copy()

        for g in range(n_groups):
            sl = group_slices[g]
            Phi_g = Phi[:, sl]

            # Partial residual
            r_g = y - Phi @ w + Phi_g @ w[sl]

            # Unconstrained solution
            v_g = A_inv_groups[g] @ (Phi_g.T @ r_g)

            # Group soft-thresholding
            G_g = gram_matrices[g] if gram_matrices is not None else None
            w[sl] = _group_soft_threshold(v_g, N * gamma, G_g)

        if np.max(np.abs(w - w_old)) < tol:
            break

    return w


# =============================================================================
# Sparse Group Lasso (group L1 + within-group L1)
# =============================================================================

def sparse_group_lasso_solve(
    Phi: np.ndarray,
    y: np.ndarray,
    group_slices: List[slice],
    reg_l2: np.ndarray,
    gamma_group: float = 0.1,
    gamma_l1: float = 0.01,
    gram_matrices: Optional[List[Optional[np.ndarray]]] = None,
    adaptive_l1: bool = True,
    max_iter: int = 500,
    tol: float = 1e-5,
) -> np.ndarray:
    """Sparse Group Lasso: group selection + within-group sparsity.

    min_w  1/(2N) ||y - Phi w||^2 + w^T diag(reg_l2) w
           + gamma_group * sum_g ||w_g||_{G_g}
           + gamma_l1 * sum_j l1_weight[j] * |w_j|

    Combines:
      - Group L1: drives entire variables/pairs to zero
      - Element L1: within active groups, drives individual harmonics to zero

    When adaptive_l1=True, the element L1 weights are derived from reg_l2
    so that the sparsity penalty respects the frequency structure.

    This is ideal for Fourier models: it selects which variables are active
    AND which harmonics within each active variable contribute.

    Args:
        Phi: (N, F) feature matrix
        y: (N,) targets
        group_slices: list of slice objects defining groups
        reg_l2: (F,) L2 regularization diagonal
        gamma_group: group L1 penalty (controls variable/pair selection)
        gamma_l1: element L1 penalty (controls within-group sparsity)
        gram_matrices: Gram matrices for group norms
        adaptive_l1: if True, L1 weights match L2 frequency structure
        max_iter: maximum iterations
        tol: convergence tolerance

    Returns:
        w: (F,) coefficient vector with group-level AND element-level sparsity
    """
    Phi = np.asarray(Phi, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    reg_l2 = np.asarray(reg_l2, dtype=np.float64)
    N, F = Phi.shape
    n_groups = len(group_slices)

    # Build adaptive L1 weights
    if adaptive_l1:
        nonzero = reg_l2[reg_l2 > 1e-15]
        if len(nonzero) > 0:
            median_reg = float(np.median(nonzero))
            l1_weights = np.sqrt(np.maximum(reg_l2, 1e-15) / median_reg)
        else:
            l1_weights = np.ones(F, dtype=np.float64)
    else:
        l1_weights = np.ones(F, dtype=np.float64)

    w = np.zeros(F, dtype=np.float64)

    # Precompute
    A_inv_groups = []
    for sl in group_slices:
        Phi_g = Phi[:, sl]
        R_g = np.diag(reg_l2[sl])
        A = Phi_g.T @ Phi_g + N * R_g
        A_inv_groups.append(np.linalg.inv(A))

    for iteration in range(max_iter):
        w_old = w.copy()

        for g in range(n_groups):
            sl = group_slices[g]
            Phi_g = Phi[:, sl]

            # Partial residual
            r_g = y - Phi @ w + Phi_g @ w[sl]

            # Unconstrained solution
            v_g = A_inv_groups[g] @ (Phi_g.T @ r_g)

            # Step 1: Element-wise soft-thresholding (within-group L1)
            # Use per-feature L1 weights matching the frequency structure
            thresholds = N * gamma_l1 * l1_weights[sl]
            v_g = np.sign(v_g) * np.maximum(np.abs(v_g) - thresholds, 0.0)

            # Step 2: Group soft-thresholding (group L1)
            G_g = gram_matrices[g] if gram_matrices is not None else None
            w[sl] = _group_soft_threshold(v_g, N * gamma_group, G_g)

        if np.max(np.abs(w - w_old)) < tol:
            break

    return w


# =============================================================================
# Convenience: solve and report sparsity
# =============================================================================

def sparse_solve(
    Phi: np.ndarray,
    y: np.ndarray,
    reg_l2: np.ndarray,
    method: str = 'elastic_net',
    group_slices: Optional[List[slice]] = None,
    gram_matrices: Optional[List[Optional[np.ndarray]]] = None,
    **kwargs,
) -> Tuple[np.ndarray, dict]:
    """Unified sparse solver with diagnostics.

    Args:
        Phi, y, reg_l2: standard ridge inputs
        method: 'lasso', 'elastic_net', 'group_lasso', 'sparse_group_lasso'
        group_slices: required for group methods
        gram_matrices: optional Gram weighting for group norms
        **kwargs: method-specific parameters

    Returns:
        w: (F,) coefficient vector
        info: dict with sparsity statistics
    """
    Phi = np.asarray(Phi, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    N, F = Phi.shape

    if method == 'lasso':
        alpha_l1 = kwargs.get('alpha_l1', 0.1)
        w = lasso_solve(Phi, y, reg_l2, alpha_l1=alpha_l1)
    elif method == 'elastic_net':
        alpha = kwargs.get('alpha', 0.1)
        l1_ratio = kwargs.get('l1_ratio', 0.5)
        w = elastic_net_solve(Phi, y, reg_l2, alpha=alpha, l1_ratio=l1_ratio)
    elif method == 'group_lasso':
        gamma = kwargs.get('gamma', 0.1)
        w = group_lasso_solve(Phi, y, group_slices, reg_l2,
                               gamma=gamma, gram_matrices=gram_matrices)
    elif method == 'sparse_group_lasso':
        gamma_group = kwargs.get('gamma_group', 0.1)
        gamma_l1 = kwargs.get('gamma_l1', 0.01)
        w = sparse_group_lasso_solve(Phi, y, group_slices, reg_l2,
                                      gamma_group=gamma_group, gamma_l1=gamma_l1,
                                      gram_matrices=gram_matrices)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Sparsity statistics
    n_zero = int(np.sum(np.abs(w) < 1e-15))
    n_nonzero = F - n_zero

    info = {
        'method': method,
        'n_features': F,
        'n_nonzero': n_nonzero,
        'n_zero': n_zero,
        'sparsity': n_zero / F,
        'rss': float(np.sum((y - Phi @ w) ** 2)),
        'mse': float(np.mean((y - Phi @ w) ** 2)),
    }

    if group_slices is not None:
        n_groups = len(group_slices)
        active_groups = [g for g in range(n_groups)
                         if np.linalg.norm(w[group_slices[g]]) > 1e-15]
        info['n_groups'] = n_groups
        info['n_active_groups'] = len(active_groups)
        info['active_groups'] = active_groups
        info['group_sparsity'] = 1.0 - len(active_groups) / n_groups

    return w, info
