"""Principled variable and group selection for Fourier components.

Three methods, from simplest to most sophisticated:

  Layer 1 — BIC marginal screening:
    For each group g, compare BIC(model with g) vs BIC(model without g).
    Include g if BIC improves. Fast (one ridge solve per group), principled
    (BIC is consistent for model selection as N→∞).

  Layer 2 — Group Lasso with BIC on the selection path:
    Solve min ½||y - Φw||² + Σ_g γ·||w_g||₂ for a grid of γ values.
    At each γ, some groups are zeroed out (hard selection). Pick the γ
    that minimizes BIC. Gold standard for structured sparsity.

  Layer 3 — One standard error rule (1SE):
    K-fold CV on the ridge or group-lasso path. Find λ_min (best CV error),
    then select the largest λ within 1 SE of that error. Conservative,
    widely recommended (Hastie, Tibshirani, Friedman).

IMPORTANT — Gram-weighted norms:
  Basis functions have DIFFERENT variances:
    linear (x-½):  Var = 1/12 ≈ 0.083
    cos(2πkx):     Var = 1/2 = 0.5
    sin(2πkx):     Var = 1/2 = 0.5
  Plus cross-terms in the Gram matrix (linear-sin coupling).

  For group selection, ||w_g||₂ is the WRONG norm — it ignores the geometry.
  The variance-relevant norm is sqrt(w_g^T G w_g), which weights each
  coefficient by its actual contribution to function variance.

  The group_lasso penalty γ·||w_g||_G (Gram-weighted) penalizes groups
  proportionally to their variance contribution, not raw coefficient magnitude.
  This is passed via the optional `gram_matrices` argument.

All methods return a list of selected group indices. They work for:
  - First-order variables (groups of 2K₁+1 coefficients)
  - Second-order pairs (groups of (2K₂+1)² coefficients)
  - Third-order triples (groups of (2K₃+1)³ coefficients)

Usage:
    from hifi_anova.training.selection import select_groups_bic, select_groups_glasso, select_groups_1se
    active_vars = select_groups_bic(Phi1, y, D, K1, strategy, lambda1)
    active_pairs = select_groups_glasso(Phi_joint, y, group_slices, n_gamma=30)
    active_vars = select_groups_1se(Phi1, y, D, K1, strategy, n_folds=5)
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from itertools import combinations

from ..core.features import basis_size


def _gram_weighted_norm(w_g: np.ndarray, G: Optional[np.ndarray] = None) -> float:
    """Compute the Gram-weighted norm of a coefficient vector.

    If G is provided: sqrt(w^T G w) — the variance-relevant norm.
    If G is None: ||w||₂ — standard Euclidean norm (fallback).

    The Gram-weighted norm reflects the ACTUAL variance contribution
    of a group, accounting for the different scales of basis functions
    (linear: 1/12, harmonics: 1/2, plus cross-terms).
    """
    if G is not None:
        val = float(w_g @ G @ w_g)
        return np.sqrt(max(0.0, val))
    return float(np.linalg.norm(w_g))


# =============================================================================
# Layer 1: BIC Marginal Screening
# =============================================================================

def select_groups_bic(
    Phi: np.ndarray,
    y: np.ndarray,
    group_slices: List[slice],
    reg_diag: np.ndarray,
    group_labels: Optional[List] = None,
    verbose: bool = True,
) -> Tuple[List[int], Dict]:
    """Select active groups by BIC marginal comparison.

    For each group g, compares:
      BIC_full  = BIC of model with all groups
      BIC_drop  = BIC of model with group g removed

    Group g is selected if removing it increases BIC (worsens the model).

    This is a leave-one-group-out screening — fast (reuses the full solve)
    and principled (BIC is consistent).

    Args:
        Phi: (N, F) full feature matrix
        y: (N,) centered targets
        group_slices: list of slice objects, one per group
        reg_diag: (F,) regularization diagonal for the full model
        group_labels: optional labels for printing (e.g., variable indices)
        verbose: print selection details

    Returns:
        selected: list of selected group indices
        info: dict with BIC values and per-group diagnostics
    """
    from .hyperopt import ridge_solve_with_diagnostics

    Phi = np.asarray(Phi, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    reg_diag = np.asarray(reg_diag, dtype=np.float64)
    N, F = Phi.shape
    n_groups = len(group_slices)

    # Full model BIC
    full_diag = ridge_solve_with_diagnostics(Phi, y, reg_diag)
    bic_full = full_diag['bic']

    # For each group: remove it and compute BIC
    selected = []
    per_group = {}

    for g in range(n_groups):
        sl = group_slices[g]
        # Build feature matrix without group g
        cols_keep = np.ones(F, dtype=bool)
        cols_keep[sl] = False
        Phi_drop = Phi[:, cols_keep]
        reg_drop = reg_diag[cols_keep]

        if Phi_drop.shape[1] == 0:
            # Degenerate: only one group
            bic_drop = N * np.log(np.var(y))
        else:
            drop_diag = ridge_solve_with_diagnostics(Phi_drop, y, reg_drop)
            bic_drop = drop_diag['bic']

        # BIC improvement from including group g
        delta_bic = bic_drop - bic_full  # positive = including g helps

        label = group_labels[g] if group_labels else g
        per_group[g] = {
            'label': label,
            'bic_with': bic_full,
            'bic_without': bic_drop,
            'delta_bic': delta_bic,
            'selected': delta_bic > 0,
        }

        if delta_bic > 0:
            selected.append(g)

    if verbose:
        n_sel = len(selected)
        print(f"  BIC selection: {n_sel}/{n_groups} groups selected")
        # Show top drops
        ranked = sorted(per_group.values(), key=lambda x: -x['delta_bic'])
        for item in ranked[:min(8, len(ranked))]:
            status = "+" if item['selected'] else "-"
            print(f"    [{status}] {item['label']}: ΔBIC={item['delta_bic']:.1f}")

    info = {
        'method': 'bic',
        'bic_full': bic_full,
        'n_groups': n_groups,
        'n_selected': len(selected),
        'per_group': per_group,
    }
    return selected, info


def select_variables_bic(
    Phi1: np.ndarray,
    y: np.ndarray,
    D: int,
    K1: int,
    reg_diag: np.ndarray,
    include_linear: bool = True,
    basis_name: str = 'fourier',
    verbose: bool = True,
) -> Tuple[List[int], Dict]:
    """BIC selection specialized for first-order variables.

    Args:
        Phi1: (N, D*block) first-order feature matrix
        y: (N,) centered targets
        D: number of variables
        K1: max harmonic
        reg_diag: (D*block,) regularization
        include_linear: whether linear term is included in basis
        basis_name: basis type ('fourier', 'legendre', 'haar')
        verbose: print details

    Returns:
        active_variables: list of selected variable indices
        info: selection diagnostics
    """
    block = basis_size(K1, include_linear, basis_name)
    group_slices = [slice(i * block, (i + 1) * block) for i in range(D)]
    labels = [f"x{i+1}" for i in range(D)]

    selected, info = select_groups_bic(
        Phi1, y, group_slices, reg_diag,
        group_labels=labels, verbose=verbose,
    )
    active = sorted(selected)
    # Ensure at least 2 variables
    if len(active) < 2:
        # Fall back to top 2 by ΔBIC
        ranked = sorted(info['per_group'].items(),
                        key=lambda x: -x[1]['delta_bic'])
        active = sorted([g for g, _ in ranked[:2]])
    info['active_variables'] = active
    return active, info


# =============================================================================
# Layer 2: Group Lasso with BIC
# =============================================================================

def _group_lasso_solve(
    Phi: np.ndarray,
    y: np.ndarray,
    group_slices: List[slice],
    reg_l2: np.ndarray,
    gamma: float,
    gram_matrices: Optional[List[Optional[np.ndarray]]] = None,
    max_iter: int = 200,
    tol: float = 1e-5,
) -> np.ndarray:
    """Solve the group-lasso problem via block coordinate descent.

    min_w  ½||y - Φw||² + w^T diag(reg_l2) w + γ Σ_g ||w_g||_G

    where ||w_g||_G = sqrt(w_g^T G_g w_g) is the Gram-weighted norm.
    If gram_matrices is None, falls back to standard ||w_g||₂.

    The Gram-weighted norm ensures the penalty reflects actual variance
    contribution, not raw coefficient magnitude. This is critical because
    linear terms (variance 1/12) and harmonic terms (variance 1/2) have
    very different scales.

    Uses block coordinate descent: for each group g:
      1. Compute partial residual: r_g = y - Φ_{-g} w_{-g}
      2. Compute unconstrained solution: v_g = (Φ_g^T Φ_g + R_g)^{-1} Φ_g^T r_g
      3. Apply group soft-thresholding: w_g = max(0, 1 - γ/||v_g||_G) · v_g

    Groups where ||v_g||_G ≤ γ get w_g = 0 exactly (hard selection).

    Args:
        Phi: (N, F) feature matrix
        y: (N,) targets
        group_slices: list of slice objects defining groups
        reg_l2: (F,) ridge penalty diagonal
        gamma: group-lasso penalty strength
        gram_matrices: list of Gram matrices per group (or None for standard norm)
        max_iter: maximum BCD iterations
        tol: convergence tolerance

    Returns:
        w: (F,) solution with some groups exactly zero
    """
    N, F = Phi.shape
    n_groups = len(group_slices)
    w = np.zeros(F, dtype=np.float64)

    # Precompute per-group quantities
    PhiTy = Phi.T @ y
    A_inv_groups = []
    for sl in group_slices:
        Phi_g = Phi[:, sl]
        PtP = Phi_g.T @ Phi_g
        R_g = np.diag(reg_l2[sl])
        A = PtP + R_g
        A_inv = np.linalg.inv(A)
        A_inv_groups.append(A_inv)

    for iteration in range(max_iter):
        w_old = w.copy()

        for g in range(n_groups):
            sl = group_slices[g]
            Phi_g = Phi[:, sl]

            # Partial residual: y minus contribution of all other groups
            r_g = y - Phi @ w + Phi_g @ w[sl]

            # Unconstrained group solution
            v_g = A_inv_groups[g] @ (Phi_g.T @ r_g)

            # Gram-weighted group soft-thresholding
            G_g = gram_matrices[g] if gram_matrices is not None else None
            v_norm = _gram_weighted_norm(v_g, G_g)
            if v_norm > gamma:
                w[sl] = (1.0 - gamma / v_norm) * v_g
            else:
                w[sl] = 0.0

        # Check convergence
        if np.max(np.abs(w - w_old)) < tol:
            break

    return w


def _compute_group_bic(Phi, y, w, group_slices):
    """BIC for a group-lasso solution.

    df = number of free parameters in active groups.
    For ridge within each active group, df_g = group_size
    (we count non-zero groups as having block_size parameters).
    """
    N = len(y)
    residuals = y - Phi @ w
    rss = float(np.sum(residuals ** 2))
    mse = rss / N

    # Count active parameters
    df = 0
    for sl in group_slices:
        if np.linalg.norm(w[sl]) > 1e-15:
            df += len(range(*sl.indices(len(w))))

    bic = N * np.log(max(mse, 1e-15)) + np.log(N) * df
    return bic, df, mse


def select_groups_glasso(
    Phi: np.ndarray,
    y: np.ndarray,
    group_slices: List[slice],
    reg_l2: np.ndarray,
    n_gamma: int = 30,
    gamma_ratio: float = 1e-3,
    gram_matrices: Optional[List[Optional[np.ndarray]]] = None,
    group_labels: Optional[List] = None,
    verbose: bool = True,
) -> Tuple[List[int], Dict]:
    """Select groups via Group Lasso with BIC on the selection path.

    Sweeps gamma from gamma_max (where everything is zeroed out) down to
    gamma_max * gamma_ratio. At each gamma, computes BIC. Selects the
    gamma with minimum BIC.

    Args:
        Phi: (N, F) feature matrix
        y: (N,) centered targets
        group_slices: list of slice objects defining groups
        reg_l2: (F,) ridge regularization diagonal
        n_gamma: number of gamma values to sweep
        gamma_ratio: ratio of smallest to largest gamma
        group_labels: optional labels for reporting
        verbose: print selection path

    Returns:
        selected: list of selected group indices
        info: dict with full path, BIC values, etc.
    """
    Phi = np.asarray(Phi, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    reg_l2 = np.asarray(reg_l2, dtype=np.float64)
    N, F = Phi.shape
    n_groups = len(group_slices)

    # Compute gamma_max: the smallest gamma that zeros out everything
    # At gamma_max, ||v_g||_G ≤ gamma for all g
    # v_g = (Phi_g^T Phi_g + R_g)^{-1} Phi_g^T y (unconstrained OLS per group)
    gamma_max = 0.0
    for g_idx, sl in enumerate(group_slices):
        Phi_g = Phi[:, sl]
        R_g = np.diag(reg_l2[sl])
        A = Phi_g.T @ Phi_g + R_g
        v_g = np.linalg.solve(A, Phi_g.T @ y)
        G_g = gram_matrices[g_idx] if gram_matrices is not None else None
        gamma_max = max(gamma_max, _gram_weighted_norm(v_g, G_g))

    gamma_max *= 1.01  # small margin
    gamma_min = gamma_max * gamma_ratio

    gammas = np.logspace(np.log10(gamma_max), np.log10(gamma_min), n_gamma)

    # Sweep the path
    path = []
    for gamma in gammas:
        w = _group_lasso_solve(Phi, y, group_slices, reg_l2, gamma,
                                gram_matrices=gram_matrices)
        bic, df, mse = _compute_group_bic(Phi, y, w, group_slices)
        active = [g for g in range(n_groups)
                  if np.linalg.norm(w[group_slices[g]]) > 1e-15]
        path.append({
            'gamma': gamma,
            'bic': bic,
            'df': df,
            'mse': mse,
            'n_active': len(active),
            'active_groups': active,
            'w': w,
        })

    # Select gamma with minimum BIC
    best_idx = int(np.argmin([p['bic'] for p in path]))
    best = path[best_idx]
    selected = best['active_groups']

    if verbose:
        print(f"  Group Lasso: swept {n_gamma} gammas, "
              f"best at γ={best['gamma']:.4f} "
              f"({best['n_active']}/{n_groups} groups, "
              f"df={best['df']:.0f}, BIC={best['bic']:.1f})")

    info = {
        'method': 'group_lasso',
        'path': path,
        'best_gamma': best['gamma'],
        'best_bic': best['bic'],
        'n_selected': len(selected),
        'n_groups': n_groups,
    }
    return selected, info


def select_variables_glasso(
    Phi1: np.ndarray,
    y: np.ndarray,
    D: int,
    K1: int,
    reg_diag: np.ndarray,
    n_gamma: int = 30,
    G1: Optional[np.ndarray] = None,
    include_linear: bool = True,
    basis_name: str = 'fourier',
    verbose: bool = True,
) -> Tuple[List[int], Dict]:
    """Group Lasso selection specialized for first-order variables.

    If G1 (Gram matrix) is provided, uses Gram-weighted norms so that
    linear terms (variance 1/12) and harmonics (variance 1/2) are
    penalized proportionally to their variance contribution.
    """
    block = basis_size(K1, include_linear, basis_name)
    group_slices = [slice(i * block, (i + 1) * block) for i in range(D)]
    labels = [f"x{i+1}" for i in range(D)]

    gram_matrices = None
    if G1 is not None:
        G1_np = np.asarray(G1, dtype=np.float64)
        gram_matrices = [G1_np] * D

    selected, info = select_groups_glasso(
        Phi1, y, group_slices, reg_diag,
        n_gamma=n_gamma, gram_matrices=gram_matrices,
        group_labels=labels, verbose=verbose,
    )
    active = sorted(selected)
    if len(active) < 2:
        # Fall back: take the 2 groups with largest ||w_g|| at best gamma
        best_w = info['path'][int(np.argmin([p['bic'] for p in info['path']]))]['w']
        norms = [(g, np.linalg.norm(best_w[group_slices[g]])) for g in range(D)]
        norms.sort(key=lambda x: -x[1])
        active = sorted([g for g, _ in norms[:2]])
    info['active_variables'] = active
    return active, info


# =============================================================================
# Layer 3: One Standard Error Rule (1SE)
# =============================================================================

def _kfold_cv_ridge(
    Phi: np.ndarray,
    y: np.ndarray,
    reg_diag: np.ndarray,
    n_folds: int = 5,
    seed: int = 42,
) -> Tuple[float, float]:
    """K-fold cross-validation for ridge regression.

    Returns:
        mean_mse: mean CV error across folds
        se_mse: standard error of the CV error
    """
    N = len(y)
    rng = np.random.RandomState(seed)
    perm = rng.permutation(N)
    fold_size = N // n_folds

    mse_folds = []
    for k in range(n_folds):
        val_idx = perm[k * fold_size: (k + 1) * fold_size]
        train_idx = np.concatenate([perm[:k * fold_size],
                                     perm[(k + 1) * fold_size:]])

        Phi_tr = Phi[train_idx]
        y_tr = y[train_idx]
        Phi_va = Phi[val_idx]
        y_va = y[val_idx]

        # Ridge solve on training fold
        A = Phi_tr.T @ Phi_tr + np.diag(reg_diag)
        w = np.linalg.solve(A, Phi_tr.T @ y_tr)

        # Validation error
        pred = Phi_va @ w
        mse = float(np.mean((y_va - pred) ** 2))
        mse_folds.append(mse)

    mean_mse = np.mean(mse_folds)
    se_mse = np.std(mse_folds, ddof=1) / np.sqrt(n_folds)
    return float(mean_mse), float(se_mse)


def select_groups_1se(
    Phi: np.ndarray,
    y: np.ndarray,
    group_slices: List[slice],
    reg_structure: np.ndarray,
    n_lambdas: int = 40,
    lambda_bounds: Tuple[float, float] = (1e-6, 1e2),
    n_folds: int = 5,
    group_labels: Optional[List] = None,
    verbose: bool = True,
    seed: int = 42,
) -> Tuple[List[int], Dict]:
    """Select groups using the 1 standard error rule on the ridge path.

    Steps:
      1. Sweep lambda on a log grid
      2. At each lambda, compute k-fold CV error and its standard error
      3. Find lambda_min (minimum CV error)
      4. Find lambda_1se: largest lambda with CV error ≤ CV_min + 1 SE
      5. At lambda_1se, identify which groups have substantial coefficients

    A group is "active" at lambda_1se if ||w_g||₂ > threshold, where
    threshold is determined by the noise level: threshold = sigma * sqrt(block_size / N).

    Args:
        Phi: (N, F) feature matrix
        y: (N,) centered targets
        group_slices: list of slice objects
        reg_structure: (F,) relative regularization weights (multiplied by lambda)
        n_lambdas: number of lambda values to sweep
        lambda_bounds: search range
        n_folds: CV folds
        group_labels: optional labels
        verbose: print details
        seed: random seed for CV splits

    Returns:
        selected: list of selected group indices
        info: dict with CV path, lambda_min, lambda_1se, etc.
    """
    Phi = np.asarray(Phi, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    reg_structure = np.asarray(reg_structure, dtype=np.float64)
    N, F = Phi.shape
    n_groups = len(group_slices)

    lambdas = np.logspace(np.log10(lambda_bounds[0]),
                          np.log10(lambda_bounds[1]), n_lambdas)

    # Sweep lambda, compute CV error at each
    cv_path = []
    for lam in lambdas:
        reg_diag = lam * reg_structure
        mean_mse, se_mse = _kfold_cv_ridge(Phi, y, reg_diag, n_folds, seed)

        # Also solve on full data for group norms
        A = Phi.T @ Phi + np.diag(reg_diag)
        w = np.linalg.solve(A, Phi.T @ y)

        group_norms = [float(np.linalg.norm(w[sl])) for sl in group_slices]
        n_active = sum(1 for gn in group_norms if gn > 1e-10)

        cv_path.append({
            'lambda': lam,
            'cv_mean': mean_mse,
            'cv_se': se_mse,
            'w': w,
            'group_norms': group_norms,
            'n_active': n_active,
        })

    cv_means = np.array([p['cv_mean'] for p in cv_path])
    cv_ses = np.array([p['cv_se'] for p in cv_path])

    # Find lambda_min
    idx_min = int(np.argmin(cv_means))
    cv_min = cv_means[idx_min]
    se_at_min = cv_ses[idx_min]
    lambda_min = lambdas[idx_min]

    # Find lambda_1se: largest lambda with cv_mean ≤ cv_min + 1*SE
    threshold_cv = cv_min + se_at_min
    # Search from high lambda (most regularized) to low
    idx_1se = idx_min
    for i in range(len(lambdas) - 1, -1, -1):
        if cv_means[i] <= threshold_cv:
            idx_1se = i
            break

    lambda_1se = lambdas[idx_1se]
    w_1se = cv_path[idx_1se]['w']

    # Determine active groups at lambda_1se
    # Threshold: a group is active if its norm exceeds noise-level expectation
    residuals_1se = y - Phi @ w_1se
    sigma_hat = float(np.std(residuals_1se))

    selected = []
    for g in range(n_groups):
        sl = group_slices[g]
        block_size = len(range(*sl.indices(F)))
        # Expected norm of noise-only group: sigma * sqrt(block_size / N)
        noise_threshold = sigma_hat * np.sqrt(block_size / N)
        group_norm = float(np.linalg.norm(w_1se[sl]))
        if group_norm > noise_threshold:
            selected.append(g)

    if verbose:
        print(f"  1SE rule: λ_min={lambda_min:.4e} (CV={cv_min:.4f}), "
              f"λ_1se={lambda_1se:.4e} (CV={cv_means[idx_1se]:.4f}), "
              f"{len(selected)}/{n_groups} groups active")

    # Ensure at least 2 groups
    if len(selected) < 2:
        norms = [(g, float(np.linalg.norm(w_1se[group_slices[g]])))
                 for g in range(n_groups)]
        norms.sort(key=lambda x: -x[1])
        selected = sorted([g for g, _ in norms[:2]])

    info = {
        'method': '1se',
        'lambda_min': lambda_min,
        'lambda_1se': lambda_1se,
        'cv_min': cv_min,
        'se_at_min': se_at_min,
        'cv_path': cv_path,
        'n_selected': len(selected),
        'n_groups': n_groups,
        'sigma_hat': sigma_hat,
    }
    return selected, info


def select_variables_1se(
    Phi1: np.ndarray,
    y: np.ndarray,
    D: int,
    K1: int,
    reg_structure: np.ndarray,
    n_folds: int = 5,
    include_linear: bool = True,
    basis_name: str = 'fourier',
    verbose: bool = True,
    seed: int = 42,
) -> Tuple[List[int], Dict]:
    """1SE selection specialized for first-order variables."""
    block = basis_size(K1, include_linear, basis_name)
    group_slices = [slice(i * block, (i + 1) * block) for i in range(D)]
    labels = [f"x{i+1}" for i in range(D)]

    selected, info = select_groups_1se(
        Phi1, y, group_slices, reg_structure,
        n_folds=n_folds, group_labels=labels,
        verbose=verbose, seed=seed,
    )
    info['active_variables'] = sorted(selected)
    return sorted(selected), info


# =============================================================================
# Unified interface
# =============================================================================

def select_active_variables_principled(
    Phi1: np.ndarray,
    y: np.ndarray,
    D: int,
    K1: int,
    reg_diag: np.ndarray,
    method: str = 'bic',
    G1: Optional[np.ndarray] = None,
    include_linear: bool = True,
    basis_name: str = 'fourier',
    verbose: bool = True,
    **kwargs,
) -> Tuple[List[int], Dict]:
    """Unified interface for principled variable selection.

    Args:
        Phi1: (N, D*block) first-order feature matrix
        y: (N,) centered targets
        D: number of variables
        K1: max harmonic
        reg_diag: (D*block,) regularization diagonal
        method: 'bic', 'group_lasso', or '1se'
        G1: (block, block) Gram matrix for Gram-weighted norms (optional)
        include_linear: whether linear term is included in basis
        basis_name: basis type ('fourier', 'legendre', 'haar')
        verbose: print details
        **kwargs: method-specific parameters

    Returns:
        active_variables: sorted list of variable indices
        info: method-specific diagnostics
    """
    Phi1 = np.asarray(Phi1, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    reg_diag = np.asarray(reg_diag, dtype=np.float64)

    if method == 'bic':
        return select_variables_bic(Phi1, y, D, K1, reg_diag,
                                    include_linear=include_linear,
                                    basis_name=basis_name, verbose=verbose)
    elif method == 'group_lasso':
        n_gamma = kwargs.get('n_gamma', 30)
        return select_variables_glasso(Phi1, y, D, K1, reg_diag,
                                        n_gamma=n_gamma, G1=G1,
                                        include_linear=include_linear,
                                        basis_name=basis_name, verbose=verbose)
    elif method == '1se':
        # For 1SE, we need the reg_structure (relative weights), not reg_diag
        # Extract by dividing out the lambda: reg_structure = reg_diag / lambda
        # Use median of nonzero entries as the effective lambda
        nonzero = reg_diag[reg_diag > 1e-15]
        if len(nonzero) > 0:
            lam_eff = float(np.median(nonzero))
            reg_structure = reg_diag / lam_eff
        else:
            reg_structure = np.ones_like(reg_diag)
        n_folds = kwargs.get('n_folds', 5)
        seed = kwargs.get('seed', 42)
        return select_variables_1se(Phi1, y, D, K1, reg_structure,
                                     n_folds=n_folds,
                                     include_linear=include_linear,
                                     basis_name=basis_name,
                                     verbose=verbose, seed=seed)
    else:
        raise ValueError(f"Unknown selection method: '{method}'. "
                         f"Options: 'bic', 'group_lasso', '1se'")


# =============================================================================
# Post-fit group pruning (for pairs, triples, or any order)
# =============================================================================

def prune_groups_postfit(
    Phi: np.ndarray,
    y: np.ndarray,
    group_slices: List[slice],
    reg_diag: np.ndarray,
    method: str = 'bic',
    gram_matrices: Optional[List[Optional[np.ndarray]]] = None,
    group_labels: Optional[List] = None,
    verbose: bool = True,
    **kwargs,
) -> Tuple[List[int], Dict]:
    """Post-fit pruning: after fitting all candidate groups, remove inactive ones.

    This is the second stage of the selection pipeline:
      1. Candidate generation (heuristic: all/either/both) -> broad set
      2. Fit all candidates jointly (ridge solve)
      3. THIS FUNCTION: apply criterion to prune -> narrow set
      4. Refit with only surviving groups

    Works for any group structure — variables, pairs, triples, or mixed.

    Args:
        Phi: (N, F) full feature matrix (all candidates)
        y: (N,) centered targets
        group_slices: list of slice objects defining each group in Phi
        reg_diag: (F,) regularization diagonal
        method: 'bic', 'group_lasso', '1se', or 'none' (keep all)
        gram_matrices: list of Gram matrices per group for weighted norms
        group_labels: optional labels for reporting
        verbose: print pruning details
        **kwargs: method-specific parameters

    Returns:
        surviving: list of group indices that survived pruning
        info: diagnostics dict
    """
    if method == 'none' or method is None:
        return list(range(len(group_slices))), {'method': 'none', 'pruned': 0}

    Phi = np.asarray(Phi, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    reg_diag = np.asarray(reg_diag, dtype=np.float64)

    if method == 'bic':
        return select_groups_bic(Phi, y, group_slices, reg_diag,
                                  group_labels=group_labels, verbose=verbose)
    elif method == 'group_lasso':
        n_gamma = kwargs.get('n_gamma', 30)
        return select_groups_glasso(Phi, y, group_slices, reg_diag,
                                     n_gamma=n_gamma, gram_matrices=gram_matrices,
                                     group_labels=group_labels, verbose=verbose)
    elif method == '1se':
        nonzero = reg_diag[reg_diag > 1e-15]
        lam_eff = float(np.median(nonzero)) if len(nonzero) > 0 else 1.0
        reg_structure = reg_diag / lam_eff
        n_folds = kwargs.get('n_folds', 5)
        seed = kwargs.get('seed', 42)
        return select_groups_1se(Phi, y, group_slices, reg_structure,
                                  n_folds=n_folds, group_labels=group_labels,
                                  verbose=verbose, seed=seed)
    else:
        raise ValueError(f"Unknown pruning method: '{method}'. "
                         f"Options: 'none', 'bic', 'group_lasso', '1se'")
