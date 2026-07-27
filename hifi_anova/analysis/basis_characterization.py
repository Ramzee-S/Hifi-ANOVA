"""Three-way basis characterization: Fourier vs Legendre vs Haar.

Fits three independent models and characterizes each variable's effect type:
  - Polynomial (Legendre captures most)
  - Oscillatory (Fourier captures most)
  - Localized (Haar captures most — steps, thresholds, regime boundaries)

Two analysis modes:
  1. Independent fits: fit all three bases, compare RMSE and per-variable Sobol.
  2. Cross-residual: fit Legendre first, project residual onto Fourier and Haar
     per variable. Gives a decomposition: polynomial + oscillatory + localized.

Usage:
    from hifi_anova.analysis.basis_characterization import (
        multi_basis_fit, cross_residual_characterization, auto_select_basis
    )

    # Quick comparison
    comp = multi_basis_fit(x_train, y_train, x_val, y_val)
    print(comp['summary'])  # which basis fits best overall

    # Per-variable characterization
    char = cross_residual_characterization(x_train, y_train, x_val, y_val)
    print_characterization_table(char)

    # Automatic basis recommendation
    rec = auto_select_basis(char)
    # {0: 'legendre', 1: 'fourier', 2: 'haar', ...}
"""

import jax.numpy as jnp
import numpy as np
from typing import Dict, List, Optional, Tuple

from ..core.features import build_first_order_features, basis_size
from ..core.gram import build_gram_matrix
from ..core.haar import HaarBasis
from ..training.ridge import weighted_ridge_solve
from ..training.regularization import build_regularization_vector


# ─────────────────────────────────────────────────────────────
# Mode 1: Independent three-basis comparison
# ─────────────────────────────────────────────────────────────

def multi_basis_fit(
    x_train: jnp.ndarray,
    y_train: jnp.ndarray,
    x_val: jnp.ndarray,
    y_val: jnp.ndarray,
    K_legendre: int = 10,
    K_fourier: int = 10,
    J_haar: int = 5,
    strategy: str = 'variance',
    lambda_order1: float = 0.001,
    verbose: bool = True,
) -> dict:
    """Fit three independent models and compare.

    Each model is a first-order ridge solve (seconds each).

    Args:
        x_train, y_train: training data, x in [0,1].
        x_val, y_val: validation data.
        K_legendre: max polynomial degree (Legendre basis, K features/var).
        K_fourier: max harmonic number (Fourier basis, 2K+1 features/var).
        J_haar: max wavelet scale (Haar basis, 2^J-1 features/var).
        strategy: regularization strategy.
        lambda_order1: regularization strength.
        verbose: print comparison table.

    Returns:
        dict with:
          models: {basis_name: {'w': array, 'f0': float, 'rmse_train': float,
                                'rmse_val': float, 'r_squared': float}}
          per_variable: {i: {'var_legendre': float, 'var_fourier': float,
                             'var_haar': float, 'best_basis': str}}
          summary: {'best_overall': str, 'character': str}
    """
    N_train, D = x_train.shape
    y_mean = float(jnp.mean(y_train))
    y_c_train = y_train - y_mean
    y_c_val = y_val - y_mean

    results = {'models': {}, 'per_variable': {}, 'summary': {}}

    configs = [
        ('legendre', K_legendre),
        ('fourier', K_fourier),
        ('haar', J_haar),
    ]

    for basis_name, K in configs:
        # Build features
        phi_train = build_first_order_features(x_train, K, basis_name=basis_name)
        phi_val = build_first_order_features(x_val, K, basis_name=basis_name)

        # Build regularization
        reg = build_regularization_vector(
            D=D, K1=K, K2=0, P=0, strategy=strategy,
            lambda_order1=lambda_order1, basis_name=basis_name)

        # Ridge solve
        w = weighted_ridge_solve(phi_train, y_c_train, reg)

        # Predictions and RMSE
        pred_train = phi_train @ w
        pred_val = phi_val @ w
        rmse_train = float(jnp.sqrt(jnp.mean((y_c_train - pred_train) ** 2)))
        rmse_val = float(jnp.sqrt(jnp.mean((y_c_val - pred_val) ** 2)))
        var_y = float(jnp.var(y_c_val))
        r_sq = 1.0 - float(jnp.mean((y_c_val - pred_val) ** 2)) / (var_y + 1e-10)

        # Per-variable variance via Gram
        G = build_gram_matrix(K, basis_name=basis_name)
        B = basis_size(K, basis_name=basis_name)
        per_var_variance = {}
        for i in range(D):
            wi = w[i * B:(i + 1) * B]
            per_var_variance[i] = float(wi @ G @ wi)

        results['models'][basis_name] = {
            'w': w, 'f0': y_mean, 'K': K,
            'rmse_train': rmse_train, 'rmse_val': rmse_val,
            'r_squared': r_sq,
            'per_var_variance': per_var_variance,
            'total_variance': sum(per_var_variance.values()),
        }

    # Per-variable: which basis captures most variance?
    for i in range(D):
        var_l = results['models']['legendre']['per_var_variance'][i]
        var_f = results['models']['fourier']['per_var_variance'][i]
        var_h = results['models']['haar']['per_var_variance'][i]
        max_var = max(var_l, var_f, var_h)
        if max_var < 1e-10:
            best = 'negligible'
        elif var_l >= var_f and var_l >= var_h:
            best = 'legendre'
        elif var_f >= var_h:
            best = 'fourier'
        else:
            best = 'haar'

        results['per_variable'][i] = {
            'var_legendre': var_l,
            'var_fourier': var_f,
            'var_haar': var_h,
            'best_basis': best,
        }

    # Overall: which basis has lowest validation RMSE?
    rmse_vals = {bn: results['models'][bn]['rmse_val'] for bn in
                 ['legendre', 'fourier', 'haar']}
    best_overall = min(rmse_vals, key=rmse_vals.get)

    if rmse_vals['haar'] < rmse_vals['legendre'] * 0.9 and \
       rmse_vals['haar'] < rmse_vals['fourier'] * 0.9:
        character = 'localized'
    elif rmse_vals['fourier'] < rmse_vals['legendre'] * 0.95:
        character = 'oscillatory'
    elif rmse_vals['legendre'] < rmse_vals['fourier'] * 0.95:
        character = 'polynomial'
    else:
        character = 'mixed'

    results['summary'] = {
        'best_overall': best_overall,
        'character': character,
        'rmse': rmse_vals,
    }

    if verbose:
        _print_multi_basis_table(results, D)

    return results


# ─────────────────────────────────────────────────────────────
# Mode 2: Cross-residual characterization
# ─────────────────────────────────────────────────────────────

def cross_residual_characterization(
    x_train: jnp.ndarray,
    y_train: jnp.ndarray,
    x_val: jnp.ndarray,
    y_val: jnp.ndarray,
    K_legendre: int = 10,
    K_fourier: int = 10,
    J_haar: int = 5,
    strategy: str = 'variance',
    lambda_legendre: float = 0.001,
    variable_names: Optional[List[str]] = None,
    verbose: bool = True,
) -> dict:
    """Cross-residual characterization: polynomial + oscillatory + localized.

    Step 1: Fit Legendre model (polynomial reference).
    Step 2: Per-variable Fourier projection of residual (oscillatory component).
    Step 3: Per-variable Haar projection of residual (localized component).
    Step 4: Remaining = unexplained.

    The Fourier and Haar fractions are upper bounds (they may overlap).
    For exact accounting, use sequential_projection_characterization().

    Args:
        x_train, y_train: training data, x in [0,1].
        x_val, y_val: validation data.
        K_legendre: max degree for Legendre basis.
        K_fourier: max harmonic for Fourier projection.
        J_haar: max scale for Haar projection.
        strategy: regularization strategy for Legendre fit.
        lambda_legendre: regularization for Legendre fit.
        variable_names: optional names for printing.
        verbose: print characterization table.

    Returns:
        dict with:
          per_variable: {i: {
              'poly_fraction': float,    # within-variable: polynomial share
              'osc_fraction': float,     # within-variable: oscillatory share
              'local_fraction': float,   # within-variable: localized share
              'residual_fraction': float, # within-variable: unexplained share
              'share_of_total': float,   # this variable's share of Var(y)
              'character': str,           # 'polynomial'/'oscillatory'/'localized'/'mixed'/'negligible'
              'poly_var': float,         # absolute variance (polynomial)
              'osc_var': float,          # absolute variance (oscillatory)
              'local_var': float,        # absolute variance (localized)
          }}
          legendre_model: {'w', 'f0', 'rmse_val', 'r_squared'}
          total_variance: float
          residual_variance: float
    """
    N_train, D = x_train.shape
    y_mean = float(jnp.mean(y_train))
    y_c_train = y_train - y_mean
    y_c_val = y_val - y_mean
    var_y = float(jnp.var(y_c_train))

    # Step 1: Fit Legendre model
    phi_L_train = build_first_order_features(x_train, K_legendre,
                                              basis_name='legendre')
    phi_L_val = build_first_order_features(x_val, K_legendre,
                                            basis_name='legendre')
    reg_L = build_regularization_vector(
        D=D, K1=K_legendre, K2=0, P=0, strategy=strategy,
        lambda_order1=lambda_legendre, basis_name='legendre')

    w_L = weighted_ridge_solve(phi_L_train, y_c_train, reg_L)

    pred_L_train = phi_L_train @ w_L
    pred_L_val = phi_L_val @ w_L
    residual_train = y_c_train - pred_L_train
    residual_val = y_c_val - pred_L_val
    rmse_val = float(jnp.sqrt(jnp.mean(residual_val ** 2)))
    r_sq = 1.0 - float(jnp.mean(residual_val ** 2)) / (var_y + 1e-10)
    residual_var = float(jnp.var(residual_train))

    # Per-variable Legendre Sobol
    G_L = build_gram_matrix(K_legendre, basis_name='legendre')
    B_L = basis_size(K_legendre, basis_name='legendre')
    poly_vars = {}
    for i in range(D):
        wi = w_L[i * B_L:(i + 1) * B_L]
        poly_vars[i] = float(wi @ G_L @ wi)

    # Steps 2 & 3: Per-variable Fourier and Haar projection of residual
    B_F = basis_size(K_fourier, basis_name='fourier')
    haar = HaarBasis(J_haar)
    B_H = haar.n_basis

    osc_vars = {}
    local_vars = {}
    for i in range(D):
        # Fourier features for variable i only
        x_i = x_train[:, i:i + 1]
        phi_F_i = build_first_order_features(x_i, K_fourier,
                                              basis_name='fourier')
        # Least-squares projection of residual onto Fourier features
        coeffs_F = jnp.linalg.lstsq(phi_F_i, residual_train, rcond=None)[0]
        osc_pred = phi_F_i @ coeffs_F
        osc_vars[i] = float(jnp.var(osc_pred))

        # Haar features for variable i only
        phi_H_i = haar.evaluate(x_train[:, i])
        coeffs_H = jnp.linalg.lstsq(phi_H_i, residual_train, rcond=None)[0]
        local_pred = phi_H_i @ coeffs_H
        local_vars[i] = float(jnp.var(local_pred))

    # Build per-variable characterization with WITHIN-VARIABLE normalization.
    # For each variable, the four fractions (poly, osc, local, resid) sum to ~1.
    # This answers: "Of variable i's total effect, what % is polynomial?"
    per_variable = {}
    for i in range(D):
        poly_v = poly_vars[i]
        osc_v = osc_vars[i]
        local_v = local_vars[i]

        # Variable's total captured signal across all three basis types
        total_captured = poly_v + osc_v + local_v
        share_of_total = total_captured / (var_y + 1e-10)

        if total_captured < var_y * 0.005:
            # Negligible variable
            per_variable[i] = {
                'poly_fraction': 0.0, 'osc_fraction': 0.0,
                'local_fraction': 0.0, 'residual_fraction': 1.0,
                'share_of_total': share_of_total,
                'character': 'negligible',
                'poly_var': poly_v, 'osc_var': osc_v, 'local_var': local_v,
            }
            continue

        # Within-variable normalization
        poly_f = poly_v / total_captured
        osc_f = osc_v / total_captured
        local_f = local_v / total_captured
        # In cross-residual mode, poly_f + osc_f + local_f = 1.0 by construction
        # (normalization above), so resid_f is always 0.0 here. Kept for API
        # compatibility with sequential_projection_characterization which can
        # produce nonzero residual fractions.
        resid_f = max(0.0, 1.0 - poly_f - osc_f - local_f)

        # Classify character
        if poly_f > 0.8:
            character = 'polynomial'
        elif osc_f > local_f and osc_f > 0.3:
            character = 'oscillatory'
        elif local_f > osc_f and local_f > 0.3:
            character = 'localized'
        elif poly_f > 0.5:
            character = 'polynomial'
        else:
            character = 'mixed'

        per_variable[i] = {
            'poly_fraction': poly_f,
            'osc_fraction': osc_f,
            'local_fraction': local_f,
            'residual_fraction': resid_f,
            'share_of_total': share_of_total,
            'character': character,
            'poly_var': poly_v, 'osc_var': osc_v, 'local_var': local_v,
        }

    result = {
        'per_variable': per_variable,
        'legendre_model': {
            'w': w_L, 'f0': y_mean,
            'rmse_val': rmse_val, 'r_squared': r_sq,
        },
        'total_variance': var_y,
        'residual_variance': residual_var,
    }

    if verbose:
        _print_characterization_table(result, D, variable_names)

    return result


# ─────────────────────────────────────────────────────────────
# Mode 2b: Sequential projection (exact decomposition)
# ─────────────────────────────────────────────────────────────

def sequential_projection_characterization(
    x_train: jnp.ndarray,
    y_train: jnp.ndarray,
    x_val: jnp.ndarray,
    y_val: jnp.ndarray,
    K_legendre: int = 10,
    K_fourier: int = 10,
    J_haar: int = 5,
    strategy: str = 'variance',
    lambda_legendre: float = 0.001,
    variable_names: Optional[List[str]] = None,
    verbose: bool = True,
) -> dict:
    """Exact decomposition via sequential orthogonal projection.

    Order: Legendre first → Fourier (projected ⊥ Legendre) → Haar (projected ⊥ both).
    The variance decomposition is ADDITIVE (sums exactly).

    Unlike cross_residual_characterization(), the Fourier and Haar fractions
    here are exact (non-overlapping).

    Returns same structure as cross_residual_characterization().
    """
    from ..core.projection import project_features_orthogonal

    N_train, D = x_train.shape
    y_mean = float(jnp.mean(y_train))
    y_c_train = y_train - y_mean
    y_c_val = y_val - y_mean
    var_y = float(jnp.var(y_c_train))

    # Step 1: Fit Legendre
    phi_L_train = build_first_order_features(x_train, K_legendre,
                                              basis_name='legendre')
    reg_L = build_regularization_vector(
        D=D, K1=K_legendre, K2=0, P=0, strategy=strategy,
        lambda_order1=lambda_legendre, basis_name='legendre')
    w_L = weighted_ridge_solve(phi_L_train, y_c_train, reg_L)
    pred_L = phi_L_train @ w_L
    residual = y_c_train - pred_L

    # Per-variable Legendre Sobol
    G_L = build_gram_matrix(K_legendre, basis_name='legendre')
    B_L = basis_size(K_legendre, basis_name='legendre')
    poly_vars = {}
    for i in range(D):
        wi = w_L[i * B_L:(i + 1) * B_L]
        poly_vars[i] = float(wi @ G_L @ wi)

    haar = HaarBasis(J_haar)
    per_variable = {}

    for i in range(D):
        # Legendre features for variable i
        phi_L_i = build_first_order_features(
            x_train[:, i:i + 1], K_legendre, basis_name='legendre')

        # Fourier features (WITHOUT linear term — Legendre owns it)
        phi_F_i = build_first_order_features(
            x_train[:, i:i + 1], K_fourier,
            include_linear=False, basis_name='fourier')

        # Project Fourier ⊥ Legendre
        if phi_F_i.shape[1] > 0:
            phi_F_proj, _ = project_features_orthogonal(phi_F_i, phi_L_i)
        else:
            phi_F_proj = phi_F_i

        # Haar features
        phi_H_i = haar.evaluate(x_train[:, i])

        # Project Haar ⊥ [Legendre | Fourier_proj]
        phi_LF_i = jnp.concatenate([phi_L_i, phi_F_proj], axis=1)
        phi_H_proj, _ = project_features_orthogonal(phi_H_i, phi_LF_i)

        # Project residual onto each projected subspace
        coeffs_F = jnp.linalg.lstsq(phi_F_proj, residual, rcond=None)[0]
        osc_var = float(jnp.var(phi_F_proj @ coeffs_F))

        coeffs_H = jnp.linalg.lstsq(phi_H_proj, residual, rcond=None)[0]
        local_var = float(jnp.var(phi_H_proj @ coeffs_H))

        poly_v = poly_vars[i]
        osc_v = osc_var
        local_v = local_var
        total_captured = poly_v + osc_v + local_v
        share_of_total = total_captured / (var_y + 1e-10)

        if total_captured < var_y * 0.005:
            per_variable[i] = {
                'poly_fraction': 0.0, 'osc_fraction': 0.0,
                'local_fraction': 0.0, 'residual_fraction': 1.0,
                'share_of_total': share_of_total, 'character': 'negligible',
                'poly_var': poly_v, 'osc_var': osc_v, 'local_var': local_v,
            }
            continue

        poly_f = poly_v / total_captured
        osc_f = osc_v / total_captured
        local_f = local_v / total_captured
        resid_f = max(0.0, 1.0 - poly_f - osc_f - local_f)

        if poly_f > 0.8:
            character = 'polynomial'
        elif osc_f > local_f and osc_f > 0.3:
            character = 'oscillatory'
        elif local_f > osc_f and local_f > 0.3:
            character = 'localized'
        elif poly_f > 0.5:
            character = 'polynomial'
        else:
            character = 'mixed'

        per_variable[i] = {
            'poly_fraction': poly_f,
            'osc_fraction': osc_f,
            'local_fraction': local_f,
            'residual_fraction': resid_f,
            'share_of_total': share_of_total,
            'character': character,
            'poly_var': poly_v, 'osc_var': osc_v, 'local_var': local_v,
        }

    # Validation RMSE
    phi_L_val = build_first_order_features(x_val, K_legendre,
                                            basis_name='legendre')
    pred_val = phi_L_val @ w_L
    rmse_val = float(jnp.sqrt(jnp.mean((y_c_val - pred_val) ** 2)))
    r_sq = 1.0 - float(jnp.mean((y_c_val - pred_val) ** 2)) / (var_y + 1e-10)

    result = {
        'per_variable': per_variable,
        'legendre_model': {
            'w': w_L, 'f0': y_mean,
            'rmse_val': rmse_val, 'r_squared': r_sq,
        },
        'total_variance': var_y,
        'residual_variance': float(jnp.var(residual)),
        'exact': True,
    }

    if verbose:
        _print_characterization_table(result, D, variable_names)

    return result


# ─────────────────────────────────────────────────────────────
# Automatic per-variable basis selection
# ─────────────────────────────────────────────────────────────

def auto_select_basis(
    characterization: dict,
    poly_threshold: float = 0.8,
    osc_threshold: float = 0.2,
    local_threshold: float = 0.2,
) -> dict:
    """Recommend per-variable basis from characterization results.

    Args:
        characterization: output from cross_residual_characterization()
            or sequential_projection_characterization().
        poly_threshold: fraction of explained variance for "polynomial" label.
        osc_threshold: minimum oscillatory fraction to recommend Fourier.
        local_threshold: minimum localized fraction to recommend Haar.

    Returns:
        dict with:
          per_variable: {i: {
              'basis': str,           # 'legendre', 'fourier', 'haar', 'legendre+haar'
              'reason': str,          # human-readable justification
              'K_recommended': int,   # suggested complexity parameter
          }}
          summary: str  # one-line description
    """
    per_var = characterization['per_variable']
    D = len(per_var)
    recommendations = {}

    for i in range(D):
        info = per_var[i]
        poly_f = info['poly_fraction']
        osc_f = info['osc_fraction']
        local_f = info['local_fraction']
        total = poly_f + osc_f + local_f

        if total < 0.02:
            recommendations[i] = {
                'basis': 'legendre',
                'reason': 'negligible effect — minimal basis',
                'K_recommended': 2,
            }
        elif poly_f > poly_threshold * total:
            # How much polynomial complexity?
            K_rec = max(2, min(10, int(4 + 6 * poly_f)))
            recommendations[i] = {
                'basis': 'legendre',
                'reason': f'polynomial ({poly_f:.0%})',
                'K_recommended': K_rec,
            }
        elif osc_f > local_f and osc_f > osc_threshold * total:
            K_rec = max(3, min(15, int(5 + 10 * osc_f)))
            recommendations[i] = {
                'basis': 'fourier',
                'reason': f'oscillatory ({osc_f:.0%} of residual)',
                'K_recommended': K_rec,
            }
        elif local_f > osc_f and local_f > local_threshold * total:
            if poly_f > 0.3 * total:
                recommendations[i] = {
                    'basis': 'legendre+haar',
                    'reason': f'polynomial ({poly_f:.0%}) + threshold ({local_f:.0%})',
                    'K_recommended': 4,  # J for Haar
                }
            else:
                recommendations[i] = {
                    'basis': 'haar',
                    'reason': f'localized ({local_f:.0%})',
                    'K_recommended': 4,
                }
        else:
            recommendations[i] = {
                'basis': 'legendre',
                'reason': f'mixed — defaulting to polynomial ({poly_f:.0%})',
                'K_recommended': 8,
            }

    # Summary
    basis_counts = {}
    for r in recommendations.values():
        b = r['basis']
        basis_counts[b] = basis_counts.get(b, 0) + 1

    parts = [f"{count} {basis}" for basis, count in
             sorted(basis_counts.items(), key=lambda x: -x[1])]
    summary = f"Recommended: {', '.join(parts)} (of {D} variables)"

    return {
        'per_variable': recommendations,
        'summary': summary,
    }


# ─────────────────────────────────────────────────────────────
# Printing helpers
# ─────────────────────────────────────────────────────────────

def _print_multi_basis_table(results: dict, D: int):
    """Print the multi-basis comparison table."""
    print("\n" + "=" * 70)
    print("THREE-BASIS COMPARISON")
    print("=" * 70)

    # Overall RMSE
    print(f"\n{'Basis':<12} {'RMSE_val':>10} {'R²':>8} {'Features':>10}")
    print("-" * 42)
    for bn in ['legendre', 'fourier', 'haar']:
        m = results['models'][bn]
        K = m['K']
        B = basis_size(K, basis_name=bn)
        nf = D * B
        marker = " <--" if bn == results['summary']['best_overall'] else ""
        print(f"{bn:<12} {m['rmse_val']:10.4f} {m['r_squared']:8.3f} "
              f"{nf:10d}{marker}")

    print(f"\nDataset character: {results['summary']['character']}")

    # Per-variable
    print(f"\n{'Variable':<10} {'Var_L':>8} {'Var_F':>8} {'Var_H':>8} "
          f"{'Best':>12}")
    print("-" * 50)
    for i in range(min(D, 20)):
        pv = results['per_variable'][i]
        print(f"x_{i:<8d} {pv['var_legendre']:8.4f} {pv['var_fourier']:8.4f} "
              f"{pv['var_haar']:8.4f} {pv['best_basis']:>12}")


def _print_characterization_table(results: dict, D: int,
                                   variable_names: Optional[List[str]] = None):
    """Print the per-variable characterization table."""
    exact = results.get('exact', False)
    mode = "EXACT" if exact else "UPPER-BOUND"

    print("\n" + "=" * 78)
    print(f"PER-VARIABLE CHARACTERIZATION ({mode})")
    print("=" * 78)
    print(f"Legendre R² = {results['legendre_model']['r_squared']:.3f}, "
          f"RMSE = {results['legendre_model']['rmse_val']:.4f}")
    print()
    print(f"{'Variable':<12} {'Share':>7} {'Poly':>8} {'Oscill':>8} "
          f"{'Local':>8} {'Character':<15}")
    print("-" * 62)

    for i in range(min(D, 30)):
        pv = results['per_variable'][i]
        name = variable_names[i] if variable_names and i < len(variable_names) \
            else f"x_{i}"
        share = pv.get('share_of_total', 0.0)
        print(f"{name:<12} {share:6.1%} {pv['poly_fraction']:7.0%} "
              f"{pv['osc_fraction']:7.0%} {pv['local_fraction']:7.0%}  "
              f"{pv['character']:<15}")


def print_characterization_table(results: dict,
                                  variable_names: Optional[List[str]] = None):
    """Public API for printing characterization results."""
    D = len(results['per_variable'])
    _print_characterization_table(results, D, variable_names)


def print_basis_recommendations(recommendations: dict,
                                variable_names: Optional[List[str]] = None):
    """Print auto-selected basis recommendations."""
    per_var = recommendations['per_variable']
    D = len(per_var)

    print("\n" + "=" * 70)
    print("AUTOMATIC BASIS SELECTION")
    print("=" * 70)
    print(f"\n{'Variable':<10} {'Basis':<16} {'K':>4}  {'Reason':<35}")
    print("-" * 68)

    for i in range(D):
        r = per_var[i]
        name = variable_names[i] if variable_names and i < len(variable_names) \
            else f"x_{i}"
        print(f"{name:<10} {r['basis']:<16} {r['K_recommended']:4d}  "
              f"{r['reason']:<35}")

    print(f"\n{recommendations['summary']}")
