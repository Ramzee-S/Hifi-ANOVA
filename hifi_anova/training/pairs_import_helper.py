"""Helper for pair selection in the trainer.

Pair selection modes (set via config 'pair_selection'):
  None / 'all'  — All C(D,2) pairs. Full reference option.
  'both'        — Only pairs where BOTH variables have S_i > threshold.
                  Most aggressive. Good when D is large and few vars matter.
  'either'      — Pairs where AT LEAST ONE variable has S_i > threshold.
                  Catches interactions involving any known-active variable.
  'auto'        — Adaptive: picks 'both', 'either', or 'all' based on
                  the resulting F vs N ratio and available memory.
  'bic'         — BIC marginal screening: include variable if BIC improves.
                  Principled (BIC-consistent). Uses ridge leave-one-out.
  'group_lasso' — Group Lasso with BIC on selection path. Gold standard
                  for structured sparsity.
  '1se'         — One standard error rule on CV ridge path. Conservative.
  list[int]     — Explicit list of active variable indices (uses 'both' logic).
"""

import jax.numpy as jnp
import numpy as np
from math import comb
from typing import Optional, List

from ..core.pairs import PairManager, select_active_variables
from ..core.features import basis_size


def _compute_first_order_sobol(D, K1, G1, w1, include_linear=True, basis_name='fourier'):
    """Compute first-order Sobol from Stage A coefficients."""
    G1_f64 = jnp.asarray(G1, dtype=jnp.float64)
    w1_f64 = jnp.asarray(w1, dtype=jnp.float64)
    block = basis_size(K1, include_linear, basis_name)

    variances = {}
    for i in range(D):
        wi = w1_f64[i * block: (i + 1) * block]
        variances[i] = float(jnp.maximum(0.0, wi @ G1_f64 @ wi))

    total = sum(variances.values())
    if total > 0:
        return {i: v / total for i, v in variances.items()}
    return {i: 0.0 for i in range(D)}


def _estimate_F(D, K1, K2, P, include_linear_1=True, include_linear_2=True, basis_name='fourier'):
    """Estimate total feature count."""
    return D * basis_size(K1, include_linear_1, basis_name) + P * basis_size(K2, include_linear_2, basis_name) ** 2


def _resolve_pair_manager(pair_selection, D, K1, K2, G1, w1, N,
                          pair_threshold=0.01, max_pair_variables=None,
                          gpu_memory_gb=4.0, verbose=True,
                          Phi1=None, y_centered=None, reg1=None,
                          strategy='variance', lambda1=0.001,
                          include_linear_1=True, include_linear_2=True,
                          basis_name='fourier'):
    """Create a PairManager based on pair_selection config.

    Args:
        pair_selection: None/'all', 'both', 'either', 'auto',
                       'bic', 'group_lasso', '1se', or list[int]
        D: number of variables
        K1, K2: max harmonics
        G1: first-order Gram matrix
        w1: first-order coefficients from Stage A
        N: number of training samples (for F/N ratio check in auto mode)
        pair_threshold: Sobol threshold for variable activity (old method)
        max_pair_variables: hard cap on active variables
        gpu_memory_gb: GPU memory budget for deciding auto mode
        verbose: print selection info
        Phi1: (N, F1) first-order features (needed for principled methods)
        y_centered: (N,) centered targets (needed for principled methods)
        reg1: (F1,) regularization diagonal (needed for principled methods)
        strategy: regularization strategy (for building reg if not provided)
        lambda1: first-order lambda (for building reg if not provided)
        include_linear_1: whether linear term is included in first-order basis
        include_linear_2: whether linear term is included in second-order basis
        basis_name: basis type ('fourier', 'legendre', 'haar')

    Returns:
        PairManager
    """
    # --- Explicit list ---
    if isinstance(pair_selection, list):
        pm = PairManager(D, active_variables=pair_selection, selection_mode='both')
        if verbose:
            print(f"  Pairs: explicit {len(pair_selection)} vars → {pm.P} pairs")
        return pm

    # --- All pairs ---
    if pair_selection is None or pair_selection == 'all':
        pm = PairManager(D)
        if verbose:
            F = _estimate_F(D, K1, K2, pm.P, include_linear_1, include_linear_2, basis_name)
            print(f"  Pairs: all {pm.P} pairs, F={F}")
        return pm

    # --- Principled selection methods (BIC, Group Lasso, 1SE) ---
    if pair_selection in ('bic', 'group_lasso', '1se'):
        from .selection import select_active_variables_principled
        from .regularization import build_regularization_vector

        # Build features and reg if not provided
        if Phi1 is None or y_centered is None:
            raise ValueError(
                f"pair_selection='{pair_selection}' requires Phi1 and y_centered. "
                f"Pass them from the trainer.")
        if reg1 is None:
            reg1 = np.asarray(build_regularization_vector(
                D, K1, 0, 0, strategy, lambda1, 0.0), dtype=np.float64)

        active, sel_info = select_active_variables_principled(
            Phi1, y_centered, D, K1, reg1,
            method=pair_selection, verbose=verbose,
        )

        if max_pair_variables is not None and len(active) > max_pair_variables:
            active = active[:max_pair_variables]

        pm = PairManager(D, active_variables=active, selection_mode='both')
        if verbose:
            F = _estimate_F(D, K1, K2, pm.P, include_linear_1, include_linear_2, basis_name)
            print(f"  Pairs ({pair_selection}→both): {len(active)}/{D} active vars "
                  f"→ {pm.P} pairs (vs {comb(D,2)} all), F={F}")
        return pm

    # --- Compute first-order Sobol for threshold-based selection ---
    sobol = _compute_first_order_sobol(D, K1, G1, w1, include_linear_1, basis_name)
    active = select_active_variables(sobol, D, threshold=pair_threshold,
                                     max_variables=max_pair_variables)
    D_act = len(active)

    # --- 'both': only pairs of active variables ---
    if pair_selection == 'both':
        pm = PairManager(D, active_variables=active, selection_mode='both')
        if verbose:
            F = _estimate_F(D, K1, K2, pm.P, include_linear_1, include_linear_2, basis_name)
            print(f"  Pairs (both): {D_act}/{D} active vars → {pm.P} pairs "
                  f"(vs {comb(D,2)} all), F={F}")
        return pm

    # --- 'either': pairs with at least one active variable ---
    if pair_selection == 'either':
        pm = PairManager(D, active_variables=active, selection_mode='either')
        if verbose:
            F = _estimate_F(D, K1, K2, pm.P, include_linear_1, include_linear_2, basis_name)
            print(f"  Pairs (either): {D_act}/{D} active vars → {pm.P} pairs "
                  f"(vs {comb(D,2)} all), F={F}")
        return pm

    # --- 'auto': adaptive based on problem size ---
    if pair_selection == 'auto':
        P_all = comb(D, 2)
        P_both = comb(D_act, 2)
        # 'either' count: active paired with everything
        active_set = set(active)
        P_either = sum(1 for i, j in __import__('itertools').combinations(range(D), 2)
                       if i in active_set or j in active_set)

        block2 = basis_size(K2, include_linear_2, basis_name) ** 2
        F1 = D * basis_size(K1, include_linear_1, basis_name)

        # Try each option, pick the most expressive that fits
        options = [
            ('all',    P_all,    F1 + P_all * block2),
            ('either', P_either, F1 + P_either * block2),
            ('both',   P_both,   F1 + P_both * block2),
        ]

        # Criteria: F < 5*N (avoid severe overfitting) AND fits in memory
        chosen = None
        for mode_name, p_count, f_total in options:
            mem_gb = N * f_total * 8 / 1e9  # matmul cost
            fits_memory = mem_gb < gpu_memory_gb * 2  # allow CPU fallback headroom
            fits_ratio = f_total < 5 * N
            if fits_memory and fits_ratio:
                chosen = (mode_name, p_count, f_total)
                break

        if chosen is None:
            # Nothing fits well — use 'both' (most conservative)
            chosen = ('both', P_both, F1 + P_both * block2)

        mode_name, p_count, f_total = chosen
        if mode_name == 'all':
            pm = PairManager(D)
        else:
            pm = PairManager(D, active_variables=active, selection_mode=mode_name)

        if verbose:
            print(f"  Pairs (auto→{mode_name}): {D_act}/{D} active vars → "
                  f"{pm.P} pairs (vs {P_all} all), F={f_total}, "
                  f"F/N={f_total/N:.1f}")
        return pm

    raise ValueError(
        f"Unknown pair_selection: '{pair_selection}'. "
        f"Options: None, 'all', 'both', 'either', 'auto', or list[int]."
    )
