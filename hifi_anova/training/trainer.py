"""Training orchestrator: staged + alternating training.

This is the main entry point for fitting an HiFiANOVA model.
"""

import jax
import jax.numpy as jnp
import equinox as eqx
import numpy as np
from typing import Dict, Optional, Tuple

from ..core.gram import build_gram_matrix, build_gram_matrix_2d, build_gram_matrix_3d
from ..core.features import (
    build_first_order_features, build_second_order_features,
    build_third_order_features,
    build_mixed_first_order_features, build_mixed_second_order_features,
    basis_size, _mixed_include_linear,
)
from ..core.pairs import PairManager, TripleManager
from ..model.mean_model import MeanModel
from ..model.variance_model import VarianceModel
from ..model.hifi_anova import HiFiANOVA
from ..model.residual_net import create_residual_mlp
from .regularization import (build_regularization_vector,
                              build_variance_regularization_vector,
                              build_mixed_regularization_vector)
from .ridge import weighted_ridge_solve
from .newton import newton_solve_log_variance
from .mode import resolve_mode, auto_decide_next_stage


def _prune_first_order_blocks(w_flat, phi_all, reg_full, y_centered,
                              D, block1, G1, method, verbose=True):
    """Zero the first-order block of any variable a group criterion rejects.

    First-order and pair/triple blocks are Hoeffding-orthogonal, so a
    leave-one-group-out test on the full design identifies variables whose
    marginal (first-order) effect is not supported by the data. Their block is
    set to exactly zero — the group-sparse step plain ridge cannot do. Because
    of the orthogonality, zeroing a rejected block barely changes predictions.

    Args:
        w_flat: (F,) fitted coefficients [first-order | pairs | triples].
        phi_all: (N, F) full design used for the fit.
        reg_full: (F,) regularization diagonal matching phi_all.
        y_centered: (N,) centered targets.
        D, block1: number of variables and first-order features per variable.
        G1: (block1, block1) first-order Gram matrix (for weighted norms).
        method: 'bic' | 'group_lasso' | '1se' | 'none'.

    Returns:
        (w_pruned, info) — a float64 numpy copy of w_flat with rejected
        first-order blocks zeroed, plus the selection diagnostics.
    """
    from .selection import prune_groups_postfit

    fo_slices = [slice(i * block1, (i + 1) * block1) for i in range(D)]
    G1_np = np.asarray(G1, dtype=np.float64)
    surviving, info = prune_groups_postfit(
        np.asarray(phi_all, dtype=np.float64),
        np.asarray(y_centered, dtype=np.float64),
        fo_slices, np.asarray(reg_full, dtype=np.float64),
        method=method, gram_matrices=[G1_np] * D,
        group_labels=[f"x{i+1}" for i in range(D)], verbose=verbose,
    )
    pruned = [i for i in range(D) if i not in set(surviving)]
    w_new = np.asarray(w_flat, dtype=np.float64).copy()
    for i in pruned:
        w_new[i * block1:(i + 1) * block1] = 0.0
    info['pruned_variables'] = pruned
    if verbose and pruned:
        print(f"    Zeroed first-order blocks: {[f'x{i+1}' for i in pruned]}")
    return w_new, info


class HiFiANOVATrainer:
    """Orchestrates the staged training procedure.

    Usage:
        # Explicit stages (backward compatible):
        trainer = HiFiANOVATrainer({'stages': ['A', 'B'], ...})

        # Named modes:
        trainer = HiFiANOVATrainer({'mode': 'second', ...})     # = stages A, B
        trainer = HiFiANOVATrainer({'mode': 'full', ...})       # = stages A, B, C
        trainer = HiFiANOVATrainer({'mode': 'heteroscedastic'}) # = stages A, B, D

        # Auto mode (decides stage-by-stage based on residual fraction):
        trainer = HiFiANOVATrainer({'mode': 'auto', 'auto_threshold': 0.01, ...})

    Modes:
        'first'           — First-order Fourier only
        'second'          — First + second-order Fourier
        'full'            — First + second + residual NN
        'heteroscedastic' — First + second + variance decomposition (no NN residual)
        'auto'            — Progressive: adds stages while residual > threshold
    """

    def __init__(self, config: Dict):
        self.config = resolve_mode(config)

    def fit(self, x_train: jnp.ndarray, y_train: jnp.ndarray,
            x_val: jnp.ndarray, y_val: jnp.ndarray,
            key: Optional[jax.Array] = None) -> Tuple:
        """Fit HiFiANOVA following the staged protocol.

        Args:
            x_train: (N, D) training inputs in [0, 1]
            y_train: (N,) training targets
            x_val: (N_val, D) validation inputs
            y_val: (N_val,) validation targets
            key: PRNG key for NN initialization

        Returns:
            (model, results_dict)
        """
        if key is None:
            key = jax.random.PRNGKey(42)

        cfg = self.config
        D = x_train.shape[1]
        K1 = cfg.get('K1', 10)
        K2 = cfg.get('K2', 5)
        K3 = cfg.get('K3', 0)
        Kh = cfg.get('Kh', 3)
        strategy = cfg.get('strategy', 'variance')
        lambda1 = cfg.get('lambda_order1', 0.001)
        lambda2 = cfg.get('lambda_order2', 0.01)
        lambda3 = cfg.get('lambda_order3', 0.1)
        lambda_h = cfg.get('lambda_h', 0.1)
        stages = cfg.get('stages', ['A', 'B', 'C', 'D'])
        auto_mode = cfg.get('_auto_mode', False)
        auto_threshold = cfg.get('_auto_threshold', 0.01)
        var_y_val = float(jnp.var(y_val))

        # --- Selection configuration ---
        # Stage 1: Which variables are active? (after Stage A)
        #   'bic', 'group_lasso', '1se' = principled methods
        #   None = no variable selection (legacy: uses pair_selection)
        variable_selection = cfg.get('variable_selection', None)

        # Stage 2: Which pair candidates to generate? (heuristic)
        #   'all', 'both', 'either' = combinatorial heuristics
        pair_candidates = cfg.get('pair_candidates', None)

        # Stage 3: Post-fit pair pruning (after Stage B fit)
        #   'bic', 'group_lasso', '1se', 'none' = pruning criterion
        pair_pruning = cfg.get('pair_pruning', 'none')

        # Post-fit FIRST-ORDER pruning: zero the whole first-order block of any
        # variable whose marginal effect the criterion deems unsupported.
        #   'bic', 'group_lasso', '1se', 'none'
        # First-order blocks are Hoeffding-orthogonal to the pair/triple blocks,
        # so a leave-one-group-out test on the full design cleanly removes a
        # spurious main effect (e.g. Ishigami x3, which is pure interaction)
        # without perturbing the interactions. Plain ridge can only shrink such a
        # block, never zero it; this is the group-sparse step that can.
        first_order_pruning = cfg.get('first_order_pruning', 'none')

        # Legacy: pair_selection does variable selection + candidate gen in one
        pair_selection = cfg.get('pair_selection', None)
        triple_selection = cfg.get('triple_selection', 'all_active')
        pair_threshold = cfg.get('pair_threshold', 0.01)
        max_pair_variables = cfg.get('max_pair_variables', None)

        # Triple pruning (post-fit)
        triple_pruning = cfg.get('triple_pruning', 'none')

        # Basis configuration
        basis_type = cfg.get('basis_type', 'full')
        basis_name = cfg.get('basis_name', 'fourier')

        # Per-order include_linear control.
        # Three levels of config (highest priority first):
        #   1. Explicit per-order: include_linear_1, include_linear_2, include_linear_3
        #   2. basis_type: 'full' (all True) or 'spectral_higher' (order 1 True, 2+ False)
        #                  'spectral_all' (all False — pure harmonics everywhere)
        #   3. Default: all True
        # For Legendre, include_linear=False drops P̃₁ (the linear polynomial).
        if basis_type == 'spectral_all':
            il1_default, il2_default, il3_default = False, False, False
        elif basis_type == 'spectral_higher':
            il1_default, il2_default, il3_default = True, False, False
        else:  # 'full'
            il1_default, il2_default, il3_default = True, True, True

        include_linear_1 = cfg.get('include_linear_1', il1_default)
        include_linear_2 = cfg.get('include_linear_2', il2_default)
        include_linear_3 = cfg.get('include_linear_3', il3_default)

        # Variance model: separate per-order include_linear
        # Defaults follow mean model settings unless overridden
        include_linear_h1 = cfg.get('include_linear_h1', include_linear_1)
        include_linear_h2 = cfg.get('include_linear_h2', include_linear_2)
        include_linear_h3 = cfg.get('include_linear_h3', include_linear_3)

        results = {}

        # ======== Mixed per-variable basis path ========
        basis_per_variable = cfg.get('basis_per_variable', None)
        if basis_per_variable is not None:
            return self._fit_mixed(
                x_train, y_train, x_val, y_val, key, cfg, D,
                basis_per_variable, stages, strategy, lambda1, lambda2,
                include_linear_h1, include_linear_h2, include_linear_h3,
            )

        # Shared infrastructure
        G1 = build_gram_matrix(K1, include_linear_1, basis_name)
        G2 = (build_gram_matrix_2d(build_gram_matrix(K2, include_linear_2, basis_name))
               if K2 > 0 else None)
        G3 = (build_gram_matrix_3d(build_gram_matrix(K3, include_linear_3, basis_name))
               if K3 > 0 else None)

        # ======== Stage A: First-order only ========
        print("=== Stage A: First-order model ===")
        phi1_train = build_first_order_features(x_train, K1,
                                                  include_linear=include_linear_1,
                                                  basis_name=basis_name)
        phi1_val = build_first_order_features(x_val, K1,
                                                include_linear=include_linear_1,
                                                basis_name=basis_name)

        # Intercept: mean of y
        f0 = float(jnp.mean(y_train))
        y_centered = y_train - f0

        # Ridge solve (first-order only)
        reg1 = build_regularization_vector(D, K1, 0, 0, strategy, lambda1, lambda2,
                                              include_linear_1=include_linear_1,
                                              basis_name=basis_name)
        w1 = weighted_ridge_solve(phi1_train, y_centered, reg1)

        # Build model (Level 0)
        mean_model = MeanModel(
            f0=jnp.array(f0, dtype=jnp.float32),
            w1=jnp.array(w1, dtype=jnp.float32),
            w2=jnp.array([], dtype=jnp.float32),
            w3=jnp.array([], dtype=jnp.float32),
            K1=K1, K2=0, K3=0, D=D,
            include_linear_1=include_linear_1, basis_name=basis_name,
        )

        model = HiFiANOVA(
            mean_model=mean_model,
            K1=K1, K2=0, K3=0, Kh=0, D=D,
            pair_indices=None, triple_indices=None,
            G1=np.array(G1), G2=None, G3=None,
            include_linear_1=include_linear_1,
            include_linear_2=include_linear_2,
            include_linear_3=include_linear_3,
            include_linear_h1=include_linear_h1,
            include_linear_h2=include_linear_h2,
            include_linear_h3=include_linear_h3,
            basis_name=basis_name,
        )

        # Evaluate Stage A
        pred_train_a = f0 + jnp.float32(phi1_train @ w1)
        pred_val_a = f0 + jnp.float32(phi1_val @ w1)
        rmse_train_a = float(jnp.sqrt(jnp.mean((y_train - pred_train_a) ** 2)))
        rmse_val_a = float(jnp.sqrt(jnp.mean((y_val - pred_val_a) ** 2)))
        results['stage_A'] = {'rmse_train': rmse_train_a, 'rmse_val': rmse_val_a}
        print(f"  RMSE train: {rmse_train_a:.4f}, val: {rmse_val_a:.4f}")

        # Auto mode: decide whether to add stage B
        if auto_mode and 'B' not in stages:
            next_s = auto_decide_next_stage(
                'A', rmse_val_a, var_y_val, threshold=auto_threshold)
            if next_s == 'B':
                stages = list(stages) + ['B']

        # First-order pruning for first-order-only models (Stage B handles it
        # otherwise, on the full design).
        stage_b_will_run = ('B' in stages) or ('D' in stages and K2 > 0)
        if first_order_pruning != 'none' and not stage_b_will_run:
            print(f"=== First-order pruning ({first_order_pruning}) ===")
            block1_fo = basis_size(K1, include_linear_1, basis_name)
            w1_pruned, fo_info = _prune_first_order_blocks(
                w1, phi1_train, reg1, y_centered, D, block1_fo, G1,
                first_order_pruning, verbose=True)
            results['first_order_pruning'] = fo_info
            model = eqx.tree_at(lambda m: m.mean_model.w1, model,
                                jnp.asarray(w1_pruned, dtype=jnp.float32))

        # ======== Pair candidate generation for Stage B ========
        pair_mgr = self._generate_pair_candidates(
            x_train, phi1_train, y_centered, w1, reg1, D, K1, K2, G1,
            variable_selection, pair_candidates, pair_selection,
            max_pair_variables, pair_threshold, strategy, lambda1,
            include_linear_1, include_linear_2, basis_name, results)

        # ======== Stage B: First + Second order ========
        if 'B' in stages or ('D' in stages and K2 > 0):
            print(f"=== Stage B: First + second-order model "
                  f"({pair_mgr.P} pairs" +
                  (f" from {len(pair_mgr.active_variables)} active vars"
                   if pair_mgr.active_variables else "") + ") ===")
            # Build second-order features.
            # For large F (>10k features), use numpy to avoid GPU OOM.
            F2_est = pair_mgr.P * basis_size(K2, include_linear_2, basis_name) ** 2
            F_total_est = D * basis_size(K1, include_linear_1, basis_name) + F2_est
            use_numpy = (F_total_est * x_train.shape[0] * 8 / 1e9) > 2.0

            if use_numpy:
                phi2_train = np.array(build_second_order_features(
                    jnp.array(x_train), K2, pair_mgr.pair_indices, include_linear=include_linear_2, basis_name=basis_name))
                phi2_val = np.array(build_second_order_features(
                    jnp.array(x_val), K2, pair_mgr.pair_indices, include_linear=include_linear_2, basis_name=basis_name))
                phi_all_train = jnp.array(np.concatenate(
                    [np.array(phi1_train), phi2_train], axis=1))
                phi_all_val = jnp.array(np.concatenate(
                    [np.array(phi1_val), phi2_val], axis=1))
            else:
                phi2_train = build_second_order_features(x_train, K2, pair_mgr.pair_indices, include_linear=include_linear_2, basis_name=basis_name)
                phi2_val = build_second_order_features(x_val, K2, pair_mgr.pair_indices, include_linear=include_linear_2, basis_name=basis_name)
                phi_all_train = jnp.concatenate([phi1_train, phi2_train], axis=1)
                phi_all_val = jnp.concatenate([phi1_val, phi2_val], axis=1)

            # Combined regularization
            reg_all = build_regularization_vector(
                D, K1, K2, pair_mgr.P, strategy, lambda1, lambda2,
                include_linear_1=include_linear_1,
                include_linear_2=include_linear_2, basis_name=basis_name,
            )

            # Ridge solve
            w_all = weighted_ridge_solve(phi_all_train, y_centered, reg_all)

            # === Post-fit pair pruning ===
            if pair_pruning != 'none' and pair_mgr.P > 0:
                from .selection import prune_groups_postfit
                F1 = D * basis_size(K1, include_linear_1, basis_name)
                block2 = basis_size(K2, include_linear_2, basis_name) ** 2
                # Build group slices for pair blocks (within the second-order portion)
                pair_group_slices = [
                    slice(F1 + p * block2, F1 + (p + 1) * block2)
                    for p in range(pair_mgr.P)
                ]
                pair_labels = [f"({pair_mgr.pair_to_variables(p)[0]},{pair_mgr.pair_to_variables(p)[1]})"
                               for p in range(pair_mgr.P)]

                # Build Gram matrices for Gram-weighted norms
                pair_gram_mats = None
                if G2 is not None:
                    G2_np = np.asarray(G2, dtype=np.float64)
                    pair_gram_mats = [G2_np] * pair_mgr.P

                print(f"  Post-fit pair pruning ({pair_pruning}):")
                surviving_pairs, prune_info = prune_groups_postfit(
                    np.asarray(phi_all_train, dtype=np.float64),
                    np.asarray(y_centered, dtype=np.float64),
                    pair_group_slices,
                    np.asarray(reg_all, dtype=np.float64),
                    method=pair_pruning,
                    gram_matrices=pair_gram_mats,
                    group_labels=pair_labels,
                    verbose=True,
                )
                results['pair_pruning'] = prune_info

                n_pruned = pair_mgr.P - len(surviving_pairs)
                if n_pruned > 0 and len(surviving_pairs) == 0:
                    # All pairs pruned — fall back to first-order only
                    print(f"  All {pair_mgr.P} pairs pruned, reverting to first-order")
                    pair_mgr = PairManager(D)
                    pair_mgr.P = 0
                    pair_mgr.pair_indices = jnp.zeros((0, 2), dtype=jnp.int32)
                    pair_mgr.active_variables = None
                    pair_mgr.selection_mode = 'pruned'
                    phi_all_train = phi1_train
                    phi_all_val = phi1_val
                    reg_all = reg1
                    w_all = weighted_ridge_solve(
                        np.asarray(phi_all_train, dtype=np.float64)
                        if use_numpy else phi_all_train,
                        y_centered, reg_all)
                    K2 = 0  # disable second-order downstream
                elif n_pruned > 0 and len(surviving_pairs) > 0:
                    # Rebuild with only surviving pairs
                    surv_indices = np.stack([
                        np.array(pair_mgr.pair_indices[p]) for p in surviving_pairs
                    ]).astype(np.int32)
                    pair_mgr = PairManager(D)  # dummy, override below
                    pair_mgr.pair_indices = jnp.array(surv_indices)
                    pair_mgr.P = len(surviving_pairs)
                    pair_mgr.active_variables = None
                    pair_mgr.selection_mode = 'pruned'

                    # Rebuild features for surviving pairs only and refit
                    if use_numpy:
                        phi2_train = np.array(build_second_order_features(
                            jnp.array(x_train), K2, pair_mgr.pair_indices,
                            include_linear=include_linear_2, basis_name=basis_name))
                        phi2_val = np.array(build_second_order_features(
                            jnp.array(x_val), K2, pair_mgr.pair_indices,
                            include_linear=include_linear_2, basis_name=basis_name))
                        phi_all_train = jnp.array(np.concatenate(
                            [np.array(phi1_train), phi2_train], axis=1))
                        phi_all_val = jnp.array(np.concatenate(
                            [np.array(phi1_val), phi2_val], axis=1))
                    else:
                        phi2_train = build_second_order_features(
                            x_train, K2, pair_mgr.pair_indices,
                            include_linear=include_linear_2, basis_name=basis_name)
                        phi2_val = build_second_order_features(
                            x_val, K2, pair_mgr.pair_indices,
                            include_linear=include_linear_2, basis_name=basis_name)
                        phi_all_train = jnp.concatenate(
                            [phi1_train, phi2_train], axis=1)
                        phi_all_val = jnp.concatenate(
                            [phi1_val, phi2_val], axis=1)

                    reg_all = build_regularization_vector(
                        D, K1, K2, pair_mgr.P, strategy, lambda1, lambda2,
                        include_linear_1=include_linear_1, include_linear_2=include_linear_2,
                        basis_name=basis_name)
                    w_all = weighted_ridge_solve(phi_all_train, y_centered, reg_all)
                    print(f"  Refit after pruning: {n_pruned} pairs removed, "
                          f"{pair_mgr.P} remaining")

            # === Third-order extension (within Stage B) ===
            triple_mgr = None
            phi3_train = phi3_val = None
            if K3 > 0:
                # Select active variables for triples.
                # For triples, first-order selection is insufficient: a variable
                # may have zero first-order effect but appear in a crucial triple.
                # Strategy:
                #  - 'bic'/'group_lasso'/'1se': use principled first-order selection
                #    for initial candidates, but extend with 'two_active' mode so
                #    triples with 2 known-active + 1 unknown are included
                #  - 'all': all C(D,3) triples
                #  - 'all_active'/'two_active'/'one_active': threshold-based selection
                if triple_selection in ('bic', 'group_lasso', '1se'):
                    from .selection import select_active_variables_principled
                    active_vars, _sel_info = select_active_variables_principled(
                        np.asarray(phi1_train, dtype=np.float64),
                        np.asarray(y_centered, dtype=np.float64),
                        D, K1, np.asarray(reg1, dtype=np.float64),
                        method=triple_selection, verbose=True,
                    )
                    # Use 'two_active' mode: includes triples where 2 of 3 vars
                    # are first-order active, catching hidden third-order variables
                    triple_sel_mode = 'two_active' if len(active_vars) >= 2 else 'all'
                elif triple_selection == 'all':
                    active_vars = list(range(D))
                    triple_sel_mode = 'all'
                else:
                    # Threshold-based fallback
                    from ..analysis.sobol import compute_sobol_indices as _compute_sobol
                    _temp_model = HiFiANOVA(
                        mean_model=MeanModel(
                            f0=jnp.array(f0, dtype=jnp.float32),
                            w1=jnp.array(w_all[:D * basis_size(K1, include_linear_1, basis_name)], dtype=jnp.float32),
                            w2=jnp.array([], dtype=jnp.float32),
                            K1=K1, K2=0, D=D,
                            include_linear_1=include_linear_1, basis_name=basis_name,
                        ),
                        K1=K1, K2=0, K3=0, Kh=0, D=D,
                        G1=np.array(G1),
                        include_linear_1=include_linear_1, basis_name=basis_name,
                    )
                    _sobol = _compute_sobol(_temp_model)
                    _fo = _sobol['mean_sobol']['first_order']
                    from ..core.pairs import select_active_variables
                    active_vars = select_active_variables(_fo, D, threshold=pair_threshold)
                    triple_sel_mode = triple_selection

                triple_mgr = TripleManager(D, active_variables=active_vars,
                                            selection_mode=triple_sel_mode)
                if triple_mgr.T > 0:
                    print(f"  Third-order: {triple_mgr.T} triples "
                          f"from {len(active_vars)} active vars")
                    phi3_train = build_third_order_features(
                        jnp.array(x_train), K3, triple_mgr.triple_indices, include_linear=include_linear_3, basis_name=basis_name)
                    phi3_val = build_third_order_features(
                        jnp.array(x_val), K3, triple_mgr.triple_indices, include_linear=include_linear_3, basis_name=basis_name)

                    # Re-solve with all three orders
                    # Handle numpy/JAX type consistency (phi2 may be numpy
                    # when use_numpy=True for large feature matrices)
                    if use_numpy:
                        phi_all_train = jnp.array(np.concatenate(
                            [np.array(phi1_train), np.array(phi2_train),
                             np.array(phi3_train)], axis=1))
                        phi_all_val = jnp.array(np.concatenate(
                            [np.array(phi1_val), np.array(phi2_val),
                             np.array(phi3_val)], axis=1))
                    else:
                        phi_all_train = jnp.concatenate(
                            [phi1_train, phi2_train, phi3_train], axis=1)
                        phi_all_val = jnp.concatenate(
                            [phi1_val, phi2_val, phi3_val], axis=1)

                    reg_all = build_regularization_vector(
                        D, K1, K2, pair_mgr.P, strategy, lambda1, lambda2,
                        K3=K3, T=triple_mgr.T, lambda_order3=lambda3, include_linear_1=include_linear_1, include_linear_2=include_linear_2, include_linear_3=include_linear_3, basis_name=basis_name,
                    )
                    w_all = weighted_ridge_solve(phi_all_train, y_centered, reg_all)
                    # === Post-fit triple pruning ===
                    if triple_pruning != 'none' and triple_mgr.T > 1:
                        from .selection import prune_groups_postfit
                        from ..core.features import basis_size as _bs
                        F1 = D * _bs(K1, include_linear_1, basis_name)
                        F2 = pair_mgr.P * _bs(K2, include_linear_2, basis_name) ** 2
                        block3 = _bs(K3, include_linear_3, basis_name) ** 3
                        triple_group_slices = [
                            slice(F1 + F2 + t * block3, F1 + F2 + (t + 1) * block3)
                            for t in range(triple_mgr.T)
                        ]
                        triple_labels = [str(triple_mgr.triple_to_variables(t))
                                          for t in range(triple_mgr.T)]

                        print(f"  Post-fit triple pruning ({triple_pruning}):")
                        surviving_triples, tprune_info = prune_groups_postfit(
                            np.asarray(phi_all_train, dtype=np.float64),
                            np.asarray(y_centered, dtype=np.float64),
                            triple_group_slices,
                            np.asarray(reg_all, dtype=np.float64),
                            method=triple_pruning,
                            group_labels=triple_labels,
                            verbose=True,
                        )
                        results['triple_pruning'] = tprune_info

                        n_tpruned = triple_mgr.T - len(surviving_triples)
                        if n_tpruned > 0:
                            surviving_triple_indices = jnp.array(
                                np.array([triple_mgr.triple_indices[t]
                                          for t in surviving_triples], dtype=np.int32))
                            triple_mgr = TripleManager(D)
                            triple_mgr.triple_indices = surviving_triple_indices
                            triple_mgr.T = len(surviving_triples)

                            phi3_train = build_third_order_features(
                                jnp.array(x_train), K3, triple_mgr.triple_indices,
                                include_linear=include_linear_3, basis_name=basis_name)
                            phi3_val = build_third_order_features(
                                jnp.array(x_val), K3, triple_mgr.triple_indices,
                                include_linear=include_linear_3, basis_name=basis_name)

                            if use_numpy:
                                phi_all_train = jnp.array(np.concatenate(
                                    [np.array(phi1_train), np.array(phi2_train),
                                     np.array(phi3_train)], axis=1))
                                phi_all_val = jnp.array(np.concatenate(
                                    [np.array(phi1_val), np.array(phi2_val),
                                     np.array(phi3_val)], axis=1))
                            else:
                                phi_all_train = jnp.concatenate(
                                    [phi1_train, phi2_train, phi3_train], axis=1)
                                phi_all_val = jnp.concatenate(
                                    [phi1_val, phi2_val, phi3_val], axis=1)

                            reg_all = build_regularization_vector(
                                D, K1, K2, pair_mgr.P, strategy, lambda1, lambda2,
                                K3=K3, T=triple_mgr.T, lambda_order3=lambda3,
                                include_linear_1=include_linear_1,
                                include_linear_2=include_linear_2,
                                include_linear_3=include_linear_3,
                                basis_name=basis_name)
                            w_all = weighted_ridge_solve(
                                phi_all_train, y_centered, reg_all)
                            print(f"  Refit: {n_tpruned} triples pruned, "
                                  f"{triple_mgr.T} remaining")

                else:
                    print("  Third-order: no triples selected (too few active vars)")

            # === Post-fit first-order pruning ===
            if first_order_pruning != 'none':
                print(f"  Post-fit first-order pruning ({first_order_pruning}):")
                block1_fo = basis_size(K1, include_linear_1, basis_name)
                w_all, fo_info = _prune_first_order_blocks(
                    w_all, phi_all_train, reg_all, y_centered,
                    D, block1_fo, G1, first_order_pruning, verbose=True)
                w_all = jnp.asarray(w_all, dtype=jnp.float32)
                results['first_order_pruning'] = fo_info

            # Split coefficients (block sizes depend on basis_type)
            from ..core.features import basis_size as _bs
            F1 = D * _bs(K1, include_linear_1, basis_name)
            F2 = pair_mgr.P * _bs(K2, include_linear_2, basis_name) ** 2
            w1_new = w_all[:F1]
            w2_new = w_all[F1:F1 + F2]
            w3_new = w_all[F1 + F2:] if (K3 > 0 and triple_mgr is not None
                                          and triple_mgr.T > 0) else jnp.array([], dtype=jnp.float32)

            # Build model (Level 1, with optional third-order)
            actual_K3 = K3 if (triple_mgr is not None and triple_mgr.T > 0) else 0
            mean_model = MeanModel(
                f0=jnp.array(f0, dtype=jnp.float32),
                w1=jnp.array(w1_new, dtype=jnp.float32),
                w2=jnp.array(w2_new, dtype=jnp.float32),
                w3=jnp.array(w3_new, dtype=jnp.float32),
                K1=K1, K2=K2, K3=actual_K3, D=D,
                include_linear_1=include_linear_1,
                include_linear_2=include_linear_2, include_linear_3=include_linear_3,
                basis_name=basis_name,
            )

            model = HiFiANOVA(
                mean_model=mean_model,
                K1=K1, K2=K2, K3=actual_K3, Kh=0, D=D,
                pair_indices=np.array(pair_mgr.pair_indices),
                triple_indices=(np.array(triple_mgr.triple_indices)
                                if triple_mgr is not None and triple_mgr.T > 0
                                else None),
                G1=np.array(G1),
                G2=np.array(G2) if G2 is not None else None,
                G3=np.array(G3) if G3 is not None and actual_K3 > 0 else None,
                include_linear_1=include_linear_1,
                include_linear_2=include_linear_2,
                include_linear_3=include_linear_3,
                include_linear_h1=include_linear_h1,
                include_linear_h2=include_linear_h2,
                include_linear_h3=include_linear_h3,
                basis_name=basis_name,
            )

            # Evaluate Stage B
            pred_train_b = f0 + jnp.float32(phi_all_train @ w_all)
            pred_val_b = f0 + jnp.float32(phi_all_val @ w_all)
            rmse_train_b = float(jnp.sqrt(jnp.mean((y_train - pred_train_b) ** 2)))
            rmse_val_b = float(jnp.sqrt(jnp.mean((y_val - pred_val_b) ** 2)))
            results['stage_B'] = {
                'rmse_train': rmse_train_b, 'rmse_val': rmse_val_b,
                'n_triples': triple_mgr.T if triple_mgr is not None else 0,
            }
            print(f"  RMSE train: {rmse_train_b:.4f}, val: {rmse_val_b:.4f}")
        elif 'C' not in stages and 'D' not in stages:
            return model, results

        # Auto mode: decide whether to add stage C (NN)
        if auto_mode and 'C' not in stages:
            best_rmse = results.get('stage_B', results['stage_A'])['rmse_val']
            next_s = auto_decide_next_stage(
                'B', best_rmse, var_y_val, threshold=auto_threshold)
            if next_s == 'C':
                stages = list(stages) + ['C']
                if 'residual_nn' not in cfg:
                    cfg['residual_nn'] = {'enabled': True}
                else:
                    cfg['residual_nn']['enabled'] = True

        # ======== Stage C: Residual (Linear or NN) ========
        if 'C' in stages:
            model, key = self._fit_stage_c(
                model, x_train, y_train, x_val, y_val, key, cfg, D, results)

        # Auto mode: decide whether to add stage D (heteroscedastic)
        if auto_mode and 'D' not in stages:
            # Check if residual variance correlates with inputs
            pred_current = model.predict_mean_only(x_val)
            r_val = np.array(y_val - pred_current)
            r2_val = r_val ** 2
            x_val_np = np.array(x_val)
            max_var_corr = 0.0
            for i in range(D):
                c = np.corrcoef(r2_val, x_val_np[:, i])[0, 1]
                if np.isfinite(c):
                    max_var_corr = max(max_var_corr, abs(c))
            best_rmse = results.get('stage_C', results.get('stage_B', results['stage_A']))['rmse_val']
            next_s = auto_decide_next_stage(
                'C', best_rmse, var_y_val, max_var_corr=max_var_corr,
                threshold=auto_threshold)
            if next_s == 'D':
                stages = list(stages) + ['D']
                if Kh == 0:
                    Kh = 3

        if 'D' not in stages:
            return model, results

        # ======== Stage D: Heteroscedastic variance ========
        if Kh > 0:
            print("=== Stage D: Heteroscedastic variance ===")
            model = self._fit_heteroscedastic(
                model, x_train, y_train, x_val, y_val,
                pair_mgr, G1, G2, K1, K2, Kh, D,
                strategy, lambda1, lambda2, lambda_h, results
            )

        # Enforce first-order pruning through any Stage C/D mean refit: the
        # heteroscedastic (Stage D) alternating solve re-estimates the mean
        # coefficients, so re-zero the rejected first-order blocks on the final
        # model. First-order/pair orthogonality makes this a no-op for the
        # interactions.
        pruned_fo = results.get('first_order_pruning', {}).get(
            'pruned_variables', [])
        if pruned_fo:
            block1_fo = basis_size(K1, include_linear_1, basis_name)
            w1_cur = np.asarray(model.mean_model.w1, dtype=np.float64).copy()
            for i in pruned_fo:
                w1_cur[i * block1_fo:(i + 1) * block1_fo] = 0.0
            model = eqx.tree_at(lambda m: m.mean_model.w1, model,
                                jnp.asarray(w1_cur, dtype=jnp.float32))

        return model, results

    def _generate_pair_candidates(self, x_train, phi1_train, y_centered,
                                 w1, reg1, D, K1, K2, G1,
                                 variable_selection, pair_candidates,
                                 pair_selection, max_pair_variables,
                                 pair_threshold, strategy, lambda1,
                                 include_linear_1, include_linear_2,
                                 basis_name, results):
        """Select active variables and generate Stage-B candidate pairs.

        Extracted verbatim from ``fit`` (behavior-preserving). Returns the
        PairManager; records variable-selection diagnostics in ``results``.
        """
        # ======== Pair candidate generation for Stage B ========
        # Two-step pipeline:
        #   1. Select active variables (variable_selection criterion)
        #   2. Generate candidate pairs (pair_candidates heuristic)
        # After fitting, optional pair_pruning removes inactive pairs.
        from .pairs_import_helper import _resolve_pair_manager
        N_train = x_train.shape[0]

        if variable_selection is not None and pair_candidates is not None:
            # New-style config: explicit separation of variable selection
            # and candidate generation
            from .selection import select_active_variables_principled
            active_vars, var_sel_info = select_active_variables_principled(
                np.asarray(phi1_train, dtype=np.float64),
                np.asarray(y_centered, dtype=np.float64),
                D, K1, np.asarray(reg1, dtype=np.float64),
                method=variable_selection, verbose=True,
            )
            if max_pair_variables is not None and len(active_vars) > max_pair_variables:
                active_vars = active_vars[:max_pair_variables]
            results['variable_selection'] = var_sel_info

            pair_mgr = PairManager(D, active_variables=active_vars,
                                    selection_mode=pair_candidates)
            if True:  # verbose
                from math import comb
                F_est = D * basis_size(K1, include_linear_1, basis_name) + pair_mgr.P * basis_size(K2, include_linear_2, basis_name)**2
                print(f"  Variable selection ({variable_selection}): "
                      f"{len(active_vars)}/{D} active")
                print(f"  Pair candidates ({pair_candidates}): "
                      f"{pair_mgr.P} pairs (vs {comb(D,2)} all), F={F_est}")
        else:
            # Legacy config: pair_selection handles everything
            pair_mgr = _resolve_pair_manager(
                pair_selection, D, K1, K2, G1, w1, N_train,
                pair_threshold, max_pair_variables,
                Phi1=phi1_train, y_centered=y_centered, reg1=reg1,
                strategy=strategy, lambda1=lambda1,
                include_linear_1=include_linear_1, include_linear_2=include_linear_2,
                basis_name=basis_name)
        return pair_mgr

    def _fit_stage_c(self, model, x_train, y_train, x_val, y_val, key, cfg,
                     D, results):
        """Stage C: fit a residual model (linear RBF/RFF/Nystrom, or NN).

        Extracted verbatim from ``fit`` (behavior-preserving). Returns the
        updated model and PRNG key; records Stage-C diagnostics in
        ``results`` in place.
        """
        # Support both old config key (residual_nn) and new (residual)
        residual_cfg = cfg.get('residual', cfg.get('residual_nn', {}))
        # A bare string (e.g. residual='rbf') is shorthand for {'type': ...}.
        if isinstance(residual_cfg, str):
            residual_cfg = {'type': residual_cfg}
        residual_type = residual_cfg.get('type', 'nn')

        # Fail loudly on an unknown residual type: otherwise Stage C would
        # silently no-op (no branch matches) and the user would get a model
        # with no residual while believing one was fitted.
        KNOWN_RESIDUAL_TYPES = ('nn', 'rbf', 'rff', 'nystrom')
        if residual_type not in KNOWN_RESIDUAL_TYPES:
            raise ValueError(
                f"Unknown residual type {residual_type!r}; expected one of "
                f"{KNOWN_RESIDUAL_TYPES}. Check the 'residual' config "
                f"(got {residual_cfg!r})."
            )

        # Backward compat: old config uses 'enabled' flag for NN
        if residual_type == 'nn' and not residual_cfg.get('enabled', False):
            pass  # Skip Stage C if NN not enabled
        elif residual_type in ('rbf', 'rff', 'nystrom'):
            # === ANALYTIC PIPELINE (linear residual) ===
            print(f"=== Stage C: Linear residual ({residual_type}) ===")
            from .analytic_residual import fit_linear_residual

            key, subkey = jax.random.split(key)
            lambda_res = residual_cfg.get('lambda_residual',
                         cfg.get('lambda_residual', 1.0))

            model, stage_c_results = fit_linear_residual(
                model, x_train, y_train, x_val, y_val,
                residual_type=residual_type,
                residual_config=residual_cfg,
                lambda_residual=lambda_res,
                key=subkey,
            )
            results['stage_C'] = stage_c_results

        elif residual_type == 'nn' and residual_cfg.get('enabled', False):
            # === SGD PIPELINE (NN residual, unchanged) ===
            print("=== Stage C: Residual NN ===")
            from .sgd import train_residual_nn

            key, subkey = jax.random.split(key)
            hidden_dims = residual_cfg.get('hidden_dims', [256, 256, 256])
            nn = create_residual_mlp(D, hidden_dims, subkey)

            model = eqx.tree_at(lambda m: m.residual_net, model, nn,
                            is_leaf=lambda x: x is None)

            model = train_residual_nn(
                model, x_train, y_train, x_val, y_val,
                lr=residual_cfg.get('lr', 0.001),
                weight_decay=residual_cfg.get('weight_decay', 0.0001),
                epochs=residual_cfg.get('epochs', 200),
                batch_size=residual_cfg.get('batch_size', 512),
                patience=residual_cfg.get('patience', 20),
                key=subkey,
            )

            # Evaluate Stage C
            pred_val_c = model.predict_mean_only(x_val)
            rmse_val_c = float(jnp.sqrt(jnp.mean((y_val - pred_val_c) ** 2)))
            results['stage_C'] = {'rmse_val': rmse_val_c}
            print(f"  RMSE val: {rmse_val_c:.4f}")
        return model, key

    # ================================================================
    # Mixed per-variable basis path
    # ================================================================

    def _fit_mixed(
        self, x_train, y_train, x_val, y_val, key, cfg, D,
        basis_per_variable, stages, strategy, lambda1, lambda2,
        include_linear_h1, include_linear_h2, include_linear_h3,
    ):
        """Fit with mixed per-variable basis assignment.

        Each variable uses its own basis (legendre/fourier/haar) with its own K.
        In mixed mode:
          - Legendre: K features, includes linear (P̃₁)
          - Fourier: 2K features, NO linear (cos/sin only)
          - Haar: 2^K-1 features, no linear

        Args:
            basis_per_variable: dict {var_idx: {'basis': str, 'K': int}} or 'auto'.
        """
        results = {}

        # --- Resolve var_specs ---
        if basis_per_variable == 'auto':
            from ..analysis.basis_characterization import (
                cross_residual_characterization, auto_select_basis)
            print("=== Mixed basis: auto-characterization ===")
            char = cross_residual_characterization(
                x_train, y_train, x_val, y_val,
                K_legendre=cfg.get('K1', 5),
                K_fourier=cfg.get('K1', 5),
                J_haar=cfg.get('K1', 4),
                strategy=strategy,
                lambda_legendre=lambda1,
                verbose=True)
            rec = auto_select_basis(char)
            var_specs_dict = []
            for i in range(D):
                r = rec['per_variable'][i]
                basis = r['basis'].split('+')[0]  # 'legendre+haar' → 'legendre'
                var_specs_dict.append({'basis': basis, 'K': r['K_recommended']})
            print(f"  Auto-selected: {rec['summary']}")
        else:
            # dict {i: {'basis': str, 'K': int}} — fill defaults for missing vars
            default_basis = cfg.get('basis_name', 'legendre')
            default_K = cfg.get('K1', 5)
            var_specs_dict = []
            for i in range(D):
                if i in basis_per_variable:
                    spec = basis_per_variable[i]
                    var_specs_dict.append({
                        'basis': spec.get('basis', default_basis),
                        'K': spec.get('K', default_K),
                    })
                else:
                    var_specs_dict.append({'basis': default_basis, 'K': default_K})

        # --- Stage A: Mixed first-order ---
        print("=== Stage A: Mixed first-order model ===")
        basis_summary = {}
        for spec in var_specs_dict:
            b = spec['basis']
            basis_summary[b] = basis_summary.get(b, 0) + 1
        summary_str = ', '.join(f"{c} {b}" for b, c in
                                 sorted(basis_summary.items(), key=lambda x: -x[1]))
        print(f"  Per-variable: {summary_str}")

        phi1_train, block_info = build_mixed_first_order_features(
            x_train, var_specs_dict)
        phi1_val, _ = build_mixed_first_order_features(x_val, var_specs_dict)

        F1 = phi1_train.shape[1]
        print(f"  Total first-order features: {F1}")

        f0 = float(jnp.mean(y_train))
        y_centered = y_train - f0

        # Ridge solve
        reg1 = build_mixed_regularization_vector(
            var_specs_dict, strategy=strategy, lambda_order1=lambda1)
        w1 = weighted_ridge_solve(phi1_train, y_centered, reg1)

        # Convert block_info to static tuple for model storage
        var_specs_tuple = block_info  # already a tuple of tuples

        # Build per-variable G1 (average for backward compat with G1 field)
        # For mixed mode, we store G1=None and use var_specs for per-variable Gram
        G1_first = build_gram_matrix(
            var_specs_dict[0]['K'],
            _mixed_include_linear(var_specs_dict[0]['basis']),
            var_specs_dict[0]['basis'])

        mean_model = MeanModel(
            f0=jnp.array(f0, dtype=jnp.float32),
            w1=jnp.array(w1, dtype=jnp.float32),
            w2=jnp.array([], dtype=jnp.float32),
            w3=jnp.array([], dtype=jnp.float32),
            K1=0, K2=0, K3=0, D=D,
            include_linear_1=True,
            basis_name='mixed',
            var_specs=var_specs_tuple,
        )

        model = HiFiANOVA(
            mean_model=mean_model,
            K1=0, K2=0, K3=0, Kh=0, D=D,
            pair_indices=None, triple_indices=None,
            G1=np.array(G1_first), G2=None, G3=None,
            include_linear_1=True,
            include_linear_h1=include_linear_h1,
            include_linear_h2=include_linear_h2,
            include_linear_h3=include_linear_h3,
            basis_name='mixed',
            var_specs=var_specs_tuple,
        )

        pred_train = f0 + jnp.float32(phi1_train @ w1)
        pred_val = f0 + jnp.float32(phi1_val @ w1)
        rmse_train = float(jnp.sqrt(jnp.mean((y_train - pred_train) ** 2)))
        rmse_val = float(jnp.sqrt(jnp.mean((y_val - pred_val) ** 2)))
        results['stage_A'] = {'rmse_train': rmse_train, 'rmse_val': rmse_val}
        print(f"  RMSE train: {rmse_train:.4f}, val: {rmse_val:.4f}")

        # --- Stage B: Mixed second-order ---
        if 'B' in stages:
            K2_mixed = cfg.get('K2', 3)
            print(f"=== Stage B: Mixed second-order ===")

            # Generate pairs from all active variables
            pair_mgr = PairManager(D)
            pair_indices = pair_mgr.pair_indices
            P = pair_mgr.P

            # Build mixed second-order features
            phi2_train, pair_bi = build_mixed_second_order_features(
                x_train, pair_indices, var_specs_dict)
            phi2_val, _ = build_mixed_second_order_features(
                x_val, pair_indices, var_specs_dict)

            F2 = phi2_train.shape[1]
            print(f"  {P} pairs, {F2} second-order features")

            # Concatenate [phi1 | phi2] and solve jointly
            phi_full_train = jnp.concatenate([phi1_train, phi2_train], axis=1)
            phi_full_val = jnp.concatenate([phi1_val, phi2_val], axis=1)

            reg_full = build_mixed_regularization_vector(
                var_specs_dict, strategy=strategy,
                lambda_order1=lambda1,
                pair_indices=pair_indices,
                lambda_order2=lambda2)

            w_full = weighted_ridge_solve(phi_full_train, y_centered, reg_full)

            w1 = w_full[:F1]
            w2 = w_full[F1:]

            pair_block_info_tuple = pair_bi

            mean_model = MeanModel(
                f0=jnp.array(f0, dtype=jnp.float32),
                w1=jnp.array(w1, dtype=jnp.float32),
                w2=jnp.array(w2, dtype=jnp.float32),
                w3=jnp.array([], dtype=jnp.float32),
                K1=0, K2=K2_mixed, K3=0, D=D,
                include_linear_1=True,
                basis_name='mixed',
                var_specs=var_specs_tuple,
                pair_block_info=pair_block_info_tuple,
            )

            model = HiFiANOVA(
                mean_model=mean_model,
                K1=0, K2=K2_mixed, K3=0, Kh=0, D=D,
                pair_indices=np.array(pair_indices),
                triple_indices=None,
                G1=np.array(G1_first), G2=None, G3=None,
                include_linear_1=True,
                include_linear_h1=include_linear_h1,
                include_linear_h2=include_linear_h2,
                include_linear_h3=include_linear_h3,
                basis_name='mixed',
                var_specs=var_specs_tuple,
                pair_block_info=pair_block_info_tuple,
            )

            pred_val = f0 + jnp.float32(phi_full_val @ w_full)
            rmse_val = float(jnp.sqrt(jnp.mean((y_val - pred_val) ** 2)))
            pred_train = f0 + jnp.float32(phi_full_train @ w_full)
            rmse_train = float(jnp.sqrt(jnp.mean((y_train - pred_train) ** 2)))
            results['stage_B'] = {'rmse_train': rmse_train, 'rmse_val': rmse_val}
            print(f"  RMSE train: {rmse_train:.4f}, val: {rmse_val:.4f}")

        results['mixed_basis'] = True
        results['var_specs'] = var_specs_dict
        return model, results

    def _fit_heteroscedastic(
        self, model, x_train, y_train, x_val, y_val,
        pair_mgr, G1, G2, K1, K2, Kh, D,
        strategy, lambda1, lambda2, lambda_h, results
    ):
        """Alternating optimization for heteroscedastic model.

        Correctly accounts for:
        - Third-order features (if K3 > 0)
        - Linear/NN residual prediction (if Stage C was run)
        - Weighted intercept f0 (recomputed each iteration)
        """
        cfg = self.config
        max_outer = cfg.get('max_outer_iter', 10)
        tol = cfg.get('alternating_tol', 1e-4)
        newton_max = cfg.get('newton_max_iter', 10)
        K3 = model.K3
        basis_name = getattr(model, 'basis_name', 'fourier')

        # --- Mean features (all orders) ---
        _il1 = getattr(model, 'include_linear_1', True)
        _il2 = getattr(model, 'include_linear_2', True)
        _il3 = getattr(model, 'include_linear_3', True)
        phi1_train = build_first_order_features(x_train, K1,
                                                  include_linear=_il1, basis_name=basis_name)
        phi2_train = (build_second_order_features(x_train, K2, pair_mgr.pair_indices, include_linear=_il2, basis_name=basis_name)
                      if K2 > 0 else None)
        phi3_train = (build_third_order_features(x_train, K3, model.triple_indices, include_linear=_il3, basis_name=basis_name)
                      if K3 > 0 and model.triple_indices is not None else None)

        phi_all_train = phi1_train
        if phi2_train is not None:
            phi_all_train = jnp.concatenate([phi_all_train, phi2_train], axis=1)
        if phi3_train is not None:
            phi_all_train = jnp.concatenate([phi_all_train, phi3_train], axis=1)

        # --- Variance features (first-order + optional second-order + optional residual) ---
        from ..core.features import basis_size as _bs
        K2h = cfg.get('K2h', 0)
        lambda_h2 = cfg.get('lambda_h2', lambda_h * 10)
        _ilh1 = cfg.get('include_linear_h1', _il1)
        _ilh2 = cfg.get('include_linear_h2', _il2)
        _ilh3 = cfg.get('include_linear_h3', _il3)
        Fh = D * _bs(Kh, _ilh1, basis_name)
        psi1_train = build_first_order_features(x_train, Kh,
                                                  include_linear=_ilh1,
                                                  basis_name=basis_name)

        # Second-order variance features (optional)
        # Two modes:
        #   var_pair_selection=None/'all': use ALL C(D,2) pairs (all-at-once mode)
        #   var_pair_selection='auto':     quick first-order variance fit to select
        #                                  variance-active variables, then pairs
        psi2_train = None
        var_pair_mgr = None
        var_pair_selection = cfg.get('var_pair_selection', None)
        if K2h > 0:
            if var_pair_selection in (None, 'all'):
                # All-at-once: use all pairs, no selection
                var_pair_mgr = PairManager(D)
            elif var_pair_selection == 'auto':
                # Sequential: quick first-order variance fit → select pairs
                from ..core.pairs import select_active_variables
                from ..core.gram import build_gram_matrix as _bgm

                _mean_pred = model.mean_model.predict(
                    phi1_train, phi2_train, phi3_train)
                if model.residual_net is not None:
                    _res = jax.vmap(model.residual_net)(x_train)
                    if _res.ndim > 1:
                        _res = _res.squeeze(-1)
                    _mean_pred = _mean_pred + _res
                _r2_init = (y_train - _mean_pred) ** 2

                _reg_h1 = build_variance_regularization_vector(
                    D, Kh, strategy, lambda_h,
                    include_linear_h1=_ilh1, basis_name=basis_name)
                _wh_init, _ = newton_solve_log_variance(
                    psi1_train, _r2_init, jnp.zeros(Fh, dtype=jnp.float64),
                    float(jnp.log(jnp.mean(_r2_init))), _reg_h1, max_iter=5)

                # Variance Sobol from quick fit → select active vars
                _Gh = jnp.asarray(_bgm(Kh, _ilh1, basis_name), dtype=jnp.float64)
                _var_sobol = {}
                _bh = _bs(Kh, _ilh1, basis_name)
                for i in range(D):
                    _wi = _wh_init[i * _bh: (i + 1) * _bh]
                    _var_sobol[i] = float(jnp.maximum(0.0, _wi @ _Gh @ _wi))
                _total = sum(_var_sobol.values())
                if _total > 0:
                    _var_sobol = {i: v / _total for i, v in _var_sobol.items()}

                var_active = select_active_variables(
                    _var_sobol, D, threshold=cfg.get('pair_threshold', 0.01))
                var_pair_mgr = PairManager(
                    D, active_variables=var_active, selection_mode='both')
            else:
                # Explicit list or other mode
                var_pair_mgr = PairManager(D)

            if var_pair_mgr.P > 0:
                psi2_train = build_second_order_features(
                    x_train, K2h, var_pair_mgr.pair_indices,
                    include_linear=_ilh2, basis_name=basis_name)
                mode_str = var_pair_selection or 'all'
                print(f"  Variance second-order: {var_pair_mgr.P} pairs "
                      f"(K2h={K2h}, mode={mode_str})")
            else:
                K2h = 0
                print(f"  Variance second-order: skipped (no pairs)")

        # Third-order variance features (optional, for small D)
        K3h = cfg.get('K3h', 0)
        lambda_h3 = cfg.get('lambda_h3', lambda_h * 100)
        psi3_train = None
        var_triple_mgr = None
        if K3h > 0:
            # Use all triples by default (small D assumed)
            var_triple_selection = cfg.get('var_triple_selection', None)
            if var_triple_selection in (None, 'all'):
                var_triple_mgr = TripleManager(D)
            else:
                # Could add 'auto' mode here like pairs
                var_triple_mgr = TripleManager(D)

            if var_triple_mgr.T > 0:
                psi3_train = build_third_order_features(
                    x_train, K3h, var_triple_mgr.triple_indices,
                    include_linear=_ilh3, basis_name=basis_name)
                print(f"  Variance third-order: {var_triple_mgr.T} triples (K3h={K3h})")
            else:
                K3h = 0

        # Optional: variance residual features (RBF/RFF for higher-order noise)
        var_residual_cfg = cfg.get('variance_residual', None)
        z_h_proj_train = None
        z_h_proj_coeffs = None
        var_residual_model = None
        M_h = 0
        lambda_h_res = cfg.get('lambda_h_residual', lambda_h * 10)

        if var_residual_cfg is not None:
            from .analytic_residual import create_residual
            from ..core.projection import project_features_orthogonal
            var_residual_type = var_residual_cfg.get('type', 'rbf')
            var_res_cfg = dict(var_residual_cfg)
            var_res_cfg.setdefault('sigma', 0.3)
            var_res_cfg.setdefault('n_centers', 150)
            var_residual_model = create_residual(
                var_residual_type, var_res_cfg, x_train, D)
            z_h_train = var_residual_model.build_features(x_train)
            M_h = z_h_train.shape[1]
            # Project against ALL variance Fourier features [psi1 | psi2 | psi3]
            psi_fourier_train = psi1_train
            if psi2_train is not None:
                psi_fourier_train = jnp.concatenate(
                    [psi_fourier_train, psi2_train], axis=1)
            if psi3_train is not None:
                psi_fourier_train = jnp.concatenate(
                    [psi_fourier_train, psi3_train], axis=1)
            z_h_proj_train, z_h_proj_coeffs = project_features_orthogonal(
                z_h_train, psi_fourier_train)
            z_h_proj_train = jnp.asarray(z_h_proj_train, dtype=jnp.float64)
            print(f"  Variance residual ({var_residual_type}): {M_h} features")

        _incl_lin_2 = getattr(model, 'include_linear_2', True)
        _incl_lin_3 = getattr(model, 'include_linear_3', True)

        F1 = D * _bs(K1, _il1, basis_name)
        F2 = pair_mgr.P * _bs(K2, _incl_lin_2, basis_name) ** 2 if K2 > 0 else 0
        Ph = var_pair_mgr.P if var_pair_mgr is not None else 0
        Th = var_triple_mgr.T if var_triple_mgr is not None else 0

        # --- Regularization ---
        T = model.triple_indices.shape[0] if model.triple_indices is not None else 0
        lambda3 = cfg.get('lambda_order3', 0.1)
        reg_mean = build_regularization_vector(
            D, K1, K2, pair_mgr.P, strategy, lambda1, lambda2,
            K3=K3, T=T, lambda_order3=lambda3,
            include_linear_1=_il1, include_linear_2=_incl_lin_2, include_linear_3=_incl_lin_3,
            basis_name=basis_name,
        )
        reg_var = build_variance_regularization_vector(
            D, Kh, strategy, lambda_h,
            K2h=K2h, Ph=Ph, lambda_h2=lambda_h2,
            K3h=K3h, Th=Th, lambda_h3=lambda_h3,
            M_h_residual=M_h, lambda_h_res=lambda_h_res,
            include_linear_h1=_ilh1, include_linear_h2=_ilh2, include_linear_h3=_ilh3,
            basis_name=basis_name,
        )

        # --- Residual prediction from Stage C (frozen during Stage D) ---
        residual_pred_train = jnp.zeros(x_train.shape[0])
        if model.residual_net is not None:
            res_out = jax.vmap(model.residual_net)(x_train)
            if res_out.ndim > 1:
                res_out = res_out.squeeze(-1)
            residual_pred_train = res_out
            print(f"  Including residual prediction (var={float(jnp.var(res_out)):.6f})")

        # --- Initialize ---
        f0 = float(model.mean_model.f0)
        y_centered = y_train - f0

        # Initial full mean prediction: Fourier + residual
        fourier_pred = model.mean_model.predict(phi1_train, phi2_train, phi3_train)
        full_mean_pred = fourier_pred + residual_pred_train
        residuals = y_train - full_mean_pred
        r2 = residuals ** 2

        h0_init = float(jnp.log(jnp.mean(r2)))

        # Augmented variance features: [psi1 | psi2 | psi3 | z_h_proj]
        psi_parts = [jnp.asarray(psi1_train, dtype=jnp.float64)]
        if psi2_train is not None:
            psi_parts.append(jnp.asarray(psi2_train, dtype=jnp.float64))
        if psi3_train is not None:
            psi_parts.append(jnp.asarray(psi3_train, dtype=jnp.float64))
        if z_h_proj_train is not None:
            psi_parts.append(z_h_proj_train)
        psi_all_train = jnp.concatenate(psi_parts, axis=1) if len(psi_parts) > 1 else psi_parts[0]

        Fh_total = psi_all_train.shape[1]
        w_h = jnp.zeros(Fh_total, dtype=jnp.float64)

        prev_loss = float('inf')

        # Initialize the log-variance intercept so `h0` is always bound before
        # the outer loop's `h0_init if outer == 0 else h0` could read it. Under
        # valid configs (max_outer >= 1) outer==0 uses h0_init and this is a
        # no-op; it removes the read-before-assignment that static analysis flags.
        h0 = h0_init

        for outer in range(max_outer):
            # --- Variance update (Newton on augmented features) ---
            w_h, h0 = newton_solve_log_variance(
                psi_all_train, r2, w_h, h0_init if outer == 0 else h0,
                reg_var, max_iter=newton_max
            )

            # Compute weights for weighted ridge
            h_pred = h0 + jnp.float64(psi_all_train) @ w_h
            sigma2 = jnp.exp(h_pred)
            weights = 1.0 / sigma2

            # --- Mean update (weighted ridge on y - residual_pred) ---
            # The residual_net prediction is frozen; Fourier coefficients adapt
            y_for_fourier = y_train - residual_pred_train

            # Recompute weighted intercept: f0 = Σ w_n y_n / Σ w_n
            w_sum = jnp.sum(weights)
            f0 = float(jnp.sum(weights * y_for_fourier) / w_sum)
            y_centered = y_for_fourier - f0

            w_all = weighted_ridge_solve(phi_all_train, y_centered, reg_mean, weights)

            # Update predictions and residuals (include residual_net)
            fourier_pred = f0 + phi_all_train @ jnp.float32(w_all)
            full_mean_pred = fourier_pred + residual_pred_train
            residuals = y_train - full_mean_pred
            r2 = residuals ** 2

            # Compute loss (NLL)
            loss = float(jnp.mean(0.5 * h_pred + 0.5 * jnp.float64(r2) / sigma2))

            # Check convergence
            if abs(prev_loss - loss) / (abs(prev_loss) + 1e-10) < tol:
                print(f"  Converged at outer iteration {outer + 1}")
                break
            prev_loss = loss

        # --- Build final model with variance ---
        w1_final = w_all[:F1]
        w2_final = w_all[F1:F1 + F2] if K2 > 0 else jnp.array([], dtype=jnp.float32)
        w3_final = (w_all[F1 + F2:] if K3 > 0 and T > 0
                    else jnp.array([], dtype=jnp.float32))

        mean_model = MeanModel(
            f0=jnp.array(f0, dtype=jnp.float32),
            w1=jnp.array(w1_final, dtype=jnp.float32),
            w2=jnp.array(w2_final, dtype=jnp.float32),
            w3=jnp.array(w3_final, dtype=jnp.float32),
            K1=K1, K2=K2, K3=K3, D=D,
            include_linear_1=_il1,
            include_linear_2=_incl_lin_2, include_linear_3=_incl_lin_3,
            basis_name=basis_name,
        )

        # Split augmented w_h into [first-order | second-order | third-order | residual]
        Fh2 = Ph * _bs(K2h, _ilh2, basis_name) ** 2 if K2h > 0 else 0
        Fh3 = Th * _bs(K3h, _ilh3, basis_name) ** 3 if K3h > 0 else 0
        w_h_fourier1 = w_h[:Fh]
        w_h_fourier2 = (w_h[Fh:Fh + Fh2] if Fh2 > 0
                        else jnp.array([], dtype=jnp.float32))
        w_h_fourier3 = (w_h[Fh + Fh2:Fh + Fh2 + Fh3] if Fh3 > 0
                        else jnp.array([], dtype=jnp.float32))
        w_h_residual = w_h[Fh + Fh2 + Fh3:] if M_h > 0 else None

        # Build fitted variance residual model (if present)
        fitted_var_residual = None
        if var_residual_model is not None and w_h_residual is not None:
            import equinox as eqx
            fitted_var_residual = eqx.tree_at(
                lambda m: m.weights, var_residual_model,
                jnp.array(w_h_residual, dtype=jnp.float32))
            fitted_var_residual = eqx.tree_at(
                lambda m: m.proj_coeffs, fitted_var_residual,
                jnp.array(z_h_proj_coeffs, dtype=jnp.float64))

        variance_model = VarianceModel(
            h0=jnp.array(h0, dtype=jnp.float32),
            w1=jnp.array(w_h_fourier1, dtype=jnp.float32),
            Kh=Kh, D=D,
            w2=jnp.array(w_h_fourier2, dtype=jnp.float32),
            K2h=K2h,
            pair_indices_h=(np.array(var_pair_mgr.pair_indices)
                            if var_pair_mgr is not None and Ph > 0 else None),
            w3=jnp.array(w_h_fourier3, dtype=jnp.float32),
            K3h=K3h,
            triple_indices_h=(np.array(var_triple_mgr.triple_indices)
                              if var_triple_mgr is not None and Th > 0 else None),
            w_var_residual=(jnp.array(w_h_residual, dtype=jnp.float32)
                            if w_h_residual is not None else None),
            variance_residual=fitted_var_residual,
            basis_name=basis_name,
            include_linear_h1=_ilh1,
            include_linear_h2=_ilh2,
            include_linear_h3=_ilh3,
        )

        model = HiFiANOVA(
            mean_model=mean_model,
            variance_model=variance_model,
            residual_net=model.residual_net,
            K1=K1, K2=K2, K3=K3, Kh=Kh, D=D,
            pair_indices=np.array(pair_mgr.pair_indices),
            triple_indices=model.triple_indices,
            G1=np.array(G1),
            G2=np.array(G2) if G2 is not None else None,
            G3=model.G3,
            include_linear_1=getattr(model, 'include_linear_1', True),
            include_linear_2=getattr(model, 'include_linear_2', True),
            include_linear_3=getattr(model, 'include_linear_3', True),
            include_linear_h1=getattr(model, 'include_linear_h1', True),
            include_linear_h2=getattr(model, 'include_linear_h2', True),
            include_linear_h3=getattr(model, 'include_linear_h3', True),
            basis_name=getattr(model, 'basis_name', 'fourier'),
        )

        # --- Evaluate on validation ---
        psi1_val = build_first_order_features(x_val, Kh, include_linear=_ilh1, basis_name=basis_name)
        phi1_val = build_first_order_features(x_val, K1, include_linear=_il1, basis_name=basis_name)
        phi2_val = (build_second_order_features(x_val, K2, pair_mgr.pair_indices, include_linear=_il2, basis_name=basis_name)
                    if K2 > 0 else None)
        phi3_val = (build_third_order_features(x_val, K3, model.triple_indices, include_linear=_il3, basis_name=basis_name)
                    if K3 > 0 and model.triple_indices is not None else None)

        mean_val = mean_model.predict(phi1_val, phi2_val, phi3_val)
        if model.residual_net is not None:
            res_out_val = jax.vmap(model.residual_net)(x_val)
            if res_out_val.ndim > 1:
                res_out_val = res_out_val.squeeze(-1)
            mean_val = mean_val + res_out_val
        rmse_val = float(jnp.sqrt(jnp.mean((y_val - mean_val) ** 2)))

        # NLL on validation (include all variance components)
        psi2_val = None
        if K2h > 0 and var_pair_mgr is not None and var_pair_mgr.P > 0:
            psi2_val = build_second_order_features(x_val, K2h, var_pair_mgr.pair_indices,
                                                    include_linear=_ilh2, basis_name=basis_name)
        psi3_val_h = None
        if K3h > 0 and var_triple_mgr is not None and var_triple_mgr.T > 0:
            psi3_val_h = build_third_order_features(x_val, K3h, var_triple_mgr.triple_indices,
                                                     include_linear=_ilh3, basis_name=basis_name)
        z_h_proj_val = None
        if fitted_var_residual is not None:
            z_h_val = var_residual_model.build_features(x_val)
            psi_fourier_val = psi1_val
            if psi2_val is not None:
                psi_fourier_val = jnp.concatenate([psi_fourier_val, psi2_val], axis=1)
            if psi3_val_h is not None:
                psi_fourier_val = jnp.concatenate([psi_fourier_val, psi3_val_h], axis=1)
            z_h_proj_val = z_h_val - psi_fourier_val @ jnp.asarray(z_h_proj_coeffs)
        h_val = variance_model.predict_log_variance(psi1_val, psi2_val, psi3_val_h, z_h_proj_val)
        sigma2_val = jnp.exp(h_val)
        nll_val = float(jnp.mean(0.5 * h_val + 0.5 * (y_val - mean_val) ** 2 / sigma2_val))

        results['stage_D'] = {
            'rmse_val': rmse_val, 'nll_val': nll_val,
            'has_variance_second_order': K2h > 0 and Ph > 0,
            'n_variance_pairs': Ph,
            'has_variance_residual': M_h > 0,
            'M_variance_residual': M_h,
        }
        parts_str = []
        if K2h > 0 and Ph > 0:
            parts_str.append(f"{Ph} var pairs")
        if M_h > 0:
            parts_str.append(f"{M_h} var residual features")
        suffix = f" ({', '.join(parts_str)})" if parts_str else ""
        print(f"  RMSE val: {rmse_val:.4f}, NLL val: {nll_val:.4f}{suffix}")

        return model


def estimate_sobol(x: jnp.ndarray, y: jnp.ndarray,
                   K1: int = 10, K2: int = 5, K3: int = 0,
                   strategy: str = 'variance',
                   lambda1: Optional[float] = None,
                   lambda2: Optional[float] = None,
                   lambda3: Optional[float] = None,
                   auto_lambda: bool = True,
                   additivity_target: float = 1.0,
                   include_linear_1: bool = True,
                   include_linear_2: bool = True,
                   include_linear_3: bool = True,
                   basis_name: str = 'fourier',
                   ) -> Dict:
    """Estimate true Sobol indices with minimal regularization.

    This is a SEPARATE mode from predictive fitting. The goal is
    unbiased Sobol recovery, not good prediction on new data.

    Two approaches:
    - auto_lambda=True: finds lambda such that sum(Sobol) ~ 1.0
      (the "additivity criterion" — unbiased estimates should sum to 1)
    - auto_lambda=False: uses provided lambda1, lambda2, lambda3

    Args:
        x: (N, D) inputs in [0, 1]
        y: (N,) targets
        K1, K2, K3: max harmonics/degrees per order
        strategy: regularization strategy
        lambda1, lambda2, lambda3: regularization (if auto_lambda=False)
        auto_lambda: if True, find lambda by additivity criterion
        additivity_target: target sum of Sobol indices (default 1.0)
        include_linear_1, include_linear_2, include_linear_3: per-order basis config
        basis_name: 'fourier' or 'legendre'

    Returns:
        Dict with sobol_first_order, sobol_second_order, sobol_third_order,
        sobol_total_order, coefficients, lambda_used, additivity_sum, etc.
    """
    from ..core.features import basis_size as _bs

    x = jnp.asarray(x)
    y = jnp.asarray(y)
    D = x.shape[1]
    N = x.shape[0]

    pair_mgr = PairManager(D)
    G1 = build_gram_matrix(K1, include_linear_1, basis_name)
    G2 = (build_gram_matrix_2d(build_gram_matrix(K2, include_linear_2, basis_name))
          if K2 > 0 else None)

    triple_mgr = None
    G3 = None
    if K3 > 0:
        triple_mgr = TripleManager(D)
        G3 = build_gram_matrix_3d(build_gram_matrix(K3, include_linear_3, basis_name))

    phi1 = build_first_order_features(x, K1, include_linear=include_linear_1,
                                       basis_name=basis_name)
    phi2 = (build_second_order_features(x, K2, pair_mgr.pair_indices,
                                         include_linear=include_linear_2,
                                         basis_name=basis_name)
            if K2 > 0 else None)
    phi3 = (build_third_order_features(x, K3, triple_mgr.triple_indices,
                                        include_linear=include_linear_3,
                                        basis_name=basis_name)
            if K3 > 0 and triple_mgr is not None and triple_mgr.T > 0 else None)

    Phi = phi1
    if phi2 is not None:
        Phi = jnp.concatenate([Phi, phi2], axis=1)
    if phi3 is not None:
        Phi = jnp.concatenate([Phi, phi3], axis=1)

    f0 = float(jnp.mean(y))
    y_c = y - f0

    block1 = _bs(K1, include_linear_1, basis_name)
    block2 = _bs(K2, include_linear_2, basis_name) ** 2 if K2 > 0 else 0
    block3 = _bs(K3, include_linear_3, basis_name) ** 3 if K3 > 0 else 0
    F1 = D * block1
    F2 = pair_mgr.P * block2 if K2 > 0 else 0
    T = triple_mgr.T if triple_mgr is not None else 0

    G1_np = np.asarray(G1, dtype=np.float64)
    G2_np = np.asarray(G2, dtype=np.float64) if G2 is not None else None
    G3_np = np.asarray(G3, dtype=np.float64) if G3 is not None else None

    def solve_and_sobol(lam1, lam2, lam3):
        """Solve ridge and extract Sobol indices."""
        reg = build_regularization_vector(
            D, K1, K2, pair_mgr.P, strategy, lam1, lam2,
            K3=K3, T=T, lambda_order3=lam3,
            include_linear_1=include_linear_1,
            include_linear_2=include_linear_2,
            include_linear_3=include_linear_3,
            basis_name=basis_name)
        w = weighted_ridge_solve(Phi, y_c, reg)
        w_np = np.asarray(w, dtype=np.float64)

        fo_vars = {}
        for i in range(D):
            wi = w_np[i*block1:(i+1)*block1]
            fo_vars[i] = max(0.0, float(wi @ G1_np @ wi))

        so_vars = {}
        if K2 > 0:
            for p in range(pair_mgr.P):
                wp = w_np[F1 + p*block2: F1 + (p+1)*block2]
                var_p = max(0.0, float(wp @ G2_np @ wp))
                i_v, j_v = pair_mgr.pair_to_variables(p)
                so_vars[(i_v, j_v)] = var_p

        to_vars = {}
        if K3 > 0 and triple_mgr is not None and G3_np is not None:
            for t_idx in range(T):
                wt = w_np[F1 + F2 + t_idx*block3: F1 + F2 + (t_idx+1)*block3]
                var_t = max(0.0, float(wt @ G3_np @ wt))
                i, j, k = (int(triple_mgr.triple_indices[t_idx, l]) for l in range(3))
                to_vars[(i, j, k)] = var_t

        total_var = (sum(fo_vars.values()) + sum(so_vars.values()) +
                     sum(to_vars.values()))

        # Additivity: sum of Sobol should be ~1 if unbiased
        var_y = float(jnp.var(y))
        additivity_sum = total_var / var_y if var_y > 0 else 0.0

        return w_np, fo_vars, so_vars, to_vars, total_var, additivity_sum

    if auto_lambda:
        # Find lambda that gives additivity_sum closest to target
        from scipy.optimize import minimize_scalar

        # Default ratios: lambda scales with parameter count per group
        beta2 = block2 / block1 if K2 > 0 and block1 > 0 else 1.0
        beta3 = block3 / block1 if K3 > 0 and block1 > 0 else 1.0

        def additivity_gap(log_lam):
            lam = 10 ** log_lam
            _, _, _, _, _, add_sum = solve_and_sobol(lam, lam * beta2, lam * beta3)
            return (add_sum - additivity_target) ** 2

        result = minimize_scalar(additivity_gap, bounds=(-10, 0), method='bounded')
        lam1_opt = 10 ** result.x
        lam2_opt = lam1_opt * beta2
        lam3_opt = lam1_opt * beta3
        print(f"  Sobol estimation: auto lambda1={lam1_opt:.2e}, "
              f"lambda2={lam2_opt:.2e}" +
              (f", lambda3={lam3_opt:.2e}" if K3 > 0 else ""))
    else:
        lam1_opt = lambda1 if lambda1 is not None else 1e-6
        lam2_opt = lambda2 if lambda2 is not None else lam1_opt * 10
        lam3_opt = lambda3 if lambda3 is not None else lam2_opt * 10

    w_np, fo_vars, so_vars, to_vars, total_var, additivity_sum = solve_and_sobol(
        lam1_opt, lam2_opt, lam3_opt)

    # Build Sobol dicts
    s1 = {i: fo_vars[i]/total_var if total_var > 0 else 0 for i in range(D)}
    s2 = {k: v/total_var if total_var > 0 else 0 for k, v in so_vars.items()}
    s3 = {k: v/total_var if total_var > 0 else 0 for k, v in to_vars.items()}

    # Total-order (first + second + third involving variable i)
    st = {}
    for i in range(D):
        t = fo_vars.get(i, 0)
        for (a, b), v in so_vars.items():
            if a == i or b == i:
                t += v
        for key, v in to_vars.items():
            if i in key:
                t += v
        st[i] = t / total_var if total_var > 0 else 0

    return {
        'sobol_first_order': s1,
        'sobol_second_order': s2,
        'sobol_third_order': s3,
        'sobol_total_order': st,
        'variance_first_order': fo_vars,
        'variance_second_order': so_vars,
        'variance_third_order': to_vars,
        'total_model_variance': total_var,
        'additivity_sum': additivity_sum,
        'lambda_order1': lam1_opt,
        'lambda_order2': lam2_opt,
        'lambda_order3': lam3_opt,
        'coefficients': w_np,
        'f0': f0,
        'mode': 'sobol_estimation',
    }
