"""Stage B: first + second(+third)-order joint ridge fit.

Extracted verbatim from ``trainer.py`` (behavior-preserving decomposition step).
Holds ``_fit_stage_b`` and ``_generate_pair_candidates`` as ``StageBMixin``,
composed by ``HiFiANOVATrainer`` so ``self`` (config/_dtype/_log/_verbose)
resolves to the trainer instance exactly as before. Relative imports are one
level deeper than in ``trainer.py`` (this module sits in ``training.stages``).
"""

from ...array_backend import xp as jnp  # switchable array backend (numpy exact core)
import numpy as np

from ...core.features import (
    build_second_order_features, build_third_order_features, basis_size,
)
from ...core.pairs import PairManager, TripleManager
from ...model.mean_model import MeanModel
from ...model.hifi_anova import HiFiANOVA
from ..regularization import build_regularization_vector
from ..ridge import weighted_ridge_solve
from ..fitted_design import build_record
from .._trainer_helpers import _split_mean_coeffs, _prune_first_order_blocks


class StageBMixin:
    """Stage-B methods mixed into :class:`HiFiANOVATrainer` (``self`` is the
    trainer instance)."""

    def _fit_stage_b(self, model, x_train, y_train, x_val, y_val, y_centered,
                     f0, phi1_train, phi1_val, reg1, pair_mgr, D, K1, K2, K3,
                     G1, G2, G3, strategy, lambda1, lambda2, lambda3,
                     pair_pruning, triple_selection, triple_pruning,
                     pair_threshold, first_order_pruning,
                     include_linear_1, include_linear_2, include_linear_3,
                     include_linear_h1, include_linear_h2, include_linear_h3,
                     basis_name, results):
        """Stage B: first + second(+third)-order joint ridge fit.

        Extracted verbatim from ``fit`` (behavior-preserving): second-order
        feature build, joint solve, optional pair/triple/first-order post-fit
        pruning with refits, third-order selection, model build, eval, and the
        full-design fitted-design record. Returns ``(model, pair_mgr, K2)`` —
        ``pair_mgr`` may be rebuilt by pruning and ``K2`` is zeroed when every
        pair is pruned (disabling second-order downstream, incl. Stage D).
        """
        self._log(f"=== Stage B: First + second-order model "
              f"({pair_mgr.P} pairs" +
              (f" from {len(pair_mgr.active_variables)} active vars"
               if pair_mgr.active_variables else "") + ") ===")
        # Term structure (X11C-S02): per-pair K2 orders and/or an order-
        # selective first-order block, resolved by the trainer. Both None on
        # every default path (byte-identical behavior).
        pair_k2 = getattr(self, '_pair_k2', None)
        fo_included = getattr(self, '_fo_included', None)
        pair_block_info = None
        # phi1_train may be a SUBSET first-order block (variable_orders), so
        # F1 is its actual width, not D * basis_size(K1).
        F1_actual = phi1_train.shape[1]
        n_fo = len(fo_included) if fo_included is not None else D

        # Build second-order features.
        # For large F (>10k features), use numpy to avoid GPU OOM.
        if pair_k2 is not None:
            F2_est = sum(basis_size(k, include_linear_2, basis_name) ** 2
                         for k in pair_k2)
        else:
            F2_est = pair_mgr.P * basis_size(K2, include_linear_2, basis_name) ** 2
        F_total_est = F1_actual + F2_est
        use_numpy = (F_total_est * x_train.shape[0] * 8 / 1e9) > 2.0

        if pair_k2 is not None:
            from ...core.features import build_second_order_features_per_pair
            phi2_train, pair_block_info = build_second_order_features_per_pair(
                x_train, pair_k2, pair_mgr.pair_indices,
                include_linear=include_linear_2, basis_name=basis_name)
            phi2_val, _ = build_second_order_features_per_pair(
                x_val, pair_k2, pair_mgr.pair_indices,
                include_linear=include_linear_2, basis_name=basis_name)
            phi_all_train = jnp.concatenate([phi1_train, phi2_train], axis=1)
            phi_all_val = jnp.concatenate([phi1_val, phi2_val], axis=1)
        elif use_numpy:
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

        # Combined regularization (per-pair blocks when pair_k2 is set)
        reg_all = build_regularization_vector(
            n_fo, K1, (pair_k2 if pair_k2 is not None else K2), pair_mgr.P,
            strategy, lambda1, lambda2,
            include_linear_1=include_linear_1,
            include_linear_2=include_linear_2, basis_name=basis_name,
        )

        # Ridge solve
        w_all = weighted_ridge_solve(phi_all_train, y_centered, reg_all)

        # === Post-fit pair pruning ===
        if pair_pruning != 'none' and pair_mgr.P > 0:
            from ..selection import prune_groups_postfit
            F1 = F1_actual
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

            self._log(f"  Post-fit pair pruning ({pair_pruning}):")
            surviving_pairs, prune_info = prune_groups_postfit(
                np.asarray(phi_all_train, dtype=np.float64),
                np.asarray(y_centered, dtype=np.float64),
                pair_group_slices,
                np.asarray(reg_all, dtype=np.float64),
                method=pair_pruning,
                gram_matrices=pair_gram_mats,
                group_labels=pair_labels,
                verbose=self._verbose,
            )
            results['pair_pruning'] = prune_info

            n_pruned = pair_mgr.P - len(surviving_pairs)
            if n_pruned > 0 and len(surviving_pairs) == 0:
                # All pairs pruned — fall back to first-order only
                self._log(f"  All {pair_mgr.P} pairs pruned, reverting to first-order")
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
                    n_fo, K1, K2, pair_mgr.P, strategy, lambda1, lambda2,
                    include_linear_1=include_linear_1, include_linear_2=include_linear_2,
                    basis_name=basis_name)
                w_all = weighted_ridge_solve(phi_all_train, y_centered, reg_all)
                self._log(f"  Refit after pruning: {n_pruned} pairs removed, "
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
                from ..selection import select_active_variables_principled
                active_vars, _sel_info = select_active_variables_principled(
                    np.asarray(phi1_train, dtype=np.float64),
                    np.asarray(y_centered, dtype=np.float64),
                    D, K1, np.asarray(reg1, dtype=np.float64),
                    method=triple_selection,
                    G1=np.asarray(G1, dtype=np.float64),
                    include_linear=include_linear_1,
                    basis_name=basis_name,
                    verbose=self._verbose,
                )
                # Use 'two_active' mode: includes triples where 2 of 3 vars
                # are first-order active, catching hidden third-order variables
                triple_sel_mode = 'two_active' if len(active_vars) >= 2 else 'all'
            elif triple_selection == 'all':
                active_vars = list(range(D))
                triple_sel_mode = 'all'
            else:
                # Threshold-based fallback
                from ...analysis.sobol import compute_sobol_indices as _compute_sobol
                _temp_model = HiFiANOVA(
                    mean_model=MeanModel(
                        f0=jnp.array(f0, dtype=self._dtype),
                        w1=jnp.array(w_all[:D * basis_size(K1, include_linear_1, basis_name)], dtype=self._dtype),
                        w2=jnp.array([], dtype=self._dtype),
                        K1=K1, K2=0, D=D,
                        include_linear_1=include_linear_1, basis_name=basis_name,
                    ),
                    K1=K1, K2=0, K3=0, Kh=0, D=D,
                    include_linear_1=include_linear_1, basis_name=basis_name,
                )
                _sobol = _compute_sobol(_temp_model)
                _fo = _sobol['mean_sobol']['first_order']
                from ...core.pairs import select_active_variables
                active_vars = select_active_variables(_fo, D, threshold=pair_threshold)
                triple_sel_mode = triple_selection

            triple_mgr = TripleManager(D, active_variables=active_vars,
                                        selection_mode=triple_sel_mode)
            if triple_mgr.T > 0:
                self._log(f"  Third-order: {triple_mgr.T} triples "
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
                    from ..selection import prune_groups_postfit
                    from ...core.features import basis_size as _bs
                    F1 = D * _bs(K1, include_linear_1, basis_name)
                    F2 = pair_mgr.P * _bs(K2, include_linear_2, basis_name) ** 2
                    block3 = _bs(K3, include_linear_3, basis_name) ** 3
                    triple_group_slices = [
                        slice(F1 + F2 + t * block3, F1 + F2 + (t + 1) * block3)
                        for t in range(triple_mgr.T)
                    ]
                    triple_labels = [str(triple_mgr.triple_to_variables(t))
                                      for t in range(triple_mgr.T)]

                    self._log(f"  Post-fit triple pruning ({triple_pruning}):")
                    surviving_triples, tprune_info = prune_groups_postfit(
                        np.asarray(phi_all_train, dtype=np.float64),
                        np.asarray(y_centered, dtype=np.float64),
                        triple_group_slices,
                        np.asarray(reg_all, dtype=np.float64),
                        method=triple_pruning,
                        group_labels=triple_labels,
                        verbose=self._verbose,
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
                        self._log(f"  Refit: {n_tpruned} triples pruned, "
                              f"{triple_mgr.T} remaining")

            else:
                self._log("  Third-order: no triples selected (too few active vars)")

        # === Post-fit first-order pruning ===
        if first_order_pruning != 'none':
            self._log(f"  Post-fit first-order pruning ({first_order_pruning}):")
            block1_fo = basis_size(K1, include_linear_1, basis_name)
            w_all, fo_info = _prune_first_order_blocks(
                w_all, phi_all_train, reg_all, y_centered,
                D, block1_fo, G1, first_order_pruning, verbose=self._verbose)
            w_all = jnp.asarray(w_all, dtype=self._dtype)
            results['first_order_pruning'] = fo_info

        # Split coefficients (block sizes depend on basis_type)
        from ...core.features import basis_size as _bs
        F1 = F1_actual
        F2 = (int(phi2_train.shape[1]) if pair_k2 is not None
              else pair_mgr.P * _bs(K2, include_linear_2, basis_name) ** 2)
        w1_new, w2_new, w3_new = _split_mean_coeffs(
            w_all, F1, F2,
            has_third=(K3 > 0 and triple_mgr is not None and triple_mgr.T > 0),
            dtype=self._dtype)

        # Build model (Level 1, with optional third-order). With an order-
        # selective first-order block the solved w1 is a subset layout —
        # scatter it into the full uniform layout with exact zeros for the
        # excluded variables (the model stays uniform; see _scatter_first_order).
        if fo_included is not None:
            from .._trainer_helpers import _scatter_first_order
            w1_model = _scatter_first_order(
                w1_new, D, _bs(K1, include_linear_1, basis_name),
                fo_included, dtype=self._dtype)
        else:
            w1_model = jnp.array(w1_new, dtype=self._dtype)
        _pair_k2_t = tuple(int(k) for k in pair_k2) if pair_k2 is not None else None
        actual_K3 = K3 if (triple_mgr is not None and triple_mgr.T > 0) else 0
        mean_model = MeanModel(
            f0=jnp.array(f0, dtype=self._dtype),
            w1=w1_model,
            w2=jnp.array(w2_new, dtype=self._dtype),
            w3=jnp.array(w3_new, dtype=self._dtype),
            K1=K1, K2=K2, K3=actual_K3, D=D,
            include_linear_1=include_linear_1,
            include_linear_2=include_linear_2, include_linear_3=include_linear_3,
            basis_name=basis_name,
            pair_block_info=pair_block_info,
            pair_k2=_pair_k2_t,
        )

        model = HiFiANOVA(
            mean_model=mean_model,
            K1=K1, K2=K2, K3=actual_K3, Kh=0, D=D,
            pair_indices=np.array(pair_mgr.pair_indices),
            triple_indices=(np.array(triple_mgr.triple_indices)
                            if triple_mgr is not None and triple_mgr.T > 0
                            else None),
            include_linear_1=include_linear_1,
            include_linear_2=include_linear_2,
            include_linear_3=include_linear_3,
            include_linear_h1=include_linear_h1,
            include_linear_h2=include_linear_h2,
            include_linear_h3=include_linear_h3,
            basis_name=basis_name,
            pair_block_info=pair_block_info,
            pair_k2=_pair_k2_t,
            fo_included=(tuple(fo_included) if fo_included is not None
                         else None),
        )

        # Evaluate Stage B
        pred_train_b = f0 + self._dtype(phi_all_train @ w_all)
        pred_val_b = f0 + self._dtype(phi_all_val @ w_all)
        rmse_train_b = float(jnp.sqrt(jnp.mean((y_train - pred_train_b) ** 2)))
        rmse_val_b = float(jnp.sqrt(jnp.mean((y_val - pred_val_b) ** 2)))
        results['stage_B'] = {
            'rmse_train': rmse_train_b, 'rmse_val': rmse_val_b,
            'n_triples': triple_mgr.T if triple_mgr is not None else 0,
        }
        self._log(f"  RMSE train: {rmse_train_b:.4f}, val: {rmse_val_b:.4f}")

        # Fitted-design record (full mean design: first + second [+ third]).
        # Overwrites the Stage-A record. Uses the real per-order penalties
        # (reg_all) and block layout — unlike the API's config-rebuilt reg,
        # which pads third-order columns with lambda_order2. Homoscedastic
        # here; Stage D re-solves this design under precision weights and
        # attaches them (Phase 3). Residual (Stage C) features are penalty-
        # only and excluded from the Sobol blocks.
        pair_grams = None
        if pair_k2 is not None:
            from ...core.gram import (build_gram_matrix as _bgm,
                                      build_gram_matrix_2d as _bgm2)
            pair_grams = [np.asarray(_bgm2(_bgm(int(k), include_linear_2,
                                               basis_name)), dtype=np.float64)
                          for k in pair_k2]
        results['fitted_design'] = build_record(
            phi_all_train, w_all, reg_all, y_train, D,
            K1, G1, include_linear_1, basis_name,
            K2=K2, P=pair_mgr.P, G2=G2, pair_indices=np.asarray(pair_mgr.pair_indices),
            include_linear_2=include_linear_2,
            K3=actual_K3,
            T=(triple_mgr.T if (triple_mgr is not None and triple_mgr.T > 0) else 0),
            G3=G3,
            triple_indices=(np.asarray(triple_mgr.triple_indices)
                            if (triple_mgr is not None and triple_mgr.T > 0)
                            else None),
            include_linear_3=include_linear_3,
            fo_included=fo_included, pair_block_info=pair_block_info,
            pair_grams=pair_grams)

        return model, pair_mgr, K2

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
        from ..pairs_import_helper import _resolve_pair_manager
        N_train = x_train.shape[0]

        if variable_selection is not None and pair_candidates is not None:
            # New-style config: explicit separation of variable selection
            # and candidate generation
            from ..selection import select_active_variables_principled
            active_vars, var_sel_info = select_active_variables_principled(
                np.asarray(phi1_train, dtype=np.float64),
                np.asarray(y_centered, dtype=np.float64),
                D, K1, np.asarray(reg1, dtype=np.float64),
                method=variable_selection,
                G1=np.asarray(G1, dtype=np.float64),
                include_linear=include_linear_1,
                basis_name=basis_name,
                verbose=self._verbose,
            )
            if max_pair_variables is not None and len(active_vars) > max_pair_variables:
                active_vars = active_vars[:max_pair_variables]
            results['variable_selection'] = var_sel_info

            pair_mgr = PairManager(D, active_variables=active_vars,
                                    selection_mode=pair_candidates)
            if True:  # verbose
                from math import comb
                F_est = D * basis_size(K1, include_linear_1, basis_name) + pair_mgr.P * basis_size(K2, include_linear_2, basis_name)**2
                self._log(f"  Variable selection ({variable_selection}): "
                      f"{len(active_vars)}/{D} active")
                self._log(f"  Pair candidates ({pair_candidates}): "
                      f"{pair_mgr.P} pairs (vs {comb(D,2)} all), F={F_est}")
        else:
            # Legacy config: pair_selection handles everything
            pair_mgr = _resolve_pair_manager(
                pair_selection, D, K1, K2, G1, w1, N_train,
                pair_threshold, max_pair_variables,
                Phi1=phi1_train, y_centered=y_centered, reg1=reg1,
                strategy=strategy, lambda1=lambda1,
                include_linear_1=include_linear_1, include_linear_2=include_linear_2,
                basis_name=basis_name, verbose=self._verbose)
        return pair_mgr
