"""Mixed per-variable basis path (Stage A + Stage B only).

Extracted verbatim from ``trainer.py`` (behavior-preserving decomposition step).
Holds ``_fit_mixed`` as ``MixedBasisMixin``, composed by ``HiFiANOVATrainer`` so
``self`` (config/_dtype/_log/_verbose) resolves to the trainer instance exactly
as before. Relative imports are one level deeper than in ``trainer.py`` (this
module sits in ``training.stages``).
"""

from ...array_backend import xp as jnp  # switchable array backend (numpy exact core)
import numpy as np

from ...core.features import (
    build_mixed_first_order_features, build_mixed_second_order_features,
)
from ...core.pairs import PairManager
from ...model.mean_model import MeanModel
from ...model.hifi_anova import HiFiANOVA
from ..regularization import build_mixed_regularization_vector
from ..ridge import weighted_ridge_solve
from ..fitted_design import build_mixed_record


class MixedBasisMixin:
    """Mixed-basis fit method mixed into :class:`HiFiANOVATrainer` (``self`` is
    the trainer instance)."""

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
        # Structural validation of the spec now that D is known (DEC-046):
        # integer indices in [0, D), supported families, positive K, no typo'd
        # nested keys. A malformed spec used to silently fall back to defaults
        # (a stringified index, a nested 'basi' typo) or crash obscurely in the
        # feature build (K<=0). Runs before the capability fence so a bad spec is
        # reported as a spec error.
        from ...validation import validate_basis_per_variable
        validate_basis_per_variable(basis_per_variable, D)

        # The mixed per-variable path implements Stage A + Stage B only (K3=0,
        # Kh=0). Requesting a nonlinear residual (Stage C), a heteroscedastic
        # variance model (Stage D), or third-order terms used to be *silently
        # ignored* here — the fit returned a mean-only A/B model that differed
        # from what the caller asked for. Fail loudly instead (DEC-036 spirit).
        unsupported = []
        if 'C' in stages:
            unsupported.append("Stage C / nonlinear residual")
        if 'D' in stages:
            unsupported.append("Stage D / heteroscedastic variance")
        res_cfg = cfg.get('residual') or cfg.get('residual_nn')
        if ('C' not in stages and res_cfg and
                (not isinstance(res_cfg, dict) or res_cfg.get('enabled', True))):
            unsupported.append("a residual model (residual=/residual_nn=)")
        if int(cfg.get('K3', 0) or 0) > 0:
            unsupported.append("third-order terms (K3>0)")
        if unsupported:
            raise NotImplementedError(
                "basis_per_variable (mixed per-variable basis) currently "
                "supports Stage A + Stage B (first- and second-order mean terms) "
                "only. Requested but not implemented on the mixed-basis path: "
                + ", ".join(unsupported) + ". These would otherwise be silently "
                "dropped. Use a uniform basis (basis_name=...) to fit them, or "
                "remove them from the mixed-basis request."
            )

        # Capability fence for selection/pruning/pair controls (DEC-045). The
        # mixed path fits ALL first-order blocks and, with Stage B, ALL pairs; it
        # does not implement variable selection, pair candidate generation, or
        # pair/first-order pruning. These used to silently no-op — they can change
        # model size and attribution, so a non-neutral value is now rejected. Each
        # is listed with its neutral value. (The one-call API neutralizes only the
        # *implicit* variable_selection='bic' default, with a warning; an explicit
        # request lands here and raises.)
        _cap_checks = [
            ('variable_selection', cfg.get('variable_selection'),
             bool(cfg.get('variable_selection')), 'None'),
            ('pair_candidates', cfg.get('pair_candidates'),
             cfg.get('pair_candidates') is not None, 'None'),
            ('pair_selection', cfg.get('pair_selection'),
             cfg.get('pair_selection') is not None, 'None'),
            ('max_pair_variables', cfg.get('max_pair_variables'),
             cfg.get('max_pair_variables') is not None, 'None'),
            ('pair_pruning', cfg.get('pair_pruning', 'none'),
             cfg.get('pair_pruning', 'none') not in (None, 'none'), "'none'"),
            ('first_order_pruning', cfg.get('first_order_pruning', 'none'),
             cfg.get('first_order_pruning', 'none') not in (None, 'none'),
             "'none'"),
        ]
        cap_bad = [(name, val, neutral)
                   for name, val, is_bad, neutral in _cap_checks if is_bad]
        if cap_bad:
            detail = "; ".join(f"{name}={val!r} (neutral: {neutral})"
                               for name, val, neutral in cap_bad)
            raise NotImplementedError(
                "basis_per_variable (mixed per-variable basis) does not support "
                "variable selection, pair candidate generation, or pair/"
                "first-order pruning — the mixed path fits all first-order blocks "
                "and all variable pairs. These controls would silently no-op, so "
                "a non-neutral value is rejected: " + detail + ". Set each to its "
                "neutral value, or use a uniform basis (basis_name=...) to run "
                "selection/pruning."
            )

        results = {}

        # --- Resolve var_specs ---
        if basis_per_variable == 'auto':
            from ...analysis.basis_characterization import (
                sequential_projection_characterization, auto_select_basis)
            self._log("=== Mixed basis: auto-characterization ===")
            # Sequential projection is the exact (non-overlapping) additive
            # decomposition; the cross-residual variant double-counts content
            # shared between the Fourier and Haar projections.
            char = sequential_projection_characterization(
                x_train, y_train, x_val, y_val,
                K_legendre=cfg.get('K1', 5),
                K_fourier=cfg.get('K1', 5),
                J_haar=cfg.get('K1', 4),
                strategy=strategy,
                lambda_legendre=lambda1,
                verbose=self._verbose)
            rec = auto_select_basis(char)
            var_specs_dict = []
            for i in range(D):
                r = rec['per_variable'][i]
                basis = r['basis'].split('+')[0]  # 'legendre+haar' → 'legendre'
                var_specs_dict.append({'basis': basis, 'K': r['K_recommended']})
            self._log(f"  Auto-selected: {rec['summary']}")
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
        self._log("=== Stage A: Mixed first-order model ===")
        basis_summary = {}
        for spec in var_specs_dict:
            b = spec['basis']
            basis_summary[b] = basis_summary.get(b, 0) + 1
        summary_str = ', '.join(f"{c} {b}" for b, c in
                                 sorted(basis_summary.items(), key=lambda x: -x[1]))
        self._log(f"  Per-variable: {summary_str}")

        phi1_train, block_info = build_mixed_first_order_features(
            x_train, var_specs_dict)
        phi1_val, _ = build_mixed_first_order_features(x_val, var_specs_dict)

        F1 = phi1_train.shape[1]
        self._log(f"  Total first-order features: {F1}")

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
        mean_model = MeanModel(
            f0=jnp.array(f0, dtype=self._dtype),
            w1=jnp.array(w1, dtype=self._dtype),
            w2=jnp.array([], dtype=self._dtype),
            w3=jnp.array([], dtype=self._dtype),
            K1=0, K2=0, K3=0, D=D,
            include_linear_1=True,
            basis_name='mixed',
            var_specs=var_specs_tuple,
        )

        model = HiFiANOVA(
            mean_model=mean_model,
            K1=0, K2=0, K3=0, Kh=0, D=D,
            pair_indices=None, triple_indices=None,
            include_linear_1=True,
            include_linear_h1=include_linear_h1,
            include_linear_h2=include_linear_h2,
            include_linear_h3=include_linear_h3,
            basis_name='mixed',
            var_specs=var_specs_tuple,
        )

        pred_train = f0 + self._dtype(phi1_train @ w1)
        pred_val = f0 + self._dtype(phi1_val @ w1)
        rmse_train = float(jnp.sqrt(jnp.mean((y_train - pred_train) ** 2)))
        rmse_val = float(jnp.sqrt(jnp.mean((y_val - pred_val) ** 2)))
        results['stage_A'] = {'rmse_train': rmse_train, 'rmse_val': rmse_val}
        self._log(f"  RMSE train: {rmse_train:.4f}, val: {rmse_val:.4f}")

        # Fitted-design record inputs (updated in Stage B if it runs): the design,
        # coefficients and penalty the mixed fit actually solved, so the one-call
        # diagnostics are record-driven (block-correct Sobol CIs on per-variable
        # bases) rather than a uniform rebuild.
        _rec_phi, _rec_w, _rec_reg, _rec_pair_bi = phi1_train, w1, reg1, None

        # --- Stage B: Mixed second-order ---
        # K2=0 is the documented switch that disables pair interactions (matches
        # the uniform path, line ~399, and the first-order estimand P_1 of the
        # Hoeffding decomposition — Manuscript_Theoryv07 §3.2). It used to be
        # metadata-only on the mixed path (pairs were fitted anyway using the
        # per-variable basis sizes); now it genuinely produces no pair features,
        # indices, result block, or CIs (DEC-045).
        K2_mixed = int(cfg.get('K2', 3) or 0)
        stage_b_requested = 'B' in stages
        stage_b_fitted = stage_b_requested and K2_mixed > 0
        if stage_b_requested and not stage_b_fitted:
            self._log("=== Stage B skipped: K2=0 disables pair interactions "
                      "(first-order / additive mixed model) ===")
        if stage_b_fitted:
            self._log("=== Stage B: Mixed second-order ===")

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
            self._log(f"  {P} pairs, {F2} second-order features")

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
                f0=jnp.array(f0, dtype=self._dtype),
                w1=jnp.array(w1, dtype=self._dtype),
                w2=jnp.array(w2, dtype=self._dtype),
                w3=jnp.array([], dtype=self._dtype),
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
                include_linear_1=True,
                include_linear_h1=include_linear_h1,
                include_linear_h2=include_linear_h2,
                include_linear_h3=include_linear_h3,
                basis_name='mixed',
                var_specs=var_specs_tuple,
                pair_block_info=pair_block_info_tuple,
            )

            pred_val = f0 + self._dtype(phi_full_val @ w_full)
            rmse_val = float(jnp.sqrt(jnp.mean((y_val - pred_val) ** 2)))
            pred_train = f0 + self._dtype(phi_full_train @ w_full)
            rmse_train = float(jnp.sqrt(jnp.mean((y_train - pred_train) ** 2)))
            results['stage_B'] = {'rmse_train': rmse_train, 'rmse_val': rmse_val}
            self._log(f"  RMSE train: {rmse_train:.4f}, val: {rmse_val:.4f}")

            _rec_phi, _rec_w, _rec_reg = phi_full_train, w_full, reg_full
            _rec_pair_bi = pair_block_info_tuple

        results['mixed_basis'] = True
        results['var_specs'] = var_specs_dict
        # Machine-readable capability metadata (DEC-045): which stages actually
        # ran, pair behavior, and that no selection/pruning was applied on the
        # mixed path (it is fenced above, so this is always False here).
        results['selection_applied'] = False
        results['mixed_capability'] = {
            'stages': ['A'] + (['B'] if stage_b_fitted else []),
            'pairs': 'all' if stage_b_fitted else 'none',
            'K2': K2_mixed,
            'stage_b_requested': stage_b_requested,
            'stage_b_fitted': stage_b_fitted,
            'selection_applied': False,
            'pruning_applied': False,
            'implicit_selection_neutralized': bool(
                cfg.get('_mixed_selection_neutralized', False)),
        }
        results['fitted_design'] = build_mixed_record(
            _rec_phi, _rec_w, _rec_reg, y_train, D,
            var_specs_tuple, pair_block_info=_rec_pair_bi, f0=f0)
        return model, results
