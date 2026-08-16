"""Training orchestrator: staged + alternating training.

This is the main entry point for fitting an HiFiANOVA model.
"""

import copy
import difflib
import warnings

import jax
from ..array_backend import xp as jnp  # switchable array backend (numpy exact core)
import equinox as eqx
import numpy as np
from typing import Dict, Optional, Tuple

from ..core.gram import build_gram_matrix, build_gram_matrix_2d, build_gram_matrix_3d
from ..core.features import (
    build_first_order_features, build_second_order_features,
    build_third_order_features,
    basis_size,
)
from ..core.pairs import (PairManager, TripleManager,
                          pair_manager_from_pairs as _pair_manager_from_pairs)
from ..model.mean_model import MeanModel
from ..model.hifi_anova import HiFiANOVA
from ..model.residual_net import create_residual_mlp
from .regularization import (build_regularization_vector)
from .ridge import weighted_ridge_solve
from ..precision import fit_dtype
from .mode import resolve_mode, auto_decide_next_stage
from .fitted_design import (
    build_record,
)
from ._trainer_helpers import (
    _prune_first_order_blocks,
)
from .stages.stage_d import (
    StageDMixin,
)
# Re-exported for the test suite: test_stage_d_estimator_identity imports these
# via ``hifi_anova.training.trainer`` (their historical public location). They
# are unused inside this module, so ruff would flag F401 — keep the intentional
# re-export explicit.
from .stages.stage_d import (  # noqa: F401
    _log_variance_bound_active,
    _resolve_stage_d_estimator,
)
from .stages.stage_b import StageBMixin
from .stages.mixed import MixedBasisMixin


# Every top-level config key the trainer recognizes. ``HiFiANOVATrainer.__init__``
# validates the (mode-resolved) config against this set and raises on anything
# outside it — an unrecognized key can no longer silently no-op (the trainer
# reads ~55 keys via ``cfg.get(...)`` and ignores the rest, so a typo like
# ``stategy=`` or ``selection_method=`` used to do nothing with no warning).
# UPDATE THIS SET WHEN ADDING A TOP-LEVEL CONFIG KEY. Nested-dict keys — the
# ``residual`` / ``residual_nn`` specs and ``basis_per_variable`` per-variable
# specs — are intentionally NOT listed; only the top level is validated. Keys
# prefixed ``_auto`` are set internally by ``resolve_mode``; ``allow_unknown_keys``
# is the opt-out escape hatch. (DEC-036)
KNOWN_CONFIG_KEYS = frozenset({
    # basis / order sizes
    'K1', 'K2', 'K3', 'Kh', 'K2h', 'K3h',
    'basis_name', 'basis_type', 'basis_per_variable',
    # user-defined term structure (X11C-S02): order-selective variable
    # membership and the variance-variable subset. K2 may also be a per-pair
    # mapping {(i, j): K2_ij} (explicit pairs + per-pair order), and
    # var_pair_selection may be an explicit [(i, j), ...] list.
    'variable_orders', 'variance_variables',
    # staging / mode / regularization strategy
    'stages', 'mode', 'strategy',
    # regularization strengths
    'lambda_order1', 'lambda_order2', 'lambda_order3',
    'lambda_h', 'lambda_h2', 'lambda_h3',
    'lambda_residual', 'lambda_h_residual',
    # variable / pair / triple selection & pruning
    'variable_selection', 'first_order_pruning',
    'pair_candidates', 'pair_selection', 'pair_pruning', 'pair_threshold',
    'max_pair_variables', 'triple_selection', 'triple_pruning',
    'var_pair_selection', 'var_triple_selection',
    'variance_selection_margin', 'variance_selection_mean_consistent',
    'variance_selection_mean_fallback', 'variance_residual',
    # residual (Stage C) — nested dict; only the top-level key is listed
    'residual', 'residual_nn',
    # heteroscedastic (Stage D) alternation
    'heteroscedastic_guard', 'min_noise_ratio', 'leverage_correction',
    'alternating_early_stop', 'alternating_tol', 'max_outer_iter',
    'newton_max_iter', 'stage_d_joint_gls_mean', 'stage_d_estimator',
    # linear-term toggles
    'include_linear_1', 'include_linear_2', 'include_linear_3',
    'include_linear_h1', 'include_linear_h2', 'include_linear_h3',
    # misc
    'precision', 'verbose', 'auto_threshold',
    'array_backend',  # provenance: 'jax' | 'numpy' (numpy exact core)
    # internal (set by resolve_mode / the one-call API) + escape hatch
    '_auto_mode', '_auto_threshold', '_mixed_selection_neutralized',
    'allow_unknown_keys',
})


class HiFiANOVATrainer(StageDMixin, StageBMixin, MixedBasisMixin):
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

    def __init__(self, config: Dict, *, progress=None, should_stop=None):
        # Optional GUI/back-end hooks (hifi_anova.progress). ``progress`` is
        # called with a JSON-friendly event dict at stage boundaries;
        # ``should_stop`` is polled between stages (and on each Stage-D outer
        # iteration) and aborts the fit with HiFiCancelled when truthy. Both
        # default to None → no events, never cancelled (unchanged behavior).
        self._progress = progress
        self._should_stop = should_stop
        # Copy the caller's config before any resolution/stage logic touches it.
        # ``resolve_mode`` already shallow-copies, but nested dicts (e.g.
        # ``residual_nn``) would still be shared and get surprise-mutated
        # (``residual_nn['enabled'] = True`` in auto/stage-C paths); a deepcopy
        # makes the caller's dict — nested dicts included — provably immune to
        # trainer-side writes. config holds scalars/strings/small dicts (never
        # data arrays), so this is cheap. (DEC-036, finding #2)
        self.config = resolve_mode(copy.deepcopy(config))
        self._validate_config_keys()
        # Value-level validation (DEC-046): stage list, numeric type/range, enum
        # choices, and nested residual/variance_residual specs. Runs regardless
        # of allow_unknown_keys (that hatch is for experimental *keys*, not a
        # bypass of type/shape/range safety). basis_per_variable index-range
        # validation needs D and happens on the mixed path (``_fit_mixed``).
        from ..validation import validate_config
        validate_config(self.config)
        # Whether the trainer prints stage-by-stage progress. The one-call API
        # forwards its ``verbose`` argument here so ``verbose=False`` actually
        # silences the trainer (previously only the final summary was hidden).
        self._verbose = bool(self.config.get('verbose', True))
        # Fit-weight dtype: float32 by default (DEC-035); float64 is opt-in via
        # config['precision']='float64' / HIFI_ANOVA_X64. Every model weight and
        # in-loop prediction cast in this trainer uses self._dtype, so the fit is
        # genuinely float64 end-to-end when requested (float64 also enables x64).
        self._dtype = fit_dtype(self.config.get('precision'))

    def _validate_config_keys(self):
        """Fail loudly on an unrecognized top-level config key (DEC-036).

        The trainer reads its ~55 keys via ``cfg.get(...)`` and silently
        ignores everything else, and the one-call API funnels arbitrary
        ``**kwargs`` into the config — so a typo (``stategy``, ``hetero``,
        ``selection_method``) used to be a no-op with no error or warning.
        Anything outside :data:`KNOWN_CONFIG_KEYS` now raises a ``ValueError``
        naming the offending key(s) and the nearest known key. Only the TOP
        level is checked (nested residual/basis specs are out of scope). Set
        ``allow_unknown_keys=True`` to bypass (forward-compat experiments).
        """
        if self.config.get('allow_unknown_keys', False):
            return
        unknown = set(self.config) - KNOWN_CONFIG_KEYS
        if not unknown:
            return
        # Suggest the nearest known key for each typo (difflib).
        hints = []
        for k in sorted(unknown):
            close = difflib.get_close_matches(k, KNOWN_CONFIG_KEYS, n=1)
            hints.append(f"{k!r}" + (f" (did you mean {close[0]!r}?)" if close else ""))
        raise ValueError(
            "Unknown config key(s): " + ", ".join(hints) + ". "
            "The trainer silently ignores unrecognized keys, so this is most "
            "likely a typo. Pass allow_unknown_keys=True to bypass this check "
            "(only top-level keys are validated)."
        )

    def _log(self, *args, **kwargs):
        """Print progress only when running verbosely."""
        if self._verbose:
            print(*args, **kwargs)

    def _resolve_term_structure(self, cfg, D, K2):
        """Resolve the user-defined term-structure keys against the config.

        Returns ``(K2_scalar, explicit_pairs, pair_k2, fo_included,
        pair_excluded_vars)``:

        * ``K2_scalar`` — the effective scalar order (``max`` of a per-pair
          mapping, else ``K2`` unchanged);
        * ``explicit_pairs`` — the exact (i, j) pair list a ``K2`` mapping
          names, else ``None`` (pairs come from the usual selection paths);
        * ``pair_k2`` — per-pair orders aligned with ``explicit_pairs``;
        * ``fo_included`` — ascending variable indices whose FIRST-ORDER block
          stays in the mean design (``None`` = all D; ``variable_orders``
          entries without order 1 are excluded — their marginal is dropped and
          no df is spent on it);
        * ``pair_excluded_vars`` — variables excluded from every pair
          (``variable_orders`` entries without order 2).

        Term structure is fully user-specified, so it composes only with
        data-INDEPENDENT pair selection (``pair_selection=None/'all'`` or an
        explicit list; a ``K2`` mapping names the pairs itself) — data-driven
        selection/pruning heuristics that read the first-order fit are
        rejected, as are mixed per-variable bases, third-order terms, and
        auto mode. Statistical honesty note: dropping a variable's marginal
        while keeping its pairs is a NON-HIERARCHICAL model — the pair terms
        absorb any true marginal effect along that variable, and the pair
        Sobol shares are conditional on that omission.
        """
        from ..validation import (validate_k2_spec, validate_variable_orders)
        K2_spec = validate_k2_spec(K2, D)
        explicit_pairs = None
        pair_k2 = None
        if isinstance(K2_spec, dict):
            explicit_pairs = sorted(K2_spec)
            pair_k2 = [K2_spec[p] for p in explicit_pairs]
            K2 = max(pair_k2)

        fo_included = None
        pair_excluded_vars = ()
        vo = cfg.get('variable_orders', None)
        if vo is not None:
            vo = validate_variable_orders(vo, D)
            fo_excluded = sorted(i for i, o in vo.items() if 1 not in o)
            pair_excluded_vars = tuple(
                sorted(i for i, o in vo.items() if 2 not in o))
            # mean-excluded membership (orders == ()): the variable carries NO
            # mean term. On a heteroscedastic fit its column serves the
            # variance model (the historical variance-only reading); on a
            # CONSTANT fit it is in NEITHER model — the column stays in X for
            # a post-hoc complement (``fit_residual``) to capture (BR-12; the
            # BR-11 projector leaves unsolved structure available to the
            # complement). Until a complement is attached such a variable is
            # inert, so it is WARNED about, not refused. The all-excluded
            # limit is the INTERCEPT-ONLY mean design (f0 only) — the
            # complement-only base: legitimate, exploratory, disclosed.
            mean_excluded = sorted(i for i, o in vo.items() if not o)
            if mean_excluded and 'D' not in (cfg.get('stages') or ()):
                warnings.warn(
                    f"variable_orders assigns variable(s) {mean_excluded} an "
                    "empty order set on a constant-noise fit: they are in "
                    "neither the mean nor a variance model. Their columns "
                    "stay available to a post-hoc complement (fit_residual); "
                    "without one they are inert.", stacklevel=2)
            if len(mean_excluded) == D:
                warnings.warn(
                    "variable_orders excludes every variable from the mean "
                    "model: this is an INTERCEPT-ONLY mean fit (f0 only). "
                    "The structured decomposition is empty — intended as the "
                    "complement-only base (fit_residual captures everything "
                    "above f0); exploratory by construction.", stacklevel=2)
            if fo_excluded:
                fo_included = [i for i in range(D) if i not in fo_excluded]

        active = (explicit_pairs is not None or fo_included is not None
                  or pair_excluded_vars)
        if active:
            if cfg.get('basis_per_variable') is not None:
                raise ValueError(
                    "user-defined term structure (a K2 mapping / "
                    "variable_orders) is not supported together with mixed "
                    "per-variable bases (basis_per_variable).")
            if cfg.get('K3', 0):
                raise ValueError(
                    "user-defined term structure (a K2 mapping / "
                    "variable_orders) does not support third-order terms; "
                    "set K3=0.")
            if cfg.get('_auto_mode', False):
                raise ValueError(
                    "mode='auto' is not supported with user-defined term "
                    "structure (a K2 mapping / variable_orders); choose the "
                    "stages explicitly.")
        if explicit_pairs is not None:
            for key in ('pair_selection', 'pair_candidates',
                        'max_pair_variables'):
                if cfg.get(key) is not None:
                    raise ValueError(
                        f"K2 given as a per-pair mapping names the retained "
                        f"pairs itself; {key!r} must not also be set.")
            if cfg.get('variable_selection') is not None:
                raise ValueError(
                    "K2 given as a per-pair mapping names the retained pairs "
                    "itself; variable_selection must be None (pass "
                    "variable_selection=None in the one-call API).")
            if cfg.get('pair_pruning', 'none') != 'none':
                raise ValueError(
                    "pair_pruning is not supported with a per-pair K2 "
                    "mapping: the pair set is user-specified, not selected.")
            bad = [p for p in explicit_pairs
                   if p[0] in pair_excluded_vars or p[1] in pair_excluded_vars]
            if bad:
                raise ValueError(
                    f"K2 names pair(s) {bad} involving variable(s) that "
                    "variable_orders excludes from order 2; remove one of the "
                    "two directives.")
        if fo_included is not None:
            if cfg.get('first_order_pruning', 'none') != 'none':
                raise ValueError(
                    "first_order_pruning is not supported with "
                    "variable_orders first-order exclusions (the first-order "
                    "set is user-specified, not selected).")
            if explicit_pairs is None:
                ps = cfg.get('pair_selection', None)
                if cfg.get('variable_selection') is not None or \
                        cfg.get('pair_candidates') is not None or \
                        (ps is not None and not isinstance(ps, list)
                         and ps != 'all'):
                    raise ValueError(
                        "variable_orders with a first-order exclusion needs a "
                        "data-independent pair set: use a per-pair K2 mapping, "
                        "pair_selection=None/'all', or an explicit "
                        "pair_selection list (data-driven selection cannot "
                        "see a variable whose first-order block is excluded). "
                        "In the one-call API also pass "
                        "variable_selection=None.")
        return K2, explicit_pairs, pair_k2, fo_included, pair_excluded_vars

    def _emit(self, event, *, stage=None, message=None, metrics=None,
              fraction=None):
        """Emit a structured progress event to the ``progress`` hook, if set.

        No-op when no hook was supplied, so the common path is free. A raising
        callback propagates (it is the caller's code); it does not corrupt the
        fit, which holds no partial global state.
        """
        if self._progress is not None:
            from ..progress import make_event
            self._progress(make_event(event, stage=stage, message=message,
                                      metrics=metrics, fraction=fraction))

    def _check_cancel(self):
        """Raise :class:`HiFiCancelled` if the ``should_stop`` hook asks to stop.

        Cooperative checkpoint — call at stage boundaries / loop iterations.
        No-op when no hook was supplied.
        """
        if self._should_stop is not None and self._should_stop():
            from ..progress import HiFiCancelled
            raise HiFiCancelled(
                "fit cancelled by should_stop() callback")

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

        # --- User-defined term structure (X11C-S02: BR-04/BR-06) ---
        # Resolve BEFORE anything reads K2 arithmetically: a K2 mapping
        # {(i, j): K2_ij} pins the exact pair set with per-pair orders, and
        # variable_orders can exclude a variable's first-order block (order-2-
        # only membership) or its pairs. All-None/scalar leaves every default
        # path byte-identical. Stashed on self for the stage mixins.
        (K2, _explicit_pairs, _pair_k2, _fo_included,
         _pair_excluded_vars) = self._resolve_term_structure(cfg, D, K2)
        self._pair_k2 = _pair_k2
        self._fo_included = _fo_included
        strategy = cfg.get('strategy', 'variance')
        lambda1 = cfg.get('lambda_order1', 0.001)
        lambda2 = cfg.get('lambda_order2', 0.01)
        lambda3 = cfg.get('lambda_order3', 0.1)
        lambda_h = cfg.get('lambda_h', 0.1)
        # Mean-only default (A, B). Reaching here without 'stages' means the
        # config bypassed resolve_mode(); keep the safe default consistent with
        # it — variance (D) and residual (C) stay opt-in. See mode.resolve_mode.
        stages = cfg.get('stages', ['A', 'B'])
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

        # Legacy: pair_selection does variable selection + candidate gen in one.
        # When it is an explicit list of active variable indices, range-check it
        # against D now that the data shape is known (DEC-046) — an out-of-range
        # index used to form a phantom pair and crash/mis-index downstream.
        pair_selection = cfg.get('pair_selection', None)
        from ..validation import validate_pair_selection
        validate_pair_selection(pair_selection, D)
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
        # Machine-readable term-structure provenance + honesty label. A pair
        # whose variable has no first-order block is a NON-HIERARCHICAL model:
        # its pair share is conditional on the omitted marginal and may absorb
        # marginal leakage — recorded so downstream consumers can label it.
        if (_pair_k2 is not None or _fo_included is not None
                or _pair_excluded_vars):
            _fo_exc = ([i for i in range(D) if i not in _fo_included]
                       if _fo_included is not None else [])
            results['term_structure'] = {
                'pair_k2': ({f"{i},{j}": int(k) for (i, j), k in
                             zip(_explicit_pairs, _pair_k2)}
                            if _pair_k2 is not None else None),
                'first_order_excluded': _fo_exc,
                'pair_excluded': list(_pair_excluded_vars),
                # variance-only variables: excluded from BOTH mean orders —
                # their column serves only the (Stage-D) variance model
                'mean_excluded': sorted(
                    set(_fo_exc) & set(_pair_excluded_vars)),
                'note': ('user-specified term structure; a variable excluded '
                         'from order 1 but kept in pairs makes the model '
                         'non-hierarchical — its pair Sobol share is '
                         'conditional on the omitted marginal (may include '
                         'marginal leakage). A variable excluded from BOTH '
                         'mean orders is variance-only: if it truly affects '
                         'the MEAN, that misfit inflates the residuals the '
                         'variance model sees (biased sigma^2(x)) — a user '
                         'assertion, not a data-driven finding.'),
            }
        self._emit('fit_start', metrics={'planned_stages': list(stages)})

        # ======== Mixed per-variable basis path ========
        basis_per_variable = cfg.get('basis_per_variable', None)
        if basis_per_variable is not None:
            self._check_cancel()
            self._emit('stage_start', stage='mixed')
            return self._fit_mixed(
                x_train, y_train, x_val, y_val, key, cfg, D,
                basis_per_variable, stages, strategy, lambda1, lambda2,
                include_linear_h1, include_linear_h2, include_linear_h3,
            )

        # Shared infrastructure. With per-pair K2 there is no single shared
        # pair Gram — per-pair Grams are built where needed (record/Sobol).
        G1 = build_gram_matrix(K1, include_linear_1, basis_name)
        G2 = (build_gram_matrix_2d(build_gram_matrix(K2, include_linear_2, basis_name))
               if K2 > 0 and _pair_k2 is None else None)
        G3 = (build_gram_matrix_3d(build_gram_matrix(K3, include_linear_3, basis_name))
               if K3 > 0 else None)

        # ======== Stage A: First-order only ========
        self._check_cancel()
        self._emit('stage_start', stage='A')
        (model, w1, f0, y_centered, phi1_train, phi1_val, reg1,
         rmse_val_a) = self._fit_stage_a(
            x_train, y_train, x_val, y_val, D, K1, strategy, lambda1, lambda2,
            G1, include_linear_1, include_linear_2, include_linear_3,
            include_linear_h1, include_linear_h2, include_linear_h3,
            basis_name, results)
        self._emit('stage_end', stage='A', metrics={'rmse_val': rmse_val_a})

        # Auto mode: decide whether to add stage B
        if auto_mode and 'B' not in stages:
            next_s = auto_decide_next_stage(
                'A', rmse_val_a, var_y_val, threshold=auto_threshold)
            if next_s == 'B':
                stages = list(stages) + ['B']

        # First-order pruning for first-order-only models (Stage B handles it
        # otherwise, on the full design).
        stage_b_will_run = K2 > 0 and ('B' in stages or 'D' in stages)
        if first_order_pruning != 'none' and not stage_b_will_run:
            self._log(f"=== First-order pruning ({first_order_pruning}) ===")
            block1_fo = basis_size(K1, include_linear_1, basis_name)
            w1_pruned, fo_info = _prune_first_order_blocks(
                w1, phi1_train, reg1, y_centered, D, block1_fo, G1,
                first_order_pruning, verbose=self._verbose)
            results['first_order_pruning'] = fo_info
            model = eqx.tree_at(lambda m: m.mean_model.w1, model,
                                jnp.asarray(w1_pruned, dtype=self._dtype))

        # ======== Pair candidate generation for Stage B ========
        if _explicit_pairs is not None:
            # A K2 mapping names the exact pair set (with per-pair orders):
            # bypass every selection path and build the manager verbatim.
            pair_mgr = _pair_manager_from_pairs(D, _explicit_pairs)
            self._log(f"  Pairs: explicit per-pair K2 mapping → "
                      f"{pair_mgr.P} pairs (orders {sorted(set(_pair_k2))})")
        elif K2 > 0:
            pair_mgr = self._generate_pair_candidates(
                x_train, phi1_train, y_centered, w1, reg1, D, K1, K2, G1,
                variable_selection, pair_candidates, pair_selection,
                max_pair_variables, pair_threshold, strategy, lambda1,
                include_linear_1, include_linear_2, basis_name, results)
        else:
            # K2=0 is the documented switch for disabling pair interactions.
            # Keep an empty manager for the Stage-D plumbing, but do not run
            # selection or construct the Fourier K=0 linear-product feature.
            pair_mgr = PairManager(
                D, active_variables=[], selection_mode='both')

        # variable_orders order-1-only entries: drop every pair touching a
        # pair-excluded variable (data-independent filter; a K2 mapping naming
        # such a pair was already rejected in _resolve_term_structure).
        if _pair_excluded_vars and pair_mgr.P > 0:
            _pi = np.asarray(pair_mgr.pair_indices)
            keep = [(int(a), int(b)) for a, b in _pi
                    if int(a) not in _pair_excluded_vars
                    and int(b) not in _pair_excluded_vars]
            if len(keep) < pair_mgr.P:
                self._log(f"  variable_orders: dropped "
                          f"{pair_mgr.P - len(keep)} pair(s) touching "
                          f"order-1-only variable(s) {list(_pair_excluded_vars)}")
                pair_mgr = _pair_manager_from_pairs(D, keep)
        # Order-2-only variables that ended up in no pair have no term at all.
        # Deliberately variance-only variables (empty order set, hetero fit)
        # are exempt — being absent from the MEAN model is their point.
        if _fo_included is not None:
            _pi = np.asarray(pair_mgr.pair_indices).reshape(-1, 2)
            _in_pairs = set(int(v) for v in _pi.ravel())
            _vo = cfg.get('variable_orders') or {}
            _var_only = {int(i) for i, o in _vo.items() if not o}
            _orphans = [i for i in range(D)
                        if i not in _fo_included and i not in _in_pairs
                        and i not in _var_only]
            if _orphans:
                warnings.warn(
                    f"variable_orders excludes the first-order block of "
                    f"variable(s) {_orphans}, but no retained pair involves "
                    "them — they are absent from the model entirely. Add the "
                    "pair(s) explicitly (e.g. a K2 mapping or a "
                    "pair_selection list) if that is not intended.",
                    stacklevel=2)

        # ======== Stage B: First + Second order ========
        if K2 > 0 and ('B' in stages or 'D' in stages):
            self._check_cancel()
            self._emit('stage_start', stage='B')
            model, pair_mgr, K2 = self._fit_stage_b(
                model, x_train, y_train, x_val, y_val, y_centered, f0,
                phi1_train, phi1_val, reg1, pair_mgr, D, K1, K2, K3,
                G1, G2, G3, strategy, lambda1, lambda2, lambda3,
                pair_pruning, triple_selection, triple_pruning,
                pair_threshold, first_order_pruning,
                include_linear_1, include_linear_2, include_linear_3,
                include_linear_h1, include_linear_h2, include_linear_h3,
                basis_name, results)
            self._emit('stage_end', stage='B',
                       metrics={'rmse_val': (results.get('stage_B') or {}).get('rmse_val')})
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
            self._check_cancel()
            self._emit('stage_start', stage='C')
            model, key = self._fit_stage_c(
                model, x_train, y_train, x_val, y_val, key, cfg, D, results)
            self._emit('stage_end', stage='C',
                       metrics={'rmse_val': (results.get('stage_C') or {}).get('rmse_val')})

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
            self._check_cancel()
            self._emit('stage_start', stage='D')
            self._log("=== Stage D: Heteroscedastic variance ===")
            model = self._fit_heteroscedastic(
                model, x_train, y_train, x_val, y_val,
                pair_mgr, G1, G2, K1, K2, Kh, D,
                strategy, lambda1, lambda2, lambda_h, results
            )
            self._emit('stage_end', stage='D',
                       metrics={'rmse_val': (results.get('stage_D') or {}).get('rmse_val')})

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
                                jnp.asarray(w1_cur, dtype=self._dtype))

        return model, results

    def _fit_stage_a(self, x_train, y_train, x_val, y_val, D, K1, strategy,
                     lambda1, lambda2, G1,
                     include_linear_1, include_linear_2, include_linear_3,
                     include_linear_h1, include_linear_h2, include_linear_h3,
                     basis_name, results):
        """Stage A: first-order-only ridge fit.

        Extracted verbatim from ``fit`` (behavior-preserving). Returns the
        Stage-A model plus the shared quantities the later stages reuse
        (coefficients, intercept/centered target, first-order features and
        penalty, validation RMSE for the auto-mode decision); records the
        Stage-A diagnostics and the first-order fitted-design record in
        ``results``.
        """
        self._log("=== Stage A: First-order model ===")
        # Order-selective mean (variable_orders): the design's first-order
        # block spans only the included variables — an excluded variable's
        # marginal is not merely shrunk to zero, it is absent from the solve
        # (no df spent). The MODEL still carries the full uniform w1 layout
        # with exact zeros in the excluded blocks (see _scatter_first_order).
        fo_included = getattr(self, '_fo_included', None)
        if fo_included is not None:
            from ..core.features import build_first_order_features_subset
            phi1_train = build_first_order_features_subset(
                x_train, K1, fo_included, include_linear=include_linear_1,
                basis_name=basis_name)
            phi1_val = build_first_order_features_subset(
                x_val, K1, fo_included, include_linear=include_linear_1,
                basis_name=basis_name)
            self._log(f"  variable_orders: first-order block spans "
                      f"{len(fo_included)}/{D} variables")
        else:
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
        n_fo = len(fo_included) if fo_included is not None else D
        reg1 = build_regularization_vector(n_fo, K1, 0, 0, strategy, lambda1, lambda2,
                                              include_linear_1=include_linear_1,
                                              basis_name=basis_name)
        w1 = weighted_ridge_solve(phi1_train, y_centered, reg1)

        # Build model (Level 0)
        if fo_included is not None:
            from ._trainer_helpers import _scatter_first_order
            w1_model = _scatter_first_order(
                w1, D, basis_size(K1, include_linear_1, basis_name),
                fo_included, dtype=self._dtype)
        else:
            w1_model = jnp.array(w1, dtype=self._dtype)
        mean_model = MeanModel(
            f0=jnp.array(f0, dtype=self._dtype),
            w1=w1_model,
            w2=jnp.array([], dtype=self._dtype),
            w3=jnp.array([], dtype=self._dtype),
            K1=K1, K2=0, K3=0, D=D,
            include_linear_1=include_linear_1, basis_name=basis_name,
        )

        model = HiFiANOVA(
            mean_model=mean_model,
            K1=K1, K2=0, K3=0, Kh=0, D=D,
            pair_indices=None, triple_indices=None,
            include_linear_1=include_linear_1,
            include_linear_2=include_linear_2,
            include_linear_3=include_linear_3,
            include_linear_h1=include_linear_h1,
            include_linear_h2=include_linear_h2,
            include_linear_h3=include_linear_h3,
            basis_name=basis_name,
            # solved-design layout on the Stage-A-only path too (BR-11/BR-12):
            # without it a first-order homoscedastic variable_orders fit left
            # the model claiming the FULL layout while the record held the
            # subset — build_phi_all_fit and the residual projector need the
            # truth. ``()`` = the intercept-only mean design.
            fo_included=(tuple(fo_included) if fo_included is not None
                         else None),
        )

        # Evaluate Stage A
        pred_train_a = f0 + self._dtype(phi1_train @ w1)
        pred_val_a = f0 + self._dtype(phi1_val @ w1)
        rmse_train_a = float(jnp.sqrt(jnp.mean((y_train - pred_train_a) ** 2)))
        rmse_val_a = float(jnp.sqrt(jnp.mean((y_val - pred_val_a) ** 2)))
        results['stage_A'] = {'rmse_train': rmse_train_a, 'rmse_val': rmse_val_a}
        self._log(f"  RMSE train: {rmse_train_a:.4f}, val: {rmse_val_a:.4f}")

        # Fitted-design record (first-order only). Overwritten below if Stage B
        # extends the design; carried to whichever return site fires so the API
        # diagnoses the design the trainer actually solved (homoscedastic here —
        # Stage D attaches precision weights). See training/fitted_design.py.
        results['fitted_design'] = build_record(
            phi1_train, w1, reg1, y_train, D,
            K1, G1, include_linear_1, basis_name,
            fo_included=fo_included)

        return (model, w1, f0, y_centered, phi1_train, phi1_val, reg1,
                rmse_val_a)

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

        # Backward compat: old config uses 'enabled' flag for NN. ``enabled`` is
        # now honored uniformly (DEC-046): an analytic residual with
        # ``enabled=False`` skips Stage C too, so a disabled residual is
        # unambiguous regardless of family (default for analytic is enabled).
        if residual_type == 'nn' and not residual_cfg.get('enabled', False):
            pass  # Skip Stage C if NN not enabled
        elif (residual_type in ('rbf', 'rff', 'nystrom')
              and residual_cfg.get('enabled', True)):
            # === ANALYTIC PIPELINE (linear residual) ===
            self._log(f"=== Stage C: Linear residual ({residual_type}) ===")
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
            self._log("=== Stage C: Residual NN ===")
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
            self._log(f"  RMSE val: {rmse_val_c:.4f}")
        return model, key

    # ================================================================
    # Mixed per-variable basis path
    # ================================================================


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
    """Experimental additivity-calibrated Sobol estimation mode.

    This is a SEPARATE mode from predictive fitting. The goal is
    sensitivity estimation under an additivity calibration, not good prediction
    on new data. The calibration is heuristic and does not guarantee unbiased
    recovery.

    Two approaches:
    - auto_lambda=True: finds lambda such that sum(Sobol) ~ 1.0
      (the experimental "additivity criterion")
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
                i, j, k = (int(triple_mgr.triple_indices[t_idx, pos]) for pos in range(3))
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
