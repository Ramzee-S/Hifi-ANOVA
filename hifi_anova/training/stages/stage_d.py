"""Stage D: alternating heteroscedastic (joint mean / log-variance) fit.

Extracted verbatim from ``trainer.py`` (behavior-preserving step 2 of the
trainer decomposition). Holds the Stage-D estimator-identity resolver and
bound-activity helper, the ``_StageDDesigns`` design bundle, and the three
Stage-D methods as ``StageDMixin`` — composed by ``HiFiANOVATrainer`` so
``self`` (``self.config``, ``self._dtype``, ``self._log``) resolves to the
trainer instance exactly as before. Relative imports are one level deeper than
in ``trainer.py`` because this module sits in the ``training.stages`` subpackage.
"""

import dataclasses
import warnings
from collections import namedtuple

from ...array_backend import xp as jnp  # switchable array backend (numpy exact core)
import numpy as np

from ...core.gram import (build_gram_matrix, build_gram_matrix_2d,
                          build_gram_matrix_3d)
from ...core.features import (
    build_first_order_features, build_second_order_features,
    build_third_order_features,
)
from ...core.pairs import PairManager, TripleManager
from ...model.mean_model import MeanModel
from ...model.variance_model import VarianceModel, LOG_VAR_CLIP
from ...model.hifi_anova import HiFiANOVA
from ...model.linear_residual import predict_residual_batch
from ..regularization import (build_regularization_vector,
                              build_variance_regularization_vector)
from ..ridge import weighted_ridge_solve, leverage_diag, debias_squared_residuals
from ..newton import newton_solve_log_variance
from ..fitted_design import (
    build_record, VarianceDesign,
    MEAN_INTERCEPT_PROFILED_JOINT_GLS, MEAN_INTERCEPT_LEGACY_FIXED,
    MEAN_INTERCEPT_UNWEIGHTED,
)
from .._trainer_helpers import _gaussian_nll, _split_mean_coeffs


# --- Stage-D estimator identity (P0-2 / X6 Session 2) -------------------------
# A truth-in-labelling selector over the alternating heteroscedastic fit. The
# DEFAULT is a *leverage-adjusted alternating quasi-likelihood with validation
# iterate selection* (leverage_correction + alternating_early_stop both on): the
# leverage-corrected residual moment makes each variance step a quasi-likelihood
# update rather than a block-coordinate-descent step on eq:model, and
# validation-best iterate selection means the fitted point need not be a
# stationary point of any single stated objective. Only ``raw_likelihood``
# (both off) is a monotone block-coordinate descent, and only while the
# log-variance solution stays interior (no ``bound_active``).
STAGE_D_ESTIMATOR_ADJUSTED = 'adjusted_quasi_likelihood'
STAGE_D_ESTIMATOR_RAW = 'raw_likelihood'
# Each selector's implied (leverage_correction, alternating_early_stop).
_STAGE_D_ESTIMATOR_FLAGS = {
    STAGE_D_ESTIMATOR_ADJUSTED: (True, True),
    STAGE_D_ESTIMATOR_RAW: (False, False),
}


def _mean_predict_on_designs(mean_model, phi1, phi2, phi3, fo_included,
                             block1):
    """Predict the structured mean from STAGE-D TRAINING designs.

    With an order-selective first-order block (``fo_included``) the stage's
    ``phi1`` is a subset layout while the model's ``w1`` is the full uniform
    layout (zeros in excluded blocks) — gather the included blocks so the
    matmul shapes agree. ``fo_included is None`` calls ``predict`` verbatim
    (byte-identical default path).
    """
    if fo_included is None:
        return mean_model.predict(phi1, phi2, phi3)
    w1 = mean_model.w1
    if len(fo_included) == 0:
        # intercept-only mean design (BR-12): no first-order block at all
        out = mean_model.f0 + jnp.zeros(phi1.shape[0], dtype=w1.dtype)
    else:
        w1_sub = jnp.concatenate([w1[i * block1:(i + 1) * block1]
                                  for i in fo_included])
        out = mean_model.f0 + phi1 @ w1_sub
    if phi2 is not None and mean_model.K2 > 0 and len(mean_model.w2) > 0:
        out = out + phi2 @ mean_model.w2
    if phi3 is not None and mean_model.K3 > 0 and len(mean_model.w3) > 0:
        out = out + phi3 @ mean_model.w3
    return out


def _log_variance_bound_active(h):
    """True iff any fitted train log-variance touches the ±``LOG_VAR_CLIP`` box the
    variance solve clamps to (P1-1 DETECT+label). Reads the module-level
    ``LOG_VAR_CLIP`` at call time so it tracks the shared clip. When active, the
    reported log-variance is a clipped value, not an interior stationary point, so
    no exact gradient/stationarity (nor monotone block-descent) claim holds."""
    h = np.asarray(h, dtype=np.float64)
    tol = 1e-9 * LOG_VAR_CLIP
    return bool(np.any(h <= -LOG_VAR_CLIP + tol)
                or np.any(h >= LOG_VAR_CLIP - tol))


def _resolve_stage_d_estimator(cfg):
    """Resolve the Stage-D estimator selector + legacy flags into the effective
    ``(leverage_correction, alternating_early_stop)`` pair plus honest
    estimator-identity metadata (P0-2).

    The public boundary distinguishes an *omitted* selector from an *explicit*
    one via config-key presence:

    * selector omitted + flags omitted  -> adjusted default (both on);
    * selector omitted + legacy flags   -> preserve the flags' behaviour and
      describe the *effective* components honestly (never rejected);
    * selector explicit + flags omitted -> resolve the flags from the selector;
    * selector explicit + explicit flags that contradict it -> ``ValueError``.

    Returns ``(lev_correct, early_stop, identity)`` where ``identity`` is a dict
    with ``estimator`` / ``objective_family`` / ``residual_update`` /
    ``iterate_selection`` (JSON-safe strings) for ``results['stage_D']``.
    """
    selector = cfg.get('stage_d_estimator', None)
    if selector is not None and selector not in _STAGE_D_ESTIMATOR_FLAGS:
        raise ValueError(
            f"stage_d_estimator must be one of "
            f"{sorted(_STAGE_D_ESTIMATOR_FLAGS)}; got {selector!r}.")

    lev_explicit = 'leverage_correction' in cfg and cfg['leverage_correction'] is not None
    es_explicit = 'alternating_early_stop' in cfg and cfg['alternating_early_stop'] is not None

    if selector is not None:
        imp_lev, imp_es = _STAGE_D_ESTIMATOR_FLAGS[selector]
        if lev_explicit and bool(cfg['leverage_correction']) != imp_lev:
            raise ValueError(
                f"Contradictory Stage-D estimator request: "
                f"stage_d_estimator={selector!r} implies "
                f"leverage_correction={imp_lev}, but leverage_correction="
                f"{bool(cfg['leverage_correction'])} was passed explicitly. "
                f"Drop one of them (the selector resolves the flag), or make "
                f"them agree.")
        if es_explicit and bool(cfg['alternating_early_stop']) != imp_es:
            raise ValueError(
                f"Contradictory Stage-D estimator request: "
                f"stage_d_estimator={selector!r} implies "
                f"alternating_early_stop={imp_es}, but alternating_early_stop="
                f"{bool(cfg['alternating_early_stop'])} was passed explicitly. "
                f"Drop one of them (the selector resolves the flag), or make "
                f"them agree.")
        lev_correct = bool(cfg['leverage_correction']) if lev_explicit else imp_lev
        early_stop = bool(cfg['alternating_early_stop']) if es_explicit else imp_es
    else:
        # No selector: legacy behaviour — the flags default on.
        lev_correct = bool(cfg.get('leverage_correction', True))
        early_stop = bool(cfg.get('alternating_early_stop', True))

    # Effective identity from the resolved components (honest even for a
    # flag-only call whose combination has no selector name).
    if lev_correct and early_stop:
        estimator = STAGE_D_ESTIMATOR_ADJUSTED
    elif not lev_correct and not early_stop:
        estimator = STAGE_D_ESTIMATOR_RAW
    else:
        estimator = 'custom'
    identity = {
        'estimator': estimator,
        'objective_family': (STAGE_D_ESTIMATOR_ADJUSTED if lev_correct
                             else 'raw_penalized_likelihood'),
        'residual_update': ('leverage_adjusted' if lev_correct
                            else 'raw_squared_residuals'),
        'iterate_selection': ('validation_best' if early_stop
                             else 'final_iterate'),
    }
    return lev_correct, early_stop, identity


_StageDDesigns = namedtuple('_StageDDesigns', [
    'phi1_train',
    'phi2_train',
    'phi3_train',
    'phi_all_train',
    'psi1_train',
    'psi2_train',
    'psi3_train',
    'K2h',
    'K3h',
    'var_pair_mgr',
    'var_triple_mgr',
    'z_h_proj_train',
    'z_h_proj_coeffs',
    'var_residual_model',
    'M_h',
    'il1',
    'il2',
    'il3',
    'ilh1',
    'ilh2',
    'ilh3',
    'incl_lin_2',
    'incl_lin_3',
    'F1',
    'F2',
    'Fh',
    'Ph',
    'Th',
    'T',
    'reg_mean',
    'reg_var',
    # Term structure (X11C-S02): per-pair K2 layout of the mean pair block
    # (None = uniform), the order-selective first-order subset (None = all D),
    # and the variance-variable subset (None = all D). All None on default paths.
    'pair_block_info',
    'fo_included',
    'variance_variables',
])


class StageDMixin:
    """Stage-D methods mixed into :class:`HiFiANOVATrainer`.

    Composed onto the trainer so ``self`` is the trainer instance; these
    methods use ``self.config`` / ``self._dtype`` / ``self._log`` and call
    each other (``self._build_stage_d_designs`` /
    ``self._homoscedastic_fallback``) exactly as when they lived on the
    trainer class.
    """

    def _homoscedastic_fallback(self, model, mean_model, log_var,
                                pair_mgr, K1, K2, K3, Kh, D, basis_name):
        """Build a homoscedastic model: given mean, constant (scalar) variance.

        Used as the safe Stage-D fallback when the variance fit is ill-posed
        (no heteroscedastic signal, or the alternating loop degraded the mean).
        Keeps ``mean_model`` intact and attaches a constant log-variance, so
        prediction and (constant-width) intervals still work.
        """
        return HiFiANOVA(
            mean_model=mean_model,
            variance_model=None,
            residual_net=model.residual_net,
            constant_log_var=jnp.asarray(log_var, dtype=self._dtype),
            K1=K1, K2=K2, K3=K3, Kh=Kh, D=D,
            pair_indices=np.array(pair_mgr.pair_indices),
            triple_indices=model.triple_indices,
            include_linear_1=getattr(model, 'include_linear_1', True),
            include_linear_2=getattr(model, 'include_linear_2', True),
            include_linear_3=getattr(model, 'include_linear_3', True),
            include_linear_h1=getattr(model, 'include_linear_h1', True),
            include_linear_h2=getattr(model, 'include_linear_h2', True),
            include_linear_h3=getattr(model, 'include_linear_h3', True),
            basis_name=basis_name,
            # Carry the mean model's ragged pair layout (per-pair K2) so the
            # fallback model's build_phi2 matches its w2 (None on default paths).
            pair_block_info=getattr(mean_model, 'pair_block_info', None),
            pair_k2=getattr(mean_model, 'pair_k2', None),
            # Carry the order-selective first-order layout (BR-06) so the
            # reverted model's epistemic posterior rebuilds a record-consistent
            # design (build_phi_all_fit). None on default paths.
            fo_included=getattr(model, 'fo_included', None),
        )

    def _build_stage_d_designs(self, model, x_train, y_train, pair_mgr,
                               K1, K2, K3, Kh, D, strategy,
                               lambda1, lambda2, lambda_h, basis_name):
        """Assemble the Stage-D training designs and penalties.

        Extracted verbatim from ``_fit_heteroscedastic`` (behavior-preserving):
        mean features (all orders), variance features (first-order, optional
        K2h second / K3h third order with their pair/triple managers, optional
        projected variance-residual block), the layout sizes, and the mean/
        variance regularization vectors. Returns a ``_StageDDesigns``; ``K2h``/
        ``K3h`` come back possibly zeroed (no pairs / no triples).
        """
        cfg = self.config
        # Term structure (X11C-S02): resolved by the trainer before the stage
        # ladder; both None on every default path.
        fo_included = getattr(self, '_fo_included', None)
        pair_k2 = getattr(self, '_pair_k2', None)
        pair_block_info = None
        # --- Mean features (all orders) ---
        _il1 = getattr(model, 'include_linear_1', True)
        _il2 = getattr(model, 'include_linear_2', True)
        _il3 = getattr(model, 'include_linear_3', True)
        if fo_included is not None:
            from ...core.features import build_first_order_features_subset
            phi1_train = build_first_order_features_subset(
                x_train, K1, fo_included, include_linear=_il1,
                basis_name=basis_name)
        else:
            phi1_train = build_first_order_features(x_train, K1,
                                                      include_linear=_il1, basis_name=basis_name)
        if pair_k2 is not None:
            from ...core.features import build_second_order_features_per_pair
            phi2_train, pair_block_info = build_second_order_features_per_pair(
                x_train, pair_k2, pair_mgr.pair_indices,
                include_linear=_il2, basis_name=basis_name)
        else:
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
        from ...core.features import basis_size as _bs
        K2h = cfg.get('K2h', 0)
        lambda_h2 = cfg.get('lambda_h2', lambda_h * 10)
        _ilh1 = cfg.get('include_linear_h1', _il1)
        _ilh2 = cfg.get('include_linear_h2', _il2)
        _ilh3 = cfg.get('include_linear_h3', _il3)
        # Variance-variable subset (BR-01): the first-order variance design
        # spans only the selected variables. NOTE the statistical meaning:
        # excluding x_j ASSERTS homoscedasticity along x_j — a user modeling
        # assumption, not a data-driven finding; the Stage-D guard still tests
        # the WHOLE variance model against a constant, but not per-direction.
        variance_variables = cfg.get('variance_variables', None)
        if variance_variables is not None:
            from ...validation import validate_variance_variables
            variance_variables = validate_variance_variables(
                variance_variables, D)
            if len(variance_variables) == D:
                variance_variables = None   # full set — uniform path
        n_var_vars = (len(variance_variables) if variance_variables is not None
                      else D)
        Fh = n_var_vars * _bs(Kh, _ilh1, basis_name)
        if variance_variables is not None:
            from ...core.features import build_first_order_features_subset
            psi1_train = build_first_order_features_subset(
                x_train, Kh, list(variance_variables), include_linear=_ilh1,
                basis_name=basis_name)
            self._log(f"  Variance model: first-order block spans "
                      f"{n_var_vars}/{D} variables "
                      f"({list(variance_variables)}); sigma²(x) is flat along "
                      "the excluded variables by user assertion")
        else:
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
            if isinstance(var_pair_selection, (list, tuple)):
                # Explicit variance-pair list (BR-05): pin the exact (i, j)
                # pairs. Previously an explicit list silently behaved as 'all'.
                from ...validation import validate_var_pair_list
                from ...core.pairs import pair_manager_from_pairs
                _pairs = validate_var_pair_list(
                    var_pair_selection, D,
                    variance_variables=variance_variables)
                var_pair_mgr = pair_manager_from_pairs(D, _pairs)
            elif var_pair_selection in (None, 'all'):
                # All-at-once: all pairs — restricted to the variance-variable
                # subset when one is active (a variance pair on a variable
                # asserted variance-flat would contradict the assertion).
                if variance_variables is not None:
                    var_pair_mgr = PairManager(
                        D, active_variables=list(variance_variables),
                        selection_mode='both')
                else:
                    var_pair_mgr = PairManager(D)
            elif var_pair_selection == 'auto':
                # Sequential: quick first-order variance fit → select pairs
                from ...core.pairs import select_active_variables
                from ...core.gram import build_gram_matrix as _bgm

                _mean_pred = _mean_predict_on_designs(
                    model.mean_model, phi1_train, phi2_train, phi3_train,
                    fo_included, _bs(K1, _il1, basis_name))
                if model.residual_net is not None:
                    _mean_pred = _mean_pred + predict_residual_batch(
                        model.residual_net, x_train)
                _r2_init = (y_train - _mean_pred) ** 2

                _reg_h1 = build_variance_regularization_vector(
                    n_var_vars, Kh, strategy, lambda_h,
                    include_linear_h1=_ilh1, basis_name=basis_name)
                _wh_init, _ = newton_solve_log_variance(
                    psi1_train, _r2_init, jnp.zeros(Fh, dtype=jnp.float64),
                    float(jnp.log(jnp.mean(_r2_init))), _reg_h1, max_iter=5)

                # Log-variance S^h from quick fit -> select active variables
                # (positions in the possibly-subset psi1 layout; keys are TRUE
                # variable indices so the pair manager gets real indices).
                _Gh = jnp.asarray(_bgm(Kh, _ilh1, basis_name), dtype=jnp.float64)
                _var_sobol = {}
                _bh = _bs(Kh, _ilh1, basis_name)
                _vv_iter = (list(variance_variables)
                            if variance_variables is not None else range(D))
                for pos, i in enumerate(_vv_iter):
                    _wi = _wh_init[pos * _bh: (pos + 1) * _bh]
                    _var_sobol[int(i)] = float(jnp.maximum(0.0, _wi @ _Gh @ _wi))
                _total = sum(_var_sobol.values())
                if _total > 0:
                    _var_sobol = {i: v / _total for i, v in _var_sobol.items()}

                var_active = select_active_variables(
                    _var_sobol, D, threshold=cfg.get('pair_threshold', 0.01))
                var_pair_mgr = PairManager(
                    D, active_variables=var_active, selection_mode='both')
            else:
                # Unknown mode string (validation restricts to 'all'/'auto'/
                # an explicit list, so this is a defensive fallback): all
                # pairs, subset-restricted like the None/'all' branch.
                if variance_variables is not None:
                    var_pair_mgr = PairManager(
                        D, active_variables=list(variance_variables),
                        selection_mode='both')
                else:
                    var_pair_mgr = PairManager(D)

            if var_pair_mgr.P > 0:
                psi2_train = build_second_order_features(
                    x_train, K2h, var_pair_mgr.pair_indices,
                    include_linear=_ilh2, basis_name=basis_name)
                mode_str = var_pair_selection or 'all'
                self._log(f"  Variance second-order: {var_pair_mgr.P} pairs "
                      f"(K2h={K2h}, mode={mode_str})")
            else:
                K2h = 0
                self._log("  Variance second-order: skipped (no pairs)")

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
                self._log(f"  Variance third-order: {var_triple_mgr.T} triples (K3h={K3h})")
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
            from ..analytic_residual import create_residual
            from ...core.projection import project_features_orthogonal
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
            self._log(f"  Variance residual ({var_residual_type}): {M_h} features")

        _incl_lin_2 = getattr(model, 'include_linear_2', True)
        _incl_lin_3 = getattr(model, 'include_linear_3', True)

        F1 = int(phi1_train.shape[1])
        F2 = (int(phi2_train.shape[1]) if phi2_train is not None else 0)
        Ph = var_pair_mgr.P if var_pair_mgr is not None else 0
        Th = var_triple_mgr.T if var_triple_mgr is not None else 0

        # --- Regularization ---
        T = model.triple_indices.shape[0] if model.triple_indices is not None else 0
        lambda3 = cfg.get('lambda_order3', 0.1)
        n_fo = len(fo_included) if fo_included is not None else D
        reg_mean = build_regularization_vector(
            n_fo, K1, (pair_k2 if pair_k2 is not None else K2), pair_mgr.P,
            strategy, lambda1, lambda2,
            K3=K3, T=T, lambda_order3=lambda3,
            include_linear_1=_il1, include_linear_2=_incl_lin_2, include_linear_3=_incl_lin_3,
            basis_name=basis_name,
        )
        reg_var = build_variance_regularization_vector(
            n_var_vars, Kh, strategy, lambda_h,
            K2h=K2h, Ph=Ph, lambda_h2=lambda_h2,
            K3h=K3h, Th=Th, lambda_h3=lambda_h3,
            M_h_residual=M_h, lambda_h_res=lambda_h_res,
            include_linear_h1=_ilh1, include_linear_h2=_ilh2, include_linear_h3=_ilh3,
            basis_name=basis_name,
        )

        return _StageDDesigns(
            phi1_train=phi1_train,
            phi2_train=phi2_train,
            phi3_train=phi3_train,
            phi_all_train=phi_all_train,
            psi1_train=psi1_train,
            psi2_train=psi2_train,
            psi3_train=psi3_train,
            K2h=K2h,
            K3h=K3h,
            var_pair_mgr=var_pair_mgr,
            var_triple_mgr=var_triple_mgr,
            z_h_proj_train=z_h_proj_train,
            z_h_proj_coeffs=z_h_proj_coeffs,
            var_residual_model=var_residual_model,
            M_h=M_h,
            il1=_il1,
            il2=_il2,
            il3=_il3,
            ilh1=_ilh1,
            ilh2=_ilh2,
            ilh3=_ilh3,
            incl_lin_2=_incl_lin_2,
            incl_lin_3=_incl_lin_3,
            F1=F1,
            F2=F2,
            Fh=Fh,
            Ph=Ph,
            Th=Th,
            T=T,
            reg_mean=reg_mean,
            reg_var=reg_var,
            pair_block_info=pair_block_info,
            fo_included=fo_included,
            variance_variables=variance_variables)

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
        # The mean model as produced by Stages A/B (before the alternating
        # variance loop can perturb it). Kept so a degenerate/unstable Stage D
        # can revert to it rather than return a corrupted mean.
        stage_b_mean_model = model.mean_model
        # Whether to auto-fall-back to constant variance when Stage D is
        # ill-posed (no heteroscedastic signal / near-noiseless residuals) or
        # ends up degrading the mean. On by default; opt out with
        # heteroscedastic_guard=False to force the raw alternating fit (e.g. when
        # speed matters and the extra held-out evaluations are unwanted — note
        # the checks reuse the already-fitted Stage-B mean, so they add no extra
        # model fit, only a few validation evaluations).
        guard = cfg.get('heteroscedastic_guard', True)
        # Near-noiseless skip: a scale-free noise-to-signal variance ratio
        # (residual variance / total variance = 1 − R²). Below this, Stage D is
        # skipped in favour of a constant variance. Two roles: (1) literal
        # degeneracy — near-zero residuals make log(r²) numerically ill-posed;
        # (2) the entry-gate enforcement of the "don't model mean lack-of-fit as
        # noise" invariant — on essentially-noiseless data the residual is
        # deterministic mean-approximation error which is structured and
        # generalizes, so a held-out-NLL guard *cannot* tell it from real
        # heteroscedastic noise (it genuinely lowers NLL). The 1e-2 default is
        # calibrated on a 48-fit sweep: essentially-noiseless fits sit at
        # 1−R² ≤ 6e-4 (SNR ≳ 1500), while genuine heteroscedastic data — down to
        # weak β=1 — sits at 1−R² ≥ 0.095, a ~160× gap; σ²-shape does NOT separate
        # them (ranges fully overlap). Per-fit overridable: raise it to keep
        # Stage D for genuinely heteroscedastic data at SNR 100–1000. See
        # (internal Stage-D joint-GLS design note / advisor decision memo).
        min_noise_ratio = cfg.get('min_noise_ratio', 1e-2)
        # Model-selection margin: keep the heteroscedastic model only if it
        # improves held-out NLL by at least this *relative* amount vs a constant
        # variance. Held-out NLL already penalizes overfitting; the margin is a
        # small, sign/scale-robust tie-breaker that biases toward the simpler
        # homoscedastic model.
        var_select_margin = cfg.get('variance_selection_margin', 2e-3)
        # Mean-consistent variance selection (X4B). When True, the keep/revert
        # NLL comparison uses the SAME Stage-D (GLS-weighted) mean on both sides,
        # so the decision isolates "does input-dependent variance help *given the
        # mean*?" instead of confounding it with the GLS-vs-unweighted mean gap
        # (the false-revert documented in an internal Stage-D guard design note).
        # Default True (advisor-approved flip): the guard selects on the variance
        # given the same mean, correct now that the joint-GLS mean is efficient.
        var_select_mean_consistent = cfg.get(
            'variance_selection_mean_consistent', True)
        # Mean-fallback outcome (X4B Step 2). When True, if the variance model
        # genuinely helps given a fixed mean but the GLS-weighted mean degraded
        # the overall package below the constant-variance revert, the fit is
        # returned as the UNIT-WEIGHT (Hoeffding-projection) mean + the
        # input-dependent variance — the dominant cell of the mean×variance grid
        # (Theorem 2, "attribution vs efficiency": attribute/predict from the
        # unit-weight fit when the precision-weighted mean is not yet efficient).
        # Default False preserves the binary keep/revert behavior.
        var_select_mean_fallback = cfg.get(
            'variance_selection_mean_fallback', False)
        # Joint-GLS Stage-D mean (Option B, root fix). When True, the alternating
        # mean update profiles the intercept jointly by weighted-centering BOTH y
        # and Φ (their precision-weighted column means) before the weighted ridge,
        # rather than fixing f0 = Σwy/Σw and solving on uncentered Φ. Fourier
        # features have ~0 UNWEIGHTED mean but nonzero WEIGHTED mean under
        # 1/σ²(x) weights, so the legacy uncentered solve is NOT the penalized-GLS
        # optimum and yields a weighted mean that loses to the unit-weight mean on
        # its own weighted objective (internal Stage-D GLS mean-fit design note).
        # Default True (advisor-approved flip): the root fix — yields the
        # penalized-GLS-optimal (efficient) weighted mean so the guard keeps
        # correct variance models. Changes σ̂/df/LOO/CI of every het fit (golden
        # re-baselined at the flip; tag stage-d-pre-flip is the rollback target).
        # NB (P0-2): the shipped default estimator is a *leverage-adjusted
        # alternating quasi-likelihood with validation iterate selection*, NOT
        # monotone block-coordinate descent on eq:model — an efficient weighted
        # mean does not make the alternation monotone once the variance step is a
        # leverage-corrected quasi-likelihood update and the iterate is chosen by
        # held-out NLL. Only stage_d_estimator='raw_likelihood' is block-
        # coordinate descent, and only while the log-variance solution is
        # interior (see _resolve_stage_d_estimator and results['stage_D']).
        joint_gls_mean = cfg.get('stage_d_joint_gls_mean', True)
        # Effective mean-estimator convention (DEC-039 provenance) of the Stage-D
        # WEIGHTED mean, recorded on the results/fitted-design record so a
        # downstream consumer or saved artifact can tell which estimator vintage
        # produced the mean. Reflects the resolved flag, not merely the request.
        # A fit that reverts to constant variance or falls back to the unit-weight
        # mean ships an unweighted-centered mean instead (set at those branches).
        weighted_mean_mode = (MEAN_INTERCEPT_PROFILED_JOINT_GLS if joint_gls_mean
                              else MEAN_INTERCEPT_LEGACY_FIXED)
        # Leverage-corrected variance solve (DEC-028). In-sample squared
        # residuals are biased low under the fitted mean — E[r_n²] ≈
        # σ_n²(1 − lev_n) — exactly where a rich mean basis fits tightly. Fed
        # raw r² the variance model underestimates σ² there, the 1/σ² weights
        # blow up, the weighted mean interpolates the low-σ region even harder,
        # and the alternating loop spirals (the failure DEC-027's guards catch
        # after the fact). Feeding the Newton solve r²/clip(1−lev, 1e-3, 1)
        # removes the feedback at its source; joint_lambda._joint_fit has
        # always used the same correction.
        #
        # Held-out trajectory selection (DEC-028): score every outer iterate on
        # validation NLL and keep the best one, rather than trusting the
        # train-NLL convergence point. Costs one validation prediction per
        # outer iteration; the val features are needed for the final
        # evaluation anyway.
        #
        # Both flags are resolved through the estimator selector (P0-2): the
        # DEFAULT (adjusted_quasi_likelihood) keeps both on — the validated
        # leverage-adjusted alternating quasi-likelihood with validation iterate
        # selection — while stage_d_estimator='raw_likelihood' turns both off for
        # a monotone block-coordinate-descent raw fit. ``est_identity`` carries
        # the honest machine-readable estimator identity onto results['stage_D'].
        lev_correct, early_stop, est_identity = _resolve_stage_d_estimator(cfg)
        max_outer = cfg.get('max_outer_iter', 10)
        if max_outer < 1:
            raise ValueError(
                f"max_outer_iter must be >= 1 for heteroscedastic fitting "
                f"(got {max_outer}); the alternating mean/variance loop needs "
                f"at least one iteration to produce mean coefficients."
            )
        tol = cfg.get('alternating_tol', 1e-4)
        newton_max = cfg.get('newton_max_iter', 10)
        K3 = model.K3
        basis_name = getattr(model, 'basis_name', 'fourier')

        from ...core.features import basis_size as _bs
        designs = self._build_stage_d_designs(
            model, x_train, y_train, pair_mgr, K1, K2, K3, Kh, D,
            strategy, lambda1, lambda2, lambda_h, basis_name)
        (phi1_train, phi2_train, phi3_train, phi_all_train, psi1_train,
         psi2_train, psi3_train, K2h, K3h, var_pair_mgr, var_triple_mgr,
         z_h_proj_train, z_h_proj_coeffs, var_residual_model, M_h,
         _il1, _il2, _il3, _ilh1, _ilh2, _ilh3, _incl_lin_2, _incl_lin_3,
         F1, F2, Fh, Ph, Th, T, reg_mean, reg_var,
         pair_block_info, fo_included, variance_variables) = designs
        _pair_k2_t = (tuple(int(k) for k in self._pair_k2)
                      if getattr(self, '_pair_k2', None) is not None else None)
        _block1 = _bs(K1, _il1, basis_name)

        # --- Residual prediction from Stage C (frozen during Stage D) ---
        residual_pred_train = jnp.zeros(x_train.shape[0])
        if model.residual_net is not None:
            res_out = predict_residual_batch(model.residual_net, x_train)
            residual_pred_train = res_out
            self._log(f"  Including residual prediction (var={float(jnp.var(res_out)):.6f})")

        # --- Initialize ---
        f0 = float(model.mean_model.f0)
        y_centered = y_train - f0

        # Initial full mean prediction: Fourier + residual
        fourier_pred = _mean_predict_on_designs(
            model.mean_model, phi1_train, phi2_train, phi3_train,
            fo_included, _block1)
        full_mean_pred = fourier_pred + residual_pred_train
        residuals = y_train - full_mean_pred
        r2 = residuals ** 2

        h0_init = float(jnp.log(jnp.mean(r2)))

        # ---- Stage-D pre-flight: near-noiseless data ----
        # If the mean already explains essentially everything, the residuals are
        # ~0 and log(r²) is numerically degenerate — the alternating loop below
        # can produce nan/inf. Skip it and use a constant variance. (The more
        # general "is the variance model actually justified?" question is
        # answered *after* fitting by the homoscedastic-vs-heteroscedastic NLL
        # comparison below — that is the authoritative gate.)
        if guard:
            _y = np.asarray(y_train, dtype=np.float64)
            _sse = float(np.sum(np.asarray(residuals, dtype=np.float64) ** 2))
            _tss = float(np.sum((_y - _y.mean()) ** 2))
            noise_ratio = _sse / _tss if _tss > 0 else 0.0
            if _tss <= 0 or noise_ratio < min_noise_ratio:
                warnings.warn(
                    f"heteroscedastic=True (Stage D) was requested, but the noise-"
                    f"to-signal ratio is {noise_ratio:.2e} (< min_noise_ratio="
                    f"{min_noise_ratio:.0e}; the mean explains ~all variation, "
                    f"train R²≈{1.0 - noise_ratio:.6f}, SNR≈{(1.0/noise_ratio - 1.0):.0f}). "
                    f"On essentially-noiseless data the residual is deterministic "
                    f"mean-approximation error, not aleatoric noise, so no variance "
                    f"model is warranted. Falling back to a constant variance and "
                    f"keeping the mean fit. If this IS genuinely heteroscedastic "
                    f"high-SNR data, lower min_noise_ratio (it is per-fit "
                    f"overridable; the 1e-2 default is calibrated to skip only "
                    f"near-noiseless fits). Or use heteroscedastic=False to skip "
                    f"Stage D, or heteroscedastic_guard=False to force the raw fit.",
                    stacklevel=2,
                )
                results['stage_D'] = {
                    'skipped': True, 'reason': 'near-noiseless',
                    'noise_ratio': noise_ratio, 'fallback': 'constant_variance',
                    # Shipped mean is the pre-D unit-weight (centered) mean.
                    'mean_intercept_mode': MEAN_INTERCEPT_UNWEIGHTED,
                    # Estimator identity of the *configured* Stage-D estimator;
                    # the outcome is a constant variance (no alternating solve
                    # ran), so convergence_reason records the skip and no
                    # log-variance bound can be active.
                    **est_identity,
                    'convergence_reason': 'near_noiseless_skip',
                    'bound_active': False,
                }
                return self._homoscedastic_fallback(
                    model, stage_b_mean_model, h0_init,
                    pair_mgr, K1, K2, model.K3, Kh, D, basis_name)

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

        # --- Validation features (built once, before the loop): used by the
        # held-out early stop each outer iteration and by the final
        # evaluation/model-selection below ---
        if fo_included is not None:
            from ...core.features import build_first_order_features_subset
            phi1_val = build_first_order_features_subset(
                x_val, K1, fo_included, include_linear=_il1,
                basis_name=basis_name)
        else:
            phi1_val = build_first_order_features(x_val, K1, include_linear=_il1, basis_name=basis_name)
        if _pair_k2_t is not None:
            from ...core.features import build_second_order_features_per_pair
            phi2_val, _ = build_second_order_features_per_pair(
                x_val, _pair_k2_t, pair_mgr.pair_indices,
                include_linear=_il2, basis_name=basis_name)
        else:
            phi2_val = (build_second_order_features(x_val, K2, pair_mgr.pair_indices, include_linear=_il2, basis_name=basis_name)
                        if K2 > 0 else None)
        phi3_val = (build_third_order_features(x_val, K3, model.triple_indices, include_linear=_il3, basis_name=basis_name)
                    if K3 > 0 and model.triple_indices is not None else None)
        phi_all_val = phi1_val
        if phi2_val is not None:
            phi_all_val = jnp.concatenate([phi_all_val, phi2_val], axis=1)
        if phi3_val is not None:
            phi_all_val = jnp.concatenate([phi_all_val, phi3_val], axis=1)
        phi_all_val64 = jnp.asarray(phi_all_val, dtype=jnp.float64)

        if variance_variables is not None:
            from ...core.features import build_first_order_features_subset
            psi1_val = build_first_order_features_subset(
                x_val, Kh, list(variance_variables), include_linear=_ilh1,
                basis_name=basis_name)
        else:
            psi1_val = build_first_order_features(x_val, Kh, include_linear=_ilh1, basis_name=basis_name)
        psi2_val = None
        if K2h > 0 and var_pair_mgr is not None and var_pair_mgr.P > 0:
            psi2_val = build_second_order_features(x_val, K2h, var_pair_mgr.pair_indices,
                                                    include_linear=_ilh2, basis_name=basis_name)
        psi3_val_h = None
        if K3h > 0 and var_triple_mgr is not None and var_triple_mgr.T > 0:
            psi3_val_h = build_third_order_features(x_val, K3h, var_triple_mgr.triple_indices,
                                                     include_linear=_ilh3, basis_name=basis_name)
        z_h_proj_val = None
        if var_residual_model is not None:
            z_h_val = var_residual_model.build_features(x_val)
            psi_fourier_val = psi1_val
            if psi2_val is not None:
                psi_fourier_val = jnp.concatenate([psi_fourier_val, psi2_val], axis=1)
            if psi3_val_h is not None:
                psi_fourier_val = jnp.concatenate([psi_fourier_val, psi3_val_h], axis=1)
            z_h_proj_val = z_h_val - psi_fourier_val @ jnp.asarray(z_h_proj_coeffs)
        psi_val_parts = [jnp.asarray(psi1_val, dtype=jnp.float64)]
        if psi2_val is not None:
            psi_val_parts.append(jnp.asarray(psi2_val, dtype=jnp.float64))
        if psi3_val_h is not None:
            psi_val_parts.append(jnp.asarray(psi3_val_h, dtype=jnp.float64))
        if z_h_proj_val is not None:
            psi_val_parts.append(jnp.asarray(z_h_proj_val, dtype=jnp.float64))
        psi_all_val = (jnp.concatenate(psi_val_parts, axis=1)
                       if len(psi_val_parts) > 1 else psi_val_parts[0])

        res_out_val = None
        residual_pred_val = jnp.zeros(x_val.shape[0], dtype=jnp.float64)
        if model.residual_net is not None:
            res_out_val = predict_residual_batch(model.residual_net, x_val)
            residual_pred_val = jnp.asarray(res_out_val, dtype=jnp.float64)
        y_val64 = jnp.asarray(y_val, dtype=jnp.float64)

        # Leverage-correction state: the residuals entering the first variance
        # solve come from the unweighted Stage-A/B mean fit, so leverage starts
        # at unit weights; afterwards it tracks the weights of the mean fit
        # that produced the current residuals. The intercept column is included
        # (unpenalized) so lev sums to the full mean df.
        #
        # The constant-variance BASELINE gets the same correction: the raw
        # in-sample mean squared residual exp(h0_init) is deflated by mean
        # overfitting (E[r²] ≈ σ²(1−lev)), which would make the homoscedastic
        # side of the model-selection comparison artificially easy to beat.
        log_s2_const = h0_init
        if lev_correct:
            _N_tr = x_train.shape[0]
            _phi_aug_np = np.concatenate(
                [np.ones((_N_tr, 1)),
                 np.asarray(phi_all_train, dtype=np.float64)], axis=1)
            _reg_aug_np = np.concatenate(
                [np.zeros(1), np.asarray(reg_mean, dtype=np.float64)])
            _w_mean_fit = np.ones(_N_tr)
            _lev0 = leverage_diag(_phi_aug_np, _reg_aug_np, _w_mean_fit)
            log_s2_const = float(np.log(np.mean(
                debias_squared_residuals(r2, _lev0))))

        # Held-out trajectory state (early stop)
        best_nll_val = np.inf
        best_iterate = None
        best_outer = -1
        val_nll_traj = []

        prev_loss = float('inf')
        # How the alternating loop terminated (estimator metadata,
        # convergence_reason): the relative train-NLL tolerance, or exhausting
        # max_outer without meeting it. Independent of iterate_selection — the
        # SHIPPED iterate may still be an earlier validation-best one (early stop).
        converged_by_tol = False

        # Initialize the log-variance intercept so `h0` is always bound before
        # the outer loop's `h0_init if outer == 0 else h0` could read it. Under
        # valid configs (max_outer >= 1) outer==0 uses h0_init and this is a
        # no-op; it removes the read-before-assignment that static analysis flags.
        h0 = h0_init

        for outer in range(max_outer):
            # Cooperative cancellation + fine-grained progress for the longest
            # stage (no-ops without the GUI hooks).
            self._check_cancel()
            self._emit('stage_progress', stage='D',
                       metrics={'outer': outer, 'max_outer': max_outer})
            # --- Variance update (Newton on augmented features) ---
            # Leverage correction: de-bias the in-sample squared residuals,
            # E[r_n²] ≈ σ_n²(1 − lev_n), before the log-variance solve (the
            # same correction as joint_lambda._joint_fit).
            r2_solve = r2
            if lev_correct:
                lev = (_lev0 if outer == 0 else
                       leverage_diag(_phi_aug_np, _reg_aug_np, _w_mean_fit))
                r2_solve = jnp.asarray(debias_squared_residuals(r2, lev))
            w_h, h0 = newton_solve_log_variance(
                psi_all_train, r2_solve, w_h, h0_init if outer == 0 else h0,
                reg_var, max_iter=newton_max
            )

            # Compute weights for weighted ridge
            h_pred = h0 + jnp.float64(psi_all_train) @ w_h
            sigma2 = jnp.exp(h_pred)
            weights = 1.0 / sigma2
            if lev_correct:
                # weights of the mean fit below == weights that produce the
                # residuals the *next* variance solve will see
                _w_mean_fit = np.asarray(weights, dtype=np.float64)

            # --- Mean update (weighted ridge on y - residual_pred) ---
            # The residual_net prediction is frozen; Fourier coefficients adapt
            y_for_fourier = y_train - residual_pred_train

            # Recompute weighted intercept: f0 = Σ w_n y_n / Σ w_n
            w_sum = jnp.sum(weights)
            if joint_gls_mean:
                # Joint-GLS (Option B): weighted-center BOTH y and Φ, so the
                # intercept is profiled jointly (the penalized-GLS optimum) and
                # the ridge penalty applies only to the non-intercept coeffs.
                # f0 is reconstructed so prediction f0 + Φ·w equals
                # ȳ_w + (Φ − Φ̄_w)·w on train.
                phi64 = jnp.asarray(phi_all_train, dtype=jnp.float64)
                phi_wmean = jnp.sum(weights[:, None] * phi64, axis=0) / w_sum
                ybar_w = jnp.sum(weights * y_for_fourier) / w_sum
                w_all = weighted_ridge_solve(
                    phi64 - phi_wmean, y_for_fourier - ybar_w, reg_mean, weights)
                f0 = float(ybar_w - phi_wmean @ jnp.asarray(w_all, dtype=jnp.float64))
            else:
                # Legacy fixed-intercept / uncentered-Φ solve (only when
                # stage_d_joint_gls_mean=False; the default is the profiled
                # joint-GLS branch above).
                f0 = float(jnp.sum(weights * y_for_fourier) / w_sum)
                y_centered = y_for_fourier - f0
                w_all = weighted_ridge_solve(phi_all_train, y_centered, reg_mean, weights)

            # Update predictions and residuals (include residual_net)
            fourier_pred = f0 + phi_all_train @ self._dtype(w_all)
            full_mean_pred = fourier_pred + residual_pred_train
            residuals = y_train - full_mean_pred
            r2 = residuals ** 2

            # Compute loss (NLL)
            loss = float(jnp.mean(0.5 * h_pred + 0.5 * jnp.float64(r2) / sigma2))

            # Held-out scoring of this iterate (early stop): validation NLL of
            # the current (mean, variance) pair. The best iterate is restored
            # after the loop.
            if early_stop:
                mean_val_it = (f0 + phi_all_val64 @ w_all) + residual_pred_val
                h_val_it = jnp.clip(h0 + psi_all_val @ w_h,
                                    -LOG_VAR_CLIP, LOG_VAR_CLIP)
                nll_val_it = float(jnp.mean(
                    0.5 * h_val_it
                    + 0.5 * (y_val64 - mean_val_it) ** 2 / jnp.exp(h_val_it)))
                val_nll_traj.append(nll_val_it)
                if np.isfinite(nll_val_it) and nll_val_it < best_nll_val:
                    best_nll_val = nll_val_it
                    best_iterate = (w_all, float(f0), w_h, float(h0))
                    best_outer = outer

            # Check convergence
            if abs(prev_loss - loss) / (abs(prev_loss) + 1e-10) < tol:
                self._log(f"  Converged at outer iteration {outer + 1}")
                converged_by_tol = True
                break
            prev_loss = loss

        n_outer_run = outer + 1
        convergence_reason = ('train_nll_tolerance' if converged_by_tol
                              else 'max_outer_iter')
        if early_stop and best_iterate is not None and best_outer != outer:
            self._log(f"  Early stop: keeping outer iteration {best_outer + 1} "
                  f"(val NLL {best_nll_val:.4f} vs {val_nll_traj[-1]:.4f} at "
                  f"iteration {n_outer_run})")
            w_all, f0, w_h, h0 = best_iterate

        # Bound-activity detection (P1-4 / P0-2, DETECT+label only — no Newton
        # redesign): whether the SHIPPED (possibly early-stop-restored) train
        # log-variance solution touches the ±LOG_VAR_CLIP box the variance solve
        # clamps to. When active, the reported log-variance is a clipped value,
        # not an interior stationary point, so no exact gradient/stationarity
        # (and no monotone block-descent) claim holds there.
        _h_train_final = float(h0) + (np.asarray(psi_all_train, dtype=np.float64)
                                      @ np.asarray(w_h, dtype=np.float64))
        bound_active = _log_variance_bound_active(_h_train_final)

        # --- Build final model with variance ---
        w1_final, w2_final, w3_final = _split_mean_coeffs(
            w_all, F1, F2, has_third=(K3 > 0 and T > 0), dtype=self._dtype)

        if fo_included is not None:
            from .._trainer_helpers import _scatter_first_order
            w1_model_final = _scatter_first_order(
                w1_final, D, _block1, fo_included, dtype=self._dtype)
        else:
            w1_model_final = jnp.array(w1_final, dtype=self._dtype)
        mean_model = MeanModel(
            f0=jnp.array(f0, dtype=self._dtype),
            w1=w1_model_final,
            w2=jnp.array(w2_final, dtype=self._dtype),
            w3=jnp.array(w3_final, dtype=self._dtype),
            K1=K1, K2=K2, K3=K3, D=D,
            include_linear_1=_il1,
            include_linear_2=_incl_lin_2, include_linear_3=_incl_lin_3,
            basis_name=basis_name,
            pair_block_info=pair_block_info,
            pair_k2=_pair_k2_t,
        )

        # Split augmented w_h into [first-order | second-order | third-order | residual]
        Fh2 = Ph * _bs(K2h, _ilh2, basis_name) ** 2 if K2h > 0 else 0
        Fh3 = Th * _bs(K3h, _ilh3, basis_name) ** 3 if K3h > 0 else 0
        w_h_fourier1 = w_h[:Fh]
        w_h_fourier2 = (w_h[Fh:Fh + Fh2] if Fh2 > 0
                        else jnp.array([], dtype=self._dtype))
        w_h_fourier3 = (w_h[Fh + Fh2:Fh + Fh2 + Fh3] if Fh3 > 0
                        else jnp.array([], dtype=self._dtype))
        w_h_residual = w_h[Fh + Fh2 + Fh3:] if M_h > 0 else None

        # Build fitted variance residual model (if present)
        fitted_var_residual = None
        if var_residual_model is not None and w_h_residual is not None:
            import equinox as eqx
            fitted_var_residual = eqx.tree_at(
                lambda m: m.weights, var_residual_model,
                jnp.array(w_h_residual, dtype=self._dtype))
            fitted_var_residual = eqx.tree_at(
                lambda m: m.proj_coeffs, fitted_var_residual,
                jnp.array(z_h_proj_coeffs, dtype=jnp.float64))

        variance_model = VarianceModel(
            h0=jnp.array(h0, dtype=self._dtype),
            w1=jnp.array(w_h_fourier1, dtype=self._dtype),
            Kh=Kh, D=D,
            w2=jnp.array(w_h_fourier2, dtype=self._dtype),
            K2h=K2h,
            pair_indices_h=(np.array(var_pair_mgr.pair_indices)
                            if var_pair_mgr is not None and Ph > 0 else None),
            w3=jnp.array(w_h_fourier3, dtype=self._dtype),
            K3h=K3h,
            triple_indices_h=(np.array(var_triple_mgr.triple_indices)
                              if var_triple_mgr is not None and Th > 0 else None),
            w_var_residual=(jnp.array(w_h_residual, dtype=self._dtype)
                            if w_h_residual is not None else None),
            variance_residual=fitted_var_residual,
            basis_name=basis_name,
            include_linear_h1=_ilh1,
            include_linear_h2=_ilh2,
            include_linear_h3=_ilh3,
            variance_variables=(tuple(variance_variables)
                                if variance_variables is not None else None),
        )

        model = HiFiANOVA(
            mean_model=mean_model,
            variance_model=variance_model,
            residual_net=model.residual_net,
            K1=K1, K2=K2, K3=K3, Kh=Kh, D=D,
            pair_indices=np.array(pair_mgr.pair_indices),
            triple_indices=model.triple_indices,
            include_linear_1=getattr(model, 'include_linear_1', True),
            include_linear_2=getattr(model, 'include_linear_2', True),
            include_linear_3=getattr(model, 'include_linear_3', True),
            include_linear_h1=getattr(model, 'include_linear_h1', True),
            include_linear_h2=getattr(model, 'include_linear_h2', True),
            include_linear_h3=getattr(model, 'include_linear_h3', True),
            basis_name=getattr(model, 'basis_name', 'fourier'),
            pair_block_info=pair_block_info,
            pair_k2=_pair_k2_t,
            fo_included=(tuple(fo_included) if fo_included is not None
                         else None),
        )

        # --- Evaluate on validation (features prebuilt above the loop) ---
        mean_val = _mean_predict_on_designs(
            mean_model, phi1_val, phi2_val, phi3_val, fo_included, _block1)
        if model.residual_net is not None:
            mean_val = mean_val + res_out_val
        rmse_val = float(jnp.sqrt(jnp.mean((y_val - mean_val) ** 2)))

        # NLL on validation (include all variance components)
        h_val = variance_model.predict_log_variance(psi1_val, psi2_val, psi3_val_h, z_h_proj_val)
        sigma2_val = jnp.exp(h_val)
        nll_val = float(jnp.mean(0.5 * h_val + 0.5 * (y_val - mean_val) ** 2 / sigma2_val))

        # Unit-weight (Hoeffding-projection) mean companion — the interpretable
        # estimand of Theorem 2. Computed once here so it is a single source of
        # truth for: (a) the mean-consistent Stage-D diagnostics below, (b) the
        # mean-fallback prediction path, and (c) the Sobol attribution companion
        # in the fitted-design record. It is the SAME penalized mean design
        # re-solved with W = I on the same target y_for_fourier.
        f0_unit = float(np.mean(np.asarray(y_for_fourier)))
        w_unit = weighted_ridge_solve(
            phi_all_train, jnp.asarray(y_for_fourier) - f0_unit, reg_mean)
        mean_val_unit = np.asarray(
            f0_unit + phi_all_val64 @ self._dtype(w_unit), dtype=np.float64)
        if model.residual_net is not None:
            mean_val_unit = mean_val_unit + np.asarray(res_out_val, dtype=np.float64)

        results['stage_D'] = {
            'rmse_val': rmse_val, 'nll_val': nll_val,
            'has_variance_second_order': K2h > 0 and Ph > 0,
            'n_variance_pairs': Ph,
            'has_variance_residual': M_h > 0,
            'M_variance_residual': M_h,
            'leverage_correction': bool(lev_correct),
            'alternating_early_stop': bool(early_stop),
            'n_outer_iterations': n_outer_run,
            'best_outer_iteration': (best_outer + 1) if best_iterate is not None else None,
            'val_nll_trajectory': [float(v) for v in val_nll_traj] or None,
            # Estimator identity + objective metadata (P0-2). ``est_identity``
            # names the effective estimator honestly (leverage-adjusted
            # alternating quasi-likelihood with validation iterate selection by
            # default; raw_likelihood / custom otherwise); convergence_reason and
            # bound_active describe the variance solve's termination and whether
            # its log-variance solution touched the clip box.
            **est_identity,
            'convergence_reason': convergence_reason,
            'bound_active': bound_active,
        }
        parts_str = []
        if K2h > 0 and Ph > 0:
            parts_str.append(f"{Ph} var pairs")
        if M_h > 0:
            parts_str.append(f"{M_h} var residual features")
        suffix = f" ({', '.join(parts_str)})" if parts_str else ""
        self._log(f"  RMSE val: {rmse_val:.4f}, NLL val: {nll_val:.4f}{suffix}")

        # ---- Stage-D model selection: did heteroscedasticity actually help? ----
        # The variance model is only justified if it beats the homoscedastic
        # (constant-variance) baseline on held-out data. Always fit that baseline
        # and compare validation NLL on one scale: the heteroscedastic model is
        # kept only if it lowers the held-out NLL. This makes constant variance
        # the *default outcome* — the data must earn the input-dependent variance
        # — and it is robust to the failure modes a pre-fit heteroscedasticity
        # test can't see: homoscedastic noise dressed up by mean-misspecification
        # (no NLL gain → rejected), and an alternating loop that blew the mean up
        # (worse NLL → rejected). It subsumes the old RMSE-ratio heuristic.
        ship_unit_mean = False
        if guard:
            # Homoscedastic baseline: the pre-Stage-D (mean-only) mean with a
            # single constant variance σ²_c = mean of its squared residuals —
            # leverage-corrected when leverage_correction is on, so the two
            # sides of the comparison estimate their variance the same way
            # (the raw in-sample level is deflated by mean overfitting).
            mean_val_preD = _mean_predict_on_designs(
                stage_b_mean_model, phi1_val, phi2_val, phi3_val,
                fo_included, _block1)
            if model.residual_net is not None:
                mean_val_preD = mean_val_preD + res_out_val
            sigma2_c = float(np.exp(log_s2_const))
            resid_val_preD = np.asarray(y_val, dtype=np.float64) - np.asarray(mean_val_preD)
            nll_homo = _gaussian_nll(resid_val_preD, sigma2_c) if sigma2_c > 0 else np.inf

            # --- Mean-consistent diagnostic grid (X4B) --------------------------
            # The keep/revert decision is really TWO questions the framework
            # separates (Theorem 2): does input-dependent variance help *given a
            # fixed mean*, and is the GLS-weighted mean better than the unit-weight
            # (Hoeffding-projection) mean. The legacy package comparison conflates
            # them, so a mildly-worse weighted mean can sink a correct variance
            # model. We always compute the 2×2 held-out NLL grid over
            # {unit mean, weighted mean} × {constant σ², input-dependent σ²(x)}
            # for transparency, and store the two gaps the manuscript reports as a
            # "heteroscedasticity × misspecification" diagnostic. Purely
            # informational — the numbers below do not change the default outcome.
            y_val_np = np.asarray(y_val, dtype=np.float64)
            h_val_np = np.asarray(h_val, dtype=np.float64)
            sigma2_val_np = np.asarray(sigma2_val, dtype=np.float64)

            def _nll_var(resid):  # NLL of σ²(x) for a given mean's residuals
                r = np.asarray(resid, dtype=np.float64)
                return float(np.mean(0.5 * h_val_np + 0.5 * r ** 2 / sigma2_val_np))

            resid_val_D = y_val_np - np.asarray(mean_val)          # weighted mean
            resid_val_unit = y_val_np - mean_val_unit             # unit mean
            nll_wmean_const = _gaussian_nll(resid_val_D, sigma2_c) if sigma2_c > 0 else np.inf
            nll_wmean_var = nll_val                               # == weighted+σ²(x)
            nll_unit_const = _gaussian_nll(resid_val_unit, sigma2_c) if sigma2_c > 0 else np.inf
            nll_unit_var = _nll_var(resid_val_unit)
            results['stage_D']['nll_grid'] = {
                'unit_mean_const_var': float(nll_unit_const),
                'unit_mean_input_var': float(nll_unit_var),
                'weighted_mean_const_var': float(nll_wmean_const),
                'weighted_mean_input_var': (
                    float(nll_wmean_var) if np.isfinite(nll_wmean_var) else None),
            }
            # variance_gain_given_mean > 0: σ²(x) beats a constant given the unit
            # mean. mean_gap_weighted_vs_unit > 0: the GLS mean is WORSE than the
            # unit mean at equal (input-dependent) variance — the efficiency gap
            # Theorem 2(ii) says should be ≤ 0 for a correct weighted solve.
            results['stage_D']['variance_gain_given_mean'] = float(
                nll_unit_const - nll_unit_var)
            results['stage_D']['mean_gap_weighted_vs_unit'] = (
                float(nll_wmean_var - nll_unit_var)
                if np.isfinite(nll_wmean_var) else None)

            pre_d = (results.get('stage_C') or results.get('stage_B')
                     or results.get('stage_A') or {})
            pre_d_rmse = pre_d.get('rmse_val') if isinstance(pre_d, dict) else None

            mean_blew_up = (not np.isfinite(rmse_val)) or (
                pre_d_rmse and np.isfinite(pre_d_rmse) and pre_d_rmse > 0
                and rmse_val > 2.0 * pre_d_rmse)
            # Relative held-out-NLL improvement, symmetric and sign-robust so the
            # decision is scale-free and never divides by a near-zero NLL:
            #   rel_improve = (nll_base − nll_het) / (|nll_base| + |nll_het|)
            # Keep heteroscedastic only if it improves by more than the margin; a
            # near-tie (rel_improve ≤ margin) keeps the simpler constant variance.
            # The baseline mean is the unit/weighted mean per the flag: mean-
            # consistent (X4B, the DEC-039 default) uses the SAME weighted mean the
            # het model uses, so the decision isolates the variance; the legacy
            # (non-default) comparison uses the pre-D unweighted mean and thus also
            # folds in the GLS-vs-unweighted gap.
            base_nll = nll_wmean_const if var_select_mean_consistent else nll_homo
            denom = abs(base_nll) + abs(nll_val) + 1e-12
            rel_improve = (base_nll - nll_val) / denom
            results['stage_D']['nll_rel_improvement'] = float(rel_improve)
            nll_not_improved = (not np.isfinite(nll_val)) or (
                rel_improve <= var_select_margin)

            results['stage_D']['nll_homoscedastic'] = base_nll
            results['stage_D']['nll_heteroscedastic'] = (
                nll_val if np.isfinite(nll_val) else None)

            # Mean-fallback eligibility (X4B Step 2): σ²(x) genuinely helps given
            # the unit-weight mean we would fall back to. Justified by the
            # unit-mean cell alone, so it is a self-contained "keep the variance,
            # drop the bad weighted mean" decision.
            denom_u = abs(nll_unit_const) + abs(nll_unit_var) + 1e-12
            unit_var_improve = ((nll_unit_const - nll_unit_var) / denom_u
                                if np.isfinite(nll_unit_var) else -np.inf)
            mean_fallback_ok = (var_select_mean_fallback and (not mean_blew_up)
                                and unit_var_improve > var_select_margin)

            if mean_blew_up or nll_not_improved:
                if mean_fallback_ok:
                    # Keep the (correct) variance model but predict from the
                    # unit-weight Hoeffding-projection mean — the dominant cell.
                    ship_unit_mean = True
                    results['stage_D']['reverted'] = False
                    results['stage_D']['selected'] = 'mean_fallback'
                    # Ships the UNIT-WEIGHT (Hoeffding-projection) mean, not the
                    # GLS-weighted one — record the shipped convention.
                    results['stage_D']['mean_intercept_mode'] = (
                        MEAN_INTERCEPT_UNWEIGHTED)
                    self._log(
                        "  Stage D: input-dependent variance improves held-out "
                        f"NLL given the mean (Δ={nll_unit_const - nll_unit_var:+.4g}), "
                        "but the GLS-weighted mean degraded the package "
                        f"(weighted+σ²(x) {nll_val:.4g} vs unit+σ²(x) "
                        f"{nll_unit_var:.4g}); keeping the variance model with the "
                        "unit-weight mean (mean-fallback).")
                    # Invariant monitor: with a correct joint-GLS mean the weighted
                    # mean cannot lose to the unit-weight mean up to noise, so a
                    # mean-fallback here signals the root fix is not holding
                    # (convergence/conditioning). Surface it as an anomaly rather
                    # than a silent route-around when joint-GLS is active.
                    results['stage_D']['mean_fallback_anomaly'] = bool(joint_gls_mean)
                    if joint_gls_mean:
                        warnings.warn(
                            "Stage D: mean-fallback fired with "
                            "stage_d_joint_gls_mean=True — the GLS-weighted mean lost "
                            "to the unit-weight mean on held-out NLL "
                            f"(weighted+σ²(x) {nll_val:.4g} vs unit+σ²(x) "
                            f"{nll_unit_var:.4g}). Under a correct joint-GLS solve "
                            "this should not occur up to noise; investigate the "
                            "alternating-loop convergence/conditioning for this fit.",
                            stacklevel=2)
                else:
                    if mean_blew_up:
                        reason = (
                            "the alternating mean/variance loop degraded the mean "
                            f"(validation RMSE {pre_d_rmse} → "
                            f"{'nan/inf' if not np.isfinite(rmse_val) else f'{rmse_val:.3g}'})")
                        hint = (" — usually too rich a mean basis (K1/K2) under "
                                "strategy='variance'; try strategy='curvature', a "
                                "larger lambda_h, or a smaller K1/K2")
                    elif np.isfinite(nll_val) and unit_var_improve > var_select_margin:
                        # Variance is genuinely helpful; the package lost only
                        # because the weighted mean is worse than the unit mean.
                        reason = (
                            "input-dependent variance improves held-out NLL given "
                            f"the mean (Δ={nll_unit_const - nll_unit_var:+.4g}), but "
                            "the GLS-weighted Stage-D mean degraded the overall fit "
                            f"(weighted+σ²(x) {nll_val:.4g} vs constant-variance "
                            f"baseline {base_nll:.4g})")
                        hint = (" — pass variance_selection_mean_fallback=True to "
                                "keep the variance model with the unit-weight mean, "
                                "or variance_selection_mean_consistent=True to select "
                                "on the variance alone")
                    else:
                        reason = (
                            f"the heteroscedastic model did not beat a constant "
                            f"variance on held-out data (validation NLL "
                            f"{nll_val:.4g} vs homoscedastic {base_nll:.4g})"
                            if np.isfinite(nll_val) else
                            "the heteroscedastic validation NLL was non-finite")
                        hint = " — the noise looks homogeneous, so no variance model is warranted"
                    warnings.warn(
                        "Stage D (heteroscedastic variance): " + reason + hint
                        + ". Reverting to the mean-only fit with a constant variance. "
                        "Pass heteroscedastic_guard=False to keep the raw variance fit.",
                        stacklevel=2,
                    )
                    results['stage_D']['reverted'] = True
                    results['stage_D']['selected'] = 'homoscedastic'
                    # Reverts to the pre-D unit-weight (centered) mean.
                    results['stage_D']['mean_intercept_mode'] = (
                        MEAN_INTERCEPT_UNWEIGHTED)
                    # The shipped variance is constant (the alternating fit was
                    # rejected), so record the revert outcome and drop any
                    # per-point bound activity. The est_identity above still names
                    # the estimator that was configured/attempted.
                    results['stage_D']['convergence_reason'] = 'reverted_homoscedastic'
                    results['stage_D']['bound_active'] = False
                    return self._homoscedastic_fallback(
                        model, stage_b_mean_model, log_s2_const,
                        pair_mgr, K1, K2, model.K3, Kh, D, basis_name)
            else:
                results['stage_D']['selected'] = 'heteroscedastic'

        # Fitted-design record for the SELECTED heteroscedastic fit: the weighted
        # (GLS) mean design plus the precision weights that produced it, so the
        # one-call diagnostics (sigma_hat/df/LOO/epistemic CI) describe the
        # weighted fit the user gets back — not an unweighted rebuild. The weights
        # are recomputed from the final (possibly early-stop-restored) variance
        # coefficients (w_h, h0), matching the weights the final mean solve used.
        # Attribution (Sobol) is handled separately from a unit-weight companion.
        h_final = float(h0) + np.asarray(psi_all_train, dtype=np.float64) @ np.asarray(w_h, dtype=np.float64)
        sample_weights = 1.0 / np.exp(h_final)
        G3_rec = None
        if K3 > 0 and T > 0:
            G3_rec = build_gram_matrix_3d(
                build_gram_matrix(K3, _incl_lin_3, basis_name))
        _pair_grams = None
        if _pair_k2_t is not None:
            _pair_grams = [np.asarray(build_gram_matrix_2d(
                build_gram_matrix(int(k), _incl_lin_2, basis_name)),
                dtype=np.float64) for k in _pair_k2_t]
        _blocks_kw = dict(
            K2=K2, P=pair_mgr.P, G2=G2,
            pair_indices=np.asarray(pair_mgr.pair_indices),
            include_linear_2=_incl_lin_2,
            K3=K3, T=T, G3=G3_rec,
            triple_indices=(np.asarray(model.triple_indices)
                            if model.triple_indices is not None else None),
            include_linear_3=_incl_lin_3,
            fo_included=fo_included, pair_block_info=pair_block_info,
            pair_grams=_pair_grams)

        if ship_unit_mean:
            # Mean-fallback (X4B Step 2): the returned fit predicts from the
            # UNIT-WEIGHT (Hoeffding-projection) mean plus the input-dependent
            # variance. Swap the model's mean to the unit companion (f0_unit,
            # w_unit computed above) and record the UNIT-weight mean fit, so the
            # one-call diagnostics (σ̂/df/LOO/CI) describe the fit we predict from.
            import equinox as eqx
            w1_u, w2_u, w3_u = _split_mean_coeffs(
                w_unit, F1, F2, has_third=(K3 > 0 and T > 0), dtype=self._dtype)
            if fo_included is not None:
                from .._trainer_helpers import _scatter_first_order
                w1_u_model = _scatter_first_order(
                    w1_u, D, _block1, fo_included, dtype=self._dtype)
            else:
                w1_u_model = jnp.array(w1_u, dtype=self._dtype)
            mean_model_unit = MeanModel(
                f0=jnp.array(f0_unit, dtype=self._dtype),
                w1=w1_u_model,
                w2=jnp.array(w2_u, dtype=self._dtype),
                w3=jnp.array(w3_u, dtype=self._dtype),
                K1=K1, K2=K2, K3=K3, D=D,
                include_linear_1=_il1,
                include_linear_2=_incl_lin_2, include_linear_3=_incl_lin_3,
                basis_name=basis_name,
                pair_block_info=pair_block_info,
                pair_k2=_pair_k2_t)
            model = eqx.tree_at(lambda m: m.mean_model, model, mean_model_unit)
            # Shipped mean is unit-weight centered (mean_intercept_mode default).
            record = build_record(
                phi_all_train, w_unit, reg_mean, y_for_fourier, D,
                K1, G1, _il1, basis_name,
                f0=f0_unit, sample_weights=None, **_blocks_kw)
        else:
            # Fitted-design record for the SELECTED heteroscedastic fit: the
            # weighted (GLS) mean design plus the precision weights that produced
            # it, so the one-call diagnostics (sigma_hat/df/LOO/epistemic CI)
            # describe the weighted fit the user gets back — not an unweighted
            # rebuild. The weights are recomputed from the final (possibly
            # early-stop-restored) variance coefficients (w_h, h0), matching the
            # weights the final mean solve used. Attribution (Sobol) is handled
            # separately from a unit-weight companion.
            # Shipped mean is the GLS-weighted Stage-D mean — record its effective
            # estimator convention (covers both the guard-kept and guard=False
            # heteroscedastic outcomes, the only two that reach this branch).
            results['stage_D']['mean_intercept_mode'] = weighted_mean_mode
            record = build_record(
                phi_all_train, w_all, reg_mean, y_for_fourier, D,
                K1, G1, _il1, basis_name,
                f0=f0, sample_weights=sample_weights,
                mean_intercept_mode=weighted_mean_mode, **_blocks_kw)

        # Surface the fitted log-variance sub-problem (the Newton solve's design,
        # penalty, and mode) so the one-call API can compute the Tier-II one-step
        # LOO jackknife of the joint model — and the Tier-III exact nested refit —
        # from the design the trainer solved (M1, DEC-031). Psi is the SAME
        # augmented-internally design [psi1|psi2|psi3|z_h_proj] the final Newton
        # step consumed; (w_h, h0) are the final (possibly early-stop-restored)
        # coefficients whose weights the mean solve above used.
        record.variance = VarianceDesign(
            Psi=np.asarray(psi_all_train, dtype=np.float64),
            reg_var=np.asarray(reg_var, dtype=np.float64),
            w_h=np.asarray(w_h, dtype=np.float64),
            h0=float(h0),
        )

        # Unit-weight interpretive companion for Sobol attribution (two-fit
        # convention, Theorem projection Part ii): the SAME penalized mean design
        # re-solved with W = I on the same target, so attribution stays a
        # Hoeffding projection (an estimand, not a heteroscedasticity artifact).
        # Only the point/CI attribution reads this; prediction/σ̂/df/LOO use the
        # weighted record above. (f0_unit, w_unit were computed above the guard.)
        record.interpretive = build_record(
            phi_all_train, w_unit, reg_mean, y_for_fourier, D,
            K1, G1, _il1, basis_name, **_blocks_kw)
        results['fitted_design'] = record

        # Carry the shipped mean's estimator convention ON the model (DEC-047
        # provenance), so a bare ``save_model(model, path)`` without the results
        # dict still persists the right vintage. Mean-fallback ships the unit-
        # weight mean; the guard-kept/guard-off path ships the GLS-weighted mean.
        model = dataclasses.replace(
            model, mean_intercept_mode=(MEAN_INTERCEPT_UNWEIGHTED if ship_unit_mean
                                        else weighted_mean_mode))

        # Bound-activity warning (P1-1, acceptance test): the SHIPPED
        # heteroscedastic variance touched the ±LOG_VAR_CLIP box, so its
        # log-variance at those points is a clipped value, not an interior
        # stationary point. Make it visible and disclaim any exact
        # gradient/stationarity claim there. (Revert / near-noiseless outcomes
        # returned earlier with bound_active=False, so this only fires for a fit
        # that actually ships a per-point variance.)
        if results['stage_D'].get('bound_active'):
            warnings.warn(
                "Stage D: the fitted log-variance reached the ±"
                f"{LOG_VAR_CLIP:g} clip bound at one or more training points "
                "(results['stage_D']['bound_active']=True). The reported "
                "log-variance there is a clipped value, not an interior "
                "stationary point — no exact gradient/stationarity (nor "
                "monotone block-coordinate-descent) claim holds for those "
                "points. Inspect the variance model / raise lambda_h if this is "
                "unexpected.",
                stacklevel=2)
        return model
