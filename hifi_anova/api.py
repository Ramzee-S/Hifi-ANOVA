"""One-call API: hifi_anova(X, y) → complete results.

The simplest way to use HiFi-ANOVA. Takes raw data, handles preprocessing,
fitting, Sobol analysis, confidence intervals, and diagnostics automatically.
Returns a single result object with everything.

Usage:
    from hifi_anova.api import hifi_anova

    result = hifi_anova(X, y, feature_names=['income', 'age', ...])

    # Predictions
    pred = result.predict(X_new)
    lower, upper = result.predict_intervals(X_new)

    # Sobol indices with CIs
    for name, (S, lo, hi) in result.sobol_ci.items():
        print(f"{name}: S = {S:.3f} [{lo:.3f}, {hi:.3f}]")

    # Quick summary
    result.summary()

    # Save / load
    result.save('my_model/')
    from hifi_anova.model.io import load_model
    loaded = load_model('my_model/')   # dict with 'model', 'transformer', 'config', ...
"""

import warnings

import jax
import numpy as np

from .array_backend import xp as jnp  # switchable array backend (numpy exact core)
from .array_backend import use_array_backend, VALID_BACKENDS
from typing import Optional, List, Dict, Tuple, Union
from dataclasses import dataclass, field, fields

# Sentinel so the one-call API can tell an *omitted* variable_selection (which
# defaults to 'bic' for uniform bases) from an *explicit* request. On mixed
# per-variable bases the implicit default is neutralized (selection unsupported);
# an explicit request is left in config and the trainer raises a capability error
# (DEC-045).
_UNSET = object()


def _json_key(k):
    """Coerce a mapping key to a JSON object key (must be a string).

    Sobol interaction blocks key on tuples ``(i, j)`` / ``(name_i, name_j)``;
    render those as ``"i,j"`` so the payload is valid JSON while staying
    human-readable. Scalars stringify directly.
    """
    if isinstance(k, tuple):
        return ",".join(str(_json_key(x)) for x in k)
    if isinstance(k, (np.floating, np.integer)):
        return str(k.item())
    return str(k)


def _jsonify(obj):
    """Recursively convert a value to a JSON-serializable form.

    numpy/jax arrays and scalars → Python lists/numbers; tuples → lists; dict
    keys → strings (see :func:`_json_key`). Unknown objects fall back to
    ``str(obj)`` so serialization never raises.
    """
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if hasattr(obj, "__array__"):  # jax (or any other) array, backend-agnostic
        return np.asarray(obj).tolist()
    if isinstance(obj, dict):
        return {_json_key(k): _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify(v) for v in obj]
    return str(obj)


def _in_fit_backend(fn):
    """Run a HiFiResult compute method under the ARRAY BACKEND the fit used.

    Post-fit calls (``predict``, ``loo``, grids for a GUI) happen outside the
    one-call API's backend scope; without this, a numpy-core fit's model would
    be pushed back through eager JAX ops — re-paying the per-shape compile tax
    the backend choice exists to remove. The fit records its backend in
    ``result.config['array_backend']``; the fit's resolved backend (DEC-056:
    default 'auto' ⇒ numpy for a non-residual config, else jax).
    """
    import functools

    @functools.wraps(fn)
    def wrapper(self, *args, **kwargs):
        name = (self.config or {}).get("array_backend") or "jax"
        with use_array_backend(name):
            return fn(self, *args, **kwargs)
    return wrapper


@dataclass
class HiFiResult:
    """Complete results from hifi_anova().

    Contains the fitted model, Sobol indices, confidence intervals,
    noise estimates, and all diagnostic information. Provides
    convenience methods for prediction and reporting.
    """
    # Core model
    model: object  # HiFiANOVA
    config: Dict
    feature_names: List[str]

    # Preprocessing
    transformer: object  # QuantileTransformer
    y_mean: float
    y_std: float

    # Training results
    train_results: Dict

    # Sobol indices
    # Full structural spectrum from ``compute_sobol_indices`` — decomposed from the
    # *fitted (predictive)* model. For a heteroscedastic (Stage-D) fit that is the
    # precision-weighted (GLS) mean, so ``sobol['mean_sobol']`` is the EFFICIENT
    # spectrum, distinct from the interpretable ``sobol_ci`` below (two-fit
    # convention, DEC-030; the two coincide for a homoscedastic fit). Use
    # ``sobol_ci`` for the headline attribution and ``sobol_ci_efficient`` for the
    # weighted first-order CIs.
    sobol: Dict
    # {name: (S, lo, hi)} first-order CORE share V_u/V_core — the INTERPRETABLE
    # (unit-weight) attribution, INVARIANT to residual augmentation (DEC-034), and
    # the HEADLINE attribution the summary reports (DEC-030).
    sobol_ci: Dict

    # Diagnostics
    sigma_hat: float  # noise estimate
    # Explained-variance R² (1 − Var(y−ŷ)/Var(y)) on the test set — the
    # framework-native, manuscript-reported convention. The textbook SSE/TSS
    # coefficient of determination is in ``r_squared_classical`` below; both
    # via hifi_anova.analysis.metrics.r_squared.
    r_squared: float
    loo_cv: float
    df: float  # effective degrees of freedom

    # Two-fit reporting surface (M2; Manuscript_Theoryv06 Thm projection Part ii).
    # For a heteroscedastic (Stage-D) fit the *efficient* (precision-weighted)
    # first-order Sobol CIs {name: (S, lo, hi)}, alongside the interpretable
    # ``sobol_ci``. ``None`` on a homoscedastic fit, where the efficient and
    # interpretable fits coincide (nothing separate to report).
    sobol_ci_efficient: Optional[Dict] = None
    # Observed efficient−interpretable gap per component — the
    # "heteroscedasticity × misspecification" diagnostic. ``None`` when
    # homoscedastic. Shape: {'first_order': {name: gap}, 'second_order':
    # {(i, j): gap}}.
    sobol_gap: Optional[Dict] = None

    # Structural fidelity 𝔉 (Manuscript_Theoryv06 §8 Eq. fidelity; M3/DEC-032).
    # Dict {'value': 𝔉, 'var_core', 'var_residual', 'cross_covariance',
    # 'orthogonality_defect', 'conditional_on_residual_variance'}. 𝔉 =
    # V_core/(V_core+Var(ĝ)) is the fraction of explained variance the interpretable
    # decomposition carries; 1−𝔉 is the honest interpretability gap. Always populated
    # (``value`` = 1.0 when no residual stage ran) so downstream can compute
    # 𝔉·S_core unconditionally; ``orthogonality_defect`` = 2·Ĉov(f̂_core,ĝ)/Var(f̂)
    # reports how far the empirically-orthogonal residual is from exact orthogonality.
    fidelity: Optional[Dict] = None
    # First-order SHARE OF FITTED VARIANCE Ŝ^fit = 𝔉·Ŝ^core with CIs {name: (S, lo, hi)}
    # — the core ``sobol_ci`` scaled by 𝔉 (= V_u/(V_core+Var ĝ)), CONDITIONAL on the
    # (QMC-estimated) residual variance. NB this is NOT the total-EFFECT index S_T
    # (first + interactions), which lives in ``sobol['mean_sobol']['total_order']``;
    # "fitted variance" names its own denominator to avoid that collision (DEC-034; a
    # field rename ``sobol_ci_total → sobol_ci_fitted_variance`` is staged for a later
    # release). ``None`` when no residual stage ran (𝔉≡1 ⇒ fitted ≡ core = ``sobol_ci``),
    # so the homoscedastic/no-residual surface collapses to the single core set.
    sobol_ci_total: Optional[Dict] = None
    # Opt-in first-order SHARE OF OBSERVED OUTPUT VARIANCE V_u/Var(Y) with CIs
    # {name: (S, lo, hi)} = ``sobol_ci_total``·(Var f̂/Var Y) (fitted-variance share
    # when a residual ran, else core). A practitioner view that DELIBERATELY confounds
    # attribution with fit quality — never co-tabulate with the canonical shares; the CI
    # treats the Var(f̂)/Var(Y) scale as fixed. NB Var(f̂)/Var(Y) equals R² only under
    # OLS-with-intercept orthogonality — for a regularized/nonlinear fit it differs (and
    # can exceed 1), which is exactly why the library computes the scaling rather than
    # advising "multiply by R²". Printed only via ``summary(observed=True)`` (DEC-034).
    sobol_ci_observed: Optional[Dict] = None
    # For a Stage-D (heteroscedastic) fit, ``sigma_hat`` is the *whitened
    # calibration scale* sqrt(RSS_w/df_res) — ≈1 when the variance model is
    # calibrated — NOT a homoscedastic noise level; the point-dependent noise is
    # sigma^2(x), available via ``sigma_x2``. False on the homoscedastic path,
    # where ``sigma_hat`` is the usual noise estimate.
    noise_scale_is_calibration: bool = False

    # Classical (SSE/TSS) coefficient of determination on the test set —
    # 1 − Σ(y−ŷ)²/Σ(y−ȳ)², sklearn ``r2_score``. Reported alongside the
    # framework-native ``r_squared`` (explained-variance score), which is the
    # value the manuscript reports. The two coincide unless the test residual
    # has a nonzero mean. See hifi_anova.analysis.metrics.
    r_squared_classical: float = 0.0

    # Predictive LOO negative log-likelihood on the common Gaussian scale — the
    # cross-model-comparable criterion (Manuscript_Theoryv06 App. C, M1/DEC-031),
    # populated on BOTH paths. Homoscedastic: 1/2 log sigma_hat^2 +
    # loo_cv/(2 sigma_hat^2) + 1/2 log 2pi. Unlike the (model-metric) weighted
    # loo_cv this lives on one scale, so homo-vs-hetero comparison is legitimate.
    loo_nll: float = 0.0
    # Which LOO tier the reported loo_cv / loo_nll are: 1 (plug-in — the
    # homoscedastic path, where tiers I/II/III coincide) or 2 (the Stage-D
    # default one-step variance jackknife). Tier III (exact nested refit) is
    # available on demand via ``result.loo(tier=3)``.
    loo_tier: int = 1
    # Tier-II regularity flags (Stage-D only; ``None`` on the homoscedastic path).
    # ``loo_tier2_guarantee_holds`` is False when a variance floor binds (KKT
    # test) or H_h is ill-conditioned — in that regime Tier III is authoritative
    # (``result.loo(tier=3)``) and ``summary()`` prints a warning line.
    loo_tier2_guarantee_holds: Optional[bool] = None
    loo_variance_floor_active: Optional[bool] = None
    variance_hessian_ill_conditioned: Optional[bool] = None
    # Residual degrees of freedom N - 2 tr(H) + tr(H^2) (ridge_analytics) —
    # used as the Student-t df for prediction intervals (sigma_hat is
    # estimated, so z quantiles undercover at small N). None ⇒ z quantiles.
    df_residual: Optional[float] = None

    # Internal (for prediction intervals)
    _Phi_train: np.ndarray = field(repr=False, default=None)
    _reg_diag: np.ndarray = field(repr=False, default=None)
    _sample_weights: np.ndarray = field(repr=False, default=None)
    _data: Dict = field(repr=False, default=None)
    # The fitted-design record (mean + variance sub-problem) — lets ``loo()``
    # recompute any tier on demand from the design the trainer solved.
    _fitted_design: object = field(repr=False, default=None)
    # Fixed-configuration inference provenance (X6 Session 3 / DEC-050).
    # Appended after all legacy fields to preserve positional construction.
    # These labels do not alter the legacy ``sobol_ci`` tuples. Per-component
    # status prevents a zero-gradient boundary block from being presented as an
    # ordinary delta-method interval; bootstrap/quadratic-form null inference is
    # deferred.
    inference_metadata: Dict = field(default_factory=dict)
    sobol_ci_status: Dict = field(default_factory=dict)

    @property
    def _pred_dtype(self):
        """Input dtype for prediction — matches the fitted model's weight dtype
        (float32 by default, float64 for a ``precision='float64'`` fit; DEC-035),
        so a float64 fit also predicts in float64 rather than re-narrowing x."""
        mm = getattr(self.model, 'mean_model', None)
        f0 = getattr(mm, 'f0', None)
        return f0.dtype if f0 is not None else jnp.float32

    @property
    def split_indices(self) -> Optional[Dict[str, np.ndarray]]:
        """Original-dataset row indices for the train/val/test splits (BR-07).

        Returns ``{'train': ndarray, 'val': ndarray, 'test': ndarray}`` of
        integer indices into the ORIGINAL ``X``/``y`` passed to ``hifi_anova``,
        each in the exact row order of the corresponding internal split. So
        ``X[result.split_indices['train']]`` reproduces the rows the fitted
        design's ``Phi`` was built from, in ``Phi`` order — the honest way to
        map a per-point diagnostic (LOO residual, leverage, worst out-of-sample
        row) back to a dataset row id, without re-deriving the seeded
        permutation. Read-only provenance; ``None`` if the result was built
        without the preprocessing split (e.g. a hand-constructed result).
        """
        data = self._data
        if not isinstance(data, dict) or 'train_indices' not in data:
            return None
        return {
            'train': np.asarray(data['train_indices']),
            'val': np.asarray(data['val_indices']),
            'test': np.asarray(data['test_indices']),
        }

    def _transform_new_inputs(self, X_new: np.ndarray) -> np.ndarray:
        """Quantile-transform new inputs, warning on out-of-training-range rows.

        Out-of-range inputs saturate at quantile 0/1. There the periodic
        (Fourier) basis components coincide at both edges, so the nonlinear
        part of the prediction cannot distinguish a far-below-range point from
        a far-above-range one, and the prediction is a CONSTANT saturation,
        not a trend extrapolation; prediction intervals do not widen either.
        """
        X_new = np.asarray(X_new, dtype=np.float64)
        q = getattr(self.transformer, 'quantiles_', None)
        if (q is not None and X_new.ndim == 2
                and X_new.shape[1] == q.shape[1]):
            out_of_range = np.any((X_new < q[0]) | (X_new > q[-1]), axis=1)
            n_out = int(np.sum(out_of_range))
            if n_out:
                warnings.warn(
                    f"{n_out}/{X_new.shape[0]} prediction inputs lie outside "
                    "the training range of at least one feature and saturate "
                    "at the quantile boundary (0/1). Predictions there are "
                    "constant extrapolations (the periodic basis coincides at "
                    "both edges) and intervals do not widen — treat them with "
                    "caution.", UserWarning, stacklevel=3)
        return np.clip(self.transformer.transform(X_new), 0, 1)

    @_in_fit_backend
    def predict(self, X_new: np.ndarray) -> np.ndarray:
        """Predict on new data (original scale).

        Args:
            X_new: (M, D) new inputs in ORIGINAL feature space

        Returns:
            (M,) predictions
        """
        X_t = self._transform_new_inputs(X_new)
        x = jnp.array(X_t, dtype=self._pred_dtype)
        return np.asarray(self.model.predict_mean_only(x))

    @_in_fit_backend
    def predict_intervals(self, X_new: np.ndarray, alpha: float = 0.05
                          ) -> Tuple[np.ndarray, np.ndarray]:
        """Prediction intervals on new data.

        Args:
            X_new: (M, D) in original feature space
            alpha: significance level (0.05 = 95%)

        Returns:
            (lower, upper) — both (M,) arrays
        """
        from .model.predict import predict_intervals
        from .training.fitted_design import MEAN_INTERCEPT_PROFILED_JOINT_GLS

        X_t = self._transform_new_inputs(X_new)
        x = jnp.array(X_t, dtype=self._pred_dtype)

        # Profiled joint-GLS Stage-D mean ⇒ the weighted epistemic posterior is the
        # augmented one (the fitted intercept is itself uncertain; Remark
        # rem:intercept). Homoscedastic / legacy-fixed / unweighted-centered means
        # keep the feature-only posterior that matches how they were solved.
        _profiled = (self._fitted_design is not None
                     and self._fitted_design.mean_intercept_mode
                     == MEAN_INTERCEPT_PROFILED_JOINT_GLS)
        result = predict_intervals(
            self.model, x,
            Phi_train=self._Phi_train,
            reg_diag=self._reg_diag,
            sigma2_hat=self.sigma_hat ** 2,
            alpha=alpha,
            weights=self._sample_weights,
            profile_intercept=_profiled,
            df_residual=self.df_residual,
        )
        return result['lower'], result['upper']

    @_in_fit_backend
    def sigma_x2(self, X_new: np.ndarray) -> np.ndarray:
        """Point-dependent aleatoric noise variance sigma^2(x) at new inputs.

        For a heteroscedastic (Stage-D) fit this is the fitted variance model
        evaluated at ``X_new``; the scalar ``sigma_hat`` is then only a whitened
        calibration meter (``noise_scale_is_calibration`` is True). For a
        homoscedastic fit the model carries no input-dependent variance and this
        returns the neutral unit variance — use ``sigma_hat**2`` as the constant
        noise level there.

        Args:
            X_new: (M, D) inputs in ORIGINAL feature space

        Returns:
            (M,) sigma^2(x)
        """
        X_t = self._transform_new_inputs(X_new)
        x = jnp.array(X_t, dtype=self._pred_dtype)
        _mean, var = self.model.predict(x)
        return np.asarray(var)

    @_in_fit_backend
    def loo(self, tier: int = 2) -> Dict:
        """Leave-one-out diagnostics at an explicit tier (M1; v06 App. C).

        Three-tier LOO hierarchy for the *joint* heteroscedastic model:

        * ``tier=1`` — plug-in: the variance model is held at its full-data
          value (optimistically biased; use to reproduce old tables).
        * ``tier=2`` — one-step variance jackknife (**the Stage-D reported
          default**): corrects the deleted variance prediction by one Newton
          step; ``O(N F_h^2)``.
        * ``tier=3`` — exact nested refit (**the oracle**, and authoritative
          whenever a variance floor binds): refits the joint estimator on each
          leave-one-out fold. ``O(N)`` full joint refits — **expensive**.

        On the homoscedastic path the three tiers coincide, so any ``tier``
        returns the same (tier-1) numbers.

        Returns a dict with ``loo_nll``, ``loo_cv`` (tiers 1/2), ``loo_tier``,
        and — for tier 2 — the regularity flags; tier 3 returns ``loo_nll`` with
        ``per_point_nll``.
        """
        if tier not in (1, 2, 3):
            raise ValueError(f"tier must be 1, 2, or 3; got {tier!r}")
        from .analysis.automl import ridge_analytics, joint_loo, exact_loo_nll

        rec = self._fitted_design
        if rec is None:
            # Legacy rebuild path: only the reported (tier-1 / homoscedastic)
            # numbers are available.
            return {'loo_nll': self.loo_nll, 'loo_cv': self.loo_cv,
                    'loo_tier': self.loo_tier}

        from .training.fitted_design import MEAN_INTERCEPT_PROFILED_JOINT_GLS
        _profiled = (rec.mean_intercept_mode
                     == MEAN_INTERCEPT_PROFILED_JOINT_GLS)
        # Profiled joint-GLS: analytics profile the SAME unpenalized intercept as
        # the shipped mean (augmented design), so the deleted-row LOO re-profiles
        # f0. Pass the uncentered response so the augmented fit recovers f0. The
        # legacy fixed-intercept / unweighted-centered modes keep the feature-only
        # instrument that matches how their mean was actually solved.
        _y_ana = (rec.y_centered + rec.f0) if _profiled else rec.y_centered
        analytics = ridge_analytics(rec.Phi, _y_ana, rec.reg_diag,
                                    weights=rec.sample_weights,
                                    profile_intercept=_profiled)
        tier1 = {'loo_nll': analytics['loo_nll'], 'loo_cv': analytics['loo_cv'],
                 'loo_tier': analytics['loo_tier']}
        # Homoscedastic (or no surfaced variance design): tiers coincide.
        if not (rec.is_weighted and rec.variance is not None):
            return tier1
        if tier == 1:
            return tier1
        if tier == 2:
            return joint_loo(analytics, rec.variance)

        # tier == 3: exact nested refit. Warn on the O(N)-refit cost; the
        # Stage-C mean residual is folded into y (= y_for_fourier) upstream, so
        # this is the joint sub-problem Stage D solved.
        import warnings
        N = rec.Phi.shape[0]
        if N > 400:
            warnings.warn(
                f"result.loo(tier=3) runs {N} exact joint refits (O(N) — this "
                f"may take a while). Tier 2 is the reported default; tier 3 is "
                f"the oracle / floor-binding authority.", stacklevel=2)
        y = rec.y_centered + rec.f0
        return exact_loo_nll(rec.Phi, y, rec.reg_diag,
                             rec.variance.Psi, rec.variance.reg_var)

    def summary(self, headline: str = "core", observed: bool = False):
        """Print a human-readable summary.

        Args:
            headline: which Sobol normalization leads the shares table when a
                residual stage ran — ``"core"`` (default; ``V_u/V_core``,
                *invariant* to the residual model) or ``"fitted_variance"``
                (``V_u/(V_core+Var ĝ) = 𝔉·core``, the SA-convention whole-fit
                share). Presentation only; both columns are always shown. On a
                homoscedastic / no-residual fit (𝔉≡1) the two coincide and a
                single column is printed. (DEC-034)
            observed: also print the opt-in *share of observed output variance*
                ``V_u/Var(Y)`` — a practitioner view that confounds attribution
                with fit quality (see ``sobol_ci_observed``). Off by default.
        """
        if headline not in ("core", "fitted_variance"):
            raise ValueError(
                f"headline must be 'core' or 'fitted_variance'; got {headline!r}")
        D = self.model.D
        print("HiFi-ANOVA Model Summary")
        print(f"  Variables: {D} ({', '.join(self.feature_names[:5])}"
              f"{'...' if D > 5 else ''})")
        print(f"  R²: {self.r_squared:.4f} (explained-variance)"
              f"  |  classical (SSE/TSS): {self.r_squared_classical:.4f}")
        if self.noise_scale_is_calibration:
            print(f"  Variance calibration (σ̂_w): {self.sigma_hat:.4f} "
                  f"(≈1 when calibrated; σ²(x) is input-dependent — see sigma_x2)")
        else:
            print(f"  Noise (σ̂): {self.sigma_hat:.4f}")
        _tier_tag = f" (Tier {self.loo_tier})" if self.noise_scale_is_calibration else ""
        print(f"  LOO-CV: {self.loo_cv:.4f}   LOO-NLL: {self.loo_nll:.4f}{_tier_tag}")
        print(f"  Effective df: {self.df:.1f}")
        _same_data_selection = bool(
            self.inference_metadata.get('structure_selected_on_same_data', False))
        if _same_data_selection:
            print("  Inference: HC3 delta-t intervals condition on transform, "
                  "basis, admitted structure, penalties, and weights; "
                  "post-selection coverage is not claimed.")
        else:
            print("  Inference: HC3 delta-t intervals condition on the fixed "
                  "transform, basis, admitted structure, penalties, and weights.")
        # Tier-II guarantee: warn when a variance floor binds or H_h is
        # ill-conditioned, where App. C makes Tier III authoritative (M1.3).
        if self.loo_tier2_guarantee_holds is False:
            why = []
            if self.loo_variance_floor_active:
                why.append("a variance floor is active")
            if self.variance_hessian_ill_conditioned:
                why.append("the variance Hessian is ill-conditioned")
            print(f"  ⚠ Tier-II LOO guarantee at risk ({'; '.join(why)}): "
                  f"Tier III is authoritative in this regime — call "
                  f"result.loo(tier=3).")
        # User-defined term structure (BR-04/BR-06/BR-01): make the equation
        # system — and the non-hierarchical honesty caveat — visible, not just
        # machine-readable in results['term_structure'].
        _ts = (self.train_results or {}).get('term_structure')
        if _ts:
            _pk2 = _ts.get('pair_k2')
            if _pk2:
                print("  Term structure: per-pair K2 "
                      + ", ".join(f"({p})→{k}" for p, k in _pk2.items()))
            _foe = _ts.get('first_order_excluded') or []
            _pe = _ts.get('pair_excluded') or []
            if _foe:
                print(f"  Term structure: first-order block excluded for "
                      f"variable(s) {_foe} (order-2-only; NON-HIERARCHICAL — "
                      f"pair shares are conditional on the omitted marginal)")
            if _pe:
                print(f"  Term structure: variable(s) {_pe} excluded from all "
                      f"pairs (order-1-only)")
        _vv = getattr(self.model.variance_model, 'variance_variables', None) \
            if self.model.variance_model is not None else None
        if _vv is not None:
            print(f"  Variance structure: σ²(x) modeled on variable(s) "
                  f"{list(_vv)} only (homoscedasticity ASSERTED elsewhere)")
        print()
        # Structural fidelity 𝔉 and the interpretability gap (v06 §8; M3/DEC-032).
        # Shown only when a residual stage ran (𝔉<1); homoscedastic/no-residual fits
        # have 𝔉≡1 and nothing extra to report.
        _fid = self.fidelity
        # ``sobol_ci_total`` carries the "share of fitted variance" (𝔉·core) — it is
        # populated only when a residual stage ran, so it doubles as the "has a
        # fitted-variance surface" flag. NB the 𝔉-scaled share is NOT the total-EFFECT
        # index S_T (first + interactions); see ``mean_sobol['total_order']``. (DEC-034)
        _has_fit = self.sobol_ci_total is not None
        if _fid is not None and _fid.get('var_residual', 0.0) > 0.0:
            _F = _fid['value']
            print(f"  Structural fidelity 𝔉: {_F:.4f} "
                  f"(interpretability gap 1−𝔉 = {1.0 - _F:.4f})")
            # 𝔉 vs R² gloss (Q2/DEC-034): 𝔉 is model-internal, not a fit-quality number.
            print("    ▸ 𝔉 = share of the fitted function's variance carried by the "
                  "interpretable structure")
            print(f"      (model-internal; distinct from R² above). "
                  f"{100.0 * (1.0 - _F):.1f}% of the fit lives in the residual ĝ.")
            _defect = _fid.get('orthogonality_defect', 0.0)
            if abs(_defect) > 0.01:
                print(f"    ⚠ orthogonality defect 2·Cov(f̂_core,ĝ)/Var(f̂) = "
                      f"{_defect:+.4f}: core and residual are not exactly "
                      f"orthogonal, so the 𝔉 identity is approximate here.")

        # Sobol shares. Homoscedastic / no-residual (𝔉≡1) prints the single core
        # column exactly as before (byte-identical). With a residual stage both the
        # core (invariant) and fitted-variance (= 𝔉·core) shares are shown; ``headline``
        # chooses which leads (presentation only). The residual row surfaces 1−𝔉.
        if not _has_fit:
            _ci_label = ("conditional intervals" if _same_data_selection
                         else "95% fixed-configuration CI")
            print(f"  Sobol Indices ({_ci_label}):")
            for name, (S, lo, hi) in sorted(self.sobol_ci.items(),
                                            key=lambda x: -x[1][0]):
                if (S > 0.01
                        and self.sobol_ci_status.get(name, 'regular') == 'regular'):
                    print(f"    {name:15s}: {S:.4f} [{lo:.4f}, {hi:.4f}]")
        else:
            _core_first = (headline == "core")
            c_lab = "core" if _core_first else "fitted-variance"
            s_lab = "fitted-variance" if _core_first else "core"
            _ci_label = ("conditional intervals" if _same_data_selection
                         else "95% fixed-configuration CI")
            print(f"  Sobol shares ({_ci_label}) — core = within interpretable structure "
                  "(invariant);")
            print("                          fitted-variance = V_u / total fitted "
                  "variance (= 𝔉·core):")
            print(f"    {'variable':15s}   {c_lab:24s} {s_lab}")

            def _rank_key(item):
                name, (S, _lo, _hi) = item
                if _core_first:
                    return -S
                st = self.sobol_ci_total.get(name)
                return -(st[0] if st else 0.0)

            for name, (S, lo, hi) in sorted(self.sobol_ci.items(), key=_rank_key):
                st = self.sobol_ci_total.get(name)
                if st is None:
                    continue
                St, lot, hit = st
                if ((S if _core_first else St) <= 0.01
                        or self.sobol_ci_status.get(name, 'regular')
                        != 'regular'):
                    continue
                core_col = f"{S:.4f} [{lo:.4f}, {hi:.4f}]"
                fit_col = f"{St:.4f} [{lot:.4f}, {hit:.4f}]"
                col0, col1 = ((core_col, fit_col) if _core_first
                              else (fit_col, core_col))
                print(f"    {name:15s}   {col0:24s} {col1}")
            # Residual row: 1−𝔉 lives in the fitted-variance column ("—" for core,
            # which has no residual share by construction).
            gap = 1.0 - (_fid['value'] if _fid is not None else 1.0)
            res_core, res_fit = "—", f"{gap:.4f}"
            col0, col1 = ((res_core, res_fit) if _core_first
                          else (res_fit, res_core))
            print(f"    {'(residual ĝ)':15s}   {col0:24s} {col1}      ← 1−𝔉")
            print("  Note: total-EFFECT indices S_T (first + interactions) are separate"
                  " —\n        result.sobol['mean_sobol']['total_order'].")

        _nonregular_null = sorted(
            name for name, status in self.sobol_ci_status.items()
            if status == 'nonregular_null')
        if _nonregular_null:
            print("  Nonregular null component(s): "
                  + ", ".join(_nonregular_null)
                  + ". Ordinary delta intervals are suppressed; bootstrap/"
                    "quadratic-form inference is deferred.")
        _nonregular_boundary = sorted(
            name for name, status in self.sobol_ci_status.items()
            if status == 'nonregular_boundary')
        if _nonregular_boundary:
            print("  Nonregular complete-share boundary component(s): "
                  + ", ".join(_nonregular_boundary)
                  + ". The full delta gradient is degenerate, so the zero-width "
                    "interval is suppressed as an ordinary coverage statement.")

        # Opt-in share of OBSERVED output variance V_u/Var(Y) (Q4/DEC-034). Confounds
        # attribution with fit quality, so it is off by default and never co-tabulated
        # with the canonical shares; the CI treats the Var(f̂)/Var(Y) scale as fixed and
        # the residual+unexplained tail is a single lump (need not sum to 1 exactly).
        if observed and self.sobol_ci_observed is not None:
            print("\n  Share of OBSERVED output variance (V_u / Var Y) — confounds "
                  "attribution\n  with fit quality; CI treats Var(f̂)/Var(Y) as fixed:")
            _obs_sum = 0.0
            for name, (S, lo, hi) in sorted(self.sobol_ci_observed.items(),
                                            key=lambda x: -x[1][0]):
                _obs_sum += S
                if (S > 0.005
                        and self.sobol_ci_status.get(name, 'regular') == 'regular'):
                    print(f"    {name:15s}: {S:.4f} [{lo:.4f}, {hi:.4f}]")
            print(f"    {'(residual+noise)':15s}: {max(0.0, 1.0 - _obs_sum):.4f}   "
                  f"(lump; components + lump ≈ 1)")

        # Heteroscedasticity × misspecification gap (two-fit convention). Only
        # meaningful for a Stage-D fit; shown only when non-negligible so the
        # homoscedastic surface is unchanged. The reported ``sobol_ci`` above is
        # the interpretable (unit-weight) attribution; a large gap flags that the
        # precision-weighted (efficient, predictive) fit attributes differently.
        _GAP_TOL = 0.01
        if self.sobol_gap is not None:
            first_gap = self.sobol_gap.get('first_order', {})
            second_gap = self.sobol_gap.get('second_order', {})
            flagged = ([(n, g) for n, g in first_gap.items()
                        if abs(g) > _GAP_TOL]
                       + [(f"({self.feature_names[i]}, {self.feature_names[j]})", g)
                          for (i, j), g in second_gap.items()
                          if abs(g) > _GAP_TOL])
            if flagged:
                print("\n  Heteroscedasticity × misspecification gap "
                      "(efficient − interpretable):")
                for label, g in sorted(flagged, key=lambda x: -abs(x[1])):
                    print(f"    {label:15s}: {g:+.4f}")

        # Second-order interactions. These come from ``self.sobol`` — the
        # structural spectrum of the *fitted (predictive)* model, which for a
        # heteroscedastic fit is the precision-weighted (GLS) mean — whereas the
        # first-order table above is the interpretable (unit-weight) ``sobol_ci``
        # headline (two-fit convention, DEC-030). The two coincide for a
        # homoscedastic fit; under heteroscedasticity they can differ slightly, so
        # the interaction section is labelled to name its convention rather than
        # silently mixing it with the interpretable first-order table.
        so = self.sobol['mean_sobol'].get('second_order', {})
        if so:
            top_pairs = sorted(so.items(), key=lambda x: -x[1])[:5]
            displayed = [(k, v) for k, v in top_pairs if v > 0.005]
            if displayed:
                _weighted = self.sobol_gap is not None   # heteroscedastic fit
                _lbl = (" (structural / predictive-fit spectrum)"
                        if _weighted else "")
                print(f"\n  Top Interactions{_lbl}:")
                for (i, j), s in displayed:
                    ni = self.feature_names[i] if i < len(self.feature_names) else f'x{i}'
                    nj = self.feature_names[j] if j < len(self.feature_names) else f'x{j}'
                    print(f"    ({ni}, {nj}): {s:.4f}")

    def save(self, path: str):
        """Save the model to a directory."""
        from .model.io import save_model
        save_model(
            self.model, path,
            config=self.config,
            transformer=self.transformer,
            feature_names=self.feature_names,
            results=self.train_results,
            overwrite=True,
        )

    def to_dict(self) -> Dict:
        """Return a JSON-serializable dict of the reporting surface.

        For GUI / web back-ends that want the results as a payload without a
        filesystem round-trip (unlike :meth:`save`, which writes a model
        directory). Includes the Sobol spectra + CIs, fidelity, scalar
        diagnostics (sigma_hat / R² / LOO / df), the inference-provenance
        metadata, feature names, config, and the per-stage ``train_results``
        metrics. Excludes non-serializable handles (the fitted ``model``, the
        transformer, and the internal design arrays); the heavy fitted-design
        record is dropped from ``train_results``. Arrays become lists, numpy/jax
        scalars become Python numbers, and Sobol interaction keys ``(i, j)``
        become ``"i,j"`` strings (see :func:`_json_key`).

        The result is directly ``json.dumps``-able.
        """
        skip = {'model', 'transformer'}
        out = {}
        for f in fields(self):
            if f.name.startswith('_') or f.name in skip:
                continue
            value = getattr(self, f.name)
            if f.name == 'train_results' and isinstance(value, dict):
                # Drop the fitted-design record (an object, not JSON) but keep
                # the stage metrics a GUI wants to show.
                value = {k: v for k, v in value.items() if k != 'fitted_design'}
            out[f.name] = _jsonify(value)
        return out

    @_in_fit_backend
    def component_curve(self, variable: Union[int, str], n_points: int = 200):
        """Get the learned component function for a variable.

        Args:
            variable: index or name
            n_points: grid points

        Returns:
            (x_grid, f_values) in [0,1] quantile space
        """
        from .analysis.component_eval import first_order_on_grid

        if isinstance(variable, str):
            variable = self.feature_names.index(variable)
        return first_order_on_grid(self.model, variable, n_points)


def _two_fit_gap(ci_interp, ci_efficient, feature_names):
    """Observed efficient−interpretable Sobol gap (M2; Thm projection Part ii).

    The "heteroscedasticity × misspecification" diagnostic: for each retained
    component the difference between the precision-weighted (*efficient*) index
    and the unit-weight (*interpretable*) index, both closed-form on the same
    blocks. Manuscript_Theoryv06 reports the *observed* gap (not the plug-in
    bound). First-order keys are feature names; second-order keys are (i, j)
    index tuples. Returns ``{'first_order': {...}, 'second_order': {...}}``.
    """
    def _name(i):
        return feature_names[i] if i < len(feature_names) else f'x{i+1}'

    first = {}
    for i, (S_e, _lo, _hi) in ci_efficient['first_order'].items():
        S_i = ci_interp['first_order'].get(i, (0.0, 0.0, 0.0))[0]
        first[_name(i)] = float(S_e - S_i)

    second = {}
    for key, (S_e, _lo, _hi) in ci_efficient.get('second_order', {}).items():
        S_i = ci_interp.get('second_order', {}).get(key, (0.0, 0.0, 0.0))[0]
        second[key] = float(S_e - S_i)

    return {'first_order': first, 'second_order': second}


def _same_data_structure_selection_reasons(config, train_results):
    """Return conservative reasons that the admitted structure was adaptive.

    Config-only controls are gated by the path that actually ran where that is
    observable. Explicit fixed lists/dicts are not classified as selection.
    """
    reasons = []
    adaptive_keys = (
        'variable_selection', 'pair_selection', 'first_order_pruning',
        'pair_pruning', 'triple_pruning', 'var_pair_selection',
        'var_triple_selection',
    )
    for key in adaptive_keys:
        value = config.get(key)
        # An explicit list/tuple of active variables is fixed structure supplied
        # by the caller, not a data-driven ``pair_selection`` rule.
        if key == 'pair_selection' and isinstance(value, (list, tuple)):
            continue
        if value not in (None, False, 'none', 'all'):
            reasons.append(key)

    if config.get('mode') == 'auto':
        reasons.append('auto_stage_selection')
    if train_results.get('selection_applied') is True:
        reasons.append('observed_selection_applied')

    stage_b = train_results.get('stage_B')
    fitted_triples = (stage_b.get('n_triples', 0)
                      if isinstance(stage_b, dict) else 0)
    triple_selection = config.get('triple_selection', 'all_active')
    if fitted_triples > 0 and triple_selection != 'all':
        reasons.append('triple_selection')

    if (config.get('basis_per_variable') == 'auto'
            and train_results.get('mixed_basis') is True):
        reasons.append('basis_per_variable_auto')

    stage_d = train_results.get('stage_D')
    if isinstance(stage_d, dict) and config.get('heteroscedastic_guard', True):
        compared_models = ('nll_homoscedastic' in stage_d
                           and 'nll_heteroscedastic' in stage_d)
        near_noiseless_skip = (stage_d.get('reason') == 'near-noiseless')
        if compared_models or near_noiseless_skip:
            reasons.append('stage_d_guard_selection')

    for key in ('variable_selection', 'first_order_pruning', 'pair_pruning',
                'triple_pruning'):
        if key in train_results and key not in reasons:
            reasons.append(key)
    return reasons


def _fixed_configuration_inference_metadata(config, train_results):
    """Return JSON-safe inference scope labels for the shipped result.

    ``True`` is intentionally conservative: any data-adaptive stage, group,
    or pruning choice that can affect the admitted structure is treated as
    same-data selection.  The intervals themselves are unchanged and remain
    conditional on the transform, basis, admitted structure, penalties, and
    weights.
    """
    selection_reasons = _same_data_structure_selection_reasons(
        config, train_results)
    return {
        'inference_scope': 'fixed_configuration',
        'structure_selected_on_same_data': bool(selection_reasons),
        'post_selection_coverage': 'not_claimed',
        'conditioned_on': [
            'transform', 'basis', 'admitted_structure', 'penalties', 'weights'],
        'interval_method': 'HC3_delta_t',
    }


def _legacy_analytics_rebuild(model, data, config, D, K1, K2, strategy):
    """Legacy post-fit analytics: rebuild an independent ridge from the model.

    Fallback for any fit path that does not yet surface a ``fitted_design``
    record (currently none of the supported one-call paths — the mixed
    per-variable basis is rejected earlier). Kept behavior-identical to the
    pre-record code so the fallback is a safe no-op on the homoscedastic
    ≤2nd-order path. Returns ``(Phi_train, reg_diag, analytics, ci)``.

    NB: this reconstruction pads third-order/residual penalties with
    ``lambda_order2`` and drops the third-order Sobol denominator — the very
    defects the record path fixes — so it is a last resort, not a parallel
    implementation.
    """
    from .analysis.automl import ridge_analytics, sobol_confidence_intervals
    from .core.gram import build_gram_matrix, build_gram_matrix_2d
    from .training.regularization import build_regularization_vector

    basis_name = model.basis_name
    include_linear_1 = model.include_linear_1
    include_linear_2 = getattr(model, 'include_linear_2', True)

    Phi_train = np.asarray(model.build_phi_all(data['x_train']), dtype=np.float64)
    G1 = np.asarray(build_gram_matrix(K1, include_linear=include_linear_1,
                                      basis_name=basis_name), dtype=np.float64)
    f0 = float(np.mean(np.asarray(data['y_train'])))
    y_c = np.asarray(data['y_train'], dtype=np.float64) - f0
    reg_diag = np.asarray(build_regularization_vector(
        D, K1, K2,
        model.pair_indices.shape[0] if model.pair_indices is not None else 0,
        strategy, config['lambda_order1'], config['lambda_order2'],
        include_linear_1=include_linear_1, include_linear_2=include_linear_2,
        basis_name=basis_name), dtype=np.float64)

    if len(reg_diag) < Phi_train.shape[1]:
        reg_diag = np.concatenate([reg_diag,
            np.full(Phi_train.shape[1] - len(reg_diag), config['lambda_order2'])])

    analytics = ridge_analytics(Phi_train, y_c, reg_diag)

    G2 = None
    if K2 > 0 and model.pair_indices is not None:
        G2 = np.asarray(build_gram_matrix_2d(
            build_gram_matrix(K2, include_linear=include_linear_2,
                              basis_name=basis_name)), dtype=np.float64)
    ci = sobol_confidence_intervals(
        Phi_train, y_c, reg_diag, D, K1, G1,
        K2=K2,
        P=model.pair_indices.shape[0] if model.pair_indices is not None else 0,
        G2=G2,
        pair_indices=(np.asarray(model.pair_indices)
                      if model.pair_indices is not None else None),
        basis_name=basis_name,
        include_linear_1=include_linear_1,
    )
    return Phi_train, reg_diag, analytics, ci


_LINEAR_RESIDUAL_FAMILIES = ('rbf', 'rff', 'nystrom')


def _residual_family(spec):
    """The residual family named by a ``residual``/``variance_residual`` spec.

    ``None`` → no spec; a bare string is the family; a dict follows the
    trainer's convention (``type``, defaulting to 'nn'); anything else is
    treated as 'nn' (conservative: routes to JAX).
    """
    if spec is None:
        return None
    if isinstance(spec, str):
        return spec
    if isinstance(spec, dict):
        return spec.get('type', 'nn')
    return 'nn'


def _resolve_array_backend(backend, *, mode, residual, config_kwargs,
                           precision=None):
    """Resolve the ``backend=`` argument to 'jax' or 'numpy'.

    'numpy' (the exact core) runs the SAME fit-path code on numpy/scipy —
    supported for everything except the JAX-native paths: the NN residual
    (``residual_nn`` / ``residual`` type 'nn'), the stage-laddering modes
    that may add it ('auto'/'full'), AND an explicit **float32** fit (the
    core is a float64 engine; float32 is a JAX-native speed mode — DEC-056).
    The linear residual families (Stage C 'rbf'/'rff'/'nystrom' and
    ``variance_residual``) are core-legal since BR-10/DEC-057 — same code on
    both backends. (RFF caveat: its random frequencies are drawn
    backend-natively, so an RFF fit is deterministic per backend but not
    comparable across backends.)
    'auto' picks numpy exactly when the config stays inside the supported
    surface; an explicit 'numpy' on an unsupported config raises instead of
    silently changing the model.
    """
    if backend is None:
        backend = 'auto'
    if backend not in VALID_BACKENDS + ('auto',):
        raise ValueError(
            f"backend must be one of {VALID_BACKENDS + ('auto',)}; "
            f"got {backend!r}")
    from . import precision as _precision
    wants_f32 = _precision.wants_float32_fit(precision)
    needs_jax = (
        mode in ('auto', 'full')
        or wants_f32
        or config_kwargs.get('residual_nn') is not None
        or any(fam is not None and fam not in _LINEAR_RESIDUAL_FAMILIES
               for fam in (_residual_family(residual),
                           _residual_family(config_kwargs.get('residual')),
                           _residual_family(
                               config_kwargs.get('variance_residual')))))
    if backend == 'auto':
        return 'jax' if needs_jax else 'numpy'
    if backend == 'numpy' and wants_f32:
        raise ValueError(
            "backend='numpy' is a float64-only exact core and does not support "
            "precision='float32' — float32 is a JAX speed mode. Use "
            "backend='jax' for a float32 fit, or backend='auto' (routes a "
            "float32 request to JAX automatically).")
    if backend == 'numpy' and needs_jax:
        raise ValueError(
            "backend='numpy' does not support the NN residual (residual_nn / "
            "residual type 'nn') or mode='auto'/'full' — those paths are "
            "JAX-native. The linear residual families ('rbf', 'rff', "
            "'nystrom') and variance_residual ARE supported on the core. Use "
            "backend='auto' (falls back to 'jax' for NN paths) or "
            "backend='jax'.")
    return backend


def hifi_anova(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: Optional[List[str]] = None,
    K1: int = 5,
    K2: int = 3,
    strategy: Optional[str] = None,
    mode: str = 'second',
    variable_selection: Optional[str] = _UNSET,
    residual: Optional[str] = None,
    heteroscedastic: bool = False,
    seed: int = 42,
    verbose: bool = True,
    precision: Optional[str] = None,
    progress=None,
    should_stop=None,
    backend: str = 'auto',
    **kwargs,
) -> HiFiResult:
    """One-call API — see ``_hifi_anova_impl`` for the full parameter list.

    ``backend``: array backend for the fit and the result's compute methods —
    'auto' (**default, DEC-056**: the float64 NumPy exact core whenever the
    config avoids the JAX-native NN paths — no per-shape XLA compilation,
    ~ms structural refits at interactive sizes; falls back to 'jax' for the
    NN residual and mode='auto'/'full'; the linear residual families
    'rbf'/'rff'/'nystrom' and ``variance_residual`` run on the core since
    BR-10/**DEC-057**), 'numpy' (force the exact core; raises on an
    unsupported config), or 'jax' (the historical backend — float32 by
    default, byte-identical to pre-DEC-056 results). The
    core is statistically identical to jax by construction (same code path)
    and float64-precise; the resolved value is recorded in
    ``result.config['array_backend']`` and scopes every ``HiFiResult`` compute
    method (predict/intervals/sigma_x2/loo/curves). Pass ``backend='jax'`` to
    reproduce the old default exactly.
    """
    resolved = _resolve_array_backend(backend, mode=mode, residual=residual,
                                      config_kwargs=kwargs, precision=precision)
    if resolved == 'numpy':
        # The NumPy exact core is a float64 engine (DEC-056) — an explicit
        # float32 already routed to JAX above, so here precision is float64 or
        # unspecified; pin float64 so the stored weights and predictions are
        # float64 (the JAX float32 *default* does not apply to the core).
        precision = 'float64'
    with use_array_backend(resolved):
        return _hifi_anova_impl(
            X, y, feature_names=feature_names, K1=K1, K2=K2,
            strategy=strategy, mode=mode,
            variable_selection=variable_selection, residual=residual,
            heteroscedastic=heteroscedastic, seed=seed, verbose=verbose,
            precision=precision, progress=progress, should_stop=should_stop,
            _array_backend=resolved, **kwargs)


def _hifi_anova_impl(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: Optional[List[str]] = None,
    K1: int = 5,
    K2: int = 3,
    strategy: Optional[str] = None,
    mode: str = 'second',
    variable_selection: Optional[str] = _UNSET,
    residual: Optional[str] = None,
    heteroscedastic: bool = False,
    seed: int = 42,
    verbose: bool = True,
    precision: Optional[str] = None,
    progress=None,
    should_stop=None,
    _array_backend: str = 'jax',
    **kwargs,
) -> HiFiResult:
    """One-call API: fit a HiFi-ANOVA model and return complete results.

    Takes raw data, handles preprocessing, fitting, Sobol analysis,
    confidence intervals, and diagnostics automatically.

    Args:
        X: (N, D) numeric feature matrix (original scale). Must be 2-D — reshape a
            single feature with X.reshape(-1, 1).
        y: (N,) numeric target values (an (N, 1) column is accepted; multi-output
            (N, k>1) is rejected — fit one model per output).
        feature_names: list of exactly D unique strings, or None to auto-name.
        K1: max harmonic for first-order (default 5; integer >= 1)
        K2: max harmonic for second-order (default 3; integer >= 0, 0 disables).
            May also be a per-pair mapping ``{(i, j): K2_ij}`` (X11C-S02 /
            BR-04): the mapping BOTH pins the exact retained pairs and gives
            each its own harmonic order (e.g. ``K2={(0, 1): 5, (2, 3): 2}``
            fits a fine x1×x2 surface and a coarse x3×x4 one). The pair set is
            then user-specified — pair selection/pruning heuristics and
            ``pair_selection``/``pair_candidates`` must not also be set, and
            third-order terms / mixed per-variable bases / mode='auto' are
            not supported with it. The scalar form is unchanged.
        strategy: regularization strategy. Default (None) resolves to
            'curvature' when heteroscedastic=True (it penalizes high-frequency
            wiggle and keeps the mean/variance alternating loop stable), and
            'variance' otherwise. Pass an explicit value to override.
        mode: model complexity ('first', 'second', 'full', 'heteroscedastic', 'auto')
        variable_selection: selection method ('bic', 'group_lasso', '1se', None).
            Defaults to 'bic'. Not supported on mixed per-variable bases
            (basis_per_variable=...): the implicit default is neutralized there
            with a warning, and an explicit non-None value raises (DEC-045).
        residual: residual type ('rbf', 'rff', 'nystrom', None)
        heteroscedastic: if True, fit a variance model (adds Stage D). Cannot be
            combined with mode='auto' (auto decides the stage ladder itself,
            including whether to add Stage D); that combination raises.
        seed: random seed
        verbose: print progress
        precision: fit-weight precision, 'float32' or 'float64'. Applies to the
            **JAX backend only** — the NumPy exact core (the default via
            backend='auto') is float64-only (DEC-056), so an explicit
            precision='float32' routes 'auto' to JAX (and is rejected by an
            explicit backend='numpy'). On the JAX path: controls the fit and
            stored weights; the post-fit analytics always run in float64;
            'float64' also enables JAX x64. Omitted (None) resolves by
            precedence: process-wide set_fit_precision() > HIFI_ANOVA_X64 env >
            'float32' default (the JAX default; DEC-035). Because the DEFAULT
            backend is the float64 core, a plain call's effective precision is
            float64. The *effective* value is recorded in result config and
            saved metadata. (See hifi_anova.precision; DEC-035/DEC-044/DEC-056.)
        progress: optional callable(event: dict) invoked at stage boundaries and
            the post-fit analysis phase, for GUI/back-end progress bars. Event
            schema in hifi_anova.progress. Default None = no events.
        should_stop: optional callable() -> bool, polled between stages and on
            each Stage-D outer iteration; returning True aborts the fit with
            hifi_anova.progress.HiFiCancelled. Default None = never cancelled.
        **kwargs: additional config overrides. Term-structure keys (X11C-S02):
            ``K2h`` (int, default 0) asks a heteroscedastic fit for a
            second-order variance model — ``log_variance_sobol['second_order']``
            is then populated per variance pair (BR-05);
            ``var_pair_selection`` additionally accepts an explicit
            ``[(i, j), ...]`` list of variance pairs;
            ``variable_orders={j: [2]}`` admits variable j to pair terms while
            excluding its first-order block (non-hierarchical model — the pair
            share is conditional on the omitted marginal; BR-06, experimental);
            ``variable_orders={j: []}`` excludes variable j from the MEAN
            model entirely while its column stays available to the VARIANCE
            model (a "variance-only" variable — heteroscedastic fits only;
            if x_j truly moves the mean, that misfit inflates the residuals
            the variance model sees, biasing sigma^2(x) — a user assertion,
            recorded in ``train_results['term_structure']['mean_excluded']``);
            ``variance_variables=[...]`` restricts the first-order variance
            model to a variable subset — excluding a variable ASSERTS
            homoscedastic noise along it (BR-01, experimental).

    Returns:
        HiFiResult with model, Sobol indices, CIs, diagnostics, and
        convenience methods for prediction and reporting.

    Note (process-global state / concurrency):
        This call enables JAX float64 process-wide
        (``jax.config.update("jax_enable_x64", True)``) because the post-fit
        analytics require it; the flag is not restored afterward. ``precision``
        and the module-level ``set_fit_precision`` / ``set_gpu_memory_limit``
        controls are likewise process-global. Consequently, concurrent fits at
        *different* precisions in one process are not isolated — run them
        sequentially, or in separate processes (e.g. a worker pool), if a GUI
        back-end must serve several at once. A single-session desktop GUI is
        unaffected. This does not make the fit itself thread-unsafe for a single
        run; it is the shared x64/precision flags that are not per-call.

    Example:
        result = hifi_anova(X, y, feature_names=['age', 'income'])
        result.summary()
        pred = result.predict(X_new)
        lo, hi = result.predict_intervals(X_new)
    """
    from .validation import require_int

    # --- Public data boundary (DEC-046) ---------------------------------------
    # Convert and shape-check X *before* unpacking ``X.shape`` (a 1-D array or a
    # Python list used to fail with an opaque "not enough values to unpack" /
    # "'list' object has no attribute 'shape'"). Deep numeric checks (NaN/inf,
    # sample sufficiency, constant target) stay in ``preprocess_data`` so its
    # specific diagnostics are preserved — this layer only makes the *shape* and
    # *dtype* boundary honest.
    X = np.asarray(X)
    if X.ndim == 0:
        raise ValueError(
            "X must be a 2-D (N, D) feature matrix; got a scalar. "
            "Reshape a single sample/feature with np.reshape(X, (n, d)).")
    if X.ndim == 1:
        raise ValueError(
            f"X must be 2-D (N, D); got a 1-D array of length {X.shape[0]}. "
            "Reshape a single feature with X.reshape(-1, 1).")
    if X.ndim != 2:
        raise ValueError(
            f"X must be 2-D (N, D); got a {X.ndim}-D array with shape {X.shape}.")
    if not np.issubdtype(X.dtype, np.number):
        raise ValueError(
            f"X must be a numeric feature matrix; got dtype {X.dtype}. Encode "
            "non-numeric columns (e.g. one-hot categoricals) before calling.")

    y = np.asarray(y)
    if y.ndim == 2 and y.shape[1] == 1:
        y = y.reshape(-1)   # a column vector (N, 1) is a supported equivalent
    if y.ndim != 1:
        raise ValueError(
            f"y must be a 1-D target array (or an (N, 1) column); got shape "
            f"{y.shape}. HiFi-ANOVA fits a scalar target — for multi-output, "
            "fit one model per output column.")
    if not np.issubdtype(y.dtype, np.number):
        raise ValueError(
            f"y must be a numeric target array; got dtype {y.dtype}.")

    jax.config.update("jax_enable_x64", True)

    from .data.preprocessing import preprocess_data
    from .training.trainer import HiFiANOVATrainer
    from .analysis.sobol import compute_sobol_indices
    from .analysis.automl import ridge_analytics, sobol_confidence_intervals

    N, D = X.shape

    # feature_names: exactly D unique strings, or None to auto-generate. These
    # used to be silently accepted at the wrong length / non-string / duplicated,
    # then either truncated (masking a caller mistake) or breaking name lookups.
    if feature_names is None:
        feature_names = [f'x{i+1}' for i in range(D)]
    else:
        if isinstance(feature_names, str) or not isinstance(
                feature_names, (list, tuple)):
            raise ValueError(
                f"feature_names must be a list of {D} strings (or None); got "
                f"{type(feature_names).__name__} ({feature_names!r}).")
        feature_names = list(feature_names)
        if len(feature_names) != D:
            raise ValueError(
                f"feature_names has {len(feature_names)} name(s) but X has {D} "
                f"feature(s); provide exactly {D} names (or None to auto-name).")
        non_str = [n for n in feature_names if not isinstance(n, str)]
        if non_str:
            raise ValueError(
                f"feature_names must all be strings; got non-string value(s) "
                f"{non_str}.")
        if len(set(feature_names)) != len(feature_names):
            dupes = sorted({n for n in feature_names
                            if feature_names.count(n) > 1})
            raise ValueError(
                f"feature_names contains duplicate(s) {dupes}; names must be "
                "unique (Sobol tables and component_curve() look up by name).")

    require_int('seed', seed, minimum=0)

    valid_modes = {'first', 'second', 'full', 'heteroscedastic', 'auto'}
    if mode not in valid_modes:
        raise ValueError(
            f"Unknown mode {mode!r}. Choose from: {sorted(valid_modes)}")

    # mode='auto' and heteroscedastic=True are contradictory (DEC-046): 'auto'
    # decides the whole stage ladder from the data — INCLUDING whether to add the
    # Stage-D variance model — while heteroscedastic=True unconditionally forces
    # Stage D. Silently, 'auto' won and the boolean was ignored. Reject the
    # ambiguous combination rather than pick a winner (this changes no default).
    if mode == 'auto' and heteroscedastic:
        raise ValueError(
            "mode='auto' and heteroscedastic=True are ambiguous together: "
            "'auto' decides the stage ladder from the data (and adds the Stage-D "
            "variance model itself when the residual is input-dependent), while "
            "heteroscedastic=True forces Stage D unconditionally. Choose one — "
            "mode='auto' alone (let auto decide), or heteroscedastic=True with "
            "an explicit mode (e.g. mode='heteroscedastic') or stages=[...].")

    # Resolve variable_selection, distinguishing an *omitted* argument (defaults
    # to 'bic') from an *explicit* request (DEC-045). Mixed per-variable bases do
    # not support variable selection: neutralize the *implicit* default there
    # (warn once, record selection_applied=False) so the fit is not silently
    # different from what was asked; an *explicit* value is left in config and the
    # trainer raises a capability error.
    _vs_explicit = variable_selection is not _UNSET
    variable_selection = variable_selection if _vs_explicit else 'bic'
    # User-defined term structure (X11C-S02): a per-pair K2 mapping names the
    # exact pair set itself, and a variable_orders first-order exclusion makes
    # data-driven selection unable to see the excluded variable — in both
    # cases the implicit variable_selection default is meaningless, so it is
    # neutralized (silently: nothing the caller asked for changes; an
    # EXPLICIT variable_selection still reaches the trainer, which raises).
    _term_structure = (isinstance(K2, dict)
                       or kwargs.get('variable_orders') is not None)
    if _term_structure and not _vs_explicit:
        variable_selection = None
    _mixed = kwargs.get('basis_per_variable') is not None
    _mixed_selection_neutralized = False
    if _mixed and not _vs_explicit and variable_selection:
        warnings.warn(
            "variable_selection defaults to 'bic', but variable selection is "
            "not supported on mixed per-variable bases (basis_per_variable=...); "
            "the implicit default is neutralized for this fit (no selection "
            "applied). Pass variable_selection=None to silence this, or use a "
            "uniform basis to run selection. This one-release migration will "
            "become an error in a future release.",
            UserWarning, stacklevel=2)
        variable_selection = None
        _mixed_selection_neutralized = True

    # Mixed per-variable bases fit in the trainer AND now drive the one-call
    # post-fit analytics through the fitted-design record: ``_fit_mixed``
    # surfaces a ``build_mixed_record`` with per-group column slices and Grams
    # (variable per order), and ``sobol_confidence_intervals`` consumes them via
    # its block-driven ``groups=`` path — so the Sobol CIs are block-correct
    # instead of crashing on the old uniform rebuild (e.g. the historical
    # ``operands could not be broadcast together with shapes (6,6) (10,10)``).
    # See build_mixed_record / DEC-030.

    # Resolve the default regularization strategy. Heteroscedastic fits run an
    # alternating mean/variance loop; since DEC-028 (leverage-corrected
    # variance solve + held-out iterate selection) it is stable under both
    # penalties, but 'curvature' (which damps high-frequency components)
    # remains the hetero default — it is what the flagship heteroscedastic
    # example uses, and reverting to 'variance' is an advisor-gated question
    # (see the internal Stage-D stability design note). An explicit `strategy=` always wins.
    if strategy is None:
        strategy = ('curvature'
                    if (heteroscedastic or mode == 'heteroscedastic')
                    else 'variance')

    # Build config
    config = {
        'K1': K1, 'K2': K2,
        'strategy': strategy,
        'lambda_order1': 0.001,
        'lambda_order2': 0.01,
        # array backend the fit runs on (provenance; scopes the result's
        # compute methods via _in_fit_backend)
        'array_backend': _array_backend,
    }

    if mode == 'auto':
        config['mode'] = 'auto'
        config['auto_threshold'] = kwargs.pop('auto_threshold', 0.01)
    elif heteroscedastic or mode == 'heteroscedastic':
        config['stages'] = ['A', 'B', 'C', 'D'] if residual else ['A', 'B', 'D']
        config['Kh'] = kwargs.pop('Kh', 3)
        config['lambda_h'] = kwargs.pop('lambda_h', 0.1)
    else:
        stage_map = {'first': ['A'], 'second': ['A', 'B'], 'full': ['A', 'B', 'C']}
        stages = stage_map.get(mode, ['A', 'B'])
        if residual:
            stages = list(set(stages) | {'C'})
            stages.sort()
        config['stages'] = stages

    if variable_selection:
        config['variable_selection'] = variable_selection
        config['pair_candidates'] = kwargs.pop('pair_candidates', 'either')
    if _mixed_selection_neutralized:
        # Machine-readable marker consumed by the trainer's mixed-capability
        # metadata (results['mixed_capability']['implicit_selection_neutralized']).
        config['_mixed_selection_neutralized'] = True

    if residual:
        res_config = {'type': residual, 'lambda_residual': kwargs.pop('lambda_residual', 1.0)}
        if residual == 'rbf':
            res_config.update({'n_centers': kwargs.pop('n_centers', min(300, N // 5)),
                               'sigma': kwargs.pop('sigma', 0.2)})
        elif residual == 'rff':
            res_config.update({'n_features': kwargs.pop('n_features', 1000),
                               'gamma': kwargs.pop('gamma', 3.0)})
        elif residual == 'nystrom':
            res_config.update({'n_inducing': kwargs.pop('n_inducing', min(300, N // 5)),
                               'kernel': kwargs.pop('kernel', 'matern52'),
                               'lengthscale': kwargs.pop('lengthscale', 0.2)})
        config['residual'] = res_config

    config.update(kwargs)  # any remaining overrides
    # Forward verbosity so verbose=False silences the trainer's stage prints,
    # not just the final summary (an explicit kwargs['verbose'] still wins).
    config.setdefault('verbose', verbose)
    # Fit precision (DEC-035, precedence fixed in DEC-044): float32 by default,
    # opt-in float64. Resolve the *effective* precision ONCE at the public
    # boundary — an omitted precision (None) obeys the documented precedence
    # (explicit arg > set_fit_precision override > HIFI_ANOVA_X64 env > float32),
    # rather than the old hard-coded 'float32' default that silently overrode the
    # global/env controls. Store the resolved string in config (so result config
    # and saved metadata record what was actually used) and pass the matching
    # dtype to preprocessing and the trainer, so inputs and every model weight
    # share one dtype. x64 stays enabled above regardless (analytics need it).
    from .precision import (fit_dtype as _resolve_fit_dtype,
                            resolve_precision as _resolve_precision)
    effective_precision = _resolve_precision(precision)
    config['precision'] = effective_precision
    _fit_dtype = _resolve_fit_dtype(effective_precision)

    # Preprocess
    data = preprocess_data(X, y, seed=seed, fit_dtype=_fit_dtype)

    # Fit
    trainer = HiFiANOVATrainer(config, progress=progress, should_stop=should_stop)
    key = jax.random.PRNGKey(seed)
    model, train_results = trainer.fit(
        data['x_train'], data['y_train'],
        data['x_val'], data['y_val'],
        key=key,
    )

    # Post-fit analysis phase (Sobol spectrum, CIs, diagnostics) — a single
    # coarse progress event; the trainer emitted the per-stage events above.
    if progress is not None:
        from .progress import make_event
        progress(make_event('phase', stage='analysis',
                            message='computing Sobol indices, intervals, diagnostics'))

    # Headline mean Sobol indices, evaluated on the fitted model.
    sobol = compute_sobol_indices(model, data['x_test'])

    # Analytics (σ̂/df/LOO) and Sobol CIs from the fitted-design record — the
    # design the trainer actually solved — so the reported diagnostics describe
    # the model the user gets back, with the *real* per-order penalties and the
    # third-order variance in the Sobol denominator (see the fitted-design record).
    # The record co-sources the CI's point estimate and covariance from one fit,
    # removing the point/CI incoherence of the old independent rebuild. Falls back
    # to the legacy rebuild only if no record was produced (e.g. a return site not
    # yet wired); the homoscedastic ≤2nd-order path is bit-for-bit identical either
    # way (tests/test_fitted_design_invariant.py).
    record = train_results.get('fitted_design')
    ci_efficient = None   # efficient (precision-weighted) index set; hetero only
    gap = None            # heteroscedasticity × misspecification gap; hetero only
    if record is not None:
        from .training.fitted_design import MEAN_INTERCEPT_PROFILED_JOINT_GLS
        Phi_train = record.Phi
        reg_diag = record.reg_diag
        # Predictive diagnostics from the (weighted, for Stage D) fit the model
        # was solved with; sample_weights is None on the homoscedastic path, so
        # this is the unweighted diagnostics there. For the profiled joint-GLS
        # Stage-D mean, the analytics profile the SAME unpenalized intercept as the
        # shipped fit (augmented design [1,Φ], Remark rem:intercept), so df /
        # leverage / residual df / LOO / covariance re-profile f0 under
        # perturbation/deletion instead of holding the fitted intercept fixed.
        # Pass the uncentered response so the augmented f0 is recovered; slopes,
        # residuals and predictions are unchanged. Legacy fixed-intercept and
        # unweighted-centered means keep the feature-only instrument that matches
        # how they were actually solved.
        _profiled = (record.mean_intercept_mode
                     == MEAN_INTERCEPT_PROFILED_JOINT_GLS)
        _y_analytics = ((record.y_centered + record.f0) if _profiled
                        else record.y_centered)
        analytics = ridge_analytics(Phi_train, _y_analytics, reg_diag,
                                    weights=record.sample_weights,
                                    profile_intercept=_profiled)
        # Attribution (Sobol point + CI) from the unit-weight companion — the
        # HC3 sandwich is already heteroscedasticity-robust, so it is NOT
        # reweighted (Theorem projection Part ii). Homoscedastic ⇒ the companion
        # is the record itself.
        attr = record.attribution_record()
        # Structural fidelity 𝔉 (M3/DEC-032), computed once in compute_sobol_indices
        # (single source of truth). Thread it into the interpretable CI so the same
        # 𝔉 scales the core shares into the reported total shares; pass None when no
        # residual stage ran (𝔉≡1) to keep the CI dict byte-identical there.
        _fid = sobol.get('fidelity', {}) if isinstance(sobol, dict) else {}
        _has_residual = _fid.get('var_residual', 0.0) > 0.0
        ci = sobol_confidence_intervals(
            attr.Phi, attr.y_centered, attr.reg_diag,
            fidelity=(_fid.get('value', 1.0) if _has_residual else None),
            **attr.sobol_ci_kwargs())
        # Two-fit reporting surface (M2, DEC-030; Manuscript_Theoryv06 Theorem
        # projection Part ii): for a genuinely weighted (Stage-D) fit also
        # compute the *efficient* (precision-weighted) closed-form index set on
        # the SAME retained blocks, with the weighted sandwich CI, and report
        # the observed efficient−interpretable gap as a "heteroscedasticity ×
        # misspecification" diagnostic. Homoscedastic ⇒ the two fits coincide,
        # so both stay None and the reported surface is byte-identical.
        if record.is_weighted:
            ci_efficient = sobol_confidence_intervals(
                record.Phi, _y_analytics, record.reg_diag,
                weights=record.sample_weights,
                profile_intercept=_profiled,
                **record.sobol_ci_kwargs())
            gap = _two_fit_gap(ci, ci_efficient, feature_names)
    else:
        Phi_train, reg_diag, analytics, ci = _legacy_analytics_rebuild(
            model, data, config, D, K1, K2, strategy)

    # Reported LOO (M1, DEC-031). ``analytics`` carries the Tier-I / homoscedastic
    # loo_nll & loo_cv; for a Stage-D fit with a surfaced variance sub-problem the
    # one-call UPGRADES to the manuscript's default Tier II (one-step variance
    # jackknife) and attaches the KKT variance-floor / H_h-conditioning flags.
    from .analysis.automl import joint_loo
    loo_nll = analytics['loo_nll']
    loo_cv = analytics['loo_cv']
    loo_tier = analytics['loo_tier']
    loo_guarantee = None
    loo_floor_active = None
    loo_ill_conditioned = None
    if (record is not None and record.is_weighted
            and record.variance is not None):
        j = joint_loo(analytics, record.variance)
        loo_nll, loo_cv, loo_tier = j['loo_nll'], j['loo_cv'], j['loo_tier']
        loo_guarantee = j['loo_tier2_guarantee_holds']
        loo_floor_active = j['loo_variance_floor_active']
        loo_ill_conditioned = j['variance_hessian_ill_conditioned']

    # Build named Sobol CI dict (interpretable / unit-weight — the headline
    # attribution).
    sobol_ci_named = {}
    for i, (S, lo, hi) in ci['first_order'].items():
        name = feature_names[i] if i < len(feature_names) else f'x{i+1}'
        sobol_ci_named[name] = (S, lo, hi)

    sobol_ci_status_named = {}
    for i, status in ci.get('component_status', {}).get('first_order', {}).items():
        name = feature_names[i] if i < len(feature_names) else f'x{i+1}'
        sobol_ci_status_named[name] = status

    inference_metadata = _fixed_configuration_inference_metadata(
        config, train_results)
    # ``HiFiResult.save`` persists train_results, so carrying the same JSON-safe
    # object here gives load_model()['results']['inference_metadata'] a stable
    # round trip without redesigning the persistence schema.
    train_results['inference_metadata'] = inference_metadata
    train_results['sobol_ci_status'] = sobol_ci_status_named

    # Total-variance first-order shares Ŝ^total = 𝔉·Ŝ^core with CIs (M3/DEC-032),
    # reported beside the core headline only when a residual stage ran (𝔉<1). A
    # homoscedastic/no-residual fit has 𝔉≡1 and collapses to the single ``sobol_ci``
    # set (``sobol_ci_total`` stays None), mirroring ``sobol_ci_efficient``.
    sobol_ci_total_named = None
    if isinstance(ci, dict) and 'total' in ci:
        sobol_ci_total_named = {}
        for i, (S, lo, hi) in ci['total']['first_order'].items():
            name = feature_names[i] if i < len(feature_names) else f'x{i+1}'
            sobol_ci_total_named[name] = (S, lo, hi)

    # Efficient (precision-weighted) first-order named CI dict — only for a
    # heteroscedastic fit; None (and identical to the interpretable set) otherwise.
    sobol_ci_efficient_named = None
    if ci_efficient is not None:
        sobol_ci_efficient_named = {}
        for i, (S, lo, hi) in ci_efficient['first_order'].items():
            name = feature_names[i] if i < len(feature_names) else f'x{i+1}'
            sobol_ci_efficient_named[name] = (S, lo, hi)

    # R^2. Two conventions (see hifi_anova.analysis.metrics): the framework-native
    # explained-variance score 1 − Var(y−ŷ)/Var(y) — the number the manuscript
    # reports — is kept as ``r_squared``; the textbook SSE/TSS coefficient of
    # determination (sklearn ``r2_score``, penalizes bias, can go negative) is
    # reported alongside as ``r_squared_classical``. They coincide unless the
    # test-set residual has a nonzero mean. The explained-variance line below is
    # kept verbatim (byte-identical numerics — golden master captures it).
    pred_test = model.predict_mean_only(data['x_test'])
    var_y = float(jnp.var(data['y_test']))
    var_r = float(jnp.var(data['y_test'] - pred_test))
    r2 = 1.0 - var_r / var_y if var_y > 0 else 0.0
    from .analysis.metrics import r_squared as _r_squared
    r2_classical = _r_squared(data['y_test'], pred_test, 'classical')

    # Opt-in share of OBSERVED output variance V_u/Var(Y) (Q4/DEC-034). Scale the
    # fitted-variance shares (or the core shares when no residual ran — they coincide
    # at 𝔉≡1) by Var(f̂)/Var(Y), treated as FIXED (same conditional logic as the total
    # CI). Var(f̂)/Var(Y) is NOT R² for a regularized/nonlinear fit — computing it here
    # is what keeps the number correct. Always populated (cheap); printed only via
    # summary(observed=True), so the default surface is unchanged.
    var_fhat = float(jnp.var(pred_test))
    _obs_scale = (var_fhat / var_y) if var_y > 0 else 0.0
    _obs_source = (sobol_ci_total_named if sobol_ci_total_named is not None
                   else sobol_ci_named)
    sobol_ci_observed_named = {
        name: (S * _obs_scale, lo * _obs_scale, hi * _obs_scale)
        for name, (S, lo, hi) in _obs_source.items()
    }

    result = HiFiResult(
        model=model,
        config=config,
        feature_names=feature_names,
        transformer=data['transformer'],
        y_mean=data['y_mean'],
        y_std=data['y_std'],
        train_results=train_results,
        sobol=sobol,
        sobol_ci=sobol_ci_named,
        sobol_ci_efficient=sobol_ci_efficient_named,
        sobol_gap=gap,
        fidelity=(sobol.get('fidelity') if isinstance(sobol, dict) else None),
        sobol_ci_total=sobol_ci_total_named,
        sobol_ci_observed=sobol_ci_observed_named,
        sigma_hat=analytics['sigma_hat'],
        r_squared=r2,
        r_squared_classical=r2_classical,
        loo_cv=loo_cv,
        df=analytics['df'],
        df_residual=float(analytics['df_residual']),
        inference_metadata=inference_metadata,
        sobol_ci_status=sobol_ci_status_named,
        noise_scale_is_calibration=bool(
            analytics.get('noise_scale_is_calibration', False)),
        loo_nll=loo_nll,
        loo_tier=loo_tier,
        loo_tier2_guarantee_holds=loo_guarantee,
        loo_variance_floor_active=loo_floor_active,
        variance_hessian_ill_conditioned=loo_ill_conditioned,
        _Phi_train=Phi_train,
        _reg_diag=reg_diag,
        _sample_weights=(record.sample_weights if record is not None else None),
        _data=data,
        _fitted_design=record,
    )

    if verbose:
        result.summary()

    if progress is not None:
        from .progress import make_event
        progress(make_event('done', message='fit complete', fraction=1.0))

    return result
