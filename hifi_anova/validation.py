"""Shared, declarative input-validation helpers (DEC-046).

Small, pure, testable checks used by the public boundary (:func:`hifi_anova`)
and the trainer (:class:`HiFiANOVATrainer`) so invalid or contradictory public
configuration fails *early* with a specific, actionable message instead of
either being silently accepted (a footgun) or blowing up obscurely deep inside
the ridge solve.

Design rules (kept deliberately narrow — this is a validation layer, not a typed
config rewrite):

* Every message names the full option path, the received value/type, and the
  valid alternatives or range.
* All checks raise :class:`ValueError` — the convention already used by the
  trainer's key allowlist and the residual-type / preprocessing checks — so
  callers (and tests) pin one exception type for bad config.
* ``bool`` is *not* an integer here: ``True``/``False`` (and ``numpy.bool_``)
  are rejected wherever an integer or number is required, because ``K1=True``
  silently meaning ``K1=1`` is exactly the kind of accident this layer exists to
  stop.
* No scientifically arbitrary upper bounds — only domains that are actually
  meaningful (non-negative counts, finite penalties, fractions in ``(0, 1)``)
  are enforced. Resource preflight is deferred.
"""

import difflib

import numpy as np

__all__ = [
    "require_int",
    "require_number",
    "require_fraction",
    "require_bool",
    "require_choice",
    "require_mapping_keys",
    "suggest",
    "validate_config",
    "validate_basis_per_variable",
    "validate_residual_spec",
    "validate_pair_selection",
    "validate_stages",
    "STAGES",
    "BASIS_FAMILIES",
    "RESIDUAL_TYPES",
]

# --- Recognized enum values (single source of truth for validation) -----------
STAGES = ("A", "B", "C", "D")
BASIS_FAMILIES = ("fourier", "legendre", "haar")
_BASIS_TYPES = ("full", "spectral_higher", "spectral_all")
_PRECISIONS = ("float32", "float64")
RESIDUAL_TYPES = ("nn", "rbf", "rff", "nystrom")
_VARIANCE_RESIDUAL_TYPES = ("rbf", "rff", "nystrom")
_SELECTION_METHODS = ("bic", "group_lasso", "1se")
_PRUNING_METHODS = ("none", "bic", "group_lasso", "1se")
# Pair/triple candidate + legacy selection heuristics (consumed by PairManager /
# _resolve_pair_manager / the variance pair/triple managers). Kept in sync with
# those call sites; an unrecognized value used to silently fall back to "all".
_PAIR_CANDIDATE_MODES = ("all", "both", "either")            # PairManager mode
_PAIR_SELECTION_MODES = ("all", "both", "either", "auto",
                         "bic", "group_lasso", "1se")        # legacy one-shot
_TRIPLE_SELECTION_MODES = ("all", "all_active", "two_active", "one_active",
                           "bic", "group_lasso", "1se")
_VAR_PAIR_SELECTION_MODES = ("all", "auto")
_VAR_TRIPLE_SELECTION_MODES = ("all",)
# Stage-D estimator selector (P0-2). Kept in sync with
# ``trainer._STAGE_D_ESTIMATOR_FLAGS``.
_STAGE_D_ESTIMATORS = ("adjusted_quasi_likelihood", "raw_likelihood")
# ``strategy`` is a fixed name, a ``sobolev[_<num>]``/``spectral[_<num>]`` family,
# or a per-order dict keyed by the orders the builder actually consumes.
_STRATEGY_NAMES = ("uniform", "variance", "smoothness", "curvature")
_STRATEGY_PREFIXES = ("sobolev", "spectral")
_STRATEGY_DICT_KEYS = ("default", "order1", "order2", "order3")

# Boolean switches — a non-bool (e.g. 0/1) here is almost always a mistake.
_BOOL_KEYS = (
    "heteroscedastic_guard", "leverage_correction", "alternating_early_stop",
    "stage_d_joint_gls_mean", "variance_selection_mean_consistent",
    "variance_selection_mean_fallback", "verbose",
    "include_linear_1", "include_linear_2", "include_linear_3",
    "include_linear_h1", "include_linear_h2", "include_linear_h3",
)

# Per-family allowed nested keys for a ``residual`` / ``variance_residual`` spec.
# ``type`` is common; ``enabled`` and ``lambda_residual`` are Stage-C
# (``residual``) concepts only — ``variance_residual`` uses the top-level
# ``lambda_h_residual`` and has no enable flag — and ``lambda_residual`` is
# ignored by the NN family (which uses ``weight_decay``). These context rules are
# applied in :func:`validate_residual_spec`. Family keys mirror the constructor
# signatures in ``analytic_residual.create_residual`` / ``residual_net`` (so
# operational options like ``center_method`` and ``signal_variance`` are kept).
_RESIDUAL_FAMILY_KEYS = {
    "rbf": {"n_centers", "sigma", "center_method"},
    "rff": {"n_features", "gamma"},
    "nystrom": {"n_inducing", "lengthscale", "kernel", "signal_variance"},
    "nn": {"hidden_dims", "lr", "weight_decay",
           "epochs", "batch_size", "patience"},
}
_RESIDUAL_ANALYTIC = ("rbf", "rff", "nystrom")
_CENTER_METHODS = ("kmeans", "random")
_NYSTROM_KERNELS = ("rbf", "matern32", "matern52")

# NumPy scalar aliases so ``numpy`` inputs (e.g. ``np.int64(5)``) are accepted
# as the corresponding Python kind, while ``numpy`` booleans are still rejected
# where an integer/number is required.
_INT_TYPES = (int, np.integer)
_FLOAT_TYPES = (float, np.floating)
_BOOL_TYPES = (bool, np.bool_)


def _typename(value):
    """Readable type name for error messages (``bool``/``int``/``float``/…)."""
    return type(value).__name__


def suggest(name, choices):
    """Return ``" (did you mean 'x'?)"`` for the nearest choice, else ``""``.

    Deterministic (``difflib.get_close_matches`` is order-stable given a fixed
    ``choices`` iterable). ``choices`` may hold non-strings; only string
    candidates are considered.
    """
    str_choices = [c for c in choices if isinstance(c, str)]
    if not isinstance(name, str):
        return ""
    close = difflib.get_close_matches(name, str_choices, n=1)
    return f" (did you mean {close[0]!r}?)" if close else ""


def require_int(name, value, *, minimum=None, maximum=None, allow_bool=False):
    """Validate that ``value`` is an integer in ``[minimum, maximum]``; return it.

    Accepts Python ``int`` and ``numpy`` integer scalars. Rejects ``bool``/
    ``numpy.bool_`` (unless ``allow_bool``) and floats — including integral
    floats like ``5.0`` — because those are almost always a caller mistake and
    otherwise fail obscurely (``'float' object cannot be interpreted as an
    integer``) far downstream.
    """
    if isinstance(value, _BOOL_TYPES) and not allow_bool:
        raise ValueError(
            f"{name} must be an integer, not a bool (got {value!r}); "
            f"True/False are not accepted as 0/1 here."
        )
    if not isinstance(value, _INT_TYPES):
        raise ValueError(
            f"{name} must be an integer; got {_typename(value)} ({value!r})."
            + (f" Pass {name}={int(value)}." if isinstance(value, _FLOAT_TYPES)
               and float(value).is_integer() else "")
        )
    ivalue = int(value)
    if minimum is not None and ivalue < minimum:
        raise ValueError(
            f"{name} must be >= {minimum}; got {ivalue}.")
    if maximum is not None and ivalue > maximum:
        raise ValueError(
            f"{name} must be <= {maximum}; got {ivalue}.")
    return ivalue


def require_number(name, value, *, minimum=None, maximum=None,
                   strict_min=False, strict_max=False, allow_none=False):
    """Validate that ``value`` is a finite real number in range; return float.

    Accepts Python ``int``/``float`` and ``numpy`` real scalars; rejects
    ``bool``, ``NaN``, and ``+/-inf`` (a non-finite penalty/tolerance silently
    breaks the solve). ``strict_min``/``strict_max`` make the corresponding
    bound exclusive. ``allow_none`` passes ``None`` through unchanged (for
    optional numeric config).
    """
    if value is None and allow_none:
        return None
    if isinstance(value, _BOOL_TYPES):
        raise ValueError(
            f"{name} must be a number, not a bool (got {value!r}).")
    if not isinstance(value, _INT_TYPES + _FLOAT_TYPES):
        raise ValueError(
            f"{name} must be a number; got {_typename(value)} ({value!r}).")
    fvalue = float(value)
    if not np.isfinite(fvalue):
        raise ValueError(
            f"{name} must be finite; got {fvalue!r}.")
    if minimum is not None:
        if strict_min and fvalue <= minimum:
            raise ValueError(f"{name} must be > {minimum}; got {fvalue}.")
        if not strict_min and fvalue < minimum:
            raise ValueError(f"{name} must be >= {minimum}; got {fvalue}.")
    if maximum is not None:
        if strict_max and fvalue >= maximum:
            raise ValueError(f"{name} must be < {maximum}; got {fvalue}.")
        if not strict_max and fvalue > maximum:
            raise ValueError(f"{name} must be <= {maximum}; got {fvalue}.")
    return fvalue


def require_fraction(name, value, *, open_lo=True, open_hi=True):
    """Validate a fraction in ``(0, 1)`` (bounds exclusive by default)."""
    return require_number(name, value, minimum=0.0, maximum=1.0,
                          strict_min=open_lo, strict_max=open_hi)


def require_bool(name, value):
    """Validate that ``value`` is a boolean (Python ``bool`` or ``numpy.bool_``).

    Rejects ``0``/``1`` and other truthy values: a boolean *switch* silently
    accepting an int hides the same class of caller mistake as ``K1=True``.
    """
    if not isinstance(value, _BOOL_TYPES):
        raise ValueError(
            f"{name} must be a bool (True/False); got {_typename(value)} "
            f"({value!r}).")
    return bool(value)


def require_choice(name, value, choices):
    """Validate that ``value`` is one of ``choices``; suggest the nearest typo.

    ``choices`` is any iterable of allowed values (usually strings). Returns the
    value unchanged when valid.
    """
    choices = list(choices)
    if value in choices:
        return value
    raise ValueError(
        f"{name} must be one of {sorted(c for c in choices if c is not None)}; "
        f"got {value!r}{suggest(value, choices)}."
    )


def require_mapping_keys(name, mapping, allowed, *, required=()):
    """Reject unknown keys in a nested spec dict, suggesting the nearest key.

    ``name`` is the full path of the mapping (e.g. ``"residual"`` or
    ``"basis_per_variable[0]"``); ``allowed`` is the iterable of recognized
    keys; ``required`` lists keys that must be present. Raises on the first
    problem (unknown key with a typo hint, or a missing required key).
    """
    if not isinstance(mapping, dict):
        raise ValueError(
            f"{name} must be a mapping; got {_typename(mapping)} ({mapping!r}).")
    allowed = set(allowed)
    unknown = [k for k in mapping if k not in allowed]
    if unknown:
        hints = ", ".join(
            f"{k!r}{suggest(k, allowed)}" for k in unknown)
        raise ValueError(
            f"Unknown key(s) in {name}: {hints}. "
            f"Recognized keys: {sorted(allowed)}."
        )
    missing = [k for k in required if k not in mapping]
    if missing:
        raise ValueError(
            f"{name} is missing required key(s): {sorted(missing)}. "
            f"Recognized keys: {sorted(allowed)}."
        )


# =============================================================================
# Composite validators (stages, basis-per-variable, residual specs, config)
# =============================================================================

def validate_stages(stages):
    """Validate an explicit ``stages`` list: letters, order, dependencies.

    Rules (a silent no-op otherwise): every element is one of ``A/B/C/D``; no
    duplicates; Stage ``A`` is present (every path builds on the first-order
    mean); the list is in canonical ``A<B<C<D`` order (the trainer runs stages
    in a fixed order — an out-of-order list used to be silently reordered); and
    the list is non-empty. Returns the list unchanged when valid.
    """
    if isinstance(stages, str) or not isinstance(stages, (list, tuple)):
        raise ValueError(
            f"stages must be a list of stage letters (subset of "
            f"{list(STAGES)}); got {_typename(stages)} ({stages!r}).")
    if len(stages) == 0:
        raise ValueError(
            f"stages is empty; provide a non-empty subset of {list(STAGES)} "
            f"including 'A' (e.g. ['A', 'B']).")
    bad = [s for s in stages if s not in STAGES]
    if bad:
        hint = "".join(suggest(b, STAGES) for b in bad if isinstance(b, str))
        raise ValueError(
            f"stages contains unknown stage(s) {bad}; valid stages are "
            f"{list(STAGES)}.{hint}")
    if len(set(stages)) != len(stages):
        dupes = sorted({s for s in stages if list(stages).count(s) > 1})
        raise ValueError(
            f"stages contains duplicate(s) {dupes}; each stage may appear at "
            f"most once (got {list(stages)}).")
    if "A" not in stages:
        raise ValueError(
            f"stages must include 'A' (the first-order mean every later stage "
            f"builds on); got {list(stages)}.")
    ordered = sorted(stages, key=STAGES.index)
    if list(stages) != ordered:
        raise ValueError(
            f"stages must be in canonical order {list(STAGES)}; got "
            f"{list(stages)} — reorder to {ordered}.")
    return list(stages)


def _validate_residual_values(name, spec):
    """Range-check the family-specific residual values (counts/scales/epochs/…).

    A recognized nested key with a nonsensical value (``n_centers=0``,
    ``sigma=-1``, ``epochs=0``) used to sail through and break — or silently
    degrade — the fit. Validate each against its meaningful domain.
    """
    for key, val in spec.items():
        path = f"{name}['{key}']"
        if key in ("n_centers", "n_inducing", "n_features"):
            require_int(path, val, minimum=1)
        elif key in ("epochs", "batch_size", "patience"):
            require_int(path, val, minimum=1)
        elif key in ("sigma", "gamma", "lengthscale", "signal_variance", "lr"):
            require_number(path, val, minimum=0.0, strict_min=True)
        elif key in ("lambda_residual", "weight_decay"):
            require_number(path, val, minimum=0.0)
        elif key == "enabled":
            require_bool(path, val)
        elif key == "center_method":
            require_choice(path, val, _CENTER_METHODS)
        elif key == "kernel":
            require_choice(path, val, _NYSTROM_KERNELS)
        elif key == "hidden_dims":
            if isinstance(val, (str, bytes)) or not isinstance(val, (list, tuple)) \
                    or len(val) == 0:
                raise ValueError(
                    f"{path} must be a non-empty list of positive layer widths; "
                    f"got {_typename(val)} ({val!r}).")
            for w in val:
                require_int(f"{path} width", w, minimum=1)


def validate_residual_spec(name, spec, *, allow_nn=True, context="residual"):
    """Validate a ``residual`` / ``variance_residual`` nested spec.

    Accepts a bare string (shorthand for ``{'type': ...}``) or a mapping, and
    **returns a normalized dict** with ``type`` present (so the caller can store
    the normalized form and never hand a bare string to ``.get()`` downstream).
    Checks the ``type`` against the recognized families, rejects unknown nested
    keys (a misspelled ``n_centrs`` used to silently fall back to the default),
    and range-checks the values. ``allow_nn=False`` for ``variance_residual``
    (no NN variance residual exists). ``context='variance_residual'`` drops the
    Stage-C-only ``enabled`` / ``lambda_residual`` keys (variance residuals use
    the top-level ``lambda_h_residual`` and are never disabled via a flag).
    """
    if isinstance(spec, str):
        spec = {"type": spec}
    elif isinstance(spec, dict):
        spec = dict(spec)   # normalized copy; caller stores this
    else:
        raise ValueError(
            f"{name} must be a mapping or a type string; got "
            f"{_typename(spec)} ({spec!r}).")
    valid_types = RESIDUAL_TYPES if allow_nn else _VARIANCE_RESIDUAL_TYPES
    rtype = spec.get("type", "nn" if allow_nn else "rbf")
    if rtype not in valid_types:
        # Keep the historical "Unknown residual type" phrasing (pinned by
        # tests/test_residual_validation.py).
        raise ValueError(
            f"Unknown residual type {rtype!r} in {name}; expected one of "
            f"{tuple(valid_types)}{suggest(rtype, valid_types)}.")
    spec["type"] = rtype
    allowed = {"type"} | _RESIDUAL_FAMILY_KEYS.get(rtype, set())
    if context == "residual":
        allowed.add("enabled")                     # Stage-C enable flag
        if rtype in _RESIDUAL_ANALYTIC:
            allowed.add("lambda_residual")         # ignored by the NN family
    require_mapping_keys(name, spec, allowed)
    _validate_residual_values(name, spec)
    return spec


def validate_basis_per_variable(bpv, D):
    """Validate a ``basis_per_variable`` spec against the ``D`` variables.

    Accepts the literal ``'auto'`` or a mapping ``{var_index: {'basis', 'K'}}``.
    Integer variable indices must lie in ``[0, D)`` (a bool or a stringified
    index like ``'0'`` used to silently miss and fall back to the default);
    ``basis`` must name a supported family; ``K`` must be a positive non-bool
    integer; unknown nested keys (``'basi'``) are rejected. Variables absent
    from the mapping take the documented per-variable default, so a partial
    mapping is allowed. Returns ``bpv`` unchanged when valid.
    """
    # Type-guard the ``'auto'`` check FIRST: a bare ``bpv == "auto"`` on a NumPy
    # array returns an array (ambiguous-truth error) instead of the actionable
    # message, and a 1-element ``np.array(['auto'])`` would sneak through.
    if isinstance(bpv, str):
        if bpv == "auto":
            return bpv
        raise ValueError(
            f"basis_per_variable must be the string 'auto' or a mapping "
            f"{{var_index: {{'basis': ..., 'K': ...}}}}; got str ({bpv!r}).")
    if not isinstance(bpv, dict):
        raise ValueError(
            f"basis_per_variable must be the string 'auto' or a mapping "
            f"{{var_index: {{'basis': ..., 'K': ...}}}}; got "
            f"{_typename(bpv)} ({bpv!r}).")
    for idx, spec in bpv.items():
        if isinstance(idx, _BOOL_TYPES) or not isinstance(idx, _INT_TYPES):
            raise ValueError(
                f"basis_per_variable keys must be integer variable indices in "
                f"[0, {D}); got key {idx!r} ({_typename(idx)}).")
        if not (0 <= int(idx) < D):
            raise ValueError(
                f"basis_per_variable index {int(idx)} is out of range for a "
                f"{D}-variable input; valid indices are 0..{D - 1}.")
        path = f"basis_per_variable[{int(idx)}]"
        if not isinstance(spec, dict):
            raise ValueError(
                f"{path} must be a mapping with 'basis' and 'K'; got "
                f"{_typename(spec)} ({spec!r}).")
        require_mapping_keys(path, spec, {"basis", "K"})
        if "basis" in spec:
            require_choice(f"{path}['basis']", spec["basis"], BASIS_FAMILIES)
        if "K" in spec:
            require_int(f"{path}['K']", spec["K"], minimum=1)
    return bpv


# Config keys grouped by numeric rule. Each is validated only when present, so a
# partial config (the common case) is fine; ``None`` skips optional slots.
_NONNEG_INT_KEYS = ("K2", "K3", "Kh", "K2h", "K3h")   # 0 disables that order
_POS_INT_KEYS = ("max_outer_iter", "newton_max_iter")  # loops need >= 1
_NONNEG_NUM_KEYS = (
    "lambda_order1", "lambda_order2", "lambda_order3",
    "lambda_h", "lambda_h2", "lambda_h3",
    "lambda_residual", "lambda_h_residual", "min_noise_ratio",
)
_POS_NUM_KEYS = ("alternating_tol",)                   # a tolerance of 0 is invalid


def validate_config(config):
    """Validate a (mode-resolved) trainer config's values in place-safe fashion.

    Complements :meth:`HiFiANOVATrainer._validate_config_keys` (which checks the
    *key* allowlist): this checks the *values* — stage list, numeric type/range,
    enum choices, and the nested ``basis_per_variable`` / ``residual`` /
    ``variance_residual`` specs — so an invalid value fails early with a
    specific message instead of silently no-oping or crashing in the solve.
    Runs regardless of ``allow_unknown_keys`` (that hatch is for experimental
    *keys*, not a bypass of type/shape/range safety). ``basis_per_variable``
    index-range validation needs ``D`` and is done on the mixed path where the
    data is known. Returns ``config`` unchanged.
    """
    # --- stages ---
    if "stages" in config:
        validate_stages(config["stages"])

    # --- integer sizes (reject bool/float) ---
    if "K1" in config:
        require_int("K1", config["K1"], minimum=1)
    for k in _NONNEG_INT_KEYS:
        if k in config:
            # K2 may alternatively be a per-pair mapping {(i, j): K2_ij}
            # (term-structure path); its structure is validated on the fit
            # path where D is known (validate_k2_spec).
            if k == "K2" and isinstance(config[k], dict):
                continue
            require_int(k, config[k], minimum=0)
    for k in _POS_INT_KEYS:
        if k in config and config[k] is not None:
            require_int(k, config[k], minimum=1)
    if config.get("max_pair_variables") is not None:
        require_int("max_pair_variables", config["max_pair_variables"], minimum=1)

    # --- penalties / thresholds / tolerances ---
    for k in _NONNEG_NUM_KEYS:
        if k in config and config[k] is not None:
            require_number(k, config[k], minimum=0.0)
    for k in _POS_NUM_KEYS:
        if k in config and config[k] is not None:
            require_number(k, config[k], minimum=0.0, strict_min=True)

    # --- thresholds / margins (finite; non-negative magnitudes) ---
    if config.get("pair_threshold") is not None:
        require_number("pair_threshold", config["pair_threshold"], minimum=0.0)
    if config.get("variance_selection_margin") is not None:
        require_number("variance_selection_margin",
                       config["variance_selection_margin"], minimum=0.0)

    # --- auto-mode residual-fraction threshold: a fraction in (0, 1) ---
    for k in ("auto_threshold", "_auto_threshold"):
        if k in config and config[k] is not None:
            require_fraction(k, config[k])

    # --- boolean switches ---
    for k in _BOOL_KEYS:
        if k in config and config[k] is not None:
            require_bool(k, config[k])

    # --- basis / precision enums ---
    if config.get("basis_name") is not None:
        require_choice("basis_name", config["basis_name"], BASIS_FAMILIES)
    if config.get("basis_type") is not None:
        require_choice("basis_type", config["basis_type"], _BASIS_TYPES)
    if config.get("precision") is not None:
        require_choice("precision", config["precision"], _PRECISIONS)
    if config.get("stage_d_estimator") is not None:
        require_choice("stage_d_estimator", config["stage_d_estimator"],
                       _STAGE_D_ESTIMATORS)

    # --- strategy: a name, a 'sobolev[_n]'/'spectral[_n]' family, or a per-order
    #     dict. A dict is only consumed by the uniform-basis MEAN builder — the
    #     variance (Stage D) and mixed builders take a single string — so reject
    #     a dict on those paths rather than let it crash mid-fit. ---
    strat = config.get("strategy")
    if strat is not None:
        if isinstance(strat, dict):
            _stages = config.get("stages") or []
            if "D" in _stages or config.get("basis_per_variable") is not None:
                raise ValueError(
                    "a per-order strategy dict is only supported for uniform-basis "
                    "mean fitting (Stages A/B/C); Stage D (heteroscedastic "
                    "variance) and mixed per-variable bases take a single strategy "
                    "string — pass strategy='<name>' for those fits.")
        _validate_strategy("strategy", strat)

    # --- selection / candidate / pruning enums (silent 'all' fallback otherwise) ---
    if config.get("variable_selection") is not None:
        require_choice("variable_selection", config["variable_selection"],
                       _SELECTION_METHODS)
    for k in ("first_order_pruning", "pair_pruning", "triple_pruning"):
        if config.get(k) is not None:
            require_choice(k, config[k], _PRUNING_METHODS)
    if config.get("pair_candidates") is not None:
        require_choice("pair_candidates", config["pair_candidates"],
                       _PAIR_CANDIDATE_MODES)
    # pair_selection is a mode string OR an explicit list of active variable
    # indices (list[int]) — NOT a list of (i, j) pairs; the downstream
    # _resolve_pair_manager forms pairs from the active set. Range vs D is
    # checked on the fit path (validate_pair_selection).
    ps = config.get("pair_selection")
    if ps is not None:
        if isinstance(ps, list):
            for i, v in enumerate(ps):
                require_int(f"pair_selection[{i}]", v, minimum=0)
        elif isinstance(ps, str):
            require_choice("pair_selection", ps, _PAIR_SELECTION_MODES)
        else:
            raise ValueError(
                "pair_selection must be a mode string "
                f"({list(_PAIR_SELECTION_MODES)}), a list of active variable "
                f"indices, or None; got {_typename(ps)} ({ps!r}).")
    if config.get("triple_selection") is not None:
        require_choice("triple_selection", config["triple_selection"],
                       _TRIPLE_SELECTION_MODES)
    # variable_orders / variance_variables: type-level check here; index-range
    # and structure checks need D and run on the fit path.
    if config.get("variable_orders") is not None:
        if not isinstance(config["variable_orders"], dict):
            raise ValueError(
                "variable_orders must be a mapping {var_index: [orders]}; got "
                f"{_typename(config['variable_orders'])} "
                f"({config['variable_orders']!r}).")
    if config.get("variance_variables") is not None:
        if not isinstance(config["variance_variables"], (list, tuple)):
            raise ValueError(
                "variance_variables must be a list of variable indices; got "
                f"{_typename(config['variance_variables'])} "
                f"({config['variance_variables']!r}).")
    # var_pair_selection is a mode string OR an explicit list of (i, j)
    # variance-pair tuples (BR-05); pair structure/range is checked on the fit
    # path (validate_var_pair_list) where D is known.
    vps = config.get("var_pair_selection")
    if vps is not None:
        if isinstance(vps, (list, tuple)):
            for n, pair in enumerate(vps):
                if (not isinstance(pair, (list, tuple)) or len(pair) != 2):
                    raise ValueError(
                        f"var_pair_selection[{n}] must be an (i, j) pair of "
                        f"variable indices; got {pair!r}. (Note this differs "
                        "from pair_selection, which lists active VARIABLES.)")
        else:
            require_choice("var_pair_selection", vps,
                           _VAR_PAIR_SELECTION_MODES)
    if config.get("var_triple_selection") is not None:
        require_choice("var_triple_selection", config["var_triple_selection"],
                       _VAR_TRIPLE_SELECTION_MODES)

    # --- nested residual specs (normalized back into config so a string
    #     shorthand like variance_residual='rbf' can't reach .get() downstream) ---
    for k in ("residual", "residual_nn"):
        if config.get(k) is not None:
            config[k] = validate_residual_spec(k, config[k], allow_nn=True,
                                               context="residual")
    if config.get("variance_residual") is not None:
        config["variance_residual"] = validate_residual_spec(
            "variance_residual", config["variance_residual"],
            allow_nn=False, context="variance_residual")

    return config


def _strategy_name_ok(s):
    """True for a valid strategy string: a base name, or a ``sobolev``/
    ``spectral`` family that is either bare or ``<prefix>_<number>``.

    Rejects malformed families that used to fail at ``float(parts[1])``
    (``sobolev_s``, ``spectral_a``) or be silently read as the bare default
    (``sobolevgarbage``).
    """
    if not isinstance(s, str):
        return False
    if s in _STRATEGY_NAMES:
        return True
    for pre in _STRATEGY_PREFIXES:
        if s == pre:
            return True
        if s.startswith(pre + "_"):
            suffix = s[len(pre) + 1:]
            try:
                float(suffix)
                return True
            except ValueError:
                return False
    return False


def _validate_strategy(name, strat):
    """A regularization strategy: a recognized name/family, or a per-order dict.

    ``strategy`` may be a string (a base name, or a ``sobolev[_<num>]`` /
    ``spectral[_<num>]`` family), or a dict keyed by the orders the builder
    consumes (``default``/``order1``/``order2``/``order3``) with such string
    values. Unknown dict keys are rejected (``ordr1`` was silently ignored →
    the model quietly fell back to ``variance``). Any other type (``42``) is
    rejected.
    """
    def _check_str(path, s):
        if not _strategy_name_ok(s):
            raise ValueError(
                f"{path} must be one of {list(_STRATEGY_NAMES)} or a "
                f"'sobolev[_<number>]'/'spectral[_<number>]' family; got "
                f"{_typename(s)} ({s!r}){suggest(s, _STRATEGY_NAMES)}.")

    if isinstance(strat, dict):
        require_mapping_keys(name, strat, _STRATEGY_DICT_KEYS)
        for order_key, s in strat.items():
            _check_str(f"{name}[{order_key!r}]", s)
    else:
        _check_str(name, strat)


def validate_k2_spec(k2, D):
    """Validate a ``K2`` value that may be a per-pair mapping (term structure).

    ``K2`` is either a non-negative int (uniform order for every pair —
    unchanged), or a mapping ``{(i, j): K2_ij}`` that BOTH names the exact
    retained pairs and assigns each its own harmonic order (positive int).
    Pair keys must be 2-tuples of distinct in-range variable indices with
    ``i < j`` (canonical order), no duplicates. Returns a canonical
    ``{(i, j): int}`` dict for a mapping, or the int unchanged.
    """
    if not isinstance(k2, dict):
        require_int("K2", k2, minimum=0)
        return k2
    if not k2:
        raise ValueError(
            "K2 given as a mapping must name at least one pair; use K2=0 to "
            "disable second-order terms.")
    out = {}
    for key, val in k2.items():
        if (not isinstance(key, tuple) or len(key) != 2):
            raise ValueError(
                f"K2 mapping keys must be (i, j) variable-index pairs; got "
                f"{key!r}. (Note: unlike pair_selection, these are PAIRS, "
                "not active-variable indices.)")
        i = require_int(f"K2 key {key!r}[0]", key[0], minimum=0)
        j = require_int(f"K2 key {key!r}[1]", key[1], minimum=0)
        if i >= D or j >= D:
            raise ValueError(
                f"K2 pair {key!r} is out of range for a {D}-variable input; "
                f"indices must be in 0..{D - 1}.")
        if i == j:
            raise ValueError(f"K2 pair {key!r} repeats a variable; a pair "
                             "needs two distinct variables.")
        if i > j:
            raise ValueError(
                f"K2 pair {key!r} is not in canonical order; write it as "
                f"({j}, {i}) with the smaller index first.")
        if (i, j) in out:
            raise ValueError(f"K2 mapping repeats pair ({i}, {j}).")
        out[(i, j)] = require_int(f"K2[{key!r}]", val, minimum=1)
    return out


def validate_variable_orders(vo, D):
    """Validate ``variable_orders`` — per-variable interaction-order membership.

    A mapping ``{var_index: orders}`` where ``orders`` is a subset of
    ``{1, 2}`` (list/tuple/set): ``[1, 2]`` is the default full membership,
    ``[2]`` admits the variable to pair terms only (its first-order block is
    EXCLUDED from the design — no df spent), ``[1]`` keeps the first-order
    block but excludes the variable from every pair, and ``[]`` excludes the
    variable from the MEAN model entirely while its column remains available
    to the VARIANCE model (a "variance-only" variable — heteroscedastic fits
    only; the trainer enforces that). Returns a canonical
    ``{int: tuple(sorted(orders))}`` dict.
    """
    if not isinstance(vo, dict):
        raise ValueError(
            f"variable_orders must be a mapping {{var_index: [orders]}} with "
            f"orders a subset of {{1, 2}}; got {_typename(vo)} ({vo!r}).")
    out = {}
    for idx, orders in vo.items():
        if isinstance(idx, _BOOL_TYPES) or not isinstance(idx, _INT_TYPES):
            raise ValueError(
                f"variable_orders keys must be integer variable indices; got "
                f"{idx!r} ({_typename(idx)}).")
        i = int(idx)
        if not (0 <= i < D):
            raise ValueError(
                f"variable_orders index {i} is out of range for a "
                f"{D}-variable input; valid indices are 0..{D - 1}.")
        if isinstance(orders, (int,)) and not isinstance(orders, bool):
            orders = (orders,)
        if not isinstance(orders, (list, tuple, set, frozenset)):
            raise ValueError(
                f"variable_orders[{i}] must be a collection of orders from "
                f"{{1, 2}} (empty = variance-only membership); got "
                f"{orders!r}.")
        o = tuple(sorted(set(int(v) for v in orders)))
        if any(v not in (1, 2) for v in o):
            raise ValueError(
                f"variable_orders[{i}] contains an unsupported order in "
                f"{orders!r}; only orders 1 and 2 are supported (third-order "
                "terms are not order-selectable).")
        out[i] = o
    return out


def validate_variance_variables(vv, D):
    """Validate ``variance_variables`` — the first-order variance-model subset.

    A non-empty list/tuple of unique variable indices in ``[0, D)``. Returns a
    sorted tuple. NOTE the statistical meaning: excluding a variable ASSERTS
    the noise is homoscedastic along it — a modeling assumption the fit will
    not test per-direction.
    """
    if not isinstance(vv, (list, tuple)):
        raise ValueError(
            f"variance_variables must be a list of variable indices; got "
            f"{_typename(vv)} ({vv!r}).")
    if not vv:
        raise ValueError(
            "variance_variables must name at least one variable (an empty "
            "variance model is not a heteroscedastic fit — use "
            "heteroscedastic=False instead).")
    out = []
    for n, v in enumerate(vv):
        i = require_int(f"variance_variables[{n}]", v, minimum=0)
        if i >= D:
            raise ValueError(
                f"variance_variables[{n}] = {i} is out of range for a "
                f"{D}-variable input; valid indices are 0..{D - 1}.")
        if i in out:
            raise ValueError(f"variance_variables repeats index {i}.")
        out.append(i)
    return tuple(sorted(out))


def validate_var_pair_list(vps, D, variance_variables=None):
    """Validate an explicit ``var_pair_selection`` list of (i, j) pairs.

    Each pair must have distinct in-range indices in canonical ``i < j`` order,
    no duplicates. When a ``variance_variables`` subset is active, every pair
    variable must be inside it (a variance pair on a variable asserted
    variance-flat is contradictory). Returns a list of ``(i, j)`` tuples.
    """
    out = []
    for n, pair in enumerate(vps):
        i = require_int(f"var_pair_selection[{n}][0]", pair[0], minimum=0)
        j = require_int(f"var_pair_selection[{n}][1]", pair[1], minimum=0)
        if i >= D or j >= D:
            raise ValueError(
                f"var_pair_selection[{n}] = ({i}, {j}) is out of range for a "
                f"{D}-variable input; indices must be in 0..{D - 1}.")
        if i == j:
            raise ValueError(
                f"var_pair_selection[{n}] repeats a variable; a pair needs "
                "two distinct variables.")
        if i > j:
            raise ValueError(
                f"var_pair_selection[{n}] = ({i}, {j}) is not in canonical "
                f"order; write it as ({j}, {i}).")
        if (i, j) in out:
            raise ValueError(f"var_pair_selection repeats pair ({i}, {j}).")
        if variance_variables is not None and (
                i not in variance_variables or j not in variance_variables):
            raise ValueError(
                f"var_pair_selection pair ({i}, {j}) uses a variable outside "
                f"variance_variables={sorted(variance_variables)}; a variance "
                "pair cannot involve a variable asserted variance-flat.")
        out.append((i, j))
    return out


def validate_pair_selection(ps, D):
    """Range-check a legacy ``pair_selection`` list against the ``D`` variables.

    ``pair_selection`` is a **list of active variable indices** (``list[int]``),
    not a list of ``(i, j)`` pairs — the downstream ``_resolve_pair_manager``
    builds pairs from the active set. Each index must be a non-bool integer in
    ``[0, D)`` (needs ``D``, so this runs on the fit path). String modes and
    ``None`` are validated earlier in :func:`validate_config` and skipped here.
    """
    if not isinstance(ps, list):
        return ps
    for i, v in enumerate(ps):
        idx = require_int(f"pair_selection[{i}]", v, minimum=0)
        if idx >= D:
            raise ValueError(
                f"pair_selection[{i}] = {idx} is out of range for a {D}-variable "
                f"input; active variable indices must be in 0..{D - 1}.")
    return ps
