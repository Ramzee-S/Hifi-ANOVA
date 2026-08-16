"""Save and load fitted HiFi-ANOVA models.

Saves the full model (Fourier coefficients, variance model, residual,
Gram matrices, config) plus the preprocessing transformer and metadata.

Uses Equinox tree serialization for the JAX arrays and pickle for the
transformer and config. Everything goes into a single directory.

For simple (uniform-basis, homoscedastic, residual-free) mean models the load
path rebuilds the pytree template from ``meta.json`` and restores the arrays via
Equinox. Structures whose pytree can't be reconstructed from metadata alone — a
variance model (heteroscedastic fit), the guard's constant-variance fallback, a
residual net, or a mixed per-variable basis — are restored from a full-model
pickle (``model.pkl``) written alongside. So every fitted model round-trips;
`save`/`load` is not limited to mean-only models.

Usage:
    from hifi_anova.model.io import save_model, load_model

    # Save
    save_model(model, 'my_model/', config=config, transformer=data['transformer'],
               feature_names=['MedInc', 'HouseAge', ...])

    # Load
    loaded = load_model('my_model/')
    model = loaded['model']
    transformer = loaded['transformer']
    pred = model.predict_mean_only(transformer.transform(X_new))
"""

import os
import json
import pickle
import numpy as np
from ..array_backend import xp as jnp  # switchable array backend (numpy exact core)
import equinox as eqx
from typing import Dict, Optional, List, Any

from .._result_aliases import canonical_result_mapping


class _NeedsFullModel(Exception):
    """Signals that a saved model's pytree structure can't be reconstructed from
    metadata alone (variance model, residual net, constant variance, or mixed
    basis), so ``load_model`` should restore it from the full-model pickle."""


def _json_safe_config_value(v):
    """Coerce one config value into a JSON-serializable form for ``meta.json``.

    Most values (scalars, lists, plain dicts) serialize directly. The one lossy
    case worth handling is a term-structure ``K2`` mapping ``{(i, j): K2_ij}``:
    JSON object keys can't be tuples, so the earlier blanket ``str(v)`` fallback
    stored the whole mapping as an opaque string (``"{(0, 1): 4, ...}"``) — a
    faithful config round-trip was impossible. We instead normalize tuple keys
    to the same ``"i,j"`` string form already used by
    ``results['term_structure']['pair_k2']`` (see ``trainer.py``), so the config
    round-trips exactly. Anything still non-serializable falls back to ``str``.
    """
    if isinstance(v, dict) and any(isinstance(k, tuple) for k in v):
        return {(",".join(str(x) for x in k) if isinstance(k, tuple) else str(k)):
                _json_safe_config_value(val) for k, val in v.items()}
    try:
        json.dumps(v)
        return v
    except (TypeError, ValueError):
        return str(v)


def save_model(
    model,
    path: str,
    config: Optional[Dict] = None,
    transformer=None,
    feature_names: Optional[List[str]] = None,
    results: Optional[Dict] = None,
    overwrite: bool = False,
):
    """Save a fitted HiFiANOVA model to a directory.

    Creates a directory with:
      model.eqx    — Equinox serialized model (JAX arrays)
      meta.json    — model config, feature names, dimensions
      transformer.pkl — sklearn QuantileTransformer (if provided)
      results.json — training results (if provided)

    Args:
        model: fitted HiFiANOVA instance
        path: directory path to save to
        config: the config dict used for training
        transformer: sklearn QuantileTransformer from preprocessing
        feature_names: list of feature names
        results: training results dict
        overwrite: if True, overwrite existing directory
    """
    if os.path.exists(path) and not overwrite:
        raise FileExistsError(f"Directory {path} exists. Use overwrite=True.")

    os.makedirs(path, exist_ok=True)

    # Save model arrays via Equinox (the portable, array-level path; used on load
    # for uniform-basis, homoscedastic, residual-free mean models).
    model_path = os.path.join(path, 'model.eqx')
    eqx.tree_serialise_leaves(model_path, model)

    # Full-model pickle: a robust fallback used by load_model for structures the
    # metadata template can't reconstruct (variance model / heteroscedastic fit,
    # constant-variance fallback, residual net, mixed basis). Models are small
    # (JAX arrays serialize to numpy), so the duplication is negligible. Written
    # best-effort; the Equinox path above still covers the simple mean model.
    try:
        with open(os.path.join(path, 'model.pkl'), 'wb') as f:
            pickle.dump(model, f)
    except Exception:  # pragma: no cover - defensive; pickle of a fitted model is expected to work
        pass

    # Save metadata
    meta = {
        'D': model.D,
        'K1': model.K1,
        'K2': model.K2,
        'K3': model.K3,
        'Kh': model.Kh,
        'basis_name': model.basis_name,
        'include_linear_1': model.include_linear_1,
        'include_linear_2': model.include_linear_2,
        'include_linear_3': model.include_linear_3,
        'has_variance_model': model.variance_model is not None,
        'has_constant_log_var': model.constant_log_var is not None,
        'has_residual': model.residual_net is not None,
        'has_linear_residual': model.has_linear_residual,
        'has_nn_residual': model.has_nn_residual,
        'is_mixed': model.is_mixed,
        # Per-pair K2 term structure (X11C-S02): a ragged w2 layout the uniform
        # deserialization template cannot rebuild — load_model falls back to
        # the full-model pickle when set. Absent on older saves (=False).
        'has_pair_k2': getattr(model, 'pair_k2', None) is not None,
        # Order-selective term structure (X11C-S03): descriptive provenance of
        # the user-defined equation system. The model restores via the pickle
        # fallback (variance model / ragged layout), so these are human-readable
        # mirrors, not load-template inputs. Absent on older saves.
        'fo_included': (list(getattr(model, 'fo_included', None))
                        if getattr(model, 'fo_included', None) is not None
                        else None),
        'variance_variables': (
            list(getattr(model.variance_model, 'variance_variables', None))
            if (model.variance_model is not None
                and getattr(model.variance_model, 'variance_variables', None)
                is not None) else None),
        # Structural counts needed to rebuild the deserialization template with
        # matching leaf shapes (pair_indices / triple_indices are now dynamic
        # integer leaves, and w2 / w3 lengths depend on P / T).
        'P': (int(model.pair_indices.shape[0])
              if model.pair_indices is not None else 0),
        'T': (int(model.triple_indices.shape[0])
              if model.triple_indices is not None else 0),
        # Fit-weight precision (DEC-035), so the deserialization template is
        # rebuilt with matching leaf dtypes for a float64 fit. Absent metadata
        # (older saves) defaults to float32 on load.
        'fit_dtype': str(model.mean_model.f0.dtype),
    }
    # Stage-D mean-estimator convention (DEC-039/DEC-047 provenance). Persist the
    # EFFECTIVE convention of the shipped mean so a reloaded heteroscedastic
    # artifact can be told apart by estimator vintage (profiled joint-GLS vs the
    # legacy fixed-intercept/uncentered solve). The authoritative source is the
    # fitted model itself (`model.mean_intercept_mode`), so a bare
    # `save_model(model, path)` without a results dict still records it; the
    # results dict is a fallback for older models predating the model field. Only
    # a Stage-D vintage is written to meta — a homoscedastic model carries the
    # default unit-weight tag, which load_model can re-infer, so it stays omitted
    # to keep older homoscedastic saves byte-identical.
    _mode = getattr(model, 'mean_intercept_mode', None)
    if _mode is None and isinstance(results, dict):
        _stage_d = results.get('stage_D')
        if isinstance(_stage_d, dict):
            _mode = _stage_d.get('mean_intercept_mode')
    if _mode and (model.variance_model is not None
                  or model.constant_log_var is not None):
        meta['mean_intercept_mode'] = _mode
    # Stage-D estimator identity (P0-2 / X6 Session 2). Mirror the honest
    # machine-readable estimator metadata onto meta.json when a heteroscedastic
    # fit's results dict carries it, so a reloaded artifact can be told apart by
    # estimator vintage without parsing results.json. Purely descriptive
    # provenance (unlike mean_intercept_mode it is not needed to reconstruct or
    # re-analyse the model), so it is sourced from results['stage_D'] only — a
    # bare save_model(model) without results simply omits it.
    if isinstance(results, dict):
        _stage_d = results.get('stage_D')
        if isinstance(_stage_d, dict) and _stage_d.get('objective_family'):
            meta['stage_d_estimator'] = {
                k: _stage_d[k] for k in (
                    'estimator', 'objective_family', 'residual_update',
                    'iterate_selection', 'convergence_reason', 'bound_active')
                if k in _stage_d}
        _inference = results.get('inference_metadata')
        if isinstance(_inference, dict):
            # Additive mirror for artifact inspection; the authoritative
            # round-trip copy also remains in results.json.
            meta['inference_metadata'] = _inference
    if feature_names is not None:
        meta['feature_names'] = feature_names
    if config is not None:
        # Filter config to JSON-serializable items. Tuple-keyed mappings (a
        # per-pair ``K2={(i, j): K2_ij}`` term structure) are normalized to the
        # ``"i,j"`` string-key form rather than stringified whole, so the config
        # round-trips faithfully (see _json_safe_config_value).
        meta['config'] = {k: _json_safe_config_value(v) for k, v in config.items()}

    with open(os.path.join(path, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    # Save transformer
    if transformer is not None:
        with open(os.path.join(path, 'transformer.pkl'), 'wb') as f:
            pickle.dump(transformer, f)

    # Save results (best-effort — skip if not serializable)
    if results is not None:
        def _make_serializable(obj):
            if isinstance(obj, (np.floating, np.integer)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, jnp.ndarray):
                return np.asarray(obj).tolist()
            return str(obj)  # fallback: convert to string

        try:
            # New artifacts physically store canonical public result keys only.
            # Warning aliases are reconstructed on load.
            results_str = json.dumps(
                canonical_result_mapping(results), default=_make_serializable)
            with open(os.path.join(path, 'results.json'), 'w') as f:
                f.write(results_str)
        except (ValueError, TypeError):
            # Circular reference or other issue — save as pickle instead
            with open(os.path.join(path, 'results.pkl'), 'wb') as f:
                pickle.dump(results, f)


def load_model(
    path: str,
    like_model=None,
) -> Dict[str, Any]:
    """Load a saved HiFiANOVA model from a directory.

    Args:
        path: directory path containing saved model
        like_model: a model with the same structure (for Equinox deserialization).
            If None, reconstructs from metadata.

    Returns:
        dict with:
          'model': HiFiANOVA instance
          'meta': metadata dict (D, K1, K2, ...)
          'config': training config (if saved)
          'feature_names': list of names (if saved)
          'transformer': sklearn transformer (if saved)
          'results': training results (if saved)
    """
    if not os.path.isdir(path):
        raise FileNotFoundError(f"No model directory at {path}")

    # Load metadata
    with open(os.path.join(path, 'meta.json'), 'r') as f:
        meta = json.load(f)

    # Stage-D mean-estimator convention (DEC-039 provenance): give artifacts that
    # predate the field a defined interpretation rather than a missing key. An
    # older *heteroscedastic* save cannot have its Stage-D mean vintage recovered
    # from metadata → 'legacy_unknown'; an older homoscedastic save always used
    # the ordinary unit-weight centered mean → that value is safely inferred.
    # Fresh saves carry the real value, so this only fills a gap.
    if 'mean_intercept_mode' not in meta:
        from ..training.fitted_design import (
            MEAN_INTERCEPT_LEGACY_UNKNOWN, MEAN_INTERCEPT_UNWEIGHTED)
        meta['mean_intercept_mode'] = (
            MEAN_INTERCEPT_LEGACY_UNKNOWN if meta.get('has_variance_model')
            else MEAN_INTERCEPT_UNWEIGHTED)

    # Reconstruct the model. Preference order:
    #   1. explicit like_model (caller supplied the exact structure)
    #   2. metadata template + Equinox leaf deserialization (simple mean models)
    #   3. full-model pickle (heteroscedastic / residual / constant-variance /
    #      mixed — structures the metadata template can't rebuild)
    model_path = os.path.join(path, 'model.eqx')
    pkl_path = os.path.join(path, 'model.pkl')
    # Reconstruct under the ARRAY BACKEND the model was fitted on (recorded in
    # config; 'jax' for pre-backend saves): the template skeleton's leaf types
    # decide whether the reloaded weights are numpy or jax arrays, and a
    # numpy-core model should round-trip as numpy — not silently migrate onto
    # the compile-paying backend. (The pickle path preserves array types by
    # itself.)
    from ..array_backend import use_array_backend
    _saved_backend = (meta.get('config') or {}).get('array_backend') or 'jax'
    if like_model is None:
        try:
            with use_array_backend(_saved_backend):
                like_model = _build_template_model(meta)
        except _NeedsFullModel:
            like_model = None
    if like_model is not None:
        with use_array_backend(_saved_backend):
            model = eqx.tree_deserialise_leaves(model_path, like_model)
    elif os.path.exists(pkl_path):
        with open(pkl_path, 'rb') as f:
            model = pickle.load(f)
    else:
        raise NotImplementedError(
            "This saved model's structure (variance model, residual net, "
            "constant variance, or mixed basis) can't be reconstructed from "
            "metadata, and no model.pkl is present (it predates full-model "
            "serialization). Re-save it with the current version, or pass an "
            "explicit `like_model` with the same structure to load_model().")

    result = {
        'model': model,
        'meta': meta,
        'config': meta.get('config', {}),
        'feature_names': meta.get('feature_names'),
    }

    # Load transformer if present
    transformer_path = os.path.join(path, 'transformer.pkl')
    if os.path.exists(transformer_path):
        with open(transformer_path, 'rb') as f:
            result['transformer'] = pickle.load(f)

    # Load results if present
    results_path = os.path.join(path, 'results.json')
    if os.path.exists(results_path):
        with open(results_path, 'r') as f:
            result['results'] = canonical_result_mapping(json.load(f))

    return result


def _build_template_model(meta: Dict):
    """Build a template HiFiANOVA model from metadata for deserialization.

    The template must have exactly the saved model's pytree structure (matching
    leaf shapes) for ``eqx.tree_deserialise_leaves``. This reconstructs the
    **mean-model** structure — first, second (P pairs) and third (T triples)
    order — for uniform-basis, homoscedastic, residual-free models.

    Structures whose pytree can't be rebuilt from metadata alone — the variance
    model (heteroscedastic fit), a constant-variance fallback, the residual net,
    and mixed per-variable bases — raise :class:`_NeedsFullModel`, which
    ``load_model`` handles by restoring from the full-model pickle instead.
    """
    from .mean_model import MeanModel
    from .hifi_anova import HiFiANOVA
    from ..core.features import basis_size

    if (meta.get('is_mixed') or meta.get('has_variance_model')
            or meta.get('has_constant_log_var') or meta.get('has_residual')
            # Per-pair K2 (ragged w2) / term-structure statics can't be rebuilt
            # from this uniform template — restore from the full-model pickle.
            or meta.get('has_pair_k2')):
        raise _NeedsFullModel

    D = meta['D']
    K1 = meta['K1']
    K2 = meta.get('K2', 0)
    K3 = meta.get('K3', 0)
    bn = meta.get('basis_name', 'fourier')
    il1 = meta.get('include_linear_1', True)
    il2 = meta.get('include_linear_2', True)
    il3 = meta.get('include_linear_3', True)
    P = int(meta.get('P', 0))
    T = int(meta.get('T', 0))

    F1 = D * basis_size(K1, il1, bn)
    B2 = basis_size(K2, il2, bn) if K2 > 0 else 0
    F2 = P * B2 * B2 if (K2 > 0 and P > 0) else 0
    B3 = basis_size(K3, il3, bn) if K3 > 0 else 0
    F3 = T * B3 * B3 * B3 if (K3 > 0 and T > 0) else 0

    # Match the saved fit precision so Equinox leaf dtypes agree (DEC-035);
    # older saves without the key default to float32.
    _dt = jnp.float64 if meta.get('fit_dtype') == 'float64' else jnp.float32

    mm = MeanModel(
        f0=jnp.array(0.0, dtype=_dt),
        w1=jnp.zeros(F1, dtype=_dt),
        w2=jnp.zeros(F2, dtype=_dt),
        K1=K1, K2=K2, D=D, K3=K3,
        w3=jnp.zeros(F3, dtype=_dt),
        include_linear_1=il1, include_linear_2=il2, include_linear_3=il3,
        basis_name=bn,
    )

    # Integer index leaves must match the saved shapes (P, 2) / (T, 3).
    pair_indices = jnp.zeros((P, 2), dtype=jnp.int32) if P > 0 else None
    triple_indices = jnp.zeros((T, 3), dtype=jnp.int32) if T > 0 else None

    return HiFiANOVA(
        mean_model=mm,
        K1=K1, K2=K2, K3=K3, Kh=meta.get('Kh', 0), D=D,
        pair_indices=pair_indices, triple_indices=triple_indices,
        include_linear_1=il1, include_linear_2=il2, include_linear_3=il3,
        basis_name=bn,
    )
