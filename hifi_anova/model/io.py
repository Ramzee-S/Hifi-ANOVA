"""Save and load fitted HiFi-ANOVA models.

Saves the full model (Fourier coefficients, variance model, residual,
Gram matrices, config) plus the preprocessing transformer and metadata.

Uses Equinox tree serialization for the JAX arrays and pickle for the
transformer and config. Everything goes into a single directory.

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
import jax.numpy as jnp
import equinox as eqx
from typing import Dict, Optional, List, Any


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

    # Save model arrays via Equinox
    model_path = os.path.join(path, 'model.eqx')
    eqx.tree_serialise_leaves(model_path, model)

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
        'has_residual': model.residual_net is not None,
        'has_linear_residual': model.has_linear_residual,
        'has_nn_residual': model.has_nn_residual,
        'is_mixed': model.is_mixed,
    }
    if feature_names is not None:
        meta['feature_names'] = feature_names
    if config is not None:
        # Filter config to JSON-serializable items
        safe_config = {}
        for k, v in config.items():
            try:
                json.dumps(v)
                safe_config[k] = v
            except (TypeError, ValueError):
                safe_config[k] = str(v)
        meta['config'] = safe_config

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
            results_str = json.dumps(results, default=_make_serializable)
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

    # Build a template model if none provided
    if like_model is None:
        like_model = _build_template_model(meta)

    # Load model arrays
    model_path = os.path.join(path, 'model.eqx')
    model = eqx.tree_deserialise_leaves(model_path, like_model)

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
            result['results'] = json.load(f)

    return result


def _build_template_model(meta: Dict):
    """Build a template HiFiANOVA model from metadata for deserialization."""
    from .mean_model import MeanModel
    from .hifi_anova import HiFiANOVA
    from ..core.features import basis_size

    D = meta['D']
    K1 = meta['K1']
    K2 = meta.get('K2', 0)
    K3 = meta.get('K3', 0)
    bn = meta.get('basis_name', 'fourier')
    il1 = meta.get('include_linear_1', True)
    il2 = meta.get('include_linear_2', True)
    il3 = meta.get('include_linear_3', True)

    B1 = basis_size(K1, il1, bn)
    F1 = D * B1

    B2 = basis_size(K2, il2, bn) if K2 > 0 else 0
    # We don't know how many pairs were selected — use empty
    F2 = 0

    mm = MeanModel(
        f0=jnp.array(0.0, dtype=jnp.float32),
        w1=jnp.zeros(F1, dtype=jnp.float32),
        w2=jnp.zeros(F2, dtype=jnp.float32),
        K1=K1, K2=K2, D=D, K3=K3,
        include_linear_1=il1, include_linear_2=il2, include_linear_3=il3,
        basis_name=bn,
    )

    return HiFiANOVA(
        mean_model=mm,
        K1=K1, K2=K2, K3=K3, Kh=meta.get('Kh', 0), D=D,
        include_linear_1=il1, include_linear_2=il2, include_linear_3=il3,
        basis_name=bn,
    )
