"""Built-in dataset registry for the console (gui1's set, ported).

Real datasets come from sklearn/OpenML (cached in ~/scikit_learn_data —
already present on this machine); synthetic ones from hifi_anova.data.
Large datasets are row-subsampled to the requested N for interactivity,
with the original size kept in the label.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

MAX_N_DEFAULT = 4000

REGISTRY = [
    {"id": "fatigue", "label": "fatigue demo (§10)", "kind": "synthetic"},
    {"id": "ishigami", "label": "Ishigami (noise-free)", "kind": "synthetic"},
    {"id": "ishigami_het", "label": "Ishigami heteroscedastic", "kind": "synthetic"},
    {"id": "heteroscedastic", "label": "synthetic heteroscedastic", "kind": "synthetic"},
    {"id": "friedman1", "label": "Friedman #1", "kind": "synthetic"},
    {"id": "california", "label": "California housing", "kind": "real"},
    {"id": "airfoil", "label": "airfoil self-noise", "kind": "real"},
    {"id": "energy", "label": "energy efficiency", "kind": "real"},
    {"id": "kin8nm", "label": "kin8nm", "kind": "real"},
    {"id": "wine", "label": "wine quality (red)", "kind": "real"},
]

_OPENML = {
    "airfoil": dict(name="airfoil_self_noise", version=1),
    "energy": dict(name="energy_efficiency", version=1),
    "kin8nm": dict(name="kin8nm", version=1),
    "wine": dict(data_id=40691),
}


def make_fatigue_demo(n: int = 4000, seed: int = 42):
    """Fatigue demonstrator (Manuscript_Theoryv07 §10)."""
    rng = np.random.default_rng(seed)
    u = rng.random((n, 6))
    p = u - 0.5
    mu = 6.0 - 3.0 * p[:, 0] - 1.0 * p[:, 0] * p[:, 1] + 0.8 * p[:, 2] + 0.5 * p[:, 4]
    h = -3.0 - 1.5 * p[:, 0] + 1.2 * p[:, 3] + 0.4 * p[:, 2] + 0.8 * p[:, 2] * p[:, 3]
    y = mu + np.exp(h / 2.0) * rng.standard_normal(n)
    names = [f"x{i}" for i in range(1, 7)]
    return u, y, names


def _xnames(x: np.ndarray) -> List[str]:
    return [f"x{i + 1}" for i in range(x.shape[1])]


def _subsample(x, y, cap: int, seed: int):
    total = len(y)
    if cap and total > cap:
        idx = np.random.default_rng(seed).permutation(total)[:cap]
        return x[idx], y[idx], total
    return x, y, None


def load(name: str, n: Optional[int] = None, seed: int = 42
         ) -> Tuple[np.ndarray, np.ndarray, List[str], str]:
    """Return (X, y, feature_names, label). Raises on unknown name."""
    cap = int(n) if n else MAX_N_DEFAULT

    if name == "fatigue":
        x, y, names = make_fatigue_demo(cap, seed)
        return x, y, names, f"fatigue_demo · N={cap} · D=6"

    if name in ("ishigami", "ishigami_het"):
        from hifi_anova.data import synthetic as syn
        out = syn.generate_ishigami(cap, seed=seed,
                                    heteroscedastic=(name == "ishigami_het"))
        x, y = np.asarray(out[0], float), np.asarray(out[1], float)
        tag = "Ishigami het" if name == "ishigami_het" else "Ishigami (noise-free)"
        return x, y, _xnames(x), f"{tag} · N={len(y)} · D={x.shape[1]}"

    if name == "heteroscedastic":
        from hifi_anova.data import synthetic as syn
        out = syn.generate_heteroscedastic(cap, seed=seed)
        x, y = np.asarray(out[0], float), np.asarray(out[1], float)
        return x, y, _xnames(x), f"synthetic het · N={len(y)} · D={x.shape[1]}"

    if name == "friedman1":
        from hifi_anova.data import test_functions as tf
        out = tf.T2_1_friedman1(n_samples=cap, seed=seed)
        x, y = np.asarray(out[0], float), np.asarray(out[1], float)
        return x, y, _xnames(x), f"Friedman#1 · N={len(y)} · D={x.shape[1]}"

    if name == "california":
        from sklearn.datasets import fetch_california_housing
        d = fetch_california_housing()
        x, y, names = (np.asarray(d.data, float), np.asarray(d.target, float),
                       list(d.feature_names))
    elif name in _OPENML:
        from sklearn.datasets import fetch_openml
        d = fetch_openml(as_frame=False, **_OPENML[name])
        x = np.asarray(d.data, float)
        y = np.asarray(d.target, float)  # wine targets are strings — cast
        names = list(d.feature_names)
    else:
        raise ValueError(f"unknown dataset: {name}")

    x, y, total = _subsample(x, y, cap, seed)
    size = f"N={len(y)}" + (f"/{total}" if total else "")
    return x, y, names, f"{name} · {size} · D={x.shape[1]}"
