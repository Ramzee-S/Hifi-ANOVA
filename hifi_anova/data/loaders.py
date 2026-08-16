"""OpenML dataset loading for real-world benchmarks."""

import numpy as np
from typing import Tuple


def load_california_housing() -> Tuple[np.ndarray, np.ndarray, list]:
    """Load California Housing dataset from sklearn."""
    from sklearn.datasets import fetch_california_housing
    data = fetch_california_housing()
    return data.data, data.target, list(data.feature_names)


def load_openml_dataset(name: str) -> Tuple[np.ndarray, np.ndarray, list]:
    """Load a dataset from OpenML by name.

    Supported: 'kin8nm', 'concrete', 'california'
    """
    if name == 'california':
        return load_california_housing()

    try:
        from sklearn.datasets import fetch_openml
        if name == 'kin8nm':
            data = fetch_openml(name='kin8nm', version=1, as_frame=False)
        elif name == 'concrete':
            # OpenML has no dataset literally named 'concrete'; this is the UCI
            # Concrete Compressive Strength set from the curated OpenML-CTR23
            # suite, pinned by id (several active versions exist by name).
            data = fetch_openml(data_id=44959, as_frame=False)
        else:
            data = fetch_openml(name=name, version=1, as_frame=False)

        X = np.array(data.data, dtype=np.float64)
        y = np.array(data.target, dtype=np.float64)
        feature_names = list(data.feature_names) if hasattr(data, 'feature_names') else [f"x{i}" for i in range(X.shape[1])]
        return X, y, feature_names
    except Exception as e:
        raise ValueError(f"Could not load dataset '{name}': {e}")
