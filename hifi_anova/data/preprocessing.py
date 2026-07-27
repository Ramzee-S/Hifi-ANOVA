"""Preprocessing: QuantileTransformer to [0,1] + train/val/test splits."""

import numpy as np
from sklearn.preprocessing import QuantileTransformer
from typing import Tuple, Optional, Dict
import jax.numpy as jnp


def preprocess_data(X: np.ndarray, y: np.ndarray,
                    val_fraction: float = 0.15,
                    test_fraction: float = 0.15,
                    seed: int = 42,
                    transformer: Optional[QuantileTransformer] = None
                    ) -> Dict:
    """Preprocess data: split, transform to uniform [0,1].

    Uses QuantileTransformer fitted on training data ONLY.

    Args:
        X: (N, D) raw features
        y: (N,) targets
        val_fraction: fraction for validation
        test_fraction: fraction for test
        seed: random seed for splitting
        transformer: pre-fitted transformer (if None, fits a new one)

    Returns:
        dict with keys:
            x_train, y_train, x_val, y_val, x_test, y_test: jnp arrays
            transformer: fitted QuantileTransformer
            y_mean, y_std: target statistics (for denormalization)
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    N = len(X)

    if np.any(np.isnan(X)):
        raise ValueError("Input X contains NaN values. Clean or impute before calling preprocess_data.")
    if np.any(np.isnan(y)):
        raise ValueError("Target y contains NaN values. Clean or impute before calling preprocess_data.")
    if val_fraction + test_fraction >= 1.0:
        raise ValueError(f"val_fraction ({val_fraction}) + test_fraction ({test_fraction}) must be < 1.0")

    rng = np.random.RandomState(seed)
    indices = rng.permutation(N)

    n_test = int(N * test_fraction)
    n_val = int(N * val_fraction)
    n_train = N - n_test - n_val

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    # Fit QuantileTransformer on training data only
    if transformer is None:
        transformer = QuantileTransformer(
            output_distribution='uniform',
            n_quantiles=min(n_train, 1000),
            random_state=seed
        )
        transformer.fit(X_train)

    # Transform all splits
    X_train_t = transformer.transform(X_train)
    X_val_t = transformer.transform(X_val)
    X_test_t = transformer.transform(X_test)

    # Clip to [0,1] (quantile transformer can produce values slightly outside)
    X_train_t = np.clip(X_train_t, 0.0, 1.0)
    X_val_t = np.clip(X_val_t, 0.0, 1.0)
    X_test_t = np.clip(X_test_t, 0.0, 1.0)

    # Target statistics
    y_mean = float(np.mean(y_train))
    y_std = float(np.std(y_train))

    return {
        'x_train': jnp.array(X_train_t, dtype=jnp.float32),
        'y_train': jnp.array(y_train, dtype=jnp.float32),
        'x_val': jnp.array(X_val_t, dtype=jnp.float32),
        'y_val': jnp.array(y_val, dtype=jnp.float32),
        'x_test': jnp.array(X_test_t, dtype=jnp.float32),
        'y_test': jnp.array(y_test, dtype=jnp.float32),
        'transformer': transformer,
        'y_mean': y_mean,
        'y_std': y_std,
        'n_train': n_train,
        'n_val': n_val,
        'n_test': n_test,
    }
