"""Preprocessing: QuantileTransformer to [0,1] + train/val/test splits."""

import warnings

import numpy as np
from sklearn.preprocessing import QuantileTransformer
from typing import Optional, Dict
from ..array_backend import xp as jnp  # switchable array backend (numpy exact core)

# Ties diagnostic thresholds: a column is flagged as discrete/heavily tied
# when it has fewer distinct training values than _TIES_MIN_UNIQUE, or when
# a single value accounts for more than _TIES_MAX_SHARE of the training rows.
_TIES_MIN_UNIQUE = 10
_TIES_MAX_SHARE = 0.2


def preprocess_data(X: np.ndarray, y: np.ndarray,
                    val_fraction: float = 0.15,
                    test_fraction: float = 0.15,
                    seed: int = 42,
                    transformer: Optional[QuantileTransformer] = None,
                    fit_dtype=jnp.float32
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
        fit_dtype: JAX dtype for the returned train/val/test arrays — the fit
            precision (default ``jnp.float32``; ``jnp.float64`` for a float64
            fit). See ``hifi_anova.precision`` (DEC-035).

    Returns:
        dict with keys:
            x_train, y_train, x_val, y_val, x_test, y_test: jnp arrays
            transformer: fitted QuantileTransformer
            y_mean, y_std: descriptive train-target statistics. NOTE: y is
                NOT standardized by this function — the model centers via its
                fitted intercept f0 and predictions need no back-transform;
                these are surfaced for reporting only.
            train_indices, val_indices, test_indices: int arrays of original
                dataset row indices for each split, in the exact row order of
                the corresponding ``x_*``/``y_*`` arrays (BR-07). So
                ``X[train_indices]`` reproduces the rows the model was fit on,
                in the same order as the fitted design's ``Phi``. Read-only
                provenance — no fitting behavior depends on them.

    Warning: if a pre-fitted ``transformer`` is passed, it must have been fit
    on training data only — a transformer fit on all rows leaks val/test
    information into the features.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    # Shape / dimensionality validation (fail early with a clear message rather
    # than deep inside the transformer or the ridge solve).
    if X.ndim != 2:
        raise ValueError(
            f"X must be 2-D (N, D); got shape {X.shape}. Reshape a single "
            f"feature with X.reshape(-1, 1).")
    y = np.reshape(y, (-1,))
    N = len(X)
    if len(y) != N:
        raise ValueError(
            f"X and y length mismatch: X has {N} rows, y has {len(y)}.")

    if np.any(np.isnan(X)):
        raise ValueError("Input X contains NaN values. Clean or impute before calling preprocess_data.")
    if np.any(np.isnan(y)):
        raise ValueError("Target y contains NaN values. Clean or impute before calling preprocess_data.")
    if not np.all(np.isfinite(X)):
        raise ValueError("Input X contains non-finite values (inf). Clean before calling preprocess_data.")
    if not np.all(np.isfinite(y)):
        raise ValueError("Target y contains non-finite values (inf). Clean before calling preprocess_data.")
    if val_fraction < 0.0 or test_fraction < 0.0:
        raise ValueError(
            f"val_fraction ({val_fraction}) and test_fraction ({test_fraction}) "
            f"must be non-negative.")
    if val_fraction + test_fraction >= 1.0:
        raise ValueError(f"val_fraction ({val_fraction}) + test_fraction ({test_fraction}) must be < 1.0")

    rng = np.random.RandomState(seed)
    indices = rng.permutation(N)

    n_test = int(N * test_fraction)
    n_val = int(N * val_fraction)
    n_train = N - n_test - n_val

    # Every split must be non-empty: the transformer fits on train, and
    # val/test drive model selection and reported R^2 downstream.
    if n_train < 2 or n_val < 1 or n_test < 1:
        raise ValueError(
            f"Not enough samples: N={N} with val_fraction={val_fraction}, "
            f"test_fraction={test_fraction} yields train/val/test = "
            f"{n_train}/{n_val}/{n_test}. Provide more data or smaller fractions.")

    # A constant target makes the standardization and R^2 ill-defined.
    if np.ptp(y) == 0.0:
        raise ValueError(
            "Target y is constant; there is nothing to fit and R^2/standardization "
            "are undefined.")

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    # Discrete / heavily-tied columns cannot be mapped to uniform marginals:
    # the quantile transform sends each tied value to a single atom, while
    # every analytic Gram variance (and hence every Sobol index) assumes
    # continuous U[0,1] marginals. Flag such columns rather than fail —
    # predictions remain usable, but their Sobol attributions are unreliable.
    tied_cols = []
    for j in range(X_train.shape[1]):
        _vals, _counts = np.unique(X_train[:, j], return_counts=True)
        if (len(_vals) < _TIES_MIN_UNIQUE
                or _counts.max() > _TIES_MAX_SHARE * n_train):
            tied_cols.append(j)
    if tied_cols:
        warnings.warn(
            f"Feature column(s) {tied_cols} are discrete or heavily tied "
            f"(< {_TIES_MIN_UNIQUE} distinct training values, or one value in "
            f"> {_TIES_MAX_SHARE:.0%} of rows). The quantile transform cannot "
            "make their marginals uniform, so the analytic Sobol indices for "
            "these variables are unreliable; treat their attributions with "
            "caution.", UserWarning, stacklevel=2)

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
        'x_train': jnp.array(X_train_t, dtype=fit_dtype),
        'y_train': jnp.array(y_train, dtype=fit_dtype),
        'x_val': jnp.array(X_val_t, dtype=fit_dtype),
        'y_val': jnp.array(y_val, dtype=fit_dtype),
        'x_test': jnp.array(X_test_t, dtype=fit_dtype),
        'y_test': jnp.array(y_test, dtype=fit_dtype),
        'transformer': transformer,
        'y_mean': y_mean,
        'y_std': y_std,
        'n_train': n_train,
        'n_val': n_val,
        'n_test': n_test,
        # Original-dataset row indices per split, in the row order of the
        # arrays above (BR-07). ``X[train_indices]`` == the training rows the
        # fitted design's Phi was built from, in Phi order.
        'train_indices': np.asarray(train_idx),
        'val_indices': np.asarray(val_idx),
        'test_indices': np.asarray(test_idx),
    }
