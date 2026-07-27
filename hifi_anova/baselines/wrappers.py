"""XGBoost baseline wrapper."""

import numpy as np


def train_xgboost_baseline(x_train, y_train, x_val, y_val, seed=42):
    """Train an XGBoost baseline with basic hyperparameter search.

    Args:
        x_train, y_train: training data
        x_val, y_val: validation data
        seed: random seed

    Returns:
        Fitted XGBoost model
    """
    try:
        from xgboost import XGBRegressor
    except ImportError:
        raise ImportError("XGBoost not installed. Run: pip install xgboost")

    model = XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=seed,
        early_stopping_rounds=20,
        eval_metric='rmse',
    )

    model.fit(
        np.array(x_train), np.array(y_train),
        eval_set=[(np.array(x_val), np.array(y_val))],
        verbose=False,
    )

    return model
