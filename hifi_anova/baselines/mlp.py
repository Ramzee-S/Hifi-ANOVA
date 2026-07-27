"""Standard MLP baseline using sklearn."""

from sklearn.neural_network import MLPRegressor
import numpy as np


def train_mlp_baseline(x_train, y_train, x_val, y_val,
                       hidden_layer_sizes=(256, 256, 256),
                       max_iter=500, seed=42):
    """Train a standard MLP baseline.

    Args:
        x_train, y_train: training data (numpy arrays)
        x_val, y_val: validation (for early stopping)
        hidden_layer_sizes: MLP architecture
        max_iter: max training epochs
        seed: random seed

    Returns:
        Fitted MLPRegressor
    """
    mlp = MLPRegressor(
        hidden_layer_sizes=hidden_layer_sizes,
        activation='relu',
        solver='adam',
        max_iter=max_iter,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=seed,
        learning_rate_init=0.001,
    )

    # Combine train and val for sklearn's internal early stopping
    X = np.vstack([np.array(x_train), np.array(x_val)])
    y = np.concatenate([np.array(y_train), np.array(y_val)])

    mlp.fit(X, y)
    return mlp
