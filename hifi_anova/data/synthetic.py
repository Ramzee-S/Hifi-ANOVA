"""Synthetic data generators: Friedman-1 and heteroscedastic variants."""

import numpy as np
from typing import Optional, Tuple


def generate_friedman1(n_samples: int = 10000, noise_std: float = 0.1,
                       n_irrelevant: int = 5, seed: int = 42
                       ) -> Tuple[np.ndarray, np.ndarray]:
    """Generate Friedman-1 regression data.

    f(x) = 10*sin(pi*x1*x2) + 20*(x3-0.5)^2 + 10*x4 + 5*x5

    Active variables: x1..x5 (0-indexed: 0..4)
    Irrelevant variables: x6..x(5+n_irrelevant)

    All inputs are uniform on [0, 1].

    Args:
        n_samples: number of samples
        noise_std: standard deviation of Gaussian noise
        n_irrelevant: number of irrelevant variables to append
        seed: random seed

    Returns:
        X: (n_samples, 5 + n_irrelevant)
        y: (n_samples,)
    """
    rng = np.random.RandomState(seed)

    D_total = 5 + n_irrelevant
    X = rng.uniform(0, 1, size=(n_samples, D_total))

    # Friedman-1 function
    y = (10.0 * np.sin(np.pi * X[:, 0] * X[:, 1])
         + 20.0 * (X[:, 2] - 0.5) ** 2
         + 10.0 * X[:, 3]
         + 5.0 * X[:, 4])

    # Add noise
    if noise_std > 0:
        y += rng.normal(0, noise_std, size=n_samples)

    return X, y


def generate_heteroscedastic(n_samples: int = 10000,
                             noise_variable: int = 2,
                             seed: int = 42
                             ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate heteroscedastic synthetic data.

    Mean: f(x) = 10*sin(pi*x1*x2) + 20*(x3-0.5)^2 + 10*x4 + 5*x5
    Variance: sigma^2(x) = (0.5 + 2*x_{noise_variable})^2

    The variance depends on one variable, providing a known ground truth
    for the variance Sobol indices.

    Args:
        n_samples: number of samples
        noise_variable: which variable drives the variance (0-indexed)
        seed: random seed

    Returns:
        X: (n_samples, 10)
        y: (n_samples,)
        sigma_true: (n_samples,) true standard deviations
    """
    rng = np.random.RandomState(seed)

    D_total = 10
    X = rng.uniform(0, 1, size=(n_samples, D_total))

    # Mean function (Friedman-1)
    mean = (10.0 * np.sin(np.pi * X[:, 0] * X[:, 1])
            + 20.0 * (X[:, 2] - 0.5) ** 2
            + 10.0 * X[:, 3]
            + 5.0 * X[:, 4])

    # Input-dependent standard deviation
    sigma_true = 0.5 + 2.0 * X[:, noise_variable]

    # Heteroscedastic noise
    y = mean + sigma_true * rng.normal(0, 1, size=n_samples)

    return X, y, sigma_true
