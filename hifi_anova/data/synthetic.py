"""Synthetic data generators: Friedman-1, Ishigami, and heteroscedastic variants."""

import numpy as np
from typing import Dict, Optional, Tuple


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


def ishigami_sobol_indices(a: float = 7.0, b: float = 0.1) -> Dict:
    """Analytic Sobol indices for the Ishigami function.

    Ground truth for ``generate_ishigami``. The classic closed forms (Sobol &
    Levitan; Marrel et al.) for ``x_i ~ U(-pi, pi)``:

        D    = a^2/8 + b*pi^4/5 + b^2*pi^8/18 + 1/2      (total variance)
        D1   = 1/2 + b*pi^4/5 + b^2*pi^8/50               (x1 main effect)
        D2   = a^2/8                                      (x2 main effect)
        D3   = 0                                          (x3 has NO main effect)
        D13  = b^2*pi^8*(1/18 - 1/50) = 8*b^2*pi^8/225    (x1-x3 interaction)

    x3 is the textbook case of a variable with zero first-order index but a
    non-zero total-order index: it acts purely through its interaction with x1.

    Returns:
        dict with 'first_order' {i: S_i}, 'total_order' {i: S_T_i},
        'partial_variances' {'D1','D2','D13'}, and 'total_variance'.
    """
    pi4 = np.pi ** 4
    pi8 = np.pi ** 8

    D1 = 0.5 + b * pi4 / 5.0 + (b ** 2) * pi8 / 50.0
    D2 = a ** 2 / 8.0
    D13 = 8.0 * (b ** 2) * pi8 / 225.0
    D = D1 + D2 + D13  # = a^2/8 + b*pi^4/5 + b^2*pi^8/18 + 1/2

    return {
        'first_order': {0: D1 / D, 1: D2 / D, 2: 0.0},
        'total_order': {0: (D1 + D13) / D, 1: D2 / D, 2: D13 / D},
        'partial_variances': {'D1': D1, 'D2': D2, 'D13': D13},
        'total_variance': D,
    }


def generate_ishigami(n_samples: int = 10000,
                      a: float = 7.0, b: float = 0.1,
                      noise_std: float = 0.0,
                      heteroscedastic: bool = False,
                      variance_variable: int = 2,
                      sigma_min: float = 0.3, sigma_max: float = 3.0,
                      seed: int = 42
                      ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate Ishigami data, optionally heteroscedastic.

    f(x) = sin(x1) + a*sin^2(x2) + b*x3^4*sin(x1),   x_i ~ U(-pi, pi)

    A canonical sensitivity-analysis benchmark with known analytic Sobol
    indices (see :func:`ishigami_sobol_indices`). Its signature feature: x3 has
    a *zero first-order* effect but a *non-zero total-order* effect — it acts
    only through the x1-x3 interaction. This makes it a clean test that
    total-order attribution catches structure that first-order attribution
    misses.

    Heteroscedastic mode (``heteroscedastic=True``) adds input-dependent noise
    whose standard deviation ramps linearly from ``sigma_min`` to ``sigma_max``
    across ``variance_variable`` (default x3). Driving the variance with x3 — a
    variable that is first-order-silent in the mean — makes it a **hidden
    heteroscedastic driver**: invisible to mean first-order importance yet
    dominating the predictive *variance*. This is the canonical demonstration
    for the dual mean+variance Sobol spectrum.

    Args:
        n_samples: number of samples.
        a, b: Ishigami coefficients (defaults a=7, b=0.1).
        noise_std: homoscedastic Gaussian noise std (ignored if heteroscedastic).
        heteroscedastic: if True, add input-dependent noise via variance_variable.
        variance_variable: which variable drives the variance (0-indexed; default x3).
        sigma_min, sigma_max: noise-std ramp endpoints in heteroscedastic mode.
        seed: random seed.

    Returns:
        X: (n_samples, 3) inputs on [-pi, pi]^3.
        y: (n_samples,) responses.
        sigma_true: (n_samples,) true noise std per sample (zeros if noiseless).
    """
    rng = np.random.RandomState(seed)
    X = rng.uniform(-np.pi, np.pi, size=(n_samples, 3))
    x1, x2, x3 = X[:, 0], X[:, 1], X[:, 2]

    f = np.sin(x1) + a * np.sin(x2) ** 2 + b * (x3 ** 4) * np.sin(x1)

    if heteroscedastic:
        # Map the driving variable to [0, 1] and ramp the noise std over it.
        u = (X[:, variance_variable] + np.pi) / (2.0 * np.pi)
        sigma_true = sigma_min + (sigma_max - sigma_min) * u
    else:
        sigma_true = np.full(n_samples, noise_std, dtype=float)

    y = f + sigma_true * rng.normal(0, 1, size=n_samples)
    return X, y, sigma_true
