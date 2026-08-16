"""Synthetic data generators: Friedman-1, Ishigami, and heteroscedastic variants."""

import numpy as np
from typing import Dict, Tuple


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
    for the log-variance indices S^h.

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


def friedman1_sobol_indices(n_quad: int = 64) -> Dict:
    """Exact Sobol indices for Friedman-1, computed to quadrature precision.

    Ground truth for ``generate_friedman1``. The function on ``x ~ U[0,1]^d`` is

        f(x) = 10 sin(pi x1 x2) + 20 (x3 - 1/2)^2 + 10 x4 + 5 x5

    (x6.. are inert). The four terms depend on disjoint variable sets, so the
    total variance is the sum of their variances. Closed forms are used where
    available; the ``sin(pi x1 x2)`` term (the only interaction) is integrated
    by tensor Gauss-Legendre quadrature, which is exact to machine precision for
    this smooth integrand.

    Component variances (uniform measure on the cube):

        Var(20 (x3-1/2)^2) = 400 (1/80 - 1/144) = 20/9
        Var(10 x4)         = 100/12 ,   Var(5 x5) = 25/12
        Var(10 sin(pi x1 x2)) = E[.^2] - E[.]^2        (2-D quadrature)
        V1 = Var_{x1}( E_{x2}[10 sin(pi x1 x2)] ) = V2   (main effects, 1-D)
        V12 = Var(10 sin(pi x1 x2)) - V1 - V2            (the (x1,x2) interaction)

    This replaces the approximate literature/SALib values: the exact first-order
    indices are S1=S2=0.19731, S3=0.09327, S4=0.34975, S5=0.08744 and the
    closed pair S12=0.07492 (they sum to 1 with the inert variables at 0).

    Args:
        n_quad: Gauss-Legendre nodes per axis (64 is already machine-exact here).

    Returns:
        dict with 'first_order' {i: S_i} for i=0..4 (inert 5..9 omitted; == 0),
        'second_order' {(0, 1): S_12}, 'partial_variances', and 'total_variance'.
    """
    nodes, w = np.polynomial.legendre.leggauss(n_quad)
    x = 0.5 * (nodes + 1.0)                    # map [-1,1] -> [0,1]
    wq = 0.5 * w                               # quadrature weights on [0,1]

    # Interaction term A = 10 sin(pi x1 x2): 2-D tensor quadrature.
    X1, X2 = np.meshgrid(x, x, indexing='ij')
    W = np.outer(wq, wq)
    A = 10.0 * np.sin(np.pi * X1 * X2)
    EA = float(np.sum(W * A))
    EA2 = float(np.sum(W * A * A))
    var_A = EA2 - EA ** 2
    h1 = (A * wq[None, :]).sum(axis=1)         # E_{x2}[A] as a function of x1
    V1 = float(np.sum(wq * (h1 - EA) ** 2))    # first-order (main) effect of x1
    V2 = V1                                    # symmetric in x1, x2
    V12 = var_A - V1 - V2                       # the (x1, x2) interaction

    # Additive univariate terms (closed form).
    V3 = 400.0 * (1.0 / 80.0 - 1.0 / 144.0)    # Var(20 (x3-1/2)^2) = 20/9
    V4 = 100.0 / 12.0                          # Var(10 x4)
    V5 = 25.0 / 12.0                           # Var(5 x5)

    D = var_A + V3 + V4 + V5                   # total variance
    return {
        'first_order': {0: V1 / D, 1: V2 / D, 2: V3 / D, 3: V4 / D, 4: V5 / D},
        'second_order': {(0, 1): V12 / D},
        'partial_variances': {'V1': V1, 'V2': V2, 'V3': V3, 'V4': V4, 'V5': V5,
                              'V12': V12},
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
    for the dual mean/log-variance spectrum.

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
