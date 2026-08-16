"""Tier 1-4 Test Functions for HiFiANOVA validation.

Each function returns (X, y, ground_truth) where ground_truth contains
analytic Sobol indices, true coefficients, and variance structure.
"""

import numpy as np
from typing import Dict, Tuple, Optional


# =============================================================================
# TIER 1: Unit-Level Validation Functions (exact analytic ground truth)
# =============================================================================

def T1_1_pure_linear(n_samples: int = 10000, noise_std: float = 0.1,
                     seed: int = 42) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """T1.1: Pure Linear
    f(x) = 5*(x1 - 0.5) + 3*(x2 - 0.5)
    D = 5, x3-x5 irrelevant

    Tests: linear basis term, coefficient recovery, basic Sobol.
    """
    rng = np.random.RandomState(seed)
    D = 5
    X = rng.uniform(0, 1, (n_samples, D))

    f = 5.0 * (X[:, 0] - 0.5) + 3.0 * (X[:, 1] - 0.5)
    y = f + noise_std * rng.randn(n_samples)

    # Analytic ground truth
    # Var(5*(x-0.5)) = 25 * Var(x-0.5) = 25/12
    # Var(3*(x-0.5)) = 9/12
    var1 = 25.0 / 12.0
    var2 = 9.0 / 12.0
    total_var = var1 + var2

    ground_truth = {
        'name': 'T1.1_pure_linear',
        'D': D,
        'mean_sobol_first_order': {0: var1/total_var, 1: var2/total_var,
                                    2: 0.0, 3: 0.0, 4: 0.0},
        'mean_sobol_second_order': {},
        'total_signal_variance': total_var,
        'noise_variance': noise_std**2,
        'coefficients': {
            'linear': {0: 5.0, 1: 3.0},
            'fourier': {},
        },
        'description': 'f(x) = 5*(x1-0.5) + 3*(x2-0.5)',
    }
    return X, y, ground_truth


def T1_2_pure_fourier(n_samples: int = 10000, noise_std: float = 0.1,
                      seed: int = 42) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """T1.2: Pure Fourier
    f(x) = 3*cos(2*pi*x1) + 2*sin(4*pi*x2)
    D = 5, x3-x5 irrelevant

    Tests: Fourier coefficient recovery.
    """
    rng = np.random.RandomState(seed)
    D = 5
    X = rng.uniform(0, 1, (n_samples, D))

    f = 3.0 * np.cos(2 * np.pi * X[:, 0]) + 2.0 * np.sin(4 * np.pi * X[:, 1])
    y = f + noise_std * rng.randn(n_samples)

    # Var(3*cos(2*pi*x)) = 9 * integral(cos^2) = 9 * 1/2 = 4.5
    # Var(2*sin(4*pi*x)) = 4 * 1/2 = 2.0
    var1 = 9.0 / 2.0  # 4.5
    var2 = 4.0 / 2.0  # 2.0
    total_var = var1 + var2

    ground_truth = {
        'name': 'T1.2_pure_fourier',
        'D': D,
        'mean_sobol_first_order': {0: var1/total_var, 1: var2/total_var,
                                    2: 0.0, 3: 0.0, 4: 0.0},
        'mean_sobol_second_order': {},
        'total_signal_variance': total_var,
        'noise_variance': noise_std**2,
        'coefficients': {
            'linear': {},
            'fourier': {(0, 'cos', 1): 3.0, (1, 'sin', 2): 2.0},
        },
        'description': 'f(x) = 3*cos(2*pi*x1) + 2*sin(4*pi*x2)',
    }
    return X, y, ground_truth


def T1_3_linear_fourier_mix(n_samples: int = 10000, noise_std: float = 0.1,
                            seed: int = 42) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """T1.3: Linear + Fourier Mix (THE critical test)
    f(x) = 4*(x1 - 0.5) + 2*sin(2*pi*x1)
    D = 3

    Tests the Gram matrix cross-term. Variable x1 has BOTH linear and sin1.
    The variance is NOT 4^2/12 + 2^2/2. It's w^T G w with the cross term.
    """
    rng = np.random.RandomState(seed)
    D = 3
    X = rng.uniform(0, 1, (n_samples, D))

    f = 4.0 * (X[:, 0] - 0.5) + 2.0 * np.sin(2 * np.pi * X[:, 0])
    y = f + noise_std * rng.randn(n_samples)

    # w = [4, 0, 2, 0, 0, ...] for variable 0 (linear=4, cos1=0, sin1=2)
    # G[0,0] = 1/12, G[2,2] = 1/2, G[0,2] = -1/(2*pi)
    # Var = w^T G w = 4^2*(1/12) + 2^2*(1/2) + 2*4*2*(-1/(2*pi))
    #     = 16/12 + 4/2 + 16*(-1/(2*pi))
    #     = 1.3333 + 2.0 - 2.5465 = 0.7869
    var1 = 16.0/12.0 + 4.0/2.0 + 2.0 * 4.0 * 2.0 * (-1.0/(2.0*np.pi))

    # WRONG answer (diagonal-only): 16/12 + 4/2 = 3.333
    wrong_var = 16.0/12.0 + 4.0/2.0

    ground_truth = {
        'name': 'T1.3_linear_fourier_mix',
        'D': D,
        'mean_sobol_first_order': {0: 1.0, 1: 0.0, 2: 0.0},
        'mean_sobol_second_order': {},
        'total_signal_variance': var1,
        'wrong_diagonal_variance': wrong_var,
        'noise_variance': noise_std**2,
        'coefficients': {
            'linear': {0: 4.0},
            'fourier': {(0, 'sin', 1): 2.0},
        },
        'gram_cross_term': -1.0/(2.0*np.pi),
        'description': ('f(x) = 4*(x1-0.5) + 2*sin(2*pi*x1). '
                       f'Correct Var={var1:.4f}, Wrong (diagonal)={wrong_var:.4f}'),
    }
    return X, y, ground_truth


def T1_4_pure_interaction(n_samples: int = 10000, noise_std: float = 0.1,
                          seed: int = 42) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """T1.4: Pure Interaction
    f(x) = 3*(x1 - 0.5)*(x2 - 0.5)
    D = 5, x3-x5 irrelevant

    Tests: second-order term. First-order Sobol should be zero.
    """
    rng = np.random.RandomState(seed)
    D = 5
    X = rng.uniform(0, 1, (n_samples, D))

    f = 3.0 * (X[:, 0] - 0.5) * (X[:, 1] - 0.5)
    y = f + noise_std * rng.randn(n_samples)

    # Var((x1-0.5)*(x2-0.5)) = E[(x1-0.5)^2]*E[(x2-0.5)^2] = (1/12)*(1/12) = 1/144
    # Var(f) = 9 * 1/144 = 9/144 = 1/16
    var_12 = 9.0 / 144.0

    ground_truth = {
        'name': 'T1.4_pure_interaction',
        'D': D,
        'mean_sobol_first_order': {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0},
        'mean_sobol_second_order': {(0, 1): 1.0},
        'total_signal_variance': var_12,
        'noise_variance': noise_std**2,
        'description': 'f(x) = 3*(x1-0.5)*(x2-0.5)',
    }
    return X, y, ground_truth


def T1_5_constant_mean_variable_noise(
    n_samples: int = 10000, seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """T1.5: Constant Mean, Variable Noise
    f(x) = 0
    sigma^2(x) = exp(2*(x1 - 0.5))
    D = 5

    All action is in the variance. Mean Sobol should be ~zero.
    Log-variance index: S1_h should be large.
    """
    rng = np.random.RandomState(seed)
    D = 5
    X = rng.uniform(0, 1, (n_samples, D))

    # Mean is zero
    log_var = 2.0 * (X[:, 0] - 0.5)
    sigma = np.exp(log_var / 2.0)
    y = sigma * rng.randn(n_samples)

    # The true log-variance function h(x) = 2*(x1-0.5)
    # This is a pure linear function of x1 in our basis
    # Log-variance index: h depends only on x1, so S1_h = 1.0
    ground_truth = {
        'name': 'T1.5_constant_mean_variable_noise',
        'D': D,
        'mean_sobol_first_order': {i: 0.0 for i in range(D)},
        'variance_sobol_first_order': {0: 1.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0},
        'true_log_variance_function': 'h(x) = 2*(x1-0.5)',
        'true_h_coefficients': {'linear': {0: 2.0}},
        'description': 'f(x) = 0, sigma^2(x) = exp(2*(x1-0.5))',
    }
    return X, y, ground_truth


# =============================================================================
# TIER 2: Integration-Level Test Functions
# =============================================================================

def T2_1_friedman1(n_samples: int = 10000, noise_std: float = 1.0,
                   seed: int = 42) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """T2.1: Modified Friedman-1 (Standard benchmark)
    f(x) = 10*sin(pi*x1*x2) + 20*(x3-0.5)^2 + 10*x4 + 5*x5
    D = 10, x6-x10 irrelevant

    Note: sin(pi*x1*x2) is NOT in our product Fourier basis.
    Tests how well truncated Fourier handles out-of-basis functions.
    """
    rng = np.random.RandomState(seed)
    D = 10
    X = rng.uniform(0, 1, (n_samples, D))

    f = (10.0 * np.sin(np.pi * X[:, 0] * X[:, 1])
         + 20.0 * (X[:, 2] - 0.5)**2
         + 10.0 * X[:, 3]
         + 5.0 * X[:, 4])
    y = f + noise_std * rng.randn(n_samples)

    # Exact Sobol indices, computed to quadrature precision (single source of
    # truth: hifi_anova.data.synthetic.friedman1_sobol_indices). Replaces the
    # earlier approximate literature values, which were off (e.g. S4 was 0.14 vs
    # exact 0.350, and the (x1,x2) pair was 0.24 vs exact 0.075).
    from .synthetic import friedman1_sobol_indices
    _exact = friedman1_sobol_indices()
    first_order = {i: _exact['first_order'].get(i, 0.0) for i in range(D)}
    ground_truth = {
        'name': 'T2.1_friedman1',
        'D': D,
        'mean_sobol_first_order': first_order,
        'mean_sobol_second_order': dict(_exact['second_order']),
        'total_variance': _exact['total_variance'],
        'noise_variance': noise_std**2,
        'tolerance': 0.15,  # Allow 15% relative error in recovery checks
        'description': ('f(x) = 10*sin(pi*x1*x2) + 20*(x3-0.5)^2 + 10*x4 + 5*x5. '
                       'sin(pi*x1*x2) is NOT in our basis.'),
    }
    return X, y, ground_truth


def T2_2_smooth_additive(n_samples: int = 10000, noise_std: float = 0.5,
                         seed: int = 42) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """T2.2: Smooth Additive (Fourier-Friendly)
    f(x) = 3*cos(2*pi*x1) + 2*sin(4*pi*x2) + 1.5*cos(2*pi*x3)*cos(2*pi*x4) + 4*(x5-0.5)
    D = 8, x6-x8 irrelevant

    This function IS exactly in our basis. Sobol indices should be recovered exactly.
    """
    rng = np.random.RandomState(seed)
    D = 8
    X = rng.uniform(0, 1, (n_samples, D))

    f = (3.0 * np.cos(2 * np.pi * X[:, 0])
         + 2.0 * np.sin(4 * np.pi * X[:, 1])
         + 1.5 * np.cos(2 * np.pi * X[:, 2]) * np.cos(2 * np.pi * X[:, 3])
         + 4.0 * (X[:, 4] - 0.5))
    y = f + noise_std * rng.randn(n_samples)

    # Exact Sobol indices:
    # Var(f1) = 9/2 = 4.5 (3^2 * 1/2)
    # Var(f2) = 4/2 = 2.0 (2^2 * 1/2)
    # Var(f34) = 1.5^2 * (1/2) * (1/2) = 0.5625
    # Var(f5) = 16/12 = 1.3333 (4^2 * 1/12)
    var1 = 9.0 / 2.0        # 4.5
    var2 = 4.0 / 2.0        # 2.0
    var34 = 2.25 * 0.25     # 0.5625
    var5 = 16.0 / 12.0      # 1.3333
    total = var1 + var2 + var34 + var5  # 8.3958

    ground_truth = {
        'name': 'T2.2_smooth_additive',
        'D': D,
        'mean_sobol_first_order': {
            0: var1/total, 1: var2/total, 2: 0.0, 3: 0.0,
            4: var5/total, 5: 0.0, 6: 0.0, 7: 0.0
        },
        'mean_sobol_second_order': {(2, 3): var34/total},
        'total_signal_variance': total,
        'noise_variance': noise_std**2,
        'per_component_variance': {
            'f1 (cos 2pi x1)': var1,
            'f2 (sin 4pi x2)': var2,
            'f34 (cos*cos interaction)': var34,
            'f5 (linear x5)': var5,
        },
        'tolerance': 0.05,  # Should be very accurate since function is in basis
        'description': ('f = 3*cos(2*pi*x1) + 2*sin(4*pi*x2) + '
                       '1.5*cos(2*pi*x3)*cos(2*pi*x4) + 4*(x5-0.5). '
                       'Exactly in the Fourier basis.'),
    }
    return X, y, ground_truth


# =============================================================================
# TIER 3: Heteroscedastic Test Functions
# =============================================================================

def T3_1_orthogonal_mean_variance(
    n_samples: int = 10000, seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """T3.1: Orthogonal Mean-Variance ("Easy Case")
    f(x) = 5*(x1-0.5) + 3*cos(2*pi*x2)
    sigma^2(x) = exp(2*(x3-0.5))
    D = 6, x4-x6 irrelevant

    Mean depends on x1,x2. Variance depends on x3. No overlap.
    """
    rng = np.random.RandomState(seed)
    D = 6
    X = rng.uniform(0, 1, (n_samples, D))

    # Mean function
    f = 5.0 * (X[:, 0] - 0.5) + 3.0 * np.cos(2 * np.pi * X[:, 1])

    # Variance function
    log_var = 2.0 * (X[:, 2] - 0.5)
    sigma = np.exp(log_var / 2.0)
    y = f + sigma * rng.randn(n_samples)

    # Mean Sobol: Var(f1) = 25/12, Var(f2) = 9/2
    var1 = 25.0 / 12.0
    var2 = 9.0 / 2.0
    total_mean = var1 + var2

    ground_truth = {
        'name': 'T3.1_orthogonal_mean_variance',
        'D': D,
        'mean_sobol_first_order': {
            0: var1/total_mean, 1: var2/total_mean,
            2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0
        },
        'variance_sobol_first_order': {
            0: 0.0, 1: 0.0, 2: 1.0, 3: 0.0, 4: 0.0, 5: 0.0
        },
        'total_mean_variance': total_mean,
        'true_log_variance': 'h(x) = 2*(x3-0.5)',
        'description': ('f = 5*(x1-0.5) + 3*cos(2*pi*x2), '
                       'sigma^2 = exp(2*(x3-0.5)). '
                       'Mean and variance have completely separate drivers.'),
    }
    return X, y, ground_truth


def T3_2_shared_variable(
    n_samples: int = 10000, seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """T3.2: Shared Variable ("Challenging Case")
    f(x) = 5*(x1-0.5) + 3*(x1-0.5)*(x2-0.5)
    sigma^2(x) = exp(1.5*(x1-0.5))
    D = 5

    x1 affects BOTH mean (main + interaction) and variance.
    """
    rng = np.random.RandomState(seed)
    D = 5
    X = rng.uniform(0, 1, (n_samples, D))

    # Mean
    f = 5.0 * (X[:, 0] - 0.5) + 3.0 * (X[:, 0] - 0.5) * (X[:, 1] - 0.5)

    # Variance
    log_var = 1.5 * (X[:, 0] - 0.5)
    sigma = np.exp(log_var / 2.0)
    y = f + sigma * rng.randn(n_samples)

    # Mean Sobol:
    # Var(main x1) = 25/12
    # Var(interaction x1*x2) = 9 * (1/12)*(1/12) = 9/144 = 1/16
    var1_main = 25.0 / 12.0
    var12 = 9.0 / 144.0
    total_mean = var1_main + var12

    ground_truth = {
        'name': 'T3.2_shared_variable',
        'D': D,
        'mean_sobol_first_order': {
            0: var1_main/total_mean, 1: 0.0,
            2: 0.0, 3: 0.0, 4: 0.0
        },
        'mean_sobol_second_order': {(0, 1): var12/total_mean},
        'variance_sobol_first_order': {
            0: 1.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0
        },
        'total_mean_variance': total_mean,
        'true_log_variance': 'h(x) = 1.5*(x1-0.5)',
        'description': ('f = 5*(x1-0.5) + 3*(x1-0.5)*(x2-0.5), '
                       'sigma^2 = exp(1.5*(x1-0.5)). '
                       'x1 drives both mean AND variance.'),
    }
    return X, y, ground_truth


def T3_3_hidden_variable(
    n_samples: int = 10000, seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """T3.3: The Hidden Variable ("The Showcase")
    f(x) = 3*cos(2*pi*x1) + 2*(x2-0.5)
    sigma^2(x) = exp(x3 + 0.5*sin(2*pi*x4) - 0.75)
    D = 8, x5-x8 irrelevant

    Mean depends on x1,x2. Variance depends on x3,x4.
    In standard analysis, x3 and x4 would appear unimportant!
    """
    rng = np.random.RandomState(seed)
    D = 8
    X = rng.uniform(0, 1, (n_samples, D))

    # Mean
    f = 3.0 * np.cos(2 * np.pi * X[:, 0]) + 2.0 * (X[:, 1] - 0.5)

    # Variance (depends on x3 and x4)
    log_var = X[:, 2] + 0.5 * np.sin(2 * np.pi * X[:, 3]) - 0.75
    sigma = np.exp(log_var / 2.0)
    y = f + sigma * rng.randn(n_samples)

    # Mean Sobol
    var1 = 9.0 / 2.0     # 3^2 * 1/2
    var2 = 4.0 / 12.0    # 2^2 * 1/12
    total_mean = var1 + var2

    # Log-variance index of h(x) = x3 + 0.5*sin(2*pi*x4) - 0.75
    # h depends on x3 (linear: coefficient 1 -> Var = 1/12)
    # and x4 (sin1: coefficient 0.5 -> Var = 0.25/2 = 0.125)
    var_h3 = 1.0 / 12.0
    var_h4 = 0.25 / 2.0
    total_h = var_h3 + var_h4

    ground_truth = {
        'name': 'T3.3_hidden_variable',
        'D': D,
        'mean_sobol_first_order': {
            0: var1/total_mean, 1: var2/total_mean,
            2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0
        },
        'variance_sobol_first_order': {
            0: 0.0, 1: 0.0,
            2: var_h3/total_h, 3: var_h4/total_h,
            4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0
        },
        'total_mean_variance': total_mean,
        'true_log_variance': 'h(x) = x3 + 0.5*sin(2*pi*x4) - 0.75',
        'showcase_message': ('Standard feature importance says x3/x4 are irrelevant. '
                            'HiFiANOVA log-variance indices identify fitted residual-scale drivers.'),
        'description': ('f = 3*cos(2*pi*x1) + 2*(x2-0.5), '
                       'sigma^2 = exp(x3 + 0.5*sin(2*pi*x4) - 0.75)'),
    }
    return X, y, ground_truth


def T3_4_interaction_noise(
    n_samples: int = 10000, seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """T3.4: Smooth Mean, Interaction Noise ("The Iceberg")
    f(x) = 4*(x1-0.5) + 2*(x2-0.5)
    sigma^2(x) = exp((x3-0.5)*(x4-0.5) * 8)
    D = 6

    Variance has a genuine INTERACTION between x3 and x4.
    """
    rng = np.random.RandomState(seed)
    D = 6
    X = rng.uniform(0, 1, (n_samples, D))

    # Mean
    f = 4.0 * (X[:, 0] - 0.5) + 2.0 * (X[:, 1] - 0.5)

    # Variance with interaction
    log_var = (X[:, 2] - 0.5) * (X[:, 3] - 0.5) * 8.0
    sigma = np.exp(log_var / 2.0)
    y = f + sigma * rng.randn(n_samples)

    # Mean Sobol
    var1 = 16.0 / 12.0
    var2 = 4.0 / 12.0
    total_mean = var1 + var2

    ground_truth = {
        'name': 'T3.4_interaction_noise',
        'D': D,
        'mean_sobol_first_order': {
            0: var1/total_mean, 1: var2/total_mean,
            2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0
        },
        'variance_has_interaction': True,
        'variance_interaction_variables': (2, 3),
        'total_mean_variance': total_mean,
        'true_log_variance': 'h(x) = 8*(x3-0.5)*(x4-0.5)',
        'description': ('f = 4*(x1-0.5) + 2*(x2-0.5), '
                       'sigma^2 = exp(8*(x3-0.5)*(x4-0.5)). '
                       'Variance has interaction.'),
    }
    return X, y, ground_truth


def T3_5_signal_noise_confusion(
    n_samples: int = 10000, seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """T3.5: Signal-Noise Confusion Stress Test
    f(x) = 5*sin(2*pi*x1) + 3*cos(4*pi*x2)
    sigma^2(x) = exp(2*cos(2*pi*x1))
    D = 5

    Variance has SAME frequency content as mean in x1 (but orthogonal phase).
    Mean: sin(2*pi*x1), Variance: cos(2*pi*x1).
    """
    rng = np.random.RandomState(seed)
    D = 5
    X = rng.uniform(0, 1, (n_samples, D))

    # Mean
    f = 5.0 * np.sin(2 * np.pi * X[:, 0]) + 3.0 * np.cos(4 * np.pi * X[:, 1])

    # Variance (same frequency as mean in x1, but cos instead of sin)
    log_var = 2.0 * np.cos(2 * np.pi * X[:, 0])
    sigma = np.exp(log_var / 2.0)
    y = f + sigma * rng.randn(n_samples)

    # Mean Sobol
    var1 = 25.0 / 2.0   # 5^2 * 1/2
    var2 = 9.0 / 2.0    # 3^2 * 1/2
    total_mean = var1 + var2

    # Log-variance index: h(x) = 2*cos(2*pi*x1), depends only on x1
    ground_truth = {
        'name': 'T3.5_signal_noise_confusion',
        'D': D,
        'mean_sobol_first_order': {
            0: var1/total_mean, 1: var2/total_mean,
            2: 0.0, 3: 0.0, 4: 0.0
        },
        'variance_sobol_first_order': {
            0: 1.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0
        },
        'total_mean_variance': total_mean,
        'true_log_variance': 'h(x) = 2*cos(2*pi*x1)',
        'stress_test_note': ('Mean has sin(2*pi*x1), variance has cos(2*pi*x1). '
                            'Same frequency, orthogonal phase. '
                            'Model must separate signal from noise structure.'),
        'description': ('f = 5*sin(2*pi*x1) + 3*cos(4*pi*x2), '
                       'sigma^2 = exp(2*cos(2*pi*x1))'),
    }
    return X, y, ground_truth


# =============================================================================
# TIER 4: Controlled Complexity Functions
# =============================================================================

def T4_1_tunable_interaction(
    n_samples: int = 10000, alpha: float = 1.0, D: int = 5,
    noise_std: float = 0.1, seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """T4.1: Tunable Interaction Strength
    f(x) = sum_i a_i*(x_i-0.5) + alpha * sum_{i<j} b_ij*(x_i-0.5)*(x_j-0.5)

    alpha=0: purely additive. alpha=1: full interactions.
    """
    rng = np.random.RandomState(seed)
    X = rng.uniform(0, 1, (n_samples, D))

    # Main effects with varying strength
    a = np.array([5.0, 3.0, 2.0, 1.0, 0.5][:D])
    f = sum(a[i] * (X[:, i] - 0.5) for i in range(D))

    # Interactions
    b = {}
    interaction_var = 0.0
    for i in range(D):
        for j in range(i+1, D):
            b_ij = 2.0 / (1 + i + j)  # Decreasing strength
            b[(i, j)] = b_ij
            f += alpha * b_ij * (X[:, i] - 0.5) * (X[:, j] - 0.5)
            interaction_var += (alpha * b_ij)**2 / 144.0

    y = f + noise_std * rng.randn(n_samples)

    # Main effect variances
    main_vars = {i: a[i]**2 / 12.0 for i in range(D)}
    total_main = sum(main_vars.values())
    total_var = total_main + interaction_var

    ground_truth = {
        'name': f'T4.1_tunable_interaction_alpha={alpha}',
        'D': D,
        'alpha': alpha,
        'main_effect_fraction': total_main / total_var if total_var > 0 else 1.0,
        'interaction_fraction': interaction_var / total_var if total_var > 0 else 0.0,
        'total_signal_variance': total_var,
        'noise_variance': noise_std**2,
        'description': f'Tunable interaction alpha={alpha}. '
                      f'Main/Total={total_main/total_var:.3f}',
    }
    return X, y, ground_truth


def T4_2_tunable_frequency(
    n_samples: int = 10000, K_true: int = 5, D: int = 3,
    noise_std: float = 0.1, seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """T4.2: Tunable Frequency Complexity
    f(x) = sum_i sum_{k=1}^{K_true} (1/k) * sin(2*pi*k*x_i)

    K_true controls the true number of harmonics.
    Tests truncation (K1 < K_true) and suppression (K1 > K_true).
    """
    rng = np.random.RandomState(seed)
    X = rng.uniform(0, 1, (n_samples, D))

    f = np.zeros(n_samples)
    per_var_variance = {}

    for i in range(D):
        var_i = 0.0
        for k in range(1, K_true + 1):
            coeff = 1.0 / k
            f += coeff * np.sin(2 * np.pi * k * X[:, i])
            var_i += coeff**2 / 2.0  # Var(c*sin(2*pi*k*x)) = c^2/2
        per_var_variance[i] = var_i

    y = f + noise_std * rng.randn(n_samples)
    total_var = sum(per_var_variance.values())

    ground_truth = {
        'name': f'T4.2_tunable_frequency_K={K_true}',
        'D': D,
        'K_true': K_true,
        'per_variable_variance': per_var_variance,
        'total_signal_variance': total_var,
        'mean_sobol_first_order': {i: per_var_variance[i]/total_var for i in range(D)},
        'noise_variance': noise_std**2,
        'description': f'sum(1/k * sin(2*pi*k*x)) for k=1..{K_true}',
    }
    return X, y, ground_truth


def T4_3_tunable_snr(
    n_samples: int = 10000, beta: float = 1.0,
    noise_variable: int = 0, seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """T4.3: Tunable Signal-to-Noise Ratio
    f(x) = 3*cos(2*pi*x1) + 2*(x2-0.5)
    sigma^2(x) = 0.5 * exp(beta * (x_{noise_var} - 0.5))

    beta controls severity of heteroscedasticity:
    beta=0: homoscedastic, beta=2: 7x noise variation.
    """
    rng = np.random.RandomState(seed)
    D = 5
    X = rng.uniform(0, 1, (n_samples, D))

    # Mean
    f = 3.0 * np.cos(2 * np.pi * X[:, 0]) + 2.0 * (X[:, 1] - 0.5)

    # Variance
    log_var = np.log(0.5) + beta * (X[:, noise_variable] - 0.5)
    sigma = np.exp(log_var / 2.0)
    y = f + sigma * rng.randn(n_samples)

    # Mean Sobol
    var1 = 9.0 / 2.0
    var2 = 4.0 / 12.0
    total_mean = var1 + var2

    ground_truth = {
        'name': f'T4.3_tunable_snr_beta={beta}',
        'D': D,
        'beta': beta,
        'noise_variable': noise_variable,
        'mean_sobol_first_order': {
            0: var1/total_mean, 1: var2/total_mean,
            2: 0.0, 3: 0.0, 4: 0.0
        },
        'variance_sobol_first_order': {
            i: (1.0 if i == noise_variable else 0.0) for i in range(D)
        } if beta > 0 else {i: 0.0 for i in range(D)},
        # max/min noise VARIANCE ratio e^beta (the sd ratio is e^{beta/2})
        'noise_ratio': float(np.exp(beta)),
        'description': (f'beta={beta}, noise variance varies by '
                        f'{np.exp(beta):.1f}x (sd by {np.exp(beta / 2.0):.1f}x)'),
    }
    return X, y, ground_truth


# =============================================================================
# Classic sensitivity-analysis benchmarks (external ground truth)
# =============================================================================
# The Sobol G-function and a simplified Morris-like function are standard SA
# test problems. Unlike the Tier 1-4 functions above they take an ``X`` array
# (inputs on the unit hypercube) and return ``y`` only; their reference indices
# are analytic (G-function) or qualitative (Morris-like: which variables are
# active; see the morris_function docstring for its simplifications vs. the
# genuine Morris 1991 function).
# Ishigami lives in :mod:`hifi_anova.data.synthetic` as ``generate_ishigami`` /
# ``ishigami_sobol_indices``.

# Standard G-function importance vector (Saltelli et al.): small a => important.
SOBOL_G_DEFAULT_A = [0.0, 1.0, 4.5, 9.0, 99.0, 99.0, 99.0, 99.0]


def sobol_g_function(X: np.ndarray, a: Optional[np.ndarray] = None) -> np.ndarray:
    """Sobol G-function: f(x) = prod_i (|4 x_i - 2| + a_i) / (1 + a_i).

    A standard sensitivity-analysis benchmark on x_i ~ U(0, 1). The importance
    of variable i is controlled by a_i: small a_i => important, large a_i =>
    inactive. First-order indices are analytic — see :func:`sobol_g_sobol`.

    Args:
        X: (N, D) inputs on [0, 1].
        a: (D,) importance vector; defaults to :data:`SOBOL_G_DEFAULT_A`[:D].

    Returns:
        y: (N,) function values.
    """
    D = X.shape[1]
    if a is None:
        a = np.array(SOBOL_G_DEFAULT_A[:D], dtype=np.float64)
    a = np.asarray(a, dtype=np.float64)
    result = np.ones(X.shape[0])
    for i in range(D):
        result *= (np.abs(4.0 * X[:, i] - 2.0) + a[i]) / (1.0 + a[i])
    return result


def sobol_g_sobol(D: int = 8, a: Optional[np.ndarray] = None) -> Dict:
    """Analytic first-order Sobol indices for the G-function.

    V_i = 1 / (3 (1 + a_i)^2),  V_T = prod_i (1 + V_i) - 1,  S_i = V_i / V_T.

    Returns:
        {i: S_i} for i in range(D).
    """
    if a is None:
        a = np.array(SOBOL_G_DEFAULT_A[:D], dtype=np.float64)
    a = np.asarray(a, dtype=np.float64)
    Vi = 1.0 / (3.0 * (1.0 + a[:D]) ** 2)
    VT = float(np.prod(1.0 + Vi) - 1.0)
    Si = Vi / VT
    return {i: float(Si[i]) for i in range(D)}


def morris_function(X: np.ndarray) -> np.ndarray:
    """SIMPLIFIED Morris-like screening function (not the full Morris 1991).

    A high-dimensional qualitative screening benchmark on x_i ~ U(0, 1):
    the first 10 variables carry strong main effects (beta=20) and pairs
    among the first 6 carry interactions (beta_ij=-15); variables 11+ are
    EXACTLY inert.

    Deliberate simplifications vs. Morris (1991): the original also has
    third-order (beta_ijl=-10) and fourth-order (beta_ijls=5) terms among the
    first 4-5 variables, and weak main effects beta_i=(-1)^i for i=11..20 —
    all dropped here. Use this function only as a qualitative
    active-vs-inert screen; do NOT compare indices computed on it against
    published Morris-function reference values.

    Args:
        X: (N, D) inputs on [0, 1], D >= 20.

    Returns:
        y: (N,) function values.
    """
    D = X.shape[1]
    if D < 20:
        raise ValueError(f"morris_function needs D >= 20, got D={D}")
    w = 2.0 * (X - 0.5)
    # x4, x6, x8 (0-indexed 3,5,7 in Morris; here matching the legacy 2,4,6)
    for i in [2, 4, 6]:
        if i < D:
            w[:, i] = 2.0 * (1.1 * X[:, i] / (X[:, i] + 0.1) - 0.5)
    beta = np.zeros(D)
    beta[:10] = 20.0
    beta_ij = np.zeros((D, D))
    for i in range(min(6, D)):
        for j in range(i + 1, min(6, D)):
            beta_ij[i, j] = -15.0
    y = np.sum(beta * w, axis=1)
    for i in range(min(6, D)):
        for j in range(i + 1, min(6, D)):
            y += beta_ij[i, j] * w[:, i] * w[:, j]
    return y


# =============================================================================
# Helper: Get all test functions
# =============================================================================

def get_all_test_functions() -> Dict:
    """Return a dict of all test function generators."""
    return {
        'T1.1': T1_1_pure_linear,
        'T1.2': T1_2_pure_fourier,
        'T1.3': T1_3_linear_fourier_mix,
        'T1.4': T1_4_pure_interaction,
        'T1.5': T1_5_constant_mean_variable_noise,
        'T2.1': T2_1_friedman1,
        'T2.2': T2_2_smooth_additive,
        'T3.1': T3_1_orthogonal_mean_variance,
        'T3.2': T3_2_shared_variable,
        'T3.3': T3_3_hidden_variable,
        'T3.4': T3_4_interaction_noise,
        'T3.5': T3_5_signal_noise_confusion,
        'T4.1': T4_1_tunable_interaction,
        'T4.2': T4_2_tunable_frequency,
        'T4.3': T4_3_tunable_snr,
    }
