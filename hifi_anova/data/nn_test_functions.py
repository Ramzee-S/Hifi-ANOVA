"""Test functions designed to stress-test the residual NN and heteroscedastic model.

Design principle: each function contains terms that are PURELY higher-order
in the Hoeffding decomposition — they have exactly zero first- and second-order
components. This means:
  - The Fourier model (order 1+2) cannot capture them AT ALL
  - The residual NN is the ONLY way to reduce RMSE
  - The Sobol indices from Fourier should be correct for the low-order terms
    regardless of the NN (if orthogonality is maintained)

Key mathematical facts used:
  - (x-0.5) has zero mean on [0,1]
  - sin(2πkx) and cos(2πkx) have zero mean on [0,1]
  - A product of zero-mean functions of DIFFERENT variables is purely
    higher-order: E[f(x1)g(x2)h(x3) | x1,x2] = f(x1)g(x2)·E[h(x3)] = 0
    so the second-order component f12 = 0.
  - Var(product of independent zero-mean functions) = product of variances
    when the functions are independent.

Variance formulas:
  Var((x-0.5))    = 1/12
  Var(sin(2πkx))  = 1/2
  Var(cos(2πkx))  = 1/2
  Var(a·f₁·f₂·f₃) = a² · Var(f₁) · Var(f₂) · Var(f₃)  [independent, zero-mean]
"""

import numpy as np
from typing import Dict, Tuple


def NNT1_pure_third_order(n_samples: int = 10000, noise_std: float = 0.5,
                          seed: int = 42) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Large purely third-order signal alongside first-order terms.

    f(x) = 5(x₁ - ½) + 3cos(2πx₂) + 10·sin(2πx₁)sin(2πx₂)sin(2πx₃)

    The third-order term sin·sin·sin is PURELY third-order in Hoeffding:
      - E[f_3way | x₁, x₂] = sin(2πx₁)sin(2πx₂)·E[sin(2πx₃)] = 0
      - So it has zero projection onto any first- or second-order component.

    The Fourier model captures terms 1-2 perfectly.
    The NN must capture term 3 to reduce RMSE.

    Variances:
      Var(5(x₁-½))           = 25/12 ≈ 2.083
      Var(3cos(2πx₂))        = 9/2 = 4.500
      Var(10·sin·sin·sin)    = 100·(½)³ = 12.500
      Total signal            = 19.083
      Fourier fraction        = 6.583/19.083 = 34.5%
      NN fraction (3rd-order) = 12.500/19.083 = 65.5%

    D = 6 (x₄-x₆ irrelevant)
    """
    rng = np.random.RandomState(seed)
    D = 6
    X = rng.uniform(0, 1, (n_samples, D))

    # Fourier-capturable terms
    term1 = 5.0 * (X[:, 0] - 0.5)
    term2 = 3.0 * np.cos(2 * np.pi * X[:, 1])

    # Purely third-order term (zero 1st and 2nd order Hoeffding components)
    term3 = (10.0 * np.sin(2 * np.pi * X[:, 0])
             * np.sin(2 * np.pi * X[:, 1])
             * np.sin(2 * np.pi * X[:, 2]))

    f = term1 + term2 + term3
    y = f + noise_std * rng.randn(n_samples)

    var_term1 = 25.0 / 12.0
    var_term2 = 9.0 / 2.0
    var_term3 = 100.0 * (0.5 ** 3)
    total_var = var_term1 + var_term2 + var_term3

    ground_truth = {
        'name': 'NNT1_pure_third_order',
        'D': D,
        'fourier_capturable_variance': var_term1 + var_term2,
        'nn_target_variance': var_term3,
        'total_signal_variance': total_var,
        'expected_nn_fraction': var_term3 / total_var,
        'expected_fourier_fraction': (var_term1 + var_term2) / total_var,
        'noise_variance': noise_std ** 2,
        # Sobol indices (Fourier part only, what the Fourier model should find)
        'fourier_sobol': {
            # x1 has both linear AND appears in 3-way — but Fourier only sees linear
            0: var_term1 / total_var,
            1: var_term2 / total_var,
            2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0,
        },
        # True total Sobol (including 3-way contribution)
        'true_total_sobol': {
            # x1 appears in term1 AND term3
            0: (var_term1 + var_term3) / total_var,  # but S1 is just var_term1/total
            1: (var_term2 + var_term3) / total_var,
            2: var_term3 / total_var,
            3: 0.0, 4: 0.0, 5: 0.0,
        },
        'true_first_order_sobol': {
            0: var_term1 / total_var,  # only the linear part
            1: var_term2 / total_var,  # only the cos part
            2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0,
        },
        'description': (
            'f = 5(x1-½) + 3cos(2πx2) + 10·sin(2πx1)sin(2πx2)sin(2πx3). '
            f'Fourier captures {(var_term1+var_term2)/total_var:.0%}, '
            f'NN must capture {var_term3/total_var:.0%} (purely 3rd-order).'
        ),
    }
    return X, y, ground_truth


def NNT2_shared_variables(n_samples: int = 10000, noise_std: float = 0.3,
                          seed: int = 42) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Variables appear in BOTH Fourier and higher-order terms.

    f(x) = 4(x₁ - ½) + 3cos(2πx₂) + 2sin(2πx₃)sin(2πx₄)
            + 6(x₁ - ½)(x₂ - ½)(x₃ - ½)

    Terms 1-3 are capturable by order-1+2 Fourier.
    Term 4 is purely 3rd-order (zero 1st and 2nd order Hoeffding components).

    Critical test: x₁, x₂, x₃ appear in BOTH the Fourier and 3rd-order parts.
    Can the NN capture the 3rd-order term WITHOUT corrupting the Fourier Sobol
    indices for these shared variables?

    Variances:
      Var(4(x₁-½))                     = 16/12 ≈ 1.333
      Var(3cos(2πx₂))                  = 9/2 = 4.500
      Var(2·sin(2πx₃)·sin(2πx₄))      = 4·(½)² = 1.000
      Var(6·(x₁-½)(x₂-½)(x₃-½))      = 36·(1/12)³ ≈ 0.02083
      Total                             = 6.854
      Fourier fraction                  = 6.833/6.854 = 99.7%
      NN fraction                       = 0.021/6.854 = 0.3%

    Note: the 3rd-order term is small — this tests sensitivity, not magnitude.

    D = 8 (x₅-x₈ irrelevant)
    """
    rng = np.random.RandomState(seed)
    D = 8
    X = rng.uniform(0, 1, (n_samples, D))

    term1 = 4.0 * (X[:, 0] - 0.5)
    term2 = 3.0 * np.cos(2 * np.pi * X[:, 1])
    term3 = 2.0 * np.sin(2 * np.pi * X[:, 2]) * np.sin(2 * np.pi * X[:, 3])
    term4 = 6.0 * (X[:, 0] - 0.5) * (X[:, 1] - 0.5) * (X[:, 2] - 0.5)

    f = term1 + term2 + term3 + term4
    y = f + noise_std * rng.randn(n_samples)

    var1 = 16.0 / 12.0
    var2 = 9.0 / 2.0
    var3 = 4.0 * 0.25
    var4 = 36.0 * (1.0 / 12.0) ** 3
    total = var1 + var2 + var3 + var4

    ground_truth = {
        'name': 'NNT2_shared_variables',
        'D': D,
        'fourier_capturable_variance': var1 + var2 + var3,
        'nn_target_variance': var4,
        'total_signal_variance': total,
        'expected_nn_fraction': var4 / total,
        'noise_variance': noise_std ** 2,
        'true_first_order_sobol': {
            0: var1 / total, 1: var2 / total,
            2: 0.0, 3: 0.0,  # x3,x4 have NO first-order (only in interactions)
            4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0,
        },
        'true_second_order_sobol': {(2, 3): var3 / total},
        'shared_variables': [0, 1, 2],  # appear in both Fourier and 3rd-order
        'description': (
            'f = 4(x1-½) + 3cos(2πx2) + 2sin(2πx3)sin(2πx4) + 6(x1-½)(x2-½)(x3-½). '
            f'3rd-order term is only {var4/total:.1%} of variance — tests sensitivity. '
            f'x1,x2,x3 shared between Fourier and NN.'
        ),
    }
    return X, y, ground_truth


def NNT3_large_third_order_fourier(n_samples: int = 10000, noise_std: float = 0.5,
                                    seed: int = 42) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Large 3rd-order term using Fourier products (sin·sin·sin).

    f(x) = 3(x₁ - ½) + 2cos(2πx₂) + 8·sin(2πx₃)sin(2πx₄)sin(2πx₅)

    Similar to NNT1 but uses different variables for each order:
      - x₁: first-order only (linear)
      - x₂: first-order only (Fourier)
      - x₃,x₄: appear in 2nd-order pair AND in 3rd-order triple
      - x₅: appears ONLY in 3rd-order triple
      Wait, no — x₃,x₄ don't have a 2nd-order term here. Let me fix.

    Revised:
    f(x) = 3(x₁ - ½) + 2cos(2πx₂) + 1.5·sin(2πx₃)sin(2πx₄)
            + 8·sin(2πx₃)sin(2πx₄)sin(2πx₅)

    Now x₃,x₄ appear in both 2nd- and 3rd-order.
    x₅ appears ONLY in the 3rd-order term.

    Variances:
      Var(3(x₁-½))                    = 9/12 = 0.75
      Var(2cos(2πx₂))                 = 4/2 = 2.0
      Var(1.5·sin·sin)                = 2.25·(½)² = 0.5625
      Var(8·sin·sin·sin)              = 64·(½)³ = 8.0
      Total                            = 11.3125
      Fourier fraction                 = 3.3125/11.3125 = 29.3%
      NN fraction                      = 8.0/11.3125 = 70.7%

    D = 8 (x₆-x₈ irrelevant)
    """
    rng = np.random.RandomState(seed)
    D = 8
    X = rng.uniform(0, 1, (n_samples, D))

    term1 = 3.0 * (X[:, 0] - 0.5)
    term2 = 2.0 * np.cos(2 * np.pi * X[:, 1])
    term3 = 1.5 * np.sin(2 * np.pi * X[:, 2]) * np.sin(2 * np.pi * X[:, 3])
    term4 = (8.0 * np.sin(2 * np.pi * X[:, 2])
             * np.sin(2 * np.pi * X[:, 3])
             * np.sin(2 * np.pi * X[:, 4]))

    f = term1 + term2 + term3 + term4
    y = f + noise_std * rng.randn(n_samples)

    var1 = 9.0 / 12.0
    var2 = 4.0 / 2.0
    var3 = 2.25 * 0.25
    var4 = 64.0 * (0.5 ** 3)
    total = var1 + var2 + var3 + var4

    ground_truth = {
        'name': 'NNT3_large_third_order_fourier',
        'D': D,
        'fourier_capturable_variance': var1 + var2 + var3,
        'nn_target_variance': var4,
        'total_signal_variance': total,
        'expected_nn_fraction': var4 / total,
        'noise_variance': noise_std ** 2,
        'true_first_order_sobol': {
            0: var1 / total, 1: var2 / total,
            2: 0.0, 3: 0.0, 4: 0.0,
            5: 0.0, 6: 0.0, 7: 0.0,
        },
        'true_second_order_sobol': {(2, 3): var3 / total},
        'description': (
            'f = 3(x1-½) + 2cos(2πx2) + 1.5sin(2πx3)sin(2πx4) '
            '+ 8sin(2πx3)sin(2πx4)sin(2πx5). '
            f'NN must capture {var4/total:.0%} of variance.'
        ),
    }
    return X, y, ground_truth


def NNT4_heteroscedastic_higher_order(
    n_samples: int = 10000, seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Heteroscedastic noise with BOTH first-order and higher-order variance terms.

    Mean:
      f(x) = 5(x₁ - ½) + 3cos(2πx₂) + 7·sin(2πx₁)sin(2πx₂)sin(2πx₃)

    Log-variance:
      h(x) = 1.5(x₄ - ½) + 3·(x₅ - ½)(x₆ - ½)(x₇ - ½)

    The mean has:
      - First-order: x₁ (linear), x₂ (Fourier) → Fourier captures
      - Third-order: sin·sin·sin → NN captures

    The log-variance has:
      - First-order: x₄ → variance model captures
      - Third-order: (x₅-½)(x₆-½)(x₇-½) → variance model CANNOT capture

    This tests:
      1. Does the NN capture the mean residual?
      2. Does the variance model capture x₄ but miss the 3-way variance term?
      3. Is the calibration degraded by the uncapturable variance interaction?
      4. Do the mean Sobol indices remain correct despite structured noise?

    D = 10 (x₈-x₁₀ irrelevant)

    Mean variances:
      Var(5(x₁-½))          = 25/12 ≈ 2.083
      Var(3cos(2πx₂))       = 9/2 = 4.500
      Var(7·sin·sin·sin)    = 49·(½)³ = 6.125
      Total mean             = 12.708

    Variance structure:
      h₀ ≈ E[h] = 0 (centered)
      Var(h_1st) = (1.5)²/12 = 0.1875
      Var(h_3rd) = 9·(1/12)³ ≈ 0.00521
    """
    rng = np.random.RandomState(seed)
    D = 10
    X = rng.uniform(0, 1, (n_samples, D))

    # Mean function
    f_term1 = 5.0 * (X[:, 0] - 0.5)
    f_term2 = 3.0 * np.cos(2 * np.pi * X[:, 1])
    f_term3 = (7.0 * np.sin(2 * np.pi * X[:, 0])
               * np.sin(2 * np.pi * X[:, 1])
               * np.sin(2 * np.pi * X[:, 2]))
    f = f_term1 + f_term2 + f_term3

    # Log-variance: first-order + third-order
    h_first = 1.5 * (X[:, 3] - 0.5)
    h_third = 3.0 * (X[:, 4] - 0.5) * (X[:, 5] - 0.5) * (X[:, 6] - 0.5)
    h = h_first + h_third

    sigma = np.exp(h / 2.0)
    y = f + sigma * rng.randn(n_samples)

    var_f1 = 25.0 / 12.0
    var_f2 = 9.0 / 2.0
    var_f3 = 49.0 * (0.5 ** 3)
    total_mean_var = var_f1 + var_f2 + var_f3

    var_h1 = 1.5 ** 2 / 12.0
    var_h3 = 9.0 * (1.0 / 12.0) ** 3
    total_h_var = var_h1 + var_h3

    ground_truth = {
        'name': 'NNT4_heteroscedastic_higher_order',
        'D': D,
        # Mean decomposition
        'fourier_capturable_mean_var': var_f1 + var_f2,
        'nn_target_mean_var': var_f3,
        'total_mean_variance': total_mean_var,
        'expected_nn_mean_fraction': var_f3 / total_mean_var,
        # Variance decomposition
        'capturable_variance_var': var_h1,
        'uncapturable_variance_var': var_h3,
        'total_h_variance': total_h_var,
        'variance_capturable_fraction': var_h1 / total_h_var,
        # Sobol ground truth
        'true_mean_first_order_sobol': {
            0: var_f1 / total_mean_var,
            1: var_f2 / total_mean_var,
            2: 0.0,  # x3 only in 3rd-order mean
            3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0,
        },
        'true_variance_first_order_sobol': {
            # Only x4 has first-order variance contribution
            3: var_h1 / total_h_var,
            # x5,x6,x7 only in 3rd-order variance (not capturable by 1st-order model)
            4: 0.0, 5: 0.0, 6: 0.0,
        },
        'description': (
            'Mean = 5(x1-½) + 3cos(2πx2) + 7sin·sin·sin. '
            'LogVar = 1.5(x4-½) + 3(x5-½)(x6-½)(x7-½). '
            f'NN captures {var_f3/total_mean_var:.0%} of mean. '
            f'Variance model captures {var_h1/total_h_var:.0%} of noise structure.'
        ),
    }
    return X, y, ground_truth


def NNT5_progressive_complexity(
    n_samples: int = 10000, noise_std: float = 0.5,
    alpha_3: float = 1.0, alpha_4: float = 0.0,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Tunable higher-order complexity for ablation studies.

    f(x) = 3(x₁ - ½) + 2cos(2πx₂) + 1.5sin(2πx₃)sin(2πx₄)
           + α₃ · 6·sin(2πx₁)sin(2πx₃)sin(2πx₅)
           + α₄ · 4·(x₂-½)(x₃-½)(x₅-½)(x₆-½)

    At α₃=0, α₄=0: purely order-1+2, Fourier captures everything.
    At α₃=1, α₄=0: 3rd-order signal, NN needed.
    At α₃=1, α₄=1: 3rd + 4th order signal.

    The 4th-order product term: (x₂-½)(x₃-½)(x₅-½)(x₆-½) is purely 4th-order
    (all lower Hoeffding components are zero because each factor has zero mean).

    D = 8 (x₇,x₈ irrelevant)

    Variances (at α₃=1, α₄=1):
      Var(3(x₁-½))                    = 9/12 = 0.75
      Var(2cos(2πx₂))                 = 4/2 = 2.0
      Var(1.5sin·sin)                  = 2.25·(½)² = 0.5625
      Var(6·sin·sin·sin)              = 36·(½)³ = 4.5
      Var(4·prod-of-4)                = 16·(1/12)⁴ = 0.000772
      Total                            ≈ 7.815
    """
    rng = np.random.RandomState(seed)
    D = 8
    X = rng.uniform(0, 1, (n_samples, D))

    term1 = 3.0 * (X[:, 0] - 0.5)
    term2 = 2.0 * np.cos(2 * np.pi * X[:, 1])
    term3 = 1.5 * np.sin(2 * np.pi * X[:, 2]) * np.sin(2 * np.pi * X[:, 3])
    term_3way = (alpha_3 * 6.0
                 * np.sin(2 * np.pi * X[:, 0])
                 * np.sin(2 * np.pi * X[:, 2])
                 * np.sin(2 * np.pi * X[:, 4]))
    term_4way = (alpha_4 * 4.0
                 * (X[:, 1] - 0.5) * (X[:, 2] - 0.5)
                 * (X[:, 4] - 0.5) * (X[:, 5] - 0.5))

    f = term1 + term2 + term3 + term_3way + term_4way
    y = f + noise_std * rng.randn(n_samples)

    var1 = 9.0 / 12.0
    var2 = 4.0 / 2.0
    var3 = 2.25 * 0.25
    var_3way = (alpha_3 * 6.0) ** 2 * (0.5 ** 3)
    var_4way = (alpha_4 * 4.0) ** 2 * (1.0 / 12.0) ** 4
    fourier_var = var1 + var2 + var3
    nn_var = var_3way + var_4way
    total = fourier_var + nn_var

    ground_truth = {
        'name': f'NNT5_progressive_a3={alpha_3}_a4={alpha_4}',
        'D': D,
        'alpha_3': alpha_3,
        'alpha_4': alpha_4,
        'fourier_capturable_variance': fourier_var,
        'nn_target_variance': nn_var,
        'total_signal_variance': total,
        'expected_nn_fraction': nn_var / total if total > 0 else 0,
        'noise_variance': noise_std ** 2,
        'description': (
            f'Progressive complexity: α₃={alpha_3}, α₄={alpha_4}. '
            f'Fourier captures {fourier_var/total:.0%}, '
            f'NN targets {nn_var/total:.0%}.'
        ),
    }
    return X, y, ground_truth


def get_nn_test_functions() -> Dict:
    """Return all NN test function generators."""
    return {
        'NNT1': NNT1_pure_third_order,
        'NNT2': NNT2_shared_variables,
        'NNT3': NNT3_large_third_order_fourier,
        'NNT4': NNT4_heteroscedastic_higher_order,
        'NNT5': NNT5_progressive_complexity,
    }
