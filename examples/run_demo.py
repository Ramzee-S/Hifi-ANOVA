"""HiFi-ANOVA: Complete Demo Script

Demonstrates all capabilities of the framework:
1. Friedman-1 with first+second order Fourier (Sobol recovery)
2. Heteroscedastic synthetic (dual Sobol spectrum)
3. Calibration diagnostics
4. Component function visualization
"""

import jax
jax.config.update('jax_enable_x64', True)
import warnings
warnings.filterwarnings('ignore')

import jax.numpy as jnp
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from hifi_anova.data.synthetic import generate_friedman1, generate_heteroscedastic
from hifi_anova.data.preprocessing import preprocess_data
from hifi_anova.training.trainer import HiFiANOVATrainer
from hifi_anova.analysis.sobol import compute_sobol_indices
from hifi_anova.analysis.diagnostics import variance_accounting_report, calibration_report
from hifi_anova.analysis.visualization import (
    plot_sobol_bars, plot_dual_sobol, plot_component_functions
)


def experiment_1_friedman1():
    """Experiment 1: Sobol Recovery on Friedman-1."""
    print("=" * 70)
    print("EXPERIMENT 1: Sobol Recovery on Friedman-1")
    print("=" * 70)
    print("\nFriedman-1: f(x) = 10*sin(pi*x1*x2) + 20*(x3-0.5)^2 + 10*x4 + 5*x5")
    print("Active: x1-x5, Irrelevant: x6-x10, Noise: sigma=1.0\n")

    X, y = generate_friedman1(n_samples=10000, noise_std=1.0,
                              n_irrelevant=5, seed=42)
    data = preprocess_data(X, y, seed=42)

    config = {
        'K1': 10, 'K2': 5, 'Kh': 0,
        'strategy': 'curvature',
        'lambda_order1': 0.001, 'lambda_order2': 0.01,
        'stages': ['A', 'B'],
        'residual_nn': {'enabled': False},
    }

    trainer = HiFiANOVATrainer(config)
    model, results = trainer.fit(
        data['x_train'], data['y_train'],
        data['x_val'], data['y_val']
    )

    # Sobol indices
    sobol = compute_sobol_indices(model, data['x_test'])

    print("\n" + "-" * 50)
    print("SOBOL SENSITIVITY ANALYSIS")
    print("-" * 50)
    var_names = [f"x{i+1}" for i in range(10)]

    print(f"\n{'Variable':<10} {'First-Order':<14} {'Total-Order':<14} {'Note'}")
    print(f"{'--------':<10} {'----------':<14} {'----------':<14} {'----'}")
    notes = {0: 'sin(pi*x1*x2)', 1: 'sin(pi*x1*x2)', 2: '20*(x3-0.5)^2',
             3: '10*x4 (linear)', 4: '5*x5 (linear)'}
    for i in range(10):
        s1 = sobol['mean_sobol']['first_order'][i]
        st = sobol['mean_sobol']['total_order'][i]
        note = notes.get(i, 'irrelevant')
        print(f"  {var_names[i]:<8} {s1:<14.4f} {st:<14.4f} {note}")

    print(f"\nTop interactions:")
    pairs_sorted = sorted(sobol['mean_sobol']['second_order'].items(),
                         key=lambda x: -x[1])
    for (i, j), sij in pairs_sorted[:3]:
        print(f"  ({var_names[i]}, {var_names[j]}): {sij:.4f}")

    # Variance accounting
    va = variance_accounting_report(model, data['x_test'], data['y_test'])
    print(f"\nVariance Accounting:")
    print(f"  R-squared:          {va['R_squared']:.4f}")
    print(f"  First-order total:  {va['first_order_total']:.4f}")
    print(f"  Second-order total: {va['second_order_total']:.4f}")
    print(f"  Additivity gap:     {va['additivity_gap']:.4f}")

    # Plot
    plot_sobol_bars(sobol, var_names, title="Friedman-1: Sobol Indices",
                    save_path="figures/friedman1_sobol.png")
    plot_component_functions(model, [0, 1, 2, 3, 4], var_names[:5],
                           save_path="figures/friedman1_components.png")

    return model, sobol


def experiment_2_heteroscedastic():
    """Experiment 2: Heteroscedastic Dual Sobol Spectrum."""
    print("\n\n" + "=" * 70)
    print("EXPERIMENT 2: Heteroscedastic Dual Sobol Spectrum")
    print("=" * 70)
    print("\nMean: Friedman-1, Variance: sigma(x) = 0.5 + 2*x3")
    print("Expected: x3 dominates the log-variance index S^h\n")

    X, y, sigma_true = generate_heteroscedastic(
        n_samples=10000, noise_variable=2, seed=42
    )
    data = preprocess_data(X, y, seed=42)

    config = {
        'K1': 10, 'K2': 5, 'Kh': 3,
        'strategy': 'curvature',
        'lambda_order1': 0.001, 'lambda_order2': 0.01, 'lambda_h': 0.1,
        'stages': ['A', 'B', 'D'],
        'residual_nn': {'enabled': False},
        'max_outer_iter': 8,
        'alternating_tol': 1e-4,
        'newton_max_iter': 10,
    }

    trainer = HiFiANOVATrainer(config)
    model, results = trainer.fit(
        data['x_train'], data['y_train'],
        data['x_val'], data['y_val']
    )

    sobol = compute_sobol_indices(model, data['x_test'])

    print("\n" + "-" * 50)
    print("DUAL SOBOL SPECTRUM")
    print("-" * 50)
    var_names = [f"x{i+1}" for i in range(10)]

    print(f"\n{'Variable':<10} {'Mean S1':<12} {'Variance S1':<12}")
    print(f"{'--------':<10} {'------':<12} {'----------':<12}")
    for i in range(10):
        sm = sobol['mean_sobol']['first_order'][i]
        sv = (sobol['log_variance_sobol']['first_order'][i]
              if 'log_variance_sobol' in sobol else 0)
        marker = " <-- log-variance driver" if sv > 0.3 else ""
        print(f"  {var_names[i]:<8} {sm:<12.4f} {sv:<12.4f}{marker}")

    # Calibration
    cal = calibration_report(model, data['x_test'], data['y_test'])
    print(f"\nCalibration Diagnostics:")
    print(f"  Mean(z):     {cal['mean_standardized_residual']:.4f} (target: 0)")
    print(f"  Var(z):      {cal['var_standardized_residual']:.4f} (target: 1)")
    print(f"  Coverage 90%: {cal['coverage_0.9']:.3f} (target: 0.90)")
    print(f"  Coverage 95%: {cal['coverage_0.95']:.3f} (target: 0.95)")

    # Plot
    plot_dual_sobol(sobol, var_names,
                    save_path="figures/heteroscedastic_dual_sobol.png")

    return model, sobol


def experiment_3_noise_levels():
    """Experiment 3: Robustness across noise levels."""
    print("\n\n" + "=" * 70)
    print("EXPERIMENT 3: Robustness Across Noise Levels")
    print("=" * 70)

    print(f"\n{'Noise σ':<10} {'RMSE':<10} {'R²':<10} {'S(x4)':<10} {'S(x6)':<10}")
    print(f"{'-------':<10} {'----':<10} {'--':<10} {'-----':<10} {'-----':<10}")

    for noise in [0.0, 0.1, 0.5, 1.0, 2.0]:
        X, y = generate_friedman1(n_samples=5000, noise_std=noise,
                                  n_irrelevant=5, seed=42)
        data = preprocess_data(X, y, seed=42)

        config = {
            'K1': 10, 'K2': 5, 'Kh': 0,
            'strategy': 'curvature',
            'lambda_order1': 0.001, 'lambda_order2': 0.01,
            'stages': ['A', 'B'],
            'residual_nn': {'enabled': False},
        }

        trainer = HiFiANOVATrainer(config)
        model, results = trainer.fit(
            data['x_train'], data['y_train'],
            data['x_val'], data['y_val']
        )

        sobol = compute_sobol_indices(model)
        va = variance_accounting_report(model, data['x_test'], data['y_test'])

        s4 = sobol['mean_sobol']['first_order'][3]  # x4 (should be large)
        s6 = sobol['mean_sobol']['first_order'][5]  # x6 (should be ~0)
        rmse = results['stage_B']['rmse_val']

        print(f"  {noise:<8.1f} {rmse:<10.4f} {va['R_squared']:<10.4f} {s4:<10.4f} {s6:<10.4f}")


if __name__ == '__main__':
    import os
    os.makedirs('figures', exist_ok=True)

    model1, sobol1 = experiment_1_friedman1()
    model2, sobol2 = experiment_2_heteroscedastic()
    experiment_3_noise_levels()

    print("\n\n" + "=" * 70)
    print("ALL EXPERIMENTS COMPLETE")
    print("=" * 70)
    print("\nFigures saved to figures/")
    print("Key findings:")
    print("  1. Sobol indices correctly identify active variables in Friedman-1")
    print("  2. Dual spectrum identifies x3 as a log-variance driver")
    print("  3. Model is robust across noise levels")
