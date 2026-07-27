"""End-to-end experiment on Friedman-1 synthetic data."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import jax
import jax.numpy as jnp
import numpy as np

from hifi_anova.data.synthetic import generate_friedman1, generate_heteroscedastic
from hifi_anova.data.preprocessing import preprocess_data
from hifi_anova.training.trainer import HiFiANOVATrainer
from hifi_anova.analysis.sobol import compute_sobol_indices


def run_friedman1_experiment():
    """Run full HiFiANOVA pipeline on Friedman-1."""
    print("=" * 60)
    print("HiFiANOVA: Friedman-1 Experiment")
    print("=" * 60)

    # Generate data
    X, y = generate_friedman1(n_samples=10000, noise_std=1.0,
                              n_irrelevant=5, seed=42)
    print(f"\nData: {X.shape[0]} samples, {X.shape[1]} variables")
    print(f"  Active: x1-x5, Irrelevant: x6-x10")
    print(f"  y range: [{y.min():.2f}, {y.max():.2f}], std: {y.std():.2f}")

    # Preprocess
    data = preprocess_data(X, y, val_fraction=0.15, test_fraction=0.15, seed=42)
    print(f"  Train: {data['n_train']}, Val: {data['n_val']}, Test: {data['n_test']}")

    # Configure and train
    config = {
        'K1': 10,
        'K2': 5,
        'Kh': 3,
        'strategy': 'curvature',
        'lambda_order1': 0.001,
        'lambda_order2': 0.01,
        'lambda_h': 0.1,
        'stages': ['A', 'B'],
        'residual_nn': {'enabled': False},
    }

    trainer = HiFiANOVATrainer(config)
    model, results = trainer.fit(
        data['x_train'], data['y_train'],
        data['x_val'], data['y_val']
    )

    # Compute Sobol indices
    print("\n=== Sobol Sensitivity Indices ===")
    sobol = compute_sobol_indices(model, data['x_test'])

    print("\nFirst-Order Sobol Indices:")
    var_names = [f"x{i+1}" for i in range(10)]
    for i in range(10):
        si = sobol['mean_sobol']['first_order'][i]
        print(f"  {var_names[i]}: {si:.4f}")

    if sobol['mean_sobol']['second_order']:
        print("\nTop Second-Order Interactions:")
        pairs_sorted = sorted(sobol['mean_sobol']['second_order'].items(),
                            key=lambda x: -x[1])
        for (i, j), sij in pairs_sorted[:5]:
            print(f"  ({var_names[i]}, {var_names[j]}): {sij:.4f}")

    print("\nTotal-Order Indices:")
    for i in range(10):
        st = sobol['mean_sobol']['total_order'][i]
        print(f"  {var_names[i]}: {st:.4f}")

    # Variance accounting
    va = sobol['variance_accounting']
    print(f"\nVariance Accounting:")
    print(f"  First-order total:  {va['first_order_total']:.4f}")
    print(f"  Second-order total: {va['second_order_total']:.4f}")
    print(f"  Total model var:    {va['total_model_variance']:.4f}")

    # Test set evaluation
    phi1_test = jnp.array(
        __import__('hifi_anova.core.features', fromlist=['build_first_order_features']).build_first_order_features(data['x_test'], config['K1'])
    )
    from hifi_anova.core.features import build_first_order_features, build_second_order_features
    from hifi_anova.core.pairs import PairManager
    pm = PairManager(10)
    phi2_test = build_second_order_features(data['x_test'], config['K2'], pm.pair_indices)
    pred_test = model.mean_model.predict(
        build_first_order_features(data['x_test'], config['K1']),
        phi2_test
    )
    rmse_test = float(jnp.sqrt(jnp.mean((data['y_test'] - pred_test) ** 2)))
    print(f"\n  Test RMSE: {rmse_test:.4f}")

    return model, results, sobol


def run_heteroscedastic_experiment():
    """Run HiFiANOVA with heteroscedastic variance on synthetic data."""
    print("\n" + "=" * 60)
    print("HiFiANOVA: Heteroscedastic Experiment")
    print("=" * 60)

    # Generate heteroscedastic data (variance driven by x3, index 2)
    X, y, sigma_true = generate_heteroscedastic(
        n_samples=10000, noise_variable=2, seed=42
    )
    print(f"\nData: {X.shape[0]} samples, variance driven by x3")

    # Preprocess
    data = preprocess_data(X, y, val_fraction=0.15, test_fraction=0.15, seed=42)

    # Configure with heteroscedastic
    config = {
        'K1': 10,
        'K2': 5,
        'Kh': 3,
        'strategy': 'curvature',
        'lambda_order1': 0.001,
        'lambda_order2': 0.01,
        'lambda_h': 0.1,
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

    # Compute dual Sobol spectrum
    sobol = compute_sobol_indices(model, data['x_test'])

    print("\n=== Dual Sobol Spectrum ===")
    print("\nMean Sobol (first-order):")
    for i in range(10):
        si = sobol['mean_sobol']['first_order'][i]
        print(f"  x{i+1}: {si:.4f}")

    if 'variance_sobol' in sobol:
        print("\nVariance Sobol (first-order):")
        for i in range(10):
            si = sobol['variance_sobol']['first_order'][i]
            print(f"  x{i+1}: {si:.4f}")
        print("\n  (x3 should dominate variance Sobol since it drives noise)")

    return model, results, sobol


if __name__ == '__main__':
    model1, results1, sobol1 = run_friedman1_experiment()
    model2, results2, sobol2 = run_heteroscedastic_experiment()
