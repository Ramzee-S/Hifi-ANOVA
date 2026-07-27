"""Heteroscedastic-Ishigami benchmark: fit, evaluate, and compare.

Two ways to use it:

1. See the HiFi-ANOVA reference numbers:
       python benchmarks/run_benchmark.py

2. Score YOUR model. Fit anything on train.csv, predict the test.csv inputs,
   then:
       from benchmarks.run_benchmark import evaluate_predictions
       evaluate_predictions(y_pred, sigma_pred=None, name='my_model')
   (`sigma_pred` optional — per-point predictive std, if your model gives one.)

Why two R^2 columns. The response is noisy, so no model can drive R^2 vs the
*observed* y to 1 — its ceiling is the noise floor. The honest generalization
metric is R^2 vs the *true* (noiseless) function, available here because the data
is synthetic. A model that overfits the heteroscedastic noise (especially where
x3 is large) will show a gap: decent train fit, worse R^2_true on test. And
beyond prediction, the point of HiFi-ANOVA is recovering the *Sobol indices* —
reported against the analytic ground truth.
"""

import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, 'ishigami_hetero')


def load_dataset():
    """Return (X_train, y_train, X_test, y_test, f_true_test, sigma_true_test)."""
    tr = np.genfromtxt(os.path.join(DATA, 'train.csv'), delimiter=',', names=True)
    te = np.genfromtxt(os.path.join(DATA, 'test.csv'), delimiter=',', names=True)
    tt = np.genfromtxt(os.path.join(DATA, 'test_truth.csv'), delimiter=',', names=True)
    Xtr = np.column_stack([tr['x1'], tr['x2'], tr['x3']])
    Xte = np.column_stack([te['x1'], te['x2'], te['x3']])
    return Xtr, tr['y'], Xte, te['y'], tt['f_true'], tt['sigma_true']


def _r2(actual, pred):
    v = float(np.var(actual))
    return 1.0 - float(np.var(actual - pred)) / v if v > 0 else 0.0


def evaluate_predictions(y_pred, sigma_pred=None, name='model', verbose=True):
    """Score test-set predictions against observed y AND the noiseless truth.

    Args:
        y_pred: (5000,) predicted mean for the rows of test.csv (same order).
        sigma_pred: optional (5000,) predicted noise std per point.
        name: label for the printed row.

    Returns a metrics dict.
    """
    _, _, _, y_obs, f_true, sigma_true = load_dataset()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    if y_pred.shape != y_obs.shape:
        raise ValueError(f"y_pred has {y_pred.shape}, expected {y_obs.shape} "
                         f"(one row per test.csv line, same order).")

    m = {
        'name': name,
        'rmse_observed': float(np.sqrt(np.mean((y_pred - y_obs) ** 2))),
        'r2_observed': _r2(y_obs, y_pred),
        'rmse_true': float(np.sqrt(np.mean((y_pred - f_true) ** 2))),
        'r2_true': _r2(f_true, y_pred),
    }
    if sigma_pred is not None:
        sigma_pred = np.asarray(sigma_pred, dtype=float).ravel()
        var = np.maximum(sigma_pred ** 2, 1e-8)
        m['nll'] = float(np.mean(0.5 * np.log(2 * np.pi * var)
                                 + (y_obs - y_pred) ** 2 / (2 * var)))
        m['sigma_corr'] = float(np.corrcoef(sigma_pred, sigma_true)[0, 1])
        for a, zc in [(0.9, 1.6449), (0.95, 1.9600)]:
            m[f'coverage_{a}'] = float(np.mean(
                np.abs(y_obs - y_pred) <= zc * sigma_pred))

    if verbose:
        print(f"\n[{name}]")
        print(f"  RMSE  vs observed y : {m['rmse_observed']:.3f}"
              f"   (R^2 = {m['r2_observed']:.3f}, capped by noise)")
        print(f"  RMSE  vs TRUE f(x)  : {m['rmse_true']:.3f}"
              f"   (R^2 = {m['r2_true']:.3f})  <- generalization")
        if sigma_pred is not None:
            print(f"  NLL                 : {m['nll']:.3f}"
                  f"   sigma corr = {m['sigma_corr']:.3f}")
            print(f"  coverage 90/95      : {m['coverage_0.9']:.3f} / "
                  f"{m['coverage_0.95']:.3f}  (target 0.90 / 0.95)")
    return m


def run_hifi_anova_reference(verbose=True):
    """Fit HiFi-ANOVA on train.csv and score it — the reference to beat."""
    import jax
    jax.config.update('jax_enable_x64', True)
    import jax.numpy as jnp
    from hifi_anova.api import hifi_anova
    from hifi_anova.data.synthetic import ishigami_sobol_indices

    Xtr, ytr, Xte, _, _, _ = load_dataset()
    result = hifi_anova(
        Xtr, ytr, feature_names=['x1', 'x2', 'x3'],
        K1=12, K2=6, strategy='curvature', mode='second',
        heteroscedastic=True, Kh=3, lambda_h=0.1,
        first_order_pruning='bic', verbose=False)

    y_pred = result.predict(Xte)
    Xte_t = jnp.asarray(np.clip(result.transformer.transform(Xte), 0, 1),
                        dtype=jnp.float32)
    _, var = result.model.predict(Xte_t)
    sigma_pred = np.sqrt(np.asarray(var))

    metrics = evaluate_predictions(y_pred, sigma_pred, name='HiFi-ANOVA',
                                   verbose=verbose)
    metrics['r2_train_observed'] = _r2(ytr, result.predict(Xtr))

    # Sensitivity recovery — the real point of the method.
    gt = ishigami_sobol_indices()
    fo = result.sobol['mean_sobol']['first_order']
    err = np.mean([abs(fo[i] - gt['first_order'][i]) for i in range(3)])
    if verbose:
        print(f"  Sobol first-order   : "
              f"x1={fo[0]:.3f} x2={fo[1]:.3f} x3={fo[2]:.3f}  "
              f"(true 0.314 / 0.442 / 0.000; MAE={err:.3f})")
        print(f"  x3 total-order      : "
              f"{result.sobol['mean_sobol']['total_order'][2]:.3f}  "
              f"(true 0.244 — pure interaction)")
    metrics['sobol_mae'] = float(err)
    return metrics


def run_baselines():
    """Off-the-shelf sklearn baselines (mean only) for the overfitting contrast."""
    from sklearn.ensemble import (GradientBoostingRegressor,
                                   RandomForestRegressor)
    from sklearn.neural_network import MLPRegressor

    Xtr, ytr, Xte, y_obs, f_true, _ = load_dataset()
    models = {
        'GradientBoosting': GradientBoostingRegressor(random_state=0),
        'RandomForest': RandomForestRegressor(n_estimators=300, random_state=0,
                                              n_jobs=-1),
        'MLP (3x128)': MLPRegressor(hidden_layer_sizes=(128, 128, 128),
                                    max_iter=1500, random_state=0),
    }
    rows = []
    for name, mdl in models.items():
        mdl.fit(Xtr, ytr)
        rows.append({
            'name': name,
            'r2_train_observed': _r2(ytr, mdl.predict(Xtr)),
            'r2_observed': _r2(y_obs, mdl.predict(Xte)),
            'r2_true': _r2(f_true, mdl.predict(Xte)),
            'sobol_mae': None,
        })
    return rows


def compare():
    """Print a unified comparison table: HiFi-ANOVA vs off-the-shelf baselines."""
    ref = run_hifi_anova_reference(verbose=False)
    try:
        rows = [ref] + run_baselines()
    except ImportError:
        print("(scikit-learn not available — showing HiFi-ANOVA only)")
        rows = [ref]

    print("\n" + "=" * 74)
    print(f"{'model':<18}{'trainR2(obs)':>13}{'testR2(obs)':>12}"
          f"{'testR2(true)':>13}{'gap':>7}{'SobolMAE':>10}")
    print("-" * 74)
    for r in rows:
        gap = r['r2_train_observed'] - r['r2_observed']
        mae = f"{r['sobol_mae']:.3f}" if r.get('sobol_mae') is not None else "  n/a"
        print(f"{r['name']:<18}{r['r2_train_observed']:>13.3f}"
              f"{r['r2_observed']:>12.3f}{r['r2_true']:>13.3f}{gap:>7.3f}{mae:>10}")
    print("-" * 74)
    print("gap = trainR2(obs) - testR2(obs): large gap => fitting the noise.")
    print("testR2(true) is the honest generalization score; SobolMAE is the")
    print("first-order sensitivity error vs analytic (n/a = no native indices).")


if __name__ == '__main__':
    import sys
    print("=" * 74)
    print("Heteroscedastic-Ishigami benchmark")
    print("=" * 74)
    if '--baselines' in sys.argv or '--compare' in sys.argv:
        compare()
    else:
        run_hifi_anova_reference()
        print("\nRun with --baselines to compare against sklearn models.")
        print("To score your own model: fit train.csv, predict test.csv, call")
        print("evaluate_predictions(y_pred, sigma_pred). Beat R^2_true and Sobol MAE.")
