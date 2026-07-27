"""Heteroscedastic Ishigami — a capabilities showcase for the dual spectrum.

The Ishigami function is the canonical sensitivity-analysis benchmark:

    f(x) = sin(x1) + a*sin^2(x2) + b*x3^4*sin(x1),   x_i ~ U(-pi, pi)

with known analytic Sobol indices. Its signature feature is x3: it has a
**zero first-order** effect but a **non-zero total-order** effect — it acts
purely through its interaction with x1.

Here we make it *heteroscedastic* by driving the noise variance with x3 as well.
That turns x3 into the textbook "hidden driver": near-invisible to mean
first-order importance, yet the sole driver of the predictive *variance*. It is
exactly the case ordinary feature-importance misses and the dual mean+variance
Sobol spectrum was built for.

The script demonstrates, end to end:
  1. Recovery of the analytic Ishigami mean Sobol indices (first- and total-order).
  2. The dual spectrum: mean vs variance sensitivity, side by side.
  3. Interaction discovery — the residual sieve rediscovering the x1-x3 pair.
  4. Calibration of the fitted heteroscedastic model.
  5. A NEW visualization: the dual-sensitivity plane, each variable an ellipse
     whose position is (mean Sobol, variance Sobol) and whose size is the
     bootstrap CI of those indices.

Run from the repo root (writes ./figures/):

    python examples/run_ishigami_heteroscedastic.py
"""

import os
import io
import contextlib

import jax
jax.config.update('jax_enable_x64', True)
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib
matplotlib.use('Agg')

from hifi_anova.data.synthetic import generate_ishigami, ishigami_sobol_indices
from hifi_anova.data.preprocessing import preprocess_data
from hifi_anova.training.trainer import HiFiANOVATrainer
from hifi_anova.analysis.sobol import compute_sobol_indices
from hifi_anova.analysis.diagnostics import calibration_report
from hifi_anova.analysis.interaction_discovery import scan_missing_pairs
from hifi_anova.analysis.visualization import (
    plot_dual_sobol, plot_component_functions, plot_sensitivity_ellipses,
)

VAR_NAMES = ['$x_1$', '$x_2$', '$x_3$']

# Mean + variance fit: first + second order mean (A, B) plus the heteroscedastic
# variance stage (D). No NN residual — the structured model is enough here.
CONFIG = {
    'K1': 12, 'K2': 6, 'Kh': 3,
    'strategy': 'curvature',
    'lambda_order1': 0.001, 'lambda_order2': 0.01, 'lambda_h': 0.1,
    'stages': ['A', 'B', 'D'],
    'residual_nn': {'enabled': False},
    'max_outer_iter': 8, 'alternating_tol': 1e-4, 'newton_max_iter': 10,
}

N_BOOT = 12          # bootstrap refits for the ellipse CIs (set 0 to skip)
BOOT_SUBSAMPLE = 4000


def _fit_silent(config, xtr, ytr, xval, yval):
    """Fit a trainer while swallowing its per-stage stdout (used in the loop)."""
    with contextlib.redirect_stdout(io.StringIO()):
        model, results = HiFiANOVATrainer(config).fit(xtr, ytr, xval, yval)
    return model, results


def bootstrap_dual_sobol_ci(data, config, n_boot, n_sub, seed=0):
    """Bootstrap CIs on first-order mean & variance Sobol indices.

    Resamples the (already transformed) training set with replacement, refits,
    and records the first-order mean and variance Sobol indices each time. The
    structural indices need no test data, so this is a direct measure of how
    stable the dual spectrum is under resampling.

    Returns (mean_ci, var_ci) as {i: (lo, hi)} at the 95% percentile interval.
    """
    xtr, ytr = np.asarray(data['x_train']), np.asarray(data['y_train'])
    xval, yval = data['x_val'], data['y_val']
    n = xtr.shape[0]
    D = xtr.shape[1]
    rng = np.random.RandomState(seed)

    mean_draws = {i: [] for i in range(D)}
    var_draws = {i: [] for i in range(D)}
    for b in range(n_boot):
        idx = rng.choice(n, size=min(n_sub, n), replace=True)
        model, _ = _fit_silent(config, xtr[idx], ytr[idx], xval, yval)
        s = compute_sobol_indices(model)
        mf = s['mean_sobol']['first_order']
        vf = (s['variance_sobol']['first_order']
              if 'variance_sobol' in s else {i: 0.0 for i in range(D)})
        for i in range(D):
            mean_draws[i].append(float(mf.get(i, 0.0)))
            var_draws[i].append(float(vf.get(i, 0.0)))
        print(f"    bootstrap {b + 1}/{n_boot}", end='\r')
    print()

    mean_ci = {i: (float(np.percentile(mean_draws[i], 2.5)),
                   float(np.percentile(mean_draws[i], 97.5))) for i in range(D)}
    var_ci = {i: (float(np.percentile(var_draws[i], 2.5)),
                  float(np.percentile(var_draws[i], 97.5))) for i in range(D)}
    return mean_ci, var_ci


def main():
    os.makedirs('figures', exist_ok=True)

    print("=" * 70)
    print("HETEROSCEDASTIC ISHIGAMI — dual mean+variance Sobol showcase")
    print("=" * 70)
    print("  f(x) = sin(x1) + 7*sin^2(x2) + 0.1*x3^4*sin(x1),   x_i ~ U(-pi,pi)")
    print("  noise std ramps 0.3 -> 3.0 across x3  (x3 drives the VARIANCE)")

    # ---- Ground truth ---------------------------------------------------
    gt = ishigami_sobol_indices(a=7.0, b=0.1)
    print("\nAnalytic mean Sobol (ground truth):")
    print(f"  {'var':<5}{'first-order':<14}{'total-order':<14}note")
    gt_notes = {0: 'sin(x1) [+x1x3]', 1: 'sin^2(x2)', 2: 'ONLY via x1x3 interaction'}
    for i in range(3):
        print(f"  x{i+1:<4}{gt['first_order'][i]:<14.4f}"
              f"{gt['total_order'][i]:<14.4f}{gt_notes[i]}")

    # ---- Fit mean + variance -------------------------------------------
    X, y, sigma_true = generate_ishigami(
        n_samples=10000, heteroscedastic=True, variance_variable=2,
        sigma_min=0.3, sigma_max=3.0, seed=1)
    data = preprocess_data(X, y, seed=1)

    print("\nFitting mean + variance (stages A, B, D)...")
    model, results = HiFiANOVATrainer(CONFIG).fit(
        data['x_train'], data['y_train'], data['x_val'], data['y_val'])
    sobol = compute_sobol_indices(model, data['x_test'])

    # ---- Dual spectrum vs ground truth ---------------------------------
    print("\n" + "-" * 70)
    print("DUAL SOBOL SPECTRUM  (fitted vs analytic)")
    print("-" * 70)
    mf = sobol['mean_sobol']['first_order']
    mt = sobol['mean_sobol']['total_order']
    vf = sobol['variance_sobol']['first_order']
    print(f"  {'var':<5}{'meanS1(fit/true)':<22}{'meanST(fit/true)':<22}{'varS1'}")
    for i in range(3):
        print(f"  x{i+1:<4}"
              f"{mf[i]:.3f} / {gt['first_order'][i]:<11.3f}"
              f"{mt[i]:.3f} / {gt['total_order'][i]:<11.3f}"
              f"{vf[i]:.3f}")

    print("\n  Reading it:")
    print("   - x1, x2 drive the MEAN (first-order); x3's mean first-order ~ 0.")
    print("   - x3's mean TOTAL-order > 0 — it lives entirely in the x1-x3 pair.")
    print(f"   - x3 dominates the VARIANCE spectrum (S^h = {vf[2]:.3f}):")
    print("     a hidden driver, invisible to mean first-order importance.")

    top_pair = max(sobol['mean_sobol']['second_order'].items(),
                   key=lambda kv: kv[1])
    print(f"\n  Top mean interaction: x{top_pair[0][0]+1}-x{top_pair[0][1]+1} "
          f"= {top_pair[1]:.4f}")

    # ---- Interaction discovery (residual sieve) ------------------------
    print("\n" + "-" * 70)
    print("INTERACTION DISCOVERY — sieve on a first-order-only fit")
    print("-" * 70)
    fo_cfg = dict(CONFIG)
    fo_cfg['stages'] = ['A']
    fo_model, _ = _fit_silent(fo_cfg, data['x_train'], data['y_train'],
                              data['x_val'], data['y_val'])
    sieve = scan_missing_pairs(fo_model, data['x_train'], data['y_train'],
                               selected_pairs=[], K2=CONFIG['K2'], verbose=False)
    print("  Residual variance a missing pair would capture (ranked):")
    for (i, j), score in sieve.ranked_pairs:
        flag = '  <-- rediscovered' if score > 0.1 else ''
        print(f"    x{i+1}-x{j+1}: {score:.4f}{flag}")

    # ---- Calibration ----------------------------------------------------
    print("\n" + "-" * 70)
    print("CALIBRATION of the heteroscedastic model")
    print("-" * 70)
    cal = calibration_report(model, data['x_test'], data['y_test'])
    print(f"  mean(z):      {cal['mean_standardized_residual']:+.3f}  (target 0)")
    print(f"  var(z):       {cal['var_standardized_residual']:.3f}  (target 1)")
    print(f"  coverage 90%: {cal['coverage_0.9']:.3f}  (target 0.90)")
    print(f"  coverage 95%: {cal['coverage_0.95']:.3f}  (target 0.95)")

    # ---- Bootstrap CIs for the ellipses --------------------------------
    mean_ci = var_ci = None
    if N_BOOT > 0:
        print("\n" + "-" * 70)
        print(f"BOOTSTRAP dual-Sobol CIs  ({N_BOOT} refits)")
        print("-" * 70)
        mean_ci, var_ci = bootstrap_dual_sobol_ci(
            data, CONFIG, N_BOOT, BOOT_SUBSAMPLE, seed=7)
        for i in range(3):
            print(f"  x{i+1}: mean S1 in [{mean_ci[i][0]:.3f}, {mean_ci[i][1]:.3f}]"
                  f"   var S1 in [{var_ci[i][0]:.3f}, {var_ci[i][1]:.3f}]")

    # ---- Figures --------------------------------------------------------
    print("\n" + "-" * 70)
    print("FIGURES")
    print("-" * 70)
    plot_dual_sobol(sobol, VAR_NAMES,
                    save_path='figures/ishigami_dual_sobol.png')
    print("  figures/ishigami_dual_sobol.png       (paired mean/variance bars)")

    plot_sensitivity_ellipses(
        sobol, VAR_NAMES, mode='glyph',
        title='Ishigami: dual-sensitivity glyphs',
        save_path='figures/ishigami_sensitivity_glyphs.png')
    print("  figures/ishigami_sensitivity_glyphs.png     (NEW: ellipse glyphs)")

    plot_sensitivity_ellipses(
        sobol, VAR_NAMES, mode='plane', mean_ci=mean_ci, var_ci=var_ci,
        ci_scale=8.0, title='Ishigami: dual-sensitivity plane (CI ×8)',
        save_path='figures/ishigami_sensitivity_plane.png')
    print("  figures/ishigami_sensitivity_plane.png      (NEW: quantitative plane)")

    plot_component_functions(model, [0, 1, 2], VAR_NAMES,
                             save_path='figures/ishigami_components.png')
    print("  figures/ishigami_components.png       (learned mean effects)")

    print("\nDone.")


if __name__ == '__main__':
    main()
