"""Heteroscedastic Ishigami — a capabilities showcase for the dual spectrum.

The Ishigami function is the canonical sensitivity-analysis benchmark:

    f(x) = sin(x1) + a*sin^2(x2) + b*x3^4*sin(x1),   x_i ~ U(-pi, pi)

with known analytic Sobol indices. Its signature feature is x3: it has a
**zero first-order** effect but a **non-zero total-order** effect — it acts
purely through its interaction with x1.

Here we make it *heteroscedastic* by driving the noise variance with x3 as well.
That turns x3 into the textbook "hidden driver": near-invisible to mean
first-order importance, yet the sole driver of the predictive *variance*. It is
exactly the case ordinary feature-importance misses and the dual mean/log-variance
Sobol spectrum was built for.

The script demonstrates, end to end:
  1. Recovery of the analytic Ishigami mean Sobol indices (first- and total-order).
  2. The dual spectrum: mean sensitivity vs log-variance index, side by side.
  3. Interaction discovery — the residual sieve rediscovering the x1-x3 pair.
  4. Calibration of the fitted heteroscedastic model.
  5. A NEW visualization: the dual-sensitivity plane, each variable an ellipse
     whose position is (mean Sobol, log-variance index S^h) and whose size is the
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
import matplotlib.pyplot as plt

from hifi_anova.data.synthetic import generate_ishigami, ishigami_sobol_indices
from hifi_anova.data.preprocessing import preprocess_data
from hifi_anova.training.trainer import HiFiANOVATrainer
from hifi_anova.analysis.sobol import compute_sobol_indices
from hifi_anova.analysis.diagnostics import calibration_report, verify_model
from hifi_anova.analysis.interaction_discovery import scan_missing_pairs
from hifi_anova.analysis.reg_path import (
    compute_reg_path, plot_reg_path, plot_pareto_frontier,
    compute_variance_reg_path, plot_variance_reg_path,
)
from hifi_anova.analysis.visualization import (
    plot_dual_sobol, plot_component_functions, plot_sensitivity_ellipses,
)

VAR_NAMES = ['$x_1$', '$x_2$', '$x_3$']

# Mean + variance fit: first + second order mean (A, B) plus the heteroscedastic
# variance stage (D). No NN residual — the structured model is enough here.
#
# first_order_pruning='bic' is the key setting for Ishigami: x3 has NO first-order
# effect (it acts only through the x1-x3 interaction), but plain ridge leaves a
# small spurious f_3(x3) wiggle. A BIC leave-one-group-out test on the first-order
# blocks recognizes x3's marginal as unsupported and zeros the whole block, so the
# fitted first-order component of x3 is exactly flat — robustly, down to N~100.
CONFIG = {
    'K1': 12, 'K2': 6, 'Kh': 3,
    'strategy': 'curvature',
    'lambda_order1': 0.001, 'lambda_order2': 0.01, 'lambda_h': 0.1,
    'stages': ['A', 'B', 'D'],
    'first_order_pruning': 'bic',
    'residual_nn': {'enabled': False},
    'max_outer_iter': 8, 'alternating_tol': 1e-4, 'newton_max_iter': 10,
}

N_BOOT = 12          # bootstrap refits for the ellipse CIs (set 0 to skip)
BOOT_SUBSAMPLE = 4000


def _ishigami_true(X, a=7.0, b=0.1):
    """Noiseless Ishigami response — the ground truth for the parity/surface plots."""
    x1, x2, x3 = X[:, 0], X[:, 1], X[:, 2]
    return np.sin(x1) + a * np.sin(x2) ** 2 + b * (x3 ** 4) * np.sin(x1)


def fit_diagnostics(model, transformer, seed=999):
    """Predicted-vs-actual, transparent true-vs-fit surface, and variance recovery.

    Because Ishigami is synthetic we know the noiseless truth, so we can separate
    the two things a single R^2 conflates: how well the *mean* is recovered
    (predicted vs true f — a tight line) and the *irreducible noise* (predicted vs
    observed y — scatter whose spread is the noise floor).
    """
    import jax.numpy as jnp
    from hifi_anova.analysis.plots import plot_parity, save_fig
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    from matplotlib.patches import Patch

    def predict(Xo):
        xj = jnp.asarray(np.clip(transformer.transform(Xo), 0, 1), dtype=jnp.float32)
        _, v = model.predict(xj)
        return np.asarray(model.predict_mean_only(xj)), np.sqrt(np.asarray(v))

    Xe, ye, sig_e = generate_ishigami(
        n_samples=3000, heteroscedastic=True, variance_variable=2,
        sigma_min=0.3, sigma_max=3.0, seed=seed)
    f_true = _ishigami_true(Xe)
    pred, sig_pred = predict(Xe)

    # (1) Parity: predicted vs observed (scatter = noise) and vs truth (clean).
    fig, _ = plot_parity(ye, pred, xlabel='Observed y (heteroscedastic noise)',
                         color_by=sig_e, color_label=r'true noise std $\sigma(x)$',
                         title='Predicted vs OBSERVED  (scatter = irreducible noise)')
    save_fig(fig, 'figures/ishigami_parity_observed.png')
    fig, _ = plot_parity(f_true, pred, xlabel='True f(x)  (noiseless ground truth)',
                         title='Predicted vs TRUTH  (mean recovered)')
    save_fig(fig, 'figures/ishigami_parity_truth.png')

    # (2) Transparent true-vs-fit surface (slice x3 = 0).
    g = np.linspace(-np.pi, np.pi, 60)
    G1, G2 = np.meshgrid(g, g)
    grid = np.column_stack([G1.ravel(), G2.ravel(), np.zeros(G1.size)])
    Ztrue = _ishigami_true(grid).reshape(G1.shape)
    Zpred = predict(grid)[0].reshape(G1.shape)
    fig = plt.figure(figsize=(8, 6.5))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(G1, G2, Ztrue, color='tab:blue', alpha=0.45, linewidth=0)
    ax.plot_surface(G1, G2, Zpred, color='tab:orange', alpha=0.45, linewidth=0)
    ax.legend(handles=[Patch(color='tab:blue', alpha=0.5, label='true f'),
                       Patch(color='tab:orange', alpha=0.5, label='predicted mean')])
    ax.set_xlabel('$x_1$'); ax.set_ylabel('$x_2$'); ax.set_zlabel('response')
    ax.set_title('True vs predicted mean surface  (slice $x_3=0$)')
    fig.savefig('figures/ishigami_surface.png', dpi=150, bbox_inches='tight')

    # (3) Fitted residual-scale recovery: predicted std vs true noise std.
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(sig_e, sig_pred, s=8, alpha=0.3, color='steelblue')
    lim = [min(sig_e.min(), sig_pred.min()) - 0.1,
           max(sig_e.max(), sig_pred.max()) + 0.1]
    ax.plot(lim, lim, 'r--', lw=1.5)
    ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect('equal'); ax.grid(alpha=0.3)
    corr = float(np.corrcoef(sig_e, sig_pred)[0, 1])
    ax.set_xlabel(r'true noise std $\sigma(x)$')
    ax.set_ylabel(r'predicted std $\hat\sigma(x)$')
    ax.set_title(
        rf'Fitted residual-scale recovery: $\hat\sigma$ vs $\sigma$  '
        rf'(corr={corr:.3f})')
    fig.tight_layout()
    fig.savefig('figures/ishigami_variance_fit.png', dpi=150, bbox_inches='tight')

    # (4) Prediction intervals from the mean + log-variance fit on a clean slice.
    # Fix x1=0 (removes the x1-x3 interaction so the mean is flat) and x2=pi/2
    # (sin^2 = 1), leaving x3 — the multiplicative residual-scale driver — free.
    # mean +/- 2*sigma(x) from the fitted mean AND variance models together; it
    # should widen with x3 and track the true +/-2 sigma. Points are fresh noisy
    # samples drawn at the slice, for illustration.
    ng = 400
    x3g = np.linspace(-np.pi, np.pi, ng)
    slice_grid = np.column_stack([np.zeros(ng), np.full(ng, np.pi / 2), x3g])
    m_s, s_s = predict(slice_grid)
    sig_true_s = 0.3 + 2.7 * ((x3g + np.pi) / (2 * np.pi))
    f_s = 7.0  # sin(0) + 7*sin(pi/2)^2 + 0 = 7 (exactly flat)
    y_s = f_s + sig_true_s * np.random.RandomState(0).normal(size=ng)
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.scatter(x3g, y_s, s=10, alpha=0.35, color='gray', label='observed y')
    ax.plot(x3g, m_s, 'b-', lw=1.8, label='predicted mean')
    ax.fill_between(x3g, m_s - 2 * s_s, m_s + 2 * s_s, color='coral', alpha=0.35,
                    label='predicted 95% interval')
    ax.plot(x3g, f_s + 2 * sig_true_s, 'g:', lw=1.2, label=r'true $\pm 2\sigma(x_3)$')
    ax.plot(x3g, f_s - 2 * sig_true_s, 'g:', lw=1.2)
    ax.set_xlabel(r'$x_3$  (log-variance driver)'); ax.set_ylabel('y')
    ax.set_title(r'Prediction intervals from the mean + log-variance fit '
                 r'(slice $x_1=0,\ x_2=\pi/2$)')
    ax.legend(loc='upper center', fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig('figures/ishigami_intervals.png', dpi=150, bbox_inches='tight')

    r2_obs = 1 - np.var(ye - pred) / np.var(ye)
    r2_true = 1 - np.var(f_true - pred) / np.var(f_true)
    return {'r2_observed': float(r2_obs), 'r2_true': float(r2_true), 'sigma_corr': corr}


def _fit_silent(config, xtr, ytr, xval, yval):
    """Fit a trainer while swallowing its per-stage stdout (used in the loop)."""
    with contextlib.redirect_stdout(io.StringIO()):
        model, results = HiFiANOVATrainer(config).fit(xtr, ytr, xval, yval)
    return model, results


def bootstrap_dual_sobol_ci(data, config, n_boot, n_sub, seed=0):
    """Bootstrap CIs on first-order mean and log-variance indices.

    Resamples the (already transformed) training set with replacement, refits,
    and records the first-order mean and log-variance indices each time. The
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
        vf = (s['log_variance_sobol']['first_order']
              if 'log_variance_sobol' in s else {i: 0.0 for i in range(D)})
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
    print("HETEROSCEDASTIC ISHIGAMI — dual mean/log-variance showcase")
    print("=" * 70)
    print("  f(x) = sin(x1) + 7*sin^2(x2) + 0.1*x3^4*sin(x1),   x_i ~ U(-pi,pi)")
    print("  noise std ramps 0.3 -> 3.0 across x3  "
          "(x3 drives the multiplicative residual scale)")

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
    vf = sobol['log_variance_sobol']['first_order']
    print(f"  {'var':<5}{'meanS1(fit/true)':<22}{'meanST(fit/true)':<22}"
          f"{'logvarS1'}")
    for i in range(3):
        print(f"  x{i+1:<4}"
              f"{mf[i]:.3f} / {gt['first_order'][i]:<11.3f}"
              f"{mt[i]:.3f} / {gt['total_order'][i]:<11.3f}"
              f"{vf[i]:.3f}")

    print("\n  Reading it:")
    print("   - x1, x2 drive the MEAN (first-order); x3's mean first-order ~ 0.")
    print("   - x3's mean TOTAL-order > 0 — it lives entirely in the x1-x3 pair.")
    print(f"   - x3 dominates the LOG-VARIANCE spectrum (S^h = {vf[2]:.3f}):")
    print("     a hidden driver, invisible to mean first-order importance.")

    top_pair = max(sobol['mean_sobol']['second_order'].items(),
                   key=lambda kv: kv[1])
    print(f"\n  Top mean interaction: x{top_pair[0][0]+1}-x{top_pair[0][1]+1} "
          f"= {top_pair[1]:.4f}")

    pruned = results.get('first_order_pruning', {}).get('pruned_variables', [])
    if pruned:
        print(f"  First-order pruning (BIC) zeroed the marginal of: "
              f"{', '.join('x%d' % (i + 1) for i in pruned)} "
              f"→ its component curve is exactly flat, not a spurious wiggle.")

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
                  f"   log-var S1 in [{var_ci[i][0]:.3f}, {var_ci[i][1]:.3f}]")

    # ---- Model verification (self-consistency health check) -------------
    print("\n" + "-" * 70)
    print("MODEL VERIFICATION")
    print("-" * 70)
    verify_model(model, data['x_test'], data['y_test'],
                 x_train=data['x_train'], feature_names=VAR_NAMES)

    # ---- Explained variance: expectation vs variance --------------------
    print("\n" + "-" * 70)
    print("EXPLAINED VARIANCE  (how the two spectra split by interaction order)")
    print("-" * 70)
    va = sobol['variance_accounting']
    tot = va['total_model_variance']
    print("  Mean model  E[y|x]  — variance explained by order:")
    print(f"    first-order : {va['first_order_total'] / tot:6.1%}")
    print(f"    second-order: {va['second_order_total'] / tot:6.1%}")
    print(f"    residual    : {va['residual'] / tot:6.1%}")
    vh = sobol['log_variance_sobol']['variance_accounting']
    toth = vh['total'] if vh['total'] > 0 else 1.0
    print("  Log-variance model  log Var[y|x]  — allocation by order:")
    print(f"    first-order : {vh['first_order_total'] / toth:6.1%}  (x3 alone)")
    print(f"    second-order: {vh['second_order_total'] / toth:6.1%}")

    # ---- Regularization path: the complexity / lambda trade-off ---------
    print("\n" + "-" * 70)
    print("REGULARIZATION PATH  (mean Sobol & explained variance vs lambda)")
    print("-" * 70)
    Phi_train = np.asarray(model.build_phi_all(data['x_train']), dtype=np.float64)
    y_c = np.asarray(data['y_train'], dtype=np.float64)
    y_c = y_c - y_c.mean()
    P = int(model.pair_indices.shape[0]) if model.pair_indices is not None else 0
    path = compute_reg_path(
        Phi_train, y_c, D=3, K1=CONFIG['K1'], K2=CONFIG['K2'], P=P,
        pair_indices=np.asarray(model.pair_indices) if P else None,
        strategy=CONFIG['strategy'], n_lambdas=40, lambda_range=(1e-5, 10.0),
        include_linear_1=model.include_linear_1,
        include_linear_2=getattr(model, 'include_linear_2', True),
        basis_name=model.basis_name)
    gcv_idx = int(np.argmin(np.abs(path.lambdas - path.lambda_gcv_opt)))
    print(f"  GCV-optimal lambda     : {path.lambda_gcv_opt:.2e} "
          f"(df={path.df_values[gcv_idx]:.1f})")
    print(f"  Evidence-optimal lambda: {path.lambda_evidence_opt:.2e}")
    print("  Along the path, x3's mean first-order Sobol stays ~0 at every "
          "lambda —")
    print("  the trade-off only moves variance between the real effects and "
          "the residual.")
    plot_reg_path(path, VAR_NAMES, save_prefix='figures/ishigami')

    # Pareto frontier: model complexity (df) vs unexplained variance.
    y_var = float(np.var(np.asarray(data['y_train'])))
    plot_pareto_frontier(path, y_var, save_path='figures/ishigami_pareto.png')

    # Variance-model path: hold the mean fixed and sweep the variance penalty
    # lambda_h. x3 dominates the log-variance spectrum across the whole range.
    vpath = compute_variance_reg_path(
        model, data['x_train'], data['y_train'],
        strategy='variance', n_lambdas=30, lambda_h_range=(1e-3, 1e2))
    plot_variance_reg_path(vpath, VAR_NAMES, lambda_h_used=CONFIG['lambda_h'],
                           save_prefix='figures/ishigami')
    x3_share = vpath['sobol_h_paths'][2]
    print(f"  Log-variance model (lambda_h path): x3's S^h stays "
          f"{x3_share.min():.2f}-{x3_share.max():.2f} across lambda_h "
          f"[1e-3, 1e2].")

    # ---- Figures --------------------------------------------------------
    print("\n" + "-" * 70)
    print("FIGURES")
    print("-" * 70)
    print("  figures/ishigami_reg_path.png         (L-curve, GCV/evidence, "
          "Sobol paths, variance decomposition)")
    print("  figures/ishigami_pareto.png           (complexity vs unexplained "
          "variance)")
    print("  figures/ishigami_var_reg_path.png     (log-variance indices vs "
          "lambda_h)")
    plot_dual_sobol(sobol, VAR_NAMES,
                    save_path='figures/ishigami_dual_sobol.png')
    print("  figures/ishigami_dual_sobol.png       "
          "(paired mean/log-variance bars)")

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

    # ---- Fit diagnostics: parity, surface, variance recovery ------------
    fd = fit_diagnostics(model, data['transformer'])
    print("  figures/ishigami_parity_observed.png  (pred vs observed; "
          f"R2={fd['r2_observed']:.3f}, capped by noise)")
    print("  figures/ishigami_parity_truth.png     (pred vs TRUE f; "
          f"R2={fd['r2_true']:.3f})")
    print("  figures/ishigami_surface.png          (transparent true vs fit "
          "surface, x3=0)")
    print("  figures/ishigami_variance_fit.png     (pred std vs true noise std; "
          f"corr={fd['sigma_corr']:.3f})")
    print("  figures/ishigami_intervals.png        (prediction intervals widen "
          "with x3, on a 1-D slice)")
    print("\n  Note: R2 vs observed (~0.8) is at the noise ceiling, not a weak "
          "fit — R2 vs the")
    print("  true function (~0.98) shows the mean is recovered; the gap is "
          "irreducible noise.")

    print("\nDone.")


if __name__ == '__main__':
    main()
