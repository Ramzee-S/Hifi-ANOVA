"""Variance accounting, calibration, and correlation diagnostics."""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Optional

from .sobol import compute_sobol_indices, compute_correlative_sobol


def variance_accounting_report(model, x_data: jnp.ndarray,
                               y_data: jnp.ndarray) -> dict:
    """Complete hierarchical variance accounting.

    Computes:
    - Per-variable first-order variance (analytic)
    - Per-pair second-order variance (analytic)
    - Residual NN variance (empirical)
    - Total Var(y) vs sum of components (additivity check)
    """
    sobol_results = compute_sobol_indices(model, x_data)
    va = sobol_results['variance_accounting']

    # Empirical total variance of y
    total_var_y = float(jnp.var(y_data))

    # Model predictions
    mean_pred, var_pred = model.predict(x_data)
    residuals = y_data - mean_pred
    empirical_residual_var = float(jnp.var(residuals))

    va['total_var_y'] = total_var_y
    va['empirical_residual_var'] = empirical_residual_var
    va['R_squared'] = 1.0 - empirical_residual_var / total_var_y if total_var_y > 0 else 0.0
    va['additivity_gap'] = abs(total_var_y - va['total_model_variance'] - empirical_residual_var) / total_var_y if total_var_y > 0 else 0.0

    return va


def calibration_report(model, x_data: jnp.ndarray,
                       y_data: jnp.ndarray) -> dict:
    """Calibration check for the heteroscedastic model.

    Computes standardized residuals z_n = (y_n - f_hat(x_n)) / sigma_hat(x_n).
    Checks:
      - mean(z) approx 0
      - var(z) approx 1
    """
    mean_pred, var_pred = model.predict(x_data)
    sigma_pred = jnp.sqrt(var_pred)

    residuals = y_data - mean_pred
    standardized = residuals / sigma_pred

    z = np.array(standardized)

    report = {
        'mean_standardized_residual': float(np.mean(z)),
        'var_standardized_residual': float(np.var(z)),
        'std_standardized_residual': float(np.std(z)),
        'skewness': float(np.mean((z - np.mean(z))**3) / np.std(z)**3),
        'kurtosis': float(np.mean((z - np.mean(z))**4) / np.std(z)**4 - 3.0),
    }

    # Coverage at various levels
    for alpha in [0.5, 0.9, 0.95, 0.99]:
        from scipy.stats import norm
        z_crit = norm.ppf((1 + alpha) / 2)
        coverage = float(np.mean(np.abs(z) <= z_crit))
        report[f'coverage_{alpha}'] = coverage

    return report


def correlation_diagnostic(model, x_data: jnp.ndarray,
                          variable_names: Optional[list] = None) -> dict:
    """Diagnose the impact of input correlations on the Sobol decomposition.

    Compares structural (analytic, independence-assuming) indices against
    correlative (empirical, correlation-aware) indices. The divergence
    between them quantifies how much input correlations affect attribution.

    Args:
        model: fitted HiFiANOVA
        x_data: (N, D) input data (transformed)
        variable_names: optional names for reporting

    Returns:
        dict with:
          structural_indices: {i: S_i^struct}
          correlative_indices: {i: S_i^corr}
          divergence: {i: |S_i^struct - S_i^corr|}
          max_divergence: scalar
          cross_correlation_matrix: (D, D)
          max_abs_cross_correlation: scalar
          recommendation: string
    """
    D = model.D
    if variable_names is None:
        variable_names = [f"x{i+1}" for i in range(D)]

    # Structural indices (analytic G, assumes independence)
    struct_results = compute_sobol_indices(model)
    structural = struct_results['mean_sobol']['first_order']

    # Correlative indices (empirical, respects data correlations)
    corr_results = compute_correlative_sobol(model, x_data)
    correlative = corr_results['first_order']

    # Divergence
    divergence = {}
    for i in range(D):
        divergence[i] = abs(structural.get(i, 0) - correlative.get(i, 0))

    max_div = max(divergence.values()) if divergence else 0.0
    max_cross = corr_results['max_abs_cross_correlation']

    # Recommendation
    if max_cross < 0.1 and max_div < 0.05:
        recommendation = (
            "Input correlations are negligible. Structural (analytic) "
            "indices are reliable and sum to 1."
        )
    elif max_cross < 0.3 and max_div < 0.15:
        recommendation = (
            "Mild input correlations detected. Structural indices are "
            "approximate. Report both types for transparency."
        )
    else:
        recommendation = (
            "Strong input correlations detected. Structural indices may "
            "be misleading. Correlative indices better reflect the data "
            "distribution but do not sum to 1. Consider the interpretive "
            "implications."
        )

    return {
        'structural_indices': structural,
        'correlative_indices': correlative,
        'divergence': divergence,
        'max_divergence': max_div,
        'cross_correlation_matrix': corr_results['cross_correlation_matrix'],
        'max_abs_cross_correlation': max_cross,
        'correlation_level': corr_results['correlation_level'],
        'sum_structural': sum(structural.values()),
        'sum_correlative': corr_results['sum_of_correlative_indices'],
        'recommendation': recommendation,
        'variable_names': variable_names,
    }


def verify_model(model, x_test, y_test, x_train=None, feature_names=None,
                 verbose=True) -> dict:
    """Self-consistency health check for a fitted HiFiANOVA model.

    Runs the diagnostic workflow end-to-end and reports pass / warn / fail for
    each check, so you can confirm a fit is trustworthy *before* reading Sobol
    indices off it. This complements the three regularization paths
    (``compute_reg_path``, ``plot_pareto_frontier``, ``compute_variance_reg_path``):
    those show how attributions move with the penalty; this confirms the fitted
    model is internally consistent at the chosen penalty.

    Checks:
      - **Sobol additivity** — structural first+second+third+residual ~ 1.
      - **Sobol bounds** — indices in [0, 1]; total-order >= first-order per var.
      - **Fit quality** — test R^2 (informational; warns if <= 0).
      - **Calibration** (heteroscedastic only) — prediction-interval coverage
        near nominal and standardized residuals with mean ~ 0, var ~ 1.
      - **Input correlation** (if ``x_train`` given) — flags when the structural
        (independence-assuming) indices may misattribute.
    It also reports variables whose first-order effect is exactly zero while
    their total-order is not (pure-interaction variables, e.g. after
    ``first_order_pruning`` or Ishigami x3).

    Args:
        model: a fitted HiFiANOVA.
        x_test, y_test: held-out data in [0,1] quantile space / original scale.
        x_train: optional training inputs, enables the correlation check.
        feature_names: optional labels.
        verbose: print a formatted report.

    Returns:
        dict: {'checks': [{'name','status','value','detail'}], 'all_pass': bool,
        'sobol': <compute_sobol_indices output>}.
    """
    checks = []

    def add(name, status, value, detail):
        checks.append({'name': name, 'status': status,
                       'value': value, 'detail': detail})

    sobol = compute_sobol_indices(model, x_test)
    ms = sobol['mean_sobol']
    D = model.D
    names = feature_names or [f"x{i+1}" for i in range(D)]

    # 1. Additivity — structural indices sum to ~1.
    s_first = sum(ms['first_order'].values())
    s_second = sum(ms.get('second_order', {}).values())
    s_third = sum(ms.get('third_order', {}).values())
    s_res = ms.get('residual', 0.0)
    total = float(s_first + s_second + s_third + s_res)
    add('Sobol additivity', 'pass' if abs(total - 1.0) < 0.02 else 'warn',
        total, f"first+second+third+residual = {total:.3f} (target 1.000)")

    # 2. Bounds and total-order >= first-order.
    bounds_ok, worst = True, None
    for i in range(D):
        s1 = float(ms['first_order'].get(i, 0.0))
        st = float(ms['total_order'].get(i, s1))
        if s1 < -1e-6 or s1 > 1.0 + 1e-6 or st < s1 - 1e-6 or st > 1.0 + 1e-6:
            bounds_ok, worst = False, (names[i], s1, st)
    add('Sobol bounds', 'pass' if bounds_ok else 'fail', bounds_ok,
        'indices in [0,1], total-order >= first-order'
        if bounds_ok else f"violated at {worst}")

    # 3. Fit quality — test R^2.
    mp = np.asarray(model.predict_mean_only(x_test))
    yt = np.asarray(y_test)
    vy = float(np.var(yt))
    r2 = 1.0 - float(np.var(yt - mp)) / vy if vy > 0 else 0.0
    add('Fit quality (test R^2)', 'pass' if r2 > 0 else 'warn', r2,
        f"R^2 = {r2:.3f}")

    # 4. Calibration — heteroscedastic models only.
    if getattr(model, 'variance_model', None) is not None:
        cal = calibration_report(model, x_test, y_test)
        c90, c95 = cal['coverage_0.9'], cal['coverage_0.95']
        vz = cal['var_standardized_residual']
        cok = abs(c90 - 0.9) < 0.06 and abs(c95 - 0.95) < 0.05 and 0.75 <= vz <= 1.3
        add('Calibration (coverage)', 'pass' if cok else 'warn', (c90, c95),
            f"cov90={c90:.2f} cov95={c95:.2f} var(z)={vz:.2f}")

    # 5. Input correlation — structural indices assume independence.
    if x_train is not None:
        try:
            lvl = compute_sobol_indices(model, x_train).get(
                'correlation_level', 'clean')
            add('Input correlation', 'pass' if lvl in ('clean', 'mild') else 'warn',
                lvl, f"correlation_level = '{lvl}'")
        except Exception:
            pass

    # Info: pure-interaction variables (zero main effect, nonzero total).
    omitted = [names[i] for i in range(D)
               if float(ms['first_order'].get(i, 0.0)) == 0.0
               and float(ms['total_order'].get(i, 0.0)) > 1e-6]
    if omitted:
        add('Pure-interaction variables', 'info', omitted,
            f"{', '.join(omitted)}: zero first-order, nonzero total-order")

    all_pass = all(c['status'] in ('pass', 'info') for c in checks)

    if verbose:
        sym = {'pass': '[PASS]', 'warn': '[WARN]', 'fail': '[FAIL]', 'info': '[info]'}
        print("Model verification")
        for c in checks:
            print(f"  {sym[c['status']]} {c['name']}: {c['detail']}")
        print(f"  => {'ALL CHECKS PASSED' if all_pass else 'SOME CHECKS NEED ATTENTION'}")

    return {'checks': checks, 'all_pass': all_pass, 'sobol': sobol}
