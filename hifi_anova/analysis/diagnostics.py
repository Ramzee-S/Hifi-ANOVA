"""Variance accounting, calibration, and correlation diagnostics."""

from ..array_backend import xp as jnp  # switchable array backend (numpy exact core)
import numpy as np
from typing import Optional

from .sobol import compute_sobol_indices, compute_correlative_sobol


def _u_center(A):
    """U-centered distance matrix (Székely–Rizzo 2014). Unlike double-centering,
    the U-centered inner product is an UNBIASED estimator of squared distance
    covariance — its expectation is 0 under independence, removing the positive
    small-sample / high-dimension bias that makes the ordinary (biased) distance
    correlation exceed fixed thresholds on genuinely independent data."""
    n = A.shape[0]
    r = A.sum(1)
    t = A.sum()
    U = (A - r[:, None] / (n - 2) - r[None, :] / (n - 2)
         + t / ((n - 1) * (n - 2)))
    np.fill_diagonal(U, 0.0)
    return U


def _input_dependence(x_data, alpha: float = 0.05, cap: int = 1000,
                      cap_perm: int = 256, n_perm: int = 199,
                      top_k: int = 8, seed: int = 0) -> dict:
    """Assess input dependence, separating STATISTICAL EVIDENCE from EFFECT SIZE.

    Independence — not merely zero correlation — is the framework's assumption, so
    we test both a LINEAR (Pearson) and a NONLINEAR (distance-correlation) channel:

      - Linear: max |Pearson| effect size + an analytic two-sided t-test p-value.
      - Nonlinear: the UNBIASED (U-centered) distance correlation as effect size
        (bias-corrected, so independent data reads ~0), and a PERMUTATION p-value
        (the Székely–Rizzo t-test is miscalibrated for 1-D marginals, so it is
        not used). Permutation is run only on the top-``top_k`` pairs by effect —
        the most likely to be significant — so cost is O(top_k · n_perm · n²),
        independent of the number of variables.

    Both channels apply a Bonferroni correction across all D(D-1)/2 pairs. A
    channel counts toward the dependence level only when it is BOTH statistically
    significant AND has effect size ≥ the level threshold — so neither a large but
    insignificant small-sample fluctuation nor a significant but negligible effect
    is reported as an assumption violation. Distance rows are subsampled to
    ``cap`` (effect) / ``cap_perm`` (permutation) for tractability.

    Returns a dict with max_abs_input_correlation, linear_significant,
    max_abs_distance_correlation, nonlinear_significant, min_pvalue,
    dependence_level ('clean'/'mild'/'strong'), and nonlinear_dominant.
    """
    from scipy import stats
    x = np.asarray(x_data, dtype=np.float64)
    n_full = x.shape[0]
    D = x.shape[1] if x.ndim > 1 else 1
    out = {'max_abs_input_correlation': 0.0, 'linear_significant': False,
           'max_abs_distance_correlation': 0.0, 'nonlinear_significant': False,
           'min_pvalue': 1.0, 'dependence_level': 'clean',
           'nonlinear_dominant': False}
    if D < 2 or n_full < 5:
        return out
    npair = D * (D - 1) // 2

    # --- Linear channel: Pearson effect + analytic two-sided t-test (Bonferroni)
    C = np.corrcoef(x.T)
    np.fill_diagonal(C, 0.0)
    out['max_abs_input_correlation'] = float(np.max(np.abs(C)))
    iu = np.triu_indices(D, 1)
    r = np.clip(C[iu], -0.999999, 0.999999)
    tt = r * np.sqrt((n_full - 2) / (1.0 - r * r))
    p_lin = np.minimum(1.0, 2.0 * stats.t.sf(np.abs(tt), df=n_full - 2) * npair)
    lin_min_p = float(np.min(p_lin))
    out['linear_significant'] = bool(lin_min_p < alpha)

    # --- Nonlinear channel: unbiased dCor effect size (subsample cap) ---
    xe = x[np.linspace(0, n_full - 1, cap).astype(int)] if n_full > cap else x
    ne = xe.shape[0]
    nl_min_p = 1.0
    if ne >= 8:
        U, dv = [], []
        for i in range(D):
            a = xe[:, i]
            Ui = _u_center(np.abs(a[:, None] - a[None, :]))
            U.append(Ui)
            dv.append(float((Ui * Ui).sum() / (ne * (ne - 3))))
        pairs = []
        for i in range(D):
            for j in range(i + 1, D):
                if dv[i] > 0 and dv[j] > 0:
                    R = float((U[i] * U[j]).sum() / (ne * (ne - 3))) \
                        / np.sqrt(dv[i] * dv[j])
                else:
                    R = 0.0
                pairs.append((max(R, 0.0), i, j))
        pairs.sort(reverse=True)
        out['max_abs_distance_correlation'] = pairs[0][0] if pairs else 0.0

        # MAX-STATISTIC (Westfall–Young) permutation test over the top-k pairs by
        # effect: FWER-controlled WITHOUT Bonferroni, so the minimum achievable
        # p is 1/(n_perm+1) regardless of D. (Per-pair p × Bonferroni would floor
        # the achievable p at npair/(n_perm+1) — e.g. 0.14 at D=8 — silently
        # disabling the channel for larger D.) Each variable is permuted
        # independently per iteration → the complete (all-independent) null.
        tested = [(i, j) for _, i, j in pairs[:top_k] if _ > 0]
        xp = xe[np.linspace(0, ne - 1, cap_perm).astype(int)] if ne > cap_perm else xe
        npp = xp.shape[0]
        if npp >= 8 and tested:
            g = np.random.default_rng(seed)
            need = {i for i, j in tested} | {j for i, j in tested}
            Up = {i: _u_center(np.abs(xp[:, i][:, None] - xp[:, i][None, :]))
                  for i in need}
            nrm = {i: np.sqrt((Up[i] * Up[i]).sum()) for i in need}  # perm-invariant
            obs = max((Up[i] * Up[j]).sum() / (nrm[i] * nrm[j]) for i, j in tested)
            ge = 1
            for _p in range(n_perm):
                P = {i: g.permutation(npp) for i in need}
                UP = {i: Up[i][np.ix_(P[i], P[i])] for i in need}
                m = max((UP[i] * UP[j]).sum() / (nrm[i] * nrm[j]) for i, j in tested)
                if m >= obs:
                    ge += 1
            nl_min_p = ge / (n_perm + 1)
            out['nonlinear_significant'] = bool(nl_min_p < alpha)
    out['min_pvalue'] = min(lin_min_p, nl_min_p)

    # --- Level: a channel counts only if SIGNIFICANT and effect ≥ threshold ---
    lin_eff = out['max_abs_input_correlation'] if out['linear_significant'] else 0.0
    nl_eff = out['max_abs_distance_correlation'] if out['nonlinear_significant'] else 0.0
    eff = max(lin_eff, nl_eff)
    out['dependence_level'] = ('clean' if eff < 0.1
                               else ('mild' if eff < 0.3 else 'strong'))
    # Nonlinear-dominant = the nonlinear channel flags it while the linear channel
    # is not meaningfully present (so a linear-only gate would have missed it).
    out['nonlinear_dominant'] = bool(out['nonlinear_significant'] and lin_eff < 0.1)
    return out


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
    # ``R_squared`` is the framework-native explained-variance score (the
    # manuscript convention); ``R_squared_classical`` is the textbook SSE/TSS
    # coefficient of determination. Both via hifi_anova.analysis.metrics.
    va['R_squared'] = 1.0 - empirical_residual_var / total_var_y if total_var_y > 0 else 0.0
    from .metrics import r_squared as _r_squared
    va['R_squared_classical'] = _r_squared(y_data, mean_pred, 'classical')
    va['additivity_gap'] = abs(total_var_y - va['total_model_variance'] - empirical_residual_var) / total_var_y if total_var_y > 0 else 0.0

    return va


def calibration_report(model, x_data: jnp.ndarray,
                       y_data: jnp.ndarray,
                       Phi_train=None, reg_diag=None,
                       sigma2_hat=None, weights=None,
                       profile_intercept: bool = False) -> dict:
    """Calibration check for the heteroscedastic model.

    Computes standardized residuals z_n = (y_n - f_hat(x_n)) / s(x_n).

    By default s(x)^2 = sigma_hat^2(x) is the NOISE variance only — this checks
    the *variance model's* calibration, not full predictive coverage: the
    implied intervals carry no Var(f_hat(x)) (parameter uncertainty), so on
    small-N / high-df fits the reported coverage overstates what noise-only
    intervals would achieve on new data. Pass ``Phi_train`` and ``reg_diag``
    (as in ``model.predict.predict_intervals``) to standardize by the FULL
    predictive scale s(x)^2 = sigma^2(x) + var_epistemic(x) instead — the
    scale ``HiFiResult.predict_intervals`` actually uses.

    Checks:
      - mean(z) approx 0
      - var(z) approx 1
    """
    if Phi_train is not None and reg_diag is not None:
        from ..model.predict import predict_intervals as _pi
        res = _pi(model, x_data, Phi_train=Phi_train, reg_diag=reg_diag,
                  sigma2_hat=sigma2_hat, weights=weights,
                  profile_intercept=profile_intercept)
        mean_pred = res['mean']
        s_pred = np.sqrt(np.maximum(res['var_total'], 1e-300))
        z = (np.asarray(y_data) - mean_pred) / s_pred
    else:
        mean_pred, var_pred = model.predict(x_data)
        sigma_pred = jnp.sqrt(var_pred)
        residuals = y_data - mean_pred
        z = np.array(residuals / sigma_pred)

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


def independence_test(x_data, alpha: float = 0.05, seed: int = 0, **kwargs) -> dict:
    """EXPERIMENTAL nonlinear independence diagnostic (not part of the core
    workflow). Wraps :func:`_input_dependence`: unbiased (U-centered) distance
    correlation as effect size + a max-statistic (Westfall–Young) permutation test
    (FWER-controlled), plus a linear Pearson channel, both gated on significance
    AND effect size. Independence — not zero correlation — is the assumption, so
    this can flag *uncorrelated-but-dependent* inputs.

    This is an assumption-sensitivity probe, NOT verification: failing to reject
    does not prove independence, and a rejection does not license correlated-input
    attribution (out of scope — see the manuscript outlook). Cost is a permutation
    test (~0.4 s); it is never run automatically. Returns the ``_input_dependence``
    dict (dependence_level, max_abs_input_correlation, max_abs_distance_correlation,
    linear_significant, nonlinear_significant, nonlinear_dominant, min_pvalue).
    """
    return _input_dependence(x_data, alpha=alpha, seed=seed, **kwargs)


def correlation_diagnostic(model, x_data: jnp.ndarray,
                           variable_names: Optional[list] = None,
                           run_independence_test: bool = False,
                           inputs_independent_by_design: bool = False) -> dict:
    """Describe how input correlation affects the Sobol attribution (descriptive).

    HiFi-ANOVA assumes an INDEPENDENT product input measure; the reported
    attribution is the structural (reference-measure) spectrum, which describes
    the fitted function under that measure. Independence is an *assumption*, not
    something this routine verifies — so by default this reports only descriptive
    information:

      - structural vs correlative first-order shares and their divergence;
      - the max pairwise ordinary Pearson correlation of the inputs — DESCRIPTIVE
        ONLY, not proof of independence (it is blind to nonlinear dependence);
      - the (component-output) correlation level, also descriptive.

    Set ``run_independence_test=True`` to additionally run the EXPERIMENTAL
    nonlinear independence test (:func:`independence_test`; unbiased distance
    correlation + permutation) — a permutation cost, off the core path. Set
    ``inputs_independent_by_design=True`` only for controlled experiments where you
    generated the inputs independently, to record that fact
    (``input_assumption_verified=True``); for observational data independence must
    be justified externally, and dependent-input attribution is out of scope
    (Shapley / generalized ANOVA — manuscript outlook).

    Args:
        model: fitted HiFiANOVA
        x_data: (N, D) input data (transformed)
        variable_names: optional names for reporting
        run_independence_test: run the experimental nonlinear test (default False)
        inputs_independent_by_design: caller assertion for controlled experiments

    Returns:
        dict with structural_indices / correlative_indices / divergence,
        max_abs_input_correlation (descriptive Pearson),
        correlation_level (descriptive), input_assumption /
        input_assumption_verified, role, recommendation; and — only when
        ``run_independence_test`` — dependence_level, max_abs_distance_correlation,
        linear_significant, nonlinear_significant, nonlinear_dominant,
        dependence_pvalue, and the full ``independence_test`` dict.
    """
    D = model.D
    if variable_names is None:
        variable_names = [f"x{i+1}" for i in range(D)]

    # Structural first-order shares under the reference (product-of-marginals)
    # measure — CORE normalization V_i/V_core (residual-excluded, matching the
    # correlative denominator; core == total for a no-residual first-order fit).
    struct_results = compute_sobol_indices(model)
    structural = struct_results['mean_sobol_core']['first_order']

    corr_results = compute_correlative_sobol(model, x_data)
    correlative = corr_results['first_order']

    divergence = {i: abs(structural.get(i, 0) - correlative.get(i, 0))
                  for i in range(D)}
    max_div = max(divergence.values()) if divergence else 0.0
    max_cross = corr_results['max_abs_cross_correlation']

    # Descriptive ordinary Pearson correlation (NOT proof of independence).
    xin = np.asarray(x_data, dtype=np.float64)
    if D > 1:
        C = np.corrcoef(xin.T)
        np.fill_diagonal(C, 0.0)
        max_input_corr = float(np.max(np.abs(C)))
    else:
        max_input_corr = 0.0
    descriptive_level = ('clean' if max_input_corr < 0.1
                         else ('mild' if max_input_corr < 0.3 else 'strong'))

    has_higher_order = bool(corr_results['second_order'] or corr_results['third_order'])

    result = {
        'structural_indices': structural,
        'correlative_indices': correlative,
        'correlative_second_order': corr_results['second_order'],
        'correlative_third_order': corr_results['third_order'],
        'divergence': divergence,
        'max_divergence': max_div,
        'cross_correlation_matrix': corr_results['cross_correlation_matrix'],
        'max_abs_cross_correlation': max_cross,
        'max_abs_input_correlation': max_input_corr,
        # Descriptive linear bucket (Pearson only) — NOT an independence verdict.
        'correlation_level': descriptive_level,
        'component_output_correlation_level':
            corr_results['component_output_correlation_level'],
        'input_assumption': 'independent_product_measure',
        'input_assumption_verified': bool(inputs_independent_by_design),
        'role': 'independence_assumption_diagnostic',
        'has_higher_order': has_higher_order,
        'sum_structural': sum(structural.values()),
        'sum_correlative': corr_results['sum_of_correlative_indices'],
        'sum_correlative_first_order': corr_results['first_order_sum'],
        'variable_names': variable_names,
        'independence_test': None,
    }

    # Base (descriptive) recommendation.
    recommendation = (
        "Structural Sobol indices describe the fitted function under the reference "
        "independent product measure. Independence is ASSUMED, not verified"
        + (" (caller asserts inputs were generated independently)."
           if inputs_independent_by_design else
           "; for observational data it must be justified externally.")
        + f" Max ordinary |Pearson| input correlation = {max_input_corr:.2f} "
        "(descriptive only — blind to nonlinear dependence, not proof of "
        "independence). The correlative block is an optional assumption-sensitivity "
        "diagnostic, not an official estimand; dependent-input attribution is out "
        "of scope."
    )

    # Optional EXPERIMENTAL nonlinear independence test (off the core path).
    if run_independence_test:
        dep = independence_test(x_data, seed=0)
        result.update({
            'dependence_level': dep['dependence_level'],
            'max_abs_distance_correlation': dep['max_abs_distance_correlation'],
            'linear_significant': dep['linear_significant'],
            'nonlinear_significant': dep['nonlinear_significant'],
            'nonlinear_dominant': dep['nonlinear_dominant'],
            'dependence_pvalue': dep['min_pvalue'],
            'independence_test': dep,
        })
        recommendation += (
            f" Experimental independence test: {dep['dependence_level']} "
            f"(max distance-corr {dep['max_abs_distance_correlation']:.2f}, "
            f"p {dep['min_pvalue']:.3g})."
        )
        if dep['nonlinear_dominant']:
            recommendation += (
                " Dependence is largely NONLINEAR (a linear gate would miss it)."
            )
    if has_higher_order:
        recommendation += (
            " NOTE: the model retains interaction components, so the first-order "
            "first-order correlative shares are a partial collection and need "
            "not sum to 1; the complete retained allocation across first, "
            "second, and third order sums to 1, while individual shares may be "
            "negative or exceed 1."
        )
    result['recommendation'] = recommendation
    return result


def verify_model(model, x_test, y_test, x_train=None, feature_names=None,
                 verbose=True, inputs_independent_by_design=False) -> dict:
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
      - **Input independence** — records that structural indices assume an
        independent product measure (``input_assumption_verified=False`` unless the
        caller asserts it), with the descriptive max ordinary |Pearson| input
        correlation when ``x_train`` is given. This is DESCRIPTIVE and does NOT run
        the (experimental, permutation-based) nonlinear independence test — call
        ``correlation_diagnostic(model, x, run_independence_test=True)`` for that.
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

    # 3. Fit quality — test R^2. Report both conventions (explained-variance,
    # the manuscript default, and classical SSE/TSS); the pass gate stays on the
    # explained-variance value for backward compatibility.
    from .metrics import r_squared as _r_squared
    mp = np.asarray(model.predict_mean_only(x_test))
    yt = np.asarray(y_test)
    vy = float(np.var(yt))
    r2 = 1.0 - float(np.var(yt - mp)) / vy if vy > 0 else 0.0
    r2_classical = _r_squared(yt, mp, 'classical')
    add('Fit quality (test R^2)', 'pass' if r2 > 0 else 'warn', r2,
        f"R^2 = {r2:.3f} (explained-var); classical {r2_classical:.3f}")

    # 4. Calibration — heteroscedastic models only.
    if getattr(model, 'variance_model', None) is not None:
        cal = calibration_report(model, x_test, y_test)
        c90, c95 = cal['coverage_0.9'], cal['coverage_0.95']
        vz = cal['var_standardized_residual']
        cok = abs(c90 - 0.9) < 0.06 and abs(c95 - 0.95) < 0.05 and 0.75 <= vz <= 1.3
        add('Calibration (coverage)', 'pass' if cok else 'warn', (c90, c95),
            f"cov90={c90:.2f} cov95={c95:.2f} var(z)={vz:.2f}")

    # 5. Input correlation — structural indices assume independence. Use the
    # actual INPUT-dependence diagnostic (linear + nonlinear distance
    # correlation), NOT the component-output correlation (which is 0 when
    # first-order coeffs are 0, even if inputs are perfectly correlated).
    # Structural indices assume an INDEPENDENT product measure. This is recorded
    # as an assumption (verified only if the caller asserts it) — NOT auto-tested;
    # the nonlinear independence test is experimental and off the core path. If
    # x_train is given, report the descriptive max ordinary |Pearson| only.
    detail = ("structural indices assume an independent product measure; "
              "independence is " + ("caller-asserted (by design)"
                                    if inputs_independent_by_design
                                    else "ASSUMED, not verified"))
    if x_train is not None:
        try:
            xt = np.asarray(x_train, dtype=np.float64)
            if xt.ndim > 1 and xt.shape[1] > 1:
                Cc = np.corrcoef(xt.T)
                np.fill_diagonal(Cc, 0.0)
                detail += (f"; max ordinary |Pearson| {np.max(np.abs(Cc)):.2f} "
                           "(descriptive only). For an experimental nonlinear "
                           "check: correlation_diagnostic(..., "
                           "run_independence_test=True)")
        except Exception:
            pass
    add('Input independence', 'info', inputs_independent_by_design, detail)

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

    return {'checks': checks, 'all_pass': all_pass, 'sobol': sobol,
            'input_assumption': 'independent_product_measure',
            'input_assumption_verified': bool(inputs_independent_by_design)}
