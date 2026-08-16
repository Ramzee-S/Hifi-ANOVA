# Confidence intervals for Sobol indices

This note documents the intervals returned by
`hifi_anova.analysis.automl.sobol_confidence_intervals`. They are
**fixed-configuration HC3/delta intervals with a Student-t reference**. They
condition on the transform, basis, admitted component structure, penalties, and
weights used by the finalized fit. They are not post-selection guarantees.

## 1. Fitted problem and residual degrees of freedom

For a fixed feature design `Phi`, the feature-only ridge problem is

```text
w_hat = A^-1 Phi^T y,       A = Phi^T Phi + R.
```

On the profiled Stage-D mean path the same algebra is applied to the augmented
design `Z = [1, Phi]`, coefficient `(f0, w)`, and penalty `diag(0, R)`. With
precision weights `W`, replace every cross-product by its weighted form. The
intercept is included in the fitted smoother but excluded from every Sobol
component energy.

Let `H` denote the applicable smoother and `M` its feature-space equivalent. The
residual effective degrees of freedom used for scale estimation and the
Student-t reference are

```text
df_res = N - 2 tr(H) + tr(H^2),
tr(H^2) = tr(M^2) = tr(S^T S),
```

where `S` is the corresponding whitened smoother under weights. The public key
is `tr_H2`; `tr_HHt` is a deprecated one-release read alias for the exact same
number. The computation has not changed.

## 2. HC3 sandwich covariance

For row `phi_n`, residual `e_n`, and leverage `h_n`, the implemented covariance
is the leverage-adjusted HC3 sandwich

```text
Cov(w_hat) = A^-1 [sum_n phi_n phi_n^T e_n^2 / (1 - h_n)^2] A^-1.
```

The weighted path uses the coherent weighted score rows. The profiled-intercept
path forms the full augmented bread and meat first, then extracts the
slope-slope covariance block. This is not generally equivalent to combining a
slope-only meat with a block of the augmented inverse.

HC3 is robust to heteroscedasticity at the fixed fitted configuration. It does
not remove ridge shrinkage bias or account for choosing a transform, basis,
structure, penalty, or weight model from the same response.

## 3. Full-gradient delta propagation

For retained component `c`, with coefficient block `w_c` and Gram matrix `G_c`,

```text
V_c = w_c^T G_c w_c,
V_total = sum_j V_j,
S_c = V_c / V_total.
```

Define the full-length component score `U_c`, equal to `2 G_c w_c` on block `c`
and zero elsewhere, and let `U = sum_j U_j`. The exact first derivative is

```text
gradient(S_c) = (U_c - S_c U) / V_total.
```

The `-S_c U` term is essential: uncertainty in every other retained component
changes the shared denominator. The implemented delta variance is

```text
Var(S_c) = gradient(S_c)^T Cov(w_hat) gradient(S_c).
```

The code evaluates this efficiently by precomputing `Cov(w_hat) U` and
`U^T Cov(w_hat) U`; it does not use an own-block-only shortcut. First-, second-,
and third-order components follow the same formula. Residual-feature columns do
not enter the structural Sobol denominator.

## 4. Student-t intervals

At a regular interior component, the reported interval uses

```text
S_c +/- t_(1-alpha/2, df_res) sqrt(Var(S_c)),
```

clipped to `[0, 1]`. The low-level result records
`interval_method = 'HC3_delta_t'`; the one-call result records the same method
inside `result.inference_metadata`.

Monte-Carlo characterization on correctly specified fixed designs found nominal
coverage across Fourier, Legendre, and Haar bases after the full denominator
gradient was restored. Those experiments validate the regular,
fixed-configuration regime; they do not establish unconditional coverage after
adaptive model selection.

## 5. Conditioning and selection scope

Every one-call result carries:

```text
result.inference_metadata = {
    'inference_scope': 'fixed_configuration',
    'structure_selected_on_same_data': True or False,
    'post_selection_coverage': 'not_claimed',
    'conditioned_on': [
        'transform', 'basis', 'admitted_structure', 'penalties', 'weights'],
    'interval_method': 'HC3_delta_t',
}
```

BIC, group lasso, the 1-SE rule, pruning, heuristic thresholds, automatic basis
recommendations, adaptive triple admission, and the Stage-D keep/revert guard
are model-selection mechanisms. They are not the manuscript's efficient-score
tests and do not inherit its FDR or post-selection guarantees. When one of these
paths selected structure on the same data, `summary()` labels the intervals as
conditional and explicitly says post-selection coverage is not claimed.

## 6. Nonregular null and complete-share boundaries

The ordinary delta approximation is nonregular when the complete gradient is
degenerate. The implementation distinguishes:

- `nonregular_null`: an exact or numerical `S_c = 0` boundary;
- `nonregular_boundary`: a degenerate complete-share boundary such as
  `S_c = 1` with every other component zero;
- `regular`: a nondegenerate delta path.

These statuses are available in the low-level `component_status` mapping and in
`result.sobol_ci_status`. Legacy CI tuples remain present for compatibility, so
an old caller may still observe `(0, 0, 0)` or `(1, 1, 1)`. `summary()` does not
present either nonregular tuple as an ordinary confidence interval.

Quadratic-form null inference, bootstrap boundary inference, and selective
inference are deferred. Do not interpret a compatibility tuple at a nonregular
boundary as a 95% coverage statement.

## 7. Stage-D two-fit convention

Under Stage D, `result.sobol` is the structural spectrum of the shipped
predictive mean, which is precision-weighted when the heteroscedastic fit is
kept. `result.sobol_ci` is the unit-weight interpretable headline attribution
and its HC3 intervals. `result.sobol_ci_efficient` provides the weighted
first-order counterpart, while `result.sobol_gap` reports efficient minus
interpretable. The fixed-configuration limitation applies to both interval
surfaces.

## 8. What is and is not covered

The intervals quantify first-order sampling variation of regular fitted Sobol
shares, conditional on the finalized configuration. They do not cover:

- bias from ridge shrinkage or additivity calibration;
- uncertainty from data-driven selection or penalty optimization;
- null or complete-share boundary inference;
- dependent-input attribution (the structural spectrum assumes a product input
  measure);
- natural-scale variance attribution;
- inferential noise attribution from cross-fitted residuals.

Current variance-side diagnostics describe the fitted log-residual scale. A
future cross-fitted path is required before making inferential noise-attribution
claims.

## 9. Symbol-to-code map

| Quantity | Public/code location |
|---|---|
| `df_res`, `tr(H^2)` | `ridge_analytics`: `df_residual`, `tr_H2` |
| HC3 coefficient covariance | `sandwich_covariance(..., hc='HC3')` |
| Full delta gradient | `sobol_confidence_intervals` internal component assembly |
| CI tuples | `first_order`, `second_order`, `third_order` |
| Boundary guard | `component_status`; `result.sobol_ci_status` |
| Scope metadata | `result.inference_metadata` |

---

*Documentation notice: Copyright (c) 2026 R. Sala. All rights reserved.
Draft, work in progress — not covered by the source-code license (PolyForm
Internal Use 1.0.0). Except for permissions arising under GitHub's Terms of
Service, applicable law, or separate written permission from the copyright
holder, no permission is granted to reproduce, distribute, modify, publish, or
create derivative works from this document. See LICENSING.md.*
