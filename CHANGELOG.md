# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **`verify_model` health check** — `analysis.diagnostics.verify_model` runs the
  diagnostic workflow end-to-end and returns a pass/warn/fail report (Sobol
  additivity, index bounds, test R², calibration coverage for heteroscedastic
  models, input-correlation level; flags pure-interaction variables). Confirms a
  fit is internally consistent before its Sobol indices are trusted. Documented
  in USER_GUIDE §10.6 with the recommended path→verify workflow, demonstrated in
  the Ishigami example, and covered by `tests/test_ishigami.py`.
- **Variance-model regularization path** — `analysis.reg_path.compute_variance_reg_path`
  / `plot_variance_reg_path`, the variance analogue of `compute_reg_path`: holds
  the fitted mean fixed and sweeps the variance penalty `lambda_h` (refitting the
  log-variance model by Newton at each point), recording the variance-Sobol
  spectrum and total explained log-variance. The Ishigami example uses it (x3's
  variance Sobol stays ~1 across the whole `lambda_h` range) and also renders the
  Pareto frontier (`plot_pareto_frontier`: complexity vs unexplained variance).
- **First-order group pruning** (`first_order_pruning` config: `'bic'` /
  `'group_lasso'` / `'1se'` / `'none'`, default `'none'`). Post-fit, it zeroes
  the entire first-order block of any variable whose marginal effect the
  criterion rejects (a leave-one-group-out test on the full design; first-order
  blocks are Hoeffding-orthogonal to the pair/triple blocks, so this does not
  disturb the interactions). This fixes the case where a variable is *pure
  interaction* — e.g. Ishigami x3 — for which plain ridge can only shrink the
  spurious main-effect block, never set it to zero. `'bic'` robustly zeros x3's
  first-order component down to N≈100. Wired into both the first-order-only and
  Stage-B paths; reuses `training.sparse` / `training.selection`. Covered by
  `tests/test_ishigami.py::test_first_order_pruning_zeros_x3`, and the Ishigami
  example now enables it.
- **Ishigami benchmark** — `hifi_anova.data.generate_ishigami` (with an optional
  heteroscedastic variant that drives the noise variance with x3) and
  `ishigami_sobol_indices`, the closed-form ground-truth Sobol indices. Ishigami
  is the canonical SA test case where x3 has a *zero first-order* but *non-zero
  total-order* index (it acts only through the x1–x3 interaction). Covered by
  `tests/test_ishigami.py` (analytic values, generator, first-order recovery,
  and the heteroscedastic variance-driver recovery).
- **Dual-sensitivity ellipse plot** — `plot_sensitivity_ellipses`, a new
  visualization of the mean+variance Sobol spectrum, in both `analysis.visualization`
  (lightweight, `save_path=` API) and `analysis.plots` (publication style,
  returns `(fig, ax)`). `mode='glyph'` draws one ellipse per variable whose
  width ∝ mean sensitivity and height ∝ variance sensitivity (shape tells the
  story: wide = mean driver, tall = variance driver); `mode='plane'` places each
  variable at (S^f, S^h) with CI ellipses.
- **`examples/run_ishigami_heteroscedastic.py`** — an end-to-end capabilities
  showcase on a heteroscedastic Ishigami: recovers the analytic mean Sobol
  indices, fits the variance spectrum (x3 emerges as a hidden driver),
  rediscovers the x1–x3 interaction with the residual sieve, checks calibration,
  bootstraps CIs for the dual spectrum, and renders the ellipse plots. It also
  prints the explained-variance split of the mean vs the variance model and
  renders the regularization path (L-curve, GCV/evidence, Sobol-vs-λ, and the
  variance-decomposition trade-off) via `compute_reg_path` / `plot_reg_path`.
- **`docs/CI_theory.md`** — theory note for the Sobol confidence intervals: the
  HC0 sandwich estimator, the delta method, the full-gradient derivation
  (including the denominator-coupling terms), why the own-block-only shortcut
  undercovers, the Monte-Carlo coverage validation across all three bases, and a
  symbol↔code map. Linked from README and USER_GUIDE §9.4.
- **`docs/USER_GUIDE.md`** — a comprehensive user manual: the two APIs, a full
  configuration reference (every trainer config key with type/default/effect),
  complexity modes and stages, all six regularization strategies (including the
  parameterized `sobolev[_s]` and `spectral[_a]`), the three bases and effect
  signatures, residual models, sensitivity/uncertainty, the diagnostic suite,
  and working with results.

### Fixed
- **Sobol confidence intervals undercovered (~90% actual at 95% nominal) — for
  every basis.** The delta-method gradient in
  `analysis/automl.py::sobol_confidence_intervals` only used the component's own
  coefficient block, dropping the denominator-coupling terms
  `∂S_i/∂w_j = −S_i·2G_j w_j / V_tot` (j ≠ i) and the corresponding covariance
  contributions. The resulting SE deficit (~13–15%) is proportional to the
  leading-order term, so it did **not** vanish with N, and grew with `S_i` (worst
  for the largest indices). Now the full gradient over all component blocks with
  the full sandwich covariance is used (cheap: one precomputed `Cov·U` vector).
  Monte-Carlo validation (in-basis ground truth, 400 noise draws): coverage
  0.94–0.96 at 95% nominal and SE/SD ≈ 1.00 for Fourier, Legendre, and Haar
  alike (previously ~0.90 / ~0.86 for all three). A deterministic coverage
  regression test (`tests/test_automl.py::TestSobolCICoverageAcrossBases`)
  guards all three bases. This also resolves the roadmap item "validate CIs for
  Legendre/Haar": the CI machinery is basis-agnostic, and Legendre/Haar were
  never worse than Fourier — all three shared the same gradient bug.
- `api.py` module docstring advertised `hifi_anova.load(...)`, which does not
  exist; corrected to `hifi_anova.model.io.load_model`.
- **`mode='heteroscedastic'` no longer silently trains a black-box NN residual.**
  It now maps to stages `['A', 'B', 'D']` (was `['A', 'B', 'C', 'D']`), so
  selecting the heteroscedastic mode no longer auto-enables the SGD-trained MLP
  residual — consistent with the one-call `hifi_anova(..., heteroscedastic=True)`
  path, the worked examples, and the "NN residual is an opt-in last resort"
  positioning in the docs. To combine a residual with the variance model, request
  it explicitly (`hifi_anova(..., heteroscedastic=True, residual='rbf')` or
  `stages=['A', 'B', 'C', 'D']`). Also fixes the README mode table, which already
  listed `A, B, D`.
- `optimize_multi_lambda` now honors `method='aic'` and `method='bic'` instead of
  silently falling back to the evidence criterion, and raises `ValueError` on an
  unknown `method`. This matches `optimize_multi_lambda_extended` (which delegates
  to it for the two-lambda case) and the USER_GUIDE §10.2 claim that the four
  methods are supported.

### Changed
- Documentation wording fixes: JAX x64 is enabled by the one-call `hifi_anova(...)`
  on first call (not merely by importing `hifi_anova.api`); the `spectral` penalty
  holds the linear term at the base penalty `λ` (not "a small stability ridge").
- Reframed the README and package descriptions to match the manuscript:
  **HiFi-ANOVA = Hoeffding Interaction–Fidelity ANOVA**. The intro now leads with
  the three contributions — the dual mean+variance Sobol spectrum, the three
  basis families (Fourier / Legendre / Haar) with per-variable effect signatures,
  and the one-solve diagnostic suite — instead of describing the method as a
  "Hoeffding-Fourier decomposition" (Fourier is only one of three bases). Also
  replaced the stale "CIs assume a Fourier basis" caveat (now fixed) with the
  penalty-strategy attribution caveat.
- Added **`LICENSING.md`** — a single overview of licensing and copyright
  covering source code (PolyForm Internal Use 1.0.0), documentation
  (all rights reserved), and third-party material — plus a REUSE-style
  `LICENSES/PolyForm-Internal-Use-1.0.0.txt`. Documentation notices now read
  "Draft, work in progress" instead of "unpublished working document" (which
  would be self-contradictory once shared) and include a GitHub-ToS carve-out.
- Changed the license from PolyForm Strict 1.0.0 to **PolyForm Internal Use
  1.0.0** for this early release: use (including internal modification) is
  permitted for the internal business operations of you and your company;
  distribution to third parties is not permitted. Still source-available, not
  open-source.
- Lowered the minimum Python from 3.11 to **3.10** (`requires-python = ">=3.10"`).
  The full 394-test suite passes on Python 3.10; this lets `pip install -e .`
  work on 3.10 machines.
- Renamed the project to **HiFi-ANOVA** (*Interpretable Regression with Analytic
  Sobol Diagnostics for Mean and Variance*). This is a breaking rename of the
  public API from the earlier `HFNet` / HiFi-Sieve names:
  - import package `hfnet` → **`hifi_anova`**
  - model class `HFNet` → **`HiFiANOVA`**
  - trainer `HFNetTrainer` → **`HiFiANOVATrainer`**
  - one-call function `hifi_sieve()` → **`hifi_anova()`**
  - Main entry points are now exported at the top level:
    `from hifi_anova import hifi_anova, HiFiANOVA, HiFiANOVATrainer, compute_sobol_indices`.

### Fixed
- **Non-Fourier confidence intervals in `hifi_anova.api`.** The one-call API
  built the Gram matrices, regularization vector, and Sobol CIs assuming a
  Fourier-with-linear basis, so for `basis_name='legendre'`/`'haar'` or
  `include_linear_1=False` the analytics either crashed or returned
  silently-wrong CIs. The basis config is now read from the fitted model and
  threaded through `build_regularization_vector`, both Gram builds, and
  `sobol_confidence_intervals`. Added a loud layout-mismatch guard in the CI
  helper. Verified end-to-end for Fourier, Legendre, and Haar bases: CI-path
  indices now agree with the authoritative `compute_sobol_indices` (max diff
  ~1e-8); previously Legendre/Haar crashed or were silently wrong.
- **Train/predict log-variance clip mismatch.** The Newton variance solver
  clamped log-variance to [-30, 30] during fitting while `predict_variance`
  clamped to [-100, 100], so the fitted objective and the prediction model could
  diverge at extreme heteroscedasticity. Both now use a single shared
  `LOG_VAR_CLIP = 30.0` constant.

### Notes
- This is a **preliminary, work-in-progress release** extracted and sanitized
  from a research workspace. The public API (`hifi_anova.api.hifi_anova` and the
  staged `HiFiANOVATrainer`) is usable but may change before 1.0.

## [0.1.0] — 2026-07-27

### Added
- Initial public packaging of the `hifi_anova` library:
  - `hifi_anova.core` — Fourier / Legendre / Haar feature construction, Gram matrices,
    interaction pairs, orthogonal projection.
  - `hifi_anova.model` — mean model, variance (heteroscedastic) model, residual
    networks, linear residuals (RBF / RFF / Nyström), Bayesian last-layer.
  - `hifi_anova.training` — staged trainer, ridge / Newton solvers, GCV hyperparameter
    optimization, group-lasso / BIC variable selection, re-decomposition.
  - `hifi_anova.analysis` — analytic Sobol indices (mean + variance spectra),
    AutoML analytics (LOO-CV, GCV, AIC/BIC, sandwich covariance), regularization
    path, interaction discovery, basis characterization, plotting.
  - `hifi_anova.data` — synthetic generators, analytic-ground-truth test functions,
    preprocessing.
  - `hifi_anova.api.hifi_anova` — one-call fit → Sobol → intervals → diagnostics.
- Test suite (394 tests) with pytest markers: `smoke`, `integration`, `slow`.
- Packaging (`pyproject.toml`), `requirements.txt`, `environment.yml`.
- Source-available license: PolyForm Internal Use License 1.0.0 (use limited to
  internal business operations; no distribution to third parties). Not open-source.

### Known limitations
- Analytic Sobol indices assume independent inputs (product measure after the
  quantile transform); correlated inputs are handled via separate "correlative"
  indices — see the README caveats.
- The interactive GUI and the manuscript / theory write-up are not part of this
  release and may be added later.

---

*Documentation notice: Copyright (c) 2026 R. Sala. All rights reserved.
Draft, work in progress — not covered by the source-code license (PolyForm
Internal Use 1.0.0). Except for permissions arising under GitHub's Terms of
Service, applicable law, or separate written permission from the copyright
holder, no permission is granted to reproduce, distribute, modify, publish, or
create derivative works from this document. See LICENSING.md.*
