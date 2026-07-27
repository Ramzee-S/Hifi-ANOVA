# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **`docs/USER_GUIDE.md`** — a comprehensive user manual: the two APIs, a full
  configuration reference (every trainer config key with type/default/effect),
  complexity modes and stages, all six regularization strategies (including the
  parameterized `sobolev[_s]` and `spectral[_a]`), the three bases and effect
  signatures, residual models, sensitivity/uncertainty, the diagnostic suite,
  and working with results.

### Fixed
- `api.py` module docstring advertised `hifi_anova.load(...)`, which does not
  exist; corrected to `hifi_anova.model.io.load_model`.

### Changed
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
