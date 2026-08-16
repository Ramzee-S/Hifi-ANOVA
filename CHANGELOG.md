# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.0a1] — 2026-08-16

First **alpha** pre-release. Bundles the backend term-structure work and the
NumPy exact-core default below, and adds the new **HiFi Console (gui3,
experimental)** — a local browser desk shipped as a source-run tool
(`pip install -e ".[gui3]"`, then `python -m gui3.server`). Pre-release: APIs
and the console UI may still move.

### Changed
- **Linear residual projection targets the SOLVED design layout (GUI3
  register BR-11).** The fitted RBF/RFF/Nyström residual now stores the mean
  model's layout fields (`var_specs`, `pair_k2`, `fo_included`) and rebuilds
  its prediction-time projection design through the same shared builder the
  model uses (`core.features.build_mean_design`, also backing
  `HiFiANOVA.build_phi1/build_phi2/build_phi_all_fit`). The fit-time
  projection targets `build_phi_all_fit` (the solved columns, `record.Phi`
  layout) instead of the full uniform `build_phi_all`. Consequences:
  - residuals now work on mixed per-variable-K, per-pair-K2, and
    order-selective (`variable_orders`) fits — layouts that previously had
    to be refused to protect the orthogonality guarantee;
  - on an order-selective fit, an excluded variable's first-order structure
    is no longer projected out of the residual (the mean never fitted it);
    uniform fits are byte-identical to before;
  - previously fitted/pickled residuals load unchanged (missing layout
    fields ⇒ the historical uniform rebuild).
- **`variable_orders={j: []}` (mean-excluded membership) is accepted on any
  fit, and excluding every variable yields an intercept-only mean (GUI3
  register BR-12).** Both former `ValueError`s are now `UserWarning`s: on a
  constant-noise fit a mean-excluded variable is in neither model but its
  column stays in X (available to a post-hoc `fit_residual` complement); the
  all-excluded limit fits `f0` only (`fo_included=()`, an empty solved
  design whose residual projection is a no-op) — an exploratory
  complement-only base, disclosed by warning. Fixes riding along: the
  Stage-A-only path now stamps `fo_included` on the model (previously the
  record held the subset while the model claimed the full layout), and
  Stage D handles the empty first-order subset.
- **Linear residual families run on the NumPy exact core (GUI3 register
  BR-10; DEC-057).** Stage C's `residual='rbf'|'rff'|'nystrom'` and Stage D's
  `variance_residual` are no longer JAX-native: they run on the switchable
  backend, and `backend='auto'` now resolves them to the float64 NumPy core
  (previously a silent JAX/float32 fallback). Details:
  - the fit/analysis paths evaluate residuals through a batched,
    backend-neutral forward (`predict_batch`) instead of `jax.vmap`;
  - residual weights are stored at the fit's precision (float64 on the core)
    — the historical float32 cast is gone;
  - RFF frequencies/phases are drawn backend-natively (seeded
    `np.random.default_rng` on numpy vs `jax.random` on jax): deterministic
    per backend, but an RFF fit is never numerically comparable across
    backends. RBF/Nyström (seeded k-means) are cross-backend reproducible;
  - `backend='numpy'` still refuses the genuinely JAX-native surface: the NN
    residual (`residual_nn` / `residual` type `'nn'`), `mode='auto'/'full'`,
    and `precision='float32'`. Pass `backend='jax'` to reproduce the old
    routing (residual weights are stored float64 there too now, so residual
    predictions can move at float32 round-off level vs pre-DEC-057).
- **NumPy exact core is the default backend; float64 is the default fit
  precision (GUI3 register BR-09; DEC-056).** `hifi_anova(..., backend=)` now
  defaults to `'auto'`, which runs the float64 NumPy exact core for the common
  (non-residual) config — no per-shape XLA compilation, statistically identical
  to JAX by construction (shared code path). Consequences:
  - a plain `hifi_anova(X, y)` call now fits in **float64** (was float32);
  - `precision='float32'` remains available on JAX and is now an explicit
    opt-in — it routes `backend='auto'` to JAX and is rejected by an explicit
    `backend='numpy'` (the core is float64-only). `backend='jax'` keeps its
    float32 default unchanged (DEC-035);
  - to reproduce pre-DEC-056 behavior exactly, pass `backend='jax'`.

### Added
- **HiFi Console (gui3) — EXPERIMENTAL browser desk.** A local FastAPI +
  WebSocket application (`python -m gui3.server`, extra `.[gui3]`) for
  interactive fitting and diagnostics: live effect faders and per-effect mutes,
  the SCAN/ROUTE second-order workflow, a layered parity view (peel the fit by
  order), the COMPLEMENT bus (post-hoc orthogonal residual, EXPLORATORY), and
  calibration / LOO / leverage scopes. New this cut: a parity
  **decomposition-ladder overlay** (per point, the vertical climb from the
  no-fit baseline f₀ through +1st/+2nd/+complement to the prediction, colored by
  contribution) with a central, retunable category palette. Ships as source in
  the alpha; not part of the installed wheel. The honesty framing of the library
  is preserved end to end (non-hierarchical / exploratory readouts are labeled,
  never presented as blessed selections).
- **User-defined equation systems (GUI3 register BR-04/BR-05/BR-06/BR-01;
  DEC-053/DEC-054).** The public API now accepts an explicit term structure,
  all additive and inert by default (every path keys on a `None` default, so
  the golden characterization is byte-identical and needed no rebaseline):
  - `K2={(i, j): K2_ij}` — a mapping pins the exact retained pairs AND gives
    each its own second-order harmonic order (per-pair Grams, ragged blocks
    end-to-end through features → penalties → model → Sobol/CI → Stage D →
    persistence). Data-driven pair selection/pruning, `K3>0`, mixed bases and
    `mode='auto'` are rejected with a mapping.
  - `variable_orders={j: [orders]}` (orders ⊆ {1, 2}) — order-selective
    membership: `[2]` admits `j` to pair terms while EXCLUDING its first-order
    block from the design (no df spent; a NON-HIERARCHICAL model, flagged in
    `results['term_structure']` and `summary()`); `[1]` drops every pair
    touching `j`.
  - `variance_variables=[...]` — restricts the first-order variance (Stage-D)
    model to a subset; excluded variables are variance-flat with `Sʰ ≡ 0` (by
    modeling assertion). Composes with `K2h`, `var_pair_selection='auto'`, and
    the mean-side term structure.
  - `K2h>0` is now documented public API (populates
    `log_variance_sobol['second_order']`), and `var_pair_selection` accepts an
    explicit `[(i, j), …]` list (previously a list silently behaved as `'all'`).
- **`summary()`** surfaces the user-defined equation system and its
  non-hierarchical honesty caveat; `meta.json` mirrors `fo_included` /
  `variance_variables` provenance.
- **Split provenance on the result (`result.split_indices`).** The fitted result
  now exposes `{'train','val','test'}` arrays of original-dataset row indices, in
  the exact row order of the fitted design; `preprocess_data` returns the same as
  `train_indices` / `val_indices` / `test_indices`. `X[result.split_indices
  ['train']]` reproduces the rows the model was fit on — the honest way to map a
  per-point diagnostic back to a dataset row id without re-deriving the seeded
  permutation. Read-only; no fitting behavior changes.

### Fixed
- **Order-selective prediction intervals (DEC-054).** `predict_intervals` on a
  `variable_orders` fit raised a matmul dimension error — the model keeps the
  full uniform first-order layout (zeros in the excluded block) while the
  fitted-design record uses the subset layout, so the epistemic posterior
  compared mismatched designs. The epistemic `Phi_new` is now built in the
  record layout (`build_phi_all_fit`); ordinary models and older pickles are
  unaffected.
- **Re-decomposition guards.** `redecompose` / `alternating_ridge_nn` now raise
  `NotImplementedError` on a term-structure model rather than silently
  mis-slicing its ragged coefficient vector with uniform block widths.
- **Interaction plots on a per-pair `K2` model (DEC-055).**
  `plot_interaction_heatmap` and `plot_interaction_grid` sized each pair's
  coefficient block from `model.K2` — which holds `max(pair_k2)` for a
  term-structure fit — so a pair with a smaller order raised a reshape error.
  The block and evaluation basis are now taken from the per-pair order.
- **Faithful `K2`-mapping config round-trip on `save_model` (DEC-055).** A
  per-pair `K2={(i, j): K2_ij}` mapping was stored in `meta.json` as an opaque
  `str()` (tuple keys aren't JSON-representable); it is now normalized to the
  `"i,j"` string-key form used elsewhere, so the retained-pair config round-trips
  exactly. The model itself already round-tripped via its pickle companion.

## [0.3.0] — 2026-08-11

Second development milestone. Consolidates the DEC-029–052 correctness/honesty
work: R² reconciliation, the Stage-D joint-GLS estimator fix and its provenance
metadata, correlative-attribution scoping to independent inputs, one-call
precision precedence, mixed-basis capability fencing, public/nested input
validation, the augmented profiled-intercept Stage-D diagnostics, estimator-identity
and honest-objective metadata, the public attribution vocabulary, and self-contained
release engineering. Behavior changes are detailed below; see `Session/DECISIONS.md`
(private) for the full rationale.

### Changed
- **Stage-D diagnostics now re-profile the intercept (DEC-048).** For profiled
  joint-GLS variance fits the downstream analytics (df, leverage, residual df,
  `sigma_hat`, PRESS/LOO, GCV/AIC/BIC) and the Sobol confidence-interval and
  epistemic prediction-interval sandwiches are computed on the augmented design
  `Z = [1, Φ]` with penalty `diag(0, R)`, so they re-profile the unpenalized
  intercept the mean fit actually uses. Fitted coefficients, residuals and
  predictions are **unchanged**; reported residual df gains ~+1 and `sigma_hat`/LOO
  shift slightly. Goldens were re-baselined on both hosts. Legacy fixed-intercept
  and homoscedastic paths are byte-identical.
- **Public log-scale attribution vocabulary canonicalized (DEC-051).** The
  log-scale variance attribution is now `log_variance_sobol` and the residual-df
  trace is `tr_H2`. One-release `DeprecationWarning` aliases keep `variance_sobol`
  and `tr_HHt` reads working with identical values and calculations; plotting
  retains legacy-key compatibility.

### Added
- **Stage-D estimator identity + honest objective metadata (DEC-049).** A
  `stage_d_estimator` selector (`adjusted_quasi_likelihood` default,
  `raw_likelihood` alternative) and `results['stage_D']` estimator metadata
  (`estimator`, `objective_family`, `residual_update`, `iterate_selection`,
  `convergence_reason`, `bound_active`), mirrored into `meta.json`. The joint-LAML
  objective reports `objective_mode`/`evidence_status`/`converged`/`bound_active`
  and only claims a Laplace evidence at a verified interior mode (otherwise
  `laplace_evidence_unverified`). No default numeric movement.

### Documentation
- Labeled fixed-configuration inference honestly (DEC-050): clarified that
  “one factorization, many diagnostics” is a fixed-configuration property
  (selection, lambda optimization and joint mean--variance training may require
  earlier refactorizations), documented mixed `'auto'` as heuristic, retained
  DEC-045's fixed mixed-basis capability boundary, and corrected the Sobol-CI
  description from HC0 to the implemented HC3 sandwich convention.
- Clarified `summary()` Sobol conventions (DEC-047): the interaction section is
  labeled as the structural/predictive-fit spectrum, distinct from the
  interpretable unit-weight `sobol_ci` headline; a bare `save_model(model, path)`
  now retains Stage-D mean-convention provenance carried on the model.

### Internal
- Self-contained release engineering (DEC-052): study-port tests dropped from both
  the git-archive release (`.gitattributes`) and the sdist (`MANIFEST.in`); a
  self-contained `public-ci.yml` (ruff, build+twine, pip smoke on 3.10/3.11/3.12,
  quick tier) ships while the private nightly-studies/golden CI stays private;
  `.pre-commit-config.yaml` ships; `LICENSING.md` completed and `SECURITY.md`
  added; Documentation and Changelog project URLs added. No numerical change.

### Fixed
- **Stage-D guard no longer false-reverts correct variance models (DEC-039).** On genuinely
  heteroscedastic data the guard discarded a correct input-dependent variance model with a
  "noise looks homogeneous" warning. Root cause: the alternating mean update fixed
  `f0 = Σwₙyₙ/Σwₙ` and solved a weighted ridge on **uncentered** Φ — not the penalized-GLS
  optimum — so the weighted mean lost to the unit-weight mean on its own weighted objective and
  dragged the variance model into a revert. Three Stage-D defaults are flipped:
  `stage_d_joint_gls_mean` (now `True` — weighted-center both `y` and Φ, profiling the intercept
  jointly: the penalized-GLS solution, restoring monotone alternating descent);
  `variance_selection_mean_consistent` (now `True` — the keep/revert comparison uses the same
  weighted mean on both sides, isolating the variance); and `min_noise_ratio` (`1e-6` → `1e-2` —
  a calibrated near-noiseless entry gate that refuses to model deterministic mean-approximation
  error as aleatoric noise). All three are per-fit overridable; the legacy estimator remains
  available behind the (transitional) flags. **Changes σ̂/df/LOO/CI/predictions of every
  heteroscedastic fit → golden re-baselined on both hosts** (byte diffs from centering FP order,
  not regressions); rollback tag `stage-d-pre-flip`. `variance_selection_mean_fallback` stays
  opt-in (default `False`) as an invariant monitor (`mean_fallback_anomaly`).
- **One-call precision precedence now obeys the documented order (DEC-044).**
  `hifi_anova()` hard-coded `precision='float32'` and passed it explicitly, so a normal
  one-call fit *ignored* `HIFI_ANOVA_X64=1` and `set_fit_precision('float64')` — both
  documented controls were dead at the public boundary (the resolver itself was correct).
  The argument now defaults to `None` (omission) and is resolved once by precedence:
  explicit argument > `set_fit_precision` override > `HIFI_ANOVA_X64` env > `float32`
  default. An explicit value always wins, so `precision='float32'` forces float32 despite a
  global/env opt-in. The *effective* precision is stored in `result.config['precision']` and
  `meta.json`. An unrecognized `HIFI_ANOVA_X64` value now warns and is ignored rather than
  being silently reinterpreted. The no-control default is unchanged (float32, golden-stable);
  new end-to-end tests cover env/override/explicit precedence, effective-value recording, and
  save/load.
- **Correlative first-order Sobol indices no longer overshoot 1 (ddof mismatch).**
  `compute_correlative_sobol` divided a `np.cov` numerator (default `ddof=1`, ÷(N−1)) by an
  `np.var` denominator (`ddof=0`, ÷N), inflating every correlative index by a factor
  `N/(N−1)` and making them sum to `N/(N−1)` instead of the exact 1 the identity
  `Σ_i Cov(f_i, f_tot) = Var(f_tot)` guarantees (Manuscript_Theoryv06 §correlative). The
  numerator now reuses the `ddof=0` covariance matrix already built in the function
  (`Σ_j cov_matrix[i,j]`), so numerator and denominator share an estimator and the indices
  sum to 1 to machine precision. The bias was ~N/(N−1) (≈2% at N=50, ≈0.1% at N=1000). Only
  the correlative indices change; the structural/analytic Sobol spectrum, all other analytics,
  and the golden master are unchanged (correlative indices are not part of the golden capture).
  A regression test pins the sum-to-1 identity across N and under correlated inputs.
- **Mixed per-variable basis silently dropped Stage C/D and third-order terms.** A
  `basis_per_variable` fit that also requested a nonlinear residual (Stage C), a
  heteroscedastic variance model (Stage D), or `K3>0` used to *silently* return a mean-only
  A/B model. `HiFiANOVATrainer._fit_mixed` now raises `NotImplementedError` naming the
  unsupported feature(s). Existing A/B mixed fits and the golden master are unchanged;
  regression tests cover the guard.
- **Mixed per-variable basis silently ignored selection/pruning/pair controls and `K2=0`
  (DEC-045).** On a `basis_per_variable` fit, `variable_selection`, `pair_candidates`,
  `pair_selection`, `max_pair_variables`, `pair_pruning`, and `first_order_pruning` were
  silent no-ops, and `K2=0` still fitted all pairs (metadata-only) instead of producing a
  first-order model. These can materially change model size and attribution. Now a
  non-neutral value of any of those controls raises `NotImplementedError` naming the option
  and its neutral value, and `K2=0` genuinely disables Stage B on the mixed path (no pair
  features, indices, result block, or CIs — matching the uniform path and the `P_1` estimand).
  The one-call API defaults `variable_selection` to `'bic'`; on mixed bases the *implicit*
  default is neutralized with a one-release migration warning (`selection_applied=False`),
  while an *explicit* value raises. New `result.train_results['mixed_capability']` records the
  stages run, pair behavior, and whether selection/pruning was applied. Supported A/A+B mixed
  numerics and the golden master are unchanged.

### Added
- **Stage-D mean-estimator convention is recorded and persisted (DEC-039 provenance).** A
  heteroscedastic fit records the *effective* mean-estimator vintage as
  `results['stage_D']['mean_intercept_mode']` and on the fitted-design record — one of
  `profiled_joint_gls` (the joint-GLS default), `legacy_fixed_intercept_uncentered_features`
  (the compatibility flag), or `unweighted_centered` (a reverted / mean-fallback / homoscedastic
  shipped mean). `save_model` persists it in `meta.json`; `load_model` gives artifacts predating
  the field a defined interpretation (`legacy_unknown` for an older heteroscedastic save whose
  vintage is unrecoverable, `unweighted_centered` for an older homoscedastic save). A
  deterministic positive control proves the reported structural Sobol spectrum is recomputed
  from the final joint-GLS mean, not a stale pre-Stage-D mean. Stage-D invariant tests that
  previously `pytest.skip`-ped when the guard reverted now assert selection (fail-fast) so a
  regression cannot masquerade as a green skip. No Stage-D numerical behavior changed.
- **Public and nested input validation fails early with actionable errors (DEC-046).**
  Invalid or contradictory public configuration is now caught at the earliest boundary with a
  specific `ValueError` — naming the option path, the received value/type, and the valid
  range/alternatives (with a typo suggestion where useful) — instead of being silently accepted
  or crashing deep in the solve. A new shared, testable layer (`hifi_anova/validation.py`)
  covers: the data boundary (`X` a 2-D numeric matrix checked *before* `X.shape`; `y` numeric
  1-D, an `(N, 1)` column accepted, `(N, k>1)` rejected as multi-output); `feature_names`
  (exactly `D` unique strings — **duplicates now raise**); nested `basis_per_variable` (family,
  index in `[0, D)`, positive integer `K`, unknown-key rejection); `residual`/`variance_residual`
  nested keys **and values** (counts `>=1`, scales/`lr` `>0`, epochs/widths, `center_method`/
  `kernel` enums — with valid constructor options like `center_method`/`signal_variance` preserved,
  a string shorthand normalized so it can't crash Stage D, and `enabled=False` now disabling an
  analytic residual too); explicit `stages` (valid letters, no duplicates, `A` present, canonical
  order); recognized enums that used to fall back silently (`basis_name`/`basis_type`/`precision`,
  pair/triple/variance selection); boolean switches (reject `0`/`1`); and numeric type/range for
  `K*`, penalties, tolerances, `auto_threshold`, iteration counts, and `seed` (a `bool` is never
  accepted as an integer — `K1=True` raises). `mode='auto'` with
  `heteroscedastic=True` is now rejected as ambiguous rather than silently letting `auto` win.
  Value validation runs even under `allow_unknown_keys=True` (that hatch is for experimental
  *keys*, not a bypass of type/shape/range safety). No supported-fit numerics, defaults, or the
  golden master change; every valid uniform/mixed/one-call/direct-trainer path is unchanged.
- **Both R² conventions are reported (explained-variance and classical SSE/TSS).** New
  `hifi_anova.analysis.metrics.r_squared(y, ŷ, definition=…)` / `r_squared_report(…)` provide
  the framework-native explained-variance score `1 − Var(y−ŷ)/Var(y)` (the manuscript
  convention, and the library default — unchanged) and the textbook coefficient of
  determination `1 − Σ(y−ŷ)²/Σ(y−ȳ)²` (sklearn `r2_score`; penalizes bias, can go negative).
  `HiFiResult` gains `r_squared_classical`; `summary()`, `variance_accounting_report`, and
  `verify_model` now surface both. Default numerics (`r_squared`) are byte-identical — the
  golden master is unchanged.

### Fixed
- **Variance-residual projection crashed with `K2h>0` (or `K3h>0`) + a variance residual (DEC-037).**
  A heteroscedastic model fit with a second/third-order variance model *and* a variance
  residual (`variance_residual={...}`) raised a `dot_general` shape mismatch the moment it
  was passed to `model.predict` or `compute_sobol_indices`. The Stage-D training projection
  orthogonalizes the variance-residual features against the full variance design
  `[psi1|psi2|psi3]`, but both consumers rebuilt only `psi1` for the new-data projection.
  Both now rebuild the same design (`K2h=0` paths are byte-identical). This combination had
  no prior test or golden coverage; a regression test now exercises it.

### Changed
- **Internal: `training/trainer.py` god-functions decomposed as pure moves (DEC-038).**
  `fit` is now a stage-orchestration + auto-mode-decision skeleton delegating to
  `_fit_stage_a` / `_fit_stage_b` (mirroring the existing `_fit_stage_c`), and
  `_fit_heteroscedastic`'s feature/penalty assembly moved into `_build_stage_d_designs`
  (a `_StageDDesigns` namedtuple), with `_split_mean_coeffs` as the single source for the
  mean-coefficient layout split. No behavior change — golden master byte-identical after
  every extraction; public `(model, results)` surface unchanged. The parent plan's item 4a
  (merge the Stage-D alternating loop with `joint_lambda._joint_fit`) was investigated and
  **rejected**: the two loops are legitimately different estimators (intercept/centering
  convention, Stage-C residual freezing, val-NLL best-iterate early stop), and their shared
  numerics (DEC-028 leverage debias, Newton/ridge solvers) already live in single shared
  functions.

- **Internal: `analysis/sobol.py` `compute_sobol_indices` decomposed into helpers (DEC-037).**
  The ~390-line function is split into focused, individually-testable helpers
  (`_mean_component_variances`, `_variance_sobol_block`, `_mean_residual_variance`,
  `_fidelity_and_core_total`, and the shared normalization routines
  `_normalized_sobol_block` / `_core_sobol_block` / `_total_order_variances`). No behavior
  change — the golden master is byte-identical across all 15 scenarios on every host; the
  public return-dict shape and keys are unchanged. Also adds golden scenarios and
  characterization tests pinning the previously-uncovered mixed-basis, Stage-C-residual,
  2nd-order-variance, `estimate_sobol`, `mode='auto'`, and degenerate paths.

- **Internal: single `ridge.kfold_indices` shared by the CV loops (DEC-037).** The k-fold split
  was hand-rolled in `selection._kfold_cv_ridge` (contiguous folds) and `joint_lambda._kfold_nll`
  (strided folds); both now delegate to one `kfold_indices(N, n_folds, seed, scheme, return_perm)`
  helper. Byte-identical (the contiguous caller reconstructs its train fold in permutation order to
  match its prior `Φᵀ Φ` summation exactly); golden master unchanged. `hyperopt` was found to have
  no k-fold (closed-form criteria), so no change there.

- **Unknown trainer config keys now raise instead of silently no-op'ing (DEC-036).**
  `HiFiANOVATrainer` validates its config against a `KNOWN_CONFIG_KEYS` allowlist and
  raises a `ValueError` (naming the offending key with a "did you mean" suggestion) on
  any unrecognized **top-level** key — previously a typo like `stategy=` or `hetero=True`,
  including via the one-call `hifi_anova(X, y, **kwargs)` path, did nothing with no error
  or warning. Set `allow_unknown_keys=True` to bypass. Nested `residual`/`basis_per_variable`
  specs are not validated in this pass. Fit numerics are unchanged (golden master unchanged).
- **`HiFiANOVATrainer` no longer mutates the caller's config dict (DEC-036).** `__init__`
  now deep-copies the config before mode resolution / stage logic, so a caller who reuses
  their config dict (for a second fit or for logging) no longer sees nested keys such as
  `residual_nn['enabled']` surprise-flipped.

### Fixed
- **Docs: `predict.py` uncertainty docstring corrected (DEC-036).** The module docstring
  claimed three prediction-uncertainty sources ("the sum of all three"); `predict_intervals`
  computes only two (aleatoric + Fourier epistemic). The docstring now matches the code and
  notes the third (residual-epistemic) source is a not-yet-wired placeholder living in
  `hifi_anova.model.bayesian_nn`. No code change.

### Added
- **Opt-in float64 fit: `hifi_anova(..., precision="float64")` (DEC-035).** The model
  fit stays **float32 by default** (unchanged); passing `precision="float64"` — or
  setting `HIFI_ANOVA_X64=1` — fits in float64 end-to-end, so the stored model weights
  and the predictions come back float64 (the post-fit analytics were already float64).
  Precision is resolved centrally in the new `hifi_anova/precision.py`
  (`resolve_precision` / `fit_dtype` / `set_fit_precision`; precedence: explicit arg >
  `set_fit_precision` > `HIFI_ANOVA_X64` env > default float32) and threaded through
  `preprocess_data(fit_dtype=…)` and the trainer's weight casts. Predictions follow the
  fitted model's dtype (`HiFiResult._pred_dtype`), and `save`/`load_model` roundtrip
  float64 (the deserialization template records and rebuilds with the saved fit dtype;
  older saves default to float32). Default path is **byte-identical** — golden master
  unchanged. Covered by `tests/test_precision.py`.

- **Sobol reporting: naming & normalization (DEC-034; presentation + one opt-in view).**
  Resolves the "total" collision on residual/NN fits. The `𝔉`-scaled share `V_u/V_f` is
  now labeled the **"share of fitted variance"** (it names its own denominator) — **not**
  the total-**effect** index `S_T`, which stays `mean_sobol['total_order']` (unchanged).
  `summary()` gains a 𝔉-vs-R² gloss, a residual row surfacing `1 − 𝔉`, and a
  `headline="core"|"fitted_variance"` display option (default `core`, the *invariant*
  attribution; presentation only). New opt-in `summary(observed=True)` /
  `HiFiResult.sobol_ci_observed` prints the *share of observed output variance*
  `V_u/Var(Y) = fitted-variance·(Var f̂/Var Y)` (scale fixed; residual+noise as one lump;
  never co-tabulated) — the library computes the scaling because `Var(f̂)/Var(Y) = R²`
  only under OLS-with-intercept orthogonality. Homoscedastic/no-residual summary
  **byte-identical** (single-column table). Field rename `sobol_ci_total →
  sobol_ci_fitted_variance` is **staged** (deprecation alias) for a later release; no
  golden re-bless (numeric analytics unchanged). Advisor-approved
  (`M3b_Sobol_naming_and_normalization_Final.md`).
- **Joint-LAML determinant-term audit + regression fence (no behavior change).**
  Audited the v06 App. D determinant-derivative ("second") term in the joint
  heteroscedastic `log λ` gradient. Finding, measured end-to-end
  (`dev/tests/m4_laml_determinant_probe.py`): the term is real and heteroscedastic-only
  (determinant-piece gap ≈ 0.72 off-optimum), but the joint λ_h selection is
  derivative-free, so there is **no defect** and **no production code change**; its
  impact on λ_h selection is measured-negligible in every regime (scalar ≤ 0.003 dec,
  multi-λ ≤ 0.054 dec to cond ≈ 1e5). Added an existence tripwire
  (`test_appD_determinant_term_detectable_and_fenced`, slow tier) and a documented
  **fence**: any future joint-LAML gradient must use JAX implicit differentiation
  through the leverage-corrected stationarity condition with the Hessian in-trace —
  AD over a frozen mode silently drops the term. Default/homoscedastic paths
  **byte-identical** (DEC-033, M4; leverage sub-finding in
  `M4b_leverage_nonstationarity_note.md`).
- **Core vs. total Sobol shares + structural fidelity 𝔉 (residual/NN fits).**
  When a residual stage runs, the two Sobol normalizations the theory keeps
  distinct (v06 §3.2/§8) are now reported *separately and always labeled*: the
  **core** share `Ŝ^core = V_u/V_core` (within the retained structured orders) and
  the **total** share `Ŝ^total = V_u/V_f` (of the whole function), bridged by the
  **structural fidelity** `𝔉 = V_core/(V_core+Var(ĝ))` with `Ŝ^total = 𝔉·Ŝ^core`.
  `result.fidelity` carries 𝔉 (`1−𝔉` is the honest interpretability gap) plus the
  **orthogonality defect** `2·Ĉov(f̂_core,ĝ)/Var(f̂)` — reported beside 𝔉, never
  folded in, since the 𝔉 identity is exact only under exact core/residual
  orthogonality. `result.sobol_ci_total` adds the total-variance CIs (the core CI
  scaled by 𝔉, conditional on the residual variance); `compute_sobol_indices`
  gains labeled `mean_sobol_core` / `mean_sobol_total` blocks and a `fidelity`
  entry. Both Sobol surfaces are reconciled onto one convention from a single
  set of variances, removing the previous unlabeled point-vs-CI scale mismatch.
  With no residual stage `𝔉≡1`, total≡core, and the homoscedastic/no-residual
  path is **byte-identical** (new fields only; `sobol_ci_total` is `None`)
  (DEC-032, M3).
- **Three-tier leave-one-out for heteroscedastic fits (`loo_nll`, `loo_tier`,
  `result.loo(tier=…)`).** A Stage-D `HiFiResult` now reports the manuscript's
  default **Tier II** one-step-jackknife LOO (v06 App. C): the variance model is
  corrected by one Newton step per deleted point, so the reported `loo_cv` is no
  longer the optimistically-biased Tier-I plug-in. A new **`result.loo_nll`** —
  the predictive leave-one-out negative log-likelihood — is populated on **both**
  paths (it lives on a common scale, so homo-vs-hetero model comparison is
  legitimate, unlike the model-metric weighted `loo_cv`). **`result.loo_tier`**
  (1 or 2) makes every number self-describing, and **`result.loo(tier=1|2|3)`**
  exposes each tier on demand — tier 3 being the exact nested refit (the oracle,
  and the authority whenever a variance floor binds). A **KKT variance-floor
  test** and an `H_h` conditioning check set `result.loo_tier2_guarantee_holds`;
  `summary()` prints a "Tier III is authoritative" warning when it is at risk. The
  homoscedastic path is byte-identical (tiers coincide; new fields only) (DEC-031).
- **Two-fit reporting surface (heteroscedastic fits).** Following the theory's
  two-fit convention (Theorem `projection` Part ii), a Stage-D `HiFiResult` now
  also exposes the **efficient** (precision-weighted) first-order Sobol CIs via
  `result.sobol_ci_efficient` — alongside the interpretable `result.sobol_ci` —
  and their observed **efficient − interpretable** gap in `result.sobol_gap`
  (`{'first_order': {...}, 'second_order': {...}}`), the "heteroscedasticity ×
  misspecification" diagnostic. `summary()` prints a gap row when it is
  non-negligible. On a homoscedastic fit both are `None` and the reported surface
  is byte-identical (DEC-030).
- **Mixed per-variable basis through the one-call API.** `hifi_anova(...,
  basis_per_variable=…)` no longer raises `NotImplementedError`: `_fit_mixed`
  surfaces a `build_mixed_record` with per-group column slices and Grams, and
  `sobol_confidence_intervals` gained a block-driven `groups=` path, so the
  post-fit Sobol CIs are block-correct for heterogeneous per-variable bases
  (previously crashed in the uniform rebuild) (DEC-030).

### Internal
- **Lint backlog cleared and the ruff gate promoted to the full set (DEC-035; no behavior
  change).** Paid down the ~116-item cosmetic backlog across `hifi_anova/` — unused imports,
  placeholder-less f-strings, dead local variables, ambiguous names (`l`), stray semicolons,
  and one intentional `__init__` import-order `# per-file-ignore`. All byte-identical
  (dead-code removal / local renames); golden master unchanged. The CI + pre-commit ruff gate
  now enforces the full `[tool.ruff.lint]` set (`E4,E7,E9,F`) instead of the `E9,F63,F7,F82`
  real-bug subset (`E501` line-length still excluded).
- **Woodbury k-fold downdate deduplicated + selectable SPD-inverse backend (DEC-035, Point 4c;
  no default behavior change).** The rank-`n_k` Woodbury leave-fold-out kernel was copy-pasted
  in `automl.kfold_cv_analytic` and `automl.stability_diagnostics`; extracted to one
  `automl._woodbury_downdate(...)`. The SPD ridge inversions (`analysis/automl.py`,
  `model/predict.py`) now route through `hifi_anova/linalg.py::spd_inverse`, whose backend is
  selectable — **default `'inv'` (`numpy.linalg.inv`, byte-identical, golden unchanged)** or
  opt-in `'cholesky'` (more stable / cheaper; `HIFI_LINALG=cholesky` or `set_linalg_method`).
  Cholesky is opt-in, not default, because on the tested well-conditioned paths it agrees to
  ~1e-13 everywhere **except** a ~1e-8 shift in one near-noiseless overfit scenario's tiny
  `sigma_hat` (an ill-conditioned near-zero quantity) — i.e. no real benefit there, and it
  would otherwise move the golden. Pinned by `tests/test_linalg_backend.py`.
- **Golden master added to CI (`.github/workflows/ci.yml`).** New `golden` job runs the
  characterization harness on push/PR. Baselines are per host; CI sets `HIFI_GOLDEN_HOST=ci`
  so one committed `dev/tests/golden/refactor_baseline.ci.json` is authoritative across
  runners. With that file committed it is a hard regression gate; without it the job
  self-bootstraps (capture→check, informational) and uploads the baseline artifact to commit.
- **Leverage-correction primitive deduplicated (DEC-035, Point 4a; no behavior change).**
  The DEC-028 in-sample de-biasing `r²/clip(1−lev, 1e-3, 1)` was copy-pasted across the
  trainer's Stage-D loop (constant-variance baseline + the alternating loop) and
  `joint_lambda._joint_fit`; the trainer comment noted they "must stay in sync." Extracted
  to `ridge.debias_squared_residuals(...)` with the `1e-3` clip floor as a named constant,
  and routed all three sites through it. **Byte-identical** — golden master unchanged
  (`rswork-ub22`, verified pre/post); pinned by `tests/test_leverage_debias.py`. (Full
  unification of the two alternating loops is deferred — blocked by the trainer's float32
  mean step vs `joint_lambda`'s float64; it lands with the opt-in `precision="float64"`
  button.)

### Fixed
- **Docs: corrected the "float64 throughout" precision claim (no code change).**
  `User_readme_functionalities.md` and `CONTRIBUTING.md` previously stated all
  numerical code runs in float64. In fact only the post-fit analytics (Sobol
  decomposition, confidence intervals, prediction) are float64; the model *fit*
  (feature construction, ridge solves, heteroscedastic alternating loop) runs in
  **float32**, because `data/preprocessing.py` casts inputs to float32 and model
  weights initialize as float32. The docs now say so. A configurable float64 fit
  (`precision=`) is planned as a follow-up (advisor-gated; `dtype_unification_brief.md`).
- **One-call diagnostics now describe the model that was actually fitted.** The
  post-fit analytics in `hifi_anova()` used to rebuild an independent, unweighted,
  config-penalty ridge, so for advanced fits the reported `sigma_hat`/`df`/`loo_cv`
  and Sobol CIs could describe a *different* model than the one returned. The
  trainer now surfaces a `FittedDesign` record
  (`hifi_anova/training/fitted_design.py`) carrying the real design, penalty,
  centered target, block layout, and (for Stage D) the GLS precision weights, and
  the API computes every diagnostic from it:
  - **Third-order fits (K3>0):** the real `lambda_order3` penalty is used (was
    padded with `lambda_order2`), and the third-order variance now enters the
    Sobol CI **denominator** (was dropped) — first/second-order CI widths change.
  - **Heteroscedastic (Stage-D) fits:** `sigma_hat`/`df`/`loo`/epistemic-CI are
    now the **weighted (GLS)** diagnostics; `sigma_hat` becomes the whitened
    calibration scale (`≈1` when calibrated), flagged by the new
    `HiFiResult.noise_scale_is_calibration`, with `σ²(x)` exposed via
    `HiFiResult.sigma_x2(x)`. Sobol attribution uses a unit-weight companion
    (two-fit convention); the HC3 CI is not reweighted.
  - The Sobol point estimate and its CI now come from one fit (removes the prior
    point/CI incoherence). The homoscedastic ≤2nd-order path is numerically
    unchanged (golden master re-blessed only to add the new K3 coverage; DEC-029).

### Added
- `HiFiResult.sigma_x2(X)` (input-dependent aleatoric variance) and
  `HiFiResult.noise_scale_is_calibration` flag.
- `ridge_analytics(..., weights=)` for weighted (GLS) diagnostics;
  `predict_intervals(..., weights=)` for the weighted epistemic posterior.

## [0.2.0] — 2026-07-31

### Added
- **Joint mean+variance regularization selection (opt-in).** New module
  `hifi_anova/training/joint_lambda.py` with `optimize_joint_lambda(...)` that
  co-selects the mean lambda and the variance penalty `lambda_h` against a single
  criterion — the "mean-fit vs variance-fit" tradeoff. Previously `lambda_h` was a
  fixed config value (or only swept diagnostically); the mean is fit weighted by
  `1/sigma^2`, so the two are coupled but were never optimized jointly. This is the
  standard location-scale (heteroscedastic-Gaussian) smoothing-parameter problem
  (Wood, Pya & Säfken 2016 — mgcv `gaulss`). Coordinate descent over a log-`lambda_h`
  grid (each point a full IRLS `(w, w_h)` refit with the mean lambda re-selected by
  whitened closed-form GCV/evidence), with **two criteria**: `'kfold_nll'` (default,
  k=5 fold-averaged held-out Gaussian NLL, plus a robust residual-capped companion)
  and `'laml'` (the joint Laplace-approximate marginal likelihood, split-free,
  reusing the Newton Hessian; block-diagonal joint Hessian with an exact-cross-block
  diagnostic). Guards (all default-on): leverage-corrected residuals `r^2/(1-lev)`,
  a `sigma^2` floor, a data-scaled `lambda_h` lower bound with boundary warnings,
  mean-first init, an optional MAP-II hyperprior on `log10 lambda_h`, and a
  `df_h <= N/10` tripwire. Returns the criterion surface, both effective df's,
  fold-wise values, and warnings. The default trainer path (fixed `lambda_h`) is
  unchanged (golden master byte-identical). Validated on the heteroscedastic
  Ishigami (x3 recovered as the dominant variance driver by both criteria), a
  homoscedastic false-positive case, and the leverage-correction guard. See DEC-012.
  Covered by `tests/test_joint_lambda.py`.
- **JAX/autodiff variant of the lambda-selection gradients (`grad='jax'`).** The
  model-selection criterion (GCV / AIC / BIC / -log_evidence) is a pure function
  of lambda, so its gradient can be obtained by `jax.grad` rather than the
  hand-derived closed form. New module `hifi_anova/training/hyperopt_jax.py`
  implements the objective in JAX (float64, primal form, mirroring
  `_criterion_valgrad_multi`) and exposes `criterion_valgrad_jax` (value + AD
  gradient w.r.t. `log10(lambda)`) and `optimize_lambdas_jax`. Wired as a new
  `grad='jax'` mode on `optimize_single_lambda`, `optimize_multi_lambda`, and
  `optimize_multi_lambda_extended` (JAX is imported lazily, only on that path, so
  the default numpy-only import is unchanged). The AD gradient matches the DEC-010
  analytic gradient to floating-point round-off (~1e-16, verified for all four
  criteria, single and multi lambda), and `grad='jax'` optima match
  `grad='analytic'`. Unlike the analytic single-lambda path, the JAX path does
  not require a strictly positive penalty shape (the evidence log-det masks the
  unpenalized support). Default stays `'numeric'`; the golden master is unchanged.
  See DEC-011. Covered by `tests/test_lambda_grad.py`.

### Performance
- **Analytic gradients for lambda selection (modular opt-in).** The GCV / evidence
  / AIC / BIC criteria are closed-form in lambda, but the optimizers took their
  gradients by finite differences (L-BFGS-B with no jacobian). New closed-form
  gradients: `RidgePathEigSolver.criterion_and_grad` (single lambda, from the
  eigendecomposition) and `_criterion_valgrad_multi` (independent per-block
  lambdas, from one A-factorization giving value + full gradient via
  `diag(A^-1)` and `diag(A^-1 C A^-1)`). Wired as an opt-in `grad` argument
  (`'numeric'` default = unchanged; `'analytic'` / `'auto'`) on
  `optimize_single_lambda`, `optimize_multi_lambda`, and
  `optimize_multi_lambda_extended`. The analytic optimizers land on the same
  optima as the numeric ones (gradients verified against finite differences to
  ~1e-6 for all four criteria). A JAX/AD variant and a joint mean+variance lambda
  objective are planned follow-ups. Covered by `tests/test_lambda_grad.py`.
- **Vectorized `analysis.sobol.compute_sobol_indices`.** The per-block variances
  `w_b^T G w_b` (per variable, per pair, per triple — for both the mean and the
  variance spectra) were computed in Python loops of tiny JAX ops, each forcing a
  host device-sync via `float(...)`. They are now batched into a single numpy
  einsum per order (`_block_variances`), cutting the call from ~88 ms to ~31 ms
  (~2.9x) on a D=10, 45-pair model. Numerically identical (verified against the
  refactor golden and the full test tier).
- **Eigendecomposition fast path for the regularization path** (`analysis.reg_path.compute_reg_path`).
  Because the sweep scales every order's penalty proportionally with `lambda_1`,
  the penalty is `lambda_1 * reg_shape` for a fixed shape — so the whole grid can
  be answered from a single eigendecomposition of the whitened Gram instead of a
  full ridge solve per grid point. New `RidgePathEigSolver`
  (`training/hyperopt.py`) reproduces every diagnostic (`w`, RSS, df, GCV, AIC,
  BIC, and the profile log-evidence — matching the dual-form evidence for `F > N`
  via the `log|K| = sum(log((mu+lambda)/lambda))` identity). Measured **~22x**
  faster on a 630x294 design at 40 lambdas, agreeing with the per-lambda solve to
  ~1e-13. A new `solver` argument (`'auto'` default / `'eig'` / `'solve'`) selects
  it; `'auto'` uses the fast path only when the penalty shape is strictly positive
  and well conditioned, and falls back to the exact solve for ill-conditioned
  shapes (e.g. `curvature`, whose weights span `(2*pi*k)^4`). The chosen path is
  recorded in `RegPathResult.solver_used`. This touches no autodiff path (lambda
  selection uses SciPy, not AD) and is covered by `tests/test_reg_path_eig.py`.

### Added
- **Worked-example showcase** (`docs/ishigami_showcase.md` + `.html`) — a visual,
  explained tour of the toolbox on the heteroscedastic Ishigami fit: the dual
  mean/log-variance spectrum and ellipse views, the learned effects (x3 flat
  after pruning), the regularization paths + Pareto + variance-λ_h, the
  fit-under-noise diagnostics (parity, surface, variance recovery), and the
  `verify_model` output. Figures committed under `docs/figures/`; linked from the
  README.
- **Heteroscedastic-Ishigami benchmark** (`benchmarks/`) — a fixed, committed
  dataset (`train.csv`, `test.csv`, `test_truth.csv`) plus a comparison harness
  (`run_benchmark.py`). Fit any model on the train split, predict the test
  inputs, and score with `evaluate_predictions(y_pred, sigma_pred)`: R² vs
  observed *and* vs the noiseless truth (honest generalization), NLL/coverage,
  and first-order Sobol MAE vs the analytic indices. `--baselines` compares
  HiFi-ANOVA against sklearn GradientBoosting/RandomForest/MLP, showing the
  overfitting gap and that HiFi-ANOVA is the only method with native, accurate
  Sobol indices. Data integrity covered by `tests/test_benchmark.py`.
- **Fit-diagnostic plots** — `analysis.plots.plot_parity` (general
  predicted-vs-actual with the 45° line and R²), and a "Fit diagnostics" section
  in the Ishigami example: predicted-vs-observed (scatter = noise, R²≈0.8) and
  predicted-vs-true (R²≈0.98) parity, a transparent true-vs-fit surface, and a
  predicted-σ vs true-σ recovery plot — making explicit how R² is read under
  heteroscedastic noise.
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
  log-variance index stays ~1 across the whole `lambda_h` range) and also renders the
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
  visualization of the mean/log-variance spectrum, in both `analysis.visualization`
  (lightweight, `save_path=` API) and `analysis.plots` (publication style,
  returns `(fig, ax)`). `mode='glyph'` draws one ellipse per variable whose
  width proportional to mean sensitivity and height proportional to the
  log-variance index (shape tells the
  story: wide = mean driver, tall = variance driver); `mode='plane'` places each
  variable at (S^f, S^h) with CI ellipses.
- **`examples/run_ishigami_heteroscedastic.py`** — an end-to-end capabilities
  showcase on a heteroscedastic Ishigami: recovers the analytic mean Sobol
  indices, fits the log-variance spectrum (x3 emerges as a hidden driver),
  rediscovers the x1–x3 interaction with the residual sieve, checks calibration,
  bootstraps CIs for the dual spectrum, and renders the ellipse plots. It also
  prints the explained-variance split of the mean vs the variance model and
  renders the regularization path (L-curve, GCV/evidence, Sobol-vs-λ, and the
  variance-decomposition trade-off) via `compute_reg_path` / `plot_reg_path`.
- **`docs/CI_theory.md`** — theory note for the Sobol confidence intervals: the
  then-used HC0 sandwich estimator (now HC3), the delta method, the
  full-gradient derivation
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
- **`UnboundLocalError` in third-order fits with the default `triple_selection`.**
  With `K3 > 0` and `triple_selection='all_active'` (the default), the trainer
  routed through a threshold-fallback branch that referenced `_bs` — a name bound
  only by a *later* local `import basis_size as _bs` inside `fit()`, which made
  `_bs` a function-local and raised `UnboundLocalError` before that import ran.
  Configs that set `triple_selection='all'` or a principled method sidestepped the
  branch, hiding it. Now uses the module-level `basis_size`. Regression test:
  `tests/test_trainer_bugfixes.py`.
- **Unbound log-variance intercept `h0` in the heteroscedastic solver.** In
  `_fit_heteroscedastic`, the outer loop's `h0_init if outer == 0 else h0` and the
  post-loop model build referenced `h0` before it was guaranteed bound (a
  read-before-assignment that only manifests in degenerate configs but that static
  analysis flags). `h0` is now initialized to `h0_init` before the loop; a no-op
  for valid configs (`max_outer_iter >= 1`).
- **Unknown `residual` type silently disabled Stage C.** A typo'd or unrecognized
  residual type (e.g. `residual='rbg'`, or a config dict `{'type': 'gaussian'}`)
  added stage `'C'` to the pipeline, but no Stage C branch matched it, so the
  stage silently did nothing — the caller got a residual-free model believing one
  had been fitted. `training/trainer.py` now validates the residual type at the
  top of Stage C and raises `ValueError` (listing the known types
  `nn`/`rbf`/`rff`/`nystrom`) instead of no-op'ing; a bare-string residual config
  is coerced to `{'type': ...}` first. The variance-residual path already raised
  via `create_residual`. Covered by `tests/test_residual_validation.py`.
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
  the three contributions — the dual mean/log-variance spectrum, the three
  basis families (Fourier / Legendre / Haar) with per-variable effect signatures,
  and the fixed-configuration one-factorization diagnostic suite — instead of
  describing the method as a
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
