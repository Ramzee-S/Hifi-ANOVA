# HiFi-ANOVA User Guide

**Interpretable regression with analytic Sobol diagnostics for the mean *and* the variance.**

HiFi-ANOVA — *Hoeffding Interaction–Fidelity ANOVA* — fits a functional-ANOVA
(Hoeffding) decomposition of both the conditional **mean** and the conditional
**variance** of a response, realized over three interchangeable basis families
(**Fourier, Legendre, Haar**). Because the fit is linear in the basis
coefficients, sensitivity indices, uncertainty estimates, and structure-discovery
diagnostics reuse the finalized ridge system — the model is interpretable *by
design*, not explained post-hoc. Selection, lambda optimization, and joint
mean--variance training can require additional factorizations before the final
configuration is fixed.

This guide documents the library as implemented. Every option, default, and code
snippet below is drawn from the source in `hifi_anova/`. It targets Python 3.10+
and assumes the `jax` / `equinox` scientific stack described in the top-level
`README.md`.

> **Mixed-basis boundary.** A different fixed Fourier/Legendre/Haar family per
> variable is currently supported for the Stage-A/B mean model, all retained
> pairs, and block-correct Sobol diagnostics. Mixed selection/pruning, Stage C,
> Stage D, and `K3>0` are not supported; see the capability matrix in §7. The
> complete automated mean/variance pipeline remains available on uniform bases.

> **Implemented scope vs manuscript program (0.3.0).** The package implements
> the v07 model geometry and fixed-configuration diagnostics. It does **not**
> expose the reusable FDR-controlled efficient-score ladder, the honest three-way
> selection/inference/reporting workflow, or post-selection guarantees. BIC,
> group lasso, the 1-SE rule, pruning, and residual-sieve thresholds are model-
> selection heuristics, not Theorem-2 tests.

> Positioning, honestly stated. Computing analytic Sobol indices from the
> coefficients of an orthogonal-basis regression is well established — it
> underlies variance decomposition in polynomial chaos expansions (PCE) and
> RS-HDMR. What is distinctive here is the combination: a **dual mean +
> log-variance** Sobol spectrum, **three basis families** with per-variable effect
> signatures, and a **fixed-configuration diagnostic suite** (closed-form LOO, GCV,
> sandwich/delta confidence intervals, regularization paths, interaction
> screening). HiFi-ANOVA trades a little predictive accuracy for interpretability;
> it is not primarily a top-tier predictor.

---

## Table of contents

1. [Overview & concepts](#1-overview--concepts)
2. [Installation & quickstart](#2-installation--quickstart)
3. [The two APIs](#3-the-two-apis)
4. [Configuration reference](#4-configuration-reference)
5. [Complexity modes & stages](#5-complexity-modes--stages)
6. [Regularization strategies](#6-regularization-strategies)
7. [Bases & effect signatures](#7-bases--effect-signatures)
8. [Residual models](#8-residual-models)
9. [Sensitivity & uncertainty](#9-sensitivity--uncertainty)
10. [Diagnostic suite](#10-diagnostic-suite)
11. [Working with results](#11-working-with-results)
12. [Assumptions & caveats](#12-assumptions--caveats)

---

## 1. Overview & concepts

### The decomposition

Given inputs `x = (x₁, …, x_D)` and a response `y`, HiFi-ANOVA writes the
conditional mean as a functional-ANOVA (Hoeffding) sum,

```
E[y | x] = f₀ + Σᵢ fᵢ(xᵢ) + Σᵢ<ⱼ fᵢⱼ(xᵢ,xⱼ) + Σ fᵢⱼₖ(...) + residual
```

where each component `fᵢ`, `fᵢⱼ`, … is expanded in a structured basis that
satisfies the Hoeffding side conditions (each basis function integrates to zero
over `[0,1]`, so components are orthogonal). Coefficients are found by a single
penalized (ridge) least-squares solve.

For heteroscedastic data, the **log-variance** is modelled with the *same*
machinery:

```
log Var[y | x] = h₀ + Σᵢ hᵢ(xᵢ) [+ pairs] [+ triples] [+ residual]
```

This yields a **dual Sobol spectrum**: every variable gets one index for its
effect on the expected outcome and one for its effect on the predictive
*uncertainty*. Variables that shape uncertainty but carry no mean signal (a
"hidden" heteroscedastic driver) show up in the log-variance spectrum while being
invisible to ordinary feature importance.

### Sobol indices, analytically

Because each component `fᵢ` is `Φᵢ wᵢ` with a known Gram matrix `Gᵢ`
(`Gᵢ[a,b] = ∫ φₐ φ_b`), its variance under a product input measure is exactly
`wᵢᵀ Gᵢ wᵢ`. Normalizing by the total gives the first-order Sobol index; pairs and
triples work the same way with tensor-product Gram matrices. No Monte-Carlo, no
extra model evaluations.

### Three bases

| `basis_name` | Features per variable | Gram matrix | Captures best |
|--------------|-----------------------|-------------|---------------|
| `'fourier'` (default) | `2K+1` (linear + cos/sin), or `2K` without linear | near-diagonal (linear–sin cross-terms) | oscillatory / periodic effects |
| `'legendre'` | `K` shifted Legendre polynomials P̃₁…P̃_K | perfectly diagonal, `G[j,j]=1/(2j+3)` | smooth polynomial effects |
| `'haar'` | `2^K − 1` wavelets (`K` reinterpreted as max scale `J`) | identity | localized effects: steps, thresholds, regime boundaries |

The three can be **mixed per variable** while preserving inter-variable
orthogonality (see [§7](#7-bases--effect-signatures)).

### Quantile space

All inputs are mapped to uniform marginals on `[0,1]` with a
`QuantileTransformer` before fitting (`preprocess_data`). "Linear" and
"low-frequency" are therefore defined in quantile space — a monotone, nonlinear
reparameterization of the original features.

---

## 2. Installation & quickstart

Requires **Python 3.10+**. From the repository root:

```bash
conda env create -f environment.yml
conda activate hifi_anova
pip install -e .
# optional: pip install -e ".[salib]"   # SALib ground-truth comparison
#           pip install -e ".[dev]"      # pytest
```

`precision=` selects the requested model/storage dtype. The default model weights
and storage are float32, while selected post-fit analytics and Stage-D linear
solves are promoted to float64. Requesting `precision="float64"` also makes the
model/storage path float64. The implementation is therefore mixed precision;
neither setting should be read as an end-to-end float32 claim.

The effective fit precision resolves by this precedence (highest first):

1. an explicit `hifi_anova(..., precision="float64")` argument (or
   `config['precision']`);
2. a process-wide `hifi_anova.precision.set_fit_precision("float64")` override;
3. the `HIFI_ANOVA_X64=1` environment variable;
4. otherwise the `"float32"` default.

An explicit argument always wins over the global override and the environment, so
`precision="float32"` forces float32 even when `HIFI_ANOVA_X64=1` is set — use it
to pin a single call against a global opt-in. An unrecognized `HIFI_ANOVA_X64`
value (e.g. a typo) is **not** silently reinterpreted: it warns and is ignored.
The *effective* precision is recorded in `result.config['precision']` and in the
saved `meta.json`. (Earlier releases hard-coded the default so the env/override
controls did not reach an ordinary one-call fit; fixed in DEC-044.)

If you call lower-level functions directly and want float64, enable x64 yourself
*and* pass float64 arrays:

```python
import jax
jax.config.update("jax_enable_x64", True)
```

### Compute backend — NumPy exact core vs. JAX

`hifi_anova(..., backend=...)` selects the linear-algebra core. The **same
fit-path code** runs on either backend (a proxy resolves array ops to
NumPy/SciPy or `jax.numpy`), so results agree to machine precision on float64
(homoscedastic fits are byte-comparable; Stage-D agrees to ~1e-11, including the
guard decisions).

- `backend="auto"` (**default**) — runs the **NumPy exact core** (float64) when
  the configuration stays inside the supported surface, and routes to JAX only
  when a JAX-native path is requested (see below).
- `backend="numpy"` — the float64 exact core. It removes the per-shape JAX
  recompile, which is why the interactive console re-fits in ~0.1 s on it. It is
  **float64-only** and **raises** (rather than silently changing the model) on
  `precision="float32"`, the NN residual (`residual_nn` / residual type `'nn'`),
  and the stage-laddering `mode='auto'/'full'`.
- `backend="jax"` — the JAX path; required for the float32 speed mode and the
  JAX-native residual/modes above.

The linear residual families (Stage C `'rbf'`/`'rff'`/`'nystrom'` and Stage-D
`variance_residual`) run on **either** backend. One caveat: **RFF** random
frequencies are drawn backend-natively, so an RFF fit is deterministic per
backend but **not numerically comparable across backends**; RBF and Nyström
(seeded k-means) are cross-backend reproducible. A plain `hifi_anova(X, y)` now
fits in float64 via the core; pass `backend="jax"` (with `precision="float32"`
if desired) to reproduce the earlier JAX/float32 behavior.

Top-level convenience imports:

```python
from hifi_anova import (
    hifi_anova,             # one-call API
    HiFiResult,             # its result object
    HiFiANOVATrainer,       # staged trainer
    estimate_sobol,         # experimental additivity-calibrated mode
    compute_sobol_indices,  # Sobol from a fitted model
)
```

### 60-second example

```python
from hifi_anova.api import hifi_anova

result = hifi_anova(X, y, feature_names=['MedInc', 'HouseAge', 'AveRooms'])

result.summary()                              # Sobol indices w/ CIs, R², noise, df
pred = result.predict(X_new)                  # predictions on new data (original scale)
lo, hi = result.predict_intervals(X_new)      # 95% prediction intervals
x_grid, f = result.component_curve('MedInc')  # learned effect of one variable
result.save('my_model/')                      # persist model + transformer + config
```

---

## 3. The two APIs

HiFi-ANOVA exposes two entry points that share the same underlying machinery.

### 3.1 One-call API — `hifi_anova(...)`

`hifi_anova(X, y, ...)` takes raw (original-scale) data and handles
preprocessing, fitting, Sobol analysis, confidence intervals, and noise
estimation, returning a single `HiFiResult`.

Full signature and defaults (`hifi_anova/api.py`):

```python
hifi_anova(
    X, y,
    feature_names=None,          # list[str] or None -> ['x1', 'x2', ...]
    K1=5,                        # max harmonic/degree/scale, first order
    K2=3,                        # max harmonic/degree/scale, second order
    strategy=None,               # None -> 'curvature' if heteroscedastic else 'variance'
    mode='second',               # 'first' | 'second' | 'full' | 'heteroscedastic' | 'auto'
    variable_selection='bic',    # 'bic' | 'group_lasso' | '1se' | None
    residual=None,               # 'rbf' | 'rff' | 'nystrom' | None
    heteroscedastic=False,       # if True, also fit the variance model
    seed=42,
    verbose=True,
    **kwargs,                    # any extra trainer config keys (see §4)
)
```

Notes on the one-call defaults:

- It always sets `lambda_order1=0.001`, `lambda_order2=0.01` internally. Override
  via `**kwargs`.
- When `variable_selection` is set, `pair_candidates` defaults to `'either'`.
- When `heteroscedastic=True`, it adds `Kh=3`, `lambda_h=0.1`, and stages
  `['A','B','D']` (or `['A','B','C','D']` if a `residual` is also requested), and
  resolves `strategy` to `'curvature'` (kept `'variance'` otherwise). Stage D is
  guarded: it reverts to a constant variance (with a warning) unless the
  heteroscedastic model beats it on held-out likelihood — see [§4.8](#48-variance-model--alternating-optimization-stage-d).
- When `residual` is set, sensible per-type sub-config defaults are filled in
  (see [§8](#8-residual-models)).
- `mode='auto'` reads `auto_threshold` (default `0.01`) from `**kwargs`.

The one-call `K1=5, K2=3` defaults are deliberately smaller than the staged
trainer's internal defaults (`K1=10, K2=5`).

### 3.2 Staged API — `HiFiANOVATrainer` + `compute_sobol_indices`

For full control, drive the trainer directly. You manage preprocessing and read
Sobol indices off the fitted model.

```python
import jax
jax.config.update('jax_enable_x64', True)

from hifi_anova.data.synthetic import generate_friedman1
from hifi_anova.data.preprocessing import preprocess_data
from hifi_anova.training.trainer import HiFiANOVATrainer
from hifi_anova.analysis.sobol import compute_sobol_indices

X, y = generate_friedman1(n_samples=10000, noise_std=1.0, n_irrelevant=5)
data = preprocess_data(X, y)   # splits + QuantileTransformer to [0,1]

config = {'mode': 'second', 'K1': 10, 'K2': 5, 'strategy': 'variance',
          'lambda_order1': 0.001, 'lambda_order2': 0.01}

trainer = HiFiANOVATrainer(config)
model, results = trainer.fit(data['x_train'], data['y_train'],
                             data['x_val'], data['y_val'])

sobol = compute_sobol_indices(model, data['x_test'])
for i in range(10):
    print(f"x{i+1}: S1={sobol['mean_sobol']['first_order'][i]:.4f}")
```

`HiFiANOVATrainer.fit(x_train, y_train, x_val, y_val, key=None)` expects inputs
already transformed to `[0,1]` and returns `(model, results_dict)`. `results`
holds per-stage diagnostics (`stage_A`, `stage_B`, …) and any selection/pruning
info.

`compute_sobol_indices(model, x_data=None)` returns the full dual spectrum
(structure detailed in [§9](#9-sensitivity--uncertainty)). Passing `x_data`
additionally computes empirical residual variance and **correlative** indices.

---

## 4. Configuration reference

The trainer is configured with a plain `dict`. Every key below is read by
`HiFiANOVATrainer.fit` (or `resolve_mode`) in the source; keys not listed are not
consulted. Unless noted, a key is optional and falls back to the default shown.

### 4.1 Harmonics / interaction orders

| Key | Type | Default | Effect |
|-----|------|---------|--------|
| `K1` | int | `10` | Max harmonic (Fourier) / degree (Legendre) / scale `J` (Haar) for first order. |
| `K2` | int | `5` | Same, second order. `0` disables pairs. |
| `K3` | int | `0` | Same, third order. `0` disables triples. |
| `Kh` | int | `3` | First-order **variance** model order (Stage D). |
| `K2h` | int | `0` | Second-order variance order. `>0` adds variance pair interactions. |
| `K3h` | int | `0` | Third-order variance order (small `D`). |

Per-variable feature counts follow `basis_size(K, include_linear, basis_name)`:
Fourier `2K+1` (or `2K`), Legendre `K`, Haar `2^K−1`.

### 4.2 Penalties (regularization strengths)

| Key | Type | Default | Effect |
|-----|------|---------|--------|
| `lambda_order1` | float | `0.001` | First-order ridge strength. |
| `lambda_order2` | float | `0.01` | Second-order ridge strength. |
| `lambda_order3` | float | `0.1` | Third-order ridge strength. |
| `lambda_h` | float | `0.1` | First-order variance penalty. |
| `lambda_h2` | float | `lambda_h * 10` | Second-order variance penalty. |
| `lambda_h3` | float | `lambda_h * 100` | Third-order variance penalty. |
| `lambda_h_residual` | float | `lambda_h * 10` | Variance-residual penalty. |
| `lambda_residual` | float | `1.0` | Mean linear-residual penalty (also settable inside the `residual` dict). |

The default first→second ratio is `0.01 / 0.001 = 10×`; per-order penalties
scale up with the (larger) parameter count of higher-order blocks. The
`estimate_sobol` and `compute_reg_path` helpers use analogous ratios
(`lambda_ratio=10`, `lambda_ratio_3=100`, `lambda_ratio_res=1000`).

### 4.3 Strategy

| Key | Type | Default | Effect |
|-----|------|---------|--------|
| `strategy` | str *or* dict | `'variance'` | Penalty shape (see [§6](#6-regularization-strategies)). A dict allows per-order strategies: `{'order1': 'curvature', 'order2': 'smoothness', 'default': 'variance'}`. |

### 4.4 Complexity: `mode` / `stages`

| Key | Type | Default | Effect |
|-----|------|---------|--------|
| `mode` | str | `'second'` | `'first'`, `'second'`, `'full'`, `'heteroscedastic'`, or `'auto'`. Resolved into `stages` by `resolve_mode`. |
| `stages` | list[str] | `['A','B']` | Explicit stage list; overrides `mode`. **The default is mean-only** (A, B) — the nonlinear residual (C) and the variance model (D) are opt-in, never fitted by accident. Request them via `mode` or an explicit `stages`. |
| `auto_threshold` | float | `0.01` | In `mode='auto'`, minimum residual fraction (`1 − R²_val`) to add the next stage. |

See [§5](#5-complexity-modes--stages) for the stage semantics.

### 4.5 Basis configuration

| Key | Type | Default | Effect |
|-----|------|---------|--------|
| `basis_name` | str | `'fourier'` | `'fourier'`, `'legendre'`, or `'haar'` (applied to all variables in non-mixed mode). |
| `basis_type` | str | `'full'` | Preset for `include_linear_*`: `'full'` (all True), `'spectral_higher'` (order 1 True, 2+ False), `'spectral_all'` (all False — pure harmonics everywhere). |
| `include_linear_1` | bool | from `basis_type` | Include the linear term in the first-order Fourier basis. |
| `include_linear_2` | bool | from `basis_type` | Same, second order. |
| `include_linear_3` | bool | from `basis_type` | Same, third order. |
| `include_linear_h1/h2/h3` | bool | follow mean settings | Same, for the variance model. |
| `basis_per_variable` | dict *or* `'auto'` | `None` | Mixed per-variable basis (see [§7](#7-bases--effect-signatures)). `None` = uniform. |

`include_linear` is only meaningful for Fourier; Legendre always includes its
linear polynomial P̃₁ and Haar has no linear term, so the flag is ignored for
those bases.

### 4.6 Selection & pruning

Pair generation is a two-step pipeline: **select active variables**, then
**generate candidate pairs**, then optionally **prune** after fitting.

| Key | Type | Default | Effect |
|-----|------|---------|--------|
| `variable_selection` | str or None | `None` | Principled first-order variable selection: `'bic'`, `'group_lasso'`, `'1se'`. `None` uses the legacy `pair_selection` path. |
| `pair_candidates` | str or None | `None` | Candidate-pair heuristic over active variables: `'all'`, `'both'` (both vars active), `'either'` (≥1 active). Only used together with `variable_selection`. |
| `pair_pruning` | str | `'none'` | Post-fit pair pruning criterion: `'bic'`, `'group_lasso'`, `'1se'`, or `'none'`. |
| `first_order_pruning` | str | `'none'` | Post-fit **first-order** pruning: `'bic'`, `'group_lasso'`, `'1se'`, or `'none'`. Zeroes the entire first-order block of any variable whose marginal effect the criterion rejects — a leave-one-group-out test on the full design. First-order blocks are Hoeffding-orthogonal to the pair/triple blocks, so this cleanly removes a *spurious* main effect (e.g. a variable that is pure interaction) without disturbing the interactions. Plain ridge can only shrink such a block; it can never set it to zero. `'bic'` is the most robust at small N. |
| `pair_selection` | str or None | `None` | Legacy one-shot variable-selection + candidate-generation. |
| `pair_threshold` | float | `0.01` | Sobol threshold for threshold-based variable activity. |
| `max_pair_variables` | int or None | `None` | Cap on number of active variables entering pair generation. |
| `triple_selection` | str | `'all_active'` | Triple candidate mode: `'all'`, `'all_active'`, `'two_active'`, `'one_active'`, or one of `'bic'/'group_lasso'/'1se'` (principled, then `'two_active'`). |
| `triple_pruning` | str | `'none'` | Post-fit triple pruning: `'bic'`, `'group_lasso'`, `'1se'`, or `'none'`. |
| `var_pair_selection` | str or None | `None` | Variance pair selection (Stage D): `None`/`'all'` = all pairs, `'auto'` = quick variance fit then select. |
| `var_triple_selection` | str or None | `None` | Variance triple selection; currently all triples. |

#### Removing spurious main effects — `first_order_pruning`

A variable can have a **zero first-order effect but a non-zero total-order
effect** — it acts *only* through an interaction. The textbook case is Ishigami
`x₃` (`f = sin x₁ + a·sin²x₂ + b·x₃⁴·sin x₁`): its main effect is exactly zero,
yet a plain ridge fit still assigns `x₃` a small, noisy first-order component
(and at small `N` a large one). Ridge can only *shrink* that block, never zero
it, and `variable_selection` does not touch first-order blocks — it only gates
pair candidates.

`first_order_pruning` closes this gap. Post-fit, it applies the selected
leave-one-group-out model-selection heuristic to the first-order blocks and
**zeros the entire block** of any variable the heuristic drops. This is not a
calibrated significance or Theorem-2 test. Because first-order and pair/triple
blocks are Hoeffding-orthogonal, removing a dropped block does not disturb the
interactions.

```python
config = {
    'K1': 6, 'K2': 4, 'strategy': 'curvature',
    'first_order_pruning': 'bic',   # 'bic' | 'group_lasso' | '1se' | 'none'
}
# Ishigami: x₃'s first-order component becomes exactly flat (S₁(x₃)=0),
# while x₁, x₂ and the x₁–x₃ interaction are untouched — robust down to N≈100.
```

Use `'bic'` as the default — it is the most robust at small `N`; `'group_lasso'`
also works for `N ≳ 1000`, and `'1se'` is the most conservative. It applies to
both the first-order-only and Stage-B paths, and is re-enforced after the
heteroscedastic (Stage D) mean refit. See
`examples/run_ishigami_heteroscedastic.py` for a worked example, whose
regularization-path panel also shows `x₃`'s first-order Sobol pinned at zero
across all `λ`.

### 4.7 Residual model (Stage C)

Provide a `residual` dict (preferred) or the legacy `residual_nn` dict. See
[§8](#8-residual-models) for trade-offs.

| Sub-key | Applies to | Default | Effect |
|---------|-----------|---------|--------|
| `type` | all | `'nn'` (in Stage C) | `'rbf'`, `'rff'`, `'nystrom'`, or `'nn'`. |
| `lambda_residual` | all | `1.0` | Residual ridge penalty. |
| `n_centers` | rbf | `min(300, N//5)`¹ | RBF centers (k-means). |
| `sigma` | rbf | `0.2` | RBF Gaussian width. |
| `n_features` | rff | `1000` | Random Fourier features. |
| `gamma` | rff | `3.0` | RFF frequency scale (small = smooth). |
| `n_inducing` | nystrom | `min(300, N//5)`¹ | Inducing points. |
| `kernel` | nystrom | `'matern52'`¹ | `'rbf'`, `'matern32'`, or `'matern52'`. |
| `lengthscale` | nystrom | `0.2` | Kernel lengthscale. |
| `hidden_dims` | nn | `[256, 256, 256]` | MLP layer widths. |
| `lr` | nn | `0.001` | Adam learning rate. |
| `weight_decay` | nn | `0.0001` | Weight decay. |
| `epochs` | nn | `200` | Max epochs. |
| `batch_size` | nn | `512` | Minibatch size. |
| `patience` | nn | `20` | Early-stopping patience. |
| `enabled` | nn | `False` | Legacy flag: the NN residual only runs if `True`. |

¹ Defaults marked with ¹ are filled in by the one-call `hifi_anova(...)` wrapper;
the linear-residual classes themselves default to `n_centers=300`,
`kernel='rbf'` when constructed directly.

### 4.8 Variance model & alternating optimization (Stage D)

| Key | Type | Default | Effect |
|-----|------|---------|--------|
| `variance_residual` | dict or None | `None` | RBF/RFF residual for higher-order variance structure. Sub-keys `type` (`'rbf'`), `sigma` (`0.3`), `n_centers` (`150`). |
| `max_outer_iter` | int | `10` | Alternating (mean ↔ variance) outer iterations. |
| `alternating_tol` | float | `1e-4` | Relative NLL tolerance for stopping the loop. |
| `newton_max_iter` | int | `10` | Inner Newton iterations for the log-variance fit. |
| `leverage_correction` | bool | `True` | Feed the variance solve leverage-corrected squared residuals `r²/(1−lev)` — raw in-sample `r²` is biased low where the mean fits tightly (`E[r²] ≈ σ²(1−lev)`), which is what destabilized the loop on rich bases — and estimate the constant-variance baseline the same way. `False` restores the raw pre-DEC-028 solve. |
| `alternating_early_stop` | bool | `True` | Score every outer iterate on held-out NLL and keep the best one, instead of trusting the train-NLL convergence point (the first iterate — one-step feasible GLS — is often the best). `results['stage_D']` reports `val_nll_trajectory`, `best_outer_iteration`, `n_outer_iterations`. |
| `heteroscedastic_guard` | bool | `True` | Master switch for the Stage-D safety net (box below). `False` forces the raw alternating fit and skips the checks (use when you know the data is heteroscedastic and want no overhead). |
| `min_noise_ratio` | float | `1e-2` | Near-noiseless entry gate, as a scale-free noise-to-signal variance ratio (`residual_var / total_var = 1 − R²`). Below it Stage D is **skipped** with an explanatory warning: on essentially-noiseless data the residual is deterministic mean-approximation error, not aleatoric noise, and a held-out-NLL guard cannot tell the two apart. The `1e-2` default is calibrated (DEC-039, 48-fit sweep): essentially-noiseless fits sit at `1−R² ≤ 6e-4` while genuine heteroscedastic data — down to a weak `β=1` — sits at `1−R² ≥ 0.095`, a ~160× gap (σ²-shape does **not** separate them). Per-fit overridable: **lower it** to keep Stage D for genuinely heteroscedastic high-SNR data (SNR 100–1000). |
| `variance_selection_margin` | float | `2e-3` | Minimum *relative* held-out-NLL improvement for the heteroscedastic model to be kept over constant variance. Larger ⇒ more conservative (prefers homoscedastic). |
| `stage_d_joint_gls_mean` | bool | `True` | Solve the Stage-D weighted mean as the **penalized-GLS optimum** — weighted-center both `y` and Φ so the intercept is profiled jointly (DEC-039). Restores monotone alternating descent and an efficient weighted mean. `False` is the legacy fixed-intercept / uncentered-Φ solve (a compatibility flag, slated for removal; see below). |
| `variance_selection_mean_consistent` | bool | `True` | Compare σ²(x) vs constant variance under the **same** (weighted) mean, so the keep/revert decision isolates the variance (DEC-039). `False` is the pre-flip package comparison. |
| `variance_selection_mean_fallback` | bool | `False` | Opt-in outcome: keep the variance model but ship the unit-weight mean if the weighted mean degrades the package. Under the joint-GLS default this should never be needed, so it doubles as an **invariant monitor** — a firing sets `results['stage_D']['mean_fallback_anomaly']`. |

> **Stage D stability.** The alternating loop reweights the mean fit by
> `1/σ²(x)`, so a poorly-posed variance model could destabilize the mean. Since
> DEC-028 the loop is **stabilized at the source** — the variance solve sees
> leverage-corrected residuals (`leverage_correction`) and the best outer
> iterate is selected on held-out NLL (`alternating_early_stop`) — and the
> DEC-027 safety net remains as a backstop, all on by default and overridable:
>
> - **Leverage correction + trajectory selection (the fix).** Raw in-sample
>   residuals under-report σ² exactly where a rich mean basis fits tightly;
>   feeding them to the variance solve created a feedback (weights blow up →
>   mean interpolates the low-σ region → residuals shrink further) that could
>   degrade the mean, especially under `strategy='variance'`. With the
>   correction on, both `'variance'` and `'curvature'` are stable; the one-call
>   API still defaults to `'curvature'` when `heteroscedastic=True`.
> - **Model selection against constant variance (the main catch).** Stage D
>   *always* compares the fitted heteroscedastic model against a **homoscedastic
>   (constant-variance) baseline** on **held-out** NLL, and keeps the variance
>   model only if it improves the held-out likelihood by `variance_selection_
>   margin`. The baseline σ̂² is leverage-corrected too (an effective-df
>   correction), so the comparison is fair on both sides. Constant variance is
>   the default *outcome*: the data must earn the input-dependent variance.
 - **Degeneracy guards.** A fit that diverges (`nan`) or inflates the mean RMSE
>   >2× reverts to the mean-only fit, each with a warning naming the cause and the
>   fix. A separate near-noiseless **entry gate** (`min_noise_ratio`, default
>   `1e-2`) skips Stage D before fitting when `1−R²` is below the gate — the
>   residual is deterministic mean-approximation error rather than aleatoric
>   noise, which the held-out-NLL guard cannot distinguish. It is per-fit
>   overridable for genuinely heteroscedastic high-SNR data.
>
> - **Joint-GLS weighted mean (DEC-039, the root fix; `stage_d_joint_gls_mean`).**
>   The alternating mean update solves the *penalized-GLS optimum* — it
>   weighted-centers both `y` and Φ and profiles the intercept jointly — rather
>   than fixing `f0 = Σwₙyₙ/Σwₙ` and solving on uncentered Φ. Fourier features
>   have ~0 unweighted mean but a nonzero *weighted* mean under `1/σ²(x)` weights,
>   so the legacy uncentered solve was not the GLS optimum and yielded a weighted
>   mean that could lose to the unit-weight mean on its own objective — dragging a
>   correct variance model into a false revert. With the fix the guard keeps
>   genuinely heteroscedastic variance models, and the keep/revert comparison uses
>   the *same* weighted mean on both sides (`variance_selection_mean_consistent`).
>   The effective estimator vintage is recorded for provenance as
>   `results['stage_D']['mean_intercept_mode']` (and on the fitted-design record /
>   saved metadata): `profiled_joint_gls` for the default, or
>   `legacy_fixed_intercept_uncentered_features` under the compatibility flag.
>   The two compatibility flags (`stage_d_joint_gls_mean`,
>   `variance_selection_mean_consistent`) are transitional — they will eventually
>   become permanent behavior and cease to exist as knobs.
>
> **Variance is opt-in**: the default fit (`mode='second'`) models only the mean
> — request Stage D via `heteroscedastic=True`, `mode='heteroscedastic'`, or
> `mode='auto'`.

### 4.9 User-defined equation systems (term structure)

By default `K1`/`K2`/`K3` request a *uniform* decomposition — every variable
enters at every requested order — and the selection/pruning heuristics of
[§4.6](#46-selection--pruning) decide which terms survive. When you instead
want to **pin an exact term structure yourself** — a specific set of pairs, a
per-pair harmonic order, an order-selective membership, or a variance model over
a named subset — the keys below express it directly. All of them are **additive
and inert by default** (every code path keys on a `None` default), so a fit that
uses none of them is byte-identical to before.

These keys **assert** a structure rather than discover one; they are the honest
alternative to a data-driven selector for the cases the selectors do not yet
cover (`Session/DECISIONS.md` DEC-053/DEC-054). Two honesty labels recur below
and are carried into `results['term_structure']` and `summary()`:

- **non-hierarchical** — a model with a pair term `(i, j)` but *no* first-order
  block for `i` (or `j`) violates the usual strong-heredity convention. This is
  a legitimate modeling choice, but it is *your* assertion, and the output says
  so rather than hiding it.
- **homoscedasticity-asserted** — a variable excluded from the variance model is
  taken to have flat `σ²(x)` (`Sʰ ≡ 0`) *by assumption*, not because the data
  showed it. The keys still span all `D`, reporting an exact zero.

| Key | Type | Default | Effect |
|-----|------|---------|--------|
| `K2` (mapping form) | `{(i, j): K2_ij}` | — | A **mapping** pins the exact retained pairs *and* gives each its own second-order order (per-pair Grams, ragged blocks end-to-end through features → penalties → model → Sobol/CI → Stage D → persistence). Pair keys must be canonical (`i < j`). Rejects data-driven pair selection/pruning, `K3 > 0`, mixed bases, and `mode='auto'` — a pinned structure and a discovery heuristic are mutually exclusive. |
| `variable_orders` | `{j: [orders]}`, orders ⊆ `{1, 2}` | `None` | Order-selective membership for variable `j`. `[1, 2]` is the default (both). `[2]` admits `j` to **pair terms only** — its first-order block is excluded from the design (no df spent; the model keeps the uniform first-order layout with exact zeros for prediction/Sobol slicing). **This is non-hierarchical** — a pair without its marginal — and is flagged as such. `[1]` drops every pair *touching* `j`, keeping its first-order term. |
| `variance_variables` | `list[int]` | `None` (all `D`) | Restrict the first-order variance model (Stage D) to a named subset. Excluded variables are **variance-flat** with `Sʰ ≡ 0` **by assertion** (homoscedasticity-asserted); df is spent only on the subset, and `get_coefficients_for_variable` returns zeros for the rest. Composes with `K2h` (variance pairs are kept only inside the subset; an explicit variance pair reaching outside it is rejected) and `var_pair_selection='auto'`. |
| `K2h` | int | `0` | Documented public API: `> 0` fits second-order **variance** interactions, populating `log_variance_sobol['second_order']`. |
| `var_pair_selection` (explicit form) | `list[(i, j)]` | `None` | An explicit list pins the exact variance pairs (previously a list silently behaved as `'all'`). `None`/`'all'` keep all pairs; `'auto'` runs a quick variance fit then selects. |

```python
from hifi_anova.api import hifi_anova

# Per-pair second-order orders: keep only (0,1) and (2,3), at orders 4 and 2.
res = hifi_anova(X, y, K1=6, K2={(0, 1): 4, (2, 3): 2})

# Order-selective membership: x2 enters pairs only (no first-order block).
# NON-HIERARCHICAL — asserted; res.train_results['term_structure'] flags it.
res = hifi_anova(X, y, K1=6, K2=4, variable_orders={2: [2]})

# Variance model over a named subset; the rest are σ²(x)-flat BY ASSERTION.
res = hifi_anova(X, y, K1=6, heteroscedastic=True, mode='heteroscedastic',
                 variance_variables=[0, 2], K2h=2)
```

`summary()` surfaces the equation system (per-pair orders, any first-order
exclusions and their non-hierarchical caveat, the variance subset), and
`results['term_structure']` records the same machine-readably; the ragged pair
layout is mirrored into `meta.json` on `save_model` (the model itself
round-trips exactly via its pickle companion). Two capabilities remain
**expert-gated** and are *not* exposed here: a data-driven pair/order selector
(BR-02/BR-03) and a data-driven variance-variable selector — both would need a
calibrated criterion before shipping, which is exactly the honesty line these
asserted keys hold.

---

## 5. Complexity modes & stages

The model is built in up to four stages. `mode` maps to a stage list
(`training/mode.py`):

| `mode` | Stages | Meaning |
|--------|--------|---------|
| `'first'` | A | First-order only. |
| `'second'` | A, B | First + second-order interactions. |
| `'full'` | A, B, C | + residual (NN by default in this path). |
| `'heteroscedastic'` | A, B, D | + input-dependent variance (no NN residual). |
| `'auto'` | grows from A | Adds the next stage while structure remains. |

**Stage meanings:**

- **Stage A — First-order.** Ridge solve on `Φ₁`. Always run. Builds the
  first-order mean model and reports train/val RMSE.
- **Stage B — Second (and optional third) order.** Selects active variables,
  generates candidate pairs, solves jointly with `Φ₁`, optionally prunes, and —
  if `K3 > 0` — adds triples with their own selection/pruning.
- **Stage C — Residual.** Fits a residual on what the structured model missed. A
  **linear** residual (`rbf`/`rff`/`nystrom`) is projected orthogonal to the
  basis features and preserves the analytic framework; an **NN** residual
  (`type: 'nn'`, requires `enabled: True`) is trained by SGD.
- **Stage D — Heteroscedastic variance.** Alternating optimization: a Newton
  solve for the log-variance coefficients, then a variance-weighted ridge for the
  mean, iterated up to `max_outer_iter`.

### Auto mode

`mode='auto'` starts at Stage A and decides stage-by-stage
(`auto_decide_next_stage`). After A and B it computes the validation residual
fraction `1 − R²_val = RMSE²_val / Var(y_val)`; if that exceeds
`auto_threshold` (default `0.01`), it adds the next stage. Because the fraction
is measured from validation RMSE, the decision is independent of the chosen
`lambda`. For Stage D it instead checks whether the squared residuals correlate
with any input (`max |corr(r², xᵢ)| > 0.1`), i.e. whether there is
heteroscedastic structure to model.

```python
config = {'mode': 'auto', 'K1': 10, 'K2': 5, 'strategy': 'variance',
          'lambda_order1': 0.001, 'lambda_order2': 0.01, 'auto_threshold': 0.01}
```

> A richer, sieve-based alternative to the RMSE threshold is available via
> `hifi_anova.analysis.interaction_discovery.auto_decide_stages`, which decides
> using where the residual variance actually lives (see [§10](#10-diagnostic-suite)).

---

## 6. Regularization strategies

The ridge penalty is `wᵀ diag(r) w`, where the per-feature vector `r` is built by
`build_regularization_vector` from a named `strategy`. Six strategies are
implemented (`training/regularization.py`) — four fixed and two parameterized:

| `strategy` | Penalizes | Per-feature weight (Fourier) | Notes |
|------------|-----------|------------------------------|-------|
| `'uniform'` | raw magnitude | `r = λ` | Standard ridge; treats all coefficients equally. Reference, not recommended for Fourier. |
| `'variance'` *(default)* | variance contribution | `r = λ · G[j,j]` | "Equal-impact" penalty — `λ` controls the variance budget per feature (linear `λ/12`, harmonics `λ/2`). |
| `'smoothness'` | rate of change | `r = λ · (2πk)² / 2` | Integrated squared first derivative (Sobolev H1); the smoothing-spline penalty. |
| `'curvature'` | curvature | `r = λ · (2πk)⁴ / 2` | Integrated squared second derivative; leaves linear effects free (cubic-spline penalty). A tiny stability ridge is added to the linear/constant term. |
| `'sobolev[_s]'` | Sobolev H^s norm | `r = λ · (1 + (2πk)²)^s` | Smoothly interpolates between uniform (`s=0`), smoothness-like (`s=1`), and curvature-like (`s=2`). The `+1` naturally regularizes the linear term without an ad-hoc fix. |
| `'spectral[_a]'` | frequency `k` directly | `r = λ · k^a` | Direct frequency weighting: `a=0` ≈ uniform, `a=2` ≈ smoothness, `a=4` ≈ curvature. The linear term (frequency `k=0`, which would give `λ·0^a=0`) is held at the base penalty `λ` for stability. |

**Suffix syntax for the parameterized strategies.** The order/exponent is passed
as an underscore suffix and parsed via `strategy.split('_')`:

- `'sobolev'` → smoothness order `s` (default `s=1.0`). Set it explicitly with
  `'sobolev_1'`, `'sobolev_2'`, `'sobolev_0.5'`, etc.
- `'spectral'` → decay exponent `a` (default `a=2.0`). Set it explicitly with
  `'spectral_1.0'`, `'spectral_2'`, `'spectral_4'`, etc.

```python
config['strategy'] = 'sobolev_2'      # curvature-like Sobolev H² penalty
config['strategy'] = 'spectral_1.0'   # linear frequency decay
```

**Per-basis handling.** The weight formulas above are for Fourier. The four fixed
strategies use each basis's own Gram diagonal (`variance`) or derivative penalty
(`smoothness`/`curvature`) for Legendre and Haar. The parameterized strategies map
the frequency index to each basis:

- **Legendre** — degree `k`: `sobolev` uses `r = λ·(1 + k(k+1))^s`; `spectral`
  uses `r = λ·k^a`.
- **Haar** — scale `j` (frequency `2^{j-1}`): `sobolev` uses
  `r = λ·(1 + 4^{j-1})^s`; `spectral` uses `r = λ·(2^{j-1})^a`, applied to all
  wavelets at that scale.

For pairs and triples the per-block penalty combines the 1-D penalties additively
(`variance` uses the tensor-product Gram diagonal).

Pass a single string to apply one strategy everywhere, or a dict for per-order
control:

```python
config['strategy'] = {'order1': 'curvature', 'order2': 'smoothness',
                      'default': 'variance'}
```

### The attribution caveat — report your strategy

Different smoothness-penalty strategies can leave predictive accuracy
essentially unchanged while shifting individual Sobol attributions substantially
— a small-scale instance of the general non-uniqueness of variable importance
across near-equivalent models. **Always report the penalty strategy alongside the
indices.** The regularization path ([§10](#10-diagnostic-suite)) lets you see how
attributions move with `λ` at a fixed strategy.

---

## 7. Bases & effect signatures

### Choosing a basis

Set `basis_name` to `'fourier'` (default), `'legendre'`, or `'haar'`. Each yields
Hoeffding-orthogonal components; the difference is which effect shapes are cheap
to represent (see the table in [§1](#1-overview--concepts)).

```python
config = {'mode': 'second', 'K1': 8, 'K2': 4, 'basis_name': 'legendre',
          'strategy': 'variance'}
```

### Mixing bases per variable

Set `basis_per_variable` to assign each variable its own family and order. In
mixed mode, Legendre owns the linear term (includes P̃₁), Fourier is cos/sin only
(no linear), and Haar has no linear term — so inter-variable orthogonality is
preserved.

```python
config = {
    'basis_per_variable': {
        0: {'basis': 'legendre', 'K': 5},   # smooth
        1: {'basis': 'fourier',  'K': 8},   # oscillatory
        2: {'basis': 'haar',     'K': 4},   # localized / threshold
    },
    'strategy': 'variance', 'lambda_order1': 0.001,
}
# variables not listed fall back to basis_name / K1
```

`basis_per_variable='auto'` runs a Legendre-first cross-residual characterization
and produces a **heuristic** basis recommendation for each variable. It is not a
joint or globally optimal basis-selection procedure. A compound
`'legendre+haar'` recommendation currently resolves to its leading Legendre
family on the mixed trainer, so pass an explicit mapping when the exact family
assignment matters.

#### Mixed-basis capability matrix

The mixed per-variable path is deliberately narrower than the uniform path. It
fits **Stage A** (all first-order blocks) and, with Stage B, **all variable
pairs** — plus explicit/auto per-variable basis specs and block-correct Sobol
point estimates and CIs. Everything else **raises** rather than silently doing
something different from what you asked (DEC-045):

| Control | Neutral value | Mixed support |
|---|---|---|
| Stage A / A+B (all pairs) | — | ✅ supported |
| `basis_per_variable` (explicit / `'auto'`) | — | ✅ supported |
| `variable_selection` | `None` | ❌ raises (implicit `'bic'` default is neutralized with a one-release warning) |
| `pair_candidates` | `None` | ❌ raises |
| `pair_selection` | `None` | ❌ raises |
| `max_pair_variables` | `None` | ❌ raises |
| `pair_pruning` | `'none'` | ❌ raises |
| `first_order_pruning` | `'none'` | ❌ raises |
| Stage C / residual, Stage D / heteroscedastic, `K3>0` | — | ❌ raises |

`K2=0` disables pair interactions on the mixed path (a first-order / additive
model — the `P_1` estimand), exactly as on the uniform path: no pair features,
indices, result block, or CIs are produced. Use a **uniform** basis
(`basis_name=...`) to run selection, pruning, residuals, variance, or triples.
`result.train_results['mixed_capability']` records the stages that ran, the pair
behavior, and whether selection/pruning was applied.

This fixed mixed path remains useful for domain-informed assignments and
small-to-moderate variable sets where retaining all pairs is acceptable. General
mixed-basis variable selection and pruning are deferred follow-up work; the
uniform-basis path provides those controls today.

### Effect signatures (characterization)

`hifi_anova.analysis.basis_characterization` splits each variable's effect into
**polynomial**, **oscillatory**, and **localized** shares by fitting Legendre and
projecting the residual onto Fourier and Haar subspaces per variable.

```python
from hifi_anova.analysis.basis_characterization import (
    multi_basis_fit, cross_residual_characterization,
    sequential_projection_characterization, auto_select_basis,
    print_characterization_table,
)

# (a) which single basis fits best overall
comp = multi_basis_fit(data['x_train'], data['y_train'],
                       data['x_val'], data['y_val'])
print(comp['summary'])   # {'best_overall': ..., 'character': 'polynomial'|'oscillatory'|'localized'|'mixed'}

# (b) per-variable poly / oscillatory / localized shares (upper-bound decomposition)
char = cross_residual_characterization(data['x_train'], data['y_train'],
                                       data['x_val'], data['y_val'])
print_characterization_table(char)

# (c) exact (non-overlapping) decomposition via sequential orthogonal projection
char_exact = sequential_projection_characterization(
    data['x_train'], data['y_train'], data['x_val'], data['y_val'])

# (d) turn a characterization into a per-variable basis recommendation
rec = auto_select_basis(char)   # {'per_variable': {...}, 'summary': '...'}
```

`cross_residual_characterization` gives within-variable fractions that are upper
bounds (subspaces may overlap); `sequential_projection_characterization` gives an
exact additive decomposition (Legendre → Fourier ⊥ Legendre → Haar ⊥ both). Each
variable is labelled `polynomial`, `oscillatory`, `localized`, `mixed`, or
`negligible`.

---

## 8. Residual models

When structured orders leave signal on the table, Stage C adds a residual. Four
residual types are available.

| `type` | Class | Linear in params? | Sobol integrity |
|--------|-------|-------------------|-----------------|
| `'rbf'` | `RBFResidual` | yes | preserved (orthogonal projection) |
| `'rff'` | `RFFResidual` | yes | preserved |
| `'nystrom'` | `NystromResidual` | yes | preserved; also gives GP posterior variance |
| `'nn'` | `eqx.nn.MLP` | no (SGD) | needs re-decomposition |

**Linear residuals preserve the analytic framework.** RBF/RFF/Nyström features
are projected orthogonal to the basis features *before* fitting, so the basis
coefficients — and therefore the Sobol indices — are identical with or without
the residual. Use them as the default when you need clean sensitivity analysis.

```python
config['residual'] = {'type': 'rbf', 'n_centers': 300, 'sigma': 0.2,
                      'lambda_residual': 1.0}
# or 'rff':     {'type': 'rff', 'n_features': 1000, 'gamma': 3.0}
# or 'nystrom': {'type': 'nystrom', 'n_inducing': 300, 'kernel': 'matern52',
#                'lengthscale': 0.2}
```

**The NN residual is a last resort.** It is a plain SGD-trained MLP; its output is
*not* orthogonal to the basis, so the raw Sobol indices are no longer clean. Enable
it explicitly and re-decompose afterward:

```python
config['residual'] = {'type': 'nn', 'enabled': True,
                      'hidden_dims': [256, 256, 256], 'epochs': 200}
# after fitting, recover clean Sobol indices:
from hifi_anova.training.redecompose import redecompose
```

Trade-offs: RBF/Nyström place centers by k-means and are good for smooth,
broad interactions; smaller `sigma` / larger `gamma` captures more localized
structure at the risk of overfitting. RFF scales to more features cheaply.
Nyström additionally exposes `posterior_variance(...)` for GP-style epistemic
uncertainty.

---

## 9. Sensitivity & uncertainty

### 9.1 The dual Sobol spectrum

`compute_sobol_indices(model, x_data=None)` returns a nested dict:

```python
sobol = compute_sobol_indices(model, data['x_test'])

sobol['mean_sobol']['first_order']    # {i: S_i}          — TOTAL shares (see below)
sobol['mean_sobol']['second_order']   # {(i, j): S_ij}
sobol['mean_sobol']['third_order']    # {(i, j, k): S_ijk}
sobol['mean_sobol']['total_order']    # {i: total-order index for i}
sobol['mean_sobol']['residual']       # fraction attributed to the residual (= 1 − 𝔉)

sobol['variance_accounting']          # absolute variances per order + totals
```

**Core vs. total shares and structural fidelity 𝔉.** When a residual/NN stage runs,
two normalizations coexist and are reported *separately, always labeled* (v06 §3.2/§8):

```python
sobol['mean_sobol_core']['first_order']   # Ŝ^core = V_u / V_core  (within retained orders)
sobol['mean_sobol_total']['first_order']  # Ŝ^total = V_u / V_f    (of the whole function)
sobol['fidelity']['value']                # 𝔉 = V_core / (V_core + Var(ĝ))
sobol['fidelity']['orthogonality_defect'] # 2·Ĉov(f̂_core, ĝ) / Var(f̂)
```

- **`core`** divides by the retained structured orders only — the *within-model*
  attribution, invariant when a residual is attached.
- **`total`** divides by the whole function's variance — reduced by the fraction of
  signal the residual carries. The two are bridged by the **structural fidelity**
  `𝔉 = V_core/(V_core+Var(ĝ))`, with `Ŝ^total = 𝔉·Ŝ^core`. `1 − 𝔉` is the honest size
  of the interpretability gap.
- **`orthogonality_defect`** reports `2·Ĉov(f̂_core,ĝ)/Var(f̂)`: the `𝔉` identity is
  exact only when core and residual are orthogonal; the defect measures how far an
  *empirically* orthogonal residual departs from that (it is **not** folded into 𝔉).
- With **no residual stage** (the common case) `𝔉 ≡ 1`, `total ≡ core`, the defect is
  `0`, and the legacy `mean_sobol` fractions are unchanged (they equal the total).

> **Naming (DEC-034).** The `𝔉`-scaled quantity `Ŝ^total = V_u/V_f` is a *share of the
> **fitted** variance* — **not** the total-**effect** index `S_T` (first-order + all
> interactions involving a variable), which is `sobol['mean_sobol']['total_order']`.
> These are different objects; the printed summary calls the `𝔉`-scaled one the
> **"share of fitted variance"** so it names its own denominator. `𝔉` itself is
> *model-internal* ("internal R² of the core against the full fit") — distinct from the
> data-vs-fit `R²` and from the population estimand `ρ_k` it estimates.

On the one-call `HiFiResult`, the headline `result.sobol_ci` is the **core**
interpretable CI (invariant to the residual model); `result.sobol_ci_total` adds the
share-of-fitted-variance CIs (`= 𝔉·core`, conditional on the residual variance) — `None`
when no residual ran — and `result.fidelity` carries the `𝔉` object above.
`result.summary(headline="fitted_variance")` leads the shares table with the
fitted-variance column instead of core (presentation only; both are always shown, and a
residual row surfaces `1 − 𝔉`).

**Opt-in — share of observed output variance.** `result.summary(observed=True)` also
prints `result.sobol_ci_observed = V_u/Var(Y)` — a practitioner view that scales the
fitted-variance shares by `Var(f̂)/Var(Y)` (treated as fixed). It *deliberately confounds
attribution with fit quality*, so it is off by default, never co-tabulated with the
canonical shares, and its residual+noise tail is reported as a single lump (components +
lump `≈ 1`). Note `Var(f̂)/Var(Y)` equals `R²` only under OLS-with-intercept
orthogonality — for a regularized/nonlinear fit it differs (and can exceed 1), so the
library computes the scaling rather than advising "multiply by `R²`".

When the model is heteroscedastic, a parallel block appears:

```python
sobol['log_variance_sobol']['first_order']         # {i: log-variance S^h_i}
sobol['log_variance_sobol']['second_order']        # {(i, j): ...}
sobol['log_variance_sobol']['total_order']         # {i: ...}
sobol['log_variance_sobol']['variance_accounting'] # per-order h-variance totals
```

`first_order[i]` drives `E[y|x]`; `log_variance_sobol['first_order'][i]` is the
log-variance index `S^h_i`, a driver of multiplicative fitted residual scale.
`variance_sobol` remains a one-release deprecated read alias returning the exact
same block. It is not natural-scale variance attribution.

**Visualizing the dual spectrum.** `hifi_anova.analysis.visualization` provides
`plot_dual_sobol` (paired mean/log-variance bars) and
`plot_sensitivity_ellipses` —
a dual-sensitivity view where each variable is an ellipse whose *width* is its
mean sensitivity and *height* its log-variance index (`mode='glyph'`), or a
scatter at `(Sᶠ, Sʰ)` with CI ellipses (`mode='plane'`). A publication-styled
`plot_sensitivity_ellipses` returning `(fig, ax)` is also in
`hifi_anova.analysis.plots`.

**Structural vs. correlative indices.** The indices above are **structural**
(`wᵀ G w`, reference product measure, assuming independent inputs); they sum to 1.
When `x_data` is passed, `compute_sobol_indices` also returns
`sobol['correlative_sobol']`, the joint-law allocation
`S^corr_u = Cov(f̂_u, f̂_tot)/Var(f̂_tot)` over **all** retained structured
components (`first_order`, `second_order`, `third_order`; the orthogonal residual
is excluded — it is carried by the structural fidelity 𝔉). The complete
collection sums to 1 **identically** (`sum_of_correlative_indices`, by linearity
of covariance, regardless of dependence); individual shares may be negative or
exceed 1, and a first-order-only subset (`first_order_sum`) need **not** sum to 1
when interactions are retained.

**This block is a diagnostic, not an official estimand**
(`role = 'independence_assumption_diagnostic'`,
`official_correlated_estimand = False`). The reported attribution is the
*structural* spectrum, which describes the fitted function under the reference
**independent product measure**; independence is an assumption
(`sobol['input_assumption_verified']` is `False` unless you pass
`inputs_independent_by_design=True` for a controlled experiment). By default
`correlation_diagnostic(model, x)` is **descriptive**: it reports the
structural-vs-correlative divergence and the max ordinary Pearson correlation
(`max_abs_input_correlation`, descriptive only — *not* proof of independence and
blind to nonlinear dependence). An **experimental, opt-in** nonlinear independence
test (unbiased distance correlation + max-statistic permutation) runs only with
`run_independence_test=True` (or via `independence_test(x)`), adding
`dependence_level` and `nonlinear_significant`; it is off the core path. For
observational data independence must be justified externally, and principled
dependent-input attribution (Shapley effects / generalized hierarchically-
orthogonal ANOVA) is out of scope — see the manuscript outlook.

### 9.2 Heteroscedastic fit — worked example

```python
from hifi_anova.data.synthetic import generate_heteroscedastic

X, y, sigma_true = generate_heteroscedastic(n_samples=10000, noise_variable=2)
data = preprocess_data(X, y)

config = {'K1': 10, 'K2': 5, 'Kh': 3, 'strategy': 'variance',
          'lambda_order1': 0.001, 'lambda_order2': 0.01, 'lambda_h': 0.1,
          'stages': ['A', 'B', 'D'], 'max_outer_iter': 10}

model, results = HiFiANOVATrainer(config).fit(
    data['x_train'], data['y_train'], data['x_val'], data['y_val'])

sobol = compute_sobol_indices(model, data['x_test'])
print(sobol['mean_sobol']['first_order'])       # mean drivers
print(sobol['log_variance_sobol']['first_order'])  # log-scale driver S^h
```

### 9.3 Sobol-estimation mode (additivity-calibrated, experimental)

Sobol indices are quadratic in the coefficients, so GCV-optimal ridge biases them
slightly downward. `estimate_sobol` is a separate experimental mode that tunes
`lambda` so the fitted indices approach a requested additivity sum. That
calibration is heuristic; it is not a proof or guarantee of unbiased recovery.

```python
from hifi_anova.training.trainer import estimate_sobol

est = estimate_sobol(data['x_train'], data['y_train'],
                     K1=10, K2=5, strategy='curvature', auto_lambda=True)
print(est['additivity_sum'])         # ~1.0 at the requested calibration target
print(est['sobol_first_order'])      # {i: S_i}
print(est['sobol_total_order'])      # {i: total-order}
print(est['lambda_order1'])          # the λ found
```

Signature (`training/trainer.py`): `estimate_sobol(x, y, K1=10, K2=5, K3=0,
strategy='variance', lambda1=None, lambda2=None, lambda3=None, auto_lambda=True,
additivity_target=1.0, include_linear_1=True, include_linear_2=True,
include_linear_3=True, basis_name='fourier')`. Returns `sobol_first_order`,
`sobol_second_order`, `sobol_third_order`, `sobol_total_order`, the per-order
variances, `additivity_sum`, the `lambda_order*` used, and the coefficients.

### 9.4 Confidence intervals

`hifi_anova.analysis.automl.sobol_confidence_intervals` puts
heteroscedasticity-robust CIs on the indices via a leverage-corrected
**HC3 sandwich estimator**
for `Cov(w)` combined with the **delta method** for `S = wᵀGw / total`:

```python
from hifi_anova.analysis.automl import sobol_confidence_intervals

ci = sobol_confidence_intervals(Phi_train, y_centered, reg_diag, D, K1, G1,
                                K2=K2, P=P, G2=G2, pair_indices=pair_indices,
                                alpha=0.05)
for i, (S, lo, hi) in ci['first_order'].items():
    print(f"x{i+1}: S = {S:.3f} [{lo:.3f}, {hi:.3f}]")
```

The delta method uses the **full gradient** of `S_i = V_i / V_tot`, including
the denominator-coupling terms `∂S_i/∂w_j = −S_i·2G_j w_j / V_tot` for `j ≠ i`
(uncertainty in the other components propagates into `S_i` through the shared
total). With that full gradient, Monte-Carlo coverage is nominal (~0.94–0.96 at
95%) for all three bases (Fourier, Legendre, Haar); a deterministic coverage
regression test in `tests/test_automl.py` guards this. The full derivation — the
sandwich estimator, the delta method, the exact gradient, why the own-block-only
shortcut undercovers, and the coverage-validation results — is in
[`docs/CI_theory.md`](CI_theory.md).

The one-call API computes first-order (and available second-order) CIs for you
and exposes them as `result.sobol_ci`.

These are fixed-configuration intervals. They condition on the transform, basis,
admitted structure, penalties, and weights; when selection used the same data,
`result.inference_metadata` records that fact and post-selection coverage is not
claimed. Degenerate null and complete-share components are marked in
`result.sobol_ci_status` and suppressed from ordinary interval presentation.

### 9.5 Fitted-design diagnostics & the two-fit convention

All one-call diagnostics (`sigma_hat`, `df`, `loo_cv`, the Sobol CIs, and the
epistemic term of `predict_intervals`) are computed from the **design the trainer
actually solved** — a `FittedDesign` record surfaced on
`result.train_results['fitted_design']` — not a separate rebuild. This makes the
reported numbers describe the model you get back, with the real per-order
penalties and the third-order variance in the Sobol denominator.

For a **heteroscedastic (Stage-D)** fit this follows the theory's *two-fit
convention* (manuscript Theorem `projection` Part ii):

`result.sobol` is the structural spectrum of the shipped predictive fit (the
precision-weighted mean when Stage D is kept). `result.sobol_ci` is instead the
unit-weight interpretable headline attribution and its fixed-configuration
intervals. They coincide without heteroscedastic weighting but can differ under
Stage D; `result.sobol_ci_efficient` and `result.sobol_gap` expose that difference.

- **Prediction & diagnostics** use the precision-weighted (GLS) fit with
  `W = diag(1/σ²(xₙ))`: `df = tr S`, weighted PRESS/LOO, a weighted epistemic
  posterior `A_w = ΦᵀWΦ + R`, and `sigma_hat` reinterpreted as the **whitened
  calibration scale** `√(RSS_w/df_res)` — it is ≈ 1 when the variance model is
  calibrated, **not** a homoscedastic noise level. `result.noise_scale_is_calibration`
  is `True` in this case, and the input-dependent noise is available as
  `result.sigma_x2(X_new)` (returns `σ²(x)`).
- **Attribution** (the Sobol point indices and their HC3 CIs, in `result.sobol_ci`)
  uses the *unit-weight* (`W = I`) companion, so a reported component is a Hoeffding
  projection (an estimand), not a heteroscedasticity artifact. The HC3 sandwich is
  already heteroscedasticity-robust, so the attribution CIs are **not** reweighted.
  This is the **interpretable** fit and stays the headline attribution.

For a heteroscedastic fit the two fits differ, and the manuscript prescribes
reporting *both* plus their observed gap:

- `result.sobol_ci_efficient` — the **efficient** (precision-weighted) first-order
  Sobol CIs, computed on the same retained blocks with the weighted sandwich. This
  is the fit used for prediction; it converges to a *different* target than the
  interpretable fit under heteroscedasticity.
- `result.sobol_gap` — the observed **efficient − interpretable** gap per component
  (`{'first_order': {name: gap}, 'second_order': {(i, j): gap}}`), the
  "heteroscedasticity × misspecification" diagnostic (Theorem `projection` Part ii).
  `summary()` prints a gap row when it is non-negligible. A large gap flags that
  weighting and the unexplained residual are jointly steering attribution.

Mixed **per-variable bases** (`basis_per_variable=`) are supported by the one-call
API: the record carries each variable's (and pair's) own block layout and Gram, and
the Sobol CIs are computed block-driven — so `hifi_anova(..., basis_per_variable=…)`
returns block-correct CIs instead of raising.

On the homoscedastic path the two fits coincide: `sobol_ci_efficient` and
`sobol_gap` are `None` (no gap row), `noise_scale_is_calibration` is `False`,
`sigma_hat` is the usual noise estimate, and `sigma_x2` returns the neutral unit
variance (use `sigma_hat**2` as the constant noise level there).

### 9.6 Leave-one-out tiers (heteroscedastic fits)

For a **heteroscedastic** fit the variance model is estimated from all the data,
so the plug-in weighted PRESS is only *Tier I* of a three-tier leave-one-out
hierarchy (theory manuscript, App. C). The one-call reports the manuscript's
default **Tier II** — a one-step Newton jackknife that also corrects the *variance*
prediction on deletion — and adds a predictive **LOO-NLL** on a common scale:

```python
result.loo_cv     # squared-error LOO (Tier-II-weighted on a Stage-D fit)
result.loo_nll    # predictive leave-one-out NLL — populated on BOTH paths
result.loo_tier   # 1 (homoscedastic; tiers coincide) or 2 (Stage-D default)

result.loo(tier=1)  # plug-in (variance held at full-data values; old tables)
result.loo(tier=2)  # the reported one-step jackknife
result.loo(tier=3)  # exact nested refit — the oracle; O(N) joint refits (slow)
```

Prefer **`loo_nll`** for comparing models: the weighted `loo_cv` is measured in a
*model-dependent* metric (the weights come from that model's variance fit), so it
is not comparable across models; the LOO-NLL is. On the homoscedastic path the
three tiers coincide, `loo_tier` is 1, and `loo_nll` reduces to the free form
`½·log σ̂² + loo_cv/(2σ̂²) + ½·log 2π`.

Tier II's `O_p(N⁻²)` guarantee is *conditional*: it does not hold at an active
variance floor or with an ill-conditioned variance Hessian. The fit checks both
(a KKT floor test — not a bare value-at-clip check — plus an `H_h` conditioning
check) and exposes `result.loo_tier2_guarantee_holds`
(`/loo_variance_floor_active` / `variance_hessian_ill_conditioned`); when the
guarantee is at risk, `summary()` prints a warning and **Tier III**
(`result.loo(tier=3)`) is authoritative. This is a *predictive* diagnostic — for
selecting `λ_h` use the K-fold / LAML criteria (`training/joint_lambda.py`), not
this single leave-one-out sweep.

---

## 10. Diagnostic suite

At a fixed design, retained block set, penalty, and weight vector, the diagnostic
pipeline comes at modest extra cost: the quantities below reuse the finalized
ridge system and its hat matrix. Configuration search and Stage-D alternation are
separate and may require additional factorizations.

### 10.1 One-solve analytics — LOO, noise, GCV

```python
from hifi_anova.analysis.automl import ridge_analytics

a = ridge_analytics(Phi_train, y_centered, reg_diag)
a['sigma_hat']   # REML-style noise estimate  sqrt(RSS / (N - df))
a['loo_cv']      # exact leave-one-out CV from leverages
a['gcv']         # generalized CV
a['df']          # effective degrees of freedom = tr(H)
a['aic'], a['bic']
```

Related helpers in the same module: `kfold_cv_analytic` (exact K-fold via
Woodbury rank updates), `stability_diagnostics` (per-fold Sobol spread, labelled
`excellent`/`good`/`moderate`/`poor`), `noise_complexity_curve` (`σ²(λ)` whose
minimum estimates the true noise), and `sample_size_diagnostics`.

### 10.2 GCV hyperparameter selection

```python
from hifi_anova.training.hyperopt import optimize_multi_lambda, optimize_multi_lambda_extended

res = optimize_multi_lambda(Phi, y_centered, D=10, K1=10, K2=5, P=45,
                            strategy='curvature', method='gcv')
print(res['lambda_order1'], res['lambda_order2'], res['df'])

# jointly optimize up to (lambda1, lambda2, lambda3, lambda_residual):
res2 = optimize_multi_lambda_extended(Phi, y_centered, D=10, K1=10, K2=5, P=45,
                                      K3=1, T=5, M_residual=200,
                                      strategy='curvature', method='gcv')
```

`method` accepts `'gcv'` (recommended, especially when `F > N`), `'evidence'`,
`'aic'`, or `'bic'`. There is also `optimize_single_lambda` for a scalar `λ`.

### 10.3 Regularization path (Sobol at every λ)

`compute_reg_path` sweeps `λ₁` on a log grid (with `λ₂ = lambda_ratio·λ₁`, etc.)
and records Sobol indices, variance decomposition, effective df, GCV, and
evidence at each point — no retraining.

```python
from hifi_anova.analysis.reg_path import compute_reg_path, plot_reg_path

path = compute_reg_path(Phi, y_centered, D=10, K1=10, K2=5, P=45,
                        pair_indices=pair_indices, strategy='curvature',
                        n_lambdas=50, lambda_range=(1e-5, 10.0))
plot_reg_path(path, save_prefix='figures/my_analysis')
```

The returned `RegPathResult` carries `sobol_paths`, `sobol_paths_2nd/3rd`, the
per-order variance curves, `lambda_gcv_opt`, and `lambda_evidence_opt`.
`plot_reg_path` renders the L-curve, model-selection criteria, Sobol paths, and a
stacked variance decomposition; `plot_pareto_frontier` shows complexity vs.
unexplained variance.

For the **variance model**, `compute_variance_reg_path(model, x, y, ...)` is the
analogue: it holds the fitted mean fixed and sweeps the variance penalty `λ_h`
(refitting the log-variance model by Newton at each point), recording the
variance-Sobol spectrum and total explained log-variance; `plot_variance_reg_path`
renders it. Use the mean path (and its Pareto view) to pick `λ`, the variance
path to pick `λ_h`, then [`verify_model`](#106-model-verification--a-one-call-health-check)
to confirm the fitted model.

### 10.4 Interaction discovery (residual sieve)

`hifi_anova.analysis.interaction_discovery` projects the model residual onto
candidate subspaces to find missing structure — each test is a tiny solve.

```python
from hifi_anova.analysis.interaction_discovery import (
    unified_residual_sieve, scan_missing_pairs, scan_missing_triples,
    scan_missing_variance_pairs,
)

sieve = unified_residual_sieve(model, data['x_train'], data['y_train'])
print(sieve)   # first/second/third-order, smooth (RBF), and noise fractions + top pairs/triples
```

`scan_missing_pairs` / `scan_missing_triples` score every unselected pair/triple
by the residual variance it would capture (with a df correction against
false positives); `scan_missing_variance_pairs` does the same for the *noise*
residual. `iterative_pair_discovery` implements fit → scan → add → refit.

### 10.5 Calibration & variance accounting

Prediction intervals combine aleatoric variance (from the variance model) with
Fourier epistemic variance (`hifi_anova.model.predict.predict_intervals`), and
`sobol['variance_accounting']` /
`sobol['log_variance_sobol']['variance_accounting']`
give the absolute variance booked to each order for auditing how much of the
signal the model actually explains.

### 10.6 Model verification — a one-call health check

`hifi_anova.analysis.diagnostics.verify_model` runs the diagnostic workflow
end-to-end and returns a pass / warn / fail report, so you can confirm a fit is
trustworthy *before* reading Sobol indices off it. It checks Sobol additivity
(structural indices sum to ~1), index bounds (total-order ≥ first-order, all in
[0,1]), test R², prediction-interval calibration (for heteroscedastic models),
and input-correlation level; it also flags pure-interaction variables (zero
first-order but non-zero total-order).

```python
from hifi_anova.analysis.diagnostics import verify_model

report = verify_model(model, x_test, y_test, x_train=x_train,
                      feature_names=names)
# -> prints a [PASS]/[WARN]/[FAIL] table; report['all_pass'] is the summary bool.
```

**Recommended workflow.** Use the three regularization paths to *choose* and
*understand* the penalty — the mean path (§10.3, `compute_reg_path`) with its
Pareto view (`plot_pareto_frontier`, complexity vs unexplained variance) for the
mean ridge `λ`, and `compute_variance_reg_path` for the variance penalty `λ_h` —
then `verify_model` to confirm the *fitted* model at the chosen penalty is
internally consistent. `examples/run_ishigami_heteroscedastic.py` walks through
all four.

---

## 11. Working with results

The one-call API returns a `HiFiResult` (`hifi_anova/api.py`) that bundles the
fitted `model`, `config`, `feature_names`, the preprocessing `transformer`, the
full `sobol` dict, named CIs in `sobol_ci`, and diagnostics
(`sigma_hat`, `r_squared`, `loo_cv`, `df`).

### Methods

```python
result = hifi_anova(X, y, feature_names=[...])

# Predictions on new ORIGINAL-scale data (transform is applied internally)
y_hat = result.predict(X_new)                     # (M,)

# Prediction intervals (aleatoric + epistemic); alpha=0.05 -> 95%
lower, upper = result.predict_intervals(X_new, alpha=0.05)

# Learned first-order effect of one variable, in [0,1] quantile space
x_grid, f_values = result.component_curve('MedInc', n_points=200)  # name or index

# Human-readable summary (ranked Sobol w/ CIs, R², noise, df, top interactions)
result.summary()

# Train/val/test split provenance — original-dataset row indices per split, in
# the fitted design's row order. X[idx['train']] / y[idx['train']] are exactly
# the rows the model was fit on, so a per-point diagnostic (LOO residual,
# leverage, worst out-of-sample row) maps back to a dataset row id.
idx = result.split_indices          # {'train': ndarray, 'val': ..., 'test': ...}
X_train, y_train = X[idx['train']], y[idx['train']]

# Persist model + transformer + config + feature names + results
result.save('my_model/')
```

### Saving and loading

`result.save(path)` writes the model, the fitted transformer, the config, and the
feature names to a directory. To reload, use the model I/O helper:

```python
from hifi_anova.model.io import load_model

bundle = load_model('my_model/')
model        = bundle['model']         # HiFiANOVA
transformer  = bundle['transformer']   # QuantileTransformer
config       = bundle['config']
names        = bundle['feature_names']

# Predict on new original-scale data manually:
import numpy as np, jax.numpy as jnp
X_t = np.clip(transformer.transform(X_new), 0, 1)
y_hat = np.asarray(model.predict_mean_only(jnp.asarray(X_t, dtype=jnp.float32)))
```

> **Note.** Loading returns a bundle dict (model + transformer + config), not a
> `HiFiResult`. Re-attach preprocessing via the transformer as shown above when
> predicting on original-scale data.

---

## 12. Assumptions & caveats

- **Independent inputs / product measure (assumed, not verified).** Analytic
  (structural) Sobol indices assume a product input measure (uniform marginals
  after the quantile transform) and describe the fitted function under that
  reference measure; they sum to 1. This is an *independence* assumption
  (`sobol['input_assumption'] = 'independent_product_measure'`,
  `input_assumption_verified = False` by default), not merely zero correlation —
  uncorrelated-but-dependent inputs violate it too. For **controlled experiments**
  pass `inputs_independent_by_design=True` to record it; for **observational
  data** justify independence externally. `correlation_diagnostic` is descriptive
  by default (structural-vs-correlative divergence + ordinary Pearson, *not* a
  proof); an experimental nonlinear test is opt-in via `run_independence_test=True`
  / `independence_test`. Read `sobol['correlative_sobol']` as an optional
  assumption-sensitivity diagnostic (`official_correlated_estimand = False`), not
  an estimand: the complete collection sums to 1; individual shares may be negative
  or exceed 1; a first-order-only subset does not sum to 1 when interactions are
  retained. Principled dependent-input attribution (Shapley / generalized ANOVA)
  is out of scope.
- **Quantile-space effects.** "Linear" / "low-frequency" is defined in quantile
  space — a monotone, nonlinear reparameterization of the original features. A
  linear effect in quantile space is not linear in the raw feature.
- **Finite basis.** Only the fitted interaction orders and harmonics are
  represented; functions with irreducible high-order structure (e.g.
  `sin(π x₁ x₂)`) are approximated, not captured exactly.
- **Shrinkage bias.** Sobol indices are quadratic in the coefficients, so
  GCV-optimal ridge biases them slightly downward. Use the Sobol-estimation mode
  The experimental `estimate_sobol` mode ([§9.3](#9-sensitivity--uncertainty))
  tunes additivity but does not guarantee unbiased recovery.
- **Not primarily a top-tier predictor.** HiFi-ANOVA trades a little predictive
  accuracy for interpretability and analytic sensitivity analysis.
- **Penalty strategy affects attributions.** Different smoothness-penalty
  strategies can leave predictive accuracy essentially unchanged while shifting
  individual Sobol attributions substantially. **Report the penalty strategy
  alongside the indices** ([§6](#6-regularization-strategies)).
- **NN residual breaks clean Sobol.** The linear residuals preserve the analytic
  framework; the SGD-trained NN residual does not, and requires re-decomposition
  (`training.redecompose`) before its Sobol indices are meaningful.

---

*Documentation notice: Copyright (c) 2026 R. Sala. All rights reserved.
Draft, work in progress — not covered by the source-code license (PolyForm
Internal Use 1.0.0). Except for permissions arising under GitHub's Terms of
Service, applicable law, or separate written permission from the copyright
holder, no permission is granted to reproduce, distribute, modify, publish, or
create derivative works from this document. See LICENSING.md.*
