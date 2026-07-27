# HiFi-ANOVA User Guide

**Interpretable regression with analytic Sobol diagnostics for the mean *and* the variance.**

HiFi-ANOVA — *Hoeffding Interaction–Fidelity ANOVA* — fits a functional-ANOVA
(Hoeffding) decomposition of both the conditional **mean** and the conditional
**variance** of a response, realized over three interchangeable basis families
(**Fourier, Legendre, Haar**). Because the fit is linear in the basis
coefficients, sensitivity indices, uncertainty estimates, and structure-discovery
diagnostics are read directly off a single ridge solve — the model is
interpretable *by design*, not explained post-hoc.

This guide documents the library as implemented. Every option, default, and code
snippet below is drawn from the source in `hifi_anova/`. It targets Python 3.10+
and assumes the `jax` / `equinox` scientific stack described in the top-level
`README.md`.

> Positioning, honestly stated. Computing analytic Sobol indices from the
> coefficients of an orthogonal-basis regression is well established — it
> underlies variance decomposition in polynomial chaos expansions (PCE) and
> RS-HDMR. What is distinctive here is the combination: a **dual mean +
> variance** Sobol spectrum, **three basis families** with per-variable effect
> signatures, and a **one-solve diagnostic suite** (closed-form LOO, GCV,
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
"hidden" heteroscedastic driver) show up in the variance spectrum while being
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

All linear algebra runs in float64. The one-call `hifi_anova(...)` enables JAX
x64 automatically (on the first call); if you call lower-level functions
directly, set it yourself:

```python
import jax
jax.config.update("jax_enable_x64", True)
```

Top-level convenience imports:

```python
from hifi_anova import (
    hifi_anova,             # one-call API
    HiFiResult,             # its result object
    HiFiANOVATrainer,       # staged trainer
    estimate_sobol,         # unbiased Sobol-estimation mode
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
    strategy='variance',         # regularization strategy
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
  `['A','B','D']` (or `['A','B','C','D']` if a `residual` is also requested).
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
| `mode` | str | — | `'first'`, `'second'`, `'full'`, `'heteroscedastic'`, or `'auto'`. Resolved into `stages` by `resolve_mode`. |
| `stages` | list[str] | `['A','B','C','D']` | Explicit stage list, used if `mode` is absent. |
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
| `pair_selection` | str or None | `None` | Legacy one-shot variable-selection + candidate-generation. |
| `pair_threshold` | float | `0.01` | Sobol threshold for threshold-based variable activity. |
| `max_pair_variables` | int or None | `None` | Cap on number of active variables entering pair generation. |
| `triple_selection` | str | `'all_active'` | Triple candidate mode: `'all'`, `'all_active'`, `'two_active'`, `'one_active'`, or one of `'bic'/'group_lasso'/'1se'` (principled, then `'two_active'`). |
| `triple_pruning` | str | `'none'` | Post-fit triple pruning: `'bic'`, `'group_lasso'`, `'1se'`, or `'none'`. |
| `var_pair_selection` | str or None | `None` | Variance pair selection (Stage D): `None`/`'all'` = all pairs, `'auto'` = quick variance fit then select. |
| `var_triple_selection` | str or None | `None` | Variance triple selection; currently all triples. |

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
| `alternating_tol` | float | `1e-4` | Relative NLL tolerance for early stopping. |
| `newton_max_iter` | int | `10` | Inner Newton iterations for the log-variance fit. |

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

`basis_per_variable='auto'` runs cross-residual characterization first and picks
a basis per variable automatically.

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

sobol['mean_sobol']['first_order']    # {i: S_i}
sobol['mean_sobol']['second_order']   # {(i, j): S_ij}
sobol['mean_sobol']['third_order']    # {(i, j, k): S_ijk}
sobol['mean_sobol']['total_order']    # {i: total-order index for i}
sobol['mean_sobol']['residual']       # fraction attributed to the residual

sobol['variance_accounting']          # absolute variances per order + totals
```

When the model is heteroscedastic, a parallel block appears:

```python
sobol['variance_sobol']['first_order']         # {i: variance-Sobol S^h_i}
sobol['variance_sobol']['second_order']        # {(i, j): ...}
sobol['variance_sobol']['total_order']         # {i: ...}
sobol['variance_sobol']['variance_accounting'] # per-order variance totals
```

`first_order[i]` drives `E[y|x]`; `variance_sobol['first_order'][i]` drives
`Var[y|x]`.

**Structural vs. correlative indices.** The indices above are **structural**
(`wᵀ G w`, assuming independent inputs); they sum to 1. When `x_data` is passed,
`compute_sobol_indices` also returns `sobol['correlative_sobol']`, computed from
the empirical covariance of component outputs. Correlative indices account for
input correlations and **need not sum to 1**; the gap between them and the
structural indices, plus `correlation_level` (`'clean'`/`'mild'`/`'strong'`),
diagnoses how much correlation is distorting the attribution.

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
print(sobol['variance_sobol']['first_order'])   # variance drivers (var 2 dominates)
```

### 9.3 Sobol-estimation mode (unbiased recovery)

Sobol indices are quadratic in the coefficients, so GCV-optimal ridge biases them
slightly downward. `estimate_sobol` is a *separate* mode aimed at unbiased
recovery rather than prediction: with `auto_lambda=True` it finds the `λ` that
makes the indices sum to ~1 (the additivity criterion).

```python
from hifi_anova.training.trainer import estimate_sobol

est = estimate_sobol(data['x_train'], data['y_train'],
                     K1=10, K2=5, strategy='curvature', auto_lambda=True)
print(est['additivity_sum'])         # ~1.0 when unbiased
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
heteroscedasticity-robust CIs on the indices via a **sandwich estimator** (HC0)
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

---

## 10. Diagnostic suite

Because the fit is a single ridge solve, an entire diagnostic pipeline comes at
negligible extra cost — everything below derives from the one hat matrix.

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
`sobol['variance_accounting']` / `sobol['variance_sobol']['variance_accounting']`
give the absolute variance booked to each order for auditing how much of the
signal the model actually explains.

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

- **Independent inputs / product measure.** Analytic (structural) Sobol indices
  assume a product input measure (uniform marginals after the quantile
  transform) and sum to 1. Under correlated inputs the structural indices
  misattribute; use the separate **correlative** indices
  (`sobol['correlative_sobol']`, which do not sum to 1) as the honest fallback,
  and watch `correlation_level`.
- **Quantile-space effects.** "Linear" / "low-frequency" is defined in quantile
  space — a monotone, nonlinear reparameterization of the original features. A
  linear effect in quantile space is not linear in the raw feature.
- **Finite basis.** Only the fitted interaction orders and harmonics are
  represented; functions with irreducible high-order structure (e.g.
  `sin(π x₁ x₂)`) are approximated, not captured exactly.
- **Shrinkage bias.** Sobol indices are quadratic in the coefficients, so
  GCV-optimal ridge biases them slightly downward. Use the Sobol-estimation mode
  (`estimate_sobol`, [§9.3](#9-sensitivity--uncertainty)) for unbiased recovery.
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
