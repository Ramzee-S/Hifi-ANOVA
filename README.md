# HiFi-ANOVA

**Interpretable Regression with Analytic Sobol Diagnostics for Mean and Variance**

*HiFi-ANOVA — Hoeffding Interaction and Fidelity Identification ANOVA.*

A framework for **interpretable regression** surrogate ML models. 
The framework has functionality that enables decomposition of both the conditional
**mean** and the conditional **variance** of a response by interaction order,
variable, and frequency content, using basis functions that satisfy the
Hoeffding (ANOVA) side conditions. Sensitivity indices are read directly off the
fitted coefficients. This should enable the creation of models that are interpretable *by design*, not explained
post-hoc. Fourier, Legendre, and Haar bases can be used for first-order and second-order terms.

📖 **Full option and API reference:** [`docs/USER_GUIDE.md`](docs/USER_GUIDE.md).

🖼️ **Worked example with figures:** [`docs/ishigami_showcase.md`](docs/ishigami_showcase.md)
— a visual tour of the toolbox on a heteroscedastic Ishigami fit (dual mean+variance
Sobol spectrum, regularization trade-offs, fit-under-noise diagnostics). A
standalone [HTML version](docs/ishigami_showcase.html) is also included.

> **Status: preliminary, work-in-progress release.** The library is usable and
> tested (~420 tests), but the API may change before a 1.0. Extracted and
> sanitized from a research codebase; the manuscript / theory write-up and the
> interactive GUI are not included here and may be added later.
>
> **Source-available (PolyForm Internal Use 1.0.0), not open-source** — use is
> limited to your and your company's internal business operations; no
> distribution to third parties. See [License](#license).

---

## Why this exists

Computing analytic Sobol sensitivity indices from the coefficients of an
orthogonal-basis regression is well established — it underlies variance
decomposition in polynomial chaos expansions (PCE) and RS-HDMR. HiFi-ANOVA
extends and integrates that idea in three directions, so that sensitivity
analysis, uncertainty quantification, and structure discovery come at negligible
cost beyond a single ridge fit:

- **A dual mean + variance Sobol spectrum.** By jointly modelling the mean and
  the log-variance with the same basis machinery, every variable gets a *pair* of
  sensitivity indices — one for its effect on the expected outcome, one for its
  effect on the predictive *uncertainty*. This surfaces "hidden" variables that
  shape uncertainty while carrying no mean signal, and that are invisible to
  standard feature-importance analysis.
- **Three basis families with per-variable effect signatures.** Fourier,
  Legendre, and Haar bases (Gram matrices ranging from near-diagonal to identity)
  can be mixed per variable while preserving inter-variable orthogonality.
  Cross-residual projection between the families splits each variable's effect
  into **polynomial**, **oscillatory**, and **localized** shares, driving
  data-adaptive basis selection.
- **A one-solve diagnostic suite.** From the single matrix factorization of the
  ridge fit come closed-form leave-one-out CV, residual noise estimation,
  heteroscedasticity-robust confidence intervals for the Sobol indices (sandwich
  estimator + delta method; theory and coverage validation in
  [`docs/CI_theory.md`](docs/CI_theory.md)), K-fold CV via Woodbury downdates,
  interaction screening by residual projection, and a full regularization path
  with sensitivity indices at every penalty level.

Everything is read off analytically, with no extra model evaluations. On
Friedman-1, first-order indices match SALib ground truth within ~0.003 mean
absolute error.

---

## Installation

Requires **Python 3.10+**.

```bash
# 1. Create an environment (conda recommended for the scientific stack)
conda env create -f environment.yml
conda activate hifi_anova

# 2. Install the package (editable)
pip install -e .

# Optional extras:
#   pip install -e ".[salib]"   # for the SALib ground-truth comparison example
#   pip install -e ".[dev]"     # for pytest
```

Or with a plain virtualenv + pip:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

For **GPU** support, install JAX per the
[official guide](https://docs.jax.dev/en/latest/installation.html) instead of
the CPU wheel pulled in by default.

All linear algebra runs in float64. The one-call `hifi_anova(...)` enables JAX x64
automatically (on the first call); if you call lower-level functions directly, set
it yourself:

```python
import jax
jax.config.update("jax_enable_x64", True)
```

---

## Quick start

### One-call API (simplest)

```python
from hifi_anova.api import hifi_anova

result = hifi_anova(X, y, feature_names=['MedInc', 'HouseAge', 'AveRooms'])

result.summary()                          # Sobol indices w/ CIs, R², noise, df
pred = result.predict(X_new)              # predictions on new data (original scale)
lo, hi = result.predict_intervals(X_new)  # 95% prediction intervals
x_grid, f = result.component_curve('MedInc')  # learned effect of one variable
result.save('my_model/')                  # persist model + transformer + config
```

The one-call API handles preprocessing, fitting, Sobol analysis, confidence
intervals, and noise estimation automatically.

### Staged API (more control)

```python
import jax
jax.config.update('jax_enable_x64', True)

from hifi_anova.data.synthetic import generate_friedman1
from hifi_anova.data.preprocessing import preprocess_data
from hifi_anova.training.trainer import HiFiANOVATrainer
from hifi_anova.analysis.sobol import compute_sobol_indices

X, y = generate_friedman1(n_samples=10000, noise_std=1.0, n_irrelevant=5)
data = preprocess_data(X, y)

config = {'mode': 'second', 'K1': 10, 'K2': 5, 'strategy': 'variance',
          'lambda_order1': 0.001, 'lambda_order2': 0.01}

trainer = HiFiANOVATrainer(config)
model, results = trainer.fit(data['x_train'], data['y_train'],
                             data['x_val'], data['y_val'])

sobol = compute_sobol_indices(model, data['x_test'])
for i in range(10):
    print(f"x{i+1}: S1={sobol['mean_sobol']['first_order'][i]:.4f}")
```

### Runnable examples

```bash
python examples/run_demo.py                     # 3 experiments + figures
python examples/run_ishigami_heteroscedastic.py # dual mean+variance showcase (see below)
python examples/run_sobol_vs_prediction.py      # prediction vs Sobol-estimation modes
python examples/run_salib_comparison.py         # SALib ground-truth check (needs [salib])
```

`run_ishigami_heteroscedastic.py` is the fullest tour: analytic Sobol recovery on
a heteroscedastic Ishigami, the dual mean+variance spectrum with **ellipse**
visualizations, `first_order_pruning` zeroing x3's spurious main effect, the
regularization paths (mean, Pareto, and variance-`λ_h`), and a `verify_model`
health check.

---

## Model complexity modes

Choose complexity with a single `mode` parameter, or let `auto` decide:

| Mode | Stages | Description |
|------|--------|-------------|
| `'first'` | A | First-order Fourier only |
| `'second'` | A, B | First + second-order interactions |
| `'full'` | A, B, C | + residual network / linear residual |
| `'heteroscedastic'` | A, B, D | + input-dependent variance |
| `'auto'` | per-stage | Adds stages while residual fraction > threshold |

```python
config = {'mode': 'auto', 'K1': 10, 'K2': 5, 'strategy': 'variance',
          'lambda_order1': 0.001, 'lambda_order2': 0.01, 'auto_threshold': 0.01}
```

---

## Regularization strategies

The `strategy` key controls *how* the ridge penalty is distributed across the
basis coefficients (the penalty is `wᵀ diag(r) w`, one entry `r[j]` per
coefficient). It is an axis of control that is largely independent of the
overall strength `λ` — it decides *which* coefficients are shrunk hardest.
Six strategies are implemented (the formulas below are for the Fourier basis at
harmonic `k`; each adapts automatically to the Legendre and Haar bases):

| `strategy` | Penalty `r[j]` | Effect |
|---|---|---|
| `'uniform'` | `λ` | Plain ridge — every coefficient penalized equally (reference; not ideal for Fourier). |
| `'variance'` *(default)* | `λ · G[j,j]` | **Equal-impact**: each coefficient penalized by the variance it contributes, so `λ` is a direct "variance budget" per feature. |
| `'smoothness'` | `λ · (2πk)² / 2` | Integrated squared **first** derivative (Sobolev H¹) — penalizes rate of change (smoothing-spline style). |
| `'curvature'` | `λ · (2πk)⁴ / 2` | Integrated squared **second** derivative — penalizes curvature, leaves linear effects free (cubic-spline penalty). |
| `'sobolev_s'` | `λ · (1 + (2πk)²)ˢ` | Sobolev **Hˢ** norm; a single knob that interpolates `uniform` (s=0) → `smoothness` (s≈1) → `curvature` (s≈2). Default `s=1`. E.g. `'sobolev_2'`. |
| `'spectral_a'` | `λ · kᵃ` | Direct frequency weighting; `a=0` ≈ uniform, `a≈2` ≈ smoothness, `a≈4` ≈ curvature. Default `a=2`. E.g. `'spectral_1.0'`. |

The last two are *parameterized*: the suffix after the underscore sets the
exponent (`'sobolev_1.5'`, `'spectral_3'`). `sobolev`'s `+1` term regularizes the
linear coefficient cleanly; `variance` and `sobolev` are the most principled
defaults for interpretable sensitivity work.

> **Report the penalty alongside the indices.** Different smoothness strategies
> can leave predictive accuracy essentially unchanged while shifting individual
> Sobol attributions substantially — a small-scale instance of the general
> non-uniqueness of variable importance across near-equivalent models. See
> [`docs/USER_GUIDE.md`](docs/USER_GUIDE.md) §6 for the full treatment, including
> the per-basis (Legendre / Haar) forms.

---

## Key capabilities

### Sobol estimation mode (unbiased sensitivity recovery)

```python
from hifi_anova.training.trainer import estimate_sobol

# Finds the lambda that makes Sobol indices sum to ~1 (additivity criterion):
# less shrinkage -> more accurate sensitivity recovery.
sobol_est = estimate_sobol(data['x_train'], data['y_train'],
                           K1=10, K2=5, strategy='curvature', auto_lambda=True)
print(sobol_est['additivity_sum'])   # should be ~1.0
```

### Analytic hyperparameter selection (GCV)

```python
from hifi_anova.training.hyperopt import optimize_multi_lambda

result = optimize_multi_lambda(Phi, y_centered, D=10, K1=10, K2=5, P=45,
                               strategy='curvature', method='gcv')
print(result['lambda_order1'], result['lambda_order2'], result['df'])
```

### Regularization path (Sobol at every λ, no retraining)

```python
from hifi_anova.analysis.reg_path import compute_reg_path, plot_reg_path

path = compute_reg_path(Phi, y_centered, D=10, K1=10, K2=5, P=45,
                        pair_indices=pair_indices, strategy='curvature',
                        n_lambdas=50, lambda_range=(1e-5, 10.0))
plot_reg_path(path, save_prefix='figures/my_analysis')
```

### Heteroscedastic model — dual Sobol spectrum

```python
config = {'K1': 10, 'K2': 5, 'Kh': 3, 'strategy': 'variance',
          'lambda_order1': 0.001, 'lambda_order2': 0.01, 'lambda_h': 0.1,
          'stages': ['A', 'B', 'D'], 'max_outer_iter': 10}

model, results = HiFiANOVATrainer(config).fit(...)
sobol = compute_sobol_indices(model, data['x_test'])
print(sobol['mean_sobol']['first_order'])       # drives E[y|x]
print(sobol['variance_sobol']['first_order'])   # drives Var[y|x]
```

### Linear residuals (RBF / RFF / Nyström) — preserve the analytic framework

```python
config['residual'] = {'type': 'rbf', 'n_centers': 300, 'sigma': 0.2,
                      'lambda_residual': 1.0}
```

Linear residuals are projected orthogonal to the Fourier features at the feature
level, so Fourier coefficients and Sobol indices are identical with or without
the residual. An SGD-trained NN residual is also available (`type: 'nn'`) as a
last resort; use `hifi_anova.training.redecompose.redecompose` to recover clean Sobol
indices afterward.

### Removing spurious main effects — `first_order_pruning`

```python
config['first_order_pruning'] = 'bic'   # 'bic' | 'group_lasso' | '1se' | 'none'
```

Zeros the entire first-order block of any variable whose marginal effect the
criterion rejects — the group-sparse step plain ridge cannot do. This cleanly
removes a *pure-interaction* variable's spurious main effect (e.g. Ishigami x3),
robustly down to N≈100, without disturbing the interactions (they are
Hoeffding-orthogonal). See [USER_GUIDE §4.6](docs/USER_GUIDE.md).

### Model verification — one-call health check

```python
from hifi_anova.analysis.diagnostics import verify_model
report = verify_model(model, x_test, y_test, x_train=x_train)  # report['all_pass']
```

Runs the diagnostic workflow end-to-end (Sobol additivity, index bounds, R²,
calibration coverage, input-correlation level) and returns a pass/warn/fail
report, confirming a fit is internally consistent before its Sobol indices are
trusted. See [USER_GUIDE §10.6](docs/USER_GUIDE.md).

See the docstrings in `hifi_anova/api.py`, `hifi_anova/training/trainer.py`, and
`hifi_anova/analysis/sobol.py` for the full option set.

---

## Benchmark — bring your own model

`benchmarks/` ships a fixed heteroscedastic-Ishigami dataset (`train.csv` /
`test.csv` / `test_truth.csv`) and a comparison harness. Fit anything on the
train split, predict the test inputs, and score it:

```python
from benchmarks.run_benchmark import evaluate_predictions
evaluate_predictions(y_pred, sigma_pred=None, name='my_model')
```

It reports R² vs observed **and** vs the noiseless truth (honest generalization),
NLL/coverage, and first-order Sobol MAE vs the analytic indices.
`python benchmarks/run_benchmark.py --baselines` compares HiFi-ANOVA to sklearn
baselines — the tree ensembles overfit the noise (large train→test gap) while
HiFi-ANOVA generalizes competitively and is the only entrant with native,
accurate Sobol indices and a calibrated input-dependent variance. See
[`benchmarks/README.md`](benchmarks/README.md).

---

## Testing

```bash
python -m pytest tests/ -m smoke   # ~1 min   core math only        (88 tests)
python -m pytest tests/            # ~3 min   + fitting & analysis   (268 tests)
python -m pytest tests/ --full     # ~10 min  + integration pipelines (~370 tests)
python -m pytest tests/ --all      # ~12 min  everything             (~420 tests)
```

The test suite includes closed-form ground-truth checks: many test functions
have analytically derived Sobol indices, variances, and coefficients (see
`hifi_anova/data/test_functions.py` and `hifi_anova/data/nn_test_functions.py`).

---

## Package layout

```
hifi_anova/
├── core/        # Fourier/Legendre/Haar features, Gram matrices, pairs, projection
├── model/       # mean model, variance model, residual nets, linear residuals, I/O
├── training/    # staged trainer, ridge/Newton solvers, GCV, selection, redecompose
├── analysis/    # Sobol indices, AutoML analytics, reg-path, interaction discovery, plots
├── data/        # synthetic generators, ground-truth test functions, preprocessing
├── baselines/   # MLP / sklearn baseline wrappers
└── api.py       # one-call hifi_anova()
```

---

## Assumptions & caveats

- **Independent inputs.** Analytic Sobol indices assume a product input measure
  (uniform marginals after the quantile transform). Under correlated inputs the
  structural indices misattribute; use the separate **correlative** indices
  (which do not sum to 1) as the honest fallback.
- **Quantile-space effects.** "Linear" / "low-frequency" is defined in quantile
  space, a nonlinear reparameterization of the original features.
- **Finite basis.** Only the fitted interaction orders are represented; functions
  with irreducible high-order structure (e.g. `sin(π x₁ x₂)`) are approximated.
- **Shrinkage bias.** Sobol indices are quadratic in the coefficients, so
  GCV-optimal ridge biases them slightly downward; use the Sobol-estimation mode
  for unbiased recovery.
- **This is not primarily a top-tier predictor** — it trades a little predictive
  accuracy for interpretability and analytic sensitivity analysis.
- **Penalty strategy affects attributions.** Different smoothness-penalty
  strategies can leave predictive accuracy essentially unchanged while shifting
  individual Sobol attributions substantially — a small-scale instance of the
  general non-uniqueness of variable importance across near-equivalent models.
  Report the penalty strategy alongside the indices.

## Relationship to prior work

The analytic-Sobol machinery is closely related to **polynomial chaos expansion
(PCE)** sensitivity analysis and **RS-HDMR**, and builds on classical functional
ANOVA (Hoeffding, Sobol) and smoothing-spline ANOVA. The distinctive
contributions here are the multi-basis (Fourier/Legendre/Haar) hierarchy with
per-variable effect characterization, the dual mean+variance Sobol spectrum, and
the fully analytic one-solve diagnostic pipeline.


## License

**Source-available, not open-source.** The **source code** is licensed under the
[PolyForm Internal Use License 1.0.0](https://polyformproject.org/licenses/internal-use/1.0.0)
— see [LICENSE](LICENSE). You may download and use the software; you may make changes and new
works based on it, **only for the internal business operations of you and your
company**. However, **Distribution to third parties is not permitted.** All other rights
are reserved. More permissive license terms are planned when the project matures. 

Copyright 2026 R. Sala (libre-labs.org). For any use beyond the license terms
(e.g. distribution or providing it to third parties), Contact the licensor.

See [LICENSING.md](LICENSING.md) for the full overview covering source code,
documentation, and third-party material.

## Documentation notice

The documentation — the Markdown files in this repository (including this
README) — is **not** covered by the source-code license and is governed
separately:

Copyright (c) 2026 R. Sala. All rights reserved.

This is a draft, work in progress. Except for permissions arising under GitHub's
Terms of Service, applicable law, or separate written permission from the
copyright holder, no permission is granted to reproduce, distribute, modify,
publish, or create derivative works from this document.
