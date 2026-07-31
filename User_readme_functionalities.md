# HiFi-ANOVA — Functionalities Guide (for users)

*A capability-oriented tour of what HiFi-ANOVA can do and how to invoke each
feature. This is the "what's in the box" overview; for the exhaustive option
reference see [`docs/USER_GUIDE.md`](docs/USER_GUIDE.md), and for a visual worked
example see [`docs/ishigami_showcase.md`](docs/ishigami_showcase.md).*

---

## What HiFi-ANOVA is

HiFi-ANOVA (**H**oeffding **I**nteraction–**Fi**delity **ANOVA**) fits an
*interpretable* regression model: a functional-ANOVA (Hoeffding) decomposition of
both the conditional **mean** and the conditional **variance** of a response, over
orthogonal basis families (**Fourier, Legendre, Haar**). Because the fit is
**linear in the basis coefficients**, sensitivity indices, uncertainty, and
diagnostics all come analytically off a single ridge solve — the model is
interpretable *by design*, not explained post-hoc.

The one thing to remember: **you get a *pair* of Sobol sensitivity indices per
variable** — one for its effect on the expected outcome (mean) and one for its
effect on the predictive *uncertainty* (variance). The second surfaces "hidden"
variables that drive uncertainty while carrying no mean signal.

---

## Installation

Requires Python 3.10+. All linear algebra runs in float64 (JAX x64 is enabled
automatically by the one-call API).

```bash
pip install -e .
# optional extras:
pip install -e ".[salib]"   # SALib ground-truth comparison
pip install -e ".[dev]"     # pytest
```

---

## 1. Fitting a model

### The one-call API — the fast path

```python
from hifi_anova import hifi_anova

result = hifi_anova(X, y, feature_names=['age', 'income', ...])
result.summary()                       # ranked Sobol indices + diagnostics
```

`hifi_anova(X, y, ...)` handles preprocessing, fitting, Sobol analysis, confidence
intervals and diagnostics, returning one `HiFiResult`. Key knobs:

| Argument | What it controls |
|---|---|
| `K1`, `K2` | max harmonic / basis richness for first- and second-order terms |
| `mode` | complexity: `'first'`, `'second'` (default), `'full'`, `'heteroscedastic'`, `'auto'` |
| `strategy` | regularization shape: `'variance'` (default), `'smoothness'`, `'curvature'`, … |
| `variable_selection` | prune spurious main effects: `'bic'` (default), `'group_lasso'`, `'1se'`, `None` |
| `residual` | add a nonlinear residual model: `'rbf'`, `'rff'`, `'nystrom'`, `None` |
| `heteroscedastic` | also fit the variance model (dual Sobol spectrum) |

### The staged API — full control

```python
from hifi_anova import HiFiANOVATrainer, compute_sobol_indices
from hifi_anova.data.preprocessing import preprocess_data

data = preprocess_data(X, y, seed=0)
trainer = HiFiANOVATrainer(config={'K1': 6, 'K2': 3, 'stages': ['A', 'B']})
model, train_results = trainer.fit(data['x_train'], data['y_train'],
                                   data['x_val'], data['y_val'], key=...)
sobol = compute_sobol_indices(model, data['x_test'])
```

The trainer runs up to four **stages** — A (first-order mean) → B (interactions) →
C (nonlinear residual) → D (heteroscedastic variance) — selectable via `stages`.

---

## 2. Sensitivity analysis (Sobol indices)

- **First / second / third-order mean indices** — `compute_sobol_indices(model, x)`
  returns per-variable, per-pair, and per-triple variance shares, read analytically
  off the coefficients (no extra model evaluations).
- **The dual mean + variance spectrum** — with a heteroscedastic fit, the same call
  returns a *variance* Sobol spectrum alongside the mean one: which variables drive
  the outcome vs which drive the uncertainty.
- **Unbiased Sobol-estimation mode** — `estimate_sobol(...)` for bias-corrected
  index recovery.
- **Confidence intervals** — heteroscedasticity-robust CIs for every index
  (sandwich estimator + delta method); `result.sobol_ci` gives `{name: (S, lo, hi)}`.
- **Ground-truth agreement** — matches SALib to ~0.003 MAE on Friedman-1
  (`examples/run_salib_comparison.py`).

---

## 3. Prediction & uncertainty

```python
pred        = result.predict(X_new)                    # mean prediction
lo, hi      = result.predict_intervals(X_new, alpha=0.05)   # 95% intervals
curve       = result.component_curve('income')         # learned 1-D effect shape
```

Prediction intervals use the closed-form posterior from the single ridge
factorization; with a heteroscedastic fit they are input-dependent.

---

## 4. Model selection & regularization

- **Closed-form criteria in λ** — GCV, marginal likelihood (evidence), AIC, BIC are
  all closed-form for the ridge fit.
- **λ optimizers** (`hifi_anova.training.hyperopt`):
  `optimize_single_lambda`, `optimize_multi_lambda`, `optimize_multi_lambda_extended`
  select the mean penalties. Each takes a `grad=` mode:
  - `'numeric'` (default) — derivative-free;
  - `'analytic'` — exact closed-form criterion gradients (faster, fewer solves);
  - `'jax'` — the same gradients by autodiff (matches analytic to ~1e-16);
  - `'auto'` — analytic when the penalty is well-conditioned, else numeric.
- **Regularization path** — `compute_reg_path(...)` sweeps λ and reports the Sobol
  spectrum at *every* penalty level from a single eigendecomposition (`solver='auto'`).
- **Joint mean + variance λ selection** *(opt-in, new)* —
  `hifi_anova.training.joint_lambda.optimize_joint_lambda(...)` co-selects the mean
  λ and the variance penalty `λ_h` against one criterion: `'kfold_nll'` (default,
  k-fold held-out likelihood) or `'laml'` (joint Laplace marginal likelihood). It
  reports the criterion surface, both effective degrees of freedom, and
  boundary/degeneracy warnings. See §6.

---

## 5. Bases & effect signatures

- **Three basis families** — Fourier (oscillatory), Legendre (polynomial), Haar
  (localized), mixable **per variable** while preserving orthogonality.
- **Effect-signature characterization** — split each variable's effect into
  polynomial / oscillatory / localized shares (via cross-basis residual projection),
  and turn that into a per-variable basis recommendation
  (`hifi_anova.analysis.basis_characterization`).

---

## 6. Heteroscedastic (variance) modeling

```python
result = hifi_anova(X, y, heteroscedastic=True, Kh=3)   # adds Stage D
```

- Fits a log-variance functional-ANOVA model by an IRLS/Newton alternation
  (`training/trainer.py::_fit_heteroscedastic`, `training/newton.py`).
- **Variance regularization path** — `compute_variance_reg_path(...)` sweeps the
  variance penalty `λ_h` with the mean fixed, recording the variance-Sobol spectrum.
- **Joint λ selection** — `optimize_joint_lambda(...)` picks a principled `λ_h`
  jointly with the mean λ instead of using a fixed value (see §4).

---

## 7. Diagnostics & health checks

From the single ridge factorization, at negligible cost:

- **Effective df, residual noise (σ̂), R², leave-one-out CV, GCV** — `ridge_analytics(...)`.
- **k-fold CV** via Woodbury downdates (no refits).
- **Interaction discovery** — screen for active interactions by residual projection
  (`analysis/interaction_discovery.py`).
- **First-order pruning** — zero out spurious pure-interaction main effects
  (`variable_selection='bic' | '1se' | 'group_lasso'`).
- **One-call health check** — `verify_model(...)` prints a PASS/WARN/FAIL table
  (`analysis/diagnostics.py`).
- **Rich plotting** — `hifi_anova.analysis.plots` (Sobol bars, component curves,
  interaction heatmaps, regularization paths, variance spectra, diagnostics).

---

## 8. Saving, loading & baselines

```python
result.save('my_model/')
from hifi_anova.model.io import load_model
loaded = load_model('my_model/')
```

Baseline comparators (MLP, sklearn wrappers) live in `hifi_anova.baselines`.
Synthetic benchmarks (Friedman-1, Ishigami incl. heteroscedastic) are in
`hifi_anova.data.synthetic`.

---

## 9. Worked examples

| Script | Shows |
|---|---|
| `examples/run_demo.py` | end-to-end one-call fit |
| `examples/run_ishigami_heteroscedastic.py` | dual mean+variance Sobol spectrum + figures |
| `examples/run_salib_comparison.py` | Sobol indices vs SALib ground truth |
| `examples/run_sobol_vs_prediction.py` | interpretability-vs-accuracy trade-off |

---

## Key things to keep in mind

- **Quantile space.** Inputs are quantile-transformed to `[0,1]` internally; effect
  curves are reported in that space.
- **Report your strategy.** Attribution depends on the regularization `strategy`;
  state which one you used when reporting indices (see USER_GUIDE §6).
- **float64 throughout.** The one-call API enables JAX x64; low-level helpers expect
  float64 designs.
- **Variance selection is intrinsically noisy** — a single squared residual carries
  little information about `σ²(x)`, so the joint-λ selector defaults to k-fold and
  emits diagnostic warnings when the variance model may be overfitting.

For the full option reference, see [`docs/USER_GUIDE.md`](docs/USER_GUIDE.md).
