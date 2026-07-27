# Heteroscedastic-Ishigami benchmark

A small, **fixed** dataset for comparing how models fit a noisy, interaction-heavy
function — and, crucially, whether they recover the *right structure* rather than
just chasing the noise. Bring your favorite ML and compare.

## The data

`ishigami_hetero/` holds three committed CSVs (regenerate with
`python benchmarks/ishigami_hetero/make_dataset.py`):

| file | columns | rows | use |
|------|---------|------|-----|
| `train.csv` | `x1,x2,x3,y` | 2000 | fit your model |
| `test.csv` | `x1,x2,x3,y` | 5000 | predict these inputs |
| `test_truth.csv` | `x1,x2,x3,f_true,sigma_true` | 5000 | the noiseless mean + true noise std, for scoring |

Inputs are `x_i ~ U(-π, π)`. The response is the **Ishigami** function
```
f(x) = sin(x1) + 7·sin²(x2) + 0.1·x3⁴·sin(x1)
```
with **heteroscedastic** Gaussian noise whose std ramps `0.3 → 3.0` across `x3`.
Two things make it a good test:

- **x3 is pure interaction.** It has *zero* first-order effect on the mean (it
  acts only through the `x1·x3⁴` term), but a non-zero total-order effect. A
  method that reports a first-order importance for x3 is picking up noise.
- **x3 is a hidden variance driver.** It carries no mean signal yet controls the
  entire noise variance — invisible to ordinary feature importance.

Analytic ground-truth first-order Sobol indices: **x1 = 0.314, x2 = 0.442,
x3 = 0.000**; x3 total-order = 0.244.

## Scoring

```python
from benchmarks.run_benchmark import evaluate_predictions
# fit your model on train.csv, predict the test.csv rows (same order) -> y_pred
evaluate_predictions(y_pred, sigma_pred=None, name='my_model')
```

Metrics:

- **R² vs observed y** — prediction under noise. Its ceiling is the noise floor
  (~0.8 here), so it *cannot* reach 1; don't mistake that for a bad fit.
- **R² vs true f(x)** — the honest generalization score (only possible because the
  data is synthetic). A model that overfits the noise shows a large
  `trainR²(obs) − testR²(obs)` gap and a lower `testR²(true)`.
- **NLL / coverage / σ-corr** — if you predict a per-point std `sigma_pred`.
- **Sobol MAE** — first-order sensitivity error vs the analytic indices (the real
  point; most black-box models have no native indices).

## Reference results

`python benchmarks/run_benchmark.py --baselines`

| model | trainR²(obs) | testR²(obs) | testR²(true) | gap | Sobol MAE |
|-------|-------------:|------------:|-------------:|----:|----------:|
| **HiFi-ANOVA** | 0.788 | 0.766 | 0.952 | **0.023** | **0.022** |
| GradientBoosting | 0.837 | 0.761 | 0.940 | 0.076 | n/a |
| RandomForest | 0.964 | 0.741 | 0.909 | **0.223** | n/a |
| MLP (3×128) | 0.829 | 0.779 | **0.961** | 0.050 | n/a |

Reading it: RandomForest overfits the heteroscedastic noise hardest (train 0.96 →
test 0.74) and generalizes worst. HiFi-ANOVA has the **smallest overfitting gap**,
generalizes competitively, and is the **only** method that returns accurate Sobol
indices and a calibrated input-dependent variance out of the box — a well-tuned
MLP edges it on pure prediction, which is exactly the "trade a little accuracy for
interpretability" positioning. Beating `testR²(true)` is one goal; beating
**Sobol MAE** while staying calibrated is the harder, more interesting one.

*(Numbers are reproducible from the committed CSVs; sklearn baselines need the
optional `scikit-learn` dependency.)*
