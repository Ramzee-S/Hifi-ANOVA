# Confidence Intervals for Sobol Indices — Theory & Implementation

**How HiFi-ANOVA puts calibrated confidence intervals on its analytic Sobol
indices, and the full-gradient delta method that makes them cover.**

This note documents the theory behind
`hifi_anova.analysis.automl.sobol_confidence_intervals`: the HC0 sandwich
estimator for the ridge coefficients, the delta method that turns coefficient
uncertainty into an interval on each Sobol index, the *full-gradient* formula the
library uses, and the Monte-Carlo evidence that the intervals are calibrated for
all three bases. It is written to stand on its own — you do not need to read the
source to follow it, but every symbol maps to a variable in the code.

---

## 1. Setup and notation

HiFi-ANOVA fits a model that is **linear in its coefficients**. After the target
is centered (`f0 = mean(y)` removed), the mean model is

```
y_c = Φ w + ε
```

where `Φ ∈ ℝ^{N×F}` is the basis design matrix (first-order blocks, then pair
blocks, then triple blocks, then any residual-feature columns), `w ∈ ℝ^F` are the
coefficients, and `ε` is noise. The penalized (ridge) solution is

```
ŵ = A⁻¹ Φᵀ y_c ,     A = ΦᵀΦ + R ,     R = diag(reg_diag)
```

with residuals `e = y_c − Φ ŵ`. (`A`, `A⁻¹ = A_inv`, and `e = residuals` are
returned by `ridge_analytics`.)

**Components and Sobol indices.** The columns of `Φ` are partitioned into
Hoeffding components — one block per variable `i` (first order), per pair `(i,j)`,
per triple `(i,j,k)`. Write `w_c` for the sub-vector of `w` on component `c`'s
block and `G_c` for that block's Gram matrix (`G_c[a,b] = ∫ φ_a φ_b` under the
product measure; `G1` for first order, the tensor products `G2`, `G3` for pairs
and triples). Because the basis satisfies the Hoeffding side conditions, each
component's variance under a product input measure is **exactly**

```
V_c = w_cᵀ G_c w_c ≥ 0 ,
```

the total explained variance is `V_tot = Σ_c V_c`, and the **Sobol index** of
component `c` is the fraction

```
S_c = V_c / V_tot .
```

No Monte-Carlo, no extra model evaluations — the index is a closed-form function
of the fitted coefficients. The rest of this note is about the *uncertainty* of
that function.

---

## 2. Uncertainty of the coefficients — the HC0 sandwich

`ŵ` is a random variable: a different noise draw `ε` gives a different fit. Its
sampling covariance is estimated with the **heteroscedasticity-robust (HC0)
sandwich estimator** (`sandwich_covariance`):

```
Cov(ŵ) ≈ A⁻¹ [ Σₙ eₙ² φₙ φₙᵀ ] A⁻¹  =  A⁻¹ Φᵀ diag(e²) Φ A⁻¹ .
```

The structure is *bread · meat · bread*: the "bread" `A⁻¹` is the ridge inverse,
and the "meat" `Φᵀ diag(e²) Φ` weights each observation by its own squared
residual. This is White's sandwich, specialized to the ridge normal equations.
Two things are worth stating plainly:

- **It is robust to heteroscedasticity.** Unlike the textbook `σ̂² A⁻¹`, the
  sandwich does not assume constant noise variance; it reads the per-point noise
  off the residuals. This matters because HiFi-ANOVA is often applied precisely
  when the noise is input-dependent.
- **It is frequentist and conditional on `λ`.** `Cov(ŵ)` describes the
  sampling spread of `ŵ` *at the chosen penalty*. It does **not** describe ridge
  shrinkage bias — see [§7](#7-what-the-interval-does-and-does-not-cover).

---

## 3. From coefficient uncertainty to an interval — the delta method

`S_c = V_c(w)/V_tot(w)` is a smooth (rational-quadratic) function of `w`. The
**delta method** propagates the coefficient covariance through a first-order
Taylor expansion:

```
Var(S_c) ≈ (∇S_c)ᵀ Cov(ŵ) (∇S_c) ,
```

and the `100(1−α)%` interval is

```
S_c  ±  z_{1−α/2} · SE(S_c) ,     SE(S_c) = √Var(S_c) ,
```

clipped to `[0, 1]` (a Sobol index is a fraction). `z_{1−α/2}` is the standard
normal quantile (`1.96` for 95%). Everything now hinges on getting `∇S_c` right.

### 3.1 The gradient — all components, not just one

Define the **component score** `U_c ≡ ∂V_c/∂w`: a *full-length* `F`-vector equal
to `2 G_c w_c` on component `c`'s block and zero everywhere else. Summing over
components gives the **total score**

```
U ≡ ∂V_tot/∂w = Σ_c U_c
```

(in code: the stacked vector `U`, filled block-by-block with `2 G_c w_c` for every
first-, second-, and third-order block; residual-feature columns stay zero because
they carry no Sobol variance). By the quotient rule,

```
        ∂   V_c        1              V_c
∇S_c = ──── ─────  = ────── U_c  −  ────── U  =  (U_c − S_c U) / V_tot .
        ∂w  V_tot     V_tot          V_tot²
```

**This is the crux.** `S_c` depends on `w` through *two* channels: its own
numerator `V_c` (the `U_c` term) **and** the shared denominator `V_tot` (the
`S_c U` term, which involves **every other component's coefficients**). Perturbing
`w_j` for `j ≠ c` changes `V_tot`, hence changes `S_c`. A correct standard error
must include that coupling.

### 3.2 The variance and its efficient evaluation

Substituting the gradient:

```
                1
Var(S_c) = ────────── (U_c − S_c U)ᵀ Cov (U_c − S_c U)
             V_tot²

                1
         = ────────── [ U_cᵀ Cov U_c  −  2 S_c · U_cᵀ Cov U  +  S_c² · Uᵀ Cov U ] .
             V_tot²
```

Because `U_c` is nonzero only on block `c`, this needs no per-component `F×F`
work. Precompute once:

```
CovU = Cov · U        (an F-vector)
UCU  = Uᵀ · CovU      (a scalar, shared by all components)
```

Then, with `own_c = 2 G_c w_c` the block-`c` score:

```
U_cᵀ Cov U_c = own_cᵀ · Cov[block_c, block_c] · own_c     (own-block covariance)
U_cᵀ Cov U   = own_cᵀ · CovU[block_c]                      (coupling to the rest)
Uᵀ Cov U     = UCU                                         (precomputed scalar)
```

so

```
                own_cᵀ Cov_cc own_c  −  2 S_c · own_cᵀ CovU_c  +  S_c² · UCU
Var(S_c)  =  ───────────────────────────────────────────────────────────────── .
                                       V_tot²
```

This is exactly `_sobol_ci` in `analysis/automl.py`. The cost is one `Cov·U`
matvec plus a slice per component — negligible on top of the ridge solve.

---

## 4. The bug this replaced, and why it undercovered

An earlier implementation used only the **own block**: it restricted the gradient
to component `c`'s own coordinates and used only the diagonal block of the
covariance, giving

```
Var_old(S_c) = (1 − S_c)² / V_tot² · own_cᵀ Cov_cc own_c .
```

Compared with the full expression, `Var_old` **drops** the `S_c` coupling on the
other blocks (`−2 S_c · U_cᵀ Cov U`), the cross-component `S_c² · UCU` term, and
all off-diagonal covariance between blocks. The omitted variance is **the same
order in `σ²` and `N` as the term that was kept** — it is a roughly *fixed
fraction* of the true variance, scaled by powers of `S_c`. Two consequences follow
directly, and both were confirmed empirically:

1. **The standard-error deficit does not shrink with sample size.** Because the
   dropped terms scale like the kept term, the *ratio* `SE_old / SE_true` is
   asymptotically constant. Measured `SE/SD ≈ 0.86` held from `N = 1,500` to
   `N = 24,000`. This is the signature that distinguishes a *gradient* bug from a
   finite-sample sandwich (HC0) effect, which would vanish as `N → ∞`.
2. **It is worst for the largest indices.** The dropped terms carry factors of
   `S_c` and `S_c²`, so a dominant variable (`S ≈ 0.4`) was undercovered far more
   than a minor one (`S ≈ 0.05`). Per-variable coverage fell to ≈ 0.86 at the top
   of the spectrum while staying near nominal at the bottom.

Net effect: ~13–15% too-small standard errors, ~90% actual coverage at 95%
nominal — **identically for all three bases**, because the omission is in the
index gradient, which is basis-agnostic.

---

## 5. Coverage validation

The interval is validated by Monte-Carlo. For each basis we draw ground-truth
coefficients **in that basis** (so the model is correctly specified and the
estimand `S_true` is exact and first-order), fix the design `X`, then over many
noise realizations refit and record the CI. We report three quantities:

- **`cov(S_true)`** — fraction of intervals that contain the true index. This is
  the headline calibration number; it should equal the nominal `1 − α`.
- **`cov(E[Ŝ])`** — coverage of the Monte-Carlo mean of the estimator. This
  isolates *CI-vs-sampling-spread* calibration from any small estimator bias.
- **`SE/SD`** — the mean delta-method SE divided by the empirical Monte-Carlo
  standard deviation of `Ŝ`. Perfect calibration is `1.00`.

Config: `D=4`, `K1=4`, `N=3000`, `R=400` noise draws, tiny ridge (`λ=1e-6`, so
shrinkage bias is negligible and `cov(S_true)` is meaningful), Gaussian noise at
signal-to-noise ≈ 4:1, nominal 95%.

| Basis    | `cov(S_true)` before → after | `SE/SD` before → after |
|----------|:---------------------------:|:----------------------:|
| Fourier  | 0.897 → **0.949**           | 0.86 → **1.00**        |
| Legendre | 0.904 → **0.959**           | 0.87 → **1.02**        |
| Haar     | 0.908 → **0.943**           | 0.87 → **0.99**        |

After the full-gradient fix, coverage is nominal and the delta-method SE matches
the true sampling spread for **all three bases**. This also settles a standing
question — whether the CIs were only valid for the Fourier basis: the machinery
is basis-agnostic; Legendre and Haar were never worse than Fourier, and all three
shared the same gradient defect and are fixed together.

**Regression guard.** `tests/test_automl.py::TestSobolCICoverageAcrossBases` runs
a deterministic (fixed-seed) coverage check, parameterized over the three bases,
asserting `coverage ≥ 0.90` and `SE/SD ∈ [0.90, 1.20]`. It **fails on the
own-block gradient** and passes on the full gradient, so the calibration cannot
silently regress. It runs in ~5 s and is in the default test tier.

---

## 6. Scope: which components, which order

The construction is identical for every interaction order. `U` is filled from the
first-order blocks (`G1`), the pair blocks (`G2`), and the triple blocks (`G3`);
`sobol_confidence_intervals` returns intervals under `first_order`,
`second_order`, and `third_order`. Any trailing residual-feature columns (RBF /
RFF / Nyström) contribute **zero** to `U` and therefore do not perturb the
indices or their intervals — consistent with the linear residual being projected
orthogonal to the basis. The first-order block width is read from
`basis_size(K1, include_linear_1, basis_name)`, and the function raises if that
layout does not fit `Φ` (guarding against a silently-wrong column slicing when
`basis_name` / `include_linear_1` disagree with how `Φ` was built).

---

## 7. What the interval does and does not cover

- **It is a frequentist CI for the *fitted* index at the chosen `λ`.** It
  captures the sampling variability of `Ŝ` induced by noise in `y`. It does
  **not** capture ridge **shrinkage bias**: Sobol indices are quadratic in the
  coefficients, so a GCV-optimal penalty biases them slightly downward. For
  (approximately) unbiased *point* recovery use the separate Sobol-estimation
  mode (`estimate_sobol`, `auto_lambda=True`), which is about bias, not variance.
- **Independent inputs / product measure.** `V_c = w_cᵀ G_c w_c` and the additive
  `V_tot` assume a product input measure (uniform marginals after the quantile
  transform). Under correlated inputs the *structural* indices misattribute; the
  honest fallback is the **correlative** indices (`compute_sobol_indices(model,
  x_data)`, which need not sum to 1). The CIs here are for the structural indices.
- **First-order (delta) approximation.** The interval is a linearization of a
  rational-quadratic map. It is accurate across the regime validated above; for an
  index extremely close to `0` or `1` the symmetric normal interval is clipped to
  `[0,1]` and is best read as approximate near the boundary.
- **HC0 in finite samples.** The sandwich "meat" uses raw squared residuals
  (HC0). This is consistent and, in the validated regime, well-calibrated; for
  very small `N` a leverage-corrected variant (HC3) would be marginally more
  conservative. The dominant miscalibration that this note fixes was the gradient,
  not the sandwich.

---

## 8. Reproducing

```python
import numpy as np, jax
jax.config.update("jax_enable_x64", True)
from hifi_anova.core.features import build_first_order_features, basis_size
from hifi_anova.core.gram import build_gram_matrix
from hifi_anova.training.regularization import build_regularization_vector
from hifi_anova.analysis.automl import sobol_confidence_intervals

# Build Φ, G1, reg for the basis of interest, then over many noise draws call
# sobol_confidence_intervals(Φ, y_centered, reg, D, K1, G1, basis_name=...,
#                            include_linear_1=...) and tally how often the
# interval contains the known S_true. See the parameterized regression test
# tests/test_automl.py::TestSobolCICoverageAcrossBases for a complete, runnable
# version.
```

---

## 9. Symbol ↔ code map

| Symbol | Meaning | Code |
|--------|---------|------|
| `A`, `A⁻¹` | `ΦᵀΦ + R` and its inverse | `ridge_analytics` → `A_inv` |
| `e` | ridge residuals | `analytics['residuals']` |
| `Cov(ŵ)` | HC0 sandwich covariance | `sandwich_covariance` → `Cov_w` |
| `G_c` | component Gram matrix | `G1`, `G2`, `G3` |
| `V_c`, `V_tot` | component / total variance | `first_order_vars`, `total_var` |
| `S_c` | Sobol index | `S` in `_sobol_ci` |
| `U`, `CovU`, `UCU` | total score, `Cov·U`, `Uᵀ Cov U` | `U`, `Cov_U`, `UCU` |
| `SE(S_c)` | delta-method standard error | `se_S` |

---

## References

- H. White (1980), *A Heteroskedasticity-Consistent Covariance Matrix Estimator*
  — the sandwich / HC0 estimator.
- The **delta method** for functions of asymptotically normal estimators (any
  standard mathematical-statistics text).
- I. M. Sobol (2001) and W. Hoeffding (1948) — the functional-ANOVA variance
  decomposition underlying `V_c = w_cᵀ G_c w_c`.
- Polynomial chaos expansion (PCE) and RS-HDMR — the established practice of
  reading analytic Sobol indices off orthogonal-basis regression coefficients,
  which HiFi-ANOVA extends (see the README "Relationship to prior work").

---

*Documentation notice: Copyright (c) 2026 R. Sala. All rights reserved.
Draft, work in progress — not covered by the source-code license (PolyForm
Internal Use 1.0.0). Except for permissions arising under GitHub's Terms of
Service, applicable law, or separate written permission from the copyright
holder, no permission is granted to reproduce, distribute, modify, publish, or
create derivative works from this document. See LICENSING.md.*
