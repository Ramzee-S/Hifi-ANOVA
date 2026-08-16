# Examples

Runnable scripts demonstrating the library. Install the package first
(`pip install -e .` from the repo root), then run from the repo root:

```bash
python examples/run_demo.py                     # 3 experiments + figures (writes ./figures/)
python examples/run_ishigami_heteroscedastic.py # dual mean/log-variance spectrum
python examples/run_sobol_vs_prediction.py      # prediction vs Sobol-estimation modes
python examples/run_salib_comparison.py         # SALib ground-truth check
```

`run_ishigami_heteroscedastic.py` is the fullest showcase: it fits the mean
*and* the log-variance of a heteroscedastic Ishigami function (whose analytic
mean Sobol indices are known), recovers them, rediscovers the x1–x3 interaction with the
residual sieve, checks calibration, and renders the **dual-sensitivity ellipse**
plots — a compact view where each variable's mean sensitivity and log-variance
index are read off the shape/position of one ellipse. It also uses
`first_order_pruning='bic'` to zero x3's spurious first-order component (x3 is
pure interaction), reports the **explained-variance split** of the mean vs the
log-variance model, and renders the **regularization path** (L-curve, GCV/evidence
selection, Sobol-vs-λ, and the variance-decomposition trade-off).

`run_salib_comparison.py` needs the optional SALib dependency:

```bash
pip install -e ".[salib]"
```

Figures are written to a `figures/` directory in the current working directory
(git-ignored).

---

*Documentation notice: Copyright (c) 2026 R. Sala. All rights reserved.
Draft, work in progress — not covered by the source-code license (PolyForm
Internal Use 1.0.0). Except for permissions arising under GitHub's Terms of
Service, applicable law, or separate written permission from the copyright
holder, no permission is granted to reproduce, distribute, modify, publish, or
create derivative works from this document. See LICENSING.md.*
