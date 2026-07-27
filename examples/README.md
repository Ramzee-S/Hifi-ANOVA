# Examples

Runnable scripts demonstrating the library. Install the package first
(`pip install -e .` from the repo root), then run from the repo root:

```bash
python examples/run_demo.py                # 3 experiments + figures (writes ./figures/)
python examples/run_sobol_vs_prediction.py # prediction vs Sobol-estimation modes
python examples/run_salib_comparison.py    # SALib ground-truth check
```

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
