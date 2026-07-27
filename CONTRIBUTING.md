# Notes for readers & evaluators

**HiFi-ANOVA is a source-available, work-in-progress research project — not an
open-source project.** It is shared as a portfolio / preview under the
[PolyForm Internal Use License 1.0.0](https://polyformproject.org/licenses/internal-use/1.0.0):
you may use it, and make changes and new works based on it, **only for the
internal business operations of you and your company**, and you may **not
distribute it to third parties**. See [LICENSE](LICENSE).

Because of that license posture, **external code contributions (pull requests,
patches) are not being accepted at this stage**, and redistribution is not
permitted. What *is* welcome:

- **Feedback, questions, and bug reports** via the issue tracker.
- **Internal evaluation** of the method on your own data.

For any use beyond the license terms — e.g. distribution or providing the
software to third parties — please contact the licensor (libre-labs.org).

## Running it locally (for evaluation)

```bash
conda env create -f environment.yml   # or use your own venv
conda activate hifi_anova
pip install -e ".[dev,salib]"
```

Verify the install:

```bash
python -m pytest tests/ -m smoke      # ~1 min, core math only
python -m pytest tests/               # ~3 min, + fitting & analysis
python -m pytest tests/ --full        # ~10 min, + integration pipelines
python -m pytest tests/ --all         # ~12 min, everything (394 tests)
```

All numerical code assumes float64. JAX x64 is enabled globally in
`conftest.py` (tests) and inside `hifi_anova.api`; if you call lower-level
functions directly, set it yourself:

```python
import jax
jax.config.update("jax_enable_x64", True)
```

## Reporting issues

Please include: the config dict / API call you used, the input shape, the
observed vs. expected behavior, and your `jax` / `equinox` / `numpy` versions.

## Known areas still in progress

See `CHANGELOG.md` (Known limitations) and the roadmap in `README.md` — e.g.
the missing Ishigami benchmark, splitting the large `analysis/plots.py` module,
validating confidence intervals for non-Fourier bases, and the structured
Hoeffding-Fourier network follow-up.

---

*Documentation notice: Copyright (c) 2026 R. Sala. All rights reserved.
Draft, work in progress — not covered by the source-code license (PolyForm
Internal Use 1.0.0). Except for permissions arising under GitHub's Terms of
Service, applicable law, or separate written permission from the copyright
holder, no permission is granted to reproduce, distribute, modify, publish, or
create derivative works from this document. See LICENSING.md.*
