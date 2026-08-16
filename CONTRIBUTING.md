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
python -m pytest tests/ --all         # ~12 min, everything
```

`precision=` selects the requested model/storage dtype. The default requests
float32 model weights, while selected post-fit analytics and Stage-D linear
solves are promoted to float64. Opt into float64 model/storage as well with
`hifi_anova(..., precision="float64")` or `HIFI_ANOVA_X64=1`; precedence is
resolved centrally in `hifi_anova/precision.py` and threaded through
preprocessing and training. If you call lower-level functions directly and want
float64, enable x64 yourself *and* pass float64 arrays:

```python
import jax
jax.config.update("jax_enable_x64", True)
```

## Code-quality tooling

Quality gates are configured in `pyproject.toml` and `.pre-commit-config.yaml`
and installed with the `dev` extra (`pip install -e ".[dev]"`):

```bash
ruff check hifi_anova          # lint — the CI gate (E4,E7,E9,F); tree passes clean
mypy hifi_anova                # optional static typing (lenient, informational)
pytest tests/ --cov=hifi_anova --cov-report=term-missing   # coverage
pre-commit install && pre-commit run --all-files            # local hooks
```

**Lint policy.** CI gates on the full `[tool.ruff.lint]` set (`E4,E7,E9,F`:
syntax, real bugs, imports, statement hygiene, and pyflakes — unused
imports/vars, ambiguous names), which the tree passes clean. The former
cosmetic backlog was paid down as behavior-preserving dead-code removals and
local renames, then checked against the characterization suite. `E501` (line
length) is still deliberately excluded pending a formatting decision (below).

**Formatting.** No auto-formatter (black / `ruff format`) is wired in: the code
is hand-formatted to a 79-column style that a formatter would rewrite wholesale.
Standardizing on one is a deliberate one-time decision, not a silent hook.

**Types & coverage** are informational for now (no `mypy` gate, no
`--cov-fail-under`): measure first, then ratchet thresholds.

## Reporting issues

Please include: the config dict / API call you used, the input shape, the
observed vs. expected behavior, and your `jax` / `equinox` / `numpy` versions.

## Known areas still in progress

See `CHANGELOG.md` (Known limitations) and the roadmap in `README.md` — e.g.
the missing Ishigami benchmark, splitting the large `analysis/plots.py` module,
validating confidence intervals for non-Fourier bases, and the structured
Hoeffding-ANOVA network follow-up.

---

*Documentation notice: Copyright (c) 2026 R. Sala. All rights reserved.
Draft, work in progress — not covered by the source-code license (PolyForm
Internal Use 1.0.0). Except for permissions arising under GitHub's Terms of
Service, applicable law, or separate written permission from the copyright
holder, no permission is granted to reproduce, distribute, modify, publish, or
create derivative works from this document. See LICENSING.md.*
