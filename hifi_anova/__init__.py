"""HiFi-ANOVA: Interpretable Regression with Analytic Sobol Diagnostics for Mean and Variance.

HiFi-ANOVA (Hoeffding Interaction-Fidelity ANOVA) is a functional-ANOVA
(Hoeffding) decomposition of the conditional mean and variance, realized over
Fourier, Legendre, and Haar bases. The core model class is :class:`HiFiANOVA`
and the staged fitter is :class:`HiFiANOVATrainer`. The one-call entry point is
:func:`hifi_anova`.

    from hifi_anova import hifi_anova, HiFiANOVATrainer, compute_sobol_indices

    result = hifi_anova(X, y)          # one-call fit -> Sobol -> intervals
"""

__version__ = "0.2.0"

import warnings as _warnings

# The model classes store fixed structural constants — Gram matrices (G1/G2/G3)
# and integer interaction-index arrays (pair_indices, triple_indices) — in
# equinox `static=True` fields. equinox warns ("A JAX array is being set as
# static!") whenever a static field holds an array, on the assumption it is a
# mistake. Here it is deliberate and required: these are compile-time constants,
# and keeping them static is exactly what keeps the joint fine-tuning optimizer
# (`eqx.filter(model, eqx.is_array)`) from ever treating them as trainable
# parameters. So we silence only that one equinox message; all other warnings
# are left untouched.
_warnings.filterwarnings(
    "ignore",
    message=r"A JAX array is being set as static",
    category=UserWarning,
)

from .model.hifi_anova import HiFiANOVA
from .training.trainer import HiFiANOVATrainer, estimate_sobol
from .analysis.sobol import compute_sobol_indices
from .api import hifi_anova, HiFiResult

__all__ = [
    "HiFiANOVA",
    "HiFiANOVATrainer",
    "hifi_anova",
    "HiFiResult",
    "estimate_sobol",
    "compute_sobol_indices",
    "__version__",
]
