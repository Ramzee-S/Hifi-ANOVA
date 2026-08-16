"""FourierComponent: coefficient block + Gram reference for Sobol computation."""

from ..array_backend import xp as jnp  # switchable array backend (numpy exact core)
from dataclasses import dataclass
from typing import Tuple


@dataclass
class FourierComponent:
    """A single structured Hoeffding (ANOVA) component (one variable or one pair).

    This is a DATA CLASS, not an Equinox module. It's a structured container
    for coefficients and metadata, used by the analysis module for Sobol extraction.
    """
    coefficients: jnp.ndarray  # (2K+1,) for order 1 or ((2K+1)^2,) for order 2
    order: int                  # 1 or 2
    variable_indices: Tuple     # (i,) or (i, j)

    def variance(self, G: jnp.ndarray) -> float:
        """Analytic variance: w^T G w. Exact, no data needed."""
        w = jnp.asarray(self.coefficients, dtype=jnp.float64)
        G = jnp.asarray(G, dtype=jnp.float64)
        return jnp.maximum(0.0, w @ G @ w)

    def curvature(self, D2: jnp.ndarray) -> float:
        """Integrated squared curvature: w^T D(2) w."""
        w = jnp.asarray(self.coefficients, dtype=jnp.float64)
        D2 = jnp.asarray(D2, dtype=jnp.float64)
        return w @ (D2 * w)
