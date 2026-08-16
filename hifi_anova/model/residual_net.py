"""ResidualMLP: wrapper around equinox MLP for residual modeling."""

import jax
import equinox as eqx
from typing import List


def create_residual_mlp(D: int, hidden_dims: List[int],
                        key: jax.Array) -> eqx.nn.MLP:
    """Create a residual MLP for capturing higher-order signal.

    Args:
        D: input dimension
        hidden_dims: list of hidden layer sizes
        key: PRNG key for initialization

    Returns:
        eqx.nn.MLP with D inputs, 1 output, GELU activation
    """
    return eqx.nn.MLP(
        in_size=D,
        out_size=1,
        width_size=hidden_dims[0] if hidden_dims else 256,
        depth=len(hidden_dims),
        activation=jax.nn.gelu,
        key=key,
    )
