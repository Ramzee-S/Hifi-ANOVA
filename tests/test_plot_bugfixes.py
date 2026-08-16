"""Regression tests for basis-awareness bugs in the plotting layer.

Two previously-latent bugs, both from hardcoded Fourier assumptions:

1. ``plot_interaction_grid`` computed the per-variable block size with
   ``basis_size(K2, incl_lin)`` (Fourier default 2K2+1) and rebuilt the basis
   without ``basis_name``. For a Legendre/Haar model the pair coefficient vector
   has length ``basis_size(K2, basis_name)**2`` (e.g. K2**2 for Legendre), so the
   ``wp.reshape(B, B)`` raised ValueError. Fixed by threading the model basis.

2. ``plot_frequency_content`` is intrinsically Fourier (linear/cos/sin harmonic
   layout, ``2K1+1`` blocks). On a non-Fourier model it silently mislabelled
   coefficients; it now raises a clear ValueError.
"""
import jax
jax.config.update("jax_enable_x64", True)

import numpy as np
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

from hifi_anova.training.trainer import HiFiANOVATrainer
from hifi_anova.analysis.plots import (
    plot_interaction_grid, plot_frequency_content,
)


def _fit_interaction_model(basis_name):
    """Fit a small D=3 model with a real x0*x1 interaction (K2>0)."""
    np.random.seed(0)
    N = 1500
    x = np.random.uniform(0, 1, (N, 3))
    y = (3.0 * (x[:, 0] - 0.5)
         + 2.0 * (x[:, 1] - 0.5)
         + 4.0 * (x[:, 0] - 0.5) * (x[:, 1] - 0.5)
         + 0.1 * np.random.randn(N))
    x, y = jnp.array(x), jnp.array(y)
    n_val = 300
    cfg = {
        "stages": ["A", "B"],
        "K1": 4, "K2": 3,
        "strategy": "uniform",
        "lambda_order1": 1e-3, "lambda_order2": 1e-2,
        "basis_name": basis_name,
    }
    model, _ = HiFiANOVATrainer(cfg).fit(
        x[n_val:], y[n_val:], x[:n_val], y[:n_val])
    return model


class TestInteractionGridBasisAware:
    """plot_interaction_grid must not assume a Fourier block size."""

    @pytest.mark.parametrize("basis_name", ["legendre", "haar", "fourier"])
    def test_grid_renders_for_basis(self, basis_name):
        model = _fit_interaction_model(basis_name)
        assert model.K2 > 0 and model.pair_indices is not None
        # Before the fix this raised ValueError on wp.reshape(B, B) for
        # legendre/haar (B computed as the Fourier 2K2+1).
        fig, axes = plot_interaction_grid(model, top_k=3)
        assert fig is not None
        assert axes.size >= 1
        plt.close(fig)

    def test_legendre_block_size_matches_coefficients(self):
        """The reshape must use the Legendre block (K2), not 2K2+1."""
        from hifi_anova.core.features import basis_size
        model = _fit_interaction_model("legendre")
        B = basis_size(model.K2, getattr(model, "include_linear_2", True),
                       "legendre")
        wp = np.asarray(model.mean_model.get_coefficients_for_pair(0))
        # Coefficient vector must be exactly B*B so wp.reshape(B, B) is valid.
        assert wp.size == B * B


class TestFrequencyContentGuard:
    """plot_frequency_content is Fourier-only and must say so clearly."""

    def test_fourier_still_works(self):
        model = _fit_interaction_model("fourier")
        fig, ax = plot_frequency_content(model)
        assert fig is not None
        plt.close(fig)

    @pytest.mark.parametrize("basis_name", ["legendre", "haar"])
    def test_non_fourier_raises_clear_error(self, basis_name):
        model = _fit_interaction_model(basis_name)
        with pytest.raises(ValueError, match="Fourier"):
            plot_frequency_content(model)
