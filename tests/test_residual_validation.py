"""Stage C rejects an unknown residual type instead of silently no-op'ing.

Regression test for the `residual=` footgun: an unknown residual type (a typo
such as `residual='rbg'`, or a dict with an unrecognized `type`) used to add
stage 'C' to the pipeline where no branch matched it, so Stage C silently did
nothing and the caller got a residual-free model believing one was fitted.
The trainer now raises ValueError early. Fast — the error fires at the top of
Stage C, right after the Stage A solve.
"""

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest

from hifi_anova.training.trainer import HiFiANOVATrainer


def _tiny_data(seed=0, n=80, d=2):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 1.0, size=(n, d))          # already in [0, 1]
    y = np.sin(2 * np.pi * x[:, 0]) + 0.5 * x[:, 1]
    x = jnp.asarray(x, dtype=jnp.float64)
    y = jnp.asarray(y, dtype=jnp.float64)
    return x, y


def _base_cfg():
    return {"K1": 4, "K2": 0, "strategy": "variance",
            "lambda_order1": 0.01, "stages": ["A", "C"]}


def test_unknown_residual_type_dict_raises():
    x, y = _tiny_data()
    cfg = _base_cfg()
    cfg["residual"] = {"type": "gaussian", "lambda_residual": 1.0}
    with pytest.raises(ValueError, match="Unknown residual type"):
        HiFiANOVATrainer(cfg).fit(x, y, x, y)


def test_unknown_residual_type_string_raises():
    """A bare-string residual config is coerced to {'type': ...} then validated."""
    x, y = _tiny_data()
    cfg = _base_cfg()
    cfg["residual"] = "rbg"          # typo for 'rbf'
    with pytest.raises(ValueError, match="Unknown residual type"):
        HiFiANOVATrainer(cfg).fit(x, y, x, y)


def test_one_call_api_unknown_residual_raises():
    """The one-call API forwards residual= as {'type': ...}; unknown -> ValueError."""
    from hifi_anova.api import hifi_anova
    rng = np.random.default_rng(1)
    X = rng.uniform(0.0, 1.0, size=(120, 3))
    y = np.sin(2 * np.pi * X[:, 0]) + X[:, 1] * X[:, 2]
    with pytest.raises(ValueError, match="Unknown residual type"):
        hifi_anova(X, y, residual="bogus")


def test_known_residual_type_not_rejected():
    """A valid type ('nn' disabled) must pass validation and simply skip Stage C."""
    x, y = _tiny_data()
    cfg = _base_cfg()
    cfg["residual"] = {"type": "nn", "enabled": False}
    # Should not raise; Stage C is skipped because the NN is not enabled.
    model, _ = HiFiANOVATrainer(cfg).fit(x, y, x, y)
    assert model is not None
