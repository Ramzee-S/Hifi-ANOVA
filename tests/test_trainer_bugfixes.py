"""Regression tests for latent trainer bugs (read-before-assignment).

These guard specific crash paths that were previously reachable with ordinary
configs. Kept small (low N) so they run in the default (quick) tier.
"""
import jax
jax.config.update("jax_enable_x64", True)

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest

from hifi_anova.data.synthetic import generate_ishigami
from hifi_anova.data.preprocessing import preprocess_data
from hifi_anova.training.trainer import HiFiANOVATrainer
from hifi_anova.analysis.sobol import compute_sobol_indices


def test_third_order_default_triple_selection_does_not_crash():
    """K3>0 with the DEFAULT triple_selection ('all_active') must fit.

    That default routes through the threshold-fallback branch that builds a
    temporary first-order model to score variables. It referenced ``_bs`` — a
    name bound only by a *later* local ``import basis_size as _bs`` in fit() — so
    Python treated ``_bs`` as a function-local and raised UnboundLocalError
    before that import ran. (A config that set triple_selection='all' or a
    principled method sidestepped the branch, hiding the bug.) The module-level
    ``basis_size`` is now used instead.
    """
    X, y, _ = generate_ishigami(n_samples=600, noise_std=0.1, seed=0)
    data = preprocess_data(X, y, seed=0)
    cfg = {
        "K1": 6, "K2": 3, "K3": 2, "strategy": "variance",
        "lambda_order1": 1e-3, "lambda_order2": 1e-2, "lambda_order3": 1e-1,
        "stages": ["A", "B"],
        # NOTE: no 'triple_selection' -> default 'all_active' (the crash path).
    }
    model, results = HiFiANOVATrainer(cfg).fit(
        data["x_train"], data["y_train"], data["x_val"], data["y_val"])

    # Sanity: it produced a usable model with well-formed Sobol indices.
    sob = compute_sobol_indices(model, data["x_test"])
    first = sob["mean_sobol"]["first_order"]
    assert set(first) == {0, 1, 2}
    assert all(np.isfinite(v) for v in first.values())
    # Ishigami mean first-order: x1~0.31, x2~0.44, x3~0 (loose bounds).
    assert first[1] > first[2]


def _fit_third_order_model():
    """Small K3>0 model (Ishigami) fit through stages A, B."""
    X, y, _ = generate_ishigami(n_samples=500, noise_std=0.1, seed=0)
    data = preprocess_data(X, y, seed=0)
    cfg = {
        "K1": 4, "K2": 2, "K3": 2, "strategy": "variance",
        "lambda_order1": 1e-3, "lambda_order2": 1e-2, "lambda_order3": 1e-1,
        "stages": ["A", "B"],
        "triple_selection": "all",  # force the single C(3,3) triple
    }
    model, _ = HiFiANOVATrainer(cfg).fit(
        data["x_train"], data["y_train"], data["x_val"], data["y_val"])
    return model, data


def test_projected_finetune_uses_third_order_features():
    """loss_fn_projected must include phi3 so third-order coeffs stay trainable.

    In the orthogonal joint-finetune, the Fourier mean was computed as
    ``predict(phi1, phi2)`` — omitting phi3. With K3>0 the loss then had *no*
    dependence on w3, so its gradient was identically zero and the third-order
    coefficients were frozen during fine-tuning. Passing phi3 restores the
    dependence; here we assert w3 actually moves.
    """
    from hifi_anova.model.residual_net import create_residual_mlp
    from hifi_anova.training.sgd import joint_finetune

    model, data = _fit_third_order_model()
    assert model.K3 > 0 and model.triple_indices is not None

    # Attach a fresh NN residual so the projected branch is taken.
    nn = create_residual_mlp(model.D, [16, 16], jax.random.PRNGKey(1))
    model = eqx.tree_at(lambda m: m.residual_net, model, nn,
                        is_leaf=lambda x: x is None)

    reg_diag = jnp.ones(model.build_phi_all(data["x_train"]).shape[1])
    w3_before = np.asarray(model.mean_model.w3).copy()
    assert w3_before.size > 0

    tuned = joint_finetune(
        model, data["x_train"], data["y_train"],
        data["x_val"], data["y_val"],
        lr=1e-2, epochs=3, batch_size=256,
        orthogonal=True, reg_diag=reg_diag,
    )
    w3_after = np.asarray(tuned.mean_model.w3)

    # Before the fix the gradient w.r.t. w3 was exactly zero -> unchanged.
    assert not np.allclose(w3_before, w3_after), (
        "third-order coefficients did not update during projected finetune")


def test_heteroscedastic_zero_outer_iter_raises():
    """max_outer_iter=0 must raise a clear ValueError, not UnboundLocalError.

    The alternating mean/variance loop assigns ``w_all`` only inside
    ``for outer in range(max_outer)``; with max_outer==0 the final model build
    read an unbound ``w_all``. Now guarded with an explicit ValueError.
    """
    X, y, _ = generate_ishigami(n_samples=400, noise_std=0.1, seed=0)
    data = preprocess_data(X, y, seed=0)
    cfg = {
        "K1": 4, "K2": 2, "Kh": 2, "strategy": "variance",
        "lambda_order1": 1e-3, "lambda_order2": 1e-2,
        "stages": ["A", "B", "D"],
        "max_outer_iter": 0,
    }
    with pytest.raises(ValueError, match="max_outer_iter"):
        HiFiANOVATrainer(cfg).fit(
            data["x_train"], data["y_train"], data["x_val"], data["y_val"])



@pytest.mark.smoke
def test_default_config_is_mean_only():
    """The DEFAULT trainer config fits the MEAN only — variance/residual opt-in.

    Regression guard: a bare ``HiFiANOVATrainer({})`` must not silently fit a
    heteroscedastic variance model (Stage D) or an NN residual (Stage C). Those
    are opt-in via mode='heteroscedastic'/'full'/'auto' or an explicit stages=.
    """
    rng = np.random.default_rng(0)
    X = rng.uniform(-np.pi, np.pi, size=(400, 3))
    y = np.sin(X[:, 0]) + 7 * np.sin(X[:, 1]) ** 2 + 0.1 * X[:, 2] ** 4 * np.sin(X[:, 0])
    data = preprocess_data(X, y, seed=0)
    model, _ = HiFiANOVATrainer({}).fit(          # no 'stages', no 'mode'
        data['x_train'], data['y_train'], data['x_val'], data['y_val'])
    assert model.variance_model is None            # variance is opt-in
    assert model.residual_net is None              # residual is opt-in
