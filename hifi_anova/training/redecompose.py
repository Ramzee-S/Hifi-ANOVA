"""Solutions for maintaining Sobol integrity with a residual NN.

Solution 1: Re-decomposition after joint fine-tuning
  Train jointly for best RMSE, then re-project the total learned function
  onto the structured (Hoeffding-ANOVA) basis using uniform evaluation points.
  The total prediction doesn't change — only the attribution between
  Fourier and NN changes.

Solution 2: Alternating ridge-NN optimization
  Instead of joint SGD, alternate between closed-form ridge for Fourier
  and SGD for NN. The Fourier coefficients are always ridge-optimal,
  so their Sobol indices are meaningful at every iteration.

Both solutions are modular — they take a fitted model and return
an updated model. The existing training pipeline doesn't change.
"""

import jax
import jax.numpy as jnp
import equinox as eqx
import optax
from typing import Optional

from ..core.features import basis_size
from ..model.mean_model import MeanModel
from ..model.hifi_anova import HiFiANOVA
from ..model.linear_residual import ProjectedResidual
from .ridge import weighted_ridge_solve


def _reject_term_structure(model, fn):
    """Guard the NN re-decomposition paths against user-defined term structure.

    Both re-decomposition routines slice the mean coefficient vector with the
    UNIFORM per-order block widths (``F1 = D·b1``, ``F2 = P·b2``). A model with
    a per-pair ``K2`` mapping / ``variable_orders`` (BR-04/BR-06) carries a
    RAGGED ``w2`` (per-pair block sizes) — and a mixed per-variable basis
    (``var_specs``) likewise — so the uniform slice would silently mis-align
    the folded projection and corrupt the coefficients. These are not reachable
    from a term-structure fit's default (mean-only) path; reject them explicitly
    rather than return a corrupted model. ``fo_included`` alone keeps the
    uniform ``w1`` but the rebuilt model would drop its fitted-design layout
    metadata, so it is rejected here too.
    """
    if (getattr(model, 'pair_block_info', None) is not None
            or getattr(model, 'pair_k2', None) is not None
            or getattr(model, 'var_specs', None) is not None
            or getattr(model, 'fo_included', None) is not None):
        raise NotImplementedError(
            f"{fn} does not support user-defined term structure (per-pair K2 "
            "mapping, variable_orders, or a mixed per-variable basis): the "
            "Fourier/NN re-decomposition assumes the uniform per-order block "
            "layout and would mis-slice the ragged coefficient vector. Re-fit "
            "without the term-structure keys to use the residual-NN stages.")


# =============================================================================
# Solution 1: Re-decomposition
# =============================================================================

def redecompose(model: HiFiANOVA,
                reg_diag: Optional[jnp.ndarray] = None,
                n_eval: int = 50000,
                seed: int = 0) -> HiFiANOVA:
    """Re-decompose a jointly-trained model into a clean Fourier + NN split.

    Computes the Hoeffding (functional-ANOVA) decomposition of the **full learned
    function** ``f_total = f_Fourier_old + f_NN`` by projecting ``f_total`` onto
    the low-order Fourier basis under the uniform input measure on ``[0, 1]^D``.

    Writing ``P`` for that L2 projection, the projection of the total is::

        f_Fourier_new = P(f_total) = f_Fourier_old + P(f_NN)

    (the old Fourier part is already in the basis, so only the NN's
    low-order-representable structure ``P(f_NN)`` is new). The residual becomes the
    orthogonal remainder::

        residual_new = f_total - f_Fourier_new = f_NN - P(f_NN) = (I - P) f_NN

    So the operator **moves the NN's low-order structure into the interpretable
    Fourier coefficients** and leaves a residual with (approximately) zero
    low-order content — a genuine Hoeffding residual. The Sobol indices of the new
    Fourier coefficients then describe the low-order structure of the *whole*
    learned function, not just the (possibly drifted) Fourier part.

    **Prediction is preserved exactly.** The projection ``P(f_NN)`` is folded into
    the mean model and the *same* term is subtracted inside a
    :class:`~hifi_anova.model.linear_residual.ProjectedResidual` wrapping the
    original residual, so ``mean_new(x) + residual_new(x) = f_total(x)`` for every
    ``x`` — independent of ``n_eval`` or the projection's accuracy (which only
    affects how the attribution is split, not the total).

    This is the corrected operator (advisor guidance, DEC-020/DEC-022). The prior
    implementation projected ``f_total - f_NN`` — which equals the *already-drifted*
    Fourier part — and so was a no-op for attribution. If the model has **no**
    residual net, the decomposition is already clean and the model is returned
    unchanged.

    Args:
        model: HiFiANOVA after joint fine-tuning (or any training).
        reg_diag: (F,) regularization for the projection of ``f_NN``. ``None`` =
            OLS, i.e. the exact L2 (Hoeffding) projection — the default and the
            statistically meaningful choice. A non-zero ``reg_diag`` shrinks how
            much NN structure is folded into the Fourier part (prediction is still
            preserved exactly).
        n_eval: number of uniform evaluation points for the projection (more =
            more accurate attribution split).
        seed: PRNG seed for the evaluation points.

    Returns:
        New HiFiANOVA. ``mean_model`` carries ``w_old + P(f_NN)`` coefficients and
        ``residual_net`` is a ``ProjectedResidual`` emitting ``(I - P) f_NN``.
        Total prediction is byte-preserved; Sobol indices reflect the Hoeffding
        decomposition of ``f_total``.
    """
    # No residual → f_total is already the Fourier part; nothing to fold in.
    if model.residual_net is None:
        return model
    _reject_term_structure(model, 'redecompose')

    D = model.D
    K1 = model.K1
    K2 = model.K2
    K3 = getattr(model, 'K3', 0)
    include_linear_1 = getattr(model, 'include_linear_1', True)
    include_linear_2 = getattr(model, 'include_linear_2', True)
    include_linear_3 = getattr(model, 'include_linear_3', True)
    basis_name_val = getattr(model, 'basis_name', 'fourier')

    # Uniform evaluation points + Fourier design under the uniform measure.
    key = jax.random.PRNGKey(seed)
    x_eval = jax.random.uniform(key, (n_eval, D))
    Phi_eval = model.build_phi_all(x_eval)

    # Evaluate the NN residual on the eval points.
    f_nn_eval = jax.vmap(model.residual_net)(x_eval)
    if f_nn_eval.ndim > 1:
        f_nn_eval = f_nn_eval.squeeze(-1)

    # === Project f_NN onto the low-order Fourier basis (P(f_NN)) ===
    # constant part + coefficients of the centered projection.
    f0_proj = jnp.mean(f_nn_eval)
    f_nn_centered = f_nn_eval - f0_proj
    if reg_diag is not None:
        w_proj = weighted_ridge_solve(Phi_eval, f_nn_centered, reg_diag)
    else:
        F = Phi_eval.shape[1]
        tiny_reg = jnp.full(F, 1e-10)
        w_proj = weighted_ridge_solve(Phi_eval, f_nn_centered, tiny_reg)

    # === Fold P(f_NN) into the Fourier coefficients: w_new = w_old + w_proj ===
    F1 = D * basis_size(K1, include_linear_1, basis_name_val)
    if K2 > 0:
        B2 = basis_size(K2, include_linear_2, basis_name_val)
        n_pairs = model.pair_indices.shape[0] if model.pair_indices is not None else 0
        F2 = n_pairs * B2 * B2
    else:
        F2 = 0

    mm = model.mean_model
    w1_old = jnp.asarray(mm.w1, dtype=jnp.float32)
    w2_old = jnp.asarray(mm.w2, dtype=jnp.float32)
    w3_old = jnp.asarray(mm.w3, dtype=jnp.float32)
    w_proj = jnp.asarray(w_proj, dtype=jnp.float32)

    w1_new = w1_old + w_proj[:F1]
    w2_new = (w2_old + w_proj[F1:F1 + F2]) if K2 > 0 else w2_old
    w3_new = (w3_old + w_proj[F1 + F2:]) if K3 > 0 else w3_old
    f0_new = jnp.asarray(mm.f0, dtype=jnp.float32) + jnp.asarray(f0_proj,
                                                                dtype=jnp.float32)

    new_mean_model = MeanModel(
        f0=f0_new, w1=w1_new, w2=w2_new,
        K1=K1, K2=K2, D=D, w3=w3_new, K3=K3,
        include_linear_1=include_linear_1,
        include_linear_2=include_linear_2,
        include_linear_3=include_linear_3,
        basis_name=basis_name_val,
    )

    # === Residual becomes (I - P) f_NN: wrap to subtract the same projection ===
    new_residual = ProjectedResidual(
        inner=model.residual_net,
        proj_coeffs=w_proj,
        f0_proj=jnp.asarray(f0_proj, dtype=jnp.float32),
        pair_indices=model.pair_indices,
        triple_indices=model.triple_indices,
        K1=K1, K2=K2, K3=K3, D=D,
        include_linear_1=include_linear_1,
        include_linear_2=include_linear_2,
        include_linear_3=include_linear_3,
        basis_name=basis_name_val,
    )

    new_model = eqx.tree_at(
        lambda m: (m.mean_model, m.residual_net), model,
        (new_mean_model, new_residual))
    return new_model


# =============================================================================
# Solution 2: Alternating ridge-NN optimization
# =============================================================================

def alternating_ridge_nn(
    model: HiFiANOVA,
    x_train: jnp.ndarray,
    y_train: jnp.ndarray,
    x_val: jnp.ndarray,
    y_val: jnp.ndarray,
    reg_diag: jnp.ndarray,
    n_outer: int = 5,
    nn_epochs_per_outer: int = 50,
    nn_lr: float = 0.001,
    nn_weight_decay: float = 0.0001,
    nn_batch_size: int = 512,
    nn_patience: int = 10,
    key: Optional[jax.Array] = None,
    verbose: bool = True,
) -> HiFiANOVA:
    """Alternating optimization: ridge for Fourier, SGD for NN.

    Instead of joint SGD on all parameters, alternates between:
      Step A: Fix NN, re-solve Fourier by ridge on NN-corrected targets
      Step B: Fix Fourier, train NN by SGD on Fourier residuals

    The Fourier coefficients are ALWAYS the ridge-optimal solution,
    so their Sobol indices are meaningful at every iteration.

    Args:
        model: HiFiANOVA with residual_net already attached (from staged training)
        x_train, y_train: training data
        x_val, y_val: validation data
        reg_diag: (F,) regularization diagonal for Fourier ridge
        n_outer: number of alternating iterations
        nn_epochs_per_outer: SGD epochs per NN step
        nn_lr, nn_weight_decay, nn_batch_size, nn_patience: NN training config
        key: PRNG key
        verbose: print progress

    Returns:
        Model with ridge-optimal Fourier coefficients and NN trained on residuals.
    """
    if key is None:
        key = jax.random.PRNGKey(42)
    _reject_term_structure(model, 'alternating_ridge_nn')

    D = model.D
    K1 = model.K1
    K2 = model.K2
    K3 = getattr(model, 'K3', 0)
    include_linear_1 = getattr(model, 'include_linear_1', True)
    include_linear_2 = getattr(model, 'include_linear_2', True)
    include_linear_3 = getattr(model, 'include_linear_3', True)
    basis_name_val = getattr(model, 'basis_name', 'fourier')

    # Build features (once)
    phi1_train = model.build_phi1(x_train)
    phi1_val = model.build_phi1(x_val)
    phi2_train = model.build_phi2(x_train)
    phi2_val = model.build_phi2(x_val)
    phi3_train = model.build_phi3(x_train) if K3 > 0 else None
    phi3_val = model.build_phi3(x_val) if K3 > 0 else None

    Phi_train = model.build_phi_all(x_train)

    F1 = D * basis_size(K1, include_linear_1, basis_name_val)
    if K2 > 0:
        B2 = basis_size(K2, include_linear_2, basis_name_val)
        n_pairs = model.pair_indices.shape[0] if model.pair_indices is not None else 0
        F2 = n_pairs * B2 * B2
    else:
        F2 = 0
    N = x_train.shape[0]

    nn = model.residual_net

    for outer in range(n_outer):
        # ---- Step A: Fourier update (ridge, closed-form) ----
        # Target for Fourier: y minus NN prediction
        if nn is not None:
            nn_pred_train = jax.vmap(nn)(x_train).squeeze(-1)
        else:
            nn_pred_train = jnp.zeros(N)

        y_for_fourier = y_train - nn_pred_train
        f0_new = float(jnp.mean(y_for_fourier))
        y_centered = y_for_fourier - f0_new

        w_new = weighted_ridge_solve(Phi_train, y_centered, reg_diag)

        # Update mean model
        w1_new = w_new[:F1]
        w2_new = w_new[F1:F1 + F2] if K2 > 0 else jnp.array([], dtype=jnp.float32)
        w3_new = w_new[F1 + F2:] if K3 > 0 else jnp.array([], dtype=jnp.float32)

        new_mean_model = MeanModel(
            f0=jnp.array(f0_new, dtype=jnp.float32),
            w1=jnp.array(w1_new, dtype=jnp.float32),
            w2=jnp.array(w2_new, dtype=jnp.float32),
            K1=K1, K2=K2, D=D,
            w3=jnp.array(w3_new, dtype=jnp.float32),
            K3=K3,
            include_linear_1=include_linear_1,
            include_linear_2=include_linear_2,
            include_linear_3=include_linear_3,
            basis_name=basis_name_val,
        )
        model = eqx.tree_at(lambda m: m.mean_model, model, new_mean_model)

        # ---- Step B: NN update (SGD on Fourier residuals) ----
        fourier_pred_train = new_mean_model.predict(phi1_train, phi2_train, phi3_train)
        fourier_pred_val = new_mean_model.predict(phi1_val, phi2_val, phi3_val)
        residuals_train = y_train - fourier_pred_train
        residuals_val = y_val - fourier_pred_val

        if nn is not None:
            optimizer = optax.adamw(nn_lr, weight_decay=nn_weight_decay)
            opt_state = optimizer.init(eqx.filter(nn, eqx.is_array))

            @eqx.filter_jit
            def loss_fn(nn, x, y_target):
                pred = jax.vmap(nn)(x).squeeze(-1)
                return jnp.mean((pred - y_target) ** 2)

            @eqx.filter_jit
            def step(nn, opt_state, x, y_target):
                loss, grads = eqx.filter_value_and_grad(loss_fn)(nn, x, y_target)
                updates, opt_state_new = optimizer.update(
                    grads, opt_state, eqx.filter(nn, eqx.is_array))
                nn_new = eqx.apply_updates(nn, updates)
                return nn_new, opt_state_new, loss

            n_batches = max(1, (N + nn_batch_size - 1) // nn_batch_size)
            best_val_loss = float('inf')
            best_nn = nn
            patience_counter = 0

            for epoch in range(nn_epochs_per_outer):
                key, subkey = jax.random.split(key)
                perm = jax.random.permutation(subkey, N)
                x_shuf = x_train[perm]
                r_shuf = residuals_train[perm]

                for b in range(n_batches):
                    s = b * nn_batch_size
                    e = min(s + nn_batch_size, N)
                    nn, opt_state, _ = step(nn, opt_state, x_shuf[s:e], r_shuf[s:e])

                vl = float(loss_fn(nn, x_val, residuals_val))
                if vl < best_val_loss:
                    best_val_loss = vl
                    best_nn = nn
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= nn_patience:
                        break

            nn = best_nn
            model = eqx.tree_at(lambda m: m.residual_net, model, nn)

        # ---- Logging ----
        if verbose:
            pred_val = new_mean_model.predict(phi1_val, phi2_val, phi3_val)
            if nn is not None:
                pred_val = pred_val + jax.vmap(nn)(x_val).squeeze(-1)
            rmse_val = float(jnp.sqrt(jnp.mean((y_val - pred_val) ** 2)))
            print(f"  Alternating iter {outer+1}/{n_outer}: RMSE_val={rmse_val:.4f}")

    return model
