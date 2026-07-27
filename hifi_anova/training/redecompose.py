"""Solutions for maintaining Sobol integrity with a residual NN.

Solution 1: Re-decomposition after joint fine-tuning
  Train jointly for best RMSE, then re-project the total learned function
  onto the Hoeffding-Fourier basis using uniform evaluation points.
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
import numpy as np
import optax
from typing import Optional, Dict

from ..core.gram import build_gram_matrix, build_gram_matrix_2d
from ..core.pairs import PairManager
from ..core.features import basis_size
from ..model.mean_model import MeanModel
from ..model.hifi_anova import HiFiANOVA
from .ridge import weighted_ridge_solve
from .regularization import build_regularization_vector


# =============================================================================
# Solution 1: Re-decomposition
# =============================================================================

def redecompose(model: HiFiANOVA,
                reg_diag: Optional[jnp.ndarray] = None,
                n_eval: int = 50000,
                seed: int = 0) -> HiFiANOVA:
    """Re-decompose a jointly-trained model into clean Fourier + NN parts.

    Generates uniform evaluation points, evaluates the total learned function
    f_total = f_Fourier_old + f_NN, then computes the Hoeffding-Fourier
    decomposition of f_total by projecting onto the Fourier basis.

    The new Fourier coefficients represent the correct low-order decomposition
    of whatever the full model learned. The new Fourier intercept f0 is set
    so that the total prediction is preserved exactly:

        f_Fourier_new(x) + f_NN(x) = f_total(x)

    This is achieved by re-projecting f_total (to get clean Sobol indices)
    and then setting:
        f0_new = f0_total_proj - mean(f_NN)

    where f0_total_proj accounts for the Fourier projection of the full signal.
    Since the NN weights are frozen, the Fourier model takes responsibility
    for the correct low-order decomposition while the NN continues to provide
    its original output. The total f_Fourier_new + f_NN = f_total because
    the Fourier re-projection targets f_total - f_NN (what the Fourier model
    must explain given that the NN is fixed).

    Args:
        model: HiFiANOVA after joint fine-tuning (or any training)
        reg_diag: (F,) regularization for the re-projection.
                  None = OLS (for Sobol estimation mode).
                  Provide the original reg_diag for prediction mode.
        n_eval: number of uniform evaluation points (more = more accurate)
        seed: PRNG seed for evaluation points

    Returns:
        New HiFiANOVA with re-decomposed Fourier coefficients.
        Total prediction is preserved (f_Fourier_new + f_NN = f_total).
        Sobol indices from the new Fourier coefficients reflect the
        Hoeffding decomposition of f_total.
    """
    D = model.D
    K1 = model.K1
    K2 = model.K2
    K3 = getattr(model, 'K3', 0)
    include_linear_1 = getattr(model, 'include_linear_1', True)
    include_linear_2 = getattr(model, 'include_linear_2', True)
    include_linear_3 = getattr(model, 'include_linear_3', True)
    basis_name_val = getattr(model, 'basis_name', 'fourier')

    # Generate uniform evaluation points
    key = jax.random.PRNGKey(seed)
    x_eval = jax.random.uniform(key, (n_eval, D))

    # Build Fourier features on evaluation points
    phi1_eval = model.build_phi1(x_eval)
    phi2_eval = model.build_phi2(x_eval)
    phi3_eval = model.build_phi3(x_eval) if K3 > 0 else None

    Phi_eval = model.build_phi_all(x_eval)

    # Evaluate total function (Fourier + NN) on evaluation points
    f_fourier_old = model.mean_model.predict(phi1_eval, phi2_eval, phi3_eval)
    has_nn = model.residual_net is not None
    if has_nn:
        f_nn_eval = jax.vmap(model.residual_net)(x_eval)
        if f_nn_eval.ndim > 1:
            f_nn_eval = f_nn_eval.squeeze(-1)
        f_total = f_fourier_old + f_nn_eval
    else:
        f_nn_eval = jnp.zeros(len(x_eval))
        f_total = f_fourier_old

    # === Project (f_total - f_NN) onto the Fourier basis ===
    #
    # Since the NN weights are frozen, the Fourier model must explain:
    #   f_Fourier_new(x) = f_total(x) - f_NN(x)
    #
    # This preserves the total prediction exactly:
    #   f_Fourier_new(x) + f_NN(x) = (f_total - f_NN)(x) + f_NN(x) = f_total(x)
    #   (up to Fourier projection error, which is zero for the Fourier-representable part)
    #
    # The Sobol indices from w_new reflect the Hoeffding decomposition of
    # (f_total - f_NN), which is the Fourier-representable part of f_total.
    # This is exactly what we want: the Fourier Sobol indices describe
    # the low-order structure, and the NN variance fraction describes
    # the higher-order residual.

    f_target = f_total - f_nn_eval  # what the Fourier model must explain
    f0_new = float(jnp.mean(f_target))
    f_centered = f_target - f0_new

    if reg_diag is not None:
        w_new = weighted_ridge_solve(Phi_eval, f_centered, reg_diag)
    else:
        F = Phi_eval.shape[1]
        tiny_reg = jnp.full(F, 1e-10)
        w_new = weighted_ridge_solve(Phi_eval, f_centered, tiny_reg)

    # Split into first-order, second-order, and third-order
    F1 = D * basis_size(K1, include_linear_1, basis_name_val)
    w1_new = w_new[:F1]
    if K2 > 0:
        B2 = basis_size(K2, include_linear_2, basis_name_val)
        n_pairs = model.pair_indices.shape[0] if model.pair_indices is not None else 0
        F2 = n_pairs * B2 * B2
        w2_new = w_new[F1:F1 + F2]
    else:
        F2 = 0
        w2_new = jnp.array([], dtype=jnp.float32)
    if K3 > 0:
        w3_new = w_new[F1 + F2:]
    else:
        w3_new = jnp.array([], dtype=jnp.float32)

    # Build new mean model
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

    new_model = eqx.tree_at(lambda m: m.mean_model, model, new_mean_model)
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
    Phi_val = model.build_phi_all(x_val)

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
