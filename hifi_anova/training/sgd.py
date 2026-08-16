"""SGD/Adam trainer for residual NN and joint fine-tuning.

Two training modes for the residual NN:
  - Standard (default): train on Fourier residuals. Approximate orthogonality.
  - Projected (orthogonal=True): project NN output away from Fourier subspace
    each forward pass. Exact orthogonality guarantee.

The projected mode is recommended when:
  - Joint fine-tuning will follow (prevents NN drifting into Fourier space)
  - Sobol index integrity is critical (NN cannot corrupt Fourier attributions)
"""

import jax
import jax.numpy as jnp
import equinox as eqx
import optax
from typing import Optional


def train_residual_nn(
    model,
    x_train: jnp.ndarray,
    y_train: jnp.ndarray,
    x_val: jnp.ndarray,
    y_val: jnp.ndarray,
    lr: float = 0.001,
    weight_decay: float = 0.0001,
    epochs: int = 200,
    batch_size: int = 512,
    patience: int = 20,
    key: jax.Array = None,
):
    """Train the residual NN on Fourier residuals.

    Minimizes MSE on residuals (what the Fourier model couldn't capture).

    Args:
        model: HiFiANOVA instance (residual_net will be trained)
        x_train, y_train: training data (x already transformed)
        x_val, y_val: validation data
        lr: learning rate
        weight_decay: AdamW weight decay
        epochs: max epochs
        batch_size: mini-batch size
        patience: early stopping patience
        key: PRNG key

    Returns:
        Updated model with trained residual_net
    """
    if key is None:
        key = jax.random.PRNGKey(0)

    # Compute Fourier predictions (targets for NN are residuals)
    phi1_train = model.build_phi1(x_train)
    phi1_val = model.build_phi1(x_val)

    phi2_train = model.build_phi2(x_train)
    phi2_val = model.build_phi2(x_val)

    phi3_train = model.build_phi3(x_train) if model.K3 > 0 else None
    phi3_val = model.build_phi3(x_val) if model.K3 > 0 else None

    fourier_pred_train = model.mean_model.predict(phi1_train, phi2_train, phi3_train)
    fourier_pred_val = model.mean_model.predict(phi1_val, phi2_val, phi3_val)

    # Residuals (NN target)
    residuals_train = y_train - fourier_pred_train
    residuals_val = y_val - fourier_pred_val

    # Extract NN for training
    nn = model.residual_net

    # Optimizer
    optimizer = optax.adamw(lr, weight_decay=weight_decay)
    opt_state = optimizer.init(eqx.filter(nn, eqx.is_array))

    @eqx.filter_jit
    def loss_fn(nn, x, y_target):
        pred = jax.vmap(nn)(x).squeeze(-1)
        return jnp.mean((pred - y_target) ** 2)

    @eqx.filter_jit
    def step(nn, opt_state, x, y_target):
        loss, grads = eqx.filter_value_and_grad(loss_fn)(nn, x, y_target)
        updates, opt_state_new = optimizer.update(
            grads, opt_state, eqx.filter(nn, eqx.is_array)
        )
        nn_new = eqx.apply_updates(nn, updates)
        return nn_new, opt_state_new, loss

    # Training loop
    N = x_train.shape[0]
    n_batches = max(1, (N + batch_size - 1) // batch_size)
    best_val_loss = float('inf')
    best_nn = nn
    patience_counter = 0

    for epoch in range(epochs):
        # Shuffle
        key, subkey = jax.random.split(key)
        perm = jax.random.permutation(subkey, N)
        x_shuffled = x_train[perm]
        r_shuffled = residuals_train[perm]

        # Mini-batch training
        epoch_loss = 0.0
        for b in range(n_batches):
            start = b * batch_size
            end = min(start + batch_size, N)
            x_batch = x_shuffled[start:end]
            r_batch = r_shuffled[start:end]
            nn, opt_state, batch_loss = step(nn, opt_state, x_batch, r_batch)
            epoch_loss += float(batch_loss)

        # Validation
        val_loss = float(loss_fn(nn, x_val, residuals_val))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_nn = nn
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    # Update model with best NN
    model = eqx.tree_at(lambda m: m.residual_net, model, best_nn)
    return model


def train_residual_nn_projected(
    model,
    x_train: jnp.ndarray,
    y_train: jnp.ndarray,
    x_val: jnp.ndarray,
    y_val: jnp.ndarray,
    reg_diag: jnp.ndarray,
    lr: float = 0.001,
    weight_decay: float = 0.0001,
    epochs: int = 200,
    batch_size: int = 512,
    patience: int = 20,
    key: jax.Array = None,
):
    """Train residual NN with orthogonal projection against Fourier subspace.

    Same as train_residual_nn, but each forward pass projects the NN output
    orthogonal to the Fourier feature space. This guarantees that the NN
    cannot learn anything already representable by the Fourier basis.

    The projection is differentiable — gradients tell the NN to focus
    exclusively on patterns outside the Fourier subspace.

    Args:
        model: HiFiANOVA with residual_net already attached
        x_train, y_train: training data
        x_val, y_val: validation data
        reg_diag: (F,) regularization diagonal from the Fourier ridge solve
        lr, weight_decay, epochs, batch_size, patience: training config
        key: PRNG key

    Returns:
        Updated model with trained (orthogonalized) residual_net
    """
    from .projection import FourierProjector, build_batch_features

    if key is None:
        key = jax.random.PRNGKey(0)

    # Build projector from training data
    projector = FourierProjector(
        build_batch_features(x_train, model),
        reg_diag
    )

    # Compute targets (full y, not just residuals — projection handles separation)
    phi1_train = model.build_phi1(x_train)
    phi1_val = model.build_phi1(x_val)
    phi2_train = model.build_phi2(x_train)
    phi2_val = model.build_phi2(x_val)

    phi3_train = model.build_phi3(x_train) if model.K3 > 0 else None
    phi3_val = model.build_phi3(x_val) if model.K3 > 0 else None

    fourier_pred_train = model.mean_model.predict(phi1_train, phi2_train, phi3_train)
    fourier_pred_val = model.mean_model.predict(phi1_val, phi2_val, phi3_val)
    residuals_train = y_train - fourier_pred_train
    residuals_val = y_val - fourier_pred_val

    nn = model.residual_net
    optimizer = optax.adamw(lr, weight_decay=weight_decay)
    opt_state = optimizer.init(eqx.filter(nn, eqx.is_array))

    @eqx.filter_jit
    def loss_fn(nn, x, y_target):
        nn_raw = jax.vmap(nn)(x).squeeze(-1)
        # Project out Fourier component
        Phi_batch = build_batch_features(x, model)
        nn_proj = projector.project(nn_raw, Phi_batch)
        return jnp.mean((nn_proj - y_target) ** 2)

    @eqx.filter_jit
    def step(nn, opt_state, x, y_target):
        loss, grads = eqx.filter_value_and_grad(loss_fn)(nn, x, y_target)
        updates, opt_state_new = optimizer.update(
            grads, opt_state, eqx.filter(nn, eqx.is_array)
        )
        nn_new = eqx.apply_updates(nn, updates)
        return nn_new, opt_state_new, loss

    N = x_train.shape[0]
    n_batches = max(1, (N + batch_size - 1) // batch_size)
    best_val_loss = float('inf')
    best_nn = nn
    patience_counter = 0

    for epoch in range(epochs):
        key, subkey = jax.random.split(key)
        perm = jax.random.permutation(subkey, N)
        x_shuffled = x_train[perm]
        r_shuffled = residuals_train[perm]

        for b in range(n_batches):
            start = b * batch_size
            end = min(start + batch_size, N)
            nn, opt_state, _ = step(nn, opt_state,
                                    x_shuffled[start:end], r_shuffled[start:end])

        val_loss = float(loss_fn(nn, x_val, residuals_val))
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_nn = nn
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    model = eqx.tree_at(lambda m: m.residual_net, model, best_nn)
    return model


def joint_finetune(
    model,
    x_train: jnp.ndarray,
    y_train: jnp.ndarray,
    x_val: jnp.ndarray,
    y_val: jnp.ndarray,
    lr: float = 0.0001,
    epochs: int = 50,
    batch_size: int = 512,
    key: jax.Array = None,
    orthogonal: bool = False,
    reg_diag: Optional[jnp.ndarray] = None,
):
    """Joint fine-tuning of ALL parameters (Fourier + NN + variance).

    Small learning rate. Short duration.

    Args:
        orthogonal: if True, project NN output away from Fourier subspace
                    during fine-tuning (prevents NN from corrupting Sobol indices)
        reg_diag: required if orthogonal=True; the Fourier regularization vector
    """
    if key is None:
        key = jax.random.PRNGKey(1)

    # Set up optional projection
    projector = None
    if orthogonal and reg_diag is not None and model.residual_net is not None:
        from .projection import FourierProjector, build_batch_features
        projector = FourierProjector(build_batch_features(x_train, model), reg_diag)

    optimizer = optax.adam(lr)
    # Filter to floating-point parameters only: the model now carries integer
    # index buffers (pair_indices/triple_indices) as dynamic leaves, and those must
    # NOT be optimized. is_inexact_array selects the trainable float params
    # (Fourier + NN + variance) and excludes the integer index buffers.
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

    @eqx.filter_jit
    def loss_fn_standard(model, x, y):
        mean, var = model.predict(x)
        if model.variance_model is not None:
            log_var = jnp.log(var)
            nll = 0.5 * log_var + 0.5 * (y - mean) ** 2 / var
            return jnp.mean(nll)
        else:
            return jnp.mean((y - mean) ** 2)

    @eqx.filter_jit
    def loss_fn_projected(model, x, y):
        """Loss with NN output projected orthogonal to Fourier subspace."""
        phi1 = model.build_phi1(x)
        phi2 = model.build_phi2(x)
        phi3 = model.build_phi3(x)

        fourier_mean = model.mean_model.predict(phi1, phi2, phi3)

        # NN output, projected
        nn_raw = jax.vmap(model.residual_net)(x).squeeze(-1)
        Phi_batch = model.build_phi_all(x)
        nn_proj = projector.project(nn_raw, Phi_batch)

        mean = fourier_mean + nn_proj

        if model.variance_model is not None:
            psi1 = model.build_psi1(x)
            log_var = model.variance_model.predict_log_variance(psi1)
            var = jnp.exp(log_var)
            nll = 0.5 * log_var + 0.5 * (y - mean) ** 2 / var
            return jnp.mean(nll)
        else:
            return jnp.mean((y - mean) ** 2)

    loss_fn = loss_fn_projected if projector is not None else loss_fn_standard

    @eqx.filter_jit
    def step(model, opt_state, x, y):
        loss, grads = eqx.filter_value_and_grad(loss_fn)(model, x, y)
        updates, opt_state_new = optimizer.update(
            grads, opt_state, eqx.filter(model, eqx.is_inexact_array)
        )
        model_new = eqx.apply_updates(model, updates)
        return model_new, opt_state_new, loss

    N = x_train.shape[0]
    n_batches = max(1, (N + batch_size - 1) // batch_size)
    best_val_loss = float('inf')
    best_model = model

    for epoch in range(epochs):
        key, subkey = jax.random.split(key)
        perm = jax.random.permutation(subkey, N)
        x_shuffled = x_train[perm]
        y_shuffled = y_train[perm]

        for b in range(n_batches):
            start = b * batch_size
            end = min(start + batch_size, N)
            model, opt_state, _ = step(model, opt_state,
                                       x_shuffled[start:end],
                                       y_shuffled[start:end])

        val_loss = float(loss_fn(model, x_val, y_val))
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model

    return best_model
