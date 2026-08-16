"""Analytic training pipeline for linear residual models (RBF, RFF, Nystrom).

Everything is ridge regression. Everything has GCV.
The feature-level projection keeps the fitted Fourier coefficients — and the
Sobol indices read off them — identical with or without the residual. (This
is an in-sample, coefficient-level guarantee; see core/projection.py for the
scope caveat vs. the population measure.)

The pipeline:
  1. Build Fourier features Phi
  2. Build residual features Z (from RBF/RFF/Nystrom)
  3. Project: Z_proj = Z - Phi @ solve(Phi^T Phi, Phi^T Z)
     After this: Phi^T @ Z_proj = 0 (exactly)
  4. Compute Fourier residuals: r = y - fourier_pred
  5. Ridge solve: alpha = ridge(Z_proj, r, lambda_res * I)
  6. Create fitted LinearResidual and attach to model

Because of the orthogonal projection, the Fourier coefficients
are IDENTICAL whether or not the residual is included. The ridge
system decouples into two independent solves.
"""

from ..array_backend import xp as jnp  # switchable array backend (numpy exact core)
import equinox as eqx
from typing import Tuple, Dict

from ..core.projection import project_features_orthogonal
from ..model.linear_residual import (
    RBFResidual, RFFResidual, NystromResidual, LinearResidualBase,
    predict_residual_batch,
)
from .ridge import weighted_ridge_solve


def create_residual(residual_type: str, residual_config: Dict,
                    x_train: jnp.ndarray, D: int,
                    key=None) -> LinearResidualBase:
    """Factory: create an unfitted linear residual model.

    Args:
        residual_type: 'rbf', 'rff', or 'nystrom'
        residual_config: dict with type-specific parameters (an optional
            ``'seed'`` key seeds k-means and the numpy-backend RFF/random
            draws; default 42)
        x_train: (N, D) training data for center selection
        D: input dimension
        key: PRNG key (jax backend only; the default k-means paths and the
            numpy backend never touch ``jax.random`` — BR-10)

    Returns:
        Unfitted LinearResidualBase subclass instance
    """
    seed = residual_config.get('seed', 42)

    if residual_type == 'rbf':
        return RBFResidual.create(
            x_train,
            n_centers=residual_config.get('n_centers', 300),
            sigma=residual_config.get('sigma', 0.2),
            method=residual_config.get('center_method', 'kmeans'),
            key=key,
            seed=seed,
        )
    elif residual_type == 'rff':
        return RFFResidual.create(
            D,
            n_features=residual_config.get('n_features', 1000),
            gamma=residual_config.get('gamma', 3.0),
            key=key,
            seed=seed,
        )
    elif residual_type == 'nystrom':
        return NystromResidual.create(
            x_train,
            n_inducing=residual_config.get('n_inducing', 300),
            lengthscale=residual_config.get('lengthscale', 0.2),
            kernel=residual_config.get('kernel', 'rbf'),
            signal_variance=residual_config.get('signal_variance', 1.0),
            key=key,
            seed=seed,
        )
    else:
        raise ValueError(f"Unknown residual type: {residual_type}")


def _build_full_fourier_features(x, model):
    """Build the low-order design in the SOLVED layout (BR-11).

    The projection targets the design the mean model was actually solved on
    (``build_phi_all_fit`` — the ``record.Phi`` layout): orthogonality is
    guaranteed to every solved column, nothing more. On a BR-06
    order-selective fit the excluded variables' first-order features are NOT
    projected out — the mean model never fitted them, so the complement is
    free to capture that structure. For a uniform fit (``fo_included`` unset)
    this is byte-identical to ``build_phi_all``. In the intercept-only limit
    (``fo_included=()``, no pairs) the design is (N, 0) and the projection is
    a no-op: the complement fits everything above f0.
    """
    return model.build_phi_all_fit(x)


def fit_linear_residual(
    model,
    x_train: jnp.ndarray,
    y_train: jnp.ndarray,
    x_val: jnp.ndarray,
    y_val: jnp.ndarray,
    residual_type: str,
    residual_config: Dict,
    lambda_residual: float = 1.0,
    key=None,
) -> Tuple:
    """Fit a linear residual model via the analytic pipeline.

    Steps:
      1. Create unfitted residual (select centers/frequencies)
      2. Build residual features Z
      3. Build Fourier features Phi
      4. Project Z orthogonal to Phi: Z_proj, C = project(Z, Phi)
      5. Compute Fourier residuals: r = y - fourier_pred
      6. Ridge solve: alpha = ridge(Z_proj, r, lambda_res * I)
      7. Create fitted residual with weights, proj_coeffs, Fourier config
      8. Attach to model

    Args:
        model: HiFiANOVA with fitted Fourier model (Stages A-B complete)
        x_train, y_train: training data
        x_val, y_val: validation data
        residual_type: 'rbf', 'rff', or 'nystrom'
        residual_config: type-specific parameters
        lambda_residual: regularization strength for residual
        key: PRNG key (jax backend only; may stay None — see create_residual)

    Returns:
        (updated_model, results_dict)
    """
    N, D = x_train.shape

    # 1. Create unfitted residual
    residual = create_residual(residual_type, residual_config, x_train, D, key)

    # 2. Build residual features
    Z_train = residual.build_features(x_train)  # (N, M)
    M = Z_train.shape[1]

    # 3. Build Fourier features
    Phi_train = _build_full_fourier_features(x_train, model)  # (N, F)

    # 4. Project Z orthogonal to Phi
    Z_train_proj, proj_coeffs = project_features_orthogonal(Z_train, Phi_train)

    # 5. Compute Fourier residuals
    phi1_train = model.build_phi1(x_train)
    phi1_val = model.build_phi1(x_val)
    phi2_train = model.build_phi2(x_train)
    phi2_val = model.build_phi2(x_val)
    phi3_train = model.build_phi3(x_train)
    phi3_val = model.build_phi3(x_val)

    fourier_pred_train = model.mean_model.predict(phi1_train, phi2_train, phi3_train)
    fourier_pred_val = model.mean_model.predict(phi1_val, phi2_val, phi3_val)
    residuals_train = y_train - fourier_pred_train

    # 6. Ridge solve on projected features
    reg_res = jnp.full(M, lambda_residual, dtype=jnp.float64)
    alpha = weighted_ridge_solve(
        jnp.asarray(Z_train_proj, dtype=jnp.float64),
        jnp.asarray(residuals_train, dtype=jnp.float64),
        reg_res,
    )

    # 7. Create fitted residual with Fourier config for prediction-time projection
    fitted_residual = _create_fitted_residual(
        residual, alpha, proj_coeffs, model
    )

    # 8. Attach to model
    updated_model = eqx.tree_at(
        lambda m: m.residual_net, model, fitted_residual,
        is_leaf=lambda x: x is None,
    )

    # Evaluate
    res_pred_val = predict_residual_batch(fitted_residual, x_val)
    total_pred_val = fourier_pred_val + res_pred_val
    rmse_val = float(jnp.sqrt(jnp.mean((y_val - total_pred_val) ** 2)))
    res_pred_train = predict_residual_batch(fitted_residual, x_train)
    res_variance = float(jnp.var(res_pred_train))

    results = {
        'rmse_val': rmse_val,
        'residual_type': residual_type,
        'n_residual_features': M,
        'residual_variance': res_variance,
        'lambda_residual': lambda_residual,
    }
    print(f"  RMSE val: {rmse_val:.4f} (residual var: {res_variance:.6f})")

    return updated_model, results


def _create_fitted_residual(unfitted, alpha, proj_coeffs, model):
    """Create a fitted residual by replacing weights and proj_coeffs,
    and storing the Fourier config for prediction-time projection."""

    # Build the constructor kwargs for the specific subclass.
    # Both weights and proj_coeffs stay at the fit's float64 precision (the
    # ridge solve is float64 on either backend): the historical float32 cast
    # of the weights was dropped with BR-10/DEC-057.
    common_kwargs = dict(
        weights=jnp.array(alpha, dtype=jnp.float64),
        proj_coeffs=jnp.array(proj_coeffs, dtype=jnp.float64),
        K1=model.K1,
        K2=model.K2,
        K3=model.K3,
        D=model.D,
        pair_indices=model.pair_indices,
        triple_indices=model.triple_indices,
        include_linear_1=getattr(model, 'include_linear_1', True),
        include_linear_2=getattr(model, 'include_linear_2', True),
        include_linear_3=getattr(model, 'include_linear_3', True),
        basis_name=getattr(model, 'basis_name', 'fourier'),
        # Solved-design layout (BR-11): the prediction-time projection must
        # rebuild the exact design the proj_coeffs were fitted against.
        var_specs=getattr(model, 'var_specs', None),
        pair_k2=getattr(model, 'pair_k2', None),
        fo_included=getattr(model, 'fo_included', None),
    )

    if isinstance(unfitted, RBFResidual):
        return RBFResidual(
            **common_kwargs,
            centers=unfitted.centers,
            sigma=unfitted.sigma,
        )
    elif isinstance(unfitted, RFFResidual):
        return RFFResidual(
            **common_kwargs,
            omega=unfitted.omega,
            bias=unfitted.bias,
            scale=unfitted.scale,
        )
    elif isinstance(unfitted, NystromResidual):
        return NystromResidual(
            **common_kwargs,
            inducing_points=unfitted.inducing_points,
            lengthscale=unfitted.lengthscale,
            kernel_type=unfitted.kernel_type,
            signal_variance=unfitted.signal_variance,
        )
    else:
        raise TypeError(f"Unknown residual type: {type(unfitted)}")
