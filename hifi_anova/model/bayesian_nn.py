"""Last-layer Bayesian treatment for the residual NN.

Provides cheap epistemic uncertainty from the residual network by treating
only the final linear layer as Bayesian (Gaussian posterior on weights),
while the hidden layers remain point estimates.

This gives per-input uncertainty at cost of one matrix-vector product:
  Var_NN(x*) = sigma^2 * z(x*)^T (Z^T Z + lambda_v I)^{-1} z(x*)

where z(x*) is the hidden feature vector from the penultimate layer.

Three types of prediction uncertainty become available:
  - Aleatoric:  sigma^2(x)  from the variance model (irreducible noise)
  - Fourier epistemic: sigma^2 * phi(x)^T (Phi^T Phi + R)^{-1} phi(x)
  - NN epistemic: sigma^2 * z(x)^T (Z^T Z + lambda_v I)^{-1} z(x)

Usage:
    bnn = BayesianLastLayer.from_trained_nn(model.residual_net, x_train, sigma2)
    mean, var_epistemic = bnn.predict(x_new)
    # or use the full uncertainty decomposition:
    unc = predict_with_uncertainty(model, x_new, bnn, fourier_posterior)
"""

import jax
import jax.numpy as jnp
import equinox as eqx
import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass


@dataclass
class BayesianLastLayer:
    """Bayesian treatment of the NN's final linear layer.

    Stores the posterior covariance of the last-layer weights and
    provides predictive uncertainty at new inputs.
    """
    # Posterior: v ~ N(v_map, sigma^2 * Sigma_v)
    # where Sigma_v = (Z^T Z + lambda_v * I)^{-1}
    v_map: np.ndarray          # (H,) MAP weights of last layer
    Sigma_v: np.ndarray        # (H, H) posterior precision^{-1}
    sigma2: float              # noise variance estimate
    lambda_v: float            # prior precision for last layer

    @classmethod
    def from_trained_nn(cls, nn: eqx.nn.MLP, x_train: jnp.ndarray,
                        sigma2: float, lambda_v: float = 0.01):
        """Construct Bayesian last layer from a trained NN.

        Extracts hidden features Z from all layers except the last,
        then computes the Bayesian posterior on the final linear mapping.

        Args:
            nn: trained eqx.nn.MLP
            x_train: (N, D) training inputs
            sigma2: estimated noise variance (from model or residual MSE)
            lambda_v: prior precision for last-layer weights (regularization)

        Returns:
            BayesianLastLayer instance
        """
        # Extract hidden features by running all layers except the last
        z_train = _extract_hidden_features(nn, x_train)  # (N, H)
        Z = np.array(z_train)
        N, H = Z.shape

        # Posterior covariance: Sigma_v = (Z^T Z + lambda_v * I)^{-1}
        ZTZ = Z.T @ Z
        A = ZTZ + lambda_v * np.eye(H)
        Sigma_v = np.linalg.inv(A)

        # MAP weights of last layer (already learned during training)
        v_map = np.array(_get_last_layer_weights(nn))  # (H,) or (H, 1)
        if v_map.ndim > 1:
            v_map = v_map.squeeze(-1)

        return cls(v_map=v_map, Sigma_v=Sigma_v, sigma2=sigma2, lambda_v=lambda_v)

    def predict(self, nn: eqx.nn.MLP, x_new: jnp.ndarray
                ) -> Tuple[np.ndarray, np.ndarray]:
        """Predict mean and epistemic variance at new inputs.

        Args:
            nn: the same trained NN (for hidden feature extraction)
            x_new: (M, D) new inputs

        Returns:
            mean: (M,) NN mean prediction (same as deterministic forward pass)
            var_epistemic: (M,) per-input epistemic variance from NN
        """
        z_new = np.array(_extract_hidden_features(nn, x_new))  # (M, H)

        # Mean prediction: v_map^T z (same as normal forward pass)
        mean = z_new @ self.v_map  # (M,)

        # Epistemic variance: sigma^2 * z^T Sigma_v z
        # Compute z^T Sigma_v z for each input
        Sigma_z = z_new @ self.Sigma_v  # (M, H)
        var_epistemic = self.sigma2 * np.sum(Sigma_z * z_new, axis=1)  # (M,)

        return mean, var_epistemic

    def predictive_std(self, nn: eqx.nn.MLP, x_new: jnp.ndarray) -> np.ndarray:
        """Return the epistemic standard deviation at new inputs."""
        _, var = self.predict(nn, x_new)
        return np.sqrt(np.maximum(var, 0.0))


@dataclass
class FourierPosterior:
    """Bayesian posterior on the Fourier coefficients (from ridge regression).

    The ridge solution is the MAP estimate under a Gaussian prior.
    The posterior covariance is sigma^2 * (Phi^T Phi + R)^{-1}.
    """
    Sigma_w: np.ndarray   # (F, F) posterior covariance / sigma^2
    sigma2: float         # noise variance

    @classmethod
    def from_ridge(cls, Phi: np.ndarray, reg_diag: np.ndarray, sigma2: float):
        """Compute Fourier posterior from the ridge solve quantities.

        Args:
            Phi: (N, F) feature matrix
            reg_diag: (F,) regularization diagonal
            sigma2: noise variance estimate

        Returns:
            FourierPosterior instance
        """
        A = Phi.T @ Phi + np.diag(reg_diag)
        Sigma_w = np.linalg.inv(A)  # (F, F)
        return cls(Sigma_w=Sigma_w, sigma2=sigma2)

    def predictive_variance(self, phi: np.ndarray) -> np.ndarray:
        """Epistemic variance of Fourier prediction at new points.

        Args:
            phi: (M, F) Fourier features at new inputs

        Returns:
            (M,) per-input epistemic variance from Fourier coefficients
        """
        # Var = sigma^2 * phi^T Sigma_w phi
        Sigma_phi = phi @ self.Sigma_w  # (M, F)
        return self.sigma2 * np.sum(Sigma_phi * phi, axis=1)  # (M,)


def predict_with_uncertainty(model, x_new: jnp.ndarray,
                             bayesian_nn: Optional[BayesianLastLayer] = None,
                             fourier_posterior: Optional[FourierPosterior] = None,
                             ) -> dict:
    """Full prediction with decomposed uncertainty.

    Returns the three-part uncertainty decomposition:
      - var_aleatoric: sigma^2(x) from variance model (irreducible)
      - var_fourier_epistemic: uncertainty in Fourier coefficients
      - var_nn_epistemic: uncertainty in NN predictions
      - var_total: sum of all three

    Args:
        model: fitted HiFiANOVA
        x_new: (M, D) new inputs
        bayesian_nn: optional BayesianLastLayer (None if no NN or no Bayesian)
        fourier_posterior: optional FourierPosterior (None skips Fourier epistemic)

    Returns:
        dict with mean, var_aleatoric, var_fourier_epistemic, var_nn_epistemic, var_total
    """
    # Standard prediction (mean + aleatoric variance)
    mean_pred, var_aleatoric = model.predict(x_new)
    mean_np = np.array(mean_pred)
    var_aleatoric_np = np.array(var_aleatoric)

    M = x_new.shape[0]

    # Fourier epistemic uncertainty
    var_fourier = np.zeros(M)
    if fourier_posterior is not None:
        phi_all = np.array(model.build_phi_all(x_new))
        var_fourier = fourier_posterior.predictive_variance(phi_all)

    # NN epistemic uncertainty
    var_nn = np.zeros(M)
    if bayesian_nn is not None and model.residual_net is not None:
        _, var_nn = bayesian_nn.predict(model.residual_net, x_new)

    var_total = var_aleatoric_np + var_fourier + var_nn

    return {
        'mean': mean_np,
        'var_aleatoric': var_aleatoric_np,
        'var_fourier_epistemic': var_fourier,
        'var_nn_epistemic': var_nn,
        'var_total': var_total,
        'std_total': np.sqrt(np.maximum(var_total, 0.0)),
    }


# --- Helper functions for extracting NN internals ---

def _extract_hidden_features(nn: eqx.nn.MLP, x: jnp.ndarray) -> jnp.ndarray:
    """Run all layers except the last to get hidden features.

    Args:
        nn: eqx.nn.MLP
        x: (N, D) inputs

    Returns:
        (N, H) hidden features from the penultimate layer
    """
    # eqx.nn.MLP stores layers as a list. We run all but the last.
    def forward_hidden(x_single):
        h = x_single
        layers = nn.layers
        # All layers except last: apply linear + activation
        for i, layer in enumerate(layers[:-1]):
            h = layer(h)
            h = nn.activation(h)
        return h

    return jax.vmap(forward_hidden)(x)


def _get_last_layer_weights(nn: eqx.nn.MLP) -> jnp.ndarray:
    """Extract weights of the last linear layer.

    Returns:
        (H, out_size) weight matrix of final layer
    """
    last_layer = nn.layers[-1]
    return last_layer.weight.T  # eqx stores as (out, in), we want (in, out)
