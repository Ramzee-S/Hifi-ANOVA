"""Residual diagnostics: analyze what the Fourier model didn't capture.

Use BEFORE training a residual NN to determine:
  - Is there remaining structure? (if not, NN will only overfit)
  - Is it in individual variables? (increase K)
  - Is it in variable pairs? (genuine higher-order interactions)
  - Is the residual homoscedastic? (variance model adequacy)

These diagnostics guide the decision tree:
  1. No structure → don't add NN
  2. Correlates with individual vars → increase K1
  3. Correlates with pairs → NN captures genuine interactions
  4. r^2 varies with inputs → add/improve variance model
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Optional, Dict
from dataclasses import dataclass


@dataclass
class ResidualDiagnostics:
    """Results from residual analysis."""
    # Per-variable correlation of r with x_i
    mean_correlations: Dict[int, float]
    # Per-variable correlation of r^2 with x_i (variance structure)
    variance_correlations: Dict[int, float]
    # Per-pair correlation of r with x_i*x_j (interaction signal)
    pair_correlations: Dict[tuple, float]
    # Spectral energy above K1 (fraction of residual power at high frequencies)
    high_freq_fraction: Dict[int, float]
    # Summary statistics
    residual_variance: float
    residual_fraction: float    # Var(r) / Var(y) — fraction of variance unexplained
    max_mean_correlation: float
    max_variance_correlation: float
    max_pair_correlation: float
    # Recommendations
    nn_recommended: bool
    increase_K_recommended: bool
    variance_model_recommended: bool
    recommendation: str


def analyze_residuals(model, x_data: jnp.ndarray, y_data: jnp.ndarray,
                      top_pairs: int = 10) -> ResidualDiagnostics:
    """Analyze Fourier residuals to determine if a residual NN is needed.

    Args:
        model: fitted HiFiANOVA (Fourier part only, no NN)
        x_data: (N, D) inputs in [0, 1]
        y_data: (N,) targets
        top_pairs: how many top interaction pairs to report

    Returns:
        ResidualDiagnostics with correlations, spectral info, and recommendations
    """
    N, D = x_data.shape

    # Compute Fourier predictions and residuals
    phi1 = model.build_phi1(x_data)
    phi2 = model.build_phi2(x_data)
    fourier_pred = model.mean_model.predict(phi1, phi2)
    residuals = np.array(y_data - fourier_pred)
    x_np = np.array(x_data)

    residual_var = float(np.var(residuals))
    total_var_y = float(np.var(np.array(y_data)))
    residual_frac = residual_var / total_var_y if total_var_y > 1e-10 else 0.0

    # --- Diagnostic 1: Correlation of r with each x_i ---
    mean_corrs = {}
    for i in range(D):
        corr = np.corrcoef(residuals, x_np[:, i])[0, 1]
        mean_corrs[i] = float(corr) if np.isfinite(corr) else 0.0

    # --- Diagnostic 2: Correlation of r^2 with each x_i (variance structure) ---
    r_squared = residuals ** 2
    var_corrs = {}
    for i in range(D):
        corr = np.corrcoef(r_squared, x_np[:, i])[0, 1]
        var_corrs[i] = float(corr) if np.isfinite(corr) else 0.0

    # --- Diagnostic 3: Correlation of r with x_i * x_j (interaction signal) ---
    pair_corrs = {}
    # Compute for all pairs, keep top ones
    all_pairs = []
    for i in range(D):
        for j in range(i + 1, D):
            product = (x_np[:, i] - 0.5) * (x_np[:, j] - 0.5)
            corr = np.corrcoef(residuals, product)[0, 1]
            c = float(corr) if np.isfinite(corr) else 0.0
            all_pairs.append(((i, j), abs(c)))
            pair_corrs[(i, j)] = c

    # Sort by absolute correlation, keep top
    all_pairs.sort(key=lambda x: -x[1])
    top_pair_corrs = {k: pair_corrs[k] for k, _ in all_pairs[:top_pairs]}

    # --- Diagnostic 4: Spectral analysis per variable ---
    # Check if residual has high-frequency content in each variable
    # by computing correlation with sin/cos at frequencies K1+1, K1+2, ...
    K_check = model.K1 + 3  # check a few frequencies above K1
    high_freq = {}
    for i in range(D):
        power_above = 0.0
        for k in range(model.K1 + 1, K_check + 1):
            cos_k = np.cos(2 * np.pi * k * x_np[:, i])
            sin_k = np.sin(2 * np.pi * k * x_np[:, i])
            power_above += np.corrcoef(residuals, cos_k)[0, 1] ** 2
            power_above += np.corrcoef(residuals, sin_k)[0, 1] ** 2
        high_freq[i] = float(power_above)

    # --- Recommendations ---
    max_mean_corr = max(abs(v) for v in mean_corrs.values())
    max_var_corr = max(abs(v) for v in var_corrs.values())
    max_pair_corr = max(abs(v) for v in pair_corrs.values()) if pair_corrs else 0.0
    max_high_freq = max(high_freq.values()) if high_freq else 0.0

    # Decision logic
    # Thresholds chosen to avoid false positives on pure-noise residuals:
    # with N=10000 and D=5, spurious pair correlations are typically <0.07.
    increase_K = max_high_freq > 0.05  # significant power above K1
    nn_needed = max_pair_corr > 0.10 or max_mean_corr > 0.1
    # Large unexplained variance (>20% of total) suggests higher-order structure
    # that may not correlate with any individual variable or pair
    # (e.g., purely 3rd-order terms like sin·sin·sin have zero 1st/2nd order projections)
    large_residual = residual_frac > 0.20
    if large_residual and not nn_needed:
        nn_needed = True
    var_model_needed = max_var_corr > 0.1

    parts = []
    if not nn_needed and not increase_K and not var_model_needed:
        parts.append("Residual appears unstructured (pure noise). No NN needed.")
    if increase_K:
        parts.append(f"High-frequency content detected above K={model.K1}. Consider increasing K1.")
    if max_mean_corr > 0.1:
        top_var = max(mean_corrs, key=lambda k: abs(mean_corrs[k]))
        parts.append(f"Residual correlates with x{top_var+1} (r={mean_corrs[top_var]:.3f}). "
                     "May indicate truncation or missing nonlinearity.")
    if max_pair_corr > 0.10:
        top_p = max(pair_corrs, key=lambda k: abs(pair_corrs[k]))
        parts.append(f"Residual correlates with (x{top_p[0]+1},x{top_p[1]+1}) interaction "
                     f"(r={pair_corrs[top_p]:.3f}). NN may capture higher-order effects.")
    if large_residual and max_pair_corr <= 0.10 and max_mean_corr <= 0.1:
        parts.append(f"Large unexplained variance ({residual_frac:.0%} of total) despite "
                     "weak variable/pair correlations. Likely higher-order interactions "
                     "(order 3+) that don't project onto individual variables or pairs. "
                     "NN recommended.")
    if var_model_needed:
        top_v = max(var_corrs, key=lambda k: abs(var_corrs[k]))
        parts.append(f"Residual variance correlates with x{top_v+1} (r={var_corrs[top_v]:.3f}). "
                     "Consider adding/strengthening the variance model.")

    recommendation = " ".join(parts) if parts else "Model appears adequate."

    return ResidualDiagnostics(
        mean_correlations=mean_corrs,
        variance_correlations=var_corrs,
        pair_correlations=top_pair_corrs,
        high_freq_fraction=high_freq,
        residual_variance=residual_var,
        residual_fraction=residual_frac,
        max_mean_correlation=max_mean_corr,
        max_variance_correlation=max_var_corr,
        max_pair_correlation=max_pair_corr,
        nn_recommended=nn_needed,
        increase_K_recommended=increase_K,
        variance_model_recommended=var_model_needed,
        recommendation=recommendation,
    )
