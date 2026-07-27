"""Regularization path analysis.

Computes model quantities (Sobol indices, variance decomposition, effective df,
GCV, evidence) along a grid of regularization strengths. Provides the data for
L-curve, Sobol path, and variance decomposition path plots.
"""

import numpy as np
import jax.numpy as jnp
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from ..core.gram import build_gram_matrix, build_gram_matrix_2d, build_gram_matrix_3d
from ..core.features import build_first_order_features, build_second_order_features, basis_size
from ..core.pairs import PairManager
from ..training.hyperopt import ridge_solve_with_diagnostics
from ..training.regularization import build_regularization_vector


@dataclass
class RegPathResult:
    """Results of a regularization path sweep."""
    lambdas: np.ndarray             # (n_lambda,) regularization values
    mse_values: np.ndarray          # (n_lambda,) training MSE
    gcv_values: np.ndarray          # (n_lambda,) GCV scores
    aic_values: np.ndarray          # (n_lambda,) AIC
    bic_values: np.ndarray          # (n_lambda,) BIC
    evidence_values: np.ndarray     # (n_lambda,) log marginal likelihood
    df_values: np.ndarray           # (n_lambda,) effective degrees of freedom
    sobol_paths: Dict[int, np.ndarray]        # {var_i: (n_lambda,)} first-order Sobol
    sobol_paths_2nd: Dict[Tuple, np.ndarray]  # {(i,j): (n_lambda,)} second-order Sobol
    sobol_paths_3rd: Dict[Tuple, np.ndarray]  # {(i,j,k): (n_lambda,)} third-order Sobol
    var_order1: np.ndarray          # (n_lambda,) total first-order variance
    var_order2: np.ndarray          # (n_lambda,) total second-order variance
    var_order3: np.ndarray          # (n_lambda,) total third-order variance
    var_residual: np.ndarray        # (n_lambda,) residual variance (if present)
    var_total: np.ndarray           # (n_lambda,) total model variance
    w_norm: np.ndarray              # (n_lambda,) ||w||^2
    lambda_gcv_opt: float           # GCV-optimal lambda
    lambda_evidence_opt: float      # evidence-optimal lambda


def compute_reg_path(
    Phi: np.ndarray,
    y: np.ndarray,
    D: int,
    K1: int,
    K2: int = 0,
    P: int = 0,
    pair_indices: Optional[np.ndarray] = None,
    K3: int = 0,
    T: int = 0,
    triple_indices: Optional[np.ndarray] = None,
    M_residual: int = 0,
    strategy: str = 'variance',
    lambda_ratio: float = 10.0,
    lambda_ratio_3: float = 100.0,
    lambda_ratio_res: float = 1000.0,
    n_lambdas: int = 50,
    lambda_range: Tuple[float, float] = (1e-6, 1e2),
    include_linear_1: bool = True,
    include_linear_2: bool = True,
    include_linear_3: bool = True,
    basis_name: str = 'fourier',
) -> RegPathResult:
    """Compute the full regularization path with third-order and residual.

    Sweeps lambda_1 over a log-spaced grid with:
      lambda_2 = lambda_ratio * lambda_1
      lambda_3 = lambda_ratio_3 * lambda_1
      lambda_res = lambda_ratio_res * lambda_1

    Args:
        Phi: (N, F) full feature matrix [Phi1 | Phi2 | Phi3 | Z_proj]
        y: (N,) centered targets
        D, K1, K2, P, K3, T, M_residual: model structure
        pair_indices, triple_indices: group index arrays
        strategy: regularization strategy
        lambda_ratio: lambda_2 / lambda_1
        lambda_ratio_3: lambda_3 / lambda_1
        lambda_ratio_res: lambda_res / lambda_1
        n_lambdas, lambda_range: sweep grid parameters
        include_linear_1: whether first-order basis includes linear term
        include_linear_2: whether second-order basis includes linear term
        include_linear_3: whether third-order basis includes linear term
        basis_name: 'fourier' or 'legendre'

    Returns:
        RegPathResult with all computed quantities including third-order
        and residual paths.
    """
    Phi_np = np.asarray(Phi, dtype=np.float64)
    y_np = np.asarray(y, dtype=np.float64)
    N, F = Phi_np.shape

    # Gram matrices for Sobol computation (basis-aware)
    G1 = np.asarray(build_gram_matrix(K1, include_linear_1, basis_name),
                    dtype=np.float64)
    G2 = (np.asarray(build_gram_matrix_2d(
               build_gram_matrix(K2, include_linear_2, basis_name)),
               dtype=np.float64)
           if K2 > 0 else None)
    G3 = (np.asarray(build_gram_matrix_3d(
               build_gram_matrix(K3, include_linear_3, basis_name)),
               dtype=np.float64)
           if K3 > 0 else None)

    # Block sizes via basis_size (correct for Fourier, Legendre, spectral)
    block1 = basis_size(K1, include_linear_1, basis_name)
    block2 = basis_size(K2, include_linear_2, basis_name) ** 2 if K2 > 0 else 0
    block3 = basis_size(K3, include_linear_3, basis_name) ** 3 if K3 > 0 else 0
    F1 = D * block1
    F2 = P * block2
    F3 = T * block3

    # Lambda grid
    lambdas = np.logspace(np.log10(lambda_range[0]),
                          np.log10(lambda_range[1]), n_lambdas)

    # Storage
    mse_vals = np.zeros(n_lambdas)
    gcv_vals = np.zeros(n_lambdas)
    aic_vals = np.zeros(n_lambdas)
    bic_vals = np.zeros(n_lambdas)
    evidence_vals = np.zeros(n_lambdas)
    df_vals = np.zeros(n_lambdas)
    var_order1_vals = np.zeros(n_lambdas)
    var_order2_vals = np.zeros(n_lambdas)
    var_order3_vals = np.zeros(n_lambdas)
    var_residual_vals = np.zeros(n_lambdas)
    var_total_vals = np.zeros(n_lambdas)
    w_norm_vals = np.zeros(n_lambdas)

    sobol_1st = {i: np.zeros(n_lambdas) for i in range(D)}
    sobol_2nd = {}
    if K2 > 0 and pair_indices is not None:
        for p in range(P):
            i, j = int(pair_indices[p, 0]), int(pair_indices[p, 1])
            sobol_2nd[(i, j)] = np.zeros(n_lambdas)
    sobol_3rd = {}
    if K3 > 0 and triple_indices is not None:
        for t in range(T):
            i, j, k = (int(triple_indices[t, l]) for l in range(3))
            sobol_3rd[(i, j, k)] = np.zeros(n_lambdas)

    for idx, lam1 in enumerate(lambdas):
        lam2 = lam1 * lambda_ratio
        lam3 = lam1 * lambda_ratio_3
        lam_res = lam1 * lambda_ratio_res

        reg = np.asarray(
            build_regularization_vector(
                D, K1, K2, P, strategy, lam1, lam2,
                K3=K3, T=T, lambda_order3=lam3,
                M_residual=M_residual, lambda_residual=lam_res,
                include_linear_1=include_linear_1,
                include_linear_2=include_linear_2,
                include_linear_3=include_linear_3,
                basis_name=basis_name,
            ),
            dtype=np.float64
        )

        diag = ridge_solve_with_diagnostics(Phi_np, y_np, reg)
        mse_vals[idx] = diag['mse']
        gcv_vals[idx] = diag['gcv']
        aic_vals[idx] = diag['aic']
        bic_vals[idx] = diag['bic']
        evidence_vals[idx] = diag['log_evidence']
        df_vals[idx] = diag['df']
        w = diag['w']
        w_norm_vals[idx] = np.sum(w ** 2)

        total_var = 0.0

        # First-order
        for i in range(D):
            wi = w[i * block1: (i + 1) * block1]
            var_i = max(0.0, float(wi @ G1 @ wi))
            sobol_1st[i][idx] = var_i
            total_var += var_i
        var_order1_vals[idx] = total_var

        # Second-order
        var_o2 = 0.0
        if K2 > 0 and pair_indices is not None:
            for p in range(P):
                wp = w[F1 + p * block2: F1 + (p + 1) * block2]
                var_p = max(0.0, float(wp @ G2 @ wp))
                i, j = int(pair_indices[p, 0]), int(pair_indices[p, 1])
                sobol_2nd[(i, j)][idx] = var_p
                var_o2 += var_p
        var_order2_vals[idx] = var_o2
        total_var += var_o2

        # Third-order
        var_o3 = 0.0
        if K3 > 0 and triple_indices is not None:
            for t in range(T):
                wt = w[F1 + F2 + t * block3: F1 + F2 + (t + 1) * block3]
                var_t = max(0.0, float(wt @ G3 @ wt))
                i, j, k = (int(triple_indices[t, l]) for l in range(3))
                sobol_3rd[(i, j, k)][idx] = var_t
                var_o3 += var_t
        var_order3_vals[idx] = var_o3
        total_var += var_o3

        # Residual (empirical variance of residual features * alpha)
        var_res = 0.0
        if M_residual > 0:
            alpha = w[F1 + F2 + F3:]
            Z_proj_block = Phi_np[:, F1 + F2 + F3:]
            res_pred = Z_proj_block @ alpha
            var_res = float(np.var(res_pred))
        var_residual_vals[idx] = var_res
        total_var += var_res

        var_total_vals[idx] = total_var

        # Normalize Sobol to fractions
        if total_var > 0:
            for i in range(D):
                sobol_1st[i][idx] /= total_var
            for key in sobol_2nd:
                sobol_2nd[key][idx] /= total_var
            for key in sobol_3rd:
                sobol_3rd[key][idx] /= total_var

    gcv_opt_idx = int(np.argmin(gcv_vals))
    evidence_opt_idx = int(np.argmax(evidence_vals))

    return RegPathResult(
        lambdas=lambdas,
        mse_values=mse_vals,
        gcv_values=gcv_vals,
        aic_values=aic_vals,
        bic_values=bic_vals,
        evidence_values=evidence_vals,
        df_values=df_vals,
        sobol_paths=sobol_1st,
        sobol_paths_2nd=sobol_2nd,
        sobol_paths_3rd=sobol_3rd,
        var_order1=var_order1_vals,
        var_order2=var_order2_vals,
        var_order3=var_order3_vals,
        var_residual=var_residual_vals,
        var_total=var_total_vals,
        w_norm=w_norm_vals,
        lambda_gcv_opt=lambdas[gcv_opt_idx],
        lambda_evidence_opt=lambdas[evidence_opt_idx],
    )


def plot_reg_path(path: RegPathResult,
                  variable_names: Optional[List[str]] = None,
                  save_prefix: Optional[str] = None,
                  show_top_n: int = 8):
    """Generate the standard regularization path plots.

    Creates 4 plots:
    1. L-curve (MSE vs df)
    2. GCV/Evidence vs lambda (model selection criteria)
    3. Sobol index paths
    4. Variance decomposition (stacked area)
    """
    import matplotlib.pyplot as plt

    D = len(path.sobol_paths)
    if variable_names is None:
        variable_names = [f"x{i+1}" for i in range(D)]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # --- Plot 1: L-curve (MSE vs effective df) ---
    ax = axes[0, 0]
    ax.loglog(path.df_values, path.mse_values, 'b-o', markersize=3)
    # Mark GCV optimal
    gcv_idx = np.argmin(np.abs(path.lambdas - path.lambda_gcv_opt))
    ax.plot(path.df_values[gcv_idx], path.mse_values[gcv_idx],
            'r*', markersize=15, label=f'GCV opt (df={path.df_values[gcv_idx]:.1f})')
    ax.set_xlabel('Effective Degrees of Freedom')
    ax.set_ylabel('Training MSE')
    ax.set_title('L-Curve')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Plot 2: Model selection criteria ---
    ax = axes[0, 1]
    ax2 = ax.twinx()
    l1 = ax.semilogx(path.lambdas, path.gcv_values, 'b-', label='GCV')
    l2 = ax2.semilogx(path.lambdas, path.evidence_values, 'r-', label='Evidence')
    ax.axvline(path.lambda_gcv_opt, color='b', linestyle='--', alpha=0.5)
    ax.axvline(path.lambda_evidence_opt, color='r', linestyle='--', alpha=0.5)
    ax.set_xlabel(r'$\lambda$')
    ax.set_ylabel('GCV', color='b')
    ax2.set_ylabel('Log Evidence', color='r')
    ax.set_title('Model Selection Criteria')
    lines = l1 + l2
    ax.legend(lines, [l.get_label() for l in lines])
    ax.grid(True, alpha=0.3)

    # --- Plot 3: Sobol index paths ---
    ax = axes[1, 0]
    # Find top-N variables by max Sobol across the path
    max_sobol = {i: np.max(path.sobol_paths[i]) for i in range(D)}
    top_vars = sorted(max_sobol.keys(), key=lambda i: -max_sobol[i])[:show_top_n]

    for i in top_vars:
        ax.semilogx(path.lambdas, path.sobol_paths[i], '-',
                    label=variable_names[i], linewidth=1.5)

    # Add top second-order interactions
    if path.sobol_paths_2nd:
        max_2nd = {k: np.max(v) for k, v in path.sobol_paths_2nd.items()}
        top_pairs = sorted(max_2nd.keys(), key=lambda k: -max_2nd[k])[:3]
        for (i, j) in top_pairs:
            if max_2nd[(i, j)] > 0.01:
                ax.semilogx(path.lambdas, path.sobol_paths_2nd[(i, j)], '--',
                           label=f"({variable_names[i]},{variable_names[j]})",
                           linewidth=1.5)

    # Add top third-order interactions
    if path.sobol_paths_3rd:
        max_3rd = {k: np.max(v) for k, v in path.sobol_paths_3rd.items()}
        top_triples = sorted(max_3rd.keys(), key=lambda k: -max_3rd[k])[:3]
        for (i, j, k) in top_triples:
            if max_3rd[(i, j, k)] > 0.01:
                ax.semilogx(path.lambdas, path.sobol_paths_3rd[(i, j, k)], ':',
                           label=f"({variable_names[i]},{variable_names[j]},{variable_names[k]})",
                           linewidth=1.5)

    ax.axvline(path.lambda_gcv_opt, color='gray', linestyle='--', alpha=0.5,
               label=r'$\lambda^*_{GCV}$')
    ax.set_xlabel(r'$\lambda$')
    ax.set_ylabel('Sobol Index')
    ax.set_title('Sobol Index Paths')
    ax.legend(loc='upper right', fontsize=8)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    # --- Plot 4: Variance decomposition (stacked area) ---
    ax = axes[1, 1]
    # Stacked: order1, order2, order3, residual
    cum1 = path.var_order1
    cum2 = cum1 + path.var_order2
    cum3 = cum2 + path.var_order3
    cum4 = cum3 + path.var_residual

    ax.fill_between(path.lambdas, 0, cum1,
                    alpha=0.6, label='First-order', color='steelblue')
    ax.fill_between(path.lambdas, cum1, cum2,
                    alpha=0.6, label='Second-order', color='coral')
    if np.any(path.var_order3 > 0):
        ax.fill_between(path.lambdas, cum2, cum3,
                        alpha=0.6, label='Third-order', color='forestgreen')
    if np.any(path.var_residual > 0):
        ax.fill_between(path.lambdas, cum3, cum4,
                        alpha=0.5, label='Residual', color='goldenrod')

    ax.axvline(path.lambda_gcv_opt, color='gray', linestyle='--', alpha=0.5,
               label=r'$\lambda^*_{GCV}$')
    ax.set_xscale('log')
    ax.set_xlabel(r'$\lambda$')
    ax.set_ylabel('Explained Variance')
    ax.set_title('Variance Decomposition Path')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_prefix:
        plt.savefig(f"{save_prefix}_reg_path.png", dpi=150, bbox_inches='tight')

    return fig


def plot_pareto_frontier(path: RegPathResult,
                         y_var: float,
                         save_path: Optional[str] = None):
    """Plot the Pareto frontier: explained variance vs model complexity.

    Each point on the curve represents a different lambda. Shows the
    tradeoff between model complexity and unexplained variance.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 6))

    explained = path.var_total
    unexplained = y_var - explained
    complexity = path.df_values

    scatter = ax.scatter(complexity, unexplained, c=np.log10(path.lambdas),
                        cmap='viridis', s=30, alpha=0.8)
    plt.colorbar(scatter, ax=ax, label=r'$\log_{10}(\lambda)$')

    # Mark optimal points
    gcv_idx = np.argmin(np.abs(path.lambdas - path.lambda_gcv_opt))
    ax.plot(complexity[gcv_idx], unexplained[gcv_idx], 'r*', markersize=15,
            label='GCV optimal')

    ax.set_xlabel('Model Complexity (Effective df)')
    ax.set_ylabel('Unexplained Variance')
    ax.set_title('Pareto Frontier: Complexity vs Unexplained Variance')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig
