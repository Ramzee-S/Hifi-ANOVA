"""Generic, stage-agnostic helpers extracted from ``trainer.py``.

Behavior-preserving split (step 1 of the trainer decomposition): these are the
free functions the staged fit uses across stages — the Gaussian NLL the trainer
reports, the flat mean-coefficient layout split shared by the Stage-B and
heteroscedastic model builds, and the post-fit first-order group pruning. They
carry no trainer state, so the stage modules can import them without pulling in
the orchestrator. Moved verbatim from ``trainer.py`` (no logic change).
"""

from ..array_backend import xp as jnp  # switchable array backend (numpy exact core)
import numpy as np


def _gaussian_nll(residuals, sigma2):
    """Mean Gaussian negative log-likelihood (dropping the constant 2π term).

    ``sigma2`` is either a scalar (homoscedastic) or a per-point array
    (heteroscedastic). Matches the NLL the trainer reports for Stage D, so the
    heteroscedastic and constant-variance models are compared on one scale.
    """
    r2 = np.asarray(residuals, dtype=np.float64) ** 2
    s2 = np.asarray(sigma2, dtype=np.float64)
    return float(np.mean(0.5 * np.log(s2) + 0.5 * r2 / s2))


def _scatter_first_order(w1_ragged, D, block1, included, *, dtype):
    """Scatter a subset-layout first-order coefficient vector to full width.

    The order-selective mean path (``variable_orders``) SOLVES on a design
    whose first-order block spans only ``included`` variables (so no df is
    spent on excluded ones), but the MODEL keeps the full uniform
    ``D * block1`` layout with exact zeros in the excluded blocks — prediction
    and Sobol slicing then work unchanged, and an excluded variable's
    first-order share is identically 0.
    """
    w1_full = np.zeros(D * block1, dtype=np.float64)
    w1_r = np.asarray(w1_ragged, dtype=np.float64)
    for pos, i in enumerate(included):
        w1_full[i * block1:(i + 1) * block1] = w1_r[pos * block1:(pos + 1) * block1]
    return jnp.asarray(w1_full, dtype=dtype)


def _split_mean_coeffs(w_all, F1, F2, *, has_third, dtype):
    """Split a flat mean-coefficient vector into (w1, w2, w3) blocks by layout.

    The mean design is laid out ``[first-order | second-order | third-order]`` with
    ``F1`` first-order and ``F2`` second-order columns, so
    ``w1 = w_all[:F1]``, ``w2 = w_all[F1:F1+F2]`` (empty when ``F2 == 0``), and
    ``w3 = w_all[F1+F2:]`` when a third-order block is present else an empty
    ``dtype`` array. Single source of truth for the layout split that the Stage-B
    and heteroscedastic model builds otherwise hand-repeat.
    """
    w1 = w_all[:F1]
    w2 = w_all[F1:F1 + F2]
    w3 = w_all[F1 + F2:] if has_third else jnp.array([], dtype=dtype)
    return w1, w2, w3


def _prune_first_order_blocks(w_flat, phi_all, reg_full, y_centered,
                              D, block1, G1, method, verbose=True):
    """Zero the first-order block of any variable a group criterion rejects.

    First-order and pair/triple blocks are Hoeffding-orthogonal, so a
    leave-one-group-out test on the full design identifies variables whose
    marginal (first-order) effect is not supported by the data. Their block is
    set to exactly zero — the group-sparse step plain ridge cannot do. Because
    of the orthogonality, zeroing a rejected block barely changes predictions.

    Args:
        w_flat: (F,) fitted coefficients [first-order | pairs | triples].
        phi_all: (N, F) full design used for the fit.
        reg_full: (F,) regularization diagonal matching phi_all.
        y_centered: (N,) centered targets.
        D, block1: number of variables and first-order features per variable.
        G1: (block1, block1) first-order Gram matrix (for weighted norms).
        method: 'bic' | 'group_lasso' | '1se' | 'none'.

    Returns:
        (w_pruned, info) — a float64 numpy copy of w_flat with rejected
        first-order blocks zeroed, plus the selection diagnostics.
    """
    from .selection import prune_groups_postfit

    fo_slices = [slice(i * block1, (i + 1) * block1) for i in range(D)]
    G1_np = np.asarray(G1, dtype=np.float64)
    surviving, info = prune_groups_postfit(
        np.asarray(phi_all, dtype=np.float64),
        np.asarray(y_centered, dtype=np.float64),
        fo_slices, np.asarray(reg_full, dtype=np.float64),
        method=method, gram_matrices=[G1_np] * D,
        group_labels=[f"x{i+1}" for i in range(D)], verbose=verbose,
    )
    pruned = [i for i in range(D) if i not in set(surviving)]
    w_new = np.asarray(w_flat, dtype=np.float64).copy()
    for i in pruned:
        w_new[i * block1:(i + 1) * block1] = 0.0
    info['pruned_variables'] = pruned
    if verbose and pruned:
        print(f"    Zeroed first-order blocks: {[f'x{i+1}' for i in pruned]}")
    return w_new, info
