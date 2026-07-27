"""Model complexity mode selection.

Provides named modes for how much of the model hierarchy to fit,
plus an 'auto' mode that decides stage-by-stage.

Modes:
  'first'            — First-order Fourier only (stages A)
  'second'           — First + second-order Fourier (stages A, B)
  'full'             — First + second + residual NN (stages A, B, C)
  'heteroscedastic'  — First + second + variance decomposition (stages A, B, D)
  'auto'             — Fit progressively, add next stage if it improves
                       validation loss beyond what noise reduction alone explains.

Auto mode decision logic:
  The key insight: the residual fraction depends on regularization strength.
  We don't compare against a fixed threshold. Instead, after each stage we
  check whether the VALIDATION RMSE improved meaningfully relative to the
  noise floor:

  1. After stage A: compute R²_val. If R²_val < (1 - threshold) → add B.
     (i.e., if more than `threshold` fraction of variance is unexplained)
  2. After stage B: same check. If residual fraction > threshold → add C (NN).
  3. After stage C: check if residual variance correlates with inputs
     (heteroscedastic structure) → add D.

  The threshold default is 0.01 (1%). This is aggressive: it says "if more
  than 1% of variance is unexplained, try the next stage." The next stage's
  own regularization / early stopping will prevent overfitting if there's
  nothing to learn. It's cheaper to try and discard than to miss structure.
"""

import numpy as np
from typing import Dict, List, Optional


MODE_STAGES = {
    'first':            ['A'],
    'second':           ['A', 'B'],
    'full':             ['A', 'B', 'C'],
    'heteroscedastic':  ['A', 'B', 'D'],
}


def resolve_mode(config: Dict) -> Dict:
    """Resolve a mode string into concrete stages and settings.

    If config has 'mode', translates to 'stages'. If 'stages' is already
    set, returns unchanged (backward compatible). 'auto' sets a marker
    for the trainer to handle dynamically.

    Args:
        config: trainer configuration dict

    Returns:
        Updated config dict with 'stages' set
    """
    config = dict(config)  # don't mutate original

    mode = config.pop('mode', None)
    if mode is None:
        return config

    if mode == 'auto':
        config['_auto_mode'] = True
        config['_auto_threshold'] = config.pop('auto_threshold', 0.01)
        config['stages'] = ['A']
        if 'residual_nn' not in config:
            config['residual_nn'] = {'enabled': True}
        elif not config['residual_nn'].get('enabled', False):
            config['residual_nn']['enabled'] = True
        return config

    if mode in MODE_STAGES:
        config['stages'] = MODE_STAGES[mode]
        if 'C' in config['stages']:
            if 'residual_nn' not in config:
                config['residual_nn'] = {'enabled': True}
            else:
                config['residual_nn']['enabled'] = True
        if 'D' in config['stages'] and config.get('Kh', 0) == 0:
            config['Kh'] = 3
        return config

    raise ValueError(
        f"Unknown mode '{mode}'. Choose from: "
        f"{list(MODE_STAGES.keys()) + ['auto']}"
    )


def auto_decide_next_stage(current_stage: str,
                           rmse_val: float,
                           var_y_val: float,
                           max_var_corr: float = 0.0,
                           threshold: float = 0.01,
                           verbose: bool = True) -> Optional[str]:
    """Decide whether to add the next stage in auto mode.

    Uses residual fraction = 1 - R²_val = Var(residual) / Var(y).
    This is computed from validation RMSE so it accounts for whatever
    regularization was applied — no dependence on lambda choice.

    Args:
        current_stage: 'A', 'B', or 'C'
        rmse_val: validation RMSE from current stage
        var_y_val: Var(y_val) — total variance of targets
        max_var_corr: max |corr(r², x_i)| — for variance model decision
        threshold: minimum residual fraction to add next stage (default 1%)
        verbose: print decision

    Returns:
        Next stage letter ('B', 'C', 'D') or None if no more stages needed
    """
    next_map = {'A': 'B', 'B': 'C', 'C': 'D'}
    next_s = next_map.get(current_stage)
    if next_s is None:
        return None

    residual_fraction = (rmse_val ** 2) / var_y_val if var_y_val > 1e-10 else 0.0

    if next_s in ('B', 'C'):
        if residual_fraction > threshold:
            if verbose:
                r2 = 1.0 - residual_fraction
                print(f"  Auto: R²_val = {r2:.4f}, residual = {residual_fraction:.1%} "
                      f"> {threshold:.1%} → adding stage {next_s}")
            return next_s
        else:
            if verbose:
                print(f"  Auto: R²_val = {1.0 - residual_fraction:.4f}, "
                      f"residual = {residual_fraction:.1%} ≤ {threshold:.1%} "
                      f"→ stopping (stage {next_s} not needed)")
            return None

    if next_s == 'D':
        if max_var_corr > 0.1:
            if verbose:
                print(f"  Auto: max |corr(r², x)| = {max_var_corr:.3f} > 0.1 "
                      f"→ adding stage D (heteroscedastic)")
            return 'D'
        else:
            if verbose:
                print(f"  Auto: max |corr(r², x)| = {max_var_corr:.3f} ≤ 0.1 "
                      f"→ no variance model needed")
            return None

    return None
