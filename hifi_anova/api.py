"""One-call API: hifi_anova(X, y) → complete results.

The simplest way to use HiFi-ANOVA. Takes raw data, handles preprocessing,
fitting, Sobol analysis, confidence intervals, and diagnostics automatically.
Returns a single result object with everything.

Usage:
    from hifi_anova.api import hifi_anova

    result = hifi_anova(X, y, feature_names=['income', 'age', ...])

    # Predictions
    pred = result.predict(X_new)
    lower, upper = result.predict_intervals(X_new)

    # Sobol indices with CIs
    for name, (S, lo, hi) in result.sobol_ci.items():
        print(f"{name}: S = {S:.3f} [{lo:.3f}, {hi:.3f}]")

    # Quick summary
    result.summary()

    # Save / load
    result.save('my_model/')
    from hifi_anova.model.io import load_model
    loaded = load_model('my_model/')   # dict with 'model', 'transformer', 'config', ...
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Optional, List, Dict, Tuple, Union
from dataclasses import dataclass, field


@dataclass
class HiFiResult:
    """Complete results from hifi_anova().

    Contains the fitted model, Sobol indices, confidence intervals,
    noise estimates, and all diagnostic information. Provides
    convenience methods for prediction and reporting.
    """
    # Core model
    model: object  # HiFiANOVA
    config: Dict
    feature_names: List[str]

    # Preprocessing
    transformer: object  # QuantileTransformer
    y_mean: float
    y_std: float

    # Training results
    train_results: Dict

    # Sobol indices
    sobol: Dict  # full sobol dict from compute_sobol_indices
    sobol_ci: Dict  # {name: (S, lo, hi)} for first-order

    # Diagnostics
    sigma_hat: float  # noise estimate
    r_squared: float
    loo_cv: float
    df: float  # effective degrees of freedom

    # Internal (for prediction intervals)
    _Phi_train: np.ndarray = field(repr=False, default=None)
    _reg_diag: np.ndarray = field(repr=False, default=None)
    _data: Dict = field(repr=False, default=None)

    def predict(self, X_new: np.ndarray) -> np.ndarray:
        """Predict on new data (original scale).

        Args:
            X_new: (M, D) new inputs in ORIGINAL feature space

        Returns:
            (M,) predictions
        """
        X_t = np.clip(self.transformer.transform(X_new), 0, 1)
        x = jnp.array(X_t, dtype=jnp.float32)
        return np.asarray(self.model.predict_mean_only(x))

    def predict_intervals(self, X_new: np.ndarray, alpha: float = 0.05
                          ) -> Tuple[np.ndarray, np.ndarray]:
        """Prediction intervals on new data.

        Args:
            X_new: (M, D) in original feature space
            alpha: significance level (0.05 = 95%)

        Returns:
            (lower, upper) — both (M,) arrays
        """
        from .model.predict import predict_intervals

        X_t = np.clip(self.transformer.transform(X_new), 0, 1)
        x = jnp.array(X_t, dtype=jnp.float32)

        result = predict_intervals(
            self.model, x,
            Phi_train=self._Phi_train,
            reg_diag=self._reg_diag,
            sigma2_hat=self.sigma_hat ** 2,
            alpha=alpha,
        )
        return result['lower'], result['upper']

    def summary(self):
        """Print a human-readable summary."""
        D = self.model.D
        print(f"HiFi-ANOVA Model Summary")
        print(f"  Variables: {D} ({', '.join(self.feature_names[:5])}"
              f"{'...' if D > 5 else ''})")
        print(f"  R²: {self.r_squared:.4f}")
        print(f"  Noise (σ̂): {self.sigma_hat:.4f}")
        print(f"  LOO-CV: {self.loo_cv:.4f}")
        print(f"  Effective df: {self.df:.1f}")
        print()
        print(f"  Sobol Indices (95% CI):")
        ranked = sorted(self.sobol_ci.items(),
                        key=lambda x: -x[1][0])
        for name, (S, lo, hi) in ranked:
            if S > 0.01:
                print(f"    {name:15s}: {S:.4f} [{lo:.4f}, {hi:.4f}]")

        # Second-order interactions
        so = self.sobol['mean_sobol'].get('second_order', {})
        if so:
            top_pairs = sorted(so.items(), key=lambda x: -x[1])[:5]
            significant = [(k, v) for k, v in top_pairs if v > 0.005]
            if significant:
                print(f"\n  Top Interactions:")
                for (i, j), s in significant:
                    ni = self.feature_names[i] if i < len(self.feature_names) else f'x{i}'
                    nj = self.feature_names[j] if j < len(self.feature_names) else f'x{j}'
                    print(f"    ({ni}, {nj}): {s:.4f}")

    def save(self, path: str):
        """Save the model to a directory."""
        from .model.io import save_model
        save_model(
            self.model, path,
            config=self.config,
            transformer=self.transformer,
            feature_names=self.feature_names,
            results=self.train_results,
            overwrite=True,
        )

    def component_curve(self, variable: Union[int, str], n_points: int = 200):
        """Get the learned component function for a variable.

        Args:
            variable: index or name
            n_points: grid points

        Returns:
            (x_grid, f_values) in [0,1] quantile space
        """
        from .analysis.component_eval import first_order_on_grid

        if isinstance(variable, str):
            variable = self.feature_names.index(variable)
        return first_order_on_grid(self.model, variable, n_points)


def hifi_anova(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: Optional[List[str]] = None,
    K1: int = 5,
    K2: int = 3,
    strategy: str = 'variance',
    mode: str = 'second',
    variable_selection: Optional[str] = 'bic',
    residual: Optional[str] = None,
    heteroscedastic: bool = False,
    seed: int = 42,
    verbose: bool = True,
    **kwargs,
) -> HiFiResult:
    """One-call API: fit a HiFi-ANOVA model and return complete results.

    Takes raw data, handles preprocessing, fitting, Sobol analysis,
    confidence intervals, and diagnostics automatically.

    Args:
        X: (N, D) feature matrix (original scale)
        y: (N,) target values
        feature_names: list of D feature names (optional)
        K1: max harmonic for first-order (default 5)
        K2: max harmonic for second-order (default 3)
        strategy: regularization strategy (default 'variance')
        mode: model complexity ('first', 'second', 'full', 'heteroscedastic', 'auto')
        variable_selection: selection method ('bic', 'group_lasso', '1se', None)
        residual: residual type ('rbf', 'rff', 'nystrom', None)
        heteroscedastic: if True, fit variance model
        seed: random seed
        verbose: print progress
        **kwargs: additional config overrides

    Returns:
        HiFiResult with model, Sobol indices, CIs, diagnostics, and
        convenience methods for prediction and reporting.

    Example:
        result = hifi_anova(X, y, feature_names=['age', 'income'])
        result.summary()
        pred = result.predict(X_new)
        lo, hi = result.predict_intervals(X_new)
    """
    jax.config.update("jax_enable_x64", True)

    from .data.preprocessing import preprocess_data
    from .training.trainer import HiFiANOVATrainer
    from .analysis.sobol import compute_sobol_indices
    from .analysis.automl import ridge_analytics, sobol_confidence_intervals
    from .core.gram import build_gram_matrix, build_gram_matrix_2d
    from .core.pairs import PairManager
    from .training.regularization import build_regularization_vector

    N, D = X.shape
    if feature_names is None:
        feature_names = [f'x{i+1}' for i in range(D)]

    # Build config
    config = {
        'K1': K1, 'K2': K2,
        'strategy': strategy,
        'lambda_order1': 0.001,
        'lambda_order2': 0.01,
    }

    if mode == 'auto':
        config['mode'] = 'auto'
        config['auto_threshold'] = kwargs.pop('auto_threshold', 0.01)
    elif heteroscedastic:
        config['stages'] = ['A', 'B', 'C', 'D'] if residual else ['A', 'B', 'D']
        config['Kh'] = kwargs.pop('Kh', 3)
        config['lambda_h'] = kwargs.pop('lambda_h', 0.1)
    else:
        stage_map = {'first': ['A'], 'second': ['A', 'B'], 'full': ['A', 'B', 'C']}
        stages = stage_map.get(mode, ['A', 'B'])
        if residual:
            stages = list(set(stages) | {'C'})
            stages.sort()
        config['stages'] = stages

    if variable_selection:
        config['variable_selection'] = variable_selection
        config['pair_candidates'] = kwargs.pop('pair_candidates', 'either')

    if residual:
        res_config = {'type': residual, 'lambda_residual': kwargs.pop('lambda_residual', 1.0)}
        if residual == 'rbf':
            res_config.update({'n_centers': kwargs.pop('n_centers', min(300, N // 5)),
                               'sigma': kwargs.pop('sigma', 0.2)})
        elif residual == 'rff':
            res_config.update({'n_features': kwargs.pop('n_features', 1000),
                               'gamma': kwargs.pop('gamma', 3.0)})
        elif residual == 'nystrom':
            res_config.update({'n_inducing': kwargs.pop('n_inducing', min(300, N // 5)),
                               'kernel': kwargs.pop('kernel', 'matern52'),
                               'lengthscale': kwargs.pop('lengthscale', 0.2)})
        config['residual'] = res_config

    config.update(kwargs)  # any remaining overrides

    # Preprocess
    data = preprocess_data(X, y, seed=seed)

    # Fit
    trainer = HiFiANOVATrainer(config)
    key = jax.random.PRNGKey(seed)
    model, train_results = trainer.fit(
        data['x_train'], data['y_train'],
        data['x_val'], data['y_val'],
        key=key,
    )

    # Sobol indices
    sobol = compute_sobol_indices(model, data['x_test'])

    # Analytics (for CIs, noise estimate). Read the basis config from the fitted
    # model (authoritative) so the CI column layout matches how Phi was built —
    # not just the Fourier default. Otherwise CIs are silently wrong for
    # Legendre/Haar bases or include_linear_1=False.
    basis_name = model.basis_name
    include_linear_1 = model.include_linear_1
    include_linear_2 = getattr(model, 'include_linear_2', True)

    Phi_train = np.asarray(model.build_phi_all(data['x_train']), dtype=np.float64)
    G1 = np.asarray(build_gram_matrix(K1, include_linear=include_linear_1,
                                      basis_name=basis_name),
                      dtype=np.float64)
    f0 = float(np.mean(np.asarray(data['y_train'])))
    y_c = np.asarray(data['y_train'], dtype=np.float64) - f0
    reg_diag = np.asarray(build_regularization_vector(
        D, K1, K2,
        model.pair_indices.shape[0] if model.pair_indices is not None else 0,
        strategy, config['lambda_order1'], config['lambda_order2'],
        include_linear_1=include_linear_1, include_linear_2=include_linear_2,
        basis_name=basis_name),
        dtype=np.float64)

    # Pad reg_diag if features exceed it (third-order, residual)
    if len(reg_diag) < Phi_train.shape[1]:
        reg_diag = np.concatenate([reg_diag,
            np.full(Phi_train.shape[1] - len(reg_diag), config['lambda_order2'])])

    analytics = ridge_analytics(Phi_train, y_c, reg_diag)

    # Sobol CIs
    G2 = None
    if K2 > 0 and model.pair_indices is not None:
        G2 = np.asarray(build_gram_matrix_2d(
            build_gram_matrix(K2, include_linear=include_linear_2,
                              basis_name=basis_name)), dtype=np.float64)
    ci = sobol_confidence_intervals(
        Phi_train, y_c, reg_diag, D, K1, G1,
        K2=K2,
        P=model.pair_indices.shape[0] if model.pair_indices is not None else 0,
        G2=G2,
        pair_indices=np.asarray(model.pair_indices) if model.pair_indices is not None else None,
        basis_name=basis_name,
        include_linear_1=include_linear_1,
    )

    # Build named Sobol CI dict
    sobol_ci_named = {}
    for i, (S, lo, hi) in ci['first_order'].items():
        name = feature_names[i] if i < len(feature_names) else f'x{i+1}'
        sobol_ci_named[name] = (S, lo, hi)

    # R^2
    pred_test = model.predict_mean_only(data['x_test'])
    var_y = float(jnp.var(data['y_test']))
    var_r = float(jnp.var(data['y_test'] - pred_test))
    r2 = 1.0 - var_r / var_y if var_y > 0 else 0.0

    result = HiFiResult(
        model=model,
        config=config,
        feature_names=feature_names,
        transformer=data['transformer'],
        y_mean=data['y_mean'],
        y_std=data['y_std'],
        train_results=train_results,
        sobol=sobol,
        sobol_ci=sobol_ci_named,
        sigma_hat=analytics['sigma_hat'],
        r_squared=r2,
        loo_cv=analytics['loo_cv'],
        df=analytics['df'],
        _Phi_train=Phi_train,
        _reg_diag=reg_diag,
        _data=data,
    )

    if verbose:
        result.summary()

    return result
