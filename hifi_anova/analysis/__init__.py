"""Analysis: Sobol indices, diagnostics, visualization, regularization paths, AutoML analytics."""

from .sobol import compute_sobol_indices, compute_correlative_sobol
from .diagnostics import variance_accounting_report, calibration_report, correlation_diagnostic
from .residual_diagnostics import analyze_residuals, ResidualDiagnostics
from .reg_path import compute_reg_path, plot_reg_path, plot_pareto_frontier, RegPathResult
from .automl import (
    ridge_analytics,
    sandwich_covariance,
    sobol_confidence_intervals,
    noise_complexity_curve,
    kfold_cv_analytic,
    stability_diagnostics,
    sample_size_diagnostics,
)
from .interaction_discovery import (
    scan_missing_pairs,
    scan_missing_variance_pairs,
    iterative_pair_discovery,
)
from .haar_diagnostic import haar_residual_analysis, haar_multi_basis_characterization
from .component_eval import (
    evaluate_first_order, evaluate_second_order,
    evaluate_all_first_order, first_order_on_grid,
    second_order_on_grid, frequency_decomposition,
    interaction_strength_matrix,
)
from .basis_characterization import (
    multi_basis_fit,
    cross_residual_characterization,
    sequential_projection_characterization,
    auto_select_basis,
    print_characterization_table,
    print_basis_recommendations,
)
