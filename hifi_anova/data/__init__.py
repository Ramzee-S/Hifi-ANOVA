"""Data: synthetic generators, loaders, preprocessing, test functions."""

from .synthetic import (generate_friedman1, generate_heteroscedastic,
                        generate_ishigami, ishigami_sobol_indices)
from .preprocessing import preprocess_data
from .nn_test_functions import get_nn_test_functions
