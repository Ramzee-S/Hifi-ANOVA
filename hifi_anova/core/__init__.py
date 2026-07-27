"""Core infrastructure: Gram matrices, basis functions, feature construction."""

from .gram import build_gram_matrix, build_gram_matrix_2d, build_gram_matrix_3d, build_derivative_penalty
from .features import (build_first_order_features, build_second_order_features,
                       build_third_order_features, build_per_variable_basis,
                       basis_size, build_mixed_first_order_features,
                       build_mixed_second_order_features)
from .pairs import PairManager, TripleManager
from .projection import project_features_orthogonal
from .haar import HaarBasis, build_haar_features
