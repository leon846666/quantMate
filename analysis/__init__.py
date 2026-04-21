"""Analysis: factor calculation, preprocessing, stock selection."""
from analysis._registry import register_factor, list_factors, get_factor
from analysis.features import (
    compute_factors,
    winsorize_cross_section,
    zscore_cross_section,
    neutralize,
    add_rank_columns,
    build_training_matrix,
    make_features,
)
from analysis.selector import filter_universe, top_n_by_score

__all__ = [
    "register_factor",
    "list_factors",
    "get_factor",
    "compute_factors",
    "winsorize_cross_section",
    "zscore_cross_section",
    "neutralize",
    "add_rank_columns",
    "build_training_matrix",
    "make_features",
    "filter_universe",
    "top_n_by_score",
]
