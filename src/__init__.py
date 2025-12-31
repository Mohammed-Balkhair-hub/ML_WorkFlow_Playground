"""
ML Workflow Playground - A collection of ML utilities for model training and evaluation.
"""

from .models import (
    LinearRegression,
    get_random_forest_regressor,
    get_random_forest_classifier,
    get_logistic_regression,
    get_xgboost_regressor,
    get_xgboost_classifier,
)

from .metrics import (
    r_squared,
    mse,
    precision,
    recall,
    f1_score,
)

from .display import display_metrics

from .model_selection import (
    cross_validate,
    grid_search_cv,
)

__all__ = [
    # Models
    "LinearRegression",
    "get_random_forest_regressor",
    "get_random_forest_classifier",
    "get_logistic_regression",
    "get_xgboost_regressor",
    "get_xgboost_classifier",
    # Metrics
    "r_squared",
    "mse",
    "precision",
    "recall",
    "f1_score",
    # Display
    "display_metrics",
    # Model Selection
    "cross_validate",
    "grid_search_cv",
]

