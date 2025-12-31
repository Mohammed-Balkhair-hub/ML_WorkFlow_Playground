"""Model implementations and wrappers."""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
try:
    from xgboost import XGBRegressor, XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


class LinearRegression:
    """Custom Linear Regression using normal equation: θ = (X^T · X)^(-1) · X^T · y"""
    
    def __init__(self, fit_intercept=True):
        """Initialize Linear Regression model."""
        self.fit_intercept = fit_intercept
        self.coefficients = None
        self.intercept = 0.0
    
    def fit(self, X, y):
        """Fit the model using normal equation."""
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).flatten()
        
        if self.fit_intercept:
            X = np.column_stack([np.ones(X.shape[0]), X])
        
        try:
            self.coefficients = np.linalg.inv(X.T @ X) @ X.T @ y
        except np.linalg.LinAlgError:
            self.coefficients = np.linalg.pinv(X) @ y
        
        if self.fit_intercept:
            self.intercept = self.coefficients[0]
            self.coefficients = self.coefficients[1:]
        else:
            self.intercept = 0.0
    
    def predict(self, X):
        """Make predictions using fitted model."""
        if self.coefficients is None:
            raise ValueError("Model must be fitted before making predictions.")
        if isinstance(X, pd.DataFrame):
            X = X.values
        return np.asarray(X, dtype=np.float64) @ self.coefficients + self.intercept


def get_random_forest_regressor(**kwargs):
    """Get Random Forest Regressor from sklearn."""
    return RandomForestRegressor(**kwargs)


def get_random_forest_classifier(**kwargs):
    """Get Random Forest Classifier from sklearn."""
    return RandomForestClassifier(**kwargs)


def get_logistic_regression(**kwargs):
    """Get Logistic Regression from sklearn."""
    return LogisticRegression(**kwargs)


def get_xgboost_regressor(**kwargs):
    """Get XGBoost Regressor."""
    if not XGBOOST_AVAILABLE:
        raise ImportError("xgboost is not installed. Install it with: pip install xgboost")
    return XGBRegressor(**kwargs)


def get_xgboost_classifier(**kwargs):
    """Get XGBoost Classifier."""
    if not XGBOOST_AVAILABLE:
        raise ImportError("xgboost is not installed. Install it with: pip install xgboost")
    return XGBClassifier(**kwargs)
