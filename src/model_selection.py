"""Model selection utilities including custom cross-validation and GridSearchCV wrapper."""

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.base import clone as sklearn_clone


def cross_validate(dataframe, model, target_col, k_folds=5, metric=None, shuffle=True, random_state=None, **model_kwargs):
    """Custom K-fold cross-validation implementation from scratch."""
    if target_col not in dataframe.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame")
    if k_folds < 2:
        raise ValueError("k_folds must be at least 2")
    
    df = dataframe.copy()
    feature_cols = [col for col in df.columns if col != target_col]
    X, y = df[feature_cols].values, df[target_col].values
    
    if shuffle:
        np.random.seed(random_state)
        indices = np.random.permutation(len(df))
        X, y = X[indices], y[indices]
    
    n_samples, fold_size = len(df), len(df) // k_folds
    all_scores = []
    
    for fold in range(k_folds):
        start_idx = fold * fold_size
        end_idx = n_samples if fold == k_folds - 1 else (fold + 1) * fold_size
        test_indices = np.arange(start_idx, end_idx)
        train_indices = np.concatenate([np.arange(0, start_idx), np.arange(end_idx, n_samples)])
        
        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]
        
        if isinstance(model, type):
            fold_model = model(**model_kwargs)
        elif callable(model):
            # Handle callable functions like get_xgboost_classifier
            fold_model = model(**model_kwargs)
        else:
            try:
                fold_model = sklearn_clone(model)
            except (AttributeError, TypeError):
                fold_model = model.__class__() if hasattr(model, '__class__') else model
                if hasattr(model, 'fit_intercept'):
                    fold_model.fit_intercept = model.fit_intercept
        
        fold_model.fit(X_train, y_train)
        y_pred = fold_model.predict(X_test)
        
        if metric is None:
            if _is_classification(y):
                from sklearn.metrics import accuracy_score
                score = accuracy_score(y_test, y_pred)
            else:
                from sklearn.metrics import mean_squared_error
                score = -mean_squared_error(y_test, y_pred)
        else:
            score = metric(y_test, y_pred)
        
        all_scores.append(score)
    
    if isinstance(model, type):
        best_model = model(**model_kwargs)
    elif callable(model):
        # Handle callable functions like get_xgboost_classifier
        best_model = model(**model_kwargs)
    else:
        try:
            best_model = sklearn_clone(model)
        except (AttributeError, TypeError):
            best_model = model.__class__() if hasattr(model, '__class__') else model
            if hasattr(model, 'fit_intercept'):
                best_model.fit_intercept = model.fit_intercept
    
    best_model.fit(X, y)
    all_scores = np.array(all_scores)
    return best_model, float(np.mean(all_scores)), float(np.std(all_scores)), all_scores.tolist()


def grid_search_cv(model, X, y, param_grid, cv=5, scoring=None, **kwargs):
    """Wrapper around sklearn's GridSearchCV for hyperparameter tuning."""
    # GridSearchCV accepts: n_jobs, refit, verbose, pre_dispatch, error_score, return_train_score
    # Other parameters like random_state, eval_metric should go to the estimator
    gridsearch_params = ['n_jobs', 'refit', 'verbose', 'pre_dispatch', 'error_score', 'return_train_score']
    estimator_kwargs = {k: v for k, v in kwargs.items() if k not in gridsearch_params}
    gridsearch_kwargs = {k: v for k, v in kwargs.items() if k in gridsearch_params}
    
    # Create base estimator with estimator-specific kwargs
    base_estimator = model(**estimator_kwargs)
    
    # Create GridSearchCV with only valid parameters
    grid_search = GridSearchCV(estimator=base_estimator, param_grid=param_grid, cv=cv, scoring=scoring, **gridsearch_kwargs)
    grid_search.fit(X, y)
    return grid_search


def _is_classification(y):
    """Helper function to determine if target is classification or regression."""
    y_array = np.array(y)
    unique_values = np.unique(y_array)
    if len(unique_values) < min(50, len(y_array) * 0.2):
        return True
    return np.allclose(y_array, np.round(y_array), atol=1e-6) and len(unique_values) < min(50, len(y_array) * 0.2)
