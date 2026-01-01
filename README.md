# ML Workflow Playground

A collection of ML utilities for model training, evaluation, and selection. Provides custom implementations from scratch for learning purposes, along with convenient wrappers for sklearn models.

## Features

- **Custom Models**: Linear Regression implemented from scratch
- **Model Wrappers**: RandomForest, LogisticRegression, XGBoost (Regressor/Classifier)
- **Metrics**: R², MSE, Precision, Recall, F1 Score (from scratch)
- **Cross-Validation**: Custom K-fold CV implementation
- **Grid Search**: Wrapper around sklearn's GridSearchCV
- **Visualization**: Display metrics and performance plots

## Installation

```bash
uv sync
```

This will create a virtual environment and install all dependencies from `pyproject.toml`.

## Quick Start

```python
from src import LinearRegression, get_random_forest_classifier, get_logistic_regression
from src import r_squared, precision, recall, f1_score, display_metrics

# Models
model = LinearRegression(fit_intercept=True)
model.fit(X_train, y_train)

rf_model = get_random_forest_classifier(n_estimators=100)
rf_model.fit(X_train, y_train)

# Metrics
r2 = r_squared(y_true, y_pred)
prec = precision(y_true, y_pred)
display_metrics(y_true, y_pred, task_type='classification')
```

## Available Models

- `LinearRegression`: Custom implementation (from scratch)
- `get_random_forest_regressor/classifier(**kwargs)`: Random Forest
- `get_logistic_regression(**kwargs)`: Logistic Regression
- `get_xgboost_regressor/classifier(**kwargs)`: XGBoost

## Available Metrics

**Regression**: `r_squared()`, `mse()`  
**Classification**: `precision()`, `recall()`, `f1_score()`

## Examples

See `notebooks/` for complete examples:
- **exam-score-playground.ipynb**: Regression task with Linear Regression
- **penguins-playground.ipynb**: Classification with XGBoost (species prediction)
- **titanic-playground.ipynb**: Classification with Logistic Regression & Random Forest (survival prediction)
- **breast_cancer_playground.ipynb**: Classification with cross-validation, grid search, and model persistence

## Notes

Custom implementations (Linear Regression, metrics, cross-validation) are from scratch for educational purposes. Other models use sklearn wrappers.

