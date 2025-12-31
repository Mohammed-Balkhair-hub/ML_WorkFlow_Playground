# ML Workflow Playground

A collection of ML utilities for model training, evaluation, and selection. This repository provides custom implementations from scratch for learning purposes, along with convenient wrappers for sklearn models.

## Features

- **Custom Linear Regression**: Implemented from scratch using the normal equation
- **Model Wrappers**: Easy-to-use functions for sklearn models (RandomForest, LogisticRegression, XGBoost)
- **Metrics from Scratch**: R², MSE, Precision, Recall, F1 Score implementations
- **Visualization**: Display metrics and model performance plots
- **Cross-Validation**: Custom K-fold CV implementation from scratch
- **Grid Search**: Wrapper around sklearn's GridSearchCV

## Installation

```bash
pip install -e .
```

Or install dependencies directly:
```bash
pip install numpy pandas scikit-learn matplotlib xgboost seaborn
```

## Quick Start

### Models

```python
from src import LinearRegression, get_random_forest_regressor, get_logistic_regression

# Custom Linear Regression (from scratch)
model = LinearRegression(fit_intercept=True)
model.fit(X_train, y_train)
predictions = model.predict(X_test)

# Sklearn model wrappers
rf_model = get_random_forest_regressor(n_estimators=100, max_depth=5)
rf_model.fit(X_train, y_train)

lr_model = get_logistic_regression(max_iter=1000)
lr_model.fit(X_train, y_train)
```

### Metrics

```python
from src import r_squared, mse, precision, recall, f1_score

# Regression metrics
r2 = r_squared(y_true, y_pred)
mse_value = mse(y_true, y_pred)

# Classification metrics
prec = precision(y_true, y_pred)
rec = recall(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
```

### Display Metrics

```python
from src import display_metrics

# For regression
display_metrics(y_true, y_pred, task_type='regression', plot=True)

# For classification
display_metrics(y_true, y_pred, task_type='classification', plot=True)
```

### Cross-Validation

```python
import pandas as pd
from src import cross_validate, LinearRegression
from sklearn.metrics import mean_squared_error

# Load your data
df = pd.read_csv('your_data.csv')

# Custom cross-validation from scratch
model, mean_score, std_score, all_scores = cross_validate(
    dataframe=df,
    model=LinearRegression,  # Model class
    target_col='target',
    k_folds=5,
    metric=mean_squared_error,
    shuffle=True,
    random_state=42
)

print(f"Mean Score: {mean_score:.4f} (+/- {std_score:.4f})")
```

### Grid Search

```python
from src import grid_search_cv
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Define parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7],
    'min_samples_split': [2, 5, 10]
}

# Perform grid search
grid_search = grid_search_cv(
    model=RandomForestClassifier,
    X=X_train,
    y=y_train,
    param_grid=param_grid,
    cv=5,
    scoring='accuracy'
)

# Access results
print(f"Best Score: {grid_search.best_score_}")
print(f"Best Parameters: {grid_search.best_params_}")
best_model = grid_search.best_estimator_
```

## Repository Structure

```
ML_WorkFlow_Playground/
├── src/
│   ├── __init__.py           # Package exports
│   ├── models.py             # Model implementations and wrappers
│   ├── metrics.py            # Metrics from scratch
│   ├── display.py            # Visualization functions
│   └── model_selection.py    # Cross-validation and GridSearch
├── notebooks/                # For EDA and experimentation
├── pyproject.toml            # Dependencies
└── README.md                 # This file
```

## Available Models

### Custom Implementation
- `LinearRegression`: Linear regression using normal equation (from scratch)

### Sklearn Wrappers
- `get_random_forest_regressor(**kwargs)`: Random Forest Regressor
- `get_random_forest_classifier(**kwargs)`: Random Forest Classifier
- `get_logistic_regression(**kwargs)`: Logistic Regression
- `get_xgboost_regressor(**kwargs)`: XGBoost Regressor
- `get_xgboost_classifier(**kwargs)`: XGBoost Classifier

## Available Metrics

### Regression
- `r_squared(y_true, y_pred)`: Coefficient of determination
- `mse(y_true, y_pred)`: Mean Squared Error

### Classification
- `precision(y_true, y_pred, average='binary')`: Precision score
- `recall(y_true, y_pred, average='binary')`: Recall score
- `f1_score(y_true, y_pred, average='binary')`: F1 score (harmonic mean of precision and recall)

## Model Selection

### Custom Cross-Validation
`cross_validate()` performs K-fold cross-validation from scratch:
- Splits data into K folds manually
- Trains model on each fold
- Evaluates using provided sklearn metric
- Returns best model (trained on full data) and statistics

### Grid Search
`grid_search_cv()` wraps sklearn's GridSearchCV for hyperparameter tuning.

## Examples

See the `notebooks/` directory for example notebooks demonstrating:
- Loading data from Kaggle
- Exploratory Data Analysis (EDA)
- Using models, metrics, and cross-validation
- Displaying results

## Notes

- Linear Regression is implemented from scratch using the normal equation for educational purposes
- Metrics are implemented from scratch (no sklearn dependencies) for learning
- Cross-validation is implemented from scratch to understand the process
- Other models use sklearn wrappers for convenience and reliability

