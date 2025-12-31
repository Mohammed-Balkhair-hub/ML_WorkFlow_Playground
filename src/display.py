"""Display functions for visualizing metrics and model performance."""

import numpy as np
from .metrics import r_squared, mse, precision, recall, f1_score


def display_metrics(y_true, y_pred, task_type):
    """Display model performance metrics in formatted way."""
    y_true, y_pred = np.array(y_true).flatten(), np.array(y_pred).flatten()
    
    if task_type not in ['regression', 'classification']:
        raise ValueError("task_type must be 'regression' or 'classification'")
    
    print("=" * 60)
    print("MODEL PERFORMANCE METRICS")
    print("=" * 60)
    
    if task_type == 'regression':
        print(f"\nRegression Metrics:")
        print(f"  R² Score:  {r_squared(y_true, y_pred):.6f}")
        print(f"  MSE:       {mse(y_true, y_pred):.6f}")
        print("=" * 60)
    else:
        print(f"\nClassification Metrics:")
        print(f"  Precision: {precision(y_true, y_pred):.6f}")
        print(f"  Recall:    {recall(y_true, y_pred):.6f}")
        print(f"  F1 Score:  {f1_score(y_true, y_pred):.6f}")
        print("=" * 60)
    print()
