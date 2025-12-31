"""ML metrics implementations from scratch."""

import numpy as np


def r_squared(y_true, y_pred):
    """Calculate R² score: R² = 1 - (SS_res / SS_tot)."""
    y_true, y_pred = np.array(y_true).flatten(), np.array(y_pred).flatten()
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1 - (ss_res / ss_tot)) if ss_tot != 0 else 0.0


def mse(y_true, y_pred):
    """Calculate Mean Squared Error."""
    y_true, y_pred = np.array(y_true).flatten(), np.array(y_pred).flatten()
    return float(np.mean((y_true - y_pred) ** 2))


def precision(y_true, y_pred):
    """Calculate precision score: TP / (TP + FP)."""
    y_true, y_pred = np.array(y_true).flatten(), np.array(y_pred).flatten()
    classes = np.unique(np.concatenate([y_true, y_pred]))
    
    if len(classes) == 2:
        pos_class = classes[1]
        tp = np.sum((y_true == pos_class) & (y_pred == pos_class))
        fp = np.sum((y_true != pos_class) & (y_pred == pos_class))
        return float(tp / (tp + fp)) if tp + fp != 0 else 0.0
    
    precisions = []
    for cls in classes:
        tp = np.sum((y_true == cls) & (y_pred == cls))
        fp = np.sum((y_true != cls) & (y_pred == cls))
        precisions.append(tp / (tp + fp) if tp + fp != 0 else 0.0)
    return float(np.mean(precisions))


def recall(y_true, y_pred):
    """Calculate recall score: TP / (TP + FN)."""
    y_true, y_pred = np.array(y_true).flatten(), np.array(y_pred).flatten()
    classes = np.unique(np.concatenate([y_true, y_pred]))
    
    if len(classes) == 2:
        pos_class = classes[1]
        tp = np.sum((y_true == pos_class) & (y_pred == pos_class))
        fn = np.sum((y_true == pos_class) & (y_pred != pos_class))
        return float(tp / (tp + fn)) if tp + fn != 0 else 0.0
    
    recalls = []
    for cls in classes:
        tp = np.sum((y_true == cls) & (y_pred == cls))
        fn = np.sum((y_true == cls) & (y_pred != cls))
        recalls.append(tp / (tp + fn) if tp + fn != 0 else 0.0)
    return float(np.mean(recalls))


def f1_score(y_true, y_pred):
    """Calculate F1 score: 2 · (precision · recall) / (precision + recall)."""
    prec = precision(y_true, y_pred)
    rec = recall(y_true, y_pred)
    return float(2 * (prec * rec) / (prec + rec)) if prec + rec != 0 else 0.0
