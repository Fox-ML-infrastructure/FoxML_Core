"""
Copyright (c) 2025 Fox ML Infrastructure

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

"""
Task-Aware Metric Evaluation

Provides task-specific evaluation metrics:
- Regression: IC (Information Coefficient), R², MSE
- Binary Classification: AUC, LogLoss, Accuracy
- Multiclass Classification: Cross-entropy, Accuracy, Macro F1
"""


import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from scripts.utils.task_types import TaskType


def eval_regression(
    y_true: np.ndarray, 
    y_pred: np.ndarray,
    return_ic: bool = True
) -> Dict[str, float]:
    """
    Evaluate regression predictions
    
    Args:
        y_true: True target values
        y_pred: Predicted values
        return_ic: If True, compute Information Coefficient (correlation)
    
    Returns:
        Dictionary of metrics
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    
    # Remove NaN pairs
    valid_mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if valid_mask.sum() == 0:
        return {"r2": np.nan, "mse": np.nan, "ic": np.nan}
    
    y_true = y_true[valid_mask]
    y_pred = y_pred[valid_mask]
    
    metrics = {}
    
    # R² (coefficient of determination)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot > 1e-10:
        metrics["r2"] = float(1.0 - (ss_res / ss_tot))
    else:
        metrics["r2"] = np.nan
    
    # MSE (Mean Squared Error)
    metrics["mse"] = float(np.mean((y_true - y_pred) ** 2))
    
    # Information Coefficient (IC) - correlation between predictions and actuals
    if return_ic and len(y_true) > 1:
        try:
            ic = float(np.corrcoef(y_true, y_pred)[0, 1])
            metrics["ic"] = ic if not np.isnan(ic) else 0.0
        except:
            metrics["ic"] = 0.0
    else:
        metrics["ic"] = 0.0
    
    # MAE (Mean Absolute Error)
    metrics["mae"] = float(np.mean(np.abs(y_true - y_pred)))
    
    return metrics


def eval_binary_classification(
    y_true: np.ndarray, 
    proba: np.ndarray
) -> Dict[str, float]:
    """
    Evaluate binary classification predictions
    
    Args:
        y_true: True binary labels (0 or 1)
        proba: Predicted probabilities for class 1 [0, 1]
    
    Returns:
        Dictionary of metrics
    """
    from sklearn.metrics import roc_auc_score, log_loss, accuracy_score
    
    y_true = np.asarray(y_true).flatten()
    proba = np.asarray(proba).flatten()
    
    # Remove NaN pairs
    valid_mask = ~(np.isnan(y_true) | np.isnan(proba))
    if valid_mask.sum() == 0:
        return {"roc_auc": np.nan, "logloss": np.nan, "accuracy": np.nan}
    
    y_true = y_true[valid_mask]
    proba = proba[valid_mask]
    
    # Clip probabilities to avoid log(0)
    eps = 1e-8
    proba = np.clip(proba, eps, 1 - eps)
    
    metrics = {}
    
    # ROC AUC
    try:
        if len(np.unique(y_true)) == 2:  # Need both classes
            metrics["roc_auc"] = float(roc_auc_score(y_true, proba))
        else:
            metrics["roc_auc"] = np.nan
    except:
        metrics["roc_auc"] = np.nan
    
    # Log Loss (binary cross-entropy)
    try:
        metrics["logloss"] = float(log_loss(y_true, proba))
    except:
        metrics["logloss"] = np.nan
    
    # Accuracy
    preds = (proba >= 0.5).astype(int)
    metrics["accuracy"] = float(accuracy_score(y_true, preds))
    
    return metrics


def eval_multiclass(
    y_true: np.ndarray, 
    proba: np.ndarray
) -> Dict[str, float]:
    """
    Evaluate multiclass classification predictions
    
    Args:
        y_true: True class labels (integers)
        proba: Predicted probability matrix [n_samples, n_classes]
    
    Returns:
        Dictionary of metrics
    """
    from sklearn.metrics import log_loss, accuracy_score, f1_score
    
    y_true = np.asarray(y_true).flatten().astype(int)
    proba = np.asarray(proba)
    
    # Remove NaN
    valid_mask = ~np.isnan(y_true)
    if valid_mask.sum() == 0:
        return {"cross_entropy": np.nan, "accuracy": np.nan, "macro_f1": np.nan}
    
    y_true = y_true[valid_mask]
    proba = proba[valid_mask]
    
    # Clip probabilities
    eps = 1e-8
    proba = np.clip(proba, eps, 1 - eps)
    # Renormalize
    proba = proba / proba.sum(axis=1, keepdims=True)
    
    metrics = {}
    
    # Cross-entropy (multiclass log loss)
    try:
        metrics["cross_entropy"] = float(log_loss(y_true, proba))
    except:
        metrics["cross_entropy"] = np.nan
    
    # Accuracy
    preds = proba.argmax(axis=1)
    metrics["accuracy"] = float(accuracy_score(y_true, preds))
    
    # Macro F1
    try:
        metrics["macro_f1"] = float(f1_score(y_true, preds, average='macro'))
    except:
        metrics["macro_f1"] = np.nan
    
    return metrics


def evaluate_by_task(
    task_type: TaskType, 
    y_true: np.ndarray, 
    raw_outputs: np.ndarray,
    return_ic: bool = True
) -> Dict[str, float]:
    """
    Evaluate predictions based on task type
    
    Args:
        task_type: TaskType enum
        y_true: True target values
        raw_outputs: Model outputs (predictions for regression, probabilities for classification)
        return_ic: If True, compute IC for regression
    
    Returns:
        Dictionary of task-appropriate metrics
    """
    if task_type == TaskType.REGRESSION:
        return eval_regression(y_true, raw_outputs, return_ic=return_ic)
    elif task_type == TaskType.BINARY_CLASSIFICATION:
        # raw_outputs should be probabilities for class 1
        return eval_binary_classification(y_true, raw_outputs)
    elif task_type == TaskType.MULTICLASS_CLASSIFICATION:
        # raw_outputs should be probability matrix [n_samples, n_classes]
        return eval_multiclass(y_true, raw_outputs)
    else:
        raise ValueError(f"Unknown task_type: {task_type}")


def compute_composite_score(
    metrics: Dict[str, float],
    task_type: TaskType
) -> float:
    """
    Compute a single composite score from metrics for ranking
    
    Args:
        metrics: Dictionary of metrics from evaluate_by_task
        task_type: TaskType enum
    
    Returns:
        Composite score (higher is better)
    """
    if task_type == TaskType.REGRESSION:
        # Use IC as primary (higher is better), R² as secondary
        ic = metrics.get("ic", 0.0)
        r2 = metrics.get("r2", 0.0)
        # Combine: IC weighted 70%, R² weighted 30%
        # CRITICAL: Don't clamp R² to [-1, 1] - preserve negative values for ranking
        # Very negative R² (e.g., -5.0) indicates "catastrophically dangerous" targets
        # that are actively misleading the model. Clamping loses this information.
        # Instead, use a soft normalization that preserves relative ordering:
        # - For positive R²: use as-is (already 0-1 range typically)
        # - For negative R²: preserve the value but apply tanh-like scaling to prevent extreme values
        if r2 > 1.0:
            r2_normalized = 1.0  # Cap extremely positive (likely leakage)
        elif r2 < -10.0:
            r2_normalized = -10.0  # Cap extremely negative (floor for catastrophic)
        else:
            r2_normalized = r2  # Preserve all values in [-10, 1] range
        composite = 0.7 * ic + 0.3 * r2_normalized
        return composite
    
    elif task_type == TaskType.BINARY_CLASSIFICATION:
        # Use AUC as primary, accuracy as secondary
        auc = metrics.get("roc_auc", 0.0)
        acc = metrics.get("accuracy", 0.0)
        # Penalize high logloss
        logloss = metrics.get("logloss", 1.0)
        logloss_penalty = max(0.0, 1.0 - logloss)  # logloss=0 → penalty=1, logloss=1 → penalty=0
        
        composite = 0.6 * auc + 0.3 * acc + 0.1 * logloss_penalty
        return composite
    
    elif task_type == TaskType.MULTICLASS_CLASSIFICATION:
        # Use accuracy as primary, macro F1 as secondary
        acc = metrics.get("accuracy", 0.0)
        f1 = metrics.get("macro_f1", 0.0)
        # Penalize high cross-entropy
        ce = metrics.get("cross_entropy", 1.0)
        ce_penalty = max(0.0, 1.0 - ce)  # ce=0 → penalty=1, ce=1 → penalty=0
        
        composite = 0.5 * acc + 0.3 * f1 + 0.2 * ce_penalty
        return composite
    
    else:
        return 0.0

