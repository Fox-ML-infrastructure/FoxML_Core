# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial

"""
Task-Aware Metrics Schema

Provides task-specific target statistics computation with cached schema loading.
Ensures regression targets never emit classification-only metrics like pos_rate,
and classification targets emit proper class balance information.
"""

from functools import lru_cache
from typing import Dict, Any, Optional, List
import numpy as np
import logging

from TRAINING.common.utils.task_types import TaskType

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _load_metrics_schema() -> Dict[str, Any]:
    """
    Load and cache metrics schema from config.
    
    Returns:
        Dict with keys: regression, binary_classification, multiclass_classification
    """
    try:
        import yaml
        from pathlib import Path
        
        # Resolve CONFIG directory relative to this file
        config_dir = Path(__file__).parents[3] / "CONFIG"
        schema_path = config_dir / "ranking" / "metrics_schema.yaml"
        
        if schema_path.exists():
            with open(schema_path, 'r') as f:
                schema = yaml.safe_load(f)
            logger.debug(f"Loaded metrics schema from {schema_path}")
            return schema
        else:
            logger.warning(f"Metrics schema not found at {schema_path}. Using defaults.")
            raise FileNotFoundError(schema_path)
    except Exception as e:
        logger.warning(f"Failed to load metrics schema: {e}. Using defaults.")
        # Fallback defaults
        return {
            "regression": {
                "target_stats": ["y_mean", "y_std", "y_min", "y_max", "y_finite_pct"],
                "exclude": ["pos_rate", "class_balance"]
            },
            "binary_classification": {
                "target_stats": ["pos_rate", "class_balance"],
                "pos_label": 1,
                "exclude": ["y_mean", "y_std"]
            },
            "multiclass_classification": {
                "target_stats": ["class_balance", "n_classes"],
                "exclude": ["pos_rate", "y_mean", "y_std"]
            }
        }


def get_task_metrics_schema(task_type: TaskType) -> Dict[str, Any]:
    """
    Get metrics schema for a specific task type.
    
    Args:
        task_type: TaskType enum (REGRESSION, BINARY_CLASSIFICATION, MULTICLASS_CLASSIFICATION)
    
    Returns:
        Dict with target_stats, exclude, and task-specific config (e.g., pos_label)
    """
    schema = _load_metrics_schema()
    key = {
        TaskType.REGRESSION: "regression",
        TaskType.BINARY_CLASSIFICATION: "binary_classification",
        TaskType.MULTICLASS_CLASSIFICATION: "multiclass_classification",
    }.get(task_type, "regression")
    return schema.get(key, schema.get("regression", {}))


def compute_target_stats(
    task_type: TaskType,
    y: np.ndarray,
    *,
    pos_label: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Compute task-appropriate target statistics.
    
    This is the single source of truth for target distribution stats.
    Replaces unconditional pos_rate computation with task-aware logic.
    
    Args:
        task_type: TaskType enum
        y: Target values array
        pos_label: Explicit positive label for binary classification.
                   If None, uses schema default (typically 1).
    
    Returns:
        Dict of stats appropriate for the task type:
        - Regression: y_mean, y_std, y_min, y_max, y_finite_pct
        - Binary: pos_rate (using pos_label), class_balance
        - Multiclass: class_balance dict, n_classes
    """
    stats: Dict[str, Any] = {}
    
    # Handle edge cases
    if y is None or not hasattr(y, '__iter__') or len(y) == 0:
        return {"y_finite_pct": 0.0}
    
    # Get clean (finite) values
    y_arr = np.asarray(y)
    finite_mask = np.isfinite(y_arr)
    y_clean = y_arr[finite_mask]
    
    if len(y_clean) == 0:
        return {"y_finite_pct": 0.0}
    
    # Compute finite percentage (useful for all task types)
    y_finite_pct = float(len(y_clean) / len(y_arr))
    
    if task_type == TaskType.REGRESSION:
        # Distribution stats for continuous targets
        stats["y_mean"] = float(np.mean(y_clean))
        stats["y_std"] = float(np.std(y_clean))
        stats["y_min"] = float(np.min(y_clean))
        stats["y_max"] = float(np.max(y_clean))
        stats["y_finite_pct"] = y_finite_pct
        # NOTE: We intentionally do NOT emit pos_rate for regression
        
    elif task_type == TaskType.BINARY_CLASSIFICATION:
        # Use explicit pos_label, or fall back to schema default
        if pos_label is None:
            schema = get_task_metrics_schema(task_type)
            pos_label = schema.get("pos_label", 1)
        
        # pos_rate = fraction of samples with positive label
        stats["pos_rate"] = float(np.mean(y_clean == pos_label))
        
        # class_balance = {label: count} for auditability
        # NOTE: Use string keys for Parquet/PyArrow compatibility (int keys fail serialization)
        unique, counts = np.unique(y_clean, return_counts=True)
        stats["class_balance"] = {str(int(u)): int(c) for u, c in zip(unique, counts)}
        stats["y_finite_pct"] = y_finite_pct
        
    elif task_type == TaskType.MULTICLASS_CLASSIFICATION:
        # class_balance for multiclass (no pos_rate - it's meaningless)
        # NOTE: Use string keys for Parquet/PyArrow compatibility (int keys fail serialization)
        unique, counts = np.unique(y_clean, return_counts=True)
        stats["class_balance"] = {str(int(u)): int(c) for u, c in zip(unique, counts)}
        stats["n_classes"] = len(unique)
        stats["y_finite_pct"] = y_finite_pct
        # NOTE: We intentionally do NOT emit pos_rate for multiclass
        
    else:
        # Unknown task type - log warning and return minimal stats
        logger.warning(f"Unknown task_type {task_type}, returning minimal stats")
        stats["y_finite_pct"] = y_finite_pct
    
    return stats


def get_excluded_metrics(task_type: TaskType) -> List[str]:
    """
    Get list of metrics that should be excluded for a task type.
    
    Useful for filtering output before persistence.
    
    Args:
        task_type: TaskType enum
    
    Returns:
        List of metric field names that should NOT appear for this task type
    """
    schema = get_task_metrics_schema(task_type)
    return schema.get("exclude", [])


def get_canonical_metric_name(task_type: TaskType, view: str, metric_type: str = "primary") -> str:
    """
    Get canonical metric name for a task type and view combination.
    
    This is the single source of truth for metric naming, replacing the
    overloaded 'auc' field that stored different metrics depending on task type.
    
    Format: <metric_base>__<view>__<aggregation>
    
    Args:
        task_type: TaskType enum (REGRESSION, BINARY_CLASSIFICATION, etc.)
        view: "CROSS_SECTIONAL" or "SYMBOL_SPECIFIC"
        metric_type: "primary" or "std" (default: "primary")
    
    Returns:
        Canonical metric name, e.g.:
        - REGRESSION + CROSS_SECTIONAL -> "spearman_ic__cs__mean"
        - BINARY_CLASSIFICATION + CROSS_SECTIONAL -> "roc_auc__cs__mean"
        - REGRESSION + SYMBOL_SPECIFIC -> "r2__sym__mean"
    
    Examples:
        >>> get_canonical_metric_name(TaskType.REGRESSION, "CROSS_SECTIONAL")
        'spearman_ic__cs__mean'
        >>> get_canonical_metric_name(TaskType.BINARY_CLASSIFICATION, "SYMBOL_SPECIFIC", "std")
        'roc_auc__sym__std'
    """
    schema = _load_metrics_schema()
    canonical = schema.get("canonical_names", {})
    
    # Map TaskType to schema key
    task_key = {
        TaskType.REGRESSION: "regression",
        TaskType.BINARY_CLASSIFICATION: "binary_classification",
        TaskType.MULTICLASS_CLASSIFICATION: "multiclass_classification",
    }.get(task_type, "regression")
    
    # Map view to schema key
    view_key = "cross_sectional" if view.upper() == "CROSS_SECTIONAL" else "symbol_specific"
    
    # Get canonical names for this task + view combination
    task_canonical = canonical.get(task_key, {})
    view_canonical = task_canonical.get(view_key, {})
    
    # Get requested metric type (primary or std)
    metric_name = view_canonical.get(metric_type)
    
    if metric_name:
        return metric_name
    
    # Fallback: construct a reasonable default
    fallback_map = {
        ("regression", "cross_sectional", "primary"): "spearman_ic__cs__mean",
        ("regression", "cross_sectional", "std"): "spearman_ic__cs__std",
        ("regression", "symbol_specific", "primary"): "r2__sym__mean",
        ("regression", "symbol_specific", "std"): "r2__sym__std",
        ("binary_classification", "cross_sectional", "primary"): "roc_auc__cs__mean",
        ("binary_classification", "cross_sectional", "std"): "roc_auc__cs__std",
        ("binary_classification", "symbol_specific", "primary"): "roc_auc__sym__mean",
        ("binary_classification", "symbol_specific", "std"): "roc_auc__sym__std",
        ("multiclass_classification", "cross_sectional", "primary"): "accuracy__cs__mean",
        ("multiclass_classification", "cross_sectional", "std"): "accuracy__cs__std",
        ("multiclass_classification", "symbol_specific", "primary"): "accuracy__sym__mean",
        ("multiclass_classification", "symbol_specific", "std"): "accuracy__sym__std",
    }
    
    fallback = fallback_map.get((task_key, view_key, metric_type))
    if fallback:
        logger.debug(f"Using fallback canonical name: {fallback}")
        return fallback
    
    # Ultimate fallback
    logger.warning(f"No canonical name found for {task_key}/{view_key}/{metric_type}, using 'primary_score'")
    return "primary_score"


def get_canonical_metric_names_for_output(task_type: TaskType, view: str, primary_value: float, std_value: float) -> Dict[str, float]:
    """
    Get a dict of canonical metric names populated with values.
    
    Use this when building metrics output for snapshots.
    Includes deprecated 'auc' field for backward compatibility.
    
    Args:
        task_type: TaskType enum
        view: "CROSS_SECTIONAL" or "SYMBOL_SPECIFIC"
        primary_value: The primary metric value (e.g., mean score)
        std_value: The standard deviation of the metric
    
    Returns:
        Dict with canonical names and backward-compat 'auc':
        {
            "spearman_ic__cs__mean": 0.058,
            "spearman_ic__cs__std": 0.103,
            "auc": 0.058,  # DEPRECATED
            "std_score": 0.103  # DEPRECATED
        }
    """
    primary_name = get_canonical_metric_name(task_type, view, "primary")
    std_name = get_canonical_metric_name(task_type, view, "std")
    
    return {
        primary_name: primary_value,
        std_name: std_value,
        # Backward compatibility fields (deprecated)
        "auc": primary_value,
        "std_score": std_value,
    }
