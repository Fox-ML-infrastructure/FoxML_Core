# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
# Copyright (c) 2025-2026 Fox ML Infrastructure LLC

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


def get_primary_metric_baseline(task_type: TaskType) -> float:
    """
    Get the null baseline for the primary metric of a task type.
    
    This is used for computing skill_mean = mean - baseline.
    
    Args:
        task_type: TaskType enum
    
    Returns:
        Baseline value:
        - Regression: 0.0 (IC has null baseline ≈ 0)
        - Classification: 0.5 (AUC has null baseline = 0.5)
    """
    if task_type == TaskType.REGRESSION:
        return 0.0
    elif task_type in (TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION):
        return 0.5
    else:
        return 0.0  # Default fallback


def build_clean_metrics_dict(
    result: 'TargetPredictabilityScore',
    target_stats: Optional[Dict[str, Any]] = None,
    n_features_pre: Optional[int] = None,
    n_features_post_prune: Optional[int] = None,
    features_safe: Optional[int] = None,
    fold_timestamps: Optional[List[Dict[str, Any]]] = None,
    leakage_info: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build a clean, grouped, non-redundant metrics dictionary.
    
    This replaces the flat, duplicate-heavy metrics output with a structured,
    task-gated, semantically unambiguous format.
    
    Structure:
    - schema: Version info
    - scope: View and task_family
    - primary_metric: Single source of truth for primary metric
    - coverage: Counts (as ints)
    - features: Feature counts (as ints)
    - y_stats or label_stats: Task-specific target stats
    - models: Model scores
    - score: Composite score and components
    
    Args:
        result: TargetPredictabilityScore object
        target_stats: Optional dict from compute_target_stats()
        n_features_pre: Number of features before pruning
        n_features_post_prune: Number of features after pruning
        features_safe: Number of safe features
        fold_timestamps: Optional fold timestamp info
        leakage_info: Optional leakage detection info
    
    Returns:
        Clean, grouped metrics dict
    """
    from TRAINING.common.utils.task_types import TaskType
    
    # Determine task family
    task_family = "regression" if result.task_type == TaskType.REGRESSION else "classification"
    
    # Get baseline for skill calculation
    baseline = get_primary_metric_baseline(result.task_type)
    
    # Get effective primary metric values (centered for classification)
    primary_mean = result.primary_metric_mean if result.primary_metric_mean is not None else result.auc
    primary_std = result.primary_metric_std if result.primary_metric_std is not None else result.std_score
    primary_se = result.primary_se if result.primary_se is not None else None
    
    # For classification, primary_mean is already AUC-excess (centered)
    # For regression, primary_mean is IC (already centered at 0)
    # Compute skill_mean (distance from baseline)
    if result.task_type in (TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION):
        # For classification, if we have raw AUC, compute excess
        if result.auc_mean_raw is not None:
            skill_mean = result.auc_excess_mean if result.auc_excess_mean is not None else (result.auc_mean_raw - baseline)
        else:
            # primary_mean is already AUC-excess
            skill_mean = primary_mean
    else:
        # For regression, primary_mean is already IC (centered at 0)
        skill_mean = primary_mean
    
    skill_se = primary_se if primary_se is not None else None
    
    # Get canonical metric name
    primary_name = result.primary_metric_name
    direction = "higher_is_better"  # All our metrics are higher-is-better
    
    # Build primary_metric object (single source of truth)
    primary_metric = {
        "family": task_family,
        "name": primary_name,
        "direction": direction,
        "baseline": baseline,
        "mean": float(primary_mean),
        "std": float(primary_std),
    }
    
    if primary_se is not None:
        primary_metric["se"] = float(primary_se)
    
    primary_metric["skill_mean"] = float(skill_mean)
    if skill_se is not None:
        primary_metric["skill_se"] = float(skill_se)
    
    # Add skill01 (normalized [0,1] score for unified routing)
    skill01 = result.skill01 if hasattr(result, 'skill01') else None
    if skill01 is not None:
        primary_metric["skill01"] = float(skill01)
    
    # Build schema group
    schema = {
        "metrics": getattr(result, 'metrics_schema_version', '1.1'),
        "scoring": getattr(result, 'scoring_schema_version', '1.1'),
    }
    
    # Build scope group
    scope = {
        "view": result.view,
        "task_family": task_family,
    }
    
    # Build coverage group (counts as ints)
    coverage = {}
    if result.n_cs_valid is not None:
        coverage["n_cs_valid"] = int(result.n_cs_valid)
    if result.n_cs_total is not None:
        coverage["n_cs_total"] = int(result.n_cs_total)
    # n_effective might come from target_stats or be computed elsewhere
    if target_stats and "n_effective" in target_stats:
        coverage["n_effective"] = int(target_stats["n_effective"])
    
    # Build features group (counts as ints)
    features = {}
    if n_features_pre is not None:
        features["pre"] = int(n_features_pre)
    if n_features_post_prune is not None:
        features["post_prune"] = int(n_features_post_prune)
    if features_safe is not None:
        features["safe"] = int(features_safe)
    
    # Build y_stats or label_stats (task-specific)
    target_stats_group = {}
    if target_stats:
        if result.task_type == TaskType.REGRESSION:
            # Regression: y_stats
            for key in ["y_mean", "y_std", "y_min", "y_max", "y_finite_pct"]:
                if key in target_stats:
                    target_stats_group[key] = target_stats[key]
        elif result.task_type == TaskType.BINARY_CLASSIFICATION:
            # Binary: label_stats with pos_rate and class_balance
            if "pos_rate" in target_stats:
                target_stats_group["pos_rate"] = target_stats["pos_rate"]
            if "class_balance" in target_stats:
                target_stats_group["class_balance"] = target_stats["class_balance"]
            if "y_finite_pct" in target_stats:
                target_stats_group["finite_pct"] = target_stats["y_finite_pct"]
        elif result.task_type == TaskType.MULTICLASS_CLASSIFICATION:
            # Multiclass: label_stats with class_balance
            if "class_balance" in target_stats:
                target_stats_group["class_balance"] = target_stats["class_balance"]
            if "n_classes" in target_stats:
                target_stats_group["n_classes"] = int(target_stats["n_classes"])
            if "y_finite_pct" in target_stats:
                target_stats_group["finite_pct"] = target_stats["y_finite_pct"]
    
    # Build models group
    models = {
        "n": int(result.n_models),
        "scores": {k: float(v) for k, v in result.model_scores.items()},
    }
    
    # Build score group
    score = {
        "composite": float(result.composite_score),
        "components": {},
    }
    
    # Add component details
    if result.mean_importance is not None:
        score["components"]["mean_importance"] = float(result.mean_importance)
    if result.consistency is not None:
        # Rename consistency to consistency_penalty for clarity
        score["components"]["consistency_penalty"] = float(result.consistency)
    if result.primary_metric_tstat is not None:
        score["components"]["skill_tstat"] = float(result.primary_metric_tstat)
    if result.primary_se is not None:
        score["components"]["primary_se"] = float(result.primary_se)
    
    # Add composite definition and version if available
    if result.composite_definition:
        score["definition"] = result.composite_definition
    if result.composite_version:
        score["version"] = result.composite_version
    if result.scoring_signature:
        score["signature"] = result.scoring_signature
    
    # Build final dict
    clean_metrics = {
        "schema": schema,
        "scope": scope,
        "primary_metric": primary_metric,
    }
    
    if coverage:
        clean_metrics["coverage"] = coverage
    if features:
        clean_metrics["features"] = features
    
    # Add task-specific stats group
    if target_stats_group:
        if result.task_type == TaskType.REGRESSION:
            clean_metrics["y_stats"] = target_stats_group
        else:
            clean_metrics["label_stats"] = target_stats_group
    
    clean_metrics["models"] = models
    clean_metrics["score"] = score
    
    # Add optional fields
    if fold_timestamps:
        clean_metrics["fold_timestamps"] = fold_timestamps
    
    if leakage_info:
        clean_metrics["leakage"] = leakage_info
    elif result.leakage_flag:
        clean_metrics["leakage_flag"] = result.leakage_flag
    
    # Add invalid_reason_counts if available
    if result.invalid_reason_counts:
        clean_metrics["coverage"]["invalid_reason_counts"] = result.invalid_reason_counts
    
    # === DUAL RANKING: Add mismatch telemetry (2026-01 filtering mismatch fix) ===
    if hasattr(result, 'score_screen') and result.score_screen is not None:
        clean_metrics["score"] = clean_metrics.get("score", {})
        clean_metrics["score"]["screen"] = float(result.score_screen)
        if hasattr(result, 'score_strict') and result.score_strict is not None:
            clean_metrics["score"]["strict"] = float(result.score_strict)
        if hasattr(result, 'strict_viability_flag') and result.strict_viability_flag is not None:
            clean_metrics["score"]["strict_viability"] = bool(result.strict_viability_flag)
        if hasattr(result, 'rank_delta') and result.rank_delta is not None:
            clean_metrics["score"]["rank_delta"] = int(result.rank_delta)
    
    if hasattr(result, 'mismatch_telemetry') and result.mismatch_telemetry:
        clean_metrics["mismatch_telemetry"] = result.mismatch_telemetry
    
    return clean_metrics


def build_clean_feature_selection_metrics(
    mean_consensus: float,
    std_consensus: float,
    top_feature_score: float,
    n_features_selected: int,
    n_successful_families: int,
    n_candidates: Optional[int] = None,
    selection_mode: Optional[str] = None,
    selection_params: Optional[Dict[str, Any]] = None,
    task_type: Optional[TaskType] = None,
    view: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Build clean, grouped metrics dict for feature selection stage.
    
    Similar structure to TARGET_RANKING but with FS-specific fields.
    
    Args:
        mean_consensus: Mean consensus score across models
        std_consensus: Std of consensus scores
        top_feature_score: Score of top feature
        n_features_selected: Number of features selected (int)
        n_successful_families: Number of successful model families (int)
        n_candidates: Optional number of candidate features
        selection_mode: Optional selection mode (rank_only, top_k, etc.)
        selection_params: Optional selection parameters
        task_type: Optional task type for task-gating
        view: Optional view type
    
    Returns:
        Clean, grouped metrics dict
    """
    # Build schema group
    schema = {
        "metrics": "1.1",
        "scoring": "1.1",
    }
    
    # Build scope group
    scope = {}
    if view:
        scope["view"] = view
    if task_type:
        task_family = "regression" if task_type == TaskType.REGRESSION else "classification"
        scope["task_family"] = task_family
    
    # Build primary_metric (consensus score)
    primary_metric = {
        "name": "consensus_score",
        "direction": "higher_is_better",
        "baseline": 0.0,
        "mean": float(mean_consensus),
        "std": float(std_consensus),
        "skill_mean": float(mean_consensus),  # Consensus already centered at 0
        "skill_se": None,  # SE not computed for FS consensus
    }
    
    # Build features group (all counts as ints)
    features = {
        "selected": int(n_features_selected),
        "successful_families": int(n_successful_families),
    }
    if n_candidates is not None:
        features["candidates"] = int(n_candidates)
    
    # Build selection group (if selection info available)
    selection = {}
    if selection_mode:
        selection["mode"] = selection_mode
    if selection_params:
        selection["params"] = selection_params
    
    # Build score group
    score = {
        "composite": float(mean_consensus),  # Use consensus as composite
        "components": {
            "top_feature_score": float(top_feature_score),
        },
    }
    
    # Build final dict
    clean_metrics = {
        "schema": schema,
        "primary_metric": primary_metric,
        "features": features,
        "score": score,
    }
    
    if scope:
        clean_metrics["scope"] = scope
    if selection:
        clean_metrics["selection"] = selection
    
    return clean_metrics
