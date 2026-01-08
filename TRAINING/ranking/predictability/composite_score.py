# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial

"""
Target Predictability Ranking

Uses multiple model families to evaluate which of your 63 targets are most predictable.
This helps prioritize compute: train models on high-predictability targets first.

Methodology:
1. For each target, train multiple model families on sample data
2. Calculate predictability scores:
   - Model R² scores (cross-validated)
   - Feature importance magnitude (mean absolute SHAP/importance)
   - Consistency across models (low std = high confidence)
3. Rank targets by composite predictability score
4. Output ranked list with recommendations

Usage:
  # Rank all enabled targets
  python SCRIPTS/rank_target_predictability.py
  
  # Test on specific symbols first
  python SCRIPTS/rank_target_predictability.py --symbols AAPL,MSFT,GOOGL
  
  # Rank specific targets
  python SCRIPTS/rank_target_predictability.py --targets peak_60m,valley_60m,swing_high_15m
"""


import argparse
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Union
import pandas as pd
import numpy as np
from dataclasses import dataclass
import yaml
import json
from collections import defaultdict
import warnings

# Add project root FIRST (before any scripts.* imports)
# TRAINING/ranking/rank_target_predictability.py -> parents[2] = repo root
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Add CONFIG directory to path for centralized config loading
_CONFIG_DIR = _REPO_ROOT / "CONFIG"
if str(_CONFIG_DIR) not in sys.path:
    sys.path.insert(0, str(_CONFIG_DIR))

# Try to import config loader
_CONFIG_AVAILABLE = False
try:
    from config_loader import get_cfg, get_safety_config
    _CONFIG_AVAILABLE = True
except ImportError:
    pass  # Logger not yet initialized, will be set up below

# Import logging config utilities
try:
    from CONFIG.logging_config_utils import get_module_logging_config, get_backend_logging_config
    _LOGGING_CONFIG_AVAILABLE = True
except ImportError:
    _LOGGING_CONFIG_AVAILABLE = False
    # Fallback: create a simple config-like object
    class _DummyLoggingConfig:
        def __init__(self):
            self.gpu_detail = False
            self.cv_detail = False
            self.edu_hints = False
            self.detail = False

# Import checkpoint utility (after path is set)
from TRAINING.orchestration.utils.checkpoint import CheckpointManager

# Import unified task type system
from TRAINING.common.utils.task_types import (
    TaskType, TargetConfig, ModelConfig, 
    is_compatible, create_model_configs_from_yaml
)
from TRAINING.common.utils.task_metrics import evaluate_by_task, compute_composite_score
from TRAINING.ranking.utils.target_validation import validate_target, check_cv_compatibility

# Suppress expected warnings (harmless)
warnings.filterwarnings('ignore', message='X does not have valid feature names')
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
warnings.filterwarnings('ignore', category=RuntimeWarning, module='sklearn')
warnings.filterwarnings('ignore', message='invalid value encountered in divide')
warnings.filterwarnings('ignore', message='invalid value encountered in true_divide')

# Setup logging with journald support
from TRAINING.orchestration.utils.logging_setup import setup_logging
logger = setup_logging(
    script_name="rank_target_predictability",
    level=logging.INFO,
    use_journald=True
)

# Composite score calculation

from TRAINING.ranking.predictability.scoring import TargetPredictabilityScore


def validate_slice(
    y_slice: np.ndarray,
    y_pred_slice: Optional[np.ndarray] = None,
    task_type: TaskType = TaskType.REGRESSION,
    min_samples: int = 10,
) -> Tuple[bool, Optional[str]]:
    """
    Validate a single slice (timestamp or symbol) for metric computation.
    
    Foundation function for Phase 3.1.1 per-slice tracking. Currently used
    as a reference implementation; full per-slice tracking requires architectural
    changes to compute metrics per-slice.
    
    Args:
        y_slice: True labels for this slice
        y_pred_slice: Optional predictions for this slice (for NaN checking)
        task_type: Task type (affects validation rules)
        min_samples: Minimum samples required per slice
    
    Returns:
        (is_valid, reason) where:
        - is_valid: True if slice is valid for metric computation
        - reason: None if valid, or error code if invalid:
            - "too_few_samples": n < min_samples
            - "nan_label": NaNs in labels
            - "nan_pred": NaNs in predictions (if provided)
            - "single_class": Classification with only one class
            - "constant_vector": Regression with constant values (spearman undefined)
    """
    # Common checks: NaNs, too few samples
    if len(y_slice) < min_samples:
        return False, "too_few_samples"
    
    if np.any(np.isnan(y_slice)):
        return False, "nan_label"
    
    if y_pred_slice is not None and np.any(np.isnan(y_pred_slice)):
        return False, "nan_pred"
    
    # Task-specific checks
    if task_type == TaskType.REGRESSION:
        # Regression: check for constant vector (spearman undefined)
        if len(y_slice) < 3:
            return False, "too_few_samples"  # Need at least 3 for spearman
        if np.std(y_slice) == 0.0:
            return False, "constant_vector"
    elif task_type in (TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION):
        # Classification: check for single class
        unique_labels = np.unique(y_slice)
        if len(unique_labels) < 2:
            return False, "single_class"
    
    return True, None


def calculate_composite_score(
    auc: float,
    std_score: float,
    mean_importance: float,
    n_models: int,
    task_type: TaskType = TaskType.REGRESSION
) -> Tuple[float, str, str]:
    """
    Calculate composite predictability score with definition and version
    
    Components:
    - Mean score: Higher is better (R² for regression, ROC-AUC/Accuracy for classification)
    - Consistency: Lower std is better
    - Importance magnitude: Higher is better
    - Model agreement: More models = more confidence
    
    Returns:
        Tuple of (composite_score, definition, version)
    """
    
    # Normalize components based on task type
    if task_type == TaskType.REGRESSION:
        # R² can be negative, so normalize to 0-1 range
        score_component = max(0, auc)  # Clamp negative R² to 0
        consistency_component = 1.0 / (1.0 + std_score)
        
        # R²-weighted importance
        if auc > 0:
            importance_component = mean_importance * (1.0 + auc)
        else:
            penalty = abs(auc) * 0.67
            importance_component = mean_importance * max(0.5, 1.0 - penalty)
        
        definition = "0.50 * score_component + 0.25 * consistency_component + 0.25 * importance_component * (1 + model_bonus)"
    else:
        # Classification: ROC-AUC and Accuracy are already 0-1
        score_component = auc  # Already 0-1
        consistency_component = 1.0 / (1.0 + std_score)
        
        # Score-weighted importance (similar logic but for 0-1 scores)
        importance_component = mean_importance * (1.0 + auc)
        
        definition = "0.50 * score_component + 0.25 * consistency_component + 0.25 * importance_component * (1 + model_bonus)"
    
    # Weighted average
    composite = (
        0.50 * score_component +        # 50% weight on score
        0.25 * consistency_component + # 25% on consistency
        0.25 * importance_component    # 25% on score-weighted importance
    )
    
    # Bonus for more models (up to 10% boost)
    model_bonus = min(0.1, n_models * 0.02)
    composite = composite * (1.0 + model_bonus)
    
    version = "v1"
    
    return composite, definition, version


def calculate_composite_score_tstat(
    primary_mean: float,  # Already centered: IC for regression, AUC-excess for classification
    primary_std: float,
    n_slices_valid: int,  # Number of valid slices (renamed from n_cs_valid for clarity)
    n_slices_total: int,  # Total slices before filtering (renamed from n_cs_total)
    task_type: TaskType = TaskType.REGRESSION,
    scoring_config: Optional[Dict[str, Any]] = None,
) -> Tuple[float, str, str, Dict[str, float], str]:
    """
    Calculate composite score using t-stat based skill normalization (Phase 3.1).
    
    Phase 3.1 fixes:
    - SE-based stability (not std-based) for cross-family comparability
    - Skill-gated composite (skill * quality, not additive) to prevent no-skill ranking high
    - Classification centering (primary_mean must be AUC-excess, not raw AUC)
    - Deterministic guards (n_valid < 2, se_floor, tcap)
    
    This provides a bounded [0,1] composite score that's comparable across
    task types and aggregation methods, using t-stat as the universal
    "signal above null" measure.
    
    Args:
        primary_mean: Mean of primary metric, ALREADY CENTERED:
            - Regression: spearman_ic (null baseline ≈ 0)
            - Classification: auc_excess = auc - 0.5 (null baseline ≈ 0)
        primary_std: Std of primary metric
        n_slices_valid: Number of valid slices used
        n_slices_total: Total slices before filtering
        task_type: Task type for se_ref selection
        scoring_config: Optional override for scoring params (loaded from yaml if None)
    
    Returns:
        Tuple of:
        - composite_score_01: Bounded [0,1] composite score
        - definition: Formula string
        - version: Scoring schema version
        - components: Dict of individual component values (for debugging/audit)
        - scoring_signature: SHA256 hash of scoring params for determinism
    """
    import numpy as np
    import hashlib
    import json
    
    # Load scoring config from yaml if not provided
    if scoring_config is None:
        try:
            from TRAINING.ranking.predictability.metrics_schema import _load_metrics_schema
            schema = _load_metrics_schema()
            scoring_config = schema.get("scoring", {})
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Failed to load scoring config from metrics_schema.yaml: {e}, using defaults")
            scoring_config = {}
    
    # Extract params with defaults
    skill_squash_k = scoring_config.get("skill_squash_k", 3.0)
    tcap = scoring_config.get("tcap", 12.0)
    se_floor = scoring_config.get("se_floor", 1e-6)
    default_se_ref = scoring_config.get("se_ref", 0.02)
    se_ref_by_task = scoring_config.get("se_ref_by_task", {})
    weights = scoring_config.get("weights", {"w_cov": 0.3, "w_stab": 0.7})
    model_bonus_cfg = scoring_config.get("model_bonus", {"enabled": True, "max_bonus": 0.10, "per_model": 0.02})
    version = scoring_config.get("version", "1.1")
    composite_form = scoring_config.get("composite_form", "skill_times_quality_v1")
    
    # Get task-specific se_ref
    task_key = {
        TaskType.REGRESSION: "regression",
        TaskType.BINARY_CLASSIFICATION: "binary_classification",
        TaskType.MULTICLASS_CLASSIFICATION: "multiclass_classification",
    }.get(task_type, "regression")
    se_ref = se_ref_by_task.get(task_key, default_se_ref)
    
    # Compute scoring_signature (hash of all effective params for determinism)
    scoring_params = {
        "k": skill_squash_k,
        "tcap": tcap,
        "se_floor": se_floor,
        "se_ref": se_ref,
        "weights": weights,
        "composite_form": composite_form,
        "model_bonus": model_bonus_cfg,
        "version": version,
    }
    # Canonical JSON (sorted keys) for deterministic hashing
    scoring_params_json = json.dumps(scoring_params, sort_keys=True, separators=(',', ':'))
    scoring_signature = hashlib.sha256(scoring_params_json.encode()).hexdigest()
    
    # 1. Compute SE (standard error) from std and n
    # SE = std / sqrt(n)
    if n_slices_valid > 1:
        primary_se = primary_std / np.sqrt(n_slices_valid)
        primary_se = max(primary_se, se_floor)  # Guard: prevent division by zero
    elif n_slices_valid == 1:
        # With only one observation, use a conservative SE estimate
        primary_se = max(primary_std, se_floor)
    else:
        primary_se = se_floor  # No valid slices
    
    # 2. Skill normalization via t-stat
    # t-stat = mean / se
    # Guards: n_valid < 2 → t = 0.0, clamp to [-tcap, tcap]
    if n_slices_valid < 2:
        skill_tstat = 0.0
    elif primary_se > 0:
        skill_tstat = primary_mean / primary_se
        skill_tstat = max(-tcap, min(tcap, skill_tstat))  # Clamp to prevent extreme values
    else:
        skill_tstat = 0.0
    
    # 3. Squash t-stat to [0,1] using sigmoid
    # sigmoid(x/k) where k controls compression
    skill_score_01 = 1.0 / (1.0 + np.exp(-skill_tstat / skill_squash_k))
    
    # 4. Coverage: fraction of valid slices
    coverage01 = n_slices_valid / n_slices_total if n_slices_total > 0 else 1.0
    
    # 5. Stability: SE-based (not std-based) for cross-family comparability
    # stability = 1 - clamp(se / se_ref, 0, 1)
    stability01 = 1.0 - min(1.0, max(0.0, primary_se / se_ref))
    
    # 6. Quality score: weighted combination of coverage and stability
    w_cov = weights.get("w_cov", 0.3)
    w_stab = weights.get("w_stab", 0.7)
    quality01 = w_cov * coverage01 + w_stab * stability01
    
    # 7. Composite: skill-gated quality (prevents no-skill targets from ranking high)
    # composite = skill * quality (multiplicative, not additive)
    composite_base = skill_score_01 * quality01
    
    # 8. Model bonus (multiplicative)
    if model_bonus_cfg.get("enabled", True):
        max_bonus = model_bonus_cfg.get("max_bonus", 0.10)
        per_model = model_bonus_cfg.get("per_model", 0.02)
        # Note: n_models not passed in current signature, skip for now
        # Will be added when model_evaluation.py is updated
        model_bonus = 0.0  # Placeholder until n_models is passed
    else:
        model_bonus = 0.0
    
    composite_score_01 = composite_base * (1.0 + model_bonus)
    
    # Clamp to [0,1] (defensive)
    composite_score_01 = max(0.0, min(1.0, composite_score_01))
    
    definition = (
        f"composite = skill * quality * (1 + model_bonus); "
        f"skill = sigmoid(tstat/{skill_squash_k}); "
        f"quality = {w_cov:.2f}*coverage + {w_stab:.2f}*stability; "
        f"stability = 1 - clamp(se/{se_ref:.3f}, 0, 1); "
        f"tstat = mean / max(se, {se_floor})"
    )
    
    components = {
        "skill_tstat": skill_tstat,
        "skill_score_01": skill_score_01,
        "primary_se": primary_se,
        "coverage01": coverage01,
        "stability01": stability01,
        "quality01": quality01,
        "model_bonus": model_bonus,
        "composite_base": composite_base,
    }
    
    return composite_score_01, definition, f"v{version}", components, scoring_signature

