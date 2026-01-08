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
    primary_mean: float,
    primary_std: float,
    n_cs_valid: int,
    n_cs_total: int,
    mean_importance: float,
    n_models: int,
    task_type: TaskType = TaskType.REGRESSION,
    scoring_config: Optional[Dict[str, Any]] = None,
) -> Tuple[float, str, str, Dict[str, float]]:
    """
    Calculate composite score using t-stat based skill normalization (P1).
    
    This provides a bounded [0,1] composite score that's comparable across
    task types and aggregation methods, using t-stat as the universal
    "signal above null" measure.
    
    Args:
        primary_mean: Mean of primary metric (e.g., spearman_ic__cs__mean)
        primary_std: Std of primary metric
        n_cs_valid: Number of valid cross-sections used
        n_cs_total: Total cross-sections before filtering
        mean_importance: Mean feature importance
        n_models: Number of model families used
        task_type: Task type for std_ref selection
        scoring_config: Optional override for scoring params (loaded from yaml if None)
    
    Returns:
        Tuple of:
        - composite_score_01: Bounded [0,1] composite score
        - definition: Formula string
        - version: Scoring schema version
        - components: Dict of individual component values (for debugging/audit)
    """
    import numpy as np
    
    # Load scoring config from yaml if not provided
    if scoring_config is None:
        try:
            from TRAINING.ranking.predictability.metrics_schema import _load_metrics_schema
            schema = _load_metrics_schema()
            scoring_config = schema.get("scoring", {})
        except Exception:
            scoring_config = {}
    
    # Extract params with defaults
    skill_squash_k = scoring_config.get("skill_squash_k", 3.0)
    default_std_ref = scoring_config.get("std_ref", 0.2)
    std_ref_by_task = scoring_config.get("std_ref_by_task", {})
    weights = scoring_config.get("weights", {"performance": 0.50, "coverage": 0.25, "stability": 0.25})
    model_bonus_cfg = scoring_config.get("model_bonus", {"enabled": True, "max_bonus": 0.10, "per_model": 0.02})
    version = scoring_config.get("version", "1.0")
    
    # Get task-specific std_ref
    task_key = {
        TaskType.REGRESSION: "regression",
        TaskType.BINARY_CLASSIFICATION: "binary_classification",
        TaskType.MULTICLASS_CLASSIFICATION: "multiclass_classification",
    }.get(task_type, "regression")
    std_ref = std_ref_by_task.get(task_key, default_std_ref)
    
    # 1. Skill normalization via t-stat
    # t-stat = mean / (std / sqrt(n)) = mean * sqrt(n) / std
    if n_cs_valid > 1 and primary_std > 0:
        skill_tstat = primary_mean * np.sqrt(n_cs_valid) / primary_std
    elif n_cs_valid == 1:
        # With only one observation, assume t-stat based on mean alone
        skill_tstat = primary_mean / 0.1 if primary_mean > 0 else 0.0
    else:
        skill_tstat = 0.0
    
    # Squash t-stat to [0,1] using sigmoid
    # sigmoid(x/k) where k controls compression
    skill_score_01 = 1.0 / (1.0 + np.exp(-skill_tstat / skill_squash_k))
    
    # 2. Coverage: fraction of valid cross-sections
    coverage = n_cs_valid / n_cs_total if n_cs_total > 0 else 1.0
    
    # 3. Stability: inverse of normalized std
    # stability = 1 - clamp(std / std_ref, 0, 1)
    stability = 1.0 - min(1.0, max(0.0, primary_std / std_ref))
    
    # 4. Weighted composite
    w_perf = weights.get("performance", 0.50)
    w_cov = weights.get("coverage", 0.25)
    w_stab = weights.get("stability", 0.25)
    
    composite_base = (
        w_perf * skill_score_01 +
        w_cov * coverage +
        w_stab * stability
    )
    
    # 5. Model bonus (multiplicative)
    if model_bonus_cfg.get("enabled", True):
        max_bonus = model_bonus_cfg.get("max_bonus", 0.10)
        per_model = model_bonus_cfg.get("per_model", 0.02)
        model_bonus = min(max_bonus, n_models * per_model)
    else:
        model_bonus = 0.0
    
    composite_score_01 = composite_base * (1.0 + model_bonus)
    
    # Clamp to [0,1] (shouldn't be needed but defensive)
    composite_score_01 = max(0.0, min(1.0, composite_score_01))
    
    definition = (
        f"composite = ({w_perf:.2f}*skill + {w_cov:.2f}*coverage + {w_stab:.2f}*stability) * (1 + model_bonus); "
        f"skill = sigmoid(tstat/{skill_squash_k}); stability = 1 - clamp(std/{std_ref}, 0, 1)"
    )
    
    components = {
        "skill_tstat": skill_tstat,
        "skill_score_01": skill_score_01,
        "coverage": coverage,
        "stability": stability,
        "model_bonus": model_bonus,
        "composite_base": composite_base,
    }
    
    return composite_score_01, definition, f"v{version}", components
