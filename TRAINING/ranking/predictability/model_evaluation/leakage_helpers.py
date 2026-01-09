# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
# Copyright (c) 2025-2026 Fox ML Infrastructure LLC

"""
Leakage Detection Helper Functions

Helper functions for detecting and analyzing data leakage.
"""

import logging
import numpy as np
from typing import Dict, Optional

from TRAINING.common.utils.task_types import TaskType

logger = logging.getLogger(__name__)

# Try to import config loader
_CONFIG_AVAILABLE = False
try:
    from config_loader import get_cfg, get_safety_config
    _CONFIG_AVAILABLE = True
except ImportError:
    pass


def compute_suspicion_score(
    train_score: float,
    cv_score: Optional[float],
    feature_importances: Dict[str, float],
    task_type: str = 'classification'
) -> float:
    """
    Compute suspicion score for perfect train accuracy.
    
    Higher score = more suspicious (likely real leakage, not just overfitting).
    
    Signals that increase suspicion:
    - CV too good to be true (cv_mean >= 0.85)
    - Generalization gap too small with perfect train (gap < 0.05)
    - Single feature domination (top1_importance / sum >= 0.40)
    
    Signals that decrease suspicion:
    - CV is normal-ish (0.55-0.75)
    - Large gap (classic overfit)
    - Feature dominance not extreme
    """
    suspicion = 0.0
    
    # Signal 1: CV too good to be true
    if cv_score is not None:
        if cv_score >= 0.85:
            suspicion += 0.4  # High suspicion
        elif cv_score >= 0.75:
            suspicion += 0.2  # Medium suspicion
        elif cv_score < 0.55:
            suspicion -= 0.2  # Low suspicion (normal performance)
    
    # Signal 2: Generalization gap (small gap with perfect train = suspicious)
    if cv_score is not None:
        gap = train_score - cv_score
        if gap < 0.05 and train_score >= 0.99:
            suspicion += 0.3  # Very suspicious: perfect train but CV also high
        elif gap > 0.20:
            suspicion -= 0.2  # Large gap = classic overfit (less suspicious)
    
    # Signal 3: Feature dominance
    if feature_importances:
        importances = list(feature_importances.values())
        if importances:
            total_importance = sum(importances)
            if total_importance > 0:
                top1_importance = max(importances)
                dominance_ratio = top1_importance / total_importance
                if dominance_ratio >= 0.50:
                    suspicion += 0.3  # Single feature dominates
                elif dominance_ratio >= 0.40:
                    suspicion += 0.2  # High dominance
                elif dominance_ratio < 0.20:
                    suspicion -= 0.1  # Low dominance (less suspicious)
    
    # Clamp to [0, 1]
    return max(0.0, min(1.0, suspicion))


def detect_leakage(
    auc: float,
    composite_score: float,
    mean_importance: float,
    target: str = "",
    model_scores: Dict[str, float] = None,
    task_type: TaskType = TaskType.REGRESSION
) -> str:
    """
    Detect potential data leakage based on suspicious patterns.
    
    Returns:
        "OK" - No signs of leakage
        "HIGH_R2" - R² > threshold (suspiciously high)
        "INCONSISTENT" - Composite score too high for R² (possible leakage)
        "SUSPICIOUS" - Multiple warning signs
    """
    flags = []
    
    # Load thresholds from config
    if _CONFIG_AVAILABLE:
        try:
            safety_cfg = get_safety_config()
            # safety_config.yaml has a top-level 'safety' key
            safety_section = safety_cfg.get('safety', {})
            leakage_cfg = safety_section.get('leakage_detection', {})
            warning_cfg = leakage_cfg.get('warning_thresholds', {})
        except Exception:
            warning_cfg = {}
    else:
        warning_cfg = {}
    
    # Determine threshold based on task type and target name
    if task_type == TaskType.REGRESSION:
        is_forward_return = target.startswith('fwd_ret_')
        if is_forward_return:
            # For forward returns: R² > 0.50 is suspicious
            reg_cfg = warning_cfg.get('regression', {}).get('forward_return', {})
            high_threshold = float(reg_cfg.get('high', 0.50))
            very_high_threshold = float(reg_cfg.get('very_high', 0.60))
            metric_name = "R²"
        else:
            # For barrier targets: R² > 0.70 is suspicious
            reg_cfg = warning_cfg.get('regression', {}).get('barrier', {})
            high_threshold = float(reg_cfg.get('high', 0.70))
            very_high_threshold = float(reg_cfg.get('very_high', 0.80))
            metric_name = "R²"
    elif task_type == TaskType.BINARY_CLASSIFICATION:
        # ROC-AUC > 0.95 is suspicious (near-perfect classification)
        class_cfg = warning_cfg.get('classification', {})
        high_threshold = float(class_cfg.get('high', 0.90))
        very_high_threshold = float(class_cfg.get('very_high', 0.95))
        metric_name = "ROC-AUC"
    else:  # MULTICLASS_CLASSIFICATION
        # Accuracy > 0.95 is suspicious
        class_cfg = warning_cfg.get('classification', {})
        high_threshold = float(class_cfg.get('high', 0.90))
        very_high_threshold = float(class_cfg.get('very_high', 0.95))
        metric_name = "Accuracy"
    
    # Check 1: Suspiciously high mean score
    if auc > very_high_threshold:
        flags.append("HIGH_SCORE")
        logger.warning(
            f"LEAKAGE WARNING: {metric_name}={auc:.3f} > {very_high_threshold:.2f} "
            f"(extremely high - likely leakage)"
        )
    elif auc > high_threshold:
        flags.append("HIGH_SCORE")
        logger.warning(
            f"LEAKAGE WARNING: {metric_name}={auc:.3f} > {high_threshold:.2f} "
            f"(suspiciously high - investigate)"
        )
    
    # Check 2: Individual model scores too high (even if mean is lower)
    if model_scores:
        high_model_count = sum(1 for score in model_scores.values() 
                              if not np.isnan(score) and score > high_threshold)
        if high_model_count >= 3:  # 3+ models with high scores
            flags.append("HIGH_SCORE")
            logger.warning(
                f"LEAKAGE WARNING: {high_model_count} models have {metric_name} > {high_threshold:.2f} "
                f"(models: {[k for k, v in model_scores.items() if not np.isnan(v) and v > high_threshold]})"
            )
    
    # Check 3: Composite score inconsistent with mean score
    # Load thresholds from config
    try:
        from CONFIG.config_loader import get_cfg
        composite_high_threshold = float(get_cfg("safety.leakage_detection.model_evaluation.composite_score_high_threshold", default=0.5, config_name="safety_config"))
        regression_score_low = float(get_cfg("safety.leakage_detection.model_evaluation.regression_score_low_threshold", default=0.2, config_name="safety_config"))
        classification_score_low = float(get_cfg("safety.leakage_detection.model_evaluation.classification_score_low_threshold", default=0.6, config_name="safety_config"))
    except Exception:
        composite_high_threshold = 0.5
        regression_score_low = 0.2
        classification_score_low = 0.6
    
    score_low_threshold = regression_score_low if task_type == TaskType.REGRESSION else classification_score_low
    if composite_score > composite_high_threshold and auc < score_low_threshold:
        flags.append("INCONSISTENT")
        logger.warning(
            f"LEAKAGE WARNING: Composite={composite_score:.3f} but {metric_name}={auc:.3f} "
            f"(inconsistent - possible leakage)"
        )
    
    # Check 4: Very high importance with low score (might indicate leaked features)
    # Load thresholds from config
    try:
        from CONFIG.config_loader import get_cfg
        importance_high_threshold = float(get_cfg("safety.leakage_detection.model_evaluation.importance_high_threshold", default=0.7, config_name="safety_config"))
        regression_score_very_low = float(get_cfg("safety.leakage_detection.model_evaluation.regression_score_very_low_threshold", default=0.1, config_name="safety_config"))
        classification_score_very_low = float(get_cfg("safety.leakage_detection.model_evaluation.classification_score_very_low_threshold", default=0.5, config_name="safety_config"))
    except Exception:
        importance_high_threshold = 0.7
        regression_score_very_low = 0.1
        classification_score_very_low = 0.5
    
    score_very_low_threshold = regression_score_very_low if task_type == TaskType.REGRESSION else classification_score_very_low
    if mean_importance > importance_high_threshold and auc < score_very_low_threshold:
        flags.append("INCONSISTENT")
        logger.warning(
            f"LEAKAGE WARNING: Importance={mean_importance:.2f} but {metric_name}={auc:.3f} "
            f"(high importance with low {metric_name} - check for leaked features)"
        )
    
    if len(flags) > 1:
        return "SUSPICIOUS"
    elif len(flags) == 1:
        return flags[0]
    else:
        return "OK"

