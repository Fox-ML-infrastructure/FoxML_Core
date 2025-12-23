"""
Copyright (c) 2025-2026 Fox ML Infrastructure LLC

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
Overfitting Detection Utilities

Shared helper for determining when to skip expensive importance computation
due to overfitting. Used by both feature selection and model evaluation.
"""

from typing import Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


def should_skip_expensive_importance(
    train_score: float,
    cv_score: Optional[float] = None,
    val_score: Optional[float] = None,
    n_features: Optional[int] = None,
    config: Optional[Dict[str, Any]] = None
) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Determine if expensive importance computation should be skipped.
    
    Uses policy-based gating with multiple conditions:
    - Train accuracy threshold
    - Train/CV gap
    - Train/Val gap
    - Feature count cap (optional)
    
    Args:
        train_score: Training score (accuracy, R², etc.)
        cv_score: Optional cross-validation mean score
        val_score: Optional validation score
        n_features: Optional number of features (for feature count cap)
        config: Optional config dict with thresholds (from safety_config.feature_importance)
    
    Returns:
        Tuple of (should_skip, reason, metadata):
        - should_skip: True if expensive importance should be skipped
        - reason: String reason for skipping (or "none" if not skipping)
        - metadata: Dict with train_score, cv_score, val_score, gaps, etc.
    """
    # Load thresholds from config
    if config is None:
        config = {}
    
    train_acc_threshold = config.get('overfit_train_acc_threshold', 0.99)
    gap_threshold = config.get('overfit_train_val_gap_threshold', 0.20)
    pvc_feature_count_cap = config.get('pvc_feature_count_cap', None)
    
    metadata = {
        'train_score': train_score,
        'cv_score': cv_score,
        'val_score': val_score,
        'n_features': n_features
    }
    
    # Check 1: Train accuracy threshold
    if train_score >= train_acc_threshold:
        metadata['train_acc_threshold'] = train_acc_threshold
        return True, f"train_acc_{train_score:.4f}_>=_{train_acc_threshold}", metadata
    
    # Check 2: Train/CV gap
    if cv_score is not None:
        gap = train_score - cv_score
        metadata['train_cv_gap'] = gap
        if gap >= gap_threshold:
            metadata['gap_threshold'] = gap_threshold
            return True, f"train_cv_gap_{gap:.4f}_>=_{gap_threshold}", metadata
    
    # Check 3: Train/Val gap
    if val_score is not None:
        gap = train_score - val_score
        metadata['train_val_gap'] = gap
        if gap >= gap_threshold:
            metadata['gap_threshold'] = gap_threshold
            return True, f"train_val_gap_{gap:.4f}_>=_{gap_threshold}", metadata
    
    # Check 4: Feature count cap (optional)
    if pvc_feature_count_cap is not None and n_features is not None:
        if n_features >= pvc_feature_count_cap:
            metadata['pvc_feature_count_cap'] = pvc_feature_count_cap
            return True, f"n_features_{n_features}_>=_{pvc_feature_count_cap}", metadata
    
    return False, "none", metadata

