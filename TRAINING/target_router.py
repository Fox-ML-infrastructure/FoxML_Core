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
Target Task Router - Maps target columns to training specifications

Routes enhanced target families (TTH, ordinal, ranking, etc.) to appropriate
training objectives, metrics, and data assembly requirements.
"""


import re
import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Tuple
import logging

logger = logging.getLogger(__name__)

@dataclass
class TaskSpec:
    """Specification for how to train a specific target."""
    task: str                      # 'regression' | 'binary' | 'multiclass' | 'ranking' | 'survival'
    objective: str                 # learner-specific objective key
    metrics: List[str]             # evaluation metrics
    needs_group: bool = False      # ranking tasks need group sizes
    n_classes: Optional[int] = None  # for multiclass
    censor_col: Optional[str] = None  # for survival (future)
    class_weighting: Optional[str] = None  # 'balanced' or None
    label_type: str = 'float32'    # numpy dtype for labels
    
    def __repr__(self):
        return f"TaskSpec({self.task}, obj={self.objective}, needs_group={self.needs_group})"


# Target patterns → task specifications
# Order matters: more specific patterns first
TARGET_PATTERNS = [
    # --- Binary classification ---
    (r'^(topk_\d+[mhd]_\d+|slip_gt_\d+bps|y_will_(peak|valley)_\d+[mhd]_[0-9.]+)$',
     TaskSpec('binary', 'binary', ['auc', 'average_precision'], class_weighting='balanced', label_type='int32')),
    
    # --- Multiclass: Asymmetric barrier hit {-1, 0, +1} ---
    (r'^hit_asym_\d+[mhd]_[0-9.]+_[0-9.]+$',
     TaskSpec('multiclass', 'multiclass', ['multi_logloss', 'macro_f1'], n_classes=3, label_type='int32')),
    
    # --- Multiclass: First touch {-1, 0, +1} ---
    (r'^(y_first_touch|hit_direction)_\d+[mhd]_[0-9.]+$',
     TaskSpec('multiclass', 'multiclass', ['multi_logloss', 'macro_f1'], n_classes=3, label_type='int32')),
    
    # --- Multiclass/Ordinal: Ordinal buckets {-3..+3} ---
    (r'^(ret_ord|mfe_ord|mdd_ord)_\d+[mhd]$',
     TaskSpec('multiclass', 'multiclass', ['multi_logloss', 'qwk'], n_classes=7, label_type='int32')),
    
    # --- Multiclass: Regimes (3-5 classes typically) ---
    (r'^regime_(trend|vol|liq)_\d+$',
     TaskSpec('multiclass', 'multiclass', ['multi_logloss', 'macro_f1'], label_type='int32')),
    
    # --- Ranking: Cross-sectional rank targets ---
    (r'^(xrank_(ret|idio)_\d+[mhd])$',
     TaskSpec('ranking', 'lambdarank', ['ndcg@10', 'map@10', 'rankic'], needs_group=True, label_type='float32')),
    
    # --- Regression: Time-to-hit (start with regression, AFT later) ---
    (r'^(tth|tth_abs)_\d+[mhd]_[0-9.]+$',
     TaskSpec('regression', 'regression', ['rmse', 'mae', 'spearman'], label_type='float32')),
    
    # --- Regression: Path quality metrics ---
    (r'^(mfe_share|time_in_profit|flipcount)_\d+[mhd]$',
     TaskSpec('regression', 'regression', ['rmse', 'mae', 'spearman'], label_type='float32')),
    
    # --- Regression: Idiosyncratic returns ---
    (r'^idio_ret_\d+[mhd]$',
     TaskSpec('regression', 'regression', ['rmse', 'mae', 'spearman', 'rankic'], label_type='float32')),
    
    # --- Regression: Forward returns (default) ---
    (r'^fwd_ret_\d+[mhd]$',
     TaskSpec('regression', 'regression', ['rmse', 'mae', 'spearman', 'rankic'], label_type='float32')),
    
    # --- Regression: Z-scores ---
    (r'^ret_zscore_\d+[mhd]$',
     TaskSpec('regression', 'regression', ['rmse', 'mae', 'spearman'], label_type='float32')),
    
    # --- Regression: MFE/MDD values ---
    (r'^(mfe|mdd)_\d+[mhd]_[0-9.]+$',
     TaskSpec('regression', 'regression', ['rmse', 'mae'], label_type='float32')),
    
    # --- Regression: Probability estimates ---
    (r'^p_(up|down)_\d+[mhd]_[0-9.]+$',
     TaskSpec('regression', 'regression', ['rmse', 'mae'], label_type='float32')),
]


def spec_from_target(col: str) -> Optional[TaskSpec]:
    """
    Route a target column name to its training specification.
    
    Args:
        col: Target column name (e.g., 'ret_ord_15m', 'xrank_ret_30m')
    
    Returns:
        TaskSpec or None if no pattern matches
    """
    for pattern, spec in TARGET_PATTERNS:
        if re.match(pattern, col):
            logger.debug(f"[Router] {col} → {spec.task} (obj={spec.objective})")
            return spec
    
    logger.warning(f"[Router] No pattern matched for target: {col}. Defaulting to regression.")
    # Default fallback: treat as regression
    return TaskSpec('regression', 'regression', ['rmse', 'mae'], label_type='float32')


def compute_class_weights(y: np.ndarray, method: str = 'balanced') -> np.ndarray:
    """
    Compute sample weights for class imbalance.
    
    Args:
        y: Label array
        method: 'balanced' or None
    
    Returns:
        Sample weights (same length as y)
    """
    if method != 'balanced':
        return np.ones(len(y), dtype=np.float32)
    
    try:
        from sklearn.utils.class_weight import compute_class_weight
        
        classes = np.unique(y[~np.isnan(y)])
        if len(classes) < 2:
            return np.ones(len(y), dtype=np.float32)
        
        class_weights = compute_class_weight('balanced', classes=classes, y=y)
        
        # Map class weights to sample weights
        weight_map = dict(zip(classes, class_weights))
        sample_weights = np.array([weight_map.get(label, 1.0) for label in y], dtype=np.float32)
        
        logger.info(f"[Class Weights] Classes: {classes}, Weights: {class_weights}")
        return sample_weights
        
    except Exception as e:
        logger.warning(f"[Class Weights] Failed to compute: {e}. Using uniform weights.")
        return np.ones(len(y), dtype=np.float32)


def encode_multiclass_labels(y: np.ndarray, n_classes: Optional[int] = None) -> Tuple[np.ndarray, dict]:
    """
    Encode multiclass labels to 0..K-1 range.
    
    Args:
        y: Raw labels (may be {-1, 0, +1} or {-3..+3} etc.)
        n_classes: Expected number of classes (for validation)
    
    Returns:
        (encoded_labels, mapping_dict) where mapping_dict maps original → encoded
    """
    unique_labels = np.unique(y[~np.isnan(y)])
    unique_labels = np.sort(unique_labels)  # Preserve order for ordinal
    
    if n_classes and len(unique_labels) != n_classes:
        logger.warning(f"[Label Encoding] Expected {n_classes} classes, found {len(unique_labels)}")
    
    # Create mapping: original label → 0-indexed
    label_map = {orig: idx for idx, orig in enumerate(unique_labels)}
    
    # Apply encoding
    encoded = np.array([label_map.get(label, -1) for label in y], dtype=np.int32)
    
    # Replace any unmapped (-1) with mode
    if (encoded == -1).any():
        mode = np.bincount(encoded[encoded >= 0]).argmax()
        encoded[encoded == -1] = mode
    
    logger.info(f"[Label Encoding] Mapped {len(label_map)} classes: {label_map}")
    
    # Inverse mapping for decoding
    inverse_map = {idx: orig for orig, idx in label_map.items()}
    
    return encoded, inverse_map


def build_ranking_groups(time_vals: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """
    Build group sizes for ranking objectives (per-timestamp cross-sections).
    
    Args:
        time_vals: Timestamp values (same length as X/y), or None
    
    Returns:
        Array of group sizes (one per unique timestamp) or None
    """
    if time_vals is None:
        logger.warning("[Ranking Groups] No time values provided. Cannot build groups.")
        return None
    
    try:
        import pandas as pd
        
        # Count samples per unique timestamp
        df = pd.DataFrame({'ts': time_vals})
        group_sizes = df.groupby('ts').size().values
        
        logger.info(f"[Ranking Groups] Built {len(group_sizes)} groups, sizes: min={group_sizes.min()}, max={group_sizes.max()}, mean={group_sizes.mean():.1f}")
        
        return group_sizes
        
    except Exception as e:
        logger.error(f"[Ranking Groups] Failed to build groups: {e}")
        return None


def prepare_labels_for_task(y: np.ndarray, spec: TaskSpec, time_vals: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], dict]:
    """
    Prepare labels and metadata for a specific task.
    
    Args:
        y: Raw target values
        spec: Task specification from router
        time_vals: Timestamp values (for ranking groups)
    
    Returns:
        (y_prepared, sample_weights, group_sizes, metadata)
        - y_prepared: Processed labels (encoded if needed)
        - sample_weights: Sample weights (for class imbalance)
        - group_sizes: Group sizes (for ranking)
        - metadata: Dict with label mapping, etc.
    """
    metadata = {
        'task': spec.task,
        'objective': spec.objective,
        'original_dtype': y.dtype
    }
    
    sample_weights = None
    group_sizes = None
    
    # Multiclass: encode labels
    if spec.task == 'multiclass':
        y_prepared, label_map = encode_multiclass_labels(y, spec.n_classes)
        metadata['label_map'] = label_map
        metadata['n_classes'] = len(label_map)
        
        # Compute class weights if requested
        if spec.class_weighting:
            sample_weights = compute_class_weights(y_prepared, spec.class_weighting)
    
    # Binary: ensure 0/1 labels
    elif spec.task == 'binary':
        y_prepared = y.astype(np.int32)
        
        # Compute class weights if requested
        if spec.class_weighting:
            sample_weights = compute_class_weights(y_prepared, spec.class_weighting)
    
    # Ranking: build group sizes
    elif spec.task == 'ranking':
        y_prepared = y.astype(spec.label_type)
        group_sizes = build_ranking_groups(time_vals)
        
        if group_sizes is None:
            logger.warning("[Ranking] No groups available. Falling back to regression.")
            spec.task = 'regression'
            spec.objective = 'regression'
            spec.needs_group = False
    
    # Regression (default)
    else:
        y_prepared = y.astype(spec.label_type)
    
    logger.info(f"[Label Prep] Task={spec.task}, y_shape={y_prepared.shape}, weights={sample_weights is not None}, groups={group_sizes is not None}")
    
    return y_prepared, sample_weights, group_sizes, metadata


def get_objective_for_family(family: str, spec: TaskSpec) -> str:
    """
    Get the correct objective string for a specific model family and task.
    
    Args:
        family: Model family name (e.g., 'LightGBM', 'XGBoost')
        spec: Task specification
    
    Returns:
        Objective string for that family
    """
    # LightGBM objectives
    if family in ['LightGBM', 'QuantileLightGBM']:
        obj_map = {
            'regression': 'regression',
            'binary': 'binary',
            'multiclass': 'multiclass',
            'ranking': 'lambdarank'
        }
        return obj_map.get(spec.task, 'regression')
    
    # XGBoost objectives
    elif family == 'XGBoost':
        obj_map = {
            'regression': 'reg:squarederror',
            'binary': 'binary:logistic',
            'multiclass': 'multi:softprob',
            'ranking': 'rank:pairwise'
        }
        return obj_map.get(spec.task, 'reg:squarederror')
    
    # Other families: use generic
    else:
        return spec.objective


def get_metrics_for_task(spec: TaskSpec) -> List[str]:
    """
    Get evaluation metrics for a task.
    
    Args:
        spec: Task specification
    
    Returns:
        List of metric names
    """
    return spec.metrics


# ============================================================================
# Integration helpers for existing training code
# ============================================================================

def route_target(target: str, time_vals: Optional[np.ndarray] = None) -> dict:
    """
    One-stop routing for a target column.
    
    Returns dict with everything needed for training:
    - spec: TaskSpec
    - prepare_fn: Function to call on (y, time_vals) → (y_prep, weights, groups, meta)
    - objective_fn: Function to call on (family) → objective string
    """
    spec = spec_from_target(target)
    
    return {
        'spec': spec,
        # Avoid boolean ambiguity on numpy arrays: use explicit None check
        'prepare_fn': lambda y, tv=None: prepare_labels_for_task(y, spec, tv if tv is not None else time_vals),
        'objective_fn': lambda family: get_objective_for_family(family, spec),
        'metrics': get_metrics_for_task(spec)
    }

