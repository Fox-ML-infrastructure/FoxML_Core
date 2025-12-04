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
  python scripts/rank_target_predictability.py
  
  # Test on specific symbols first
  python scripts/rank_target_predictability.py --symbols AAPL,MSFT,GOOGL
  
  # Rank specific targets
  python scripts/rank_target_predictability.py --targets peak_60m,valley_60m,swing_high_15m
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
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Import checkpoint utility (after path is set)
from scripts.utils.checkpoint import CheckpointManager

# Import unified task type system
from scripts.utils.task_types import (
    TaskType, TargetConfig, ModelConfig, 
    is_compatible, create_model_configs_from_yaml
)
from scripts.utils.task_metrics import evaluate_by_task, compute_composite_score
from scripts.utils.target_validation import validate_target, check_cv_compatibility

# Suppress expected warnings (harmless)
warnings.filterwarnings('ignore', message='X does not have valid feature names')
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
warnings.filterwarnings('ignore', category=RuntimeWarning, module='sklearn')
warnings.filterwarnings('ignore', message='invalid value encountered in divide')
warnings.filterwarnings('ignore', message='invalid value encountered in true_divide')

# Setup logging with journald support
from scripts.utils.logging_setup import setup_logging
logger = setup_logging(
    script_name="rank_target_predictability",
    level=logging.INFO,
    use_journald=True
)


@dataclass
class TargetPredictabilityScore:
    """Predictability assessment for a single target"""
    target_name: str
    target_column: str
    task_type: TaskType  # REGRESSION, BINARY_CLASSIFICATION, or MULTICLASS_CLASSIFICATION
    mean_score: float  # Mean score (R² for regression, ROC-AUC for binary, accuracy for multiclass)
    std_score: float  # Std of scores
    mean_importance: float  # Mean absolute importance
    consistency: float  # 1 - CV(score) - lower is better
    n_models: int
    model_scores: Dict[str, float]
    composite_score: float = 0.0
    leakage_flag: str = "OK"  # "OK", "SUSPICIOUS", "HIGH_SCORE", "INCONSISTENT"
    suspicious_features: Dict[str, List[Tuple[str, float]]] = None  # {model: [(feature, imp), ...]}
    fold_timestamps: List[Dict[str, Any]] = None  # List of {fold_idx, train_start, train_end, test_start, test_end} per fold
    
    # Backward compatibility: mean_r2 property
    @property
    def mean_r2(self) -> float:
        """Backward compatibility: returns mean_score"""
        return self.mean_score
    
    @property
    def std_r2(self) -> float:
        """Backward compatibility: returns std_score"""
        return self.std_score
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        result = {
            'target_name': self.target_name,
            'target_column': self.target_column,
            'task_type': self.task_type.name if hasattr(self, 'task_type') else 'REGRESSION',
            'mean_score': float(self.mean_score),
            'std_score': float(self.std_score),
            'mean_r2': float(self.mean_score),  # Backward compatibility
            'std_r2': float(self.std_score),  # Backward compatibility
            'mean_importance': float(self.mean_importance),
            'consistency': float(self.consistency),
            'n_models': int(self.n_models),
            'model_scores': {k: float(v) for k, v in self.model_scores.items()},
            'composite_score': float(self.composite_score),
            'leakage_flag': self.leakage_flag
        }
        if self.fold_timestamps is not None:
            result['fold_timestamps'] = self.fold_timestamps
        return result
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'TargetPredictabilityScore':
        """Create from dictionary"""
        # Handle suspicious_features if present
        suspicious = d.pop('suspicious_features', None)
        
        # Backward compatibility: handle old format with mean_r2/std_r2
        if 'mean_r2' in d and 'mean_score' not in d:
            d['mean_score'] = d['mean_r2']
        if 'std_r2' in d and 'std_score' not in d:
            d['std_score'] = d['std_r2']
        
        # Handle task_type (may be missing in old checkpoints)
        if 'task_type' not in d:
            # Try to infer from target name or default to REGRESSION
            d['task_type'] = TaskType.REGRESSION
        
        # Convert task_type string to enum if needed
        if isinstance(d.get('task_type'), str):
            d['task_type'] = TaskType[d['task_type']]
        
        obj = cls(**d)
        if suspicious:
            obj.suspicious_features = suspicious
        return obj


def load_target_configs() -> Dict[str, Dict]:
    """Load target configurations"""
    config_path = _REPO_ROOT / "CONFIG" / "target_configs.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config['targets']


def discover_all_targets(symbol: str, data_dir: Path) -> Dict[str, TargetConfig]:
    """
    Auto-discover all valid targets from data (non-degenerate).
    
    Discovers:
    - y_* targets (barrier, swing, MFE/MDD targets)
    - fwd_ret_* targets (forward return targets)
    
    Returns dict of {target_name: TargetConfig} for all valid targets found.
    """
    import pandas as pd
    import numpy as np
    
    # Load sample data to discover targets
    symbol_dir = data_dir / f"symbol={symbol}"
    parquet_file = symbol_dir / f"{symbol}.parquet"
    
    if not parquet_file.exists():
        raise FileNotFoundError(f"Cannot discover targets: {parquet_file} not found")
    
    df = pd.read_parquet(parquet_file)
    
    # Find all target columns
    # 1. y_* targets (barrier, swing, MFE/MDD)
    y_targets = [c for c in df.columns if c.startswith('y_')]
    # 2. fwd_ret_* targets (forward returns)
    fwd_ret_targets = [c for c in df.columns if c.startswith('fwd_ret_')]
    
    all_targets = y_targets + fwd_ret_targets
    
    # Filter out degenerate targets (single class or zero variance)
    valid_targets = {}
    degenerate_count = 0
    first_touch_count = 0
    sparse_count = 0
    
    for target_col in all_targets:
        y = df[target_col].dropna()
        
        # FIX 2: Check for sparsity - need minimum samples for statistical validity
        # If target has < 100 non-NaN values, it's too sparse for reliable CV
        if len(y) < 100:
            sparse_count += 1
            continue
        
        # Also check as percentage of total dataframe (catch extremely sparse targets)
        if len(y) < len(df) * 0.01:  # Less than 1% of data
            sparse_count += 1
            continue
        
        unique_vals = y.unique()
        n_unique = len(unique_vals)
        
        # Skip degenerate targets (single class)
        if n_unique == 1:
            degenerate_count += 1
            continue
        
        # FIX 3: Check for extreme class imbalance (e.g., single positive sample)
        # For classification targets (n_unique <= 10), ensure minimum class count
        if n_unique <= 10:  # Heuristic for classification
            class_counts = y.value_counts()
            min_class_count = class_counts.min()
            if min_class_count < 2:  # Need at least 2 samples per class for CV
                degenerate_count += 1
                continue
        
        # For regression targets (fwd_ret_*), also check variance
        if target_col.startswith('fwd_ret_'):
            std = y.std()
            if std < 1e-6:  # Zero or near-zero variance
                degenerate_count += 1
                continue
        
        # Skip first_touch targets (they're leaked - correlated with hit_direction features)
        if 'first_touch' in target_col:
            first_touch_count += 1
            continue
        
        # Infer task type from data
        task_type = TaskType.from_target_column(target_col, y)
        
        # FIX 1: Use full target_col as key to avoid collisions
        # (e.g., y_squeeze and y_will_squeeze both become "squeeze" with old logic)
        # Store display_name for UI/logging purposes
        if target_col.startswith('y_'):
            display_name = target_col.replace('y_will_', '').replace('y_', '')
        else:  # fwd_ret_*
            display_name = target_col  # Keep full name for forward returns
        
        # Extract horizon if possible
        horizon = None
        import re
        horizon_match = re.search(r'(\d+)[mhd]', target_col)
        if horizon_match:
            horizon_val = int(horizon_match.group(1))
            if 'd' in target_col:
                horizon = horizon_val * 1440  # days to minutes
            elif 'h' in target_col:
                horizon = horizon_val * 60  # hours to minutes
            else:
                horizon = horizon_val  # minutes
        
        # Create TargetConfig object
        valid_targets[target_col] = TargetConfig(
            name=target_col,
            target_column=target_col,
            task_type=task_type,
            horizon=horizon,
            display_name=display_name,
            description=f"Auto-discovered target: {target_col}",
            use_case=f"{task_type.name} target",
            top_n=60,
            method='mean',
            enabled=True
        )
    
    logger.info(f"  Discovered {len(valid_targets)} valid targets")
    logger.info(f"    - y_* targets: {len([t for t in valid_targets.values() if t.target_column.startswith('y_')])}")
    logger.info(f"    - fwd_ret_* targets: {len([t for t in valid_targets.values() if t.target_column.startswith('fwd_ret_')])}")
    logger.info(f"  Skipped {degenerate_count} degenerate targets (single class/zero variance/extreme imbalance)")
    if sparse_count > 0:
        logger.info(f"  Skipped {sparse_count} sparse targets (< 100 samples or < 1% of data)")
    if first_touch_count > 0:
        logger.info(f"  Skipped {first_touch_count} first_touch targets (leaked)")
    
    return valid_targets


def load_sample_data(
    symbol: str,
    data_dir: Path,
    max_samples: int = 10000
) -> pd.DataFrame:
    """Load sample data for a symbol"""
    symbol_dir = data_dir / f"symbol={symbol}"
    parquet_file = symbol_dir / f"{symbol}.parquet"
    
    if not parquet_file.exists():
        logger.warning(f"  Symbol {symbol} not found in dataset, skipping")
        raise FileNotFoundError(f"Data not found: {parquet_file}")
    
    df = pd.read_parquet(parquet_file)
    
    # Sample if too large
    if len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=42)
    
    return df


def prepare_features_and_target(
    df: pd.DataFrame,
    target_column: str,
    target_config: Optional[TargetConfig] = None
) -> Tuple[np.ndarray, np.ndarray, List[str], TaskType]:
    """
    Prepare features and target for modeling
    
    Returns:
        X: Feature matrix
        y: Target array
        feature_names: List of feature names
        task_type: TaskType enum
    """
    # Check target exists
    if target_column not in df.columns:
        raise ValueError(f"Target '{target_column}' not in data")
    
    # Drop NaN in target
    df = df.dropna(subset=[target_column])
    
    if df.empty:
        raise ValueError("No valid data after dropping NaN in target")
    
    # Get target config or infer task type
    if target_config is None:
        y_sample = df[target_column].dropna()
        task_type = TaskType.from_target_column(target_column, y_sample.to_numpy())
    else:
        task_type = target_config.task_type
    
    # LEAKAGE PREVENTION: Filter out leaking features (target-aware)
    from scripts.utils.leakage_filtering import filter_features_for_target
    
    all_columns = df.columns.tolist()
    # Use target-aware filtering to exclude temporal overlap features
    # Enable verbose logging to see what's being filtered
    safe_columns = filter_features_for_target(all_columns, target_column, verbose=True)
    
    # Log filtering summary
    excluded_count = len(all_columns) - len(safe_columns) - 1  # -1 for target itself
    logger.info(f"  Filtered out {excluded_count} potentially leaking features (kept {len(safe_columns)} safe features)")
    
    # Keep only safe features + target
    safe_columns_with_target = safe_columns + [target_column]
    df = df[safe_columns_with_target]
    
    # Prepare features (exclude target explicitly)
    X = df.drop(columns=[target_column], errors='ignore')
    
    # Drop object dtypes
    object_cols = X.select_dtypes(include=['object']).columns.tolist()
    if object_cols:
        X = X.drop(columns=object_cols)
    
    y = df[target_column]
    feature_names = X.columns.tolist()
    
    return X.to_numpy(), y.to_numpy(), feature_names, task_type


def load_multi_model_config(config_path: Path = None) -> Dict[str, Any]:
    """Load multi-model feature selection configuration"""
    if config_path is None:
        config_path = _REPO_ROOT / "CONFIG" / "multi_model_feature_selection.yaml"
    
    if not config_path.exists():
        logger.debug(f"Multi-model config not found: {config_path}, using defaults")
        return None
    
    try:
        import yaml
        with open(config_path) as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded multi-model config from {config_path}")
        return config
    except Exception as e:
        logger.warning(f"Failed to load multi-model config: {e}")
        return None


def get_model_config(model_name: str, multi_model_config: Dict[str, Any]) -> Dict[str, Any]:
    """Get config for a specific model from multi_model_config"""
    if multi_model_config is None:
        return {}
    
    model_families = multi_model_config.get('model_families', {})
    if not model_families or not isinstance(model_families, dict):
        return {}
    
    model_spec = model_families.get(model_name)
    if model_spec is None or not isinstance(model_spec, dict):
        logger.warning(f"Model '{model_name}' not found in config or is None/empty. Using empty config.")
        return {}
    
    config = model_spec.get('config', {})
    if config is None:
        logger.warning(f"Config for '{model_name}' is None. Using empty config.")
        return {}
    
    if not isinstance(config, dict):
        logger.warning(f"Config for '{model_name}' is not a dict (got {type(config)}). Using empty config.")
        return {}
    
    return config


def _detect_leaking_features(
    feature_names: List[str],
    importances: np.ndarray,
    model_name: str,
    threshold: float = 0.50,
    force_report: bool = False  # If True, always report top features even if no leak detected
) -> List[Tuple[str, float]]:
    """
    Detect features with suspiciously high importance (likely data leakage).
    
    Returns list of (feature_name, importance) tuples for suspicious features.
    """
    if len(feature_names) != len(importances):
        logger.warning(f"  Feature count mismatch: {len(feature_names)} names vs {len(importances)} importances")
        return []
    
    # Normalize importances to sum to 1
    total_importance = np.sum(importances)
    if total_importance == 0:
        logger.warning(f"  Total importance is zero for {model_name}")
        return []
    
    normalized_importance = importances / total_importance
    
    # Create sorted list of (feature, importance) pairs
    feature_imp_pairs = list(zip(feature_names, normalized_importance))
    feature_imp_pairs.sort(key=lambda x: x[1], reverse=True)
    
    # Find features with importance above threshold
    suspicious = []
    for feat, imp in feature_imp_pairs:
        if imp >= threshold:
            suspicious.append((feat, float(imp)))
            logger.error(
                f"  🚨 LEAK DETECTED: {feat} has {imp:.1%} importance in {model_name} "
                f"(threshold: {threshold:.1%}) - likely data leakage!"
            )
    
    # Also check if top feature dominates (even if below threshold)
    if len(normalized_importance) > 0:
        top_feature, top_importance = feature_imp_pairs[0]
        
        # If top feature has >30% and is much larger than second, flag it
        if top_importance >= 0.30 and len(feature_imp_pairs) > 1:
            second_importance = feature_imp_pairs[1][1]
            if top_importance > second_importance * 3:  # 3x larger than second
                if (top_feature, top_importance) not in suspicious:
                    suspicious.append((top_feature, float(top_importance)))
                    logger.warning(
                        f"  ⚠️  SUSPICIOUS: {top_feature} has {top_importance:.1%} importance "
                        f"(3x larger than next feature: {feature_imp_pairs[1][0]}={second_importance:.1%}) - investigate for leakage"
                    )
    
    # CRITICAL: If we suspect a leak (force_report=True) or found suspicious features,
    # always print top 10 features to help identify the leak
    if force_report or suspicious:
        logger.error(f"  📊 TOP 10 FEATURES BY IMPORTANCE ({model_name}):")
        logger.error(f"  {'='*70}")
        for i, (feat, imp) in enumerate(feature_imp_pairs[:10], 1):
            marker = "🚨" if (feat, imp) in suspicious else "  "
            logger.error(f"    {marker} {i:2d}. {feat:50s} = {imp:.2%}")
        
        # Also check cumulative importance of top features
        top_5_importance = sum(imp for _, imp in feature_imp_pairs[:5])
        top_10_importance = sum(imp for _, imp in feature_imp_pairs[:10])
        logger.error(f"  📈 Cumulative: Top 5 = {top_5_importance:.1%}, Top 10 = {top_10_importance:.1%}")
        if top_5_importance > 0.80:
            logger.error(f"  ⚠️  WARNING: Top 5 features account for {top_5_importance:.1%} of importance - likely leakage!")
        
        # Provide actionable next steps
        logger.error(f"  💡 NEXT STEPS:")
        logger.error(f"     1. Review the top features above - they likely contain future information")
        logger.error(f"     2. Check feature importance CSV for full analysis")
        logger.error(f"     3. Add leaking features to CONFIG/excluded_features.yaml")
        logger.error(f"     4. Restart Python process and re-run to apply new filters")
    
    return suspicious


def train_and_evaluate_models(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    task_type: TaskType,
    model_families: List[str] = None,
    multi_model_config: Dict[str, Any] = None,
    target_column: str = None,  # For leak reporting and horizon extraction
    data_interval_minutes: int = 5,  # Data bar interval (default: 5-minute bars)
    time_vals: Optional[np.ndarray] = None  # Timestamps for each sample (for fold timestamp tracking)
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, float], float, Dict[str, List[Tuple[str, float]]], Dict[str, Dict[str, float]], List[Dict[str, Any]]]:
    """
    Train multiple models and return task-aware metrics + importance magnitude
    
    Args:
        X: Feature matrix
        y: Target array
        feature_names: List of feature names
        task_type: TaskType enum (REGRESSION, BINARY_CLASSIFICATION, MULTICLASS_CLASSIFICATION)
        model_families: List of model family names to use
        multi_model_config: Multi-model config dict
    
    Returns:
        model_metrics: Dict of {model_name: {metric_name: value}} per model (full metrics)
        model_scores: Dict of {model_name: primary_score} per model (backward compat)
        mean_importance: Mean absolute feature importance
        all_suspicious_features: Dict of {model_name: [(feature, importance), ...]}
        all_feature_importances: Dict of {model_name: {feature: importance}}
        fold_timestamps: List of {fold_idx, train_start, train_end, test_start, test_end} per fold
    
    Always returns 6 values, even on error (returns empty dicts, 0.0, and empty list)
    """
    # Initialize return values (ensures we always return 6 values)
    model_metrics = {}
    model_scores = {}
    importance_magnitudes = []
    all_suspicious_features = {}  # {model_name: [(feature, importance), ...]}
    all_feature_importances = {}  # {model_name: {feature: importance}} for detailed export
    fold_timestamps = []  # List of fold timestamp info
    
    try:
        from sklearn.model_selection import cross_val_score
        from sklearn.preprocessing import StandardScaler
        import lightgbm as lgb
        from scripts.utils.purged_time_series_split import PurgedTimeSeriesSplit
        from scripts.utils.leakage_filtering import _extract_horizon, _load_leakage_config
        from scripts.utils.feature_pruning import quick_importance_prune
    except Exception as e:
        logger.warning(f"Failed to import required libraries: {e}")
        return {}, {}, 0.0, {}, {}, []
    
    # Helper function for CV with early stopping (for gradient boosting models)
    def cross_val_score_with_early_stopping(model, X, y, cv, scoring, early_stopping_rounds=50, n_jobs=1):
        """
        Cross-validation with early stopping support for gradient boosting models.
        
        cross_val_score doesn't support early stopping callbacks, so we need a manual loop.
        This prevents overfitting by stopping when validation performance plateaus.
        """
        scores = []
        for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y)):
            try:
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                # Clone model for this fold
                from sklearn.base import clone
                fold_model = clone(model)
                
                # Train with early stopping
                # Check if model supports early stopping (LightGBM/XGBoost)
                supports_eval_set = hasattr(fold_model, 'fit') and 'eval_set' in fold_model.fit.__code__.co_varnames
                supports_early_stopping = hasattr(fold_model, 'fit') and 'early_stopping_rounds' in fold_model.fit.__code__.co_varnames
                
                if supports_eval_set:
                    # LightGBM style: uses callbacks
                    # Check by module name for reliability (str(type()) can be fragile)
                    model_module = type(fold_model).__module__
                    if 'lightgbm' in model_module.lower():
                        fold_model.fit(
                            X_train, y_train,
                            eval_set=[(X_val, y_val)],
                            callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)]
                        )
                    # XGBoost style: early_stopping_rounds is set in constructor (XGBoost 2.0+)
                    # Don't pass it to fit() - it's already in the model
                    elif 'xgboost' in model_module.lower():
                        import xgboost as xgb
                        # XGBoost 2.0+ has early_stopping_rounds in constructor, not fit()
                        # Check if model already has it set, otherwise use eval_set only
                        fold_model.fit(
                            X_train, y_train,
                            eval_set=[(X_val, y_val)],
                            verbose=False
                        )
                    else:
                        # Fallback: try eval_set without callbacks
                        fold_model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
                else:
                    # Standard fit for models without early stopping
                    fold_model.fit(X_train, y_train)
                
                # Evaluate on validation set
                if scoring == 'r2':
                    from sklearn.metrics import r2_score
                    y_pred = fold_model.predict(X_val)
                    score = r2_score(y_val, y_pred)
                elif scoring == 'roc_auc':
                    from sklearn.metrics import roc_auc_score
                    y_proba = fold_model.predict_proba(X_val)[:, 1] if hasattr(fold_model, 'predict_proba') else fold_model.predict(X_val)
                    if len(np.unique(y_val)) == 2:
                        score = roc_auc_score(y_val, y_proba)
                    else:
                        score = np.nan
                elif scoring == 'accuracy':
                    from sklearn.metrics import accuracy_score
                    y_pred = fold_model.predict(X_val)
                    score = accuracy_score(y_val, y_pred)
                else:
                    # Fallback to default scorer
                    from sklearn.metrics import get_scorer
                    scorer = get_scorer(scoring)
                    score = scorer(fold_model, X_val, y_val)
                
                scores.append(score)
            except Exception as e:
                logger.debug(f"  Fold {fold_idx + 1} failed: {e}")
                scores.append(np.nan)
        
        return np.array(scores)
    
    # ARCHITECTURAL IMPROVEMENT: Pre-prune low-importance features before expensive training
    # This reduces noise and prevents "Curse of Dimensionality" issues
    # Drop features with < 0.01% cumulative importance using a fast LightGBM model
    original_feature_count = len(feature_names)
    if original_feature_count > 100:  # Only prune if we have many features
        logger.info(f"  Pre-pruning features: {original_feature_count} features")
        
        # Determine task type string for pruning
        if task_type == TaskType.REGRESSION:
            task_str = 'regression'
        elif task_type == TaskType.BINARY_CLASSIFICATION:
            task_str = 'classification'
        else:
            task_str = 'classification'
        
        try:
            X_pruned, feature_names_pruned, pruning_stats = quick_importance_prune(
                X, y, feature_names,
                cumulative_threshold=0.0001,  # 0.01% cumulative importance
                min_features=50,  # Always keep at least 50
                task_type=task_str,
                n_estimators=50,  # Fast model for quick pruning
                random_state=42
            )
            
            if pruning_stats.get('dropped_count', 0) > 0:
                logger.info(f"  ✅ Pruned: {original_feature_count} → {len(feature_names_pruned)} features "
                          f"(dropped {pruning_stats['dropped_count']} low-importance features)")
                X = X_pruned
                feature_names = feature_names_pruned
            else:
                logger.info(f"  No features pruned (all above threshold)")
        except Exception as e:
            logger.warning(f"  Feature pruning failed: {e}, using all features")
            # Continue with original features
    
    # Get CV config
    cv_config = multi_model_config.get('cross_validation', {}) if multi_model_config else {}
    cv_folds = cv_config.get('cv_folds', 3)
    cv_n_jobs = cv_config.get('n_jobs', 1)
    
    # CRITICAL: Use PurgedTimeSeriesSplit to prevent temporal leakage
    # Standard K-Fold shuffles data randomly, which destroys time patterns
    # TimeSeriesSplit respects time order but doesn't prevent overlap leakage
    # PurgedTimeSeriesSplit enforces a gap between train/test = target horizon
    
    # Calculate purge_overlap based on target horizon
    # Extract target horizon (in minutes) from target column name
    leakage_config = _load_leakage_config()
    target_horizon_minutes = _extract_horizon(target_column, leakage_config) if target_column else None
    
    # Auto-detect data interval from timestamps if available, otherwise use parameter
    # CRITICAL: Using wrong interval causes data leakage (e.g., 1m data with 5m assumption leaks 4 minutes)
    if time_vals is not None and len(time_vals) > 1:
        try:
            # Convert to pandas Timestamp if needed
            # Handle both numeric (nanoseconds) and datetime timestamps
            if isinstance(time_vals[0], (int, float, np.integer, np.floating)):
                # Handle numeric timestamps (nanoseconds or Unix timestamp)
                time_series = pd.to_datetime(time_vals, unit='ns')
            elif isinstance(time_vals, np.ndarray) and time_vals.dtype.kind == 'M':
                # Already datetime64 array
                time_series = pd.Series(time_vals)
            else:
                time_series = pd.Series(time_vals)
            
            # Ensure time_series is datetime type for proper diff calculation
            if not pd.api.types.is_datetime64_any_dtype(time_series):
                time_series = pd.to_datetime(time_series)
            
            # CRITICAL: For panel data, multiple rows share the same timestamp
            # Calculate diff on UNIQUE timestamps, not all rows (otherwise median will be 0)
            unique_times = time_series.unique()
            unique_times_sorted = pd.Series(unique_times).sort_values()
            
            # Calculate median time difference between unique timestamps
            time_diffs = unique_times_sorted.diff().dropna()
            # time_diffs should be TimedeltaIndex when time_series is datetime
            if isinstance(time_diffs, pd.TimedeltaIndex) and len(time_diffs) > 0:
                median_diff_minutes = time_diffs.median().total_seconds() / 60.0
            elif len(time_diffs) > 0:
                # Fallback: if diff didn't produce Timedeltas, calculate manually
                median_diff = time_diffs.median()
                if isinstance(median_diff, pd.Timedelta):
                    median_diff_minutes = median_diff.total_seconds() / 60.0
                elif isinstance(median_diff, (int, float, np.integer, np.floating)):
                    # Assume nanoseconds if numeric
                    median_diff_minutes = float(median_diff) / 1e9 / 60.0
                else:
                    raise ValueError(f"Unexpected median_diff type: {type(median_diff)}")
            else:
                # No differences (all timestamps identical) - use default
                median_diff_minutes = data_interval_minutes
                logger.warning(f"  All timestamps identical, cannot detect interval, using parameter: {data_interval_minutes}m")
            
            # Round to common intervals (1m, 5m, 15m, 30m, 60m)
            common_intervals = [1, 5, 15, 30, 60]
            detected_interval = min(common_intervals, key=lambda x: abs(x - median_diff_minutes))
            
            # Only use auto-detection if it's close to a common interval (within 20% tolerance)
            if abs(median_diff_minutes - detected_interval) / detected_interval < 0.2:
                data_interval_minutes = detected_interval
                logger.info(f"  Auto-detected data interval: {median_diff_minutes:.1f}m → {data_interval_minutes}m (from timestamps)")
            else:
                # Fall back to parameter if detection is unclear
                logger.warning(f"  Auto-detection unclear ({median_diff_minutes:.1f}m), using parameter: {data_interval_minutes}m")
        except Exception as e:
            logger.warning(f"  Failed to auto-detect interval from timestamps: {e}, using parameter: {data_interval_minutes}m")
    else:
        # Use parameter value (default: 5)
        logger.info(f"  Using data interval from parameter: {data_interval_minutes}m")
    
    # Convert horizon from minutes to number of bars
    purge_buffer_bars = 5  # Safety buffer (5 bars = 25 minutes)
    
    # ARCHITECTURAL FIX: Use time-based purging instead of row-count based
    # This prevents leakage when data interval doesn't match assumptions
    if target_horizon_minutes is not None:
        # Create Timedelta for purge window (target horizon + safety buffer)
        purge_buffer_minutes = purge_buffer_bars * data_interval_minutes
        purge_time = pd.Timedelta(minutes=target_horizon_minutes + purge_buffer_minutes)
        logger.info(f"  Target horizon: {target_horizon_minutes}m, purge_time: {purge_time}")
    else:
        # Fallback: use a conservative default (60m + 25m buffer = 85m)
        purge_time = pd.Timedelta(minutes=85)
        logger.warning(f"  Could not extract target horizon from '{target_column}', using default purge_time={purge_time}")
    
    # Create purged time series split with time-based purging
    # CRITICAL: Validate time_vals alignment and sorting before using time-based purging
    if time_vals is not None and len(time_vals) == len(X):
        # Ensure time_vals is sorted (required for binary search in PurgedTimeSeriesSplit)
        time_series = pd.Series(time_vals) if not isinstance(time_vals, pd.Series) else time_vals
        if not time_series.is_monotonic_increasing:
            logger.warning("⚠️  time_vals is not sorted! Sorting X, y, and time_vals together")
            sort_idx = np.argsort(time_vals)
            X = X[sort_idx]
            y = y[sort_idx]
            time_vals = time_series.iloc[sort_idx].values if isinstance(time_series, pd.Series) else time_series[sort_idx]
            logger.info(f"  Sorted data by timestamp (preserving alignment)")
        
        tscv = PurgedTimeSeriesSplit(
            n_splits=cv_folds, 
            purge_overlap_time=purge_time,
            time_column_values=time_vals
        )
        logger.info(f"  Using PurgedTimeSeriesSplit (TIME-BASED): {cv_folds} folds, purge_time={purge_time}")
    else:
        # CRITICAL: Row-count based purging is INVALID for panel data (multiple symbols per timestamp)
        # With 50 symbols, 1 bar = 50 rows. Using row counts causes catastrophic leakage.
        # We MUST fail loudly rather than silently producing invalid results.
        raise ValueError(
            f"CRITICAL: time_vals is required for panel data (cross-sectional). "
            f"Row-count based purging is INVALID when multiple symbols share the same timestamp. "
            f"With {len(X)} samples, row-count purging would cause 100% data leakage. "
            f"Please ensure cross_sectional_data.py returns time_vals."
        )
    
    # Capture fold timestamps if time_vals is provided
    if time_vals is not None and len(time_vals) == len(X):
        try:
            for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(X, y)):
                train_times = time_vals[train_idx]
                test_times = time_vals[test_idx]
                fold_timestamps.append({
                    'fold_idx': fold_idx + 1,
                    'train_start': pd.Timestamp(train_times.min()) if len(train_times) > 0 else None,
                    'train_end': pd.Timestamp(train_times.max()) if len(train_times) > 0 else None,
                    'test_start': pd.Timestamp(test_times.min()) if len(test_times) > 0 else None,
                    'test_end': pd.Timestamp(test_times.max()) if len(test_times) > 0 else None,
                    'train_samples': len(train_idx),
                    'test_samples': len(test_idx)
                })
            logger.info(f"  Captured timestamps for {len(fold_timestamps)} folds")
        except Exception as e:
            logger.warning(f"  Failed to capture fold timestamps: {e}")
            fold_timestamps = []
    
    if model_families is None:
        # Load from multi-model config if available
        if multi_model_config:
            model_families_dict = multi_model_config.get('model_families', {})
            if model_families_dict is None or not isinstance(model_families_dict, dict):
                logger.warning("model_families in config is None or not a dict. Using defaults.")
                model_families = ['lightgbm', 'random_forest', 'neural_network']
            else:
                model_families = [
                    name for name, config in model_families_dict.items()
                    if config is not None and isinstance(config, dict) and config.get('enabled', False)
                ]
            logger.debug(f"Using {len(model_families)} models from config: {', '.join(model_families)}")
        else:
            model_families = ['lightgbm', 'random_forest', 'neural_network']
    
    # Create ModelConfig objects for this task type
    model_configs = create_model_configs_from_yaml(multi_model_config, task_type) if multi_model_config else []
    # Filter to only enabled model families
    model_configs = [mc for mc in model_configs if mc.name in model_families]
    
    # Note: model_metrics, model_scores, importance_magnitudes already initialized at function start
    
    # Determine task characteristics
    unique_vals = np.unique(y[~np.isnan(y)])
    is_binary = task_type == TaskType.BINARY_CLASSIFICATION
    is_multiclass = task_type == TaskType.MULTICLASS_CLASSIFICATION
    is_classification = is_binary or is_multiclass
    
    # Select scoring metric based on task type
    if task_type == TaskType.REGRESSION:
        scoring = 'r2'
    elif task_type == TaskType.BINARY_CLASSIFICATION:
        scoring = 'roc_auc'
    else:  # MULTICLASS_CLASSIFICATION
        scoring = 'accuracy'
    
    # Helper function to detect perfect correlation (data leakage)
    def _check_for_perfect_correlation(y_true: np.ndarray, y_pred: np.ndarray, model_name: str) -> bool:
        """
        Check if predictions are perfectly correlated with targets (indicates leakage).
        Returns True if perfect correlation detected.
        """
        try:
            # For classification, check if predictions match exactly
            if task_type in {TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION}:
                if len(y_true) == len(y_pred):
                    accuracy = np.mean(y_true == y_pred)
                    if accuracy >= 0.999:  # 99.9% accuracy = suspicious
                        metric_name = "training accuracy"  # Clarify this is training, not CV
                        logger.warning(f"  LEAKAGE WARNING: {model_name} has {accuracy:.1%} {metric_name} - likely data leakage!")
                        return True
            
            # For regression, check correlation
            elif task_type == TaskType.REGRESSION:
                if len(y_true) == len(y_pred):
                    corr = np.corrcoef(y_true, y_pred)[0, 1]
                    if not np.isnan(corr) and abs(corr) >= 0.999:
                        logger.warning(f"  LEAKAGE WARNING: {model_name} has correlation {corr:.4f} - likely data leakage!")
                        return True
        except Exception:
            pass
        return False
    
    # Helper function to compute and store full task-aware metrics
    def _compute_and_store_metrics(model_name: str, model, X: np.ndarray, y: np.ndarray,
                                   primary_score: float, task_type: TaskType):
        """
        Compute full task-aware metrics and store in both model_metrics and model_scores.
        
        Args:
            model_name: Name of the model
            model: Fitted model
            X: Feature matrix (for predictions)
            y: True target values
            primary_score: Primary score from CV (R², AUC, or accuracy)
            task_type: TaskType enum
        """
        # Defensive check: ensure model_scores and model_metrics are dicts
        nonlocal model_scores, model_metrics
        if model_scores is None or not isinstance(model_scores, dict):
            logger.warning(f"model_scores is None or not a dict in _compute_and_store_metrics, reinitializing")
            model_scores = {}
        if model_metrics is None or not isinstance(model_metrics, dict):
            logger.warning(f"model_metrics is None or not a dict in _compute_and_store_metrics, reinitializing")
            model_metrics = {}
        
        # Store primary score for backward compatibility
        model_scores[model_name] = primary_score
        
        # Compute full task-aware metrics
        try:
            if task_type == TaskType.REGRESSION:
                y_pred = model.predict(X)
                # Check for perfect correlation (leakage)
                if _check_for_perfect_correlation(y, y_pred, model_name):
                    logger.error(f"  CRITICAL: {model_name} shows signs of data leakage! Check feature filtering.")
                full_metrics = evaluate_by_task(task_type, y, y_pred, return_ic=True)
            elif task_type == TaskType.BINARY_CLASSIFICATION:
                if hasattr(model, 'predict_proba'):
                    y_proba = model.predict_proba(X)[:, 1]  # Probability of class 1
                    y_pred = (y_proba >= 0.5).astype(int)
                else:
                    # Fallback for models without predict_proba
                    y_pred = model.predict(X)
                    y_proba = np.clip(y_pred, 0, 1)  # Assume predictions are probabilities
                # Check for perfect correlation (leakage)
                if _check_for_perfect_correlation(y, y_pred, model_name):
                    logger.error(f"  CRITICAL: {model_name} shows signs of data leakage! Check feature filtering.")
                full_metrics = evaluate_by_task(task_type, y, y_proba)
            else:  # MULTICLASS_CLASSIFICATION
                if hasattr(model, 'predict_proba'):
                    y_proba = model.predict_proba(X)
                    y_pred = y_proba.argmax(axis=1)
                else:
                    # Fallback: one-hot encode predictions
                    y_pred = model.predict(X)
                    n_classes = len(np.unique(y[~np.isnan(y)]))
                    y_proba = np.eye(n_classes)[y_pred.astype(int)]
                # Check for perfect correlation (leakage)
                if _check_for_perfect_correlation(y, y_pred, model_name):
                    logger.error(f"  CRITICAL: {model_name} shows signs of data leakage! Check feature filtering.")
                full_metrics = evaluate_by_task(task_type, y, y_proba)
            
            # Store full metrics
            model_metrics[model_name] = full_metrics
        except Exception as e:
            logger.warning(f"Failed to compute full metrics for {model_name}: {e}")
            # Fallback to primary score only
            if task_type == TaskType.REGRESSION:
                model_metrics[model_name] = {'r2': primary_score}
            elif task_type == TaskType.BINARY_CLASSIFICATION:
                model_metrics[model_name] = {'roc_auc': primary_score}
            else:
                model_metrics[model_name] = {'accuracy': primary_score}
    
        # Helper function to update both model_scores and model_metrics
        # NOTE: This is now mainly for backward compat - full metrics computed after training
    def _update_model_score(model_name: str, score: float):
        """Update model_scores (backward compat) - full metrics computed separately"""
        model_scores[model_name] = score
    
        # Check for degenerate target BEFORE training models
    # Check for degenerate target BEFORE training models
    # A target is degenerate if it has < 2 unique values or one class has < 2 samples
    unique_vals = np.unique(y[~np.isnan(y)])
    if len(unique_vals) < 2:
        logger.debug(f"    Skipping: Target has only {len(unique_vals)} unique value(s)")
        return {}, {}, 0.0, {}, {}, []  # model_metrics, model_scores, mean_importance, suspicious_features, feature_importances, fold_timestamps
    
    # For classification, check class balance
    if is_binary or is_multiclass:
        class_counts = np.bincount(y[~np.isnan(y)].astype(int))
        min_class_count = class_counts[class_counts > 0].min()
        if min_class_count < 2:
            logger.debug(f"    Skipping: Smallest class has only {min_class_count} sample(s)")
            return {}, {}, 0.0, {}, {}, []  # model_metrics, model_scores, mean_importance, suspicious_features, feature_importances, fold_timestamps
    
    # LightGBM
    if 'lightgbm' in model_families:
        try:
            # GPU settings (will fallback to CPU if GPU not available)
            gpu_params = {}
            try:
                # Try CUDA first (fastest)
                test_model = lgb.LGBMRegressor(device='cuda', n_estimators=1, verbose=-1)
                test_model.fit(np.random.rand(10, 5), np.random.rand(10))
                gpu_params = {'device': 'cuda', 'gpu_device_id': 0}
                logger.info("  Using GPU (CUDA) for LightGBM")
            except:
                try:
                    # Try OpenCL
                    test_model = lgb.LGBMRegressor(device='gpu', n_estimators=1, verbose=-1)
                    test_model.fit(np.random.rand(10, 5), np.random.rand(10))
                    gpu_params = {'device': 'gpu', 'gpu_platform_id': 0, 'gpu_device_id': 0}
                    logger.info("  Using GPU (OpenCL) for LightGBM")
                except:
                    logger.info("  Using CPU for LightGBM")
            
            # Get config values
            lgb_config = get_model_config('lightgbm', multi_model_config)
            # Defensive check: ensure config is a dict
            if not isinstance(lgb_config, dict):
                lgb_config = {}
            # Remove objective and device from config (we set these explicitly)
            lgb_config_clean = {k: v for k, v in lgb_config.items() if k not in ['device', 'objective', 'metric']}
            
            if is_binary:
                model = lgb.LGBMClassifier(
                    objective='binary',
                    **lgb_config_clean,
                    **gpu_params
                )
            elif is_multiclass:
                n_classes = len(unique_vals)
                model = lgb.LGBMClassifier(
                    objective='multiclass',
                    num_class=n_classes,
                    **lgb_config_clean,
                    **gpu_params
                )
            else:
                model = lgb.LGBMRegressor(
                    objective='regression',
                    **lgb_config_clean,
                    **gpu_params
                )
            
            # CRITICAL FIX: Use manual CV loop with early stopping for gradient boosting
            # Get early stopping rounds from config (default: 50)
            early_stopping_rounds = lgb_config.get('early_stopping_rounds', 50) if isinstance(lgb_config, dict) else 50
            
            logger.info(f"  Using CV with early stopping (rounds={early_stopping_rounds}) for LightGBM")
            scores = cross_val_score_with_early_stopping(
                model, X, y, cv=tscv, scoring=scoring, 
                early_stopping_rounds=early_stopping_rounds, n_jobs=1  # n_jobs=1 for early stopping compatibility
            )
            valid_scores = scores[~np.isnan(scores)]
            primary_score = valid_scores.mean() if len(valid_scores) > 0 else np.nan
            
            # Train once on full data (with early stopping on a validation split) to get importance
            # CRITICAL: Use time-aware split (last 20% as validation) - don't shuffle time series data
            # Guard against empty arrays
            if len(X) < 10:
                logger.warning(f"  ⚠️  Too few samples ({len(X)}) for train/val split, fitting on all data")
                split_idx = len(X)
            else:
                split_idx = int(len(X) * 0.8)
                split_idx = max(1, split_idx)  # Ensure at least 1 sample in validation
            
            if split_idx < len(X):
                X_train_final, X_val_final = X[:split_idx], X[split_idx:]
                y_train_final, y_val_final = y[:split_idx], y[split_idx:]
            else:
                # Fallback: use all data if too small
                X_train_final, X_val_final = X, X
                y_train_final, y_val_final = y, y
            model.fit(
                X_train_final, y_train_final,
                eval_set=[(X_val_final, y_val_final)],
                callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)]
            )
            
            # CRITICAL: Check for suspiciously high scores (likely leakage)
            has_leak = False
            if not np.isnan(primary_score) and primary_score >= 0.99:
                # Use task-appropriate metric name
                if task_type == TaskType.REGRESSION:
                    metric_name = "R²"
                elif task_type == TaskType.BINARY_CLASSIFICATION:
                    metric_name = "ROC-AUC"
                else:
                    metric_name = "Accuracy"
                logger.error(f"  🚨 LEAKAGE ALERT: lightgbm {metric_name}={primary_score:.4f} >= 0.99 - likely data leakage!")
                logger.error(f"    Features: {len(feature_names)} features")
                logger.error(f"    Analyzing feature importances to identify leaks...")
                has_leak = True
            
            # LEAK DETECTION: Analyze feature importance for suspicious patterns
            importances = model.feature_importances_
            suspicious_features = _detect_leaking_features(
                feature_names, importances, model_name='lightgbm',
                threshold=0.50,  # Flag if single feature has >50% importance
                force_report=has_leak  # Always report top features if score indicates leak
            )
            if suspicious_features:
                all_suspicious_features['lightgbm'] = suspicious_features
            
            # Store all feature importances for detailed export
            all_feature_importances['lightgbm'] = {
                feat: float(imp) for feat, imp in zip(feature_names, importances)
            }
            
            # Compute and store full task-aware metrics
            _compute_and_store_metrics('lightgbm', model, X, y, primary_score, task_type)
            # Use percentage of total importance in top 10% features (0-1 scale, interpretable)
            total_importance = np.sum(importances)
            if total_importance > 0:
                top_k = max(1, int(len(importances) * 0.1))  # Top 10% of features
                top_importance_sum = np.sum(np.sort(importances)[-top_k:])
                # Normalize to 0-1: what % of total importance is in top 10%?
                importance_ratio = top_importance_sum / total_importance
            else:
                importance_ratio = 0.0
            importance_magnitudes.append(importance_ratio)
            
        except Exception as e:
            logger.warning(f"LightGBM failed: {e}")
    
    # Random Forest
    if 'random_forest' in model_families:
        try:
            from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
            
            # Get config values
            rf_config = get_model_config('random_forest', multi_model_config)
            
            if is_binary or is_multiclass:
                model = RandomForestClassifier(**rf_config)
            else:
                model = RandomForestRegressor(**rf_config)
            
            scores = cross_val_score(model, X, y, cv=tscv, scoring=scoring, n_jobs=cv_n_jobs, error_score=np.nan)
            valid_scores = scores[~np.isnan(scores)]
            primary_score = valid_scores.mean() if len(valid_scores) > 0 else np.nan
            
            # ⚠️ IMPORTANCE BIAS WARNING: This fits on the full dataset (in-sample)
            # Deep trees/GBMs can memorize noise, making feature importance biased.
            # TODO: Future enhancement - use permutation importance calculated on CV test folds
            # For now, this is acceptable but be aware that importance may be inflated
            model.fit(X, y)
            
            # Check for suspicious scores
            has_leak = not np.isnan(primary_score) and primary_score >= 0.99
            
            # LEAK DETECTION: Analyze feature importance
            importances = model.feature_importances_
            suspicious_features = _detect_leaking_features(
                feature_names, importances, model_name='random_forest', 
                threshold=0.50, force_report=has_leak
            )
            if suspicious_features:
                all_suspicious_features['random_forest'] = suspicious_features
            
            # Store all feature importances for detailed export
            all_feature_importances['random_forest'] = {
                feat: float(imp) for feat, imp in zip(feature_names, importances)
            }
            
            # Compute and store full task-aware metrics
            _compute_and_store_metrics('random_forest', model, X, y, primary_score, task_type)
            # Use percentage of total importance in top 10% features (0-1 scale, interpretable)
            total_importance = np.sum(importances)
            if total_importance > 0:
                top_k = max(1, int(len(importances) * 0.1))  # Top 10% of features
                top_importance_sum = np.sum(np.sort(importances)[-top_k:])
                # Normalize to 0-1: what % of total importance is in top 10%?
                importance_ratio = top_importance_sum / total_importance
            else:
                importance_ratio = 0.0
            importance_magnitudes.append(importance_ratio)
            
        except Exception as e:
            logger.warning(f"RandomForest failed: {e}")
    
    # Neural Network
    if 'neural_network' in model_families:
        try:
            from sklearn.neural_network import MLPRegressor, MLPClassifier
            from sklearn.impute import SimpleImputer
            from sklearn.exceptions import ConvergenceWarning
            from sklearn.compose import TransformedTargetRegressor
            from sklearn.pipeline import Pipeline
            
            # Get config values
            nn_config = get_model_config('neural_network', multi_model_config)
            
            if is_binary or is_multiclass:
                # For classification: Pipeline handles imputation and scaling within CV folds
                steps = [
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler()),
                    ('model', MLPClassifier(**nn_config))
                ]
                pipeline = Pipeline(steps)
                model = pipeline
                y_for_training = y
            else:
                # For regression: Pipeline for features + TransformedTargetRegressor for target
                # This ensures no data leakage - all scaling/imputation happens within CV folds
                feature_steps = [
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler()),
                    ('model', MLPRegressor(**nn_config))
                ]
                feature_pipeline = Pipeline(feature_steps)
                model = TransformedTargetRegressor(
                    regressor=feature_pipeline,
                    transformer=StandardScaler()
                )
                y_for_training = y
            
            # Neural networks need special handling for degenerate targets
            # Suppress convergence warnings (they're noisy and we handle failures gracefully)
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=ConvergenceWarning)
                try:
                    # Pipeline ensures imputation/scaling happens within each CV fold (no leakage)
                    scores = cross_val_score(model, X, y_for_training, cv=tscv, scoring=scoring, n_jobs=cv_n_jobs, error_score=np.nan)
                    valid_scores = scores[~np.isnan(scores)]
                    primary_score = valid_scores.mean() if len(valid_scores) > 0 else np.nan
                except ValueError as e:
                    if "least populated class" in str(e) or "too few" in str(e):
                        logger.debug(f"    Neural Network: Target too imbalanced for CV")
                        primary_score = np.nan
                        model_metrics['neural_network'] = {'roc_auc': np.nan} if task_type == TaskType.BINARY_CLASSIFICATION else {'r2': np.nan} if task_type == TaskType.REGRESSION else {'accuracy': np.nan}
                        model_scores['neural_network'] = np.nan
                    else:
                        raise
            
            # Fit on raw data (Pipeline handles preprocessing internally)
            # ⚠️ IMPORTANCE BIAS WARNING: This fits on the full dataset (in-sample)
            # See comment above for details
            if not np.isnan(primary_score):
                model.fit(X, y_for_training)
                
                # Compute and store full task-aware metrics (Pipeline handles preprocessing)
                _compute_and_store_metrics('neural_network', model, X, y_for_training, primary_score, task_type)
            
            baseline_score = model.score(X, y_for_training)
            
            perm_scores = []
            for i in range(min(10, X.shape[1])):  # Sample 10 features
                X_perm = X.copy()
                np.random.shuffle(X_perm[:, i])
                perm_score = model.score(X_perm, y_for_training)
                perm_scores.append(abs(baseline_score - perm_score))
            
            importance_magnitudes.append(np.mean(perm_scores))
            
        except Exception as e:
            logger.warning(f"NeuralNetwork failed: {e}")
    
    # XGBoost
    if 'xgboost' in model_families:
        try:
            import xgboost as xgb
            
            # Get config values
            xgb_config = get_model_config('xgboost', multi_model_config)
            # Defensive check: ensure config is a dict
            if not isinstance(xgb_config, dict):
                xgb_config = {}
            # Remove task-specific parameters (we set these explicitly based on task type)
            # CRITICAL: Extract early_stopping_rounds from config - it goes in constructor for XGBoost 2.0+
            early_stopping_rounds = xgb_config.get('early_stopping_rounds', None)
            xgb_config_clean = {k: v for k, v in xgb_config.items() 
                              if k not in ['objective', 'eval_metric', 'early_stopping_rounds']}
            
            # XGBoost 2.0+ requires early_stopping_rounds in constructor, not fit()
            if early_stopping_rounds is not None:
                xgb_config_clean['early_stopping_rounds'] = early_stopping_rounds
            
            if is_binary:
                model = xgb.XGBClassifier(
                    objective='binary:logistic',
                    **xgb_config_clean
                )
            elif is_multiclass:
                n_classes = len(unique_vals)
                model = xgb.XGBClassifier(
                    objective='multi:softprob',
                    num_class=n_classes,
                    **xgb_config_clean
                )
            else:
                model = xgb.XGBRegressor(
                    objective='reg:squarederror',
                    **xgb_config_clean
                )
            
            # CRITICAL FIX: Use manual CV loop with early stopping for gradient boosting
            # Get early stopping rounds from config (default: 50)
            # NOTE: For XGBoost 2.0+, early_stopping_rounds is set in constructor above, not passed to fit()
            early_stopping_rounds = xgb_config.get('early_stopping_rounds', 50) if isinstance(xgb_config, dict) else 50
            
            logger.info(f"  Using CV with early stopping (rounds={early_stopping_rounds}) for XGBoost")
            try:
                # XGBoost uses same early stopping interface as LightGBM
                scores = cross_val_score_with_early_stopping(
                    model, X, y, cv=tscv, scoring=scoring,
                    early_stopping_rounds=early_stopping_rounds, n_jobs=1
                )
                valid_scores = scores[~np.isnan(scores)]
                primary_score = valid_scores.mean() if len(valid_scores) > 0 else np.nan
            except ValueError as e:
                if "Invalid classes" in str(e) or "Expected" in str(e):
                    logger.debug(f"    XGBoost: Target degenerate in some CV folds")
                    primary_score = np.nan
                    model_metrics['xgboost'] = {'roc_auc': np.nan} if task_type == TaskType.BINARY_CLASSIFICATION else {'r2': np.nan} if task_type == TaskType.REGRESSION else {'accuracy': np.nan}
                    model_scores['xgboost'] = np.nan
                else:
                    raise
            
            # Train once on full data (with early stopping) to get importance and full metrics
            # CRITICAL: Use time-aware split (last 20% as validation) - don't shuffle time series data
            if not np.isnan(primary_score):
                # Guard against empty arrays
                if len(X) < 10:
                    logger.warning(f"  ⚠️  Too few samples ({len(X)}) for train/val split, fitting on all data")
                    split_idx = len(X)
                else:
                    split_idx = int(len(X) * 0.8)
                    split_idx = max(1, split_idx)  # Ensure at least 1 sample in validation
                
                if split_idx < len(X):
                    X_train_final, X_val_final = X[:split_idx], X[split_idx:]
                    y_train_final, y_val_final = y[:split_idx], y[split_idx:]
                else:
                    # Fallback: use all data if too small
                    X_train_final, X_val_final = X, X
                    y_train_final, y_val_final = y, y
                # XGBoost 2.0+: early_stopping_rounds is set in constructor, not passed to fit()
                # The model already has it configured from the constructor above
                model.fit(
                    X_train_final, y_train_final,
                    eval_set=[(X_val_final, y_val_final)],
                    verbose=False
                )
                
                # Check for suspicious scores
                has_leak = primary_score >= 0.99
                
                # Compute and store full task-aware metrics
                _compute_and_store_metrics('xgboost', model, X, y, primary_score, task_type)
                
                # LEAK DETECTION: Analyze feature importance
                if hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                    suspicious_features = _detect_leaking_features(
                        feature_names, importances, model_name='xgboost', 
                        threshold=0.50, force_report=has_leak
                    )
                    if suspicious_features:
                        all_suspicious_features['xgboost'] = suspicious_features
                    
                    # Store all feature importances for detailed export
                    all_feature_importances['xgboost'] = {
                        feat: float(imp) for feat, imp in zip(feature_names, importances)
                    }
            if hasattr(model, 'feature_importances_'):
                # Use percentage of total importance in top 10% features (0-1 scale, interpretable)
                importances = model.feature_importances_
                total_importance = np.sum(importances)
                if total_importance > 0:
                    top_k = max(1, int(len(importances) * 0.1))  # Top 10% of features
                    top_importance_sum = np.sum(np.sort(importances)[-top_k:])
                    # Normalize to 0-1: what % of total importance is in top 10%?
                    importance_ratio = top_importance_sum / total_importance
                else:
                    importance_ratio = 0.0
                importance_magnitudes.append(importance_ratio)
        except Exception as e:
            logger.warning(f"XGBoost failed: {e}")
    
    # CatBoost
    if 'catboost' in model_families:
        try:
            import catboost as cb
            
            # Get config values
            cb_config = get_model_config('catboost', multi_model_config)
            # Defensive check: ensure config is a dict
            if not isinstance(cb_config, dict):
                cb_config = {}
            # Remove task-specific parameters (we set these explicitly based on task type)
            cb_config_clean = {k: v for k, v in cb_config.items() if k not in ['loss_function']}
            
            if is_binary:
                model = cb.CatBoostClassifier(**cb_config_clean)
            elif is_multiclass:
                n_classes = len(unique_vals)
                model = cb.CatBoostClassifier(**cb_config_clean)
            else:
                model = cb.CatBoostRegressor(**cb_config_clean)
            
            try:
                scores = cross_val_score(model, X, y, cv=tscv, scoring=scoring, n_jobs=cv_n_jobs, error_score=np.nan)
                valid_scores = scores[~np.isnan(scores)]
                primary_score = valid_scores.mean() if len(valid_scores) > 0 else np.nan
            except (ValueError, TypeError) as e:
                if "Invalid classes" in str(e) or "Expected" in str(e):
                    logger.debug(f"    CatBoost: Target degenerate in some CV folds")
                    primary_score = np.nan
                    model_metrics['catboost'] = {'roc_auc': np.nan} if task_type == TaskType.BINARY_CLASSIFICATION else {'r2': np.nan} if task_type == TaskType.REGRESSION else {'accuracy': np.nan}
                    model_scores['catboost'] = np.nan
                else:
                    raise
            
            if not np.isnan(primary_score):
                model.fit(X, y)
                
                # Compute and store full task-aware metrics
                _compute_and_store_metrics('catboost', model, X, y, primary_score, task_type)
            importance = model.get_feature_importance()
            if len(importance) > 0:
                total_importance = np.sum(importance)
                if total_importance > 0:
                    top_k = max(1, int(len(importance) * 0.1))
                    top_importance_sum = np.sum(np.sort(importance)[-top_k:])
                    importance_ratio = top_importance_sum / total_importance
                else:
                    importance_ratio = 0.0
            else:
                importance_ratio = 0.0
            importance_magnitudes.append(importance_ratio)
        except ImportError:
            logger.warning("CatBoost not available (pip install catboost)")
        except Exception as e:
            logger.warning(f"CatBoost failed: {e}")
    
    # Lasso
    if 'lasso' in model_families:
        try:
            from sklearn.linear_model import Lasso
            from sklearn.impute import SimpleImputer
            from sklearn.pipeline import Pipeline
            
            # Get config values
            lasso_config = get_model_config('lasso', multi_model_config)
            
            # CRITICAL FIX: Pipeline ensures imputation + scaling happens within each CV fold (no leakage)
            # Lasso requires scaling for proper convergence (features must be on similar scales)
            steps = [
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler()),  # Required for Lasso convergence
                ('model', Lasso(**lasso_config))
            ]
            pipeline = Pipeline(steps)
            
            scores = cross_val_score(pipeline, X, y, cv=tscv, scoring=scoring, n_jobs=cv_n_jobs, error_score=np.nan)
            valid_scores = scores[~np.isnan(scores)]
            primary_score = valid_scores.mean() if len(valid_scores) > 0 else np.nan
            
            # ⚠️ IMPORTANCE BIAS WARNING: This fits on the full dataset (in-sample)
            # See comment above for details
            pipeline.fit(X, y)
            
            # Compute and store full task-aware metrics (Lasso is regression-only)
            if not np.isnan(primary_score) and task_type == TaskType.REGRESSION:
                _compute_and_store_metrics('lasso', pipeline, X, y, primary_score, task_type)
            
            # Extract coefficients from the fitted model
            model = pipeline.named_steps['model']
            importance = np.abs(model.coef_)
            if len(importance) > 0:
                total_importance = np.sum(importance)
                if total_importance > 0:
                    top_k = max(1, int(len(importance) * 0.1))
                    top_importance_sum = np.sum(np.sort(importance)[-top_k:])
                    importance_ratio = top_importance_sum / total_importance
                else:
                    importance_ratio = 0.0
            else:
                importance_ratio = 0.0
            importance_magnitudes.append(importance_ratio)
        except Exception as e:
            logger.warning(f"Lasso failed: {e}")
    
    # Mutual Information
    if 'mutual_information' in model_families:
        try:
            from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
            from sklearn.impute import SimpleImputer
            
            # Mutual information doesn't handle NaN - need to impute
            imputer = SimpleImputer(strategy='median')
            X_imputed = imputer.fit_transform(X)
            
            # Get config values
            mi_config = get_model_config('mutual_information', multi_model_config)
            
            # Suppress warnings for zero-variance features
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                if is_binary or is_multiclass:
                    importance = mutual_info_classif(X_imputed, y, 
                                                    random_state=mi_config['random_state'],
                                                    discrete_features=mi_config['discrete_features'])
                else:
                    importance = mutual_info_regression(X_imputed, y, 
                                                       random_state=mi_config['random_state'],
                                                       discrete_features=mi_config['discrete_features'])
            
            # Handle NaN/inf
            importance = np.nan_to_num(importance, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Mutual information doesn't have R², so we use a proxy based on max MI
            # Normalize to 0-1 scale for importance
            if len(importance) > 0 and np.max(importance) > 0:
                importance_normalized = importance / np.max(importance)
                total_importance = np.sum(importance_normalized)
                if total_importance > 0:
                    top_k = max(1, int(len(importance_normalized) * 0.1))
                    top_importance_sum = np.sum(np.sort(importance_normalized)[-top_k:])
                    importance_ratio = top_importance_sum / total_importance
                else:
                    importance_ratio = 0.0
            else:
                importance_ratio = 0.0
            
            # For mutual information, we can't compute R² directly
            # Use a proxy: higher MI concentration = better predictability
            # Scale to approximate R² range (0-0.3 for good targets)
            model_scores['mutual_information'] = min(0.3, importance_ratio * 0.3)
            importance_magnitudes.append(importance_ratio)
        except Exception as e:
            logger.warning(f"Mutual Information failed: {e}")
    
    # Univariate Selection
    if 'univariate_selection' in model_families:
        try:
            from sklearn.feature_selection import f_regression, f_classif
            from sklearn.impute import SimpleImputer
            
            # F-tests don't handle NaN - need to impute
            imputer = SimpleImputer(strategy='median')
            X_imputed = imputer.fit_transform(X)
            
            # Suppress division by zero warnings (expected for zero-variance features)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                if is_binary or is_multiclass:
                    scores, pvalues = f_classif(X_imputed, y)
                else:
                    scores, pvalues = f_regression(X_imputed, y)
            
            # Handle NaN/inf in scores (from zero-variance features)
            scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Normalize F-statistics
            if len(scores) > 0 and np.max(scores) > 0:
                importance = scores / np.max(scores)
                total_importance = np.sum(importance)
                if total_importance > 0:
                    top_k = max(1, int(len(importance) * 0.1))
                    top_importance_sum = np.sum(np.sort(importance)[-top_k:])
                    importance_ratio = top_importance_sum / total_importance
                else:
                    importance_ratio = 0.0
            else:
                importance_ratio = 0.0
            
            # F-statistics don't have R², use proxy
            model_scores['univariate_selection'] = min(0.3, importance_ratio * 0.3)
            importance_magnitudes.append(importance_ratio)
        except Exception as e:
            logger.warning(f"Univariate Selection failed: {e}")
    
    # RFE
    if 'rfe' in model_families:
        try:
            from sklearn.feature_selection import RFE
            from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
            from sklearn.impute import SimpleImputer
            
            # RFE uses RandomForest which handles NaN, but let's impute for consistency
            imputer = SimpleImputer(strategy='median')
            X_imputed = imputer.fit_transform(X)
            
            # Get config values
            rfe_config = get_model_config('rfe', multi_model_config)
            n_features_to_select = min(rfe_config['n_features_to_select'], X_imputed.shape[1])
            step = rfe_config['step']
            
            # Use random_forest config for RFE estimator
            rf_config = get_model_config('random_forest', multi_model_config)
            
            if is_binary or is_multiclass:
                estimator = RandomForestClassifier(**rf_config)
            else:
                estimator = RandomForestRegressor(**rf_config)
            
            selector = RFE(estimator, n_features_to_select=n_features_to_select, step=step)
            selector.fit(X_imputed, y)
            
            # Get R² using cross-validation on selected features (proper validation)
            selected_features = selector.support_
            if np.any(selected_features):
                X_selected = X_imputed[:, selected_features]
                # Quick RF for scoring (use smaller config)
                quick_rf_config = get_model_config('random_forest', multi_model_config).copy()
                # Use smaller model for quick scoring
                quick_rf_config['n_estimators'] = 50
                quick_rf_config['max_depth'] = 8
                
                if is_binary or is_multiclass:
                    quick_rf = RandomForestClassifier(**quick_rf_config)
                else:
                    quick_rf = RandomForestRegressor(**quick_rf_config)
                
                # Use cross-validation for proper validation (not training score)
                scores = cross_val_score(quick_rf, X_selected, y, cv=tscv, scoring=scoring, n_jobs=cv_n_jobs, error_score=np.nan)
                valid_scores = scores[~np.isnan(scores)]
                model_scores['rfe'] = valid_scores.mean() if len(valid_scores) > 0 else np.nan
            else:
                model_scores['rfe'] = np.nan
            
            # Convert ranking to importance
            ranking = selector.ranking_
            importance = 1.0 / (ranking + 1e-6)
            if len(importance) > 0:
                total_importance = np.sum(importance)
                if total_importance > 0:
                    top_k = max(1, int(len(importance) * 0.1))
                    top_importance_sum = np.sum(np.sort(importance)[-top_k:])
                    importance_ratio = top_importance_sum / total_importance
                else:
                    importance_ratio = 0.0
            else:
                importance_ratio = 0.0
            importance_magnitudes.append(importance_ratio)
        except Exception as e:
            logger.warning(f"RFE failed: {e}")
    
    # Boruta
    if 'boruta' in model_families:
        try:
            from boruta import BorutaPy
            from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
            from sklearn.impute import SimpleImputer
            
            # Boruta uses RandomForest which handles NaN, but let's impute for consistency
            imputer = SimpleImputer(strategy='median')
            X_imputed = imputer.fit_transform(X)
            
            # Get config values
            boruta_config = get_model_config('boruta', multi_model_config)
            
            # Use random_forest config for Boruta estimator
            rf_config = get_model_config('random_forest', multi_model_config)
            
            if is_binary or is_multiclass:
                rf = RandomForestClassifier(**rf_config)
            else:
                rf = RandomForestRegressor(**rf_config)
            
            boruta = BorutaPy(rf, n_estimators='auto', verbose=0, 
                            random_state=boruta_config['random_state'],
                            max_iter=boruta_config['max_iter'])
            boruta.fit(X_imputed, y)
            
            # Get R² using cross-validation on selected features (proper validation)
            selected_features = boruta.support_
            if np.any(selected_features):
                X_selected = X_imputed[:, selected_features]
                # Quick RF for scoring (use smaller config)
                quick_rf_config = get_model_config('random_forest', multi_model_config).copy()
                # Use smaller model for quick scoring
                quick_rf_config['n_estimators'] = 50
                quick_rf_config['max_depth'] = 8
                
                if is_binary or is_multiclass:
                    quick_rf = RandomForestClassifier(**quick_rf_config)
                else:
                    quick_rf = RandomForestRegressor(**quick_rf_config)
                
                # Use cross-validation for proper validation (not training score)
                scores = cross_val_score(quick_rf, X_selected, y, cv=tscv, scoring=scoring, n_jobs=cv_n_jobs, error_score=np.nan)
                valid_scores = scores[~np.isnan(scores)]
                model_scores['boruta'] = valid_scores.mean() if len(valid_scores) > 0 else np.nan
            else:
                model_scores['boruta'] = np.nan
            
            # Convert to importance
            ranking = boruta.ranking_
            selected = boruta.support_
            importance = np.where(selected, 1.0, np.where(ranking == 2, 0.5, 0.1))
            if len(importance) > 0:
                total_importance = np.sum(importance)
                if total_importance > 0:
                    top_k = max(1, int(len(importance) * 0.1))
                    top_importance_sum = np.sum(np.sort(importance)[-top_k:])
                    importance_ratio = top_importance_sum / total_importance
                else:
                    importance_ratio = 0.0
            else:
                importance_ratio = 0.0
            importance_magnitudes.append(importance_ratio)
        except ImportError:
            logger.warning("Boruta not available (pip install Boruta)")
        except Exception as e:
            logger.warning(f"Boruta failed: {e}")
    
    # Stability Selection
    if 'stability_selection' in model_families:
        try:
            from sklearn.linear_model import LassoCV, LogisticRegressionCV
            from sklearn.impute import SimpleImputer
            
            # Stability selection uses Lasso/LogisticRegression which don't handle NaN
            imputer = SimpleImputer(strategy='median')
            X_imputed = imputer.fit_transform(X)
            
            # Get config values
            stability_config = get_model_config('stability_selection', multi_model_config)
            n_bootstrap = stability_config['n_bootstrap']
            random_state = stability_config['random_state']
            stability_cv = stability_config['cv']
            stability_n_jobs = stability_config['n_jobs']
            stability_cs = stability_config['Cs']
            stability_scores = np.zeros(X_imputed.shape[1])
            bootstrap_r2_scores = []
            
            # Use lasso config for stability selection models
            lasso_config = get_model_config('lasso', multi_model_config)
            
            for _ in range(n_bootstrap):
                indices = np.random.choice(len(X_imputed), size=len(X_imputed), replace=True)
                X_boot, y_boot = X_imputed[indices], y[indices]
                
                try:
                    # Use TimeSeriesSplit for internal CV (even though bootstrap breaks temporal order,
                    # this maintains consistency with the rest of the codebase)
                    if is_binary or is_multiclass:
                        model = LogisticRegressionCV(Cs=stability_cs, cv=tscv, 
                                                    random_state=random_state,
                                                    max_iter=lasso_config['max_iter'], 
                                                    n_jobs=stability_n_jobs)
                    else:
                        model = LassoCV(cv=tscv, random_state=random_state,
                                      max_iter=lasso_config['max_iter'], 
                                      n_jobs=stability_n_jobs)
                    
                    model.fit(X_boot, y_boot)
                    coef = model.coef_[0] if len(model.coef_.shape) > 1 else model.coef_
                    stability_scores += (np.abs(coef) > 1e-6).astype(int)
                    
                    # Get R² using cross-validation (proper validation, not training score)
                    # Note: Bootstrap samples break temporal order, but we still use TimeSeriesSplit
                    # for consistency (it won't help here, but maintains the pattern)
                    # Use a quick model for CV scoring
                    if is_binary or is_multiclass:
                        cv_model = LogisticRegressionCV(Cs=[1.0], cv=tscv, random_state=random_state, 
                                                        max_iter=lasso_config['max_iter'], n_jobs=1)
                    else:
                        cv_model = LassoCV(cv=tscv, random_state=random_state,
                                          max_iter=lasso_config['max_iter'], n_jobs=1)
                    cv_scores = cross_val_score(cv_model, X_boot, y_boot, cv=tscv, scoring=scoring, n_jobs=1, error_score=np.nan)
                    valid_cv_scores = cv_scores[~np.isnan(cv_scores)]
                    if len(valid_cv_scores) > 0:
                        bootstrap_r2_scores.append(valid_cv_scores.mean())
                except:
                    continue
            
            # Average R² across bootstraps
            if bootstrap_r2_scores:
                model_scores['stability_selection'] = np.mean(bootstrap_r2_scores)
            else:
                model_scores['stability_selection'] = np.nan
            
            # Normalize stability scores to importance
            importance = stability_scores / n_bootstrap
            if len(importance) > 0:
                total_importance = np.sum(importance)
                if total_importance > 0:
                    top_k = max(1, int(len(importance) * 0.1))
                    top_importance_sum = np.sum(np.sort(importance)[-top_k:])
                    importance_ratio = top_importance_sum / total_importance
                else:
                    importance_ratio = 0.0
            else:
                importance_ratio = 0.0
            importance_magnitudes.append(importance_ratio)
        except Exception as e:
            logger.warning(f"Stability Selection failed: {e}")
    
    # Histogram Gradient Boosting
    if 'histogram_gradient_boosting' in model_families:
        try:
            from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
            
            # Get config values
            hgb_config = get_model_config('histogram_gradient_boosting', multi_model_config)
            # Defensive check: ensure config is a dict
            if not isinstance(hgb_config, dict):
                hgb_config = {}
            # Remove task-specific parameters (loss is set automatically by classifier/regressor)
            hgb_config_clean = {k: v for k, v in hgb_config.items() if k != 'loss'}
            
            if is_binary or is_multiclass:
                model = HistGradientBoostingClassifier(**hgb_config_clean)
            else:
                model = HistGradientBoostingRegressor(**hgb_config_clean)
            
            scores = cross_val_score(model, X, y, cv=tscv, scoring=scoring, n_jobs=cv_n_jobs, error_score=np.nan)
            valid_scores = scores[~np.isnan(scores)]
            primary_score = valid_scores.mean() if len(valid_scores) > 0 else np.nan
            
            # Train once to get importance and full metrics
            model.fit(X, y)
            
            # Compute and store full task-aware metrics
            if not np.isnan(primary_score):
                _compute_and_store_metrics('histogram_gradient_boosting', model, X, y, primary_score, task_type)
            if hasattr(model, 'feature_importances_'):
                # Use percentage of total importance in top 10% features (0-1 scale, interpretable)
                importances = model.feature_importances_
                total_importance = np.sum(importances)
                if total_importance > 0:
                    top_k = max(1, int(len(importances) * 0.1))  # Top 10% of features
                    top_importance_sum = np.sum(np.sort(importances)[-top_k:])
                    # Normalize to 0-1: what % of total importance is in top 10%?
                    importance_ratio = top_importance_sum / total_importance
                else:
                    importance_ratio = 0.0
                importance_magnitudes.append(importance_ratio)
        except Exception as e:
            logger.warning(f"Histogram Gradient Boosting failed: {e}")
    
    mean_importance = np.mean(importance_magnitudes) if importance_magnitudes else 0.0
    
    # model_scores already contains primary scores (backward compatible)
    # model_metrics contains full metrics dict
    # all_suspicious_features contains leak detection results (aggregated across all models)
    # all_feature_importances contains detailed per-feature importances for export
    return model_metrics, model_scores, mean_importance, all_suspicious_features, all_feature_importances, fold_timestamps


def _save_feature_importances(
    target_column: str,
    symbol: str,
    feature_importances: Dict[str, Dict[str, float]],
    output_dir: Path = None
) -> None:
    """
    Save detailed per-model, per-feature importance scores to CSV files.
    
    Creates structure:
    {output_dir}/feature_importances/
      {target_name}/
        {symbol}/
          lightgbm_importances.csv
          xgboost_importances.csv
          random_forest_importances.csv
          ...
    
    Args:
        target_column: Name of the target being evaluated
        symbol: Symbol being evaluated
        feature_importances: Dict of {model_name: {feature: importance}}
        output_dir: Base output directory (defaults to results/)
    """
    if output_dir is None:
        output_dir = _REPO_ROOT / "results"
    
    # Create directory structure
    target_name_clean = target_column.replace('/', '_').replace('\\', '_')
    importances_dir = output_dir / "feature_importances" / target_name_clean / symbol
    importances_dir.mkdir(parents=True, exist_ok=True)
    
    # Save per-model CSV files
    for model_name, importances in feature_importances.items():
        if not importances:
            continue
        
        # Create DataFrame sorted by importance
        df = pd.DataFrame([
            {'feature': feat, 'importance': imp}
            for feat, imp in importances.items()
        ])
        df = df.sort_values('importance', ascending=False)
        
        # Normalize to percentages
        total = df['importance'].sum()
        if total > 0:
            df['importance_pct'] = (df['importance'] / total * 100).round(2)
            df['cumulative_pct'] = df['importance_pct'].cumsum().round(2)
        else:
            df['importance_pct'] = 0.0
            df['cumulative_pct'] = 0.0
        
        # Reorder columns
        df = df[['feature', 'importance', 'importance_pct', 'cumulative_pct']]
        
        # Save to CSV
        csv_file = importances_dir / f"{model_name}_importances.csv"
        df.to_csv(csv_file, index=False)
    
    logger.info(f"  💾 Saved feature importances to: {importances_dir}")


def _log_suspicious_features(
    target_column: str,
    symbol: str,
    suspicious_features: Dict[str, List[Tuple[str, float]]]
) -> None:
    """
    Log suspicious features to a file for later analysis.
    
    Args:
        target_column: Name of the target being evaluated
        symbol: Symbol being evaluated
        suspicious_features: Dict of {model_name: [(feature, importance), ...]}
    """
    leak_report_file = _REPO_ROOT / "results" / "leak_detection_report.txt"
    leak_report_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(leak_report_file, 'a') as f:
        f.write(f"\n{'='*80}\n")
        f.write(f"Target: {target_column} | Symbol: {symbol}\n")
        f.write(f"{'='*80}\n")
        
        for model_name, features in suspicious_features.items():
            if features:
                f.write(f"\n{model_name.upper()} - Suspicious Features:\n")
                f.write(f"{'-'*80}\n")
                for feat, imp in sorted(features, key=lambda x: x[1], reverse=True):
                    f.write(f"  {feat:50s} | Importance: {imp:.1%}\n")
                f.write("\n")
    
    logger.info(f"  Leak detection report saved to: {leak_report_file}")


def detect_leakage(
    mean_score: float,
    composite_score: float,
    mean_importance: float,
    target_name: str = "",
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
    
    # Determine threshold based on task type and target name
    if task_type == TaskType.REGRESSION:
        is_forward_return = target_name.startswith('fwd_ret_')
        if is_forward_return:
            # For forward returns: R² > 0.50 is suspicious
            high_threshold = 0.50
            very_high_threshold = 0.60
            metric_name = "R²"
        else:
            # For barrier targets: R² > 0.70 is suspicious
            high_threshold = 0.70
            very_high_threshold = 0.80
            metric_name = "R²"
    elif task_type == TaskType.BINARY_CLASSIFICATION:
        # ROC-AUC > 0.95 is suspicious (near-perfect classification)
        high_threshold = 0.90
        very_high_threshold = 0.95
        metric_name = "ROC-AUC"
    else:  # MULTICLASS_CLASSIFICATION
        # Accuracy > 0.95 is suspicious
        high_threshold = 0.90
        very_high_threshold = 0.95
        metric_name = "Accuracy"
    
    # Check 1: Suspiciously high mean score
    if mean_score > very_high_threshold:
        flags.append("HIGH_SCORE")
        logger.warning(
            f"LEAKAGE WARNING: {metric_name}={mean_score:.3f} > {very_high_threshold:.2f} "
            f"(extremely high - likely leakage)"
        )
    elif mean_score > high_threshold:
        flags.append("HIGH_SCORE")
        logger.warning(
            f"LEAKAGE WARNING: {metric_name}={mean_score:.3f} > {high_threshold:.2f} "
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
    # If composite is very high (> 0.5) but score is low (< 0.2 for regression, < 0.6 for classification), something's wrong
    score_low_threshold = 0.2 if task_type == TaskType.REGRESSION else 0.6
    if composite_score > 0.5 and mean_score < score_low_threshold:
        flags.append("INCONSISTENT")
        logger.warning(
            f"LEAKAGE WARNING: Composite={composite_score:.3f} but {metric_name}={mean_score:.3f} "
            f"(inconsistent - possible leakage)"
        )
    
    # Check 4: Very high importance with low score (might indicate leaked features)
    score_very_low_threshold = 0.1 if task_type == TaskType.REGRESSION else 0.5
    if mean_importance > 0.7 and mean_score < score_very_low_threshold:
        flags.append("INCONSISTENT")
        logger.warning(
            f"LEAKAGE WARNING: Importance={mean_importance:.2f} but {metric_name}={mean_score:.3f} "
            f"(high importance with low {metric_name} - check for leaked features)"
        )
    
    if len(flags) > 1:
        return "SUSPICIOUS"
    elif len(flags) == 1:
        return flags[0]
    else:
        return "OK"


def calculate_composite_score(
    mean_score: float,
    std_score: float,
    mean_importance: float,
    n_models: int,
    task_type: TaskType = TaskType.REGRESSION
) -> float:
    """
    Calculate composite predictability score
    
    Components:
    - Mean score: Higher is better (R² for regression, ROC-AUC/Accuracy for classification)
    - Consistency: Lower std is better
    - Importance magnitude: Higher is better
    - Model agreement: More models = more confidence
    """
    
    # Normalize components based on task type
    if task_type == TaskType.REGRESSION:
        # R² can be negative, so normalize to 0-1 range
        score_component = max(0, mean_score)  # Clamp negative R² to 0
        consistency_component = 1.0 / (1.0 + std_score)
        
        # R²-weighted importance
        if mean_score > 0:
            importance_component = mean_importance * (1.0 + mean_score)
        else:
            penalty = abs(mean_score) * 0.67
            importance_component = mean_importance * max(0.5, 1.0 - penalty)
    else:
        # Classification: ROC-AUC and Accuracy are already 0-1
        score_component = mean_score  # Already 0-1
        consistency_component = 1.0 / (1.0 + std_score)
        
        # Score-weighted importance (similar logic but for 0-1 scores)
        importance_component = mean_importance * (1.0 + mean_score)
    
    # Weighted average
    composite = (
        0.50 * score_component +        # 50% weight on score
        0.25 * consistency_component + # 25% on consistency
        0.25 * importance_component    # 25% on score-weighted importance
    )
    
    # Bonus for more models (up to 10% boost)
    model_bonus = min(0.1, n_models * 0.02)
    composite = composite * (1.0 + model_bonus)
    
    return composite


def evaluate_target_predictability(
    target_name: str,
    target_config: Dict[str, Any] | TargetConfig,
    symbols: List[str],
    data_dir: Path,
    model_families: List[str],
    multi_model_config: Dict[str, Any] = None,
    output_dir: Path = None,
    min_cs: int = 10,
    max_cs_samples: Optional[int] = None,
    max_rows_per_symbol: int = 50000
) -> TargetPredictabilityScore:
    """Evaluate predictability of a single target across symbols"""
    
    # Convert dict config to TargetConfig if needed
    if isinstance(target_config, dict):
        target_column = target_config['target_column']
        display_name = target_config.get('display_name', target_name)
        # Infer task type from column name (will be refined with actual data)
        task_type = TaskType.from_target_column(target_column)
        target_config_obj = TargetConfig(
            name=target_name,
            target_column=target_column,
            task_type=task_type,
            display_name=display_name,
            **{k: v for k, v in target_config.items() 
               if k not in ['target_column', 'display_name']}
        )
    else:
        target_config_obj = target_config
        target_column = target_config_obj.target_column
        display_name = target_config_obj.display_name or target_name
    logger.info(f"\n{'='*60}")
    logger.info(f"Evaluating: {display_name} ({target_column})")
    logger.info(f"{'='*60}")
    
    # Load all symbols at once (cross-sectional data loading)
    from scripts.utils.cross_sectional_data import load_mtf_data_for_ranking, prepare_cross_sectional_data_for_ranking
    from scripts.utils.leakage_filtering import filter_features_for_target
    
    logger.info(f"Loading data for {len(symbols)} symbols (max {max_rows_per_symbol} rows per symbol)...")
    mtf_data = load_mtf_data_for_ranking(data_dir, symbols, max_rows_per_symbol=max_rows_per_symbol)
    
    if not mtf_data:
        logger.error(f"No data loaded for any symbols")
        return TargetPredictabilityScore(
            target_name=target_name,
            target_column=target_column,
            task_type=TaskType.REGRESSION,
            mean_score=-999.0,
            std_score=1.0,
            mean_importance=0.0,
            consistency=0.0,
            n_models=0,
            model_scores={}
        )
    
    # Apply leakage filtering to feature list BEFORE preparing data
    # Get all columns from first symbol to determine available features
    sample_df = next(iter(mtf_data.values()))
    all_columns = sample_df.columns.tolist()
    safe_columns = filter_features_for_target(all_columns, target_column, verbose=True)
    
    excluded_count = len(all_columns) - len(safe_columns) - 1  # -1 for target itself
    logger.info(f"Filtered out {excluded_count} potentially leaking features (kept {len(safe_columns)} safe features)")
    
    # Prepare cross-sectional data (matches training pipeline)
    X, y, feature_names, symbols_array, time_vals = prepare_cross_sectional_data_for_ranking(
        mtf_data, target_column, min_cs=min_cs, max_cs_samples=max_cs_samples, feature_names=safe_columns
    )
    
    if X is None or y is None:
        logger.error(f"Failed to prepare cross-sectional data for {target_column}")
        return TargetPredictabilityScore(
            target_name=target_name,
            target_column=target_column,
            task_type=TaskType.REGRESSION,
            mean_score=-999.0,
            std_score=1.0,
            mean_importance=0.0,
            consistency=0.0,
            n_models=0,
            model_scores={}
        )
    
    logger.info(f"Cross-sectional data: {len(X)} samples, {X.shape[1]} features")
    logger.info(f"Symbols: {len(set(symbols_array))} unique symbols")
    
    # Infer task type from data
    y_sample = pd.Series(y).dropna()
    task_type = TaskType.from_target_column(target_column, y_sample.to_numpy())
    
    # Validate target
    is_valid, error_msg = validate_target(y, task_type=task_type)
    if not is_valid:
        logger.warning(f"Skipping: {error_msg}")
        return TargetPredictabilityScore(
            target_name=target_name,
            target_column=target_column,
            task_type=task_type,
            mean_score=-999.0,
            std_score=1.0,
            mean_importance=0.0,
            consistency=0.0,
            n_models=0,
            model_scores={}
        )
    
    # Check if target is degenerate
    unique_vals = np.unique(y[~np.isnan(y)])
    if len(unique_vals) < 2:
        logger.warning(f"Skipping: Target has only {len(unique_vals)} unique value(s)")
        return TargetPredictabilityScore(
            target_name=target_name,
            target_column=target_column,
            task_type=task_type,
            mean_score=-999.0,
            std_score=1.0,
            mean_importance=0.0,
            consistency=0.0,
            n_models=0,
            model_scores={}
        )
    
    # For classification, check if classes are too imbalanced for CV
    if len(unique_vals) <= 10:  # Likely classification
        class_counts = np.bincount(y[~np.isnan(y)].astype(int))
        min_class_count = class_counts[class_counts > 0].min()
        if min_class_count < 2:
            logger.warning(f"Skipping: Smallest class has only {min_class_count} sample(s) (too few for CV)")
            return TargetPredictabilityScore(
                target_name=target_name,
                target_column=target_column,
                task_type=task_type,
                mean_score=-999.0,
                std_score=1.0,
                mean_importance=0.0,
                consistency=0.0,
                n_models=0,
                model_scores={}
            )
    
    # Train and evaluate on cross-sectional data (single evaluation, not per-symbol)
    all_model_scores = []
    all_importances = []
    all_suspicious_features = {}
    fold_timestamps = None  # Initialize fold_timestamps for later use
    
    try:
        # Auto-detect data interval from timestamps if available
        detected_interval = 5  # Default fallback
        if time_vals is not None and len(time_vals) > 1:
            try:
                if isinstance(time_vals[0], (int, float)):
                    time_series = pd.to_datetime(time_vals, unit='ns')
                else:
                    time_series = pd.Series(time_vals)
                time_diffs = time_series.diff().dropna()
                median_diff_minutes = time_diffs.median().total_seconds() / 60.0
                common_intervals = [1, 5, 15, 30, 60]
                detected_interval = min(common_intervals, key=lambda x: abs(x - median_diff_minutes))
                if abs(median_diff_minutes - detected_interval) / detected_interval >= 0.2:
                    detected_interval = 5  # Fallback if unclear
            except Exception:
                pass  # Use default
        
        result = train_and_evaluate_models(
            X, y, feature_names, task_type, model_families, multi_model_config,
            target_column=target_column,
            data_interval_minutes=detected_interval,  # Auto-detected or default
            time_vals=time_vals  # Pass timestamps for fold tracking
        )
        
        if result is None or len(result) != 6:
            logger.warning(f"train_and_evaluate_models returned unexpected value: {result}")
            return TargetPredictabilityScore(
                target_name=target_name,
                target_column=target_column,
                task_type=task_type,
                mean_score=-999.0,
                std_score=1.0,
                mean_importance=0.0,
                consistency=0.0,
                n_models=0,
                model_scores={}
            )
        
        model_metrics, primary_scores, importance, suspicious_features, feature_importances, fold_timestamps = result
        
        # Save aggregated feature importances (cross-sectional, not per-symbol)
        if feature_importances and output_dir:
            _save_feature_importances(target_column, "CROSS_SECTIONAL", feature_importances, output_dir)
        
        # Store suspicious features
        if suspicious_features:
            all_suspicious_features = suspicious_features
            _log_suspicious_features(target_column, "CROSS_SECTIONAL", suspicious_features)
        
        # Ensure primary_scores is a dict
        if primary_scores is None:
            logger.warning(f"primary_scores is None, skipping")
            return TargetPredictabilityScore(
                target_name=target_name,
                target_column=target_column,
                task_type=task_type,
                mean_score=-999.0,
                std_score=1.0,
                mean_importance=0.0,
                consistency=0.0,
                n_models=0,
                model_scores={}
            )
        if not isinstance(primary_scores, dict):
            logger.warning(f"primary_scores is not a dict (got {type(primary_scores)}), skipping")
            return TargetPredictabilityScore(
                target_name=target_name,
                target_column=target_column,
                task_type=task_type,
                mean_score=-999.0,
                std_score=1.0,
                mean_importance=0.0,
                consistency=0.0,
                n_models=0,
                model_scores={}
            )
        
        all_model_scores.append(primary_scores)
        all_importances.append(importance)
        
        scores_str = ", ".join([f"{k}={v:.3f}" for k, v in primary_scores.items()])
        logger.info(f"Scores: {scores_str}, importance={importance:.2f}")
        
    except Exception as e:
        import traceback
        error_msg = str(e)
        tb_str = traceback.format_exc()
        logger.warning(f"Failed: {error_msg}")
        logger.error(f"Full traceback:\n{tb_str}")
        return TargetPredictabilityScore(
            target_name=target_name,
            target_column=target_column,
            task_type=task_type,
            mean_score=-999.0,
            std_score=1.0,
            mean_importance=0.0,
            consistency=0.0,
            n_models=0,
            model_scores={}
        )
    
    if not all_model_scores:
        logger.warning(f"No successful evaluations for {target_name} (skipping)")
        return TargetPredictabilityScore(
            target_name=target_name,
            target_column=target_column,
            task_type=TaskType.REGRESSION,  # Default, will be updated if target succeeds
            mean_score=-999.0,  # Flag for degenerate/failed targets
            std_score=1.0,
            mean_importance=0.0,
            consistency=0.0,
            n_models=0,
            model_scores={}
        )
    
    # Aggregate across models (skip NaN scores)
    # Note: With cross-sectional data, we only have one evaluation, not per-symbol
    all_scores_by_model = defaultdict(list)
    for scores_dict in all_model_scores:
        # Defensive check: skip None or non-dict entries
        if scores_dict is None or not isinstance(scores_dict, dict):
            logger.warning(f"Skipping invalid scores_dict: {type(scores_dict)}")
            continue
        for model_name, score in scores_dict.items():
            if not (np.isnan(score) if isinstance(score, (float, np.floating)) else False):
                all_scores_by_model[model_name].append(score)
    
    # Calculate statistics (only from models that succeeded)
    model_means = {model: np.mean(scores) for model, scores in all_scores_by_model.items() if scores}
    if not model_means:
        logger.warning(f"No successful model evaluations for {target_name}")
        return TargetPredictabilityScore(
            target_name=target_name,
            target_column=target_column,
            task_type=TaskType.REGRESSION,  # Default
            mean_score=-999.0,
            std_score=1.0,
            mean_importance=0.0,
            consistency=0.0,
            n_models=0,
            model_scores={},
            leakage_flag="OK",
            suspicious_features=None
        )
    
    mean_score = np.mean(list(model_means.values()))
    std_score = np.std(list(model_means.values())) if len(model_means) > 1 else 0.0
    mean_importance = np.mean(all_importances)
    consistency = 1.0 - (std_score / (abs(mean_score) + 1e-6))
    
    # Determine task type (already inferred from data above)
    final_task_type = task_type
    
    # Get metric name for logging
    if final_task_type == TaskType.REGRESSION:
        metric_name = "R²"
    elif final_task_type == TaskType.BINARY_CLASSIFICATION:
        metric_name = "ROC-AUC"
    else:  # MULTICLASS_CLASSIFICATION
        metric_name = "Accuracy"
    
    # Composite score (normalize scores appropriately)
    composite = calculate_composite_score(
        mean_score, std_score, mean_importance, len(all_scores_by_model), final_task_type
    )
    
    # Detect potential leakage (use task-appropriate thresholds)
    leakage_flag = detect_leakage(mean_score, composite, mean_importance, 
                                  target_name=target_name, model_scores=model_means, task_type=final_task_type)
    
    result = TargetPredictabilityScore(
        target_name=target_name,
        target_column=target_column,
        task_type=final_task_type,
        mean_score=mean_score,
        std_score=std_score,
        mean_importance=mean_importance,
        consistency=consistency,
        n_models=len(all_scores_by_model),
        model_scores=model_means,
        composite_score=composite,
        leakage_flag=leakage_flag,
        suspicious_features=all_suspicious_features if all_suspicious_features else None,
        fold_timestamps=fold_timestamps
    )
    
    # Log with leakage warning if needed
    leakage_indicator = f" [{leakage_flag}]" if leakage_flag != "OK" else ""
    logger.info(f"Summary: {metric_name}={mean_score:.3f}±{std_score:.3f}, "
               f"importance={mean_importance:.2f}, composite={composite:.3f}{leakage_indicator}")
    
    # Store suspicious features in result for summary report
    result.suspicious_features = all_suspicious_features if all_suspicious_features else None
    
    return result


def save_leak_report_summary(
    output_dir: Path,
    all_leaks: Dict[str, Dict[str, List[Tuple[str, float]]]]
) -> None:
    """
    Save a summary of all detected leaks across all targets.
    
    Args:
        output_dir: Directory to save the report
        all_leaks: Dict of {target_name: {model_name: [(feature, importance), ...]}}
    """
    report_file = output_dir / "leak_detection_summary.txt"
    
    with open(report_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("LEAK DETECTION SUMMARY\n")
        f.write("="*80 + "\n\n")
        f.write("This report lists features with suspiciously high importance (>50%)\n")
        f.write("which may indicate data leakage (future information in features).\n\n")
        
        total_leaks = sum(len(leaks) for leaks in all_leaks.values())
        f.write(f"Total targets with suspicious features: {len(all_leaks)}\n")
        f.write(f"Total suspicious feature detections: {total_leaks}\n\n")
        
        for target_name, model_leaks in sorted(all_leaks.items()):
            f.write(f"\n{'='*80}\n")
            f.write(f"Target: {target_name}\n")
            f.write(f"{'='*80}\n")
            
            for model_name, features in model_leaks.items():
                if features:
                    f.write(f"\n{model_name.upper()}:\n")
                    f.write(f"{'-'*80}\n")
                    for feat, imp in sorted(features, key=lambda x: x[1], reverse=True):
                        f.write(f"  {feat:60s} | {imp:.1%}\n")
        
        f.write(f"\n\n{'='*80}\n")
        f.write("RECOMMENDATIONS:\n")
        f.write(f"{'='*80}\n")
        f.write("1. Review features with >50% importance - they likely contain future information\n")
        f.write("2. Check for:\n")
        f.write("   - Centered moving averages (center=True)\n")
        f.write("   - Backward shifts (.shift(-1) instead of .shift(1))\n")
        f.write("   - High/Low data that matches target definition\n")
        f.write("   - Features computed from the same barrier logic as the target\n")
        f.write("3. Add suspicious features to leakage_filtering.py exclusion list\n")
        f.write("4. Re-run ranking after fixing leaks\n")
    
    logger.info(f"Leak detection summary saved to: {report_file}")


def save_rankings(
    results: List[TargetPredictabilityScore],
    output_dir: Path
):
    """Save target predictability rankings"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Sort by composite score
    results = sorted(results, key=lambda x: x.composite_score, reverse=True)
    
    # Create DataFrame
    df = pd.DataFrame([{
        'rank': i + 1,
        'target_name': r.target_name,
        'target_column': r.target_column,
        'composite_score': r.composite_score,
        'task_type': r.task_type.name,
        'mean_score': r.mean_score,
        'std_score': r.std_score,
        'mean_r2': r.mean_score,  # Backward compatibility
        'std_r2': r.std_score,  # Backward compatibility
        'mean_importance': r.mean_importance,
        'consistency': r.consistency,
        'n_models': r.n_models,
        'leakage_flag': r.leakage_flag,
        **{f'{model}_r2': score for model, score in r.model_scores.items()},
        'recommendation': _get_recommendation(r)
    } for i, r in enumerate(results)])
    
    # Log suspicious targets
    suspicious = df[df['leakage_flag'] != 'OK']
    if len(suspicious) > 0:
        logger.warning(f"\nFOUND {len(suspicious)} SUSPICIOUS TARGETS (possible leakage):")
        for _, row in suspicious.iterrows():
            logger.warning(
                f"  {row['target_name']:25s} | R²={row['mean_r2']:.3f} | "
                f"Composite={row['composite_score']:.3f} | Flag: {row['leakage_flag']}"
            )
        logger.warning("Review these targets - they may have leaked features or be degenerate!")
    
    # Save CSV
    df.to_csv(output_dir / "target_predictability_rankings.csv", index=False)
    logger.info(f"\nSaved rankings to target_predictability_rankings.csv")
    
    # Save YAML with recommendations
    yaml_data = {
        'target_rankings': [
            {
            'rank': i + 1,
            'target': r.target_name,
            'composite_score': float(r.composite_score),
            'task_type': r.task_type.name,
            'mean_score': float(r.mean_score),
            'mean_r2': float(r.mean_score),  # Backward compatibility
            'leakage_flag': r.leakage_flag,
            'recommendation': _get_recommendation(r)
            }
            for i, r in enumerate(results)
        ]
    }
    
    with open(output_dir / "target_predictability_rankings.yaml", 'w') as f:
        yaml.dump(yaml_data, f, default_flow_style=False)
    
    logger.info(f"Saved YAML to target_predictability_rankings.yaml")


def _get_recommendation(score: TargetPredictabilityScore) -> str:
    """Get recommendation based on predictability score"""
    if score.composite_score >= 0.7:
        return "PRIORITIZE - Strong predictive signal"
    elif score.composite_score >= 0.5:
        return "ENABLE - Good predictive signal"
    elif score.composite_score >= 0.3:
        return "TEST - Moderate signal, worth exploring"
    else:
        return "DEPRIORITIZE - Weak signal, low ROI"


def main():
    parser = argparse.ArgumentParser(
        description="Rank target predictability across model families"
    )
    parser.add_argument("--symbols", type=str, default="AAPL,MSFT,GOOGL,TSLA,JPM",
                       help="Symbols to test on (default: 5 representative stocks)")
    parser.add_argument("--data-dir", type=Path,
                       default=_REPO_ROOT / "data/data_labeled/interval=5m")
    parser.add_argument("--output-dir", type=Path,
                       default=_REPO_ROOT / "results/target_rankings")
    parser.add_argument("--targets", type=str,
                       help="Specific targets to evaluate (comma-separated), default: all enabled")
    parser.add_argument("--discover-all", action="store_true",
                       help="Auto-discover and rank ALL targets from data (ignores config)")
    parser.add_argument("--model-families", type=str,
                       default=None,
                       help="Model families to use (default: use all enabled from multi_model_feature_selection.yaml)")
    parser.add_argument("--multi-model-config", type=Path,
                       default=None,
                       help="Path to multi-model config (default: CONFIG/multi_model_feature_selection.yaml)")
    parser.add_argument("--resume", action="store_true",
                       help="Resume from checkpoint if available")
    parser.add_argument("--clear-checkpoint", action="store_true",
                       help="Clear existing checkpoint and start fresh")
    parser.add_argument("--min-cs", type=int, default=10,
                       help="Minimum cross-sectional size per timestamp (default: 10)")
    parser.add_argument("--max-cs-samples", type=int, default=None,
                       help="Maximum samples per timestamp for cross-sectional sampling (default: 1000)")
    parser.add_argument("--max-rows-per-symbol", type=int, default=50000,
                       help="Maximum rows to load per symbol (most recent rows, default: 50000)")
    
    args = parser.parse_args()
    
    # Parse inputs
    symbols = [s.strip() for s in args.symbols.split(',')]
    
    # Load multi-model config
    multi_model_config = None
    if args.multi_model_config:
        multi_model_config = load_multi_model_config(args.multi_model_config)
    else:
        multi_model_config = load_multi_model_config()  # Try default path
    
    # Determine model families
    if args.model_families:
        model_families = [m.strip() for m in args.model_families.split(',')]
    elif multi_model_config:
        # Use enabled models from config
        model_families_dict = multi_model_config.get('model_families', {})
        if model_families_dict is None or not isinstance(model_families_dict, dict):
            logger.warning("model_families in config is None or not a dict. Using defaults.")
            model_families = ['lightgbm', 'random_forest', 'neural_network']
        else:
            model_families = [
                name for name, config in model_families_dict.items()
                if config is not None and isinstance(config, dict) and config.get('enabled', False)
            ]
        logger.info(f"Using {len(model_families)} model families from config: {', '.join(model_families)}")
    else:
        # Default fallback
        model_families = ['lightgbm', 'random_forest', 'neural_network']
        logger.info(f"Using default model families: {', '.join(model_families)}")
    
    logger.info("="*80)
    logger.info("Target Predictability Ranking")
    logger.info("="*80)
    logger.info(f"Test symbols: {', '.join(symbols)}")
    logger.info(f"Model families: {', '.join(model_families)}")
    
    # Discover or load targets
    if args.discover_all:
        logger.info("Auto-discovering ALL targets from data...")
        targets_to_eval = discover_all_targets(symbols[0], args.data_dir)
        logger.info(f"Found {len(targets_to_eval)} valid targets\n")
    else:
        # Load target configs
        target_configs = load_target_configs()
        
        # Filter targets
        if args.targets:
            requested = [t.strip() for t in args.targets.split(',')]
            targets_to_eval = {k: v for k, v in target_configs.items() if k in requested}
        else:
            # Only evaluate enabled targets
            targets_to_eval = {k: v for k, v in target_configs.items() if v.get('enabled', False)}
        
        logger.info(f"Evaluating {len(targets_to_eval)} targets\n")
    
    # Initialize checkpoint manager
    checkpoint_file = args.output_dir / "checkpoint.json"
    checkpoint = CheckpointManager(
        checkpoint_file=checkpoint_file,
        item_key_fn=lambda item: item if isinstance(item, str) else item[0]  # target_name
    )
    
    # Clear checkpoint if requested
    if args.clear_checkpoint:
        checkpoint.clear()
        logger.info("Cleared checkpoint - starting fresh")
    
    # Load completed targets
    completed = checkpoint.load_completed()
    logger.info(f"Found {len(completed)} completed targets in checkpoint")
    
    # Evaluate each target
    results = []
    total_targets = len(targets_to_eval)
    completed_count = 0
    skipped_count = 0
    
    for idx, (target_name, target_config) in enumerate(targets_to_eval.items(), 1):
        # Check if already completed
        if target_name in completed:
            if args.resume:
                logger.info(f"[{idx}/{total_targets}] Skipping {target_name} (already completed)")
                result = TargetPredictabilityScore.from_dict(completed[target_name])
                if result.mean_r2 != -999.0:
                    results.append(result)
                skipped_count += 1
                continue
            elif not args.resume:
                # If not resuming, skip silently
                skipped_count += 1
                continue
        
        # Evaluate target
        logger.info(f"[{idx}/{total_targets}] Evaluating {target_name}...")
        try:
            result = evaluate_target_predictability(
                target_name, target_config, symbols, args.data_dir, model_families, multi_model_config,
                output_dir=args.output_dir, min_cs=args.min_cs, max_cs_samples=args.max_cs_samples,
                max_rows_per_symbol=args.max_rows_per_symbol
            )
            
            # Save checkpoint after each target
            checkpoint.save_item(target_name, result.to_dict())
            
            # Skip degenerate targets (marked with mean_score = -999)
            if result.mean_score != -999.0:
                results.append(result)
                completed_count += 1
            else:
                logger.info(f"  Skipped degenerate target: {target_name}")
        
        except Exception as e:
            logger.error(f"  Failed to evaluate {target_name}: {e}")
            checkpoint.mark_failed(target_name, str(e))
            # Continue with next target
    
    logger.info(f"\nCompleted: {completed_count}, Skipped: {skipped_count}, Total: {total_targets}")
    
    # Get all results (including from checkpoint)
    all_results = results
    if args.resume:
        # Merge with checkpoint results
        checkpoint_results = [
            TargetPredictabilityScore.from_dict(v)
            for k, v in completed.items()
            if k not in [r.target_name for r in results]  # Avoid duplicates
        ]
        all_results = results + checkpoint_results
    
    # Save rankings
    save_rankings(all_results, args.output_dir)
    
    # Print summary
    logger.info("="*80)
    logger.info("TARGET PREDICTABILITY RANKINGS")
    logger.info("="*80)
    
    all_results = sorted(all_results, key=lambda x: x.composite_score, reverse=True)
    
    for i, result in enumerate(all_results, 1):
        leakage_indicator = f" [{result.leakage_flag}]" if result.leakage_flag != "OK" else ""
        logger.info(f"\n{i:2d}. {result.target_name:25s} | Score: {result.composite_score:.3f}{leakage_indicator}")
        # Use task-appropriate metric name
        if result.task_type == TaskType.REGRESSION:
            metric_name = "R²"
        elif result.task_type == TaskType.BINARY_CLASSIFICATION:
            metric_name = "ROC-AUC"
        else:
            metric_name = "Accuracy"
        logger.info(f"    {metric_name}: {result.mean_score:.3f} ± {result.std_score:.3f}")
        logger.info(f"    Importance: {result.mean_importance:.2f}")
        logger.info(f"    Recommendation: {_get_recommendation(result)}")
        if result.leakage_flag != "OK":
            logger.info(f"    LEAKAGE FLAG: {result.leakage_flag}")
    
    logger.info("\n" + "="*80)
    logger.info("Target ranking complete!")
    logger.info(f"Results saved to: {args.output_dir}")
    logger.info(f"Checkpoint saved to: {checkpoint_file}")
    logger.info("="*80)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

