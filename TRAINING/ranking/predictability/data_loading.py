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
from TRAINING.utils.checkpoint import CheckpointManager

# Import unified task type system
from TRAINING.utils.task_types import (
    TaskType, TargetConfig, ModelConfig, 
    is_compatible, create_model_configs_from_yaml
)
from TRAINING.utils.task_metrics import evaluate_by_task, compute_composite_score
from TRAINING.utils.target_validation import validate_target, check_cv_compatibility

# Suppress expected warnings (harmless)
warnings.filterwarnings('ignore', message='X does not have valid feature names')
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
warnings.filterwarnings('ignore', category=RuntimeWarning, module='sklearn')
warnings.filterwarnings('ignore', message='invalid value encountered in divide')
warnings.filterwarnings('ignore', message='invalid value encountered in true_divide')

# Setup logging with journald support
from TRAINING.utils.logging_setup import setup_logging
logger = setup_logging(
    script_name="rank_target_predictability",
    level=logging.INFO,
    use_journald=True
)

# Data loading and configuration utilities

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
    max_samples: Optional[int] = None  # Load from config if None
) -> pd.DataFrame:
    """
    Load sample data for a symbol.
    
    Args:
        symbol: Symbol to load data for
        data_dir: Directory containing data files
        max_samples: Maximum number of samples to load (loads from config if None)
    
    Returns:
        DataFrame with loaded data
    """
    # Load from config if not provided
    if max_samples is None:
        if _CONFIG_AVAILABLE:
            try:
                from config_loader import get_cfg
                max_samples = int(get_cfg("pipeline.data_limits.default_max_samples_ranking", default=10000, config_name="pipeline_config"))
            except Exception:
                max_samples = 10000
        else:
            max_samples = 10000
    """Load sample data for a symbol"""
    symbol_dir = data_dir / f"symbol={symbol}"
    parquet_file = symbol_dir / f"{symbol}.parquet"
    
    if not parquet_file.exists():
        logger.warning(f"  Symbol {symbol} not found in dataset, skipping")
        raise FileNotFoundError(f"Data not found: {parquet_file}")
    
    df = pd.read_parquet(parquet_file)
    
    # Sample if too large - use deterministic seed based on symbol
    if len(df) > max_samples:
        # Generate stable seed from symbol name for deterministic sampling
        from TRAINING.common.determinism import stable_seed_from
        sample_seed = stable_seed_from([symbol, "data_sampling"])
        df = df.sample(n=max_samples, random_state=sample_seed)
    
    return df


def prepare_features_and_target(
    df: pd.DataFrame,
    target_column: str,
    target_config: Optional[TargetConfig] = None,
    explicit_interval: Optional[Union[int, str]] = None,  # Explicit interval from config
    experiment_config: Optional[Any] = None  # Optional ExperimentConfig (for data.bar_interval)
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
    
    # LEAKAGE PREVENTION: Filter out leaking features (target-aware, with registry validation)
    from TRAINING.utils.leakage_filtering import filter_features_for_target
    from TRAINING.utils.data_interval import detect_interval_from_dataframe
    
    # Detect data interval for horizon conversion (use explicit_interval if provided)
    detected_interval = detect_interval_from_dataframe(
        df, 
        timestamp_column='ts', 
        default=5,
        explicit_interval=explicit_interval,
        experiment_config=experiment_config
    )
    
    all_columns = df.columns.tolist()
    # Use target-aware filtering with registry validation
    # Enable verbose logging to see what's being filtered
    safe_columns = filter_features_for_target(
        all_columns, 
        target_column, 
        verbose=True,
        use_registry=True,  # Enable registry validation
        data_interval_minutes=detected_interval,
        for_ranking=True  # Use permissive rules for ranking (allow basic OHLCV/TA)
    )
    
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
    """Load multi-model configuration for target ranking
    
    Checks new location first (CONFIG/target_ranking/multi_model.yaml),
    then falls back to feature_selection config, then legacy location.
    """
    if config_path is None:
        # Try new target_ranking location first
        new_path = _REPO_ROOT / "CONFIG" / "target_ranking" / "multi_model.yaml"
        feature_selection_path = _REPO_ROOT / "CONFIG" / "feature_selection" / "multi_model.yaml"
        legacy_path = _REPO_ROOT / "CONFIG" / "multi_model_feature_selection.yaml"
        
        if new_path.exists():
            config_path = new_path
            logger.debug(f"Using target ranking config: {config_path}")
        elif feature_selection_path.exists():
            config_path = feature_selection_path
            logger.debug(f"Using feature selection config (shared): {config_path}")
        elif legacy_path.exists():
            config_path = legacy_path
            logger.warning(
                f"⚠️  DEPRECATED: Using legacy config location: {legacy_path}\n"
                f"   Please migrate to: CONFIG/target_ranking/multi_model.yaml"
            )
        else:
            logger.debug(f"Multi-model config not found in any location, using defaults")
            return None
    
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


