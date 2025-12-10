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

# TargetPredictabilityScore class definition

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
    # Auto-fix and rerun tracking
    autofix_info: Optional['AutoFixInfo'] = None  # AutoFixInfo from auto-fixer (if leakage was detected)
    leakage_flags: Dict[str, bool] = None  # Detailed leakage flags: {"perfect_train_acc": bool, "high_auc": bool, etc.}
    status: str = "OK"  # "OK", "SUSPICIOUS_STRONG", "LEAKAGE_UNRESOLVED", "LEAKAGE_UNRESOLVED_MAX_RETRIES"
    attempts: int = 1  # Number of evaluation attempts (for auto-rerun tracking)
    
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


