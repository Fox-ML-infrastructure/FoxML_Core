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
Unified Task Type System for Target Ranking

Defines TaskType enum and configuration classes (TargetConfig, ModelConfig)
that enable a single pipeline to handle both regression and classification targets.
"""


from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Dict, Any, Set, Optional, List
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class TaskType(Enum):
    """Task type for model training"""
    REGRESSION = auto()
    BINARY_CLASSIFICATION = auto()
    MULTICLASS_CLASSIFICATION = auto()
    
    @classmethod
    def from_target_column(cls, target_column: str, y: Optional[np.ndarray] = None) -> 'TaskType':
        """
        Infer task type from target column name and/or data
        
        Args:
            target_column: Name of target column
            y: Optional target values to inspect
        
        Returns:
            TaskType enum
        """
        # Check column name patterns first
        if target_column.startswith('fwd_ret_'):
            return cls.REGRESSION
        
        # Binary classification patterns
        if any(target_column.startswith(prefix) for prefix in [
            'y_will_peak', 'y_will_valley', 'y_will_swing', 
            'y_will_peak_mfe', 'y_will_valley_mdd'
        ]):
            # Could be binary or multiclass - check data if available
            if y is not None:
                unique_vals = np.unique(y[~np.isnan(y)])
                if len(unique_vals) == 2 and set(unique_vals).issubset({0, 1, 0.0, 1.0}):
                    return cls.BINARY_CLASSIFICATION
                elif len(unique_vals) <= 10:
                    return cls.MULTICLASS_CLASSIFICATION
        
        # Multiclass patterns
        if any(target_column.startswith(prefix) for prefix in [
            'y_first_touch', 'hit_direction', 'hit_asym'
        ]):
            return cls.MULTICLASS_CLASSIFICATION
        
        # If we have data, inspect it
        if y is not None:
            unique_vals = np.unique(y[~np.isnan(y)])
            n_unique = len(unique_vals)
            
            if n_unique == 2 and set(unique_vals).issubset({0, 1, 0.0, 1.0}):
                return cls.BINARY_CLASSIFICATION
            elif n_unique <= 10:
                # Check if integer-like (classification) vs continuous (regression)
                if all(isinstance(v, (int, np.integer)) or 
                       (isinstance(v, float) and float(v).is_integer()) 
                       for v in unique_vals):
                    return cls.MULTICLASS_CLASSIFICATION
            
            # Default to regression for continuous values
            return cls.REGRESSION
        
        # Default fallback: assume regression
        return cls.REGRESSION


@dataclass
class TargetConfig:
    """Configuration for a target"""
    name: str
    target_column: str
    task_type: TaskType
    horizon: Optional[int] = None  # in bars/minutes/days
    description: str = ""
    display_name: Optional[str] = None
    
    # Optional metadata
    class_thresholds: Dict[str, float] = field(default_factory=dict)
    # e.g., {"long": 0.005, "short": -0.005} for binarizing returns
    
    # Legacy fields (for compatibility)
    top_n: int = 60
    method: str = "mean"
    enabled: bool = True
    use_case: str = ""
    
    def __post_init__(self):
        """Set display_name if not provided"""
        if self.display_name is None:
            self.display_name = self.name


@dataclass
class ModelConfig:
    """Configuration for a model family"""
    name: str
    constructor: Callable[..., Any]  # e.g., lambda **kw: LGBMRegressor(**kw)
    supported_tasks: Set[TaskType]
    default_params: Dict[str, Any] = field(default_factory=dict)
    
    # What metric should be primary for this model-task combo
    primary_metric_by_task: Dict[TaskType, str] = field(default_factory=dict)
    
    # Extra capabilities if needed later
    tags: Set[str] = field(default_factory=set)
    
    # Legacy fields (for compatibility with existing configs)
    enabled: bool = True
    weight: float = 1.0
    importance_method: str = "native"
    
    def __post_init__(self):
        """Set default primary metrics if not specified"""
        if not self.primary_metric_by_task:
            self.primary_metric_by_task = {
                TaskType.REGRESSION: "r2",
                TaskType.BINARY_CLASSIFICATION: "roc_auc",
                TaskType.MULTICLASS_CLASSIFICATION: "accuracy"
            }


def is_compatible(target: TargetConfig, model: ModelConfig) -> bool:
    """
    Check if a model can train on a given target
    
    Args:
        target: TargetConfig
        model: ModelConfig
    
    Returns:
        True if model supports the target's task type
    """
    return target.task_type in model.supported_tasks


def create_model_configs_from_yaml(
    multi_model_config: Dict[str, Any],
    task_type: TaskType
) -> List[ModelConfig]:
    """
    Create ModelConfig objects from YAML config
    
    Args:
        multi_model_config: Loaded YAML config dict
        task_type: Task type to filter models for
    
    Returns:
        List of ModelConfig objects that support the given task_type
    """
    from typing import TYPE_CHECKING
    
    model_configs = []
    model_families = multi_model_config.get('model_families', {})
    
    # Defensive check: ensure model_families is a dict
    if model_families is None or not isinstance(model_families, dict):
        logger.warning(f"model_families in config is None or not a dict (got {type(model_families)}). Returning empty list.")
        return []
    
    for model_name, model_spec in model_families.items():
        # Defensive check: skip None or non-dict model specs
        if model_spec is None:
            logger.warning(f"Model '{model_name}' has None config. Skipping.")
            continue
        
        if not isinstance(model_spec, dict):
            logger.warning(f"Model '{model_name}' config is not a dict (got {type(model_spec)}). Skipping.")
            continue
        
        if not model_spec.get('enabled', False):
            continue
        
        # Determine supported tasks based on model type
        # Most models support both regression and classification
        supported_tasks = {TaskType.REGRESSION, TaskType.BINARY_CLASSIFICATION, 
                          TaskType.MULTICLASS_CLASSIFICATION}
        
        # Some models are regression-only or classification-only
        if model_name in ['lasso', 'ridge', 'elastic_net']:
            supported_tasks = {TaskType.REGRESSION}
        elif model_name in ['mutual_information', 'univariate_selection']:
            # These support both but are feature selectors, not predictors
            supported_tasks = {TaskType.REGRESSION, TaskType.BINARY_CLASSIFICATION,
                              TaskType.MULTICLASS_CLASSIFICATION}
        
        # Filter by requested task_type
        if task_type not in supported_tasks:
            continue
        
        # Get constructor based on model name and task type
        # Defensive check: ensure config is not None
        model_config = model_spec.get('config')
        if model_config is None or not isinstance(model_config, dict):
            model_config = {}
        constructor = _get_model_constructor(model_name, task_type, model_config)
        
        if constructor is None:
            continue
        
        # Create ModelConfig
        # Defensive check: ensure default_params is not None
        default_params = model_spec.get('config')
        if default_params is None or not isinstance(default_params, dict):
            default_params = {}
        config = ModelConfig(
            name=model_name,
            constructor=constructor,
            supported_tasks=supported_tasks,
            default_params=default_params,
            enabled=model_spec.get('enabled', True),
            weight=model_spec.get('weight', 1.0),
            importance_method=model_spec.get('importance_method', 'native')
        )
        
        model_configs.append(config)
    
    return model_configs


def _get_model_constructor(
    model_name: str,
    task_type: TaskType,
    config: Dict[str, Any]
) -> Optional[Callable]:
    """
    Get model constructor function for a given model name and task type
    
    Returns:
        Constructor function or None if model not available
    """
    import lightgbm as lgb
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    from sklearn.neural_network import MLPRegressor, MLPClassifier
    from sklearn.linear_model import Lasso, LogisticRegression
    
    # Defensive check: ensure config is a dict
    if config is None or not isinstance(config, dict):
        logger.warning(f"Config for {model_name} is None or not a dict (got {type(config)}), using empty config")
        config = {}
    
    # Remove task-specific params from config (we'll set them explicitly)
    # NOTE: 'device' is NOT removed - it's needed for GPU acceleration (e.g., LightGBM device='gpu')
    config_clean = {k: v for k, v in config.items() 
                    if k not in ['objective', 'metric', 'eval_metric', 'loss_function']}
    
    # CRITICAL FIX: Enforce reproducibility with default random_state
    # Without this, results vary between runs, making it impossible to verify improvements
    if 'random_state' not in config_clean:
        config_clean['random_state'] = 42
    
    if model_name == 'lightgbm':
        # CRITICAL FIX: Standardize constructor signatures - all accept **kwargs for consistency
        # This prevents polymorphism crashes where training loop doesn't know which signature to use
        if task_type == TaskType.REGRESSION:
            return lambda **kwargs: lgb.LGBMRegressor(objective='regression', **config_clean)
        elif task_type == TaskType.BINARY_CLASSIFICATION:
            return lambda **kwargs: lgb.LGBMClassifier(objective='binary', **config_clean)
        elif task_type == TaskType.MULTICLASS_CLASSIFICATION:
            # Extract n_classes from kwargs if provided, otherwise infer from data later
            def make_multiclass_lgbm(**kwargs):
                n_classes = kwargs.get('n_classes', None)
                if n_classes is not None:
                    return lgb.LGBMClassifier(objective='multiclass', num_class=n_classes, **config_clean)
                else:
                    return lgb.LGBMClassifier(objective='multiclass', **config_clean)
            return make_multiclass_lgbm
    
    elif model_name == 'xgboost':
        try:
            import xgboost as xgb
            # CRITICAL FIX: Standardize constructor signatures - all accept **kwargs for consistency
            if task_type == TaskType.REGRESSION:
                return lambda **kwargs: xgb.XGBRegressor(objective='reg:squarederror', **config_clean)
            elif task_type == TaskType.BINARY_CLASSIFICATION:
                return lambda **kwargs: xgb.XGBClassifier(objective='binary:logistic', **config_clean)
            elif task_type == TaskType.MULTICLASS_CLASSIFICATION:
                # Extract n_classes from kwargs if provided, otherwise infer from data later
                def make_multiclass_xgb(**kwargs):
                    n_classes = kwargs.get('n_classes', None)
                    if n_classes is not None:
                        return xgb.XGBClassifier(objective='multi:softprob', num_class=n_classes, **config_clean)
                    else:
                        return xgb.XGBClassifier(objective='multi:softprob', **config_clean)
                return make_multiclass_xgb
        except ImportError:
            return None
    
    elif model_name == 'random_forest':
        # Standardized signature: all accept **kwargs (ignored for RandomForest)
        if task_type == TaskType.REGRESSION:
            return lambda **kwargs: RandomForestRegressor(**config_clean)
        elif task_type in {TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION}:
            return lambda **kwargs: RandomForestClassifier(**config_clean)
    
    elif model_name == 'neural_network':
        # Standardized signature: all accept **kwargs (ignored for MLP)
        if task_type == TaskType.REGRESSION:
            return lambda **kwargs: MLPRegressor(**config_clean)
        elif task_type in {TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION}:
            return lambda **kwargs: MLPClassifier(**config_clean)
    
    elif model_name == 'catboost':
        try:
            import catboost as cb
            # Standardized signature: all accept **kwargs (CatBoost handles n_classes automatically)
            if task_type == TaskType.REGRESSION:
                return lambda **kwargs: cb.CatBoostRegressor(**config_clean)
            elif task_type in {TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION}:
                return lambda **kwargs: cb.CatBoostClassifier(**config_clean)
        except ImportError:
            return None
    
    elif model_name == 'lasso':
        # Standardized signature: all accept **kwargs (ignored for Lasso)
        if task_type == TaskType.REGRESSION:
            return lambda **kwargs: Lasso(**config_clean)
        # Lasso doesn't support classification directly
        return None
    
    # Add more models as needed...
    
    return None

