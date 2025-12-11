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
Configuration Cleaner Utility

Systematic helper to prevent duplicate argument errors and unknown parameter errors
when passing configs to model constructors. This maintains SST (Single Source of Truth)
while ensuring only valid, non-duplicated keys are passed to estimators.
"""

import inspect
import logging
from typing import Dict, Any, Type

logger = logging.getLogger(__name__)


def clean_config_for_estimator(
    estimator_cls: Type,
    raw_config: Dict[str, Any],
    extra_kwargs: Dict[str, Any] = None,
    family_name: str = None
) -> Dict[str, Any]:
    """
    Clean configuration dictionary for estimator instantiation.
    
    This helper prevents duplicate argument errors and unknown parameter errors by:
    1. Removing keys that are also passed explicitly via extra_kwargs (duplicates)
    2. Removing keys not in the estimator's __init__ signature (unknown params)
    3. Logging what was stripped for visibility
    
    This maintains SST (Single Source of Truth) - values still come from config/defaults,
    but we ensure only valid, non-duplicated keys are passed to the constructor.
    
    Args:
        estimator_cls: The estimator class to instantiate
        raw_config: Raw config dictionary (may contain invalid/duplicate keys)
        extra_kwargs: Dictionary of parameters to pass explicitly (will remove from config).
                     If None, only unknown params are removed.
        family_name: Model family name (for logging). If None, uses estimator class name.
    
    Returns:
        Cleaned config dictionary safe to pass to estimator constructor
    
    Example:
        >>> from lightgbm import LGBMRegressor
        >>> config = {'n_estimators': 100, 'random_seed': 42, 'invalid_param': 123}
        >>> extra = {'random_seed': 42}
        >>> clean = clean_config_for_estimator(LGBMRegressor, config, extra, 'lightgbm')
        >>> model = LGBMRegressor(**clean, **extra)  # Safe - no duplicates or invalid params
    """
    if raw_config is None:
        return {}
    
    if not isinstance(raw_config, dict):
        logger.warning(f"[{family_name or 'unknown'}] Config is not a dict (got {type(raw_config)}), using empty dict")
        return {}
    
    config = raw_config.copy()
    
    if extra_kwargs is None:
        extra_kwargs = {}
    
    if family_name is None:
        family_name = getattr(estimator_cls, '__name__', 'unknown')
    
    # Get estimator __init__ signature to determine valid parameters
    try:
        sig = inspect.signature(estimator_cls.__init__)
        valid_params = set(sig.parameters.keys()) - {"self", "args", "kwargs"}
    except (TypeError, ValueError, AttributeError):
        # Fallback: if we can't inspect, assume all keys are valid (conservative)
        logger.debug(f"[{family_name}] Could not inspect {estimator_cls.__name__} signature, skipping unknown param filtering")
        valid_params = None
    
    dropped_unknown = []
    dropped_duplicates = []
    
    # Remove keys that we will pass explicitly (avoid duplicates)
    for k in list(config.keys()):
        if k in extra_kwargs:
            config.pop(k, None)
            dropped_duplicates.append(k)
    
    # Remove keys the estimator doesn't know about (unknown params)
    if valid_params is not None:
        for k in list(config.keys()):
            if k not in valid_params:
                config.pop(k, None)
                dropped_unknown.append(k)
    
    # Special handling for known parameter conflicts
    # CatBoost: depth and max_depth are synonyms - only one should be used
    if 'catboost' in (family_name or '').lower():
        if 'depth' in config and 'max_depth' in config:
            # Prefer depth (CatBoost's native param), remove max_depth
            config.pop('max_depth', None)
            dropped_duplicates.append('max_depth')
            logger.debug(f"[{family_name}] Removed max_depth (duplicate of depth for CatBoost)")
    
    # RandomForest: verbose must be >= 0, not -1
    if 'random_forest' in (family_name or '').lower() and 'verbose' in config:
        verbose_val = config.get('verbose')
        if verbose_val == -1 or (isinstance(verbose_val, (int, float)) and verbose_val < 0):
            config.pop('verbose', None)
            dropped_unknown.append('verbose')
            logger.debug(f"[{family_name}] Removed invalid verbose={verbose_val} (must be >= 0 for RandomForest)")
    
    # MLPRegressor: learning_rate must be string, not float
    if 'neural_network' in (family_name or '').lower() and 'learning_rate' in config:
        lr_val = config.get('learning_rate')
        if isinstance(lr_val, (int, float)):
            # MLPRegressor expects 'constant', 'adaptive', or 'invscaling'
            config.pop('learning_rate', None)
            dropped_unknown.append('learning_rate')
            logger.debug(f"[{family_name}] Removed invalid learning_rate={lr_val} (MLPRegressor expects string: 'constant', 'adaptive', or 'invscaling')")
    
    # Log what was stripped (use WARNING temporarily to surface issues, then drop to DEBUG)
    if dropped_unknown or dropped_duplicates:
        logger.debug(
            f"[{family_name}] stripped unknown={dropped_unknown} "
            f"duplicate={dropped_duplicates} before estimator init"
        )
    
    return config
