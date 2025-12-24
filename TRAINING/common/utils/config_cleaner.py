# MIT License - see LICENSE file

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
    # CRITICAL: This prevents "multiple values for argument" errors
    for k in list(config.keys()):
        if k in extra_kwargs:
            config.pop(k, None)
            dropped_duplicates.append(k)
    
    # Special case: Remove 'verbose' from config if it exists (common source of double argument errors)
    # verbose is often set explicitly in code, so remove from config to avoid conflicts
    # This is a defensive measure - explicit verbose= in code takes precedence
    if 'verbose' in config and 'verbose' not in extra_kwargs:
        # Only remove if not in extra_kwargs (if in extra_kwargs, it was already removed above)
        # But we still want to remove it to prevent conflicts with explicit verbose= in code
        verbose_val = config.pop('verbose', None)
        if verbose_val is not None:
            logger.debug(f"[{family_name}] Removed verbose={verbose_val} from config (will use explicit value if provided)")
    
    # Special case: Remove 'random_state' from config if it exists and will be passed explicitly
    # random_state is often set explicitly in code (from determinism system), so remove from config to avoid conflicts
    # This is a defensive measure - explicit random_state= in code takes precedence
    if 'random_state' in config and 'random_state' not in extra_kwargs:
        # Only remove if not in extra_kwargs (if in extra_kwargs, it was already removed above)
        # But we still want to remove it to prevent conflicts with explicit random_state= in code
        random_state_val = config.pop('random_state', None)
        if random_state_val is not None:
            logger.debug(f"[{family_name}] Removed random_state={random_state_val} from config (will use explicit value if provided)")
    
    # Remove keys the estimator doesn't know about (unknown params)
    if valid_params is not None:
        for k in list(config.keys()):
            if k not in valid_params:
                config.pop(k, None)
                dropped_unknown.append(k)
    
    # Special handling for known parameter conflicts and invalid values
    # These need to be checked AFTER unknown param removal because they might be valid params with invalid values
    
    # CatBoost: depth and max_depth are synonyms - only one should be used
    if 'catboost' in (family_name or '').lower():
        if 'depth' in config and 'max_depth' in config:
            # Prefer depth (CatBoost's native param), remove max_depth
            config.pop('max_depth', None)
            dropped_duplicates.append('max_depth')
            logger.debug(f"[{family_name}] Removed max_depth (duplicate of depth for CatBoost)")
    
    # RandomForest: verbose must be >= 0, not -1 (even though verbose is a valid param)
    if 'random_forest' in (family_name or '').lower() and 'verbose' in config:
        verbose_val = config.get('verbose')
        if verbose_val == -1 or (isinstance(verbose_val, (int, float)) and verbose_val < 0):
            config.pop('verbose', None)
            dropped_unknown.append('verbose')
            logger.debug(f"[{family_name}] Removed invalid verbose={verbose_val} (must be >= 0 for RandomForest)")
    
    # MLPRegressor/MLPClassifier: verbose must be >= 0, not -1 (sklearn doesn't accept negative values)
    if 'neural_network' in (family_name or '').lower() and 'verbose' in config:
        verbose_val = config.get('verbose')
        if verbose_val == -1 or (isinstance(verbose_val, (int, float)) and verbose_val < 0):
            # Convert -1 to 0 (silent), keep non-negative values as-is
            config['verbose'] = 0
            logger.debug(f"[{family_name}] Sanitized verbose={verbose_val} -> 0 (MLPRegressor/MLPClassifier requires >= 0)")
    
    # MLPRegressor: learning_rate must be string, not float (even though learning_rate is a valid param)
    if 'neural_network' in (family_name or '').lower() and 'learning_rate' in config:
        lr_val = config.get('learning_rate')
        if isinstance(lr_val, (int, float)):
            # MLPRegressor expects 'constant', 'adaptive', or 'invscaling'
            config.pop('learning_rate', None)
            dropped_unknown.append('learning_rate')
            logger.debug(f"[{family_name}] Removed invalid learning_rate={lr_val} (MLPRegressor expects string: 'constant', 'adaptive', or 'invscaling')")
    
    # CatBoost: only one of random_state or random_seed should be set
    if 'catboost' in (family_name or '').lower():
        if 'random_state' in config and 'random_seed' in config:
            # Prefer random_seed (CatBoost's native param), remove random_state
            config.pop('random_state', None)
            dropped_duplicates.append('random_state')
            logger.debug(f"[{family_name}] Removed random_state (duplicate of random_seed for CatBoost)")
        elif 'random_state' in config:
            # Check if random_seed will be passed explicitly (in extra_kwargs)
            if extra_kwargs and 'random_seed' in extra_kwargs:
                # random_seed is already in extra_kwargs, just remove random_state to avoid conflict
                config.pop('random_state', None)
                dropped_duplicates.append('random_state')
                logger.debug(f"[{family_name}] Removed random_state (random_seed already in extra_kwargs for CatBoost)")
            else:
                # Convert random_state to random_seed (CatBoost's preferred name)
                config['random_seed'] = config.pop('random_state')
                logger.debug(f"[{family_name}] Converted random_state -> random_seed for CatBoost")
        
        # CatBoost: only one of iterations, n_estimators, num_boost_round, num_trees should be set
        # These are all synonyms - prefer 'iterations' (CatBoost's native param)
        iteration_synonyms = ['iterations', 'n_estimators', 'num_boost_round', 'num_trees']
        found_iteration_params = [p for p in iteration_synonyms if p in config]
        if len(found_iteration_params) > 1:
            # Keep 'iterations' if present, otherwise keep the first one found
            if 'iterations' in found_iteration_params:
                param_to_keep = 'iterations'
            else:
                param_to_keep = found_iteration_params[0]
            
            # Remove all others
            for param in found_iteration_params:
                if param != param_to_keep:
                    config.pop(param, None)
                    dropped_duplicates.append(param)
                    logger.debug(f"[{family_name}] Removed {param} (duplicate of {param_to_keep} for CatBoost)")
        
        # Also check if any iteration param is in extra_kwargs
        if extra_kwargs:
            extra_iteration_params = [p for p in iteration_synonyms if p in extra_kwargs]
            if extra_iteration_params:
                # Remove iteration params from config if they're in extra_kwargs
                for param in iteration_synonyms:
                    if param in config:
                        config.pop(param, None)
                        dropped_duplicates.append(param)
                        logger.debug(f"[{family_name}] Removed {param} from config (already in extra_kwargs for CatBoost)")
    
    # Log what was stripped (use WARNING temporarily to surface issues, then drop to DEBUG)
    if dropped_unknown or dropped_duplicates:
        logger.debug(
            f"[{family_name}] stripped unknown={dropped_unknown} "
            f"duplicate={dropped_duplicates} before estimator init"
        )
    
    return config
