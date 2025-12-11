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
Centralized Configuration Loader

Loads model and training configurations from YAML files.
Supports variants, overrides, and environment-based selection.
"""


import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

# Resolve CONFIG directory (parent of this file)
CONFIG_DIR = Path(__file__).resolve().parent

# Cache for defaults config (loaded once)
_DEFAULTS_CACHE = None

def load_defaults_config() -> Dict[str, Any]:
    """
    Load global defaults configuration (Single Source of Truth).
    
    Returns:
        Dictionary with default values organized by category
    """
    global _DEFAULTS_CACHE
    if _DEFAULTS_CACHE is not None:
        return _DEFAULTS_CACHE
    
    defaults_file = CONFIG_DIR / "defaults.yaml"
    if not defaults_file.exists():
        logger.warning(f"Defaults config not found: {defaults_file}, using empty defaults")
        _DEFAULTS_CACHE = {}
        return _DEFAULTS_CACHE
    
    try:
        with open(defaults_file, 'r') as f:
            loaded = yaml.safe_load(f)
            if loaded is None:
                logger.warning(f"Defaults config {defaults_file} is empty or invalid YAML, using empty defaults")
                _DEFAULTS_CACHE = {}
            else:
                _DEFAULTS_CACHE = loaded
                logger.debug(f"Loaded defaults config from {defaults_file}")
    except Exception as e:
        logger.error(f"Failed to load defaults config {defaults_file}: {e}")
        _DEFAULTS_CACHE = {}
    
    return _DEFAULTS_CACHE


def inject_defaults(config: Dict[str, Any], model_family: Optional[str] = None) -> Dict[str, Any]:
    """
    Inject default values from defaults.yaml into a config dictionary.
    
    This ensures Single Source of Truth (SST) - common settings are defined once
    and automatically applied unless explicitly overridden.
    
    Args:
        config: Configuration dictionary to inject defaults into
        model_family: Optional model family name (for model-specific defaults)
    
    Returns:
        Config dictionary with defaults injected
    """
    defaults = load_defaults_config()
    if not defaults:
        logger.warning("Defaults config is empty or failed to load - defaults will not be injected. Config will use explicit values only.")
        return config
    
    # Log when defaults injection starts (debug level to avoid spam)
    logger.debug(f"Injecting defaults into config (model_family={model_family or 'N/A'})")
    
    # Get random_state from determinism system if not set in defaults
    random_state = None
    if 'randomness' in defaults:
        random_state = defaults['randomness'].get('random_state')
        if random_state is None:
            # Load from determinism system (SST) - load directly to avoid circular dependency
            try:
                pipeline_config_file = CONFIG_DIR / "training_config" / "pipeline_config.yaml"
                if pipeline_config_file.exists():
                    with open(pipeline_config_file, 'r') as f:
                        pipeline_config = yaml.safe_load(f)
                    if pipeline_config is None:
                        logger.warning("pipeline_config.yaml is empty or invalid YAML, using fallback random_state=42")
                        random_state = 42  # FALLBACK_DEFAULT_OK
                    else:
                        random_state = pipeline_config.get('pipeline', {}).get('determinism', {}).get('base_seed', 42)
                else:
                    logger.warning("pipeline_config.yaml not found, using fallback random_state=42")
                    random_state = 42  # FALLBACK_DEFAULT_OK
            except Exception as e:
                logger.warning(f"Failed to load random_state from pipeline_config.yaml: {e}, using fallback random_state=42")
                random_state = 42  # FALLBACK_DEFAULT_OK
        defaults['randomness']['random_state'] = random_state
        defaults['randomness']['random_seed'] = defaults['randomness'].get('random_seed') or random_state
    
    # Determine which defaults category to apply based on model family
    defaults_to_apply = {}
    
    # Always apply randomness and performance defaults
    if 'randomness' in defaults:
        defaults_to_apply.update(defaults['randomness'])
    if 'performance' in defaults:
        defaults_to_apply.update(defaults['performance'])
    
    # Apply model-specific defaults
    if model_family:
        model_lower = model_family.lower()
        if 'tree' in model_lower or model_lower in ['lightgbm', 'xgboost', 'catboost', 'random_forest', 'histogram_gradient_boosting']:
            if 'tree_models' in defaults:
                defaults_to_apply.update(defaults['tree_models'])
        elif ('neural' in model_lower or 'mlp' in model_lower or 'lstm' in model_lower or 
              'cnn' in model_lower or 'transformer' in model_lower or 'multi_task' in model_lower or
              'vae' in model_lower or 'gan' in model_lower or 'meta_learning' in model_lower or
              'reward_based' in model_lower):
            if 'neural_networks' in defaults:
                defaults_to_apply.update(defaults['neural_networks'])
        elif model_lower in ['lasso', 'ridge', 'elastic_net', 'linear']:
            if 'linear_models' in defaults:
                defaults_to_apply.update(defaults['linear_models'])
    
    # Inject defaults into config (only if key doesn't exist)
    injected_keys = []
    for key, value in defaults_to_apply.items():
        if key not in config:
            config[key] = value
            injected_keys.append(key)
    
    # Log what was injected (debug level to avoid spam, but useful for troubleshooting)
    if injected_keys:
        logger.debug(f"   Injected {len(injected_keys)} defaults: {', '.join(injected_keys[:10])}{'...' if len(injected_keys) > 10 else ''}")
    
    return config


def load_model_config(
    model_family: str,
    variant: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Load configuration for a specific model family.
    
    Args:
        model_family: Model family name (e.g., "lightgbm", "xgboost")
        variant: Configuration variant (e.g., "conservative", "balanced", "aggressive")
        overrides: Dictionary of parameters to override
        
    Returns:
        Dictionary with model configuration
        
    Example:
        >>> config = load_model_config("lightgbm")
        >>> config = load_model_config("lightgbm", variant="conservative")
        >>> config = load_model_config("lightgbm", overrides={"learning_rate": 0.02})
    """
    # Normalize model family name
    model_family_lower = model_family.lower().replace("trainer", "").replace("_", "")
    
    # Map common aliases
    aliases = {
        "quantilelightgbm": "quantile_lightgbm",
        "gmmregime": "gmm_regime",
        "changepoint": "change_point",
        "ftrlproximal": "ftrl_proximal",
        "rewardbased": "reward_based",
        "metalearning": "meta_learning",
        "multitask": "multi_task",
    }
    
    if model_family_lower in aliases:
        model_family_lower = aliases[model_family_lower]
    
    # Construct config file path
    config_file = CONFIG_DIR / "model_config" / f"{model_family_lower}.yaml"
    
    # Initialize result dict (will be populated from file or defaults)
    result = {}
    config = {}
    
    if not config_file.exists():
        logger.warning(f"Config file not found: {config_file}, using defaults only")
    else:
        # Load YAML
        try:
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            if config is None:
                logger.warning(f"Config file {config_file} is empty or invalid YAML, using defaults only")
                config = {}
            else:
                logger.debug(f"Loaded model config: {model_family} from {config_file.name}")
                # Start with default hyperparameters from file
                result = config.get("hyperparameters", {}).copy()
        except Exception as e:
            logger.error(f"Failed to load config {config_file}: {e}, using defaults only")
            config = {}
    
    # Inject global defaults (SST) FIRST - only for keys not already set
    # This ensures models get random_state, n_jobs, etc. even if config file doesn't exist
    result = inject_defaults(result, model_family=model_family_lower)
    
    # Apply variant if specified (overrides defaults)
    if variant and "variants" in config:
        if variant in config["variants"]:
            variant_config = config["variants"][variant]
            result.update(variant_config)
            logger.info(f"Applied variant '{variant}' for {model_family}")
        else:
            logger.warning(f"Variant '{variant}' not found for {model_family}, using defaults")
    
    # Apply overrides LAST (highest priority)
    if overrides:
        result.update(overrides)
        logger.info(f"Applied {len(overrides)} overrides for {model_family}")
    
    return result


def load_training_config(config_name: str) -> Dict[str, Any]:
    """
    Load training workflow configuration.
    
    Args:
        config_name: Config file name (without .yaml extension)
        
    Returns:
        Dictionary with training configuration
        
    Example:
        >>> config = load_training_config("first_batch_specs")
        >>> config = load_training_config("sequential_config")
    """
    config_file = CONFIG_DIR / "training_config" / f"{config_name}.yaml"
    
    if not config_file.exists():
        logger.warning(f"Training config not found: {config_file}")
        return {}
    
    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        if config is None:
            logger.warning(f"Training config {config_file} is empty or invalid YAML, using empty config")
            return {}
        return config
    except Exception as e:
        logger.error(f"Failed to load training config {config_file}: {e}")
        return {}


def get_variant_from_env(model_family: str, default: str = "balanced") -> str:
    """
    Get configuration variant from environment variable.
    
    Environment variable format: {MODEL_FAMILY}_VARIANT
    Example: LIGHTGBM_VARIANT=conservative
    
    Args:
        model_family: Model family name
        default: Default variant if env var not set
        
    Returns:
        Variant name
    """
    env_var = f"{model_family.upper()}_VARIANT"
    return os.getenv(env_var, default)


def list_available_configs() -> Dict[str, list]:
    """
    List all available configuration files.
    
    Returns:
        Dictionary with 'model_configs' and 'training_configs' lists
    """
    model_configs = []
    training_configs = []
    
    # List model configs
    model_config_dir = CONFIG_DIR / "model_config"
    if model_config_dir.exists():
        model_configs = [
            f.stem for f in model_config_dir.glob("*.yaml")
        ]
    
    # List training configs
    training_config_dir = CONFIG_DIR / "training_config"
    if training_config_dir.exists():
        training_configs = [
            f.stem for f in training_config_dir.glob("*.yaml")
        ]
    
    return {
        "model_configs": sorted(model_configs),
        "training_configs": sorted(training_configs)
    }


def get_config_variants(model_family: str) -> list:
    """
    Get available variants for a model family.
    
    Args:
        model_family: Model family name
        
    Returns:
        List of variant names
    """
    config_file = CONFIG_DIR / "model_config" / f"{model_family.lower()}.yaml"
    
    if not config_file.exists():
        return []
    
    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        if config is None:
            logger.warning(f"Config file {config_file} is empty or invalid YAML, no variants available")
            return []
        return list(config.get("variants", {}).keys())
    except Exception as e:
        logger.error(f"Failed to load variants for {model_family}: {e}")
        return []


# Convenience functions for common models

def load_lightgbm_config(variant: Optional[str] = None, **overrides) -> Dict[str, Any]:
    """Load LightGBM configuration"""
    return load_model_config("lightgbm", variant=variant, overrides=overrides)

def load_xgboost_config(variant: Optional[str] = None, **overrides) -> Dict[str, Any]:
    """Load XGBoost configuration"""
    return load_model_config("xgboost", variant=variant, overrides=overrides)

def load_ensemble_config(variant: Optional[str] = None, **overrides) -> Dict[str, Any]:
    """Load Ensemble configuration"""
    return load_model_config("ensemble", variant=variant, overrides=overrides)

def load_multi_task_config(variant: Optional[str] = None, **overrides) -> Dict[str, Any]:
    """Load Multi-Task configuration"""
    return load_model_config("multi_task", variant=variant, overrides=overrides)


# Training config convenience functions

def get_cfg(path: str, default: Any = None, config_name: str = "pipeline_config") -> Any:
    """
    Get a nested config value using dot notation.
    
    Args:
        path: Dot-separated path to config value (e.g., "pipeline.isolation_timeout_seconds")
        default: Default value if path not found
        config_name: Name of training config file (without .yaml)
        
    Returns:
        Config value or default
        
    Example:
        >>> timeout = get_cfg("pipeline.isolation_timeout_seconds", default=7200)
        >>> batch_size = get_cfg("preprocessing.validation.test_size", default=0.2)
    """
    config = load_training_config(config_name)
    if not config:
        return default
    
    keys = path.split(".")
    value = config
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default
    return value


def get_pipeline_config() -> Dict[str, Any]:
    """Load pipeline configuration"""
    return load_training_config("pipeline_config")


def get_gpu_config() -> Dict[str, Any]:
    """Load GPU configuration"""
    return load_training_config("gpu_config")


def get_memory_config() -> Dict[str, Any]:
    """Load memory configuration"""
    return load_training_config("memory_config")


def get_preprocessing_config() -> Dict[str, Any]:
    """Load preprocessing configuration"""
    return load_training_config("preprocessing_config")


def get_threading_config() -> Dict[str, Any]:
    """Load threading configuration"""
    return load_training_config("threading_config")


def get_safety_config() -> Dict[str, Any]:
    """Load safety configuration with optional schema validation"""
    cfg = load_training_config("safety_config")
    
    # Optional: Validate schema if available (prevents silent failures)
    try:
        from CONFIG.config_schemas import validate_safety_config
        import os
        # Use strict mode if FOXML_STRICT_MODE=1, otherwise graceful degradation
        strict_mode = os.getenv("FOXML_STRICT_MODE", "0") == "1"
        validate_safety_config(cfg, strict=strict_mode)
    except ImportError:
        pass  # Schema validation not available, skip
    except ValueError as e:
        # Validation failed - behavior depends on strict mode
        # (validate_safety_config already handles strict vs non-strict)
        raise  # Re-raise if strict, or already logged if non-strict
    
    return cfg


def get_callbacks_config() -> Dict[str, Any]:
    """Load callbacks configuration"""
    return load_training_config("callbacks_config")


def get_optimizer_config() -> Dict[str, Any]:
    """Load optimizer configuration"""
    return load_training_config("optimizer_config")


def get_system_config() -> Dict[str, Any]:
    """Load system configuration"""
    return load_training_config("system_config")


def get_family_timeout(family: str, default: int = 7200) -> int:
    """
    Get timeout for a specific family, with fallback to default.
    
    Args:
        family: Model family name
        default: Default timeout in seconds
        
    Returns:
        Timeout in seconds
    """
    # Check for family-specific timeout in pipeline config
    pipeline = get_pipeline_config()
    family_timeouts = pipeline.get("pipeline", {}).get("family_timeouts", {})
    if family in family_timeouts:
        return family_timeouts[family]
    
    # Fallback to general isolation timeout
    timeout = get_cfg("pipeline.isolation_timeout_seconds", default=default)
    return timeout


if __name__ == "__main__":
    # Test the loader
    import json
    
    print("=" * 80)
    print("Configuration Loader Test")
    print("=" * 80)
    print()
    
    # List available configs
    available = list_available_configs()
    print(f"Available model configs: {len(available['model_configs'])}")
    for config in available['model_configs']:
        print(f"  - {config}")
    print()
    
    print(f"Available training configs: {len(available['training_configs'])}")
    for config in available['training_configs']:
        print(f"  - {config}")
    print()
    
    # Test loading a config
    print("Testing LightGBM config load:")
    config = load_lightgbm_config()
    print(json.dumps(config, indent=2))
    print()
    
    # Test variant
    print("Testing LightGBM config with 'conservative' variant:")
    config = load_lightgbm_config(variant="conservative")
    print(json.dumps(config, indent=2))
    print()
    
    # Test overrides
    print("Testing LightGBM config with overrides:")
    config = load_lightgbm_config(learning_rate=0.02, max_depth=10)
    print(json.dumps(config, indent=2))
    print()
    
    # Test variants listing
    print("Available variants for LightGBM:")
    variants = get_config_variants("lightgbm")
    for v in variants:
        print(f"  - {v}")

