# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
# Copyright (c) 2025-2026 Fox ML Infrastructure LLC

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

from CONFIG.config_migrator import migrate_config

logger = logging.getLogger(__name__)

# Resolve CONFIG directory (parent of this file)
CONFIG_DIR = Path(__file__).resolve().parent

# Cache for defaults config (loaded once)
_DEFAULTS_CACHE = None

def clear_config_cache() -> None:
    """
    Clear all config caches to force reload on next access.
    Useful when config files are modified and you want changes to take effect immediately.
    """
    global _DEFAULTS_CACHE
    _DEFAULTS_CACHE = None
    logger.info("✅ Cleared config cache - configs will be reloaded on next access")

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
        config: Configuration dictionary to inject defaults into (can be None)
        model_family: Optional model family name (for model-specific defaults)
    
    Returns:
        Config dictionary with defaults injected
    """
    # Handle None config (initialize empty dict)
    if config is None:
        config = {}
    
    # Ensure config is a dict (not list, str, etc.)
    if not isinstance(config, dict):
        logger.warning(f"Config is not a dict (got {type(config)}), initializing empty dict")
        config = {}
    
    # SST: Migrate deprecated config keys to canonical names
    config = migrate_config(config, warn=True, recursive=True)
    
    defaults = load_defaults_config()
    if not defaults or not isinstance(defaults, dict):
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
            # Use get_config_path() for path resolution (SST-compliant)
            try:
                pipeline_config_file = get_config_path("pipeline_config")
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
    
    # Try new location first (models/), then fallback to old (model_config/)
    config_file = CONFIG_DIR / "models" / f"{model_family_lower}.yaml"
    old_config_file = CONFIG_DIR / "model_config" / f"{model_family_lower}.yaml"
    
    # Use old location if new doesn't exist
    if not config_file.exists() and old_config_file.exists():
        config_file = old_config_file
        logger.debug(f"Using legacy location: model_config/{model_family_lower}.yaml (consider migrating to models/{model_family_lower}.yaml)")
    
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
    Load training workflow configuration from canonical paths.
    
    Args:
        config_name: Config file name (without .yaml extension)
        
    Returns:
        Dictionary with training configuration
        
    Example:
        >>> config = load_training_config("first_batch_specs")
        >>> config = load_training_config("sequential_config")
    """
    # Map old config names to new names
    name_mapping = {
        "intelligent_training_config": "intelligent",
        "safety_config": "safety",
        "preprocessing_config": "preprocessing",
        "optimizer_config": "optimizer",
        "callbacks_config": "callbacks",
        "routing_config": "routing",
        "stability_config": "stability",
        "decision_policies": "decisions",
        "family_config": "families",
        "sequential_config": "sequential",
        "first_batch_specs": "first_batch",
        "gpu_config": "gpu",
        "memory_config": "memory",
        "threading_config": "threading",
        "pipeline_config": "pipeline",
        "system_config": "system",  # Maps to core/system.yaml
        "target_ranking_config": "target_configs",  # Maps to ranking/targets/configs.yaml
        "multi_model": "target_ranking_multi_model",  # Maps to ranking/targets/multi_model.yaml
        "multi_model_feature_selection": "feature_selection_multi_model",  # Maps to ranking/features/multi_model.yaml
    }
    
    # Try new location first (pipeline/training/ or pipeline/)
    new_name = name_mapping.get(config_name, config_name)
    
    # Determine if it's a training-specific config or pipeline-level config
    training_configs = {"intelligent", "safety", "preprocessing", "optimizer", "callbacks", 
                       "routing", "stability", "decisions", "families", "sequential", "first_batch",
                       "reproducibility"}
    
    if new_name in training_configs:
        # Training-specific configs go in pipeline/training/
        config_file = CONFIG_DIR / "pipeline" / "training" / f"{new_name}.yaml"
    elif new_name == "system":
        # System config goes in core/
        config_file = CONFIG_DIR / "core" / "system.yaml"
    elif new_name == "target_configs":
        # Target ranking config goes in ranking/targets/
        config_file = CONFIG_DIR / "ranking" / "targets" / "configs.yaml"
    elif new_name == "target_ranking_multi_model":
        # Target ranking multi_model config goes in ranking/targets/
        config_file = CONFIG_DIR / "ranking" / "targets" / "multi_model.yaml"
    elif new_name == "feature_selection_multi_model":
        # Feature selection multi_model config goes in ranking/features/
        config_file = CONFIG_DIR / "ranking" / "features" / "multi_model.yaml"
    else:
        # Pipeline-level configs go in pipeline/
        config_file = CONFIG_DIR / "pipeline" / f"{new_name}.yaml"
    
    # Load from canonical location
    if config_file.exists():
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
    
    # Config not found
    logger.warning(f"Training config '{config_name}' not found at {config_file.relative_to(CONFIG_DIR)}")
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
    List all available configuration files from canonical locations.
    
    Returns:
        Dictionary with 'model_configs' and 'training_configs' lists
    """
    model_configs = []
    training_configs = []
    
    # List model configs (check both models/ and model_config/ for backward compatibility)
    models_dir = CONFIG_DIR / "models"
    if models_dir.exists():
        model_configs = [
            f.stem for f in models_dir.glob("*.yaml")
        ]
    
    # Also check legacy model_config/ directory
    model_config_dir = CONFIG_DIR / "model_config"
    if model_config_dir.exists():
        legacy_models = [
            f.stem for f in model_config_dir.glob("*.yaml")
        ]
        # Merge without duplicates
        model_configs = sorted(list(set(model_configs + legacy_models)))
    
    # List training configs from canonical pipeline/training/ location
    training_dir = CONFIG_DIR / "pipeline" / "training"
    if training_dir.exists():
        training_configs = [
            f.stem for f in training_dir.glob("*.yaml")
        ]
    
    # Also list pipeline-level configs
    pipeline_dir = CONFIG_DIR / "pipeline"
    if pipeline_dir.exists():
        pipeline_configs = [
            f.stem for f in pipeline_dir.glob("*.yaml")
            if f.name not in ["pipeline.yaml"]  # Exclude main pipeline.yaml (it's loaded separately)
        ]
        training_configs = sorted(list(set(training_configs + pipeline_configs)))
    
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

def validate_config_exists(config_path: str, config_name: str) -> bool:
    """
    Verify config path exists before using fallback defaults.
    
    Args:
        config_path: Dot-separated path to config value (e.g., "pipeline.determinism.base_seed")
        config_name: Name of config file (without .yaml)
        
    Returns:
        True if config file exists, False otherwise
    """
    try:
        config_file = get_config_path(config_name)
        if config_file.exists():
            return True
        else:
            logger.debug(f"Config file '{config_name}' not found at expected path: {config_file}")
            return False
    except Exception as e:
        logger.debug(f"Error validating config '{config_name}': {e}")
        return False


def get_cfg(path: str, default: Any = None, config_name: str = "pipeline_config") -> Any:
    """
    Get a nested config value using dot notation.
    
    SINGLE SOURCE OF TRUTH: Always loads from CONFIG files first, then falls back to default.
    
    Args:
        path: Dot-separated path to config value (e.g., "pipeline.isolation_timeout_seconds")
        default: Default value if path not found (should match config file default)
        config_name: Name of training config file (without .yaml)
        
    Returns:
        Config value or default
        
    Example:
        >>> timeout = get_cfg("pipeline.isolation_timeout_seconds", default=7200)
        >>> batch_size = get_cfg("preprocessing.validation.test_size", default=0.2)
    """
    # Handle legacy "training_config" name - fallback to intelligent_training_config or pipeline_config
    fallback_configs = []
    if config_name == "training_config":
        # "training_config" is a legacy name - try intelligent_training_config first, then pipeline_config
        fallback_configs = ["intelligent_training_config", "pipeline_config"]
        config_name = "intelligent_training_config"  # Try this first
    
    # Handle "model_config" - not a training config, models are per-family files
    if config_name == "model_config":
        logger.warning(
            f"get_cfg() called with config_name='model_config', but model configs are per-family files in models/. "
            f"Use load_model_config(model_family) instead. Returning default value for path '{path}'."
        )
        return default
    
    config = load_training_config(config_name)
    
    # If config not found and we have fallbacks, try them
    if not config and fallback_configs:
        for fallback_name in fallback_configs:
            config = load_training_config(fallback_name)
            if config:
                logger.debug(f"Config '{config_name}' not found, using fallback '{fallback_name}'")
                break
    
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


def get_config_path(config_name: str) -> Path:
    """
    Get the path to a config file using canonical paths only.
    
    This provides a centralized way to resolve config file paths.
    
    Args:
        config_name: Name of config file (e.g., "excluded_features", "feature_registry", 
                    "intelligent_training_config", "lightgbm")
    
    Returns:
        Path to config file (may not exist - caller should check)
    
    Examples:
        >>> get_config_path("excluded_features")  # Returns CONFIG/data/excluded_features.yaml
        >>> get_config_path("lightgbm")  # Returns CONFIG/models/lightgbm.yaml
        >>> get_config_path("intelligent_training_config")  # Returns CONFIG/pipeline/training/intelligent.yaml
    """
    # Map of config names to canonical paths
    config_mappings = {
        # Core configs
        "logging_config": "core/logging.yaml",
        "system_config": "core/system.yaml",
        
        # Data configs
        "excluded_features": "data/excluded_features.yaml",
        "feature_registry": "data/feature_registry.yaml",
        "feature_target_schema": "data/feature_target_schema.yaml",
        "feature_groups": "data/feature_groups.yaml",
        
        # Training configs
        "intelligent_training_config": "pipeline/training/intelligent.yaml",
        "safety_config": "pipeline/training/safety.yaml",
        "preprocessing_config": "pipeline/training/preprocessing.yaml",
        "optimizer_config": "pipeline/training/optimizer.yaml",
        "callbacks_config": "pipeline/training/callbacks.yaml",
        "routing_config": "pipeline/training/routing.yaml",
        "stability_config": "pipeline/training/stability.yaml",
        "decision_policies": "pipeline/training/decisions.yaml",
        "family_config": "pipeline/training/families.yaml",
        "sequential_config": "pipeline/training/sequential.yaml",
        "first_batch_specs": "pipeline/training/first_batch.yaml",
        "reproducibility": "pipeline/training/reproducibility.yaml",
        
        # Pipeline configs
        "gpu_config": "pipeline/gpu.yaml",
        "memory_config": "pipeline/memory.yaml",
        "threading_config": "pipeline/threading.yaml",
        "pipeline_config": "pipeline/pipeline.yaml",
        
        # Ranking configs
        "target_ranking_multi_model": "ranking/targets/multi_model.yaml",
        "target_configs": "ranking/targets/configs.yaml",
        "feature_selection_multi_model": "ranking/features/multi_model.yaml",
        "feature_selection_config": "ranking/features/config.yaml",
        
        # Core configs
        "identity_config": "core/identity_config.yaml",
    }
    
    # Special handling for experiment configs
    if config_name.startswith("experiment:"):
        # Format: "experiment:exp_name" -> experiments/exp_name.yaml
        exp_name = config_name.replace("experiment:", "", 1)
        return get_experiment_config_path(exp_name)
    
    # Check if we have a mapping
    if config_name in config_mappings:
        canonical_path = config_mappings[config_name]
        return CONFIG_DIR / canonical_path
    
    # For model configs, check models/ first, then model_config/
    model_file = CONFIG_DIR / "models" / f"{config_name}.yaml"
    if model_file.exists():
        return model_file
    old_model_file = CONFIG_DIR / "model_config" / f"{config_name}.yaml"
    if old_model_file.exists():
        return old_model_file
    
    # Check if it's an experiment config (experiments/ directory)
    exp_file = CONFIG_DIR / "experiments" / f"{config_name}.yaml"
    if exp_file.exists():
        return exp_file
    
    # Default: assume it's in the root or use the name directly
    return CONFIG_DIR / f"{config_name}.yaml"


def get_experiment_config_path(exp_name: str) -> Path:
    """
    Get path to experiment config file.
    
    Args:
        exp_name: Experiment name (without .yaml extension)
        
    Returns:
        Path to experiment config file
        
    Example:
        >>> exp_path = get_experiment_config_path("e2e_full_targets_test")
        >>> # Returns CONFIG_DIR / "experiments" / "e2e_full_targets_test.yaml"
    """
    return CONFIG_DIR / "experiments" / f"{exp_name}.yaml"


def load_experiment_config(exp_name: str) -> Dict[str, Any]:
    """
    Load experiment config by name.
    
    Experiment configs are the TOP-LEVEL config (highest priority after CLI args).
    They override intelligent_training_config and defaults.
    
    **Config Precedence (Highest to Lowest):**
    1. CLI arguments (highest priority)
    2. **Experiment config** (this function) - overrides everything below
    3. Intelligent training config (pipeline/training/intelligent.yaml)
    4. Pipeline configs (pipeline/training/*.yaml)
    5. Defaults (defaults.yaml) - lowest priority
    
    **Fallback Behavior:**
    - If experiment config exists: values in it override intelligent_training_config
    - Missing values in experiment config fall back to intelligent_training_config
    - Missing values there fall back to defaults.yaml
    - If experiment config file doesn't exist: raises FileNotFoundError (no fallback)
    
    **SST Principle:**
    Experiment configs only need to specify values that differ from defaults.
    Missing values automatically fall back through the precedence chain.
    
    Args:
        exp_name: Experiment name (without .yaml extension)
        
    Returns:
        Dictionary with experiment configuration (loaded as-is, merging happens in intelligent_trainer.py)
        
    Raises:
        FileNotFoundError: If experiment config file doesn't exist (no fallback)
        
    Example:
        >>> config = load_experiment_config("e2e_full_targets_test")
        >>> data_dir = config.get("data", {}).get("data_dir")
    """
    exp_path = get_experiment_config_path(exp_name)
    if not exp_path.exists():
        raise FileNotFoundError(f"Experiment config not found: {exp_path}")
    
    try:
        with open(exp_path, 'r') as f:
            config = yaml.safe_load(f)
        if config is None:
            logger.warning(f"Experiment config {exp_path} is empty or invalid YAML, using empty config")
            config = {}
        
        # Experiment configs are top-level - they override everything
        # They only need to specify values that differ from defaults/intelligent_training_config
        # Missing values will fall back to intelligent_training_config, then defaults
        # (This happens in intelligent_trainer.py, not here)
        
        # Note: We don't inject defaults here because experiment configs should be
        # loaded as-is. The merging with defaults/intelligent_training_config happens
        # in intelligent_trainer.py where the precedence is enforced.
        
        logger.debug(f"Loaded experiment config: {exp_name} from {exp_path}")
        return config
    except Exception as e:
        logger.error(f"Failed to load experiment config {exp_path}: {e}")
        raise


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

