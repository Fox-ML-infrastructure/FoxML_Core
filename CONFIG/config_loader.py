"""
Copyright (c) 2025 Jennifer Lewis

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
    
    if not config_file.exists():
        logger.warning(f"Config file not found: {config_file}, using empty config")
        return {}
    
    # Load YAML
    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Failed to load config {config_file}: {e}")
        return {}
    
    # Start with default hyperparameters
    result = config.get("hyperparameters", {}).copy()
    
    # Apply variant if specified
    if variant and "variants" in config:
        if variant in config["variants"]:
            variant_config = config["variants"][variant]
            result.update(variant_config)
            logger.info(f"Applied variant '{variant}' for {model_family}")
        else:
            logger.warning(f"Variant '{variant}' not found for {model_family}, using defaults")
    
    # Apply overrides
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

