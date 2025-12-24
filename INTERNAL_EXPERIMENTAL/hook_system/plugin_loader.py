"""
Plugin Loader

Explicit plugin loading for hook registration.

This provides controlled, config-driven plugin loading instead of
relying on implicit imports that may happen at unpredictable times.

Usage:
    from TRAINING.common.plugin_loader import load_plugins_from_config
    
    # Load plugins from config
    load_plugins_from_config()
    
    # Or load explicitly
    from TRAINING.common.pipeline_hooks import load_plugins
    load_plugins(['TRAINING.common.my_feature'])
"""

import logging
from typing import List, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


def load_plugins_from_config(config_path: Optional[Path] = None) -> dict:
    """
    Load plugins from config file.
    
    Looks for config key: `pipeline.enabled_plugins` (list of module names)
    
    Args:
        config_path: Optional path to config file (defaults to CONFIG/base.yaml)
    
    Returns:
        Dict with 'loaded' and 'failed' lists
    """
    try:
        from CONFIG.config_loader import get_cfg
        
        # Try to get plugin list from config
        plugins = get_cfg(
            "pipeline.enabled_plugins",
            default=[],
            config_name="base"
        )
        
        if not plugins:
            logger.debug("No plugins configured in pipeline.enabled_plugins")
            return {'loaded': [], 'failed': []}
        
        if not isinstance(plugins, list):
            logger.warning(f"pipeline.enabled_plugins must be a list, got {type(plugins)}")
            return {'loaded': [], 'failed': []}
        
        # CRITICAL: Preserve config order (YAML lists are ordered, so this is deterministic)
        # Config order may express plugin precedence, so we preserve it rather than sorting
        # Optional: warn if list is not sorted (helps catch accidental misordering)
        if plugins != sorted(plugins):
            logger.debug(
                f"Plugin list is not sorted alphabetically: {plugins}. "
                f"Preserving config order (this may be intentional for precedence)."
            )
        
        # Load plugins in config order (preserves intended precedence)
        from TRAINING.common.pipeline_hooks import load_plugins
        return load_plugins(plugins, fail_on_error=False)
        
    except ImportError:
        logger.debug("Config loader not available, skipping config-driven plugin loading")
        return {'loaded': [], 'failed': []}
    except Exception as e:
        logger.warning(f"Failed to load plugins from config: {e}")
        return {'loaded': [], 'failed': []}


def load_default_plugins() -> dict:
    """
    Load default plugins (if any).
    
    This is a convenience function for common plugins that should
    always be loaded if available.
    
    Returns:
        Dict with 'loaded' and 'failed' lists
    """
    default_plugins = [
        # Add default plugins here as they're created
        # Example: 'TRAINING.common.leakage_detection',
    ]
    
    if not default_plugins:
        return {'loaded': [], 'failed': []}
    
    from TRAINING.common.pipeline_hooks import load_plugins
    return load_plugins(default_plugins, fail_on_error=False)
