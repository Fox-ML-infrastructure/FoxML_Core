# MIT License - see LICENSE file

"""
Configuration Loading for Multi-Model Feature Selection

Functions for loading and managing multi-model feature selection configuration.
"""

import logging
import sys
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger(__name__)

# Add project root for _REPO_ROOT
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def load_multi_model_config(config_path: Path = None) -> Dict[str, Any]:
    """Load multi-model feature selection configuration
    
    Uses centralized config loader if available, otherwise falls back to manual path resolution.
    Checks new location first (CONFIG/ranking/features/multi_model.yaml),
    then old location (CONFIG/feature_selection/multi_model.yaml).
    """
    if config_path is None:
        # Try using centralized config loader first
        try:
            from CONFIG.config_loader import get_config_path
            config_path = get_config_path("feature_selection_multi_model")
            if config_path.exists():
                logger.debug(f"Using centralized config loader: {config_path}")
            else:
                # Fallback to manual resolution
                config_path = None
        except (ImportError, AttributeError):
            # Config loader not available, use manual resolution
            config_path = None
        
        if config_path is None:
            # Manual path resolution (fallback)
            # Try newest location first (ranking/features/)
            newest_path = _REPO_ROOT / "CONFIG" / "ranking" / "features" / "multi_model.yaml"
            # Then old location (feature_selection/)
            old_path = _REPO_ROOT / "CONFIG" / "feature_selection" / "multi_model.yaml"
            
            if newest_path.exists():
                config_path = newest_path
                logger.debug(f"Using new config location: {config_path}")
            elif old_path.exists():
                config_path = old_path
                logger.debug(f"Using old config location: {config_path} (consider migrating to ranking/features/)")
            else:
                logger.warning(f"Config not found in new ({newest_path}) or old ({old_path}) locations, using defaults")
                return get_default_config()
    
    if not config_path.exists():
        logger.warning(f"Config not found: {config_path}, using defaults")
        return get_default_config()
    
    try:
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded multi-model config from: {config_path}")
        return config
    except Exception as e:
        logger.error(f"Failed to load config from {config_path}: {e}, using defaults")
        return get_default_config()


def get_default_config() -> Dict[str, Any]:
    """Get default multi-model feature selection configuration."""
    return {
        'model_families': {
            'lightgbm': {
                'enabled': True,
                'importance_method': 'native',
                'weight': 1.0,
                'config': {}
            },
            'xgboost': {
                'enabled': True,
                'importance_method': 'native',
                'weight': 1.0,
                'config': {}
            },
            'random_forest': {
                'enabled': True,
                'importance_method': 'native',
                'weight': 1.0,
                'config': {}
            }
        },
        'aggregation': {
            'method': 'mean',  # 'mean', 'median', 'max', 'weighted_mean'
            'top_fraction': 0.10,  # Top 10% of features
            'min_consensus': 2,  # Minimum number of models that must agree
            'fallback': {
                'enabled': True,
                'method': 'uniform',
                'threshold': 0.0
            }
        },
        'parallel': {
            'enabled': True,
            'max_workers': None  # Auto-detect
        }
    }

