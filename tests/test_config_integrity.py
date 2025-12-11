"""
Config Integrity Tests

Validates that all config files load correctly and match expected schemas.
Prevents silent failures from wrong key names or missing sections.
"""

import pytest
from pathlib import Path
from typing import Dict, Any

# Import config loaders
try:
    from CONFIG.config_loader import (
        get_safety_config,
        get_system_config,
        get_pipeline_config,
        load_training_config
    )
    from CONFIG.config_schemas import validate_safety_config
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False
    pytest.skip("Config loader not available", allow_module_level=True)


def test_safety_config_loads():
    """Test that safety_config.yaml loads and has correct structure"""
    cfg = get_safety_config()
    
    # Must have 'safety' top-level key
    assert 'safety' in cfg, f"safety_config missing 'safety' key. Keys: {list(cfg.keys())}"
    
    safety_section = cfg['safety']
    assert isinstance(safety_section, dict), f"safety_config.safety must be dict, got {type(safety_section)}"
    
    # Must have 'leakage_detection' section
    assert 'leakage_detection' in safety_section, \
        f"safety_config.safety missing 'leakage_detection'. Keys: {list(safety_section.keys())}"
    
    leakage = safety_section['leakage_detection']
    assert isinstance(leakage, dict), f"leakage_detection must be dict, got {type(leakage)}"
    
    # Validate critical keys exist
    required_keys = [
        'auto_fix_max_features_per_run',
        'auto_fix_min_confidence',
        'auto_fix_enabled'
    ]
    
    missing = [k for k in required_keys if k not in leakage]
    assert not missing, \
        f"safety_config.safety.leakage_detection missing keys: {missing}. Available: {list(leakage.keys())}"
    
    # Validate types
    assert isinstance(leakage['auto_fix_max_features_per_run'], int), \
        f"auto_fix_max_features_per_run must be int, got {type(leakage['auto_fix_max_features_per_run'])}"
    assert isinstance(leakage['auto_fix_min_confidence'], (int, float)), \
        f"auto_fix_min_confidence must be numeric, got {type(leakage['auto_fix_min_confidence'])}"
    assert isinstance(leakage['auto_fix_enabled'], bool), \
        f"auto_fix_enabled must be bool, got {type(leakage['auto_fix_enabled'])}"


def test_safety_config_schema_validation():
    """Test schema validation function"""
    cfg = get_safety_config()
    # Should not raise
    validate_safety_config(cfg)


def test_system_config_loads():
    """Test that system_config.yaml loads correctly"""
    cfg = get_system_config()
    assert isinstance(cfg, dict), f"system_config must be dict, got {type(cfg)}"
    
    # Must have 'system' top-level key
    assert 'system' in cfg, f"system_config missing 'system' key. Keys: {list(cfg.keys())}"


def test_pipeline_config_loads():
    """Test that pipeline_config.yaml loads correctly"""
    cfg = get_pipeline_config()
    assert isinstance(cfg, dict), f"pipeline_config must be dict, got {type(cfg)}"


def test_config_path_access():
    """Test that config paths work correctly (catches wrong nesting)"""
    from CONFIG.config_loader import get_cfg
    
    # Test safety config access with correct path
    max_features = get_cfg(
        "safety.leakage_detection.auto_fix_max_features_per_run",
        default=None,
        config_name="safety_config"
    )
    assert max_features is not None, "Could not access auto_fix_max_features_per_run via get_cfg"
    assert isinstance(max_features, int), f"max_features must be int, got {type(max_features)}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
