"""
Smoke Test: Import Verification

Quick test to ensure all critical modules can be imported without errors.
Catches syntax errors, import failures, and missing dependencies early.
"""

import pytest
import sys
from pathlib import Path

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def test_import_config_modules():
    """Test that config modules can be imported"""
    from CONFIG.config_loader import (
        get_safety_config,
        get_system_config,
        get_pipeline_config,
        get_cfg
    )
    from CONFIG.config_schemas import (
        ExperimentConfig,
        FeatureSelectionConfig,
        TargetRankingConfig,
        TrainingConfig,
        validate_safety_config
    )
    assert True  # If we get here, imports worked


def test_import_training_modules():
    """Test that core training modules can be imported"""
    # These might fail if dependencies missing, but should not fail on syntax errors
    try:
        from TRAINING.common.strict_mode import strict_assert, STRICT_MODE
        from TRAINING.common.safety import set_global_numeric_guards
        assert True
    except ImportError as e:
        # Missing dependencies are OK for smoke test
        # Syntax errors are NOT OK
        if "No module named" in str(e):
            pytest.skip(f"Optional dependency missing: {e}")
        else:
            raise  # Re-raise if it's a syntax/import structure error


def test_import_ranking_modules():
    """Test that ranking modules can be imported"""
    try:
        # These might have optional dependencies
        from TRAINING.ranking.target_ranker import rank_targets
        assert True
    except ImportError as e:
        if "No module named" in str(e):
            pytest.skip(f"Optional dependency missing: {e}")
        else:
            raise


def test_config_validation_smoke():
    """Test that config validation works"""
    from CONFIG.config_loader import get_safety_config
    from CONFIG.config_schemas import validate_safety_config
    
    cfg = get_safety_config()
    # Should not raise (config should be valid)
    validate_safety_config(cfg, strict=False)
    assert True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
