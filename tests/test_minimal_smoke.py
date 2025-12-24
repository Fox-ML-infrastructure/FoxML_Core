"""
Minimal Smoke Test (runs without pytest)

Quick verification that core modules can be imported and basic functionality works.
"""
import sys
from pathlib import Path

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def test_config_imports():
    """Test config modules can be imported"""
    from CONFIG.config_loader import get_safety_config, get_system_config
    from CONFIG.config_schemas import validate_safety_config, ExperimentConfig
    assert True


def test_config_validation():
    """Test config validation works"""
    from CONFIG.config_loader import get_safety_config
    from CONFIG.config_schemas import validate_safety_config
    
    cfg = get_safety_config()
    # Should not raise (config should be valid)
    validate_safety_config(cfg, strict=False)
    assert True


def test_strict_mode():
    """Test strict mode module works"""
    from TRAINING.common.strict_mode import strict_assert, STRICT_MODE
    # Should not raise
    strict_assert(True, "This should not fail")
    assert isinstance(STRICT_MODE, bool)


if __name__ == "__main__":
    print("Running minimal smoke tests...")
    test_config_imports()
    print("✅ Config imports")
    test_config_validation()
    print("✅ Config validation")
    test_strict_mode()
    print("✅ Strict mode")
    print("\n✅ All smoke tests passed!")
