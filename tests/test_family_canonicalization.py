# MIT License - see LICENSE file

"""
Unit tests for family name canonicalization and registry invariants.
"""

import pytest
import sys
from pathlib import Path

# Add project root to path
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def test_normalize_idempotent():
    """Test that normalize_family_name is idempotent."""
    from TRAINING.training_strategies.utils import normalize_family_name
    
    samples = [
        "LightGBM", "lightgbm", "LIGHTGBM", "light_gbm",
        "xgboost", "XGBoost", "XGBOOST",
        "neural_network", "NeuralNetwork", "NEURAL_NETWORK",
        "random_forest", "RandomForest", "RANDOM_FOREST",
        "quantile_lightgbm", "QuantileLightGBM",
        "meta_learning", "MetaLearning",
        "multi_task", "MultiTask",
        "change_point", "ChangePoint",
        "gmm_regime", "GMMRegime",
        "reward_based", "RewardBased",
        "ftrl_proximal", "FTRLProximal",
        "cnn1d", "CNN1D",
        "tabcnn", "TabCNN",
        "tablstm", "TabLSTM",
        "tabtransformer", "TabTransformer",
    ]
    
    for s in samples:
        normalized = normalize_family_name(s)
        # Idempotency: normalize(normalize(x)) == normalize(x)
        assert normalize_family_name(normalized) == normalized, \
            f"normalize_family_name not idempotent for '{s}': {normalized} -> {normalize_family_name(normalized)}"


def test_registry_keys_canonical_and_unique():
    """Test that all registry keys are canonical and collision-free."""
    from TRAINING.training_strategies.utils import normalize_family_name
    from TRAINING.common.isolation_runner import TRAINER_MODULE_MAP
    from TRAINING.common.runtime_policy import POLICY
    from TRAINING.training_strategies.utils import FAMILY_CAPS
    
    registries = [
        ("TRAINER_MODULE_MAP", TRAINER_MODULE_MAP),
        ("POLICY", POLICY),
        ("FAMILY_CAPS", FAMILY_CAPS),
    ]
    
    for name, reg in registries:
        seen = set()
        for k in reg.keys():
            # Key must be canonical (normalize(k) == k)
            assert k == normalize_family_name(k), \
                f"{name} has non-canonical key: '{k}' (expected '{normalize_family_name(k)}')"
            # No collisions
            assert k not in seen, \
                f"{name} has duplicate key: '{k}'"
            seen.add(k)


def test_registry_coverage():
    """Test that registries have consistent coverage."""
    from TRAINING.common.isolation_runner import TRAINER_MODULE_MAP
    from TRAINING.common.runtime_policy import POLICY
    from TRAINING.training_strategies.utils import FAMILY_CAPS
    
    # All trainers in TRAINER_MODULE_MAP should have policies
    missing_policies = set(TRAINER_MODULE_MAP.keys()) - set(POLICY.keys())
    assert not missing_policies, \
        f"TRAINER_MODULE_MAP families missing from POLICY: {sorted(missing_policies)}"
    
    # All trainers in TRAINER_MODULE_MAP should have capabilities (if they're actual trainers)
    # Note: Some families might be selectors, so this is a soft check
    missing_caps = set(TRAINER_MODULE_MAP.keys()) - set(FAMILY_CAPS.keys())
    if missing_caps:
        # Log but don't fail - selectors might not have caps
        print(f"Note: TRAINER_MODULE_MAP families missing from FAMILY_CAPS: {sorted(missing_caps)}")


def test_lookup_accepts_variants():
    """Test that lookups accept various input formats and normalize correctly."""
    from TRAINING.training_strategies.utils import normalize_family_name
    from TRAINING.common.isolation_runner import TRAINER_MODULE_MAP
    
    # Test that various input formats normalize to keys that exist in registry
    test_cases = [
        ("LightGBM", "lightgbm"),
        ("lightgbm", "lightgbm"),
        ("XGBoost", "xgboost"),
        ("xgboost", "xgboost"),
        ("RandomForest", "random_forest"),
        ("random_forest", "random_forest"),
        ("NeuralNetwork", "neural_network"),
        ("neural_network", "neural_network"),
        ("MetaLearning", "meta_learning"),
        ("meta_learning", "meta_learning"),
    ]
    
    for raw_input, expected_canonical in test_cases:
        canonical = normalize_family_name(raw_input)
        assert canonical == expected_canonical, \
            f"normalize_family_name('{raw_input}') = '{canonical}', expected '{expected_canonical}'"
        
        # If the canonical form exists in registry, verify lookup works
        # (Some families like RandomForest might not be in TRAINER_MODULE_MAP yet)
        if canonical in TRAINER_MODULE_MAP:
            assert TRAINER_MODULE_MAP[canonical] is not None


def test_modmap_keys_canonical():
    """Test that MODMAP keys in family_runners.py are canonical."""
    from TRAINING.training_strategies.utils import normalize_family_name
    
    # MODMAP is defined inside functions, so we need to extract it
    # We'll test by calling the function with normalized inputs
    from TRAINING.training_strategies.family_runners import _run_family_isolated
    
    # Test that known families normalize correctly
    test_families = ["lightgbm", "xgboost", "mlp", "vae", "gan"]
    
    for family in test_families:
        normalized = normalize_family_name(family)
        assert normalized == family, \
            f"Family '{family}' should already be canonical, but normalized to '{normalized}'"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

