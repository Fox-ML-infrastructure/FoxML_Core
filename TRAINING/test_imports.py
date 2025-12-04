#!/usr/bin/env python3

"""
Copyright (c) 2025 Fox ML Infrastructure

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
Import Verification Script for TRAINING Module

Tests all critical imports to ensure the module structure is correct.
Run this after moving to a new workspace or making structural changes.
"""


import sys
from pathlib import Path

# Add paths
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_TRAINING_ROOT = Path(__file__).resolve().parent

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(_TRAINING_ROOT))

# Test results
passed = 0
failed = 0
warnings = 0

def test_import(module_path, class_or_func, description):
    """Test a single import"""
    global passed, failed, warnings
    try:
        module = __import__(module_path, fromlist=[class_or_func])
        getattr(module, class_or_func)
        print(f"✅ {description:50s} PASS")
        passed += 1
        return True
    except ImportError as e:
        print(f"❌ {description:50s} FAIL: {e}")
        failed += 1
        return False
    except SyntaxError as e:
        print(f"⚠️  {description:50s} SYNTAX ERROR: {e}")
        warnings += 1
        return False
    except Exception as e:
        print(f"⚠️  {description:50s} WARN: {e}")
        warnings += 1
        return False

print("=" * 80)
print("TRAINING Module Import Verification")
print("=" * 80)
print()

# Test path resolution
print("📁 Path Resolution:")
print(f"   PROJECT_ROOT:  {_PROJECT_ROOT}")
print(f"   TRAINING_ROOT: {_TRAINING_ROOT}")
print(f"   Expected PROJECT_ROOT:  /home/Jennifer/trader")
print(f"   Expected TRAINING_ROOT: /home/Jennifer/trader/TRAINING")
print()

# Test core utilities
print("🔧 Core Utilities (common/):")
test_import("common.safety", "set_global_numeric_guards", "common.safety")
test_import("common.threads", "temp_environ", "common.threads")
test_import("common.isolation_runner", "child_isolated", "common.isolation_runner")
test_import("common.tf_runtime", "ensure_tf_initialized", "common.tf_runtime")
test_import("common.determinism", "set_seed", "common.determinism")
print()

# Test model trainers (CPU-only, safe to import)
print("🤖 Model Trainers (model_fun/ - CPU only):")
test_import("model_fun.lightgbm_trainer", "LightGBMTrainer", "model_fun.lightgbm_trainer")
test_import("model_fun.xgboost_trainer", "XGBoostTrainer", "model_fun.xgboost_trainer")
test_import("model_fun.ensemble_trainer", "EnsembleTrainer", "model_fun.ensemble_trainer")
test_import("model_fun.quantile_lightgbm_trainer", "QuantileLightGBMTrainer", "model_fun.quantile_lightgbm_trainer")
test_import("model_fun.ngboost_trainer", "NGBoostTrainer", "model_fun.ngboost_trainer")
test_import("model_fun.gmm_regime_trainer", "GMMRegimeTrainer", "model_fun.gmm_regime_trainer")
test_import("model_fun.reward_based_trainer", "RewardBasedTrainer", "model_fun.reward_based_trainer")
test_import("model_fun.base_trainer", "BaseModelTrainer", "model_fun.base_trainer")
print()

# Test strategies
print("📊 Training Strategies (strategies/):")
test_import("strategies.base", "BaseTrainingStrategy", "strategies.base")
test_import("strategies.single_task", "SingleTaskStrategy", "strategies.single_task")
test_import("strategies.multi_task", "MultiTaskStrategy", "strategies.multi_task")
test_import("strategies.cascade", "CascadeStrategy", "strategies.cascade")
print()

# Test utilities
print("🛠️  Utilities (utils/):")
test_import("utils.feature_selection", "select_top_features", "utils.feature_selection")
test_import("utils.core_utils", "safe_column_filter", "utils.core_utils")
test_import("utils.validation", "validate_data", "utils.validation")
test_import("utils.target_resolver", "TargetResolver", "utils.target_resolver")
print()

# Test data processing
print("📦 Data Processing (data_processing/):")
test_import("data_processing.data_loader", "load_data", "data_processing.data_loader")
test_import("data_processing.data_utils", "check_data_quality", "data_processing.data_utils")
print()

# Test models
print("🎯 Model Registry (models/):")
test_import("models.registry", "MODEL_REGISTRY", "models.registry")
test_import("models.factory", "create_model", "models.factory")
test_import("models.family_router", "route_to_family", "models.family_router")
print()

# Test preprocessing
print("🔄 Preprocessing (preprocessing/):")
test_import("preprocessing.mega_script_data_preprocessor", "MegaScriptDataPreprocessor", "preprocessing.mega_script_data_preprocessor")
print()

# Test processing
print("⚡ Advanced Processing (processing/):")
test_import("processing.cross_sectional", "CrossSectionalProcessor", "processing.cross_sectional")
test_import("processing.polars_optimizer", "optimize_polars", "processing.polars_optimizer")
print()

# Test __init__.py files exist
print("📄 __init__.py Files:")
init_files = [
    _TRAINING_ROOT / "__init__.py",
    _TRAINING_ROOT / "common" / "__init__.py",
    _TRAINING_ROOT / "model_fun" / "__init__.py",
    _TRAINING_ROOT / "strategies" / "__init__.py",
    _TRAINING_ROOT / "utils" / "__init__.py",
    _TRAINING_ROOT / "data_processing" / "__init__.py",
    _TRAINING_ROOT / "models" / "__init__.py",
    _TRAINING_ROOT / "core" / "__init__.py",
    _TRAINING_ROOT / "datasets" / "__init__.py",
    _TRAINING_ROOT / "features" / "__init__.py",
    _TRAINING_ROOT / "live" / "__init__.py",
    _TRAINING_ROOT / "memory" / "__init__.py",
    _TRAINING_ROOT / "preprocessing" / "__init__.py",
    _TRAINING_ROOT / "processing" / "__init__.py",
    _TRAINING_ROOT / "tests" / "__init__.py",
    _TRAINING_ROOT / "tools" / "__init__.py",
    _TRAINING_ROOT / "examples" / "__init__.py",
]

for init_file in init_files:
    rel_path = str(init_file.relative_to(_TRAINING_ROOT))
    if init_file.exists():
        print(f"✅ {rel_path:40s} EXISTS")
        passed += 1
    else:
        print(f"❌ {rel_path:40s} MISSING")
        failed += 1

print()

# Summary
print("=" * 80)
print("📊 Summary:")
print(f"   ✅ Passed:   {passed}")
print(f"   ❌ Failed:   {failed}")
print(f"   ⚠️  Warnings: {warnings}")
print("=" * 80)

if failed == 0:
    print("🎉 ALL TESTS PASSED! Module structure is correct.")
    sys.exit(0)
else:
    print("⚠️  SOME TESTS FAILED. Check the errors above.")
    sys.exit(1)

