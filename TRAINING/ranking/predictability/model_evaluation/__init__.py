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
Model Evaluation Module

Modular components for model evaluation and target predictability ranking.
"""

from .config_helpers import get_importance_top_fraction
from .leakage_helpers import compute_suspicion_score
# detect_leakage is imported from leakage_detection.py, not leakage_helpers.py
# (leakage_helpers.detect_leakage has old signature without X, y, time_vals, symbols parameters)
from TRAINING.ranking.predictability.leakage_detection import detect_leakage
from .reporting import log_canonical_summary, save_feature_importances, log_suspicious_features

# Import from parent file (functions that weren't extracted yet)
# These are still in model_evaluation.py (parent file, not the folder)
import sys
from pathlib import Path
_parent_file = Path(__file__).parent.parent / "model_evaluation.py"
if _parent_file.exists():
    import importlib.util
    spec = importlib.util.spec_from_file_location("model_evaluation_main", _parent_file)
    model_evaluation_main = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_evaluation_main)
    
    train_and_evaluate_models = model_evaluation_main.train_and_evaluate_models
    evaluate_target_predictability = model_evaluation_main.evaluate_target_predictability
    _enforce_final_safety_gate = model_evaluation_main._enforce_final_safety_gate
    # validate_target might not exist, check first
    validate_target = getattr(model_evaluation_main, 'validate_target', None)
else:
    raise ImportError(f"Could not find model_evaluation.py at {_parent_file}")

__all__ = [
    'get_importance_top_fraction',
    'compute_suspicion_score',
    'detect_leakage',
    'log_canonical_summary',
    'save_feature_importances',
    'log_suspicious_features',
    # Functions still in main file
    'train_and_evaluate_models',
    'evaluate_target_predictability',
    '_enforce_final_safety_gate',
    'validate_target',
]

