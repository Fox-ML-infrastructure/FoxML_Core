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
Multi-Model Feature Selection Module

Modular components for multi-model feature selection pipeline.
"""

from .types import ModelFamilyConfig, ImportanceResult
from .config_loader import load_multi_model_config, get_default_config
from .importance_extractors import (
    safe_load_dataframe,
    extract_native_importance,
    extract_shap_importance,
    extract_permutation_importance
)

# Import from parent file (functions that weren't extracted yet)
# These are still in multi_model_feature_selection.py (parent file, not the folder)
import sys
from pathlib import Path
_parent_file = Path(__file__).parent.parent / "multi_model_feature_selection.py"
if _parent_file.exists():
    import importlib.util
    spec = importlib.util.spec_from_file_location("multi_model_feature_selection_main", _parent_file)
    multi_model_main = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(multi_model_main)
    
    process_single_symbol = multi_model_main.process_single_symbol
    aggregate_multi_model_importance = multi_model_main.aggregate_multi_model_importance
    save_multi_model_results = multi_model_main.save_multi_model_results
else:
    raise ImportError(f"Could not find multi_model_feature_selection.py at {_parent_file}")

__all__ = [
    'ModelFamilyConfig',
    'ImportanceResult',
    'load_multi_model_config',
    'get_default_config',
    'safe_load_dataframe',
    'extract_native_importance',
    'extract_shap_importance',
    'extract_permutation_importance',
    # Functions still in main file
    'process_single_symbol',
    'aggregate_multi_model_importance',
    'save_multi_model_results',
]

