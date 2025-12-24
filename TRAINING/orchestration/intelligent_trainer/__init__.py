# MIT License - see LICENSE file

"""
Intelligent Trainer Module

Modular components for intelligent training orchestrator.
"""

from .utils import json_default, get_sample_size_bin

# Re-export IntelligentTrainer and main from the sibling module file
# This is needed because Python imports the package dir over the .py file
import importlib.util
import sys
from pathlib import Path

# Load the intelligent_trainer.py module file (sibling to this package)
_module_path = Path(__file__).parent.parent / "intelligent_trainer.py"
if _module_path.exists():
    _spec = importlib.util.spec_from_file_location("intelligent_trainer_module", _module_path)
    _module = importlib.util.module_from_spec(_spec)
    sys.modules["TRAINING.orchestration.intelligent_trainer_module"] = _module
    _spec.loader.exec_module(_module)
    
    # Re-export the main class and function
    IntelligentTrainer = _module.IntelligentTrainer
    main = _module.main
else:
    # Fallback: class not available
    IntelligentTrainer = None
    main = None

__all__ = [
    'json_default',
    'get_sample_size_bin',
    'IntelligentTrainer',
    'main',
]

