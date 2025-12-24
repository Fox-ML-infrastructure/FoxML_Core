# MIT License - see LICENSE file

"""
Backward compatibility wrapper for TRAINING.unified_training_interface

This module has been moved to TRAINING.orchestration.interfaces.unified_training_interface
All imports are re-exported here to maintain backward compatibility.
"""

# Re-export everything from the new location
from TRAINING.orchestration.interfaces.unified_training_interface import *

__all__ = ['UnifiedTrainingInterface']
