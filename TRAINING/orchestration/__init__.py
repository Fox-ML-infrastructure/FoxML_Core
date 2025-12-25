# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial

"""
Training Orchestration Module

Provides intelligent training pipeline with integrated target ranking
and feature selection, while preserving all existing functionality.
"""

# Lazy imports to avoid RuntimeWarning when running as module
# This prevents the warning: 'TRAINING.orchestration.intelligent_trainer' found in sys.modules
# when running: python -m TRAINING.orchestration.intelligent_trainer
#
# The warning occurs because:
# 1. __init__.py imports intelligent_trainer
# 2. When running as module, Python imports the package first
# 3. This triggers __init__.py which imports intelligent_trainer
# 4. Then Python tries to run intelligent_trainer as a script, but it's already in sys.modules
#
# Solution: Use lazy imports via __getattr__ (Python 3.7+)

def _lazy_import():
    """Lazy import to avoid circular import issues when running as module."""
    from .intelligent_trainer import IntelligentTrainer, main
    return IntelligentTrainer, main

# Provide lazy access via __getattr__ (Python 3.7+)
# This defers the import until the attribute is actually accessed
def __getattr__(name):
    if name in ('IntelligentTrainer', 'main'):
        IntelligentTrainer, main = _lazy_import()
        if name == 'IntelligentTrainer':
            return IntelligentTrainer
        elif name == 'main':
            return main
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

# For backward compatibility, also provide direct access (but lazy)
__all__ = ['IntelligentTrainer', 'main']

