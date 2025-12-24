# MIT License - see LICENSE file

"""
Training Strategies Module

Implements different training approaches:
- Single-task: Separate models per target
- Multi-task: Shared encoder + separate heads
- Cascade: Stacking/gating approach
"""


from .single_task import SingleTaskStrategy
from .multi_task import MultiTaskStrategy  
from .cascade import CascadeStrategy
from .base import BaseTrainingStrategy

__all__ = [
    'BaseTrainingStrategy',
    'SingleTaskStrategy',
    'MultiTaskStrategy', 
    'CascadeStrategy'
]
