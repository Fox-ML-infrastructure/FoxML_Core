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
