# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
# Copyright (c) 2025-2026 Fox ML Infrastructure LLC

"""
Backward compatibility wrapper for TRAINING.target_router

This module has been moved to TRAINING.orchestration.routing.target_router
All imports are re-exported here to maintain backward compatibility.
"""

# Re-export everything from the new location
from TRAINING.orchestration.routing.target_router import *

__all__ = ['TaskSpec', 'route_target']

