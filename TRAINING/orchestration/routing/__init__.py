# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial

"""
Target Routing Module

Routes target columns to training specifications.
"""

from .target_router import TaskSpec, route_target

__all__ = ['TaskSpec', 'route_target']

