# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
# Copyright (c) 2025-2026 Fox ML Infrastructure LLC

"""
Feature Engineering Module

Provides comprehensive feature builders for market data:
- ComprehensiveFeatureBuilder: 200+ features for ranking pipeline
- SimpleFeatureComputer: Basic feature computation
- StreamingFeatureBuilder: Memory-efficient streaming processing with Polars
"""


from .comprehensive_builder import ComprehensiveFeatureBuilder
from .simple_features import SimpleFeatureComputer

__all__ = [
    "ComprehensiveFeatureBuilder",
    "SimpleFeatureComputer",
]

