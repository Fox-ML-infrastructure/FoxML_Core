# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
# Copyright (c) 2025-2026 Fox ML Infrastructure LLC

"""
Processing Pipeline Module

Provides end-to-end data processing pipelines:
- barrier_pipeline: Smart barrier target processing with resumability
- normalize: Session normalization and RTH grid enforcement
"""


from .normalize import normalize_interval, assert_bars_per_day

__all__ = [
    "normalize_interval",
    "assert_bars_per_day",
]

