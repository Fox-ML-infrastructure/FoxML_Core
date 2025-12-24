# MIT License - see LICENSE file

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

