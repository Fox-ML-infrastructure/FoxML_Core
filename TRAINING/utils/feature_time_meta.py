"""
Feature Time Semantics Metadata

Defines per-feature time semantics for multi-interval + embargo-aware feature pipeline.
This enables features with different native intervals (5m, 15m, 1d) to be safely combined
without lookahead bias.

Key concept: Features are aligned by "availability timestamp" (feature_ts + embargo),
not by feature timestamp. This prevents leakage when mixing intervals.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class FeatureTimeMeta:
    """
    Time semantics metadata for a single feature.
    
    This answers:
    - What timeline is it sampled on? (native_interval_minutes)
    - When is it available? (embargo_minutes, publish_offset_minutes)
    - How far back does it reach? (lookback_bars or lookback_minutes)
    - How to align it onto base grid? (as-of join + staleness cap)
    
    Attributes:
        name: Feature name
        native_interval_minutes: Native sampling interval (None => defaults to base_interval)
        embargo_minutes: Availability delay (when feature becomes known after bar close)
        lookback_bars: Lookback period in bars (if period is bar-count, e.g., rsi_21 = 21 bars)
        lookback_minutes: Lookback period in minutes (if period is wall-clock time, e.g., rsi_105m)
        max_staleness_minutes: Cap on forward-fill (None = no cap, use latest available)
        publish_offset_minutes: Offset if timestamps are bar-start not bar-end (default: 0)
    
    Note: Either lookback_bars OR lookback_minutes should be set, not both.
    If both are None, lookback will be inferred from feature name (fallback).
    """
    name: str
    native_interval_minutes: Optional[float] = None  # None => defaults to base_interval
    embargo_minutes: float = 0.0  # Availability delay
    lookback_bars: Optional[int] = None  # If period is bar-count
    lookback_minutes: Optional[float] = None  # If period is wall-clock time
    max_staleness_minutes: Optional[float] = None  # Cap forward-fill
    publish_offset_minutes: float = 0.0  # If timestamps are bar-start not bar-end
    
    def __post_init__(self):
        """Validate that lookback_bars and lookback_minutes are not both set."""
        if self.lookback_bars is not None and self.lookback_minutes is not None:
            raise ValueError(
                f"FeatureTimeMeta for {self.name}: Cannot set both lookback_bars and lookback_minutes. "
                f"Use lookback_bars for bar-count periods (e.g., rsi_21), "
                f"lookback_minutes for wall-clock periods (e.g., rsi_105m)."
            )


def effective_lookback_minutes(
    meta: FeatureTimeMeta,
    base_interval_minutes: float,
    inferred_lookback_minutes: Optional[float] = None
) -> float:
    """
    Compute effective lookback for leakage safety (lookback + embargo).
    
    For leakage safety around a split boundary, the relevant span is:
    effective_lookback_minutes = lookback_minutes + embargo_minutes
    
    Reason: At time t, the newest raw data used by that feature is ≤ t - embargo.
    The window extends further back by lookback.
    
    Args:
        meta: FeatureTimeMeta for the feature
        base_interval_minutes: Base training grid interval (for defaulting native_interval)
        inferred_lookback_minutes: Optional inferred lookback from name (if meta doesn't have explicit lookback)
    
    Returns:
        Effective lookback in minutes (lookback + embargo)
    """
    # Use native interval or default to base
    interval = meta.native_interval_minutes or base_interval_minutes
    
    # Compute lookback in minutes
    if meta.lookback_minutes is not None:
        lookback = meta.lookback_minutes
    elif meta.lookback_bars is not None:
        lookback = meta.lookback_bars * interval
    elif inferred_lookback_minutes is not None:
        # Use inferred lookback from name (fallback)
        lookback = inferred_lookback_minutes
    else:
        # Fallback: conservative default (should not happen in practice if metadata is complete)
        lookback = 1440.0  # 1 day conservative default
    
    # Effective lookback includes embargo
    return lookback + meta.embargo_minutes
