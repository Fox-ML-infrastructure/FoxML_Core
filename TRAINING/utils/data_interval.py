"""
Copyright (c) 2025-2026 Fox ML Infrastructure LLC

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
Utility for auto-detecting data bar interval from timestamps

CRITICAL: Using wrong interval causes data leakage in PurgedTimeSeriesSplit.
For example, if data is 1-minute bars but code assumes 5-minute bars:
- 60m target horizon = 60 bars (correct)
- But code calculates: 60m / 5m = 12 bars (WRONG - leaks 48 minutes!)
"""


import pandas as pd
import numpy as np
from typing import Optional, Union, List, Any
import logging
import re

logger = logging.getLogger(__name__)


def normalize_interval(interval: Union[str, int]) -> int:
    """
    Normalize interval to minutes (canonical internal representation).
    
    Accepts:
    - String: "5m", "15m", "1h", "300s" (normalized to minutes)
    - Integer: 5 (assumed to be minutes)
    
    Args:
        interval: Interval as string or int
    
    Returns:
        Interval in minutes (int)
    
    Raises:
        ValueError: If interval format is invalid
    """
    if interval is None:
        raise ValueError("Interval cannot be None")
    
    # Integer: assume minutes
    if isinstance(interval, int):
        if interval < 1:
            raise ValueError(f"Interval must be >= 1 minute, got {interval}")
        return interval
    
    # String: parse
    if not isinstance(interval, str):
        raise ValueError(f"Interval must be str or int, got {type(interval)}")
    
    interval_str = interval.lower().strip()
    if not interval_str:
        raise ValueError("Interval string cannot be empty")
    
    # Try patterns: "5m", "15m", "1h", "300s"
    # Pattern 1: "5m", "15m", "1h"
    match = re.match(r'^(\d+)([mh])$', interval_str)
    if match:
        value = int(match.group(1))
        unit = match.group(2)
        if unit == 'h':
            return value * 60
        elif unit == 'm':
            return value
    
    # Pattern 2: "300s", "60s" (seconds)
    match = re.match(r'^(\d+)s$', interval_str)
    if match:
        seconds = int(match.group(1))
        minutes = seconds / 60.0
        if minutes.is_integer():
            return int(minutes)
        else:
            raise ValueError(f"Interval {interval} (={seconds}s) does not convert to whole minutes")
    
    raise ValueError(f"Invalid interval format: {interval}. Expected format: '5m', '15m', '1h', '300s', or integer")


def _parse_interval_string(interval_str: str) -> Optional[int]:
    """
    Parse interval string like "5m", "15m", "1h" to minutes.
    
    DEPRECATED: Use normalize_interval() instead.
    
    Args:
        interval_str: String like "5m", "15m", "1h", "30m"
    
    Returns:
        Interval in minutes, or None if parsing fails
    """
    try:
        return normalize_interval(interval_str)
    except ValueError:
        return None


def _detect_timestamp_unit(delta: float) -> Optional[tuple]:
    """
    Detect timestamp unit by trying different conversions.
    
    Args:
        delta: Raw timestamp difference (numeric)
    
    Returns:
        Tuple of (unit_name, minutes) if detected, None otherwise
    """
    # Try different units: ns, us, ms, s
    # For each, convert to minutes and check if it's a reasonable bar interval
    candidates = [
        ("ns", 1e9),
        ("us", 1e6),
        ("ms", 1e3),
        ("s", 1.0),
    ]
    
    for unit_name, per_second in candidates:
        minutes = delta / (60.0 * per_second)
        
        # Check if it's a reasonable bar interval (0.01 minutes to 1 day)
        if 0.01 <= minutes <= 1440:
            # Check if it's close to a round number (within 1% tolerance)
            rounded = round(minutes)
            if abs(minutes - rounded) / max(rounded, 0.01) < 0.01:
                return (unit_name, rounded)
    
    return None


def detect_interval_from_timestamps(
    timestamps: Union[pd.Series, np.ndarray, List],
    default: Optional[int] = 5,
    explicit_interval: Optional[Union[int, str]] = None
) -> Optional[int]:
    """
    Auto-detect data bar interval (in minutes) from timestamp differences.
    
    Args:
        timestamps: Series, array, or list of timestamps (datetime-like)
        default: Default interval to use if detection fails (default: 5 minutes)
        explicit_interval: If provided, use this interval and skip auto-detection.
                          Can be int (minutes) or str like "5m", "15m", "1h"
    
    Returns:
        Detected interval in minutes (rounded to common intervals: 1, 5, 15, 30, 60)
    """
    # If explicit interval is set, use it (called from detect_interval_from_dataframe with precedence)
    if explicit_interval is not None:
        try:
            minutes = normalize_interval(explicit_interval)
            logger.info(f"Using explicit interval: {explicit_interval} = {minutes}m")
            return minutes
        except ValueError as e:
            logger.warning(f"Failed to parse explicit interval '{explicit_interval}': {e}, falling back to auto-detect")
    
    if timestamps is None or len(timestamps) < 2:
        logger.warning(f"Insufficient timestamps for interval detection, using default: {default}m")
        return default
    
    try:
        # Convert to pandas Series if needed
        if not isinstance(timestamps, pd.Series):
            if isinstance(timestamps[0], (int, float)):
                # Don't assume unit yet - we'll detect it
                time_series = pd.to_datetime(timestamps, unit='ns')
            else:
                time_series = pd.Series(timestamps)
        else:
            time_series = timestamps
        
        # Calculate time differences
        time_diffs = time_series.diff().dropna()
        
        if len(time_diffs) == 0:
            logger.warning(f"No valid time differences, using default: {default}m")
            return default
        
        # MEDIUM TERM FIX: Proper unit detection
        median_diff_minutes = None
        
        # Check if we have Timedelta objects (pandas datetime diff)
        if hasattr(time_diffs.iloc[0], 'total_seconds'):
            # Already converted to Timedelta - convert to minutes
            diff_minutes = time_diffs.apply(lambda x: abs(x.total_seconds()) / 60.0)
            median_diff_minutes = float(diff_minutes.median())
        else:
            # Numeric deltas - need to detect unit
            raw_median = float(time_diffs.median())
            # Use absolute value to handle unsorted timestamps or wraparound
            abs_raw_median = abs(raw_median)
            
            # Try to detect unit (using absolute value)
            detected = _detect_timestamp_unit(abs_raw_median)
            if detected:
                unit_name, minutes = detected
                median_diff_minutes = minutes
                logger.debug(f"Detected timestamp unit: {unit_name}, median delta = {raw_median} {unit_name} = {minutes}m")
            else:
                # Fallback: try assuming nanoseconds (most common for Unix timestamps)
                minutes_from_ns = abs_raw_median / 1e9 / 60.0
                if 0.01 <= minutes_from_ns <= 1440:
                    median_diff_minutes = minutes_from_ns
                    logger.debug(f"Assuming nanoseconds, median delta = {raw_median} ns = {minutes_from_ns:.2f}m")
                else:
                    # Sanity check failed - this is likely wrong
                    logger.warning(
                        f"Timestamp delta {raw_median} doesn't map to reasonable interval "
                        f"(tried ns: {minutes_from_ns:.1f}m). Using default: {default}m"
                    )
                    return default
        
        # SANITY BOUND: Reject anything > 1 day (1440 minutes)
        if median_diff_minutes is None or median_diff_minutes > 1440:
            logger.warning(
                f"Detected interval {median_diff_minutes:.1f}m is > 1 day (likely unit bug), "
                f"using default: {default}m"
            )
            return default
        
        # Round to common intervals (1m, 5m, 15m, 30m, 60m)
        common_intervals = [1, 5, 15, 30, 60]
        detected_interval = min(common_intervals, key=lambda x: abs(x - median_diff_minutes))
        
        # Only use auto-detection if it's close to a common interval (within 20% tolerance)
        if abs(median_diff_minutes - detected_interval) / detected_interval < 0.2:
            logger.info(f"Auto-detected data interval: {median_diff_minutes:.1f}m → {detected_interval}m")
            return detected_interval
        else:
            logger.warning(
                f"Auto-detection unclear ({median_diff_minutes:.1f}m doesn't match common intervals), "
                f"using default: {default}m"
            )
            return default if default is not None else None
            
    except Exception as e:
        logger.warning(f"Failed to auto-detect interval from timestamps: {e}")
        return default if default is not None else None


def detect_interval_from_dataframe(
    df: pd.DataFrame,
    timestamp_column: str = 'ts',
    default: int = 5,
    explicit_interval: Optional[Union[int, str]] = None,
    experiment_config: Optional[Any] = None  # ExperimentConfig type (avoid circular import)
) -> int:
    """
    Auto-detect data bar interval from a dataframe's timestamp column.
    
    Precedence order:
    1. Explicit function arg (explicit_interval)
    2. Experiment config (experiment_config.data.bar_interval)
    3. Auto-detect from timestamps
    4. Fallback to default with LOUD warning
    
    Args:
        df: DataFrame with timestamp column
        timestamp_column: Name of timestamp column (default: 'ts')
        default: Default interval if detection fails (default: 5 minutes)
        explicit_interval: If provided, use this interval and skip auto-detection.
                          Can be int (minutes) or str like "5m", "15m", "1h"
        experiment_config: Optional ExperimentConfig object (for accessing data.bar_interval)
    
    Returns:
        Detected interval in minutes
    """
    # PRECEDENCE 1: Explicit function arg wins
    if explicit_interval is not None:
        try:
            minutes = normalize_interval(explicit_interval)
            logger.info(f"Using explicit interval from function arg: {explicit_interval} = {minutes}m")
            return minutes
        except ValueError as e:
            logger.warning(f"Invalid explicit_interval '{explicit_interval}': {e}, falling back to config/auto-detect")
    
    # PRECEDENCE 2: Experiment config
    if experiment_config is not None:
        # Try to get bar_interval from config
        bar_interval = None
        try:
            # Check if it's an ExperimentConfig with data.bar_interval
            if hasattr(experiment_config, 'data') and hasattr(experiment_config.data, 'bar_interval'):
                bar_interval = experiment_config.data.bar_interval
            # Also check direct bar_interval property (convenience)
            elif hasattr(experiment_config, 'bar_interval'):
                bar_interval = experiment_config.bar_interval
            # Legacy: check interval field
            elif hasattr(experiment_config, 'interval'):
                bar_interval = experiment_config.interval
        except Exception as e:
            logger.debug(f"Could not access bar_interval from config: {e}")
        
        if bar_interval is not None:
            try:
                minutes = normalize_interval(bar_interval)
                logger.info(f"Using bar interval from experiment config: {bar_interval} = {minutes}m")
                return minutes
            except ValueError as e:
                logger.warning(f"Invalid bar_interval in config '{bar_interval}': {e}, falling back to auto-detect")
    
    # PRECEDENCE 3: Auto-detect from timestamps
    if timestamp_column not in df.columns:
        logger.warning(
            f"Timestamp column '{timestamp_column}' not found and no config interval set. "
            f"Falling back to default: {default}m"
        )
        logger.warning(
            "⚠️  INTERVAL AUTO-DETECTION FAILED: Falling back to 5m. "
            "Set data.bar_interval in your experiment config to silence this warning."
        )
        return default
    
    timestamps = df[timestamp_column]
    # Pass default to avoid "Nonem" in warning messages
    detected = detect_interval_from_timestamps(timestamps, default=default, explicit_interval=None)
    
    # PRECEDENCE 4: Fallback with LOUD warning
    if detected is None:
        logger.warning(
            "⚠️  INTERVAL AUTO-DETECTION FAILED: Falling back to 5m. "
            "Set data.bar_interval in your experiment config to silence this warning."
        )
        return default
    
    return detected
