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
        
        # Check if we have Timedelta objects (pandas datetime diff) or numeric
        is_timedelta = hasattr(time_diffs.iloc[0], 'total_seconds')
        
        # DEBUG: Log delta statistics to diagnose negative/unsorted timestamp issues
        if is_timedelta:
            # Timedelta objects - convert to nanoseconds for logging
            deltas_ns = np.array([td.total_seconds() * 1e9 for td in time_diffs])
        else:
            # Numeric deltas - assume already in nanoseconds
            deltas_ns = time_diffs.values.astype(np.float64)
        
        if len(deltas_ns) > 0:
            logger.debug(
                f"data_interval debug: n={len(deltas_ns)}, "
                f"min_delta_ns={deltas_ns.min()}, max_delta_ns={deltas_ns.max()}, "
                f"median_delta_ns={np.median(deltas_ns)}"
            )
        
        # GUARD: Filter out negative deltas (indicates unsorted timestamps)
        positive_mask = deltas_ns > 0
        if not positive_mask.any():
            logger.error(
                f"All timestamp deltas are non-positive (min={deltas_ns.min()}, max={deltas_ns.max()}); "
                f"timestamps appear unsorted. Using default: {default}m"
            )
            return default
        
        # Filter to only positive deltas
        if is_timedelta:
            time_diffs = time_diffs[positive_mask]
        else:
            time_diffs = time_diffs[positive_mask]
        
        # CRITICAL: Filter out insane gaps (> 1 day) BEFORE computing median
        # This prevents outliers (weekends, data gaps, bad rows) from contaminating detection
        # Load max reasonable minutes from config
        try:
            from CONFIG.config_loader import get_cfg
            MAX_REASONABLE_MINUTES = float(get_cfg("pipeline.data_interval.max_reasonable_minutes", default=1440.0, config_name="pipeline_config"))
            MAX_GAP_FACTOR = float(get_cfg("pipeline.data_interval.max_gap_factor", default=10.0, config_name="pipeline_config"))
        except Exception:
            MAX_REASONABLE_MINUTES = 1440.0  # 1 day (fallback)
            MAX_GAP_FACTOR = 10.0  # Ignore gaps > 10x median (fallback)
        
        if is_timedelta:
            # Convert to minutes first, then filter
            diff_minutes = time_diffs.apply(lambda x: x.total_seconds() / 60.0)
            # Step 1: Filter out gaps > 1 day
            sane_mask = diff_minutes <= MAX_REASONABLE_MINUTES
            sane_diff_minutes = diff_minutes[sane_mask]
            
            if len(sane_diff_minutes) == 0:
                # All gaps are insane - log max gap for debugging
                max_gap_minutes = float(diff_minutes.max())
                logger.warning(
                    f"All timestamp deltas are huge (max={max_gap_minutes:.1f}m = {max_gap_minutes/1440:.1f} days); "
                    f"cannot infer bar size. Using default: {default}m"
                )
                return default
            
            # Step 2: Compute rough median to identify base cadence
            rough_median = float(sane_diff_minutes.median())
            
            # Step 3: Filter out large gaps relative to median (overnight/weekend gaps)
            # This prevents 270m or 1210m gaps from contaminating detection when base cadence is 5m
            gap_threshold = rough_median * MAX_GAP_FACTOR
            small_gaps_mask = sane_diff_minutes <= gap_threshold
            small_gaps_minutes = sane_diff_minutes[small_gaps_mask]
            
            if len(small_gaps_minutes) == 0:
                # All gaps are large relative to median - use rough median anyway
                logger.debug(f"All gaps are large relative to median ({rough_median:.1f}m), using rough median")
                median_diff_minutes = rough_median
            else:
                # Use median of small gaps only (the actual bar cadence)
                median_diff_minutes = float(small_gaps_minutes.median())
                
                # Log if we filtered out large gaps
                n_filtered = len(sane_diff_minutes) - len(small_gaps_minutes)
                if n_filtered > 0:
                    logger.debug(
                        f"Filtered out {n_filtered} large gaps (>{gap_threshold:.1f}m = {MAX_GAP_FACTOR}x median) "
                        f"before computing final median. Using {len(small_gaps_minutes)} small gaps. "
                        f"Rough median: {rough_median:.1f}m → Final median: {median_diff_minutes:.1f}m"
                    )
            
            # Debug: log if we filtered out any insane gaps
            n_filtered = len(diff_minutes) - len(sane_diff_minutes)
            if n_filtered > 0:
                logger.debug(
                    f"Filtered out {n_filtered} insane timestamp gaps (>{MAX_REASONABLE_MINUTES}m) "
                    f"before computing median. Using {len(sane_diff_minutes)} sane deltas."
                )
        else:
            # Numeric deltas - need to detect unit first, then filter
            # Strategy: try to convert all deltas to minutes (assuming nanoseconds), filter, then detect
            # This is safe because we'll validate the result anyway
            
            # Try assuming nanoseconds (most common) and convert to minutes
            deltas_ns = time_diffs.values.astype(np.float64)
            deltas_minutes = deltas_ns / 1e9 / 60.0
            
            # Step 1: Filter out insane gaps (> 1 day)
            sane_mask = deltas_minutes <= MAX_REASONABLE_MINUTES
            sane_deltas_minutes = deltas_minutes[sane_mask]
            
            if len(sane_deltas_minutes) == 0:
                # All gaps are insane - try to give helpful error
                max_gap_ns = float(deltas_ns.max())
                max_gap_minutes = max_gap_ns / 1e9 / 60.0
                logger.warning(
                    f"All timestamp deltas are huge (max={max_gap_minutes:.1f}m = {max_gap_minutes/1440:.1f} days); "
                    f"cannot infer bar size. Using default: {default}m"
                )
                return default
            
            # Step 2: Compute rough median to identify base cadence
            rough_median = float(np.median(sane_deltas_minutes))
            
            # Step 3: Filter out large gaps relative to median (overnight/weekend gaps)
            gap_threshold = rough_median * MAX_GAP_FACTOR
            small_gaps_mask = sane_deltas_minutes <= gap_threshold
            small_gaps_minutes = sane_deltas_minutes[small_gaps_mask]
            
            if len(small_gaps_minutes) == 0:
                # All gaps are large relative to median - use rough median anyway
                logger.debug(f"All gaps are large relative to median ({rough_median:.1f}m), using rough median")
                median_sane_minutes = rough_median
            else:
                # Use median of small gaps only (the actual bar cadence)
                median_sane_minutes = float(np.median(small_gaps_minutes))
                
                # Log if we filtered out large gaps
                n_filtered_large = len(sane_deltas_minutes) - len(small_gaps_minutes)
                if n_filtered_large > 0:
                    logger.debug(
                        f"Filtered out {n_filtered_large} large gaps (>{gap_threshold:.1f}m = {MAX_GAP_FACTOR}x median) "
                        f"before computing final median. Using {len(small_gaps_minutes)} small gaps. "
                        f"Rough median: {rough_median:.1f}m → Final median: {median_sane_minutes:.1f}m"
                    )
            
            # Debug: log if we filtered out any insane gaps (> 1 day)
            n_filtered_insane = len(deltas_minutes) - len(sane_deltas_minutes)
            if n_filtered_insane > 0:
                logger.debug(
                    f"Filtered out {n_filtered_insane} insane timestamp gaps (>{MAX_REASONABLE_MINUTES}m) "
                    f"before computing median. Using {len(sane_deltas_minutes)} sane deltas."
                )
            
            # Try to detect unit from the median of sane deltas
            # Use the raw median value (in original units) for unit detection
            raw_median = float(time_diffs[sane_mask].median())
            abs_raw_median = abs(raw_median)
            
            detected = _detect_timestamp_unit(abs_raw_median)
            if detected:
                unit_name, minutes = detected
                median_diff_minutes = minutes
                logger.debug(f"Detected timestamp unit: {unit_name}, median delta = {raw_median} {unit_name} = {minutes}m")
            else:
                # Use the median we computed from sane deltas (already in minutes, assuming ns)
                median_diff_minutes = median_sane_minutes
                logger.debug(f"Using median of sane deltas (assuming nanoseconds): {median_diff_minutes:.2f}m")
        
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
