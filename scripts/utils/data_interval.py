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
Utility for auto-detecting data bar interval from timestamps

CRITICAL: Using wrong interval causes data leakage in PurgedTimeSeriesSplit.
For example, if data is 1-minute bars but code assumes 5-minute bars:
- 60m target horizon = 60 bars (correct)
- But code calculates: 60m / 5m = 12 bars (WRONG - leaks 48 minutes!)
"""


import pandas as pd
import numpy as np
from typing import Optional, Union, List
import logging

logger = logging.getLogger(__name__)


def detect_interval_from_timestamps(
    timestamps: Union[pd.Series, np.ndarray, List],
    default: int = 5
) -> int:
    """
    Auto-detect data bar interval (in minutes) from timestamp differences.
    
    Args:
        timestamps: Series, array, or list of timestamps (datetime-like)
        default: Default interval to use if detection fails (default: 5 minutes)
    
    Returns:
        Detected interval in minutes (rounded to common intervals: 1, 5, 15, 30, 60)
    """
    if timestamps is None or len(timestamps) < 2:
        logger.warning(f"Insufficient timestamps for interval detection, using default: {default}m")
        return default
    
    try:
        # Convert to pandas Series if needed
        if not isinstance(timestamps, pd.Series):
            if isinstance(timestamps[0], (int, float)):
                # Assume nanoseconds if numeric
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
        
        # Convert to minutes (handle both Timedelta and numeric)
        if hasattr(time_diffs.iloc[0], 'total_seconds'):
            diff_minutes = time_diffs.apply(lambda x: x.total_seconds() / 60.0)
        else:
            # Already in minutes or seconds - assume minutes for now
            diff_minutes = time_diffs
        
        # Use median (more robust to outliers than mean)
        median_diff_minutes = float(diff_minutes.median())
        
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
            return default
            
    except Exception as e:
        logger.warning(f"Failed to auto-detect interval from timestamps: {e}, using default: {default}m")
        return default


def detect_interval_from_dataframe(
    df: pd.DataFrame,
    timestamp_column: str = 'ts',
    default: int = 5
) -> int:
    """
    Auto-detect data bar interval from a dataframe's timestamp column.
    
    Args:
        df: DataFrame with timestamp column
        timestamp_column: Name of timestamp column (default: 'ts')
        default: Default interval if detection fails (default: 5 minutes)
    
    Returns:
        Detected interval in minutes
    """
    if timestamp_column not in df.columns:
        logger.warning(f"Timestamp column '{timestamp_column}' not found, using default: {default}m")
        return default
    
    timestamps = df[timestamp_column]
    return detect_interval_from_timestamps(timestamps, default=default)

