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
Purged Time Series Split for Financial ML

This CV splitter prevents temporal leakage by enforcing a "purge" gap
between training and test sets equal to the target horizon.

Critical for financial ML where standard K-Fold would leak future information.

ARCHITECTURAL IMPROVEMENT: Now uses TIME-BASED purging instead of row-count based.
This fixes the critical bug where different bar intervals (1m vs 5m) would cause
severe data leakage.
"""


import numpy as np
import pandas as pd
from sklearn.model_selection._split import _BaseKFold
from sklearn.utils import indexable
from typing import Optional, Union

try:
    from sklearn.utils.validation import _num_samples
except ImportError:
    # Fallback for older sklearn versions
    def _num_samples(x):
        """Get number of samples in array-like object"""
        if hasattr(x, '__len__'):
            return len(x)
        elif hasattr(x, 'shape'):
            return x.shape[0]
        else:
            raise TypeError(f"Cannot determine number of samples in {type(x)}")


class PurgedTimeSeriesSplit(_BaseKFold):
    """
    Time Series Split with TIME-BASED Embargo/Purge to prevent overlap leakage.
    
    This is CRITICAL for financial ML. Standard K-Fold shuffles data randomly,
    which destroys time patterns and allows training on future data to predict past data.
    
    ARCHITECTURAL FIX: Uses actual timestamps to calculate purge gaps, not row counts.
    This prevents leakage when data interval doesn't match assumptions (e.g., 1m vs 5m bars).
    
    **CRITICAL FOR PANEL DATA:**
    This splitter correctly handles panel data where multiple rows share the same timestamp
    (e.g., 50 symbols = 50 rows per bar). With panel data:
    - Row-count purging is INVALID: 17 rows ≠ 17 bars (with 50 symbols, 17 rows ≈ 20 seconds)
    - Time-based purging is REQUIRED: Purges ALL rows with timestamps in the purge window
    
    Args:
        n_splits: Number of folds (default: 5)
        purge_overlap_time: pd.Timedelta for the purge gap (e.g., pd.Timedelta(minutes=60))
                           REQUIRED for panel data. If None, falls back to purge_overlap (row-count based)
        purge_overlap: Number of rows to drop (DEPRECATED - DANGEROUS for panel data)
                      Only used if purge_overlap_time is None
                      WARNING: With panel data, this causes catastrophic leakage!
        time_column_values: Array of timestamps for each sample (REQUIRED for time-based purging)
                           Must be sorted in ascending order
                           Each timestamp can appear multiple times (panel data structure)
    
    Example:
        # Time-based (REQUIRED for panel data):
        import pandas as pd
        timestamps = df['ts'].values  # Sorted timestamps (may have duplicates for panel data)
        purge_time = pd.Timedelta(minutes=60)  # 60-minute target horizon
        cv = PurgedTimeSeriesSplit(
            n_splits=5, 
            purge_overlap_time=purge_time,
            time_column_values=timestamps
        )
        
        # Legacy row-count based (DEPRECATED - DO NOT USE for panel data):
        # cv = PurgedTimeSeriesSplit(n_splits=5, purge_overlap=17)  # DANGEROUS!
    """
    
    def __init__(
        self, 
        n_splits=5, 
        purge_overlap_time: Optional[pd.Timedelta] = None,
        purge_overlap: int = 0,  # Legacy parameter for backward compatibility
        time_column_values: Optional[Union[np.ndarray, pd.Series, list]] = None
    ):
        super().__init__(n_splits, shuffle=False, random_state=None)
        self.purge_overlap_time = purge_overlap_time
        self.purge_overlap = purge_overlap  # Legacy support
        self.time_vals = time_column_values
        
        # Validate time-based mode
        if purge_overlap_time is not None:
            if time_column_values is None:
                raise ValueError(
                    "CRITICAL: time_column_values is REQUIRED when using purge_overlap_time. "
                    "Panel data (multiple symbols per timestamp) requires time-based purging. "
                    "Row-count purging causes catastrophic leakage (1 bar = N rows, not 1 row)."
                )
            
            # Validate panel data structure: Check if multiple rows share same timestamp
            if len(time_column_values) > 0:
                if isinstance(time_column_values, pd.Series):
                    unique_timestamps = time_column_values.nunique()
                else:
                    unique_timestamps = len(np.unique(time_column_values))
                total_rows = len(time_column_values)
                avg_rows_per_timestamp = total_rows / unique_timestamps if unique_timestamps > 0 else 1
                
                if avg_rows_per_timestamp > 1.1:  # More than 10% overhead suggests panel data
                    import warnings
                    warnings.warn(
                        f"Panel data detected: {avg_rows_per_timestamp:.1f} rows per timestamp on average. "
                        f"Time-based purging is REQUIRED. Row-count purging would cause leakage."
                    )
            # Convert to pandas Series for easier manipulation
            if not isinstance(time_column_values, pd.Series):
                if isinstance(time_column_values, (list, np.ndarray)):
                    # Try to infer if it's numeric (nanoseconds) or datetime
                    if len(time_column_values) > 0:
                        first_val = time_column_values[0]
                        if isinstance(first_val, (int, float, np.integer, np.floating)):
                            # Assume nanoseconds
                            self.time_vals = pd.to_datetime(time_column_values, unit='ns')
                        else:
                            self.time_vals = pd.Series(time_column_values)
                    else:
                        self.time_vals = pd.Series(time_column_values)
                else:
                    self.time_vals = pd.Series(time_column_values)
            else:
                self.time_vals = time_column_values
            
            # Ensure sorted
            if not self.time_vals.is_monotonic_increasing:
                import warnings
                warnings.warn(
                    "time_column_values is not sorted. Sorting now (this may affect fold boundaries)."
                )
                sort_idx = np.argsort(self.time_vals)
                self.time_vals = self.time_vals.iloc[sort_idx] if isinstance(self.time_vals, pd.Series) else pd.Series(self.time_vals).iloc[sort_idx]
    
    def split(self, X, y=None, groups=None):
        """
        Generate indices to split data into training and test set.
        
        Uses TIME-BASED purging when purge_overlap_time is provided (recommended).
        Falls back to row-count based purging for backward compatibility.
        
        Args:
            X: Feature matrix
            y: Target vector (optional)
            groups: Group labels (optional, not used but required by interface)
        
        Yields:
            (train_indices, test_indices) tuples
        """
        X, y, groups = indexable(X, y, groups)
        n_samples = _num_samples(X)
        indices = np.arange(n_samples)
        
        # Calculate fold sizes (distribute samples evenly across folds)
        fold_sizes = np.full(self.n_splits, n_samples // self.n_splits, dtype=int)
        fold_sizes[:n_samples % self.n_splits] += 1
        
        current = 0
        
        for fold_idx, fold_size in enumerate(fold_sizes):
            start, stop = current, current + fold_size
            test_indices = indices[start:stop]
            
            # TIME-BASED PURGING (CRITICAL for panel data)
            # Panel data structure: Multiple rows share the same timestamp (e.g., 50 symbols = 50 rows per bar)
            # With 50 symbols, 1 bar = 50 rows. Row-count purging (17 rows) = ~20 seconds, not 60 minutes!
            # Time-based purging correctly handles this by purging ALL rows with timestamps in the purge window.
            if self.purge_overlap_time is not None and self.time_vals is not None:
                # Find the timestamp of the first test sample
                test_start_time = self.time_vals.iloc[start] if isinstance(self.time_vals, pd.Series) else self.time_vals[start]
                
                # Calculate the cutoff time (test_start - purge_window)
                # This ensures no training data overlaps with the test target horizon
                cutoff_time = test_start_time - self.purge_overlap_time
                
                # CRITICAL: Find ALL rows with timestamp <= cutoff_time
                # searchsorted(..., side='right') returns the rightmost position where cutoff_time would be inserted,
                # which means it includes ALL rows with timestamp <= cutoff_time.
                # This correctly handles panel data where multiple rows share the same timestamp.
                if isinstance(self.time_vals, pd.Series):
                    # Use pandas searchsorted for Series
                    train_stop_idx = self.time_vals.searchsorted(cutoff_time, side='right')
                else:
                    # Use numpy searchsorted for array
                    train_stop_idx = np.searchsorted(self.time_vals, cutoff_time, side='right')
                
                # Ensure we don't go negative
                train_stop_idx = max(0, train_stop_idx)
                
                # Validate: Ensure no timestamp overlap between train and test
                # This is a safety check to catch any edge cases
                if train_stop_idx > 0:
                    train_indices = indices[:train_stop_idx]
                    # Verify no timestamp leakage (all train timestamps < cutoff_time)
                    # train_max_time should be <= cutoff_time (which is test_min_time - purge_overlap_time)
                    # If train_max_time > cutoff_time, there's a problem
                    train_max_time = self.time_vals.iloc[train_stop_idx - 1] if isinstance(self.time_vals, pd.Series) else self.time_vals[train_stop_idx - 1]
                    test_min_time = test_start_time
                    # CRITICAL: train_max_time should be <= cutoff_time (not >= test_min_time - purge)
                    # The correct check: train_max_time should NOT exceed the cutoff
                    if train_max_time > cutoff_time:
                        import warnings
                        warnings.warn(
                            f"⚠️  CRITICAL: Timestamp overlap detected in fold {fold_idx + 1}: "
                            f"train_max={train_max_time} > cutoff={cutoff_time} (test_min={test_min_time} - purge={self.purge_overlap_time})"
                        )
                    
                    yield train_indices, test_indices
                else:
                    # If purge eats the whole train set (early folds), skip this fold
                    # This is expected behavior when there's not enough historical data
                    # Only warn once per splitter instance to avoid spam
                    if not hasattr(self, '_warned_empty_fold'):
                        import warnings
                        warnings.warn(
                            f"Fold {fold_idx + 1}: purge_overlap_time={self.purge_overlap_time} is too large. "
                            f"Train set would be empty (test_start={test_start_time}, cutoff={cutoff_time}). "
                            f"Skipping this fold. (This warning will only appear once per CV splitter)"
                        )
                        self._warned_empty_fold = True
            
            # LEGACY ROW-COUNT BASED PURGING (DEPRECATED - DANGEROUS FOR PANEL DATA)
            # CRITICAL WARNING: Row-count purging is INVALID for panel data where multiple symbols
            # share the same timestamp. With 50 symbols, 1 bar = 50 rows. Purging 17 rows = ~20 seconds,
            # not 60 minutes! This causes 100% data leakage.
            else:
                import warnings
                warnings.warn(
                    f"⚠️  CRITICAL: Using row-count based purging (DEPRECATED). "
                    f"This is INVALID for panel data (cross-sectional). "
                    f"If you have multiple symbols per timestamp, this will cause catastrophic data leakage. "
                    f"Please provide purge_overlap_time and time_column_values for time-based purging.",
                    UserWarning
                )
                # The 'Train' set is everything BEFORE the test set
                # BUT we must cut off the end of the training set to prevent leakage
                train_stop = start - self.purge_overlap
                
                if train_stop > 0:
                    train_indices = indices[:train_stop]
                    yield train_indices, test_indices
                else:
                    # If purge eats the whole train set (early folds), skip this fold
                    warnings.warn(
                        f"Fold {fold_idx + 1}: purge_overlap={self.purge_overlap} is too large. "
                        f"Train set would be empty (start={start}, train_stop={train_stop}). "
                        f"Skipping this fold."
                    )
            
            current = stop
    
    def get_n_splits(self, X=None, y=None, groups=None):
        """Returns the number of splitting iterations in the cross-validator"""
        return self.n_splits

