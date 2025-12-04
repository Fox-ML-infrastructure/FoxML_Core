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
Cross-sectional data loading utilities for ranking scripts.

This module provides functions to load and prepare cross-sectional data
that matches the training pipeline's data structure.
"""


import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
import warnings

logger = logging.getLogger(__name__)


def load_mtf_data_for_ranking(
    data_dir: Path,
    symbols: List[str],
    max_rows_per_symbol: Optional[int] = None
) -> Dict[str, pd.DataFrame]:
    """
    Load MTF data for multiple symbols (matches training pipeline structure).
    
    Args:
        data_dir: Directory containing symbol data
        symbols: List of symbols to load
        max_rows_per_symbol: Optional limit on rows per symbol (most recent rows)
                            Default: None (load all). For ranking, use 10000-50000 for speed.
    
    Returns:
        Dictionary mapping symbol -> DataFrame
    """
    """
    Load MTF data for multiple symbols (matches training pipeline structure).
    
    Args:
        data_dir: Directory containing symbol data
        symbols: List of symbols to load
        max_rows_per_symbol: Optional limit on rows per symbol (most recent)
    
    Returns:
        Dictionary mapping symbol -> DataFrame
    """
    mtf_data = {}
    
    for symbol in symbols:
        # Try different possible file locations (matching training pipeline)
        possible_paths = [
            data_dir / f"symbol={symbol}" / f"{symbol}.parquet",  # New structure
            data_dir / f"{symbol}.parquet",  # Direct file
            data_dir / f"{symbol}_mtf.parquet",  # Legacy format
        ]
        
        symbol_file = None
        for path in possible_paths:
            if path.exists():
                symbol_file = path
                break
        
        if symbol_file and symbol_file.exists():
            try:
                df = pd.read_parquet(symbol_file)
                
                # Apply row limit if specified (most recent rows)
                if max_rows_per_symbol and len(df) > max_rows_per_symbol:
                    df = df.tail(max_rows_per_symbol)
                    logger.debug(f"Limited {symbol} to {max_rows_per_symbol} most recent rows")
                
                mtf_data[symbol] = df
                logger.debug(f"Loaded {symbol}: {df.shape}")
            except Exception as e:
                logger.error(f"Error loading {symbol}: {e}")
        else:
            logger.warning(f"File not found for {symbol}. Tried: {possible_paths}")
    
    logger.info(f"Loaded {len(mtf_data)} symbols: {list(mtf_data.keys())}")
    return mtf_data


def prepare_cross_sectional_data_for_ranking(
    mtf_data: Dict[str, pd.DataFrame],
    target_column: str,
    min_cs: int = 10,
    max_cs_samples: Optional[int] = None,
    feature_names: Optional[List[str]] = None
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[List[str]], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Prepare cross-sectional data for ranking (simplified version of training pipeline).
    
    Args:
        mtf_data: Dictionary of symbol -> DataFrame
        target_column: Target column name
        min_cs: Minimum cross-sectional size per timestamp
        max_cs_samples: Maximum samples per timestamp (default: 1000)
        feature_names: Optional list of feature names (auto-discovered if None)
    
    Returns:
        Tuple of (X, y, feature_names, symbols, time_vals) or (None,)*5 on error
    """
    if not mtf_data:
        logger.error("No data provided")
        return (None,) * 5
    
    # Default max_cs_samples to match training pipeline
    if max_cs_samples is None:
        max_cs_samples = 1000
        logger.info(f"Using default max_cs_samples={max_cs_samples}")
    
    logger.info(f"🎯 Building cross-sectional data for target: {target_column}")
    logger.info(f"📊 Cross-sectional sampling: min_cs={min_cs}, max_cs_samples={max_cs_samples}")
    
    # Combine all symbol data
    all_data = []
    for symbol, df in mtf_data.items():
        if target_column not in df.columns:
            logger.debug(f"Skipping {symbol}: target '{target_column}' not found")
            continue
        
        df_copy = df.copy()
        df_copy['symbol'] = symbol
        all_data.append(df_copy)
    
    if not all_data:
        logger.error(f"Target '{target_column}' not found in any symbol")
        return (None,) * 5
    
    # CRITICAL: Ensure we have a time column for panel data validation
    # Without timestamps, we cannot use time-based purging and will cause data leakage
    
    combined_df = pd.concat(all_data, ignore_index=True)
    logger.info(f"Combined data shape: {combined_df.shape}")
    
    # Normalize time column name
    time_col = "timestamp" if "timestamp" in combined_df.columns else ("ts" if "ts" in combined_df.columns else None)
    
    # CRITICAL: Sort by timestamp IMMEDIATELY after combining
    # This ensures data is always sorted and prevents warnings later
    if time_col is not None:
        combined_df = combined_df.sort_values(time_col).reset_index(drop=True)
        logger.debug(f"Sorted combined data by {time_col}")
    
    # Enforce min_cs: filter timestamps that don't meet cross-sectional size
    # But be lenient: if we have fewer symbols than min_cs, use what we have
    if time_col is not None:
        cs = combined_df.groupby(time_col)["symbol"].transform("size")
        # Adjust min_cs to be at most the number of symbols we have
        effective_min_cs = min(min_cs, len(mtf_data))
        combined_df = combined_df[cs >= effective_min_cs]
        logger.info(f"After min_cs={effective_min_cs} filter (requested {min_cs}, have {len(mtf_data)} symbols): {combined_df.shape}")
        
        if len(combined_df) == 0:
            logger.warning(f"No data after min_cs filter - all timestamps have < {effective_min_cs} symbols")
            return (None,) * 5
    else:
        # CRITICAL: Panel data REQUIRES timestamps for time-based purging
        # Without timestamps, row-count purging causes catastrophic leakage (1 bar = N rows, not 1 row)
        logger.error("CRITICAL: No time column found in panel data. Time-based purging is REQUIRED.")
        logger.error("  Panel data structure: multiple symbols per timestamp means row-count purging is invalid.")
        logger.error("  Example: With 50 symbols, 1 bar = 50 rows. Purging 17 rows = ~20 seconds, not 60 minutes!")
        return (None,) * 5
        
        # Apply cross-sectional sampling per timestamp
        # CRITICAL: Shuffle symbols within each timestamp to avoid bias
        # If data is sorted alphabetically, we'd always sample AAPL, AMZN, etc. and miss ZZZ
        if max_cs_samples:
            # Add random shuffle column per timestamp group
            # CRITICAL FIX: Use deterministic timestamp-based seeding instead of hash()
            # hash() output changes every Python restart (salted for security), breaking reproducibility
            # Using timestamp integer ensures same shuffle for same timestamp across all runs
            def _get_deterministic_shuffle(group):
                """Generate deterministic shuffle key based on timestamp"""
                timestamp = group.name  # The timestamp value for this group
                # Convert timestamp to integer seed (works for pd.Timestamp, datetime, or numeric)
                if isinstance(timestamp, pd.Timestamp):
                    seed = int(timestamp.timestamp()) % (2**31)  # Use timestamp as seed
                elif isinstance(timestamp, (int, float)):
                    seed = int(timestamp) % (2**31)
                else:
                    # Fallback: use string hash but with fixed seed
                    seed = hash(str(timestamp)) % (2**31)
                return np.random.RandomState(seed).permutation(len(group))
            
            combined_df["_shuffle_key"] = combined_df.groupby(time_col)["symbol"].transform(_get_deterministic_shuffle)
            combined_df = (combined_df
                           .sort_values([time_col, "_shuffle_key"])
                           .groupby(time_col, group_keys=False)
                           .head(max_cs_samples)
                           .drop(columns=["_shuffle_key"]))
            logger.info(f"After max_cs_samples={max_cs_samples} filter (shuffled per timestamp): {combined_df.shape}")
            # Data is already sorted by [time_col, _shuffle_key], so it's sorted by time_col
    
    # Auto-discover features if not provided
    if feature_names is None:
        feature_names = [col for col in combined_df.columns 
                        if not any(col.startswith(prefix) for prefix in 
                                 ['fwd_ret_', 'will_peak', 'will_valley', 'mdd_', 'mfe_', 'y_will_',
                                  'tth_', 'p_', 'barrier_', 'hit_'])
                        and col not in (['symbol', time_col, target_column] if time_col else ['symbol', target_column])]
        logger.info(f"Auto-discovered {len(feature_names)} features")
    
    # Extract target
    if target_column not in combined_df.columns:
        logger.error(f"Target '{target_column}' not in combined data")
        return (None,) * 5
    
    y = combined_df[target_column].values
    y = pd.Series(y).replace([np.inf, -np.inf], np.nan).values
    
    # Extract features
    feature_df = combined_df[feature_names].copy()
    
    # Convert to numeric and handle infinities
    for col in feature_df.columns:
        feature_df.loc[:, col] = pd.to_numeric(feature_df[col], errors='coerce')
    feature_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Drop columns that are entirely NaN
    before_cols = feature_df.shape[1]
    feature_df = feature_df.dropna(axis=1, how='all')
    dropped = before_cols - feature_df.shape[1]
    if dropped:
        logger.info(f"🔧 Dropped {dropped} all-NaN feature columns")
        # Update feature_names to match
        feature_names = [f for f in feature_names if f in feature_df.columns]
    
    # Ensure only numeric dtypes
    numeric_cols = [c for c in feature_df.columns if pd.api.types.is_numeric_dtype(feature_df[c])]
    if len(numeric_cols) != feature_df.shape[1]:
        non_numeric_dropped = feature_df.shape[1] - len(numeric_cols)
        feature_df = feature_df[numeric_cols]
        feature_names = [f for f in feature_names if f in numeric_cols]
        logger.info(f"🔧 Dropped {non_numeric_dropped} non-numeric feature columns")
    
    # Check if we have any features left
    if len(feature_names) == 0:
        logger.error("No features remaining after filtering")
        return (None,) * 5
    
    # Build feature matrix
    X = feature_df.to_numpy(dtype=np.float32, copy=False)
    
    # Check if we have any data
    if X.shape[0] == 0:
        logger.error("Feature matrix is empty - no data to process")
        return (None,) * 5
    
    # Clean data: remove rows with invalid target or too many NaN features
    target_valid = ~np.isnan(y) & np.isfinite(y)
    
    # Compute feature NaN ratio safely (suppress warning for empty slices)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        if X.shape[0] > 0 and X.shape[1] > 0:
            feature_nan_ratio = np.isnan(X).mean(axis=1)
        else:
            feature_nan_ratio = np.ones(X.shape[0])  # All invalid if empty
            logger.warning("Feature matrix has zero columns - all features invalid")
    
    feature_valid = feature_nan_ratio <= 0.5  # Allow up to 50% NaN in features
    
    valid_mask = target_valid & feature_valid
    
    if not valid_mask.any():
        logger.error(f"No valid data after cleaning:")
        logger.error(f"  Target: {len(y)} total, {target_valid.sum()} valid ({np.isnan(y).sum()} NaN, {np.sum(~np.isfinite(y))} inf)")
        logger.error(f"  Features: {X.shape[0]} rows, {X.shape[1]} cols, {feature_valid.sum()} valid rows")
        if X.shape[0] > 0:
            logger.error(f"  Feature NaN: {np.isnan(X).sum()} total, mean per row: {feature_nan_ratio.mean():.2%}")
        return (None,) * 5
    
    X_clean = X[valid_mask]
    y_clean = y[valid_mask]
    symbols_clean = combined_df['symbol'].values[valid_mask]
    time_vals = combined_df[time_col].values[valid_mask] if time_col else None
    
    # CRITICAL FIX: Do NOT impute here - this causes data leakage!
    # Imputation must happen INSIDE the CV loop (fit on train, transform test)
    # The imputation will be handled by sklearn Pipeline in train_and_evaluate_models
    # We only remove rows with >50% NaN features, but keep NaN values for proper CV imputation
    
    # CRITICAL: Ensure time_vals is sorted (required for PurgedTimeSeriesSplit)
    # Data should already be sorted from earlier steps, but verify and fix if needed
    if time_vals is not None and len(time_vals) > 1:
        time_series = pd.Series(time_vals) if not isinstance(time_vals, pd.Series) else time_vals
        if not time_series.is_monotonic_increasing:
            # Sort silently - data should be pre-sorted, but handle edge cases
            sort_idx = np.argsort(time_vals)
            X_clean = X_clean[sort_idx]
            y_clean = y_clean[sort_idx]
            symbols_clean = symbols_clean[sort_idx]
            time_vals = time_series.iloc[sort_idx].values if isinstance(time_series, pd.Series) else time_series[sort_idx]
            logger.debug(f"  Re-sorted data by timestamp (should be rare)")
    
    logger.info(f"✅ Cleaned data: {len(X_clean)} samples, {X_clean.shape[1]} features")
    logger.info(f"   Removed {len(X) - len(X_clean)} rows due to cleaning")
    logger.info(f"   ⚠️  Note: NaN values preserved for CV-safe imputation (no leakage)")
    
    return X_clean, y_clean, feature_names, symbols_clean, time_vals

